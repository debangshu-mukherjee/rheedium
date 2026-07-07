"""Functions for simulating RHEED patterns and diffraction patterns.

Extended Summary
----------------
This module provides functions for simulating Reflection High-Energy Electron
Diffraction (RHEED) patterns using kinematic approximations with proper atomic
form factors and surface physics. It includes utilities for calculating
electron wavelengths, incident wavevectors, diffraction intensities with CTRs,
and complete RHEED patterns from crystal structures.

Routine Listings
----------------
:func:`compute_kinematic_intensities_with_ctrs`
    Calculate kinematic intensities with CTR contributions.
:func:`detector_extent_mm`
    Convert detector pixel calibration and beam centre to imshow extent.
:func:`ewald_simulator`
    Simulate RHEED using exact Ewald sphere-CTR intersection.
:func:`find_kinematic_reflections`
    Find reflections satisfying kinematic conditions.
:func:`kinematic_amplitude`
    Render a single coherent kinematic Ewald pattern as complex amplitude.
:func:`log_compress_image`
    Apply normalized log compression for screen-style visualization.
:func:`multislice_propagate`
    Propagate electron wave through potential slices along z (transmission
    geometry -- not valid for grazing-incidence RHEED).
:func:`multislice_amplitude`
    Return reciprocal-space transmission multislice amplitude before
    modulus-squared.
:func:`multislice_detector_amplitude`
    Render transmission multislice amplitudes onto a dense detector field.
:func:`multislice_simulator`
    Simulate a transmission diffraction pattern from potential slices;
    RHEED users should use the reflection multislice pipeline instead.
:func:`project_on_detector_geometry`
    Project wavevectors with full detector geometry support.
:func:`render_pattern_to_image`
    Rasterize a sparse RHEEDPattern onto a dense detector image grid.
:func:`render_amplitude_to_field`
    Rasterize sparse complex reflection amplitudes onto a dense detector field.
:func:`render_ctr_amplitude_to_field`
    Rasterize sparse CTR amplitudes onto elongated complex detector streaks.
:func:`simulate_detector_image`
    High-level kinematic detector-image orchestration with beam broadening.
:func:`sliced_crystal_to_projected_potential_slices`
    Convert SlicedCrystal to projected-potential slices for multislice
    simulation.

Notes
-----
All functions support JAX transformations and automatic differentiation for
gradient-based optimization and inverse problems.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Final, Tuple
from jax.experimental import checkify
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.procs.distribution_binds import (
    BEAM_AXIS_IDS,
    STRUCTURE_AXIS_IDS,
    bind_kinematic_axis_distribution,
    bind_multislice_axis_distribution,
)
from rheedium.procs.surface_modifier import (
    bind_step_edge_distribution,
    bind_twin_wall_distribution,
)
from rheedium.tools import (
    gauss_hermite_nodes_weights,
    incidence_angles_to_radians,
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.types import (
    BeamModeDistribution,
    BeamSpec,
    CrystalStructure,
    DetectorGeometry,
    Distribution,
    KinematicAxisUpdate,
    MultisliceAxisUpdate,
    OrientationDistribution,
    PotentialSlices,
    ReductionMode,
    RenderParams,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    SurfaceCTRParams,
    create_distribution,
    create_potential_slices,
    create_rheed_pattern,
    detector_beam_center_px,
    detector_image_shape_px,
    detector_pixel_size_mm,
    detector_psf_sigma_pixels,
    identify_surface_atoms,
    orientation_to_distribution,
    scalar_float,
    scalar_int,
    scalar_num,
)
from rheedium.types import (
    detector_distance_mm as detector_geometry_distance_mm,
)
from rheedium.types import (
    detector_extent_mm as detector_geometry_extent_mm,
)
from rheedium.ucell import reciprocal_lattice_vectors

from .beam_averaging import (
    apply_distributions,
    decompose_beam_modes,
    detector_psf_convolve,
)
from .finite_domain import (
    compute_shell_sigma,
    extent_to_rod_sigma,
    find_ctr_ewald_intersection,
)
from .form_factors import (
    atomic_scattering_factor,
    projected_potential,
)
from .surface_rods import (
    ctr_truncation_amplitude,
    ctr_truncation_intensity,
    roughness_damping,
)

_VALID_THRESHOLD: Final[float] = 0.5
_KINEMATIC_KERNEL: Final[str] = "kinematic"
_MULTISLICE_KERNEL: Final[str] = "multislice"
_SUPPORTED_LAYER0_KERNELS: Final[frozenset[str]] = frozenset(
    {_KINEMATIC_KERNEL, _MULTISLICE_KERNEL}
)


def _validate_layer0_kernel(kernel: str) -> None:
    """Validate the public Layer-0 kernel selector."""
    if kernel not in _SUPPORTED_LAYER0_KERNELS:
        raise ValueError(
            "Unsupported kernel. Expected one of: "
            f"{', '.join(sorted(_SUPPORTED_LAYER0_KERNELS))}."
        )


def _resolve_layer_attenuation(
    layer_attenuation: scalar_float,
    ctr_regularization: scalar_float | None,
) -> scalar_float:
    r"""Resolve the CTR attenuation ε, honoring the deprecated alias.

    ``ctr_regularization`` was the additive constant in the legacy
    ``1 / (sin^2(pi l) + reg)`` shape. It is deprecated in favor of the
    per-layer amplitude attenuation ε of the semi-infinite truncation
    factor. When given, it is converted so the legacy Bragg-peak cap
    ``1/reg`` is preserved exactly: the new cap is
    ``1/(1 - exp(-ε))^2``, so ``ε = -log(1 - sqrt(reg))`` for
    ``0 < reg < 1``.
    """
    if ctr_regularization is None:
        return layer_attenuation
    warnings.warn(
        "ctr_regularization is deprecated; pass layer_attenuation "
        "(per-layer amplitude attenuation epsilon) instead. The given "
        "value is converted via epsilon = -log(1 - sqrt(reg)) so the "
        "legacy Bragg-peak cap 1/reg is preserved exactly.",
        DeprecationWarning,
        stacklevel=3,
    )
    reg: Float[Array, ""] = jnp.clip(
        jnp.asarray(ctr_regularization, dtype=jnp.float64), 1e-12, 1.0 - 1e-12
    )
    epsilon: Float[Array, ""] = -jnp.log(1.0 - jnp.sqrt(reg))
    return epsilon


@jaxtyped(typechecker=beartype)
def _compose_ewald_intensity(
    sf_intensity: scalar_float,
    l_value: scalar_float,
    q_z: scalar_float,
    surface_roughness: scalar_float,
    layer_attenuation: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    r"""Combine sparse Ewald intensity terms into the unified CTR model.

    The intensity is
    ``sf_intensity * T(l)**ctr_power * R(q_z)**roughness_power`` where
    ``T(l) = 1 / (1 - 2 e^{-eps} cos(2 pi l) + e^{-2 eps})`` is the
    semi-infinite truncation-rod intensity factor
    (:func:`~rheedium.simul.ctr_truncation_intensity`) and
    ``R(q_z) = exp(-sigma^2 q_z^2)`` is the roughness **intensity**
    factor (the square of
    :func:`~rheedium.simul.roughness_damping`). ``ctr_power`` and
    ``roughness_power`` are non-physical diagnostic knobs; both default
    to 1.0, which is the physical model.
    """
    ctr_modulation: Float[Array, ""] = ctr_truncation_intensity(
        l_values=jnp.asarray(l_value, dtype=jnp.float64),
        layer_attenuation=layer_attenuation,
    )
    roughness_intensity: Float[Array, ""] = (
        roughness_damping(
            q_z=jnp.asarray(q_z, dtype=jnp.float64),
            sigma_height=surface_roughness,
        )
        ** 2
    )
    total_intensity: Float[Array, ""] = (
        sf_intensity
        * ctr_modulation**ctr_power
        * roughness_intensity**roughness_power
    )
    return total_intensity, ctr_modulation, roughness_intensity


@jaxtyped(typechecker=beartype)
def project_on_detector_geometry(
    k_out: Float[Array, "N 3"],
    geometry: DetectorGeometry,
) -> Float[Array, "N 2"]:
    """Project output wavevectors onto detector with full geometry support.

    This function supports flat, tilted, and curved detector screens through
    the shared detector-geometry carrier.

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Output wavevectors in 1/Å. Shape (N, 3) where each row is [kx, ky, kz].
    geometry : DetectorGeometry
        Detector geometry configuration specifying distance, tilt, curvature,
        and center offsets.

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [horizontal, vertical] coordinates on detector in mm.

    Notes
    -----
    For zero tilt and infinite curvature, this reduces to the simple
    ray-tracing formula :math:`(k_y d/k_x, k_z d/k_x)`.

    1. **Ray-plane intersection** --
       Compute intersection parameter :math:`t` for each
       wavevector with the (possibly tilted) detector plane.
    2. **Tilt correction** --
       Rotate coordinates when tilt angle is non-zero.
    3. **Curvature correction** --
       Map flat coordinates onto cylindrical surface when
       curvature radius is finite.
    4. **Apply offsets** --
       Shift by centre offsets.

    See Also
    --------
    DetectorGeometry : Shared detector carrier for projection and rendering.
    """
    distance: float = geometry.distance
    tilt_rad: Float[Array, ""] = jnp.deg2rad(geometry.tilt_angle)
    curvature: float = geometry.curvature_radius
    offset_h: float = geometry.center_offset_h
    offset_v: float = geometry.center_offset_v
    kx: Float[Array, "N"] = k_out[:, 0]
    ky: Float[Array, "N"] = k_out[:, 1]
    kz: Float[Array, "N"] = k_out[:, 2]
    cos_tilt: Float[Array, ""] = jnp.cos(tilt_rad)
    sin_tilt: Float[Array, ""] = jnp.sin(tilt_rad)
    denom: Float[Array, "N"] = kx * cos_tilt + kz * sin_tilt + 1e-10
    t_intersect: Float[Array, "N"] = distance / denom
    x_int: Float[Array, "N"] = kx * t_intersect
    y_int: Float[Array, "N"] = ky * t_intersect
    z_int: Float[Array, "N"] = kz * t_intersect
    detector_h: Float[Array, "N"] = y_int
    detector_v: Float[Array, "N"] = -x_int * sin_tilt + z_int * cos_tilt
    is_curved: Bool[Array, ""] = jnp.isfinite(curvature)

    def _apply_curvature(
        coords: tuple[Float[Array, "N"], Float[Array, "N"]],
    ) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
        """Apply cylindrical curvature correction."""
        h, v = coords
        theta: Float[Array, "N"] = h / curvature
        h_curved: Float[Array, "N"] = curvature * jnp.sin(theta)
        return h_curved, v

    def _no_curvature(
        coords: tuple[Float[Array, "N"], Float[Array, "N"]],
    ) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
        """No curvature correction for flat detector."""
        return coords

    detector_h, detector_v = jax.lax.cond(
        is_curved,
        _apply_curvature,
        _no_curvature,
        (detector_h, detector_v),
    )
    detector_h: Float[Array, "N"] = detector_h - offset_h
    detector_v: Float[Array, "N"] = detector_v - offset_v
    detector_coords: Float[Array, "N 2"] = jnp.stack(
        [detector_h, detector_v], axis=-1
    )
    return detector_coords


def _detector_grid(
    geometry: DetectorGeometry,
) -> tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """Build the dense pixel grid for a detector geometry."""
    height_px, width_px = detector_image_shape_px(geometry)
    x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_grid, y_grid


def _detector_points_to_pixels(
    detector_points: Float[Array, "N 2"],
    geometry: DetectorGeometry,
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Map detector millimetre coordinates to dense image pixels."""
    x_mm_per_px, y_mm_per_px = detector_pixel_size_mm(geometry)
    center_x_px, center_y_px = detector_beam_center_px(geometry)
    x_pixels: Float[Array, "N"] = detector_points[:, 0] / x_mm_per_px
    y_pixels: Float[Array, "N"] = detector_points[:, 1] / y_mm_per_px
    return x_pixels + center_x_px, y_pixels + center_y_px


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, "3"],
    gs: Float[Array, "M 3"],
    z_sign: scalar_float = 1.0,
    tolerance_inv_ang: scalar_float | None = None,
) -> Tuple[Int[Array, "N"], Float[Array, "N 3"]]:
    r"""Find kinematically allowed reflections.

    Computes k_out = k_in + G for all reciprocal lattice vectors G, then
    filters based on elastic scattering condition
    :math:`|k_out| ≈ |k_in|` and z-direction constraint. Returns fixed-size
    arrays for JIT compatibility, with -1 marking invalid entries.

    :see: :class:`~.test_simulator.TestFindKinematicReflections`

    Parameters
    ----------
    k_in : Float[Array, "3"]
        Incident wavevector.
    gs : Float[Array, "M 3"]
        Array of reciprocal lattice vectors.
    z_sign : scalar_float, optional
        If +1, keep reflections with positive z in k_out.
        If -1, keep reflections with negative z.
        Default: 1.0
    tolerance_inv_ang : scalar_float | None, optional
        Absolute Ewald-shell half-thickness in inverse Ångstroms for the
        elastic condition :math:`||k_{out}| - |k_{in}|| <` tolerance_inv_ang.
        Same name and meaning as in
        :func:`~rheedium.simul.ewald_allowed_reflections`. If None
        (default), derived from beam parameters as :math:`3 \sigma_{shell}`
        via :func:`~rheedium.simul.compute_shell_sigma` with
        ΔE/E = 1e-4 and 1 mrad beam divergence. At 20 kV
        (k = 73.2 1/Å) this admits :math:`|\Delta k| \lesssim 0.25` 1/Å.

    Returns
    -------
    allowed_indices : Int[Array, "M"]
        Indices of allowed reflections in gs array. Invalid entries are -1.
        Use `allowed_indices >= 0` to filter valid results.
    k_out : Float[Array, "M 3"]
        Output wavevectors for allowed reflections. Padded entries
        (`allowed_indices == -1`) are exactly zero.

    Notes
    -----
    Returns fixed-size arrays for JIT compatibility. Filter results using:
        valid_mask = allowed_indices >= 0
        valid_indices = allowed_indices[valid_mask]
        valid_k_out = k_out[valid_mask]

    1. **Outgoing wavevectors** --
       :math:`k_{out} = k_{in} + G` for all G vectors.
    2. **Elastic condition** --
       Filter by :math:`||k_{out}| - |k_{in}|| <` tolerance_inv_ang
       (absolute, 1/Å).
    3. **z-direction filter** --
       Keep reflections with correct z-sign.
    4. **Fixed-size output** --
       Use :func:`jnp.where` with fill_value for JIT
       compatibility; padded k_out rows are zeroed at the source.

    See Also
    --------
    incident_wavevector : Calculate incident wavevector from angles.
    generate_reciprocal_points : Generate reciprocal lattice vectors.
    """
    k_out_all: Float[Array, "M 3"] = k_in + gs
    k_in_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    if tolerance_inv_ang is None:
        tol_abs: Float[Array, ""] = 3.0 * compute_shell_sigma(
            k_magnitude=k_in_mag,
            energy_spread_frac=1e-4,
            beam_divergence_rad=1e-3,
        )
    else:
        tol_abs = jnp.asarray(tolerance_inv_ang, dtype=jnp.float64)
    k_out_mags: Float[Array, "M"] = jnp.linalg.norm(k_out_all, axis=1)
    elastic_condition: Bool[Array, "M"] = (
        jnp.abs(k_out_mags - k_in_mag) < tol_abs
    )
    z_condition: Bool[Array, "M"] = k_out_all[:, 2] * z_sign > 0
    allowed: Bool[Array, "M"] = elastic_condition & z_condition
    allowed_indices: Int[Array, "M"] = jnp.where(
        allowed, size=gs.shape[0], fill_value=-1
    )[0]
    safe_indices: Int[Array, "M"] = jnp.maximum(allowed_indices, 0)
    valid: Bool[Array, "M"] = allowed_indices >= 0
    k_out: Float[Array, "M 3"] = jnp.where(
        valid[:, None], k_out_all[safe_indices], 0.0
    )
    return allowed_indices, k_out


@jaxtyped(typechecker=beartype)
def compute_kinematic_intensities_with_ctrs(  # noqa: PLR0913, PLR0915
    crystal: CrystalStructure,
    g_allowed: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    k_out: Float[Array, "N 3"],
    hkl_indices: Int[Array, "N 3"] | None = None,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    detector_acceptance_inv_ang: scalar_float = 0.01,  # noqa: ARG001
    surface_fraction: scalar_float = 0.3,  # noqa: ARG001
    surface_config: SurfaceConfig | None = None,
    ctr_mixing_mode: str = "rod",
    ctr_weight: scalar_float = 1.0,
    hk_tolerance: scalar_float = 0.1,
    layer_attenuation: scalar_float = 0.01,
    parameterization: str = "lobato",
) -> Float[Array, "N"]:
    r"""Calculate kinematic diffraction intensities with CTR truncation.

    For each reflection, computes the single-cell structure factor by
    summing atomic contributions (form factor × phase factor). The phase
    is computed as G·r where G vectors already include the 2π factor from
    reciprocal lattice generation. For reflections whose (h, k) are
    near-integer, the intensity **is** the unified truncation-rod model
    :math:`|F_{cell}(q)|^2 \times T(l) \times e^{-q_z^2 \sigma^2}` —
    one model, with no additive kinematic + CTR double counting.

    :see: :class:`~.test_simulator.TestComputeKinematicIntensitiesExtended`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure containing atomic positions and types.
    g_allowed : Float[Array, "N 3"]
        Allowed reciprocal lattice vectors.
    k_in : Float[Array, "3"]
        Incident wavevector.
    k_out : Float[Array, "N 3"]
        Output wavevectors.
    hkl_indices : Int[Array, "N 3"] | None, optional
        Miller indices corresponding to each G vector. If provided, these
        are used directly for CTR gating. If None, indices are recovered
        from g_allowed using the reciprocal lattice basis.
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors.
        Default: 300.0
    surface_roughness : scalar_float, optional
        RMS surface roughness in angstroms.
        Default: 0.0
    detector_acceptance_inv_ang : scalar_float, optional
        Gaussian σ of the detector q_z acceptance window in 1/Å (this
        parameter was never an angle). Retained for API stability; the
        unified rod model evaluates the truncation factor at the
        reflection's own l, so this window is not applied here.
        Default: 0.01
    surface_fraction : scalar_float, optional
        Legacy height-fraction knob retained for API compatibility. It is
        only honored when an explicit height-based ``surface_config`` is
        constructed by the caller; with ``surface_config=None`` no surface
        enhancement is applied and this value is ignored.
        Default: 0.3
    surface_config : SurfaceConfig | None, optional
        Configuration for surface atom identification. Supports methods
        "none", "height", "coordination", "layers", "explicit".
        If None, defaults to ``SurfaceConfig(method="none")`` — no atom is
        treated as a surface atom, because the bulk basis handled here is
        implicitly repeated by the CTR factor and therefore has no surface
        atoms. Slab-based callers that want thermal surface enhancement
        must opt in explicitly.
    ctr_mixing_mode : str, optional
        CTR handling mode:
        - "rod" (default): unified truncation-rod model
          :math:`|F|^2 T(l) e^{-q_z^2 \sigma^2}` for near-integer (h, k)
        - "none": bare :math:`|F|^2` (bulk-like, no CTR factor)
        - "coherent"/"incoherent": deprecated aliases of "rod" (emit a
          DeprecationWarning); the old additive kinematic + CTR paths
          double-counted the rod and were removed.
        Default: "rod"
    ctr_weight : scalar_float, optional
        Blend factor in [0, 1] between the bare kinematic model (0.0)
        and the full truncation-rod model (1.0): the applied factor is
        ``(1 - w) + w * T(l) * exp(-q_z^2 sigma^2)``.
        Default: 1.0
    hk_tolerance : scalar_float, optional
        Tolerance for validating near-integer h,k indices. CTR is only
        applied when :math:`|h - round(h)| < tolerance` and same for k.
        Default: 0.1
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon` of the
        truncation factor. Default: 0.01
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    intensities : Float[Array, "N"]
        Diffraction intensities for each allowed reflection.

    Notes
    -----
    Surface atom identification supports multiple strategies:
    - "height": Top fraction by z-coordinate (simple, fast)
    - "coordination": Atoms with fewer neighbors (better for steps)
    - "layers": Topmost N complete layers (good for flat surfaces)
    - "explicit": User-provided mask (full control)

    1. **Extract atomic data** --
       Positions, atomic numbers, and Miller indices
       from crystal and G vectors.
    2. **Identify surface atoms** --
       Apply configured method (height, coordination,
       layers, or explicit mask).
    3. **Per-reflection intensity** --
       For each G: compute :math:`q = k_{out} - k_{in}`,
       evaluate the structure factor with form factors and
       Debye-Waller, gate the rod factor on near-integer (h, k).
    4. **Apply the unified rod model** --
       Multiply :math:`|F|^2` by the truncation factor
       :math:`T(l)` at the reflection's continuous l and by the
       roughness intensity factor; "none" skips both.

    See Also
    --------
    atomic_scattering_factor : Compute form factors with Debye-Waller.
    ctr_truncation_intensity : Semi-infinite truncation-rod factor.
    roughness_damping : Roughness amplitude factor (squared here).
    identify_surface_atoms : Identify surface atoms from positions.
    """
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    if ctr_mixing_mode in {"coherent", "incoherent"}:
        warnings.warn(
            "ctr_mixing_mode='coherent'/'incoherent' are deprecated "
            "aliases of 'rod': the additive kinematic + CTR paths "
            "double-counted the truncation rod and were removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        ctr_mixing_mode = "rod"
    if ctr_mixing_mode not in {"rod", "none"}:
        raise ValueError(
            "ctr_mixing_mode must be one of 'rod', 'none' (or the "
            "deprecated aliases 'coherent'/'incoherent')."
        )
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    occupancies: Float[Array, "M"] = (
        jnp.ones(atom_positions.shape[0], dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *crystal.cell_lengths,
        *crystal.cell_angles,
        in_degrees=True,
    )
    inv_recip: Float[Array, "3 3"] = jnp.linalg.inv(recip_vecs)
    # Continuous (h, k, l) of each reflection's momentum transfer in the
    # reciprocal basis; the l column feeds the truncation factor.
    hkl_q: Float[Array, "N 3"] = jnp.matmul(k_out - k_in, inv_recip)
    # Miller indices used for near-integer (h, k) gating; prefer explicit
    hkl_all: Float[Array, "N 3"]
    if hkl_indices is not None:
        hkl_all = jnp.asarray(hkl_indices, dtype=jnp.float64)
    else:
        hkl_all = jnp.matmul(g_allowed, inv_recip)

    # Default: no surface enhancement. The bulk basis here is repeated by
    # the CTR factor, so no basis atom is a surface atom; slab-based paths
    # that genuinely want enhancement must opt in via surface_config.
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="none")
    )
    is_surface_atom: Bool[Array, "M"] = identify_surface_atoms(
        atom_positions, config
    )

    def _calculate_reflection_intensity(
        idx: Int[Array, ""],
    ) -> Float[Array, ""]:
        g_vec: Float[Array, "3"] = g_allowed[idx]
        k_out_vec: Float[Array, "3"] = k_out[idx]
        q_vector: Float[Array, "3"] = k_out_vec - k_in

        def _atomic_contribution(
            atom_idx: Int[Array, ""],
        ) -> Complex[Array, ""]:
            atomic_num: scalar_int = atomic_numbers[atom_idx]
            atom_pos: Float[Array, "3"] = atom_positions[atom_idx]
            is_surface: bool = is_surface_atom[atom_idx]
            form_factor: scalar_float = occupancies[atom_idx] * jnp.squeeze(
                atomic_scattering_factor(
                    atomic_number=atomic_num,
                    q_vector=q_vector,
                    temperature=temperature,
                    is_surface=is_surface,
                    parameterization=parameterization,
                )
            )
            phase: scalar_float = jnp.dot(g_vec, atom_pos)
            contribution: Complex[Array, ""] = form_factor * jnp.exp(
                1j * phase
            )
            return contribution

        n_atoms: Int[Array, ""] = atom_positions.shape[0]
        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contributions: Complex[Array, "M"] = jax.vmap(_atomic_contribution)(
            atom_indices
        )
        structure_factor: Complex[Array, ""] = jnp.sum(contributions)

        # Validate h,k are near-integer before applying the rod factor
        h_val: Float[Array, ""] = hkl_all[idx, 0]
        k_val: Float[Array, ""] = hkl_all[idx, 1]
        h_deviation: Float[Array, ""] = jnp.abs(h_val - jnp.round(h_val))
        k_deviation: Float[Array, ""] = jnp.abs(k_val - jnp.round(k_val))
        is_near_integer: Bool[Array, ""] = (h_deviation < hk_tolerance) & (
            k_deviation < hk_tolerance
        )

        kinematic_intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2

        if ctr_mixing_mode == "none":
            # Bulk-like: bare |F|^2, no truncation-rod factor
            total_intensity: Float[Array, ""] = kinematic_intensity
        else:
            # Unified truncation-rod model (ONE model, no double count):
            # |F(q)|^2 * T(l) * exp(-q_z^2 sigma^2) at the reflection's
            # own continuous l along the (h, k) rod.
            l_value: Float[Array, ""] = hkl_q[idx, 2]
            truncation: Float[Array, ""] = ctr_truncation_intensity(
                l_values=l_value,
                layer_attenuation=layer_attenuation,
            )
            roughness_intensity: Float[Array, ""] = (
                roughness_damping(
                    q_z=q_vector[2],
                    sigma_height=surface_roughness,
                )
                ** 2
            )
            rod_factor: Float[Array, ""] = truncation * roughness_intensity
            blended_factor: Float[Array, ""] = (
                1.0 - ctr_weight
            ) + ctr_weight * rod_factor
            applied_factor: Float[Array, ""] = jnp.where(
                is_near_integer, blended_factor, 1.0
            )
            total_intensity: Float[Array, ""] = (
                kinematic_intensity * applied_factor
            )

        return total_intensity

    n_reflections: Int[Array, ""] = g_allowed.shape[0]
    reflection_indices: Int[Array, "N"] = jnp.arange(n_reflections)
    intensities: Float[Array, "N"] = jax.vmap(_calculate_reflection_intensity)(
        reflection_indices
    )
    return intensities


@jaxtyped(typechecker=beartype)
def ewald_simulator(  # noqa: PLR0913, PLR0915
    crystal: CrystalStructure,
    energy_kev: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    layer_attenuation: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 1.0,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    ctr_regularization: scalar_float | None = None,
) -> RHEEDPattern:
    r"""Simulate RHEED pattern using exact Ewald sphere-CTR intersection.

    This is the physically correct approach for surface diffraction: for each
    (h,k) crystal truncation rod, solve for the l value where the rod
    intersects the Ewald sphere, then compute the diffracted intensity at
    that point.

    :see: :class:`~.test_simulator.TestEwaldSimulator`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    energy_kev : scalar_num, optional
        Electron beam energy in kiloelectron volts. Default: 20.0
    theta_deg : scalar_num, optional
        Grazing angle of incidence in degrees (angle from surface).
        Default: 2.0
    phi_deg : scalar_num, optional
        Azimuthal angle in degrees (in-plane rotation). Default: 0.0
    hmax : scalar_int, optional
        Maximum h Miller index for CTR grid. Default: 5
    kmax : scalar_int, optional
        Maximum k Miller index for CTR grid. Default: 5
    detector_distance : scalar_float, optional
        Distance from sample to detector plane in mm. Default: 1000.0
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors. Default: 300.0
    surface_roughness : scalar_float, optional
        RMS surface roughness in Ångstroms for CTR damping. Default: 0.0
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon \geq 0` in the
        semi-infinite truncation factor
        :math:`1/(1 - 2 e^{-\epsilon}\cos 2\pi l + e^{-2\epsilon})`,
        absorbing the finite penetration depth. Default: 0.01
    ctr_power : scalar_float, optional
        Non-physical diagnostic exponent applied to the CTR truncation
        term. The physical model is ``1.0`` (default); set to ``0.0`` to
        disable CTR weighting while leaving the geometry unchanged.
    roughness_power : scalar_float, optional
        Non-physical diagnostic exponent applied to the roughness
        intensity factor :math:`e^{-\sigma^2 q_z^2}`. The physical model
        is ``1.0`` (default); set to ``0.0`` to disable roughness damping
        while leaving the geometry unchanged.
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.
    ctr_regularization : scalar_float | None, optional
        Deprecated alias of the ε parameterization. The legacy additive
        constant of ``1/(sin^2(pi l) + reg)`` is converted so the
        Bragg-peak cap matches: ``eps = 2 asinh(sqrt(reg)/2)``. Emits a
        DeprecationWarning when not None. Default: None
    surface_config : SurfaceConfig | None, optional
        Configuration for surface atom identification. Default: None,
        which maps to ``SurfaceConfig(method="none")`` — no surface
        thermal enhancement. This simulator treats the crystal as a
        semi-infinite repeated bulk (the :math:`1/\sin^2(\pi l)` CTR
        factor); in that picture no basis atom is a surface atom, so
        enhanced Debye-Waller factors would be unphysical. Slab-based
        workflows must opt in with an explicit config.

    Returns
    -------
    pattern : RHEEDPattern
        RHEED pattern with detector positions and intensities.

    Notes
    -----
    The algorithm solves the Ewald sphere constraint for each (h,k) rod:

    .. math::

        |k_{in} + h a^* + k b^* + l c^*|^2 = |k_{in}|^2

    This gives a quadratic equation in l with solutions:

    .. math::

        l = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}

    where:
    - :math:`a = |c^*|^2`
    - :math:`b = 2 (k_{in} + h a^* + k b^*) \cdot c^*`
    - :math:`c = |k_{in} + h a^* + k b^*|^2 - |k_{in}|^2`

    Both solutions with :math:`k_{out,z} > 0` (upward scattering) are kept.

    Intensity includes:
    - Structure factor :math:`|F(G)|^2` with selected atomic form factors
    - Debye-Waller thermal damping
    - CTR intensity modulation :math:`1/\sin^2(\pi l)`
    - Surface roughness damping :math:`\exp(-\sigma^2 q_z^2)`

    1. **Beam parameters** --
       Wavelength and incident wavevector from voltage and
       angles.
    2. **Reciprocal basis** --
       Compute :math:`a^*, b^*, c^*` from cell parameters.
    3. **Rod-sphere intersection** --
       For each (h, k) rod, solve quadratic for l where
       the rod intersects the Ewald sphere.
    4. **Structure factors** --
       Lobato form factors by default with Debye-Waller and surface
       enhancement at each intersection.
    5. **CTR modulation** --
       :math:`1/\\sin^2(\\pi l)` with roughness damping.
    6. **Assemble pattern** --
       Project onto detector and normalize intensities.

    See Also
    --------
    find_ctr_ewald_intersection : Solve rod-sphere intersection geometry.
    kinematic_simulator : Deprecated bulk-like simulator.
    """
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    layer_attenuation = _resolve_layer_attenuation(
        layer_attenuation, ctr_regularization
    )
    energy_kev: Float[Array, ""] = jnp.asarray(energy_kev)
    theta_deg: Float[Array, ""] = jnp.asarray(theta_deg)
    phi_deg: Float[Array, ""] = jnp.asarray(phi_deg)
    # Keep hmax/kmax as Python ints for static array sizing in JIT
    hmax_int: int = int(hmax)
    kmax_int: int = int(kmax)

    # Compute incident wavevector
    lam_ang: Float[Array, ""] = wavelength_ang(energy_kev)
    k_in: Float[Array, "3"] = incident_wavevector(lam_ang, theta_deg, phi_deg)

    # Get reciprocal lattice vectors
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *crystal.cell_lengths, *crystal.cell_angles, in_degrees=True
    )
    recip_a: Float[Array, "3"] = recip_vecs[0]
    recip_b: Float[Array, "3"] = recip_vecs[1]
    recip_c: Float[Array, "3"] = recip_vecs[2]

    # Build (h,k) grid for CTRs
    hh: Int[Array, "n_h n_k"]
    kk: Int[Array, "n_h n_k"]
    hh, kk = jnp.meshgrid(
        jnp.arange(-hmax_int, hmax_int + 1, dtype=jnp.int32),
        jnp.arange(-kmax_int, kmax_int + 1, dtype=jnp.int32),
        indexing="ij",
    )
    h_flat: Int[Array, "N"] = hh.ravel()
    k_flat: Int[Array, "N"] = kk.ravel()
    n_rods: int = h_flat.shape[0]

    # Find Ewald-CTR intersection for each rod using vmap
    def _find_intersection(
        idx: Int[Array, ""],
    ) -> Tuple[Float[Array, "2"], Float[Array, "2 3"], Float[Array, "2"]]:
        h_val: Int[Array, ""] = h_flat[idx]
        k_val: Int[Array, ""] = k_flat[idx]
        return find_ctr_ewald_intersection(
            h=h_val,
            k=k_val,
            k_in=k_in,
            recip_a=recip_a,
            recip_b=recip_b,
            recip_c=recip_c,
        )

    rod_indices: Int[Array, "N"] = jnp.arange(n_rods, dtype=jnp.int32)
    l_all: Float[Array, "N 2"]
    k_out_all: Float[Array, "N 2 3"]
    valid_all: Float[Array, "N 2"]
    l_all, k_out_all, valid_all = jax.vmap(_find_intersection)(rod_indices)
    l_all = l_all.reshape(-1)
    k_out_all = k_out_all.reshape(-1, 3)
    valid_all = valid_all.reshape(-1)
    h_all: Int[Array, "N2"] = jnp.repeat(h_flat, 2)
    k_all: Int[Array, "N2"] = jnp.repeat(k_flat, 2)
    hk_linear_indices: Int[Array, "N2"] = jnp.repeat(rod_indices, 2)
    n_candidates: int = l_all.shape[0]

    # Filter to valid intersections
    valid_mask: Bool[Array, "N2"] = valid_all > _VALID_THRESHOLD
    valid_indices: Int[Array, "K"] = jnp.where(
        valid_mask, size=n_candidates, fill_value=-1
    )[0]
    safe_indices: Int[Array, "K"] = jnp.maximum(valid_indices, 0)

    l_valid: Float[Array, "K"] = l_all[safe_indices]
    k_out: Float[Array, "K 3"] = k_out_all[safe_indices]
    h_valid: Int[Array, "K"] = h_all[safe_indices]
    k_valid: Int[Array, "K"] = k_all[safe_indices]

    # Compute G vectors at intersection points
    g_vectors: Float[Array, "K 3"] = (
        h_valid[:, None] * recip_a
        + k_valid[:, None] * recip_b
        + l_valid[:, None] * recip_c
    )

    # Get atomic data
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    occupancies: Float[Array, "M"] = (
        jnp.ones(atom_positions.shape[0], dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    n_atoms: int = atom_positions.shape[0]

    # Surface atom identification. Default: no surface enhancement — the
    # bulk unit cell is repeated by the 1/sin^2(pi*l) CTR factor, so no
    # basis atom is a surface atom; slab models must opt in explicitly.
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="none")
    )
    is_surface_atom: Bool[Array, "M"] = identify_surface_atoms(
        atom_positions, config
    )

    # Compute intensities for each valid reflection
    def _compute_intensity(
        refl_idx: Int[Array, ""],
    ) -> Float[Array, ""]:
        g_vec: Float[Array, "3"] = g_vectors[refl_idx]
        k_out_vec: Float[Array, "3"] = k_out[refl_idx]
        l_val: Float[Array, ""] = l_valid[refl_idx]
        is_valid: Bool[Array, ""] = valid_indices[refl_idx] >= 0

        # Momentum transfer
        q_vector: Float[Array, "3"] = k_out_vec - k_in
        q_z: Float[Array, ""] = q_vector[2]

        # Structure factor with form factors and Debye-Waller
        def _atomic_contrib(atom_idx: Int[Array, ""]) -> Complex[Array, ""]:
            z_num: Int[Array, ""] = atomic_numbers[atom_idx]
            pos: Float[Array, "3"] = atom_positions[atom_idx]
            is_surface: Bool[Array, ""] = is_surface_atom[atom_idx]
            ff: Float[Array, ""] = occupancies[atom_idx] * jnp.squeeze(
                atomic_scattering_factor(
                    atomic_number=z_num,
                    q_vector=q_vector,
                    temperature=temperature,
                    is_surface=is_surface,
                    parameterization=parameterization,
                )
            )
            phase: Float[Array, ""] = jnp.dot(g_vec, pos)
            return ff * jnp.exp(1j * phase)

        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contribs: Complex[Array, "M"] = jax.vmap(_atomic_contrib)(atom_indices)
        structure_factor: Complex[Array, ""] = jnp.sum(contribs)
        sf_intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2

        intensity: Float[Array, ""] = _compose_ewald_intensity(
            sf_intensity=sf_intensity,
            l_value=l_val,
            q_z=q_z,
            surface_roughness=surface_roughness,
            layer_attenuation=layer_attenuation,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
        )[0]
        return jnp.where(is_valid, intensity, 0.0)

    n_valid: int = valid_indices.shape[0]
    refl_indices: Int[Array, "K"] = jnp.arange(n_valid, dtype=jnp.int32)
    intensities: Float[Array, "K"] = jax.vmap(_compute_intensity)(refl_indices)

    # Normalize intensities
    max_intensity: Float[Array, ""] = jnp.maximum(jnp.max(intensities), 1e-10)
    intensities: Float[Array, "K"] = intensities / max_intensity

    # Project onto detector
    detector_points: Float[Array, "K 2"] = project_on_detector_geometry(
        k_out,
        DetectorGeometry(distance=detector_distance),
    )

    # Create pattern
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=jnp.where(
            valid_indices >= 0, hk_linear_indices[safe_indices], -1
        ),
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern


@jaxtyped(typechecker=beartype)
def _ewald_amplitude_pattern(  # noqa: PLR0913, PLR0915
    crystal: CrystalStructure,
    energy_kev: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    layer_attenuation: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 1.0,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
) -> Tuple[RHEEDPattern, Complex[Array, "K"]]:
    """Return sparse Ewald pattern plus complex per-reflection amplitudes."""
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    energy_kev_arr: Float[Array, ""] = jnp.asarray(energy_kev)
    theta_deg_arr: Float[Array, ""] = jnp.asarray(theta_deg)
    phi_deg_arr: Float[Array, ""] = jnp.asarray(phi_deg)
    hmax_int: int = int(hmax)
    kmax_int: int = int(kmax)

    lam_ang: Float[Array, ""] = wavelength_ang(energy_kev_arr)
    k_in: Float[Array, "3"] = incident_wavevector(
        lam_ang,
        theta_deg_arr,
        phi_deg_arr,
    )
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *crystal.cell_lengths,
        *crystal.cell_angles,
        in_degrees=True,
    )
    recip_a: Float[Array, "3"] = recip_vecs[0]
    recip_b: Float[Array, "3"] = recip_vecs[1]
    recip_c: Float[Array, "3"] = recip_vecs[2]

    hh: Int[Array, "n_h n_k"]
    kk: Int[Array, "n_h n_k"]
    hh, kk = jnp.meshgrid(
        jnp.arange(-hmax_int, hmax_int + 1, dtype=jnp.int32),
        jnp.arange(-kmax_int, kmax_int + 1, dtype=jnp.int32),
        indexing="ij",
    )
    h_flat: Int[Array, "N"] = hh.ravel()
    k_flat: Int[Array, "N"] = kk.ravel()
    n_rods: int = h_flat.shape[0]

    def _find_intersection(
        idx: Int[Array, ""],
    ) -> Tuple[Float[Array, "2"], Float[Array, "2 3"], Float[Array, "2"]]:
        return find_ctr_ewald_intersection(
            h=h_flat[idx],
            k=k_flat[idx],
            k_in=k_in,
            recip_a=recip_a,
            recip_b=recip_b,
            recip_c=recip_c,
        )

    rod_indices: Int[Array, "N"] = jnp.arange(n_rods, dtype=jnp.int32)
    l_all: Float[Array, "N 2"]
    k_out_all: Float[Array, "N 2 3"]
    valid_all: Float[Array, "N 2"]
    l_all, k_out_all, valid_all = jax.vmap(_find_intersection)(rod_indices)
    l_all = l_all.reshape(-1)
    k_out_all = k_out_all.reshape(-1, 3)
    valid_all = valid_all.reshape(-1)
    h_all: Int[Array, "N2"] = jnp.repeat(h_flat, 2)
    k_all: Int[Array, "N2"] = jnp.repeat(k_flat, 2)
    hk_linear_indices: Int[Array, "N2"] = jnp.repeat(rod_indices, 2)
    n_candidates: int = l_all.shape[0]

    valid_mask: Bool[Array, "N2"] = valid_all > _VALID_THRESHOLD
    valid_indices: Int[Array, "K"] = jnp.where(
        valid_mask,
        size=n_candidates,
        fill_value=-1,
    )[0]
    safe_indices: Int[Array, "K"] = jnp.maximum(valid_indices, 0)

    l_valid: Float[Array, "K"] = l_all[safe_indices]
    k_out: Float[Array, "K 3"] = k_out_all[safe_indices]
    h_valid: Int[Array, "K"] = h_all[safe_indices]
    k_valid: Int[Array, "K"] = k_all[safe_indices]
    g_vectors: Float[Array, "K 3"] = (
        h_valid[:, None] * recip_a
        + k_valid[:, None] * recip_b
        + l_valid[:, None] * recip_c
    )

    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    occupancies: Float[Array, "M"] = (
        jnp.ones(atom_positions.shape[0], dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    n_atoms: int = atom_positions.shape[0]
    # Default: no surface enhancement (bulk basis repeated by the CTR
    # factor has no surface atoms); slab models must opt in explicitly.
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="none")
    )
    is_surface_atom: Bool[Array, "M"] = identify_surface_atoms(
        atom_positions,
        config,
    )

    def _compute_amplitude(
        refl_idx: Int[Array, ""],
    ) -> Complex[Array, ""]:
        g_vec: Float[Array, "3"] = g_vectors[refl_idx]
        k_out_vec: Float[Array, "3"] = k_out[refl_idx]
        l_val: Float[Array, ""] = l_valid[refl_idx]
        is_valid: Bool[Array, ""] = valid_indices[refl_idx] >= 0
        q_vector: Float[Array, "3"] = k_out_vec - k_in
        q_z: Float[Array, ""] = q_vector[2]

        def _atomic_contrib(atom_idx: Int[Array, ""]) -> Complex[Array, ""]:
            z_num: Int[Array, ""] = atomic_numbers[atom_idx]
            pos: Float[Array, "3"] = atom_positions[atom_idx]
            is_surface: Bool[Array, ""] = is_surface_atom[atom_idx]
            form_factor: Float[Array, ""] = occupancies[
                atom_idx
            ] * jnp.squeeze(
                atomic_scattering_factor(
                    atomic_number=z_num,
                    q_vector=q_vector,
                    temperature=temperature,
                    is_surface=is_surface,
                    parameterization=parameterization,
                )
            )
            phase: Float[Array, ""] = jnp.dot(g_vec, pos)
            return form_factor * jnp.exp(1j * phase)

        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contributions: Complex[Array, "M"] = jax.vmap(_atomic_contrib)(
            atom_indices
        )
        structure_factor: Complex[Array, ""] = jnp.sum(contributions)
        _, ctr_modulation, roughness_intensity = _compose_ewald_intensity(
            sf_intensity=jnp.abs(structure_factor) ** 2,
            l_value=l_val,
            q_z=q_z,
            surface_roughness=surface_roughness,
            layer_attenuation=layer_attenuation,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
        )
        # Preserve the complex phase of the truncation amplitude while
        # letting the diagnostic powers act on the magnitudes, so that
        # |amplitude|^2 matches _compose_ewald_intensity exactly.
        truncation_amp: Complex[Array, ""] = ctr_truncation_amplitude(
            l_values=l_val,
            layer_attenuation=layer_attenuation,
        )
        truncation_phase: Complex[Array, ""] = truncation_amp / jnp.sqrt(
            ctr_modulation
        )
        amplitude_scale: Float[Array, ""] = ctr_modulation ** (
            0.5 * ctr_power
        ) * roughness_intensity ** (0.5 * roughness_power)
        amplitude: Complex[Array, ""] = (
            structure_factor * truncation_phase * amplitude_scale
        )
        return jnp.where(is_valid, amplitude, 0.0 + 0.0j)

    n_valid: int = valid_indices.shape[0]
    refl_indices: Int[Array, "K"] = jnp.arange(n_valid, dtype=jnp.int32)
    raw_amplitudes: Complex[Array, "K"] = jax.vmap(_compute_amplitude)(
        refl_indices
    )
    raw_intensities: Float[Array, "K"] = jnp.abs(raw_amplitudes) ** 2
    max_intensity: Float[Array, ""] = jnp.maximum(
        jnp.max(raw_intensities),
        1e-10,
    )
    amplitudes: Complex[Array, "K"] = raw_amplitudes / jnp.sqrt(max_intensity)
    intensities: Float[Array, "K"] = jnp.abs(amplitudes) ** 2
    detector_points: Float[Array, "K 2"] = project_on_detector_geometry(
        k_out,
        DetectorGeometry(distance=detector_distance),
    )
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=jnp.where(
            valid_indices >= 0,
            hk_linear_indices[safe_indices],
            -1,
        ),
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern, amplitudes


@jaxtyped(typechecker=beartype)
def render_pattern_to_image(
    pattern: RHEEDPattern,
    geometry: DetectorGeometry,
    spot_sigma_px: scalar_float,
) -> Float[Array, "H W"]:
    """Rasterize a sparse RHEEDPattern onto a dense detector image.

    Parameters
    ----------
    pattern : RHEEDPattern
        Sparse detector pattern with positions in millimetres.
    geometry : DetectorGeometry
        Detector geometry carrying output shape, pixel calibration, and beam
        centre.
    spot_sigma_px : scalar_float
        Gaussian spot width in detector pixels.

    Returns
    -------
    image : Float[Array, "H W"]
        Rasterized image normalized to unit maximum.

    Notes
    -----
    This is the dense-image bridge between sparse reciprocal-space
    simulations and experiment-like detector rendering. Each sparse hit is
    painted as a Gaussian on a calibrated pixel grid.
    """
    sigma_px: Float[Array, ""] = jnp.asarray(spot_sigma_px, dtype=jnp.float64)
    x_pixels: Float[Array, "N"]
    y_pixels: Float[Array, "N"]
    x_pixels, y_pixels = _detector_points_to_pixels(
        pattern.detector_points,
        geometry,
    )
    intensities: Float[Array, "N"] = jnp.asarray(
        pattern.intensities, dtype=jnp.float64
    )

    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = _detector_grid(geometry)

    def _render_one_spot(
        x0_px: Float[Array, ""],
        y0_px: Float[Array, ""],
        intensity: Float[Array, ""],
    ) -> Float[Array, "H W"]:
        return intensity * jnp.exp(
            -((x_grid - x0_px) ** 2 + (y_grid - y0_px) ** 2)
            / (2.0 * sigma_px**2)
        )

    image: Float[Array, "H W"] = jnp.sum(
        jax.vmap(_render_one_spot)(x_pixels, y_pixels, intensities),
        axis=0,
    )
    max_intensity: Float[Array, ""] = jnp.maximum(jnp.max(image), 1e-12)
    return image / max_intensity


@jaxtyped(typechecker=beartype)
def render_amplitude_to_field(
    pattern: RHEEDPattern,
    amplitudes: Complex[Array, "N"],
    geometry: DetectorGeometry,
    spot_sigma_px: scalar_float,
) -> Complex[Array, "H W"]:
    """Rasterize sparse complex amplitudes onto a dense detector field.

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    pattern : RHEEDPattern
        Sparse detector pattern carrying detector coordinates in millimetres.
    amplitudes : Complex[Array, "N"]
        Complex reflection amplitudes aligned with ``pattern.detector_points``.
    geometry : DetectorGeometry
        Detector geometry carrying output shape, pixel calibration, and beam
        centre.
    spot_sigma_px : scalar_float
        Gaussian intensity width in detector pixels.

    Returns
    -------
    field : Complex[Array, "H W"]
        Dense coherent detector amplitude field.

    Notes
    -----
    1. Convert detector millimetres to pixel coordinates.
    2. Deposit each sparse amplitude with a square-root Gaussian envelope.
    3. Leave normalization to the downstream intensity reducer.

    See Also
    --------
    render_pattern_to_image : Rasterize sparse intensities.
    """
    sigma_px: Float[Array, ""] = jnp.asarray(spot_sigma_px, dtype=jnp.float64)
    amplitudes_arr: Complex[Array, "N"] = jnp.asarray(
        amplitudes,
        dtype=jnp.complex128,
    )
    if amplitudes_arr.shape[0] != pattern.detector_points.shape[0]:
        raise ValueError("amplitudes length must match detector_points")

    x_pixels: Float[Array, "N"]
    y_pixels: Float[Array, "N"]
    x_pixels, y_pixels = _detector_points_to_pixels(
        pattern.detector_points,
        geometry,
    )
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = _detector_grid(geometry)

    def _render_one_amplitude(
        x0_px: Float[Array, ""],
        y0_px: Float[Array, ""],
        amplitude: Complex[Array, ""],
    ) -> Complex[Array, "H W"]:
        envelope: Float[Array, "H W"] = jnp.exp(
            -((x_grid - x0_px) ** 2 + (y_grid - y0_px) ** 2)
            / (4.0 * sigma_px**2)
        )
        return amplitude * envelope

    return jnp.sum(
        jax.vmap(_render_one_amplitude)(
            x_pixels,
            y_pixels,
            amplitudes_arr,
        ),
        axis=0,
    )


@jaxtyped(typechecker=beartype)
def kinematic_amplitude(  # noqa: PLR0913
    crystal: CrystalStructure,
    energy_kev: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    layer_attenuation: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 1.0,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    render_ctrs_as_streaks: bool = False,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    ctr_regularization: scalar_float | None = None,
) -> Complex[Array, "H W"]:
    """Render one coherent kinematic Ewald pattern as detector amplitude.

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
    energy_kev, theta_deg, phi_deg : scalar_num, optional
        Beam energy and incidence geometry.
    hmax, kmax : scalar_int, optional
        In-plane CTR grid bounds.
    detector_distance_mm : scalar_float, optional
        Sample-to-detector distance in millimetres.
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller damping.
    surface_roughness : scalar_float, optional
        RMS surface roughness in Ångstroms.
    layer_attenuation, ctr_power, roughness_power : scalar_float, optional
        CTR truncation and roughness controls with the same meaning and
        defaults as in :func:`ewald_simulator`.
    ctr_regularization : scalar_float | None, optional
        Deprecated alias of the ε parameterization; see
        :func:`ewald_simulator`. Default: None
    image_shape_px : Tuple[int, int], optional
        Detector field shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float], optional
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float], optional
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.
    spot_sigma_px : scalar_float, optional
        Gaussian intensity width in detector pixels.
    render_ctrs_as_streaks : bool, optional
        If True, render sparse amplitudes as elongated CTR streak amplitudes
        instead of compact spot amplitudes. Default: False.
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` or ``"kirkland"``.
    surface_config : SurfaceConfig | None, optional
        Surface atom identification configuration.

    Returns
    -------
    amplitude : Complex[Array, "H W"]
        Dense coherent detector amplitude field.

    Notes
    -----
    1. Compute sparse Ewald intersections with complex structure factors.
    2. Apply CTR and roughness factors at amplitude level.
    3. Rasterize the amplitudes onto the shared detector grid.

    See Also
    --------
    render_amplitude_to_field : Dense complex detector renderer.
    _ewald_amplitude_pattern : Sparse kinematic Ewald amplitude simulator.
    """
    layer_attenuation = _resolve_layer_attenuation(
        layer_attenuation, ctr_regularization
    )
    sparse_pattern: RHEEDPattern
    amplitudes: Complex[Array, "N"]
    sparse_pattern, amplitudes = _ewald_amplitude_pattern(
        crystal=crystal,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        hmax=hmax,
        kmax=kmax,
        detector_distance=detector_distance_mm,
        temperature=temperature,
        surface_roughness=surface_roughness,
        layer_attenuation=layer_attenuation,
        ctr_power=ctr_power,
        roughness_power=roughness_power,
        parameterization=parameterization,
        surface_config=surface_config,
    )
    geometry: DetectorGeometry = DetectorGeometry(
        distance=detector_distance_mm,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
    )
    if render_ctrs_as_streaks:
        return render_ctr_amplitude_to_field(
            pattern=sparse_pattern,
            amplitudes=amplitudes,
            geometry=geometry,
            spot_sigma_px=spot_sigma_px,
        )
    return render_amplitude_to_field(
        pattern=sparse_pattern,
        amplitudes=amplitudes,
        geometry=geometry,
        spot_sigma_px=spot_sigma_px,
    )


@jaxtyped(typechecker=beartype)
def _kinematic_finite_domain_amplitude(  # noqa: PLR0913
    crystal: CrystalStructure,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    domain_size_angstrom: scalar_float,
    domain_aspect_ratio: Tuple[float, float, float],
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    layer_attenuation: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    render_ctrs_as_streaks: bool,
    parameterization: str,
    surface_config: SurfaceConfig | None,
    energy_spread_frac: scalar_float,  # noqa: ARG001
    beam_divergence_rad: scalar_float,  # noqa: ARG001
) -> Complex[Array, "H W"]:
    """Render kinematic amplitudes with finite-domain spot broadening.

    The sparse amplitudes come from exact rod-sphere intersections, where
    the rod-based lateral overlap weight is identically 1, so the
    finite-domain physics enters through the lateral rod widths that
    broaden the rendered spots.
    """
    sparse_pattern: RHEEDPattern
    amplitudes: Complex[Array, "N"]
    sparse_pattern, amplitudes = _ewald_amplitude_pattern(
        crystal=crystal,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        hmax=hmax,
        kmax=kmax,
        detector_distance=detector_distance_mm,
        temperature=temperature,
        surface_roughness=surface_roughness,
        layer_attenuation=layer_attenuation,
        ctr_power=ctr_power,
        roughness_power=roughness_power,
        parameterization=parameterization,
        surface_config=surface_config,
    )
    aspect_ratio: Float[Array, "3"] = jnp.asarray(
        domain_aspect_ratio,
        dtype=jnp.float64,
    )
    domain_extent_ang: Float[Array, "3"] = (
        jnp.asarray(domain_size_angstrom, dtype=jnp.float64) * aspect_ratio
    )
    lam_ang: Float[Array, ""] = wavelength_ang(energy_kev)
    k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    rod_sigma: Float[Array, "3"] = extent_to_rod_sigma(domain_extent_ang)
    finite_amplitudes: Complex[Array, "N"] = amplitudes
    mean_pixel_size_mm: Float[Array, ""] = jnp.mean(
        jnp.asarray(pixel_size_mm, dtype=jnp.float64)
    )
    finite_sigma_px: Float[Array, ""] = (
        detector_distance_mm
        * jnp.mean(rod_sigma[:2])
        / jnp.maximum(k_magnitude * mean_pixel_size_mm, 1e-12)
    )
    effective_spot_sigma_px: Float[Array, ""] = jnp.sqrt(
        jnp.asarray(spot_sigma_px, dtype=jnp.float64) ** 2 + finite_sigma_px**2
    )
    geometry: DetectorGeometry = DetectorGeometry(
        distance=detector_distance_mm,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
    )
    if render_ctrs_as_streaks:
        return render_ctr_amplitude_to_field(
            pattern=sparse_pattern,
            amplitudes=finite_amplitudes,
            geometry=geometry,
            spot_sigma_px=effective_spot_sigma_px,
        )
    return render_amplitude_to_field(
        pattern=sparse_pattern,
        amplitudes=finite_amplitudes,
        geometry=geometry,
        spot_sigma_px=effective_spot_sigma_px,
    )


@jaxtyped(typechecker=beartype)
def render_ctr_amplitude_to_field(
    pattern: RHEEDPattern,
    amplitudes: Complex[Array, "N"],
    geometry: DetectorGeometry,
    spot_sigma_px: scalar_float,
) -> Complex[Array, "H W"]:
    """Rasterize sparse complex amplitudes as detector CTR streaks.

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    pattern : RHEEDPattern
        Sparse detector pattern with detector coordinates in millimetres.
    amplitudes : Complex[Array, "N"]
        Complex reflection amplitudes aligned with ``pattern.detector_points``.
    geometry : DetectorGeometry
        Detector geometry carrying output shape, pixel calibration, and beam
        centre.
    spot_sigma_px : scalar_float
        Gaussian spot-core width in detector pixels.

    Returns
    -------
    field : Complex[Array, "H W"]
        Dense coherent detector field with elongated CTR envelopes.

    Notes
    -----
    1. Render each reflection as a compact Bragg core plus an elongated CTR
       halo.
    2. Put the halo in quadrature with the core so a single reflection's
       ``|field|²`` reproduces the legacy intensity renderer exactly.
    3. Leave normalization to the downstream intensity reducer.
    """
    sigma_x_px: Float[Array, ""] = jnp.asarray(
        spot_sigma_px,
        dtype=jnp.float64,
    )
    sigma_y_px: Float[Array, ""] = jnp.maximum(4.0 * sigma_x_px, 2.5)
    spot_core_weight: Float[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
    streak_halo_weight: Float[Array, ""] = jnp.asarray(0.6, dtype=jnp.float64)
    amplitudes_arr: Complex[Array, "N"] = jnp.asarray(
        amplitudes,
        dtype=jnp.complex128,
    )
    if amplitudes_arr.shape[0] != pattern.detector_points.shape[0]:
        raise ValueError("amplitudes length must match detector_points")

    valid_mask: Bool[Array, "N"] = pattern.G_indices >= 0
    x_pixels: Float[Array, "N"]
    y_pixels: Float[Array, "N"]
    x_pixels, y_pixels = _detector_points_to_pixels(
        pattern.detector_points,
        geometry,
    )
    valid_amplitudes: Complex[Array, "N"] = jnp.where(
        valid_mask,
        amplitudes_arr,
        0.0 + 0.0j,
    )
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = _detector_grid(geometry)

    def _render_one_streak(
        x0_px: Float[Array, ""],
        y0_px: Float[Array, ""],
        amplitude: Complex[Array, ""],
    ) -> Complex[Array, "H W"]:
        core_envelope: Float[Array, "H W"] = jnp.exp(
            -0.25 * ((x_grid - x0_px) / sigma_x_px) ** 2
            - 0.25 * ((y_grid - y0_px) / sigma_x_px) ** 2
        )
        halo_envelope: Float[Array, "H W"] = jnp.exp(
            -0.25 * ((x_grid - x0_px) / sigma_x_px) ** 2
            - 0.25 * ((y_grid - y0_px) / sigma_y_px) ** 2
        )
        streak_envelope: Complex[Array, "H W"] = (
            jnp.sqrt(spot_core_weight) * core_envelope
            + 1j * jnp.sqrt(streak_halo_weight) * halo_envelope
        )
        return amplitude * streak_envelope

    return jnp.sum(
        jax.vmap(_render_one_streak)(
            x_pixels,
            y_pixels,
            valid_amplitudes,
        ),
        axis=0,
    )


@jaxtyped(typechecker=beartype)
def render_ctr_streaks_to_image(
    pattern: RHEEDPattern,
    geometry: DetectorGeometry,
    spot_sigma_px: scalar_float,
) -> Float[Array, "H W"]:
    """Rasterize a sparse RHEEDPattern onto a dense image as CTR streaks.

    Drop-in replacement for :func:`render_pattern_to_image` that renders each
    reflection as a compact Bragg spot core plus a vertically elongated crystal
    truncation rod (CTR) streak, giving the characteristic RHEED streak
    appearance instead of isotropic spots. Streak elongation is along the
    detector ``y`` axis (the surface-normal projection).

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    pattern : RHEEDPattern
        Sparse detector pattern with positions in millimetres. ``G_indices``
        marks valid reflections (``>= 0``); invalid entries are skipped.
    geometry : DetectorGeometry
        Detector geometry carrying output shape, pixel calibration, and beam
        centre.
    spot_sigma_px : scalar_float
        Gaussian spot-core width in detector pixels. The streak halo is
        elongated to ``max(4 * spot_sigma_px, 2.5)`` pixels in ``y``.

    Returns
    -------
    image : Float[Array, "H W"]
        Rasterized streak image normalized to unit maximum.

    See Also
    --------
    render_pattern_to_image : Isotropic-spot rasterizer.
    """
    return _render_ctr_streaks_to_image(pattern, geometry, spot_sigma_px)


@jaxtyped(typechecker=beartype)
def _render_ctr_streaks_to_image(
    pattern: RHEEDPattern,
    geometry: DetectorGeometry,
    spot_sigma_px: scalar_float,
) -> Float[Array, "H W"]:
    """Render detector-anchored streaks with visible Bragg spot cores."""
    sigma_x_px: Float[Array, ""] = jnp.asarray(
        spot_sigma_px,
        dtype=jnp.float64,
    )
    sigma_y_px: Float[Array, ""] = jnp.maximum(4.0 * sigma_x_px, 2.5)
    spot_core_weight: Float[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
    streak_halo_weight: Float[Array, ""] = jnp.asarray(0.6, dtype=jnp.float64)

    valid_mask: Bool[Array, "N"] = pattern.G_indices >= 0
    x_pixels: Float[Array, "N"]
    y_pixels: Float[Array, "N"]
    x_pixels, y_pixels = _detector_points_to_pixels(
        pattern.detector_points,
        geometry,
    )
    intensities: Float[Array, "N"] = jnp.where(
        valid_mask,
        jnp.asarray(pattern.intensities, dtype=jnp.float64),
        0.0,
    )

    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = _detector_grid(geometry)

    def _render_one_spot_core(
        x0_px: Float[Array, ""],
        y0_px: Float[Array, ""],
        intensity: Float[Array, ""],
    ) -> Float[Array, "H W"]:
        return intensity * jnp.exp(
            -0.5 * ((x_grid - x0_px) / sigma_x_px) ** 2
            - 0.5 * ((y_grid - y0_px) / sigma_x_px) ** 2
        )

    def _render_one_streak_halo(
        x0_px: Float[Array, ""],
        y0_px: Float[Array, ""],
        intensity: Float[Array, ""],
    ) -> Float[Array, "H W"]:
        return intensity * jnp.exp(
            -0.5 * ((x_grid - x0_px) / sigma_x_px) ** 2
            - 0.5 * ((y_grid - y0_px) / sigma_y_px) ** 2
        )

    spot_core_image: Float[Array, "H W"] = jnp.sum(
        jax.vmap(_render_one_spot_core)(x_pixels, y_pixels, intensities),
        axis=0,
    )
    streak_halo_image: Float[Array, "H W"] = jnp.sum(
        jax.vmap(_render_one_streak_halo)(x_pixels, y_pixels, intensities),
        axis=0,
    )
    image: Float[Array, "H W"] = (
        spot_core_weight * spot_core_image
        + streak_halo_weight * streak_halo_image
    )
    max_intensity: Float[Array, ""] = jnp.maximum(jnp.max(image), 1e-12)
    return image / max_intensity


@beartype
def detector_extent_mm(
    geometry: DetectorGeometry,
) -> Tuple[float, float, float, float]:
    """Compute matplotlib-style detector extent in millimetres.

    Parameters
    ----------
    geometry : DetectorGeometry
        Detector geometry carrying output shape, pixel calibration, and beam
        centre.

    Returns
    -------
    extent_mm : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` extent in millimetres.
    """
    return detector_geometry_extent_mm(geometry)


@jaxtyped(typechecker=beartype)
def log_compress_image(
    image: Float[Array, "H W"],
    gain: scalar_float = 25.0,
    dynamic_range_floor: scalar_float = 0.0,
) -> Float[Array, "H W"]:
    """Apply normalized log compression for detector-style display.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Non-negative detector image.
    gain : scalar_float, optional
        Logarithmic gain factor. Default: 25.0
    dynamic_range_floor : scalar_float, optional
        Normalized detector cutoff in ``[0, 1)`` applied before display
        compression. Pixels below the cutoff are hidden, and pixels above
        it are rescaled onto the visible range. Default: 0.0

    Returns
    -------
    compressed_image : Float[Array, "H W"]
        Log-compressed image normalized to ``[0, 1]``.
    """
    gain_safe: Float[Array, ""] = jnp.maximum(
        jnp.asarray(gain, dtype=jnp.float64), 1e-12
    )
    floor_safe: Float[Array, ""] = jnp.clip(
        jnp.asarray(dynamic_range_floor, dtype=jnp.float64),
        0.0,
        1.0 - 1e-12,
    )
    normalized: Float[Array, "H W"] = image / jnp.maximum(
        jnp.max(image), 1e-12
    )
    visible: Float[Array, "H W"] = jnp.where(
        normalized >= floor_safe,
        (normalized - floor_safe) / jnp.maximum(1.0 - floor_safe, 1e-12),
        0.0,
    )
    return jnp.log1p(gain_safe * visible) / jnp.log1p(gain_safe)


@jaxtyped(typechecker=beartype)
def _bind_kinematic_distributions(  # noqa: PLR0913, PLR0915
    distributions: tuple[Distribution, ...],
    crystal: CrystalStructure,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    layer_attenuation: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
    angular_divergence_mrad: scalar_float,
    energy_spread_ev: scalar_float,
    domain_aspect_ratio: Tuple[float, float, float],
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    render_ctrs_as_streaks: bool,
    parameterization: str,
    surface_config: SurfaceConfig | None,
    defect_surface_layer_depth_angstrom: scalar_float,
) -> Callable[[Float[Array, "D"]], Complex[Array, "H W"]]:
    """Bind composed distribution samples to the kinematic kernel."""
    axis_dims: tuple[int, ...] = tuple(
        distribution.samples.shape[1] for distribution in distributions
    )
    twin_builder: Callable[[Float[Array, "2"]], CrystalStructure] = (
        bind_twin_wall_distribution(
            slab=crystal,
            surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
        )
    )
    step_builder: Callable[[Float[Array, "3"]], CrystalStructure] = (
        bind_step_edge_distribution(
            slab=crystal,
            surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
        )
    )

    axis_binds: tuple[
        Callable[[Float[Array, "D_axis"]], KinematicAxisUpdate],
        ...,
    ] = tuple(
        distribution.bind(
            lambda dist: bind_kinematic_axis_distribution(
                dist,
                twin_builder=twin_builder,
                step_builder=step_builder,
            )
        )
        for distribution in distributions
    )

    def _bound(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
        sample_crystal: CrystalStructure = crystal
        sample_energy_kev: Float[Array, ""] = jnp.asarray(
            energy_kev,
            dtype=jnp.float64,
        )
        sample_theta_deg: Float[Array, ""] = jnp.asarray(
            theta_deg,
            dtype=jnp.float64,
        )
        sample_phi_deg: Float[Array, ""] = jnp.asarray(
            phi_deg,
            dtype=jnp.float64,
        )
        sample_domain_size_angstrom: Any | None = None
        cursor: int = 0
        for axis_bind, sample_dim in zip(axis_binds, axis_dims, strict=True):
            axis_sample: Float[Array, "D_axis"] = sample[
                cursor : cursor + sample_dim
            ]
            cursor += sample_dim
            axis_update: KinematicAxisUpdate = axis_bind(axis_sample)
            if axis_update.crystal is not None:
                sample_crystal = axis_update.crystal
            if axis_update.domain_size_angstrom is not None:
                sample_domain_size_angstrom = axis_update.domain_size_angstrom
            sample_energy_kev = (
                sample_energy_kev + axis_update.energy_delta_kev
            )
            sample_theta_deg = sample_theta_deg + axis_update.theta_delta_deg
            sample_phi_deg = sample_phi_deg + axis_update.phi_delta_deg

        if sample_domain_size_angstrom is not None:
            energy_spread_frac: Float[Array, ""] = jnp.asarray(
                energy_spread_ev,
                dtype=jnp.float64,
            ) / jnp.maximum(sample_energy_kev * 1000.0, 1e-12)
            return _kinematic_finite_domain_amplitude(
                crystal=sample_crystal,
                energy_kev=sample_energy_kev,
                theta_deg=sample_theta_deg,
                phi_deg=sample_phi_deg,
                domain_size_angstrom=sample_domain_size_angstrom,
                domain_aspect_ratio=domain_aspect_ratio,
                hmax=hmax,
                kmax=kmax,
                detector_distance_mm=detector_distance_mm,
                temperature=temperature,
                surface_roughness=surface_roughness,
                layer_attenuation=layer_attenuation,
                ctr_power=ctr_power,
                roughness_power=roughness_power,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=spot_sigma_px,
                render_ctrs_as_streaks=render_ctrs_as_streaks,
                parameterization=parameterization,
                surface_config=surface_config,
                energy_spread_frac=energy_spread_frac,
                beam_divergence_rad=angular_divergence_mrad * 1.0e-3,
            )

        return kinematic_amplitude(
            crystal=sample_crystal,
            energy_kev=sample_energy_kev,
            theta_deg=sample_theta_deg,
            phi_deg=sample_phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            layer_attenuation=layer_attenuation,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
            parameterization=parameterization,
            surface_config=surface_config,
        )

    return _bound


@jaxtyped(typechecker=beartype)
def _validate_multislice_kernel_axes(
    distributions: tuple[Distribution, ...],
    phi_deg: scalar_num,
) -> None:
    """Reject distribution axes the reflection kernel cannot honor.

    The edge-on reflection pipeline currently supports ``phi_deg == 0``
    only, so any axis that rotates the azimuth (grain populations, size
    envelopes bound through azimuth updates, generic one-dimensional phi
    axes) raises :class:`NotImplementedError`, as do beam axes whose
    samples carry nonzero phi offsets. Beam-energy/theta axes and
    structure axes (twins, steps) are fully supported.
    """
    if abs(float(phi_deg)) > 1e-12:
        raise NotImplementedError(
            "reflection multislice currently supports phi_deg=0 only"
        )
    for distribution in distributions:
        axis_id: str | None = distribution.axis_id
        if axis_id in STRUCTURE_AXIS_IDS:
            continue
        if axis_id in BEAM_AXIS_IDS:
            max_phi_offset: float = float(
                jnp.max(jnp.abs(distribution.samples[:, 1]))
            )
            if max_phi_offset > 1e-12:
                raise NotImplementedError(
                    "kernel='multislice' routes through the edge-on "
                    "reflection pipeline, which supports phi_deg=0 only; "
                    f"beam axis {axis_id!r} carries nonzero phi offsets."
                )
            continue
        raise NotImplementedError(
            f"Distribution axis {axis_id!r} is not supported by the "
            "reflection multislice kernel: it binds an azimuth rotation "
            "or finite-domain envelope, and the edge-on pipeline supports "
            "phi_deg=0 with periodic transverse geometry only. Use the "
            "kinematic kernel for these axes."
        )


def _bind_multislice_distributions(  # noqa: PLR0913
    distributions: tuple[Distribution, ...],
    crystal: CrystalStructure,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    detector_distance_mm: scalar_float,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    bandwidth_limit: scalar_float,
    parameterization: str,
    defect_surface_layer_depth_angstrom: scalar_float,
    dx_slice: float,
    dy: float,
    dz: float,
    propagation_length_ang: float,
    vacuum_above: float,
    cap_width: float,
) -> Callable[[Float[Array, "D"]], Complex[Array, "H W"]]:
    """Bind composed distribution samples to the reflection multislice kernel.

    Builds edge-on potential slices from the crystal once, then binds
    theta/energy offsets (and per-sample structure rebuilds for twin/step
    axes) to :func:`rheedium.simul.reflection_detector_amplitude`, exactly
    mirroring the kinematic kernel's beam-axis semantics. Azimuth-rotating
    axes raise :class:`NotImplementedError` because the reflection module
    supports ``phi_deg == 0`` only.
    """
    from .reflection_multislice import (
        _edge_on_slices_like,
        crystal_to_edge_on_slices,
        reflection_detector_amplitude,
    )

    _validate_multislice_kernel_axes(distributions, phi_deg)
    template_slices = crystal_to_edge_on_slices(
        crystal,
        dx_slice=dx_slice,
        dy=dy,
        dz=dz,
        vacuum_above=vacuum_above,
        cap_width=cap_width,
        parameterization=parameterization,
    )
    axis_dims: tuple[int, ...] = tuple(
        distribution.samples.shape[1] for distribution in distributions
    )
    twin_builder: Callable[
        [Float[Array, "2"]],
        CrystalStructure,
    ] = bind_twin_wall_distribution(
        slab=crystal,
        surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
    )
    step_builder: Callable[
        [Float[Array, "3"]],
        CrystalStructure,
    ] = bind_step_edge_distribution(
        slab=crystal,
        surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
    )
    axis_binds: tuple[
        Callable[[Float[Array, "D_axis"]], MultisliceAxisUpdate],
        ...,
    ] = tuple(
        distribution.bind(
            lambda dist: bind_multislice_axis_distribution(
                dist,
                twin_builder=twin_builder,
                step_builder=step_builder,
            )
        )
        for distribution in distributions
    )

    def _bound(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
        sample_slices = template_slices
        sample_energy_kev: Float[Array, ""] = jnp.asarray(
            energy_kev,
            dtype=jnp.float64,
        )
        sample_theta_deg: Float[Array, ""] = jnp.asarray(
            theta_deg,
            dtype=jnp.float64,
        )
        cursor: int = 0
        for axis_bind, sample_dim in zip(axis_binds, axis_dims, strict=True):
            axis_sample: Float[Array, "D_axis"] = sample[
                cursor : cursor + sample_dim
            ]
            cursor += sample_dim
            axis_update: MultisliceAxisUpdate = axis_bind(axis_sample)
            if axis_update.crystal is not None:
                sample_slices = _edge_on_slices_like(
                    axis_update.crystal,
                    template=template_slices,
                    parameterization=parameterization,
                )
            sample_energy_kev = (
                sample_energy_kev + axis_update.energy_delta_kev
            )
            sample_theta_deg = sample_theta_deg + axis_update.theta_delta_deg

        return reflection_detector_amplitude(
            slices=sample_slices,
            energy_kev=sample_energy_kev,
            theta_deg=sample_theta_deg,
            detector_distance_mm=detector_distance_mm,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            bandwidth_limit=bandwidth_limit,
            propagation_length_ang=propagation_length_ang,
        )

    return _bound


def _normalize_detector_image(
    detector_image: Float[Array, "H W"],
) -> Float[Array, "H W"]:
    """Normalize a detector image to unit maximum."""
    max_intensity: Float[Array, ""] = jnp.maximum(
        jnp.max(detector_image),
        1e-12,
    )
    return detector_image / max_intensity


@jaxtyped(typechecker=beartype)
def _apply_detector_psf_and_normalize(
    detector_image: Float[Array, "H W"],
    psf_sigma_pixels: scalar_float,
) -> Float[Array, "H W"]:
    """Apply detector PSF and normalize the detector image."""
    convolved: Float[Array, "H W"] = detector_psf_convolve(
        detector_image=detector_image,
        psf_sigma_pixels=psf_sigma_pixels,
    )
    return _normalize_detector_image(convolved)


@jaxtyped(typechecker=beartype)
def _legacy_instrument_distribution(
    angular_divergence_mrad: scalar_float,
    energy_spread_ev: scalar_float,
    n_angular_samples: int,
    n_energy_samples: int,
) -> Distribution:
    """Convert legacy broadening widths to a generic instrument axis."""
    angle_nodes: Float[Array, "A"]
    angle_weights: Float[Array, "A"]
    energy_nodes: Float[Array, "E"]
    energy_weights: Float[Array, "E"]
    angle_nodes, angle_weights = gauss_hermite_nodes_weights(n_angular_samples)
    energy_nodes, energy_weights = gauss_hermite_nodes_weights(
        n_energy_samples
    )
    sqrt2: Float[Array, ""] = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    sqrt_pi: Float[Array, ""] = jnp.sqrt(
        jnp.asarray(jnp.pi, dtype=jnp.float64)
    )
    theta_offsets: Float[Array, "A"] = (
        sqrt2
        * jnp.asarray(angular_divergence_mrad, dtype=jnp.float64)
        * 1.0e-3
        * angle_nodes
    )
    energy_offsets: Float[Array, "E"] = (
        sqrt2 * jnp.asarray(energy_spread_ev, dtype=jnp.float64) * energy_nodes
    )
    theta_grid: Float[Array, "A E"]
    energy_grid: Float[Array, "A E"]
    theta_grid, energy_grid = jnp.meshgrid(
        theta_offsets,
        energy_offsets,
        indexing="ij",
    )
    phi_grid: Float[Array, "A E"] = jnp.zeros_like(theta_grid)
    weight_grid: Float[Array, "A E"] = (
        angle_weights[:, None] * energy_weights[None, :]
    ) / (sqrt_pi**2)
    samples: Float[Array, "N 3"] = jnp.stack(
        [
            theta_grid.ravel(),
            phi_grid.ravel(),
            energy_grid.ravel(),
        ],
        axis=-1,
    )
    return create_distribution(
        samples=samples,
        weights=weight_grid.ravel(),
        reduction=ReductionMode.INCOHERENT,
        axis_id="legacy_instrument",
    )


def _detector_image_distributions(  # noqa: PLR0913
    beam_modes: BeamModeDistribution | None,
    orientation_distribution: OrientationDistribution | None,
    distribution: Distribution | None,
    angular_divergence_mrad: scalar_float,
    energy_spread_ev: scalar_float,
    n_angular_samples: int,
    n_energy_samples: int,
    n_beam_modes_per_axis: int,
    n_beam_modes_out_of_plane: int | None,
    n_beam_energy_points: int,
    n_mosaic_points: scalar_int,
) -> tuple[Distribution, ...]:
    """Normalize public ensemble inputs into ordered distribution axes."""
    distributions: list[Distribution] = []
    if beam_modes is None:
        distributions.append(
            _legacy_instrument_distribution(
                angular_divergence_mrad=angular_divergence_mrad,
                energy_spread_ev=energy_spread_ev,
                n_angular_samples=n_angular_samples,
                n_energy_samples=n_energy_samples,
            )
        )
    else:
        distributions.append(
            decompose_beam_modes(
                beam_modes,
                n_modes_per_axis=n_beam_modes_per_axis,
                n_modes_out_of_plane=n_beam_modes_out_of_plane,
                n_energy_points=n_beam_energy_points,
            )
        )
    if orientation_distribution is not None:
        distributions.append(
            orientation_to_distribution(
                orientation_distribution,
                n_mosaic_points=n_mosaic_points,
            )
        )
    if distribution is not None:
        distributions.append(distribution)
    return tuple(distributions)


@jaxtyped(typechecker=beartype)
def simulate_detector_image(
    crystal: CrystalStructure,
    beam: BeamSpec | None = None,
    surface: SurfaceCTRParams | None = None,
    detector: DetectorGeometry | None = None,
    render: RenderParams | None = None,
) -> Float[Array, "H W"]:
    """Simulate a broadened detector image from a Layer-0 amplitude kernel.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
    beam : BeamSpec, optional
        Electron beam, incidence geometry, and beam-mode sampling metadata.
    surface : SurfaceCTRParams, optional
        CTR bounds, roughness, temperature, and domain-surface controls.
    detector : DetectorGeometry, optional
        Detector distance, dense image calibration, and PSF width.
    render : RenderParams, optional
        Rendering, kernel, quadrature, and optional ensemble distributions.

    Returns
    -------
    detector_image : Float[Array, "H W"]
        Dense detector image normalized to unit maximum.

    Notes
    -----
    This orchestrator is the Layer-1 detector integrator:

    1. Normalize public ensemble inputs into ordered ``Distribution`` axes.
    2. Bind those axes to the selected coherent amplitude kernel.
    3. Reduce with :func:`apply_distributions`.
    4. Apply detector PSF and normalize.

    ``render.kernel="multislice"`` runs the edge-on *reflection* multislice
    pipeline (:mod:`rheedium.simul.reflection_multislice`): edge-on slices
    are built from ``crystal`` directly, so ``render.potential_slices`` is
    deprecated and ignored for this kernel. The reflection module supports
    ``phi_deg == 0`` only; nonzero azimuths (including distribution axes
    that rotate the azimuth) raise :class:`NotImplementedError`.
    """
    if beam is None:
        beam = BeamSpec()
    if surface is None:
        surface = SurfaceCTRParams()
    if detector is None:
        detector = DetectorGeometry(distance=1000.0, psf_sigma_pixels=1.2)
    if render is None:
        render = RenderParams()

    _validate_layer0_kernel(render.kernel)
    if (
        render.orientation_distribution is not None
        and render.distribution is not None
    ):
        raise ValueError(
            "Pass either orientation_distribution or distribution, not both."
        )
    if beam.beam_modes is not None and render.distribution is not None:
        raise ValueError("Pass either beam_modes or distribution, not both.")
    distributions: tuple[Distribution, ...] = _detector_image_distributions(
        beam_modes=beam.beam_modes,
        orientation_distribution=render.orientation_distribution,
        distribution=render.distribution,
        angular_divergence_mrad=beam.angular_divergence_mrad,
        energy_spread_ev=beam.energy_spread_ev,
        n_angular_samples=render.n_angular_samples,
        n_energy_samples=render.n_energy_samples,
        n_beam_modes_per_axis=beam.n_beam_modes_per_axis,
        n_beam_modes_out_of_plane=beam.n_beam_modes_out_of_plane,
        n_beam_energy_points=beam.n_beam_energy_points,
        n_mosaic_points=render.n_mosaic_points,
    )
    bound: Callable[[Float[Array, "D"]], Complex[Array, "H W"]]
    if render.kernel == _KINEMATIC_KERNEL:
        bound = _bind_kinematic_distributions(
            distributions=distributions,
            crystal=crystal,
            energy_kev=beam.energy_kev,
            theta_deg=beam.theta_deg,
            phi_deg=beam.phi_deg,
            hmax=surface.hmax,
            kmax=surface.kmax,
            detector_distance_mm=detector_geometry_distance_mm(detector),
            temperature=surface.temperature,
            surface_roughness=surface.surface_roughness,
            layer_attenuation=_resolve_layer_attenuation(
                surface.layer_attenuation,
                surface.ctr_regularization,
            ),
            ctr_power=surface.ctr_power,
            roughness_power=surface.roughness_power,
            angular_divergence_mrad=beam.angular_divergence_mrad,
            energy_spread_ev=beam.energy_spread_ev,
            domain_aspect_ratio=surface.finite_domain_aspect_ratio,
            image_shape_px=detector_image_shape_px(detector),
            pixel_size_mm=detector_pixel_size_mm(detector),
            beam_center_px=detector_beam_center_px(detector),
            spot_sigma_px=render.spot_sigma_px,
            render_ctrs_as_streaks=render.render_ctrs_as_streaks,
            parameterization=render.parameterization,
            surface_config=surface.surface_config,
            defect_surface_layer_depth_angstrom=(
                surface.defect_surface_layer_depth_angstrom
            ),
        )
    else:
        if render.potential_slices is not None:
            warnings.warn(
                "render.potential_slices is deprecated and ignored for "
                "kernel='multislice': the RHEED multislice kernel now runs "
                "the edge-on reflection pipeline, which builds its own "
                "slices from the crystal. Use multislice_propagate / "
                "multislice_simulator directly for transmission-geometry "
                "potential slices.",
                DeprecationWarning,
                stacklevel=2,
            )
        bound = _bind_multislice_distributions(
            distributions=distributions,
            crystal=crystal,
            energy_kev=beam.energy_kev,
            theta_deg=beam.theta_deg,
            phi_deg=beam.phi_deg,
            detector_distance_mm=detector_geometry_distance_mm(detector),
            image_shape_px=detector_image_shape_px(detector),
            pixel_size_mm=detector_pixel_size_mm(detector),
            beam_center_px=detector_beam_center_px(detector),
            spot_sigma_px=render.spot_sigma_px,
            bandwidth_limit=render.bandwidth_limit,
            parameterization=render.parameterization,
            defect_surface_layer_depth_angstrom=(
                surface.defect_surface_layer_depth_angstrom
            ),
            dx_slice=render.dx_slice,
            dy=render.dy,
            dz=render.dz,
            propagation_length_ang=render.propagation_length_ang,
            vacuum_above=render.vacuum_above,
            cap_width=render.cap_width,
        )
    detector_image: Float[Array, "H W"] = apply_distributions(
        distributions,
        bound,
    )
    return _apply_detector_psf_and_normalize(
        detector_image,
        psf_sigma_pixels=detector_psf_sigma_pixels(detector),
    )


@jaxtyped(typechecker=beartype)
def simulate_detector_image_instrument(
    crystal: CrystalStructure,
    beam: BeamSpec,
    surface: SurfaceCTRParams | None = None,
    detector: DetectorGeometry | None = None,
    render: RenderParams | None = None,
) -> Float[Array, "H W"]:
    """Simulate a detector image by incoherently summing beam modes."""
    if surface is None:
        surface = SurfaceCTRParams()
    if detector is None:
        detector = DetectorGeometry(distance=1000.0, psf_sigma_pixels=1.2)
    if render is None:
        render = RenderParams(render_ctrs_as_streaks=False)
    if beam.beam_modes is None:
        raise ValueError(
            "simulate_detector_image_instrument requires "
            "beam.beam_modes to be set."
        )
    if render.kernel != _KINEMATIC_KERNEL:
        raise ValueError(
            "simulate_detector_image_instrument currently supports only "
            "render.kernel='kinematic'; use simulate_detector_image with "
            "RenderParams(kernel='multislice') for the reflection "
            "multislice path."
        )
    return simulate_detector_image(
        crystal=crystal,
        beam=beam,
        surface=surface,
        detector=detector,
        render=render,
    )


@jaxtyped(typechecker=beartype)
def sliced_crystal_to_projected_potential_slices(
    sliced_crystal: SlicedCrystal,
    slice_thickness: scalar_float = 2.0,
    pixel_size: scalar_float = 0.1,
    parameterization: str = "lobato",
) -> PotentialSlices:
    r"""Convert a SlicedCrystal into projected-potential slices.

    This function takes a surface-oriented crystal slab and generates 3D
    potential slices suitable for multislice electron diffraction simulations.
    The potential is calculated from atomic positions using the selected
    parameterization (Lobato-van Dyck by default) for accurate projected
    atomic potentials.

    Each atom contributes a raw projected potential in Volt-Angstrom
    units. The electron-specimen interaction constant is applied later
    when the transmission function is built during propagation.

    :see: :class:`~.test_simulator.TestSlicedCrystalToProjectedPotentialSlices`

    Parameters
    ----------
    sliced_crystal : SlicedCrystal
        Surface-oriented crystal structure with atoms and extents.
    slice_thickness : scalar_float, optional
        Thickness of each potential slice in Ångstroms. Default: 2.0 Å
        Determines the z-spacing between consecutive slices.
    pixel_size : scalar_float, optional
        Real-space pixel size in Ångstroms. Default: 0.1 Å
        Sets the lateral resolution of the potential grid.
    parameterization : str, optional
        Atomic potential model: ``"lobato"`` (default) or ``"kirkland"``.
        Not JIT-compiled; resolved at trace time.
        Used for projected-potential evaluation.

    Returns
    -------
    potential_slices : PotentialSlices
        3D projected-potential array in Volt-Angstrom with calibration
        information.

    Notes
    -----
    - The potential includes proper atomic scattering factors
    - Assumes independent atom approximation
    - Periodic boundary conditions in x-y plane
    - Non-periodic in z-direction (surface slab)
    - Each atom's projected potential is weighted by its site
      occupancy (``sliced_crystal.occupancies``; ones when absent)

    1. **Grid dimensions** --
       Compute nx, ny from extents and pixel size, nz from
       depth and slice thickness.
    2. **Per-slice potential** --
       For each z-range, select atoms in slice, project
       potentials onto xy grid with periodic wrapping, and sum.
    3. **Package result** --
       Return :class:`PotentialSlices` with calibration.

    See Also
    --------
    projected_potential : Projected atomic potential calculation.
    create_potential_slices : Create PotentialSlices from array.
    multislice_propagate : Transmission propagation through the slices.
    multislice_simulator : Sparse transmission diffraction pattern.
    reflection_multislice_simulator : The RHEED (reflection) multislice
        path, which builds its own edge-on slices from the crystal.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create surface slab
    >>> bulk = rh.inout.parse_cif("SrTiO3.cif")
    >>> slab = rh.ucell.bulk_to_slice(
    ...     bulk_crystal=bulk, orientation=jnp.array([1, 1, 1]), depth=20.0
    ... )
    >>>
    >>> # Convert to potential slices
    >>> potential = rh.simul.sliced_crystal_to_projected_potential_slices(
    ...     sliced_crystal=slab, slice_thickness=2.0, pixel_size=0.1
    ... )
    """
    slice_thickness: Float[Array, ""] = jnp.asarray(
        slice_thickness, dtype=jnp.float64
    )
    pixel_size: Float[Array, ""] = jnp.asarray(pixel_size, dtype=jnp.float64)
    positions: Float[Array, "N 3"] = sliced_crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = sliced_crystal.cart_positions[:, 3]
    occupancies: Float[Array, "N"] = (
        jnp.ones(positions.shape[0], dtype=jnp.float64)
        if sliced_crystal.occupancies is None
        else jnp.asarray(sliced_crystal.occupancies, dtype=jnp.float64)
    )
    x_extent: Float[Array, ""] = sliced_crystal.x_extent
    y_extent: Float[Array, ""] = sliced_crystal.y_extent
    depth: Float[Array, ""] = sliced_crystal.depth
    nx: int = int(jnp.ceil(x_extent / pixel_size))
    ny: int = int(jnp.ceil(y_extent / pixel_size))
    nz: int = int(jnp.ceil(depth / slice_thickness))
    # Periodic grid: n samples spaced L/n, excluding the endpoint L
    # (0 and L are the same periodic point). This matches the
    # fftfreq(n, L/n) convention used by the Fresnel propagators.
    x_coords: Float[Array, "nx"] = jnp.arange(nx) * (x_extent / nx)
    y_coords: Float[Array, "ny"] = jnp.arange(ny) * (y_extent / ny)
    xx: Float[Array, "nx ny"]
    yy: Float[Array, "nx ny"]
    xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
    n_atoms: int = positions.shape[0]

    def _calculate_slice_potential(
        slice_idx: Int[Array, ""],
    ) -> Float[Array, "nx ny"]:
        """Calculate potential for a single slice."""
        z_start: Float[Array, ""] = slice_idx * slice_thickness
        z_end: Float[Array, ""] = (slice_idx + 1) * slice_thickness
        z_positions: Float[Array, "N"] = positions[:, 2]
        in_slice: Bool[Array, "N"] = jnp.logical_and(
            z_positions >= z_start, z_positions < z_end
        )

        def _atom_contribution(
            atom_idx: Int[Array, ""],
        ) -> Float[Array, "nx ny"]:
            """Calculate contribution from single atom to potential."""
            pos: Float[Array, "3"] = positions[atom_idx]
            z_number: Int[Array, ""] = atomic_numbers[atom_idx].astype(
                jnp.int32
            )
            is_in_slice: Bool[Array, ""] = in_slice[atom_idx]
            dx: Float[Array, "nx ny"] = xx - pos[0]
            dy: Float[Array, "nx ny"] = yy - pos[1]
            dx = dx - x_extent * jnp.round(dx / x_extent)
            dy = dy - y_extent * jnp.round(dy / y_extent)
            r: Float[Array, "nx ny"] = jnp.sqrt(dx**2 + dy**2)
            atom_potential: Float[Array, "nx ny"] = occupancies[
                atom_idx
            ] * projected_potential(
                z_number,
                r,
                parameterization,
            )
            return jnp.where(is_in_slice, atom_potential, 0.0)

        atom_indices: Int[Array, "N"] = jnp.arange(n_atoms)
        contributions: Float[Array, "N nx ny"] = jax.vmap(_atom_contribution)(
            atom_indices
        )
        slice_potential: Float[Array, "nx ny"] = jnp.sum(contributions, axis=0)
        return slice_potential

    slice_indices: Int[Array, "nz"] = jnp.arange(nz)
    all_slices: Float[Array, "nz nx ny"] = jax.vmap(
        _calculate_slice_potential
    )(slice_indices)
    potential_slices: PotentialSlices = create_potential_slices(
        slices=all_slices,
        slice_thickness=slice_thickness,
        x_calibration=pixel_size,
        y_calibration=pixel_size,
    )
    return potential_slices


@jaxtyped(typechecker=beartype)
def _refraction_wavevector_components(
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    inner_potential_v0: scalar_float,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Return the refracted wavevector components at a crystal surface.

    Snell's law for electrons at a mean-inner-potential step conserves the
    momentum component parallel to the surface and rescales only the
    surface-normal component:

    .. math::

        k_\parallel = k \cos\theta \quad \text{(unchanged)}, \qquad
        k_{z,\text{inside}} = k \sqrt{\sin^2\theta + V_0 / V}.

    Parameters
    ----------
    energy_kev : scalar_float
        Accelerating voltage in kilovolts.
    theta_deg : scalar_float
        Grazing incidence angle in degrees, measured from the surface.
    inner_potential_v0 : scalar_float
        Mean inner potential of the crystal in volts.

    Returns
    -------
    k_parallel : Float[Array, ""]
        Surface-parallel wavevector magnitude in inverse Angstroms; it is
        identical inside and outside the crystal.
    k_z_inside : Float[Array, ""]
        Surface-normal wavevector magnitude inside the crystal in inverse
        Angstroms; this is what a z-resolved solver would propagate with,
        and it matches the Fresnel-step ``k_perp_in`` used by
        ``reflection_multislice._flat_step_specular_reflectivity``.
    """
    lam_ang: Float[Array, ""] = wavelength_ang(energy_kev)
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta_rad: Float[Array, ""]
    theta_rad, _ = incidence_angles_to_radians(theta_deg, 0.0)
    voltage_v: Float[Array, ""] = (
        jnp.asarray(energy_kev, dtype=jnp.float64) * 1000.0
    )
    k_parallel: Float[Array, ""] = k_mag * jnp.cos(theta_rad)
    k_z_inside: Float[Array, ""] = k_mag * jnp.sqrt(
        jnp.sin(theta_rad) ** 2 + inner_potential_v0 / voltage_v
    )
    return k_parallel, k_z_inside


_TRANSMISSION_TILT_GUARD_MESSAGE: Final[str] = (
    "multislice_propagate: the in-plane tilt k*cos(theta) exceeds the "
    "bandwidth-limited grid Nyquist pi*bandwidth_limit/min(dx, dy), so the "
    "tilted incident wave cannot be represented on this grid (it would "
    "alias). This kernel is a TRANSMISSION calculation along z and is not "
    "valid for grazing-incidence RHEED; use the reflection pipeline instead "
    "(simulate_detector_image with RenderParams(kernel='multislice') or "
    "rheedium.simul.reflection_multislice_simulator)."
)


def _guard_transmission_tilt(
    psi: Complex[Array, "nx ny"],
    k_parallel: Float[Array, ""],
    tilt_limit: Float[Array, ""],
) -> Complex[Array, "nx ny"]:
    """Raise when the in-plane tilt is unrepresentable on the grid.

    With concrete inputs this raises :class:`ValueError` eagerly; under
    tracing (``jit``/``vmap``) it attaches an ``equinox.error_if`` runtime
    check to the wavefield instead.
    """
    exceeds = k_parallel > tilt_limit
    try:
        exceeds_concrete = bool(exceeds)
    except jax.errors.ConcretizationTypeError:
        return eqx.error_if(psi, exceeds, _TRANSMISSION_TILT_GUARD_MESSAGE)
    if exceeds_concrete:
        raise ValueError(_TRANSMISSION_TILT_GUARD_MESSAGE)
    return psi


@jaxtyped(typechecker=beartype)
def multislice_propagate(
    potential_slices: PotentialSlices,
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> Complex[Array, "nx ny"]:
    r"""Propagate an electron wave through potential slices along z.

    **Transmission geometry along z -- NOT valid for grazing-incidence
    RHEED.** Slices are x-y planes stacked along the surface normal and the
    wave marches through them exactly as in a TEM multislice calculation.
    For RHEED use the edge-on reflection pipeline instead
    (``simulate_detector_image`` with ``RenderParams(kernel="multislice")``
    or :func:`rheedium.simul.reflection_multislice_simulator`). A
    representability guard raises when the requested in-plane tilt
    :math:`k\cos\theta` exceeds the bandwidth-limited grid Nyquist
    frequency, which makes grazing-incidence misuse impossible instead of
    silently aliased.

    The algorithm alternates between transmission through a slice,
    :math:`\psi' = \psi \exp(i\sigma V)`, and Fresnel propagation,
    :math:`\psi \to \mathrm{FFT}^{-1}[\mathrm{FFT}[\psi] P(k_x, k_y)]`
    with :math:`P = \exp(-i\pi\lambda\Delta z (k_x^2 + k_y^2))`. The
    initial wave is a tilted plane wave with phase
    :math:`k_\parallel(\hat{x}\cos\varphi + \hat{y}\sin\varphi)\cdot r`.

    :see: :class:`~.test_simulator.TestMultislicePropagate`

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials with shape (nz, nx, ny)
    energy_kev : scalar_float
        Accelerating voltage in kilovolts
    theta_deg : scalar_float
        Angle of the beam from the x-y (slice) plane in degrees. For a
        transmission calculation this is close to 90 degrees (beam along
        the slice-stacking z axis); the representability guard rejects
        grazing values on realistic grids.
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
        phi=0: beam along +x axis, phi=90: beam along +y axis
    inner_potential_v0 : scalar_float, optional
        Inner (mean) potential of the crystal in volts (default: 0.0).
        Refraction at the surface conserves the surface-parallel momentum:
        the tilt phase uses :math:`k_\parallel = k\cos\theta` unchanged,
        while the surface-normal component inside the crystal is
        :math:`k_{z,\text{inside}} = k\sqrt{\sin^2\theta + V_0/V}` (see
        ``_refraction_wavevector_components``; a z-resolved solver would
        propagate with that component).
    bandwidth_limit : scalar_float, optional
        Fraction of Nyquist frequency to retain (default: 2/3).
        Applied as a low-pass filter in Fourier space to prevent aliasing
        artifacts from the non-linear transmission function. A value of
        2/3 is standard; use 1.0 to disable bandwidth limiting.

    Returns
    -------
    exit_wave : Complex[Array, "nx ny"]
        Complex exit wave after propagation through all slices

    Raises
    ------
    ValueError
        When ``k cos(theta) > bandwidth_limit * pi / min(dx, dy)``: the
        tilted incident wave cannot be represented on the grid. Under
        tracing the same condition raises through ``equinox.error_if``.

    Notes
    -----
    The transmission function is
    :math:`T(x,y) = \exp(i \sigma V(x,y))`, where
    :math:`\sigma = 2 \pi m e / (h^2 k)` is the interaction constant.

    1. **Guard** --
       Raise if the in-plane tilt exceeds the bandwidth-limited
       grid Nyquist.
    2. **Initialise wave** --
       Tilted plane wave from the k-parallel-conserving
       :math:`k_{in,x}`, :math:`k_{in,y}`.
    3. **Build propagator** --
       Fresnel propagator
       :math:`P = \exp(-i \pi \lambda \Delta z
       (k_x^2 + k_y^2))` and bandwidth aperture.
    4. **Scan slices** --
       For each slice: transmit
       :math:`\psi' = \psi \exp(i \sigma V)`, then
       propagate in Fourier space.
    5. **Return exit wave** --
       Complex wave after all slices.

    See Also
    --------
    wavelength_ang : Compute electron wavelength from voltage.
    sliced_crystal_to_projected_potential_slices : Create projected-potential
        slices from crystal.
    multislice_simulator : Sparse transmission diffraction pattern.
    reflection_multislice_simulator : The RHEED (reflection) multislice path.

    References
    ----------
    - Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy,
      2nd ed.
    - Cowley & Moodie (1957). Acta Cryst. 10, 609-619.
    """
    v_slices: Float[Array, " nz nx ny"] = potential_slices.slices
    dz: scalar_float = potential_slices.slice_thickness
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    nx: int = v_slices.shape[1]
    ny: int = v_slices.shape[2]
    lam_ang: scalar_float = wavelength_ang(energy_kev)
    sigma: scalar_float = interaction_constant(energy_kev, lam_ang)
    x: Float[Array, " nx"] = jnp.arange(nx) * dx
    y: Float[Array, " ny"] = jnp.arange(ny) * dy
    kx: Float[Array, " nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, " ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, " nx ny"]
    ky_grid: Float[Array, " nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    azimuth_angle_rad: scalar_float
    _, azimuth_angle_rad = incidence_angles_to_radians(
        theta_deg,
        phi_deg,
    )
    k_parallel: Float[Array, ""]
    k_parallel, _ = _refraction_wavevector_components(
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        inner_potential_v0=inner_potential_v0,
    )
    k_in_x: scalar_float = k_parallel * jnp.cos(azimuth_angle_rad)
    k_in_y: scalar_float = k_parallel * jnp.sin(azimuth_angle_rad)
    x_grid: Float[Array, " nx ny"]
    y_grid: Float[Array, " nx ny"]
    x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
    phase_init: Float[Array, "nx ny"] = k_in_x * x_grid + k_in_y * y_grid
    psi: Complex[Array, "nx ny"] = jnp.exp(1j * phase_init)
    tilt_limit: Float[Array, ""] = (
        jnp.asarray(bandwidth_limit, dtype=jnp.float64)
        * jnp.pi
        / jnp.minimum(
            jnp.asarray(dx, dtype=jnp.float64),
            jnp.asarray(dy, dtype=jnp.float64),
        )
    )
    psi = _guard_transmission_tilt(psi, k_parallel, tilt_limit)
    kx_max: scalar_float = 0.5 / dx
    ky_max: scalar_float = 0.5 / dy
    k_cutoff_x: scalar_float = bandwidth_limit * kx_max
    k_cutoff_y: scalar_float = bandwidth_limit * ky_max
    bandwidth_aperture: Float[Array, "nx ny"] = jnp.exp(
        -0.5
        * (
            (jnp.abs(kx_grid) / k_cutoff_x) ** 8
            + (jnp.abs(ky_grid) / k_cutoff_y) ** 8
        )
    )
    propagator: Complex[Array, "nx ny"] = jnp.exp(
        -1j * jnp.pi * lam_ang * dz * (kx_grid**2 + ky_grid**2)
    )

    def _propagate_one_slice(
        psi_in: Complex[Array, "nx ny"],
        v_slice: Float[Array, "nx ny"],
    ) -> tuple[Complex[Array, "nx ny"], None]:
        """Propagate through one slice: transmit then propagate."""
        transmission: Complex[Array, "nx ny"] = jnp.exp(1j * sigma * v_slice)
        psi_transmitted: Complex[Array, "nx ny"] = psi_in * transmission
        psi_k: Complex[Array, "nx ny"] = jnp.fft.fft2(psi_transmitted)
        psi_k_propagated: Complex[Array, "nx ny"] = (
            psi_k * propagator * bandwidth_aperture
        )
        psi_out: Complex[Array, "nx ny"] = jnp.fft.ifft2(psi_k_propagated)
        return psi_out, None

    psi_exit: Complex[Array, "nx ny"]
    psi_exit, _ = jax.lax.scan(_propagate_one_slice, psi, v_slices)
    return psi_exit


@jaxtyped(typechecker=beartype)
def multislice_amplitude(
    potential_slices: PotentialSlices,
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> Complex[Array, "nx ny"]:
    """Return the coherent transmission multislice diffraction amplitude.

    Transmission geometry along z -- NOT valid for grazing-incidence RHEED;
    use the reflection pipeline (``kernel='multislice'`` via
    ``simulate_detector_image`` or ``reflection_multislice_simulator``).

    :see: :class:`~.test_multislice.TestMultisliceAmplitude`

    Parameters
    ----------
    potential_slices : PotentialSlices
        Projected potential slices with shape ``(nz, nx, ny)``.
    energy_kev : scalar_float
        Accelerating voltage in kilovolts.
    theta_deg : scalar_float
        Angle of the beam from the slice plane in degrees (near 90 for
        transmission; grazing values are rejected by the representability
        guard in :func:`multislice_propagate`).
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees. Default: 0.0.
    inner_potential_v0 : scalar_float, optional
        Inner potential used by :func:`multislice_propagate`. Default: 0.0.
    bandwidth_limit : scalar_float, optional
        Fraction of Nyquist frequency retained during propagation.
        Default: 2/3.

    Returns
    -------
    amplitude : Complex[Array, "nx ny"]
        Complex reciprocal-space exit-wave amplitude before the modulus
        squared reduction.

    Notes
    -----
    1. Propagate the tilted incident wave through all slices.
    2. Fourier transform the exit wave into reciprocal space.
    3. Return the complex field so Layer 1 can choose coherent or incoherent
       reduction explicitly.

    See Also
    --------
    multislice_simulator : Sparse transmission pattern using
        ``|amplitude|^2``.
    """
    exit_wave: Complex[Array, "nx ny"] = multislice_propagate(
        potential_slices=potential_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    amplitude: Complex[Array, "nx ny"] = jnp.fft.fft2(exit_wave)
    return amplitude


@jaxtyped(typechecker=beartype)
def _multislice_amplitude_pattern(  # noqa: PLR0913
    potential_slices: PotentialSlices,
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance_mm: scalar_float = 100.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> tuple[RHEEDPattern, Complex[Array, "N"]]:
    """Project transmission multislice amplitudes onto the detector.

    Transmission geometry along z -- NOT valid for grazing-incidence
    RHEED. Exit-wave momenta are ``k_in + 2 pi fftfreq`` (fftfreq is in
    cycles per Angstrom while ``k_in`` is in radians per Angstrom).
    """
    exit_wave_k: Complex[Array, "nx ny"] = multislice_amplitude(
        potential_slices=potential_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    nx: int = potential_slices.slices.shape[1]
    ny: int = potential_slices.slices.shape[2]
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, "nx ny"]
    ky_grid: Float[Array, "nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    lam_ang: scalar_float = wavelength_ang(energy_kev)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    polar_angle_rad: scalar_float
    azimuth_angle_rad: scalar_float
    polar_angle_rad, azimuth_angle_rad = incidence_angles_to_radians(
        theta_deg,
        phi_deg,
    )
    k_in: Float[Array, "3"] = k_mag * jnp.array(
        [
            jnp.cos(polar_angle_rad) * jnp.cos(azimuth_angle_rad),
            jnp.cos(polar_angle_rad) * jnp.sin(azimuth_angle_rad),
            jnp.sin(polar_angle_rad),
        ]
    )
    k_out_x: Float[Array, "nx ny"] = k_in[0] + 2.0 * jnp.pi * kx_grid
    k_out_y: Float[Array, "nx ny"] = k_in[1] + 2.0 * jnp.pi * ky_grid
    k_out_z_squared: Float[Array, "nx ny"] = k_mag**2 - k_out_x**2 - k_out_y**2
    valid_mask: Bool[Array, "nx ny"] = k_out_z_squared > 0
    k_out_z: Float[Array, "nx ny"] = jnp.where(
        valid_mask,
        jnp.sqrt(jnp.maximum(k_out_z_squared, 0.0)),
        0.0,
    )
    k_out_flat: Float[Array, "N 3"] = jnp.column_stack(
        [k_out_x.ravel(), k_out_y.ravel(), k_out_z.ravel()]
    )
    detector_points: Float[Array, "N 2"] = project_on_detector_geometry(
        k_out_flat,
        DetectorGeometry(distance=detector_distance_mm),
    )
    valid_flat: Bool[Array, "N"] = valid_mask.ravel()
    amplitudes: Complex[Array, "N"] = jnp.where(
        valid_flat,
        exit_wave_k.ravel(),
        0.0 + 0.0j,
    )
    max_intensity: Float[Array, ""] = jnp.maximum(
        jnp.max(jnp.abs(amplitudes) ** 2),
        1e-12,
    )
    normalized_amplitudes: Complex[Array, "N"] = amplitudes / jnp.sqrt(
        max_intensity
    )
    intensities: Float[Array, "N"] = jnp.abs(normalized_amplitudes) ** 2
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=jnp.arange(nx * ny, dtype=jnp.int32),
        k_out=k_out_flat,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern, normalized_amplitudes


@jaxtyped(typechecker=beartype)
def multislice_detector_amplitude(  # noqa: PLR0913
    potential_slices: PotentialSlices,
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance_mm: scalar_float = 100.0,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> Complex[Array, "H W"]:
    """Render transmission multislice amplitudes onto a detector field.

    Transmission geometry along z -- NOT valid for grazing-incidence
    RHEED; the public ``kernel='multislice'`` detector path uses
    :func:`rheedium.simul.reflection_detector_amplitude` instead.
    """
    pattern, amplitudes = _multislice_amplitude_pattern(
        potential_slices=potential_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        detector_distance_mm=detector_distance_mm,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    geometry: DetectorGeometry = DetectorGeometry(
        distance=detector_distance_mm,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
    )
    return render_amplitude_to_field(
        pattern=pattern,
        amplitudes=amplitudes,
        geometry=geometry,
        spot_sigma_px=spot_sigma_px,
    )


@jaxtyped(typechecker=beartype)
def multislice_simulator(
    potential_slices: PotentialSlices,
    energy_kev: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance: scalar_float = 100.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> RHEEDPattern:
    r"""Simulate a transmission diffraction pattern from potential slices.

    **Transmission geometry along z -- NOT valid for grazing-incidence
    RHEED.** The wave marches through x-y slices stacked along z exactly as
    in a TEM multislice calculation; for RHEED use the reflection pipeline
    (``simulate_detector_image`` with ``RenderParams(kernel="multislice")``
    or :func:`rheedium.simul.reflection_multislice_simulator`).

    The pipeline is:

    1. Propagate the electron wave through the slices
       (:func:`multislice_propagate`, including its representability
       guard against aliased tilts).
    2. Fourier transform the exit wave to get the reciprocal-space
       amplitude.
    3. Apply the Ewald-sphere constraint for elastic scattering, where
       :math:`|k_{out}| = |k_{in}| = 2\pi / \lambda` and
       :math:`k_{out,\{x,y\}} = k_{in,\{x,y\}} + 2\pi\,\mathrm{fftfreq}`
       (fftfreq is cycles per Angstrom, ``k_in`` is radians per Angstrom).
    4. Project diffracted beams onto the detector with
       :func:`project_on_detector_geometry` -- the same convention used by
       every other pattern producer in this module.
    5. Calculate the intensity as :math:`|\text{amplitude}|^2`.

    :see: :class:`~.test_simulator.TestMultisliceSimulator`

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials from
        sliced_crystal_to_projected_potential_slices()
    energy_kev : scalar_float
        Accelerating voltage in kilovolts
    theta_deg : scalar_float
        Angle of the beam from the x-y (slice) plane in degrees; near 90
        for a transmission calculation. Grazing values raise the
        representability guard in :func:`multislice_propagate`.
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
    detector_distance : scalar_float, optional
        Distance from sample to detector screen in mm (default: 100.0)
    inner_potential_v0 : scalar_float, optional
        Inner (mean) potential of the crystal in volts (default: 0.0).
        Surface refraction conserves the surface-parallel momentum (see
        :func:`multislice_propagate`).
    bandwidth_limit : scalar_float, optional
        Fraction of Nyquist frequency to retain (default: 2/3).
        Applied as a low-pass filter to prevent aliasing artifacts.

    Returns
    -------
    pattern : RHEEDPattern
        Transmission diffraction pattern with detector coordinates and
        intensities. All ``nx * ny`` reciprocal-grid channels are returned
        with fixed shapes: evanescent (non-propagating) channels are padded
        with ``g_indices == -1`` and zeroed ``intensities``, following the
        package's ``-1``-padded fixed-size convention (``k_out`` keeps the
        in-plane components with ``k_z = 0`` because the pattern validator
        requires non-zero wavevectors).

    Notes
    -----
    The multislice algorithm captures dynamical transmission diffraction
    effects: multiple scattering, absorption (with an imaginary potential),
    and thickness-dependent intensity oscillations.

    1. **Exit wave** --
       Propagate through all slices via
       :func:`multislice_propagate`.
    2. **Fourier transform** --
       FFT of exit wave to reciprocal space.
    3. **Ewald constraint** --
       :math:`k_{out,z}^2 = k^2 - k_{out,x}^2
       - k_{out,y}^2`; keep real solutions.
    4. **Detector projection** --
       :func:`project_on_detector_geometry` on the fixed-size channel set.
    5. **Assemble pattern** --
       Zero padded slots and create :class:`RHEEDPattern`.

    See Also
    --------
    multislice_propagate : Core propagation algorithm
    reflection_multislice_simulator : The RHEED (reflection) multislice path
    sliced_crystal_to_projected_potential_slices : Convert SlicedCrystal to
        projected-potential slices

    References
    ----------
    - Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy,
      2nd ed.
    - Cowley & Moodie (1957). Acta Cryst. 10, 609-619.
    """
    exit_wave_k: Complex[Array, "nx ny"] = multislice_amplitude(
        potential_slices=potential_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    nx: int = potential_slices.slices.shape[1]
    ny: int = potential_slices.slices.shape[2]
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, "nx ny"]
    ky_grid: Float[Array, "nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    lam_ang: scalar_float = wavelength_ang(energy_kev)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    polar_angle_rad: scalar_float
    azimuth_angle_rad: scalar_float
    polar_angle_rad, azimuth_angle_rad = incidence_angles_to_radians(
        theta_deg,
        phi_deg,
    )
    k_in: Float[Array, "3"] = k_mag * jnp.array(
        [
            jnp.cos(polar_angle_rad) * jnp.cos(azimuth_angle_rad),
            jnp.cos(polar_angle_rad) * jnp.sin(azimuth_angle_rad),
            jnp.sin(polar_angle_rad),
        ]
    )
    k_out_x: Float[Array, "nx ny"] = k_in[0] + 2.0 * jnp.pi * kx_grid
    k_out_y: Float[Array, "nx ny"] = k_in[1] + 2.0 * jnp.pi * ky_grid
    k_out_z_squared: Float[Array, "nx ny"] = k_mag**2 - k_out_x**2 - k_out_y**2
    valid_mask: Bool[Array, "nx ny"] = k_out_z_squared > 0
    k_out_z: Float[Array, "nx ny"] = jnp.where(
        valid_mask, jnp.sqrt(jnp.maximum(k_out_z_squared, 0.0)), 0.0
    )
    intensity_k: Float[Array, "nx ny"] = jnp.where(
        valid_mask, jnp.abs(exit_wave_k) ** 2, 0.0
    )
    valid_flat: Bool[Array, "n"] = valid_mask.ravel()
    k_out_flat: Float[Array, "n 3"] = jnp.column_stack(
        [k_out_x.ravel(), k_out_y.ravel(), k_out_z.ravel()]
    )
    detector_points: Float[Array, "n 2"] = project_on_detector_geometry(
        k_out_flat,
        DetectorGeometry(distance=detector_distance),
    )
    intensity_padded: Float[Array, "n"] = jnp.where(
        valid_flat, intensity_k.ravel(), 0.0
    )
    channel_indices: Int[Array, "n"] = jnp.arange(nx * ny, dtype=jnp.int32)
    grid_indices: Int[Array, "n"] = jnp.where(valid_flat, channel_indices, -1)
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=grid_indices,
        k_out=k_out_flat,
        detector_points=detector_points,
        intensities=intensity_padded,
    )
    return pattern


# --------------------------------------------------------------------------
# Opt-in runtime-checked simulator variants
# --------------------------------------------------------------------------
# Each ``checked_*`` below wraps a simulator with ``checkify.checkify``: a
# functional, JIT-compatible transform that runs the same computation while
# watching for floating-point faults, so a bad run fails loudly instead of
# silently propagating NaN/Inf.
#
# Contract (see jax.experimental.checkify): a checked call returns the pair
# ``(err, out)``, not just ``out``. Always surface failures with the error's
# ``throw`` method (or inspect it via ``get``); ignoring the error silently
# disables every check. Recommended order: apply eqx.filter_jit outside the
# wrapper at public boundaries.
#
# Enabled error sets: nan_checks (NaN/Inf from any op) and div_checks
# (division or remainder by zero). Both are automatic; no code changes needed.
#
# user_checks is intentionally NOT enabled. A checkify check placed inside a
# shared simulator raises a "not functionalized" error under plain jax.jit or
# jax.grad, which would break the raw, differentiable call path that does not
# go through checkify. Re-enable user_checks only alongside checks that live
# in code reached solely via these wrappers.
#
# Differentiability: these variants return ``(err, out)`` and so are not
# drop-in differentiable. Differentiate the raw simulators (ewald_simulator
# etc.); use the checked variants for validation or debugging runs.
_CHECKIFY_ERRORS = checkify.nan_checks | checkify.div_checks
checked_ewald_simulator = checkify.checkify(
    ewald_simulator,
    errors=_CHECKIFY_ERRORS,
)
checked_simulate_detector_image = checkify.checkify(
    simulate_detector_image,
    errors=_CHECKIFY_ERRORS,
)
checked_multislice_propagate = checkify.checkify(
    multislice_propagate,
    errors=_CHECKIFY_ERRORS,
)
checked_multislice_simulator = checkify.checkify(
    multislice_simulator,
    errors=_CHECKIFY_ERRORS,
)


__all__: list[str] = [
    "checked_ewald_simulator",
    "checked_multislice_propagate",
    "checked_multislice_simulator",
    "checked_simulate_detector_image",
    "compute_kinematic_intensities_with_ctrs",
    "detector_extent_mm",
    "ewald_simulator",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "kinematic_amplitude",
    "log_compress_image",
    "multislice_amplitude",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector_geometry",
    "render_amplitude_to_field",
    "render_ctr_amplitude_to_field",
    "render_pattern_to_image",
    "simulate_detector_image",
    "simulate_detector_image_instrument",
    "sliced_crystal_to_projected_potential_slices",
]
