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
:func:`ewald_simulator_with_orientation_distribution`
    Simulate and incoherently combine an orientation distribution of
    Ewald patterns.
:func:`ewald_simulator`
    Simulate RHEED using exact Ewald sphere-CTR intersection.
:func:`find_kinematic_reflections`
    Find reflections satisfying kinematic conditions.
:func:`kinematic_amplitude`
    Render a single coherent kinematic Ewald pattern as complex amplitude.
:func:`log_compress_image`
    Apply normalized log compression for screen-style visualization.
:func:`multislice_propagate`
    Propagate electron wave through potential slices using multislice
    algorithm.
:func:`multislice_amplitude`
    Return reciprocal-space multislice amplitude before modulus-squared.
:func:`multislice_simulator`
    Simulate RHEED pattern from potential slices using multislice
    (dynamical).
:func:`project_on_detector`
    Project wavevectors onto detector plane.
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

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Final, Tuple
from jax.core import Tracer
from jax.experimental import checkify
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.procs.surface_modifier import (
    bind_step_edge_distribution,
    bind_twin_wall_distribution,
)
from rheedium.tools import (
    gauss_hermite_nodes_weights,
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.types import (
    BeamModeDistribution,
    CrystalStructure,
    DetectorGeometry,
    Distribution,
    OrientationDistribution,
    PotentialSlices,
    ReductionMode,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    create_distribution,
    create_potential_slices,
    create_rheed_pattern,
    discretize_orientation,
    discretize_orientation_static,
    identify_surface_atoms,
    orientation_to_distribution,
    scalar_float,
    scalar_int,
    scalar_num,
)
from rheedium.ucell import reciprocal_lattice_vectors

from .beam_averaging import (
    apply_distribution,
    apply_distributions,
    decompose_beam_modes,
    detector_psf_convolve,
)
from .form_factors import (
    atomic_scattering_factor,
    projected_potential,
)
from .surface_rods import integrated_ctr_amplitude, integrated_rod_intensity

_VALID_THRESHOLD: Final[float] = 0.5
_KINEMATIC_KERNEL: Final[str] = "kinematic"
_AZIMUTH_AXIS_IDS: Final[frozenset[str]] = frozenset(
    {"trivial", "orientation", "azimuth", "phi", "test_phi"}
)
_BEAM_AXIS_IDS: Final[frozenset[str]] = frozenset(
    {"beam_modes", "coherent_beam", "legacy_instrument"}
)
_STRUCTURE_AXIS_IDS: Final[frozenset[str]] = frozenset({"twins", "steps"})
_GRAIN_AXIS_IDS: Final[frozenset[str]] = frozenset({"grains"})
_UNSUPPORTED_AXIS_IDS: Final[frozenset[str]] = frozenset({"size"})


def _validate_layer0_kernel(kernel: str) -> None:
    """Validate the public Layer-0 kernel selector."""
    if kernel != _KINEMATIC_KERNEL:
        raise ValueError(
            "Unsupported kernel. Only 'kinematic' is currently implemented."
        )


@jaxtyped(typechecker=beartype)
def _compose_ewald_intensity(
    sf_intensity: scalar_float,
    l_value: scalar_float,
    q_z: scalar_float,
    surface_roughness: scalar_float,
    ctr_regularization: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Combine sparse Ewald intensity terms with tunable powers."""
    sin_pi_l: Float[Array, ""] = jnp.sin(jnp.pi * l_value)
    ctr_modulation: Float[Array, ""] = 1.0 / (sin_pi_l**2 + ctr_regularization)
    roughness_damping: Float[Array, ""] = jnp.exp(
        -0.5 * (surface_roughness**2) * q_z**2
    )
    total_intensity: Float[Array, ""] = (
        sf_intensity
        * ctr_modulation**ctr_power
        * roughness_damping**roughness_power
    )
    return total_intensity, ctr_modulation, roughness_damping


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out: Float[Array, "N 3"],
    detector_distance: scalar_float,
) -> Float[Array, "N 2"]:
    r"""Project output wavevectors onto detector plane.

    Uses ray-tracing projection to a vertical detector screen at distance d.
    The scale factor is computed as d/k_x (with small epsilon to avoid
    division by zero), then multiplied by k_y and k_z to get horizontal
    and vertical detector coordinates.

    :see: :class:`~.test_simulator.TestProjectOnDetector`

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Output wavevectors.
    detector_distance : scalar_float
        Distance from sample to detector in mm.

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [horizontal, vertical] coordinates on detector in mm.

    Notes
    -----
    1. **Scale factor** --
       :math:`s = d / (k_x + \\epsilon)` for each
       wavevector.
    2. **Detector coordinates** --
       :math:`(k_y \\times s,\\; k_z \\times s)`.

    See Also
    --------
    project_on_detector_geometry : Projection with tilt and curvature support.
    """
    scale: Float[Array, "N"] = detector_distance / (k_out[:, 0] + 1e-10)
    detector_h: Float[Array, "N"] = k_out[:, 1] * scale
    detector_v: Float[Array, "N"] = k_out[:, 2] * scale
    detector_coords: Float[Array, "N 2"] = jnp.stack(
        [detector_h, detector_v], axis=-1
    )
    return detector_coords


@jaxtyped(typechecker=beartype)
def project_on_detector_geometry(
    k_out: Float[Array, "N 3"],
    geometry: DetectorGeometry,
) -> Float[Array, "N 2"]:
    """Project output wavevectors onto detector with full geometry support.

    This function extends the basic projection to support tilted and curved
    detector screens. For a flat, untilted detector at the default distance,
    this is equivalent to `project_on_detector`.

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
    For small tilt angles and infinite curvature, this reduces
    to the simple ray-tracing formula in
    :func:`project_on_detector`.

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
    project_on_detector : Simple projection for flat, untilted detectors.
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


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, "3"],
    gs: Float[Array, "M 3"],
    z_sign: scalar_float = 1.0,
    tolerance: scalar_float = 0.05,
) -> Tuple[Int[Array, "N"], Float[Array, "N 3"]]:
    """Find kinematically allowed reflections.

    Computes k_out = k_in + G for all reciprocal lattice vectors G, then
    filters based on elastic scattering condition |k_out| ≈ |k_in| and
    z-direction constraint. Returns fixed-size arrays for JIT compatibility,
    with -1 marking invalid entries.

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
    tolerance : scalar_float, optional
        Tolerance for reflection condition :math:`|k_{out}| = |k_{in}|`.
        Default: 0.05

    Returns
    -------
    allowed_indices : Int[Array, "M"]
        Indices of allowed reflections in gs array. Invalid entries are -1.
        Use `allowed_indices >= 0` to filter valid results.
    k_out : Float[Array, "M 3"]
        Output wavevectors for allowed reflections. Invalid entries
        correspond to `allowed_indices == -1`.

    Notes
    -----
    Returns fixed-size arrays for JIT compatibility. Filter results using:
        valid_mask = allowed_indices >= 0
        valid_indices = allowed_indices[valid_mask]
        valid_k_out = k_out[valid_mask]

    1. **Outgoing wavevectors** --
       :math:`k_{out} = k_{in} + G` for all G vectors.
    2. **Elastic condition** --
       Filter by :math:`||k_{out}| - |k_{in}|| <` tolerance.
    3. **z-direction filter** --
       Keep reflections with correct z-sign.
    4. **Fixed-size output** --
       Use :func:`jnp.where` with fill_value for JIT
       compatibility.

    See Also
    --------
    incident_wavevector : Calculate incident wavevector from angles.
    generate_reciprocal_points : Generate reciprocal lattice vectors.
    """
    k_out_all: Float[Array, "M 3"] = k_in + gs
    k_in_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    k_out_mags: Float[Array, "M"] = jnp.linalg.norm(k_out_all, axis=1)
    elastic_condition: Bool[Array, "M"] = (
        jnp.abs(k_out_mags - k_in_mag) < tolerance
    )
    z_condition: Bool[Array, "M"] = k_out_all[:, 2] * z_sign > 0
    allowed: Bool[Array, "M"] = elastic_condition & z_condition
    allowed_indices: Int[Array, "M"] = jnp.where(
        allowed, size=gs.shape[0], fill_value=-1
    )[0]
    safe_indices: Int[Array, "M"] = jnp.maximum(allowed_indices, 0)
    k_out: Float[Array, "M 3"] = k_out_all[safe_indices]
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
    detector_acceptance: scalar_float = 0.01,
    surface_fraction: scalar_float = 0.3,
    surface_config: SurfaceConfig | None = None,
    ctr_mixing_mode: str = "incoherent",
    ctr_weight: scalar_float = 1.0,
    hk_tolerance: scalar_float = 0.1,
    parameterization: str = "lobato",
) -> Float[Array, "N"]:
    """Calculate kinematic diffraction intensities with CTR contributions.

    For each reflection, computes the structure factor by summing atomic
    contributions (form factor × phase factor). The phase is computed as
    G·r where G vectors already include the 2π factor from reciprocal
    lattice generation. CTR contributions are mixed according to the
    specified mode.

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
    detector_acceptance : scalar_float, optional
        Detector angular acceptance in reciprocal angstroms.
        Default: 0.01
    surface_fraction : scalar_float, optional
        Fraction of atoms considered as surface atoms (for backward
        compatibility). Used only if surface_config is None.
        Default: 0.3
    surface_config : SurfaceConfig | None, optional
        Configuration for surface atom identification. Supports multiple
        methods: "height" (default), "coordination", "layers", "explicit".
        If None, uses height-based method with surface_fraction parameter.
    ctr_mixing_mode : str, optional
        How to combine kinematic and CTR contributions:
        - "coherent": Add complex amplitudes, then square (interference)
        - "incoherent": Add intensities directly (no interference)
        - "none": Only kinematic scattering, no CTR contribution
        Default: "incoherent"
    ctr_weight : scalar_float, optional
        Weight factor for CTR contribution (0.0-1.0). Controls the
        relative strength of streak vs spot intensity.
        Default: 1.0
    hk_tolerance : scalar_float, optional
        Tolerance for validating near-integer h,k indices. CTR is only
        applied when |h - round(h)| < tolerance and same for k.
        Default: 0.1
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    intensities : Float[Array, "N"]
        Diffraction intensities for each allowed reflection.

    Notes
    -----
    The coherent mode is physically more accurate as it accounts for
    interference between kinematic scattering and CTR contributions.
    However, incoherent mode may be more stable numerically and is
    the historical default behavior.

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
       evaluate structure factor with form factors and
       Debye-Waller, gate CTR on near-integer (h, k).
    4. **Mix contributions** --
       Coherent (add amplitudes), incoherent (add
       intensities), or no CTR.

    See Also
    --------
    atomic_scattering_factor : Compute form factors with Debye-Waller.
    integrated_ctr_amplitude : CTR amplitude for coherent mixing.
    integrated_rod_intensity : CTR intensity for incoherent mixing.
    identify_surface_atoms : Identify surface atoms from positions.
    """
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    # Recover Miller indices for CTR gating; prefer explicit indices when given
    hkl_all: Float[Array, "N 3"]
    if hkl_indices is not None:
        hkl_all = jnp.asarray(hkl_indices, dtype=jnp.float64)
    else:
        recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
            *crystal.cell_lengths,
            *crystal.cell_angles,
            in_degrees=True,
        )
        inv_recip: Float[Array, "3 3"] = jnp.linalg.inv(recip_vecs)
        hkl_all = jnp.matmul(g_allowed, inv_recip)

    # Use provided config or create one from surface_fraction for compatibility
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="height", height_fraction=surface_fraction)
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
            form_factor: scalar_float = jnp.squeeze(
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

        # Validate h,k are near-integer before applying CTR
        h_val: Float[Array, ""] = hkl_all[idx, 0]
        k_val: Float[Array, ""] = hkl_all[idx, 1]
        h_deviation: Float[Array, ""] = jnp.abs(h_val - jnp.round(h_val))
        k_deviation: Float[Array, ""] = jnp.abs(k_val - jnp.round(k_val))
        is_near_integer: Bool[Array, ""] = (h_deviation < hk_tolerance) & (
            k_deviation < hk_tolerance
        )

        hk_index: Int[Array, "2"] = jnp.array(
            [
                jnp.round(h_val).astype(jnp.int32),
                jnp.round(k_val).astype(jnp.int32),
            ]
        )
        q_z_value: Float[Array, ""] = q_vector[2]
        q_z_range: Float[Array, "2"] = jnp.array(
            [q_z_value - detector_acceptance, q_z_value + detector_acceptance]
        )

        # Calculate intensity based on mixing mode
        kinematic_intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2

        if ctr_mixing_mode == "none":
            # No CTR contribution
            total_intensity: Float[Array, ""] = kinematic_intensity
        elif ctr_mixing_mode == "coherent":
            # Coherent mixing: add complex amplitudes, then square
            ctr_amplitude: Complex[Array, ""] = integrated_ctr_amplitude(
                hk_index=hk_index,
                q_z_range=q_z_range,
                crystal=crystal,
                surface_roughness=surface_roughness,
                detector_acceptance=detector_acceptance,
                temperature=temperature,
                parameterization=parameterization,
            )
            # Apply weight and near-integer mask
            weighted_ctr: Complex[Array, ""] = (
                ctr_weight * ctr_amplitude * is_near_integer
            )
            total_amplitude: Complex[Array, ""] = (
                structure_factor + weighted_ctr
            )
            total_intensity: Float[Array, ""] = jnp.abs(total_amplitude) ** 2
        else:
            # Incoherent mixing (default): add intensities
            ctr_intensity: Float[Array, ""] = integrated_rod_intensity(
                hk_index=hk_index,
                q_z_range=q_z_range,
                crystal=crystal,
                surface_roughness=surface_roughness,
                detector_acceptance=detector_acceptance,
                temperature=temperature,
                parameterization=parameterization,
            )
            # Apply weight and near-integer mask
            weighted_ctr_intensity: Float[Array, ""] = (
                ctr_weight * ctr_intensity * is_near_integer
            )
            total_intensity: Float[Array, ""] = (
                kinematic_intensity + weighted_ctr_intensity
            )

        return total_intensity

    n_reflections: Int[Array, ""] = g_allowed.shape[0]
    reflection_indices: Int[Array, "N"] = jnp.arange(n_reflections)
    intensities: Float[Array, "N"] = jax.vmap(_calculate_reflection_intensity)(
        reflection_indices
    )
    return intensities


@jaxtyped(typechecker=beartype)
def find_ctr_ewald_intersection(
    h: scalar_int,
    k: scalar_int,
    k_in: Float[Array, "3"],
    recip_a: Float[Array, "3"],
    recip_b: Float[Array, "3"],
    recip_c: Float[Array, "3"],
) -> Tuple[Float[Array, "2"], Float[Array, "2 3"], Float[Array, "2"]]:
    """Find where a crystal truncation rod intersects the Ewald sphere.

    Solves the quadratic rod-sphere equation and returns both valid
    upward-scattering branches, if present.

    Parameters
    ----------
    h : scalar_int
        Miller index h for the rod.
    k : scalar_int
        Miller index k for the rod.
    k_in : Float[Array, "3"]
        Incident wavevector.
    recip_a : Float[Array, "3"]
        First reciprocal lattice vector (a*).
    recip_b : Float[Array, "3"]
        Second reciprocal lattice vector (b*).
    recip_c : Float[Array, "3"]
        Third reciprocal lattice vector (c*), defines rod direction.

    Returns
    -------
    l_intersect : Float[Array, "2"]
        The plus and minus branch l values, zeroed when invalid.
    k_out : Float[Array, "2 3"]
        Outgoing wavevectors for the plus and minus branches.
    valid : Float[Array, "2"]
        Branch validity flags as floats in ``{0.0, 1.0}``.
    """
    g_hk: Float[Array, "3"] = h * recip_a + k * recip_b
    c_star: Float[Array, "3"] = recip_c
    k_mag_sq: Float[Array, ""] = jnp.dot(k_in, k_in)
    p_vec: Float[Array, "3"] = k_in + g_hk
    a_coef: Float[Array, ""] = jnp.dot(c_star, c_star)
    b_coef: Float[Array, ""] = 2.0 * jnp.dot(p_vec, c_star)
    c_coef: Float[Array, ""] = jnp.dot(p_vec, p_vec) - k_mag_sq
    discriminant: Float[Array, ""] = b_coef**2 - 4.0 * a_coef * c_coef
    safe_disc: Float[Array, ""] = jnp.maximum(discriminant, 1e-30)
    sqrt_disc: Float[Array, ""] = jnp.sqrt(safe_disc)
    l_plus: Float[Array, ""] = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    l_minus: Float[Array, ""] = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    k_out_plus: Float[Array, "3"] = p_vec + l_plus * c_star
    k_out_minus: Float[Array, "3"] = p_vec + l_minus * c_star
    valid_plus: Float[Array, ""] = (discriminant >= 0) & (k_out_plus[2] > 0)
    valid_minus: Float[Array, ""] = (discriminant >= 0) & (k_out_minus[2] > 0)
    l_intersect: Float[Array, "2"] = jnp.array([l_plus, l_minus])
    k_out: Float[Array, "2 3"] = jnp.stack([k_out_plus, k_out_minus], axis=0)
    valid_mask: Bool[Array, "2"] = jnp.array([valid_plus, valid_minus])
    l_intersect = jnp.where(valid_mask, l_intersect, 0.0)
    k_out = jnp.where(valid_mask[:, None], k_out, 0.0)
    valid: Float[Array, "2"] = valid_mask.astype(jnp.float64)
    return l_intersect, k_out, valid


@jaxtyped(typechecker=beartype)
def ewald_simulator(  # noqa: PLR0913, PLR0915
    crystal: CrystalStructure,
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
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
    voltage_kv : scalar_num, optional
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
    ctr_regularization : scalar_float, optional
        Additive regularization in the CTR factor ``1 / (sin^2(pi l) + eps)``.
        Default: 0.01
    ctr_power : scalar_float, optional
        Exponent applied to the CTR modulation term. Set to ``0.0`` to disable
        CTR weighting while leaving the geometry unchanged. Default: 1.0
    roughness_power : scalar_float, optional
        Exponent applied to the roughness damping term. Set to ``0.0`` to
        disable roughness damping while leaving the geometry unchanged.
        Default: 0.25
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.
    surface_config : SurfaceConfig | None, optional
        Configuration for surface atom identification. Default: None

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
    voltage_kv: Float[Array, ""] = jnp.asarray(voltage_kv)
    theta_deg: Float[Array, ""] = jnp.asarray(theta_deg)
    phi_deg: Float[Array, ""] = jnp.asarray(phi_deg)
    # Keep hmax/kmax as Python ints for static array sizing in JIT
    hmax_int: int = int(hmax)
    kmax_int: int = int(kmax)

    # Compute incident wavevector
    lam_ang: Float[Array, ""] = wavelength_ang(voltage_kv)
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
    n_atoms: int = atom_positions.shape[0]

    # Surface atom identification
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="height", height_fraction=0.3)
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
            ff: Float[Array, ""] = jnp.squeeze(
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
            ctr_regularization=ctr_regularization,
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
    detector_points: Float[Array, "K 2"] = project_on_detector(
        k_out, detector_distance
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
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
) -> Tuple[RHEEDPattern, Complex[Array, "K"]]:
    """Return sparse Ewald pattern plus complex per-reflection amplitudes."""
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    voltage_kv_arr: Float[Array, ""] = jnp.asarray(voltage_kv)
    theta_deg_arr: Float[Array, ""] = jnp.asarray(theta_deg)
    phi_deg_arr: Float[Array, ""] = jnp.asarray(phi_deg)
    hmax_int: int = int(hmax)
    kmax_int: int = int(kmax)

    lam_ang: Float[Array, ""] = wavelength_ang(voltage_kv_arr)
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
    n_atoms: int = atom_positions.shape[0]
    config: SurfaceConfig = (
        surface_config
        if surface_config is not None
        else SurfaceConfig(method="height", height_fraction=0.3)
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
            form_factor: Float[Array, ""] = jnp.squeeze(
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
        _, ctr_modulation, roughness_damping = _compose_ewald_intensity(
            sf_intensity=jnp.abs(structure_factor) ** 2,
            l_value=l_val,
            q_z=q_z,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
        )
        amplitude_scale: Float[Array, ""] = ctr_modulation ** (
            0.5 * ctr_power
        ) * roughness_damping ** (0.5 * roughness_power)
        amplitude: Complex[Array, ""] = structure_factor * amplitude_scale
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
    detector_points: Float[Array, "K 2"] = project_on_detector(
        k_out,
        detector_distance,
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
def _discretize_orientation_for_sparse_pattern(
    orientation_distribution: OrientationDistribution,
    n_mosaic_points: scalar_int,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Discretize orientations while keeping eager sparse patterns compact."""
    if isinstance(orientation_distribution.mosaic_fwhm_deg, Tracer):
        return discretize_orientation(
            orientation_distribution,
            n_mosaic_points=n_mosaic_points,
        )
    return discretize_orientation_static(
        orientation_distribution,
        n_mosaic_points=n_mosaic_points,
    )


@jaxtyped(typechecker=beartype)
def _incoherent_pattern_union(
    pattern_bank: RHEEDPattern,
    weights: Float[Array, "N_orientations"],
) -> RHEEDPattern:
    """Flatten a weighted bank of sparse patterns into one pattern."""
    weighted_intensities: Float[Array, "N_orientations N_reflections"] = (
        pattern_bank.intensities * weights[:, None]
    )
    return create_rheed_pattern(
        g_indices=pattern_bank.G_indices.reshape(-1),
        k_out=pattern_bank.k_out.reshape(-1, 3),
        detector_points=pattern_bank.detector_points.reshape(-1, 2),
        intensities=weighted_intensities.reshape(-1),
    )


@jaxtyped(typechecker=beartype)
def ewald_simulator_with_orientation_distribution(  # noqa: PLR0913
    crystal: CrystalStructure,
    orientation_distribution: OrientationDistribution,
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    n_mosaic_points: scalar_int = 7,
) -> RHEEDPattern:
    r"""Simulate a weighted union of Ewald patterns over orientations.

    This wrapper promotes :class:`~rheedium.types.OrientationDistribution`
    to a first-class simulator input. The azimuthal support is discretized,
    :func:`ewald_simulator` is evaluated at each quadrature angle, and the
    resulting sparse spot patterns are combined as an incoherent detector
    intensity sum.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    orientation_distribution : OrientationDistribution
        Probability distribution over azimuthal orientations in degrees.
    voltage_kv : scalar_num, optional
        Electron beam energy in kiloelectron volts. Default: 20.0
    theta_deg : scalar_num, optional
        Grazing angle of incidence in degrees. Default: 2.0
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
    ctr_regularization : scalar_float, optional
        Additive regularization in the sparse CTR intensity factor.
        Default: 0.01
    ctr_power : scalar_float, optional
        Exponent applied to the CTR modulation term. Default: 1.0
    roughness_power : scalar_float, optional
        Exponent applied to the roughness damping term. Default: 0.25
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.
    surface_config : SurfaceConfig | None, optional
        Configuration for surface atom identification. Default: None
    n_mosaic_points : scalar_int, optional
        Quadrature points per orientation peak for mosaic broadening.
        Default: 7. For purely discrete variants, ``1`` gives the most
        compact sparse output.

    Returns
    -------
    pattern : RHEEDPattern
        Sparse detector pattern containing the weighted union of all
        orientation-specific reflections. Intensities add incoherently.

    Notes
    -----
    The output keeps one detector spot entry per simulated orientation sample.
    Reflections from different orientations are not merged by Miller index,
    because their detector coordinates generally differ. Downstream detector
    rendering should therefore sum the returned spot intensities directly.
    """
    angles_deg: Float[Array, "N"]
    weights: Float[Array, "N"]
    angles_deg, weights = _discretize_orientation_for_sparse_pattern(
        orientation_distribution,
        n_mosaic_points=n_mosaic_points,
    )

    def _simulate_at_orientation(phi_deg: scalar_float) -> RHEEDPattern:
        return ewald_simulator(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            parameterization=parameterization,
            surface_config=surface_config,
        )

    pattern_bank: RHEEDPattern = jax.vmap(_simulate_at_orientation)(angles_deg)
    return _incoherent_pattern_union(pattern_bank, weights)


@jaxtyped(typechecker=beartype)
def render_pattern_to_image(
    pattern: RHEEDPattern,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
) -> Float[Array, "H W"]:
    """Rasterize a sparse RHEEDPattern onto a dense detector image.

    Parameters
    ----------
    pattern : RHEEDPattern
        Sparse detector pattern with positions in millimetres.
    image_shape_px : Tuple[int, int]
        Output detector image shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float]
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)`` as
        ``(center_x_px, center_y_px)``.
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
    height_px, width_px = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px
    sigma_px: Float[Array, ""] = jnp.asarray(spot_sigma_px, dtype=jnp.float64)

    x_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 0] / x_mm_per_px + center_x_px
    )
    y_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 1] / y_mm_per_px + center_y_px
    )
    intensities: Float[Array, "N"] = jnp.asarray(
        pattern.intensities, dtype=jnp.float64
    )

    x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")

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
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
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
    image_shape_px : Tuple[int, int]
        Output detector field shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float]
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.
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
    height_px, width_px = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px
    sigma_px: Float[Array, ""] = jnp.asarray(spot_sigma_px, dtype=jnp.float64)
    amplitudes_arr: Complex[Array, "N"] = jnp.asarray(
        amplitudes,
        dtype=jnp.complex128,
    )
    if amplitudes_arr.shape[0] != pattern.detector_points.shape[0]:
        raise ValueError("amplitudes length must match detector_points")

    x_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 0] / x_mm_per_px + center_x_px
    )
    y_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 1] / y_mm_per_px + center_y_px
    )
    x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")

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
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    render_ctrs_as_streaks: bool = False,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
) -> Complex[Array, "H W"]:
    """Render one coherent kinematic Ewald pattern as detector amplitude.

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
    voltage_kv, theta_deg, phi_deg : scalar_num, optional
        Beam energy and incidence geometry.
    hmax, kmax : scalar_int, optional
        In-plane CTR grid bounds.
    detector_distance_mm : scalar_float, optional
        Sample-to-detector distance in millimetres.
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller damping.
    surface_roughness : scalar_float, optional
        RMS surface roughness in Ångstroms.
    ctr_regularization, ctr_power, roughness_power : scalar_float, optional
        CTR and roughness weighting controls passed to
        :func:`ewald_simulator`.
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
    sparse_pattern: RHEEDPattern
    amplitudes: Complex[Array, "N"]
    sparse_pattern, amplitudes = _ewald_amplitude_pattern(
        crystal=crystal,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        hmax=hmax,
        kmax=kmax,
        detector_distance=detector_distance_mm,
        temperature=temperature,
        surface_roughness=surface_roughness,
        ctr_regularization=ctr_regularization,
        ctr_power=ctr_power,
        roughness_power=roughness_power,
        parameterization=parameterization,
        surface_config=surface_config,
    )
    if render_ctrs_as_streaks:
        return render_ctr_amplitude_to_field(
            pattern=sparse_pattern,
            amplitudes=amplitudes,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
        )
    return render_amplitude_to_field(
        pattern=sparse_pattern,
        amplitudes=amplitudes,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
        spot_sigma_px=spot_sigma_px,
    )


@jaxtyped(typechecker=beartype)
def render_ctr_amplitude_to_field(
    pattern: RHEEDPattern,
    amplitudes: Complex[Array, "N"],
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
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
    image_shape_px : Tuple[int, int]
        Output detector field shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float]
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.
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
    height_px, width_px = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px
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
    x_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 0] / x_mm_per_px + center_x_px
    )
    y_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 1] / y_mm_per_px + center_y_px
    )
    valid_amplitudes: Complex[Array, "N"] = jnp.where(
        valid_mask,
        amplitudes_arr,
        0.0 + 0.0j,
    )
    x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")

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
def _render_ctr_streaks_to_image(
    pattern: RHEEDPattern,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
) -> Float[Array, "H W"]:
    """Render detector-anchored streaks with visible Bragg spot cores."""
    height_px, width_px = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px
    sigma_x_px: Float[Array, ""] = jnp.asarray(
        spot_sigma_px,
        dtype=jnp.float64,
    )
    sigma_y_px: Float[Array, ""] = jnp.maximum(4.0 * sigma_x_px, 2.5)
    spot_core_weight: Float[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
    streak_halo_weight: Float[Array, ""] = jnp.asarray(0.6, dtype=jnp.float64)

    valid_mask: Bool[Array, "N"] = pattern.G_indices >= 0
    x_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 0] / x_mm_per_px + center_x_px
    )
    y_pixels: Float[Array, "N"] = (
        pattern.detector_points[:, 1] / y_mm_per_px + center_y_px
    )
    intensities: Float[Array, "N"] = jnp.where(
        valid_mask,
        jnp.asarray(pattern.intensities, dtype=jnp.float64),
        0.0,
    )

    x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")

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
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Compute matplotlib-style detector extent in millimetres.

    Parameters
    ----------
    image_shape_px : Tuple[int, int]
        Detector image shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float]
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.

    Returns
    -------
    extent_mm : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` extent in millimetres.
    """
    height_px, width_px = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px
    return (
        -center_x_px * x_mm_per_px,
        (width_px - center_x_px) * x_mm_per_px,
        -center_y_px * y_mm_per_px,
        (height_px - center_y_px) * y_mm_per_px,
    )


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
def _bind_kinematic_distribution(  # noqa: PLR0913, PLR0915
    distribution: Distribution,
    crystal: CrystalStructure,
    voltage_kv: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    ctr_regularization: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    render_ctrs_as_streaks: bool,
    parameterization: str,
    surface_config: SurfaceConfig | None,
    defect_surface_layer_depth_angstrom: scalar_float,
) -> Callable[[Float[Array, "D"]], Complex[Array, "H W"]]:
    """Bind distribution samples to the kinematic amplitude kernel."""
    axis_id: str | None = distribution.axis_id
    sample_dim: int = distribution.samples.shape[1]
    theta_nominal: Float[Array, ""] = jnp.asarray(theta_deg, dtype=jnp.float64)
    phi_nominal: Float[Array, ""] = jnp.asarray(phi_deg, dtype=jnp.float64)
    voltage_nominal: Float[Array, ""] = jnp.asarray(
        voltage_kv,
        dtype=jnp.float64,
    )

    def _call_kinematic(
        sample_crystal: CrystalStructure,
        sample_voltage_kv: scalar_num,
        sample_theta_deg: scalar_num,
        sample_phi_deg: scalar_num,
    ) -> Complex[Array, "H W"]:
        return kinematic_amplitude(
            crystal=sample_crystal,
            voltage_kv=sample_voltage_kv,
            theta_deg=sample_theta_deg,
            phi_deg=sample_phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
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

    if axis_id in _UNSUPPORTED_AXIS_IDS:
        raise ValueError(
            f"Distribution axis {axis_id!r} has no detector-image bind yet; "
            "use its dedicated finite-domain API until the shared size "
            "kernel lands."
        )

    if axis_id in _BEAM_AXIS_IDS:
        if sample_dim != 3:
            raise ValueError(
                f"Beam-like distribution axis {axis_id!r} requires samples "
                "with shape (N, 3)."
            )

        def _beam_bound(sample: Float[Array, "3"]) -> Complex[Array, "H W"]:
            theta_sample_deg: Float[Array, ""] = theta_nominal + jnp.rad2deg(
                sample[0]
            )
            phi_sample_deg: Float[Array, ""] = phi_nominal + jnp.rad2deg(
                sample[1]
            )
            voltage_sample_kv: Float[Array, ""] = (
                voltage_nominal + 1.0e-3 * sample[2]
            )
            return _call_kinematic(
                crystal,
                voltage_sample_kv,
                theta_sample_deg,
                phi_sample_deg,
            )

        return _beam_bound

    if axis_id in _STRUCTURE_AXIS_IDS:
        if axis_id == "twins":
            structure_builder: Callable[
                [Float[Array, "2"]],
                CrystalStructure,
            ] = bind_twin_wall_distribution(
                slab=crystal,
                surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
            )

            def _twin_bound(
                sample: Float[Array, "2"],
            ) -> Complex[Array, "H W"]:
                return _call_kinematic(
                    structure_builder(sample),
                    voltage_kv,
                    theta_deg,
                    phi_deg,
                )

            return _twin_bound

        structure_builder_step: Callable[
            [Float[Array, "3"]],
            CrystalStructure,
        ] = bind_step_edge_distribution(
            slab=crystal,
            surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
        )

        def _step_bound(sample: Float[Array, "3"]) -> Complex[Array, "H W"]:
            return _call_kinematic(
                structure_builder_step(sample),
                voltage_kv,
                theta_deg,
                phi_deg,
            )

        return _step_bound

    if axis_id in _GRAIN_AXIS_IDS:
        if sample_dim != 2:
            raise ValueError(
                "Grain distributions require samples "
                "[orientation_angle_deg, grain_size_angstrom]."
            )

        def _grain_bound(sample: Float[Array, "2"]) -> Complex[Array, "H W"]:
            return _call_kinematic(
                crystal,
                voltage_kv,
                theta_deg,
                phi_deg + sample[0],
            )

        return _grain_bound

    if axis_id in _AZIMUTH_AXIS_IDS or (axis_id is None and sample_dim == 1):

        def _azimuth_bound(sample: Float[Array, "1"]) -> Complex[Array, "H W"]:
            return _call_kinematic(
                crystal,
                voltage_kv,
                theta_deg,
                phi_deg + sample[0],
            )

        return _azimuth_bound

    raise ValueError(
        f"Distribution axis {axis_id!r} with sample dimension {sample_dim} "
        "does not have a registered kinematic bind."
    )


@jaxtyped(typechecker=beartype)
def _apply_kinematic_distribution(  # noqa: PLR0913
    distribution: Distribution,
    crystal: CrystalStructure,
    voltage_kv: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    ctr_regularization: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    render_ctrs_as_streaks: bool,
    parameterization: str,
    surface_config: SurfaceConfig | None,
    defect_surface_layer_depth_angstrom: scalar_float = 1.0,
) -> Float[Array, "H W"]:
    """Apply a registered distribution bind to the kinematic kernel."""
    bound: Callable[[Float[Array, "D"]], Complex[Array, "H W"]] = (
        _bind_kinematic_distribution(
            distribution=distribution,
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
            parameterization=parameterization,
            surface_config=surface_config,
            defect_surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
        )
    )

    return apply_distribution(distribution, bound)


@jaxtyped(typechecker=beartype)
def _bind_kinematic_distributions(  # noqa: PLR0913, PLR0915
    distributions: tuple[Distribution, ...],
    crystal: CrystalStructure,
    voltage_kv: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    ctr_regularization: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
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
    axis_ids: tuple[str | None, ...] = tuple(
        distribution.axis_id for distribution in distributions
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

    for axis_id, sample_dim in zip(axis_ids, axis_dims, strict=True):
        if axis_id in _UNSUPPORTED_AXIS_IDS:
            raise ValueError(
                f"Distribution axis {axis_id!r} has no detector-image bind "
                "yet; use its dedicated finite-domain API until the shared "
                "size kernel lands."
            )
        if axis_id in _BEAM_AXIS_IDS and sample_dim != 3:
            raise ValueError(
                f"Beam-like distribution axis {axis_id!r} requires samples "
                "with shape (N, 3)."
            )
        if axis_id == "twins" and sample_dim != 2:
            raise ValueError("Twin distributions require samples (N, 2).")
        if axis_id == "steps" and sample_dim != 3:
            raise ValueError("Step distributions require samples (N, 3).")
        if axis_id in _GRAIN_AXIS_IDS and sample_dim != 2:
            raise ValueError(
                "Grain distributions require samples "
                "[orientation_angle_deg, grain_size_angstrom]."
            )
        if (
            axis_id not in _BEAM_AXIS_IDS
            and axis_id not in _STRUCTURE_AXIS_IDS
            and axis_id not in _GRAIN_AXIS_IDS
            and axis_id not in _AZIMUTH_AXIS_IDS
            and not (axis_id is None and sample_dim == 1)
        ):
            raise ValueError(
                f"Distribution axis {axis_id!r} with sample dimension "
                f"{sample_dim} does not have a registered kinematic bind."
            )

    def _bound(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
        sample_crystal: CrystalStructure = crystal
        sample_voltage_kv: Float[Array, ""] = jnp.asarray(
            voltage_kv,
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
        cursor: int = 0
        for axis_id, sample_dim in zip(axis_ids, axis_dims, strict=True):
            axis_sample: Float[Array, "D_axis"] = sample[
                cursor : cursor + sample_dim
            ]
            cursor += sample_dim
            if axis_id in _BEAM_AXIS_IDS:
                sample_theta_deg = sample_theta_deg + jnp.rad2deg(
                    axis_sample[0]
                )
                sample_phi_deg = sample_phi_deg + jnp.rad2deg(axis_sample[1])
                sample_voltage_kv = sample_voltage_kv + 1.0e-3 * axis_sample[2]
            elif axis_id == "twins":
                sample_crystal = twin_builder(axis_sample)
            elif axis_id == "steps":
                sample_crystal = step_builder(axis_sample)
            elif axis_id in _GRAIN_AXIS_IDS:
                sample_phi_deg = sample_phi_deg + axis_sample[0]
            else:
                sample_phi_deg = sample_phi_deg + axis_sample[0]

        return kinematic_amplitude(
            crystal=sample_crystal,
            voltage_kv=sample_voltage_kv,
            theta_deg=sample_theta_deg,
            phi_deg=sample_phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
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
def _apply_kinematic_beam_distribution(  # noqa: PLR0913
    distribution: Distribution,
    crystal: CrystalStructure,
    voltage_kv: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num,
    hmax: scalar_int,
    kmax: scalar_int,
    detector_distance_mm: scalar_float,
    temperature: scalar_float,
    surface_roughness: scalar_float,
    ctr_regularization: scalar_float,
    ctr_power: scalar_float,
    roughness_power: scalar_float,
    image_shape_px: Tuple[int, int],
    pixel_size_mm: Tuple[float, float],
    beam_center_px: Tuple[float, float],
    spot_sigma_px: scalar_float,
    render_ctrs_as_streaks: bool,
    parameterization: str,
    surface_config: SurfaceConfig | None,
) -> Float[Array, "H W"]:
    """Apply a beam-mode-style distribution to the kinematic kernel."""
    theta_nominal: Float[Array, ""] = jnp.asarray(theta_deg, dtype=jnp.float64)
    phi_nominal: Float[Array, ""] = jnp.asarray(phi_deg, dtype=jnp.float64)
    voltage_nominal: Float[Array, ""] = jnp.asarray(
        voltage_kv,
        dtype=jnp.float64,
    )

    def _amplitude_at_beam_sample(
        sample: Float[Array, "3"],
    ) -> Complex[Array, "H W"]:
        theta_sample_deg: Float[Array, ""] = theta_nominal + jnp.rad2deg(
            sample[0]
        )
        phi_sample_deg: Float[Array, ""] = phi_nominal + jnp.rad2deg(sample[1])
        voltage_sample_kv: Float[Array, ""] = (
            voltage_nominal + 1.0e-3 * sample[2]
        )
        return kinematic_amplitude(
            crystal=crystal,
            voltage_kv=voltage_sample_kv,
            theta_deg=theta_sample_deg,
            phi_deg=phi_sample_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
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

    return apply_distribution(
        distribution,
        _amplitude_at_beam_sample,
    )


@jaxtyped(typechecker=beartype)
def simulate_detector_image(  # noqa: PLR0913
    crystal: CrystalStructure,
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    distribution: Distribution | None = None,
    beam_modes: BeamModeDistribution | None = None,
    n_beam_modes_per_axis: int = 3,
    n_beam_modes_out_of_plane: int | None = None,
    n_beam_energy_points: int = 1,
    n_mosaic_points: scalar_int = 7,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
    defect_surface_layer_depth_angstrom: scalar_float = 1.0,
    kernel: str = _KINEMATIC_KERNEL,
) -> Float[Array, "H W"]:
    """Simulate a broadened kinematic detector image from a crystal.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
    voltage_kv, theta_deg, phi_deg : scalar_num, optional
        Beam energy and incidence geometry. Defaults are 20 keV, 2°, 0°.
    hmax, kmax : scalar_int, optional
        In-plane CTR grid bounds. Default: 5, 5.
    detector_distance_mm : scalar_float, optional
        Sample-to-detector distance in millimetres. Default: 1000.0
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller damping. Default: 300.0
    surface_roughness : scalar_float, optional
        RMS surface roughness in Ångstroms. Default: 0.0
    ctr_regularization : scalar_float, optional
        Additive regularization in the sparse CTR intensity factor.
        Default: 0.01
    ctr_power : scalar_float, optional
        Exponent applied to the CTR modulation term. Default: 1.0
    roughness_power : scalar_float, optional
        Exponent applied to the roughness damping term. Default: 0.25
    image_shape_px : Tuple[int, int], optional
        Detector image shape as ``(height_px, width_px)``.
    pixel_size_mm : Tuple[float, float], optional
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : Tuple[float, float], optional
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.
    spot_sigma_px : scalar_float, optional
        Sparse-hit rasterization width in pixels. Default: 1.4
    angular_divergence_mrad : scalar_float, optional
        Beam angular divergence. Default: 0.35 mrad
    energy_spread_ev : scalar_float, optional
        Beam energy spread. Default: 0.35 eV
    psf_sigma_pixels : scalar_float, optional
        Detector PSF width in pixels. Default: 1.2
    n_angular_samples, n_energy_samples : int, optional
        Gauss-Hermite quadrature sizes for beam broadening.
    orientation_distribution : OrientationDistribution | None, optional
        Optional azimuthal orientation distribution. When supplied, its
        angles are interpreted relative to ``phi_deg``.
    distribution : Distribution | None, optional
        Generic Layer-1 latent distribution. The first sample coordinate is
        interpreted as an azimuthal offset in degrees relative to the current
        ``phi_deg`` sample.
    beam_modes : BeamModeDistribution | None, optional
        Explicit GSM beam-mode producer. When supplied, the amplitude-rendered
        path uses the generic Layer-1 reducer over
        ``[delta_theta_rad, delta_phi_rad, delta_energy_ev]`` beam samples.
        Default: None.
    n_beam_modes_per_axis : int, optional
        In-plane transverse GSM mode count when ``beam_modes`` is supplied.
        Default: 3.
    n_beam_modes_out_of_plane : int | None, optional
        Optional out-of-plane transverse GSM mode count. Default: None.
    n_beam_energy_points : int, optional
        Longitudinal energy quadrature count when ``beam_modes`` is supplied.
        Default: 1.
    n_mosaic_points : scalar_int, optional
        Orientation quadrature points per peak when
        ``orientation_distribution`` is supplied. Default: 7
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.
    surface_config : SurfaceConfig | None, optional
        Surface atom identification configuration. Default: None
    render_ctrs_as_streaks : bool, optional
        If True, render each valid crystal truncation rod as a continuous
        detector streak in the dense image. If False, rasterize only the
        discrete Ewald intersections as circular spots. Default: True
    defect_surface_layer_depth_angstrom : scalar_float, optional
        Surface-layer depth affected by structure-bound defect distributions
        such as ``axis_id="twins"`` or ``axis_id="steps"``. Default: 1.0
    kernel : str, optional
        Layer-0 coherent amplitude kernel selector. Currently only
        ``"kinematic"`` is implemented. Default: ``"kinematic"``.

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
    """
    _validate_layer0_kernel(kernel)
    if orientation_distribution is not None and distribution is not None:
        raise ValueError(
            "Pass either orientation_distribution or distribution, not both."
        )
    if beam_modes is not None and distribution is not None:
        raise ValueError("Pass either beam_modes or distribution, not both.")
    distributions: tuple[Distribution, ...] = _detector_image_distributions(
        beam_modes=beam_modes,
        orientation_distribution=orientation_distribution,
        distribution=distribution,
        angular_divergence_mrad=angular_divergence_mrad,
        energy_spread_ev=energy_spread_ev,
        n_angular_samples=n_angular_samples,
        n_energy_samples=n_energy_samples,
        n_beam_modes_per_axis=n_beam_modes_per_axis,
        n_beam_modes_out_of_plane=n_beam_modes_out_of_plane,
        n_beam_energy_points=n_beam_energy_points,
        n_mosaic_points=n_mosaic_points,
    )
    bound: Callable[[Float[Array, "D"]], Complex[Array, "H W"]] = (
        _bind_kinematic_distributions(
            distributions=distributions,
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
            parameterization=parameterization,
            surface_config=surface_config,
            defect_surface_layer_depth_angstrom=defect_surface_layer_depth_angstrom,
        )
    )
    detector_image: Float[Array, "H W"] = apply_distributions(
        distributions,
        bound,
    )
    return _apply_detector_psf_and_normalize(
        detector_image,
        psf_sigma_pixels=psf_sigma_pixels,
    )


@jaxtyped(typechecker=beartype)
def simulate_detector_image_instrument(  # noqa: PLR0913
    crystal: CrystalStructure,
    beam_modes: BeamModeDistribution,
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    psf_sigma_pixels: scalar_float = 1.2,
    n_modes_per_axis: int = 3,
    n_modes_out_of_plane: int | None = None,
    n_energy_points: int = 1,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    kernel: str = _KINEMATIC_KERNEL,
) -> Float[Array, "H W"]:
    """Simulate a detector image by incoherently summing beam modes.

    :see: :class:`~.test_simulator.TestDetectorImageOrchestrator`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
    beam_modes : BeamModeDistribution
        GSM source parameters defining angular and longitudinal modes.
    voltage_kv, theta_deg, phi_deg : scalar_num, optional
        Nominal beam energy and incidence geometry.
    hmax, kmax : scalar_int, optional
        In-plane CTR grid bounds.
    detector_distance_mm : scalar_float, optional
        Sample-to-detector distance in millimetres.
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller damping.
    surface_roughness : scalar_float, optional
        RMS surface roughness in Ångstroms.
    ctr_regularization, ctr_power, roughness_power : scalar_float, optional
        CTR and roughness controls passed to :func:`kinematic_amplitude`.
    image_shape_px, pixel_size_mm, beam_center_px : tuple, optional
        Detector grid calibration.
    spot_sigma_px : scalar_float, optional
        Anti-aliasing spot width in pixels.
    psf_sigma_pixels : scalar_float, optional
        Detector PSF width applied after incoherent modal summation.
    n_modes_per_axis : int, optional
        In-plane transverse GSM mode count. Default: 3.
    n_modes_out_of_plane : int | None, optional
        Optional out-of-plane transverse GSM mode count. Default: None.
    n_energy_points : int, optional
        Longitudinal energy quadrature count. Default: 1.
    parameterization : str, optional
        Atomic form-factor model.
    surface_config : SurfaceConfig | None, optional
        Surface atom identification configuration.
    kernel : str, optional
        Layer-0 coherent amplitude kernel selector. Currently only
        ``"kinematic"`` is implemented. Default: ``"kinematic"``.

    Returns
    -------
    detector_image : Float[Array, "H W"]
        Dense detector image normalized to unit maximum.

    Notes
    -----
    The beam-mode sample contract is
    ``[delta_theta_rad, delta_phi_rad, delta_energy_ev]``. Each sample is one
    coherent kinematic amplitude call; samples are reduced incoherently by the
    generic Layer-1 distribution reducer.
    """
    _validate_layer0_kernel(kernel)
    distribution: Distribution = decompose_beam_modes(
        beam_modes,
        n_modes_per_axis=n_modes_per_axis,
        n_modes_out_of_plane=n_modes_out_of_plane,
        n_energy_points=n_energy_points,
    )
    modal_image: Float[Array, "H W"] = _apply_kinematic_beam_distribution(
        distribution,
        crystal=crystal,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        hmax=hmax,
        kmax=kmax,
        detector_distance_mm=detector_distance_mm,
        temperature=temperature,
        surface_roughness=surface_roughness,
        ctr_regularization=ctr_regularization,
        ctr_power=ctr_power,
        roughness_power=roughness_power,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
        spot_sigma_px=spot_sigma_px,
        render_ctrs_as_streaks=False,
        parameterization=parameterization,
        surface_config=surface_config,
    )
    detector_image: Float[Array, "H W"] = detector_psf_convolve(
        detector_image=modal_image,
        psf_sigma_pixels=psf_sigma_pixels,
    )
    max_intensity: Float[Array, ""] = jnp.maximum(
        jnp.max(detector_image),
        1e-12,
    )
    return detector_image / max_intensity


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
    multislice_propagate : Propagate wave through potential slices.
    multislice_simulator : Complete multislice RHEED simulation.

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
    x_extent: Float[Array, ""] = sliced_crystal.x_extent
    y_extent: Float[Array, ""] = sliced_crystal.y_extent
    depth: Float[Array, ""] = sliced_crystal.depth
    nx: int = int(jnp.ceil(x_extent / pixel_size))
    ny: int = int(jnp.ceil(y_extent / pixel_size))
    nz: int = int(jnp.ceil(depth / slice_thickness))
    x_coords: Float[Array, "nx"] = jnp.linspace(0, x_extent, nx)
    y_coords: Float[Array, "ny"] = jnp.linspace(0, y_extent, ny)
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
            atom_potential: Float[Array, "nx ny"] = projected_potential(
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
def multislice_propagate(
    potential_slices: PotentialSlices,
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> Complex[Array, "nx ny"]:
    r"""Propagate electron wave through potential slices.

    This implements the multislice algorithm for dynamical electron
    diffraction, which accounts for multiple scattering events. The
    algorithm alternates between:
    1. Transmission through a slice: ψ' = ψ × exp(iσV)
    2. Fresnel propagation: ψ → FFT⁻¹[FFT[ψ] × P(kx,ky)]

    The interaction constant σ = 2π/(λV) is computed in simplified form
    suitable for high-energy electrons. The Fresnel propagator in reciprocal
    space is P(kx,ky) = exp(-iπλΔz(kx² + ky²)) which accounts for free-space
    propagation between slices. The initial wave is a tilted plane wave with
    phase k_in_x*x + k_in_y*y at z=0.

    :see: :class:`~.test_simulator.TestMultislicePropagate`

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials with shape (nz, nx, ny)
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts
    theta_deg : scalar_float
        Grazing incidence angle in degrees
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
        phi=0: beam along +x axis, phi=90: beam along +y axis
    inner_potential_v0 : scalar_float, optional
        Inner (mean) potential of the crystal in volts (default: 0.0).
        This causes refraction at the surface. Typical values are 10-20 V
        for most materials. When non-zero, the electron wavelength inside
        the crystal is shortened, and the beam refracts at the surface
        according to Snell's law for electrons.
    bandwidth_limit : scalar_float, optional
        Fraction of Nyquist frequency to retain (default: 2/3).
        Applied as a low-pass filter in Fourier space to prevent aliasing
        artifacts from the non-linear transmission function. A value of
        2/3 is standard; use 1.0 to disable bandwidth limiting.

    Returns
    -------
    exit_wave : Complex[Array, "nx ny"]
        Complex exit wave after propagation through all slices

    Notes
    -----
    The transmission function is
    :math:`T(x,y) = \exp(i \sigma V(x,y))`, where
    :math:`\sigma = 2 \pi m e / (h^2 k)` is the interaction constant.

    The Fresnel propagator in reciprocal space is
    :math:`P(k_x, k_y, \Delta z) = \exp(-i \pi \lambda \Delta z
    (k_x^2 + k_y^2))`.

    For RHEED geometry with grazing incidence, we:

    1. Start with a tilted plane wave.
    2. Propagate through slices perpendicular to the surface normal.
    3. Account for the projection of ``k_in`` onto the surface.

    1. **Initialise wave** --
       Tilted plane wave from :math:`k_{in,x}` and
       :math:`k_{in,y}` (with refraction if
       :math:`V_0 \\neq 0`).
    2. **Build propagator** --
       Fresnel propagator
       :math:`P = \\exp(-i \\pi \\lambda \\Delta z
       (k_x^2 + k_y^2))` and bandwidth aperture.
    3. **Scan slices** --
       For each slice: transmit
       :math:`\\psi' = \\psi \\exp(i \\sigma V)`, then
       propagate in Fourier space.
    4. **Return exit wave** --
       Complex wave after all slices.

    See Also
    --------
    wavelength_ang : Compute electron wavelength from voltage.
    sliced_crystal_to_projected_potential_slices : Create projected-potential
        slices from crystal.
    multislice_simulator : Complete multislice RHEED simulation.

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
    lam_ang: scalar_float = wavelength_ang(voltage_kv)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    sigma: scalar_float = interaction_constant(voltage_kv, lam_ang)
    x: Float[Array, " nx"] = jnp.arange(nx) * dx
    y: Float[Array, " ny"] = jnp.arange(ny) * dy
    kx: Float[Array, " nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, " ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, " nx ny"]
    ky_grid: Float[Array, " nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)
    voltage_v: scalar_float = voltage_kv * 1000.0
    refraction_index: scalar_float = jnp.sqrt(
        jnp.maximum(1.0 + inner_potential_v0 / voltage_v, 1e-12)
    )
    cos_theta_crystal: scalar_float = jnp.clip(
        jnp.cos(theta_rad) / refraction_index,
        -1.0,
        1.0,
    )
    theta_crystal: scalar_float = jnp.arccos(cos_theta_crystal)
    k_in_x: scalar_float = k_mag * jnp.cos(theta_crystal) * jnp.cos(phi_rad)
    k_in_y: scalar_float = k_mag * jnp.cos(theta_crystal) * jnp.sin(phi_rad)
    x_grid: Float[Array, " nx ny"]
    y_grid: Float[Array, " nx ny"]
    x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
    phase_init: Float[Array, "nx ny"] = k_in_x * x_grid + k_in_y * y_grid
    psi: Complex[Array, "nx ny"] = jnp.exp(1j * phase_init)
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
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> Complex[Array, "nx ny"]:
    """Return the coherent multislice diffraction amplitude.

    :see: :class:`~.test_multislice.TestMultisliceAmplitude`

    Parameters
    ----------
    potential_slices : PotentialSlices
        Projected potential slices with shape ``(nz, nx, ny)``.
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts.
    theta_deg : scalar_float
        Grazing incidence angle in degrees.
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
    multislice_simulator : Legacy sparse pattern path using ``|amplitude|^2``.
    """
    exit_wave: Complex[Array, "nx ny"] = multislice_propagate(
        potential_slices=potential_slices,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    amplitude: Complex[Array, "nx ny"] = jnp.fft.fft2(exit_wave)
    return amplitude


@jaxtyped(typechecker=beartype)
def multislice_simulator(
    potential_slices: PotentialSlices,
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance: scalar_float = 100.0,
    inner_potential_v0: scalar_float = 0.0,
    bandwidth_limit: scalar_float = 2.0 / 3.0,
) -> RHEEDPattern:
    r"""Simulate a RHEED pattern from potential slices with multislice.

    This function implements the complete multislice RHEED simulation
    pipeline:

    1. Propagate the electron wave through the crystal
       (:func:`multislice_propagate`).
    2. Fourier transform the exit wave to get the reciprocal-space amplitude.
    3. Apply the Ewald-sphere constraint for elastic scattering, where
       :math:`|k_{out}| = |k_{in}| = 2\pi / \lambda`.
    4. Project diffracted beams onto the detector using the angle
       approximation :math:`\theta_x \approx k_x / k_z`,
       :math:`\theta_y \approx k_y / k_z`.
    5. Calculate the intensity as :math:`|\text{amplitude}|^2`.

    The Ewald-sphere constraint gives
    :math:`k_{out,z}^2 = k^2 - k_{out,x}^2 - k_{out,y}^2`.
    Only real solutions (positive k_out_z²) correspond to propagating waves;
    evanescent waves don't reach the detector.

    :see: :class:`~.test_simulator.TestMultisliceSimulator`

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials from
        sliced_crystal_to_projected_potential_slices()
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts (typically 10-30 keV for RHEED)
    theta_deg : scalar_float
        Grazing incidence angle in degrees (typically 1-5°)
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
    detector_distance : scalar_float, optional
        Distance from sample to detector screen in mm (default: 100.0)
    inner_potential_v0 : scalar_float, optional
        Inner (mean) potential of the crystal in volts (default: 0.0).
        This causes refraction at the surface. Typical values are 10-20 V.
    bandwidth_limit : scalar_float, optional
        Fraction of Nyquist frequency to retain (default: 2/3).
        Applied as a low-pass filter to prevent aliasing artifacts.

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with detector coordinates and intensities.
        The g_indices field contains flattened grid indices since Miller
        indices are not well-defined for multislice simulation.

    Notes
    -----
    The multislice algorithm captures dynamical diffraction effects including:
    - Multiple scattering events
    - Absorption and inelastic processes (if imaginary potential included)
    - Thickness-dependent intensity oscillations
    - Kikuchi lines from diffuse scattering

    Unlike the kinematic approximation, multislice is quantitatively accurate
    for thick samples and strong scattering conditions.

    For RHEED geometry, the exit wave is projected onto the Ewald sphere
    to satisfy elastic scattering constraint :math:`|k_{out}| = |k_{in}|`.

    1. **Exit wave** --
       Propagate through all slices via
       :func:`multislice_propagate`.
    2. **Fourier transform** --
       FFT of exit wave to reciprocal space.
    3. **Ewald constraint** --
       :math:`k_{out,z}^2 = k^2 - k_{out,x}^2
       - k_{out,y}^2`; keep real solutions.
    4. **Detector projection** --
       :math:`\\theta_x \\approx k_x / k_z`,
       :math:`\\theta_y \\approx k_y / k_z`.
    5. **Assemble pattern** --
       Filter non-zero intensities and create
       :class:`RHEEDPattern`.

    See Also
    --------
    multislice_propagate : Core propagation algorithm
    simulate_rheed_pattern : Kinematic approximation simulator
    sliced_crystal_to_projected_potential_slices : Convert SlicedCrystal to
        projected-potential slices

    References
    ----------
    - Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy,
      2nd ed.
    - Ichimiya & Cohen (2004). Reflection High-Energy Electron Diffraction.
    """
    exit_wave_k: Complex[Array, "nx ny"] = multislice_amplitude(
        potential_slices=potential_slices,
        voltage_kv=voltage_kv,
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
    lam_ang: scalar_float = wavelength_ang(voltage_kv)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)
    k_in: Float[Array, "3"] = k_mag * jnp.array(
        [
            jnp.cos(theta_rad) * jnp.cos(phi_rad),
            jnp.cos(theta_rad) * jnp.sin(phi_rad),
            jnp.sin(theta_rad),
        ]
    )
    k_out_x: Float[Array, "nx ny"] = k_in[0] + kx_grid
    k_out_y: Float[Array, "nx ny"] = k_in[1] + ky_grid
    k_out_z_squared: Float[Array, "nx ny"] = k_mag**2 - k_out_x**2 - k_out_y**2
    valid_mask: Bool[Array, "nx ny"] = k_out_z_squared > 0
    k_out_z: Float[Array, "nx ny"] = jnp.where(
        valid_mask, jnp.sqrt(k_out_z_squared), 0.0
    )
    theta_x: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_x / k_out_z, 0.0
    )
    theta_y: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_y / k_out_z, 0.0
    )
    det_x: Float[Array, "nx ny"] = detector_distance * theta_x
    det_y: Float[Array, "nx ny"] = detector_distance * theta_y
    intensity_k: Float[Array, "nx ny"] = jnp.abs(exit_wave_k) ** 2
    intensity_k: Float[Array, "nx ny"] = jnp.where(
        valid_mask, intensity_k, 0.0
    )
    det_x_flat: Float[Array, "n"] = det_x.ravel()
    det_y_flat: Float[Array, "n"] = det_y.ravel()
    intensity_flat: Float[Array, "n"] = intensity_k.ravel()
    k_out_x_flat: Float[Array, "n"] = k_out_x.ravel()
    k_out_y_flat: Float[Array, "n"] = k_out_y.ravel()
    k_out_z_flat: Float[Array, "n"] = k_out_z.ravel()
    nonzero_mask: Bool[Array, "n"] = intensity_flat > 0
    det_x_filtered: Float[Array, "m"] = det_x_flat[nonzero_mask]
    det_y_filtered: Float[Array, "m"] = det_y_flat[nonzero_mask]
    intensity_filtered: Float[Array, "m"] = intensity_flat[nonzero_mask]
    k_out_filtered: Float[Array, "m 3"] = jnp.column_stack(
        [
            k_out_x_flat[nonzero_mask],
            k_out_y_flat[nonzero_mask],
            k_out_z_flat[nonzero_mask],
        ]
    )
    detector_points: Float[Array, "m 2"] = jnp.column_stack(
        [det_x_filtered, det_y_filtered]
    )
    grid_indices: Int[Array, "m"] = jnp.where(nonzero_mask)[0]
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=grid_indices,
        k_out=k_out_filtered,
        detector_points=detector_points,
        intensities=intensity_filtered,
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
    "ewald_simulator_with_orientation_distribution",
    "ewald_simulator",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "kinematic_amplitude",
    "log_compress_image",
    "multislice_amplitude",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector",
    "project_on_detector_geometry",
    "render_amplitude_to_field",
    "render_ctr_amplitude_to_field",
    "render_pattern_to_image",
    "simulate_detector_image",
    "simulate_detector_image_instrument",
    "sliced_crystal_to_projected_potential_slices",
]
