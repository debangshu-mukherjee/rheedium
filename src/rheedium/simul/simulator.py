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
:func:`ewald_simulator`
    Simulate RHEED using exact Ewald sphere-CTR intersection.
:func:`find_kinematic_reflections`
    Find reflections satisfying kinematic conditions.
:func:`multislice_propagate`
    Propagate electron wave through potential slices using multislice
    algorithm.
:func:`multislice_simulator`
    Simulate RHEED pattern from potential slices using multislice
    (dynamical).
:func:`project_on_detector`
    Project wavevectors onto detector plane.
:func:`project_on_detector_geometry`
    Project wavevectors with full detector geometry support.
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
from beartype.typing import Final, Tuple
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.tools import (
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.types import (
    CrystalStructure,
    DetectorGeometry,
    PotentialSlices,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    create_potential_slices,
    create_rheed_pattern,
    identify_surface_atoms,
    scalar_float,
    scalar_int,
    scalar_num,
)
from rheedium.ucell import reciprocal_lattice_vectors

from .form_factors import (
    atomic_scattering_factor,
    projected_potential,
)
from .surface_rods import integrated_ctr_amplitude, integrated_rod_intensity

_VALID_THRESHOLD: Final[float] = 0.5


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
def compute_kinematic_intensities_with_ctrs(  # noqa: PLR0913
    crystal: CrystalStructure,
    g_allowed: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    k_out: Float[Array, "N 3"],
    hkl_indices: Int[Array, "N 3"] | None = None,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.5,
    detector_acceptance: scalar_float = 0.01,
    surface_fraction: scalar_float = 0.3,
    surface_config: SurfaceConfig | None = None,
    ctr_mixing_mode: str = "incoherent",
    ctr_weight: scalar_float = 1.0,
    hk_tolerance: scalar_float = 0.1,
) -> Float[Array, "N"]:
    """Calculate kinematic diffraction intensities with CTR contributions.

    For each reflection, computes the structure factor by summing atomic
    contributions (form factor × phase factor). The phase is computed as
    G·r where G vectors already include the 2π factor from reciprocal
    lattice generation. CTR contributions are mixed according to the
    specified mode.

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
        Default: 0.5
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
            form_factor: scalar_float = atomic_scattering_factor(
                atomic_number=atomic_num,
                q_vector=q_vector,
                temperature=temperature,
                is_surface=is_surface,
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
) -> Tuple[Float[Array, ""], Float[Array, "3"], Float[Array, ""]]:
    """Find where a crystal truncation rod intersects the Ewald sphere.

    Solves quadratic equation for rod-sphere intersection and returns
    the solution with upward scattering (k_out_z > 0).

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
    l_intersect : Float[Array, ""]
        The l value at intersection (NaN if no valid intersection).
    k_out : Float[Array, "3"]
        Outgoing wavevector at intersection.
    valid : Float[Array, ""]
        1.0 if intersection exists and is physical, 0.0 otherwise.
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
    use_plus: Float[Array, ""] = valid_plus & (
        ~valid_minus | (k_out_plus[2] > k_out_minus[2])
    )
    l_intersect: Float[Array, ""] = jnp.where(use_plus, l_plus, l_minus)
    k_out: Float[Array, "3"] = jnp.where(use_plus, k_out_plus, k_out_minus)
    any_valid: Float[Array, ""] = valid_plus | valid_minus
    l_intersect: Float[Array, ""] = jnp.where(any_valid, l_intersect, 0.0)
    k_out: Float[Array, "3"] = jnp.where(any_valid, k_out, 0.0)
    valid: Float[Array, ""] = any_valid.astype(jnp.float64)
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
    surface_roughness: scalar_float = 0.5,
    surface_config: SurfaceConfig | None = None,
) -> RHEEDPattern:
    r"""Simulate RHEED pattern using exact Ewald sphere-CTR intersection.

    This is the physically correct approach for surface diffraction: for each
    (h,k) crystal truncation rod, solve for the l value where the rod
    intersects the Ewald sphere, then compute the diffracted intensity at
    that point.

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
        RMS surface roughness in Ångstroms for CTR damping. Default: 0.5
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

    The solution with :math:`k_{out,z} > 0` (upward scattering) is selected.

    Intensity includes:
    - Structure factor :math:`|F(G)|^2` with Kirkland form factors
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
       Kirkland form factors with Debye-Waller and surface
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
    ) -> Tuple[Float[Array, ""], Float[Array, "3"], Float[Array, ""]]:
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
    l_all: Float[Array, "N"]
    k_out_all: Float[Array, "N 3"]
    valid_all: Float[Array, "N"]
    l_all, k_out_all, valid_all = jax.vmap(_find_intersection)(rod_indices)

    # Filter to valid intersections
    valid_mask: Bool[Array, "N"] = valid_all > _VALID_THRESHOLD
    valid_indices: Int[Array, "K"] = jnp.where(
        valid_mask, size=n_rods, fill_value=-1
    )[0]
    safe_indices: Int[Array, "K"] = jnp.maximum(valid_indices, 0)

    l_valid: Float[Array, "K"] = l_all[safe_indices]
    k_out: Float[Array, "K 3"] = k_out_all[safe_indices]
    h_valid: Int[Array, "K"] = h_flat[safe_indices]
    k_valid: Int[Array, "K"] = k_flat[safe_indices]

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
            ff: Float[Array, ""] = atomic_scattering_factor(
                atomic_number=z_num,
                q_vector=q_vector,
                temperature=temperature,
                is_surface=is_surface,
            )
            phase: Float[Array, ""] = jnp.dot(g_vec, pos)
            return ff * jnp.exp(1j * phase)

        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contribs: Complex[Array, "M"] = jax.vmap(_atomic_contrib)(atom_indices)
        structure_factor: Complex[Array, ""] = jnp.sum(contribs)
        sf_intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2

        # CTR intensity modulation: 1/sin²(πl) with regularization
        sin_pi_l: Float[Array, ""] = jnp.sin(jnp.pi * l_val)
        ctr_modulation: Float[Array, ""] = 1.0 / (sin_pi_l**2 + 0.01)

        # Roughness damping: exp(-½σ²q_z²)
        roughness_damping: Float[Array, ""] = jnp.exp(
            -0.5 * (surface_roughness**2) * q_z**2
        )

        # Total intensity
        intensity: Float[Array, ""] = (
            sf_intensity * ctr_modulation * roughness_damping
        )
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
        g_indices=valid_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern


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

    def _calculate_slice_potential(slice_idx: int) -> Float[Array, "nx ny"]:
        """Calculate potential for a single slice."""
        z_start: Float[Array, ""] = slice_idx * slice_thickness
        z_end: Float[Array, ""] = (slice_idx + 1) * slice_thickness
        z_positions: Float[Array, "N"] = positions[:, 2]
        in_slice: Bool[Array, "N"] = jnp.logical_and(
            z_positions >= z_start, z_positions < z_end
        )

        def _atom_contribution(atom_idx: int) -> Float[Array, "nx ny"]:
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
    The transmission function is:
        T(x,y) = exp(iσV(x,y))
    where σ = 2πme/(h²k) is the interaction constant.

    The Fresnel propagator in reciprocal space is:
        P(kx,ky,Δz) = exp(-iπλΔz(kx² + ky²))

    For RHEED geometry with grazing incidence, we:
    1. Start with a tilted plane wave
    2. Propagate through slices perpendicular to surface normal
    3. Account for the projection of k_in onto the surface

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
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron
       Microscopy, 2nd ed.
    .. [2] Cowley & Moodie (1957). Acta Cryst. 10, 609-619.
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

    This function implements the complete multislice RHEED simulation pipeline:
    1. Propagate electron wave through crystal (multislice_propagate)
    2. Fourier transform exit wave to get reciprocal space amplitude
    3. Apply Ewald sphere constraint for elastic scattering where
       |k_out| = |k_in| = 2π/λ
    4. Project diffracted beams onto detector using angle approximation
       θ_x ≈ k_x/k_z, θ_y ≈ k_y/k_z
    5. Calculate intensity as |amplitude|²

    The Ewald sphere constraint gives k_out_z² = k_mag² - k_out_x² - k_out_y².
    Only real solutions (positive k_out_z²) correspond to propagating waves;
    evanescent waves don't reach the detector.

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
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron
       Microscopy, 2nd ed.
    .. [2] Ichimiya & Cohen (2004). Reflection High-Energy Electron
       Diffraction
    """
    exit_wave: Complex[Array, "nx ny"] = multislice_propagate(
        potential_slices=potential_slices,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        inner_potential_v0=inner_potential_v0,
        bandwidth_limit=bandwidth_limit,
    )
    exit_wave_k: Complex[Array, "nx ny"] = jnp.fft.fft2(exit_wave)
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


__all__: list[str] = [
    "compute_kinematic_intensities_with_ctrs",
    "ewald_simulator",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector",
    "project_on_detector_geometry",
    "sliced_crystal_to_projected_potential_slices",
]
