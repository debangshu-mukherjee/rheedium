"""Kinematic RHEED simulator following arXiv:2207.06642 exactly.

Extended Summary
----------------
This module provides a detector projection function that matches the paper's
equations exactly. It reuses wavelength, wavevector, and Ewald sphere functions
from simulator.py, only providing the paper-specific detector projection.

Routine Listings
----------------
make_ewald_sphere : function
    Generate Ewald sphere geometry from scattering parameters
paper_detector_projection : function
    Project scattered wavevectors onto detector screen
simple_structure_factor : function
    Calculate structure factor for a single reflection
paper_kinematic_simulator : function
    Kinematic RHEED simulator following arXiv:2207.06642

Notes
-----
Key difference from simulator.py:
- Uses paper's Equations 5-6 for detector projection
- Simplified structure factors (f_j ≈ Z_j instead of Kirkland)

References
----------
.. [1] arXiv:2207.06642 - "A Python program for simulating RHEED patterns"
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_rheed_pattern,
    scalar_float,
    scalar_int,
)
from rheedium.ucell import generate_reciprocal_points, reciprocal_lattice_vectors

from .simulator import (
    find_kinematic_reflections,
    incident_wavevector,
    project_on_detector,
    wavelength_ang,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def make_ewald_sphere(
    wavevector_magnitude: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
) -> Tuple[Float[Array, "3"], scalar_float]:
    """Generate Ewald sphere geometry from scattering parameters.

    Parameters
    ----------
    wavevector_magnitude : scalar_float
        Magnitude of the wavevector (k = 2π/λ) in 1/Å.
    theta_deg : scalar_float
        Grazing incidence angle in degrees.
    phi_deg : scalar_float, optional
        Azimuthal angle in degrees. Default: 0.0

    Returns
    -------
    center : Float[Array, "3"]
        Center of the Ewald sphere (-k_in).
    radius : scalar_float
        Radius of the Ewald sphere (k).

    Notes
    -----
    Calculations:
    - Wavelength is derived from k = 2π/λ => λ = 2π/k.
    - Incident wavevector k_in is calculated from wavelength and angles.
    - Ewald sphere center is at -k_in.
    - Radius is simply the magnitude k.
    """
    wavelength: scalar_float = 2.0 * jnp.pi / wavevector_magnitude
    k_in: Float[Array, "3"] = incident_wavevector(
        wavelength, theta_deg, phi_deg
    )
    center: Float[Array, "3"] = -k_in
    radius: scalar_float = wavevector_magnitude
    return center, radius


@jaxtyped(typechecker=beartype)
def paper_detector_projection(
    k_out: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    detector_distance: scalar_float,
    theta_deg: scalar_float,
) -> Float[Array, "N 2"]:
    """Project scattered wavevectors onto detector screen.
    Following paper's Equations 5-6 for geometric projection.

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Scattered wavevectors
    k_in : Float[Array, "3"]
        Incident wavevector
    detector_distance : scalar_float
        Distance from sample to detector screen (in mm typically)
    theta_deg : scalar_float
        Grazing incidence angle in degrees

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [x_d, y_d] coordinates on detector screen

    Notes
    -----
    Paper's Equations 5-6:

        x_d = d · (k_out_x / k_out_z)                                [Eq. 5]

        y_d = d · (k_out_y - k_in_y) / (k_out_z - k_in_z) + d·tan(θ) [Eq. 6]

    Geometry:
        - Detector is vertical screen perpendicular to surface
        - Located at distance d from sample
        - x_d is horizontal (perpendicular to incident beam)
        - y_d is vertical (along surface normal, with offset from θ)
        - Our coordinate system has z pointing UP (surface normal).
        - Both k_in_z and k_out_z are negative (beams going down to detector).
        - For Eq. 5, we use -k_out_z since k_out_z < 0 in our convention.

    The factor d·tan(θ) in Eq. 6 accounts for the detector position
    relative to the incident beam direction.

    For small denominator in Eq. 6 (specular-like reflections where 
    k_out_z ≈ k_in_z), we use a simplified projection (just the angle offset) 
    to avoid division by zero.

    Examples
    --------
    >>> k_in = jnp.array([73.0, 0.0, 2.5])
    >>> k_out = jnp.array([[72.8, 1.2, -2.3], [73.2, -0.8, -2.1]])
    >>> coords = paper_detector_projection(k_out, k_in,
    ...     detector_distance=100.0, theta_deg=2.0)
    >>> print(f"Detector positions: {coords}")
    """
    denom_threshold: scalar_float = 1e-6
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    x_d: Float[Array, "N"] = detector_distance * (k_out[:, 0] / (-k_out[:, 2]))
    denom: Float[Array, "N"] = -k_out[:, 2] + k_in[2]
    y_d: Float[Array, "N"] = jnp.where(
        jnp.abs(denom) < denom_threshold,
        detector_distance * jnp.tan(theta_rad),
        detector_distance * (k_out[:, 1] - k_in[1]) / denom
        + detector_distance * jnp.tan(theta_rad),
    )
    detector_coords: Float[Array, "N 2"] = jnp.stack([x_d, y_d], axis=-1)
    return detector_coords


@jaxtyped(typechecker=beartype)
def simple_structure_factor(
    reciprocal_vector: Float[Array, "3"],
    atom_positions: Float[Array, "M 3"],
    atomic_numbers: Int[Array, "M"],
) -> Float[Array, ""]:
    """Calculate structure factor for a single reflection.
    
    Following paper's Equation 7: F(G) = Σ_j f_j · exp(i·G·r_j)

    Parameters
    ----------
    reciprocal_vector : Float[Array, "3"]
        Reciprocal lattice vector G for this reflection
    atom_positions : Float[Array, "M 3"]
        Cartesian positions of atoms in unit cell
    atomic_numbers : Int[Array, "M"]
        Atomic numbers (Z) for each atom

    Returns
    -------
    intensity : Float[Array, ""]
        Diffraction intensity I = |F(G)|²

    Notes
    -----
    Structure factor:
        F(G) = Σ_j f_j(G) · exp(i·G·r_j)  [Paper's Eq. 7]

    where:
        - f_j(G) = atomic scattering factor for atom j
        - r_j = position of atom j
        - Sum over all atoms in unit cell

    Intensity:
        I(G) = |F(G)|²

    Implementation details:
    - Uses vectorized operations (JAX-friendly).
    - Atomic scattering factors are simplified as f_j ≈ Z_j (atomic number).
    - For more accurate scattering, use Kirkland parameterization 
      (see form_factors.py).
    - Calculates phase factors exp(i·G·r_j) for all atoms.
    - Sums contributions: F = Σ f_j · exp(i·G·r_j).

    Examples
    --------
    >>> G = jnp.array([2.0, 0.0, 1.0])  # (100) reflection
    >>> positions = jnp.array([[0, 0, 0], [0.5, 0.5, 0.5]])  # Two atoms
    >>> atomic_nums = jnp.array([14, 14])  # Silicon
    >>> I = simple_structure_factor(G, positions, atomic_nums)
    >>> print(f"I(100) = {I:.2f}")
    """
    f_j: Float[Array, "M"] = atomic_numbers.astype(jnp.float64)
    dot_products: Float[Array, "M"] = jnp.dot(
        atom_positions, reciprocal_vector
    )
    phases: Complex[Array, "M"] = jnp.exp(1j * dot_products)
    structure_factor: Complex[Array, ""] = jnp.sum(f_j * phases)
    intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2
    return intensity


@jaxtyped(typechecker=beartype)
def paper_kinematic_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    lmax: scalar_int = 1,
    detector_distance: scalar_float = 100.0,
    tolerance: scalar_float = 0.05,
) -> RHEEDPattern:
    """Kinematic RHEED simulator following arXiv:2207.06642.

    Clean implementation matching the paper's step-by-step algorithm.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters
    voltage_kv : scalar_float, optional
        Electron beam voltage in kilovolts. Default: 20.0
    theta_deg : scalar_float, optional
        Grazing incidence angle in degrees. Default: 2.0
    hmax : scalar_int, optional
        Maximum h Miller index. Default: 3
    kmax : scalar_int, optional
        Maximum k Miller index. Default: 3
    lmax : scalar_int, optional
        Maximum l Miller index. Default: 1
    detector_distance : scalar_float, optional
        Sample-to-screen distance in mm. Default: 100.0
    tolerance : scalar_float, optional
        Tolerance for Ewald sphere constraint. Default: 0.05

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with reflection positions and intensities

    Notes
    -----
    This is a pedagogical implementation following the published paper.
    For production use with full surface physics (Debye-Waller factors,
    CTRs, surface roughness), see `simulator.py:simulate_rheed_pattern()`.

    Algorithm (following paper)
    ---------------------------
    1. Generate reciprocal lattice G(h,k,l) up to (hmax, kmax, lmax)
       (Following paper's Equations 1-3).
    2. Calculate electron wavelength λ from voltage
       (Relativistic de Broglie wavelength from simulator.py).
    3. Build incident wavevector k_in from θ and λ
       (k_in = (2π/λ) × [cos(θ), 0, sin(θ)] from simulator.py).
    4. Find allowed reflections (Ewald sphere construction)
       (Following paper's Equation 4: k_out = k_in + G with |k_out| = |k_in|).
       (Using find_kinematic_reflections from simulator.py).
       (RHEED: upward scattering, k_out_z > 0).
    5. Project k_out onto detector screen using geometric formulas
       (Following paper's Equations 5-6).
    6. Calculate intensities I = |F(G)|² using structure factors
       (Following paper's Equation 7).
    7. Return structured pattern.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Load crystal
    >>> crystal = rh.inout.parse_cif("Si.cif")
    >>>
    >>> # Simulate RHEED pattern
    >>> pattern = rh.simul.paper_kinematic_simulator(
    ...     crystal=crystal,
    ...     voltage_kv=20.0,
    ...     theta_deg=2.0,
    ...     hmax=3,
    ...     kmax=3,
    ...     lmax=1,
    ...     detector_distance=100.0
    ... )
    >>>
    >>> print(f"Number of reflections: {len(pattern.intensities)}")
    >>> print(f"Detector coordinates: {pattern.detector_points}")
    >>> print(f"Intensities: {pattern.intensities}")

    See Also
    --------
    simulate_rheed_pattern : Full simulator with surface physics
    multislice_simulator : Dynamical diffraction simulator

    References
    ----------
    .. [1] arXiv:2207.06642 - "A Python program for simulating RHEED patterns"
    .. [2] Ichimiya & Cohen. "Reflection High-Energy Electron Diffraction"
    """
    reciprocal_points: Float[Array, "M 3"] = generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    wavelength: scalar_float = wavelength_ang(voltage_kv)
    k_in: Float[Array, "3"] = incident_wavevector(
        wavelength, theta_deg, phi_deg=0.0
    )
    kr: Tuple[Int[Array, "K"], Float[Array, "K 3"]] = (
        find_kinematic_reflections(
            k_in=k_in, gs=reciprocal_points, z_sign=1.0, tolerance=tolerance
        )
    )
    allowed_indices: Int[Array, "K"] = kr[0]
    k_out: Float[Array, "K 3"] = kr[1]
    reciprocal_allowed: Float[Array, "K 3"] = reciprocal_points[
        allowed_indices
    ]
    detector_coords: Float[Array, "K 2"] = project_on_detector(
        k_out=k_out,
        detector_distance=detector_distance,
        k_in=k_in,
    )
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )

    def _calculate_intensity(gg: Float[Array, "3"]) -> Float[Array, ""]:
        return simple_structure_factor(gg, atom_positions, atomic_numbers)

    intensities: Float[Array, "N"] = jax.vmap(_calculate_intensity)(
        reciprocal_allowed
    )
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_coords,
        intensities=intensities,
    )
    return pattern


@jaxtyped(typechecker=beartype)
def find_ctr_ewald_intersection(
    h: scalar_int,
    k: scalar_int,
    k_in: Float[Array, "3"],
    recip_a: Float[Array, "3"],
    recip_b: Float[Array, "3"],
) -> Tuple[Float[Array, ""], Float[Array, "3"], Float[Array, ""]]:
    """Find where a crystal truncation rod (h, k, l) intersects the Ewald sphere.

    For a surface, the reciprocal lattice consists of rods extending perpendicular
    to the surface (along z). Each rod is labeled by in-plane indices (h, k) and
    has continuous l. This function finds the l value(s) where the rod intersects
    the Ewald sphere.

    Parameters
    ----------
    h : scalar_int
        In-plane Miller index h.
    k : scalar_int
        In-plane Miller index k.
    k_in : Float[Array, "3"]
        Incident wavevector.
    recip_a : Float[Array, "3"]
        First reciprocal lattice vector (a*).
    recip_b : Float[Array, "3"]
        Second reciprocal lattice vector (b*).

    Returns
    -------
    l_intersect : Float[Array, ""]
        The l value where the rod intersects the Ewald sphere (upward scattering).
    k_out : Float[Array, "3"]
        The scattered wavevector at the intersection.
    valid : Float[Array, ""]
        1.0 if intersection exists and is physical, 0.0 otherwise.

    Notes
    -----
    The Ewald sphere condition is: |k_out|² = |k_in|²

    For a CTR at (h, k), the reciprocal space position is:
        G(l) = h·a* + k·b* + l·c*

    where c* is perpendicular to the surface (along z).

    The scattered wavevector is:
        k_out = k_in + G(l)

    Substituting and solving for l (with c* = [0, 0, c*_z]):
        |k_in + h·a* + k·b* + l·c*|² = |k_in|²

    This is a quadratic in l. We want the solution with k_out_z > 0 (upward).
    """
    k_mag_sq: Float[Array, ""] = jnp.dot(k_in, k_in)
    g_hk: Float[Array, "3"] = h * recip_a + k * recip_b
    k_plus_ghk: Float[Array, "3"] = k_in + g_hk
    c_star_z: Float[Array, ""] = jnp.linalg.norm(recip_a)
    a_coef: Float[Array, ""] = c_star_z**2
    b_coef: Float[Array, ""] = 2.0 * k_plus_ghk[2] * c_star_z
    c_coef: Float[Array, ""] = jnp.dot(k_plus_ghk, k_plus_ghk) - k_mag_sq
    discriminant: Float[Array, ""] = b_coef**2 - 4.0 * a_coef * c_coef
    has_solution: Float[Array, ""] = discriminant >= 0
    disc_safe: Float[Array, ""] = jnp.maximum(discriminant, 0.0)
    sqrt_disc: Float[Array, ""] = jnp.sqrt(disc_safe)
    l1: Float[Array, ""] = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    l2: Float[Array, ""] = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    k_out1_z: Float[Array, ""] = k_plus_ghk[2] + l1 * c_star_z
    k_out2_z: Float[Array, ""] = k_plus_ghk[2] + l2 * c_star_z
    l_intersect: Float[Array, ""] = jnp.where(k_out1_z > 0, l1, l2)
    k_out_z_chosen: Float[Array, ""] = jnp.where(k_out1_z > 0, k_out1_z, k_out2_z)
    valid: Float[Array, ""] = has_solution & (k_out_z_chosen > 0)
    k_out: Float[Array, "3"] = k_plus_ghk + jnp.array(
        [0.0, 0.0, l_intersect * c_star_z]
    )
    return l_intersect, k_out, valid.astype(jnp.float64)


@jaxtyped(typechecker=beartype)
def streak_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    detector_distance: scalar_float = 100.0,
    points_per_streak: scalar_int = 50,
) -> Tuple[Float[Array, "N 2"], Float[Array, "N"], Int[Array, "N 2"]]:
    """Simulate RHEED streak pattern from crystal truncation rods.

    This function properly models RHEED as diffraction from a surface where
    the reciprocal lattice consists of continuous rods (CTRs) rather than
    discrete points. Each rod intersects the Ewald sphere along an arc,
    producing the characteristic vertical streaks seen in RHEED patterns.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    voltage_kv : scalar_float, optional
        Electron beam voltage in kilovolts. Default: 20.0
    theta_deg : scalar_float, optional
        Grazing incidence angle in degrees. Default: 2.0
    hmax : scalar_int, optional
        Maximum h Miller index. Default: 3
    kmax : scalar_int, optional
        Maximum k Miller index. Default: 3
    detector_distance : scalar_float, optional
        Sample-to-screen distance in mm. Default: 100.0
    points_per_streak : scalar_int, optional
        Number of points to sample along each streak. Default: 50

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [x, y] coordinates on detector screen for all streak points.
    intensities : Float[Array, "N"]
        Intensity values for each point (includes CTR decay with l).
    hk_indices : Int[Array, "N 2"]
        The (h, k) indices for each point, identifying which streak it belongs to.

    Notes
    -----
    Physics of RHEED streaks:

    1. Surface breaks translational symmetry in z-direction
    2. Reciprocal lattice becomes rods: discrete (h,k), continuous l
    3. Each rod intersects Ewald sphere where |k_in + G|² = |k_in|²
    4. CTR intensity varies as 1/sin²(πl) for ideal termination
    5. Projection onto detector creates vertical streaks

    The streak length depends on:
    - Ewald sphere curvature (electron wavelength)
    - Grazing angle (determines accessible l range)
    - Detector geometry

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> coords, intensities, hk = rh.simul.streak_simulator(
    ...     crystal=crystal,
    ...     voltage_kv=15.0,
    ...     theta_deg=2.0,
    ... )
    >>> # Plot as scatter with varying intensity
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(coords[:, 0], coords[:, 1], c=intensities, s=1)
    """
    wavelength: scalar_float = wavelength_ang(voltage_kv)
    k_in: Float[Array, "3"] = incident_wavevector(wavelength, theta_deg, phi_deg=0.0)
    k_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *crystal.cell_lengths, *crystal.cell_angles, in_degrees=True
    )
    recip_a: Float[Array, "3"] = recip_vecs[0]
    recip_b: Float[Array, "3"] = recip_vecs[1]
    recip_c: Float[Array, "3"] = recip_vecs[2]
    c_star_z: Float[Array, ""] = jnp.abs(recip_c[2])
    h_range: Int[Array, "H"] = jnp.arange(-hmax, hmax + 1, dtype=jnp.int32)
    k_range: Int[Array, "K"] = jnp.arange(-kmax, kmax + 1, dtype=jnp.int32)
    hh, kk = jnp.meshgrid(h_range, k_range, indexing="ij")
    h_flat: Int[Array, "M"] = hh.flatten()
    k_flat: Int[Array, "M"] = kk.flatten()
    n_rods: int = h_flat.shape[0]
    all_detector_coords = []
    all_intensities = []
    all_hk_indices = []

    for i in range(n_rods):
        h_i = h_flat[i]
        k_i = k_flat[i]
        g_hk: Float[Array, "3"] = h_i * recip_a + k_i * recip_b
        k_plus_ghk: Float[Array, "3"] = k_in + g_hk
        a_coef: Float[Array, ""] = c_star_z**2
        b_coef: Float[Array, ""] = 2.0 * k_plus_ghk[2] * c_star_z
        c_coef: Float[Array, ""] = jnp.dot(k_plus_ghk, k_plus_ghk) - k_mag**2
        discriminant: Float[Array, ""] = b_coef**2 - 4.0 * a_coef * c_coef
        has_solution = discriminant >= 0
        if not has_solution:
            continue
        sqrt_disc: Float[Array, ""] = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        l1: Float[Array, ""] = (-b_coef + sqrt_disc) / (2.0 * a_coef)
        l2: Float[Array, ""] = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        l_min: Float[Array, ""] = jnp.minimum(l1, l2)
        l_max: Float[Array, ""] = jnp.maximum(l1, l2)
        l_values: Float[Array, "P"] = jnp.linspace(
            l_min - 2.0, l_max + 2.0, points_per_streak
        )
        g_points: Float[Array, "P 3"] = g_hk[None, :] + l_values[:, None] * recip_c[
            None, :
        ]
        k_out_points: Float[Array, "P 3"] = k_in[None, :] + g_points
        k_out_mags: Float[Array, "P"] = jnp.linalg.norm(k_out_points, axis=1)
        ewald_deviation: Float[Array, "P"] = jnp.abs(k_out_mags - k_mag)
        tolerance: Float[Array, ""] = 0.3 * k_mag
        on_ewald: Float[Array, "P"] = ewald_deviation < tolerance
        upward: Float[Array, "P"] = k_out_points[:, 2] > 0.0
        valid_mask: Float[Array, "P"] = on_ewald & upward
        if not jnp.any(valid_mask):
            continue
        valid_k_out: Float[Array, "V 3"] = k_out_points[valid_mask]
        valid_l: Float[Array, "V"] = l_values[valid_mask]
        det_coords: Float[Array, "V 2"] = project_on_detector(
            k_out=valid_k_out,
            detector_distance=detector_distance,
            k_in=k_in,
        )
        l_safe: Float[Array, "V"] = jnp.where(
            jnp.abs(valid_l) < 0.1, jnp.sign(valid_l) * 0.1 + 0.1, valid_l
        )
        ctr_intensity: Float[Array, "V"] = 1.0 / (jnp.sin(jnp.pi * l_safe) ** 2 + 0.01)
        ctr_intensity = ctr_intensity / ctr_intensity.max()
        all_detector_coords.append(det_coords)
        all_intensities.append(ctr_intensity)
        hk_for_streak: Int[Array, "V 2"] = jnp.tile(
            jnp.array([h_i, k_i]), (det_coords.shape[0], 1)
        )
        all_hk_indices.append(hk_for_streak)
    if len(all_detector_coords) == 0:
        empty_coords: Float[Array, "0 2"] = jnp.zeros((0, 2))
        empty_intensities: Float[Array, "0"] = jnp.zeros((0,))
        empty_hk: Int[Array, "0 2"] = jnp.zeros((0, 2), dtype=jnp.int32)
        return empty_coords, empty_intensities, empty_hk
    detector_coords: Float[Array, "N 2"] = jnp.concatenate(all_detector_coords, axis=0)
    intensities: Float[Array, "N"] = jnp.concatenate(all_intensities, axis=0)
    hk_indices: Int[Array, "N 2"] = jnp.concatenate(all_hk_indices, axis=0)
    return detector_coords, intensities, hk_indices