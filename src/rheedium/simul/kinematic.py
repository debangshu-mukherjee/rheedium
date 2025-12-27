"""Kinematic RHEED simulator.

Extended Summary
----------------
This module provides kinematic RHEED simulation functions including
structure factor calculation, and complete pattern simulation.
Implements the algorithm from arXiv:2207.06642.

Routine Listings
----------------
find_ctr_ewald_intersection : function
    Find where a crystal truncation rod intersects the Ewald sphere
kinematic_ctr_simulator : function
    RHEED simulation using continuous crystal truncation rods (streaks)
kinematic_spot_simulator : function
    RHEED simulation using discrete 3D reciprocal lattice (spots)
make_ewald_sphere : function
    Generate Ewald sphere geometry from scattering parameters
simple_structure_factor : function
    Calculate structure factor for a single reflection

Notes
-----
Key difference from simulator.py:
- Simplified structure factors (f_j ≈ Z_j instead of Kirkland)
- For detector projection, use :func:`project_on_detector` from simulator

References
----------
.. [1] arXiv:2207.06642 - "A Python program for simulating RHEED patterns"
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_rheed_pattern,
    scalar_float,
    scalar_int,
)
from rheedium.ucell import (
    generate_reciprocal_points,
    reciprocal_lattice_vectors,
)

from .simul_utils import incident_wavevector, wavelength_ang
from .simulator import find_kinematic_reflections, project_on_detector

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
def simple_structure_factor(
    reciprocal_vector: Float[Array, "3"],
    atom_positions: Float[Array, "M 3"],
    atomic_numbers: Int[Array, "M"],
) -> Float[Array, ""]:
    r"""Calculate structure factor for a single reflection.

    Following paper's Equation 7:

    .. math::

        F(G) = \sum_j f_j \cdot \exp(i \cdot G \cdot r_j)

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
        Diffraction intensity :math:`I = |F(G)|^2`

    Notes
    -----
    Structure factor (Paper's Eq. 7):

    .. math::

        F(G) = \sum_j f_j(G) \cdot \exp(i \cdot G \cdot r_j)

    where:

    - :math:`f_j(G)` = atomic scattering factor for atom j
    - :math:`r_j` = position of atom j
    - Sum over all atoms in unit cell

    Intensity:

    .. math::

        I(G) = |F(G)|^2

    Implementation details:

    - Uses vectorized operations (JAX-friendly).
    - Atomic scattering factors are simplified as :math:`f_j \approx Z_j`
      (atomic number).
    - For more accurate scattering, use Kirkland parameterization
      (see form_factors.py).
    - Calculates phase factors :math:`\exp(i \cdot G \cdot r_j)` for all atoms.
    - Sums contributions: :math:`F = \sum f_j \cdot \exp(i \cdot G \cdot r_j)`.

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
def kinematic_spot_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    lmax: scalar_int = 1,
    detector_distance: scalar_float = 100.0,
    tolerance: scalar_float = 0.05,
) -> RHEEDPattern:
    """Kinematic RHEED spot simulator using discrete 3D reciprocal lattice.

    Simulates RHEED pattern as discrete spots where integer (h,k,l) reciprocal
    lattice points intersect the Ewald sphere. Useful for bulk-like diffraction
    or when only spot positions matter.

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
    lmax : scalar_int, optional
        Maximum l Miller index. Default: 1
    detector_distance : scalar_float, optional
        Sample-to-screen distance in mm. Default: 100.0
    tolerance : scalar_float, optional
        Tolerance for Ewald sphere constraint. Default: 0.05

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with spot positions and intensities.

    Notes
    -----
    This simulator treats the reciprocal lattice as discrete 3D points.
    For surface-sensitive RHEED with continuous crystal truncation rods
    (CTRs) and streak patterns, use `kinematic_ctr_simulator` instead.

    Algorithm
    ---------
    1. Generate reciprocal lattice G(h,k,l) up to (hmax, kmax, lmax)
    2. Calculate electron wavelength λ from voltage
    3. Build incident wavevector k_in from θ and λ
    4. Find allowed reflections via Ewald sphere construction
    5. Project k_out onto detector screen
    6. Calculate intensities :math:`I = |F(G)|^2` using structure factors

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> pattern = rh.simul.kinematic_spot_simulator(
    ...     crystal=crystal,
    ...     voltage_kv=20.0,
    ...     theta_deg=2.0,
    ...     hmax=3, kmax=3, lmax=5,
    ... )
    >>> print(f"Found {len(pattern.intensities)} spots")

    See Also
    --------
    kinematic_ctr_simulator : CTR-based simulator with streaks
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
    all_indices, all_k_out = find_kinematic_reflections(
        k_in=k_in, gs=reciprocal_points, z_sign=1.0, tolerance=tolerance
    )

    # Filter to valid reflections only (indices >= 0)
    valid_mask: Bool[Array, "M"] = all_indices >= 0
    allowed_indices: Int[Array, "K"] = all_indices[valid_mask]
    k_out: Float[Array, "K 3"] = all_k_out[valid_mask]
    reciprocal_allowed: Float[Array, "K 3"] = reciprocal_points[
        allowed_indices
    ]

    detector_coords: Float[Array, "K 2"] = project_on_detector(
        k_out=k_out,
        detector_distance=detector_distance,
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
    recip_c: Float[Array, "3"],
) -> Tuple[Float[Array, ""], Float[Array, "3"], Float[Array, ""]]:
    """Find where a crystal truncation rod intersects the Ewald sphere.

    Description
    -----------
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
    sqrt_disc: Float[Array, ""] = jnp.sqrt(jnp.maximum(discriminant, 0.0))
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
    l_intersect = jnp.where(any_valid, l_intersect, jnp.nan)
    k_out = jnp.where(any_valid, k_out, jnp.nan)
    valid: Float[Array, ""] = any_valid.astype(jnp.float64)
    return l_intersect, k_out, valid


@jaxtyped(typechecker=beartype)
def kinematic_ctr_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    phi_deg: scalar_float = 0.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    detector_distance: scalar_float = 100.0,
    l_min: scalar_float = 0.0,
    l_max: scalar_float = 5.0,
    n_points_per_rod: scalar_int = 100,
) -> RHEEDPattern:
    """RHEED simulation with continuous CTR streaks.

    Description
    -----------
    Generates vertical streaks by sampling l continuously along each
    allowed (h,k) rod. Applies structure factor extinction and CTR
    intensity modulation.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    voltage_kv : scalar_float, optional
        Electron beam voltage in kilovolts. Default: 20.0
    theta_deg : scalar_float, optional
        Grazing incidence angle in degrees. Default: 2.0
    phi_deg : scalar_float, optional
        Azimuthal angle in degrees. Default: 0.0
    hmax : scalar_int, optional
        Maximum h Miller index. Default: 3
    kmax : scalar_int, optional
        Maximum k Miller index. Default: 3
    detector_distance : scalar_float, optional
        Sample-to-screen distance in mm. Default: 100.0
    l_min : scalar_float, optional
        Minimum l value to sample along rods. Default: 0.0
    l_max : scalar_float, optional
        Maximum l value to sample along rods. Default: 5.0
    n_points_per_rod : scalar_int, optional
        Number of l points to sample per rod. Default: 100

    Returns
    -------
    pattern : RHEEDPattern
        RHEED pattern with streak positions and intensities.

    Notes
    -----
    This function is fully JAX-compatible using jax.vmap for the rod loop.
    Fixed-size output arrays are used with masking for invalid entries.
    """
    k_in = incident_wavevector(wavelength_ang(voltage_kv), theta_deg, phi_deg)
    recip_vecs = reciprocal_lattice_vectors(
        *crystal.cell_lengths, *crystal.cell_angles, in_degrees=True
    )
    recip_a, recip_b, recip_c = recip_vecs[0], recip_vecs[1], recip_vecs[2]
    atom_pos = crystal.cart_positions[:, :3]
    atom_z = crystal.cart_positions[:, 3].astype(jnp.int32)

    hh, kk = jnp.meshgrid(
        jnp.arange(-hmax, hmax + 1, dtype=jnp.int32),
        jnp.arange(-kmax, kmax + 1, dtype=jnp.int32),
        indexing="ij",
    )
    h_flat, k_flat = hh.flatten(), kk.flatten()
    n_rods = h_flat.shape[0]
    l_values = jnp.linspace(l_min, l_max, n_points_per_rod)

    def _process_rod(rod_idx: Int[Array, ""]) -> Tuple[
        Float[Array, "P 2"],
        Float[Array, "P 3"],
        Float[Array, "P"],
        Bool[Array, "P"],
    ]:
        """Process single rod: (det_coords, k_out, intensity, valid)."""
        h, k = h_flat[rod_idx], k_flat[rod_idx]
        g_hk = h * recip_a + k * recip_b
        # Test SF at l=0.5 to check extinction
        g_test = g_hk + 0.5 * recip_c
        sf_test = simple_structure_factor(g_test, atom_pos, atom_z)
        sf_valid = sf_test >= 1.0
        # Compute G and k_out for all l values
        g_vecs = g_hk + l_values[:, None] * recip_c
        k_out_vecs = k_in + g_vecs
        point_valid = (k_out_vecs[:, 2] > 0) & sf_valid
        # Detector projection
        scale = detector_distance / (k_out_vecs[:, 0] + 1e-10)
        det_coords = jnp.stack(
            [k_out_vecs[:, 1] * scale, k_out_vecs[:, 2] * scale], axis=-1
        )
        # CTR modulation with regularization near l=0
        l_eps = 0.1
        l_safe = jnp.where(
            jnp.abs(l_values) < l_eps,
            jnp.sign(l_values) * l_eps + l_eps,
            l_values,
        )
        ctr_mod = 1.0 / (jnp.sin(jnp.pi * l_safe) ** 2 + 0.01)
        intensity = sf_test * ctr_mod
        intensity = intensity / jnp.maximum(jnp.max(intensity), 1e-10)
        return det_coords, k_out_vecs, intensity, point_valid

    # Process all rods in parallel with vmap
    rod_indices = jnp.arange(n_rods, dtype=jnp.int32)
    all_det, all_kout, all_int, all_valid = jax.vmap(_process_rod)(rod_indices)

    # Flatten and filter valid points
    shape = (n_rods, n_points_per_rod)
    rod_idx_exp = jnp.broadcast_to(rod_indices[:, None], shape)
    det_flat, kout_flat = all_det.reshape(-1, 2), all_kout.reshape(-1, 3)
    int_flat, valid_flat = all_int.reshape(-1), all_valid.reshape(-1)
    idx_flat = rod_idx_exp.reshape(-1)

    n_total = n_rods * n_points_per_rod
    valid_idx = jnp.where(valid_flat, size=n_total, fill_value=-1)[0]
    safe_idx = jnp.maximum(valid_idx, 0)

    det_coords = det_flat[safe_idx]
    k_out_all = kout_flat[safe_idx]
    intensities = jnp.where(valid_idx >= 0, int_flat[safe_idx], 0.0)
    g_indices = idx_flat[safe_idx]

    return create_rheed_pattern(
        g_indices=g_indices,
        k_out=k_out_all,
        detector_points=det_coords,
        intensities=intensities,
    )
