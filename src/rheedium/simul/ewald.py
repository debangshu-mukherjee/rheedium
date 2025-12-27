"""Ewald sphere builder functions.

Extended Summary
----------------
This module provides functions for building angle-independent Ewald sphere
data for RHEED simulation. The build_ewald_data function pre-computes
reciprocal lattice geometry and structure factors from crystal structure
and beam voltage, enabling efficient reuse across multiple beam orientations.

Routine Listings
----------------
build_ewald_data : function
    Build EwaldData from CrystalStructure, voltage, and lattice bounds
ewald_allowed_reflections : function
    Find reflections satisfying Ewald sphere condition for given beam angles

Notes
-----
The separation of angle-independent (EwaldData) from angle-dependent (beam
orientation) calculations enables efficient azimuthal scans where structure
factors need only be computed once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    EwaldData,
    create_ewald_data,
    scalar_float,
    scalar_int,
)
from rheedium.ucell import (
    generate_reciprocal_points,
    reciprocal_lattice_vectors,
)

from .finite_domain import (
    compute_shell_sigma,
    extent_to_rod_sigma,
    rod_ewald_overlap,
)
from .form_factors import (
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
)
from .simul_utils import wavelength_ang

jax.config.update("jax_enable_x64", True)


def _compute_structure_factor_single(
    g_vector: Float[Array, "3"],
    atom_positions: Float[Array, "M 3"],
    atomic_numbers: Int[Array, "M"],
    temperature: scalar_float,
) -> Complex[Array, ""]:
    """Compute structure factor F(G) for a single reciprocal vector.

    Description
    -----------
    Calculates the crystallographic structure factor including Kirkland
    atomic form factors and Debye-Waller thermal damping:

    .. math::

        F(G) = \sum_j f_j(|G|) \cdot \exp(-W_j) \cdot \exp(i \cdot G \cdot r_j)

    Parameters
    ----------
    g_vector : Float[Array, "3"]
        Reciprocal lattice vector G in 1/Ångstroms.
    atom_positions : Float[Array, "M 3"]
        Cartesian positions of M atoms in Ångstroms.
    atomic_numbers : Int[Array, "M"]
        Atomic numbers Z for M atoms.
    temperature : scalar_float
        Temperature in Kelvin for Debye-Waller calculation.

    Returns
    -------
    structure_factor : Complex[Array, ""]
        Complex structure factor F(G).

    Flow
    ----
    1. Compute :math:`|G|` for form factor lookup
    2. For each atom: compute :math:`f(|G|)`, Debye-Waller, and phase
    3. Sum all atomic contributions
    """
    g_magnitude: Float[Array, ""] = jnp.linalg.norm(g_vector)

    def _atomic_contribution(atom_idx: Int[Array, ""]) -> Complex[Array, ""]:
        atomic_num: Int[Array, ""] = atomic_numbers[atom_idx]
        atom_pos: Float[Array, "3"] = atom_positions[atom_idx]
        form_factor: Float[Array, ""] = kirkland_form_factor(
            atomic_number=atomic_num,
            q_magnitude=g_magnitude,
        )
        mean_sq_disp: Float[Array, ""] = get_mean_square_displacement(
            atomic_number=atomic_num,
            temperature=temperature,
            is_surface=False,
        )
        debye_waller: Float[Array, ""] = debye_waller_factor(
            q_magnitude=g_magnitude,
            mean_square_displacement=mean_sq_disp,
        )
        phase: Float[Array, ""] = jnp.dot(g_vector, atom_pos)
        contribution: Complex[Array, ""] = (
            form_factor * debye_waller * jnp.exp(1.0j * phase)
        )
        return contribution

    n_atoms: Int[Array, ""] = atom_positions.shape[0]
    atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
    contributions: Complex[Array, "M"] = jax.vmap(_atomic_contribution)(
        atom_indices
    )
    structure_factor: Complex[Array, ""] = jnp.sum(contributions)
    return structure_factor


@jaxtyped(typechecker=beartype)
def build_ewald_data(
    crystal: CrystalStructure,
    voltage_kv: scalar_float,
    hmax: scalar_int,
    kmax: scalar_int,
    lmax: scalar_int,
    temperature: scalar_float = 300.0,
) -> EwaldData:
    """Build angle-independent EwaldData from crystal and beam parameters.

    Description
    -----------
    Constructs an EwaldData PyTree containing all angle-independent quantities
    needed for RHEED simulation: Ewald sphere geometry, reciprocal lattice
    points, and pre-computed structure factors with atomic form factors and
    thermal damping.

    Parameters
    ----------
    crystal : CrystalStructure
        Symmetry-expanded crystal structure with atomic positions and cell
        parameters. Must have complete unit cell (all symmetry-equivalent
        atoms present).
    voltage_kv : scalar_float
        Electron beam accelerating voltage in kilovolts. Typical RHEED
        values: 10-30 kV.
    hmax : scalar_int
        Maximum Miller index h. Range: [-hmax, +hmax].
    kmax : scalar_int
        Maximum Miller index k. Range: [-kmax, +kmax].
    lmax : scalar_int
        Maximum Miller index l. Range: [-lmax, +lmax].
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factor calculation.
        Default: 300.0 K (room temperature).

    Returns
    -------
    ewald_data : EwaldData
        Complete angle-independent Ewald sphere data ready for use with
        kinematic_from_ewald() or similar angle-dependent functions.

    Flow
    ----
    1. Compute relativistic electron wavelength from voltage
    2. Calculate wavevector magnitude :math:`k = 2\pi/\lambda` (= sphere radius)
    3. Generate reciprocal lattice basis vectors from crystal cell
    4. Create Miller index grid for specified (hmax, kmax, lmax)
    5. Transform Miller indices to reciprocal space vectors G
    6. Compute :math:`|G|` for each reciprocal point
    7. Calculate structure factors F(G) with form factors and DW
    8. Compute intensities :math:`I(G) = |F(G)|^2`
    9. Package into EwaldData PyTree

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ewald = rh.simul.build_ewald_data(
    ...     crystal=crystal,
    ...     voltage_kv=15.0,
    ...     hmax=3, kmax=3, lmax=2,
    ...     temperature=300.0
    ... )
    >>> print(f"λ = {float(ewald.wavelength_ang):.4f} Å")
    >>> print(f"k_mag = {float(ewald.k_magnitude):.2f} 1/Å")
    >>> print(f"N_G = {ewald.g_vectors.shape[0]}")

    See Also
    --------
    kinematic_from_ewald : Apply beam angles to get allowed reflections
    create_ewald_data : Factory function for manual construction
    """
    voltage_kv_arr: Float[Array, ""] = jnp.asarray(
        voltage_kv, dtype=jnp.float64
    )
    temperature_arr: Float[Array, ""] = jnp.asarray(
        temperature, dtype=jnp.float64
    )
    wavelength: Float[Array, ""] = wavelength_ang(voltage_kv_arr)
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / wavelength
    sphere_rad: Float[Array, ""] = k_mag
    cell_lengths: Float[Array, "3"] = crystal.cell_lengths
    cell_angles: Float[Array, "3"] = crystal.cell_angles
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        cell_lengths[0],
        cell_lengths[1],
        cell_lengths[2],
        cell_angles[0],
        cell_angles[1],
        cell_angles[2],
        in_degrees=True,
    )
    g_vecs: Float[Array, "N 3"] = generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    hs: Int[Array, "n_h"] = jnp.arange(-hmax, hmax + 1, dtype=jnp.int32)
    ks: Int[Array, "n_k"] = jnp.arange(-kmax, kmax + 1, dtype=jnp.int32)
    ls: Int[Array, "n_l"] = jnp.arange(-lmax, lmax + 1, dtype=jnp.int32)
    hh: Int[Array, "n_h n_k n_l"]
    kk: Int[Array, "n_h n_k n_l"]
    ll: Int[Array, "n_h n_k n_l"]
    hh, kk, ll = jnp.meshgrid(hs, ks, ls, indexing="ij")
    hkl: Int[Array, "N 3"] = jnp.stack(
        [hh.ravel(), kk.ravel(), ll.ravel()], axis=-1
    )
    g_mags: Float[Array, "N"] = jnp.linalg.norm(g_vecs, axis=-1)
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )

    def _compute_sf(g_vec: Float[Array, "3"]) -> Complex[Array, ""]:
        return _compute_structure_factor_single(
            g_vector=g_vec,
            atom_positions=atom_positions,
            atomic_numbers=atomic_numbers,
            temperature=temperature_arr,
        )

    structure_factors: Complex[Array, "N"] = jax.vmap(_compute_sf)(g_vecs)
    intensities: Float[Array, "N"] = jnp.abs(structure_factors) ** 2
    ewald_data: EwaldData = create_ewald_data(
        wavelength_ang=wavelength,
        k_magnitude=k_mag,
        sphere_radius=sphere_rad,
        recip_vectors=recip_vecs,
        hkl_grid=hkl,
        g_vectors=g_vecs,
        g_magnitudes=g_mags,
        structure_factors=structure_factors,
        intensities=intensities,
    )
    return ewald_data


@jaxtyped(typechecker=beartype)
def ewald_allowed_reflections(
    ewald: EwaldData,
    theta_deg: scalar_float,
    phi_deg: scalar_float,
    tolerance: scalar_float = 0.05,
    domain_extent_ang: Float[Array, "3"] | None = None,
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
) -> tuple[Int[Array, "N"], Float[Array, "N 3"], Float[Array, "N"]]:
    """Find reflections satisfying Ewald sphere condition for given beam angles.

    Description
    -----------
    Given pre-computed EwaldData and beam orientation angles, find which
    reciprocal lattice points lie on the Ewald sphere. Supports two modes:

    1. **Binary mode** (default): Returns reflections within tolerance of
       exact Ewald sphere intersection.
    2. **Finite domain mode**: Returns all reflections with continuous
       overlap-weighted intensities based on domain size and beam parameters.

    Parameters
    ----------
    ewald : EwaldData
        Pre-computed Ewald sphere data containing reciprocal lattice points,
        structure factors, and beam wavelength.
    theta_deg : scalar_float
        Grazing incidence angle in degrees (angle from surface plane).
        Typical RHEED values: 1-5 degrees.
    phi_deg : scalar_float
        Azimuthal angle in degrees (rotation about surface normal).
        0 degrees = beam along x-axis.
    tolerance : scalar_float, optional
        Fractional tolerance for Ewald sphere intersection condition
        :math:`||k_{out}| - |k_{in}|| / |k_{in}| <` tolerance.
        Only used in binary mode (when domain_extent_ang is None).
        Default: 0.05 (5%).
    domain_extent_ang : Float[Array, "3"], optional
        Physical domain size [Lx, Ly, Lz] in Ångstroms. If provided, enables
        finite domain mode with continuous overlap weighting. Default: None
    energy_spread_frac : scalar_float, optional
        Fractional energy spread ΔE/E for shell thickness calculation.
        Only used in finite domain mode. Default: 1e-4
    beam_divergence_rad : scalar_float, optional
        Beam angular divergence in radians for shell thickness calculation.
        Only used in finite domain mode. Default: 1e-3

    Returns
    -------
    allowed_indices : Int[Array, "N"]
        Indices into ewald.hkl_grid for allowed reflections.
        In binary mode, only indices within tolerance are valid (rest are -1).
        In finite domain mode, all indices with k_out_z > 0 are included.
    k_out : Float[Array, "N 3"]
        Outgoing wavevectors for allowed reflections.
    intensities : Float[Array, "N"]
        Structure factor intensities. In binary mode: :math:`I(G) = |F(G)|^2`.
        In finite domain mode: :math:`I(G) = |F(G)|^2 \\times \\text{overlap}`.

    Flow
    ----
    1. Compute incident wavevector :math:`k_{in}` from theta, phi, and :math:`|k|`
    2. For each G vector: compute :math:`k_{out} = k_{in} + G`
    3. Check Ewald condition (binary) or compute overlap (finite domain)
    4. Filter to reflections with upward scattering (:math:`k_{out,z} > 0`)
    5. Return allowed indices, k_out vectors, and intensities

    Examples
    --------
    Binary mode (default):

    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ewald = rh.simul.build_ewald_data(crystal, voltage_kv=15.0, hmax=3, kmax=3, lmax=2)
    >>> indices, k_out, intensities = rh.simul.ewald_allowed_reflections(
    ...     ewald=ewald,
    ...     theta_deg=2.0,
    ...     phi_deg=0.0,
    ... )

    Finite domain mode:

    >>> import jax.numpy as jnp
    >>> domain = jnp.array([100., 100., 50.])  # 100x100x50 Å domain
    >>> indices, k_out, intensities = rh.simul.ewald_allowed_reflections(
    ...     ewald=ewald,
    ...     theta_deg=2.0,
    ...     phi_deg=0.0,
    ...     domain_extent_ang=domain,
    ... )

    See Also
    --------
    build_ewald_data : Build EwaldData from crystal structure
    finite_domain_intensities : Alternative API for finite domain physics
    """
    theta_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(theta_deg, dtype=jnp.float64)
    )
    phi_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(phi_deg, dtype=jnp.float64)
    )

    k_mag: Float[Array, ""] = ewald.k_magnitude

    # Incident wavevector: grazing incidence from (cos(phi), sin(phi), 0) direction
    # tilted up by theta from the surface (z=0 plane)
    # k_in points INTO the surface, so z-component is negative
    k_in: Float[Array, "3"] = k_mag * jnp.array(
        [
            jnp.cos(theta_rad) * jnp.cos(phi_rad),
            jnp.cos(theta_rad) * jnp.sin(phi_rad),
            -jnp.sin(theta_rad),
        ]
    )

    # Compute all k_out = k_in + G
    g_vecs: Float[Array, "M 3"] = ewald.g_vectors
    k_out_all: Float[Array, "M 3"] = k_in + g_vecs

    # Check for upward scattering (k_out_z > 0)
    upward_mask: Bool[Array, "M"] = k_out_all[:, 2] > 0

    if domain_extent_ang is not None:
        # Finite domain mode: compute continuous overlap weights
        domain_arr: Float[Array, "3"] = jnp.asarray(
            domain_extent_ang, dtype=jnp.float64
        )
        rod_sigma: Float[Array, "2"] = extent_to_rod_sigma(domain_arr)
        shell_sigma: Float[Array, ""] = compute_shell_sigma(
            k_magnitude=k_mag,
            energy_spread_frac=energy_spread_frac,
            beam_divergence_rad=beam_divergence_rad,
        )

        # Compute overlap factors for all G vectors
        overlap: Float[Array, "M"] = rod_ewald_overlap(
            g_vectors=g_vecs,
            k_in=k_in,
            k_magnitude=k_mag,
            rod_sigma=rod_sigma,
            shell_sigma=shell_sigma,
        )

        # Apply upward scattering mask to overlap
        overlap = jnp.where(upward_mask, overlap, 0.0)

        # In finite domain mode, return all indices with nonzero overlap
        # Use threshold to filter very small overlaps for efficiency
        overlap_threshold: float = 1e-6
        is_allowed: Bool[Array, "M"] = overlap > overlap_threshold

        # Get indices of allowed reflections
        allowed_indices: Int[Array, "N"] = jnp.where(
            is_allowed, size=g_vecs.shape[0], fill_value=-1
        )[0]

        # Extract allowed k_out and intensities weighted by overlap
        k_out: Float[Array, "N 3"] = k_out_all[allowed_indices]
        base_intensities: Float[Array, "N"] = ewald.intensities[allowed_indices]
        overlap_weights: Float[Array, "N"] = overlap[allowed_indices]
        intensities: Float[Array, "N"] = base_intensities * overlap_weights

    else:
        # Binary mode: use tolerance-based filtering (original behavior)
        k_out_mags: Float[Array, "M"] = jnp.linalg.norm(k_out_all, axis=-1)
        relative_error: Float[Array, "M"] = jnp.abs(k_out_mags - k_mag) / k_mag

        # Ewald condition satisfied AND upward scattering
        is_allowed: Bool[Array, "M"] = (relative_error < tolerance) & upward_mask

        # Get indices of allowed reflections
        allowed_indices: Int[Array, "N"] = jnp.where(
            is_allowed, size=g_vecs.shape[0], fill_value=-1
        )[0]

        # Extract allowed k_out and intensities (no overlap weighting)
        k_out: Float[Array, "N 3"] = k_out_all[allowed_indices]
        intensities: Float[Array, "N"] = ewald.intensities[allowed_indices]

    return allowed_indices, k_out, intensities
