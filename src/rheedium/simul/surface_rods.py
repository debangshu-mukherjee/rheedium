"""Surface reciprocal lattice rod calculations for RHEED simulations.

Extended Summary
----------------
This module provides functions for calculating Crystal Truncation Rod (CTR)
intensities, which are continuous scattering features along surface-normal
reciprocal space directions. CTRs are essential for accurate RHEED pattern
simulation as they produce the characteristic streaks observed experimentally.

Routine Listings
----------------
calculate_ctr_intensity : function
    Calculate continuous intensity along crystal truncation rods with form
    factors.
gaussian_rod_profile : function
    Gaussian lateral width profile of rods due to finite correlation length
lorentzian_rod_profile : function
    Lorentzian lateral width profile of rods due to finite correlation length
roughness_damping : function
    Gaussian roughness damping factor for CTR intensities
rod_profile_function : function
    Lateral width profile of rods due to finite correlation length
surface_structure_factor : function
    Calculate structure factor for surface with q_z dependence
integrated_rod_intensity : function
    Integrate CTR intensity over finite detector acceptance

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
CTR calculations follow the kinematic approximation with proper surface
physics.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.types import CrystalStructure, scalar_bool, scalar_float

from .form_factors import atomic_scattering_factor

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def calculate_ctr_intensity(
    hk_indices: Int[Array, "N 2"],
    q_z: Float[Array, "M"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    temperature: scalar_float = 300.0,
) -> Float[Array, "N M"]:
    """Calculate continuous intensity along crystal truncation rods.

    Description
    -----------
    Computes the intensity distribution along CTRs for given in-plane
    reciprocal lattice points (h,k). The intensity varies continuously
    along q_z due to the finite crystal thickness and surface termination.

    Parameters
    ----------
    hk_indices : Int[Array, "N 2"]
        In-plane Miller indices (h,k) for each rod. Shape (N, 2) where
        N is the number of rods to calculate.
    q_z : Float[Array, "M"]
        Perpendicular momentum transfer values in 1/Å where intensity
        is calculated. Shape (M,) for M points along each rod.
    crystal : CrystalStructure
        Crystal structure containing atomic positions and cell parameters
    surface_roughness : scalar_float
        RMS surface roughness σ_h in Angstroms
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors. Default: 300.0

    Returns
    -------
    intensities : Float[Array, "N M"]
        CTR intensities for each (h,k) rod at each q_z value.
        Shape (N, M) where N is number of rods, M is number of q_z points.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Extract atomic positions and numbers from crystal structure
    2. Build reciprocal lattice vectors from cell parameters
    3. For each (h,k) index, calculate in-plane q vector
    4. For each q_z value, construct full 3D q vector
    5. Calculate structure factor with atomic form factors
    6. Apply roughness damping to intensity
    7. Return intensity array for all rods and q_z values
    """
    atomic_positions: Float[Array, "n_atoms 3"] = crystal.cart_positions[
        :, :3
    ]
    atomic_numbers: Int[Array, "n_atoms"] = crystal.cart_positions[
        :, 3
    ].astype(jnp.int32)
    cell_lengths: Float[Array, "3"] = crystal.cell_lengths
    reciprocal_a: Float[Array, ""] = 2.0 * jnp.pi / cell_lengths[0]
    reciprocal_b: Float[Array, ""] = 2.0 * jnp.pi / cell_lengths[1]

    def calculate_single_rod_intensity(
        hk: Int[Array, "2"],
    ) -> Float[Array, "M"]:
        """Calculate intensity for a single (h,k) rod at all q_z values."""
        h_val: Float[Array, ""] = jnp.float64(hk[0])
        k_val: Float[Array, ""] = jnp.float64(hk[1])
        q_x: Float[Array, ""] = h_val * reciprocal_a
        q_y: Float[Array, ""] = k_val * reciprocal_b

        def calculate_at_qz(qz_val: Float[Array, ""]) -> Float[Array, ""]:
            """Calculate intensity at single q_z value."""
            q_vector: Float[Array, "3"] = jnp.array([q_x, q_y, qz_val])
            structure_factor: Complex[Array, ""] = surface_structure_factor(
                q_vector=q_vector,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                temperature=temperature,
                is_surface=True,
            )
            damping: Float[Array, ""] = roughness_damping(
                q_z=qz_val, sigma_height=surface_roughness
            )
            intensity: Float[Array, ""] = (
                jnp.abs(structure_factor) ** 2 * damping
            )
            return intensity

        rod_intensities: Float[Array, "M"] = jax.vmap(calculate_at_qz)(q_z)
        return rod_intensities

    all_intensities: Float[Array, "N M"] = jax.vmap(
        calculate_single_rod_intensity
    )(hk_indices)
    return all_intensities


@jaxtyped(typechecker=beartype)
def roughness_damping(
    q_z: Float[Array, "..."],
    sigma_height: scalar_float,
) -> Float[Array, "..."]:
    """Gaussian roughness damping factor for CTR intensities.

    Description
    -----------
    Calculates the damping factor due to surface roughness, which
    reduces the CTR intensity especially at large q_z values. Assumes
    Gaussian height distribution with RMS roughness σ_h.

    Parameters
    ----------
    q_z : Float[Array, "..."]
        Perpendicular momentum transfer in 1/Å. Can be scalar or array.
    sigma_height : scalar_float
        RMS surface roughness in Angstroms

    Returns
    -------
    damping : Float[Array, "..."]
        Damping factor exp(-½q_z²σ_h²) between 0 and 1

    Notes
    -----
    The algorithm proceeds as follows:

    1. Ensure roughness is non-negative
    2. Calculate exponent W = ½q_z²σ_h²
    3. Return exp(-W) damping factor
    4. Handle edge case of zero roughness (no damping)
    """
    sigma: Float[Array, ""] = jnp.maximum(
        jnp.asarray(sigma_height, dtype=jnp.float64), 0.0
    )
    epsilon: Float[Array, ""] = jnp.asarray(1e-10, dtype=jnp.float64)
    half: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
    q_z_squared: Float[Array, "..."] = jnp.square(q_z)
    sigma_squared: Float[Array, ""] = jnp.square(sigma)
    exponent: Float[Array, "..."] = half * q_z_squared * sigma_squared
    damping: Float[Array, ". .."] = jnp.exp(-exponent)
    damping_final: Float[Array, "..."] = jnp.where(
        sigma < epsilon, jnp.ones_like(q_z), damping
    )
    return damping_final


@jaxtyped(typechecker=beartype)
def gaussian_rod_profile(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
) -> Float[Array, "..."]:
    """Gaussian lateral width profile of rods due to finite correlation length.

    Description
    -----------
    Calculates the Gaussian lateral intensity profile of CTRs perpendicular
    to the rod direction. The width in reciprocal space is inversely
    proportional to the real-space correlation length.

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized Gaussian intensity profile perpendicular to rod

    Notes
    -----
    The algorithm proceeds as follows:

    1. Ensure correlation length is positive (clip to minimum value)
    2. Convert correlation length to reciprocal space width σ_q = 1/ξ

    3. Normalize q_perpendicular by σ_q
    4. Calculate Gaussian profile: exp(-½(q_⊥/σ_q)²)

    5. Return normalized profile with peak value of 1.0
    """
    xi: Float[Array, ""] = jnp.maximum(
        jnp.asarray(correlation_length, dtype=jnp.float64), 1e-10
    )
    sigma_q: Float[Array, ""] = 1.0 / xi
    half: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
    q_perp_normalized: Float[Array, "..."] = q_perpendicular / sigma_q
    profile: Float[Array, "..."] = jnp.exp(
        -half * jnp.square(q_perp_normalized)
    )
    return profile


@jaxtyped(typechecker=beartype)
def lorentzian_rod_profile(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
) -> Float[Array, "..."]:
    """Lorentzian lateral width profile of rods with finite correlation length.

    Description
    -----------
    Calculates the Lorentzian lateral intensity profile of CTRs perpendicular
    to the rod direction. This profile corresponds to exponentially decaying
    surface correlations.

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized Lorentzian intensity profile perpendicular to rod

    Notes
    -----
    The algorithm proceeds as follows:

    1. Ensure correlation length is positive (clip to minimum value)
    2. Calculate product q_⊥ × ξ
    3. Calculate Lorentzian profile: 1/(1 + (q_⊥ξ)²)
    4. Return normalized profile with peak value of 1.0
    """
    xi: Float[Array, ""] = jnp.maximum(
        jnp.asarray(correlation_length, dtype=jnp.float64), 1e-10
    )
    q_xi_product: Float[Array, "..."] = q_perpendicular * xi
    profile: Float[Array, "..."] = 1.0 / (1.0 + jnp.square(q_xi_product))
    return profile


@jaxtyped(typechecker=beartype)
def rod_profile_function(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
    profile_type: str = "gaussian",
) -> Float[Array, "..."]:
    """Lateral width profile of rods due to finite correlation length.

    Description
    -----------
    Calculates the lateral intensity profile of CTRs perpendicular to
    the rod direction using JAX-safe conditional logic. Finite correlation
    length of surface features causes rods to have finite width in reciprocal
    space.

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms
    profile_type : str, optional
        Type of profile: "gaussian" or "lorentzian".
        Default is "gaussian".

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized intensity profile perpendicular to rod

    Notes
    -----
    The algorithm proceeds as follows:

    1. Use JAX-safe conditional to select profile type
    2. Call appropriate profile function
    3. Return selected profile
    """
    is_lorentzian: Bool[Array, ""] = jnp.asarray(
        profile_type == "lorentzian", dtype=jnp.bool_
    )
    profile: Float[Array, "..."] = jax.lax.cond(
        is_lorentzian,
        lambda: lorentzian_rod_profile(q_perpendicular, correlation_length),
        lambda: gaussian_rod_profile(q_perpendicular, correlation_length),
    )
    return profile


@jaxtyped(typechecker=beartype)
def surface_structure_factor(
    q_vector: Float[Array, "3"],
    atomic_positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    temperature: scalar_float = 300.0,
    is_surface: scalar_bool = True,
) -> Complex[Array, ""]:
    """Calculate structure factor for surface with q_z dependence.

    Description
    -----------
    Computes the complex structure factor F(q) for a surface, including
    atomic form factors and Debye-Waller factors. Surface atoms are
    treated with enhanced thermal vibrations.

    Parameters
    ----------
    q_vector : Float[Array, "3"]
        3D scattering vector in 1/Å
    atomic_positions : Float[Array, "N 3"]
        Cartesian atomic positions in Angstroms
    atomic_numbers : Int[Array, "N"]
        Atomic numbers for each atom
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0
    is_surface : scalar_bool, optional
        If True, use surface-enhanced thermal factors. Default: True

    Returns
    -------
    structure_factor : Complex[Array, ""]
        Complex structure factor F(q)

    Notes
    -----
    The algorithm proceeds as follows:

    1. Calculate phase factors exp(iq·r) for each atom
    2. Get atomic scattering factors with Debye-Waller
    3. Sum weighted contributions from all atoms
    4. Return complex structure factor
    """
    n_atoms: int = atomic_positions.shape[0]
    phases: Float[Array, "N"] = jnp.einsum(
        "i,ji->j", q_vector, atomic_positions
    )
    phase_factors: Complex[Array, "N"] = jnp.exp(1j * phases)

    def get_atom_scattering(atom_idx: Int[Array, ""]) -> Float[Array, ""]:
        """Get scattering factor for single atom."""
        atomic_num: Int[Array, ""] = atomic_numbers[atom_idx]
        q_vec_expanded: Float[Array, "1 3"] = q_vector[jnp.newaxis, :]
        scattering: Float[Array, "1"] = atomic_scattering_factor(
            atomic_number=atomic_num,
            q_vector=q_vec_expanded,
            temperature=temperature,
            is_surface=is_surface,
        )
        return jnp.squeeze(scattering)

    atom_indices: Int[Array, "N"] = jnp.arange(n_atoms)
    scattering_factors: Float[Array, "N"] = jax.vmap(get_atom_scattering)(
        atom_indices
    )
    weighted_contributions: Complex[Array, "N"] = (
        scattering_factors * phase_factors
    )
    structure_factor: Complex[Array, ""] = jnp.sum(weighted_contributions)
    return structure_factor


@jaxtyped(typechecker=beartype)
def integrated_rod_intensity(
    hk_index: Int[Array, "2"],
    q_z_range: Float[Array, "2"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    detector_acceptance: scalar_float,
    n_integration_points: int = 50,
    temperature: scalar_float = 300.0,
) -> scalar_float:
    """Integrate CTR intensity over finite detector acceptance.

    Description
    -----------
    Calculates the total intensity collected by a detector with finite
    angular acceptance by integrating the CTR intensity over a range
    of q_z values. This accounts for the finite detector pixel size.

    Parameters
    ----------
    hk_index : Int[Array, "2"]
        In-plane Miller indices (h, k) for the rod
    q_z_range : Float[Array, "2"]
        Range of q_z values (min, max) in 1/Å to integrate over
    crystal : CrystalStructure
        Crystal structure for calculation
    surface_roughness : scalar_float
        RMS surface roughness in Angstroms
    detector_acceptance : scalar_float
        Angular acceptance of detector in radians
    n_integration_points : int, optional
        Number of integration points. Default: 50
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0

    Returns
    -------
    integrated_intensity : scalar_float
        Total integrated intensity over detector acceptance

    Notes
    -----
    The algorithm proceeds as follows:

    1. Create q_z array spanning integration range
    2. Calculate CTR intensity at all q_z points
    3. Apply detector acceptance window function
    4. Integrate using trapezoidal rule
    5. Return total integrated intensity
    """
    q_z_values: Float[Array, "n_points"] = jnp.linspace(
        q_z_range[0], q_z_range[1], n_integration_points
    )
    intensities: Float[Array, "1 n_points"] = calculate_ctr_intensity(
        hk_indices=hk_index[None, :],
        q_z=q_z_values,
        crystal=crystal,
        surface_roughness=surface_roughness,
        temperature=temperature,
    )
    rod_intensities: Float[Array, "n_points"] = intensities[0]
    q_z_center: Float[Array, ""] = jnp.mean(q_z_values)
    q_z_width: Float[Array, ""] = detector_acceptance

    acceptance_window: Float[Array, "n_points"] = jnp.exp(
        -0.5 * jnp.square((q_z_values - q_z_center) / q_z_width)
    )
    weighted_intensities: Float[Array, "n_points"] = (
        rod_intensities * acceptance_window
    )
    integrated_intensity: Float[Array, ""] = jnp.trapezoid(
        weighted_intensities, q_z_values
    )
    return integrated_intensity
