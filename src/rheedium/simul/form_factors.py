"""Atomic form factors and scattering calculations for electron diffraction.

Extended Summary
----------------
This module provides functions for calculating atomic form factors using
Kirkland parameterization, Debye-Waller temperature factors, and combined
atomic scattering factors for quantitative RHEED simulations.

Routine Listings
----------------
kirkland_form_factor : function
    Calculate atomic form factor f(q) using Kirkland parameterization
debye_waller_factor : function
    Calculate Debye-Waller damping factor for thermal vibrations
atomic_scattering_factor : function
    Combined form factor with Debye-Waller damping
get_mean_square_displacement : function
    Calculate mean square displacement for given temperature
load_kirkland_parameters : function
    Load Kirkland scattering parameters from data file

Notes
-----
All functions support JAX transformations and automatic differentiation.
Form factors use the Kirkland parameterization optimized for electron
scattering.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.inout import kirkland_potentials
from rheedium.types import scalar_bool, scalar_float, scalar_int

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def load_kirkland_parameters(
    atomic_number: scalar_int,
) -> Tuple[Float[Array, " 6"], Float[Array, " 6"]]:
    """Load Kirkland scattering parameters for a given atomic number.

    Description
    -----------
    Extracts the Kirkland parameterization coefficients for atomic form
    factors from the preloaded data. The Kirkland model uses 6 Gaussian
    terms to approximate the atomic scattering factor.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)

    Returns
    -------
    a_coeffs : Float[Array, " 6"]
        Amplitude coefficients for Gaussian terms
    b_coeffs : Float[Array, " 6"]
        Width coefficients for Gaussian terms in Ų

    Flow
    ----
    - Validate atomic number is in valid range [1, 103]
    - Load full Kirkland potential parameters matrix
    - Extract row for specified atomic number
    - Split into amplitude coefficients (even indices 0,2,4,6,8,10)
    - Split into width coefficients (odd indices 1,3,5,7,9,11)
    - Return both coefficient arrays
    """
    min_atomic_number: Int[Array, " "] = jnp.asarray(1, dtype=jnp.int32)
    max_atomic_number: Int[Array, " "] = jnp.asarray(103, dtype=jnp.int32)
    atomic_number_clipped: Int[Array, " "] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32),
        min_atomic_number,
        max_atomic_number,
    )
    kirkland_data: Float[Array, " 103 12"] = kirkland_potentials()
    atomic_index: Int[Array, " "] = atomic_number_clipped - 1
    atom_params: Float[Array, " 12"] = kirkland_data[atomic_index]
    a_indices: Int[Array, " 6"] = jnp.array(
        [0, 2, 4, 6, 8, 10], dtype=jnp.int32
    )
    b_indices: Int[Array, " 6"] = jnp.array(
        [1, 3, 5, 7, 9, 11], dtype=jnp.int32
    )
    a_coeffs: Float[Array, " 6"] = atom_params[a_indices]
    b_coeffs: Float[Array, " 6"] = atom_params[b_indices]
    return a_coeffs, b_coeffs


@jaxtyped(typechecker=beartype)
def kirkland_form_factor(
    atomic_number: scalar_int,
    q_magnitude: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Calculate atomic form factor f(q) using Kirkland parameterization.

    Description
    -----------
    Computes the atomic scattering factor for electrons using the Kirkland
    parameterization, which represents the form factor as a sum of Gaussians.
    This is optimized for electron diffraction calculations.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    q_magnitude : Float[Array, " ..."]
        Magnitude of scattering vector in 1/Å

    Returns
    -------
    form_factor : Float[Array, " ..."]
        Atomic form factor f(q) in electron scattering units

    Flow
    ----
    - Load Kirkland parameters for the element
    - Calculate q/(4π) term used in exponentials
    - Compute each Gaussian term: aᵢ exp(-bᵢ(q/4π)²)
    - Sum all six Gaussian contributions
    - Return total form factor

    Notes
    -----
    Uses the sum of Gaussians approximation:
    f(q) = Σᵢ aᵢ exp(-bᵢ(q/4π)²)
    where i runs from 1 to 6 for the Kirkland parameterization.
    """
    a_coeffs: Float[Array, " 6"]
    b_coeffs: Float[Array, " 6"]
    a_coeffs, b_coeffs = load_kirkland_parameters(atomic_number)
    four_pi: Float[Array, " "] = jnp.asarray(4.0 * jnp.pi, dtype=jnp.float64)
    q_over_4pi: Float[Array, " ..."] = q_magnitude / four_pi
    q_over_4pi_squared: Float[Array, " ..."] = jnp.square(q_over_4pi)
    expanded_q_squared: Float[Array, "... 1"] = q_over_4pi_squared[
        ..., jnp.newaxis
    ]
    expanded_b_coeffs: Float[Array, " 1 6"] = b_coeffs[jnp.newaxis, :]
    exponent_terms: Float[Array, "... 6"] = (
        -expanded_b_coeffs * expanded_q_squared
    )
    gaussian_terms: Float[Array, "... 6"] = jnp.exp(exponent_terms)
    expanded_a_coeffs: Float[Array, " 1 6"] = a_coeffs[jnp.newaxis, :]
    weighted_gaussians: Float[Array, "... 6"] = (
        expanded_a_coeffs * gaussian_terms
    )
    form_factor: Float[Array, " ..."] = jnp.sum(weighted_gaussians, axis=-1)
    return form_factor


@jaxtyped(typechecker=beartype)
def get_mean_square_displacement(
    atomic_number: scalar_int,
    temperature: scalar_float,
    is_surface: Optional[scalar_bool] = False,
) -> scalar_float:
    """Calculate mean square displacement for thermal vibrations.

    Description
    -----------
    Estimates the mean square displacement ⟨u²⟩ for atomic thermal
    vibrations using the Debye model. Surface atoms typically have
    enhanced vibrations compared to bulk atoms.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element
    temperature : scalar_float
        Temperature in Kelvin
    is_surface : scalar_bool, optional
        If True, apply surface enhancement factor. Default: False

    Returns
    -------
    mean_square_displacement : scalar_float
        Mean square displacement ⟨u²⟩ in Ų

    Flow
    ----
    - Define base Debye-Waller B factor at room temperature
    - Scale B factor by atomic number (heavier atoms vibrate less)
    - Apply temperature scaling relative to room temperature
    - Apply surface enhancement if specified (2x bulk value)
    - Convert B factor to mean square displacement
    - Return ⟨u²⟩ value

    Notes
    -----
    Uses simplified Debye model with empirical scaling.
    B = 8π²⟨u²⟩, so ⟨u²⟩ = B/(8π²)
    Surface enhancement is applied ONLY here to avoid double-application.
    """
    room_temperature: Float[Array, " "] = jnp.asarray(300.0, dtype=jnp.float64)
    base_b_factor: Float[Array, " "] = jnp.asarray(0.5, dtype=jnp.float64)
    atomic_number_float: Float[Array, " "] = jnp.asarray(
        atomic_number, dtype=jnp.float64
    )
    atomic_scaling: Float[Array, " "] = jnp.sqrt(
        jnp.asarray(12.0, dtype=jnp.float64) / atomic_number_float
    )
    temperature_float: Float[Array, " "] = jnp.asarray(
        temperature, dtype=jnp.float64
    )
    temperature_ratio: Float[Array, " "] = temperature_float / room_temperature
    b_factor: Float[Array, " "] = (
        base_b_factor * atomic_scaling * temperature_ratio
    )
    surface_enhancement: Float[Array, " "] = jnp.asarray(
        2.0, dtype=jnp.float64
    )
    enhanced_b_factor: Float[Array, " "] = jnp.where(
        is_surface, b_factor * surface_enhancement, b_factor
    )
    eight_pi_squared: Float[Array, " "] = jnp.asarray(
        8.0 * jnp.pi**2, dtype=jnp.float64
    )
    mean_square_displacement: Float[Array, " "] = (
        enhanced_b_factor / eight_pi_squared
    )
    return mean_square_displacement


@jaxtyped(typechecker=beartype)
def debye_waller_factor(
    q_magnitude: Float[Array, " ..."],
    mean_square_displacement: scalar_float,
) -> Float[Array, " ..."]:
    """Calculate Debye-Waller damping factor for thermal vibrations.

    Description
    -----------
    Computes the Debye-Waller temperature factor that accounts for
    reduction in scattering intensity due to thermal atomic vibrations.

    Parameters
    ----------
    q_magnitude : Float[Array, " ..."]
        Magnitude of scattering vector in 1/Å
    mean_square_displacement : scalar_float
        Mean square displacement ⟨u²⟩ in Ų

    Returns
    -------
    dw_factor : Float[Array, " ..."]
        Debye-Waller damping factor exp(-W)

    Flow
    ----
    - Validate mean square displacement is non-negative
    - Calculate W = ½⟨u²⟩q²
    - Compute exp(-W) damping factor
    - Return Debye-Waller factor

    Notes
    -----
    The Debye-Waller factor is:
    exp(-W) = exp(-½⟨u²⟩q²)

    Surface enhancement should be applied when calculating the
    mean_square_displacement, NOT in this function, to avoid
    double-application of the enhancement factor.
    """
    msd: Float[Array, " "] = jnp.asarray(
        mean_square_displacement, dtype=jnp.float64
    )
    epsilon: Float[Array, " "] = jnp.asarray(1e-10, dtype=jnp.float64)
    msd_safe: Float[Array, " "] = jnp.maximum(msd, epsilon)
    q_squared: Float[Array, " ..."] = jnp.square(q_magnitude)
    half: Float[Array, " "] = jnp.asarray(0.5, dtype=jnp.float64)
    w_exponent: Float[Array, " ..."] = half * msd_safe * q_squared
    dw_factor: Float[Array, " ..."] = jnp.exp(-w_exponent)
    return dw_factor


@jaxtyped(typechecker=beartype)
def atomic_scattering_factor(
    atomic_number: scalar_int,
    q_vector: Float[Array, "... 3"],
    temperature: Optional[scalar_float] = 300.0,
    is_surface: Optional[scalar_bool] = False,
) -> Float[Array, " ..."]:
    """Calculate combined atomic scattering factor with thermal damping.

    Description
    -----------
    Computes the total atomic scattering factor by combining the
    q-dependent form factor with the Debye-Waller temperature factor.
    This gives the effective scattering amplitude including thermal effects.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    q_vector : Float[Array, "... 3"]
        Scattering vector in 1/Å (can be batched)
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0
    is_surface : scalar_bool, optional
        If True, use surface-enhanced thermal vibrations. Default: False

    Returns
    -------
    scattering_factor : Float[Array, " ..."]
        Total atomic scattering factor f(q)×exp(-W)

    Flow
    ----
    - Calculate magnitude of q vector
    - Compute atomic form factor f(q) using Kirkland parameterization
    - Calculate mean square displacement for temperature with surface 
      enhancement.
    - Compute Debye-Waller factor exp(-W) using the MSD
    - Multiply form factor by Debye-Waller factor
    - Return combined scattering factor

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Silicon atom at room temperature
    >>> q_vec = jnp.array([1.0, 0.0, 0.0])  # 1/Å
    >>> f_si = rh.simul.atomic_scattering_factor(
    ...     atomic_number=14,  # Silicon
    ...     q_vector=q_vec,
    ...     temperature=300.0,
    ...     is_surface=False
    ... )
    >>> print(f"Si scattering factor at q=1.0: {f_si:.3f}")
    """
    q_magnitude: Float[Array, " ..."] = jnp.linalg.norm(q_vector, axis=-1)
    form_factor: Float[Array, " ..."] = kirkland_form_factor(
        atomic_number, q_magnitude
    )
    mean_square_disp: scalar_float = get_mean_square_displacement(
        atomic_number, temperature, is_surface
    )
    dw_factor: Float[Array, " ..."] = debye_waller_factor(
        q_magnitude, mean_square_disp
    )
    scattering_factor: Float[Array, " ..."] = form_factor * dw_factor
    return scattering_factor
