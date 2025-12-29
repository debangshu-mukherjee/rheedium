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
kirkland_projected_potential : function
    Calculate projected atomic potential for multislice simulations
debye_waller_factor : function
    Calculate Debye-Waller damping factor for thermal vibrations
atomic_scattering_factor : function
    Combined form factor with Debye-Waller damping
get_mean_square_displacement : function
    Calculate mean square displacement for given temperature
get_debye_temperature : function
    Get element-specific Debye temperature
load_kirkland_parameters : function
    Load Kirkland scattering parameters from data file

Notes
-----
All functions support JAX transformations and automatic differentiation.
Form factors use the Kirkland parameterization optimized for electron
scattering.

Debye temperatures are from:
- Kittel, Introduction to Solid State Physics (8th ed.)
- CRC Handbook of Chemistry and Physics
- Various experimental sources for less common elements
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.inout import kirkland_potentials
from rheedium.types import scalar_bool, scalar_float, scalar_int


# Element-specific Debye temperatures in Kelvin (index = atomic_number - 1)
# Value of 0.0 indicates no reliable data; fallback to generic model
# Sources: Kittel, CRC Handbook, experimental literature
# fmt: off
DEBYE_TEMPERATURES: Float[Array, "103"] = jnp.array([
    # Z=1-10: H, He, Li, Be, B, C, N, O, F, Ne
    110.0, 25.0, 344.0, 1440.0, 1250.0, 2230.0, 63.0, 91.0, 53.0, 75.0,
    # Z=11-20: Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca
    158.0, 400.0, 428.0, 645.0, 195.0, 200.0, 115.0, 92.0, 91.0, 230.0,
    # Z=21-30: Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
    360.0, 420.0, 380.0, 630.0, 410.0, 470.0, 445.0, 450.0, 343.0, 327.0,
    # Z=31-40: Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr
    320.0, 374.0, 282.0, 90.0, 58.0, 72.0, 56.0, 147.0, 280.0, 291.0,
    # Z=41-50: Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn
    275.0, 450.0, 351.0, 600.0, 480.0, 274.0, 225.0, 209.0, 108.0, 200.0,
    # Z=51-60: Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd
    211.0, 153.0, 55.0, 64.0, 38.0, 110.0, 142.0, 146.0, 85.0, 157.0,
    # Z=61-70: Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb
    158.0, 166.0, 127.0, 170.0, 176.0, 186.0, 191.0, 188.0, 179.0, 120.0,
    # Z=71-80: Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg
    183.0, 252.0, 240.0, 400.0, 430.0, 500.0, 420.0, 240.0, 165.0, 72.0,
    # Z=81-90: Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th
    78.0, 105.0, 119.0, 81.0, 0.0, 64.0, 0.0, 89.0, 124.0, 163.0,
    # Z=91-100: Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm
    159.0, 207.0, 163.0, 171.0, 121.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # Z=101-103: Md, No, Lr
    0.0, 0.0, 0.0,
], dtype=jnp.float64)
# fmt: on

# Atomic masses in atomic mass units (index = atomic_number - 1)
# Used for proper Debye model calculation
# fmt: off
ATOMIC_MASSES: Float[Array, "103"] = jnp.array([
    # Z=1-10
    1.008, 4.003, 6.941, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
    # Z=11-20
    22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.95, 39.10, 40.08,
    # Z=21-30
    44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38,
    # Z=31-40
    69.72, 72.63, 74.92, 78.97, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,
    # Z=41-50
    92.91, 95.95, 98.00, 101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,
    # Z=51-60
    121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,
    # Z=61-70
    145.0, 150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.0,
    # Z=71-80
    175.0, 178.5, 180.9, 183.8, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,
    # Z=81-90
    204.4, 207.2, 209.0, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,
    # Z=91-100
    231.0, 238.0, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
    # Z=101-103
    258.0, 259.0, 262.0,
], dtype=jnp.float64)
# fmt: on


@jaxtyped(typechecker=beartype)
def get_debye_temperature(
    atomic_number: scalar_int,
) -> Float[Array, ""]:
    """Get element-specific Debye temperature.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)

    Returns
    -------
    theta_d : Float[Array, ""]
        Debye temperature in Kelvin. Returns 0.0 if no data available.

    Notes
    -----
    Debye temperatures are from experimental measurements compiled from:
    - Kittel, Introduction to Solid State Physics
    - CRC Handbook of Chemistry and Physics
    - Various experimental literature

    A value of 0.0 indicates no reliable data is available for that element.
    """
    atomic_idx: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32) - 1, 0, 102
    )
    return DEBYE_TEMPERATURES[atomic_idx]


@jaxtyped(typechecker=beartype)
def get_atomic_mass(
    atomic_number: scalar_int,
) -> Float[Array, ""]:
    """Get atomic mass for an element.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)

    Returns
    -------
    mass : Float[Array, ""]
        Atomic mass in atomic mass units (amu)
    """
    atomic_idx: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32) - 1, 0, 102
    )
    return ATOMIC_MASSES[atomic_idx]


@jaxtyped(typechecker=beartype)
def load_kirkland_parameters(
    atomic_number: scalar_int,
) -> Tuple[Float[Array, "6"], Float[Array, "6"]]:
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
    a_coeffs : Float[Array, "6"]
        Amplitude coefficients for Gaussian terms
    b_coeffs : Float[Array, "6"]
        Width coefficients for Gaussian terms in Ų

    Notes
    -----
    The algorithm proceeds as follows:

    1. Validate atomic number is in valid range [1, 103]
    2. Load full Kirkland potential parameters matrix
    3. Extract row for specified atomic number
    4. Split into amplitude coefficients (even indices 0,2,4,6,8,10)
    5. Split into width coefficients (odd indices 1,3,5,7,9,11)
    6. Return both coefficient arrays
    """
    min_atomic_number: Int[Array, ""] = jnp.asarray(1, dtype=jnp.int32)
    max_atomic_number: Int[Array, ""] = jnp.asarray(103, dtype=jnp.int32)
    atomic_number_clipped: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32),
        min_atomic_number,
        max_atomic_number,
    )
    kirkland_data: Float[Array, "103 12"] = kirkland_potentials()
    atomic_index: Int[Array, ""] = atomic_number_clipped - 1
    atom_params: Float[Array, "12"] = kirkland_data[atomic_index]
    a_indices: Int[Array, "6"] = jnp.array(
        [0, 2, 4, 6, 8, 10], dtype=jnp.int32
    )
    b_indices: Int[Array, "6"] = jnp.array(
        [1, 3, 5, 7, 9, 11], dtype=jnp.int32
    )
    a_coeffs: Float[Array, "6"] = atom_params[a_indices]
    b_coeffs: Float[Array, "6"] = atom_params[b_indices]
    return a_coeffs, b_coeffs


@jaxtyped(typechecker=beartype)
def kirkland_form_factor(
    atomic_number: scalar_int,
    q_magnitude: Float[Array, "..."],
) -> Float[Array, "..."]:
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
    q_magnitude : Float[Array, "..."]
        Magnitude of scattering vector in 1/Å

    Returns
    -------
    form_factor : Float[Array, "..."]
        Atomic form factor f(q) in electron scattering units

    Notes
    -----
    The algorithm proceeds as follows:

    1. Load Kirkland parameters for the element
    2. Calculate q/(4π) term used in exponentials
    3. Compute each Gaussian term: aᵢ exp(-bᵢ(q/4π)²)
    4. Sum all six Gaussian contributions
    5. Return total form factor

    Uses the sum of Gaussians approximation:
    f(q) = Σᵢ aᵢ exp(-bᵢ(q/4π)²)
    where i runs from 1 to 6 for the Kirkland parameterization.
    """
    a_coeffs: Float[Array, "6"]
    b_coeffs: Float[Array, "6"]
    a_coeffs, b_coeffs = load_kirkland_parameters(atomic_number)
    four_pi: Float[Array, ""] = jnp.asarray(4.0 * jnp.pi, dtype=jnp.float64)
    q_over_4pi: Float[Array, "..."] = q_magnitude / four_pi
    q_over_4pi_squared: Float[Array, "..."] = jnp.square(q_over_4pi)
    expanded_q_squared: Float[Array, "... 1"] = q_over_4pi_squared[
        ..., jnp.newaxis
    ]
    expanded_b_coeffs: Float[Array, "1 6"] = b_coeffs[jnp.newaxis, :]
    exponent_terms: Float[Array, "... 6"] = (
        -expanded_b_coeffs * expanded_q_squared
    )
    gaussian_terms: Float[Array, "... 6"] = jnp.exp(exponent_terms)
    expanded_a_coeffs: Float[Array, "1 6"] = a_coeffs[jnp.newaxis, :]
    weighted_gaussians: Float[Array, "... 6"] = (
        expanded_a_coeffs * gaussian_terms
    )
    form_factor: Float[Array, "..."] = jnp.sum(weighted_gaussians, axis=-1)
    return form_factor


@jaxtyped(typechecker=beartype)
def kirkland_projected_potential(
    atomic_number: scalar_int,
    r: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Calculate projected atomic potential using Kirkland parameterization.

    Description
    -----------
    Computes the 2D projected atomic potential for multislice calculations
    using Kirkland parameterization. This is the integral of the 3D atomic
    potential along the beam direction.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    r : Float[Array, "..."]
        Radial distance from atom center in Angstroms

    Returns
    -------
    potential : Float[Array, "..."]
        Projected potential in Volt·Angstrom

    Notes
    -----
    The Kirkland projected potential is given by:

        V(r) = Σᵢ [aᵢ × K₀(2π·r·√bᵢ) + cᵢ × K₀(2π·r·√dᵢ)]

    where K₀ is the modified Bessel function of zeroth order. For numerical
    stability at small r, we use an asymptotic approximation.

    The first 6 parameters (a,b) describe the real part and the last 6
    (c,d) describe additional terms. The potential is in Volt·Angstroms.

    The conversion factor (4π²a₀e)/(m₀e) includes:
    - a₀ = 0.529177 Å (Bohr radius)
    - e = elementary charge
    - Kirkland uses different units, so we include a conversion

    References
    ----------
    Kirkland, E.J. "Advanced Computing in Electron Microscopy" (2010)
    """
    # Load Kirkland parameters
    a_coeffs: Float[Array, "6"]
    b_coeffs: Float[Array, "6"]
    a_coeffs, b_coeffs = load_kirkland_parameters(atomic_number)

    # Physical constants
    # Conversion factor for Kirkland parameterization to Volt·Angstrom
    # V(r) = Σ aᵢ × K₀(2π·r·√bᵢ) where the coefficients give V·Å²
    two_pi: Float[Array, ""] = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)

    # Avoid division by zero at r=0
    r_safe: Float[Array, "..."] = jnp.maximum(r, 1e-10)

    # Compute potential as sum of Gaussian approximations
    # For numerical stability, we use a Gaussian approximation:
    # K₀(x) ≈ -ln(x/2) - γ for small x, ≈ √(π/2x)exp(-x) for large x
    # Here we use a Gaussian form that matches the integral behavior
    expanded_r: Float[Array, "... 1"] = r_safe[..., jnp.newaxis]
    expanded_b_coeffs: Float[Array, "1 6"] = b_coeffs[jnp.newaxis, :]
    expanded_a_coeffs: Float[Array, "1 6"] = a_coeffs[jnp.newaxis, :]

    # Kirkland form: V(r) = Σ aᵢ/(bᵢ) × exp(-πr²/bᵢ) (Gaussian approximation)
    # This integrates to give the correct form factor in reciprocal space
    exponent_terms: Float[Array, "... 6"] = (
        -jnp.pi * expanded_r**2 / expanded_b_coeffs
    )
    gaussian_terms: Float[Array, "... 6"] = jnp.exp(exponent_terms)

    # Weight by aᵢ/bᵢ to get proper normalization
    weighted_terms: Float[Array, "... 6"] = (
        expanded_a_coeffs / expanded_b_coeffs * gaussian_terms
    )

    # Sum contributions and scale by 2π (from Fourier relationship)
    potential: Float[Array, "..."] = two_pi * jnp.sum(weighted_terms, axis=-1)

    return potential


@jaxtyped(typechecker=beartype)
def get_mean_square_displacement(
    atomic_number: scalar_int,
    temperature: scalar_float,
    is_surface: Optional[scalar_bool] = False,
    surface_enhancement: Optional[scalar_float] = 2.0,
) -> scalar_float:
    """Calculate mean square displacement for thermal vibrations.

    Uses element-specific Debye temperatures when available for accurate
    thermal displacement calculations. Falls back to a generic model for
    elements without tabulated Debye temperatures.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    temperature : scalar_float
        Temperature in Kelvin
    is_surface : scalar_bool, optional
        If True, apply surface enhancement factor. Default: False
    surface_enhancement : scalar_float, optional
        Enhancement factor for surface atoms. Default: 2.0

    Returns
    -------
    mean_square_displacement : scalar_float
        Mean square displacement ⟨u²⟩ in Ų

    Notes
    -----
    The Debye model for mean square displacement is:

        ⟨u²⟩ = (3 * hbar²) / (m * k_B * Θ_D) * [Φ(Θ_D/T) + 1/4]

    where:
    - hbar = reduced Planck constant
    - m = atomic mass
    - k_B = Boltzmann constant
    - Θ_D = Debye temperature
    - Φ(x) = Debye function ≈ 1/x for high T, → 0 for low T

    In the high-temperature limit (T >> Θ_D):
        ⟨u²⟩ ≈ (3 * hbar² * T) / (m * k_B * Θ_D²)

    For elements without Debye temperature data (Θ_D = 0), falls back to
    the generic scaling ⟨u²⟩ ∝ sqrt(12/Z) * T / 300K.

    Surface enhancement is applied ONLY here to avoid double-application.
    """
    # Physical constants in SI units
    hbar: float = 1.054571817e-34  # J·s (reduced Planck constant)
    k_B: float = 1.380649e-23  # J/K (Boltzmann constant)
    amu_to_kg: float = 1.66053906660e-27  # kg per amu

    # Get element-specific properties
    theta_d: Float[Array, ""] = get_debye_temperature(atomic_number)
    mass_amu: Float[Array, ""] = get_atomic_mass(atomic_number)
    mass_kg: Float[Array, ""] = mass_amu * amu_to_kg

    temperature_float: Float[Array, ""] = jnp.asarray(
        temperature, dtype=jnp.float64
    )
    atomic_number_float: Float[Array, ""] = jnp.asarray(
        atomic_number, dtype=jnp.float64
    )

    # Debye model MSD calculation (high-temperature approximation)
    # ⟨u²⟩ = 3 * hbar² * T / (m * k_B * Θ_D²)
    # Convert to Ų (1 m² = 1e20 Ų)
    m2_to_ang2: float = 1e20

    def debye_msd() -> Float[Array, ""]:
        """Calculate MSD using Debye model with element-specific Θ_D."""
        # Add small epsilon to avoid division by zero
        theta_d_safe: Float[Array, ""] = jnp.maximum(theta_d, 1.0)
        numerator: Float[Array, ""] = 3.0 * hbar**2 * temperature_float
        denominator: Float[Array, ""] = mass_kg * k_B * theta_d_safe**2
        msd_m2: Float[Array, ""] = numerator / denominator
        return msd_m2 * m2_to_ang2

    def fallback_msd() -> Float[Array, ""]:
        """Calculate MSD using generic sqrt(12/Z)*T scaling."""
        room_temp: Float[Array, ""] = jnp.asarray(300.0, dtype=jnp.float64)
        base_b: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
        z_scaling: Float[Array, ""] = jnp.sqrt(12.0 / atomic_number_float)
        t_ratio: Float[Array, ""] = temperature_float / room_temp
        b_factor: Float[Array, ""] = base_b * z_scaling * t_ratio
        eight_pi_sq: Float[Array, ""] = 8.0 * jnp.pi**2
        return b_factor / eight_pi_sq

    # Use Debye model if Θ_D is available, otherwise use fallback
    has_debye_temp: Float[Array, ""] = theta_d > 0.0
    msd: Float[Array, ""] = jnp.where(
        has_debye_temp, debye_msd(), fallback_msd()
    )

    # Apply surface enhancement
    surface_factor: Float[Array, ""] = jnp.asarray(
        surface_enhancement, dtype=jnp.float64
    )
    mean_square_displacement: Float[Array, ""] = jnp.where(
        is_surface, msd * surface_factor, msd
    )

    return mean_square_displacement


@jaxtyped(typechecker=beartype)
def debye_waller_factor(
    q_magnitude: Float[Array, "..."],
    mean_square_displacement: scalar_float,
) -> Float[Array, "..."]:
    """Calculate Debye-Waller damping factor for thermal vibrations.

    Description
    -----------
    Computes the Debye-Waller temperature factor that accounts for
    reduction in scattering intensity due to thermal atomic vibrations.

    Parameters
    ----------
    q_magnitude : Float[Array, "..."]
        Magnitude of scattering vector in 1/Å
    mean_square_displacement : scalar_float
        Mean square displacement ⟨u²⟩ in Ų

    Returns
    -------
    dw_factor : Float[Array, "..."]
        Debye-Waller damping factor exp(-W)

    Notes
    -----
    The algorithm proceeds as follows:

    1. Validate mean square displacement is non-negative
    2. Calculate W = ½⟨u²⟩q²
    3. Compute exp(-W) damping factor
    4. Return Debye-Waller factor

    The Debye-Waller factor is:
    exp(-W) = exp(-½⟨u²⟩q²)

    Surface enhancement should be applied when calculating the
    mean_square_displacement, NOT in this function, to avoid
    double-application of the enhancement factor.
    """
    msd: Float[Array, ""] = jnp.asarray(
        mean_square_displacement, dtype=jnp.float64
    )
    epsilon: Float[Array, ""] = jnp.asarray(1e-10, dtype=jnp.float64)
    msd_safe: Float[Array, ""] = jnp.maximum(msd, epsilon)
    q_squared: Float[Array, "..."] = jnp.square(q_magnitude)
    half: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
    w_exponent: Float[Array, "..."] = half * msd_safe * q_squared
    dw_factor: Float[Array, "..."] = jnp.exp(-w_exponent)
    return dw_factor


@jaxtyped(typechecker=beartype)
def atomic_scattering_factor(
    atomic_number: scalar_int,
    q_vector: Float[Array, "... 3"],
    temperature: Optional[scalar_float] = 300.0,
    is_surface: Optional[scalar_bool] = False,
) -> Float[Array, "..."]:
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
    scattering_factor : Float[Array, "..."]
        Total atomic scattering factor f(q)×exp(-W)

    Notes
    -----
    The algorithm proceeds as follows:

    1. Calculate magnitude of q vector
    2. Compute atomic form factor f(q) using Kirkland parameterization
    3. Calculate mean square displacement for temperature with surface
       enhancement.
    4. Compute Debye-Waller factor exp(-W) using the MSD
    5. Multiply form factor by Debye-Waller factor
    6. Return combined scattering factor

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
    q_magnitude: Float[Array, "..."] = jnp.linalg.norm(q_vector, axis=-1)
    form_factor: Float[Array, "..."] = kirkland_form_factor(
        atomic_number, q_magnitude
    )
    mean_square_disp: scalar_float = get_mean_square_displacement(
        atomic_number, temperature, is_surface
    )
    dw_factor: Float[Array, "..."] = debye_waller_factor(
        q_magnitude, mean_square_disp
    )
    scattering_factor: Float[Array, "..."] = form_factor * dw_factor
    return scattering_factor
