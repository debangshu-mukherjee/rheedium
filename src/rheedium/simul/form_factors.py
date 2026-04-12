"""Atomic form factors and scattering calculations for electron diffraction.

Extended Summary
----------------
This module provides functions for calculating atomic form factors using
Kirkland parameterization, Debye-Waller temperature factors, and combined
atomic scattering factors for quantitative RHEED simulations.

Routine Listings
----------------
:func:`kirkland_form_factor`
    Calculate atomic form factor f(q) using Kirkland
    parameterization.
:func:`kirkland_projected_potential`
    Calculate projected atomic potential for multislice
    simulations.
:func:`load_lobato_parameters`
    Load Lobato-van Dyck scattering parameters from
    data file.
:func:`lobato_form_factor`
    Calculate atomic form factor f_e(q) using
    Lobato-van Dyck parameterization.
:func:`lobato_projected_potential`
    Calculate projected atomic potential using
    Lobato-van Dyck parameterization.
:func:`projected_potential`
    Projected potential with selectable parameterization
    (Lobato or Kirkland).
:func:`debye_waller_factor`
    Calculate Debye-Waller damping factor for thermal vibrations.
:func:`atomic_scattering_factor`
    Combined form factor with Debye-Waller damping.
:func:`get_mean_square_displacement`
    Calculate mean square displacement for given temperature.
:func:`get_debye_temperature`
    Get element-specific Debye temperature.
:func:`load_kirkland_parameters`
    Load Kirkland scattering parameters from data file.

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

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.inout import (
    atomic_masses,
    debye_temperatures,
    kirkland_potentials,
    lobato_potentials,
)
from rheedium.tools import bessel_k0, bessel_k1
from rheedium.types import (
    AMU_TO_KG,
    BOLTZMANN_CONSTANT_JK,
    HBAR_JS,
    M2_TO_ANG2,
    KirklandParameters,
    create_kirkland_parameters,
    scalar_bool,
    scalar_float,
    scalar_int,
)

DEBYE_TEMPERATURES: Float[Array, "103"] = debye_temperatures()
ATOMIC_MASSES: Float[Array, "103"] = atomic_masses()
LOBATO_PARAMS: Float[Array, "103 10"] = lobato_potentials()


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

    1. **Clip index** --
       Map atomic number to zero-based array index,
       clamped to [0, 102].
    2. **Table lookup** --
       Return :data:`DEBYE_TEMPERATURES` entry. Zero
       indicates no reliable data.
    """
    atomic_idx: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32) - 1, 0, 102
    )
    theta_d: Float[Array, ""] = DEBYE_TEMPERATURES[atomic_idx]
    return theta_d


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

    Notes
    -----
    1. **Clip index** --
       Map atomic number to zero-based array index,
       clamped to [0, 102].
    2. **Table lookup** --
       Return :data:`ATOMIC_MASSES` entry in amu.
    """
    atomic_idx: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32) - 1, 0, 102
    )
    mass: Float[Array, ""] = ATOMIC_MASSES[atomic_idx]
    return mass


@jaxtyped(typechecker=beartype)
def load_kirkland_parameters(
    atomic_number: scalar_int,
) -> KirklandParameters:
    """Load Kirkland scattering parameters for a given atomic number.

    Extracts the Kirkland parameterization coefficients for atomic form
    factors from the preloaded data. The Kirkland model uses three
    Lorentzian terms followed by three Gaussian terms.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)

    Returns
    -------
    parameters : KirklandParameters
        Structured coefficient container with Lorentzian amplitudes and
        scales plus Gaussian amplitudes and scales.

    Notes
    -----
    1. **Validate range** --
       Clip atomic number to [1, 103].
    2. **Load parameter matrix** --
       Full Kirkland potential parameters (103 × 12).
    3. **Extract row** --
       Select row for the specified atomic number.
    4. **Split coefficients** --
       Return alternating coefficients so downstream code can split
       them into the first three Lorentzian pairs and last three
       Gaussian pairs.

    See Also
    --------
    kirkland_form_factor : Compute form factor using these parameters
    kirkland_projected_potential : Compute projected potential
    rheedium.inout.kirkland_potentials : Load raw Kirkland data from file
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
    amplitudes: Float[Array, "6"] = atom_params[a_indices]
    scales: Float[Array, "6"] = atom_params[b_indices]
    parameters: KirklandParameters = create_kirkland_parameters(
        lorentzian_amplitudes=amplitudes[:3],
        lorentzian_scales=scales[:3],
        gaussian_amplitudes=amplitudes[3:],
        gaussian_scales=scales[3:],
    )
    return parameters


@jaxtyped(typechecker=beartype)
def kirkland_form_factor(
    atomic_number: scalar_int,
    q_magnitude: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Calculate atomic form factor f(q) using Kirkland parameterization.

    Computes the atomic scattering factor for electrons using the Kirkland
    parameterization:

    .. math::

        f_e(s) = \sum_{i=1}^{3} \frac{a_i}{s^2 + b_i}
        + \sum_{i=1}^{3} c_i \exp(-d_i s^2)

    where :math:`s = q / (4\pi)`.

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
    1. **Load parameters** --
       Kirkland :math:`a_i, b_i` for the element.
    2. **Prepare q term** --
       :math:`s = q / (4\\pi)`.
    3. **Lorentzian terms** --
       :math:`a_i / (s^2 + b_i)` for :math:`i = 1 \\ldots 3`.
    4. **Gaussian terms** --
       :math:`c_i \\exp(-d_i s^2)` for :math:`i = 1 \\ldots 3`.
    5. **Sum contributions** --
       Add both families of terms.

    See Also
    --------
    load_kirkland_parameters : Load the Kirkland coefficients
    atomic_scattering_factor : Form factor with Debye-Waller damping
    debye_waller_factor : Thermal damping factor
    """
    parameters: KirklandParameters = load_kirkland_parameters(atomic_number)
    four_pi: Float[Array, ""] = jnp.asarray(4.0 * jnp.pi, dtype=jnp.float64)
    q_over_4pi: Float[Array, "..."] = q_magnitude / four_pi
    q_over_4pi_squared: Float[Array, "... 1"] = jnp.square(q_over_4pi)[
        ..., jnp.newaxis
    ]
    lorentzian_terms: Float[Array, "... 3"] = parameters.lorentzian_amplitudes[
        jnp.newaxis, :
    ] / (q_over_4pi_squared + parameters.lorentzian_scales[jnp.newaxis, :])
    gaussian_terms: Float[Array, "... 3"] = parameters.gaussian_amplitudes[
        jnp.newaxis, :
    ] * jnp.exp(
        -parameters.gaussian_scales[jnp.newaxis, :] * q_over_4pi_squared
    )
    form_factor: Float[Array, "..."] = jnp.sum(
        lorentzian_terms + gaussian_terms,
        axis=-1,
    )
    return form_factor


@jaxtyped(typechecker=beartype)
def kirkland_projected_potential(
    atomic_number: scalar_int,
    r: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Calculate projected atomic potential using Kirkland parameterization.

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
    The Kirkland projected potential is the analytic real-space partner
    of the mixed Lorentzian/Gaussian reciprocal-space fit:

    .. math::

        V_z(r) = 4\pi^2 a_0 e
        \left[
            \sum_{i=1}^{3} a_i K_0(2\pi r \sqrt{b_i}) +
            \sum_{i=1}^{3} \frac{c_i}{d_i}
            \exp\!\left(-\frac{\pi^2 r^2}{d_i}\right)
        \right]

    1. **Load parameters** --
       Kirkland :math:`a_i, b_i` for the element.
    2. **Safe radial distance** --
       Clamp :math:`r` to avoid singularity at zero.
    3. **Lorentzian terms** --
       Evaluate :math:`a_i K_0(2\pi r \sqrt{b_i})`.
    4. **Gaussian terms** --
       Evaluate :math:`(c_i / d_i) \exp(-\pi^2 r^2 / d_i)`.
    5. **Sum and scale** --
       Multiply by :math:`4\pi^2 a_0 e`.

    See Also
    --------
    load_kirkland_parameters : Load the Kirkland coefficients
    kirkland_form_factor : Reciprocal-space form factor

    References
    ----------
    Kirkland, E.J. "Advanced Computing in Electron Microscopy" (2010)
    """
    parameters: KirklandParameters = load_kirkland_parameters(atomic_number)
    two_pi: Float[Array, ""] = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    prefactor: Float[Array, ""] = jnp.asarray(
        47.87801 * 2.0 * jnp.pi,
        dtype=jnp.float64,
    )
    r_safe: Float[Array, "..."] = jnp.maximum(r, 1e-10)
    expanded_r: Float[Array, "... 1"] = r_safe[..., jnp.newaxis]
    lorentzian_arguments: Float[Array, "... 3"] = (
        two_pi
        * expanded_r
        * jnp.sqrt(parameters.lorentzian_scales[jnp.newaxis, :])
    )
    lorentzian_terms: Float[Array, "... 3"] = parameters.lorentzian_amplitudes[
        jnp.newaxis, :
    ] * bessel_k0(lorentzian_arguments)
    gaussian_terms: Float[Array, "... 3"] = (
        parameters.gaussian_amplitudes[jnp.newaxis, :]
        / parameters.gaussian_scales[jnp.newaxis, :]
    ) * jnp.exp(
        -(jnp.pi**2)
        * expanded_r**2
        / parameters.gaussian_scales[jnp.newaxis, :]
    )
    potential: Float[Array, "..."] = prefactor * jnp.sum(
        lorentzian_terms + gaussian_terms,
        axis=-1,
    )
    return potential


@jaxtyped(typechecker=beartype)
def load_lobato_parameters(
    atomic_number: scalar_int,
) -> Tuple[Float[Array, "5"], Float[Array, "5"]]:
    """Load Lobato-van Dyck scattering parameters for a given element.

    Extracts the Lobato-van Dyck (2014) parameterization coefficients
    for electron scattering factors from the preloaded data. The model
    uses 5 terms of the form a_i (2 + b_i q^2) / (1 + b_i q^2)^2.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)

    Returns
    -------
    a_coeffs : Float[Array, "5"]
        Amplitude coefficients for the 5 Lobato terms
    b_coeffs : Float[Array, "5"]
        Width coefficients for the 5 Lobato terms in Angstroms^2

    Notes
    -----
    1. **Validate range** --
       Clip atomic number to [1, 103].
    2. **Extract row** --
       Select row for the specified atomic number.
    3. **Split coefficients** --
       Amplitude :math:`a_i` at even indices (0, 2, 4, 6, 8),
       width :math:`b_i` at odd indices (1, 3, 5, 7, 9).

    See Also
    --------
    lobato_form_factor : Compute form factor using these parameters
    lobato_projected_potential : Compute projected potential
    rheedium.inout.lobato_potentials : Load raw Lobato data from file

    References
    ----------
    Lobato, I.I. and Van Dyck, D. (2014). Acta Cryst. A70, 636--649.
    """
    min_atomic_number: Int[Array, ""] = jnp.asarray(1, dtype=jnp.int32)
    max_atomic_number: Int[Array, ""] = jnp.asarray(103, dtype=jnp.int32)
    atomic_number_clipped: Int[Array, ""] = jnp.clip(
        jnp.asarray(atomic_number, dtype=jnp.int32),
        min_atomic_number,
        max_atomic_number,
    )
    atomic_index: Int[Array, ""] = atomic_number_clipped - 1
    atom_params: Float[Array, "10"] = LOBATO_PARAMS[atomic_index]
    a_indices: Int[Array, "5"] = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
    b_indices: Int[Array, "5"] = jnp.array([1, 3, 5, 7, 9], dtype=jnp.int32)
    a_coeffs: Float[Array, "5"] = atom_params[a_indices]
    b_coeffs: Float[Array, "5"] = atom_params[b_indices]
    return (a_coeffs, b_coeffs)


@jaxtyped(typechecker=beartype)
def lobato_form_factor(
    atomic_number: scalar_int,
    q_magnitude: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Electron scattering factor using Lobato-van Dyck parameterization.

    Computes the atomic scattering factor for electrons using the
    Lobato-van Dyck (2014) parameterization, which obeys all physical
    constraints including the correct high-q Bethe asymptotic
    :math:`f_e \propto q^{-2}`.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    q_magnitude : Float[Array, "..."]
        Magnitude of scattering vector in 1/Angstrom.
        Here q = \|g\| where g is the reciprocal lattice vector.

    Returns
    -------
    form_factor : Float[Array, "..."]
        Electron scattering factor f_e(q) in Angstrom

    Notes
    -----
    The Lobato-van Dyck scattering factor is

    .. math::

        f_e(q) = \sum_{i=1}^{5}
        a_i \frac{2 + b_i\,q^2}{(1 + b_i\,q^2)^2}

    where :math:`g = q/(2\pi)` is the scattering vector magnitude in
    the convention of the original paper
    (:math:`g = 2\sin\theta/\lambda`). The coefficients
    :math:`a_i, b_i` are tabulated for Z = 1--103.

    1. **Load parameters** --
       Lobato :math:`a_i, b_i` for the element.
    2. **Prepare g term** --
       :math:`g^2 = (q / (2\pi))^2`.
    3. **Rational terms** --
       :math:`a_i (2 + b_i g^2) / (1 + b_i g^2)^2`
       for :math:`i = 1 \ldots 5`.
    4. **Sum contributions** --
       :math:`f_e = \sum_i` rational terms.

    See Also
    --------
    load_lobato_parameters : Load the Lobato coefficients
    kirkland_form_factor : Alternative Kirkland parameterization

    References
    ----------
    Lobato, I.I. and Van Dyck, D. (2014). Acta Cryst. A70, 636--649.
    """
    a_coeffs: Float[Array, "5"]
    b_coeffs: Float[Array, "5"]
    a_coeffs, b_coeffs = load_lobato_parameters(atomic_number)
    two_pi: Float[Array, ""] = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    g_squared: Float[Array, "..."] = jnp.square(q_magnitude / two_pi)
    expanded_s_sq: Float[Array, "... 1"] = g_squared[..., jnp.newaxis]
    expanded_a: Float[Array, "1 5"] = a_coeffs[jnp.newaxis, :]
    expanded_b: Float[Array, "1 5"] = b_coeffs[jnp.newaxis, :]
    b_s_sq: Float[Array, "... 5"] = expanded_b * expanded_s_sq
    numerator: Float[Array, "... 5"] = expanded_a * (2.0 + b_s_sq)
    denominator: Float[Array, "... 5"] = jnp.square(1.0 + b_s_sq)
    rational_terms: Float[Array, "... 5"] = numerator / denominator
    form_factor: Float[Array, "..."] = jnp.sum(rational_terms, axis=-1)
    return form_factor


@jaxtyped(typechecker=beartype)
def lobato_projected_potential(
    atomic_number: scalar_int,
    r: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Projected atomic potential using Lobato-van Dyck parameterization.

    Computes the 2D projected atomic potential for multislice calculations
    using the Lobato-van Dyck (2014) parameterization. This is the integral
    of the 3D atomic potential along the beam direction, expressed
    analytically in terms of modified Bessel functions K_0 and K_1.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    r : Float[Array, "..."]
        Radial distance from atom centre in Angstroms

    Returns
    -------
    potential : Float[Array, "..."]
        Projected potential in Volt-Angstrom

    Notes
    -----
    The Lobato-van Dyck projected potential (Eq. 16 of [1]_) is

    .. math::

        V_z(r) = \frac{2\pi\,h^2}{m_e\,e} \sum_{i=1}^{5} a_i
        \left[
            \frac{\pi}{\sqrt{b_i}}\,K_0\!\bigl(\tfrac{2\pi r}
            {\sqrt{b_i}}\bigr)
            + \frac{2\pi^2 r}{b_i}\,K_1\!\bigl(\tfrac{2\pi r}
            {\sqrt{b_i}}\bigr)
        \right]

    where K_0 and K_1 are modified Bessel functions of the second
    kind, obtained from :mod:`rheedium.tools`.

    The prefactor :math:`h^2 / (2\pi m_e e)` evaluates to
    47.87801 V-Angstrom^2.

    1. **Load parameters** --
       Lobato :math:`a_i, b_i` for the element.
    2. **Safe radial distance** --
       Clamp :math:`r` to avoid singularity at zero.
    3. **Bessel arguments** --
       :math:`x_i = 2\pi r / \sqrt{b_i}` for each term.
    4. **K_0 and K_1 contributions** --
       Evaluate Bessel functions and combine with
       prefactors.
    5. **Sum and scale** --
       Multiply sum by :math:`2\pi \times 47.87801`.

    See Also
    --------
    load_lobato_parameters : Load the Lobato coefficients
    lobato_form_factor : Reciprocal-space form factor
    kirkland_projected_potential : Alternative Kirkland parameterization
    bessel_k0 : Modified Bessel K_0
    bessel_k1 : Modified Bessel K_1

    References
    ----------
    .. [1] Lobato, I.I. and Van Dyck, D. (2014). Acta Cryst. A70,
       636--649, Eq. (16).
    """
    a_coeffs: Float[Array, "5"]
    b_coeffs: Float[Array, "5"]
    a_coeffs, b_coeffs = load_lobato_parameters(atomic_number)
    r_safe: Float[Array, "..."] = jnp.maximum(r, 1e-10)
    two_pi: Float[Array, ""] = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    prefactor: Float[Array, ""] = jnp.asarray(
        47.87801 * 2.0 * jnp.pi, dtype=jnp.float64
    )
    expanded_r: Float[Array, "... 1"] = r_safe[..., jnp.newaxis]
    expanded_a: Float[Array, "1 5"] = a_coeffs[jnp.newaxis, :]
    expanded_b: Float[Array, "1 5"] = b_coeffs[jnp.newaxis, :]
    sqrt_b: Float[Array, "1 5"] = jnp.sqrt(expanded_b)
    bessel_arg: Float[Array, "... 5"] = two_pi * expanded_r / sqrt_b
    k0_vals: Float[Array, "... 5"] = bessel_k0(bessel_arg)
    k1_vals: Float[Array, "... 5"] = bessel_k1(bessel_arg)
    k0_contribution: Float[Array, "... 5"] = (
        expanded_a * jnp.pi / sqrt_b * k0_vals
    )
    k1_contribution: Float[Array, "... 5"] = (
        expanded_a * two_pi * jnp.pi * expanded_r / expanded_b * k1_vals
    )
    per_term: Float[Array, "... 5"] = k0_contribution + k1_contribution
    potential: Float[Array, "..."] = prefactor * jnp.sum(per_term, axis=-1)
    return potential


@jaxtyped(typechecker=beartype)
def projected_potential(
    atomic_number: scalar_int,
    r: Float[Array, "..."],
    parameterization: str = "lobato",
) -> Float[Array, "..."]:
    r"""Projected atomic potential with selectable parameterization.

    Dispatches to either the Lobato-van Dyck or Kirkland projected
    potential calculation via ``jax.lax.cond``. Both parameterizations
    share the same call signature and return shape.

    Parameters
    ----------
    atomic_number : scalar_int
        Atomic number (Z) of the element (1-103)
    r : Float[Array, "..."]
        Radial distance from atom centre in Angstroms
    parameterization : str, optional
        Potential model: ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    potential : Float[Array, "..."]
        Projected potential in Volt-Angstrom

    Notes
    -----
    The ``parameterization`` string is resolved to a boolean predicate
    at trace time and passed to ``jax.lax.cond``, which compiles both
    branches but executes only the selected one at runtime.

    1. **Resolve predicate** --
       ``use_lobato = (parameterization == "lobato")``.
    2. **Dispatch** --
       ``jax.lax.cond`` selects :func:`lobato_projected_potential`
       or :func:`kirkland_projected_potential`.

    See Also
    --------
    lobato_projected_potential : Lobato-van Dyck parameterization
    kirkland_projected_potential : Kirkland parameterization
    """
    use_lobato: Float[Array, ""] = jnp.asarray(parameterization == "lobato")
    potential: Float[Array, "..."] = jax.lax.cond(
        use_lobato,
        lobato_projected_potential,
        kirkland_projected_potential,
        atomic_number,
        r,
    )
    return potential


@jaxtyped(typechecker=beartype)
def get_mean_square_displacement(
    atomic_number: scalar_int,
    temperature: scalar_float,
    is_surface: scalar_bool = False,
    surface_enhancement: scalar_float = 2.0,
) -> Float[Array, ""]:
    r"""Calculate mean square displacement for thermal vibrations.

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
    mean_square_displacement : Float[Array, ""]
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

    1. **Retrieve Debye temperature** --
       Look up element-specific :math:`\\Theta_D`.
    2. **Debye model MSD** --
       :math:`\\langle u^2 \\rangle = 3 \\hbar^2 T
       / (m k_B \\Theta_D^2)` (high-T limit).
    3. **Fallback model** --
       If :math:`\\Theta_D = 0`, use generic scaling
       :math:`\\propto \\sqrt{12/Z} \\times T/300`.
    4. **Surface enhancement** --
       Multiply by enhancement factor if atom is
       flagged as surface.

    See Also
    --------
    get_debye_temperature : Element-specific Debye temperatures
    get_atomic_mass : Atomic masses for MSD calculation
    debye_waller_factor : Convert MSD to damping factor
    """
    hbar: float = HBAR_JS
    k_b: float = BOLTZMANN_CONSTANT_JK
    amu_to_kg: float = AMU_TO_KG
    theta_d: Float[Array, ""] = get_debye_temperature(atomic_number)
    mass_amu: Float[Array, ""] = get_atomic_mass(atomic_number)
    mass_kg: Float[Array, ""] = mass_amu * amu_to_kg
    temperature_float: Float[Array, ""] = jnp.asarray(
        temperature, dtype=jnp.float64
    )
    atomic_number_float: Float[Array, ""] = jnp.asarray(
        atomic_number, dtype=jnp.float64
    )
    m2_to_ang2: float = M2_TO_ANG2

    def debye_msd() -> Float[Array, ""]:
        """Calculate MSD using Debye model with element-specific Θ_D."""
        theta_d_safe: Float[Array, ""] = jnp.maximum(theta_d, 1.0)
        numerator: Float[Array, ""] = 3.0 * hbar**2 * temperature_float
        denominator: Float[Array, ""] = mass_kg * k_b * theta_d_safe**2
        msd_m2: Float[Array, ""] = numerator / denominator
        return msd_m2 * m2_to_ang2

    def fallback_msd() -> Float[Array, ""]:
        """Calculate MSD using generic sqrt(12/Z)*T scaling."""
        room_temp: Float[Array, ""] = jnp.asarray(300.0, dtype=jnp.float64)
        base_b: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
        z_scaling: Float[Array, ""] = jnp.sqrt(12.0 / atomic_number_float)
        t_ratio: Float[Array, ""] = temperature_float / room_temp
        b_factor: Float[Array, ""] = base_b * z_scaling * t_ratio
        eight_pi_sq: Float[Array, ""] = jnp.asarray(8.0) * (jnp.pi**2)
        return b_factor / eight_pi_sq

    has_debye_temp: Float[Array, ""] = theta_d > 0.0
    msd: Float[Array, ""] = jnp.where(
        has_debye_temp, debye_msd(), fallback_msd()
    )
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
    r"""Calculate Debye-Waller damping factor for thermal vibrations.

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
    Surface enhancement should be applied when calculating the
    mean_square_displacement, NOT in this function, to avoid
    double-application of the enhancement factor.

    1. **Validate MSD** --
       Ensure :math:`\\langle u^2 \\rangle \\geq 0`.
    2. **Compute exponent** --
       :math:`W = \\tfrac{1}{2} \\langle u^2 \\rangle q^2`.
    3. **Evaluate factor** --
       :math:`\\exp(-W)`.

    See Also
    --------
    get_mean_square_displacement : Calculate MSD from temperature
    atomic_scattering_factor : Combines form factor with Debye-Waller
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
    temperature: scalar_float = 300.0,
    is_surface: scalar_bool = False,
) -> Float[Array, "..."]:
    r"""Calculate combined atomic scattering factor with thermal damping.

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
    1. **q magnitude** --
       :math:`|q| = \\|q\\|`.
    2. **Form factor** --
       Evaluate Kirkland :math:`f(|q|)`.
    3. **Mean square displacement** --
       Element- and temperature-dependent
       :math:`\\langle u^2 \\rangle` with optional surface
       enhancement.
    4. **Debye-Waller factor** --
       :math:`\\exp(-\\tfrac{1}{2} \\langle u^2 \\rangle q^2)`.
    5. **Combined factor** --
       :math:`f(q) \\times \\exp(-W)`.

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
    ...     is_surface=False,
    ... )
    >>> print(f"Si scattering factor at q=1.0: {f_si:.3f}")

    See Also
    --------
    kirkland_form_factor : Calculate form factor without thermal damping
    get_mean_square_displacement : Calculate thermal displacement
    debye_waller_factor : Calculate thermal damping factor
    """
    q_magnitude: Float[Array, "..."] = jnp.linalg.norm(q_vector, axis=-1)
    form_factor: Float[Array, "..."] = kirkland_form_factor(
        atomic_number, q_magnitude
    )
    mean_square_disp: Float[Array, ""] = get_mean_square_displacement(
        atomic_number, temperature, is_surface
    )
    dw_factor: Float[Array, "..."] = debye_waller_factor(
        q_magnitude, mean_square_disp
    )
    scattering_factor: Float[Array, "..."] = form_factor * dw_factor
    return scattering_factor


__all__: list[str] = [
    "atomic_scattering_factor",
    "debye_waller_factor",
    "get_atomic_mass",
    "get_debye_temperature",
    "get_mean_square_displacement",
    "kirkland_form_factor",
    "kirkland_projected_potential",
    "load_kirkland_parameters",
    "load_lobato_parameters",
    "lobato_form_factor",
    "lobato_projected_potential",
    "projected_potential",
]
