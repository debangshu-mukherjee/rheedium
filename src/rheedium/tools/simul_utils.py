"""Shared utility functions for RHEED simulation modules.

Extended Summary
----------------
This module provides common utility functions used across multiple simulation
modules. These functions are placed here to avoid circular imports between
simulator.py, ewald.py, finite_domain.py, and kinematic.py.

Routine Listings
----------------
:func:`incident_wavevector`
    Calculate incident electron wavevector from beam parameters.
:func:`incidence_angles_to_radians`
    Convert public grazing/azimuth degrees to internal radian angles.
:func:`wavelength_ang`
    Calculate relativistic electron wavelength in angstroms.
:func:`interaction_constant`
    Relativistic electron interaction constant.

Notes
-----
These functions live in :mod:`rheedium.tools`. Prefer importing from
``rheedium.tools`` rather than the old deleted
``rheedium.simul.simul_utils`` module path.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Union
from jaxtyping import Array, Float, Num, jaxtyped

from rheedium.types import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    H_OVER_SQRT_2ME_ANG_VSQRT,
    PLANCK_CONSTANT_JS,
    RELATIVISTIC_COEFF_PER_V,
    SPEED_OF_LIGHT_MS,
    scalar_num,
)


@jaxtyped(typechecker=beartype)
def wavelength_ang(
    energy_kev: Union[scalar_num, Num[Array, "..."]],
) -> Float[Array, "..."]:
    r"""Calculate the relativistic electron wavelength in angstroms.

    Extended Summary
    ----------------
    Uses the full relativistic de Broglie wavelength formula:

        lambda = h / sqrt(2 * m_e * e * V * (1 + e*V / (2 * m_e * c^2)))

    This is more accurate than simplified approximations, especially at
    higher voltages (>=30 keV) where the difference can be several percent.

    :see: :class:`~.test_simul_utils.TestWavelengthAng`

    Parameters
    ----------
    energy_kev : Union[scalar_num, Num[Array, "..."]]
        Electron energy in kiloelectron volts.
        Could be either a scalar or an array.

    Returns
    -------
    wavelength : Float[Array, "..."]
        Electron wavelength in angstroms.

    Notes
    -----
    Physical constants used:
    - h = 6.62607015e-34 J·s (Planck constant, exact)
    - m_e = 9.1093837015e-31 kg (electron mass)
    - e = 1.602176634e-19 C (elementary charge, exact)
    - c = 299792458 m/s (speed of light, exact)

    The formula simplifies to:
        lambda(Å) = 12.2643 / sqrt(V * (1 + 0.978476e-6 * V))

    where V is in volts and the coefficient 0.978476e-6 = e / (2 * m_e * c^2).

    1. **Convert voltage** --
       Multiply kV by 1000 to obtain voltage in Volts.
    2. **Relativistic correction** --
       Compute corrected voltage
       :math:`V_{corr} = V (1 + eV / 2 m_e c^2)`.
    3. **Wavelength calculation** --
       Compute :math:`\\lambda = h / \\sqrt{2 m_e e V_{corr}}`
       and return in Ångstroms.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> lam = rh.tools.wavelength_ang(jnp.asarray(20.0))  # 20 keV
    >>> f"λ = {lam:.4f} Å"
    λ = 0.0859 Å
    """
    # Convert kV to V
    voltage_v: Float[Array, "..."] = (
        jnp.asarray(energy_kev, dtype=jnp.float64) * 1000.0
    )

    corrected_voltage: Float[Array, "..."] = voltage_v * (
        1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v
    )

    wavelength: Float[Array, "..."] = H_OVER_SQRT_2ME_ANG_VSQRT / jnp.sqrt(
        corrected_voltage
    )
    return wavelength


@jaxtyped(typechecker=beartype)
def incidence_angles_to_radians(
    theta_deg: scalar_num,
    phi_deg: scalar_num = 0.0,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Convert public grazing/azimuth degrees to internal radian angles.

    :see: :class:`~.test_simul_utils.TestIncidenceAnglesToRadians`

    Parameters
    ----------
    theta_deg : scalar_num
        Public grazing angle in degrees.
    phi_deg : scalar_num, optional
        Public azimuthal angle in degrees. Default: 0.0

    Returns
    -------
    polar_angle_rad : Float[Array, ""]
        Internal grazing/polar angle in radians.
    azimuth_angle_rad : Float[Array, ""]
        Internal azimuthal angle in radians.

    Notes
    -----
    1. Convert the public grazing angle ``theta_deg`` to radians.
    2. Convert the public azimuthal angle ``phi_deg`` to radians.
    3. Return the internal polar/azimuth pair used by detector kernels.
    """
    polar_angle_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(theta_deg, dtype=jnp.float64)
    )
    azimuth_angle_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(phi_deg, dtype=jnp.float64)
    )
    return polar_angle_rad, azimuth_angle_rad


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num = 0.0,
) -> Float[Array, "3"]:
    r"""Calculate the incident electron wavevector for RHEED geometry.

    :see: :class:`~.test_simul_utils.TestIncidentWavevector`

    Parameters
    ----------
    lam_ang : scalar_num
        Electron wavelength in angstroms.
    theta_deg : scalar_num
        Grazing angle of incidence in degrees (angle from surface).
    phi_deg : scalar_num, optional
        Azimuthal angle in degrees (in-plane rotation).
        phi=0: beam along +x axis (default, gives horizontal streaks)
        phi=90: beam along +y axis (gives vertical streaks)
        Default: 0.0

    Returns
    -------
    k_in : Float[Array, "3"]
        Incident wavevector [k_x, k_y, k_z] in reciprocal angstroms.
        The beam propagates in the surface plane at azimuthal angle phi,
        with a downward z-component determined by the grazing angle theta.

    Notes
    -----
    1. **Compute wavevector magnitude** --
       :math:`k = 2\\pi / \\lambda`.
    2. **Convert angles** --
       Convert grazing and azimuthal angles from degrees
       to radians.
    3. **Decompose into components** --
       Split :math:`k` into in-plane (:math:`k_x`, :math:`k_y`)
       and surface-normal (:math:`k_z`) components using
       trigonometric projection.
    """
    k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    polar_angle_rad: Float[Array, ""]
    azimuth_angle_rad: Float[Array, ""]
    polar_angle_rad, azimuth_angle_rad = incidence_angles_to_radians(
        theta_deg,
        phi_deg,
    )

    # In-plane component magnitude
    k_parallel: Float[Array, ""] = k_magnitude * jnp.cos(polar_angle_rad)

    # Split in-plane component into x and y based on azimuthal angle
    k_x: Float[Array, ""] = k_parallel * jnp.cos(azimuth_angle_rad)
    k_y: Float[Array, ""] = k_parallel * jnp.sin(azimuth_angle_rad)
    k_z: Float[Array, ""] = -k_magnitude * jnp.sin(polar_angle_rad)

    k_in: Float[Array, "3"] = jnp.array([k_x, k_y, k_z])
    return k_in


@jaxtyped(typechecker=beartype)
def interaction_constant(
    energy_kev: scalar_num,
    wavelength_ang: scalar_num,
) -> Float[Array, ""]:
    r"""Relativistic electron interaction constant σ in 1/(V·Å).

    Extended Summary
    ----------------
    Computes the relativistic interaction constant :math:`\\sigma`
    used in multislice calculations. Includes relativistic mass
    correction via the Lorentz factor.

    :see: :class:`~.test_simul_utils.TestInteractionConstant`

    Notes
    -----
    1. **Convert units** --
       Convert voltage from kV to V and wavelength from
       Ångstroms to metres.
    2. **Compute Lorentz factor** --
       Calculate relativistic :math:`\\gamma` from accelerating voltage.
    3. **Evaluate interaction constant** --
       :math:`\\sigma = (2\\pi m_e e \\lambda / h^2)
       \\times \\gamma` in SI, then convert
       to :math:`1/(V \\cdot \\text{Å})`.

    Parameters
    ----------
    energy_kev : scalar_num
        Accelerating voltage in kilovolts.
    wavelength_ang : scalar_num
        Relativistic electron wavelength in angstroms.

    Returns
    -------
    sigma : Float[Array, ""]
        Interaction constant σ (1 / (Volt · Ångstrom)).
    """
    voltage_v: Float[Array, ""] = (
        jnp.asarray(energy_kev, dtype=jnp.float64) * 1000.0
    )
    lam_m: Float[Array, ""] = (
        jnp.asarray(wavelength_ang, dtype=jnp.float64) * 1e-10
    )
    gamma: Float[Array, ""] = 1.0 + (ELEMENTARY_CHARGE_C * voltage_v) / (
        ELECTRON_MASS_KG * SPEED_OF_LIGHT_MS * SPEED_OF_LIGHT_MS
    )
    sigma_si: Float[Array, ""] = (
        2.0
        * jnp.pi
        * ELECTRON_MASS_KG
        * ELEMENTARY_CHARGE_C
        * lam_m
        / (PLANCK_CONSTANT_JS**2)
    ) * gamma

    # Convert from 1/(V·m) to 1/(V·Å)
    sigma_ang: Float[Array, ""] = sigma_si * 1e-10
    return sigma_ang


__all__: list[str] = [
    "incidence_angles_to_radians",
    "incident_wavevector",
    "interaction_constant",
    "wavelength_ang",
]
