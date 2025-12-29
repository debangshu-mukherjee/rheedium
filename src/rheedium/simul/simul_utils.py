"""Shared utility functions for RHEED simulation modules.

Extended Summary
----------------
This module provides common utility functions used across multiple simulation
modules. These functions are placed here to avoid circular imports between
simulator.py, ewald.py, finite_domain.py, and kinematic.py.

Routine Listings
----------------
incident_wavevector : function
    Calculate incident electron wavevector from beam parameters
wavelength_ang : function
    Calculate relativistic electron wavelength in angstroms

Notes
-----
These functions are re-exported from the main simul module for backward
compatibility. Import from rheedium.simul, not rheedium.simul.simul_utils.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Union
from jaxtyping import Array, Float, Num, jaxtyped

from rheedium.types import scalar_float, scalar_num


@jaxtyped(typechecker=beartype)
def wavelength_ang(
    voltage_kv: Union[scalar_num, Num[Array, "..."]],
) -> Float[Array, "..."]:
    """Calculate the relativistic electron wavelength in angstroms.

    Uses the full relativistic de Broglie wavelength formula:

        lambda = h / sqrt(2 * m_e * e * V * (1 + e*V / (2 * m_e * c^2)))

    This is more accurate than simplified approximations, especially at
    higher voltages (>=30 keV) where the difference can be several percent.

    Parameters
    ----------
    voltage_kv : Union[scalar_num, Num[Array, "..."]]
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

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> lam = rh.simul.wavelength_ang(jnp.asarray(20.0))  # 20 keV
    >>> print(f"λ = {lam:.4f} Å")
    λ = 0.0859 Å
    """
    # Convert kV to V
    voltage_v: Float[Array, "..."] = (
        jnp.asarray(voltage_kv, dtype=jnp.float64) * 1000.0
    )

    # Exact relativistic correction coefficient: e / (2 * m_e * c^2)
    # = 1.602176634e-19 / (2 * 9.1093837015e-31 * (299792458)^2)
    # = 0.978476e-6 V^-1
    relativistic_coeff: float = 0.978476e-6

    # Relativistically corrected voltage: V * (1 + e*V / (2*m_e*c^2))
    corrected_voltage: Float[Array, "..."] = voltage_v * (
        1.0 + relativistic_coeff * voltage_v
    )

    # h / sqrt(2 * m_e * e) in Å·V^0.5 units
    # = 6.62607015e-34 / sqrt(2 * 9.1093837015e-31 * 1.602176634e-19)
    # = 12.2643 Å·V^0.5
    h_over_sqrt_2me: float = 12.2643

    wavelength: Float[Array, "..."] = h_over_sqrt_2me / jnp.sqrt(
        corrected_voltage
    )
    return wavelength


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
) -> Float[Array, "3"]:
    """Calculate the incident electron wavevector for RHEED geometry.

    Parameters
    ----------
    lam_ang : scalar_float
        Electron wavelength in angstroms.
    theta_deg : scalar_float
        Grazing angle of incidence in degrees (angle from surface).
    phi_deg : scalar_float, optional
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
    """
    k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta_rad: Float[Array, ""] = jnp.deg2rad(theta_deg)
    phi_rad: Float[Array, ""] = jnp.deg2rad(phi_deg)

    # In-plane component magnitude
    k_parallel: Float[Array, ""] = k_magnitude * jnp.cos(theta_rad)

    # Split in-plane component into x and y based on azimuthal angle
    k_x: Float[Array, ""] = k_parallel * jnp.cos(phi_rad)
    k_y: Float[Array, ""] = k_parallel * jnp.sin(phi_rad)
    k_z: Float[Array, ""] = -k_magnitude * jnp.sin(theta_rad)

    k_in: Float[Array, "3"] = jnp.array([k_x, k_y, k_z])
    return k_in
