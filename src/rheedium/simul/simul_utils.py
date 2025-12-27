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

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def wavelength_ang(
    voltage_kv: Union[scalar_num, Num[Array, "..."]],
) -> Float[Array, "..."]:
    """Calculate the relativistic electron wavelength in angstroms.

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
    Uses relativistic corrections for accurate wavelength at high energies.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> lam = rh.simul.wavelength_ang(jnp.asarray(20.0))  # 20 keV
    >>> print(f"λ = {lam:.4f} Å")
    λ = 0.0859 Å
    """
    rest_mass_energy_kev: Float[Array, "..."] = 511.0
    # Convert kV to V for the formula
    voltage_v: Float[Array, "..."] = voltage_kv * 1000.0
    corrected_voltage: Float[Array, "..."] = voltage_v * (
        1.0 + voltage_v / (2.0 * rest_mass_energy_kev * 1000.0)
    )
    h_over_2me: Float[Array, "..."] = 12.26
    wavelength: Float[Array, "..."] = h_over_2me / jnp.sqrt(corrected_voltage)
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
