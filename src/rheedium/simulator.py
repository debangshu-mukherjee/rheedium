from functools import partial
from typing import Any, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
# from typeguard import typechecked as typechecker
from jax import lax
from jaxtyping import Array, Complex, Float, Int, jaxtyped

jax.config.update("jax_enable_x64", True)

@jaxtyped(typechecker=typechecker)
def wavelength_ang(
    voltage_kV: int | float | Float[Array, "*"]
    ) -> Float[Array, "*"]:
    """
    Description
    -----------
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `voltage_kV` (int | float | Float[Array, "*"]):
        The microscope accelerating voltage in kilo
        electronVolts

    Returns
    -------
    - `in_angstroms (Float[Array, "*"]):
        The electron wavelength in angstroms

    Flow
    ----
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: Float[Array, "*"] = jnp.float64(9.109383e-31)  # mass of an electron
    e: Float[Array, "*"] = jnp.float64(1.602177e-19)  # charge of an electron
    c: Float[Array, "*"] = jnp.float64(299792458.0)  # speed of light
    h: Float[Array, "*"] = jnp.float64(6.62607e-34)  # Planck's constant

    voltage: Float[Array, "*"] = jnp.multiply(
        jnp.float64(voltage_kV), jnp.float64(1000)
    )
    eV = jnp.multiply(e, voltage)
    numerator: Float[Array, "*"] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, "*"] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, "*"] = jnp.sqrt(
        numerator / denominator
    )  # in meters
    in_angstroms: Float[Array, "*"] = 1e10 * wavelength_meters  # in angstroms
    return in_angstroms