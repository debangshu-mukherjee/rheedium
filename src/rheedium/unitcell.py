"""
=========================================================
Unit Cell Operations (:mod:`rheedium.unitcell`)
=========================================================

This package contains the modules for the calculations of
unit cell operations and conversion to Ewald sphere.
"""

from typing import Tuple, SupportsFloat

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax
from jaxtyping import Array, Float, Num, jaxtyped

import rheedium

jax.config.update("jax_enable_x64", True)
num_type = type[SupportsFloat]


@jaxtyped(typechecker=typechecker)
def wavelength_ang(voltage_kV: num_type | Float[Array, ""]) -> Float[Array, ""]:
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
    - `voltage_kV` (num_type | Float[Array, ""]):
        The microscope accelerating voltage in kilo
        electronVolts

    Returns
    -------
    - `in_angstroms (Float[Array, ""]):
        The electron wavelength in angstroms

    Flow
    ----
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: Float[Array, ""] = jnp.float64(9.109383e-31)  # mass of an electron
    e: Float[Array, ""] = jnp.float64(1.602177e-19)  # charge of an electron
    c: Float[Array, ""] = jnp.float64(299792458.0)  # speed of light
    h: Float[Array, ""] = jnp.float64(6.62607e-34)  # Planck's constant

    voltage: Float[Array, ""] = jnp.multiply(jnp.float64(voltage_kV), jnp.float64(1000))
    eV = jnp.multiply(e, voltage)
    numerator: Float[Array, ""] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, ""] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, ""] = jnp.sqrt(numerator / denominator)  # in meters
    in_angstroms: Float[Array, ""] = 1e10 * wavelength_meters  # in angstroms
    return in_angstroms


@jaxtyped(typechecker=typechecker)
def reciprocal_unitcell(unitcell: Num[Array, "3 3"]) -> Float[Array, "3 3"]:
    """
    Description
    -----------
    Calculate the reciprocal cell of a unit cell.

    Parameters
    ----------
    - `unitcell` (Num[Array, "3 3"]):
        The unit cell.

    Returns
    -------
    - `reciprocal_cell` (Float[Array, "3 3"]):
        The reciprocal cell.

    Flow
    ----
    - Calculate the reciprocal cell
    - Check if the matrix is well-conditioned
    - If not, replace the values with NaN
    """
    # Optional: Check that matrix is well-conditioned
    condition_number = jnp.linalg.cond(unitcell)
    is_well_conditioned = condition_number < 1e10  # threshold can be adjusted

    # Calculate reciprocal cell
    reciprocal_cell_uncond: Float[Array, "3 3"] = (
        2 * jnp.pi * jnp.transpose(jnp.linalg.inv(unitcell))
    )

    reciprocal_cell: Float[Array, "3 3"] = jnp.where(
        is_well_conditioned,
        reciprocal_cell_uncond,
        jnp.full_like(reciprocal_cell_uncond, 0.0),
    )
    return reciprocal_cell


@jaxtyped(typechecker=typechecker)
def reciprocal_uc_angles(
    unitcell_abc: Num[Array, "3"],
    unitcell_angles: Num[Array, "3"],
    in_degrees: bool | None = True,
    out_degrees: bool | None = False,
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Description
    -----------
    Calculate the reciprocal unit cell when the sides (a, b, c) and
    the angles (alpha, beta, gamma) are given.

    Parameters
    ----------
    - `unitcell_abc` (Num[Array, "3"]):
        The sides of the unit cell.
    - `unitcell_angles` (Num[Array, "3"]):
        The angles of the unit cell.
    - `in_degrees` (bool | None):
        Whether the angles are in degrees or radians.
        If None, it will be assumed that the angles are
        in degrees.
        Default is True.
    - `out_degrees` (bool | None):
        Whether the angles should be in degrees or radians.
        If None, it will be assumed that the angles should
        be in radians.
        Default is False.

    Returns
    -------
    - `reciprocal_abc` (Float[Array, "3"]):
        The sides of the reciprocal unit cell.
    - `reciprocal_angles` (Float[Array, "3"]):
        The angles of the reciprocal unit cell.

    Flow
    ----
    - Convert the angles to radians if they are in degrees
    - Calculate the cos and sin values of the angles
    - Calculate the volume factor of the unit cell
    - Calculate the unit cell volume
    - Calculate the reciprocal lattice parameters
    - Calculate the reciprocal angles
    - Convert the angles to degrees if they are in radians
    """
    # Convert to radians if the angles are in degrees
    if in_degrees:
        unitcell_angles = jnp.radians(unitcell_angles)

    # Calculate cos and sin values of the angles
    cos_angles: Float[Array, "3"] = jnp.cos(unitcell_angles)
    sin_angles: Float[Array, "3"] = jnp.sin(unitcell_angles)

    # Calculate the volume factor of the unit cell
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )

    # Calculate unit cell volume
    volume: Float[Array, ""] = jnp.prod(unitcell_abc) * volume_factor

    # Calculate reciprocal lattice parameters
    reciprocal_abc: Float[Array, "3"] = (
        jnp.array(
            [
                unitcell_abc[1] * unitcell_abc[2] * sin_angles[0],
                unitcell_abc[2] * unitcell_abc[0] * sin_angles[1],
                unitcell_abc[0] * unitcell_abc[1] * sin_angles[2],
            ]
        )
        / volume
    )

    # Calculate reciprocal angles
    reciprocal_angles = jnp.arccos(
        (cos_angles[:, None] * cos_angles[None, :] - cos_angles[None, :])
        / (sin_angles[:, None] * sin_angles[None, :])
    )
    reciprocal_angles: Float[Array, "3"] = jnp.array(
        [reciprocal_angles[1, 2], reciprocal_angles[2, 0], reciprocal_angles[0, 1]]
    )

    if out_degrees:
        reciprocal_angles = jnp.degrees(reciprocal_angles)

    return (reciprocal_abc, reciprocal_angles)


@jaxtyped(typechecker=typechecker)
def get_unit_cell_matrix(
    unitcell_abc: Num[Array, "3"],
    unitcell_angles: Num[Array, "3"],
    in_degrees: bool | None = True,
) -> Float[Array, "3 3"]:
    """
    Description
    -----------
    Calculate the transformation matrix for a unit cell using JAX.

    Parameters
    ----------
    - `unitcell_abc` (Num[Array, "3"]):
        Length of the unit cell edges (a, b, c) in Angstroms.
    - `unitcell_angles` (Num[Array, "3"]):
        Angles between the edges (alpha, beta, gamma) in degrees or radians.
    - `in_degrees` (bool | None):
        Whether the angles are in degrees or radians.
        Default is True.

    Returns
    -------
    - `matrix` (Float[Array, "3 3"]):
        3x3 transformation matrix
    """
    # Convert to radians if needed
    angles_rad: Num[Array, "3"]
    if in_degrees:
        angles_rad = jnp.radians(unitcell_angles)
    else:
        angles_rad = unitcell_angles

    # Calculate trigonometric values
    cos_angles: Float[Array, "3"] = jnp.cos(angles_rad)
    sin_angles: Float[Array, "3"] = jnp.sin(angles_rad)

    # Calculate volume factor
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )

    # Create the transformation matrix
    matrix: Float[Array, "3 3"] = jnp.zeros(shape=(3, 3), dtype=jnp.float64)

    # Update matrix elements
    matrix = matrix.at[0, 0].set(unitcell_abc[0])
    matrix = matrix.at[0, 1].set(unitcell_abc[1] * cos_angles[2])
    matrix = matrix.at[0, 2].set(unitcell_abc[2] * cos_angles[1])
    matrix = matrix.at[1, 1].set(unitcell_abc[1] * sin_angles[2])
    matrix = matrix.at[1, 2].set(
        unitcell_abc[2]
        * (cos_angles[0] - cos_angles[1] * cos_angles[2])
        / sin_angles[2]
    )
    matrix = matrix.at[2, 2].set(unitcell_abc[2] * volume_factor / sin_angles[2])

    return matrix
