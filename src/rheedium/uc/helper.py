import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, Num, jaxtyped

from rheedium import uc

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def wavelength_ang(voltage_kV: Num[Array, ""]) -> Float[Array, ""]:
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
    - `voltage_kV` (Num[Array, ""]):
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
    m: Float[Array, ""] = jnp.float64(9.109383e-31)
    e: Float[Array, ""] = jnp.float64(1.602177e-19)
    c: Float[Array, ""] = jnp.float64(299792458.0)
    h: Float[Array, ""] = jnp.float64(6.62607e-34)
    voltage: Float[Array, ""] = jnp.multiply(jnp.float64(voltage_kV), jnp.float64(1000))
    eV = jnp.multiply(e, voltage)
    numerator: Float[Array, ""] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, ""] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, ""] = jnp.sqrt(numerator / denominator)
    in_angstroms: Float[Array, ""] = 1e10 * wavelength_meters
    return in_angstroms


@jaxtyped(typechecker=beartype)
def angle_in_degrees(u: Float[Array, "c"], v: Float[Array, "c"]) -> Float[Array, ""]:
    """
    Description
    -----------
    Calculate the angle in degrees between two vectors u and v.

    Parameters
    ----------
    - `u` (Float[Array, "c"]):
        The first vector.
    - `v` (Float[Array, "c"]):
        The second vector.

    Returns
    -------
    - `cos_val_degrees` (Float[Array, ""]):
        The angle in degrees between u and v.

    Flow
    ----
    - Calculate the dot product of u and v
    - Calculate the cosine of the angle
    - Convert the cosine to degrees
    """
    dot_uv: Float[Array, ""] = jnp.dot(u, v)
    cos_val: Float[Array, ""] = dot_uv / (
        jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-32
    )
    cos_val_clamped: Float[Array, ""] = jnp.clip(cos_val, -1.0, 1.0)
    cos_val_degrees: Float[Array, ""] = (jnp.arccos(cos_val_clamped) * 180.0) / jnp.pi
    return cos_val_degrees


@jaxtyped(typechecker=beartype)
def compute_lengths_angles(
    cell_vectors: Float[Array, "3 3"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Description
    -----------
    Given a (3, 3) array of lattice vectors (a_vec, b_vec, c_vec),
    compute (a, b, c) in Å and (alpha, beta, gamma) in degrees.

    Parameters
    ----------
    - `cell_vectors` (Float[Array, "3 3"]):
        The 3x3 array of lattice vectors in Cartesian coordinates.

    Returns
    -------
    - `lengths` (Float[Array, "3"]):
        The lengths of the lattice vectors (a, b, c) in Å.
    - `angles` (Float[Array, "3"]):
        The angles between the lattice vectors (alpha, beta, gamma) in degrees.
    """
    a_vec: Float[Array, "3"] = cell_vectors[0]
    b_vec: Float[Array, "3"] = cell_vectors[1]
    c_vec: Float[Array, "3"] = cell_vectors[2]
    a_len: Float[Array, ""] = jnp.linalg.norm(a_vec)
    b_len: Float[Array, ""] = jnp.linalg.norm(b_vec)
    c_len: Float[Array, ""] = jnp.linalg.norm(c_vec)
    alpha: Float[Array, ""] = uc.angle_in_degrees(b_vec, c_vec)
    beta: Float[Array, ""] = uc.angle_in_degrees(a_vec, c_vec)
    gamma: Float[Array, ""] = uc.angle_in_degrees(a_vec, b_vec)
    lengths: Float[Array, "3"] = jnp.array([a_len, b_len, c_len])
    angles: Float[Array, "3"] = jnp.array([alpha, beta, gamma])
    return (lengths, angles)
