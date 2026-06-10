"""Lattice-vector conversion helpers for crystal parsers.

Extended Summary
----------------
This module contains small lattice-geometry helpers shared by multiple
structure parsers. Keeping them separate avoids import cycles between the
format-specific parsers and the higher-level crystal loader.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def lattice_to_cell_params(
    lattice: Float[Array, "3 3"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    r"""Convert lattice vectors to crystallographic cell parameters.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Lattice vectors as rows: ``[a, b, c]`` in Cartesian coordinates.

    Returns
    -------
    cell_lengths : Float[Array, "3"]
        Unit-cell lengths ``[a, b, c]`` in Angstroms.
    cell_angles : Float[Array, "3"]
        Unit-cell angles ``[alpha, beta, gamma]`` in degrees.
    """
    a_vec: Float[Array, "3"] = lattice[0]
    b_vec: Float[Array, "3"] = lattice[1]
    c_vec: Float[Array, "3"] = lattice[2]

    a: Float[Array, ""] = jnp.linalg.norm(a_vec)
    b: Float[Array, ""] = jnp.linalg.norm(b_vec)
    c: Float[Array, ""] = jnp.linalg.norm(c_vec)
    cell_lengths: Float[Array, "3"] = jnp.array([a, b, c])

    cos_alpha: Float[Array, ""] = jnp.dot(b_vec, c_vec) / (b * c)
    cos_beta: Float[Array, ""] = jnp.dot(a_vec, c_vec) / (a * c)
    cos_gamma: Float[Array, ""] = jnp.dot(a_vec, b_vec) / (a * b)

    cos_alpha = jnp.clip(cos_alpha, min=-1.0, max=1.0)
    cos_beta = jnp.clip(cos_beta, min=-1.0, max=1.0)
    cos_gamma = jnp.clip(cos_gamma, min=-1.0, max=1.0)

    alpha: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_alpha))
    beta: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_beta))
    gamma: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_gamma))
    cell_angles: Float[Array, "3"] = jnp.array([alpha, beta, gamma])

    return cell_lengths, cell_angles


__all__: list[str] = ["lattice_to_cell_params"]
