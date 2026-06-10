"""Quadrature helpers shared across the rheedium package."""

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812


@jaxtyped(typechecker=beartype)
def gauss_hermite_nodes_weights(
    n_points: int,
) -> Tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Compute Gauss-Hermite quadrature nodes and weights."""
    np_nodes: Float[NDArray, "N"]
    np_weights: Float[NDArray, "N"]
    np_nodes, np_weights = np.polynomial.hermite.hermgauss(n_points)
    nodes: Float[Array, " N"] = jnp.asarray(np_nodes, dtype=jnp.float64)
    weights: Float[Array, " N"] = jnp.asarray(np_weights, dtype=jnp.float64)
    return nodes, weights


__all__: list[str] = [
    "gauss_hermite_nodes_weights",
]
