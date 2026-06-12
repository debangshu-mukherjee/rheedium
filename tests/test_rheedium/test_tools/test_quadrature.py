"""Tests for shared quadrature helpers in rheedium.tools."""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.tools.quadrature import gauss_hermite_nodes_weights


class TestGaussHermiteNodesWeights(chex.TestCase):
    """Tests for Gauss-Hermite quadrature computation."""

    def test_correct_count(self) -> None:
        """Returned arrays have the requested number of points."""
        n: int
        for n in [3, 5, 7, 9]:
            nodes: Float[Array, "..."]
            weights: Float[Array, "..."]
            nodes, weights = gauss_hermite_nodes_weights(n)
            chex.assert_shape(nodes, (n,))
            chex.assert_shape(weights, (n,))

    def test_weights_positive(self) -> None:
        """All Gauss-Hermite weights are positive."""
        weights: Float[Array, "..."]
        _, weights = gauss_hermite_nodes_weights(7)
        assert jnp.all(weights > 0.0)

    def test_nodes_symmetric(self) -> None:
        """Nodes are symmetric about zero."""
        nodes: Float[Array, "..."]
        nodes, _ = gauss_hermite_nodes_weights(7)
        sorted_nodes: Float[Array, "..."] = jnp.sort(nodes)
        chex.assert_trees_all_close(
            sorted_nodes,
            -jnp.flip(sorted_nodes),
            atol=1e-12,
        )

    def test_weights_sum(self) -> None:
        """Weights sum to sqrt(pi)."""
        weights: Float[Array, "..."]
        _, weights = gauss_hermite_nodes_weights(7)
        chex.assert_trees_all_close(
            jnp.sum(weights),
            jnp.sqrt(jnp.pi),
            atol=1e-12,
        )
