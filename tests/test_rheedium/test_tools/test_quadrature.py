"""Tests for shared quadrature helpers in rheedium.tools."""

import chex
import jax.numpy as jnp

from rheedium.tools.quadrature import gauss_hermite_nodes_weights


class TestGaussHermiteNodesWeights(chex.TestCase):
    """Tests for Gauss-Hermite quadrature computation."""

    def test_correct_count(self):
        """Returned arrays have the requested number of points."""
        for n in [3, 5, 7, 9]:
            nodes, weights = gauss_hermite_nodes_weights(n)
            chex.assert_shape(nodes, (n,))
            chex.assert_shape(weights, (n,))

    def test_weights_positive(self):
        """All Gauss-Hermite weights are positive."""
        _, weights = gauss_hermite_nodes_weights(7)
        assert jnp.all(weights > 0.0)

    def test_nodes_symmetric(self):
        """Nodes are symmetric about zero."""
        nodes, _ = gauss_hermite_nodes_weights(7)
        sorted_nodes = jnp.sort(nodes)
        chex.assert_trees_all_close(
            sorted_nodes,
            -jnp.flip(sorted_nodes),
            atol=1e-12,
        )

    def test_weights_sum(self):
        """Weights sum to sqrt(pi)."""
        _, weights = gauss_hermite_nodes_weights(7)
        chex.assert_trees_all_close(
            jnp.sum(weights),
            jnp.sqrt(jnp.pi),
            atol=1e-12,
        )
