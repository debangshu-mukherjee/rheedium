"""Tests for shared quadrature helpers in rheedium.tools."""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.tools.quadrature import gauss_hermite_nodes_weights


class TestGaussHermiteNodesWeights(chex.TestCase):
    """Tests for Gauss-Hermite quadrature computation.

    :see: :func:`~rheedium.tools.gauss_hermite_nodes_weights`
    """

    def test_correct_count(self) -> None:
        r"""Returned arrays have the requested number of points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Returned arrays
        have the requested number of points.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_quadrature``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n: int
        for n in [3, 5, 7, 9]:
            nodes: Float[Array, "..."]
            weights: Float[Array, "..."]
            nodes, weights = gauss_hermite_nodes_weights(n)
            chex.assert_shape(nodes, (n,))
            chex.assert_shape(weights, (n,))

    def test_weights_positive(self) -> None:
        r"""All Gauss-Hermite weights are positive.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All Gauss-Hermite
        weights are positive.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_quadrature``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        weights: Float[Array, "..."]
        _, weights = gauss_hermite_nodes_weights(7)
        assert jnp.all(weights > 0.0)

    def test_nodes_symmetric(self) -> None:
        r"""Nodes are symmetric about zero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Nodes are
        symmetric about zero.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_quadrature``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        nodes: Float[Array, "..."]
        nodes, _ = gauss_hermite_nodes_weights(7)
        sorted_nodes: Float[Array, "..."] = jnp.sort(nodes)
        chex.assert_trees_all_close(
            sorted_nodes,
            -jnp.flip(sorted_nodes),
            atol=1e-12,
        )

    def test_weights_sum(self) -> None:
        r"""Weights sum to sqrt(pi).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Weights sum to
        sqrt(pi).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_quadrature``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        weights: Float[Array, "..."]
        _, weights = gauss_hermite_nodes_weights(7)
        chex.assert_trees_all_close(
            jnp.sum(weights),
            jnp.sqrt(jnp.pi),
            atol=1e-12,
        )
