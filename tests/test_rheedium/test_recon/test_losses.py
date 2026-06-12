"""Tests for recon/losses.py.

Verifies the weighted residual and weighted mean-squared error helpers
used by reconstruction routines.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.recon import (
    weighted_image_residual,
    weighted_mean_squared_error,
)
from rheedium.types.custom_types import scalar_float


class TestWeightedLosses(chex.TestCase):
    """Tests for weighted residual and loss builders."""

    def test_weighted_image_residual_scales_by_sqrt_weights(self) -> None:
        """Residual weights should enter as square roots."""
        simulated: Float[Array, "rows cols"] = jnp.array(
            [[3.0, 4.0], [5.0, 6.0]]
        )
        experimental: Float[Array, "rows cols"] = jnp.ones((2, 2))
        weight_map: Float[Array, "rows cols"] = jnp.array(
            [[1.0, 0.0], [4.0, 0.25]]
        )

        residual: Float[Array, "rows cols"] = weighted_image_residual(
            simulated_image=simulated,
            experimental_image=experimental,
            weight_map=weight_map,
        )

        expected: Float[Array, "rows cols"] = jnp.array(
            [[2.0, 0.0], [8.0, 2.5]]
        )
        chex.assert_trees_all_close(residual, expected, atol=1e-12)

    def test_weighted_mean_squared_error_normalizes_by_weight_sum(
        self,
    ) -> None:
        """Weighted MSE should divide by the sum of retained weights."""
        simulated: Float[Array, "rows cols"] = jnp.array(
            [[2.0, 3.0], [4.0, 5.0]]
        )
        experimental: Float[Array, "rows cols"] = jnp.array(
            [[1.0, 1.0], [1.0, 1.0]]
        )
        weight_map: Float[Array, "rows cols"] = jnp.array(
            [[1.0, 1.0], [0.0, 0.0]]
        )

        loss: scalar_float = weighted_mean_squared_error(
            simulated_image=simulated,
            experimental_image=experimental,
            weight_map=weight_map,
        )

        chex.assert_trees_all_close(loss, 2.5, atol=1e-12)
