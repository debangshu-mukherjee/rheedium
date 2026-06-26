"""Tests for recon/losses.py.

Verifies the weighted residual and weighted mean-squared error helpers
used by reconstruction routines.
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from rheedium.recon import (
    checked_weighted_image_residual,
    checked_weighted_mean_squared_error,
    weighted_image_residual,
    weighted_mean_squared_error,
)
from rheedium.types.custom_types import scalar_float


class TestWeightedLosses(chex.TestCase):
    """Tests for weighted residual and loss builders.

    :see: :func:`~rheedium.recon.weighted_image_residual`
    :see: :func:`~rheedium.recon.weighted_mean_squared_error`
    """

    def test_weighted_image_residual_scales_by_sqrt_weights(self) -> None:
        r"""Residual weights should enter as square roots.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Residual weights
        should enter as square roots.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_losses``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
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
        r"""Weighted MSE should divide by the sum of retained weights.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Weighted MSE
        should divide by the sum of retained weights.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_losses``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
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


class TestCheckedWeightedLosses(chex.TestCase):
    """Tests for opt-in checkified reconstruction losses."""

    def test_checked_weighted_image_residual_valid(self) -> None:
        r"""Checked residual should allow finite outputs under JIT.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Checked residual
        should allow finite outputs under JIT.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_losses``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        simulated: Float[Array, "rows cols"] = jnp.array(
            [[3.0, 4.0], [5.0, 6.0]]
        )
        experimental: Float[Array, "rows cols"] = jnp.ones((2, 2))
        weight_map: Float[Array, "rows cols"] = jnp.ones((2, 2))

        err: Any
        residual: Float[Array, "rows cols"]
        err, residual = jax.jit(checked_weighted_image_residual)(
            simulated,
            experimental,
            weight_map,
        )
        err.throw()

        expected: Float[Array, "rows cols"] = simulated - experimental
        chex.assert_trees_all_close(residual, expected, atol=1e-12)

    def test_checked_weighted_mean_squared_error_rejects_nan(self) -> None:
        r"""Checked MSE should report NaN-producing inputs under JIT.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Checked MSE should
        report NaN-producing inputs under JIT.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_losses``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        simulated: Float[Array, "rows cols"] = jnp.array(
            [[jnp.nan, 2.0], [3.0, 4.0]]
        )
        experimental: Float[Array, "rows cols"] = jnp.ones((2, 2))
        weight_map: Float[Array, "rows cols"] = jnp.ones((2, 2))

        err: Any
        loss: scalar_float
        err, loss = jax.jit(checked_weighted_mean_squared_error)(
            simulated,
            experimental,
            weight_map,
        )

        del loss
        with pytest.raises(Exception, match="nan"):
            err.throw()
