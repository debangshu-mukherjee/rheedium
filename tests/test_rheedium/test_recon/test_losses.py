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
    affine_intensity_marginalization,
    affine_marginalized_residual,
    checked_weighted_image_residual,
    checked_weighted_mean_squared_error,
    entropy_prior,
    huber_image_loss,
    log_intensity_loss,
    normalized_cross_correlation_loss,
    smoothness_prior,
    sparsity_prior,
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


class TestDifferentiableLosses(chex.TestCase):
    """Tests for differentiable reconstruction loss extensions.

    :see: :func:`~rheedium.recon.log_intensity_loss`
    :see: :func:`~rheedium.recon.affine_intensity_marginalization`
    :see: :func:`~rheedium.recon.entropy_prior`
    """

    def test_log_huber_and_ncc_losses_have_finite_gradients(self) -> None:
        r"""Extended image losses should be differentiable in scale.

        Extended Summary
        ----------------
        Verifies that log-intensity, Huber, and NCC image losses return finite
        values and allow ``jax.grad`` to flow through a scalar forward
        parameter.

        Notes
        -----
        It builds positive synthetic images, combines the losses into one
        scalar objective, and checks the gradient with chex finite assertions.
        """
        base: Float[Array, "rows cols"] = jnp.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=jnp.float64,
        )
        target: Float[Array, "rows cols"] = 1.5 * base + 0.25

        def objective(scale: scalar_float) -> scalar_float:
            simulated: Float[Array, "rows cols"] = scale * base
            loss: scalar_float = (
                log_intensity_loss(simulated, target)
                + huber_image_loss(simulated, target, delta=0.5)
                + normalized_cross_correlation_loss(simulated, target)
            )
            return loss

        value: scalar_float = objective(jnp.asarray(1.0, dtype=jnp.float64))
        gradient: scalar_float = jax.grad(objective)(
            jnp.asarray(1.0, dtype=jnp.float64)
        )

        chex.assert_tree_all_finite(value)
        chex.assert_tree_all_finite(gradient)

    def test_affine_marginalization_removes_scale_and_background(self) -> None:
        r"""Analytic calibration should remove affine intensity mismatch.

        Extended Summary
        ----------------
        Verifies that the closed-form scale/background solve recovers a planted
        calibration and drives the marginalized residual to zero.

        Notes
        -----
        It constructs an experimental image from an exact affine transform of a
        simulated image and checks both recovered nuisance parameters and the
        final residual.
        """
        simulated: Float[Array, "rows cols"] = jnp.array(
            [[0.5, 1.0], [2.0, 4.0]],
            dtype=jnp.float64,
        )
        experimental: Float[Array, "rows cols"] = 2.5 * simulated + 0.75

        scale: scalar_float
        background: scalar_float
        scale, background = affine_intensity_marginalization(
            simulated,
            experimental,
        )
        residual: Float[Array, "rows cols"] = affine_marginalized_residual(
            simulated,
            experimental,
        )

        chex.assert_trees_all_close(scale, 2.5, atol=1e-10)
        chex.assert_trees_all_close(background, 0.75, atol=1e-10)
        chex.assert_trees_all_close(
            residual, jnp.zeros_like(residual), atol=1e-10
        )

    def test_priors_prefer_entropy_smoothness_and_sparsity(self) -> None:
        r"""Regularizers should rank simple reference distributions sensibly.

        Extended Summary
        ----------------
        Verifies that the entropy prior favors diffuse weights, smoothness
        favors slowly varying profiles, and sparsity favors zero vectors.

        Notes
        -----
        It compares paired synthetic vectors chosen so each prior has an
        unambiguous ordering.
        """
        uniform: Float[Array, "weights"] = jnp.array(
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            dtype=jnp.float64,
        )
        peaked: Float[Array, "weights"] = jnp.array(
            [0.98, 0.01, 0.01],
            dtype=jnp.float64,
        )
        smooth: Float[Array, "weights"] = jnp.array(
            [0.2, 0.25, 0.3],
            dtype=jnp.float64,
        )
        rough: Float[Array, "weights"] = jnp.array(
            [0.2, 0.9, 0.1],
            dtype=jnp.float64,
        )

        self.assertLess(
            float(entropy_prior(uniform)), float(entropy_prior(peaked))
        )
        self.assertLess(
            float(smoothness_prior(smooth)),
            float(smoothness_prior(rough)),
        )
        self.assertLess(
            float(sparsity_prior(jnp.zeros(3, dtype=jnp.float64))),
            float(sparsity_prior(peaked)),
        )
