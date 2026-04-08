"""Tests for recon/losses.py and recon/optimizers.py.

Verifies the first optimizer-based reconstruction routines against a
small linear forward model so convergence is deterministic and cheap to
evaluate.
"""

import chex
import jax.numpy as jnp

import rheedium.recon as recon
from rheedium.recon import (
    ReconstructionResult,
    adagrad_reconstruction,
    adam_reconstruction,
    gauss_newton_least_squares,
    gauss_newton_reconstruction,
    weighted_image_residual,
    weighted_mean_squared_error,
)


def _linear_forward_model(
    params: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Map a simple two-parameter pytree to a 2-D detector image."""
    scale = params["scale"]
    offset = params["offset"]
    return jnp.array(
        [
            [2.0 * scale + offset, scale - offset],
            [scale + 0.5 * offset, -scale + 2.0 * offset],
        ],
        dtype=jnp.float64,
    )


def _true_params() -> dict[str, jnp.ndarray]:
    """Reference parameters for reconstruction tests."""
    return {
        "scale": jnp.float64(1.5),
        "offset": jnp.float64(-0.25),
    }


def _initial_params() -> dict[str, jnp.ndarray]:
    """Initial guess for reconstruction tests."""
    return {
        "scale": jnp.float64(0.0),
        "offset": jnp.float64(0.0),
    }


class TestWeightedLosses(chex.TestCase):
    """Tests for weighted residual and loss builders."""

    def test_weighted_image_residual_scales_by_sqrt_weights(self):
        """Residual weights should enter as square roots."""
        simulated = jnp.array([[3.0, 4.0], [5.0, 6.0]])
        experimental = jnp.ones((2, 2))
        weight_map = jnp.array([[1.0, 0.0], [4.0, 0.25]])

        residual = weighted_image_residual(
            simulated_image=simulated,
            experimental_image=experimental,
            weight_map=weight_map,
        )

        expected = jnp.array([[2.0, 0.0], [8.0, 2.5]])
        chex.assert_trees_all_close(residual, expected, atol=1e-12)

    def test_weighted_mean_squared_error_normalizes_by_weight_sum(self):
        """Weighted MSE should divide by the sum of retained weights."""
        simulated = jnp.array([[2.0, 3.0], [4.0, 5.0]])
        experimental = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        weight_map = jnp.array([[1.0, 1.0], [0.0, 0.0]])

        loss = weighted_mean_squared_error(
            simulated_image=simulated,
            experimental_image=experimental,
            weight_map=weight_map,
        )

        chex.assert_trees_all_close(loss, 2.5, atol=1e-12)


class TestGaussNewtonReconstruction(chex.TestCase):
    """Tests for Gauss-Newton-based reconstruction."""

    def test_gauss_newton_reconstruction_recovers_linear_parameters(self):
        """Gauss-Newton should recover a linear image model in one step."""
        true_params = _true_params()
        target = _linear_forward_model(true_params)

        result = gauss_newton_reconstruction(
            initial_params=_initial_params(),
            forward_model=_linear_forward_model,
            experimental_image=target,
            damping=jnp.float64(1e-12),
            max_iterations=5,
            tolerance=jnp.float64(1e-10),
        )

        self.assertIsInstance(result, ReconstructionResult)
        self.assertTrue(bool(result.converged))
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=1e-8,
        )
        chex.assert_tree_all_finite(result.objective_history)
        self.assertLessEqual(int(result.iterations), 2)

    def test_gauss_newton_low_level_accepts_parameter_pytrees(self):
        """The low-level least-squares solver should work on dict pytrees."""
        true_params = _true_params()
        target = _linear_forward_model(true_params)

        def residual_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            return _linear_forward_model(params) - target

        result = gauss_newton_least_squares(
            initial_params=_initial_params(),
            residual_fn=residual_fn,
            damping=jnp.float64(1e-12),
            max_iterations=5,
            tolerance=jnp.float64(1e-10),
        )

        self.assertTrue(bool(result.converged))
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=1e-8,
        )


class TestAdaptiveGradientReconstruction(chex.TestCase):
    """Tests for Adam- and Adagrad-based reconstruction."""

    def test_adam_reconstruction_reduces_loss(self):
        """Adam should substantially reduce the image-matching loss."""
        true_params = _true_params()
        target = _linear_forward_model(true_params)
        initial_loss = float(
            weighted_mean_squared_error(
                _linear_forward_model(_initial_params()),
                target,
            )
        )

        result = adam_reconstruction(
            initial_params=_initial_params(),
            forward_model=_linear_forward_model,
            experimental_image=target,
            learning_rate=jnp.float64(0.1),
            max_iterations=300,
            tolerance=jnp.float64(1e-10),
        )

        self.assertLess(
            float(result.objective_history[-1]),
            initial_loss * 1e-3,
        )
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=5e-3,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=5e-3,
        )

    def test_adagrad_reconstruction_reduces_loss(self):
        """Adagrad should substantially reduce the image-matching loss."""
        true_params = _true_params()
        target = _linear_forward_model(true_params)
        initial_loss = float(
            weighted_mean_squared_error(
                _linear_forward_model(_initial_params()),
                target,
            )
        )

        result = adagrad_reconstruction(
            initial_params=_initial_params(),
            forward_model=_linear_forward_model,
            experimental_image=target,
            learning_rate=jnp.float64(0.75),
            max_iterations=600,
            tolerance=jnp.float64(1e-10),
        )

        self.assertLess(
            float(result.objective_history[-1]),
            initial_loss * 1e-2,
        )
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=2e-2,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=2e-2,
        )


class TestReconNamespace(chex.TestCase):
    """Tests for public recon exports."""

    def test_namespace_exports_optimizer_entry_points(self):
        """Optimizer APIs should be re-exported from rheedium.recon."""
        self.assertIs(
            recon.gauss_newton_reconstruction,
            gauss_newton_reconstruction,
        )
        self.assertIs(recon.adam_reconstruction, adam_reconstruction)
        self.assertIs(recon.adagrad_reconstruction, adagrad_reconstruction)
        self.assertIs(recon.weighted_image_residual, weighted_image_residual)
