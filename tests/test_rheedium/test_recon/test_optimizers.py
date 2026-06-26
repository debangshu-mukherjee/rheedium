"""Tests for recon/optimizers.py.

Verifies the first optimizer-based reconstruction routines against a
small linear forward model so convergence is deterministic and cheap to
evaluate.
"""

from typing import Any

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    ReconstructionResult,
    adagrad_reconstruction,
    adam_reconstruction,
    gauss_newton_least_squares,
    gauss_newton_reconstruction,
    weighted_mean_squared_error,
)
from rheedium.types.custom_types import scalar_float


def _linear_forward_model(
    params: dict[str, scalar_float],
) -> Float[Array, "rows cols"]:
    """Map a simple two-parameter pytree to a 2-D detector image."""
    scale: float = params["scale"]
    offset: float = params["offset"]
    return jnp.array(
        [
            [2.0 * scale + offset, scale - offset],
            [scale + 0.5 * offset, -scale + 2.0 * offset],
        ],
        dtype=jnp.float64,
    )


def _true_params() -> dict[str, scalar_float]:
    """Return reference parameters for reconstruction tests."""
    return {
        "scale": jnp.float64(1.5),
        "offset": jnp.float64(-0.25),
    }


def _initial_params() -> dict[str, scalar_float]:
    """Return the initial guess for reconstruction tests."""
    return {
        "scale": jnp.float64(0.0),
        "offset": jnp.float64(0.0),
    }


class TestGaussNewtonReconstruction(chex.TestCase):
    """Tests for Gauss-Newton-based reconstruction.

    :see: :func:`~rheedium.recon.gauss_newton_least_squares`
    :see: :func:`~rheedium.recon.gauss_newton_reconstruction`
    """

    def test_gauss_newton_reconstruction_recovers_linear_parameters(
        self,
    ) -> None:
        r"""Gauss-Newton should recover a linear image model in one step.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gauss-Newton
        should recover a linear image model in one step.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_optimizers``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        true_params: dict[str, scalar_float] = _true_params()
        target: Float[Array, "rows cols"] = _linear_forward_model(true_params)

        result: ReconstructionResult = gauss_newton_reconstruction(
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

    def test_gauss_newton_low_level_accepts_parameter_pytrees(self) -> None:
        r"""The low-level least-squares solver should work on dict pytrees.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The low-level
        least-squares solver should work on dict pytrees.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_optimizers``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        true_params: dict[str, scalar_float] = _true_params()
        target: Float[Array, "rows cols"] = _linear_forward_model(true_params)

        def residual_fn(
            params: dict[str, scalar_float],
        ) -> Float[Array, "rows cols"]:
            return _linear_forward_model(params) - target

        result: ReconstructionResult = gauss_newton_least_squares(
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
    """Tests for Adam- and Adagrad-based reconstruction.

    :see: :func:`~rheedium.recon.adagrad_reconstruction`
    :see: :func:`~rheedium.recon.adam_reconstruction`
    """

    def test_adam_reconstruction_reduces_loss(self) -> None:
        r"""Adam should substantially reduce the image-matching loss.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Adam should
        substantially reduce the image-matching loss.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_optimizers``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        true_params: Any = _true_params()
        target: Any = _linear_forward_model(true_params)
        initial_loss: Any = float(
            weighted_mean_squared_error(
                _linear_forward_model(_initial_params()),
                target,
            )
        )

        result: Float[Array, "..."] = adam_reconstruction(
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

    def test_adagrad_reconstruction_reduces_loss(self) -> None:
        r"""Adagrad should substantially reduce the image-matching loss.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Adagrad should
        substantially reduce the image-matching loss.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_optimizers``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        true_params: Any = _true_params()
        target: Any = _linear_forward_model(true_params)
        initial_loss: Any = float(
            weighted_mean_squared_error(
                _linear_forward_model(_initial_params()),
                target,
            )
        )

        result: Float[Array, "..."] = adagrad_reconstruction(
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

    def test_namespace_exports_optimizer_entry_points(self) -> None:
        r"""Optimizer APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Optimizer APIs
        should be re-exported from rheedium.recon.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_optimizers``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        self.assertIs(
            recon.gauss_newton_reconstruction,
            gauss_newton_reconstruction,
        )
        self.assertIs(recon.adam_reconstruction, adam_reconstruction)
        self.assertIs(recon.adagrad_reconstruction, adagrad_reconstruction)
