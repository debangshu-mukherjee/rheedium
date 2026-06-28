"""Migration guards for the retired recon optimizer APIs.

Verifies that the optimistix/optax ``solve`` surface preserves the
deterministic linear reconstruction behavior that used to be covered by the
hand-rolled Gauss-Newton, Adam, and Adagrad tests.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    solve,
    weighted_mean_squared_error,
)
from rheedium.types import ReconProblem, ReconResult
from rheedium.types.custom_types import scalar_float


def _linear_forward_model(
    params: dict[str, scalar_float],
) -> Float[Array, "rows cols"]:
    """Map a simple two-parameter pytree to a 2-D detector image."""
    scale: scalar_float = params["scale"]
    offset: scalar_float = params["offset"]
    image: Float[Array, "rows cols"] = jnp.array(
        [
            [2.0 * scale + offset, scale - offset],
            [scale + 0.5 * offset, -scale + 2.0 * offset],
        ],
        dtype=jnp.float64,
    )
    return image


def _true_params() -> dict[str, scalar_float]:
    """Return reference parameters for reconstruction tests."""
    params: dict[str, scalar_float] = {
        "scale": jnp.float64(1.5),
        "offset": jnp.float64(-0.25),
    }
    return params


def _initial_params() -> dict[str, scalar_float]:
    """Return the initial guess for reconstruction tests."""
    params: dict[str, scalar_float] = {
        "scale": jnp.float64(0.0),
        "offset": jnp.float64(0.0),
    }
    return params


def _linear_problem() -> tuple[ReconProblem, dict[str, scalar_float]]:
    """Return a planted linear reconstruction problem."""
    true_params: dict[str, scalar_float] = _true_params()
    target: Float[Array, "rows cols"] = _linear_forward_model(true_params)
    problem: ReconProblem = ReconProblem(
        forward=_linear_forward_model,
        measured=target,
        loss_fn=weighted_mean_squared_error,
    )
    return problem, true_params


class TestReconOptimizerMigration(chex.TestCase):
    """Regression tests for replacing hand-rolled optimizers with solve.

    :see: :func:`~rheedium.recon.solve`
    """

    def test_lm_solve_replaces_gauss_newton_reconstruction(self) -> None:
        r"""LM solve should recover the former Gauss-Newton linear fixture.

        Extended Summary
        ----------------
        Verifies the K2 regression guard for the retired Gauss-Newton path:
        the general least-squares ``solve`` API recovers the same planted
        dictionary-parameter image model to tight tolerance.

        Notes
        -----
        This pins the replacement behavior in the same test module that used
        to cover the deleted hand-rolled optimizer.
        """
        problem: ReconProblem
        true_params: dict[str, scalar_float]
        problem, true_params = _linear_problem()

        result: ReconResult = solve(
            problem=problem,
            initial_latent=_initial_params(),
            max_steps=16,
            atol=1e-10,
            rtol=1e-10,
        )

        self.assertIsInstance(result, ReconResult)
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
        chex.assert_trees_all_close(result.loss, 0.0, atol=1e-12)

    def test_bfgs_solve_replaces_scalar_minimization_path(self) -> None:
        r"""BFGS solve should recover the planted scalar-loss fixture.

        Extended Summary
        ----------------
        Verifies that the scalar minimisation branch of the new solver surface
        covers the low-level objective-minimization use case formerly handled
        by bespoke optimizer loops.

        Notes
        -----
        The loss is the public image MSE helper, so this also protects the
        loss-function calling convention used by reconstruction wrappers.
        """
        problem: ReconProblem
        true_params: dict[str, scalar_float]
        problem, true_params = _linear_problem()

        result: ReconResult = solve(
            problem=problem,
            initial_latent=_initial_params(),
            mode="bfgs",
            max_steps=64,
            atol=1e-10,
            rtol=1e-10,
        )

        self.assertTrue(bool(result.converged))
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=1e-6,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=1e-6,
        )

    def test_adamw_solve_replaces_adaptive_gradient_smoke(self) -> None:
        r"""AdamW solve should substantially reduce the old adaptive loss.

        Extended Summary
        ----------------
        Verifies that the optax-backed first-order path keeps the practical
        behavior previously covered by Adam/Adagrad smoke tests: starting from
        the same poor initialization, it drives the image-matching loss down
        and approaches the planted parameters.

        Notes
        -----
        The tolerance is intentionally looser than the LM/BFGS paths because
        this is the stochastic/high-dimensional solver branch.
        """
        problem: ReconProblem
        true_params: dict[str, scalar_float]
        problem, true_params = _linear_problem()
        initial_loss: Float[Array, ""] = weighted_mean_squared_error(
            _linear_forward_model(_initial_params()),
            problem.measured,
        )

        result: ReconResult = solve(
            problem=problem,
            initial_latent=_initial_params(),
            mode="adamw",
            max_steps=500,
            learning_rate=0.08,
            atol=1e-10,
        )

        self.assertLess(float(result.loss), float(initial_loss) * 1e-3)
        chex.assert_trees_all_close(
            result.params["scale"],
            true_params["scale"],
            atol=1e-2,
        )
        chex.assert_trees_all_close(
            result.params["offset"],
            true_params["offset"],
            atol=1e-2,
        )

    def test_legacy_optimizer_symbols_are_not_exported(self) -> None:
        r"""Retired hand-rolled optimizer names should be absent.

        Extended Summary
        ----------------
        Verifies the zero-legacy half of K2: the old optimizer functions and
        result container are not public aliases for the new solver surface.

        Notes
        -----
        Absence is checked directly on ``rheedium.recon`` so accidental shims
        or compatibility exports fail loudly.
        """
        retired_names: tuple[str, ...] = (
            "ReconstructionResult",
            "adagrad_optimize",
            "adagrad_reconstruction",
            "adam_optimize",
            "adam_reconstruction",
            "gauss_newton_least_squares",
            "gauss_newton_reconstruction",
        )
        for retired_name in retired_names:
            self.assertFalse(hasattr(recon, retired_name), retired_name)
