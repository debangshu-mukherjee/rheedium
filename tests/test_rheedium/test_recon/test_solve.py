"""Tests for recon/solve.py.

Verifies the general optimistix/optax reconstruction surface on lightweight
synthetic inverse problems with deterministic linear forward models.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    ReconProblem,
    ReconResult,
    multistart,
    reconstruct_incoherent_weights,
    solve,
)

_LINEAR_MATRIX: Float[Array, "pixels params"] = jnp.array(
    [[2.0, 1.0], [1.0, -1.0], [0.5, 2.0]],
    dtype=jnp.float64,
)


def _linear_forward(params: Float[Array, "params"]) -> Float[Array, "pixels"]:
    """Map a two-parameter vector to synthetic detector pixels."""
    pixels: Float[Array, "pixels"] = _LINEAR_MATRIX @ params
    return pixels


def _linear_problem() -> tuple[ReconProblem, Float[Array, "params"]]:
    """Return a well-conditioned synthetic reconstruction problem."""
    true_params: Float[Array, "params"] = jnp.array(
        [1.25, -0.5],
        dtype=jnp.float64,
    )
    measured: Float[Array, "pixels"] = _linear_forward(true_params)
    problem: ReconProblem = ReconProblem(
        forward=_linear_forward,
        measured=measured,
    )
    return problem, true_params


class TestReconSolve(chex.TestCase):
    """Tests for the general reconstruction solver.

    :see: :class:`~rheedium.recon.ReconProblem`
    :see: :func:`~rheedium.recon.solve`
    """

    def test_least_squares_solve_recovers_linear_parameters(self) -> None:
        r"""Least-squares solve should recover planted linear parameters.

        Extended Summary
        ----------------
        Verifies that the default Levenberg-Marquardt path solves a synthetic
        overdetermined linear inverse problem to tight tolerance.

        Notes
        -----
        It initializes from zero, solves against exact synthetic data, and
        checks final parameters, loss, convergence, and finite residuals.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()

        result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.zeros(2, dtype=jnp.float64),
            max_steps=32,
            rtol=1e-10,
            atol=1e-10,
        )

        self.assertIsInstance(result, ReconResult)
        self.assertTrue(bool(result.converged))
        chex.assert_trees_all_close(result.params, true_params, atol=1e-8)
        chex.assert_trees_all_close(result.loss, 0.0, atol=1e-12)
        chex.assert_tree_all_finite(result.residual)

    def test_optax_adamw_solve_reduces_scalar_loss(self) -> None:
        r"""Optax-backed solve should reduce the synthetic image loss.

        Extended Summary
        ----------------
        Verifies that the first-order path wraps an optax gradient
        transformation through the same public solve surface.

        Notes
        -----
        It compares the final AdamW loss against the initial loss rather than
        requiring exact convergence from a fixed small iteration budget.
        """
        problem: ReconProblem
        _true_params: Float[Array, "params"]
        problem, _true_params = _linear_problem()
        initial_latent: Float[Array, "params"] = jnp.array(
            [-2.0, 2.0],
            dtype=jnp.float64,
        )
        initial_residual: Float[Array, "pixels"] = (
            problem.residual_from_latent(initial_latent)
        )
        initial_loss: Float[Array, ""] = jnp.mean(initial_residual**2)

        result: ReconResult = solve(
            problem=problem,
            initial_latent=initial_latent,
            mode="adamw",
            max_steps=300,
            learning_rate=0.05,
            atol=1e-8,
        )

        self.assertLess(float(result.loss), float(initial_loss) * 1e-2)
        chex.assert_tree_all_finite(result.params)

    def test_multistart_returns_lowest_loss_result(self) -> None:
        r"""Multistart should choose the start with the best final loss.

        Extended Summary
        ----------------
        Verifies deterministic best-result selection across a leading start
        axis of initial latent guesses.

        Notes
        -----
        It includes the exact solution as one start and checks that the
        returned result has essentially zero loss.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()
        starts: Float[Array, "starts params"] = jnp.stack(
            [
                jnp.array([-10.0, 10.0], dtype=jnp.float64),
                true_params,
            ]
        )

        result: ReconResult = multistart(
            problem=problem,
            initial_latents=starts,
            max_steps=8,
            atol=1e-12,
        )

        chex.assert_trees_all_close(result.params, true_params, atol=1e-10)
        chex.assert_trees_all_close(result.loss, 0.0, atol=1e-12)

    def test_reconstruct_incoherent_weights_recovers_planted_shape(
        self,
    ) -> None:
        r"""Incoherent weight fast path should recover planted weights.

        Extended Summary
        ----------------
        Verifies the convex linear distribution-reconstruction path for
        incoherent intensity sums.

        Notes
        -----
        It builds a small independent intensity library, mixes it with known
        simplex weights, and solves the regularized normal equations.
        """
        intensity_library: Float[Array, "samples rows cols"] = jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.5]],
                [[0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 1.0]],
            ],
            dtype=jnp.float64,
        )
        true_weights: Float[Array, "samples"] = jnp.array(
            [0.2, 0.5, 0.3],
            dtype=jnp.float64,
        )
        measured: Float[Array, "rows cols"] = jnp.einsum(
            "n,nhw->hw",
            true_weights,
            intensity_library,
        )

        weights: Float[Array, "samples"] = reconstruct_incoherent_weights(
            intensity_library=intensity_library,
            measured_image=measured,
            ridge=1e-12,
        )

        chex.assert_trees_all_close(weights, true_weights, atol=1e-8)


class TestReconSolveNamespace(chex.TestCase):
    """Tests for public solve exports."""

    def test_namespace_exports_solve_entry_points(self) -> None:
        r"""Solve APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the new solver
        surface documented by the reconstruction optimization plan.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertIs(recon.ReconProblem, ReconProblem)
        self.assertIs(recon.ReconResult, ReconResult)
        self.assertIs(recon.solve, solve)
        self.assertIs(recon.multistart, multistart)
