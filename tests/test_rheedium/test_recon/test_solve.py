"""Tests for recon/solve.py.

Verifies the general optimistix/optax reconstruction surface on lightweight
synthetic inverse problems with deterministic linear forward models.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    DistributionAxisSpec,
    ReconProblem,
    ReconResult,
    build_incoherent_intensity_library,
    create_distribution_axis_spec,
    multistart,
    reconstruct_distribution,
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


def _amplitude_templates() -> Float[Array, "samples rows cols"]:
    """Return independent synthetic amplitude templates."""
    templates: Float[Array, "samples rows cols"] = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    return templates


def _one_hot_perturbation(
    base_templates: Float[Array, "samples rows cols"],
    sample: Float[Array, "samples"],
) -> Float[Array, "rows cols"]:
    """Select one amplitude template from a one-hot sample coordinate."""
    amplitude: Float[Array, "rows cols"] = jnp.einsum(
        "n,nhw->hw",
        sample,
        base_templates,
    )
    return amplitude


def _identity_forward(
    amplitude: Float[Array, "rows cols"],
) -> Float[Array, "rows cols"]:
    """Return a synthetic coherent amplitude image unchanged."""
    return amplitude


class TestReconDistributionReconstruction(chex.TestCase):
    """Tests for base-object distribution reconstruction.

    :see: :class:`~rheedium.recon.DistributionAxisSpec`
    :see: :func:`~rheedium.recon.create_distribution_axis_spec`
    :see: :func:`~rheedium.recon.build_incoherent_intensity_library`
    :see: :func:`~rheedium.recon.reconstruct_distribution`
    """

    def test_reconstruct_distribution_recovers_planted_axis_weights(
        self,
    ) -> None:
        r"""Base-axis reconstruction should recover planted weights.

        Extended Summary
        ----------------
        Verifies the updated plan's library-builder path: a base object and
        perturbation-axis specification build an incoherent intensity library,
        then the convex solver recovers the planted mixing distribution.

        Notes
        -----
        It uses one-hot axis samples to select independent amplitude templates,
        mixes their intensities with known weights, and checks the recovered
        distribution plus its one-sigma band.
        """
        samples: Float[Array, "samples sample_dim"] = jnp.eye(
            3,
            dtype=jnp.float64,
        )
        axis_spec: DistributionAxisSpec = create_distribution_axis_spec(
            samples=samples,
            perturbation_fn=_one_hot_perturbation,
            forward_model=_identity_forward,
            output_kind="amplitude",
            axis_id="synthetic_axis",
        )
        base_templates: Float[Array, "samples rows cols"] = (
            _amplitude_templates()
        )
        intensity_library: Float[Array, "samples rows cols"] = (
            build_incoherent_intensity_library(
                base_object=base_templates,
                axis_spec=axis_spec,
            )
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

        distribution, band = reconstruct_distribution(
            measured_image=measured,
            base_object=base_templates,
            axis_spec=axis_spec,
            ridge=1e-12,
            noise_variance=0.05,
        )

        chex.assert_trees_all_close(
            distribution.weights,
            true_weights,
            atol=1e-8,
        )
        chex.assert_trees_all_close(distribution.samples, samples, atol=1e-12)
        self.assertEqual(distribution.axis_id, "synthetic_axis")
        chex.assert_shape(band, (3,))
        chex.assert_tree_all_finite(band)


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
        self.assertIs(
            recon.create_distribution_axis_spec,
            create_distribution_axis_spec,
        )
        self.assertIs(
            recon.build_incoherent_intensity_library,
            build_incoherent_intensity_library,
        )
        self.assertIs(recon.reconstruct_distribution, reconstruct_distribution)
