"""Tests for recon/orientation.py.

These tests use a lightweight synthetic forward model so orientation
weight recovery is deterministic, cheap, and independent of the full
RHEED simulator cost.
"""

import chex
import jax.numpy as jnp

import rheedium.recon as recon
from rheedium.recon import (
    OrientationFitResult,
    compute_fisher_information,
    estimate_weight_uncertainty,
    fit_orientation_weights,
    orientation_loss,
)
from rheedium.types import (
    OrientationDistribution,
    create_discrete_orientation,
    integrate_over_orientation,
)


def _synthetic_pattern(phi_deg: jnp.ndarray) -> jnp.ndarray:
    """Map an orientation angle to a distinct positive detector image."""
    x = jnp.asarray(phi_deg, dtype=jnp.float64) / 10.0
    return jnp.array(
        [
            [1.0 + x, 0.5 + x**2, 0.25 + x**3],
            [1.5 + x**4, 0.1 + x**5, 0.2 + x + x**2],
        ],
        dtype=jnp.float64,
    )


def _true_distribution() -> OrientationDistribution:
    """Reference discrete orientation mixture for inverse tests."""
    return create_discrete_orientation(
        angles_deg=jnp.array([0.0, 10.0, 20.0]),
        weights=jnp.array([0.15, 0.55, 0.30]),
        distribution_id="synthetic_truth",
    )


class TestOrientationLoss(chex.TestCase):
    """Tests for the forward loss used in orientation fitting."""

    def test_orientation_loss_is_zero_for_matching_distribution(self):
        """The loss should vanish when the trial distribution matches data."""
        distribution = _true_distribution()
        observed = integrate_over_orientation(
            _synthetic_pattern,
            distribution,
            n_mosaic_points=1,
        )

        loss = orientation_loss(
            distribution=distribution,
            simulate_fn=_synthetic_pattern,
            observed_pattern=observed,
            normalize=False,
            n_mosaic_points=1,
        )

        chex.assert_trees_all_close(loss, 0.0, atol=1e-10)


class TestOrientationFitting(chex.TestCase):
    """Tests for weight recovery on a fixed orientation support."""

    def test_fit_orientation_weights_recovers_synthetic_weights(self):
        """The inverse fit should recover the synthetic mixture weights."""
        true_distribution = _true_distribution()
        observed = integrate_over_orientation(
            _synthetic_pattern,
            true_distribution,
            n_mosaic_points=1,
        )

        result = fit_orientation_weights(
            observed_pattern=observed,
            simulate_fn=_synthetic_pattern,
            candidate_angles_deg=true_distribution.discrete_angles_deg,
            learning_rate=0.05,
            n_iterations=400,
            convergence_tol=1e-9,
            regularization_strength=0.0,
            entropy_weight=0.0,
            n_mosaic_points=1,
            normalize=False,
        )

        self.assertIsInstance(result, OrientationFitResult)
        chex.assert_tree_all_finite(result.loss_history)
        self.assertLess(float(result.final_loss), 1e-8)
        chex.assert_trees_all_close(
            result.fitted_distribution.discrete_weights,
            true_distribution.discrete_weights,
            atol=5e-2,
        )
        self.assertLess(
            float(jnp.max(jnp.abs(result.residual_pattern))),
            1e-3,
        )


class TestOrientationUncertainty(chex.TestCase):
    """Tests for Fisher-based orientation uncertainty estimates."""

    def test_fisher_information_tracks_discrete_weight_count(self):
        """Fisher shape should depend on discrete weights, not quadrature."""
        distribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 20.0]),
            weights=jnp.array([0.4, 0.6]),
        )

        fisher = compute_fisher_information(
            simulate_fn=_synthetic_pattern,
            distribution=distribution,
            noise_variance=0.5,
            normalize=False,
            n_mosaic_points=5,
        )

        chex.assert_shape(fisher, (2, 2))
        chex.assert_tree_all_finite(fisher)
        chex.assert_trees_all_close(fisher, fisher.T, atol=1e-10)
        self.assertTrue(bool(jnp.all(jnp.linalg.eigvalsh(fisher) >= -1e-10)))

    def test_estimate_weight_uncertainty_returns_one_sigma_per_weight(self):
        """Uncertainty output should match the fitted weight dimension."""
        distribution = _true_distribution()
        observed = integrate_over_orientation(
            _synthetic_pattern,
            distribution,
            n_mosaic_points=1,
        )
        result = OrientationFitResult(
            fitted_distribution=distribution,
            final_loss=jnp.array(0.0, dtype=jnp.float64),
            loss_history=jnp.array([0.0], dtype=jnp.float64),
            converged=True,
            n_iterations=1,
            residual_pattern=jnp.zeros_like(observed),
        )

        uncertainties = estimate_weight_uncertainty(
            result=result,
            simulate_fn=_synthetic_pattern,
            noise_variance=1.0,
            normalize=False,
            n_mosaic_points=3,
        )

        chex.assert_shape(uncertainties, (3,))
        chex.assert_tree_all_finite(uncertainties)
        self.assertTrue(bool(jnp.all(uncertainties >= 0.0)))


class TestReconNamespace(chex.TestCase):
    """Tests for public recon exports."""

    def test_namespace_exports_orientation_entry_points(self):
        """Orientation APIs should be re-exported from rheedium.recon."""
        self.assertIs(recon.orientation_loss, orientation_loss)
        self.assertIs(recon.fit_orientation_weights, fit_orientation_weights)
        self.assertIs(
            recon.compute_fisher_information,
            compute_fisher_information,
        )
        self.assertIs(
            recon.estimate_weight_uncertainty,
            estimate_weight_uncertainty,
        )
