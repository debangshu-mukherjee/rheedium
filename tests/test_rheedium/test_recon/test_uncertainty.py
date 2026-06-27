"""Tests for recon/uncertainty.py.

Verifies generalized Fisher-information and Laplace covariance helpers on a
small linear residual model with an analytic reference matrix.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    LaplaceUncertainty,
    PosteriorSamples,
    compute_fisher_information,
    covariance_from_fisher,
    fisher_information_from_residual,
    laplace_inverse_mass_matrix,
    laplace_uncertainty,
    posterior_from_samples,
    sample_posterior,
)
from rheedium.types import (
    OrientationDistribution,
    create_discrete_orientation,
    integrate_over_orientation,
)
from rheedium.types.custom_types import scalar_float

_SENSITIVITY: Float[Array, "residuals params"] = jnp.array(
    [[2.0, 0.0], [1.0, -1.0], [0.5, 2.0]],
    dtype=jnp.float64,
)


def _linear_residual(
    params: Float[Array, "params"],
) -> Float[Array, "residuals"]:
    """Return a linear residual vector for uncertainty tests."""
    residual: Float[Array, "residuals"] = _SENSITIVITY @ params
    return residual


def _orientation_pattern(phi_deg: scalar_float) -> Float[Array, "rows cols"]:
    """Return a lightweight orientation-dependent detector image."""
    x: scalar_float = jnp.asarray(phi_deg, dtype=jnp.float64) / 10.0
    pattern: Float[Array, "rows cols"] = jnp.array(
        [
            [1.0 + x, 0.25 + x**2],
            [0.5 + 0.2 * x, 0.75 + x**3],
        ],
        dtype=jnp.float64,
    )
    return pattern


class TestReconUncertainty(chex.TestCase):
    """Tests for Fisher and Laplace reconstruction uncertainty.

    :see: :func:`~rheedium.recon.fisher_information_from_residual`
    :see: :func:`~rheedium.recon.laplace_uncertainty`
    """

    def test_fisher_information_matches_linear_reference(self) -> None:
        r"""Fisher information should match the analytic linear result.

        Extended Summary
        ----------------
        Verifies that residual Jacobians are flattened correctly for arbitrary
        parameter pytrees by comparing against ``J.T @ J / variance``.

        Notes
        -----
        It evaluates the Fisher matrix at a nonzero parameter vector and uses
        chex closeness checks against the analytic expression.
        """
        params: Float[Array, "params"] = jnp.array(
            [1.0, -0.25],
            dtype=jnp.float64,
        )
        noise_variance: Float[Array, ""] = jnp.asarray(
            0.5,
            dtype=jnp.float64,
        )

        fisher: Float[Array, "params params"] = (
            fisher_information_from_residual(
                residual_fn=_linear_residual,
                params=params,
                noise_variance=noise_variance,
            )
        )

        expected: Float[Array, "params params"] = (
            _SENSITIVITY.T @ _SENSITIVITY
        ) / noise_variance
        chex.assert_trees_all_close(fisher, expected, atol=1e-12)

    def test_laplace_uncertainty_returns_psd_covariance(self) -> None:
        r"""Laplace uncertainty should produce finite covariance summaries.

        Extended Summary
        ----------------
        Verifies that the covariance inversion and correlation normalization
        return finite, symmetric uncertainty summaries.

        Notes
        -----
        It checks covariance symmetry, non-negative eigenvalues, finite
        standard deviations, and unit diagonal correlation entries.
        """
        params: Float[Array, "params"] = jnp.array(
            [0.5, 0.25],
            dtype=jnp.float64,
        )

        uncertainty: LaplaceUncertainty = laplace_uncertainty(
            residual_fn=_linear_residual,
            params=params,
            regularization=1e-8,
        )

        chex.assert_tree_all_finite(uncertainty.covariance)
        chex.assert_trees_all_close(
            uncertainty.covariance,
            uncertainty.covariance.T,
            atol=1e-12,
        )
        self.assertTrue(
            bool(
                jnp.all(jnp.linalg.eigvalsh(uncertainty.covariance) >= -1e-10)
            )
        )
        chex.assert_tree_all_finite(uncertainty.standard_deviation)
        chex.assert_trees_all_close(
            jnp.diag(uncertainty.correlation),
            jnp.ones(2, dtype=jnp.float64),
            atol=1e-8,
        )

    def test_laplace_covariance_matches_empirical_linear_spread(self) -> None:
        r"""Laplace covariance should match noisy linear fit spread.

        Extended Summary
        ----------------
        Verifies the K4 calibration gate in a linear Gaussian setting: the
        analytic Laplace covariance from the Fisher matrix matches the
        empirical spread of many noisy least-squares reconstructions.

        Notes
        -----
        A fixed PRNG key makes the empirical comparison reproducible while
        still exercising a genuine ensemble of synthetic measurements.
        """
        noise_sigma: Float[Array, ""] = jnp.asarray(0.05, dtype=jnp.float64)
        true_params: Float[Array, "params"] = jnp.array(
            [0.5, -0.25],
            dtype=jnp.float64,
        )
        clean_measurement: Float[Array, "residuals"] = (
            _SENSITIVITY @ true_params
        )
        key: Array = jax.random.PRNGKey(7)
        noise: Float[Array, "draws residuals"] = (
            noise_sigma
            * jax.random.normal(
                key,
                (2048, _SENSITIVITY.shape[0]),
                dtype=jnp.float64,
            )
        )
        pseudo_inverse: Float[Array, "params residuals"] = jnp.linalg.pinv(
            _SENSITIVITY
        )
        estimates: Float[Array, "draws params"] = (
            clean_measurement[None, :] + noise
        ) @ pseudo_inverse.T
        centered: Float[Array, "draws params"] = estimates - jnp.mean(
            estimates,
            axis=0,
            keepdims=True,
        )
        empirical_covariance: Float[Array, "params params"] = (
            centered.T @ centered / (estimates.shape[0] - 1)
        )

        uncertainty: LaplaceUncertainty = laplace_uncertainty(
            residual_fn=_linear_residual,
            params=true_params,
            noise_variance=noise_sigma**2,
            regularization=1e-12,
        )

        chex.assert_trees_all_close(
            empirical_covariance,
            uncertainty.covariance,
            rtol=0.2,
            atol=5e-5,
        )

    def test_orientation_fisher_reduces_to_general_residual_fisher(
        self,
    ) -> None:
        r"""General Fisher helper should reduce to orientation Fisher.

        Extended Summary
        ----------------
        Verifies the K4 orientation regression: the generic residual-Jacobian
        Fisher matrix matches the orientation-specific Fisher computation when
        both are evaluated in the same softmax-logit parameterization.

        Notes
        -----
        The mosaic width is held fixed, matching the documented conditional
        uncertainty contract of ``compute_fisher_information``.
        """
        weights: Float[Array, "weights"] = jnp.array(
            [0.3, 0.7],
            dtype=jnp.float64,
        )
        distribution: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0], dtype=jnp.float64),
            weights=weights,
        )
        logits: Float[Array, "weights"] = jnp.log(weights)
        noise_variance: Float[Array, ""] = jnp.asarray(
            0.25,
            dtype=jnp.float64,
        )

        def residual_from_logits(
            trial_logits: Float[Array, "weights"],
        ) -> Float[Array, "pixels"]:
            trial_distribution: OrientationDistribution = (
                create_discrete_orientation(
                    angles_deg=distribution.discrete_angles_deg,
                    weights=jax.nn.softmax(trial_logits),
                )
            )
            pattern: Float[Array, "rows cols"] = integrate_over_orientation(
                _orientation_pattern,
                trial_distribution,
                n_mosaic_points=1,
            )
            residual: Float[Array, "pixels"] = jnp.ravel(pattern)
            return residual

        general_fisher: Float[Array, "weights weights"] = (
            fisher_information_from_residual(
                residual_fn=residual_from_logits,
                params=logits,
                noise_variance=noise_variance,
            )
        )
        orientation_fisher: Float[Array, "weights weights"] = (
            compute_fisher_information(
                simulate_fn=_orientation_pattern,
                distribution=distribution,
                noise_variance=noise_variance,
                normalize=False,
                n_mosaic_points=1,
            )
        )

        chex.assert_trees_all_close(
            general_fisher,
            orientation_fisher,
            atol=1e-10,
        )


class TestReconPosteriorUncertainty(chex.TestCase):
    """Tests for blackjax posterior UQ and diagnostics.

    :see: :class:`~rheedium.recon.PosteriorSamples`
    :see: :func:`~rheedium.recon.laplace_inverse_mass_matrix`
    :see: :func:`~rheedium.recon.posterior_from_samples`
    :see: :func:`~rheedium.recon.sample_posterior`
    """

    def test_blackjax_sampler_recovers_gaussian_posterior(self) -> None:
        r"""NUTS posterior samples should recover a Gaussian reference.

        Extended Summary
        ----------------
        Verifies the K4 posterior-first UQ path on a known Gaussian posterior:
        blackjax NUTS chains converge by R-hat/ESS, recover the analytic mean
        and covariance, and produce credible intervals containing the true
        location.

        Notes
        -----
        The test uses a fixed inverse mass matrix and short chains so it stays
        deterministic and cheap while still exercising the actual sampler.
        """
        mean: Float[Array, "params"] = jnp.array(
            [0.25, -0.5],
            dtype=jnp.float64,
        )
        precision: Float[Array, "params params"] = jnp.array(
            [[4.0, 0.0], [0.0, 1.0]],
            dtype=jnp.float64,
        )

        def log_probability(
            params: Float[Array, "params"],
        ) -> Float[Array, ""]:
            delta: Float[Array, "params"] = params - mean
            logp: Float[Array, ""] = -0.5 * delta @ precision @ delta
            return logp

        starts: Float[Array, "chains params"] = jnp.array(
            [
                [0.0, 0.0],
                [0.5, -1.0],
                [0.1, -0.75],
                [0.4, -0.25],
            ],
            dtype=jnp.float64,
        )
        posterior: PosteriorSamples = sample_posterior(
            log_probability_fn=log_probability,
            initial_position=starts,
            key=jax.random.PRNGKey(11),
            num_samples=192,
            num_warmup=0,
            step_size=0.45,
            inverse_mass_matrix=jnp.diag(precision),
            adapt=False,
            r_hat_threshold=1.2,
            min_effective_sample_size=20.0,
        )

        self.assertTrue(bool(posterior.converged))
        self.assertLess(float(jnp.max(posterior.r_hat)), 1.2)
        self.assertGreater(
            float(jnp.min(posterior.effective_sample_size)), 20.0
        )
        chex.assert_trees_all_close(posterior.mean, mean, atol=0.18)
        chex.assert_trees_all_close(
            jnp.diag(posterior.covariance),
            jnp.array([0.25, 1.0], dtype=jnp.float64),
            rtol=0.4,
            atol=0.12,
        )
        self.assertTrue(
            bool(
                jnp.all(posterior.credible_interval[0] < mean)
                and jnp.all(mean < posterior.credible_interval[1])
            )
        )

    def test_multistart_sampler_preserves_bimodal_posterior(self) -> None:
        r"""Multistart NUTS chains should retain separated posterior modes.

        Extended Summary
        ----------------
        Verifies the K4 multimodal guard: rows of the initial-position array
        act like K3 multistart chains, so a bimodal posterior is represented by
        samples near both modes rather than collapsed to one local basin.

        Notes
        -----
        The intentionally high R-hat is itself diagnostic: the posterior is
        multimodal and should not be summarized as one mixed Gaussian without
        that warning.
        """
        mode_scale: Float[Array, ""] = jnp.asarray(0.25, dtype=jnp.float64)

        def log_probability(
            params: Float[Array, "params"],
        ) -> Float[Array, ""]:
            x: Float[Array, ""] = params[0]
            left: Float[Array, ""] = -0.5 * ((x + 2.0) / mode_scale) ** 2
            right: Float[Array, ""] = -0.5 * ((x - 2.0) / mode_scale) ** 2
            logp: Float[Array, ""] = jax.nn.logsumexp(
                jnp.array([left, right], dtype=jnp.float64)
            )
            return logp

        starts: Float[Array, "chains params"] = jnp.array(
            [[-2.0], [-1.8], [2.0], [1.8]],
            dtype=jnp.float64,
        )
        posterior: PosteriorSamples = sample_posterior(
            log_probability_fn=log_probability,
            initial_position=starts,
            key=jax.random.PRNGKey(13),
            num_samples=96,
            num_warmup=0,
            step_size=0.15,
            inverse_mass_matrix=jnp.array([16.0], dtype=jnp.float64),
            adapt=False,
            r_hat_threshold=10.0,
            min_effective_sample_size=1.0,
        )
        draws: Float[Array, "draws"] = jnp.ravel(posterior.samples[:, :, 0])

        self.assertTrue(bool(jnp.any(draws < -1.0)))
        self.assertTrue(bool(jnp.any(draws > 1.0)))
        self.assertGreater(float(posterior.r_hat[0]), 1.2)

    def test_laplace_inverse_mass_and_sample_summary(self) -> None:
        r"""Laplace precision should warm-start posterior summaries.

        Extended Summary
        ----------------
        Verifies the bridge from the fast Laplace approximation to blackjax:
        Fisher/JᵀJ precision becomes a positive inverse-mass matrix, and
        posterior summaries computed from explicit samples report finite
        intervals, covariance, and diagnostics.

        Notes
        -----
        This test avoids random sampling so the summary path is pinned exactly.
        """
        uncertainty: LaplaceUncertainty = laplace_uncertainty(
            residual_fn=_linear_residual,
            params=jnp.array([0.0, 0.0], dtype=jnp.float64),
            noise_variance=0.5,
            regularization=1e-8,
        )
        inverse_mass: Float[Array, "params"] = laplace_inverse_mass_matrix(
            uncertainty,
            diagonal=True,
        )
        self.assertTrue(bool(jnp.all(inverse_mass > 0.0)))
        chex.assert_trees_all_close(
            inverse_mass,
            jnp.diag(uncertainty.fisher_information) + 1e-6,
            atol=1e-12,
        )

        base_draws: Float[Array, "draws params"] = jnp.array(
            [
                [-1.0, -0.5],
                [-0.5, 0.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 2.0],
            ],
            dtype=jnp.float64,
        )
        samples: Float[Array, "chains draws params"] = jnp.stack(
            [base_draws, base_draws + 0.1],
            axis=0,
        )
        posterior: PosteriorSamples = posterior_from_samples(
            samples=samples,
            min_effective_sample_size=1.0,
            r_hat_threshold=2.0,
        )

        chex.assert_tree_all_finite(posterior.covariance)
        chex.assert_tree_all_finite(posterior.credible_interval)
        self.assertTrue(bool(jnp.all(posterior.effective_sample_size > 0.0)))

    def test_free_form_weight_posterior_band_contains_planted_shape(
        self,
    ) -> None:
        r"""Posterior bands should contain a planted free-form shape.

        Extended Summary
        ----------------
        Verifies the K4 free-form distribution-band contract directly in
        recovered-weight space: posterior samples around a planted simplex
        shape summarize to credible intervals that contain the true weights.

        Notes
        -----
        This isolates the posterior band calculation from sampler randomness;
        the actual blackjax path is exercised by the Gaussian and multimodal
        tests above.
        """
        true_weights: Float[Array, "weights"] = jnp.array(
            [0.2, 0.5, 0.3],
            dtype=jnp.float64,
        )
        offsets: Float[Array, "draws weights"] = jnp.array(
            [
                [-0.04, 0.03, 0.01],
                [-0.02, 0.01, 0.01],
                [0.0, 0.0, 0.0],
                [0.02, -0.01, -0.01],
                [0.04, -0.03, -0.01],
                [0.01, 0.02, -0.03],
            ],
            dtype=jnp.float64,
        )
        first_chain: Float[Array, "draws weights"] = true_weights + offsets
        second_chain: Float[Array, "draws weights"] = true_weights - offsets
        samples: Float[Array, "chains draws weights"] = jnp.stack(
            [first_chain, second_chain],
            axis=0,
        )
        posterior: PosteriorSamples = posterior_from_samples(
            samples=samples,
            credibility=0.8,
            r_hat_threshold=2.0,
            min_effective_sample_size=1.0,
        )

        self.assertTrue(
            bool(
                jnp.all(posterior.credible_interval[0] <= true_weights)
                and jnp.all(true_weights <= posterior.credible_interval[1])
            )
        )
        chex.assert_trees_all_close(
            posterior.mean,
            true_weights,
            atol=1e-12,
        )


class TestReconUncertaintyNamespace(chex.TestCase):
    """Tests for public uncertainty exports."""

    def test_namespace_exports_uncertainty_entry_points(self) -> None:
        r"""Uncertainty APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the Fisher and
        Laplace helpers documented as public reconstruction APIs.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertIs(recon.LaplaceUncertainty, LaplaceUncertainty)
        self.assertIs(recon.PosteriorSamples, PosteriorSamples)
        self.assertIs(
            recon.fisher_information_from_residual,
            fisher_information_from_residual,
        )
        self.assertIs(recon.covariance_from_fisher, covariance_from_fisher)
        self.assertIs(recon.laplace_uncertainty, laplace_uncertainty)
        self.assertIs(
            recon.laplace_inverse_mass_matrix,
            laplace_inverse_mass_matrix,
        )
        self.assertIs(recon.posterior_from_samples, posterior_from_samples)
        self.assertIs(recon.sample_posterior, sample_posterior)
