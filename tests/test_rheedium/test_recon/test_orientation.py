"""Tests for recon/orientation.py.

These tests use a lightweight synthetic forward model so orientation
weight recovery is deterministic, cheap, and independent of the full
RHEED simulator cost.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import rheedium.recon.orientation as orientation_module
from rheedium import recon
from rheedium.recon import (
    compute_fisher_information,
    estimate_weight_uncertainty,
    fit_orientation_weights,
    orientation_loss,
)
from rheedium.types import (
    OrientationDistribution,
    OrientationFitResult,
    create_discrete_orientation,
    create_gaussian_orientation,
    create_orientation_distribution,
    integrate_over_orientation,
)
from rheedium.types.custom_types import scalar_float


def _synthetic_pattern(phi_deg: scalar_float) -> Float[Array, "rows cols"]:
    """Map an orientation angle to a distinct positive detector image."""
    x: scalar_float = jnp.asarray(phi_deg, dtype=jnp.float64) / 10.0
    return jnp.array(
        [
            [1.0 + x, 0.5 + x**2, 0.25 + x**3],
            [1.5 + x**4, 0.1 + x**5, 0.2 + x + x**2],
        ],
        dtype=jnp.float64,
    )


def _true_distribution() -> OrientationDistribution:
    """Return reference discrete orientation mixture for inverse tests."""
    return create_discrete_orientation(
        angles_deg=jnp.array([0.0, 10.0, 20.0]),
        weights=jnp.array([0.15, 0.55, 0.30]),
        distribution_id="synthetic_truth",
    )


def _zero_pixel_gaussian_pattern(
    phi_deg: scalar_float,
) -> Float[Array, "rows cols"]:
    """Return a thresholded Gaussian spot with many exact-zero pixels."""
    axis: Float[Array, "pixels"] = jnp.linspace(-1.0, 1.0, 16)
    x_grid: Float[Array, "rows cols"]
    y_grid: Float[Array, "rows cols"]
    x_grid, y_grid = jnp.meshgrid(axis, axis, indexing="xy")
    center: scalar_float = jnp.asarray(phi_deg, dtype=jnp.float64) / 20.0
    raw_spot: Float[Array, "rows cols"] = jnp.exp(
        -((x_grid - center) ** 2 + y_grid**2) / 0.08
    )
    return jnp.where(raw_spot > 0.08, raw_spot, 0.0)


class TestOrientationLoss(chex.TestCase):
    """Tests for the forward loss used in orientation fitting.

    :see: :func:`~rheedium.recon.orientation_loss`
    """

    def test_orientation_loss_is_zero_for_matching_distribution(self) -> None:
        r"""The loss should vanish when the trial distribution matches data.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The loss should
        vanish when the trial distribution matches data.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_orientation``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: OrientationDistribution = _true_distribution()
        observed: Float[Array, "rows cols"] = integrate_over_orientation(
            _synthetic_pattern,
            distribution,
            n_mosaic_points=1,
        )

        loss: scalar_float = orientation_loss(
            distribution=distribution,
            simulate_fn=_synthetic_pattern,
            observed_pattern=observed,
            normalize=False,
            n_mosaic_points=1,
        )

        chex.assert_trees_all_close(loss, 0.0, atol=1e-10)

    def test_orientation_loss_soft_mask_weights_residual_once(self) -> None:
        r"""Soft masks should scale squared residuals linearly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Soft masks
        should scale squared residuals linearly.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        distribution: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0]),
            weights=jnp.array([1.0]),
        )

        def simulate_fn(_phi_deg: scalar_float) -> Float[Array, "1 2"]:
            return jnp.array([[0.0, 1.0]], dtype=jnp.float64)

        observed: Float[Array, "1 2"] = jnp.zeros((1, 2), dtype=jnp.float64)
        mask: Float[Array, "1 2"] = jnp.array([[1.0, 0.5]], dtype=jnp.float64)

        loss: scalar_float = orientation_loss(
            distribution=distribution,
            simulate_fn=simulate_fn,
            observed_pattern=observed,
            mask=mask,
            normalize=False,
            n_mosaic_points=1,
        )

        chex.assert_trees_all_close(loss, 1.0 / 3.0, atol=1e-12)


class TestOrientationFitting(chex.TestCase):
    """Tests for weight recovery on a fixed orientation support.

    :see: :func:`~rheedium.recon.fit_orientation_weights`
    """

    def test_fit_orientation_weights_recovers_synthetic_weights(self) -> None:
        r"""The inverse fit should recover the synthetic mixture weights.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The inverse fit
        should recover the synthetic mixture weights.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_orientation``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        true_distribution: OrientationDistribution = _true_distribution()
        observed: Float[Array, "rows cols"] = integrate_over_orientation(
            _synthetic_pattern,
            true_distribution,
            n_mosaic_points=1,
        )

        result: OrientationFitResult = fit_orientation_weights(
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

    def test_softplus_large_latent_has_finite_unit_gradient(self) -> None:
        r"""Softplus should remain finite for a large positive latent.

        Extended Summary
        ----------------
        Verifies the asymptotically linear forward value and unit derivative at
        a latent large enough to overflow the naive exponential formulation.

        Notes
        -----
        The ``1000.0`` audit boundary previously produced both an infinite
        forward value and a ``NaN`` gradient.
        """
        latent: Float[Array, ""] = jnp.asarray(1000.0, dtype=jnp.float64)

        value: Float[Array, ""] = orientation_module._softplus(latent)
        gradient: Float[Array, ""] = jax.grad(orientation_module._softplus)(
            latent
        )

        chex.assert_tree_all_finite(value)
        chex.assert_tree_all_finite(gradient)
        chex.assert_trees_all_close(value, 1000.0, atol=0.0)
        chex.assert_trees_all_close(gradient, 1.0, atol=0.0)

    def test_zero_pixel_mosaic_gradients_and_fit_are_finite(self) -> None:
        r"""Zero-intensity pixels should not NaN orientation gradients.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero-
        intensity pixels should not NaN orientation gradients.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        base_image: Float[Array, "rows cols"] = _zero_pixel_gaussian_pattern(
            0.0
        )
        zero_fraction: Float[Array, ""] = jnp.mean(base_image == 0.0)
        self.assertGreaterEqual(float(zero_fraction), 0.4)

        def loss(
            center_deg: scalar_float, fwhm_deg: scalar_float
        ) -> scalar_float:
            distribution = create_orientation_distribution(
                angles_deg=jnp.atleast_1d(center_deg),
                weights=jnp.ones(1, dtype=jnp.float64),
                mosaic_fwhm_deg=fwhm_deg,
            )
            pattern: Float[Array, "rows cols"] = integrate_over_orientation(
                _zero_pixel_gaussian_pattern,
                distribution,
                n_mosaic_points=5,
            )
            return jnp.mean(pattern)

        grad_center: scalar_float
        grad_fwhm: scalar_float
        grad_center, grad_fwhm = jax.grad(loss, argnums=(0, 1))(
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.4, dtype=jnp.float64),
        )
        chex.assert_tree_all_finite(grad_center)
        chex.assert_tree_all_finite(grad_fwhm)

        observed: Float[Array, "rows cols"] = integrate_over_orientation(
            _zero_pixel_gaussian_pattern,
            create_gaussian_orientation(center_deg=0.0, fwhm_deg=0.4),
            n_mosaic_points=5,
        )
        result: OrientationFitResult = fit_orientation_weights(
            observed_pattern=observed,
            simulate_fn=_zero_pixel_gaussian_pattern,
            candidate_angles_deg=jnp.array([-0.5, 0.0, 0.5]),
            learning_rate=0.03,
            n_iterations=10,
            convergence_tol=1e-12,
            regularization_strength=0.0,
            entropy_weight=0.0,
            n_mosaic_points=5,
            normalize=False,
        )

        chex.assert_tree_all_finite(
            result.fitted_distribution.discrete_weights
        )
        chex.assert_tree_all_finite(result.fitted_distribution.mosaic_fwhm_deg)


class TestOrientationUncertainty(chex.TestCase):
    """Tests for Fisher-based orientation uncertainty estimates.

    :see: :func:`~rheedium.recon.compute_fisher_information`
    :see: :func:`~rheedium.recon.estimate_weight_uncertainty`
    """

    def test_fisher_information_tracks_discrete_weight_count(self) -> None:
        r"""Fisher shape should depend on discrete weights, not quadrature.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fisher shape
        should depend on discrete weights, not quadrature.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_orientation``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 20.0]),
            weights=jnp.array([0.4, 0.6]),
        )

        fisher: Float[Array, "weights weights"] = compute_fisher_information(
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

    def test_estimate_weight_uncertainty_returns_one_sigma_per_weight(
        self,
    ) -> None:
        r"""Uncertainty output should match the fitted weight dimension.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Uncertainty output
        should match the fitted weight dimension.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_orientation``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: OrientationDistribution = _true_distribution()
        observed: Float[Array, "rows cols"] = integrate_over_orientation(
            _synthetic_pattern,
            distribution,
            n_mosaic_points=1,
        )
        result: OrientationFitResult = OrientationFitResult(
            fitted_distribution=distribution,
            final_loss=jnp.array(0.0, dtype=jnp.float64),
            loss_history=jnp.array([0.0], dtype=jnp.float64),
            converged=True,
            n_iterations=1,
            residual_pattern=jnp.zeros_like(observed),
        )

        uncertainties: Float[Array, "weights"] = estimate_weight_uncertainty(
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

    def test_namespace_exports_orientation_entry_points(self) -> None:
        r"""Orientation APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation APIs
        should be re-exported from rheedium.recon.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_recon.test_orientation``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
