"""Tests for orientation-distribution probability types and integration."""

import chex
import jax
import jax.numpy as jnp
from jax import tree_util

from rheedium.types import (
    OrientationDistribution,
    create_discrete_orientation,
    create_gaussian_orientation,
    create_mixed_orientation,
    discretize_orientation,
    discretize_orientation_static,
    integrate_over_orientation,
)


class TestOrientationDistributionFactories(chex.TestCase):
    """Tests for OrientationDistribution factory helpers."""

    def test_create_discrete_orientation_defaults_equal_weights(self):
        """Discrete variants default to equal probability weights."""
        dist = create_discrete_orientation(jnp.array([33.7, -33.7]))

        assert isinstance(dist, OrientationDistribution)
        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.0, atol=1e-12)

    def test_create_mixed_orientation_clips_negative_weights(self):
        """Factory weights are clipped to a valid probability simplex."""
        dist = create_mixed_orientation(
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, -2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.5, 0.0, 0.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.3, atol=1e-12)

    def test_create_gaussian_orientation_builds_single_peak(self):
        """Gaussian orientation uses one center peak plus mosaic width."""
        dist = create_gaussian_orientation(center_deg=12.5, fwhm_deg=0.8)

        chex.assert_trees_all_close(
            dist.discrete_angles_deg,
            jnp.array([12.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([1.0]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.8, atol=1e-12)

    def test_orientation_distribution_is_pytree(self):
        """OrientationDistribution should flatten and unflatten cleanly."""
        dist = create_discrete_orientation(
            angles_deg=jnp.array([10.0, -10.0]),
            weights=jnp.array([0.25, 0.75]),
            distribution_id="twins",
        )

        flat, treedef = tree_util.tree_flatten(dist)
        reconstructed = treedef.unflatten(flat)

        assert isinstance(reconstructed, OrientationDistribution)
        chex.assert_trees_all_close(
            reconstructed.discrete_angles_deg,
            dist.discrete_angles_deg,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            reconstructed.discrete_weights,
            dist.discrete_weights,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            reconstructed.mosaic_fwhm_deg,
            dist.mosaic_fwhm_deg,
            atol=1e-12,
        )
        assert reconstructed.distribution_id == "twins"


class TestOrientationDiscretization(chex.TestCase):
    """Tests for discretizing orientation probability distributions."""

    def test_discretize_orientation_returns_normalized_weights(self):
        """Quadrature weights remain a proper probability distribution."""
        dist = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        angles_deg, weights = discretize_orientation(dist, n_mosaic_points=5)

        chex.assert_shape(angles_deg, (10,))
        chex.assert_shape(weights, (10,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-12)
        chex.assert_trees_all_equal(jnp.all(weights >= 0.0), True)

    def test_discretize_orientation_static_returns_discrete_support(self):
        """Static discretization avoids redundant quadrature for sharp peaks."""
        dist = create_discrete_orientation(
            angles_deg=jnp.array([15.0, -15.0]),
            weights=jnp.array([0.6, 0.4]),
        )

        angles_deg, weights = discretize_orientation_static(
            dist, n_mosaic_points=7
        )

        chex.assert_shape(angles_deg, (2,))
        chex.assert_trees_all_close(
            angles_deg,
            jnp.array([15.0, -15.0]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            weights,
            jnp.array([0.6, 0.4]),
            atol=1e-12,
        )

    def test_discretize_orientation_static_normalizes_manual_weights(self):
        """Manual OrientationDistribution instances still behave probabilistically."""
        dist = OrientationDistribution(
            discrete_angles_deg=jnp.array([0.0, 90.0]),
            discrete_weights=jnp.array([0.0, 0.0]),
            mosaic_fwhm_deg=jnp.array(0.0),
            distribution_id=None,
        )

        _, weights = discretize_orientation_static(dist)
        chex.assert_trees_all_close(
            weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )


class TestOrientationIntegration(chex.TestCase):
    """Tests for orientation-driven incoherent pattern integration."""

    def test_integrate_over_orientation_computes_incoherent_sum(self):
        """The final pattern is the weighted intensity sum over variants."""
        dist = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        def simulate_fn(phi_deg):
            return jnp.ones((2, 2), dtype=jnp.float64) * phi_deg**2

        pattern = integrate_over_orientation(simulate_fn, dist, 5)
        chex.assert_trees_all_close(pattern, 75.0, atol=1e-6)

    def test_grad_flows_through_orientation_angle(self):
        """Orientation integration remains differentiable in angle space."""

        def loss(angle_deg):
            dist = create_discrete_orientation(jnp.atleast_1d(angle_deg))
            pattern = integrate_over_orientation(
                lambda phi_deg: jnp.ones((2, 2), dtype=jnp.float64)
                * phi_deg**2,
                dist,
                3,
            )
            return jnp.sum(pattern)

        grad_value = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(grad_value)
        chex.assert_trees_all_close(grad_value, 16.0, atol=1e-6)

    def test_jit_compiles_orientation_integration(self):
        """Orientation integration should compile under jax.jit."""

        @jax.jit
        def run(center_deg):
            dist = create_gaussian_orientation(
                center_deg=center_deg, fwhm_deg=0.0
            )
            return integrate_over_orientation(
                lambda phi_deg: jnp.ones((3, 3), dtype=jnp.float64)
                * (phi_deg + 1.0),
                dist,
                3,
            )

        pattern = run(jnp.float64(4.0))
        chex.assert_shape(pattern, (3, 3))
        chex.assert_trees_all_close(pattern, 5.0, atol=1e-6)
