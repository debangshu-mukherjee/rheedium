"""Tests for orientation-distribution probability types and integration."""

from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import tree_util
from jaxtyping import Array, Float

from rheedium.types import (
    OrientationDistribution,
    SizeDistribution,
    create_discrete_orientation,
    create_gaussian_orientation,
    create_lognormal_size,
    create_mixed_orientation,
    discretize_orientation,
    discretize_orientation_static,
    integrate_over_orientation,
)
from rheedium.types.custom_types import scalar_float


def assert_rejects(
    fn: Callable[..., object],
    *args: object,
    match: str | None = None,
    **kwargs: object,
) -> None:
    """Assert a call rejects eagerly and under ``eqx.filter_jit``."""
    with pytest.raises(Exception, match=match):
        fn(*args, **kwargs)

    with pytest.raises(Exception, match=match):
        eqx.filter_jit(lambda: fn(*args, **kwargs))()


class TestOrientationDistributionFactories(chex.TestCase):
    """Tests for OrientationDistribution factory helpers."""

    def test_create_discrete_orientation_defaults_equal_weights(self) -> None:
        """Discrete variants default to equal probability weights."""
        dist: OrientationDistribution = create_discrete_orientation(
            jnp.array([33.7, -33.7])
        )

        assert isinstance(dist, OrientationDistribution)
        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.0, atol=1e-12)

    def test_create_mixed_orientation_rejects_negative_weights(self) -> None:
        """Factory weights must be valid probabilities."""
        assert_rejects(
            create_mixed_orientation,
            match="weights must be non-negative",
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, -2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

    def test_create_mixed_orientation_normalizes_weights(self) -> None:
        """Factory weights are normalized when valid."""
        dist: OrientationDistribution = create_mixed_orientation(
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, 2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.25, 0.5, 0.25]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.3, atol=1e-12)

    def test_create_gaussian_orientation_builds_single_peak(self) -> None:
        """Gaussian orientation uses one center peak plus mosaic width."""
        dist: OrientationDistribution = create_gaussian_orientation(
            center_deg=12.5, fwhm_deg=0.8
        )

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

    def test_orientation_distribution_is_pytree(self) -> None:
        """OrientationDistribution should flatten and unflatten cleanly."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([10.0, -10.0]),
            weights=jnp.array([0.25, 0.75]),
            distribution_id="twins",
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(dist)
        reconstructed: OrientationDistribution = treedef.unflatten(flat)

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

    def test_create_discrete_orientation_rejects_nan_angle(self) -> None:
        """Factory angles must be finite."""
        assert_rejects(
            create_discrete_orientation,
            jnp.array([0.0, jnp.nan]),
            match="angles_deg must be finite",
        )

    def test_create_gaussian_orientation_rejects_negative_fwhm(self) -> None:
        """Mosaic FWHM must be non-negative."""
        assert_rejects(
            create_gaussian_orientation,
            fwhm_deg=-0.1,
            match="mosaic_fwhm_deg must be non-negative",
        )

    def test_create_mixed_orientation_rejects_zero_weight_sum(self) -> None:
        """Factory weights must have positive total probability."""
        assert_rejects(
            create_mixed_orientation,
            angles_deg=jnp.array([0.0, 90.0]),
            weights=jnp.array([0.0, 0.0]),
            match="weights must have positive total probability",
        )


class TestSizeDistributionFactories(chex.TestCase):
    """Tests for size-distribution factory helpers."""

    def test_create_lognormal_size_valid(self) -> None:
        """Valid lognormal size parameters should be preserved."""
        dist: SizeDistribution = create_lognormal_size(
            mean_ang=100.0,
            sigma_ang=30.0,
            min_size_ang=10.0,
            max_size_ang=500.0,
        )

        assert isinstance(dist, SizeDistribution)
        chex.assert_trees_all_close(dist.mean_ang, 100.0)
        chex.assert_trees_all_close(dist.sigma_ang, 30.0)
        chex.assert_trees_all_close(dist.min_size_ang, 10.0)
        chex.assert_trees_all_close(dist.max_size_ang, 500.0)

    def test_create_lognormal_size_rejects_negative_mean(self) -> None:
        """Mean domain size must be positive."""
        assert_rejects(
            create_lognormal_size,
            mean_ang=-1.0,
            match="mean_ang must be positive",
        )

    def test_create_lognormal_size_rejects_negative_sigma(self) -> None:
        """Size spread must be non-negative."""
        assert_rejects(
            create_lognormal_size,
            sigma_ang=-1.0,
            match="sigma_ang must be non-negative",
        )

    def test_create_lognormal_size_rejects_invalid_bounds(self) -> None:
        """Maximum size must exceed minimum size."""
        assert_rejects(
            create_lognormal_size,
            min_size_ang=100.0,
            max_size_ang=10.0,
            match="max_size_ang must be greater than min_size_ang",
        )


class TestOrientationDiscretization(chex.TestCase):
    """Tests for discretizing orientation probability distributions."""

    def test_discretize_orientation_returns_normalized_weights(self) -> None:
        """Quadrature weights remain a proper probability distribution."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        angles_deg: Float[Array, "10"]
        weights: Float[Array, "10"]
        angles_deg, weights = discretize_orientation(dist, n_mosaic_points=5)

        chex.assert_shape(angles_deg, (10,))
        chex.assert_shape(weights, (10,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-12)
        chex.assert_trees_all_equal(jnp.all(weights >= 0.0), True)

    def test_discretize_orientation_static_returns_discrete_support(
        self,
    ) -> None:
        """Avoid redundant quadrature for sharp discrete peaks."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([15.0, -15.0]),
            weights=jnp.array([0.6, 0.4]),
        )

        angles_deg: Float[Array, "2"]
        weights: Float[Array, "2"]
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

    def test_discretize_orientation_static_normalizes_manual_weights(
        self,
    ) -> None:
        """Normalize manual OrientationDistribution weights."""
        dist: OrientationDistribution = OrientationDistribution(
            discrete_angles_deg=jnp.array([0.0, 90.0]),
            discrete_weights=jnp.array([0.0, 0.0]),
            mosaic_fwhm_deg=jnp.array(0.0),
            distribution_id=None,
        )

        angles_deg: Float[Array, "2"]
        weights: Float[Array, "2"]
        angles_deg, weights = discretize_orientation_static(dist)
        chex.assert_trees_all_close(
            weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )


class TestOrientationIntegration(chex.TestCase):
    """Tests for orientation-driven incoherent pattern integration."""

    def test_integrate_over_orientation_computes_incoherent_sum(self) -> None:
        """The final pattern is the weighted intensity sum over variants."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        def simulate_fn(phi_deg: scalar_float) -> Float[Array, "2 2"]:
            return jnp.ones((2, 2), dtype=jnp.float64) * phi_deg**2

        pattern: Float[Array, "2 2"] = integrate_over_orientation(
            simulate_fn, dist, 5
        )
        chex.assert_trees_all_close(pattern, 75.0, atol=1e-6)

    def test_grad_flows_through_orientation_angle(self) -> None:
        """Orientation integration remains differentiable in angle space."""

        def loss(angle_deg: scalar_float) -> scalar_float:
            dist: OrientationDistribution = create_discrete_orientation(
                jnp.atleast_1d(angle_deg)
            )
            pattern: Float[Array, "2 2"] = integrate_over_orientation(
                lambda phi_deg: jnp.ones((2, 2), dtype=jnp.float64)
                * phi_deg**2,
                dist,
                3,
            )
            return jnp.sum(pattern)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(grad_value)
        chex.assert_trees_all_close(grad_value, 16.0, atol=1e-6)

    def test_jit_compiles_orientation_integration(self) -> None:
        """Orientation integration should compile under jax.jit."""

        @jax.jit
        def run(center_deg: scalar_float) -> Float[Array, "3 3"]:
            dist: OrientationDistribution = create_gaussian_orientation(
                center_deg=center_deg, fwhm_deg=0.0
            )
            return integrate_over_orientation(
                lambda phi_deg: jnp.ones((3, 3), dtype=jnp.float64)
                * (phi_deg + 1.0),
                dist,
                3,
            )

        pattern: Float[Array, "3 3"] = run(jnp.float64(4.0))
        chex.assert_shape(pattern, (3, 3))
        chex.assert_trees_all_close(pattern, 5.0, atol=1e-6)
