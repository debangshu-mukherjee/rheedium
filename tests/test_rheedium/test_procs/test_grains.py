"""Test suite for procs/grains.py."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rheedium.procs.grains import (
    apply_misorientation_distribution,
    grain_distribution_average,
)


class TestGrainDistributionAverage(chex.TestCase):
    """Tests for grain_distribution_average."""

    def test_computes_weighted_intensity_average(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
                jnp.ones((2, 2)) * 5.0,
            ],
            axis=0,
        )
        result = grain_distribution_average(
            patterns,
            jnp.array([0.2, 0.3, 0.5]),
        )

        chex.assert_trees_all_close(result, 3.6, atol=1e-6)

    def test_clips_negative_grain_weights(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
                jnp.ones((2, 2)) * 5.0,
            ],
            axis=0,
        )
        result = grain_distribution_average(
            patterns,
            jnp.array([1.0, -2.0, 1.0]),
        )

        chex.assert_trees_all_close(result, 3.0, atol=1e-6)

    def test_grad_flows_through_grain_fraction(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )

        def objective(weight):
            return jnp.sum(
                grain_distribution_average(
                    patterns,
                    jnp.array([weight, 1.0]),
                )
            )

        grad_value = jax.grad(objective)(0.5)
        chex.assert_trees_all_close(
            float(grad_value),
            -8.0 / 2.25,
            atol=1e-6,
        )

    def test_jit_compiles(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )
        compiled = jax.jit(
            lambda fractions: grain_distribution_average(patterns, fractions)
        )

        result = compiled(jnp.array([1.0, 3.0]))
        chex.assert_trees_all_close(result, 2.5, atol=1e-6)

    def test_vmap_supports_batched_fraction_vectors(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )

        def first_pixel(fractions):
            return grain_distribution_average(patterns, fractions)[0, 0]

        batch = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = jax.vmap(first_pixel)(batch)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 2.0, 3.0]),
            atol=1e-6,
        )


class TestApplyMisorientationDistribution(chex.TestCase):
    """Tests for apply_misorientation_distribution."""

    def test_selects_patterns_near_distribution_center(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        result = apply_misorientation_distribution(
            patterns,
            jnp.array([-1.0, 0.0, 1.0]),
            jnp.ones((3,)),
            0.0,
            0.05,
        )

        chex.assert_trees_all_close(result, 4.0, atol=1e-3)

    def test_broad_width_recovers_nearly_uniform_average(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        result = apply_misorientation_distribution(
            patterns,
            jnp.array([-1.0, 0.0, 1.0]),
            jnp.ones((3,)),
            0.0,
            100.0,
        )

        chex.assert_trees_all_close(result, 14.0 / 3.0, atol=1e-4)

    def test_grad_flows_through_distribution_center(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )

        def objective(mean_angle):
            return jnp.sum(
                apply_misorientation_distribution(
                    patterns,
                    jnp.array([-1.0, 0.0, 1.0]),
                    jnp.ones((3,)),
                    mean_angle,
                    0.5,
                )
            )

        grad_value = jax.grad(objective)(0.0)
        assert np.isfinite(float(grad_value))
        assert float(grad_value) > 0.0

    def test_jit_compiles(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        compiled = jax.jit(
            lambda mean_angle: apply_misorientation_distribution(
                patterns,
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.ones((3,)),
                mean_angle,
                0.5,
            )
        )

        result = compiled(0.0)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(np.asarray(result)))

    def test_vmap_supports_batched_distribution_centers(self):
        patterns = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )

        def first_pixel(mean_angle):
            return apply_misorientation_distribution(
                patterns,
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.ones((3,)),
                mean_angle,
                0.05,
            )[0, 0]

        result = jax.vmap(first_pixel)(jnp.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 4.0, 9.0]),
            atol=1e-3,
        )
