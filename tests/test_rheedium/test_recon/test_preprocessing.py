"""Test suite for recon/preprocessing.py.

Verifies differentiable preprocessing pipeline for experimental RHEED
images: soft masking, background subtraction, log transform,
normalization, and the full pipeline. Includes gradient tests to
ensure jax.grad flows through all operations.
"""

import chex
import jax
import jax.numpy as jnp

from rheedium.recon.preprocessing import (
    log_intensity_transform,
    normalize_image,
    preprocess_experimental,
    soft_threshold_mask,
    subtract_background,
)

H = 32
W = 32


class TestSoftThresholdMask(chex.TestCase):
    """Tests for soft_threshold_mask."""

    def test_shape_preserved(self):
        """Output shape matches distance field shape."""
        dist = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask = soft_threshold_mask(dist, jnp.float64(0.5))
        chex.assert_shape(mask, (H, W))

    def test_values_in_unit_interval(self):
        """All mask values are in (0, 1)."""
        dist = jnp.linspace(-1.0, 2.0, H * W).reshape(H, W)
        mask = soft_threshold_mask(dist, jnp.float64(0.5), jnp.float64(10.0))
        self.assertTrue(jnp.all(mask > 0.0))
        self.assertTrue(jnp.all(mask < 1.0))

    def test_above_threshold_near_one(self):
        """Pixels well above threshold have mask near 1."""
        dist = jnp.ones((H, W)) * 10.0
        mask = soft_threshold_mask(dist, jnp.float64(0.5), jnp.float64(10.0))
        self.assertTrue(jnp.all(mask > 0.99))

    def test_below_threshold_near_zero(self):
        """Pixels well below threshold have mask near 0."""
        dist = jnp.ones((H, W)) * (-10.0)
        mask = soft_threshold_mask(dist, jnp.float64(0.5), jnp.float64(10.0))
        self.assertTrue(jnp.all(mask < 0.01))

    def test_at_threshold_is_half(self):
        """At the threshold, mask equals 0.5."""
        dist = jnp.ones((H, W)) * 0.5
        mask = soft_threshold_mask(dist, jnp.float64(0.5), jnp.float64(10.0))
        chex.assert_trees_all_close(mask, jnp.full((H, W), 0.5), atol=1e-12)

    def test_higher_sharpness_steeper(self):
        """Higher sharpness produces steeper transition."""
        dist = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask_soft = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(1.0)
        )
        mask_sharp = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(100.0)
        )
        range_soft = jnp.max(mask_soft) - jnp.min(mask_soft)
        range_sharp = jnp.max(mask_sharp) - jnp.min(mask_sharp)
        self.assertTrue(range_sharp >= range_soft)


class TestSubtractBackground(chex.TestCase):
    """Tests for subtract_background."""

    def test_correct_subtraction(self):
        """Background is correctly subtracted."""
        img = jnp.ones((H, W)) * 100.0
        bg = jnp.ones((H, W)) * 30.0
        result = subtract_background(img, bg)
        chex.assert_trees_all_close(result, jnp.full((H, W), 70.0), atol=1e-12)

    def test_nonnegative_clipping(self):
        """Result is clipped to non-negative when background exceeds image."""
        img = jnp.ones((H, W)) * 10.0
        bg = jnp.ones((H, W)) * 50.0
        result = subtract_background(img, bg)
        self.assertTrue(jnp.all(result >= 0.0))
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        img = jnp.ones((H, W))
        bg = jnp.zeros((H, W))
        result = subtract_background(img, bg)
        chex.assert_shape(result, (H, W))


class TestLogIntensityTransform(chex.TestCase):
    """Tests for log_intensity_transform."""

    def test_zero_input_is_zero(self):
        """log(1 + 0/eps) = 0 for zero input."""
        img = jnp.zeros((H, W))
        result = log_intensity_transform(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_monotonically_increasing(self):
        """Transform preserves ordering of pixel intensities."""
        vals = jnp.array([0.0, 1.0, 100.0, 10000.0])
        img = vals.reshape(2, 2)
        result = log_intensity_transform(img)
        flat = result.ravel()
        self.assertTrue(jnp.all(jnp.diff(flat) > 0.0))

    def test_compresses_dynamic_range(self):
        """Output range is smaller than input range."""
        img = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result = log_intensity_transform(img)
        input_range = jnp.max(img) - jnp.min(img)
        output_range = jnp.max(result) - jnp.min(result)
        self.assertTrue(output_range < input_range)

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        img = jnp.ones((H, W))
        result = log_intensity_transform(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self):
        """No NaN or Inf in output."""
        img = jnp.ones((H, W)) * 1e8
        result = log_intensity_transform(img)
        chex.assert_tree_all_finite(result)


class TestNormalizeImage(chex.TestCase):
    """Tests for normalize_image."""

    def test_output_range(self):
        """Output min is 0 and max is 1."""
        img = jnp.linspace(5.0, 500.0, H * W).reshape(H, W)
        result = normalize_image(img)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_uniform_image(self):
        """Uniform image normalizes to zero (no range)."""
        img = jnp.ones((H, W)) * 42.0
        result = normalize_image(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-6)

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        img = jnp.ones((H, W))
        result = normalize_image(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self):
        """No NaN or Inf in output."""
        img = jnp.linspace(0.0, 100.0, H * W).reshape(H, W)
        result = normalize_image(img)
        chex.assert_tree_all_finite(result)


class TestPreprocessExperimental(chex.TestCase):
    """Tests for the full preprocessing pipeline."""

    def test_minimal_call(self):
        """Pipeline works with only raw_image (no optional args)."""
        raw = jnp.ones((H, W)) * 100.0
        result = preprocess_experimental(raw)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_background(self):
        """Pipeline works with background subtraction."""
        raw = jnp.ones((H, W)) * 100.0
        bg = jnp.ones((H, W)) * 30.0
        result = preprocess_experimental(raw, background=bg)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_mask(self):
        """Pipeline works with soft mask."""
        raw = jnp.linspace(10.0, 500.0, H * W).reshape(H, W)
        mask = jnp.ones((H, W)) * 0.8
        result = preprocess_experimental(raw, beam_shadow_mask=mask)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_log_scale(self):
        """Pipeline works with log scale enabled."""
        raw = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result = preprocess_experimental(raw, log_scale=True)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_full_pipeline(self):
        """Pipeline works with all options enabled."""
        raw = jnp.linspace(10.0, 1e4, H * W).reshape(H, W)
        bg = jnp.ones((H, W)) * 5.0
        mask = jnp.ones((H, W)) * 0.9
        result = preprocess_experimental(
            raw,
            background=bg,
            beam_shadow_mask=mask,
            log_scale=True,
            log_epsilon=jnp.float64(1e-4),
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)
        self.assertTrue(jnp.all(result >= 0.0))
        self.assertTrue(jnp.all(result <= 1.0))

    def test_output_normalized(self):
        """Output is in [0, 1] range."""
        raw = jnp.linspace(0.0, 1000.0, H * W).reshape(H, W)
        result = preprocess_experimental(raw)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_nonnegative(self):
        """All output pixels are non-negative."""
        raw = jnp.ones((H, W)) * 50.0
        bg = jnp.ones((H, W)) * 100.0
        result = preprocess_experimental(raw, background=bg)
        self.assertTrue(jnp.all(result >= 0.0))


class TestPreprocessingGradients(chex.TestCase):
    """Gradient tests for preprocessing functions."""

    def test_grad_through_soft_mask(self):
        """jax.grad flows through soft_threshold_mask."""

        def loss(threshold):
            dist = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
            mask = soft_threshold_mask(dist, threshold, jnp.float64(10.0))
            return jnp.sum(mask)

        grad_val = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(jnp.abs(grad_val) > 1e-10)

    def test_grad_through_background_subtraction(self):
        """jax.grad flows through subtract_background."""

        def loss(bg_level):
            img = jnp.ones((H, W)) * 100.0
            bg = jnp.ones((H, W)) * bg_level
            result = subtract_background(img, bg)
            return jnp.sum(result)

        grad_val = jax.grad(loss)(jnp.float64(30.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_log_transform(self):
        """jax.grad flows through log_intensity_transform."""

        def loss(epsilon):
            img = jnp.ones((H, W)) * 100.0
            result = log_intensity_transform(img, epsilon)
            return jnp.sum(result)

        grad_val = jax.grad(loss)(jnp.float64(1e-4))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(jnp.abs(grad_val) > 1e-10)

    def test_grad_through_normalize(self):
        """jax.grad flows through normalize_image."""

        def loss(scale):
            img = jnp.linspace(0.0, 1.0, H * W).reshape(H, W) * scale
            result = normalize_image(img)
            return jnp.sum(result)

        grad_val = jax.grad(loss)(jnp.float64(100.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_full_preprocess(self):
        """jax.grad flows through preprocess_experimental without NaN."""

        def loss(bg_level):
            raw = jnp.linspace(10.0, 500.0, H * W).reshape(H, W)
            bg = jnp.ones((H, W)) * bg_level
            mask = jnp.ones((H, W)) * 0.9
            result = preprocess_experimental(
                raw,
                background=bg,
                beam_shadow_mask=mask,
                log_scale=True,
            )
            return jnp.sum(result)

        grad_val = jax.grad(loss)(jnp.float64(5.0))
        chex.assert_tree_all_finite(grad_val)
