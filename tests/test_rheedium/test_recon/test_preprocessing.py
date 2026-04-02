"""Test suite for recon/preprocessing.py.

Verifies differentiable preprocessing pipeline for experimental RHEED
images: soft masking, background subtraction, log transform,
normalization, and the full pipeline. Includes gradient tests to
ensure jax.grad flows through all operations.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.recon.preprocessing import (
    log_intensity_transform,
    normalize_image,
    preprocess_experimental,
    soft_threshold_mask,
    subtract_background,
)
from rheedium.types import scalar_float

H: int = 32
W: int = 32


class TestSoftThresholdMask(chex.TestCase):
    """Tests for soft_threshold_mask."""

    def test_shape_preserved(self) -> None:
        """Output shape matches distance field shape."""
        dist: Float[Array, "H W"] = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask: Float[Array, "H W"] = soft_threshold_mask(dist, jnp.float64(0.5))
        chex.assert_shape(mask, (H, W))

    def test_values_in_unit_interval(self) -> None:
        """All mask values are in (0, 1)."""
        dist: Float[Array, "H W"] = jnp.linspace(-1.0, 2.0, H * W).reshape(
            H, W
        )
        mask: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask > 0.0))
        self.assertTrue(jnp.all(mask < 1.0))

    def test_above_threshold_near_one(self) -> None:
        """Pixels well above threshold have mask near 1."""
        dist: Float[Array, "H W"] = jnp.ones((H, W)) * 10.0
        mask: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask > 0.99))

    def test_below_threshold_near_zero(self) -> None:
        """Pixels well below threshold have mask near 0."""
        dist: Float[Array, "H W"] = jnp.ones((H, W)) * (-10.0)
        mask: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask < 0.01))

    def test_at_threshold_is_half(self) -> None:
        """At the threshold, mask equals 0.5."""
        dist: Float[Array, "H W"] = jnp.ones((H, W)) * 0.5
        mask: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        chex.assert_trees_all_close(mask, jnp.full((H, W), 0.5), atol=1e-12)

    def test_higher_sharpness_steeper(self) -> None:
        """Higher sharpness produces steeper transition."""
        dist: Float[Array, "H W"] = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask_soft: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(1.0)
        )
        mask_sharp: Float[Array, "H W"] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(100.0)
        )
        range_soft: scalar_float = jnp.max(mask_soft) - jnp.min(mask_soft)
        range_sharp: scalar_float = jnp.max(mask_sharp) - jnp.min(mask_sharp)
        self.assertTrue(range_sharp >= range_soft)


class TestSubtractBackground(chex.TestCase):
    """Tests for subtract_background."""

    def test_correct_subtraction(self) -> None:
        """Background is correctly subtracted."""
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
        bg: Float[Array, "H W"] = jnp.ones((H, W)) * 30.0
        result: Float[Array, "H W"] = subtract_background(img, bg)
        chex.assert_trees_all_close(result, jnp.full((H, W), 70.0), atol=1e-12)

    def test_nonnegative_clipping(self) -> None:
        """Result is clipped to non-negative when background exceeds image."""
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 10.0
        bg: Float[Array, "H W"] = jnp.ones((H, W)) * 50.0
        result: Float[Array, "H W"] = subtract_background(img, bg)
        self.assertTrue(jnp.all(result >= 0.0))
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        img: Float[Array, "H W"] = jnp.ones((H, W))
        bg: Float[Array, "H W"] = jnp.zeros((H, W))
        result: Float[Array, "H W"] = subtract_background(img, bg)
        chex.assert_shape(result, (H, W))


class TestLogIntensityTransform(chex.TestCase):
    """Tests for log_intensity_transform."""

    def test_zero_input_is_zero(self) -> None:
        """log(1 + 0/eps) = 0 for zero input."""
        img: Float[Array, "H W"] = jnp.zeros((H, W))
        result: Float[Array, "H W"] = log_intensity_transform(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_monotonically_increasing(self) -> None:
        """Transform preserves ordering of pixel intensities."""
        vals: Float[Array, " 4"] = jnp.array([0.0, 1.0, 100.0, 10000.0])
        img: Float[Array, "2 2"] = vals.reshape(2, 2)
        result: Float[Array, "2 2"] = log_intensity_transform(img)
        flat: Float[Array, " 4"] = result.ravel()
        self.assertTrue(jnp.all(jnp.diff(flat) > 0.0))

    def test_compresses_dynamic_range(self) -> None:
        """Output range is smaller than input range."""
        img: Float[Array, "H W"] = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result: Float[Array, "H W"] = log_intensity_transform(img)
        input_range: scalar_float = jnp.max(img) - jnp.min(img)
        output_range: scalar_float = jnp.max(result) - jnp.min(result)
        self.assertTrue(output_range < input_range)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        img: Float[Array, "H W"] = jnp.ones((H, W))
        result: Float[Array, "H W"] = log_intensity_transform(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 1e8
        result: Float[Array, "H W"] = log_intensity_transform(img)
        chex.assert_tree_all_finite(result)


class TestNormalizeImage(chex.TestCase):
    """Tests for normalize_image."""

    def test_output_range(self) -> None:
        """Output min is 0 and max is 1."""
        img: Float[Array, "H W"] = jnp.linspace(5.0, 500.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "H W"] = normalize_image(img)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_uniform_image(self) -> None:
        """Uniform image normalizes to zero (no range)."""
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 42.0
        result: Float[Array, "H W"] = normalize_image(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-6)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        img: Float[Array, "H W"] = jnp.ones((H, W))
        result: Float[Array, "H W"] = normalize_image(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        img: Float[Array, "H W"] = jnp.linspace(0.0, 100.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "H W"] = normalize_image(img)
        chex.assert_tree_all_finite(result)


class TestPreprocessExperimental(chex.TestCase):
    """Tests for the full preprocessing pipeline."""

    def test_minimal_call(self) -> None:
        """Pipeline works with only raw_image (no optional args)."""
        raw: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
        result: Float[Array, "H W"] = preprocess_experimental(raw)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_background(self) -> None:
        """Pipeline works with background subtraction."""
        raw: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
        bg: Float[Array, "H W"] = jnp.ones((H, W)) * 30.0
        result: Float[Array, "H W"] = preprocess_experimental(
            raw, background=bg
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_mask(self) -> None:
        """Pipeline works with soft mask."""
        raw: Float[Array, "H W"] = jnp.linspace(10.0, 500.0, H * W).reshape(
            H, W
        )
        mask: Float[Array, "H W"] = jnp.ones((H, W)) * 0.8
        result: Float[Array, "H W"] = preprocess_experimental(
            raw, beam_shadow_mask=mask
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_log_scale(self) -> None:
        """Pipeline works with log scale enabled."""
        raw: Float[Array, "H W"] = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result: Float[Array, "H W"] = preprocess_experimental(
            raw, log_scale=True
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_full_pipeline(self) -> None:
        """Pipeline works with all options enabled."""
        raw: Float[Array, "H W"] = jnp.linspace(10.0, 1e4, H * W).reshape(H, W)
        bg: Float[Array, "H W"] = jnp.ones((H, W)) * 5.0
        mask: Float[Array, "H W"] = jnp.ones((H, W)) * 0.9
        result: Float[Array, "H W"] = preprocess_experimental(
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

    def test_output_normalized(self) -> None:
        """Output is in [0, 1] range."""
        raw: Float[Array, "H W"] = jnp.linspace(0.0, 1000.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "H W"] = preprocess_experimental(raw)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_nonnegative(self) -> None:
        """All output pixels are non-negative."""
        raw: Float[Array, "H W"] = jnp.ones((H, W)) * 50.0
        bg: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
        result: Float[Array, "H W"] = preprocess_experimental(
            raw, background=bg
        )
        self.assertTrue(jnp.all(result >= 0.0))


class TestPreprocessingGradients(chex.TestCase):
    """Gradient tests for preprocessing functions."""

    def test_grad_through_soft_mask(self) -> None:
        """jax.grad flows through soft_threshold_mask."""

        def loss(threshold: scalar_float) -> scalar_float:
            dist: Float[Array, "H W"] = jnp.linspace(0.0, 1.0, H * W).reshape(
                H, W
            )
            mask: Float[Array, "H W"] = soft_threshold_mask(
                dist, threshold, jnp.float64(10.0)
            )
            return jnp.sum(mask)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(jnp.abs(grad_val) > 1e-10)

    def test_grad_through_background_subtraction(self) -> None:
        """jax.grad flows through subtract_background."""

        def loss(bg_level: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
            bg: Float[Array, "H W"] = jnp.ones((H, W)) * bg_level
            result: Float[Array, "H W"] = subtract_background(img, bg)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(30.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_log_transform(self) -> None:
        """jax.grad flows through log_intensity_transform."""

        def loss(epsilon: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
            result: Float[Array, "H W"] = log_intensity_transform(img, epsilon)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(1e-4))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(jnp.abs(grad_val) > 1e-10)

    def test_grad_through_normalize(self) -> None:
        """jax.grad flows through normalize_image."""

        def loss(scale: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = (
                jnp.linspace(0.0, 1.0, H * W).reshape(H, W) * scale
            )
            result: Float[Array, "H W"] = normalize_image(img)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(100.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_full_preprocess(self) -> None:
        """jax.grad flows through preprocess_experimental without NaN."""

        def loss(bg_level: scalar_float) -> scalar_float:
            raw: Float[Array, "H W"] = jnp.linspace(
                10.0, 500.0, H * W
            ).reshape(H, W)
            bg: Float[Array, "H W"] = jnp.ones((H, W)) * bg_level
            mask: Float[Array, "H W"] = jnp.ones((H, W)) * 0.9
            result: Float[Array, "H W"] = preprocess_experimental(
                raw,
                background=bg,
                beam_shadow_mask=mask,
                log_scale=True,
            )
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(5.0))
        chex.assert_tree_all_finite(grad_val)
