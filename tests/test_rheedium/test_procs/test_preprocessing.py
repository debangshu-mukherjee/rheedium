"""Test suite for procs/preprocessing.py.

Verifies differentiable preprocessing pipeline for experimental RHEED
images: soft masking, background subtraction, log transform,
normalization, and the full pipeline. Includes gradient tests to
ensure jax.grad flows through all operations.
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from rheedium.procs.preprocessing import (
    log_intensity_transform,
    normalize_image,
    preprocess_experimental,
    soft_threshold_mask,
    subtract_background,
)
from rheedium.types.custom_types import scalar_float

H: int = 32
W: int = 32


class TestSoftThresholdMask(chex.TestCase):
    """Tests for soft_threshold_mask.

    :see: :func:`~rheedium.procs.soft_threshold_mask`
    """

    def test_shape_preserved(self) -> None:
        r"""Output shape matches distance field shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches distance field shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask: Bool[Array, "..."] = soft_threshold_mask(dist, jnp.float64(0.5))
        chex.assert_shape(mask, (H, W))

    def test_values_in_unit_interval(self) -> None:
        r"""All mask values are in (0, 1).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All mask values
        are in (0, 1).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.linspace(-1.0, 2.0, H * W).reshape(
            H, W
        )
        mask: Bool[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask > 0.0))
        self.assertTrue(jnp.all(mask < 1.0))

    def test_above_threshold_near_one(self) -> None:
        r"""Pixels well above threshold have mask near 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pixels well above
        threshold have mask near 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.ones((H, W)) * 10.0
        mask: Bool[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask > 0.99))

    def test_below_threshold_near_zero(self) -> None:
        r"""Pixels well below threshold have mask near 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pixels well below
        threshold have mask near 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.ones((H, W)) * (-10.0)
        mask: Bool[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        self.assertTrue(jnp.all(mask < 0.01))

    def test_at_threshold_is_half(self) -> None:
        r"""At the threshold, mask equals 0.5.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: At the threshold,
        mask equals 0.5.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.ones((H, W)) * 0.5
        mask: Bool[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(10.0)
        )
        chex.assert_trees_all_close(mask, jnp.full((H, W), 0.5), atol=1e-12)

    def test_higher_sharpness_steeper(self) -> None:
        r"""Higher sharpness produces steeper transition.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Higher sharpness
        produces steeper transition.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Float[Array, "..."] = jnp.linspace(0.0, 1.0, H * W).reshape(H, W)
        mask_soft: Float[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(1.0)
        )
        mask_sharp: Float[Array, "..."] = soft_threshold_mask(
            dist, jnp.float64(0.5), jnp.float64(100.0)
        )
        range_soft: scalar_float = jnp.max(mask_soft) - jnp.min(mask_soft)
        range_sharp: scalar_float = jnp.max(mask_sharp) - jnp.min(mask_sharp)
        self.assertTrue(range_sharp >= range_soft)


class TestSubtractBackground(chex.TestCase):
    """Tests for subtract_background.

    :see: :func:`~rheedium.procs.subtract_background`
    """

    def test_correct_subtraction(self) -> None:
        r"""Background is correctly subtracted.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Background is
        correctly subtracted.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W)) * 100.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 30.0
        result: Float[Array, "..."] = subtract_background(img, bg)
        chex.assert_trees_all_close(result, jnp.full((H, W), 70.0), atol=1e-12)

    def test_nonnegative_clipping(self) -> None:
        r"""Result is clipped to non-negative when background exceeds image.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Result is clipped
        to non-negative when background exceeds image.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W)) * 10.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 50.0
        result: Float[Array, "..."] = subtract_background(img, bg)
        self.assertTrue(jnp.all(result >= 0.0))
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_shape_preserved(self) -> None:
        r"""Output shape matches input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W))
        bg: Float[Array, "..."] = jnp.zeros((H, W))
        result: Float[Array, "..."] = subtract_background(img, bg)
        chex.assert_shape(result, (H, W))


class TestLogIntensityTransform(chex.TestCase):
    """Tests for log_intensity_transform.

    :see: :func:`~rheedium.procs.log_intensity_transform`
    """

    def test_zero_input_is_zero(self) -> None:
        r"""log(1 + 0/eps) = 0 for zero input.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: log(1 + 0/eps) = 0
        for zero input.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.zeros((H, W))
        result: Float[Array, "..."] = log_intensity_transform(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-12)

    def test_monotonically_increasing(self) -> None:
        r"""Transform preserves ordering of pixel intensities.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Transform
        preserves ordering of pixel intensities.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        vals: Float[Array, "..."] = jnp.array([0.0, 1.0, 100.0, 10000.0])
        img: Any = vals.reshape(2, 2)
        result: Float[Array, "..."] = log_intensity_transform(img)
        flat: Any = result.ravel()
        self.assertTrue(jnp.all(jnp.diff(flat) > 0.0))

    def test_compresses_dynamic_range(self) -> None:
        r"""Output range is smaller than input range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output range is
        smaller than input range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result: Float[Array, "..."] = log_intensity_transform(img)
        input_range: scalar_float = jnp.max(img) - jnp.min(img)
        output_range: scalar_float = jnp.max(result) - jnp.min(result)
        self.assertTrue(output_range < input_range)

    def test_shape_preserved(self) -> None:
        r"""Output shape matches input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W))
        result: Float[Array, "..."] = log_intensity_transform(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self) -> None:
        r"""No NaN or Inf in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W)) * 1e8
        result: Float[Array, "..."] = log_intensity_transform(img)
        chex.assert_tree_all_finite(result)


class TestNormalizeImage(chex.TestCase):
    """Tests for normalize_image.

    :see: :func:`~rheedium.procs.normalize_image`
    """

    def test_output_range(self) -> None:
        r"""Output min is 0 and max is 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output min is 0
        and max is 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.linspace(5.0, 500.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "..."] = normalize_image(img)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_uniform_image(self) -> None:
        r"""Uniform image normalizes to zero (no range).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Uniform image
        normalizes to zero (no range).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W)) * 42.0
        result: Float[Array, "..."] = normalize_image(img)
        chex.assert_trees_all_close(result, jnp.zeros((H, W)), atol=1e-6)

    def test_shape_preserved(self) -> None:
        r"""Output shape matches input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.ones((H, W))
        result: Float[Array, "..."] = normalize_image(img)
        chex.assert_shape(result, (H, W))

    def test_finite_values(self) -> None:
        r"""No NaN or Inf in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "..."] = jnp.linspace(0.0, 100.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "..."] = normalize_image(img)
        chex.assert_tree_all_finite(result)


class TestPreprocessExperimental(chex.TestCase):
    """Tests for the full preprocessing pipeline.

    :see: :func:`~rheedium.procs.preprocess_experimental`
    """

    def test_minimal_call(self) -> None:
        r"""Pipeline works with only raw_image (no optional args).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pipeline works
        with only raw_image (no optional args).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.ones((H, W)) * 100.0
        result: Float[Array, "..."] = preprocess_experimental(raw)
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_background(self) -> None:
        r"""Pipeline works with background subtraction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pipeline works
        with background subtraction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.ones((H, W)) * 100.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 30.0
        result: Float[Array, "..."] = preprocess_experimental(
            raw, background=bg
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_mask(self) -> None:
        r"""Pipeline works with soft mask.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pipeline works
        with soft mask.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.linspace(10.0, 500.0, H * W).reshape(
            H, W
        )
        mask: Bool[Array, "..."] = jnp.ones((H, W)) * 0.8
        result: Float[Array, "..."] = preprocess_experimental(
            raw, beam_shadow_mask=mask
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_with_log_scale(self) -> None:
        r"""Pipeline works with log scale enabled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pipeline works
        with log scale enabled.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.linspace(1.0, 1e6, H * W).reshape(H, W)
        result: Float[Array, "..."] = preprocess_experimental(
            raw, log_scale=True
        )
        chex.assert_shape(result, (H, W))
        chex.assert_tree_all_finite(result)

    def test_full_pipeline(self) -> None:
        r"""Pipeline works with all options enabled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pipeline works
        with all options enabled.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.linspace(10.0, 1e4, H * W).reshape(H, W)
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 5.0
        mask: Bool[Array, "..."] = jnp.ones((H, W)) * 0.9
        result: Float[Array, "..."] = preprocess_experimental(
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
        r"""Output is in [0, 1] range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output is in [0,
        1] range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.linspace(0.0, 1000.0, H * W).reshape(
            H, W
        )
        result: Float[Array, "..."] = preprocess_experimental(raw)
        chex.assert_trees_all_close(jnp.min(result), 0.0, atol=1e-12)
        chex.assert_trees_all_close(jnp.max(result), 1.0, atol=1e-12)

    def test_nonnegative(self) -> None:
        r"""All output pixels are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All output pixels
        are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        raw: Float[Array, "..."] = jnp.ones((H, W)) * 50.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 100.0
        result: Float[Array, "..."] = preprocess_experimental(
            raw, background=bg
        )
        self.assertTrue(jnp.all(result >= 0.0))


class TestPreprocessingGradients(chex.TestCase):
    """Gradient tests for preprocessing functions."""

    def test_grad_through_soft_mask(self) -> None:
        r"""jax.grad flows through soft_threshold_mask.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through soft_threshold_mask.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""jax.grad flows through subtract_background.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through subtract_background.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(bg_level: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
            bg: Float[Array, "H W"] = jnp.ones((H, W)) * bg_level
            result: Float[Array, "H W"] = subtract_background(img, bg)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(30.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_log_transform(self) -> None:
        r"""jax.grad flows through log_intensity_transform.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through log_intensity_transform.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(epsilon: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W)) * 100.0
            result: Float[Array, "H W"] = log_intensity_transform(img, epsilon)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(1e-4))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(jnp.abs(grad_val) > 1e-10)

    def test_grad_through_normalize(self) -> None:
        r"""jax.grad flows through normalize_image.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through normalize_image.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(scale: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = (
                jnp.linspace(0.0, 1.0, H * W).reshape(H, W) * scale
            )
            result: Float[Array, "H W"] = normalize_image(img)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(100.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_full_preprocess(self) -> None:
        r"""jax.grad flows through preprocess_experimental without NaN.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through preprocess_experimental without NaN.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_preprocessing``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
