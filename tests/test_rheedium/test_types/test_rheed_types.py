import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import tree_util
from jaxtyping import TypeCheckError

from rheedium.types.rheed_types import (
    RHEEDImage,
    RHEEDPattern,
    create_rheed_image,
    create_rheed_pattern,
)


class TestRHEEDPattern(chex.TestCase):
    """Comprehensive test suite for RHEEDPattern PyTree."""

    def setUp(self) -> None:
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_rheed_pattern_valid(self) -> None:
        """Test creation of valid RHEEDPattern instances."""
        n_reflections = 10
        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jax.random.normal(self.rng, (n_reflections, 3))
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        detector_points = jax.random.normal(self.rng, (n_reflections, 2)) * 100
        intensities = jax.random.uniform(
            self.rng, (n_reflections,), minval=0, maxval=1000
        )

        var_create_rheed_pattern = self.variant(create_rheed_pattern)
        pattern = var_create_rheed_pattern(
            g_indices, k_out, detector_points, intensities
        )

        chex.assert_shape(pattern.g_indices, (n_reflections,))
        chex.assert_shape(pattern.k_out, (n_reflections, 3))
        chex.assert_shape(pattern.detector_points, (n_reflections, 2))
        chex.assert_shape(pattern.intensities, (n_reflections,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_pattern_pytree(self) -> None:
        """Test PyTree registration and operations."""
        n_reflections = 5
        g_indices = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 3))
        detector_points = jnp.ones((n_reflections, 2)) * 10
        intensities = jnp.ones(n_reflections) * 100

        var_create_rheed_pattern = self.variant(create_rheed_pattern)
        pattern = var_create_rheed_pattern(
            g_indices, k_out, detector_points, intensities
        )

        flat, treedef = tree_util.tree_flatten(pattern)
        reconstructed = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(pattern, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("single_reflection", 1, 100),
        ("small_pattern", 10, 500),
        ("medium_pattern", 100, 1000),
        ("large_pattern", 1000, 10000),
    )
    def test_rheed_pattern_various_sizes(
        self, n_reflections: int, max_intensity: float
    ) -> None:
        """Test RHEEDPattern with various numbers of reflections."""
        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jax.random.normal(self.rng, (n_reflections, 3))
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        detector_points = jax.random.normal(self.rng, (n_reflections, 2)) * 100
        intensities = jax.random.uniform(
            self.rng, (n_reflections,), minval=0, maxval=max_intensity
        )

        var_create_rheed_pattern = self.variant(create_rheed_pattern)
        pattern = var_create_rheed_pattern(
            g_indices, k_out, detector_points, intensities
        )

        chex.assert_shape(pattern.g_indices, (n_reflections,))
        chex.assert_shape(pattern.k_out, (n_reflections, 3))
        chex.assert_shape(pattern.detector_points, (n_reflections, 2))
        chex.assert_shape(pattern.intensities, (n_reflections,))
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)
        chex.assert_trees_all_equal(
            jnp.all(jnp.linalg.norm(pattern.k_out, axis=1) > 0), True
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_pattern_jit_compilation(self) -> None:
        """Test JIT compilation of RHEEDPattern operations."""

        def create_and_process(
            g_indices: jnp.ndarray,
            k_out: jnp.ndarray,
            detector_points: jnp.ndarray,
            intensities: jnp.ndarray,
        ) -> jnp.ndarray:
            pattern = create_rheed_pattern(
                g_indices, k_out, detector_points, intensities
            )
            return jnp.sum(pattern.intensities) + jnp.sum(pattern.k_out)

        var_create_and_process = self.variant(create_and_process)

        n_reflections = 5
        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 3))
        detector_points = jnp.ones((n_reflections, 2))
        intensities = jnp.ones(n_reflections)

        result = var_create_and_process(
            g_indices, k_out, detector_points, intensities
        )
        expected = jnp.sum(intensities) + jnp.sum(k_out)
        chex.assert_trees_all_close(result, expected)

    def test_rheed_pattern_validation_errors(self) -> None:
        """Test that invalid inputs are properly handled during JIT compilation."""
        n_reflections = 5

        g_indices = jnp.arange(n_reflections + 1, dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 3))
        detector_points = jnp.ones((n_reflections, 2))
        intensities = jnp.ones(n_reflections)
        # Shape mismatch should raise ValueError
        with pytest.raises(ValueError, match=".*shape.*"):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 2))
        detector_points = jnp.ones((n_reflections, 2))
        intensities = jnp.ones(n_reflections)
        # jaxtyping catches type errors before internal validation
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jnp.zeros((n_reflections, 3))
        detector_points = jnp.ones((n_reflections, 2))
        intensities = jnp.ones(n_reflections)
        with pytest.raises(ValueError, match=".*non-zero.*"):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 3))
        detector_points = jnp.ones((n_reflections, 2))
        intensities = -jnp.ones(n_reflections)
        with pytest.raises(ValueError, match=".*non-negative.*"):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_pattern_vmap(self) -> None:
        """Test vmap operations over batches of RHEED patterns."""
        batch_size = 4
        n_reflections = 5

        g_indices_batch = jnp.tile(
            jnp.arange(n_reflections, dtype=jnp.int32), (batch_size, 1)
        )
        k_out_batch = jnp.ones((batch_size, n_reflections, 3))
        detector_points_batch = jnp.ones((batch_size, n_reflections, 2))
        intensities_batch = jnp.ones((batch_size, n_reflections))

        vmapped_create = jax.vmap(create_rheed_pattern)
        patterns = vmapped_create(
            g_indices_batch,
            k_out_batch,
            detector_points_batch,
            intensities_batch,
        )

        chex.assert_shape(patterns.g_indices, (batch_size, n_reflections))
        chex.assert_shape(patterns.k_out, (batch_size, n_reflections, 3))
        chex.assert_shape(
            patterns.detector_points, (batch_size, n_reflections, 2)
        )
        chex.assert_shape(patterns.intensities, (batch_size, n_reflections))


class TestRHEEDImage(chex.TestCase):
    """Comprehensive test suite for RHEEDImage PyTree."""

    def setUp(self) -> None:
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_rheed_image_valid(self) -> None:
        """Test creation of valid RHEEDImage instances."""
        height, width = 256, 512
        img_array = jax.random.uniform(
            self.rng, (height, width), minval=0, maxval=1000
        )
        incoming_angle = 2.0
        calibration = 0.01
        electron_wavelength = 0.037
        detector_distance = 1000.0

        var_create_rheed_image = self.variant(create_rheed_image)
        image = var_create_rheed_image(
            img_array,
            incoming_angle,
            calibration,
            electron_wavelength,
            detector_distance,
        )

        chex.assert_shape(image.img_array, (height, width))
        chex.assert_trees_all_equal(image.incoming_angle, incoming_angle)
        chex.assert_trees_all_equal(image.calibration, calibration)
        chex.assert_trees_all_equal(
            image.electron_wavelength, electron_wavelength
        )
        chex.assert_trees_all_equal(image.detector_distance, detector_distance)

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_image_pytree(self) -> None:
        """Test PyTree registration and operations."""
        img_array = jnp.ones((128, 256))
        incoming_angle = 1.5
        calibration = jnp.array([0.01, 0.015])
        electron_wavelength = 0.04
        detector_distance = 800.0

        var_create_rheed_image = self.variant(create_rheed_image)
        image = var_create_rheed_image(
            img_array,
            incoming_angle,
            calibration,
            electron_wavelength,
            detector_distance,
        )

        flat, treedef = tree_util.tree_flatten(image)
        reconstructed = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(image.img_array, reconstructed.img_array)
        chex.assert_trees_all_equal(
            image.incoming_angle, reconstructed.incoming_angle
        )
        chex.assert_trees_all_close(
            image.calibration, reconstructed.calibration
        )
        chex.assert_trees_all_equal(
            image.electron_wavelength, reconstructed.electron_wavelength
        )
        chex.assert_trees_all_equal(
            image.detector_distance, reconstructed.detector_distance
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_low_angle", 64, 64, 0.5, 0.025),
        ("medium_standard", 128, 256, 2.0, 0.037),
        ("large_square", 512, 512, 5.0, 0.05),
        ("wide_high_angle", 256, 1024, 10.0, 0.1),
    )
    def test_rheed_image_various_params(
        self, height: int, width: int, angle: float, wavelength: float
    ) -> None:
        """Test RHEEDImage with various image sizes and parameters."""
        img_array = jax.random.uniform(
            self.rng, (height, width), minval=0, maxval=1000
        )
        calibration = 0.01
        detector_distance = 1000.0

        var_create_rheed_image = self.variant(create_rheed_image)
        image = var_create_rheed_image(
            img_array, angle, calibration, wavelength, detector_distance
        )

        chex.assert_shape(image.img_array, (height, width))
        max_angle = 90
        chex.assert_trees_all_equal(
            (image.incoming_angle >= 0) & (image.incoming_angle <= max_angle),
            True,
        )
        # Scalar fields are validated in the create_rheed_image function
        chex.assert_trees_all_equal(jnp.all(image.img_array >= 0), True)
        chex.assert_trees_all_equal(
            jnp.all(jnp.isfinite(image.img_array)), True
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_image_calibration_types(self) -> None:
        """Test RHEEDImage with scalar and array calibration."""
        img_array = jnp.ones((128, 128))
        incoming_angle = 2.0
        electron_wavelength = 0.037
        detector_distance = 1000.0

        scalar_calibration = 0.01
        var_create_rheed_image = self.variant(create_rheed_image)
        image_scalar = var_create_rheed_image(
            img_array,
            incoming_angle,
            scalar_calibration,
            electron_wavelength,
            detector_distance,
        )
        chex.assert_trees_all_equal(
            jnp.isscalar(image_scalar.calibration)
            or image_scalar.calibration.ndim == 0,
            True,
        )

        array_calibration = jnp.array([0.01, 0.015])
        image_array = var_create_rheed_image(
            img_array,
            incoming_angle,
            array_calibration,
            electron_wavelength,
            detector_distance,
        )
        chex.assert_shape(image_array.calibration, (2,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_image_jit_compilation(self) -> None:
        """Test JIT compilation of RHEEDImage operations."""

        def create_and_process(
            img_array: jnp.ndarray,
            angle: float,
            calibration: float,
            wavelength: float,
            distance: float,
        ) -> jnp.ndarray:
            image = create_rheed_image(
                img_array, angle, calibration, wavelength, distance
            )
            return jnp.sum(image.img_array) * image.incoming_angle

        var_create_and_process = self.variant(create_and_process)

        img_array = jnp.ones((64, 64))
        angle = 2.0
        calibration = 0.01
        wavelength = 0.037
        distance = 1000.0

        result = var_create_and_process(
            img_array, angle, calibration, wavelength, distance
        )
        expected = jnp.sum(img_array) * angle
        chex.assert_trees_all_close(result, expected)

    def test_rheed_image_validation_errors(self) -> None:
        """Test that invalid inputs are properly handled during JIT compilation."""
        wrong_shape_img = jnp.ones((64,))
        # jaxtyping catches type errors before internal validation
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_image)(
                wrong_shape_img, 2.0, 0.01, 0.037, 1000.0
            )

        img_array = jnp.ones((64, 64))
        invalid_angle = 100.0
        with pytest.raises(ValueError, match=".*angle.*"):
            jax.jit(create_rheed_image)(
                img_array, invalid_angle, 0.01, 0.037, 1000.0
            )

        img_array = jnp.ones((64, 64))
        negative_wavelength = -0.037
        with pytest.raises(ValueError, match=".*wavelength.*"):
            jax.jit(create_rheed_image)(
                img_array, 2.0, 0.01, negative_wavelength, 1000.0
            )

        img_array = jnp.ones((64, 64))
        negative_calibration = -0.01
        with pytest.raises(ValueError, match=".*calibration.*"):
            jax.jit(create_rheed_image)(
                img_array, 2.0, negative_calibration, 0.037, 1000.0
            )

        img_array = -jnp.ones((64, 64))
        with pytest.raises(ValueError, match=".*non-negative.*"):
            jax.jit(create_rheed_image)(img_array, 2.0, 0.01, 0.037, 1000.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_image_tree_map(self) -> None:
        """Test that RHEEDImage works correctly with tree_map operations."""
        img_array = jnp.ones((128, 256))
        incoming_angle = 2.0
        calibration = 0.01
        electron_wavelength = 0.037
        detector_distance = 1000.0

        var_create_rheed_image = self.variant(create_rheed_image)
        image = var_create_rheed_image(
            img_array,
            incoming_angle,
            calibration,
            electron_wavelength,
            detector_distance,
        )

        def scale_intensities(x: jnp.ndarray) -> jnp.ndarray:
            if isinstance(x, jnp.ndarray) and x.shape == img_array.shape:
                return x * 2.0
            return x

        scaled_image = tree_util.tree_map(scale_intensities, image)
        chex.assert_trees_all_close(scaled_image.img_array, img_array * 2.0)
        chex.assert_trees_all_equal(
            scaled_image.incoming_angle, incoming_angle
        )


class TestRHEEDIntegration(chex.TestCase):
    """Test integrated operations with RHEED data structures."""

    def setUp(self) -> None:
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_combined_rheed_structures(self) -> None:
        """Test combining RHEEDPattern and RHEEDImage in a single structure."""
        n_reflections = 10
        pattern = create_rheed_pattern(
            jnp.arange(n_reflections, dtype=jnp.int32),
            jnp.ones((n_reflections, 3)),
            jnp.ones((n_reflections, 2)),
            jnp.ones(n_reflections),
        )

        image = create_rheed_image(
            jnp.ones((128, 256)), 2.0, 0.01, 0.037, 1000.0
        )

        # Use variant on a function that processes the combined structure
        def process_combined(p: RHEEDPattern, i: RHEEDImage) -> tuple:
            combined = {"pattern": p, "image": i}
            flat, treedef = tree_util.tree_flatten(combined)
            reconstructed = tree_util.tree_unflatten(treedef, flat)
            return (combined, reconstructed)

        var_process_combined = self.variant(process_combined)
        combined, reconstructed = var_process_combined(pattern, image)

        chex.assert_trees_all_close(
            combined["pattern"], reconstructed["pattern"]
        )
        chex.assert_trees_all_close(
            combined["image"].img_array, reconstructed["image"].img_array
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient_flow_rheed_pattern(self) -> None:
        """Test that gradients flow through RHEEDPattern correctly."""

        def loss_fn(intensities: jnp.ndarray) -> jnp.ndarray:
            pattern = create_rheed_pattern(
                jnp.array([0, 1, 2], dtype=jnp.int32),
                jnp.ones((3, 3)),
                jnp.ones((3, 2)),
                intensities,
            )
            return jnp.sum(pattern.intensities**2)

        var_grad_fn = self.variant(jax.grad(loss_fn))
        intensities = jnp.array([1.0, 2.0, 3.0])
        grads = var_grad_fn(intensities)

        expected_grads = 2 * intensities
        chex.assert_trees_all_close(grads, expected_grads)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient_flow_rheed_image(self) -> None:
        """Test that gradients flow through RHEEDImage correctly."""

        def loss_fn(img_array: jnp.ndarray) -> jnp.ndarray:
            image = create_rheed_image(img_array, 2.0, 0.01, 0.037, 1000.0)
            return jnp.sum(image.img_array**2)

        var_grad_fn = self.variant(jax.grad(loss_fn))
        img_array = jnp.ones((32, 32))
        grads = var_grad_fn(img_array)

        expected_grads = 2 * img_array
        chex.assert_trees_all_close(grads, expected_grads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
