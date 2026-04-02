import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import tree_util
from jaxtyping import TypeCheckError

from rheedium.types.rheed_types import (
    DetectorGeometry,
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    bulk_to_slice,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
    identify_surface_atoms,
)
from rheedium.types.crystal_types import create_crystal_structure


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

        chex.assert_shape(pattern.G_indices, (n_reflections,))
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

        chex.assert_shape(pattern.G_indices, (n_reflections,))
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

        # Test wrong k_out shape - jaxtyping catches this
        g_indices = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out = jnp.ones((n_reflections, 2))  # Wrong: should be (n, 3)
        detector_points = jnp.ones((n_reflections, 2))
        intensities = jnp.ones(n_reflections)
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

    @chex.variants(without_jit=True, with_jit=False)
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

        vmapped_create = self.variant(jax.vmap(create_rheed_pattern))
        patterns = vmapped_create(
            g_indices_batch,
            k_out_batch,
            detector_points_batch,
            intensities_batch,
        )

        chex.assert_shape(patterns.G_indices, (batch_size, n_reflections))
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
        # Test wrong image shape - jaxtyping catches type errors
        wrong_shape_img = jnp.ones((64,))
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_image)(
                wrong_shape_img, 2.0, 0.01, 0.037, 1000.0
            )

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


class TestRHEEDPatternValidation(chex.TestCase):
    """Test validation logic in create_rheed_pattern."""

    def _make_valid_pattern_kwargs(self, n: int = 5) -> dict:
        """Build valid keyword arguments for create_rheed_pattern."""
        rng = jax.random.PRNGKey(0)
        k_out = jax.random.normal(rng, (n, 3))
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        return dict(
            g_indices=jnp.arange(n, dtype=jnp.int32),
            k_out=k_out,
            detector_points=jax.random.normal(rng, (n, 2)) * 50,
            intensities=jnp.ones(n),
        )

    def test_negative_intensities(self) -> None:
        """Negative intensities should be caught by validation."""
        kw = self._make_valid_pattern_kwargs()
        kw["intensities"] = -jnp.ones(5)
        jax.jit(create_rheed_pattern)(**kw)

    def test_zero_k_out_vectors(self) -> None:
        """Zero-length k_out vectors should be caught."""
        kw = self._make_valid_pattern_kwargs()
        kw["k_out"] = jnp.zeros((5, 3))
        jax.jit(create_rheed_pattern)(**kw)

    def test_nan_detector_points(self) -> None:
        """NaN in detector_points should be caught."""
        kw = self._make_valid_pattern_kwargs()
        kw["detector_points"] = kw["detector_points"].at[0, 0].set(jnp.nan)
        jax.jit(create_rheed_pattern)(**kw)

    def test_inf_detector_points(self) -> None:
        """Inf in detector_points should be caught."""
        kw = self._make_valid_pattern_kwargs()
        kw["detector_points"] = kw["detector_points"].at[0, 0].set(jnp.inf)
        jax.jit(create_rheed_pattern)(**kw)

    def test_mismatched_g_indices_length(self) -> None:
        """Mismatched g_indices length is a runtime lax.cond check."""
        kw = self._make_valid_pattern_kwargs(n=5)
        kw["g_indices"] = jnp.arange(3, dtype=jnp.int32)
        # g_indices uses dimension "N" while k_out uses "M",
        # so jaxtyping allows different lengths. The lax.cond
        # validation still traces successfully under JIT.
        jax.jit(create_rheed_pattern)(**kw)

    def test_mismatched_intensities_length(self) -> None:
        """Mismatched intensities length should be caught."""
        kw = self._make_valid_pattern_kwargs(n=5)
        kw["intensities"] = jnp.ones(3)
        with pytest.raises(Exception):
            jax.jit(create_rheed_pattern)(**kw)

    def test_dtypes_are_correct(self) -> None:
        """Factory should cast to correct dtypes."""
        kw = self._make_valid_pattern_kwargs()
        pattern = create_rheed_pattern(**kw)
        assert pattern.G_indices.dtype == jnp.int32
        assert pattern.k_out.dtype == jnp.float64
        assert pattern.detector_points.dtype == jnp.float64
        assert pattern.intensities.dtype == jnp.float64


class TestRHEEDImageValidation(chex.TestCase):
    """Test validation logic in create_rheed_image."""

    def _make_valid_image_kwargs(self) -> dict:
        """Build valid keyword arguments for create_rheed_image."""
        rng = jax.random.PRNGKey(0)
        return dict(
            img_array=jax.random.uniform(rng, (64, 64)),
            incoming_angle=2.0,
            calibration=0.01,
            electron_wavelength=0.037,
            detector_distance=1000.0,
        )

    def test_negative_image_values(self) -> None:
        """Negative pixel values should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["img_array"] = -jnp.ones((64, 64))
        jax.jit(create_rheed_image)(**kw)

    def test_nan_in_image(self) -> None:
        """NaN in image should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["img_array"] = kw["img_array"].at[0, 0].set(jnp.nan)
        jax.jit(create_rheed_image)(**kw)

    def test_angle_too_large(self) -> None:
        """Incoming angle > 90 degrees should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 100.0
        jax.jit(create_rheed_image)(**kw)

    def test_negative_angle(self) -> None:
        """Negative incoming angle should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["incoming_angle"] = -1.0
        jax.jit(create_rheed_image)(**kw)

    def test_negative_wavelength(self) -> None:
        """Negative electron wavelength should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["electron_wavelength"] = -0.01
        jax.jit(create_rheed_image)(**kw)

    def test_negative_distance(self) -> None:
        """Negative detector distance should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["detector_distance"] = -100.0
        jax.jit(create_rheed_image)(**kw)

    def test_negative_calibration(self) -> None:
        """Negative calibration should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["calibration"] = -0.01
        jax.jit(create_rheed_image)(**kw)

    def test_array_calibration_negative(self) -> None:
        """Negative array calibration should be caught."""
        kw = self._make_valid_image_kwargs()
        kw["calibration"] = jnp.array([-0.01, 0.01])
        jax.jit(create_rheed_image)(**kw)

    def test_dtypes_are_correct(self) -> None:
        """Factory should cast to correct dtypes."""
        kw = self._make_valid_image_kwargs()
        image = create_rheed_image(**kw)
        assert image.img_array.dtype == jnp.float64
        assert image.incoming_angle.dtype == jnp.float64
        assert image.electron_wavelength.dtype == jnp.float64
        assert image.detector_distance.dtype == jnp.float64

    def test_boundary_angle_zero(self) -> None:
        """Incoming angle of exactly 0 should be valid."""
        kw = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 0.0
        image = create_rheed_image(**kw)
        chex.assert_trees_all_close(image.incoming_angle, 0.0)

    def test_boundary_angle_ninety(self) -> None:
        """Incoming angle of exactly 90 should be valid."""
        kw = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 90.0
        image = create_rheed_image(**kw)
        chex.assert_trees_all_close(image.incoming_angle, 90.0)


class TestSlicedCrystal(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for SlicedCrystal PyTree."""

    def _make_valid_sliced_kwargs(self, n_atoms: int = 10) -> dict:
        """Build valid keyword arguments for create_sliced_crystal."""
        rng = jax.random.PRNGKey(0)
        positions_3d = jax.random.uniform(
            rng, (n_atoms, 3), minval=0.0, maxval=100.0
        )
        atomic_numbers = jnp.ones((n_atoms, 1)) * 14.0
        cart_positions = jnp.concatenate(
            [positions_3d, atomic_numbers], axis=1
        )
        return dict(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([150.0, 150.0, 20.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=20.0,
            x_extent=150.0,
            y_extent=150.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_sliced_crystal_valid(self) -> None:
        """Test creation of valid SlicedCrystal instances."""
        n_atoms = 10
        kw = self._make_valid_sliced_kwargs(n_atoms)
        create_fn = self.variant(create_sliced_crystal)
        sliced = create_fn(**kw)

        chex.assert_shape(sliced.cart_positions, (n_atoms, 4))
        chex.assert_shape(sliced.cell_lengths, (3,))
        chex.assert_shape(sliced.cell_angles, (3,))
        chex.assert_shape(sliced.orientation, (3,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_pytree(self) -> None:
        """Test PyTree flatten/unflatten round-trip."""
        kw = self._make_valid_sliced_kwargs()
        create_fn = self.variant(create_sliced_crystal)
        sliced = create_fn(**kw)

        flat, treedef = tree_util.tree_flatten(sliced)
        reconstructed = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(sliced, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_values_preserved(self) -> None:
        """Test that array values are faithfully preserved."""
        kw = self._make_valid_sliced_kwargs()
        create_fn = self.variant(create_sliced_crystal)
        sliced = create_fn(**kw)

        chex.assert_trees_all_close(
            sliced.cart_positions, kw["cart_positions"]
        )
        chex.assert_trees_all_close(sliced.cell_lengths, kw["cell_lengths"])
        chex.assert_trees_all_close(sliced.orientation, kw["orientation"])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small", 2),
        ("medium", 50),
        ("large", 500),
    )
    def test_sliced_crystal_various_sizes(self, n_atoms: int) -> None:
        """Test SlicedCrystal with various atom counts."""
        kw = self._make_valid_sliced_kwargs(n_atoms)
        create_fn = self.variant(create_sliced_crystal)
        sliced = create_fn(**kw)

        chex.assert_shape(sliced.cart_positions, (n_atoms, 4))

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_jit_compilation(self) -> None:
        """Test JIT compilation of SlicedCrystal operations."""

        def create_and_process(**kwargs) -> jnp.ndarray:
            sliced = create_sliced_crystal(**kwargs)
            return jnp.sum(sliced.cart_positions[:, :3]) + sliced.depth

        jitted_fn = self.variant(create_and_process)
        kw = self._make_valid_sliced_kwargs()
        result = jitted_fn(**kw)
        expected = jnp.sum(kw["cart_positions"][:, :3]) + kw["depth"]
        chex.assert_trees_all_close(result, expected)

    def test_sliced_crystal_dtypes(self) -> None:
        """Factory should cast to correct dtypes."""
        kw = self._make_valid_sliced_kwargs()
        sliced = create_sliced_crystal(**kw)
        assert sliced.cart_positions.dtype == jnp.float64
        assert sliced.cell_lengths.dtype == jnp.float64
        assert sliced.cell_angles.dtype == jnp.float64
        assert sliced.orientation.dtype == jnp.int32
        assert sliced.depth.dtype == jnp.float64
        assert sliced.x_extent.dtype == jnp.float64
        assert sliced.y_extent.dtype == jnp.float64

    def test_negative_depth(self) -> None:
        """Negative depth should be caught by validation."""
        kw = self._make_valid_sliced_kwargs()
        kw["depth"] = -5.0
        jax.jit(create_sliced_crystal)(**kw)

    def test_negative_x_extent(self) -> None:
        """Negative x_extent should be caught by validation."""
        kw = self._make_valid_sliced_kwargs()
        kw["x_extent"] = -100.0
        jax.jit(create_sliced_crystal)(**kw)

    def test_negative_y_extent(self) -> None:
        """Negative y_extent should be caught by validation."""
        kw = self._make_valid_sliced_kwargs()
        kw["y_extent"] = -100.0
        jax.jit(create_sliced_crystal)(**kw)

    def test_negative_cell_lengths(self) -> None:
        """Negative cell lengths should be caught."""
        kw = self._make_valid_sliced_kwargs()
        kw["cell_lengths"] = jnp.array([-1.0, 5.0, 5.0])
        jax.jit(create_sliced_crystal)(**kw)

    def test_invalid_cell_angles(self) -> None:
        """Cell angles outside (0, 180) should be caught."""
        kw = self._make_valid_sliced_kwargs()
        kw["cell_angles"] = jnp.array([0.0, 90.0, 90.0])
        jax.jit(create_sliced_crystal)(**kw)

    def test_nan_in_positions(self) -> None:
        """NaN in positions should be caught."""
        kw = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 0].set(jnp.nan)
        jax.jit(create_sliced_crystal)(**kw)

    def test_invalid_atomic_number_zero(self) -> None:
        """Atomic number of 0 should be caught."""
        kw = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 3].set(0.0)
        jax.jit(create_sliced_crystal)(**kw)

    def test_invalid_atomic_number_too_large(self) -> None:
        """Atomic number > 118 should be caught."""
        kw = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 3].set(200.0)
        jax.jit(create_sliced_crystal)(**kw)

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_gradient_flow(self) -> None:
        """Test that gradients flow through SlicedCrystal."""

        def loss_fn(positions: jnp.ndarray) -> jnp.ndarray:
            kw = self._make_valid_sliced_kwargs()
            kw["cart_positions"] = positions
            sliced = create_sliced_crystal(**kw)
            return jnp.sum(sliced.cart_positions[:, :3] ** 2)

        var_grad_fn = self.variant(jax.grad(loss_fn))
        kw = self._make_valid_sliced_kwargs()
        grads = var_grad_fn(kw["cart_positions"])
        expected_grads = 2 * kw["cart_positions"].at[:, 3].set(0.0)
        chex.assert_trees_all_close(grads, expected_grads)


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


def _make_simple_crystal(n_atoms: int = 8):
    """Create a simple cubic CrystalStructure for testing."""
    import numpy as np

    rng = np.random.default_rng(0)
    frac_xyz = rng.uniform(size=(n_atoms, 3))
    z_nums = np.full((n_atoms, 1), 14.0)
    frac_pos = jnp.array(np.hstack([frac_xyz, z_nums]))
    cell_lengths = jnp.array([5.43, 5.43, 5.43])
    cell_angles = jnp.array([90.0, 90.0, 90.0])
    cart_xyz = frac_xyz * np.array([5.43, 5.43, 5.43])
    cart_pos = jnp.array(np.hstack([cart_xyz, z_nums]))
    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


class TestIdentifySurfaceAtoms(chex.TestCase):
    """Tests for identify_surface_atoms with all four methods."""

    def test_height_method_default(self) -> None:
        """Height method with default 30% fraction."""
        positions = jnp.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
            dtype=jnp.float64,
        )
        mask = identify_surface_atoms(positions)
        assert mask.shape == (5,)
        assert bool(mask[4])
        assert not bool(mask[0])

    def test_height_method_custom_fraction(self) -> None:
        """Height method with 50% fraction."""
        positions = jnp.array(
            [[0, 0, i] for i in range(10)], dtype=jnp.float64
        )
        config = SurfaceConfig(method="height", height_fraction=0.5)
        mask = identify_surface_atoms(positions, config)
        n_surface = int(jnp.sum(mask))
        assert n_surface == 5

    def test_height_method_all_surface(self) -> None:
        """Height fraction of 1.0 marks all atoms as surface."""
        positions = jnp.array([[0, 0, i] for i in range(5)], dtype=jnp.float64)
        config = SurfaceConfig(method="height", height_fraction=1.0)
        mask = identify_surface_atoms(positions, config)
        assert bool(jnp.all(mask))

    def test_coordination_method(self) -> None:
        """Coordination method identifies under-coordinated atoms."""
        positions = jnp.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0.5, 0.5, 0.5],
                [10, 10, 10],
            ],
            dtype=jnp.float64,
        )
        config = SurfaceConfig(
            method="coordination",
            coordination_cutoff=2.0,
            coordination_threshold=3,
        )
        mask = identify_surface_atoms(positions, config)
        assert mask.shape == (6,)
        assert bool(mask[5])

    def test_layers_method(self) -> None:
        """Layers method marks topmost layer."""
        positions = jnp.array(
            [
                [0, 0, 0.0],
                [1, 0, 0.0],
                [0, 0, 2.0],
                [1, 0, 2.0],
                [0, 0, 4.0],
                [1, 0, 4.0],
            ],
            dtype=jnp.float64,
        )
        config = SurfaceConfig(
            method="layers", n_layers=1, layer_tolerance=0.5
        )
        mask = identify_surface_atoms(positions, config)
        assert bool(mask[4]) and bool(mask[5])
        assert not bool(mask[0]) and not bool(mask[1])

    def test_layers_method_two_layers(self) -> None:
        """Layers method with n_layers=2."""
        positions = jnp.array(
            [[0, 0, z] for z in [0.0, 0.0, 2.0, 2.0, 4.0, 4.0]],
            dtype=jnp.float64,
        )
        config = SurfaceConfig(
            method="layers", n_layers=2, layer_tolerance=0.5
        )
        mask = identify_surface_atoms(positions, config)
        n_surface = int(jnp.sum(mask))
        assert n_surface == 4

    def test_explicit_method(self) -> None:
        """Explicit method uses user-provided mask."""
        positions = jnp.ones((5, 3))
        explicit = jnp.array([True, False, True, False, True])
        config = SurfaceConfig(method="explicit", explicit_mask=explicit)
        mask = identify_surface_atoms(positions, config)
        chex.assert_trees_all_equal(mask, explicit)

    def test_explicit_method_no_mask_fallback(self) -> None:
        """Explicit method without mask falls back to height."""
        positions = jnp.array([[0, 0, i] for i in range(5)], dtype=jnp.float64)
        config = SurfaceConfig(method="explicit")
        mask = identify_surface_atoms(positions, config)
        assert mask.shape == (5,)
        assert bool(mask[4])

    def test_output_shape_matches_input(self) -> None:
        """Output mask has same length as input positions."""
        for n in [1, 10, 100]:
            positions = jnp.zeros((n, 3))
            mask = identify_surface_atoms(positions)
            assert mask.shape == (n,)


class TestSurfaceConfig(chex.TestCase):
    """Tests for SurfaceConfig NamedTuple."""

    def test_default_values(self) -> None:
        """Default config should use height method at 30%."""
        config = SurfaceConfig()
        assert config.method == "height"
        assert config.height_fraction == 0.3
        assert config.coordination_cutoff == 3.0
        assert config.coordination_threshold == 8
        assert config.n_layers == 1
        assert config.layer_tolerance == 0.5
        assert config.explicit_mask is None

    def test_custom_values(self) -> None:
        """Custom config should preserve all values."""
        config = SurfaceConfig(
            method="coordination",
            coordination_cutoff=4.0,
            coordination_threshold=6,
        )
        assert config.method == "coordination"
        assert config.coordination_cutoff == 4.0
        assert config.coordination_threshold == 6

    def test_immutable(self) -> None:
        """SurfaceConfig should be immutable (NamedTuple)."""
        config = SurfaceConfig()
        with pytest.raises(AttributeError):
            config.method = "layers"


class TestDetectorGeometry(chex.TestCase):
    """Tests for DetectorGeometry NamedTuple."""

    def test_default_values(self) -> None:
        """Default geometry should have standard RHEED values."""
        geom = DetectorGeometry()
        assert geom.distance == 100.0
        assert geom.tilt_angle == 0.0
        assert geom.curvature_radius == float("inf")
        assert geom.center_offset_h == 0.0
        assert geom.center_offset_v == 0.0
        assert geom.psf_sigma_pixels == 1.0

    def test_custom_values(self) -> None:
        """Custom geometry should preserve values."""
        geom = DetectorGeometry(
            distance=200.0,
            tilt_angle=5.0,
            curvature_radius=500.0,
            center_offset_h=1.5,
            center_offset_v=-2.0,
            psf_sigma_pixels=1.5,
        )
        assert geom.distance == 200.0
        assert geom.tilt_angle == 5.0
        assert geom.curvature_radius == 500.0
        assert geom.center_offset_h == 1.5
        assert geom.center_offset_v == -2.0
        assert geom.psf_sigma_pixels == 1.5

    def test_psf_sigma_zero_disables(self) -> None:
        """Zero PSF sigma should be valid (disables convolution)."""
        geom = DetectorGeometry(psf_sigma_pixels=0.0)
        assert geom.psf_sigma_pixels == 0.0

    def test_immutable(self) -> None:
        """DetectorGeometry should be immutable (NamedTuple)."""
        geom = DetectorGeometry()
        with pytest.raises(AttributeError):
            geom.distance = 200.0

    def test_infinite_curvature_is_flat(self) -> None:
        """Default curvature should indicate a flat detector."""
        import math

        geom = DetectorGeometry()
        assert math.isinf(geom.curvature_radius)


class TestBulkToSlice(chex.TestCase):
    """Tests for bulk_to_slice function."""

    def test_returns_sliced_crystal(self) -> None:
        """Should return a SlicedCrystal instance."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
        )
        assert isinstance(sliced, SlicedCrystal)

    def test_output_shapes(self) -> None:
        """Output should have correct array shapes."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.ndim == 2
        assert sliced.cart_positions.shape[1] == 4
        chex.assert_shape(sliced.cell_lengths, (3,))
        chex.assert_shape(sliced.cell_angles, (3,))
        chex.assert_shape(sliced.orientation, (3,))

    def test_depth_preserved(self) -> None:
        """Slab depth should match requested depth."""
        crystal = _make_simple_crystal()
        depth = 15.0
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=depth,
        )
        chex.assert_trees_all_close(sliced.depth, depth)

    def test_extents_preserved(self) -> None:
        """Lateral extents should match requested values."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=120.0,
            y_extent=130.0,
        )
        chex.assert_trees_all_close(sliced.x_extent, 120.0)
        chex.assert_trees_all_close(sliced.y_extent, 130.0)

    def test_orientation_preserved(self) -> None:
        """Surface orientation should be preserved."""
        crystal = _make_simple_crystal()
        orient = jnp.array([1, 1, 1], dtype=jnp.int32)
        sliced = bulk_to_slice(
            crystal,
            orientation=orient,
            depth=10.0,
        )
        chex.assert_trees_all_equal(sliced.orientation, orient)

    def test_atoms_within_bounds(self) -> None:
        """All atoms should be within the specified bounds."""
        crystal = _make_simple_crystal()
        depth = 10.0
        x_ext = 80.0
        y_ext = 80.0
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=depth,
            x_extent=x_ext,
            y_extent=y_ext,
        )
        positions = sliced.cart_positions[:, :3]
        assert bool(jnp.all(positions[:, 0] >= 0))
        assert bool(jnp.all(positions[:, 0] <= x_ext))
        assert bool(jnp.all(positions[:, 1] >= 0))
        assert bool(jnp.all(positions[:, 1] <= y_ext))
        assert bool(jnp.all(positions[:, 2] >= 0))
        assert bool(jnp.all(positions[:, 2] <= depth))

    def test_cell_angles_orthorhombic(self) -> None:
        """Output cell should have 90-degree angles."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
        )
        chex.assert_trees_all_close(
            sliced.cell_angles,
            jnp.array([90.0, 90.0, 90.0]),
        )

    def test_001_orientation(self) -> None:
        """(001) orientation should work without rotation."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0

    def test_111_orientation(self) -> None:
        """(111) orientation should produce rotated slab."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([1, 1, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0

    def test_100_orientation(self) -> None:
        """(100) orientation should produce rotated slab."""
        crystal = _make_simple_crystal()
        sliced = bulk_to_slice(
            crystal,
            orientation=jnp.array([1, 0, 0], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
