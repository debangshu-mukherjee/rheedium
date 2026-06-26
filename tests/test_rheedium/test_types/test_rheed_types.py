"""Test suite for rheedium.types.rheed_types PyTrees."""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import tree_util
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    Integer,
    Num,
    PRNGKeyArray,
    TypeCheckError,
)

from rheedium.types.custom_types import scalar_float
from rheedium.types.rheed_types import (
    DetectorGeometry,
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
    identify_surface_atoms,
)

from ..._assertions import assert_rejects


class TestRHEEDPattern(chex.TestCase):
    """Comprehensive test suite for RHEEDPattern PyTree.

    :see: :class:`~rheedium.types.RHEEDPattern`
    :see: :func:`~rheedium.types.create_rheed_pattern`
    """

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_rheed_pattern_valid(self) -> None:
        """Test creation of valid RHEEDPattern instances."""
        n_reflections: int = 10
        g_indices: Integer[Array, "..."] = jnp.arange(
            n_reflections, dtype=jnp.int32
        )
        k_out: Float[Array, "reflections coords"] = jax.random.normal(
            self.rng, (n_reflections, 3)
        )
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        detector_points: Float[Array, "reflections detector_coords"] = (
            jax.random.normal(self.rng, (n_reflections, 2)) * 100
        )
        intensities: Float[Array, "reflections"] = jax.random.uniform(
            self.rng, (n_reflections,), minval=0, maxval=1000
        )

        var_create_rheed_pattern: Callable[..., Any] = self.variant(
            create_rheed_pattern
        )
        pattern: RHEEDPattern = var_create_rheed_pattern(
            g_indices, k_out, detector_points, intensities
        )

        chex.assert_shape(pattern.G_indices, (n_reflections,))
        chex.assert_shape(pattern.k_out, (n_reflections, 3))
        chex.assert_shape(pattern.detector_points, (n_reflections, 2))
        chex.assert_shape(pattern.intensities, (n_reflections,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_pattern_pytree(self) -> None:
        """Test PyTree registration and operations."""
        n_reflections: int = 5
        g_indices: Integer[Array, "..."] = jnp.array(
            [0, 1, 2, 3, 4], dtype=jnp.int32
        )
        k_out: Float[Array, "..."] = jnp.ones((n_reflections, 3))
        detector_points: Float[Array, "..."] = (
            jnp.ones((n_reflections, 2)) * 10
        )
        intensities: Float[Array, "..."] = jnp.ones(n_reflections) * 100

        var_create_rheed_pattern: Callable[..., Any] = self.variant(
            create_rheed_pattern
        )
        pattern: RHEEDPattern = var_create_rheed_pattern(
            g_indices, k_out, detector_points, intensities
        )

        flat: Any
        treedef: Any
        flat, treedef = tree_util.tree_flatten(pattern)
        reconstructed: Any = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(pattern, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("single_reflection", 1, 100),
        ("small_pattern", 10, 500),
        ("medium_pattern", 100, 1000),
        ("large_pattern", 1000, 10000),
    )
    def test_rheed_pattern_various_sizes(
        self, n_reflections: int, max_intensity: int
    ) -> None:
        """Test RHEEDPattern with various numbers of reflections."""
        g_indices: Integer[Array, "..."] = jnp.arange(
            n_reflections, dtype=jnp.int32
        )
        k_out: Float[Array, "reflections coords"] = jax.random.normal(
            self.rng, (n_reflections, 3)
        )
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        detector_points: Float[Array, "reflections detector_coords"] = (
            jax.random.normal(self.rng, (n_reflections, 2)) * 100
        )
        intensities: Float[Array, "reflections"] = jax.random.uniform(
            self.rng, (n_reflections,), minval=0, maxval=max_intensity
        )

        var_create_rheed_pattern: Callable[..., Any] = self.variant(
            create_rheed_pattern
        )
        pattern: RHEEDPattern = var_create_rheed_pattern(
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
            g_indices: Int[Array, "N"],
            k_out: Float[Array, "N 3"],
            detector_points: Float[Array, "N 2"],
            intensities: Float[Array, "N"],
        ) -> Num[Array, ""]:
            pattern: RHEEDPattern = create_rheed_pattern(
                g_indices, k_out, detector_points, intensities
            )
            return jnp.sum(pattern.intensities) + jnp.sum(pattern.k_out)

        var_create_and_process: Callable[..., Any] = self.variant(
            create_and_process
        )

        n_reflections: int = 5
        g_indices: Int[Array, "N"] = jnp.arange(n_reflections, dtype=jnp.int32)
        k_out: Float[Array, "N 3"] = jnp.ones((n_reflections, 3))
        detector_points: Float[Array, "N 2"] = jnp.ones((n_reflections, 2))
        intensities: Float[Array, "N"] = jnp.ones(n_reflections)

        result: Float[Array, "..."] = var_create_and_process(
            g_indices, k_out, detector_points, intensities
        )
        expected: scalar_float = jnp.sum(intensities) + jnp.sum(k_out)
        chex.assert_trees_all_close(result, expected)

    def test_rheed_pattern_validation_errors(self) -> None:
        """Test invalid inputs are handled during JIT compilation."""
        n_reflections: int = 5

        # Test wrong k_out shape - jaxtyping catches this
        g_indices: Integer[Array, "..."] = jnp.arange(
            n_reflections, dtype=jnp.int32
        )
        k_out: Float[Array, "..."] = jnp.ones(
            (n_reflections, 2)
        )  # Wrong: should be (n, 3)
        detector_points: Float[Array, "..."] = jnp.ones((n_reflections, 2))
        intensities: Float[Array, "..."] = jnp.ones(n_reflections)
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_pattern)(
                g_indices, k_out, detector_points, intensities
            )

    @chex.variants(without_jit=True, with_jit=False)
    def test_rheed_pattern_vmap(self) -> None:
        """Test vmap operations over batches of RHEED patterns."""
        batch_size: int = 4
        n_reflections: int = 5

        g_indices_batch: Float[Array, "..."] = jnp.tile(
            jnp.arange(n_reflections, dtype=jnp.int32), (batch_size, 1)
        )
        k_out_batch: Float[Array, "..."] = jnp.ones(
            (batch_size, n_reflections, 3)
        )
        detector_points_batch: Float[Array, "..."] = jnp.ones(
            (batch_size, n_reflections, 2)
        )
        intensities_batch: Float[Array, "..."] = jnp.ones(
            (batch_size, n_reflections)
        )

        vmapped_create: Callable[..., Any] = self.variant(
            jax.vmap(create_rheed_pattern)
        )
        patterns: Float[Array, "..."] = vmapped_create(
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
    """Comprehensive test suite for RHEEDImage PyTree.

    :see: :class:`~rheedium.types.RHEEDImage`
    :see: :func:`~rheedium.types.create_rheed_image`
    """

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_rheed_image_valid(self) -> None:
        """Test creation of valid RHEEDImage instances."""
        height: tuple[Any, ...]
        width: tuple[Any, ...]
        height, width = 256, 512
        img_array: Float[Array, "height width"] = jax.random.uniform(
            self.rng, (height, width), minval=0, maxval=1000
        )
        incoming_angle: float = 2.0
        calibration: float = 0.01
        electron_wavelength: float = 0.037
        detector_distance: float = 1000.0

        var_create_rheed_image: Callable[..., Any] = self.variant(
            create_rheed_image
        )
        image: RHEEDImage = var_create_rheed_image(
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
        img_array: Float[Array, "..."] = jnp.ones((128, 256))
        incoming_angle: float = 1.5
        calibration: Float[Array, "..."] = jnp.array([0.01, 0.015])
        electron_wavelength: float = 0.04
        detector_distance: float = 800.0

        var_create_rheed_image: Callable[..., Any] = self.variant(
            create_rheed_image
        )
        image: RHEEDImage = var_create_rheed_image(
            img_array,
            incoming_angle,
            calibration,
            electron_wavelength,
            detector_distance,
        )

        flat: Any
        treedef: Any
        flat, treedef = tree_util.tree_flatten(image)
        reconstructed: Any = tree_util.tree_unflatten(treedef, flat)

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
        img_array: Float[Array, "height width"] = jax.random.uniform(
            self.rng, (height, width), minval=0, maxval=1000
        )
        calibration: float = 0.01
        detector_distance: float = 1000.0

        var_create_rheed_image: Callable[..., Any] = self.variant(
            create_rheed_image
        )
        image: RHEEDImage = var_create_rheed_image(
            img_array, angle, calibration, wavelength, detector_distance
        )

        chex.assert_shape(image.img_array, (height, width))
        max_angle: int = 90
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
        img_array: Float[Array, "..."] = jnp.ones((128, 128))
        incoming_angle: float = 2.0
        electron_wavelength: float = 0.037
        detector_distance: float = 1000.0

        scalar_calibration: float = 0.01
        var_create_rheed_image: Callable[..., Any] = self.variant(
            create_rheed_image
        )
        image_scalar: RHEEDImage = var_create_rheed_image(
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

        array_calibration: Float[Array, "..."] = jnp.array([0.01, 0.015])
        image_array: RHEEDImage = var_create_rheed_image(
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
            img_array: Float[Array, "H W"],
            angle: float,
            calibration: float,
            wavelength: float,
            distance: float,
        ) -> Num[Array, ""]:
            image: RHEEDImage = create_rheed_image(
                img_array, angle, calibration, wavelength, distance
            )
            return jnp.sum(image.img_array) * image.incoming_angle

        var_create_and_process: Callable[..., Any] = self.variant(
            create_and_process
        )

        img_array: Float[Array, "64 64"] = jnp.ones((64, 64))
        angle: float = 2.0
        calibration: float = 0.01
        wavelength: float = 0.037
        distance: float = 1000.0

        result: Float[Array, "..."] = var_create_and_process(
            img_array, angle, calibration, wavelength, distance
        )
        expected: Num[Array, ""] = jnp.sum(img_array) * angle
        chex.assert_trees_all_close(result, expected)

    def test_rheed_image_validation_errors(self) -> None:
        """Test invalid inputs are handled during JIT compilation."""
        # Test wrong image shape - jaxtyping catches type errors
        wrong_shape_img: Float[Array, "..."] = jnp.ones((64,))
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_image)(
                wrong_shape_img, 2.0, 0.01, 0.037, 1000.0
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rheed_image_tree_map(self) -> None:
        """Test that RHEEDImage works correctly with tree_map operations."""
        img_array: Float[Array, "..."] = jnp.ones((128, 256))
        incoming_angle: float = 2.0
        calibration: float = 0.01
        electron_wavelength: float = 0.037
        detector_distance: float = 1000.0

        var_create_rheed_image: Callable[..., Any] = self.variant(
            create_rheed_image
        )
        image: RHEEDImage = var_create_rheed_image(
            img_array,
            incoming_angle,
            calibration,
            electron_wavelength,
            detector_distance,
        )

        def scale_intensities(x: Num[Array, "..."]) -> Num[Array, "..."]:
            if isinstance(x, jnp.ndarray) and x.shape == img_array.shape:
                return x * 2.0
            return x

        scaled_image: Float[Array, "..."] = tree_util.tree_map(
            scale_intensities, image
        )
        chex.assert_trees_all_close(scaled_image.img_array, img_array * 2.0)
        chex.assert_trees_all_equal(
            scaled_image.incoming_angle, incoming_angle
        )


class TestRHEEDPatternValidation(chex.TestCase):
    """Test validation logic in create_rheed_pattern."""

    def _make_valid_pattern_kwargs(
        self, n: int = 5
    ) -> dict[str, Num[Array, "..."]]:
        """Build valid keyword arguments for create_rheed_pattern."""
        rng: PRNGKeyArray = jax.random.PRNGKey(0)
        k_out: Float[Array, "N 3"] = jax.random.normal(rng, (n, 3))
        k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
        return {
            "g_indices": jnp.arange(n, dtype=jnp.int32),
            "k_out": k_out,
            "detector_points": jax.random.normal(rng, (n, 2)) * 50,
            "intensities": jnp.ones(n),
        }

    def test_negative_intensities(self) -> None:
        """Negative intensities should be caught by validation."""
        kw: Any = self._make_valid_pattern_kwargs()
        kw["intensities"] = -jnp.ones(5)
        assert_rejects(
            create_rheed_pattern,
            match="intensities must be non-negative",
            **kw,
        )

    def test_zero_k_out_vectors(self) -> None:
        """Zero-length k_out vectors should be caught."""
        kw: Any = self._make_valid_pattern_kwargs()
        kw["k_out"] = jnp.zeros((5, 3))
        assert_rejects(
            create_rheed_pattern,
            match="k_out vectors must be non-zero",
            **kw,
        )

    def test_nan_detector_points(self) -> None:
        """NaN in detector_points should be caught."""
        kw: Any = self._make_valid_pattern_kwargs()
        kw["detector_points"] = kw["detector_points"].at[0, 0].set(jnp.nan)
        assert_rejects(
            create_rheed_pattern,
            match="detector_points contain non-finite values",
            **kw,
        )

    def test_inf_detector_points(self) -> None:
        """Inf in detector_points should be caught."""
        kw: Any = self._make_valid_pattern_kwargs()
        kw["detector_points"] = kw["detector_points"].at[0, 0].set(jnp.inf)
        assert_rejects(
            create_rheed_pattern,
            match="detector_points contain non-finite values",
            **kw,
        )

    def test_mismatched_g_indices_length(self) -> None:
        """Mismatched g_indices length should be caught."""
        kw: Any = self._make_valid_pattern_kwargs(n=5)
        kw["g_indices"] = jnp.arange(3, dtype=jnp.int32)
        assert_rejects(
            create_rheed_pattern,
            match="g_indices length must match reflections",
            **kw,
        )

    def test_mismatched_intensities_length(self) -> None:
        """Mismatched intensities length should be caught."""
        kw: Any = self._make_valid_pattern_kwargs(n=5)
        kw["intensities"] = jnp.ones(3)
        with pytest.raises(TypeCheckError):
            jax.jit(create_rheed_pattern)(**kw)

    def test_dtypes_are_correct(self) -> None:
        """Factory should cast to correct dtypes."""
        kw: Any = self._make_valid_pattern_kwargs()
        pattern: RHEEDPattern = create_rheed_pattern(**kw)
        assert pattern.G_indices.dtype == jnp.int32
        assert pattern.k_out.dtype == jnp.float64
        assert pattern.detector_points.dtype == jnp.float64
        assert pattern.intensities.dtype == jnp.float64


class TestRHEEDImageValidation(chex.TestCase):
    """Test validation logic in create_rheed_image."""

    def _make_valid_image_kwargs(self) -> dict[str, object]:
        """Build valid keyword arguments for create_rheed_image."""
        rng: PRNGKeyArray = jax.random.PRNGKey(0)
        return {
            "img_array": jax.random.uniform(rng, (64, 64)),
            "incoming_angle": 2.0,
            "calibration": 0.01,
            "electron_wavelength": 0.037,
            "detector_distance": 1000.0,
        }

    def test_negative_image_values(self) -> None:
        """Negative pixel values should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["img_array"] = -jnp.ones((64, 64))
        assert_rejects(
            create_rheed_image,
            match="img_array must be non-negative",
            **kw,
        )

    def test_nan_in_image(self) -> None:
        """NaN in image should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["img_array"] = kw["img_array"].at[0, 0].set(jnp.nan)
        assert_rejects(
            create_rheed_image,
            match="img_array contains non-finite values",
            **kw,
        )

    def test_angle_too_large(self) -> None:
        """Incoming angle > 90 degrees should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 100.0
        assert_rejects(
            create_rheed_image,
            match="incoming_angle must be between 0 and 90 degrees",
            **kw,
        )

    def test_negative_angle(self) -> None:
        """Negative incoming angle should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["incoming_angle"] = -1.0
        assert_rejects(
            create_rheed_image,
            match="incoming_angle must be between 0 and 90 degrees",
            **kw,
        )

    def test_negative_wavelength(self) -> None:
        """Negative electron wavelength should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["electron_wavelength"] = -0.01
        assert_rejects(
            create_rheed_image,
            match="electron_wavelength must be positive",
            **kw,
        )

    def test_negative_distance(self) -> None:
        """Negative detector distance should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["detector_distance"] = -100.0
        assert_rejects(
            create_rheed_image,
            match="detector_distance must be positive",
            **kw,
        )

    def test_negative_calibration(self) -> None:
        """Negative calibration should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["calibration"] = -0.01
        assert_rejects(
            create_rheed_image,
            match="calibration must be positive",
            **kw,
        )

    def test_array_calibration_negative(self) -> None:
        """Negative array calibration should be caught."""
        kw: Any = self._make_valid_image_kwargs()
        kw["calibration"] = jnp.array([-0.01, 0.01])
        assert_rejects(
            create_rheed_image,
            match="calibration must be positive",
            **kw,
        )

    def test_dtypes_are_correct(self) -> None:
        """Factory should cast to correct dtypes."""
        kw: Any = self._make_valid_image_kwargs()
        image: RHEEDImage = create_rheed_image(**kw)
        assert image.img_array.dtype == jnp.float64
        assert image.incoming_angle.dtype == jnp.float64
        assert image.electron_wavelength.dtype == jnp.float64
        assert image.detector_distance.dtype == jnp.float64

    def test_boundary_angle_zero(self) -> None:
        """Incoming angle of exactly 0 should be valid."""
        kw: Any = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 0.0
        image: RHEEDImage = create_rheed_image(**kw)
        chex.assert_trees_all_close(image.incoming_angle, 0.0)

    def test_boundary_angle_ninety(self) -> None:
        """Incoming angle of exactly 90 should be valid."""
        kw: Any = self._make_valid_image_kwargs()
        kw["incoming_angle"] = 90.0
        image: RHEEDImage = create_rheed_image(**kw)
        chex.assert_trees_all_close(image.incoming_angle, 90.0)


class TestSlicedCrystal(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for SlicedCrystal PyTree.

    :see: :class:`~rheedium.types.SlicedCrystal`
    :see: :func:`~rheedium.types.create_sliced_crystal`
    """

    def _make_valid_sliced_kwargs(self, n_atoms: int = 10) -> dict[str, Any]:
        """Build valid keyword arguments for create_sliced_crystal."""
        rng: PRNGKeyArray = jax.random.PRNGKey(0)
        positions_3d: Float[Array, "atoms coords"] = jax.random.uniform(
            rng, (n_atoms, 3), minval=0.0, maxval=100.0
        )
        atomic_numbers: Float[Array, "..."] = jnp.ones((n_atoms, 1)) * 14.0
        cart_positions: Float[Array, "..."] = jnp.concatenate(
            [positions_3d, atomic_numbers], axis=1
        )
        return {
            "cart_positions": cart_positions,
            "cell_lengths": jnp.array([150.0, 150.0, 20.0]),
            "cell_angles": jnp.array([90.0, 90.0, 90.0]),
            "orientation": jnp.array([0, 0, 1], dtype=jnp.int32),
            "depth": 20.0,
            "x_extent": 150.0,
            "y_extent": 150.0,
        }

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_sliced_crystal_valid(self) -> None:
        """Test creation of valid SlicedCrystal instances."""
        n_atoms: int = 10
        kw: Any = self._make_valid_sliced_kwargs(n_atoms)
        create_fn: Callable[..., Any] = self.variant(create_sliced_crystal)
        sliced: SlicedCrystal = create_fn(**kw)

        chex.assert_shape(sliced.cart_positions, (n_atoms, 4))
        chex.assert_shape(sliced.cell_lengths, (3,))
        chex.assert_shape(sliced.cell_angles, (3,))
        chex.assert_shape(sliced.orientation, (3,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_pytree(self) -> None:
        """Test PyTree flatten/unflatten round-trip."""
        kw: Any = self._make_valid_sliced_kwargs()
        create_fn: Callable[..., Any] = self.variant(create_sliced_crystal)
        sliced: SlicedCrystal = create_fn(**kw)

        flat: Any
        treedef: Any
        flat, treedef = tree_util.tree_flatten(sliced)
        reconstructed: Any = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(sliced, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_values_preserved(self) -> None:
        """Test that array values are faithfully preserved."""
        kw: Any = self._make_valid_sliced_kwargs()
        create_fn: Callable[..., Any] = self.variant(create_sliced_crystal)
        sliced: SlicedCrystal = create_fn(**kw)

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
        kw: Any = self._make_valid_sliced_kwargs(n_atoms)
        create_fn: Callable[..., Any] = self.variant(create_sliced_crystal)
        sliced: SlicedCrystal = create_fn(**kw)

        chex.assert_shape(sliced.cart_positions, (n_atoms, 4))

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_jit_compilation(self) -> None:
        """Test JIT compilation of SlicedCrystal operations."""

        def create_and_process(**kwargs: object) -> Num[Array, ""]:
            sliced: SlicedCrystal = create_sliced_crystal(**kwargs)
            return jnp.sum(sliced.cart_positions[:, :3]) + sliced.depth

        jitted_fn: Callable[..., Any] = self.variant(create_and_process)
        kw: Any = self._make_valid_sliced_kwargs()
        result: Float[Array, "..."] = jitted_fn(**kw)
        expected: scalar_float = (
            jnp.sum(kw["cart_positions"][:, :3]) + kw["depth"]
        )
        chex.assert_trees_all_close(result, expected)

    def test_sliced_crystal_dtypes(self) -> None:
        """Factory should cast to correct dtypes."""
        kw: Any = self._make_valid_sliced_kwargs()
        sliced: SlicedCrystal = create_sliced_crystal(**kw)
        assert sliced.cart_positions.dtype == jnp.float64
        assert sliced.cell_lengths.dtype == jnp.float64
        assert sliced.cell_angles.dtype == jnp.float64
        assert sliced.orientation.dtype == jnp.int32
        assert sliced.depth.dtype == jnp.float64
        assert sliced.x_extent.dtype == jnp.float64
        assert sliced.y_extent.dtype == jnp.float64

    def test_negative_depth(self) -> None:
        """Negative depth should be caught by validation."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["depth"] = -5.0
        assert_rejects(
            create_sliced_crystal,
            match="depth must be positive",
            **kw,
        )

    def test_negative_x_extent(self) -> None:
        """Negative x_extent should be caught by validation."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["x_extent"] = -100.0
        assert_rejects(
            create_sliced_crystal,
            match="x_extent must be positive",
            **kw,
        )

    def test_negative_y_extent(self) -> None:
        """Negative y_extent should be caught by validation."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["y_extent"] = -100.0
        assert_rejects(
            create_sliced_crystal,
            match="y_extent must be positive",
            **kw,
        )

    def test_negative_cell_lengths(self) -> None:
        """Negative cell lengths should be caught."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["cell_lengths"] = jnp.array([-1.0, 5.0, 5.0])
        assert_rejects(
            create_sliced_crystal,
            match="cell_lengths must be positive",
            **kw,
        )

    def test_invalid_cell_angles(self) -> None:
        """Cell angles outside (0, 180) should be caught."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["cell_angles"] = jnp.array([0.0, 90.0, 90.0])
        assert_rejects(
            create_sliced_crystal,
            match="cell_angles must be between 0 and 180 degrees",
            **kw,
        )

    def test_nan_in_positions(self) -> None:
        """NaN in positions should be caught."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 0].set(jnp.nan)
        assert_rejects(
            create_sliced_crystal,
            match="cart_positions contain non-finite values",
            **kw,
        )

    def test_invalid_atomic_number_zero(self) -> None:
        """Atomic number of 0 should be caught."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 3].set(0.0)
        assert_rejects(
            create_sliced_crystal,
            match="atomic numbers must be in",
            **kw,
        )

    def test_invalid_atomic_number_too_large(self) -> None:
        """Atomic number > 118 should be caught."""
        kw: Any = self._make_valid_sliced_kwargs()
        kw["cart_positions"] = kw["cart_positions"].at[0, 3].set(200.0)
        assert_rejects(
            create_sliced_crystal,
            match="atomic numbers must be in",
            **kw,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_sliced_crystal_gradient_flow(self) -> None:
        """Test that gradients flow through SlicedCrystal."""

        def loss_fn(positions: Float[Array, "N 4"]) -> Num[Array, ""]:
            kw: Any = self._make_valid_sliced_kwargs()
            kw["cart_positions"] = positions
            sliced: SlicedCrystal = create_sliced_crystal(**kw)
            return jnp.sum(sliced.cart_positions[:, :3] ** 2)

        var_grad_fn: Callable[..., Any] = self.variant(jax.grad(loss_fn))
        kw = self._make_valid_sliced_kwargs()
        grads: Any = var_grad_fn(kw["cart_positions"])
        expected_grads: Float[Array, "..."] = 2 * kw["cart_positions"].at[
            :, 3
        ].set(0.0)
        chex.assert_trees_all_close(grads, expected_grads)


class TestRHEEDIntegration(chex.TestCase):
    """Test integrated operations with RHEED data structures."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_combined_rheed_structures(self) -> None:
        """Test combining RHEEDPattern and RHEEDImage in a single structure."""
        n_reflections: int = 10
        pattern: RHEEDPattern = create_rheed_pattern(
            jnp.arange(n_reflections, dtype=jnp.int32),
            jnp.ones((n_reflections, 3)),
            jnp.ones((n_reflections, 2)),
            jnp.ones(n_reflections),
        )

        image: RHEEDImage = create_rheed_image(
            jnp.ones((128, 256)), 2.0, 0.01, 0.037, 1000.0
        )

        # Use variant on a function that processes the combined structure
        def process_combined(
            p: RHEEDPattern, i: RHEEDImage
        ) -> tuple[dict[str, RHEEDPattern | RHEEDImage], Any]:
            combined: Any = {"pattern": p, "image": i}
            flat: Any
            treedef: Any
            flat: Any
            treedef: Any
            flat, treedef = tree_util.tree_flatten(combined)
            reconstructed: Any = tree_util.tree_unflatten(treedef, flat)
            return (combined, reconstructed)

        var_process_combined: Callable[..., Any] = self.variant(
            process_combined
        )
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

        def loss_fn(intensities: Float[Array, "3"]) -> Num[Array, ""]:
            pattern: RHEEDPattern = create_rheed_pattern(
                jnp.array([0, 1, 2], dtype=jnp.int32),
                jnp.ones((3, 3)),
                jnp.ones((3, 2)),
                intensities,
            )
            return jnp.sum(pattern.intensities**2)

        var_grad_fn: Callable[..., Any] = self.variant(jax.grad(loss_fn))
        intensities: Float[Array, "3"] = jnp.array([1.0, 2.0, 3.0])
        grads: Float[Array, "3"] = var_grad_fn(intensities)

        expected_grads: Float[Array, "3"] = 2 * intensities
        chex.assert_trees_all_close(grads, expected_grads)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient_flow_rheed_image(self) -> None:
        """Test that gradients flow through RHEEDImage correctly."""

        def loss_fn(img_array: Float[Array, "H W"]) -> Num[Array, ""]:
            image: RHEEDImage = create_rheed_image(
                img_array, 2.0, 0.01, 0.037, 1000.0
            )
            return jnp.sum(image.img_array**2)

        var_grad_fn: Callable[..., Any] = self.variant(jax.grad(loss_fn))
        img_array: Float[Array, "32 32"] = jnp.ones((32, 32))
        grads: Float[Array, "32 32"] = var_grad_fn(img_array)

        expected_grads: Float[Array, "32 32"] = 2 * img_array
        chex.assert_trees_all_close(grads, expected_grads)


class TestIdentifySurfaceAtoms(chex.TestCase):
    """Tests for identify_surface_atoms with all four methods.

    :see: :func:`~rheedium.types.identify_surface_atoms`
    """

    def test_height_method_default(self) -> None:
        """Height method with default 30% fraction."""
        positions: Float[Array, "..."] = jnp.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
            dtype=jnp.float64,
        )
        mask: Bool[Array, "..."] = identify_surface_atoms(positions)
        assert mask.shape == (5,)
        assert bool(mask[4])
        assert not bool(mask[0])

    def test_height_method_custom_fraction(self) -> None:
        """Height method with 50% fraction."""
        positions: Float[Array, "..."] = jnp.array(
            [[0, 0, i] for i in range(10)], dtype=jnp.float64
        )
        config: Any = SurfaceConfig(method="height", height_fraction=0.5)
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        n_surface: scalar_float = int(jnp.sum(mask))
        assert n_surface == 5

    def test_height_method_all_surface(self) -> None:
        """Height fraction of 1.0 marks all atoms as surface."""
        positions: Float[Array, "..."] = jnp.array(
            [[0, 0, i] for i in range(5)], dtype=jnp.float64
        )
        config: Any = SurfaceConfig(method="height", height_fraction=1.0)
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        assert bool(jnp.all(mask))

    def test_coordination_method(self) -> None:
        """Coordination method identifies under-coordinated atoms."""
        positions: Float[Array, "..."] = jnp.array(
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
        config: Any = SurfaceConfig(
            method="coordination",
            coordination_cutoff=2.0,
            coordination_threshold=3,
        )
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        assert mask.shape == (6,)
        assert bool(mask[5])

    def test_layers_method(self) -> None:
        """Layers method marks topmost layer."""
        positions: Float[Array, "..."] = jnp.array(
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
        config: Any = SurfaceConfig(
            method="layers", n_layers=1, layer_tolerance=0.5
        )
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        assert bool(mask[4])
        assert bool(mask[5])
        assert not bool(mask[0])
        assert not bool(mask[1])

    def test_layers_method_two_layers(self) -> None:
        """Layers method with n_layers=2."""
        positions: Float[Array, "..."] = jnp.array(
            [[0, 0, z] for z in [0.0, 0.0, 2.0, 2.0, 4.0, 4.0]],
            dtype=jnp.float64,
        )
        config: Any = SurfaceConfig(
            method="layers", n_layers=2, layer_tolerance=0.5
        )
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        n_surface: scalar_float = int(jnp.sum(mask))
        assert n_surface == 4

    def test_explicit_method(self) -> None:
        """Explicit method uses user-provided mask."""
        positions: Float[Array, "..."] = jnp.ones((5, 3))
        explicit: Bool[Array, "..."] = jnp.array(
            [True, False, True, False, True]
        )
        config: Any = SurfaceConfig(method="explicit", explicit_mask=explicit)
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        chex.assert_trees_all_equal(mask, explicit)

    def test_explicit_method_no_mask_fallback(self) -> None:
        """Explicit method without mask falls back to height."""
        positions: Float[Array, "..."] = jnp.array(
            [[0, 0, i] for i in range(5)], dtype=jnp.float64
        )
        config: Any = SurfaceConfig(method="explicit")
        mask: Bool[Array, "..."] = identify_surface_atoms(positions, config)
        assert mask.shape == (5,)
        assert bool(mask[4])

    def test_output_shape_matches_input(self) -> None:
        """Output mask has same length as input positions."""
        n: int
        for n in [1, 10, 100]:
            positions: Float[Array, "..."] = jnp.zeros((n, 3))
            mask: Bool[Array, "..."] = identify_surface_atoms(positions)
            assert mask.shape == (n,)


class TestSurfaceConfig(chex.TestCase):
    """Tests for SurfaceConfig NamedTuple.

    :see: :class:`~rheedium.types.SurfaceConfig`
    """

    def test_default_values(self) -> None:
        """Default config should use height method at 30%."""
        config: Any = SurfaceConfig()
        assert config.method == "height"
        assert config.height_fraction == 0.3
        assert config.coordination_cutoff == 3.0
        assert config.coordination_threshold == 8
        assert config.n_layers == 1
        assert config.layer_tolerance == 0.5
        assert config.explicit_mask is None

    def test_custom_values(self) -> None:
        """Custom config should preserve all values."""
        config: Any = SurfaceConfig(
            method="coordination",
            coordination_cutoff=4.0,
            coordination_threshold=6,
        )
        assert config.method == "coordination"
        assert config.coordination_cutoff == 4.0
        assert config.coordination_threshold == 6

    def test_immutable(self) -> None:
        """SurfaceConfig should be immutable (NamedTuple)."""
        config: Any = SurfaceConfig()
        with pytest.raises(AttributeError):
            config.method = "layers"


class TestDetectorGeometry(chex.TestCase):
    """Tests for DetectorGeometry NamedTuple.

    :see: :class:`~rheedium.types.DetectorGeometry`
    """

    def test_default_values(self) -> None:
        """Default geometry should have standard RHEED values."""
        geom: Any = DetectorGeometry()
        assert geom.distance == 100.0
        assert geom.tilt_angle == 0.0
        assert geom.curvature_radius == float("inf")
        assert geom.center_offset_h == 0.0
        assert geom.center_offset_v == 0.0
        assert geom.psf_sigma_pixels == 1.0

    def test_custom_values(self) -> None:
        """Custom geometry should preserve values."""
        geom: Any = DetectorGeometry(
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
        geom: Any = DetectorGeometry(psf_sigma_pixels=0.0)
        assert geom.psf_sigma_pixels == 0.0

    def test_immutable(self) -> None:
        """DetectorGeometry should be immutable (NamedTuple)."""
        geom: Any = DetectorGeometry()
        with pytest.raises(AttributeError):
            geom.distance = 200.0

    def test_infinite_curvature_is_flat(self) -> None:
        """Default curvature should indicate a flat detector."""
        geom: Any = DetectorGeometry()
        assert jnp.isinf(geom.curvature_radius)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
