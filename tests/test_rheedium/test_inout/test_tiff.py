"""Test suite for inout/tiff.py.

Verifies TIFF sequence loading, metadata extraction, normalization,
and beam center detection. Uses temporary TIFF files created with
tifffile for reproducible I/O tests.
"""

import tempfile
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tifffile
from jaxtyping import Array, Float, PRNGKeyArray
from numpy.typing import NDArray

from rheedium.inout.tiff import (
    FrameMetadata,
    detect_beam_center,
    extract_frame_metadata,
    load_tiff_as_rheed_image,
    load_tiff_sequence,
    normalize_sequence,
)
from rheedium.types.custom_types import scalar_float
from rheedium.types.rheed_types import RHEEDImage

H = 32
W = 48


def _write_multipage_tiff(
    path: Path,
    n_frames: int = 5,
) -> Float[NDArray, "frames height width"]:
    """Write a multi-page TIFF and return the data."""
    rng: np.random.Generator = np.random.default_rng(42)
    data: Float[NDArray, "frames height width"] = rng.uniform(
        10.0, 1000.0, size=(n_frames, H, W)
    ).astype(np.float32)
    tifffile.imwrite(str(path), data, photometric="minisblack")
    return data


def _write_single_frame_tiffs(
    dirpath: Path,
    n_frames: int = 5,
) -> Float[NDArray, "frames height width"]:
    """Write individual TIFF files to a directory and return data."""
    rng: np.random.Generator = np.random.default_rng(42)
    data: Float[NDArray, "frames height width"] = rng.uniform(
        10.0, 1000.0, size=(n_frames, H, W)
    ).astype(np.float32)
    dirpath.mkdir(parents=True, exist_ok=True)
    i: int
    for i in range(n_frames):
        filename: Any = f"frame_{i:04d}.tif"
        tifffile.imwrite(str(dirpath / filename), data[i])
    return data


class TestLoadTiffSequence(chex.TestCase):
    """Tests for load_tiff_sequence."""

    def setUp(self) -> None:
        """Create temporary directory for test files."""
        super().setUp()
        self._tmpdir: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.tmp_path: Path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self._tmpdir.cleanup()
        super().tearDown()

    def test_multipage_shape(self) -> None:
        """Multi-page TIFF loads with correct shape."""
        n_frames: int = 5
        _write_multipage_tiff(self.tmp_path / "stack.tif", n_frames)
        seq: Any
        meta: Any
        seq, meta = load_tiff_sequence(self.tmp_path / "stack.tif")
        chex.assert_shape(seq, (n_frames, H, W))
        self.assertEqual(len(meta), n_frames)

    def test_multipage_dtype(self) -> None:
        """Loaded data is float64 JAX array."""
        _write_multipage_tiff(self.tmp_path / "stack.tif", 3)
        seq: Any
        seq, _ = load_tiff_sequence(self.tmp_path / "stack.tif")
        self.assertEqual(seq.dtype, jnp.float64)

    def test_multipage_values(self) -> None:
        """Loaded values match written data."""
        expected: Float[Array, "..."] = _write_multipage_tiff(
            self.tmp_path / "stack.tif", 3
        )
        seq: Any
        seq, _ = load_tiff_sequence(self.tmp_path / "stack.tif")
        chex.assert_trees_all_close(
            seq, jnp.asarray(expected, dtype=jnp.float64), atol=1e-4
        )

    def test_directory_shape(self) -> None:
        """Directory of TIFFs loads with correct shape."""
        n_frames: int = 4
        _write_single_frame_tiffs(self.tmp_path / "frames", n_frames)
        seq: Any
        meta: Any
        seq, meta = load_tiff_sequence(self.tmp_path / "frames")
        chex.assert_shape(seq, (n_frames, H, W))
        self.assertEqual(len(meta), n_frames)

    def test_directory_values(self) -> None:
        """Directory values match written data."""
        expected: Float[Array, "..."] = _write_single_frame_tiffs(
            self.tmp_path / "frames", 3
        )
        seq: Any
        seq, _ = load_tiff_sequence(self.tmp_path / "frames")
        chex.assert_trees_all_close(
            seq, jnp.asarray(expected, dtype=jnp.float64), atol=1e-4
        )

    def test_metadata_indices(self) -> None:
        """Frame indices are sequential starting from 0."""
        n_frames: int = 4
        _write_multipage_tiff(self.tmp_path / "stack.tif", n_frames)
        meta: Any
        _, meta = load_tiff_sequence(self.tmp_path / "stack.tif")
        indices: Any = [m.frame_index for m in meta]
        self.assertEqual(indices, list(range(n_frames)))

    def test_single_frame_file(self) -> None:
        """Single-frame TIFF loads as (1, H, W)."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32) * 42.0
        tifffile.imwrite(str(self.tmp_path / "single.tif"), data)
        seq: Any
        meta: Any
        seq, meta = load_tiff_sequence(self.tmp_path / "single.tif")
        chex.assert_shape(seq, (1, H, W))
        self.assertEqual(len(meta), 1)

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            load_tiff_sequence("/nonexistent/path.tif")

    def test_empty_directory(self) -> None:
        """Raises ValueError for directory with no TIFFs."""
        empty_dir: Any = self.tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No TIFF files found"):
            load_tiff_sequence(empty_dir)

    def test_invalid_sort_by(self) -> None:
        """Raises ValueError for invalid sort_by."""
        _write_multipage_tiff(self.tmp_path / "stack.tif", 2)
        with pytest.raises(ValueError, match="sort_by must be"):
            load_tiff_sequence(self.tmp_path / "stack.tif", sort_by="invalid")

    def test_corrupt_file_raises_runtime_error(self) -> None:
        """A present but undecodable TIFF file raises RuntimeError."""
        corrupt: Any = self.tmp_path / "corrupt.tif"
        corrupt.write_bytes(b"not a real tiff payload")
        with pytest.raises(RuntimeError, match="Failed to decode TIFF"):
            load_tiff_sequence(corrupt)

    def test_finite_values(self) -> None:
        """No NaN or Inf in loaded data."""
        _write_multipage_tiff(self.tmp_path / "stack.tif", 3)
        seq: Any
        seq, _ = load_tiff_sequence(self.tmp_path / "stack.tif")
        chex.assert_tree_all_finite(seq)


class TestExtractFrameMetadata(chex.TestCase):
    """Tests for extract_frame_metadata."""

    def setUp(self) -> None:
        """Create temporary directory for test files."""
        super().setUp()
        self._tmpdir: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.tmp_path: Path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self._tmpdir.cleanup()
        super().tearDown()

    def test_returns_named_tuple(self) -> None:
        """Returns a FrameMetadata instance."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32)
        fpath: Path = self.tmp_path / "meta.tif"
        tifffile.imwrite(str(fpath), data)
        tif: Any
        with tifffile.TiffFile(str(fpath)) as tif:
            meta: Any = extract_frame_metadata(tif.pages[0], 7)
        self.assertIsInstance(meta, FrameMetadata)
        self.assertEqual(meta.frame_index, 7)

    def test_description_string(self) -> None:
        """Description is a string (possibly empty)."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32)
        fpath: Path = self.tmp_path / "desc.tif"
        tifffile.imwrite(str(fpath), data)
        tif: Any
        with tifffile.TiffFile(str(fpath)) as tif:
            meta: Any = extract_frame_metadata(tif.pages[0])
        self.assertIsInstance(meta.description, str)


class TestNormalizeSequence(chex.TestCase):
    """Tests for normalize_sequence."""

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        seq: Float[Array, "..."] = jnp.ones((5, H, W)) * 500.0
        result: Float[Array, "..."] = normalize_sequence(seq)
        chex.assert_shape(result, (5, H, W))

    def test_output_range(self) -> None:
        """Each frame is normalized to [0, 1]."""
        rng: np.random.Generator = np.random.default_rng(99)
        np_data: Float[NDArray, "frames height width"] = rng.uniform(
            10.0, 1000.0, size=(5, H, W)
        )
        seq: Float[Array, "..."] = jnp.asarray(np_data, dtype=jnp.float64)
        result: Float[Array, "..."] = normalize_sequence(seq)
        t: Any
        for t in range(5):
            frame: Any = result[t]
            chex.assert_trees_all_close(jnp.min(frame), 0.0, atol=1e-10)
            chex.assert_trees_all_close(jnp.max(frame), 1.0, atol=1e-10)

    def test_with_background(self) -> None:
        """Background subtraction reduces values."""
        seq: Float[Array, "..."] = jnp.ones((3, H, W)) * 500.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 200.0
        result: Float[Array, "..."] = normalize_sequence(seq, background=bg)
        chex.assert_shape(result, (3, H, W))
        chex.assert_tree_all_finite(result)

    def test_with_flat_field(self) -> None:
        """Flat-field correction applied without errors."""
        rng: np.random.Generator = np.random.default_rng(99)
        np_data: Float[NDArray, "frames height width"] = rng.uniform(
            10.0, 1000.0, size=(3, H, W)
        )
        seq: Float[Array, "..."] = jnp.asarray(np_data, dtype=jnp.float64)
        flat: Float[Array, "..."] = jnp.ones((H, W)) * 0.8
        result: Float[Array, "..."] = normalize_sequence(seq, flat_field=flat)
        chex.assert_shape(result, (3, H, W))
        chex.assert_tree_all_finite(result)

    def test_with_all_corrections(self) -> None:
        """Full correction pipeline works."""
        rng: np.random.Generator = np.random.default_rng(99)
        np_data: Float[NDArray, "frames height width"] = rng.uniform(
            100.0, 1000.0, size=(3, H, W)
        )
        seq: Float[Array, "..."] = jnp.asarray(np_data, dtype=jnp.float64)
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 50.0
        flat: Float[Array, "..."] = jnp.ones((H, W)) * 0.9
        result: Float[Array, "..."] = normalize_sequence(
            seq, background=bg, flat_field=flat
        )
        chex.assert_shape(result, (3, H, W))
        chex.assert_tree_all_finite(result)
        self.assertTrue(jnp.all(result >= 0.0))
        self.assertTrue(jnp.all(result <= 1.0))

    def test_nonnegative(self) -> None:
        """Output is non-negative even when background exceeds signal."""
        seq: Float[Array, "..."] = jnp.ones((2, H, W)) * 10.0
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 100.0
        result: Float[Array, "..."] = normalize_sequence(seq, background=bg)
        self.assertTrue(jnp.all(result >= 0.0))

    def test_uniform_frames(self) -> None:
        """Uniform frames normalize to zero (no range)."""
        seq: Float[Array, "..."] = jnp.ones((3, H, W)) * 42.0
        result: Float[Array, "..."] = normalize_sequence(seq)
        chex.assert_trees_all_close(result, jnp.zeros((3, H, W)), atol=1e-6)


class TestDetectBeamCenter(chex.TestCase):
    """Tests for detect_beam_center."""

    def test_shape(self) -> None:
        """Output is a 2-element array."""
        img: Float[Array, "..."] = jnp.zeros((H, W))
        img = img.at[H // 2, W // 2].set(1000.0)
        center: Float[Array, "..."] = detect_beam_center(img)
        chex.assert_shape(center, (2,))

    def test_centered_spot(self) -> None:
        """Detects a centered spot correctly."""
        row_center: Float[Array, "..."] = H // 2
        col_center: Float[Array, "..."] = W // 2
        y: Float[Array, "..."] = jnp.arange(H, dtype=jnp.float64)
        x: Float[Array, "..."] = jnp.arange(W, dtype=jnp.float64)
        yy: Float[Array, "..."]
        xx: Float[Array, "..."]
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        img: Float[Array, "..."] = jnp.exp(
            -((yy - row_center) ** 2 + (xx - col_center) ** 2) / (2.0 * 3.0**2)
        )
        center: Float[Array, "..."] = detect_beam_center(img, jnp.float64(3.0))
        chex.assert_trees_all_close(
            center,
            jnp.array([row_center, col_center], dtype=jnp.float64),
            atol=1.0,
        )

    def test_offset_spot(self) -> None:
        """Detects an off-center spot correctly."""
        row_center: Float[Array, "..."] = H // 4
        col_center: Float[Array, "..."] = 3 * W // 4
        y: Float[Array, "..."] = jnp.arange(H, dtype=jnp.float64)
        x: Float[Array, "..."] = jnp.arange(W, dtype=jnp.float64)
        yy: Float[Array, "..."]
        xx: Float[Array, "..."]
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        img: Float[Array, "..."] = jnp.exp(
            -((yy - row_center) ** 2 + (xx - col_center) ** 2) / (2.0 * 3.0**2)
        )
        center: Float[Array, "..."] = detect_beam_center(img, jnp.float64(3.0))
        chex.assert_trees_all_close(
            center,
            jnp.array([row_center, col_center], dtype=jnp.float64),
            atol=1.5,
        )

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        img: Float[Array, "..."] = jnp.ones((H, W)) * 50.0
        center: Float[Array, "..."] = detect_beam_center(img)
        chex.assert_tree_all_finite(center)

    def test_noisy_spot(self) -> None:
        """Detects spot in noisy image."""
        row_center: Float[Array, "..."] = H // 2
        col_center: Float[Array, "..."] = W // 2
        y: Float[Array, "..."] = jnp.arange(H, dtype=jnp.float64)
        x: Float[Array, "..."] = jnp.arange(W, dtype=jnp.float64)
        yy: Float[Array, "height width"]
        xx: Float[Array, "height width"]
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        signal: Float[Array, "height width"] = 1000.0 * jnp.exp(
            -((yy - row_center) ** 2 + (xx - col_center) ** 2) / (2.0 * 3.0**2)
        )
        key: PRNGKeyArray = jax.random.PRNGKey(42)
        noise: Float[Array, "height width"] = (
            jax.random.normal(key, (H, W)) * 10.0
        )
        img: Float[Array, "..."] = jnp.maximum(signal + noise, 0.0)
        center: Float[Array, "..."] = detect_beam_center(img, jnp.float64(5.0))
        chex.assert_trees_all_close(
            center,
            jnp.array([row_center, col_center], dtype=jnp.float64),
            atol=2.0,
        )


class TestLoadTiffAsRheedImage(chex.TestCase):
    """Tests for load_tiff_as_rheed_image."""

    def setUp(self) -> None:
        """Create temporary directory for test files."""
        super().setUp()
        self._tmpdir: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.tmp_path: Path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self._tmpdir.cleanup()
        super().tearDown()

    def test_returns_rheed_image(self) -> None:
        """Returns a RHEEDImage instance."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32) * 500.0
        tifffile.imwrite(str(self.tmp_path / "frame.tif"), data)
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "frame.tif",
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
        )
        self.assertIsInstance(img, RHEEDImage)

    def test_image_shape(self) -> None:
        """Image array has correct shape."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32)
        tifffile.imwrite(str(self.tmp_path / "frame.tif"), data)
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "frame.tif",
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
        )
        chex.assert_shape(img.img_array, (H, W))

    def test_wavelength_correct(self) -> None:
        """Electron wavelength is physically reasonable for 20 keV."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32)
        tifffile.imwrite(str(self.tmp_path / "frame.tif"), data)
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "frame.tif",
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
        )
        chex.assert_trees_all_close(
            img.electron_wavelength, 0.0859, atol=0.001
        )

    def test_with_background(self) -> None:
        """Background subtraction is applied."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32) * 500.0
        tifffile.imwrite(str(self.tmp_path / "frame.tif"), data)
        bg: Float[Array, "..."] = jnp.ones((H, W)) * 200.0
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "frame.tif",
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
            background=bg,
        )
        chex.assert_trees_all_close(
            img.img_array,
            jnp.full((H, W), 300.0),
            atol=1.0,
        )

    def test_multipage_takes_first(self) -> None:
        """Multi-page TIFF uses only the first frame."""
        data: Float[NDArray, "..."] = np.stack(
            [
                np.ones((H, W), dtype=np.float32) * 100.0,
                np.ones((H, W), dtype=np.float32) * 999.0,
            ]
        )
        tifffile.imwrite(
            str(self.tmp_path / "multi.tif"),
            data,
            photometric="minisblack",
        )
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "multi.tif",
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
        )
        chex.assert_trees_all_close(
            img.img_array,
            jnp.full((H, W), 100.0),
            atol=1.0,
        )

    def test_parameters_stored(self) -> None:
        """Beam and detector parameters are stored correctly."""
        data: Float[NDArray, "..."] = np.ones((H, W), dtype=np.float32)
        tifffile.imwrite(str(self.tmp_path / "frame.tif"), data)
        img: Any = load_tiff_as_rheed_image(
            self.tmp_path / "frame.tif",
            incoming_angle_deg=3.5,
            energy_kev=15.0,
            detector_distance_mm=400.0,
            calibration=0.05,
        )
        chex.assert_trees_all_close(img.incoming_angle, 3.5, atol=1e-10)
        chex.assert_trees_all_close(img.detector_distance, 400.0, atol=1e-10)
        chex.assert_trees_all_close(img.calibration, 0.05, atol=1e-10)


class TestGradients(chex.TestCase):
    """Gradient tests for tiff functions."""

    def test_grad_through_normalize(self) -> None:
        """jax.grad flows through normalize_sequence."""

        def loss(scale: scalar_float) -> scalar_float:
            seq: Float[Array, "3 H W"] = (
                jnp.linspace(1.0, 100.0, 3 * H * W).reshape(3, H, W) * scale
            )
            result: Float[Array, "3 H W"] = normalize_sequence(seq)
            return jnp.sum(result)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_detect_beam_center(self) -> None:
        """jax.grad flows through detect_beam_center."""

        def loss(peak_row: scalar_float) -> scalar_float:
            y: Float[Array, "H"] = jnp.arange(H, dtype=jnp.float64)
            x: Float[Array, "W"] = jnp.arange(W, dtype=jnp.float64)
            yy: Float[Array, "H W"]
            xx: Float[Array, "H W"]
            yy, xx = jnp.meshgrid(y, x, indexing="ij")
            img: Float[Array, "H W"] = jnp.exp(
                -((yy - peak_row) ** 2 + (xx - W / 2.0) ** 2) / (2.0 * 3.0**2)
            )
            center: Float[Array, "2"] = detect_beam_center(
                img, jnp.float64(3.0)
            )
            return jnp.sum(center)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(H / 2.0))
        chex.assert_tree_all_finite(grad_val)
