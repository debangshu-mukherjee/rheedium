"""Tests for rheedium.audit metrics and reference metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float
from numpy.typing import NDArray

from rheedium.audit.metrics import (
    dominant_peak_positions,
    extract_streak_profile,
    normalized_cross_correlation,
    peak_centroid,
    peak_centroid_error_px,
    rod_spacing_error_px,
    specular_offset_px,
    streak_fwhm_px,
)
from rheedium.audit.reference_benchmark import load_reference_cases
from rheedium.audit.reference_types import (
    REQUIRED_REFERENCE_METADATA_KEYS,
    ReferenceCase,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = (
    _REPO_ROOT / "tests" / "test_data" / "reference_data" / "synthetic"
)


def _synthetic_three_peak_image(
    peak_positions_px: tuple[int, ...],
) -> Float[Array, "image_height image_width"]:
    """Create a simple three-peak detector image for metric tests."""
    image_height: int = 80
    image_width: int = 96
    x_axis: Float[Array, "image_width"] = jnp.arange(
        image_width, dtype=jnp.float64
    )
    y_axis: Float[Array, "image_height"] = jnp.arange(
        image_height, dtype=jnp.float64
    )
    x_grid: Float[Array, "image_height image_width"]
    y_grid: Float[Array, "image_height image_width"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    image: Float[Array, "image_height image_width"] = jnp.zeros(
        (image_height, image_width)
    )

    x0_px: int
    for x0_px in peak_positions_px:
        image = image + jnp.exp(
            -((x_grid - float(x0_px)) ** 2 + (y_grid - 42.0) ** 2)
            / (2.0 * 2.0**2)
        )

    return image / jnp.max(image)


class TestReferenceMetadata(chex.TestCase):
    """Tests for the stored audit benchmark metadata bundle.

    :see: :class:`~rheedium.audit.ReferenceCase`
    :see: :class:`~rheedium.audit.ReferenceMetadata`
    """

    def test_reference_metadata_complete(self) -> None:
        """Each stored reference case has the required metadata fields."""
        metadata_paths: Any = sorted(_REFERENCE_DIR.glob("*_metadata.json"))
        assert len(metadata_paths) >= 2

        metadata_path: Any
        for metadata_path in metadata_paths:
            metadata: Any = json.loads(metadata_path.read_text())
            missing_keys: Any = [
                key
                for key in REQUIRED_REFERENCE_METADATA_KEYS
                if key not in metadata
            ]
            assert missing_keys == []
            image_path: Path = _REFERENCE_DIR / metadata["image_path"]
            assert image_path.exists()

            data: Any
            with np.load(image_path) as data:
                image: Float[NDArray, "..."] = np.asarray(
                    data["image"], dtype=np.float64
                )

            assert list(image.shape) == metadata["image_shape_px"]
            assert np.all(np.isfinite(image))
            assert np.all(image >= 0.0)

    def test_load_reference_cases_reads_images(self) -> None:
        """The loader returns the shipped reference cases with images."""
        cases: Any = load_reference_cases(_REFERENCE_DIR)
        assert len(cases) >= 2
        case: Any
        for case in cases:
            assert isinstance(case, ReferenceCase)
            assert tuple(case.image.shape) == case.metadata.image_shape_px


class TestAuditMetrics(chex.TestCase):
    """Tests for pixel-space realism metrics.

    :see: :func:`~rheedium.audit.dominant_peak_positions`
    :see: :func:`~rheedium.audit.extract_streak_profile`
    :see: :func:`~rheedium.audit.normalized_cross_correlation`
    :see: :func:`~rheedium.audit.peak_centroid_error_px`
    :see: :func:`~rheedium.audit.peak_centroid`
    :see: :func:`~rheedium.audit.rod_spacing_error_px`
    :see: :func:`~rheedium.audit.specular_offset_px`
    :see: :func:`~rheedium.audit.streak_fwhm_px`
    """

    def test_metrics_translation_invariant(self) -> None:
        """Spacing and width metrics ignore rigid image translations."""
        image: Float[Array, "..."] = _synthetic_three_peak_image((18, 42, 66))
        shifted_image: Float[Array, "..."] = _synthetic_three_peak_image(
            (24, 48, 72)
        )
        peak_positions: Float[Array, "..."] = dominant_peak_positions(
            image, axis="horizontal", n_peaks=3, min_separation_px=6
        )
        shifted_peak_positions: Float[Array, "..."] = dominant_peak_positions(
            shifted_image,
            axis="horizontal",
            n_peaks=3,
            min_separation_px=6,
        )
        peak: Any = peak_centroid(image)
        shifted_peak: Any = peak_centroid(shifted_image)
        profile: Float[Array, "..."] = extract_streak_profile(image, peak)
        shifted_profile: Float[Array, "..."] = extract_streak_profile(
            shifted_image, shifted_peak
        )

        chex.assert_trees_all_close(
            rod_spacing_error_px(peak_positions, shifted_peak_positions),
            0.0,
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            streak_fwhm_px(profile),
            streak_fwhm_px(shifted_profile),
            atol=1e-10,
        )

    def test_metrics_report_zero_error_on_identical_patterns(self) -> None:
        """Identical detector images produce perfect agreement metrics."""
        image: Float[Array, "..."] = _synthetic_three_peak_image((22, 48, 74))
        peak: Any = peak_centroid(image)
        peak_positions: Float[Array, "..."] = dominant_peak_positions(
            image, axis="horizontal", n_peaks=3, min_separation_px=6
        )
        profile: Float[Array, "..."] = extract_streak_profile(image, peak)

        assert normalized_cross_correlation(image, image) == pytest.approx(
            1.0, abs=1e-12
        )
        assert peak_centroid_error_px(image, image) == pytest.approx(
            0.0, abs=1e-12
        )
        assert rod_spacing_error_px(peak_positions, peak_positions) == (
            pytest.approx(0.0, abs=1e-12)
        )
        assert specular_offset_px(peak, peak) == pytest.approx(0.0, abs=1e-12)
        assert streak_fwhm_px(profile) > 0.0
