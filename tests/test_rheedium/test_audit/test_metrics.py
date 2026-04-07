"""Tests for rheedium.audit metrics and reference metadata."""

from __future__ import annotations

import json
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from rheedium.audit import (
    REQUIRED_REFERENCE_METADATA_KEYS,
    ReferenceCase,
    dominant_peak_positions,
    extract_streak_profile,
    load_reference_cases,
    normalized_cross_correlation,
    peak_centroid,
    peak_centroid_error_px,
    rod_spacing_error_px,
    specular_offset_px,
    streak_fwhm_px,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIR: Path = (
    _REPO_ROOT / "tests" / "test_data" / "reference_data" / "synthetic"
)


def _synthetic_three_peak_image(
    peak_positions_px: tuple[int, int, int],
) -> Float[Array, "H W"]:
    """Create a simple three-peak detector image for metric tests."""
    image_height: int = 80
    image_width: int = 96
    x_axis: Float[Array, "W"] = jnp.arange(image_width, dtype=jnp.float64)
    y_axis: Float[Array, "H"] = jnp.arange(image_height, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    image: Float[Array, "H W"] = jnp.zeros((image_height, image_width))

    for x0_px in peak_positions_px:
        image = image + jnp.exp(
            -((x_grid - float(x0_px)) ** 2 + (y_grid - 42.0) ** 2)
            / (2.0 * 2.0**2)
        )

    return image / jnp.max(image)


class TestReferenceMetadata(chex.TestCase):
    """Tests for the stored audit benchmark metadata bundle."""

    def test_reference_metadata_complete(self) -> None:
        """Each stored reference case has the required metadata fields."""
        metadata_paths = sorted(_REFERENCE_DIR.glob("*_metadata.json"))
        assert len(metadata_paths) >= 2

        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text())
            missing_keys = [
                key
                for key in REQUIRED_REFERENCE_METADATA_KEYS
                if key not in metadata
            ]
            assert missing_keys == []
            image_path = _REFERENCE_DIR / metadata["image_path"]
            assert image_path.exists()

            with np.load(image_path) as data:
                image = np.asarray(data["image"], dtype=np.float64)

            assert list(image.shape) == metadata["image_shape_px"]
            assert np.all(np.isfinite(image))
            assert np.all(image >= 0.0)

    def test_load_reference_cases_reads_images(self) -> None:
        """The loader returns the shipped reference cases with images."""
        cases = load_reference_cases(_REFERENCE_DIR)
        assert len(cases) >= 2
        for case in cases:
            assert isinstance(case, ReferenceCase)
            assert tuple(case.image.shape) == case.metadata.image_shape_px


class TestAuditMetrics(chex.TestCase):
    """Tests for pixel-space realism metrics."""

    def test_metrics_translation_invariant(self) -> None:
        """Spacing and width metrics ignore rigid image translations."""
        image = _synthetic_three_peak_image((18, 42, 66))
        shifted_image = _synthetic_three_peak_image((24, 48, 72))
        peak_positions = dominant_peak_positions(
            image, axis="horizontal", n_peaks=3, min_separation_px=6
        )
        shifted_peak_positions = dominant_peak_positions(
            shifted_image,
            axis="horizontal",
            n_peaks=3,
            min_separation_px=6,
        )
        peak = peak_centroid(image)
        shifted_peak = peak_centroid(shifted_image)
        profile = extract_streak_profile(image, peak)
        shifted_profile = extract_streak_profile(shifted_image, shifted_peak)

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
        image = _synthetic_three_peak_image((22, 48, 74))
        peak = peak_centroid(image)
        peak_positions = dominant_peak_positions(
            image, axis="horizontal", n_peaks=3, min_separation_px=6
        )
        profile = extract_streak_profile(image, peak)

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
