"""Tests for detector geometry carrier helpers."""

import chex

from rheedium.types.detector import (
    DetectorGeometry,
    detector_beam_center_px,
    detector_distance_mm,
    detector_extent_mm,
    detector_image_shape_px,
    detector_pixel_size_mm,
    detector_psf_sigma_pixels,
)


class TestDetectorModule(chex.TestCase):
    """Detector geometry should live outside rheed_types."""

    def test_detector_geometry_helpers_extract_public_fields(self) -> None:
        """Extract distance and PSF values from the shared carrier."""
        geometry = DetectorGeometry(
            distance=250.0,
            image_shape_px=(100, 200),
            pixel_size_mm=(1.5, 3.0),
            beam_center_px=(80.0, 5.0),
            psf_sigma_pixels=1.75,
        )

        assert detector_distance_mm(geometry) == 250.0
        assert detector_image_shape_px(geometry) == (100, 200)
        assert detector_pixel_size_mm(geometry) == (1.5, 3.0)
        assert detector_beam_center_px(geometry) == (80.0, 5.0)
        assert detector_extent_mm(geometry) == (
            -120.0,
            180.0,
            -15.0,
            285.0,
        )
        assert detector_psf_sigma_pixels(geometry) == 1.75
