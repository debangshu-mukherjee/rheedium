"""Tests for detector geometry carrier helpers."""

import chex

from rheedium.types.detector import (
    DetectorGeometry,
    detector_distance_mm,
    detector_psf_sigma_pixels,
)


class TestDetectorModule(chex.TestCase):
    """Detector geometry should live outside rheed_types."""

    def test_detector_geometry_helpers_extract_public_fields(self) -> None:
        """Extract distance and PSF values from the shared carrier."""
        geometry = DetectorGeometry(distance=250.0, psf_sigma_pixels=1.75)

        assert detector_distance_mm(geometry) == 250.0
        assert detector_psf_sigma_pixels(geometry) == 1.75
