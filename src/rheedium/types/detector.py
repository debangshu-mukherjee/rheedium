"""Detector geometry carrier and detector-grid helpers.

Extended Summary
----------------
This module owns detector geometry types shared by kinematic and multislice
detector-image paths. The public ``DetectorGeometry`` carrier preserves the
legacy RHEED geometry fields, while helper functions bind the common dense
detector-grid arguments used by Layer-0 renderers.

Routine Listings
----------------
:class:`DetectorGeometry`
    Configuration for detector distance, tilt, curvature, offsets, dense image
    calibration, and PSF.
:func:`detector_distance_mm`
    Extract the sample-to-detector distance used by Layer-0 kernels.
:func:`detector_image_shape_px`
    Extract dense detector image shape.
:func:`detector_pixel_size_mm`
    Extract detector pixel calibration.
:func:`detector_beam_center_px`
    Extract beam-centre pixel coordinate.
:func:`detector_psf_sigma_pixels`
    Extract detector PSF width.
"""

from typing import NamedTuple


class DetectorGeometry(NamedTuple):
    """Configuration for RHEED detector geometry.

    :see: :class:`~.test_rheed_types.TestDetectorGeometry`

    Attributes
    ----------
    distance : float
        Perpendicular distance from sample to detector center in mm.
        Default: 100.0
    tilt_angle : float
        Tilt angle of the detector about the horizontal axis in degrees.
        Positive tilt rotates the top of the screen away from the sample.
        Default: 0.0 (vertical screen)
    curvature_radius : float
        Radius of curvature of the detector screen in mm.
        Use ``float("inf")`` for flat screen. Default: flat.
    center_offset_h : float
        Horizontal offset of detector center from beam axis in mm.
    center_offset_v : float
        Vertical offset of detector center from beam axis in mm.
    image_shape_px : tuple[int, int]
        Dense detector image shape as ``(height_px, width_px)``.
    pixel_size_mm : tuple[float, float]
        Detector calibration as ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)`` as
        ``(center_x_px, center_y_px)``.
    psf_sigma_pixels : float
        Point spread function 1-sigma width in pixels. Use 0.0 to disable PSF
        convolution. Default: 1.0.
    """

    distance: float = 100.0
    tilt_angle: float = 0.0
    curvature_radius: float = float("inf")
    center_offset_h: float = 0.0
    center_offset_v: float = 0.0
    image_shape_px: tuple[int, int] = (192, 192)
    pixel_size_mm: tuple[float, float] = (1.5, 3.0)
    beam_center_px: tuple[float, float] = (96.0, 8.0)
    psf_sigma_pixels: float = 1.0


def detector_distance_mm(geometry: DetectorGeometry) -> float:
    """Return detector distance in millimetres for Layer-0 kernels."""
    return geometry.distance


def detector_image_shape_px(geometry: DetectorGeometry) -> tuple[int, int]:
    """Return dense detector image shape as ``(height_px, width_px)``."""
    return geometry.image_shape_px


def detector_pixel_size_mm(geometry: DetectorGeometry) -> tuple[float, float]:
    """Return detector pixel calibration in millimetres per pixel."""
    return geometry.pixel_size_mm


def detector_beam_center_px(geometry: DetectorGeometry) -> tuple[float, float]:
    """Return detector beam-centre pixel coordinate."""
    return geometry.beam_center_px


def detector_extent_mm(
    geometry: DetectorGeometry,
) -> tuple[float, float, float, float]:
    """Return matplotlib-style detector extent in millimetres."""
    height_px, width_px = geometry.image_shape_px
    x_mm_per_px, y_mm_per_px = geometry.pixel_size_mm
    center_x_px, center_y_px = geometry.beam_center_px
    return (
        -center_x_px * x_mm_per_px,
        (width_px - center_x_px) * x_mm_per_px,
        -center_y_px * y_mm_per_px,
        (height_px - center_y_px) * y_mm_per_px,
    )


def detector_psf_sigma_pixels(geometry: DetectorGeometry) -> float:
    """Return detector PSF width in pixels."""
    return geometry.psf_sigma_pixels


__all__: list[str] = [
    "DetectorGeometry",
    "detector_beam_center_px",
    "detector_distance_mm",
    "detector_extent_mm",
    "detector_image_shape_px",
    "detector_pixel_size_mm",
    "detector_psf_sigma_pixels",
]
