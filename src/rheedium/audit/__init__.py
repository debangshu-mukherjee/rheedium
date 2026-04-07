"""Audit utilities for benchmark-driven RHEED validation.

Extended Summary
----------------
This module provides experiment-facing audit tools for comparing
simulated detector images against stored reference cases. The initial
focus is realism benchmarking: image similarity metrics, peak and
streak measurements, and a lightweight benchmark runner that
re-simulates reference configurations and reports quantitative errors.

Routine Listings
----------------
:func:`benchmark_reference_case`
    Compare one stored reference case against a regenerated simulation.
:func:`benchmark_reference_suite`
    Run the full reference bundle and return an aggregate summary.
:func:`dominant_peak_positions`
    Extract the strongest peak positions from an image projection.
:func:`extract_streak_profile`
    Extract a vertical or horizontal streak profile near a peak.
:func:`load_reference_cases`
    Load stored reference images and metadata from disk.
:func:`normalized_cross_correlation`
    Measure image similarity on a normalized scale.
:func:`peak_centroid`
    Compute the centroid of the brightest image region.
:func:`peak_centroid_error_px`
    Compare bright-region centroids between two images.
:func:`rod_spacing_error_px`
    Compare peak-to-peak spacing between two patterns.
:func:`simulate_detector_image_from_metadata`
    Regenerate a reference detector image from its stored metadata.
:func:`specular_offset_px`
    Measure pixel-space displacement between two anchor positions.
:func:`streak_fwhm_px`
    Measure full width at half maximum from a 1-D profile.

Notes
-----
The benchmark fixtures currently shipped with the test suite are
synthetic detector images generated from rheedium itself. They are used
to validate the audit pipeline and data format until calibrated
experimental references are added.
"""

from importlib import import_module
from typing import Any

from .metrics import (
    dominant_peak_positions,
    extract_streak_profile,
    normalized_cross_correlation,
    peak_centroid,
    peak_centroid_error_px,
    rod_spacing_error_px,
    specular_offset_px,
    streak_fwhm_px,
)
from .reference_types import (
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ReferenceCase,
    ReferenceMetadata,
    REQUIRED_REFERENCE_METADATA_KEYS,
)

_BENCHMARK_EXPORTS: frozenset[str] = frozenset(
    {
        "benchmark_reference_case",
        "benchmark_reference_suite",
        "load_reference_cases",
        "simulate_detector_image_from_metadata",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose benchmark helpers without eager module import."""
    if name in _BENCHMARK_EXPORTS:
        benchmark_module = import_module(".reference_benchmark", __name__)
        return getattr(benchmark_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    "benchmark_reference_case",
    "benchmark_reference_suite",
    "BenchmarkCaseResult",
    "BenchmarkSuiteResult",
    "dominant_peak_positions",
    "extract_streak_profile",
    "load_reference_cases",
    "normalized_cross_correlation",
    "peak_centroid",
    "peak_centroid_error_px",
    "ReferenceCase",
    "ReferenceMetadata",
    "REQUIRED_REFERENCE_METADATA_KEYS",
    "rod_spacing_error_px",
    "simulate_detector_image_from_metadata",
    "specular_offset_px",
    "streak_fwhm_px",
]
