r"""Audit utilities for RHEED simulation validation.

Extended Summary
----------------
This module provides two complementary audit tracks for the rheedium
simulation pipeline:

1. **Physics invariants** (:mod:`.invariants`) verify that any correct
   RHEED simulator must obey explicit physical laws — elastic-scattering
   closure, Friedel symmetry, form-factor positivity and monotonicity,
   relativistic wavelength consistency. These checks are stateless with
   respect to disk and catch sign errors, missing factors, normalization
   mistakes, and frame-of-reference bugs that regression comparison
   against stored images cannot.
2. **Reference benchmarking** (:mod:`.reference_benchmark`,
   :mod:`.metrics`) compares regenerated simulations against stored
   detector images using image-space similarity metrics. These detect
   *changes* relative to a baseline rather than physics correctness, and
   are intended as a guardrail against unintended drift.

Routine Listings
----------------
:class:`InvariantResult`
    Structured pass/fail container for one physics invariant check.
:func:`benchmark_reference_case`
    Compare one stored reference case against a regenerated simulation.
:func:`benchmark_reference_suite`
    Run the full reference bundle and return an aggregate summary.
:func:`check_elastic_closure_ewald`
    Verify ``ewald_simulator`` reflections lie on the Ewald sphere.
:func:`check_form_factor_kirkland_lobato_close`
    Coarse cross-check that the two parameterizations agree.
:func:`check_form_factor_monotonic_decrease`
    Verify electron form factors decrease monotonically with q.
:func:`check_form_factor_positivity`
    Verify electron form factors are positive over their valid range.
:func:`check_friedel_law_structure_factor`
    Verify :math:`I(\\mathbf{G}) = I(-\\mathbf{G})` for the kinematic
    structure factor.
:func:`check_wavelength_relativistic_consistency`
    Verify ``wavelength_ang`` matches an independent CODATA evaluation.
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
:func:`run_default_invariants`
    Run the full default physics-invariant suite.
:func:`simulate_detector_image_from_metadata`
    Regenerate a reference detector image from its stored metadata.
:func:`specular_offset_px`
    Measure pixel-space displacement between two anchor positions.
:func:`streak_fwhm_px`
    Measure full width at half maximum from a 1-D profile.

Notes
-----
The benchmark fixtures currently shipped with the test suite are
synthetic detector images generated from rheedium itself. They validate
the benchmark pipeline and data format until calibrated experimental
references are added. The physics-invariant suite, by contrast, needs
no fixtures and is suitable for use as a CI gate against silent
simulation regressions.
"""

from importlib import import_module
from typing import Any

from .invariants import (
    InvariantResult,
    check_elastic_closure_ewald,
    check_form_factor_kirkland_lobato_close,
    check_form_factor_monotonic_decrease,
    check_form_factor_positivity,
    check_friedel_law_structure_factor,
    check_wavelength_relativistic_consistency,
    run_default_invariants,
)
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
    REQUIRED_REFERENCE_METADATA_KEYS,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ReferenceCase,
    ReferenceMetadata,
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
    "check_elastic_closure_ewald",
    "check_form_factor_kirkland_lobato_close",
    "check_form_factor_monotonic_decrease",
    "check_form_factor_positivity",
    "check_friedel_law_structure_factor",
    "check_wavelength_relativistic_consistency",
    "dominant_peak_positions",
    "extract_streak_profile",
    "InvariantResult",
    "load_reference_cases",
    "normalized_cross_correlation",
    "peak_centroid",
    "peak_centroid_error_px",
    "ReferenceCase",
    "ReferenceMetadata",
    "REQUIRED_REFERENCE_METADATA_KEYS",
    "rod_spacing_error_px",
    "run_default_invariants",
    "simulate_detector_image_from_metadata",
    "specular_offset_px",
    "streak_fwhm_px",
]
