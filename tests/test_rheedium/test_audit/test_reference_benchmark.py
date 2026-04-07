"""Tests for rheedium.audit benchmark loading and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from rheedium.audit import (
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ReferenceCase,
    ReferenceMetadata,
    benchmark_reference_case,
    benchmark_reference_suite,
    load_reference_cases,
    simulate_detector_image_from_metadata,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_REFERENCE_DIR: Path = _REPO_ROOT / "tests" / "test_data" / "rheed_reference"
_REFERENCE_CASES = load_reference_cases(_REFERENCE_DIR)
_REFERENCE_CASE = _REFERENCE_CASES[0]
_SIMULATED_IMAGE = simulate_detector_image_from_metadata(
    _REFERENCE_CASE.metadata,
    repo_root=_REPO_ROOT,
)
_CASE_RESULT = benchmark_reference_case(_REFERENCE_CASE, repo_root=_REPO_ROOT)
_TEMP_DIR = TemporaryDirectory()
_SUMMARY_OUTPUT_PATH = Path(_TEMP_DIR.name) / "summary.json"
_SUITE_SUMMARY = benchmark_reference_suite(
    reference_dir=_REFERENCE_DIR,
    output_path=_SUMMARY_OUTPUT_PATH,
    repo_root=_REPO_ROOT,
)
_SUITE_PAYLOAD = json.loads(_SUMMARY_OUTPUT_PATH.read_text())


def test_load_reference_cases_returns_typed_cases() -> None:
    """Reference cases load as typed metadata-plus-image objects."""
    assert len(_REFERENCE_CASES) >= 2
    for case in _REFERENCE_CASES:
        assert isinstance(case, ReferenceCase)
        assert isinstance(case.metadata, ReferenceMetadata)
        assert case.metadata.metadata_path is not None
        assert tuple(case.image.shape) == case.metadata.image_shape_px
        assert np.all(np.isfinite(case.image))


def test_simulate_detector_image_from_metadata_matches_shape() -> None:
    """Regenerated images preserve the stored detector grid shape."""
    assert (
        tuple(_SIMULATED_IMAGE.shape)
        == _REFERENCE_CASE.metadata.image_shape_px
    )
    assert np.all(np.isfinite(_SIMULATED_IMAGE))
    assert float(np.max(_SIMULATED_IMAGE)) == pytest.approx(1.0, abs=1e-12)


def test_benchmark_reference_case_matches_synthetic_fixture() -> None:
    """Synthetic fixtures benchmark back to a near-perfect match."""
    assert isinstance(_CASE_RESULT, BenchmarkCaseResult)
    assert _CASE_RESULT.reference_id == _REFERENCE_CASE.metadata.reference_id
    assert _CASE_RESULT.normalized_cross_correlation == pytest.approx(
        1.0, abs=1e-12
    )
    assert _CASE_RESULT.specular_offset_px == pytest.approx(0.0, abs=1e-12)
    assert _CASE_RESULT.peak_centroid_error_px == pytest.approx(0.0, abs=1e-12)
    assert _CASE_RESULT.rod_spacing_error_px == pytest.approx(0.0, abs=1e-12)
    assert _CASE_RESULT.streak_fwhm_abs_error_px == pytest.approx(
        0.0, abs=1e-12
    )


def test_benchmark_reference_suite_writes_json_summary() -> None:
    """The suite runner writes a JSON summary with one entry per case."""
    assert isinstance(_SUITE_SUMMARY, BenchmarkSuiteResult)
    assert _SUITE_SUMMARY.reference_count >= 2
    assert _SUMMARY_OUTPUT_PATH.exists()
    assert _SUITE_PAYLOAD["reference_count"] == _SUITE_SUMMARY.reference_count
    assert _SUITE_PAYLOAD["mean_normalized_cross_correlation"] == (
        pytest.approx(1.0, abs=1e-12)
    )
    assert _SUITE_PAYLOAD["mean_specular_offset_px"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert _SUITE_PAYLOAD["max_streak_fwhm_abs_error_px"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert len(_SUITE_PAYLOAD["cases"]) == _SUITE_SUMMARY.reference_count
