"""Tests for rheedium.audit benchmark loading and aggregation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest

from rheedium.audit.reference_benchmark import (
    benchmark_reference_case,
    benchmark_reference_suite,
    load_reference_cases,
    simulate_detector_image_from_metadata,
)
from rheedium.audit.reference_types import (
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ReferenceCase,
    ReferenceMetadata,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = (
    _REPO_ROOT / "tests" / "test_data" / "reference_data" / "synthetic"
)

if os.environ.get("BUILDING_DOCS"):
    # Skip the heavy import-time simulation during documentation builds so
    # autodoc can import this module (tests are not executed there). The
    # placeholders below are only referenced inside test bodies, which do
    # not run under Sphinx. See docs/source/api/tests.rst.
    _REFERENCE_CASES = None
    _REFERENCE_CASE = None
    _SIMULATED_IMAGE = None
    _CASE_RESULT = None
    _TEMP_DIR = None
    _SUMMARY_OUTPUT_PATH = None
    _SUITE_SUMMARY = None
    _SUITE_PAYLOAD = None
else:
    _REFERENCE_CASES = load_reference_cases(_REFERENCE_DIR)
    _REFERENCE_CASE = _REFERENCE_CASES[0]
    _SIMULATED_IMAGE = simulate_detector_image_from_metadata(
        _REFERENCE_CASE.metadata,
        repo_root=_REPO_ROOT,
    )
    _CASE_RESULT = benchmark_reference_case(
        _REFERENCE_CASE, repo_root=_REPO_ROOT
    )
    _TEMP_DIR = TemporaryDirectory()
    _SUMMARY_OUTPUT_PATH = Path(_TEMP_DIR.name) / "summary.json"
    _SUITE_SUMMARY = benchmark_reference_suite(
        reference_dir=_REFERENCE_DIR,
        output_path=_SUMMARY_OUTPUT_PATH,
        repo_root=_REPO_ROOT,
    )
    _SUITE_PAYLOAD = json.loads(_SUMMARY_OUTPUT_PATH.read_text())


def test_load_reference_cases_returns_typed_cases() -> None:
    r"""Reference cases load as typed metadata-plus-image objects.

    :see: :obj:`~rheedium.audit.load_reference_cases`

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: Reference cases load
    as typed metadata-plus-image objects.

    Notes
    -----
    It constructs the representative inputs inside the test body, keeping the
    fixture and assertion path local to the documented case.

    The existing assertions in the function body compare the observed result
    with the expected contract for this module.

    The documented check is rendered from
    ``tests.test_rheedium.test_audit.test_reference_benchmark``, so the Test
    Reference exposes both the guarantee and the implementation path.
    """
    assert _REFERENCE_CASES is not None
    assert len(_REFERENCE_CASES) >= 2
    case: Any
    for case in _REFERENCE_CASES:
        assert isinstance(case, ReferenceCase)
        assert isinstance(case.metadata, ReferenceMetadata)
        assert case.metadata.metadata_path is not None
        assert tuple(case.image.shape) == case.metadata.image_shape_px
        assert np.all(np.isfinite(case.image))


def test_simulate_detector_image_from_metadata_matches_shape() -> None:
    r"""Regenerated images preserve the stored detector grid shape.

    :see: :obj:`~rheedium.audit.simulate_detector_image_from_metadata`

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: Regenerated images
    preserve the stored detector grid shape.

    Notes
    -----
    It constructs the representative inputs inside the test body, keeping the
    fixture and assertion path local to the documented case.

    The existing assertions in the function body compare the observed result
    with the expected contract for this module.

    The documented check is rendered from
    ``tests.test_rheedium.test_audit.test_reference_benchmark``, so the Test
    Reference exposes both the guarantee and the implementation path.
    """
    assert _SIMULATED_IMAGE is not None
    assert _REFERENCE_CASE is not None
    assert (
        tuple(_SIMULATED_IMAGE.shape)
        == _REFERENCE_CASE.metadata.image_shape_px
    )
    assert np.all(np.isfinite(_SIMULATED_IMAGE))
    assert float(np.max(_SIMULATED_IMAGE)) == pytest.approx(1.0, abs=1e-12)


def test_rg1_pixelwise_reference_images_match_pre_refactor_fixtures() -> None:
    r"""RG1: regenerated detector images match stored pre-refactor pixels.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: RG1: regenerated
    detector images match stored pre-refactor pixels.

    Notes
    -----
    It constructs the representative inputs inside the test body, keeping the
    fixture and assertion path local to the documented case.

    Numerical expectations are checked with tolerance-aware closeness
    assertions, which is appropriate for floating-point JAX arrays.

    The documented check is rendered from
    ``tests.test_rheedium.test_audit.test_reference_benchmark``, so the Test
    Reference exposes both the guarantee and the implementation path.
    """
    assert _REFERENCE_CASES is not None
    case: ReferenceCase
    for case in _REFERENCE_CASES:
        simulated = simulate_detector_image_from_metadata(
            case.metadata,
            repo_root=_REPO_ROOT,
        )
        np.testing.assert_allclose(simulated, case.image, atol=1e-12, rtol=0.0)


def test_benchmark_reference_case_matches_synthetic_fixture() -> None:
    r"""Synthetic fixtures benchmark back to a near-perfect match.

    :see: :obj:`~rheedium.audit.BenchmarkCaseResult`

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: Synthetic fixtures
    benchmark back to a near-perfect match.

    Notes
    -----
    It constructs the representative inputs inside the test body, keeping the
    fixture and assertion path local to the documented case.

    The existing assertions in the function body compare the observed result
    with the expected contract for this module.

    The documented check is rendered from
    ``tests.test_rheedium.test_audit.test_reference_benchmark``, so the Test
    Reference exposes both the guarantee and the implementation path.
    """
    assert _REFERENCE_CASE is not None
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
    r"""The suite runner writes a JSON summary with one entry per case.

    :see: :obj:`~rheedium.audit.BenchmarkSuiteResult`

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: The suite runner
    writes a JSON summary with one entry per case.

    Notes
    -----
    It constructs the representative inputs inside the test body, keeping the
    fixture and assertion path local to the documented case.

    The existing assertions in the function body compare the observed result
    with the expected contract for this module.

    The documented check is rendered from
    ``tests.test_rheedium.test_audit.test_reference_benchmark``, so the Test
    Reference exposes both the guarantee and the implementation path.
    """
    assert _SUMMARY_OUTPUT_PATH is not None
    assert _SUITE_PAYLOAD is not None
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
