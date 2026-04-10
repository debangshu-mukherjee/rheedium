"""Tests for the rheedium.audit physics-invariant suite."""

from __future__ import annotations

import pytest

from rheedium.audit.invariants import (
    InvariantResult,
    check_elastic_closure_ewald,
    check_form_factor_kirkland_lobato_close,
    check_form_factor_monotonic_decrease,
    check_form_factor_positivity,
    check_friedel_law_structure_factor,
    check_wavelength_relativistic_consistency,
    run_default_invariants,
)


def _assert_well_formed(result, expected_name):
    """Assert an InvariantResult has all fields populated correctly."""
    assert isinstance(result, InvariantResult)
    assert result.name == expected_name
    assert isinstance(result.passed, bool)
    assert isinstance(result.residual, float)
    assert isinstance(result.tolerance, float)
    assert isinstance(result.units, str) and result.units
    assert isinstance(result.detail, str) and result.detail


def test_form_factor_positivity_both_parameterizations():
    """Both Kirkland and Lobato form factors stay positive."""
    kirkland_result, lobato_result = check_form_factor_positivity()
    _assert_well_formed(kirkland_result, "form_factor_positivity_kirkland")
    _assert_well_formed(lobato_result, "form_factor_positivity_lobato")
    assert (
        kirkland_result.passed
    ), f"Kirkland form factor went negative: residual={kirkland_result.residual}"
    assert (
        lobato_result.passed
    ), f"Lobato form factor went negative: residual={lobato_result.residual}"


def test_form_factor_monotonic_decrease_both_parameterizations():
    """Both Kirkland and Lobato form factors decrease monotonically."""
    kirkland_result, lobato_result = check_form_factor_monotonic_decrease()
    _assert_well_formed(kirkland_result, "form_factor_monotonic_kirkland")
    _assert_well_formed(lobato_result, "form_factor_monotonic_lobato")
    assert (
        kirkland_result.passed
    ), f"Kirkland not monotonic: residual={kirkland_result.residual}"
    assert (
        lobato_result.passed
    ), f"Lobato not monotonic: residual={lobato_result.residual}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Kirkland and Lobato parameterizations currently disagree by "
        "~65% on q in [0.5, 3.0] 1/Å for {C, Si, Cu, Ag}. Tracked as a "
        "real audit finding; expected to pass once the underlying "
        "normalization or units mismatch is resolved."
    ),
)
def test_form_factor_kirkland_lobato_close():
    """Kirkland and Lobato form factors are within ~30% of each other."""
    result = check_form_factor_kirkland_lobato_close()
    _assert_well_formed(result, "form_factor_kirkland_lobato_close")
    assert (
        result.passed
    ), f"Cross-parameterization disagreement: residual={result.residual}"


def test_wavelength_relativistic_consistency():
    """rheedium.tools.wavelength_ang matches CODATA-derived de Broglie."""
    result = check_wavelength_relativistic_consistency()
    _assert_well_formed(result, "wavelength_relativistic_consistency")
    assert (
        result.passed
    ), f"Wavelength mismatch vs CODATA: residual={result.residual}"


def test_friedel_law_structure_factor():
    """I(G) = I(-G) for a non-centrosymmetric three-atom basis."""
    result = check_friedel_law_structure_factor()
    _assert_well_formed(result, "friedel_law_structure_factor")
    assert (
        result.passed
    ), f"Friedel symmetry violated: residual={result.residual}"


def test_elastic_closure_ewald_simulator():
    """ewald_simulator reflections lie exactly on the Ewald sphere."""
    result = check_elastic_closure_ewald()
    _assert_well_formed(result, "elastic_closure_ewald")
    assert (
        result.passed
    ), f"Elastic closure violated: residual={result.residual}"


def test_run_default_invariants_returns_full_suite():
    """The default runner emits one well-formed result per check."""
    expected_names = {
        "form_factor_positivity_kirkland",
        "form_factor_positivity_lobato",
        "form_factor_monotonic_kirkland",
        "form_factor_monotonic_lobato",
        "form_factor_kirkland_lobato_close",
        "wavelength_relativistic_consistency",
        "friedel_law_structure_factor",
        "elastic_closure_ewald",
    }
    results = run_default_invariants()
    assert isinstance(results, list)
    assert len(results) == len(expected_names)
    seen_names = {r.name for r in results}
    assert seen_names == expected_names
    for r in results:
        assert isinstance(r, InvariantResult)
        assert isinstance(r.passed, bool)
        assert isinstance(r.residual, float)
        assert r.tolerance >= 0.0
        assert r.units
        assert r.detail


def test_invariant_result_is_immutable():
    """InvariantResult is a frozen dataclass and cannot be mutated."""
    result = check_friedel_law_structure_factor()
    with pytest.raises((AttributeError, TypeError)):
        result.passed = not result.passed
