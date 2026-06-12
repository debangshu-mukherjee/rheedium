"""Tests for the rheedium.audit physics-invariant suite."""

from __future__ import annotations

from typing import Any

import pytest
from jaxtyping import Array, Float

from rheedium.audit.invariants import (
    InvariantResult,
    check_elastic_closure_ewald,
    check_form_factor_monotonic_decrease,
    check_form_factor_positivity,
    check_friedel_law_structure_factor,
    check_wavelength_relativistic_consistency,
    run_default_invariants,
)


def _assert_well_formed(result: InvariantResult, expected_name: str) -> None:
    """Assert an InvariantResult has all fields populated correctly."""
    assert isinstance(result, InvariantResult)
    assert result.name == expected_name
    assert isinstance(result.passed, bool)
    assert isinstance(result.residual, float)
    assert isinstance(result.tolerance, float)
    assert isinstance(result.units, str)
    assert result.units
    assert isinstance(result.detail, str)
    assert result.detail


def test_form_factor_positivity_both_parameterizations() -> None:
    """Both Kirkland and Lobato form factors stay positive."""
    kirkland_result: Float[Array, "..."]
    lobato_result: Float[Array, "..."]
    kirkland_result, lobato_result = check_form_factor_positivity()
    _assert_well_formed(kirkland_result, "form_factor_positivity_kirkland")
    _assert_well_formed(lobato_result, "form_factor_positivity_lobato")
    assert kirkland_result.passed, (
        "Kirkland form factor went negative: "
        f"residual={kirkland_result.residual}"
    )
    assert lobato_result.passed, (
        f"Lobato form factor went negative: residual={lobato_result.residual}"
    )


def test_form_factor_monotonic_decrease_both_parameterizations() -> None:
    """Both Kirkland and Lobato form factors decrease monotonically."""
    kirkland_result: Float[Array, "..."]
    lobato_result: Float[Array, "..."]
    kirkland_result, lobato_result = check_form_factor_monotonic_decrease()
    _assert_well_formed(kirkland_result, "form_factor_monotonic_kirkland")
    _assert_well_formed(lobato_result, "form_factor_monotonic_lobato")
    assert kirkland_result.passed, (
        f"Kirkland not monotonic: residual={kirkland_result.residual}"
    )
    assert lobato_result.passed, (
        f"Lobato not monotonic: residual={lobato_result.residual}"
    )


def test_wavelength_relativistic_consistency() -> None:
    """rheedium.tools.wavelength_ang matches CODATA-derived de Broglie."""
    result: Float[Array, "..."] = check_wavelength_relativistic_consistency()
    _assert_well_formed(result, "wavelength_relativistic_consistency")
    assert result.passed, (
        f"Wavelength mismatch vs CODATA: residual={result.residual}"
    )


def test_friedel_law_structure_factor() -> None:
    """I(G) = I(-G) for a non-centrosymmetric three-atom basis."""
    result: Float[Array, "..."] = check_friedel_law_structure_factor()
    _assert_well_formed(result, "friedel_law_structure_factor")
    assert result.passed, (
        f"Friedel symmetry violated: residual={result.residual}"
    )


def test_elastic_closure_ewald_simulator() -> None:
    """ewald_simulator reflections lie exactly on the Ewald sphere."""
    result: Float[Array, "..."] = check_elastic_closure_ewald()
    _assert_well_formed(result, "elastic_closure_ewald")
    assert result.passed, (
        f"Elastic closure violated: residual={result.residual}"
    )


def test_run_default_invariants_returns_full_suite() -> None:
    """The default runner emits one well-formed result per check."""
    expected_names: Float[Array, "..."] = {
        "form_factor_positivity_kirkland",
        "form_factor_positivity_lobato",
        "form_factor_monotonic_kirkland",
        "form_factor_monotonic_lobato",
        "wavelength_relativistic_consistency",
        "friedel_law_structure_factor",
        "elastic_closure_ewald",
    }
    results: Float[Array, "..."] = run_default_invariants()
    assert isinstance(results, list)
    assert len(results) == len(expected_names)
    seen_names: Any = {r.name for r in results}
    assert seen_names == expected_names
    r: Any
    for r in results:
        assert isinstance(r, InvariantResult)
        assert isinstance(r.passed, bool)
        assert isinstance(r.residual, float)
        assert r.tolerance >= 0.0
        assert r.units
        assert r.detail


def test_invariant_result_is_immutable() -> None:
    """InvariantResult is a frozen dataclass and cannot be mutated."""
    result: Float[Array, "..."] = check_friedel_law_structure_factor()
    with pytest.raises((AttributeError, TypeError)):
        result.passed = not result.passed
