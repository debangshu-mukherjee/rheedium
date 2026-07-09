# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.15"]
# ///
"""Run rheedium's default physics-invariant audit suite.

The automaton is a thin process wrapper around ``rheedium.audit``. It runs the
stateless invariant checks, serializes a pass/fail table, and emits aggregate
residual metrics that can be used as a CI or agent-side health signal.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from beartype.typing import Any

import rheedium as rh
from rheedium.audit import InvariantResult
from rheedium.harness import Param, experiment


def _result_rows(results: list[InvariantResult]) -> list[dict[str, Any]]:
    """Convert invariant dataclasses to JSON-safe rows."""
    return [asdict(result) for result in results]


def _residual_ratios(results: list[InvariantResult]) -> np.ndarray[Any, Any]:
    """Return residual/tolerance ratios with exact-zero tolerances handled."""
    ratios: list[float] = []
    for result in results:
        if result.tolerance > 0.0:
            ratios.append(result.residual / result.tolerance)
        elif result.residual == 0.0:
            ratios.append(0.0)
        else:
            ratios.append(float("inf"))
    return np.asarray(ratios, dtype=np.float64)


@experiment(
    name="audit-invariants",
    params=[
        Param(
            "fail_on_violation",
            bool,
            default=False,
            help="Raise an error if any invariant fails.",
        ),
    ],
    returns={
        "metrics": {
            "n_invariants": {"type": "integer"},
            "n_passed": {"type": "integer"},
            "all_passed": {"type": "boolean"},
            "max_residual_ratio": {"type": "number"},
        },
        "artifacts": {"roles": ["audit_report", "audit_arrays"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the default invariant suite and emit an audit report."""
    results: list[InvariantResult] = rh.audit.run_default_invariants()
    rows: list[dict[str, Any]] = _result_rows(results)
    passed: list[bool] = [result.passed for result in results]
    residual_ratios: np.ndarray[Any, Any] = _residual_ratios(results)
    all_passed: bool = all(passed)
    if args.fail_on_violation and not all_passed:
        failed_names: list[str] = [
            result.name for result in results if not result.passed
        ]
        raise RuntimeError(f"invariant violation(s): {failed_names}")

    report_artifact = ctx.save_json(
        "audit_invariants.json",
        {"results": rows, "all_passed": all_passed},
        role="audit_report",
    )
    array_artifact = ctx.save_array(
        "audit_invariants.npz",
        {
            "name": np.asarray([result.name for result in results]),
            "passed": np.asarray(passed, dtype=np.bool_),
            "residual": np.asarray(
                [result.residual for result in results],
                dtype=np.float64,
            ),
            "tolerance": np.asarray(
                [result.tolerance for result in results],
                dtype=np.float64,
            ),
            "residual_ratio": residual_ratios,
        },
        role="audit_arrays",
    )
    metrics: dict[str, Any] = {
        "n_invariants": len(results),
        "n_passed": int(sum(passed)),
        "n_failed": int(len(results) - sum(passed)),
        "all_passed": all_passed,
        "max_residual_ratio": float(np.max(residual_ratios)),
    }
    return {
        "metrics": metrics,
        "artifacts": [report_artifact, array_artifact],
        "invariants": rows,
    }


if __name__ == "__main__":
    main()
