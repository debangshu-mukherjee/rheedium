"""Contract tests for :mod:`rheedium.harness.automaton`.

Extended Summary
----------------
These tests exercise the automaton process boundary implemented in
:mod:`rheedium.harness.automaton` through the public
:mod:`rheedium.harness` package facade, which re-exports it. They cover the
package re-export contract, the ``--describe`` schema, machine-input
precedence, pre-flight validation, structured error emission, and
deterministic result keys.

Notes
-----
:see: rheedium.harness.automaton
:see: rheedium.harness
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rheedium import harness
from rheedium.harness import (
    DESCRIBE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    Param,
    automaton,
    experiment,
)


def _last_json(stdout: str) -> dict[str, Any]:
    """Parse the last stdout line as JSON."""
    return json.loads(stdout.strip().splitlines()[-1])


def _dummy_experiment() -> Any:
    """Create a tiny harness-backed experiment for tests."""

    @experiment(
        name="dummy",
        params=[
            Param(
                "scale",
                float,
                default=1.0,
                help="Scale factor.",
                bounds=(0.0, 10.0),
            ),
            Param("count", int, default=1, help="Loop count."),
            Param("label", str, default="base", help="Run label."),
        ],
        returns={"metrics": {"scale": {"type": "number"}}},
    )
    def main(args: SimpleNamespace, ctx: Any) -> dict[str, Any]:
        """Run a tiny dummy automaton."""
        artifact = ctx.save_json(
            "payload.json",
            {"scale": args.scale, "count": args.count},
        )
        return {
            "metrics": {
                "scale": args.scale,
                "count": args.count,
                "label": args.label,
            },
            "artifacts": [artifact],
        }

    return main


def test_package_reexports_automaton_surface() -> None:
    r"""The package facade re-exports the implementation module verbatim.

    Extended Summary
    ----------------
    Guards the package split: :mod:`rheedium.harness` is a thin facade whose
    public names are defined in :mod:`rheedium.harness.automaton`. Each name in
    the package ``__all__`` must resolve to the identical object on the
    implementation module, so importing from either location is equivalent.

    Notes
    -----
    It iterates over ``rheedium.harness.__all__`` and asserts each name is
    present on both modules and bound to the same object via identity.
    """
    for name in harness.__all__:
        assert hasattr(automaton, name), f"{name} missing from automaton"
        assert getattr(harness, name) is getattr(automaton, name)


def test_describe_emits_parameter_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""``--describe`` emits the agent-facing schema document.

    Extended Summary
    ----------------
    Verifies that a harness-decorated experiment exposes the declared
    parameter metadata and return schema without running the experiment body.

    Notes
    -----
    It calls the wrapped dummy experiment with ``--describe``, parses the final
    stdout JSON line, and checks the schema version plus representative nested
    fields from the parameter and return declarations.
    """
    main = _dummy_experiment()

    assert main(["--describe"]) == 0

    payload = _last_json(capsys.readouterr().out)
    assert payload["schema_version"] == DESCRIBE_SCHEMA_VERSION
    assert payload["experiment"] == "dummy"
    assert payload["params_schema"]["properties"]["scale"]["maximum"] == 10.0
    assert payload["returns"]["metrics"]["scale"]["type"] == "number"


def test_params_json_merges_with_cli_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""Explicit CLI flags override values supplied by ``--params``.

    Extended Summary
    ----------------
    Verifies the machine-input precedence rule: JSON supplied through
    ``--params`` forms the base parameter set, and explicit CLI flags override
    individual values before validation and execution.

    Notes
    -----
    It runs the dummy experiment with an inline JSON document and an overriding
    ``--scale`` flag, then inspects the emitted result and the JSON artifact
    written beneath the temporary output directory.
    """
    main = _dummy_experiment()
    params_doc = json.dumps({"scale": 2.0, "count": 4, "label": "json"})

    code = main(
        [
            "--params",
            params_doc,
            "--scale",
            "3.5",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["params"]["scale"] == 3.5
    assert payload["params"]["count"] == 4
    assert payload["metrics"]["label"] == "json"
    artifact = tmp_path / payload["artifacts"][0]["path"]
    assert artifact.exists()


def test_validate_mode_does_not_call_body(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""``--validate`` checks params without running the experiment body.

    Extended Summary
    ----------------
    Verifies the pre-flight validation path used by agents before launching an
    expensive pipeline: valid parameters should return an ``ok`` result without
    invoking experiment code.

    Notes
    -----
    It decorates a body that would raise if executed, calls it with
    ``--validate``, and confirms the result contains the validated parameters
    plus the lightweight ``{"valid": True}`` metric.
    """

    @experiment(
        name="validate-only",
        params=[Param("value", int, help="Required integer.")],
    )
    def main(args: SimpleNamespace, ctx: Any) -> dict[str, Any]:
        """Raise if validation accidentally executes the body."""
        del args, ctx
        raise AssertionError("body should not run")

    code = main(["--value", "7", "--validate", "--outdir", str(tmp_path)])

    assert code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["metrics"] == {"valid": True}
    assert payload["params"] == {"value": 7}


def test_param_bounds_emit_structured_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""Out-of-range parameters emit a structured error result.

    Extended Summary
    ----------------
    Verifies that semantic ``Param.bounds`` metadata is enforced before the
    experiment body runs and that failures are categorized for agent handling.

    Notes
    -----
    It passes a scale value above the declared maximum, then checks for the
    validation exit code, ``ParamOutOfRange`` error kind, and offending field
    in the emitted JSON result.
    """
    main = _dummy_experiment()

    code = main(["--scale", "11.0", "--outdir", str(tmp_path)])

    assert code == 2
    payload = _last_json(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error_kind"] == "ParamOutOfRange"
    assert payload["field"] == "scale"


def test_result_key_is_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""Identical params and seed produce the same result key.

    Extended Summary
    ----------------
    Verifies that the harness derives stable idempotency keys from the
    experiment name, validated parameters, seed, and rheedium version.

    Notes
    -----
    It runs the same dummy experiment twice with identical arguments, parses
    both result payloads, and compares the emitted ``result_key`` values.
    """
    main = _dummy_experiment()
    argv = ["--scale", "2.0", "--seed", "9", "--outdir", str(tmp_path)]

    assert main(argv) == 0
    first = _last_json(capsys.readouterr().out)
    assert main(argv) == 0
    second = _last_json(capsys.readouterr().out)

    assert first["result_key"] == second["result_key"]


def test_emit_sanitizes_nonfinite_floats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""``emit`` maps JSON-forbidden NaN and infinity values to null.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: ``emit`` maps
    JSON-forbidden NaN and infinity values to null.

    Notes
    -----
    It constructs the representative inputs inside the test body,
    keeping the fixture and assertion path local to the documented case.
    """
    harness.emit(
        {
            "metric": float("nan"),
            "nested": [float("inf"), float("-inf"), 1.0],
        }
    )

    payload = _last_json(capsys.readouterr().out)
    assert payload == {"metric": None, "nested": [None, None, 1.0]}


def test_unchecked_reexecs_with_runtime_check_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""``--unchecked`` re-execs once with the import-time env var set.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: ``--unchecked``
    re-execs once with the import-time env var set.

    Notes
    -----
    It constructs the representative inputs inside the test body,
    keeping the fixture and assertion path local to the documented case.
    """
    main = _dummy_experiment()
    calls: list[tuple[str, list[str]]] = []

    def _fake_execv(executable: str, argv: list[str]) -> None:
        calls.append((executable, argv))
        raise RuntimeError("execv called")

    monkeypatch.delenv("RHEEDIUM_DISABLE_RUNTIME_CHECKS", raising=False)
    monkeypatch.setattr(automaton.os, "execv", _fake_execv)
    monkeypatch.setattr(automaton.sys, "argv", ["dummy.py", "--unchecked"])

    with pytest.raises(RuntimeError, match="execv called"):
        main(["--unchecked"])

    assert automaton.os.environ["RHEEDIUM_DISABLE_RUNTIME_CHECKS"] == "1"
    assert calls == [
        (
            automaton.sys.executable,
            [automaton.sys.executable, "dummy.py", "--unchecked"],
        )
    ]


def test_unchecked_env_guard_prevents_reexec_loop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The env guard lets the re-executed process continue normally.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: The env guard
    lets the re-executed process continue normally.

    Notes
    -----
    It constructs the representative inputs inside the test body,
    keeping the fixture and assertion path local to the documented case.
    """
    main = _dummy_experiment()

    def _unexpected_execv(executable: str, argv: list[str]) -> None:
        del executable, argv
        raise AssertionError("execv should not be called")

    monkeypatch.setenv("RHEEDIUM_DISABLE_RUNTIME_CHECKS", "1")
    monkeypatch.setattr(automaton.os, "execv", _unexpected_execv)

    code = main(["--unchecked", "--validate", "--outdir", str(tmp_path)])

    assert code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["status"] == "ok"
