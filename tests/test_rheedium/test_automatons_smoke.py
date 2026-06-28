"""Smoke tests for top-level automaton scripts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT: Path = Path(__file__).parents[2]
_AUTOMATON_DIR: Path = _REPO_ROOT / "automatons"
_EXPERIMENT_SCRIPTS: list[Path] = [
    path
    for path in sorted(_AUTOMATON_DIR.glob("*.py"))
    if path.name != "bump_pin.py"
]


def _last_json(stdout: str) -> dict[str, Any]:
    """Parse the last stdout line as a JSON object."""
    return json.loads(stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "script",
    _EXPERIMENT_SCRIPTS,
    ids=[path.name for path in _EXPERIMENT_SCRIPTS],
)
def test_automaton_smoke_run(script: Path, tmp_path: Path) -> None:
    r"""Each experiment automaton runs under ``uv run`` in smoke mode.

    Extended Summary
    ----------------
    Verifies the executable handoff contract for top-level automaton scripts:
    each experiment script must run from its PEP 723 entry point, exit
    successfully, and emit a final JSON result with declared artifacts.

    Notes
    -----
    It launches the script with ``uv run --with-editable .`` so the test uses
    the working tree while still exercising the same command shape an agent
    uses. The test then parses the final stdout line and checks every artifact
    path against the requested output directory.
    """
    outdir: Path = tmp_path / script.stem
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with-editable",
            ".",
            str(script.relative_to(_REPO_ROOT)),
            "--smoke",
            "--outdir",
            str(outdir),
            "--json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = _last_json(result.stdout)
    assert payload["status"] == "ok"
    assert payload["experiment"] == "forward-kinematic"
    assert payload["artifacts"]
    for artifact in payload["artifacts"]:
        assert (outdir / str(artifact["path"])).exists()


@pytest.mark.parametrize(
    "script",
    _EXPERIMENT_SCRIPTS,
    ids=[path.name for path in _EXPERIMENT_SCRIPTS],
)
def test_automaton_describe(script: Path) -> None:
    r"""Each experiment automaton exposes an introspection schema.

    Extended Summary
    ----------------
    Verifies that every experiment script supports ``--describe`` and returns
    the machine-readable parameter schema needed by automated callers.

    Notes
    -----
    It invokes each script through ``uv run --with-editable . --describe``,
    parses the final JSON line, and checks for the canonical schema version and
    the exemplar kinematic beam-energy parameter.
    """
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with-editable",
            ".",
            str(script.relative_to(_REPO_ROOT)),
            "--describe",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = _last_json(result.stdout)
    assert payload["schema_version"] == "rheedium.automaton.describe.v1"
    assert payload["params_schema"]["type"] == "object"
    assert "energy_kev" in payload["params_schema"]["properties"]


def test_automaton_index_mentions_scripts() -> None:
    r"""The human-readable index lists every top-level automaton script.

    Extended Summary
    ----------------
    Verifies that ``automatons/INDEX.md`` stays synchronized with the script
    files present in the automaton directory.

    Notes
    -----
    It reads the committed Markdown index and asserts that every ``*.py`` file
    in ``automatons/`` appears by filename, including operational helpers such
    as the pin rewriter.
    """
    index_text: str = (_AUTOMATON_DIR / "INDEX.md").read_text(encoding="utf-8")
    for path in sorted(_AUTOMATON_DIR.glob("*.py")):
        assert path.name in index_text
