"""Smoke tests for top-level automaton scripts."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
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
_LOOP_B_REPRO_SCRIPTS: tuple[Path, ...] = (
    _AUTOMATON_DIR / "forward_kinematic.py",
    _AUTOMATON_DIR / "forward_multislice.py",
    _AUTOMATON_DIR / "forward_reflection.py",
    _AUTOMATON_DIR / "match_measured_to_simulated.py",
    _AUTOMATON_DIR / "screen_xyz_ensemble.py",
)
_KINEMATIC_EXPORT_PROBE: str = textwrap.dedent(
    """
    import json

    import jax
    import jax.numpy as jnp
    import rheedium as rh
    from rheedium.tools import (
        deserialize_exported,
        export_forward,
        serialize_exported,
    )
    from rheedium.types import CrystalStructure


    def forward_sum(frac, cart, lengths, angles):
        crystal = CrystalStructure(
            frac_positions=frac,
            cart_positions=cart,
            cell_lengths=lengths,
            cell_angles=angles,
        )
        pattern = rh.simul.ewald_simulator(
            crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
        )
        return jnp.sum(pattern.intensities)


    (n,) = jax.export.symbolic_shape("n")
    exported = export_forward(
        forward_sum,
        jax.ShapeDtypeStruct((n, 4), jnp.float64),
        jax.ShapeDtypeStruct((n, 4), jnp.float64),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jax.ShapeDtypeStruct((3,), jnp.float64),
    )
    blob = serialize_exported(exported)
    reloaded = deserialize_exported(blob)

    frac2 = jnp.array([[0.0, 0.0, 0.0, 12.0], [0.5, 0.5, 0.5, 8.0]])
    cart2 = jnp.array([[0.0, 0.0, 0.0, 12.0], [2.1, 2.1, 2.1, 8.0]])
    frac3 = jnp.array(
        [
            [0.0, 0.0, 0.0, 12.0],
            [0.5, 0.5, 0.5, 8.0],
            [0.25, 0.25, 0.25, 12.0],
        ]
    )
    cart3 = jnp.array(
        [
            [0.0, 0.0, 0.0, 12.0],
            [2.1, 2.1, 2.1, 8.0],
            [1.05, 1.05, 1.05, 12.0],
        ]
    )
    lengths = jnp.array([4.21, 4.21, 4.21])
    angles = jnp.array([90.0, 90.0, 90.0])
    result = {
        "atom_counts": [2, 3],
        "blob_bytes": len(blob),
        "n2": float(reloaded.call(frac2, cart2, lengths, angles)),
        "n3": float(reloaded.call(frac3, cart3, lengths, angles)),
    }
    print(json.dumps(result, sort_keys=True))
    """
)


def _last_json(stdout: str) -> dict[str, Any]:
    """Parse the last stdout line as a JSON object."""
    return json.loads(stdout.strip().splitlines()[-1])


def _run_automaton(script: Path, outdir: Path) -> dict[str, Any]:
    """Run one automaton in smoke mode and parse its result JSON."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with-editable",
            ".",
            str(script.relative_to(_REPO_ROOT)),
            "--smoke",
            "--seed",
            "123",
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
    return _last_json(result.stdout)


def _artifact_manifest(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Return deterministic artifact fields from a result payload."""
    return [
        {
            "role": str(artifact["role"]),
            "mime": str(artifact["mime"]),
            "path": str(artifact["path"]),
        }
        for artifact in payload["artifacts"]
    ]


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
    payload = _run_automaton(script, outdir)
    assert payload["status"] == "ok"
    assert isinstance(payload["experiment"], str)
    assert payload["experiment"]
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
    parses the final JSON line, and checks for the canonical schema version
    plus non-empty parameter metadata.
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
    assert payload["params"]
    assert payload["params_schema"]["properties"]


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


@pytest.mark.parametrize(
    "script",
    _LOOP_B_REPRO_SCRIPTS,
    ids=[path.name for path in _LOOP_B_REPRO_SCRIPTS],
)
def test_loop_b_smoke_runs_are_reproducible(
    script: Path,
    tmp_path: Path,
) -> None:
    r"""Loop B smoke runs are reproducible for a fixed seed and request.

    Extended Summary
    ----------------
    Verifies the G3 reproducibility gate for the theory-in-the-loop
    automatons: identical script, arguments, output directory, seed, and
    rheedium version should emit the same deterministic science payload.

    Notes
    -----
    It runs each Loop B automaton twice in smoke mode through the normal
    ``uv run --with-editable .`` entry point, then compares the result key,
    metrics, artifact manifest, and any ranked rows while intentionally
    ignoring wall-clock runtime and binary preview encoding.
    """
    outdir: Path = tmp_path / script.stem

    first = _run_automaton(script, outdir)
    second = _run_automaton(script, outdir)

    assert first["result_key"] == second["result_key"]
    assert first["metrics"] == second["metrics"]
    assert _artifact_manifest(first) == _artifact_manifest(second)
    if "ranked" in first or "ranked" in second:
        assert first["ranked"] == second["ranked"]


def test_g3_kinematic_export_runs_two_atom_counts() -> None:
    r"""One exported kinematic artifact runs over two atom counts.

    Extended Summary
    ----------------
    Verifies the G3 export gate for Loop B: the kinematic Ewald forward path is
    lowered once with a symbolic atom-count axis, serialized as a StableHLO
    artifact, reloaded, and called for two different structure sizes.

    Notes
    -----
    The export probe runs in a fresh subprocess with ``EQX_ON_ERROR=nan`` set
    before importing rheedium, matching the documented requirement for
    stripping Equinox runtime-check host callbacks from exported artifacts.
    """
    env: dict[str, str] = dict(os.environ)
    env["EQX_ON_ERROR"] = "nan"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with-editable",
            ".",
            "python",
            "-c",
            _KINEMATIC_EXPORT_PROBE,
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = _last_json(result.stdout)
    assert payload["atom_counts"] == [2, 3]
    assert payload["blob_bytes"] > 0
    assert payload["n2"] > 0.0
    assert payload["n3"] > 0.0
