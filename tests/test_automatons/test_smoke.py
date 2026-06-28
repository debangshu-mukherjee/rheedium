"""Smoke tests for top-level automaton scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT: Path = Path(__file__).parents[2]
_AUTOMATON_DIR: Path = _REPO_ROOT / "automatons"
_AGENT_GUIDE: Path = (
    _REPO_ROOT / "docs/source/guides/running-experiments-as-an-agent.md"
)
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
_LOOP_C_GATE_SCRIPTS: tuple[Path, ...] = (
    _AUTOMATON_DIR / "fit_orientation_beam.py",
    _AUTOMATON_DIR / "reconstruct_distribution.py",
    _AUTOMATON_DIR / "invert_structure.py",
    _AUTOMATON_DIR / "recipe_deviation.py",
)
_LOOP_A_GATE_SCRIPTS: tuple[Path, ...] = (
    _AUTOMATON_DIR / "rheed_ingest.py",
    _AUTOMATON_DIR / "growth_monitor.py",
)
_G6_GATE_SCRIPTS: tuple[Path, ...] = (
    _AUTOMATON_DIR / "azimuthal_sweep.py",
    _AUTOMATON_DIR / "parameter_grid.py",
    _AUTOMATON_DIR / "ensemble_average.py",
    _AUTOMATON_DIR / "reconstruct_orientation.py",
    _AUTOMATON_DIR / "convergence_study.py",
)
_G7_GATE_SCRIPTS: tuple[Path, ...] = (
    _AUTOMATON_DIR / "audit_invariants.py",
    _AUTOMATON_DIR / "export_model.py",
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


def _automaton_command(script: Path, *args: str) -> list[str]:
    """Return the local editable command for an automaton script."""
    return [
        "uv",
        "run",
        "--with-editable",
        ".",
        "python",
        str(script.relative_to(_REPO_ROOT)),
        *args,
    ]


def _run_automaton(script: Path, outdir: Path) -> dict[str, Any]:
    """Run one automaton in smoke mode and parse its result JSON."""
    result = subprocess.run(
        _automaton_command(
            script,
            "--smoke",
            "--seed",
            "123",
            "--outdir",
            str(outdir),
            "--json",
        ),
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
    each experiment script must run against the local editable tree, exit
    successfully, and emit a final JSON result with declared artifacts.

    Notes
    -----
    It launches the script with ``uv run --with-editable . python`` so tests
    use the working tree as the source of truth even when the PEP 723 pin has
    been bumped ahead of PyPI publication. The test then parses the final
    stdout line and checks every artifact path against the requested output
    directory.
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
    It invokes each script through the local editable harness, parses the final
    JSON line, and checks for the canonical schema version plus non-empty
    parameter metadata.
    """
    result = subprocess.run(
        _automaton_command(script, "--describe"),
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
    It reads the committed Markdown index and asserts that the backticked
    ``*.py`` entries exactly match the files in ``automatons/``, including
    operational helpers such as the pin rewriter.
    """
    index_text: str = (_AUTOMATON_DIR / "INDEX.md").read_text(encoding="utf-8")
    actual_scripts: set[str] = {
        path.name for path in sorted(_AUTOMATON_DIR.glob("*.py"))
    }
    indexed_scripts: set[str] = set(re.findall(r"`([^`]+\.py)`", index_text))
    assert indexed_scripts == actual_scripts


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
    It runs each Loop B automaton twice in smoke mode through the local
    editable command, then compares the result key, metrics, artifact manifest,
    and any ranked rows while intentionally ignoring wall-clock runtime and
    binary preview encoding.
    """
    outdir: Path = tmp_path / script.stem

    first = _run_automaton(script, outdir)
    second = _run_automaton(script, outdir)

    assert first["params"] == second["params"]
    assert first["rheedium_version"] == second["rheedium_version"]
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


@pytest.mark.parametrize(
    "script",
    _LOOP_C_GATE_SCRIPTS,
    ids=[path.name for path in _LOOP_C_GATE_SCRIPTS],
)
def test_g5_loop_c_smoke_recovers_planted_signal(
    script: Path,
    tmp_path: Path,
) -> None:
    r"""Loop C recon automatons recover their planted smoke fixtures.

    Extended Summary
    ----------------
    Verifies the G5 gate for inversion automatons: the four thin wrappers over
    ``rheedium.recon`` run through the normal subprocess contract and recover a
    known synthetic signal, distribution, structure latent, or recipe gap.

    Notes
    -----
    It intentionally checks science-facing metrics rather than only process
    success: fit errors must be tiny for exact linear fixtures, the planted
    distribution must reconstruct, and recipe deviation must flag the
    deliberately mismatched intended recipe as critical.
    """
    payload = _run_automaton(script, tmp_path / script.stem)
    metrics: dict[str, Any] = payload["metrics"]
    assert payload["wall_seconds"] < 60.0

    if script.name == "fit_orientation_beam.py":
        assert metrics["orientation_l2_error"] < 1e-6
        assert metrics["beam_l2_error"] < 1e-6
        assert metrics["residual_mse"] < 1e-12
    elif script.name == "reconstruct_distribution.py":
        assert metrics["weight_l1_error"] < 1e-6
        assert metrics["max_band"] > 0.0
        assert metrics["best_sample_index"] == 1
    elif script.name == "invert_structure.py":
        assert metrics["param_l2_error"] < 1e-6
        assert metrics["final_loss"] < 1e-12
        assert metrics["converged"] is True
    elif script.name == "recipe_deviation.py":
        assert metrics["severity"] == 2
        assert metrics["max_abs_z"] >= 3.0
        assert metrics["first_deviation"] == pytest.approx(0.5)
        assert metrics["first_z_score"] == pytest.approx(5.0)
    else:
        raise AssertionError(f"unexpected Loop C script: {script.name}")


@pytest.mark.parametrize(
    "script",
    _LOOP_A_GATE_SCRIPTS,
    ids=[path.name for path in _LOOP_A_GATE_SCRIPTS],
)
def test_g4_loop_a_smoke_emits_growth_state(
    script: Path,
    tmp_path: Path,
) -> None:
    r"""Loop A automatons emit the planted online-growth observables.

    Extended Summary
    ----------------
    Verifies the G4 mission-loop gate: ``rheed_ingest`` emits stateless
    per-frame observables from the committed RHEED frame fixture, and
    ``growth_monitor`` recovers the planted oscillation period from the
    committed rolling series fixture.

    Notes
    -----
    The scripts run through the same local editable smoke contract as all other
    automatons. Assertions focus on the science payload: center/spacing/state
    for the single frame, and period/roughness/transition flags for the rolling
    series.
    """
    payload = _run_automaton(script, tmp_path / script.stem)
    metrics: dict[str, Any] = payload["metrics"]

    if script.name == "rheed_ingest.py":
        assert metrics["specular_center_y"] == 12
        assert metrics["specular_center_x"] == 32
        assert metrics["specular_intensity"] > 0.0
        assert metrics["streak_spacing_px"] == pytest.approx(10.0)
        assert metrics["is_2d"] is True
        assert metrics["surface_state"] == "2d_streaky"
    elif script.name == "growth_monitor.py":
        assert metrics["n_frames"] == 24
        assert metrics["dominant_period_frames"] == pytest.approx(
            6.0,
            abs=0.1,
        )
        assert metrics["oscillation_count"] == pytest.approx(4.0, abs=0.1)
        assert metrics["roughness_trend"] == "stable"
        assert metrics["oscillation_detected"] is True
    else:
        raise AssertionError(f"unexpected Loop A script: {script.name}")


def test_g4_rheed_ingest_is_per_frame_stateless(tmp_path: Path) -> None:
    r"""Repeated RHEED frame ingest emits the same state for the same frame.

    Extended Summary
    ----------------
    Verifies the resolved open-decision #5 contract for Loop A:
    ``rheed_ingest`` is per-frame-stateless, so identical frame, params, seed,
    and rheedium version produce identical science payloads.

    Notes
    -----
    The test compares the deterministic result key, metrics, and growth-state
    payload while ignoring wall-clock runtime and binary preview encoding.
    """
    script = _AUTOMATON_DIR / "rheed_ingest.py"
    first = _run_automaton(script, tmp_path / "first")
    second = _run_automaton(script, tmp_path / "second")

    assert first["result_key"] == second["result_key"]
    assert first["metrics"] == second["metrics"]
    assert first["growth_state"] == second["growth_state"]


@pytest.mark.parametrize(
    "script",
    _G6_GATE_SCRIPTS,
    ids=[path.name for path in _G6_GATE_SCRIPTS],
)
def test_g6_diagnostic_smoke_emits_promised_metrics(
    script: Path,
    tmp_path: Path,
) -> None:
    r"""G6 diagnostics emit their ensemble and convergence gate metrics.

    Extended Summary
    ----------------
    Verifies the diagnostics/ensemble gate: sweep automatons produce
    reproducible numeric grids, ``ensemble_average`` reports mode and
    effective-count metrics, ``reconstruct_orientation`` recovers a planted
    synthetic orientation distribution, and ``convergence_study`` emits a
    monotone residual-vs-N series.

    Notes
    -----
    The test runs each script through the same subprocess smoke contract as
    every other top-level automaton, then asserts the science-facing metrics
    that distinguish G6 from a generic "process exited" check.
    """
    payload = _run_automaton(script, tmp_path / script.stem)
    metrics: dict[str, Any] = payload["metrics"]

    if script.name == "azimuthal_sweep.py":
        assert metrics["n_angles"] >= 3
        assert metrics["max_integrated_intensity"] >= 0.0
        assert len(payload["sweep"]) == metrics["n_angles"]
    elif script.name == "parameter_grid.py":
        assert metrics["n_grid_points"] >= 4
        assert metrics["grid_shape"][0] >= 2
        assert metrics["grid_shape"][1] >= 2
    elif script.name == "ensemble_average.py":
        assert metrics["mode_count"] >= 3
        assert metrics["effective_count"] > 1.0
        assert sum(payload["ensemble"]["weights"]) == pytest.approx(1.0)
    elif script.name == "reconstruct_orientation.py":
        assert metrics["weight_l1_error"] < 0.20
        assert metrics["final_loss"] < metrics["initial_loss"]
        assert metrics["gradient_finite"] is True
    elif script.name == "convergence_study.py":
        assert metrics["n_mode_counts"] >= 4
        assert metrics["residual_monotone"] is True
        assert metrics["final_residual"] < metrics["initial_residual"]
    else:
        raise AssertionError(f"unexpected G6 script: {script.name}")


@pytest.mark.parametrize(
    "script",
    _G7_GATE_SCRIPTS,
    ids=[path.name for path in _G7_GATE_SCRIPTS],
)
def test_g7_ops_smoke_emits_deployment_proofs(
    script: Path,
    tmp_path: Path,
) -> None:
    r"""G7 ops automatons emit audit and StableHLO deployment proofs.

    Extended Summary
    ----------------
    Verifies the operations/hardening gate: ``audit_invariants`` reports a
    passing physics-invariant suite, and ``export_model`` writes a serialized
    StableHLO artifact that reloads in a separate process and matches the
    in-process kinematic forward call.

    Notes
    -----
    It runs both scripts through the same subprocess smoke contract as the
    user-facing automata so the proof covers PEP 723 execution, working-tree
    overrides, artifact emission, and final-line result JSON together.
    """
    payload = _run_automaton(script, tmp_path / script.stem)
    metrics: dict[str, Any] = payload["metrics"]

    if script.name == "audit_invariants.py":
        assert metrics["n_invariants"] >= 7
        assert metrics["all_passed"] is True
        assert metrics["n_failed"] == 0
        assert len(payload["invariants"]) == metrics["n_invariants"]
    elif script.name == "export_model.py":
        assert metrics["artifact_bytes"] > 0
        assert metrics["separate_process_ok"] is True
        assert metrics["atom_counts_verified"] == [2, 3]
        assert metrics["same_result_max_abs_error"] < 1e-10
        assert (
            payload["export"]["rheedium_version"]
            == payload["rheedium_version"]
        )
        artifact_paths = {
            artifact["role"]: artifact["path"]
            for artifact in payload["artifacts"]
        }
        stablehlo_path = (
            tmp_path / script.stem / artifact_paths["stablehlo_artifact"]
        )
        assert stablehlo_path.suffix == ".stablehlo"
        assert stablehlo_path.stat().st_size == metrics["artifact_bytes"]
    else:
        raise AssertionError(f"unexpected G7 script: {script.name}")


def test_g7_bump_pin_rewrites_and_is_idempotent(tmp_path: Path) -> None:
    r"""The release pin rewriter updates every automaton header once.

    Extended Summary
    ----------------
    Verifies the G7 pin-hardening requirement without mutating the real
    automaton directory: a temporary automaton root with plain and CUDA pins is
    rewritten to one pinned ``rheedium`` version, and a second run is a no-op.

    Notes
    -----
    The test executes ``automatons/bump_pin.py`` through ``uv run`` so the
    script path and command shape match release automation rather than relying
    on an import-only unit test.
    """
    root = tmp_path / "automatons"
    root.mkdir()
    first = root / "first.py"
    second = root / "second.py"
    first.write_text(
        '"rheedium==2026.1.1"\n"numpy==2.0.0"\n',
        encoding="utf-8",
    )
    second.write_text(
        '"rheedium[cuda]==2026.1.1"\n',
        encoding="utf-8",
    )

    command = [
        "uv",
        "run",
        str((_AUTOMATON_DIR / "bump_pin.py").relative_to(_REPO_ROOT)),
        "2026.6.8",
        "--root",
        str(root),
    ]
    first_run = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    second_run = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert "updated 2 file(s)" in first_run.stdout
    assert "updated 0 file(s)" in second_run.stdout
    assert '"rheedium==2026.6.8"' in first.read_text(encoding="utf-8")
    assert '"rheedium==2026.6.8"' in second.read_text(encoding="utf-8")


def test_g7_automatons_are_not_packaged_in_wheel(tmp_path: Path) -> None:
    r"""The top-level automaton directory is excluded from the wheel.

    Extended Summary
    ----------------
    Verifies the G7 wheel-exclusion gate: experiment scripts are tracked in
    the repository and executable via PEP 723, but remain outside the packaged
    ``rheedium`` wheel.

    Notes
    -----
    It builds a wheel into a temporary directory and inspects the archive
    member names directly, avoiding assumptions about the generated wheel
    filename.
    """
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
    assert not any(name.startswith("automatons/") for name in names)


def test_g7_automatons_smoke_ci_is_blocking() -> None:
    r"""The CI automaton smoke job is no longer informational.

    Extended Summary
    ----------------
    Verifies the repo-side half of making ``automatons-smoke`` a required
    branch status: the workflow job does not use ``continue-on-error`` and its
    display name no longer labels it informational.

    Notes
    -----
    GitHub branch-protection settings live outside the repository, so this
    test guards the workflow configuration that makes the status check
    enforceable.
    """
    workflow = (_REPO_ROOT / ".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"\n  automatons-smoke:\n(?P<body>.*?)(?=\n  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
        flags=re.DOTALL,
    )
    assert match is not None
    job_body = match.group("body")
    assert "continue-on-error" not in job_body
    assert "informational" not in job_body.lower()


def test_g8_agent_guide_exists_with_contract_sections() -> None:
    r"""The agent-running guide documents the automaton handoff contract.

    Extended Summary
    ----------------
    Verifies the G8 guide exists and covers the concrete invocation topics an
    automated caller needs: discovery, parameter introspection, structured
    inputs, result JSON, smoke mode, and the three lab loops.

    Notes
    -----
    This is intentionally a lightweight source-level documentation contract;
    the Sphinx build test covers rendering.
    """
    text = _AGENT_GUIDE.read_text(encoding="utf-8")
    required_terms = [
        "# Running Experiments as an Agent",
        "--describe",
        "--params",
        "--json",
        "--smoke",
        "--with-editable",
        "schema/automaton_result.schema.json",
        "Loop A",
        "Loop B",
        "Loop C",
    ]
    for term in required_terms:
        assert term in text


def test_g8_agent_guide_registered_in_guides_index() -> None:
    r"""The guide landing page links the agent-running guide.

    Extended Summary
    ----------------
    Verifies that ``docs/source/guides/index.md`` exposes the new guide in the
    human-facing guide overview table.

    Notes
    -----
    The root toctree is checked separately so navigation and landing-page
    discoverability fail independently.
    """
    index_text = (_REPO_ROOT / "docs/source/guides/index.md").read_text(
        encoding="utf-8"
    )
    assert (
        "[Running Experiments as an Agent](running-experiments-as-an-agent.md)"
    ) in index_text


def test_g8_agent_guide_registered_in_root_toctree() -> None:
    r"""The root Sphinx toctree includes the agent-running guide.

    Extended Summary
    ----------------
    Verifies that the new guide participates in the rendered documentation
    navigation instead of existing only as an orphan Markdown file.

    Notes
    -----
    The check reads ``docs/source/index.rst`` directly because that file owns
    the hidden Guides toctree used by the HTML sidebar.
    """
    root_index = (_REPO_ROOT / "docs/source/index.rst").read_text(
        encoding="utf-8"
    )
    assert "guides/running-experiments-as-an-agent" in root_index


def test_g8_agent_guide_points_to_live_automaton_catalogs() -> None:
    r"""The guide points agents at the live automaton catalog files.

    Extended Summary
    ----------------
    Verifies that the guide links to the checked-in catalog and README rather
    than duplicating a stale list of runnable scripts.

    Notes
    -----
    ``test_automaton_index_mentions_scripts`` remains the authoritative
    directory-sync check for the catalog itself.
    """
    text = _AGENT_GUIDE.read_text(encoding="utf-8")
    assert "automatons/INDEX.md" in text
    assert "automatons/README.md" in text
    assert "automatons/export_model.py" in text


def test_g8_automatons_plan_graduated_to_implemented() -> None:
    r"""The automaton plan has graduated out of ``plans/partial``.

    Extended Summary
    ----------------
    Verifies the final G8 bookkeeping: after the docs gate closes, the plan
    lives under ``plans/implemented`` and records A0 plus G0-G8 closure.

    Notes
    -----
    This guards against future edits reviving the old partial-plan path after
    the implementation has landed.
    """
    partial = _REPO_ROOT / "plans/partial/automatons_plan.md"
    implemented = _REPO_ROOT / "plans/implemented/automatons_plan.md"
    assert not partial.exists()
    assert implemented.exists()
    text = implemented.read_text(encoding="utf-8")
    assert "Status: **implemented" in text
    assert "A0 + G0-G8 closed" in text
    assert "68 `tests/test_automatons` tests passing" in text


def test_pins_match_pyproject_version() -> None:
    r"""Every automaton PEP 723 pin matches the current package version.

    Extended Summary
    ----------------
    Verifies the CONTRIBUTING *Versioning & release pins* rule: each
    ``automatons/*.py`` header pins ``rheedium==<version>`` to the exact
    ``[project].version`` in ``pyproject.toml``, so a version bump cannot
    silently drift from the script pins. ``bump_pin.py`` carries no pin and is
    skipped.

    Notes
    -----
    Reads ``[project].version`` from ``pyproject.toml`` and scans every other
    automaton script for ``rheedium==`` / ``rheedium[cuda]==`` PEP 723 pins,
    asserting each equals that version. Running ``automatons/bump_pin.py``
    repairs any drift this catches.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        version: str = str(tomllib.load(handle)["project"]["version"])
    expected: str = f"rheedium=={version}"
    pin_re = re.compile(r"rheedium(?:\[cuda\])?==[0-9][^\"']*")
    for path in sorted(_AUTOMATON_DIR.glob("*.py")):
        if path.name == "bump_pin.py":
            continue
        pins: list[str] = pin_re.findall(path.read_text(encoding="utf-8"))
        assert pins, f"{path.name} has no rheedium PEP 723 pin"
        for pin in pins:
            normalized: str = pin.replace("[cuda]", "")
            assert normalized == expected, (
                f"{path.name} pins {pin!r}, expected {expected!r}; "
                "run automatons/bump_pin.py"
            )
