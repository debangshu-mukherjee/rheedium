# Plan: `agents/` — Self-Contained, Agent-Runnable Experiment Scripts

Scope: `rheedium` — add a directory of **single-file, fully self-contained
experiment scripts**, each runnable end-to-end by an automated experiment agent
with one command:

```bash
uv run agents/<experiment>.py [--args ...]
```

Each file wraps a large multi-step rheedium pipeline (load → simulate → sweep →
reduce → report) behind one callable entry point, carries **PEP 723 inline
script metadata** declaring its dependencies, and emits a **machine-readable
result** an agent can parse. The defining property: because almost everything
the science needs is already a transitive dependency of `rheedium`, the inline
dependency list is usually just `["rheedium"]`.

Status: **proposed** — not yet implemented. This document is authoritative and
self-contained.

---

## 0. TL;DR

1. New top-level `agents/` directory (sibling to `tutorials/`), **not** part of
   the installed wheel. Each `*.py` is a standalone PEP 723 script.
2. Shared boilerplate (arg parsing, seeding, result emission, runtime knobs)
   lives **inside the package** as `rheedium.harness`, so every script's sole
   declared dependency stays `rheedium` and scripts never import each other.
3. A fixed file template + a strict **handoff contract**: typed CLI in, a JSON
   result object on stdout + artifacts in `--outdir`, deterministic seeds,
   meaningful exit codes, and a `--describe` self-introspection mode.
4. Smoke-tested in CI by actually running each script under `uv run` in a tiny
   `--smoke` mode, guaranteeing the "independently executable" promise stays
   true.

---

## 1. The handoff contract (why this exists)

An automated experiment agent should be able to, with **zero knowledge of
rheedium's internals**:

1. **Discover** what experiments exist and what each does (`agents/INDEX.md` +
   per-script `--describe`).
2. **Introspect** a script's parameters and their types/defaults without reading
   source (`--describe` emits a JSON parameter schema; `--help` is human-facing).
3. **Run** it with parameters and an output directory.
4. **Parse** the outcome: a single JSON object on the **last line of stdout**
   (`{"status", "metrics", "artifacts", "rheedium_version", "seed", ...}`), plus
   artifact files (figures, `.npz`, `.json`) under `--outdir`.
5. **Trust** the result: deterministic given `(script, args, seed,
   rheedium_version)`; non-zero exit on failure with a human message on stderr.

This contract — not any single experiment — is the real deliverable. Everything
below serves it.

---

## 2. Placement (decision + rationale)

**Recommended: top-level `agents/`**, a sibling of `tutorials/`, tracked in git
but excluded from the wheel.

Why not `src/rheedium/agents/`:

- These are **scripts, not library modules**. A module shipped inside the wheel
  that also declares `rheedium` as a PEP 723 dependency is conceptually circular
  (the package depending on itself to run its own file) and bloats the
  distribution.
- `tutorials/` already establishes the "runnable, not-installed, uses rheedium"
  pattern; `agents/` is its non-interactive, machine-facing sibling.

Library code that the scripts share goes the other way — **into** the package as
`rheedium.harness` (§5), where it is importable, typed, and tested like the rest
of `src/`. Net: scripts stay thin and runnable; reusable logic stays in the
installed package; each script's only dependency remains `rheedium`.

**Naming:** no collision. The planning docs now live in the tracked `plans/`
directory (formerly `.agents/`), so `agents/` is free to use cleanly:
`plans/` = what to build, `agents/` = runnable experiments. Distinct names,
distinct purposes.

Alternative (if `import rheedium.agents.<x>` discoverability is explicitly
wanted): make `src/rheedium/agents/` a real subpackage of dual-purpose modules
(importable **and** PEP 723-runnable). Viable but ships scripts in the wheel and
muddies the script/library boundary. Deferred unless required.

---

## 3. The PEP 723 dependency contract

Every script begins with an inline metadata block. The baseline is deliberately
minimal:

```python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.6"]
# ///
```

- **Single dependency, pinned.** `rheedium` pins to a released version so an
  agent re-running the script anywhere gets byte-identical behavior
  (reproducibility is a first-class requirement for experiments). Everything
  else (jax, equinox, jaxtyping, beartype, ase, ...) arrives transitively — this
  is the "wrap very few things" payoff.
- **GPU variant** when a script needs it: `dependencies = ["rheedium[cuda]==…"]`
  (the existing `cuda` extra). Keep CPU the default.
- **Dev-vs-handoff resolution.** During local development against unpublished
  changes, override the pin to the working tree without editing the script:
  ```bash
  uv run --with-editable . agents/<x>.py        # local source wins
  ```
  Document this in `agents/README.md`. For agents running off a branch, a git
  source is the portable form:
  `dependencies = ["rheedium @ git+https://github.com/debangshu-mukherjee/rheedium@<ref>"]`.
- **One pin, one place.** The pinned version string is identical across scripts;
  a tiny `agents/bump_pin.py` (itself a PEP 723 script) rewrites every header on
  release so they never drift.

Rationale for pinning vs floating: floating (`rheedium>=…`) maximizes "runs with
latest" but breaks reproducibility; pinning is correct for experiments. The dev
override covers the inner loop.

---

## 4. File anatomy (the template)

Every script is structured identically so an agent (and a human) can predict it:

```python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.6"]
# ///
"""One-line summary of the experiment.

Extended description: inputs, the multi-step pipeline it runs, and the metrics
and artifacts it produces. This docstring is what an agent reads first.
"""

from __future__ import annotations

import rheedium as rh
from rheedium.harness import experiment, emit, Param   # the shared contract


@experiment(
    name="kinematic-forward",
    params=[
        Param("crystal", str, help="Path to a CIF/XYZ/POSCAR file."),
        Param("voltage_kv", float, default=20.0),
        Param("theta_deg", float, default=2.0),
    ],
)
def main(args, ctx):
    """Run the pipeline; return a dict of metrics; write artifacts to ctx.outdir."""
    crystal = rh.inout.parse_cif(args.crystal)
    pattern = rh.simul.ewald_simulator(
        crystal, voltage_kv=args.voltage_kv, theta_deg=args.theta_deg
    )
    fig_path = ctx.save_figure("pattern.png", rh.plots.plot_rheed(pattern))
    return {
        "metrics": {"n_reflections": int(pattern.intensities.shape[0])},
        "artifacts": [fig_path],
    }


if __name__ == "__main__":
    main()
```

Fixed elements:

1. **PEP 723 header** (§3).
2. **Module docstring** = the experiment's human/agent-readable spec.
3. **`@experiment(...)`** decorator from `rheedium.harness` declares the typed
   parameter set once; it builds the argparse CLI, `--help`, `--describe`,
   `--smoke`, `--seed`, and `--outdir` for free.
4. **`main(args, ctx)`** runs the pipeline and returns `{"metrics", "artifacts"}`.
5. **`if __name__ == "__main__": main()`** — the only execution path.

No script imports another script (independence). All shared behavior is in
`rheedium.harness`, which is `rheedium`, the one declared dependency.

---

## 5. `rheedium.harness` — the in-package support library

A new **tested, typed** subpackage `src/rheedium/harness/` (per CONTRIBUTING:
`@jaxtyped(typechecker=beartype)` where arrays flow, numpydoc, `:see:` tests,
exported via `__init__.__all__` + Routine Listings). It carries the contract so
scripts stay declarative:

- **`Param` / `@experiment`** — declarative typed-parameter spec → argparse CLI;
  generates `--help` and `--describe` (a JSON schema of params: name, type,
  default, help) so agents introspect without reading source.
- **`ExperimentContext` (`ctx`)** — resolved `outdir` (created), `seed`, a
  `save_figure` / `save_array` / `save_json` artifact API returning relative
  paths, and a structured stderr logger.
- **`emit(result)`** — writes the canonical JSON result object as the **final
  stdout line**: `{"status": "ok"|"error", "experiment", "rheedium_version",
  "jax_backend", "seed", "params", "metrics", "artifacts", "wall_seconds"}`.
  Exceptions are caught at the boundary → `{"status": "error", "error": "..."}`,
  non-zero exit, traceback to stderr.
- **Runtime knobs (reuse this session's work):** the harness standardizes
  - `rheedium.tools.enable_compilation_cache(...)` (opt-in via `--cache`),
  - `RHEEDIUM_DISABLE_RUNTIME_CHECKS` for fast validated-data runs (`--unchecked`),
  - device selection / `tools.distribute_batched` for sweep-class experiments,
  - deterministic seeding of `jax.random` from `--seed`.
  These are exactly the levers the parallelization/compilation guide documents;
  the harness wires them once so every script gets them uniformly.

Because the harness is in `rheedium`, scripts depend on **only** `rheedium`, and
the harness itself is covered by the normal test suite.

---

## 6. Initial experiment catalog

Each maps a rheedium capability to a one-command experiment (Phase 2+):

| Script | Pipeline | Key metrics / artifacts |
|---|---|---|
| `forward_kinematic.py` | crystal → `ewald_simulator` → image | pattern PNG, reflection count, peak table |
| `forward_multislice.py` | crystal → multislice → image | pattern PNG, intensity stats |
| `forward_reflection.py` | crystal → `reflection_multislice_simulator` | specular position, reflectivity |
| `azimuthal_sweep.py` | φ-sweep via `simul.sweeps` + `distribute_batched` | per-angle metrics `.npz`, montage |
| `parameter_grid.py` | energy × θ grid | grid `.npz`, summary heatmap |
| `ensemble_average.py` | beam GSM / orientation / size `Distribution` average | averaged image, mode/effective counts |
| `reconstruct_orientation.py` | inverse fit (synthetic target → `recon`) | fitted angle, loss curve, grad-finite check |
| `audit_invariants.py` | run `rheedium.audit` invariants | pass/fail table, residual-vs-tolerance JSON |
| `export_model.py` | AOT-export forward model via `tools.export_forward` | serialized artifact, polymorphism report |
| `convergence_study.py` | mode-count / grid convergence sweep | residual-vs-N curve |

Phase 1 ships the harness + **one** exemplar (`forward_kinematic.py`) to lock
the contract end-to-end before fanning out.

---

## 7. Lint / type policy for `agents/`

- **Ruff applies** (line-length 79, double quotes, import sorting) — these are
  still rheedium source.
- **Relaxed jaxtyping**: scripts are orchestration entry points, not numerical
  kernels; the heavy `@jaxtyped(typechecker=beartype)`-on-everything rule is **not**
  required in `agents/` (mirrors how the numerics live in the package, not the
  scripts). A per-directory ruff/ty config carve-out (like the existing
  `[tool.ty.overrides]` for `tests/**`) relaxes annotation-completeness rules for
  `agents/**`.
- `agents/**` is **excluded from `[tool.uv.build-backend]`** packaging so it is
  not shipped in the wheel (it is not under `src/`, so this is automatic;
  assert it in the build test).
- `rheedium.harness` (in `src/`) follows the **full** CONTRIBUTING standard.

---

## 8. Testing & CI

The "fully and independently executable" promise is only real if CI proves it:

1. **Harness unit tests** (`tests/test_rheedium/test_harness/`): `Param`/CLI
   parsing, `--describe` schema shape, `emit` JSON contract, error-path exit
   code, artifact-path resolution, seed determinism.
2. **Script smoke tests**: a parametrized test discovers every `agents/*.py` and
   runs it via **`uv run --with-editable . agents/<x>.py --smoke --outdir <tmp>`**
   in a subprocess, asserting (a) exit code 0, (b) the last stdout line parses as
   the result JSON with `status == "ok"`, (c) declared artifacts exist. `--smoke`
   forces a tiny grid so each runs in seconds.
3. **CI job** `agents-smoke` (non-blocking at first, then blocking once stable),
   mirroring the existing informational-job pattern in `test.yml`: it executes
   each script under `uv run` in a clean ephemeral env to catch a header that
   declares the wrong dependency or a script that silently needs more than
   `rheedium`.
4. **`--describe` contract test**: every script must emit a valid param schema
   (so the agent-introspection promise can't rot).

---

## 9. Phasing

1. **Phase 1 — contract + harness + one exemplar.** `rheedium.harness`
   (`Param`, `@experiment`, `ExperimentContext`, `emit`, runtime knobs);
   `agents/forward_kinematic.py`; `agents/README.md` + `agents/INDEX.md`; harness
   unit tests + the smoke test for the one script; CI `agents-smoke` job
   (non-blocking). Lock the JSON/`--describe` contract here.
2. **Phase 2 — forward experiments.** `forward_multislice`, `forward_reflection`,
   `azimuthal_sweep`, `parameter_grid` (the sweep ones exercise
   `distribute_batched` + the compilation cache).
3. **Phase 3 — ensemble & inverse.** `ensemble_average` (beam GSM / orientation /
   size Distributions), `reconstruct_orientation` (recon), `convergence_study`.
4. **Phase 4 — ops experiments.** `audit_invariants`, `export_model` (reuses
   `tools.export_forward`); promote `agents-smoke` to blocking; `bump_pin.py`
   release hook.
5. **Phase 5 — docs.** A "Running experiments as an agent" guide under
   `docs/source/guides/` describing the handoff contract, `--describe`, and the
   result schema.

Phase 1 is independently shippable and proves the whole idea; later phases are
additive.

---

## 10. Risks

- **Dependency leakage** — a script silently needing a package not transitively
  provided by `rheedium`. Mitigation: the `agents-smoke` CI job runs in a clean
  ephemeral `uv` env from the PEP 723 header alone; a missing dep fails loudly.
- **Version pin vs local dev drift** — the pinned header won't reflect
  unpublished changes. Mitigation: documented `--with-editable .` override; CI
  smoke uses it so tests track the working tree, not the last release.
- **Reproducibility holes** — uncontrolled randomness or backend variance.
  Mitigation: harness seeds `jax.random` from `--seed` and records
  `rheedium_version` + `jax_backend` in every result; experiments avoid
  `Math.random`-style nondeterminism.
- **Wheel bloat / circular semantics** — avoided by keeping scripts in top-level
  `agents/` (not `src/`); a build test asserts they are absent from the wheel.
- **Contract rot** — `--describe`/result JSON drifting from what agents expect.
  Mitigation: schema tests in §8; one canonical `emit` in the harness, never
  hand-rolled per script.
  (The former `.agents/` vs `agents/` naming concern is moot: planning docs now
  live in `plans/`.)
- **Long-running experiments under smoke** — mitigated by a mandatory `--smoke`
  mode every script must honor (tiny grids), enforced by the smoke test.

---

## 11. Diff surface

| Path | Change |
|---|---|
| `src/rheedium/harness/__init__.py` | new subpackage: `Param`, `experiment`, `ExperimentContext`, `emit`, runtime-knob setup; exports + Routine Listings |
| `src/rheedium/__init__.py` | add `harness` to the submodule imports + `__all__` + Routine Listings |
| `agents/forward_kinematic.py` | Phase-1 exemplar PEP 723 script |
| `agents/README.md`, `agents/INDEX.md` | usage, dev override, experiment index |
| `agents/bump_pin.py` | release-time pin rewriter (PEP 723 script) |
| `tests/test_rheedium/test_harness/` | harness unit tests |
| `tests/.../test_agents_smoke.py` | discover + `uv run` every `agents/*.py --smoke` |
| `pyproject.toml` | `[tool.ty.overrides]`/ruff carve-out for `agents/**`; confirm `agents/` excluded from `uv_build` |
| `.github/workflows/test.yml` | `agents-smoke` job (ephemeral `uv run` per script) |
| `docs/source/guides/` | "Running experiments as an agent" guide (Phase 5) |

The science stays in `rheedium`; `agents/` only orchestrates it into one-command,
agent-parseable experiments whose sole declared dependency is `rheedium` itself.

---

## 12. Open decisions

1. **Directory name** — `agents/` (now collision-free, planning docs live in
   `plans/`) vs `experiments/` / `protocols/`. Recommendation: `agents/`.
2. **Pin policy** — exact `==` per release (recommended, reproducible) vs a git
   source per branch vs floating `>=`. Recommendation: `==`, with the documented
   `--with-editable .` dev override.
3. **CLI layer** — stdlib `argparse` (keeps the dependency list at just
   `rheedium`) vs `tyro`/`typer` (nicer typed CLIs but an extra PEP 723 dep).
   Recommendation: `argparse` in the harness, to preserve the single-dependency
   property.
4. **`agents-smoke` blocking** — start non-blocking (informational), promote to
   required once stable. Recommendation: yes, staged.
