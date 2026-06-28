# Plan: `automatons/` — Self-Contained, Agent-Runnable Experiment Units

Scope: `rheedium` — add a directory of **single-file, fully self-contained
experiment scripts**, each runnable end-to-end by an automated experiment agent
with one command:

```bash
uv run automatons/<experiment>.py [--args ...]
```

Each file wraps a large multi-step rheedium pipeline (load → simulate → sweep →
reduce → report) behind one callable entry point, carries **PEP 723 inline
script metadata** declaring its dependencies, and emits a **machine-readable
result** an agent can parse. The defining property: because almost everything
the science needs is already a transitive dependency of `rheedium`, the inline
dependency list is usually just `["rheedium"]`.

Status: **proposed** — not yet implemented, and **sequenced last** in the
roadmap: [framework](plans/implemented/distribution_framework_plan.md) →
[rationalization](plans/implemented/rationalization_refactor_plan.md) →
[recon (inversion)](../implemented/recon_optimization_plan.md) → automatons. It
begins only after all three land. The reason for last: these scripts call
`rheedium`'s public API heavily — the *rationalized* forward surface (the
collapsed ~6-arg `simulate_detector_image`, config carriers, unified sweeps) and
the **recon `solve` / `recipe_deviation` API that Loop C consumes** — so each
automaton is written once and pins (PEP 723) to the stable release that carries
both, not rewritten after the API changes underneath it. This document is
authoritative and self-contained.

---

## Mission — the three autonomous-lab loops

These automatons are the callable tools of an **agent-driven autonomous MBE
laboratory**. A recipe-giving LLM agent drives growth on a molecular-beam-epitaxy
system; **RHEED is the real-time measure of truth**, and `rheedium` turns RHEED
into quantitative state the agent can act on. Three loops run together:

**Loop A — online (experiment).** Live RHEED frames stream off the MBE detector
during growth. An automaton ingests each frame (or a rolling window) and extracts
growth observables — specular-spot intensity oscillations (layer-by-layer RHEED
oscillations), streak spacing and sharpness, spotty-vs-streaky (2D↔3D /
roughening) — emitting a state object the agent uses to decide *continue / hold /
adjust flux / change temperature / stop*. This is the real-time feedback channel,
and the **first place `rheedium` enters the loop (stage 1)**.

**Loop B — theory-in-the-loop.** In parallel, **MatEnsemble** proposes candidate
structures and writes **`.xyz` files**. An automaton feeds each `.xyz` to
`rheedium`, simulates its RHEED pattern, and scores it — run over an ensemble,
this ranks candidate structures by how well their *simulated* RHEED matches the
*measured* one.

**Loop C — inversion (the differentiable payoff).** The keystone, and the reason
`rheedium` is written in JAX and kept differentiable end-to-end. An automaton
**rapidly inverts** a measured RHEED pattern — differentiating the forward model
by gradient descent — to recover, with credible bands, the **orientation and beam
parameters** (given a CIF), the **structure / composition / thickness**, and even
the **probability distributions** over those latents (orientation spread,
grain-size distribution, beam coherence) — answering the two questions the agent
needs in real time: *what is actually being grown*, and *how far is it from the
recipe the LLM agent intended*. That recipe-vs-actual
deviation is the control signal. The speed work that makes this loop viable
inside a growth-control cadence — persistent compilation cache, AOT-exported
forward kernels, the unchecked fast path — already exists in `rheedium.tools`.
And where the inversion is **model-degenerate** — distinct material mechanisms
that produce the same distribution (recon §2.3) — Loop C does not guess: it
**designs the tie-breaking experiment**, choosing the next perturbation
(temperature / flux / time) that maximizes the predicted divergence between the
competing mechanisms' distribution-evolution (expected information gain), turning
non-identifiability into the agent's next action.

The loops close on each other: **A** supplies the measured ground truth, **B**
brackets it with a discrete candidate set (a forward-and-rank comparison), and
**C** refines and quantifies it by gradient inversion — so the agent can both say
*what is on the surface* and *steer the recipe back toward intent*.

Three consequences for the design:

- **MatEnsemble couples to `rheedium` through files, not imports.** Its only
  interface is the `.xyz` it writes, which `rheedium.inout` already parses — so the
  theory automatons keep the single-`rheedium`-dependency property (§3); neither
  side imports the other. (Same for the MBE detector: its output is image frames
  `rheedium.inout` reads.)
- **Online tracking is a stream, not a one-shot.** The run→emit→exit contract
  (§1) still fits, via **per-frame invocation**: the agent drives the loop, calling
  the automaton once per frame and accumulating emitted state. Temporal observables
  (oscillation period, drift) need history — supplied by a rolling series file the
  automaton reads/appends, *not* a long-running process. (Per-frame-stateless vs
  windowed vs a `--watch` daemon is an open decision — §12.)
- **Loop C requires gradients through the *entire* chain**, which is exactly the
  CONTRIBUTING invertibility principle (no premature, non-differentiable
  reduction). The inversion automaton is the consumer that makes that principle
  pay off; a hard reduction anywhere in the forward model would silently break it.

---

## 0. TL;DR

1. New top-level `automatons/` directory (sibling to `tutorials/`), **not** part of
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

1. **Discover** what experiments exist and what each does (`automatons/INDEX.md` +
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

## Agent-consumer contract — lean agents, heavy backend

The §1 contract makes an automaton *runnable*; this section makes it *ergonomic
for an LLM-driven agent to choose, call safely, and chain* — the design that keeps
the agent layer thin. The principle: **the agent only moves bytes and makes
decisions; `rheedium` owns the heavy lifting — compute, analysis, plotting, and
serialization.** Everything an agent needs to invoke an automaton *safely,
composably, and at scale* is **declared by the automaton and enforced once by the
harness**, never re-implemented in the agent or in the INTERSECT/MCP bridge that
fronts it. These are extensions to `Param` / `@experiment` / `emit` / `--describe`,
not new machinery, and they stay stdlib-only (`json`), preserving the
single-`rheedium`-dependency property (open decision #3).

**A. Symmetric JSON I/O — JSON in, JSON out.**
`emit()` already returns a JSON result on the final stdout line; the missing half
is JSON *input*. The harness adds `--params <file | - | json-string>`: a JSON
object validated against the **same schema `--describe` emits**, populating the
same args namespace (explicit flags override; the merged result is validated).
Flat CLI flags stay the human path; JSON is the machine path. This is what lets
**nested params** (distributions, sweep value lists, config carriers, recipe
dicts, artifact refs) be expressed at all — flat argparse cannot — and makes a
slice of one automaton's `emit()` directly feedable as another's `--params`. A
strict `--json` mode writes *only* the result object to stdout so the last-line
parse cannot be fooled by stray output.

**B. Self-describing, safe parameters.**
- **Semantic metadata on `Param`** — `unit`, `bounds=(min, max)`, `choices`,
  `example`, beyond name/type/default/help. An agent needs
  `voltage_kv: {min: 5, max: 50, unit: "kV"}` to call sanely, and the harness
  pre-validates before an expensive run.
- **A declared result schema** — `returns=` on `@experiment`, surfaced in
  `--describe`, so the agent knows the metric keys/types/units that come back
  *without running*; the consuming MCP tool's return type follows from it.
- **A structured error taxonomy** — the error result carries `error_kind` (a
  small enum: `InvalidStructure`, `ParamOutOfRange`, `Unsupported`,
  `ResourceExhausted`, `NumericalFailure`) plus the offending `field`, so the
  agent branches (ask the user / clamp & retry / downscale / re-init) instead of
  scraping a traceback. The `beartype` boundary already *detects* these —
  categorize them.
- **A pure `--validate` mode** — validate a structure file and params against the
  schema and return structured errors *without running* the pipeline (distinct
  from `--smoke`, which runs a tiny one). Fast, precise pre-flight feedback
  ("XYZ malformed at line N", "theta_deg=12 exceeds max 5").

**C. Results as portable artifacts, never JAX arrays.**
A `jax.Array` is neither JSON-serializable nor backend-portable, so it must never
cross the process boundary. The rule: **PNG for eyes, `.npz` for math, JSON for
metrics.**
- **Guarantee a rendered image.** Any pattern-producing automaton (`forward_*`,
  `screen_*`, `rheed_ingest`) *declares* an image artifact in its manifest — not
  an incidental `save_figure`. Rendering lives in `rheedium` (`rh.plots`), so the
  agent/UI never plots.
- **Typed artifact manifest.** Each `artifacts` entry is
  `{role, mime, path, preview_b64?}` (e.g. `role: "detector_image",
  mime: "image/png"`) so a consumer renders generically with zero per-automaton
  code.
- **Inline `preview_b64`.** A small base64 thumbnail dropped straight into the
  `emit()` JSON, so an interactive agent shows an image with no
  write→read→encode round-trip; the full-resolution PNG stays an artifact.
- **Keep the numeric array** as a serialized `.npz` artifact (on request) for
  Loop C, residuals, and any quantitative downstream — images for display do not
  replace arrays for math.
- **Render controls as params** — `cmap` (e.g. `"phosphor"`), image size,
  normalization / log-scale — so the agent can request a specific render
  (`rh.plots.plot_rheed` / `render_pattern_to_image` already exist).

**D. Long-run & scale ergonomics.**
- **Progress streaming.** For inversions and large sweeps, optional NDJSON
  progress lines (`{type: "progress", frac, iter, loss}`) before the final result
  — mirrors INTERSECT snapshot telemetry and the agent's streaming UI. (`--serve`
  streams one result *per input*; this streams *within* a run.)
- **Cooperative cancellation / `--deadline`.** Honor a wall-clock ceiling /
  SIGTERM at checkpoints, emitting a partial `status: "timeout"`. An agent must be
  able to abort a runaway inversion within a growth-control cadence. (Complements
  the CI budget assertion at G5 — that is offline; this is runtime.)
- **Cost estimate (`--estimate`).** Given params, return
  `{est_wall_s, needs_gpu, est_mem_gb}` without running, so the agent can warn the
  user or pick CPU vs GPU before launching.
- **Result cache / idempotency key.** Determinism over
  `(script, args, seed, rheedium_version)` makes a content-addressed `result_key`
  derivable; `emit()` includes it and an optional `--cache-results` short-circuits
  identical requests — important for an MCP server fronting many agents
  re-requesting the same simulation. (Distinct from the *compilation* cache.)

**E. Composition — chaining without shuttling blobs.**
- **Structure by value or reference.** Automatons accept structure *content*
  inline (small) or a path / `artifact://` reference (large), so a remote caller
  need not stage a temp file and large payloads (CIF/XYZ text, frames, arrays)
  move by reference, not through the LLM/JSON.
- **Artifact-reference inputs.** A stable `artifact://<run_id>/<name>` scheme so
  one automaton's output wires as another's input (`screen_xyz_ensemble` →
  `match_measured_to_simulated` → `recipe_deviation`) without re-passing data —
  the composition story that maps directly onto iHub campaigns.

Net: the automaton declares *what it takes, what it returns, what it costs, and
what can go wrong*; the harness enforces it once; the agent reads `--describe`,
sends `--params`, and renders the manifest. Heavy code, analysis, and plotting
stay in `rheedium`; the agent stays lean.

---

## 2. Placement (decision + rationale)

**Recommended: top-level `automatons/`**, a sibling of `tutorials/`, tracked in git
but excluded from the wheel.

Why not `src/rheedium/automatons/`:

- These are **scripts, not library modules**. A module shipped inside the wheel
  that also declares `rheedium` as a PEP 723 dependency is conceptually circular
  (the package depending on itself to run its own file) and bloats the
  distribution.
- `tutorials/` already establishes the "runnable, not-installed, uses rheedium"
  pattern; `automatons/` is its non-interactive, machine-facing sibling.

Library code that the scripts share goes the other way — **into** the package as
`rheedium.harness` (§5), where it is importable, typed, and tested like the rest
of `src/`. Net: scripts stay thin and runnable; reusable logic stays in the
installed package; each script's only dependency remains `rheedium`.

**Naming — why `automatons/`, not `agents/`:** the *callers* are the automated
agents; these scripts are the self-contained units they invoke. Naming the
directory `agents/` conflates the two (an agent running an "agent"). An
*automaton* — a self-operating machine that carries out a whole sequence on its
own — names the unit precisely: an agent runs an automaton. Planning docs are
separate, in the tracked `plans/` directory, so there is no directory collision
either (`plans/` = what to build, `automatons/` = runnable experiment units).

Alternative (if `import rheedium.automatons.<x>` discoverability is explicitly
wanted): make `src/rheedium/automatons/` a real subpackage of dual-purpose modules
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
  uv run --with-editable . automatons/<x>.py        # local source wins
  ```
  Document this in `automatons/README.md`. For agents running off a branch, a git
  source is the portable form:
  `dependencies = ["rheedium @ git+https://github.com/debangshu-mukherjee/rheedium@<ref>"]`.
- **One pin, one place.** The pinned version string is identical across scripts;
  a tiny `automatons/bump_pin.py` (itself a PEP 723 script) rewrites every header on
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

from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import experiment, emit, Param   # the shared contract


@jaxtyped(typechecker=beartype)
def specular_intensity(pattern_image: Float[Array, "h w"]) -> Float[Array, ""]:
    """Array-handling helpers carry the full decorator stack (CONTRIBUTING)."""
    return rh.simul.specular_peak_intensity(pattern_image)


@experiment(
    name="kinematic-forward",
    params=[
        Param("crystal", str, help="Path to a CIF/XYZ/POSCAR file."),
        Param("voltage_kv", float, default=20.0),
        Param("theta_deg", float, default=2.0),
    ],
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
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

The array-handling helpers (`specular_intensity` above) carry
`@jaxtyped(typechecker=beartype)` + `jaxtyping` annotations exactly as in `src/`;
only the argparse-facing `main(args, ctx)` glue uses ordinary annotations (it
touches no arrays). `jaxtyping` and `beartype` arrive transitively with
`rheedium`, so the dependency list stays `["rheedium"]`.

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
- **AOT plumbing (so automatons opt in uniformly):** the harness exposes the AOT
  strategy as two harness-level capabilities, not per-script boilerplate:
  - `--export <path>` — lower this automaton's forward kernel to portable
    StableHLO via `tools.export_forward` (symbolic atom count where the kernel is
    polymorphic; bucketed concrete grids where it is not), for version-pinned,
    recompile-free reuse on a control/agent node. Backs the `export_model`
    automaton (Phase 7).
  - `--serve` — run the automaton as a **long-running process** that loads the
    forward kernel once (cache-warm or from an exported artifact) and reads
    successive inputs from stdin/a watched path, emitting one result per input.
    This is the real-time escape hatch for Loop A: it amortizes the per-process
    JAX startup that AOT alone cannot remove (see "AOT compilation strategy").
- **Warmup / first-call JIT latency (`--warmup`, persistent cache on by default).**
  JAX specializes compilation on *static* shapes (detector `image_shape`,
  `hmax`/`kmax`, atom-count buckets), so the first call for each new shape pays a
  full compile — **measured ~160 s for a 192×192 kinematic forward sim on CPU**
  (rheed_agent INTERSECT capability, Jun 2026). The harness owns the fix so no
  script hand-rolls it:
  1. the **persistent compilation cache is on by default**, so each
     `(computation, shape, backend)` compiles **once ever** and later runs — even
     fresh `uv run` processes — load the XLA artifact from disk rather than
     recompiling;
  2. a **`--warmup`** pass (and an **automatic warmup at `--serve` startup**)
     compiles the kernel for every shape the automaton will serve — the shapes
     declared in `--describe` / supplied via `--params` — *before* accepting real
     requests, so the first request is not slow;
  3. **`--estimate`** (Agent-consumer contract) reports whether the requested
     shape is **cache-warm** (fast) or **cold** (first-compile), so the agent can
     set expectations.

  **Crucial gotcha — warmup is shape-specific.** Warming a 64×64 kernel does
  **not** cover a 192×192 request (JAX recompiles), so the warmup set must match
  the shapes actually served, or rely on bucketed concrete grids / an AOT-exported
  kernel. For fixed-shape, high-rate paths (Loop A), AOT export removes JIT
  entirely (see "AOT compilation strategy"); for variable interactive shapes, the
  persistent cache + a declared `--warmup` set is the right tool.
- **Agent-consumer plumbing (see "Agent-consumer contract"):** `--params`
  (JSON-in, validated against the `--describe` schema), `--validate`
  (schema-check without running), `--estimate` (predicted cost), `--deadline`
  (cooperative cancellation), and a strict `--json` output mode. `Param` gains
  `unit` / `bounds` / `choices` / `example`; `@experiment` gains `returns=` (the
  result schema, surfaced in `--describe`); `emit` gains a typed artifact manifest
  (`role` / `mime` / `preview_b64`), an `error_kind` taxonomy, and a
  content-addressed `result_key`; plus optional NDJSON progress events. One
  canonical implementation in the harness, never hand-rolled per script.

Because the harness is in `rheedium`, scripts depend on **only** `rheedium`, and
the harness itself is covered by the normal test suite.

---

## 6. Initial experiment catalog

The **mission-critical** automatons serve the three loops directly; the rest are
supporting capabilities a forward kernel or a diagnostic needs.

**Loop B — theory-in-the-loop (MatEnsemble → rheedium):**

| Script | Pipeline | Key metrics / artifacts |
|---|---|---|
| `forward_kinematic.py` | crystal `.xyz`/CIF → `ewald_simulator` → image | pattern PNG, reflection count, peak table |
| `forward_multislice.py` | crystal → multislice → image | pattern PNG, intensity stats |
| `forward_reflection.py` | crystal → `reflection_multislice_simulator` | specular position, reflectivity |
| `screen_xyz_ensemble.py` | dir/glob of MatEnsemble `.xyz` → simulate each → rank vs a measured/target pattern | ranked candidates `.json`, best-match overlay |
| `match_measured_to_simulated.py` | **the bridge** — measured RHEED frame + simulated set → similarity scores | scores `.json`, best match + residual image |

**Loop C — inversion (differentiable, "what is actually being grown"):**

| Script | Pipeline | Key metrics / artifacts |
|---|---|---|
| `fit_orientation_beam.py` | measured RHEED + a CIF → `recon.fit_geometry_beam` (orientation + beam params; structure fixed) | recovered `(θ, φ, azimuth)` + `BeamModeDistribution` + covariance, fit overlay |
| `reconstruct_distribution.py` | measured RHEED + target latent axis → `recon` distribution reconstruction (parametric family or free-form weights) | recovered probability distribution (orientation spread / grain-size / beam coherence) + **credible band** `.json`, shape plot |
| `invert_structure.py` | measured RHEED frame → differentiable fit via `recon` (grad descent through the forward model) | recovered structure/composition/thickness, fit-loss curve, residual image |
| `recipe_deviation.py` | intended recipe + measured RHEED → inverted actual + gap | per-parameter intended-vs-actual delta `.json`, severity flag (the control signal) |

**Loop A — online RHEED (experiment, real-time truth):**

| Script | Pipeline | Key metrics / artifacts |
|---|---|---|
| `rheed_ingest.py` | one live RHEED frame (TIFF/HDF5) → growth observables | specular intensity, streak spacing/sharpness, 2D/3D flag, state `.json` |
| `growth_monitor.py` | RHEED time-series / rolling window → growth dynamics | oscillation period + count, roughness trend, transition flags |

**Supporting / diagnostic:**

| Script | Pipeline | Key metrics / artifacts |
|---|---|---|
| `azimuthal_sweep.py` | φ-sweep via `simul.sweeps` + `distribute_batched` | per-angle metrics `.npz`, montage |
| `parameter_grid.py` | energy × θ grid | grid `.npz`, summary heatmap |
| `ensemble_average.py` | beam GSM / orientation / size `Distribution` average | averaged image, mode/effective counts |
| `reconstruct_orientation.py` | inverse fit (synthetic target → `recon`) | fitted angle, loss curve, grad-finite check |
| `audit_invariants.py` | run `rheedium.audit` invariants | pass/fail table, residual-vs-tolerance JSON |
| `export_model.py` | AOT-export forward model via `tools.export_forward` | serialized artifact, polymorphism report |
| `convergence_study.py` | mode-count / grid convergence sweep | residual-vs-N curve |

Phases 1–2 ship the harness and **one** exemplar (`forward_kinematic.py`, the
per-structure core of Loops B and C) to lock the contract end-to-end; the three
mission loops are the first fan-out (§9).

---

## AOT compilation strategy — where HLO lowering pays off

The autonomous lab is the first context with production constraints that justify
ahead-of-time (AOT) lowering of forward kernels to portable StableHLO
(`tools.export_forward`, §5): fixed shapes called at high frequency, a real-time
control cadence, fresh-process invocation, and a need for reproducible,
version-locked numerics the agent can audit — none of which hold for interactive
research. But AOT's value is uneven across the loops, and is
**portability/reproducibility first, latency second** (the persistent compilation
cache already captures most in-process speed). Apply it where the shape is fixed
and the call count is high:

| Loop | AOT fit | What to lower / how |
|---|---|---|
| **A — online ingest** | **Strongest** | Detector emits a fixed `H×W` frame every time → one shape, run every frame. Lower the fixed-shape ingest kernel once for the whole growth run; no shape polymorphism, no FFT-polymorphism issue. |
| **B — kinematic screen** | **Strong** | Kinematic forward is shape-polymorphic in atom count → **one exported artifact handles any MatEnsemble `.xyz`** (symbolic `N`). A multislice back-end needs *bucketed concrete grids* — the FFT cannot polymorphize. |
| **C — inversion** | **Weak (direct)** | Iterative in-process optimization → cache + JIT in a *warm* process is the right tool; export only a *frozen, fixed-iteration* grad step if a deployable inversion is needed. |

Two constraints that shape the architecture, not just the flag:

- **Per-process JAX startup (~seconds) dominates the real-time path, and AOT does
  not fix it.** Every `uv run automaton.py` is a fresh interpreter that pays
  Python+JAX import before any lowered binary runs. For Loop A at frame rate that
  startup — not compilation — is the bottleneck, so the real-time loop wants a
  **long-running `--serve` process** (open decision #5), where in-process JIT +
  cache suffices. AOT export earns its keep once you have *escaped* the per-call
  subprocess: a persistent inference server, or the control/agent node running an
  **exported, version-pinned forward model** without the rheedium dev stack (the
  `export_model` automaton, Phase 7).
- **Ship StableHLO, not raw compiled binaries, across the fleet.** Raw
  `.compile()` PJRT executables are machine- and jaxlib-specific (the `SIGILL`
  machine-feature mismatch we measured); `jax.export` StableHLO is the portable,
  compat-window form. Namespace any persistent cache *per architecture*.

**Net:** AOT here means *reproducible, portable, recompile-free forward models for
deployment beyond the dev box* — Loop A's fixed-shape ingest kernel and Loop B's
atom-count-polymorphic kinematic engine — with raw latency a secondary,
cache-shared benefit. The harness owns the plumbing so each automaton opts in
uniformly (§5).

---

## 7. Lint / type policy for `automatons/`

- **Ruff applies** (line-length 79, double quotes, import sorting) — these are
  still rheedium source.
- **Full jaxtyping + beartype**, same standard as `src/`. `jaxtyping` and
  `beartype` already ship with `rheedium` (the sole declared dependency), so
  there is **zero added dependency cost** — and the runtime `@jaxtyped(typechecker
  =beartype)` checking is a *feature* here: a malformed detector frame or a bad
  `.xyz` fails loud at the function boundary instead of producing a quietly-wrong
  growth decision. Every array-handling function in an automaton carries
  `@jaxtyped(typechecker=beartype)` with `jaxtyping` shape/dtype annotations and
  imports typing constructs from `beartype.typing`, exactly as CONTRIBUTING
  requires. (Pure-orchestration glue that touches no arrays — argparse wiring —
  needs only ordinary annotations.)
- **`automatons/` is added to the checked scope:** `[tool.ty.src].include` gains
  `automatons`, at the **src-strict** rule set (no `tests/**`-style relaxation).
  The explicit `@jaxtyped(typechecker=beartype)` decorators fire at runtime on
  every call under plain `uv run` (no pytest `--jaxtyping-packages` flag needed —
  that is only for import-hook checking of *undecorated* code), so the
  `automatons-smoke` job already exercises them in CI.
- **Fast-path escape:** the runtime checks add per-call overhead. The hot loops
  (`--serve`, Loop C inversion) may disable them for the inner compute — the same
  `RHEEDIUM_DISABLE_RUNTIME_CHECKS` / jaxtyping-deactivation lever the harness
  already manages — while keeping them on at the input boundary where bad data
  enters. Default: **checks on**; opt out only on the measured hot path.
- `automatons/**` is **excluded from `[tool.uv.build-backend]`** packaging so it is
  not shipped in the wheel (it is not under `src/`, so this is automatic;
  assert it in the build test).
- `rheedium.harness` (in `src/`) follows the **full** CONTRIBUTING standard.

---

## 8. Testing & CI

The "fully and independently executable" promise is only real if CI proves it:

1. **Harness unit tests** (`tests/test_rheedium/test_harness/`): `Param`/CLI
   parsing, `--describe` schema shape, `emit` JSON contract, error-path exit
   code, artifact-path resolution, seed determinism.
2. **Script smoke tests**: a parametrized test discovers every `automatons/*.py` and
   runs it via **`uv run --with-editable . automatons/<x>.py --smoke --outdir <tmp>`**
   in a subprocess, asserting (a) exit code 0, (b) the last stdout line parses as
   the result JSON with `status == "ok"`, (c) declared artifacts exist. `--smoke`
   forces a tiny grid so each runs in seconds.
3. **CI job** `automatons-smoke` (non-blocking at first, then blocking once stable),
   mirroring the existing informational-job pattern in `test.yml`: it executes
   each script under `uv run` in a clean ephemeral env to catch a header that
   declares the wrong dependency or a script that silently needs more than
   `rheedium`.
4. **`--describe` contract test**: every script must emit a valid param schema
   (so the agent-introspection promise can't rot).

---

## 9. Phasing & gates

Each phase ends at a **gate**: an objectively checkable set of conditions that
must *all* be green before the next phase starts. A gate is a command whose
output decides pass/fail, not a review opinion. The discipline: no experiment
automaton is written before the contract is locked (schemas at **G0**, harness at
**G1**), and no fan-out before the exemplar gate (**G2**) proves the whole loop
end-to-end.

### Entry — Gate A0 (roadmap precondition for *any* work below)

Both the
[rationalization refactor](plans/implemented/rationalization_refactor_plan.md) (R0–R6)
and the [recon optax solver](../implemented/recon_optimization_plan.md) (K0–KG6) are
**complete** — which transitively requires the
[distribution framework](plans/implemented/distribution_framework_plan.md). A
`rheedium` release carrying the rationalized API (the ~6-arg
`simulate_detector_image` + config carriers + unified sweeps) **and** the frozen
recon inverse API (`ReconProblem` / `solve` / `multistart` / `recipe_deviation`,
which Loop C's `invert_structure` + `recipe_deviation` automatons import) is
**published**, so the PEP 723 pin (§3) targets it. Until A0 holds this plan does
not start — building automatons against the pre-refactor or pre-solver API would
mean rewriting every one afterward.

### Phase 0 — Contract design (paper, no code)

*Tasks:* freeze the **result-JSON schema** and the **`--describe` param schema**
as committed artifacts (a `schema/` JSON-Schema file or a versioned note). The
param schema now carries the **semantic metadata** (unit / bounds / choices /
example) and **doubles as the `--params` JSON-input schema**; the result schema
includes the **`returns` block**, the **typed artifact manifest**
(`role`/`mime`/`preview_b64`), and the **`error_kind` taxonomy** (see
"Agent-consumer contract"). Enumerate the exact harness API (`Param`,
`experiment`, `ExperimentContext`, `emit`) and the CLI flags every automaton
inherits (`--outdir`, `--seed`, `--smoke`, `--describe`, `--cache`, `--unchecked`,
plus `--params`, `--validate`, `--estimate`, `--deadline`, `--json`).

**Gate G0:** both JSON schemas exist as committed files and are reviewed; the
param schema validates a sample `--params` document and the result schema
validates a sample `emit()` with artifact manifest + error taxonomy; all later
phases *validate against* them rather than redefining them.

### Phase 1 — Harness (the foundation)

*Tasks:* implement `src/rheedium/harness/` to the full CONTRIBUTING standard
(typed, numpydoc, `:see:` tests); wire the runtime knobs
(`enable_compilation_cache`, `RHEEDIUM_DISABLE_RUNTIME_CHECKS`,
`distribute_batched`, seeded `jax.random`); export via `__init__` + add to
`rheedium.__all__` / Routine Listings; unit tests for CLI parsing, `--describe`
output, the `emit` contract, error-path exit code, artifact-path resolution, and
seed determinism.

**Gate G1:** `uv run pytest tests/test_rheedium/test_harness` green; `emit` and
`--describe` outputs validate against the G0 schemas (snapshot test);
`uv run ty check` and `uv run ruff check` clean. **No automaton scripts exist
until G1 passes.**

### Phase 2 — First exemplar, end-to-end (prove the loop)

*Tasks:* `automatons/forward_kinematic.py` on the §4 template; `automatons/README.md`
+ `automatons/INDEX.md`; the smoke-test harness that discovers `automatons/*.py`
and runs each via `uv run --with-editable . … --smoke`; the `automatons-smoke`
CI job (non-blocking), mirroring `type-safety-scan`; the build assertion that
`automatons/` is absent from the wheel.

**Gate G2:** `uv run automatons/forward_kinematic.py --smoke --outdir <tmp>` exits
0, emits a final-line result JSON with `status == "ok"`, and writes the declared
artifacts; the same runs **green in the clean ephemeral CI env** (proving the
single-`rheedium` dependency); `--describe` validates against the G0 schema. The
template is now frozen — later phases only fill it in.

### Phase 3 — Loop B: theory-in-the-loop (MatEnsemble → rheedium)

The first mission loop. `forward_kinematic` (G2) is the per-structure core;
this phase scales it to an ensemble and closes the comparison.

*Tasks:* `screen_xyz_ensemble` (a dir/glob of MatEnsemble `.xyz` → simulate each
→ rank vs a measured/target pattern, batched with `distribute_batched`);
`match_measured_to_simulated` (**the bridge** — measured frame + simulated set →
similarity scores + best match); richer simulate kernels `forward_multislice` /
`forward_reflection` as alternative back-ends for the screen. The screening
engine is the **AOT-exported kinematic kernel** — one StableHLO artifact,
symbolic in atom count, reused across the whole `.xyz` ensemble (a multislice
back-end falls back to bucketed concrete grids).

**Gate G3:** given a fixture directory of `.xyz` files + one measured/target
pattern, `screen_xyz_ensemble --smoke` ranks the candidates and emits valid
result JSON; `match_measured_to_simulated` returns a score for measured-vs-each
and identifies the best; both smoke-green in the ephemeral CI env with valid
`--describe`; a same-`--seed` reproducibility check passes; the exported kinematic
artifact runs over ≥2 different atom counts from one lowering (proves the
polymorphic-`N` engine).

### Phase 4 — Loop A: online RHEED (real-time experiment truth)

*Tasks:* `rheed_ingest` (one live RHEED frame TIFF/HDF5 → growth observables:
specular intensity, streak spacing/sharpness, 2D/3D flag); `growth_monitor`
(a rolling RHEED series → oscillation period/count, roughness trend, transition
flags), reading/appending a series file rather than running as a daemon. Because
the detector resolution is fixed, `rheed_ingest`'s kernel is **AOT-lowered once**
(one shape for the whole run); the frame-rate path runs it under the harness
`--serve` mode so the per-process JAX startup is paid once, not per frame.

**Gate G4:** on a committed RHEED frame/series fixture, `rheed_ingest --smoke`
emits a growth-state JSON with the observables, and `growth_monitor` recovers a
known oscillation period from a synthetic intensity series to tolerance; both
smoke-green; per-frame statelessness verified (same frame ⇒ same state); a
`--serve` run processes a short frame sequence with one warm-up and steady
per-frame latency thereafter.

### Phase 5 — Loop C: inverse — *what is actually being grown*

The keystone, and the payoff of the differentiable architecture. Where Loop B
*brackets* with a discrete candidate set, Loop C *refines* via the
[recon optax solver](../implemented/recon_optimization_plan.md): it differentiates the
forward model and fits the active latents to the measured RHEED by gradient
descent — rapidly enough for closed-loop control, **with credible bands**. It
spans the recon §2.2 ladder, easiest first: **`fit_orientation_beam`** (given a
CIF + a measured pattern, recover the orientation and beam parameters that
generated it — the fast, well-posed alignment/beam-metrology case, symmetry
handled by multistart); **`reconstruct_distribution`** (recover a *probability
distribution* — orientation spread, grain-size, or beam coherence — as a
parametric family or free-form weights, with a calibrated band); then
**`invert_structure`** (the harder structure/composition/thickness fit); and
**`recipe_deviation`** (the per-parameter gap between intended recipe and inverted
reality, normalized to a z-score — the control signal the agent acts on).

*Tasks:* the four automatons above, each a thin wrapper over the frozen recon API
(`fit_geometry_beam` / `reconstruct_distribution` / `solve` / `recipe_deviation`);
all lean on the speed machinery — compilation cache, AOT-exported forward kernels,
the unchecked fast path — so an inversion completes inside a growth-control
cadence.

**Gate G5:** on a synthetic pattern generated from a *known* CIF + orientation +
beam, `fit_orientation_beam --smoke` recovers the orientation and beam parameters
to tolerance (finite gradients, loss decreasing, multistart resolving the symmetry
orbit); `reconstruct_distribution` recovers a *planted distribution shape* with a
calibrated band; `invert_structure` recovers a known structure; `recipe_deviation`
reports the correct gap against a deliberately-mismatched intended recipe; all
smoke-green in the ephemeral CI env with valid `--describe`; a wall-clock budget
assertion guards the "rapid" claim.

### Phase 6 — Diagnostics & ensemble

*Tasks:* `azimuthal_sweep`, `parameter_grid` (sweeps exercising
`distribute_batched` + the compilation cache); `ensemble_average` (beam GSM /
orientation / size `Distribution`); `reconstruct_orientation` (recon, the narrow
single-parameter case of Loop C); `convergence_study`.

**Gate G6:** smoke-green; `ensemble_average` reports mode / effective-count
metrics; `convergence_study` emits a monotone residual-vs-N series; each emits a
valid `--describe`.

### Phase 7 — Ops automatons + hardening

*Tasks:* `audit_invariants`; `export_model` — lower a chosen forward model to a
**portable, version-pinned StableHLO artifact** (via `tools.export_forward`) for
recompile-free reuse on a control/agent node without the rheedium dev stack: the
kinematic engine symbolic in atom count, multislice as bucketed concrete grids,
and the artifact tagged with `rheedium_version` + the input schema. This is the
deployment surface for Loops A and B beyond the dev box. Plus
`automatons/bump_pin.py`; promote `automatons-smoke` from non-blocking to
**required**.

**Gate G7:** `automatons-smoke` is a required status check on the default branch;
the wheel-exclusion test is green; `bump_pin.py` is idempotent (a second run is a
no-op) and rewrites every header to one pinned version (tested); `export_model`
produces a **StableHLO** artifact that **deserializes and runs in a separate
process** (proving recompile-free portability), with a recorded `rheedium_version`
and a same-result check against the in-process forward call.

### Phase 8 — Docs

*Tasks:* a "Running experiments as an agent" guide under `docs/source/guides/`
(handoff contract, `--describe`, result schema, the dev `--with-editable`
override, and the three lab loops); register it in the guides `index` + toctree;
keep `INDEX.md` complete.

**Gate G8:** `cd docs && uv run make html` builds with the new guide; a test
asserts `INDEX.md` lists exactly the `automatons/*.py` present (no drift).

### Gate summary

| Gate | One-line pass condition |
|---|---|
| **A0** | rationalization + recon-optax complete (⇒ framework complete); rationalized API + frozen recon inverse API published; PEP 723 pin targets it |
| G0 | result + `--describe` JSON schemas committed and reviewed |
| G1 | harness tests + schema-snapshot green; `ty`/`ruff` clean |
| G2 | exemplar runs `--smoke` green in clean ephemeral CI; template frozen |
| G3 | **Loop B** — `screen_xyz_ensemble` ranks `.xyz`, bridge scores measured-vs-sim; reproducible; exported kinematic artifact runs over ≥2 atom counts from one lowering |
| G4 | **Loop A** — `rheed_ingest` emits growth state; `growth_monitor` recovers a known oscillation period; `--serve` warms up once then steady per-frame latency |
| G5 | **Loop C** — `fit_orientation_beam` recovers orientation+beam from a CIF; `reconstruct_distribution` recovers a planted shape w/ band; `invert_structure` recovers a known structure; `recipe_deviation` reports the gap; within budget |
| G6 | diagnostics/ensemble smoke-green; valid `--describe` |
| G7 | `automatons-smoke` required; wheel-exclusion + `bump_pin` green; `export_model` StableHLO deserializes + runs in a separate process |
| G8 | docs build with the guide; `INDEX.md` matches the directory |

Phases 0–2 are the critical path — they lock the contract and prove the loop.
Phases 3–5 are the **three mission loops** (B theory, A online, C inverse), each
gated only by the one before it; 6–8 are diagnostics, ops, and docs. Once G2
holds the mission loops can be reordered or parallelized.

---

## 10. Risks

- **Dependency leakage** — a script silently needing a package not transitively
  provided by `rheedium`. Mitigation: the `automatons-smoke` CI job runs in a clean
  ephemeral `uv` env from the PEP 723 header alone; a missing dep fails loudly.
- **Version pin vs local dev drift** — the pinned header won't reflect
  unpublished changes. Mitigation: documented `--with-editable .` override; CI
  smoke uses it so tests track the working tree, not the last release.
- **Reproducibility holes** — uncontrolled randomness or backend variance.
  Mitigation: harness seeds `jax.random` from `--seed` and records
  `rheedium_version` + `jax_backend` in every result; experiments avoid
  `Math.random`-style nondeterminism.
- **Wheel bloat / circular semantics** — avoided by keeping scripts in top-level
  `automatons/` (not `src/`); a build test asserts they are absent from the wheel.
- **Contract rot** — `--describe`/result JSON drifting from what agents expect.
  Mitigation: schema tests in §8; one canonical `emit` in the harness, never
  hand-rolled per script.
  (No directory-naming collision: planning docs live in `plans/`, runnable units
  in `automatons/` — see §2.)
- **Long-running experiments under smoke** — mitigated by a mandatory `--smoke`
  mode every script must honor (tiny grids), enforced by the smoke test.

---

## 11. Diff surface

| Path | Change |
|---|---|
| `src/rheedium/harness/__init__.py` | new subpackage: `Param`, `experiment`, `ExperimentContext`, `emit`, runtime-knob setup, AOT plumbing (`--export` via `tools.export_forward`, warm `--serve` mode); **warmup** (`--warmup` + auto-warm at `--serve` startup; persistent compilation cache on by default; shape-specific); **agent-consumer plumbing** (`--params`/`--validate`/`--estimate`/`--deadline`/`--json`; semantic `Param`; `returns=` schema; typed artifact manifest + `preview_b64`; `error_kind`; `result_key`; NDJSON progress); exports + Routine Listings |
| `schema/automaton_params.schema.json`, `schema/automaton_result.schema.json` | committed JSON-Schema files frozen at G0; `--describe`/`emit` and `--params` validate against them |
| `src/rheedium/__init__.py` | add `harness` to the submodule imports + `__all__` + Routine Listings |
| `automatons/forward_kinematic.py` | Phase-2 exemplar PEP 723 script |
| `automatons/README.md`, `automatons/INDEX.md` | usage, dev override, experiment index |
| `automatons/bump_pin.py` | release-time pin rewriter (PEP 723 script) |
| `tests/test_rheedium/test_harness/` | harness unit tests |
| `tests/.../test_automatons_smoke.py` | discover + `uv run` every `automatons/*.py --smoke` |
| `pyproject.toml` | add `automatons` to `[tool.ty.src].include` at the src-strict rule set (no relaxation); confirm `automatons/` excluded from `uv_build` |
| `.github/workflows/test.yml` | `automatons-smoke` job (ephemeral `uv run` per script) |
| `docs/source/guides/` | "Running experiments as an agent" guide (Phase 8) |

The science stays in `rheedium`; `automatons/` only orchestrates it into one-command,
agent-parseable experiments whose sole declared dependency is `rheedium` itself.

---

## 12. Open decisions

1. **Directory name** — `automatons/` (the unit a calling agent runs) vs
   `agents/` (rejected: conflates the unit with its agent caller) vs
   `experiments/` / `protocols/`. Recommendation: `automatons/`.
2. **Pin policy** — exact `==` per release (recommended, reproducible) vs a git
   source per branch vs floating `>=`. Recommendation: `==`, with the documented
   `--with-editable .` dev override.
3. **CLI layer** — stdlib `argparse` (keeps the dependency list at just
   `rheedium`) vs `tyro`/`typer` (nicer typed CLIs but an extra PEP 723 dep).
   Recommendation: `argparse` in the harness, to preserve the single-dependency
   property.
4. **`automatons-smoke` blocking** — start non-blocking (informational), promote to
   required once stable. Recommendation: yes, staged.
5. **Online streaming model (Loop A)** — per-frame-stateless (agent calls once
   per frame, history in a rolling series file) vs a windowed batch vs a
   long-running `--watch` daemon. Recommendation: **per-frame-stateless**, so
   the run→emit→exit contract and the agent-drives-the-loop model hold; revisit
   `--watch` only if invocation overhead dominates the frame rate.
6. **Loop C inversion budget** — the wall-clock ceiling an `invert_structure`
   run must meet to be "rapid" enough for growth-control cadence, and which
   speed levers it defaults to (compilation cache on, AOT-exported kernel,
   unchecked fast path). Recommendation: set a concrete budget at G5 and assert
   it; default the cache on and the checks off for inversion runs.
7. **Structure passing** (agent-consumer) — inline content vs filesystem path vs
   `artifact://` reference. Recommendation: accept all three; small structures
   inline, large payloads (frames, arrays, big `.xyz`) by reference, never routed
   through the LLM/JSON.
8. **`--params` precedence** — JSON document as the base with explicit CLI flags
   overriding (recommended) vs mutually exclusive JSON-or-flags. Recommendation:
   merge then validate against the schema.
9. **Inline `preview_b64`** — always emit a small thumbnail (recommended for
   interactive agents; full-res PNG remains an artifact) vs only under an opt-in
   `--preview`. Recommendation: always, capped to a small size to bound result
   size.
