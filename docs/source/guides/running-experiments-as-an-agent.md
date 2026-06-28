# Running Experiments as an Agent

Rheedium automatons are process-bound experiment units under `automatons/`.
Each script wraps a complete workflow, declares its PEP 723 dependency pin,
accepts structured inputs, writes typed artifacts, and emits one
machine-readable JSON result as the final stdout line. The calling agent owns
orchestration; the automaton owns a single scientific run.

Use the catalog first:

- `automatons/INDEX.md` lists every runnable automaton and its role.
- `automatons/README.md` shows the shared command shape and smoke examples.

## Invocation Contract

Every top-level automaton except `bump_pin.py` follows the same harness
contract:

```bash
uv run automatons/forward_kinematic.py \
  --smoke \
  --outdir /tmp/rh-forward \
  --json
```

For local development against the working tree, use the editable override:

```bash
uv run --with-editable . python automatons/forward_kinematic.py \
  --smoke \
  --outdir /tmp/rh-forward \
  --json
```

That form keeps the local `pyproject.toml` version as the test source of truth.
Running the script path directly asks uv to resolve the PEP 723 pin first, which
is right for published handoff but can fail before a bumped local version exists
on PyPI.

The final stdout line is a JSON object matching
[`schema/automaton_result.schema.json`](../../../schema/automaton_result.schema.json).
It includes:

- `status`, `experiment`, `rheedium_version`, `jax_backend`, `seed`
- `params`, the fully merged and validated request
- `metrics`, the compact science-facing summary
- `artifacts`, typed relative paths under `--outdir`
- `result_key`, a deterministic key from experiment name, params, seed, and
  rheedium version

Anything before the final line is human log output. Agents should parse the last
line only.

## Discover Parameters

Use `--describe` before calling an unfamiliar automaton:

```bash
uv run --with-editable . python automatons/export_model.py --describe
```

The description payload contains the parameter list, JSON Schema properties,
return-role declarations, and shared harness flags. This lets an agent build a
validated request without reading Python source.

For structured inputs, pass `--params` as a JSON string, a file path, or `-` for
stdin:

```bash
uv run --with-editable . python automatons/recipe_deviation.py \
  --params recipe_request.json \
  --outdir /tmp/rh-recipe \
  --json
```

Use `--validate` to check merged params without running the science path, and
`--estimate` to request a lightweight cost estimate when a script declares one.

## Smoke, Seeds, and Reproducibility

Every automaton has a `--smoke` path that uses tiny built-in fixtures or
committed test data. Agents should use smoke mode for capability checks,
scheduling probes, and CI. Normal runs omit `--smoke` and provide real input
files.

Use `--seed` whenever deterministic ranking or synthetic fixtures matter:

```bash
uv run --with-editable . python automatons/screen_xyz_ensemble.py \
  --smoke \
  --seed 123 \
  --outdir /tmp/rh-screen \
  --json
```

For identical script, params, seed, and rheedium version, the harness records
the same `result_key`. Binary previews and wall-clock time are not part of the
scientific identity.

## Artifacts and Plots

Artifacts are relative to `--outdir` and carry a `role`, `mime`, and `path`.
Common roles include detector images, numeric `.npz` arrays, rankings,
distribution payloads, audit reports, and StableHLO export manifests. Pattern
and diagnostic images use the project phosphor colormap as the primary
colorscheme, so an agent can render previews consistently without per-script
special cases.

Prefer the compact `metrics` object for routing decisions and open artifacts
only when the agent needs detailed rows, arrays, images, or deployment bytes.

## Runtime Knobs

The harness exposes the same runtime controls everywhere:

- `--cache` enables the persistent XLA compilation cache before the first
  compile.
- `--unchecked` requests the fast path for already-validated data.
- `--deadline <seconds>` turns a wall-clock limit into a structured timeout.
- `--warmup` records the warmup intent for scripts that provide warmable
  kernels.

For portable deployment, `automatons/export_model.py` writes a version-tagged
StableHLO artifact and manifest, then proves that the artifact deserializes and
runs in a separate Python process. See also
[Parallelization and Compilation](parallelization-and-compilation.md) for the
shape-polymorphic export rules.

## The Three Lab Loops

Loop A is online RHEED truth. `rheed_ingest.py` turns one detector frame into
growth observables, and `growth_monitor.py` turns a rolling series into
oscillation period and roughness state. The resolved operating mode is
per-frame stateless: run, emit, exit; the calling agent accumulates history.

Loop B is theory in the loop. `forward_kinematic.py`,
`forward_multislice.py`, and `forward_reflection.py` generate simulated
patterns, while `screen_xyz_ensemble.py` and
`match_measured_to_simulated.py` rank candidate structures or simulated images
against measured data.

Loop C is inversion. `fit_orientation_beam.py`,
`reconstruct_distribution.py`, `invert_structure.py`, and
`recipe_deviation.py` recover latent state and expose the recipe-vs-actual
control signal. These wrap the frozen `rheedium.recon` API and produce compact
metrics plus detailed fit artifacts.

Diagnostics and ops support the loops: sweeps, parameter grids, ensemble
averages, orientation reconstruction, convergence studies, invariant audits,
pin rewriting, and StableHLO export live alongside the mission scripts.

## Agent Pattern

A robust agent loop is:

1. Read `automatons/INDEX.md` to choose a tool.
2. Run `--describe` and validate a planned request.
3. Run with `--outdir`, `--seed`, and `--json`.
4. Parse the final stdout line.
5. Branch on `status`, `error_kind`, and `metrics`.
6. Open artifacts only when the compact metrics are insufficient.

This keeps the scientific code in `rheedium`, the experiment wiring in
`automatons/`, and the autonomous decision logic in the calling agent.
