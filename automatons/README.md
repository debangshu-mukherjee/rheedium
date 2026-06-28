# Rheedium Automatons

`automatons/` contains single-file, agent-runnable experiment units. Each
script is executable with `uv run` and emits a machine-readable JSON result on
the final stdout line.

```bash
uv run automatons/forward_kinematic.py --smoke --outdir /tmp/rh-auto
uv run automatons/forward_multislice.py --smoke --outdir /tmp/rh-multi
uv run automatons/forward_reflection.py --smoke --outdir /tmp/rh-refl
uv run automatons/screen_xyz_ensemble.py --smoke --outdir /tmp/rh-screen
uv run automatons/match_measured_to_simulated.py --smoke --outdir /tmp/rh-match
uv run automatons/fit_orientation_beam.py --smoke --outdir /tmp/rh-fit
uv run automatons/reconstruct_distribution.py --smoke --outdir /tmp/rh-dist
uv run automatons/invert_structure.py --smoke --outdir /tmp/rh-invert
uv run automatons/recipe_deviation.py --smoke --outdir /tmp/rh-dev
uv run automatons/rheed_ingest.py --smoke --outdir /tmp/rh-ingest
uv run automatons/growth_monitor.py --smoke --outdir /tmp/rh-growth
```

During local development, run against the working tree instead of the PEP 723
pin:

```bash
uv run --with-editable . automatons/forward_kinematic.py --smoke
```

Every automaton supports shared harness flags:

- `--describe` emits the parameter schema and declared return schema.
- `--params <file | - | json-string>` supplies machine JSON input.
- `--validate` checks merged params without running the pipeline.
- `--estimate` reports a lightweight cost estimate when available.
- `--outdir`, `--seed`, `--smoke`, `--cache`, `--unchecked`, `--deadline`, and
  `--json` are available uniformly.

Artifacts are written under `--outdir`; result JSON contains typed manifest
entries with `role`, `mime`, and relative `path`.
