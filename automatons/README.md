# Rheedium Automatons

`automatons/` contains single-file, agent-runnable experiment units. Each
script is executable with `uv run` and emits a machine-readable JSON result on
the final stdout line.

```bash
uv run automatons/forward_kinematic.py --smoke --outdir /tmp/rh-auto
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
