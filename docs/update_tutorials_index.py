#!/usr/bin/env python3
"""Export Marimo tutorial notebooks and regenerate the tutorials index.

For each ``tutorials/*.py`` Marimo notebook this script:

1. Exports it to a self-contained, browser-runnable HTML bundle via
   ``marimo export html-wasm`` into ``docs/source/_extra/marimo/<stem>/``.
   Sphinx is configured (via ``html_extra_path``) to copy ``_extra``
   verbatim into the build root, so each notebook ends up at
   ``<build>/marimo/<stem>/index.html``.
2. Regenerates ``tutorials/index.rst`` with one link per notebook
   pointing at the corresponding generated HTML bundle.

Run as part of the docs build (see ``.readthedocs.yaml``) or manually
with ``python docs/update_tutorials_index.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent
TUTORIALS_DIR = REPO_ROOT / "tutorials"
EXTRA_DIR = DOCS_DIR / "source" / "_extra" / "marimo"


def export_notebook(notebook: Path, out_dir: Path) -> None:
    """Export a single Marimo notebook to a wasm HTML bundle."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(notebook),
            "-o",
            str(out_dir),
            "--mode",
            "run",
            "-f",
        ],
        check=True,
    )


def write_index(notebooks: list[Path]) -> None:
    """Write ``tutorials/index.rst`` linking to each exported notebook."""
    lines = [
        "Tutorials",
        "=========",
        "",
        "Interactive `Marimo <https://marimo.io>`_ notebooks demonstrating",
        "how to use Rheedium for RHEED pattern simulation. Each tutorial",
        "below runs entirely in your browser via Pyodide — no installation",
        "required.",
        "",
        ".. note::",
        "",
        "   To run a notebook locally with full performance, clone the",
        "   repository, install Marimo (``pip install marimo``), and",
        "   launch ``marimo edit tutorials/<notebook>.py``.",
        "",
        "Available tutorials",
        "-------------------",
        "",
    ]
    for nb in notebooks:
        title = nb.stem.replace("_", " ")
        lines.append(f"- `{title} <../marimo/{nb.stem}/index.html>`_")
    lines.append("")

    index_file = TUTORIALS_DIR / "index.rst"
    index_file.write_text("\n".join(lines))
    print(f"Wrote {index_file} with {len(notebooks)} tutorials")


def main() -> None:
    notebooks = sorted(TUTORIALS_DIR.glob("*.py"))
    if not notebooks:
        print("No Marimo notebooks found in tutorials/", file=sys.stderr)
        return

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    for nb in notebooks:
        out_dir = EXTRA_DIR / nb.stem
        print(f"Exporting {nb.name} -> {out_dir.relative_to(REPO_ROOT)}")
        export_notebook(nb, out_dir)

    write_index(notebooks)


if __name__ == "__main__":
    main()
