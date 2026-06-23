#!/usr/bin/env python3
"""Regenerate the tutorial notebook index.

The tutorials are Jupyter notebooks, optionally paired with Jupytext percent
scripts. Sphinx renders the ``.ipynb`` files directly through myst-nb, so this
script only refreshes the ``tutorials/index.rst`` toctree.
"""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent
TUTORIALS_DIR = REPO_ROOT / "tutorials"
INDEX_FILE = TUTORIALS_DIR / "index.rst"


def write_index(notebooks: list[Path]) -> None:
    """Write ``tutorials/index.rst`` with a Sphinx notebook toctree."""
    lines = [
        "Tutorials",
        "=========",
        "",
        (
            "Tutorial notebooks for Rheedium. They are authored as Jupyter "
            "notebooks paired"
        ),
        (
            "with Jupytext percent scripts where available, so they can be "
            "opened directly in"
        ),
        (
            "VS Code or Jupyter while still keeping reviewable Python sources "
            "in git."
        ),
        "",
        ".. note::",
        "",
        (
            "   For remote development, open "
            "``tutorials/<notebook>.ipynb`` in VS Code over"
        ),
        (
            "   Remote-SSH and select the project kernel. After editing paired "
            "notebooks,"
        ),
        (
            "   run ``jupytext --sync tutorials/<notebook>.ipynb`` before "
            "committing."
        ),
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    lines.extend(f"   {notebook.stem}" for notebook in notebooks)
    lines.append("")
    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {INDEX_FILE.relative_to(REPO_ROOT)} with {len(notebooks)} tutorials")


def main() -> None:
    """Regenerate the tutorial index from committed notebook files."""
    notebooks = sorted(TUTORIALS_DIR.glob("0*.ipynb"))
    write_index(notebooks)


if __name__ == "__main__":
    main()
