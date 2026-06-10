#!/usr/bin/env python3
"""Generate Read the Docs friendly tutorial pages from Marimo notebooks.

For each ``tutorials/*.py`` Marimo notebook this script:

1. Generates a static reStructuredText page in ``docs/source/tutorials``
   that links to the notebook source and includes the full notebook script
   with ``literalinclude``.
2. Regenerates ``docs/source/tutorials/index.rst`` with a ``toctree``
   pointing at those generated tutorial pages.

This avoids depending on ``marimo export html`` during Read the Docs builds,
which is fragile for heavy JAX-based notebooks. The notebooks remain fully
editable and runnable locally with Marimo, while RTD gets a plain Sphinx
representation that is reliable to build.
"""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent
TUTORIALS_DIR = REPO_ROOT / "tutorials"
DOCS_TUTORIALS_DIR = DOCS_DIR / "source" / "tutorials"


def tutorial_title(notebook: Path) -> str:
    """Convert a tutorial filename into a readable title."""
    return notebook.stem.replace("_", " ")


def write_tutorial_page(notebook: Path) -> None:
    """Write a static tutorial page for one Marimo notebook."""
    page_path = DOCS_TUTORIALS_DIR / f"{notebook.stem}.rst"
    source_rel = Path("../../../tutorials") / notebook.name
    title = tutorial_title(notebook)
    title_underline = "=" * len(title)
    page_lines = [
        title,
        title_underline,
        "",
        (
            "This tutorial is authored as a "
            "`Marimo <https://marimo.io>`_ notebook,"
        ),
        (
            "but Read the Docs renders it as a static source page for "
            "reliability."
        ),
        "",
        "To run it interactively on your own machine:",
        "",
        ".. code-block:: bash",
        "",
        f"   marimo edit tutorials/{notebook.name}",
        "",
        f"Notebook source: ``tutorials/{notebook.name}``",
        "",
        ".. literalinclude:: " + str(source_rel),
        "   :language: python",
        "   :linenos:",
        "",
    ]
    page_path.write_text("\n".join(page_lines))
    print(f"Wrote {page_path.relative_to(REPO_ROOT)}")


def write_index(notebooks: list[Path]) -> None:
    """Write ``docs/source/tutorials/index.rst`` with a Sphinx toctree."""
    lines = [
        "Tutorials",
        "=========",
        "",
        "Tutorial notebooks for Rheedium. The original notebooks are written",
        (
            "in `Marimo <https://marimo.io>`_, but on Read the Docs "
            "they are shown"
        ),
        "as static source pages so the documentation build does not depend on",
        "live notebook export or execution.",
        "",
        ".. note::",
        "",
        (
            "   For true notebook interactivity, clone the repository, "
            "install Marimo"
        ),
        "   (``pip install marimo``), and launch",
        "   ``marimo edit tutorials/<notebook>.py``.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for nb in notebooks:
        lines.append(f"   {nb.stem}")
    lines.append("")

    index_file = DOCS_TUTORIALS_DIR / "index.rst"
    index_file.write_text("\n".join(lines))
    print(f"Wrote {index_file} with {len(notebooks)} tutorials")


def main() -> None:
    """Generate static tutorial pages and their index."""
    notebooks = sorted(TUTORIALS_DIR.glob("*.py"))
    DOCS_TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)
    for nb in notebooks:
        write_tutorial_page(nb)

    write_index(notebooks)


if __name__ == "__main__":
    main()
