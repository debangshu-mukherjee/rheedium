#!/usr/bin/env python3
"""Strip outputs and volatile metadata from Jupyter notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _strip_notebook(path: Path) -> bool:
    """Strip outputs from one notebook.

    Parameters
    ----------
    path : Path
        Notebook file to update in place.

    Returns
    -------
    changed : bool
        True when the notebook was modified.
    """
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs") != []:
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

        metadata = cell.get("metadata")
        if isinstance(metadata, dict) and "trusted" in metadata:
            del metadata["trusted"]
            changed = True

    metadata = notebook.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        del metadata["widgets"]
        changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def main(argv: list[str]) -> int:
    """Strip notebook outputs for all paths passed on the command line."""
    for filename in argv:
        path = Path(filename)
        if path.suffix == ".ipynb" and path.exists():
            _strip_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
