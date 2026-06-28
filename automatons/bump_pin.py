# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Rewrite rheedium PEP 723 pins across automaton scripts."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

_PIN_RE = re.compile(r'rheedium(?:\[cuda\])?==[0-9][^"]*')
_PYPROJECT: Path = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Return the ``[project].version`` string from the repo pyproject.toml."""
    with _PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    version: str = str(data["project"]["version"])
    return version


def _rewrite_pin(path: Path, version: str) -> bool:
    """Rewrite one automaton pin and return whether the file changed.

    Reads and writes bytes so existing line endings survive verbatim on every
    platform; text-mode I/O would translate newlines (LF to CRLF on Windows)
    and flip the whole file. Only the pin substring changes.
    """
    text: str = path.read_bytes().decode("utf-8")
    rewritten: str = _PIN_RE.sub(f"rheedium=={version}", text)
    if rewritten == text:
        return False
    path.write_bytes(rewritten.encode("utf-8"))
    return True


def main() -> None:
    """Run the pin rewriter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help=(
            "Version string, e.g. 2026.6.9. Defaults to [project].version "
            "in pyproject.toml when omitted."
        ),
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Automaton directory to rewrite.",
    )
    args = parser.parse_args()

    version: str = args.version or _pyproject_version()
    changed: list[str] = []
    for path in sorted(args.root.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        if _rewrite_pin(path, version):
            changed.append(path.name)
    print(
        f"pinned rheedium=={version}; updated {len(changed)} file(s)"
        + (f": {', '.join(changed)}" if changed else "")
    )


if __name__ == "__main__":
    main()
