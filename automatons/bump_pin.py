# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Rewrite rheedium PEP 723 pins across automaton scripts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_PIN_RE = re.compile(r'rheedium(?:\[cuda\])?==[0-9][^"]*')


def _rewrite_pin(path: Path, version: str) -> bool:
    """Rewrite one automaton pin and return whether the file changed."""
    text: str = path.read_text(encoding="utf-8")
    rewritten: str = _PIN_RE.sub(f"rheedium=={version}", text)
    if rewritten == text:
        return False
    path.write_text(rewritten, encoding="utf-8")
    return True


def main() -> None:
    """Run the pin rewriter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="Version string, for example 2026.6.8."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Automaton directory to rewrite.",
    )
    args = parser.parse_args()

    changed: list[str] = []
    for path in sorted(args.root.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        if _rewrite_pin(path, args.version):
            changed.append(path.name)
    print(
        f"updated {len(changed)} file(s)"
        + (f": {', '.join(changed)}" if changed else "")
    )


if __name__ == "__main__":
    main()
