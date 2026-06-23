"""Validate that every README image renders on both GitHub and PyPI.

PyPI's long-description renderer (used for the project page) shows images only
when they are served as **raster** files over an **absolute https URL** -- it
drops SVGs and ignores repository-relative paths. GitHub is more permissive, so
relative or SVG links silently pass review there and break only on PyPI.

This script is the guard against that drift. It checks that every ``<img>`` /
Markdown image in ``README.md``:

1. uses an absolute ``https://`` URL (no repo-relative paths);
2. is a raster format (``.png`` / ``.jpg``); and
3. for images hosted in this repository (``raw.githubusercontent.com`` on the
   pinned branch), resolves to a file that actually exists on disk -- i.e. the
   cached PNG was committed.

It also verifies the converse: every figure the generator marks as a README
PNG (:data:`generate_figures.README_PNG_FIGURES`) is referenced by the README,
so the cache and the document cannot fall out of sync.

Run it directly (used by CI)::

    uv run python docs/check_readme_figures.py

Exit code is ``0`` when everything is consistent and ``1`` otherwise, with a
human-readable list of problems on stderr.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
README: Final[Path] = PROJECT_ROOT / "README.md"

# Must match the branch encoded in the README's raw URLs and the figure cache.
RAW_HOST: Final[str] = "raw.githubusercontent.com"
RAW_PREFIX: Final[str] = (
    f"https://{RAW_HOST}/debangshu-mukherjee/rheedium/main/"
)
RASTER_SUFFIXES: Final[tuple[str, ...]] = (".png", ".jpg", ".jpeg")

# Any image src: HTML <img src="..."> or Markdown ![alt](...).
_IMG_SRC: Final[re.Pattern[str]] = re.compile(r'<img[^>]*\bsrc="([^"]+)"')
_MD_IMG: Final[re.Pattern[str]] = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _readme_image_urls(text: str) -> list[str]:
    """Return every image URL referenced in the README, in document order."""
    return [*_IMG_SRC.findall(text), *_MD_IMG.findall(text)]


def _check_image(url: str) -> list[str]:
    """Return a list of problems for a single image URL (empty if valid).

    The strict raster-and-exists rules apply only to figures hosted in this
    repository (the pinned ``raw.githubusercontent.com`` prefix), which we
    generate and commit. External images -- shields.io / codecov / Zenodo
    badges and the like -- only need to be absolute URLs; their formats are
    outside our control and render fine on PyPI as-is.
    """
    problems: list[str] = []
    if not url.startswith("https://"):
        problems.append(
            f"{url!r}: not an absolute https URL "
            "(PyPI will not render relative paths)."
        )
        return problems
    if not url.startswith(RAW_PREFIX):
        # External (badge / third-party) image: absolute https is enough.
        return problems
    rel: str = url[len(RAW_PREFIX) :]
    if not url.lower().endswith(RASTER_SUFFIXES):
        problems.append(
            f"{url!r}: in-repo figure is not a raster image "
            "(PyPI does not reliably render SVG; reference the committed PNG)."
        )
    elif not (PROJECT_ROOT / rel).is_file():
        problems.append(
            f"{url!r}: cached file {rel!r} is missing -- regenerate it "
            "(uv run python docs/source/guides/figures/generate_figures.py) "
            "and commit the PNG."
        )
    return problems


def _check_generator_sync(urls: list[str]) -> list[str]:
    """Ensure every generator-marked README PNG is actually referenced."""
    # Imported lazily via a runtime sys.path insertion; the static checker
    # cannot follow it, hence the suppression.
    sys.path.insert(0, str(PROJECT_ROOT / "docs" / "source" / "guides"))
    try:
        from figures.generate_figures import (  # ty: ignore[unresolved-import]
            README_PNG_FIGURES,
        )
    except ImportError as exc:  # pragma: no cover - import wiring
        return [f"could not import README_PNG_FIGURES: {exc}"]

    referenced: str = "\n".join(urls)
    missing: list[str] = sorted(
        stem
        for stem in README_PNG_FIGURES
        if f"{stem}.png" not in referenced
    )
    return [
        f"figure {stem!r} is generated as a README PNG but not referenced "
        "in README.md (remove it from README_PNG_FIGURES or add it back)."
        for stem in missing
    ]


def main() -> int:
    """Validate README image links; return process exit code."""
    text: str = README.read_text(encoding="utf-8")
    urls: list[str] = _readme_image_urls(text)

    problems: list[str] = []
    for url in urls:
        problems.extend(_check_image(url))
    problems.extend(_check_generator_sync(urls))

    if problems:
        print("README figure check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"README figure check OK: {len(urls)} image(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
