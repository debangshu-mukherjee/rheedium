"""Build the MyST-NB execution cache used by the documentation."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from jupyter_cache import get_cache
from jupyter_cache.executors import load_executor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "docs" / "build" / ".jupyter_cache"
DEFAULT_NOTEBOOK_PATTERNS = (
    "docs/source/tutorials/0*.ipynb",
    "docs/source/interactive_notebooks/*.ipynb",
)


def configure_execution_environment() -> None:
    """Use deterministic, docs-friendly defaults for notebook execution."""
    os.environ.setdefault("BUILDING_DOCS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("RHEEDIUM_TUTORIAL_FAST_DOCS", "1")
    os.environ.setdefault("RHEEDIUM_VISUALIZER_BACKEND", "x3d")


def discover_notebooks(patterns: tuple[str, ...]) -> list[Path]:
    """Find documentation notebooks in a stable order."""
    notebooks: list[Path] = []
    for pattern in patterns:
        notebooks.extend(PROJECT_ROOT.glob(pattern))
    return sorted(
        path.absolute()
        for path in notebooks
        if path.is_file() and ".ipynb_checkpoints" not in path.parts
    )


def parse_notebook_paths(paths: list[str]) -> list[Path]:
    """Normalize user-supplied notebook paths."""
    notebooks: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        notebooks.append(path.absolute())
    return notebooks


def sync_project_notebooks(
    cache_path: Path,
    notebooks: list[Path],
    *,
    prune_stale: bool,
) -> None:
    """Stage the current docs notebooks and remove stale project entries."""
    cache = get_cache(str(cache_path))
    expected_uris = {str(path) for path in notebooks}

    for record in cache.list_project_records():
        if prune_stale and record.uri not in expected_uris:
            cache.remove_nb_from_project(record.pk)

    for notebook in notebooks:
        cache.add_nb_to_project(
            str(notebook),
            read_data={"name": "nbformat", "type": "plugin"},
        )


def build_cache(
    *,
    cache_path: Path,
    notebooks: list[Path],
    executor_name: str,
    timeout: int,
    force: bool,
    prune_stale: bool,
) -> int:
    """Execute stale notebooks and store their outputs in jupyter-cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sync_project_notebooks(
        cache_path,
        notebooks,
        prune_stale=prune_stale,
    )

    cache = get_cache(str(cache_path))
    notebook_uris = [str(path) for path in notebooks]
    project_records = cache.list_project_records(filter_uris=notebook_uris)
    stale_records = (
        project_records
        if force
        else cache.list_unexecuted(filter_uris=notebook_uris)
    )
    skipped_count = len(project_records) - len(stale_records)

    print(f"Cache path: {cache_path}")
    print(f"Project notebooks: {len(project_records)}")
    print(f"Skipped unchanged: {skipped_count}")

    if not stale_records:
        print("Executed successfully: 0")
        print("Execution exceptions: 0")
        print("Execution errors: 0")
        return 0

    logger = logging.getLogger("rheedium.notebook_cache")
    executor = load_executor(executor_name, cache, logger=logger)
    result = executor.run_and_cache(
        filter_uris=notebook_uris,
        timeout=timeout,
        force=force,
    )

    print(f"Executed successfully: {len(result.succeeded)}")
    print(f"Execution exceptions: {len(result.excepted)}")
    print(f"Execution errors: {len(result.errored)}")

    if result.excepted or result.errored:
        for uri in result.excepted + result.errored:
            print(f"Failed notebook: {uri}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute documentation notebooks into the MyST-NB/Jupyter cache."
        )
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Jupyter cache directory used by docs/source/conf.py.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help=(
            "Notebook path to execute. May be repeated. Defaults to all docs "
            "tutorial and interactive notebooks."
        ),
    )
    parser.add_argument(
        "--executor",
        choices=(
            "local-serial",
            "local-parallel",
            "temp-serial",
            "temp-parallel",
        ),
        default="local-serial",
        help="jupyter-cache executor implementation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-cell execution timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute notebooks even when matching cached outputs exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the docs notebooks and populate the shared docs cache."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_execution_environment()

    cache_path = args.cache_path
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    notebooks = (
        parse_notebook_paths(args.notebook)
        if args.notebook
        else discover_notebooks(DEFAULT_NOTEBOOK_PATTERNS)
    )
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    missing = [path for path in notebooks if not path.exists()]
    if missing:
        for path in missing:
            print(f"Notebook not found: {path}", file=sys.stderr)
        return 1

    return build_cache(
        cache_path=cache_path.absolute(),
        notebooks=notebooks,
        executor_name=args.executor,
        timeout=args.timeout,
        force=args.force,
        prune_stale=not args.notebook,
    )


if __name__ == "__main__":
    raise SystemExit(main())
