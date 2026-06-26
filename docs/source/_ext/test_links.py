"""Sphinx extension: resolve compact ``:see:`` references into test pages.

Throughout the rheedium docstrings, functions and classes point at the test
that exercises them via lines such as::

    :see: :class:`~.test_optimizers.TestGaussNewtonReconstruction`
    :see: :func:`~.test_wavelength_relativistic_consistency`

The referenced objects live in ``tests/`` and are documented in the Test
Reference. This extension scans ``tests/`` once at build start, maps every
test class and top-level test function to its import path, and rewrites compact
relative references into fully qualified Python-domain references so Sphinx can
link source API docs to rendered validation pages.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any

_ROLE_RE = re.compile(
    r":(?P<role>class|func|meth|obj):`(?P<target>~?\.?[\w.]+)`"
)


def _scan_tests(
    repo_root: str,
) -> tuple[dict[str, dict[str, tuple[str, str]]], dict[str, tuple[str, str]]]:
    """Index every documented test object under ``tests/``."""
    by_module: dict[str, dict[str, tuple[str, str]]] = {}
    global_names: dict[str, tuple[str, str]] = {}
    tests_dir = os.path.join(repo_root, "tests")
    for dirpath, _dirnames, filenames in os.walk(tests_dir):
        for filename in filenames:
            if not (filename.startswith("test_") and filename.endswith(".py")):
                continue
            abspath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(abspath, repo_root)
            module_name = os.path.splitext(relpath)[0].replace(os.sep, ".")
            try:
                with open(abspath, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=abspath)
            except (OSError, SyntaxError):
                continue

            names: dict[str, tuple[str, str]] = {}
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    qualified_name = f"{module_name}.{node.name}"
                    names.setdefault(node.name, ("class", qualified_name))
                    global_names.setdefault(
                        node.name,
                        ("class", qualified_name),
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        qualified_name = f"{module_name}.{node.name}"
                        names.setdefault(node.name, ("func", qualified_name))
                        global_names.setdefault(
                            node.name,
                            ("func", qualified_name),
                        )
            by_module[filename[:-3]] = names
    return by_module, global_names


def _resolve(
    core: str,
    by_module: dict[str, dict[str, tuple[str, str]]],
    global_names: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Resolve a stripped test target to ``(role, qualified_name)``."""
    parts = core.split(".")
    if len(parts) >= 2:
        module_basename = parts[-2]
        name = parts[-1]
        entry = by_module.get(module_basename)
        if entry and name in entry:
            return entry[name]

    name = parts[-1]
    if name in global_names:
        return global_names[name]
    return None


def _make_processor(app: Any) -> Any:
    """Create an autodoc docstring processor with a cached test index."""
    state: dict[str, Any] = {"by_module": None, "global_names": None}

    def process_docstring(
        _app: Any,
        _what: str,
        _name: str,
        _obj: object,
        _options: object,
        lines: list[str],
    ) -> None:
        """Rewrite compact test refs before Sphinx parses the docstring."""
        if state["by_module"] is None:
            repo_root = os.path.abspath(
                os.path.join(app.srcdir, os.pardir, os.pardir)
            )
            state["by_module"], state["global_names"] = _scan_tests(repo_root)

        def replace(match: re.Match[str]) -> str:
            target = match.group("target")
            core = target.lstrip("~.")
            resolved = _resolve(
                core,
                state["by_module"],
                state["global_names"],
            )
            if resolved is None:
                return match.group(0)
            role, qualified_name = resolved
            return f":{role}:`~{qualified_name}`"

        for index, line in enumerate(lines):
            if "test_" in line and _ROLE_RE.search(line):
                lines[index] = _ROLE_RE.sub(replace, line)

    return process_docstring


def setup(app: Any) -> dict[str, object]:
    """Register the compact test-reference resolver."""
    app.connect("autodoc-process-docstring", _make_processor(app))
    return {"version": "1.0", "parallel_read_safe": True}
