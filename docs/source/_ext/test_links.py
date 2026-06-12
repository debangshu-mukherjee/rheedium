"""Sphinx extension: turn ``:see:`` test references into GitHub source links.

Throughout the rheedium docstrings, functions and classes point at the test
that exercises them via lines such as::

    :see: :class:`~.test_optimizers.TestGaussNewtonReconstruction`
    :see: :func:`~.test_wavelength_relativistic_consistency`

The referenced objects live in ``tests/`` and are *not* part of the
documented API, so the ``:class:`` / ``:func:`` cross-references never resolve
and render as plain, non-clickable text.

This extension scans ``tests/`` once at build start, maps every test class and
function to its file and line number, and rewrites those references (in place,
via ``autodoc-process-docstring``) into anonymous external hyperlinks pointing
at the corresponding source on GitHub. References that cannot be resolved are
left untouched.
"""

from __future__ import annotations

import ast
import os
import re

# Matches ``:class:`<target>``` / ``:func:`<target>``` role usages. The target
# may carry a leading ``~`` (show last component only) and/or ``.`` (relative).
_ROLE_RE = re.compile(r":(?:class|func|meth|obj):`(~?\.?[\w.]+)`")

# Default GitHub blob base; overridable via the ``test_links_github_base``
# config value in conf.py.
_DEFAULT_GITHUB_BASE = (
    "https://github.com/debangshu-mukherjee/rheedium/blob/main/"
)


def _scan_tests(repo_root):
    """Index every class/function defined under ``tests/``.

    Returns
    -------
    by_module : dict[str, dict]
        ``{module_basename: {"path": relpath, "names": {name: lineno}}}``.
    global_names : dict[str, tuple[str, int]]
        ``{name: (relpath, lineno)}`` across all test files (first wins).
    """
    by_module = {}
    global_names = {}
    tests_dir = os.path.join(repo_root, "tests")
    for dirpath, _dirnames, filenames in os.walk(tests_dir):
        for filename in filenames:
            if not (filename.startswith("test_") and filename.endswith(".py")):
                continue
            abspath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(abspath, repo_root).replace(os.sep, "/")
            try:
                with open(abspath, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=abspath)
            except (OSError, SyntaxError):
                continue
            names = {}
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    names.setdefault(node.name, node.lineno)
                    global_names.setdefault(node.name, (relpath, node.lineno))
            module_basename = filename[:-3]
            by_module[module_basename] = {"path": relpath, "names": names}
    return by_module, global_names


def _resolve(core, by_module, global_names):
    """Resolve a stripped target to ``(relpath, lineno, display)`` or None."""
    parts = core.split(".")
    if len(parts) >= 2:
        module_basename, name = parts[0], parts[-1]
        entry = by_module.get(module_basename)
        if entry and name in entry["names"]:
            return entry["path"], entry["names"][name], name
    name = parts[-1]
    if name in global_names:
        relpath, lineno = global_names[name]
        return relpath, lineno, name
    return None


def _make_processor(app):
    state = {"by_module": None, "global_names": None}

    def process_docstring(_app, _what, _name, _obj, _options, lines):
        if state["by_module"] is None:
            repo_root = os.path.abspath(
                os.path.join(app.srcdir, os.pardir, os.pardir)
            )
            state["by_module"], state["global_names"] = _scan_tests(repo_root)
        base = getattr(
            app.config, "test_links_github_base", _DEFAULT_GITHUB_BASE
        )

        def replace(match):
            target = match.group(1)
            core = target.lstrip("~.")
            resolved = _resolve(
                core, state["by_module"], state["global_names"]
            )
            if resolved is None:
                return match.group(0)
            relpath, lineno, display = resolved
            url = f"{base}{relpath}#L{lineno}"
            return f"`{display} <{url}>`__"

        for index, line in enumerate(lines):
            if "test_" in line and _ROLE_RE.search(line):
                lines[index] = _ROLE_RE.sub(replace, line)

    return process_docstring


def setup(app):
    app.add_config_value(
        "test_links_github_base", _DEFAULT_GITHUB_BASE, "env"
    )
    app.connect("autodoc-process-docstring", _make_processor(app))
    return {"version": "1.0", "parallel_read_safe": True}
