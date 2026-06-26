"""Tests for the published testing-documentation contract.

Extended Summary
----------------
Validates that the test suite itself is documented as a public validation
reference: every test module and ``Test*`` class has prose, every ``test_*``
method has explicit what/how sections, and tests avoid source-package export
patterns.
"""

import ast
from pathlib import Path

import chex

_EXTENDED_SUMMARY = "Extended Summary\n----------------"
_NOTES = "Notes\n-----"


def _repo_root() -> Path:
    """Return the repository root for this test module."""
    root: Path = Path(__file__).resolve().parents[2]
    return root


def _test_paths(root: Path) -> list[Path]:
    """Return every rendered ``tests.test_rheedium`` test module path."""
    paths: list[Path] = sorted(
        (root / "tests" / "test_rheedium").rglob("test_*.py")
    )
    return paths


def _parse_module(path: Path) -> ast.Module:
    """Parse a Python test module."""
    module: ast.Module = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    return module


def _qualified_name(path: Path, node: ast.AST) -> str:
    """Return a stable location string for one AST node."""
    node_name: str = getattr(node, "name", "<module>")
    qualified_name: str = f"{path}:{node.lineno}:{node_name}"
    return qualified_name


def _has_what_how_sections(docstring: str) -> bool:
    """Return whether a test docstring has the required sections."""
    has_sections: bool = _EXTENDED_SUMMARY in docstring and _NOTES in docstring
    return has_sections


class TestTestingDocumentationContract(chex.TestCase):
    """Validate the rendered test suite documentation contract."""

    def test_test_modules_and_classes_have_docstrings(self) -> None:
        r"""Every rendered test module and ``Test*`` class is documented.

        Extended Summary
        ----------------
        Verifies that the Test Reference has prose at the module and class
        levels, so each page states the validation scope before listing
        individual test methods.

        Notes
        -----
        It parses every ``tests/test_rheedium/**/test_*.py`` module with the
        standard library AST and records missing module or class docstrings as
        stable file/line diagnostics.
        """
        root: Path = _repo_root()
        missing: list[str] = []
        for path in _test_paths(root):
            module: ast.Module = _parse_module(path)
            if ast.get_docstring(module) is None:
                missing.append(f"{path}:1:<module>")
            for node in module.body:
                is_test_class: bool = isinstance(
                    node, ast.ClassDef
                ) and node.name.startswith("Test")
                if is_test_class and ast.get_docstring(node) is None:
                    missing.append(_qualified_name(path, node))

        self.assertEqual(missing, [])

    def test_test_methods_have_what_how_docstrings(self) -> None:
        r"""Every ``test_*`` method documents what it checks and how.

        Extended Summary
        ----------------
        Verifies the T3 gate from ``testing_docs.md``: each test method has a
        summary plus explicit ``Extended Summary`` and ``Notes`` sections.

        Notes
        -----
        It scans top-level tests and class methods through the AST, then
        reports any missing section by file, line number, and function name so
        the failure is directly actionable.
        """
        root: Path = _repo_root()
        missing: list[str] = []
        for path in _test_paths(root):
            module: ast.Module = _parse_module(path)
            for node in ast.walk(module):
                is_test_function: bool = isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                if not is_test_function:
                    continue
                if not node.name.startswith("test_"):
                    continue
                docstring: str = ast.get_docstring(node) or ""
                if not _has_what_how_sections(docstring):
                    missing.append(_qualified_name(path, node))

        self.assertEqual(missing, [])

    def test_test_modules_avoid_source_only_doc_patterns(self) -> None:
        r"""Tests do not expose source-style exports or routine listings.

        Extended Summary
        ----------------
        Verifies the CONTRIBUTING rule that test modules document validation
        behavior directly instead of mimicking source package API surfaces.

        Notes
        -----
        It inspects module-level assignments for ``__all__`` and checks module
        docstrings for ``Routine Listings``, reporting any violation with a
        stable module path.
        """
        root: Path = _repo_root()
        violations: list[str] = []
        for path in _test_paths(root):
            module: ast.Module = _parse_module(path)
            module_docstring: str = ast.get_docstring(module) or ""
            if "Routine Listings" in module_docstring:
                violations.append(f"{path}:Routine Listings")
            for node in module.body:
                is_all_assignment: bool = False
                if isinstance(node, ast.Assign):
                    is_all_assignment = any(
                        isinstance(target, ast.Name) and target.id == "__all__"
                        for target in node.targets
                    )
                elif isinstance(node, ast.AnnAssign):
                    is_all_assignment = (
                        isinstance(node.target, ast.Name)
                        and node.target.id == "__all__"
                    )
                if is_all_assignment:
                    violations.append(f"{path}:{node.lineno}:__all__")

        self.assertEqual(violations, [])
