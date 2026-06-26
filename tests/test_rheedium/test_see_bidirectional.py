"""Tests for source/test documentation cross-reference contracts.

Extended Summary
----------------
Validates the testing-documentation contract from ``testing_docs.md``: source
``:see:`` targets resolve to real test objects, source-targeted tests link back
to source objects, and the rendered Test Reference covers every test module.
"""

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import chex

_SEE_RE = re.compile(
    r":see:\s+:(?P<role>class|func|meth|obj):`(?P<target>[^`]+)`"
)
_AUTOMODULE_RE = re.compile(r"^\.\. automodule:: (?P<module>tests\..+)$")


def _repo_root() -> Path:
    """Return the repository root for this test module."""
    root: Path = Path(__file__).resolve().parents[2]
    return root


def _module_name(root: Path, path: Path) -> str:
    """Return the import path represented by a Python file."""
    rel_path: Path = path.relative_to(root).with_suffix("")
    module_name: str = ".".join(rel_path.parts)
    return module_name


def _source_module_name(root: Path, path: Path) -> str:
    """Return the import path represented by a source Python file."""
    rel_path: Path = path.relative_to(root / "src").with_suffix("")
    module_name: str = ".".join(rel_path.parts)
    return module_name


def _parse_module(path: Path) -> ast.Module:
    """Parse one Python source file into an AST module."""
    module: ast.Module = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    return module


def _source_symbols(root: Path) -> set[str]:
    """Collect resolvable source object names and exported aliases."""
    symbols: set[str] = set()
    for path in (root / "src" / "rheedium").rglob("*.py"):
        module_name: str = _source_module_name(root, path)
        module: ast.Module = _parse_module(path)
        for node in module.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                symbols.add(f"{module_name}.{node.name}")
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                has_all_target: bool = any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                if has_all_target:
                    value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            ):
                value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                package_name: str = module_name.removesuffix(".__init__")
                for item in value.elts:
                    if isinstance(item, ast.Constant) and isinstance(
                        item.value, str
                    ):
                        symbols.add(f"{package_name}.{item.value}")
    return symbols


def _test_objects(
    root: Path,
) -> tuple[dict[tuple[str, str], str], Counter[str]]:
    """Index documented test classes and top-level test functions."""
    objects: dict[tuple[str, str], str] = {}
    name_counts: Counter[str] = Counter()
    for path in (root / "tests" / "test_rheedium").rglob("test_*.py"):
        module_name: str = _module_name(root, path)
        module_basename: str = path.stem
        module: ast.Module = _parse_module(path)
        for node in module.body:
            is_test_function: bool = isinstance(
                node, ast.FunctionDef
            ) and node.name.startswith("test_")
            if isinstance(node, ast.ClassDef) or is_test_function:
                qualified_name: str = f"{module_name}.{node.name}"
                objects[(module_name, node.name)] = qualified_name
                objects[(module_basename, node.name)] = qualified_name
                name_counts[node.name] += 1
                if name_counts[node.name] == 1:
                    objects[("", node.name)] = qualified_name
                elif ("", node.name) in objects:
                    del objects[("", node.name)]
    return objects, name_counts


def _resolve_test_target(
    target: str,
    test_objects: dict[tuple[str, str], str],
) -> str | None:
    """Resolve a compact ``:see:`` test target to a test object."""
    core: str = target.lstrip("~.")
    parts: list[str] = core.split(".")
    name: str = parts[-1]
    if len(parts) >= 2:
        full_module: str = ".".join(parts[:-1])
        if (full_module, name) in test_objects:
            resolved_full: str = test_objects[(full_module, name)]
            return resolved_full
        module_basename: str = parts[-2]
        if (module_basename, name) in test_objects:
            resolved_module: str = test_objects[(module_basename, name)]
            return resolved_module
    if ("", name) in test_objects:
        resolved_global: str = test_objects[("", name)]
        return resolved_global
    return None


def _source_forward_targets(
    root: Path,
) -> tuple[dict[str, set[str]], list[str]]:
    """Map test objects targeted by source ``:see:`` refs to source symbols."""
    test_objects: dict[tuple[str, str], str]
    test_objects, _name_counts = _test_objects(root)
    source_symbols: set[str] = _source_symbols(root)
    targets: dict[str, set[str]] = defaultdict(set)
    unresolved: list[str] = []
    for path in (root / "src" / "rheedium").rglob("*.py"):
        module_name: str = _source_module_name(root, path)
        module: ast.Module = _parse_module(path)
        for node in module.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            docstring: str = ast.get_docstring(node) or ""
            for match in _SEE_RE.finditer(docstring):
                target: str = match.group("target")
                resolved: str | None = _resolve_test_target(
                    target,
                    test_objects,
                )
                source_symbol: str = f"{module_name}.{node.name}"
                if resolved is None:
                    unresolved.append(f"{source_symbol} -> {target}")
                    continue
                aliases: set[str] = {
                    symbol
                    for symbol in source_symbols
                    if symbol.endswith(f".{node.name}")
                }
                aliases.add(source_symbol)
                targets[resolved].update(aliases)
    mapped_targets: dict[str, set[str]] = dict(targets)
    return mapped_targets, unresolved


def _test_object_docstrings(root: Path) -> dict[str, str]:
    """Collect docstrings for rendered test classes and top-level tests."""
    docstrings: dict[str, str] = {}
    for path in (root / "tests" / "test_rheedium").rglob("test_*.py"):
        module_name: str = _module_name(root, path)
        module: ast.Module = _parse_module(path)
        for node in module.body:
            is_test_function: bool = isinstance(
                node, ast.FunctionDef
            ) and node.name.startswith("test_")
            if isinstance(node, ast.ClassDef) or is_test_function:
                qualified_name: str = f"{module_name}.{node.name}"
                docstrings[qualified_name] = ast.get_docstring(node) or ""
    return docstrings


def _source_backrefs(docstring: str, source_symbols: set[str]) -> set[str]:
    """Return valid source-symbol backrefs from one test docstring."""
    refs: set[str] = set()
    for match in _SEE_RE.finditer(docstring):
        target: str = match.group("target").lstrip("~.")
        if target in source_symbols:
            refs.add(target)
    return refs


def _documented_test_modules(root: Path) -> set[str]:
    """Collect test modules rendered by ``docs/source/tests`` pages."""
    modules: set[str] = set()
    docs_root: Path = root / "docs" / "source" / "tests"
    for path in docs_root.glob("*.rst"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match: re.Match[str] | None = _AUTOMODULE_RE.match(line)
            if match is not None:
                modules.add(match.group("module"))
    return modules


class TestBidirectionalSeeReferences(chex.TestCase):
    """Validate source/test ``:see:`` links for rendered test docs."""

    def test_source_see_targets_resolve_to_test_objects(self) -> None:
        """Every source ``:see:`` target names an existing test object."""
        root: Path = _repo_root()
        _targets: dict[str, set[str]]
        unresolved: list[str]
        _targets, unresolved = _source_forward_targets(root)

        self.assertEqual(unresolved, [])

    def test_source_targeted_tests_link_back_to_source_objects(self) -> None:
        """Every source-targeted test object links back to source code."""
        root: Path = _repo_root()
        targets: dict[str, set[str]]
        unresolved: list[str]
        targets, unresolved = _source_forward_targets(root)
        docstrings: dict[str, str] = _test_object_docstrings(root)
        source_symbols: set[str] = _source_symbols(root)

        missing_backrefs: list[str] = []
        invalid_targets: list[str] = []
        for test_object in sorted(targets):
            docstring: str = docstrings[test_object]
            backrefs: set[str] = _source_backrefs(docstring, source_symbols)
            if not backrefs:
                missing_backrefs.append(test_object)
            if not (backrefs & targets[test_object]):
                invalid_targets.append(test_object)

        self.assertEqual(unresolved, [])
        self.assertEqual(missing_backrefs, [])
        self.assertEqual(invalid_targets, [])

    def test_test_reference_pages_cover_all_test_modules(self) -> None:
        """Every mirrored test module is included in the Test Reference."""
        root: Path = _repo_root()
        expected: set[str] = {
            _module_name(root, path)
            for path in (root / "tests" / "test_rheedium").rglob("test_*.py")
        }
        documented: set[str] = _documented_test_modules(root)
        missing: list[str] = sorted(expected - documented)

        self.assertEqual(missing, [])
