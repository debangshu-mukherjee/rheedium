"""Tests for import-time configuration in the top-level rheedium package.

The runtime-check toggle (``RHEEDIUM_DISABLE_RUNTIME_CHECKS``) acts before the
first equinox import, so it can only be observed in a fresh interpreter. Each
case launches a subprocess that imports rheedium and reports the resolved
``EQX_ON_ERROR`` value.
"""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import chex
import pytest

_PROBE: str = (
    "import os, rheedium; "
    "print('EQX=' + os.environ.get('EQX_ON_ERROR', '<unset>'))"
)
_REPO_ROOT: Path = Path(__file__).parents[2]
_TYPE_CONSTRUCTOR_EXEMPTIONS: set[tuple[str, str]] = {
    ("src/rheedium/plots/figuring.py", "create_phosphor_colormap"),
    ("src/rheedium/procs/surface_builder.py", "create_surface_slab"),
}


def _resolved_eqx_on_error(**overrides: str) -> str:
    """Import rheedium in a subprocess and return its ``EQX_ON_ERROR``."""
    env: dict[str, str] = dict(os.environ)
    env.pop("EQX_ON_ERROR", None)
    env.pop("RHEEDIUM_DISABLE_RUNTIME_CHECKS", None)
    env.update(overrides)
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=300,
    )
    line: str = next(
        row for row in result.stdout.splitlines() if row.startswith("EQX=")
    )
    return line.removeprefix("EQX=")


def _base_name(base: ast.expr) -> str:
    """Return a dotted-ish display name for one class base expression."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        prefix: str = _base_name(base.value)
        return f"{prefix}.{base.attr}" if prefix else base.attr
    return ""


class TestRuntimeCheckToggle(chex.TestCase):
    """Tests for the ``RHEEDIUM_DISABLE_RUNTIME_CHECKS`` import-time knob."""

    def test_checks_on_by_default(self) -> None:
        r"""Without the opt-in, rheedium leaves ``EQX_ON_ERROR`` unset.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Without the
        opt-in, rheedium leaves ``EQX_ON_ERROR`` unset.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_package_init``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        assert _resolved_eqx_on_error() == "<unset>"

    def test_opt_in_disables_checks(self) -> None:
        r"""The opt-in sets ``EQX_ON_ERROR=off`` before the equinox import.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The opt-in sets
        ``EQX_ON_ERROR=off`` before the equinox import.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_package_init``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        assert (
            _resolved_eqx_on_error(RHEEDIUM_DISABLE_RUNTIME_CHECKS="1")
            == "off"
        )

    def test_explicit_value_is_respected(self) -> None:
        r"""An explicit ``EQX_ON_ERROR`` wins over the opt-in default.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: An explicit
        ``EQX_ON_ERROR`` wins over the opt-in default.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_package_init``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        assert (
            _resolved_eqx_on_error(
                RHEEDIUM_DISABLE_RUNTIME_CHECKS="1",
                EQX_ON_ERROR="nan",
            )
            == "nan"
        )


class TestNamingGuards(chex.TestCase):
    """Tests for public naming consistency."""

    def test_code_uses_energy_kev_not_old_beam_name(self) -> None:
        r"""W7: code and tests should use energy_kev for keV beam energy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: W7: code and tests
        should use energy_kev for keV beam energy.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_package_init``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        old_name = "voltage" + "_kv"
        offenders: list[str] = []
        for root_name in ("src", "tests"):
            root = _REPO_ROOT / root_name
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix in {".py", ".json"}
                ):
                    text = path.read_text(encoding="utf-8")
                    if old_name in text:
                        offenders.append(str(path.relative_to(_REPO_ROOT)))
        self.assertEqual(offenders, [])

    def test_removed_forwarding_exports_stay_removed(self) -> None:
        r"""Compatibility forwarding exports should stay deleted.

        Extended Summary
        ----------------
        Verifies the updated export contract: symbols are public from the
        owning module and subpackage, not from additional forwarding modules.

        Notes
        -----
        It imports the former forwarding modules that still exist for other
        helpers and checks that the retired public attributes are absent. It
        also asserts the removed simulator layer modules are not importable.
        """
        inout_crystal = importlib.import_module("rheedium.inout.crystal")
        rheed_types = importlib.import_module("rheedium.types.rheed_types")

        self.assertFalse(hasattr(inout_crystal, "lattice_to_cell_params"))
        self.assertFalse(hasattr(rheed_types, "DetectorGeometry"))
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rheedium.simul.layer0")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rheedium.simul.layer1")

    def test_structured_type_definitions_stay_under_types(self) -> None:
        r"""PyTree carriers and constructors should stay under types.

        Extended Summary
        ----------------
        Verifies the types-centralization rule by scanning source modules for
        non-types `eqx.Module` and `NamedTuple` definitions, plus top-level
        `create_*` constructors outside the documented non-type exceptions.

        Notes
        -----
        This is an architectural guard rather than a behavioral test: it keeps
        future feature work from reintroducing local carriers in recon, procs,
        inout, or other consuming subpackages.
        """
        offenders: list[str] = []
        source_root: Path = _REPO_ROOT / "src" / "rheedium"
        for path in source_root.rglob("*.py"):
            relative_path: str = path.relative_to(_REPO_ROOT).as_posix()
            if relative_path.startswith("src/rheedium/types/"):
                continue
            tree: ast.Module = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    base_names: set[str] = {
                        _base_name(base) for base in node.bases
                    }
                    if base_names & {
                        "NamedTuple",
                        "typing.NamedTuple",
                        "beartype.typing.NamedTuple",
                        "eqx.Module",
                        "equinox.Module",
                    }:
                        offenders.append(f"{relative_path}:{node.name}")
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name.startswith("create_")
                    and (relative_path, node.name)
                    not in _TYPE_CONSTRUCTOR_EXEMPTIONS
                ):
                    offenders.append(f"{relative_path}:{node.name}")
        self.assertEqual(offenders, [])
