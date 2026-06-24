"""Tests for import-time configuration in the top-level rheedium package.

The runtime-check toggle (``RHEEDIUM_DISABLE_RUNTIME_CHECKS``) acts before the
first equinox import, so it can only be observed in a fresh interpreter. Each
case launches a subprocess that imports rheedium and reports the resolved
``EQX_ON_ERROR`` value.
"""

import os
import subprocess
import sys

import chex

_PROBE: str = (
    "import os, rheedium; "
    "print('EQX=' + os.environ.get('EQX_ON_ERROR', '<unset>'))"
)


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


class TestRuntimeCheckToggle(chex.TestCase):
    """Tests for the ``RHEEDIUM_DISABLE_RUNTIME_CHECKS`` import-time knob."""

    def test_checks_on_by_default(self) -> None:
        """Without the opt-in, rheedium leaves ``EQX_ON_ERROR`` unset."""
        assert _resolved_eqx_on_error() == "<unset>"

    def test_opt_in_disables_checks(self) -> None:
        """The opt-in sets ``EQX_ON_ERROR=off`` before the equinox import."""
        assert (
            _resolved_eqx_on_error(RHEEDIUM_DISABLE_RUNTIME_CHECKS="1")
            == "off"
        )

    def test_explicit_value_is_respected(self) -> None:
        """An explicit ``EQX_ON_ERROR`` wins over the opt-in default."""
        assert (
            _resolved_eqx_on_error(
                RHEEDIUM_DISABLE_RUNTIME_CHECKS="1",
                EQX_ON_ERROR="nan",
            )
            == "nan"
        )
