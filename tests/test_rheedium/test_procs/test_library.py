"""Test suite for recon/library.py.

Tests the pre-parameterized surface reconstruction library functions.
All slabs are loaded from pre-built .npz fixtures — no JIT at test
time.
"""

from pathlib import Path
from typing import Any

import chex
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name: str) -> dict[str, Float[NDArray, "..."]]:
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


class TestSi111_1x1(chex.TestCase):  # noqa: N801
    """Tests for si111_1x1 library function.

    :see: :func:`~rheedium.procs.si111_1x1`
    """

    def test_has_silicon_atoms(self) -> None:
        r"""Slab should contain Si atoms (Z=14).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain Si atoms (Z=14).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("si111_1x1.npz")
        assert np.any(np.abs(d["cart_positions"][:, 3] - 14.0) < 0.5)

    def test_nonzero_atom_count(self) -> None:
        r"""Slab should have at least one atom.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should have
        at least one atom.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("si111_1x1.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_custom_lattice_parameter(self) -> None:
        r"""Custom lattice parameter should change cell dimensions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Custom lattice
        parameter should change cell dimensions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        default: Any = _load("si111_1x1.npz")
        custom: Any = _load("si111_1x1_custom.npz")
        assert float(custom["cell_lengths"][0]) != float(
            default["cell_lengths"][0]
        )

    def test_cell_c_includes_vacuum(self) -> None:
        r"""Cell c should be slab_depth + vacuum_gap = 35.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cell c should be
        slab_depth + vacuum_gap = 35.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("si111_1x1.npz")
        chex.assert_trees_all_close(
            float(d["cell_lengths"][2]), 35.0, atol=1e-6
        )


class TestSi111_7x7(chex.TestCase):  # noqa: N801
    """Tests for si111_7x7 library function.

    :see: :func:`~rheedium.procs.si111_7x7`
    """

    def test_more_atoms_than_1x1(self) -> None:
        r"""7x7 slab should have more atoms than 1x1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 7x7 slab should
        have more atoms than 1x1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d_1x1: Any = _load("si111_1x1.npz")
        d_7x7: Any = _load("si111_7x7.npz")
        assert (
            d_7x7["cart_positions"].shape[0]
            > (d_1x1["cart_positions"].shape[0])
        )

    def test_has_twelve_extra_atoms(self) -> None:
        r"""7x7 should have 12 more atoms than 1x1 (adatoms).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 7x7 should have 12
        more atoms than 1x1 (adatoms).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d_1x1: Any = _load("si111_1x1.npz")
        d_7x7: Any = _load("si111_7x7.npz")
        diff: Any = (
            d_7x7["cart_positions"].shape[0] - d_1x1["cart_positions"].shape[0]
        )
        assert diff == 12


class TestSi100_2x1(chex.TestCase):  # noqa: N801
    """Tests for si100_2x1 library function.

    :see: :func:`~rheedium.procs.si100_2x1`
    """

    def test_has_atoms(self) -> None:
        r"""Slab should have atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should have
        atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("si100_2x1.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_has_silicon_atoms(self) -> None:
        r"""Slab should contain Si atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain Si atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("si100_2x1.npz")
        assert np.any(np.abs(d["cart_positions"][:, 3] - 14.0) < 0.5)


class TestGaAs001_2x4(chex.TestCase):  # noqa: N801
    """Tests for gaas001_2x4 library function.

    :see: :func:`~rheedium.procs.gaas001_2x4`
    """

    def test_has_ga_and_as(self) -> None:
        r"""Slab should contain both Ga (31) and As (33).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain both Ga (31) and As (33).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("gaas001_2x4.npz")
        z: Any = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 31.0) < 0.5)
        assert np.any(np.abs(z - 33.0) < 0.5)

    def test_lattice_parameter(self) -> None:
        r"""Default lattice parameter should be ~5.653 A.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Default lattice
        parameter should be ~5.653 A.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("gaas001_2x4.npz")
        a: float = float(d["cell_lengths"][0])
        assert 4.0 < a < 8.0


class TestMgO001BulkTerminated(chex.TestCase):
    """Tests for mgo001_bulk_terminated library function.

    :see: :func:`~rheedium.procs.mgo001_bulk_terminated`
    """

    def test_has_mg_and_o(self) -> None:
        r"""Slab should contain both Mg (12) and O (8).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain both Mg (12) and O (8).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("mgo001.npz")
        z: Any = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 12.0) < 0.5)
        assert np.any(np.abs(z - 8.0) < 0.5)

    def test_stoichiometry_mg_to_o(self) -> None:
        r"""Mg:O ratio should be close to 1:1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mg:O ratio should
        be close to 1:1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("mgo001.npz")
        z: Any = d["cart_positions"][:, 3]
        n_mg: int = int(np.sum(np.abs(z - 12.0) < 0.5))
        n_o: int = int(np.sum(np.abs(z - 8.0) < 0.5))
        ratio: Any = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_c_includes_vacuum(self) -> None:
        r"""Cell c should be 25 + 15 = 40.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cell c should be
        25 + 15 = 40.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("mgo001.npz")
        chex.assert_trees_all_close(
            float(d["cell_lengths"][2]), 40.0, atol=1e-6
        )


class TestSrTiO3_001_2x1(chex.TestCase):  # noqa: N801
    """Tests for srtio3_001_2x1 library function.

    :see: :func:`~rheedium.procs.srtio3_001_2x1`
    """

    def test_has_sr_ti_o(self) -> None:
        r"""Slab should contain Sr (38), Ti (22), and O (8).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain Sr (38), Ti (22), and O (8).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("srtio3_001.npz")
        z: Any = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 38.0) < 0.5)
        assert np.any(np.abs(z - 22.0) < 0.5)
        assert np.any(np.abs(z - 8.0) < 0.5)

    def test_nonzero_atom_count(self) -> None:
        r"""Slab should have atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should have
        atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_library``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        d: Any = _load("srtio3_001.npz")
        assert d["cart_positions"].shape[0] > 0
