"""Test suite for recon/library.py.

Tests the pre-parameterized surface reconstruction library functions.
All slabs are loaded from pre-built .npz fixtures — no JIT at test
time.
"""

from pathlib import Path

import chex
import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name):
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


class TestSi111_1x1(chex.TestCase):
    """Tests for si111_1x1 library function."""

    def test_has_silicon_atoms(self):
        """Slab should contain Si atoms (Z=14)."""
        d = _load("si111_1x1.npz")
        assert np.any(np.abs(d["cart_positions"][:, 3] - 14.0) < 0.5)

    def test_nonzero_atom_count(self):
        """Slab should have at least one atom."""
        d = _load("si111_1x1.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_custom_lattice_parameter(self):
        """Custom lattice parameter should change cell dimensions."""
        default = _load("si111_1x1.npz")
        custom = _load("si111_1x1_custom.npz")
        assert float(custom["cell_lengths"][0]) != float(
            default["cell_lengths"][0]
        )

    def test_cell_c_includes_vacuum(self):
        """Cell c should be slab_depth + vacuum_gap = 35."""
        d = _load("si111_1x1.npz")
        chex.assert_trees_all_close(
            float(d["cell_lengths"][2]), 35.0, atol=1e-6
        )


class TestSi111_7x7(chex.TestCase):
    """Tests for si111_7x7 library function."""

    def test_more_atoms_than_1x1(self):
        """7x7 slab should have more atoms than 1x1."""
        d_1x1 = _load("si111_1x1.npz")
        d_7x7 = _load("si111_7x7.npz")
        assert d_7x7["cart_positions"].shape[0] > (
            d_1x1["cart_positions"].shape[0]
        )

    def test_has_twelve_extra_atoms(self):
        """7x7 should have 12 more atoms than 1x1 (adatoms)."""
        d_1x1 = _load("si111_1x1.npz")
        d_7x7 = _load("si111_7x7.npz")
        diff = (
            d_7x7["cart_positions"].shape[0] - d_1x1["cart_positions"].shape[0]
        )
        assert diff == 12


class TestSi100_2x1(chex.TestCase):
    """Tests for si100_2x1 library function."""

    def test_has_atoms(self):
        """Slab should have atoms."""
        d = _load("si100_2x1.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_has_silicon_atoms(self):
        """Slab should contain Si atoms."""
        d = _load("si100_2x1.npz")
        assert np.any(np.abs(d["cart_positions"][:, 3] - 14.0) < 0.5)


class TestGaAs001_2x4(chex.TestCase):
    """Tests for gaas001_2x4 library function."""

    def test_has_ga_and_as(self):
        """Slab should contain both Ga (31) and As (33)."""
        d = _load("gaas001_2x4.npz")
        z = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 31.0) < 0.5)
        assert np.any(np.abs(z - 33.0) < 0.5)

    def test_lattice_parameter(self):
        """Default lattice parameter should be ~5.653 A."""
        d = _load("gaas001_2x4.npz")
        a = float(d["cell_lengths"][0])
        assert 4.0 < a < 8.0


class TestMgO001BulkTerminated(chex.TestCase):
    """Tests for mgo001_bulk_terminated library function."""

    def test_has_mg_and_o(self):
        """Slab should contain both Mg (12) and O (8)."""
        d = _load("mgo001.npz")
        z = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 12.0) < 0.5)
        assert np.any(np.abs(z - 8.0) < 0.5)

    def test_stoichiometry_mg_to_o(self):
        """Mg:O ratio should be close to 1:1."""
        d = _load("mgo001.npz")
        z = d["cart_positions"][:, 3]
        n_mg = int(np.sum(np.abs(z - 12.0) < 0.5))
        n_o = int(np.sum(np.abs(z - 8.0) < 0.5))
        ratio = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_c_includes_vacuum(self):
        """Cell c should be 25 + 15 = 40."""
        d = _load("mgo001.npz")
        chex.assert_trees_all_close(
            float(d["cell_lengths"][2]), 40.0, atol=1e-6
        )


class TestSrTiO3_001_2x1(chex.TestCase):
    """Tests for srtio3_001_2x1 library function."""

    def test_has_sr_ti_o(self):
        """Slab should contain Sr (38), Ti (22), and O (8)."""
        d = _load("srtio3_001.npz")
        z = d["cart_positions"][:, 3]
        assert np.any(np.abs(z - 38.0) < 0.5)
        assert np.any(np.abs(z - 22.0) < 0.5)
        assert np.any(np.abs(z - 8.0) < 0.5)

    def test_nonzero_atom_count(self):
        """Slab should have atoms."""
        d = _load("srtio3_001.npz")
        assert d["cart_positions"].shape[0] > 0
