"""Test suite for recon/library.py.

Tests the pre-parameterized surface reconstruction library functions.
Each library function should return a valid CrystalStructure with
atoms, correct cell parameters, and expected atomic species.
"""

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from rheedium.recon import (
    gaas001_2x4,
    mgo001_bulk_terminated,
    si100_2x1,
    si111_1x1,
    si111_7x7,
    srtio3_001_2x1,
)
from rheedium.types import CrystalStructure


class TestSi111_1x1(chex.TestCase):
    """Tests for si111_1x1 library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = si111_1x1()
        assert isinstance(slab, CrystalStructure)

    def test_has_silicon_atoms(self) -> None:
        """Slab should contain Si atoms (Z=14)."""
        slab: CrystalStructure = si111_1x1()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        assert np.any(np.abs(z_nums - 14.0) < 0.5)

    def test_nonzero_atom_count(self) -> None:
        """Slab should have at least one atom."""
        slab: CrystalStructure = si111_1x1()
        assert slab.cart_positions.shape[0] > 0

    def test_custom_lattice_parameter(self) -> None:
        """Custom lattice parameter should change cell dimensions."""
        slab_default: CrystalStructure = si111_1x1()
        slab_custom: CrystalStructure = si111_1x1(a_lattice_angstrom=6.0)
        a_default: float = float(slab_default.cell_lengths[0])
        a_custom: float = float(slab_custom.cell_lengths[0])
        assert a_custom != a_default

    def test_cell_c_includes_vacuum(self) -> None:
        """Cell c should be slab_depth + vacuum_gap."""
        slab: CrystalStructure = si111_1x1(
            slab_depth_angstrom=20.0,
            vacuum_gap_angstrom=15.0,
        )
        c: float = float(slab.cell_lengths[2])
        chex.assert_trees_all_close(c, 35.0, atol=1e-6)


class TestSi111_7x7(chex.TestCase):
    """Tests for si111_7x7 library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = si111_7x7()
        assert isinstance(slab, CrystalStructure)

    def test_more_atoms_than_1x1(self) -> None:
        """7x7 slab should have more atoms than 1x1 (adatoms)."""
        slab_1x1: CrystalStructure = si111_1x1()
        slab_7x7: CrystalStructure = si111_7x7()
        n_1x1: int = slab_1x1.cart_positions.shape[0]
        n_7x7: int = slab_7x7.cart_positions.shape[0]
        assert n_7x7 > n_1x1

    def test_has_twelve_extra_atoms(self) -> None:
        """7x7 should have 12 more atoms than 1x1 (adatoms)."""
        slab_1x1: CrystalStructure = si111_1x1()
        slab_7x7: CrystalStructure = si111_7x7()
        n_1x1: int = slab_1x1.cart_positions.shape[0]
        n_7x7: int = slab_7x7.cart_positions.shape[0]
        assert n_7x7 - n_1x1 == 12


class TestSi100_2x1(chex.TestCase):
    """Tests for si100_2x1 library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = si100_2x1()
        assert isinstance(slab, CrystalStructure)

    def test_has_silicon_atoms(self) -> None:
        """Slab should contain Si atoms."""
        slab: CrystalStructure = si100_2x1()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        assert np.any(np.abs(z_nums - 14.0) < 0.5)

    def test_has_dimer_atoms(self) -> None:
        """2x1 slab should have more atoms than bare (001) slab."""
        from rheedium.recon.library import (
            _SI_DIAMOND_FRAC,
            _SI_DIAMOND_Z,
            _build_bulk_crystal,
        )
        from rheedium.recon.surface_builder import create_surface_slab

        bulk: CrystalStructure = _build_bulk_crystal(
            frac_coords=_SI_DIAMOND_FRAC,
            atomic_numbers=_SI_DIAMOND_Z,
            a=5.431,
            b=5.431,
            c=5.431,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        bare: CrystalStructure = create_surface_slab(
            bulk_crystal=bulk,
            surface_normal_miller=jnp.array([1, 0, 0], dtype=jnp.int32),
            slab_thickness_angstrom=20.0,
            vacuum_gap_angstrom=15.0,
        )
        recon: CrystalStructure = si100_2x1()
        assert recon.cart_positions.shape[0] > (bare.cart_positions.shape[0])


class TestGaAs001_2x4(chex.TestCase):
    """Tests for gaas001_2x4 library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = gaas001_2x4()
        assert isinstance(slab, CrystalStructure)

    def test_has_ga_and_as(self) -> None:
        """Slab should contain both Ga (31) and As (33)."""
        slab: CrystalStructure = gaas001_2x4()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        has_ga: bool = np.any(np.abs(z_nums - 31.0) < 0.5)
        has_as: bool = np.any(np.abs(z_nums - 33.0) < 0.5)
        assert has_ga
        assert has_as

    def test_lattice_parameter(self) -> None:
        """Default lattice parameter should be ~5.653 A."""
        slab: CrystalStructure = gaas001_2x4()
        a: float = float(slab.cell_lengths[0])
        assert 4.0 < a < 8.0


class TestMgO001BulkTerminated(chex.TestCase):
    """Tests for mgo001_bulk_terminated library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = mgo001_bulk_terminated()
        assert isinstance(slab, CrystalStructure)

    def test_has_mg_and_o(self) -> None:
        """Slab should contain both Mg (12) and O (8)."""
        slab: CrystalStructure = mgo001_bulk_terminated()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        has_mg: bool = np.any(np.abs(z_nums - 12.0) < 0.5)
        has_o: bool = np.any(np.abs(z_nums - 8.0) < 0.5)
        assert has_mg
        assert has_o

    def test_stoichiometry_mg_to_o(self) -> None:
        """Mg:O ratio should be close to 1:1."""
        slab: CrystalStructure = mgo001_bulk_terminated()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        n_mg: int = int(np.sum(np.abs(z_nums - 12.0) < 0.5))
        n_o: int = int(np.sum(np.abs(z_nums - 8.0) < 0.5))
        ratio: float = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_c_includes_vacuum(self) -> None:
        """Cell c should be slab_depth + vacuum_gap."""
        slab: CrystalStructure = mgo001_bulk_terminated(
            slab_depth_angstrom=25.0,
            vacuum_gap_angstrom=15.0,
        )
        c: float = float(slab.cell_lengths[2])
        chex.assert_trees_all_close(c, 40.0, atol=1e-6)


class TestSrTiO3_001_2x1(chex.TestCase):
    """Tests for srtio3_001_2x1 library function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = srtio3_001_2x1()
        assert isinstance(slab, CrystalStructure)

    def test_has_sr_ti_o(self) -> None:
        """Slab should contain Sr (38), Ti (22), and O (8)."""
        slab: CrystalStructure = srtio3_001_2x1()
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        has_sr: bool = np.any(np.abs(z_nums - 38.0) < 0.5)
        has_ti: bool = np.any(np.abs(z_nums - 22.0) < 0.5)
        has_o: bool = np.any(np.abs(z_nums - 8.0) < 0.5)
        assert has_sr
        assert has_ti
        assert has_o

    def test_nonzero_atom_count(self) -> None:
        """Slab should have atoms."""
        slab: CrystalStructure = srtio3_001_2x1()
        assert slab.cart_positions.shape[0] > 0
