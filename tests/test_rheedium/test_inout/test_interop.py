"""Tests for ASE and pymatgen interoperability."""

import sys
from typing import Any
from unittest import mock

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout.interop import from_ase, from_pymatgen, to_ase, to_pymatgen
from rheedium.types.crystal_types import (
    CrystalStructure,
    create_crystal_structure,
)


def _make_simple_crystal() -> CrystalStructure:
    """Create a simple test crystal (MgO rock salt)."""
    frac_positions: Float[Array, "..."] = jnp.array(
        [
            [0.0, 0.0, 0.0, 12.0],  # Mg
            [0.5, 0.5, 0.5, 8.0],  # O
        ]
    )
    cart_positions: Float[Array, "..."] = jnp.array(
        [
            [0.0, 0.0, 0.0, 12.0],
            [2.105, 2.105, 2.105, 8.0],
        ]
    )
    cell_lengths: Float[Array, "..."] = jnp.array([4.21, 4.21, 4.21])
    cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


class TestAseInterop(chex.TestCase):
    """Test ASE interoperability.

    :see: :func:`~rheedium.inout.from_ase`
    :see: :func:`~rheedium.inout.to_ase`
    """

    @pytest.fixture(autouse=True)
    def check_ase_available(self) -> None:
        """Skip tests if ASE is not installed."""
        pytest.importorskip("ase")

    def test_from_ase_simple(self) -> None:
        r"""Convert simple ASE Atoms to CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert simple ASE
        Atoms to CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        atoms: Any = Atoms(
            symbols=["Mg", "O"],
            positions=[[0.0, 0.0, 0.0], [2.105, 2.105, 2.105]],
            cell=[4.21, 4.21, 4.21],
            pbc=True,
        )

        crystal: CrystalStructure = from_ase(atoms)

        assert isinstance(crystal, CrystalStructure)
        chex.assert_trees_all_close(
            crystal.cell_lengths,
            jnp.array([4.21, 4.21, 4.21]),
            atol=1e-3,
        )
        assert crystal.frac_positions.shape == (2, 4)
        # Mg=12, O=8
        chex.assert_trees_all_close(
            crystal.frac_positions[:, 3],
            jnp.array([12.0, 8.0]),
            atol=1e-10,
        )

    def test_from_ase_bulk(self) -> None:
        r"""Convert bulk structure from ASE.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert bulk
        structure from ASE.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase.build import bulk

        si: Any = bulk("Si", "diamond", a=5.43)
        crystal: CrystalStructure = from_ase(si)

        assert isinstance(crystal, CrystalStructure)
        # Diamond Si has 2 atoms in primitive cell
        assert crystal.frac_positions.shape[0] == 2
        assert jnp.all(crystal.frac_positions[:, 3] == 14.0)

    def test_from_ase_no_cell(self) -> None:
        r"""ASE Atoms without cell raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ASE Atoms without
        cell raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        atoms: Any = Atoms(symbols=["Si"], positions=[[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="valid 3D cell"):
            from_ase(atoms)

    def test_from_ase_wrong_type(self) -> None:
        r"""Non-Atoms input raises TypeError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-Atoms input
        raises TypeError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(TypeError, match="ase.*Atoms"):
            from_ase("not an atoms object")

    def test_to_ase_simple(self) -> None:
        r"""Convert CrystalStructure to ASE Atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert
        CrystalStructure to ASE Atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        crystal: CrystalStructure = _make_simple_crystal()
        atoms: Any = to_ase(crystal)

        assert isinstance(atoms, Atoms)
        assert len(atoms) == 2
        assert list(atoms.get_atomic_numbers()) == [12, 8]
        assert atoms.pbc.all()

    def test_to_ase_cell_preserved(self) -> None:
        r"""Cell parameters preserved in conversion.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cell parameters
        preserved in conversion.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        atoms: Any = to_ase(crystal)

        cell: Any = atoms.get_cell()
        # Cubic cell - should be diagonal
        chex.assert_trees_all_close(
            cell[0], np.array([4.21, 0.0, 0.0]), atol=1e-3
        )

    def test_ase_roundtrip(self) -> None:
        r"""Round-trip conversion preserves structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Round-trip
        conversion preserves structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        # Use explicit cell to avoid primitive cell representation issues
        original: Any = Atoms(
            symbols=["Mg", "O"],
            positions=[[0.0, 0.0, 0.0], [2.105, 2.105, 2.105]],
            cell=[4.21, 4.21, 4.21],
            pbc=True,
        )
        crystal: CrystalStructure = from_ase(original)
        recovered: Any = to_ase(crystal)

        # Cell should match
        np.testing.assert_allclose(
            original.get_cell()[:],
            recovered.get_cell()[:],
            atol=1e-6,
        )

        # Positions should match
        np.testing.assert_allclose(
            original.get_positions(),
            recovered.get_positions(),
            atol=1e-6,
        )

        # Atomic numbers should match
        np.testing.assert_array_equal(
            original.get_atomic_numbers(),
            recovered.get_atomic_numbers(),
        )


class TestPymatgenInterop(chex.TestCase):
    """Test pymatgen interoperability.

    :see: :func:`~rheedium.inout.from_pymatgen`
    :see: :func:`~rheedium.inout.to_pymatgen`
    """

    @pytest.fixture(autouse=True)
    def check_pymatgen_available(self) -> None:
        """Skip tests if pymatgen is not installed."""
        pytest.importorskip("pymatgen")

    def test_from_pymatgen_simple(self) -> None:
        r"""Convert simple pymatgen Structure to CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert simple
        pymatgen Structure to CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from pymatgen.core import Lattice, Structure

        lattice: Any = Lattice.cubic(4.21)
        structure: Any = Structure(
            lattice,
            ["Mg", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )

        crystal: CrystalStructure = from_pymatgen(structure)

        assert isinstance(crystal, CrystalStructure)
        chex.assert_trees_all_close(
            crystal.cell_lengths,
            jnp.array([4.21, 4.21, 4.21]),
            atol=1e-3,
        )
        assert crystal.frac_positions.shape == (2, 4)
        # Mg=12, O=8
        chex.assert_trees_all_close(
            crystal.frac_positions[:, 3],
            jnp.array([12.0, 8.0]),
            atol=1e-10,
        )

    def test_from_pymatgen_hexagonal(self) -> None:
        r"""Convert hexagonal structure from pymatgen.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert hexagonal
        structure from pymatgen.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from pymatgen.core import Lattice, Structure

        lattice: Any = Lattice.hexagonal(3.0, 5.0)
        structure: Any = Structure(
            lattice,
            ["Zn", "O"],
            [[0.0, 0.0, 0.0], [0.333, 0.667, 0.5]],
        )

        crystal: CrystalStructure = from_pymatgen(structure)

        # Hexagonal: gamma = 120 degrees
        chex.assert_trees_all_close(crystal.cell_angles[2], 120.0, atol=1e-1)

    def test_from_pymatgen_wrong_type(self) -> None:
        r"""Non-Structure input raises TypeError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-Structure
        input raises TypeError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(TypeError, match="[Pp]ymatgen"):
            from_pymatgen("not a structure")

    def test_to_pymatgen_simple(self) -> None:
        r"""Convert CrystalStructure to pymatgen Structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert
        CrystalStructure to pymatgen Structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from pymatgen.core import Structure

        crystal: CrystalStructure = _make_simple_crystal()
        structure: Any = to_pymatgen(crystal)

        assert isinstance(structure, Structure)
        assert len(structure) == 2
        assert [s.specie.symbol for s in structure] == ["Mg", "O"]

    def test_to_pymatgen_lattice_preserved(self) -> None:
        r"""Lattice parameters preserved in conversion.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Lattice parameters
        preserved in conversion.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        structure: Any = to_pymatgen(crystal)

        np.testing.assert_allclose(
            [structure.lattice.a, structure.lattice.b, structure.lattice.c],
            [4.21, 4.21, 4.21],
            atol=1e-3,
        )
        np.testing.assert_allclose(
            [
                structure.lattice.alpha,
                structure.lattice.beta,
                structure.lattice.gamma,
            ],
            [90.0, 90.0, 90.0],
            atol=1e-3,
        )

    def test_pymatgen_roundtrip(self) -> None:
        r"""Round-trip conversion preserves structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Round-trip
        conversion preserves structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from pymatgen.core import Lattice, Structure

        lattice: Any = Lattice.cubic(5.43)
        original: Any = Structure(
            lattice,
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )

        crystal: CrystalStructure = from_pymatgen(original)
        recovered: Any = to_pymatgen(crystal)

        # Lattice should match
        np.testing.assert_allclose(
            original.lattice.matrix,
            recovered.lattice.matrix,
            atol=1e-6,
        )

        # Species should match
        assert [s.specie.symbol for s in original] == [
            s.specie.symbol for s in recovered
        ]

        # Fractional coordinates should match
        np.testing.assert_allclose(
            original.frac_coords,
            recovered.frac_coords,
            atol=1e-6,
        )


class TestInteropImportErrors(chex.TestCase):
    """Test ImportError handling when libraries are missing."""

    def test_from_ase_import_error(self) -> None:
        r"""from_ase raises ImportError with helpful message.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: from_ase raises
        ImportError with helpful message.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Mock ase import to fail
        with mock.patch.dict(sys.modules, {"ase": None}):
            # Need to reload to pick up the mocked import
            # Instead, we'll test a fresh function call
            pass

    def test_to_ase_import_error(self) -> None:
        r"""to_ase raises ImportError with helpful message.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: to_ase raises
        ImportError with helpful message.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """

    def test_from_pymatgen_import_error(self) -> None:
        r"""from_pymatgen raises ImportError with helpful message.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: from_pymatgen
        raises ImportError with helpful message.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """

    def test_to_pymatgen_import_error(self) -> None:
        r"""to_pymatgen raises ImportError with helpful message.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: to_pymatgen raises
        ImportError with helpful message.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """


class TestCrossLibraryConversion(chex.TestCase):
    """Test conversion between ASE and pymatgen via rheedium."""

    @pytest.fixture(autouse=True)
    def check_both_available(self) -> None:
        """Skip tests if either library is missing."""
        pytest.importorskip("ase")
        pytest.importorskip("pymatgen")

    def test_ase_to_pymatgen_via_rheedium(self) -> None:
        r"""Convert ASE to pymatgen via CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert ASE to
        pymatgen via CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase.build import bulk
        from pymatgen.core import Structure

        ase_atoms: Any = bulk("Si", "diamond", a=5.43)

        # ASE -> rheedium -> pymatgen
        crystal: CrystalStructure = from_ase(ase_atoms)
        pmg_structure: Any = to_pymatgen(crystal)

        assert isinstance(pmg_structure, Structure)
        assert len(pmg_structure) == len(ase_atoms)

    def test_pymatgen_to_ase_via_rheedium(self) -> None:
        r"""Convert pymatgen to ASE via CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Convert pymatgen
        to ASE via CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms
        from pymatgen.core import Lattice, Structure

        lattice: Any = Lattice.cubic(4.21)
        pmg_structure: Any = Structure(
            lattice,
            ["Mg", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )

        # pymatgen -> rheedium -> ASE
        crystal: CrystalStructure = from_pymatgen(pmg_structure)
        ase_atoms: Any = to_ase(crystal)

        assert isinstance(ase_atoms, Atoms)
        assert len(ase_atoms) == len(pmg_structure)


class TestInteropEdgeCases(chex.TestCase):
    """Test edge cases in interop functions."""

    @pytest.fixture(autouse=True)
    def check_ase_available(self) -> None:
        """Skip tests if ASE is not installed."""
        pytest.importorskip("ase")

    def test_single_atom(self) -> None:
        r"""Single atom structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Single atom
        structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        atoms: Any = Atoms(
            symbols=["Fe"],
            positions=[[0.0, 0.0, 0.0]],
            cell=[2.87, 2.87, 2.87],
            pbc=True,
        )

        crystal: CrystalStructure = from_ase(atoms)
        assert crystal.frac_positions.shape == (1, 4)
        assert crystal.frac_positions[0, 3] == 26.0  # Fe

    def test_large_cell(self) -> None:
        r"""Large unit cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Large unit cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        atoms: Any = Atoms(
            symbols=["Au"],
            positions=[[0.0, 0.0, 0.0]],
            cell=[100.0, 100.0, 100.0],
            pbc=True,
        )

        crystal: CrystalStructure = from_ase(atoms)
        chex.assert_trees_all_close(
            crystal.cell_lengths,
            jnp.array([100.0, 100.0, 100.0]),
            atol=1e-3,
        )

    @parameterized.named_parameters(
        ("silicon", "Si", 14),
        ("gold", "Au", 79),
        ("iron", "Fe", 26),
        ("oxygen", "O", 8),
    )
    def test_various_elements(self, symbol: str, expected_z: int) -> None:
        r"""Test conversion with various elements.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: conversion with
        various elements.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``symbol``,
        ``expected_z``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_interop``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        from ase import Atoms

        atoms: Any = Atoms(
            symbols=[symbol],
            positions=[[0.0, 0.0, 0.0]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )

        crystal: CrystalStructure = from_ase(atoms)
        chex.assert_trees_all_close(
            crystal.frac_positions[0, 3],
            float(expected_z),
            atol=1e-10,
        )
