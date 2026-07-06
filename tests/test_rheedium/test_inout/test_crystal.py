"""Tests for crystal structure parsing and conversion utilities."""

import tempfile
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Integer

from rheedium.inout.crystal import (
    _infer_lattice_from_positions,
    parse_crystal,
    xyz_to_crystal,
)
from rheedium.inout.lattice import lattice_to_cell_params
from rheedium.types.crystal_types import CrystalStructure, create_xyz_data


class TestInferLatticeFromPositions(chex.TestCase):
    """Test lattice inference from atomic positions.

    :see: :func:`~rheedium.inout.crystal._infer_lattice_from_positions`
    """

    def test_single_atom(self) -> None:
        r"""Single atom should create minimum extent cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Single atom should
        create minimum extent cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        lattice: Any = _infer_lattice_from_positions(
            positions, padding_ang=2.0
        )

        # Single atom: extent = 0, but minimum is 1.0
        # Final = max(0 + 2*2, 1.0) = 4.0
        expected: Float[Array, "..."] = jnp.diag(jnp.array([4.0, 4.0, 4.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_atoms_with_extent(self) -> None:
        r"""Atoms spanning 0-10 A in each direction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Atoms spanning
        0-10 A in each direction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        lattice: Any = _infer_lattice_from_positions(
            positions, padding_ang=2.0
        )

        # Extent = 10, + 2*2 = 14 in each direction
        expected: Float[Array, "..."] = jnp.diag(jnp.array([14.0, 14.0, 14.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_asymmetric_extent(self) -> None:
        r"""Atoms with different extents in each direction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Atoms with
        different extents in each direction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 10.0, 15.0],
            ]
        )
        lattice: Any = _infer_lattice_from_positions(
            positions, padding_ang=1.0
        )

        # Extents: x=5, y=10, z=15, + 2*1 = 7, 12, 17
        expected: Float[Array, "..."] = jnp.diag(jnp.array([7.0, 12.0, 17.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_minimum_extent_enforced(self) -> None:
        r"""Minimum extent of 1 A should be enforced.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Minimum extent of
        1 A should be enforced.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[5.0, 5.0, 5.0]]
        )  # Single atom
        lattice: Any = _infer_lattice_from_positions(
            positions, padding_ang=0.0
        )

        # No padding, extent = 0, minimum = 1.0
        expected: Float[Array, "..."] = jnp.diag(jnp.array([1.0, 1.0, 1.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)


class TestXyzToCrystal(chex.TestCase):
    """Test XYZ to CrystalStructure conversion.

    :see: :func:`~rheedium.inout.xyz_to_crystal`
    """

    def test_with_explicit_lattice(self) -> None:
        r"""XYZData with explicit cell_vectors override.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XYZData with
        explicit cell_vectors override.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [2.1, 2.1, 2.1]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array([12, 8])  # Mg, O
        lattice: Float[Array, "..."] = jnp.eye(3) * 4.2

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        # Check structure
        assert crystal.cart_positions.shape == (2, 4)
        assert crystal.frac_positions.shape == (2, 4)
        chex.assert_trees_all_close(
            crystal.cell_lengths, jnp.array([4.2, 4.2, 4.2]), atol=1e-6
        )
        chex.assert_trees_all_close(
            crystal.cell_angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-6
        )

        # Check atomic numbers preserved
        chex.assert_trees_all_close(
            crystal.cart_positions[:, 3], jnp.array([12.0, 8.0]), atol=1e-10
        )

        # Check fractional coordinates
        expected_frac: Float[Array, "..."] = positions / 4.2
        chex.assert_trees_all_close(
            crystal.frac_positions[:, :3], expected_frac, atol=1e-10
        )

    def test_with_cell_vectors_override(self) -> None:
        r"""Test explicit cell_vectors parameter overrides xyz_data.lattice.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: explicit
        cell_vectors parameter overrides xyz_data.lattice.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers: Integer[Array, "..."] = jnp.array([14])  # Si
        xyz_lattice: Float[Array, "..."] = jnp.eye(3) * 5.43
        override_lattice: Float[Array, "..."] = jnp.eye(3) * 10.0

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=xyz_lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(
            xyz_data, cell_vectors=override_lattice
        )

        # Should use override, not xyz_data.lattice
        chex.assert_trees_all_close(
            crystal.cell_lengths, jnp.array([10.0, 10.0, 10.0]), atol=1e-6
        )

    def test_non_orthorhombic_lattice(self) -> None:
        r"""XYZ with non-orthorhombic (hexagonal) lattice.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XYZ with
        non-orthorhombic (hexagonal) lattice.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 3.2
        c: float = 5.2
        # Hexagonal lattice vectors
        lattice: Float[Array, "..."] = jnp.array(
            [[a, 0, 0], [-a / 2, a * jnp.sqrt(3) / 2, 0], [0, 0, c]]
        )

        positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, c / 2]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array([30, 8])  # Zn, O

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        # Check hexagonal angles
        chex.assert_trees_all_close(crystal.cell_angles[2], 120.0, atol=1e-5)
        chex.assert_trees_all_close(
            crystal.cell_angles[:2], jnp.array([90.0, 90.0]), atol=1e-5
        )

    def test_preserves_atomic_numbers(self) -> None:
        r"""Atomic numbers correctly transferred to 4th column.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Atomic numbers
        correctly transferred to 4th column.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array(
            [26, 8, 14]
        )  # Fe, O, Si

        xyz_data: Float[Array, "..."] = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=jnp.eye(3) * 5.0,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        # Both frac and cart should have same atomic numbers
        expected: Float[Array, "..."] = jnp.array([26.0, 8.0, 14.0])
        chex.assert_trees_all_close(
            crystal.frac_positions[:, 3], expected, atol=1e-10
        )
        chex.assert_trees_all_close(
            crystal.cart_positions[:, 3], expected, atol=1e-10
        )

    def test_cartesian_positions_preserved(self) -> None:
        r"""Cartesian positions should be preserved exactly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cartesian
        positions should be preserved exactly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array([6, 7])  # C, N
        lattice: Float[Array, "..."] = jnp.eye(3) * 10.0

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        chex.assert_trees_all_close(
            crystal.cart_positions[:, :3], positions, atol=1e-10
        )

    def test_returns_crystal_structure(self) -> None:
        r"""Should return CrystalStructure instance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should return
        CrystalStructure instance.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers: Integer[Array, "..."] = jnp.array([1])
        lattice: Float[Array, "..."] = jnp.eye(3) * 5.0

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        assert isinstance(crystal, CrystalStructure)

    def test_unit_cubic_lattice_preserved(self) -> None:
        r"""A genuine 1 Angstrom cubic lattice is not replaced.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: an explicit
        identity lattice (1 Angstrom cubic cell) in ``xyz_data.lattice``
        survives ingestion unchanged. It must not be mistaken for a
        missing-lattice sentinel and silently replaced by a bounding-box
        cell inferred from the atomic positions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array([[0.25, 0.25, 0.25]])
        atomic_numbers: Integer[Array, "..."] = jnp.array([6])
        lattice: Float[Array, "..."] = jnp.eye(3)

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        chex.assert_trees_all_close(
            crystal.cell_lengths,
            jnp.array([1.0, 1.0, 1.0]),
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            crystal.frac_positions[:, :3],
            jnp.array([[0.25, 0.25, 0.25]]),
            atol=1e-10,
        )

    def test_missing_lattice_uses_bounding_box(self) -> None:
        r"""A lattice-free XYZData falls back to the bounding-box cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: when
        ``xyz_data.lattice`` is None and no ``cell_vectors`` override is
        supplied, the cell is inferred from the atomic bounding box plus
        symmetric padding (extent 2 + 2*2 padding = 6 Angstroms per axis).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array([6, 6])

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )

        assert xyz_data.lattice is None

        crystal: CrystalStructure = xyz_to_crystal(xyz_data, padding_ang=2.0)

        chex.assert_trees_all_close(
            crystal.cell_lengths,
            jnp.array([6.0, 6.0, 6.0]),
            atol=1e-10,
        )


class TestParseCrystal(chex.TestCase):
    """Test unified crystal parser.

    :see: :func:`~rheedium.inout.parse_crystal`
    """

    def test_parse_cif(self) -> None:
        r"""Parse CIF file via parse_crystal.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse CIF file via
        parse_crystal.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
data_MgO
_cell_length_a 4.212
_cell_length_b 4.212
_cell_length_c 4.212
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mg 0.0 0.0 0.0
O  0.5 0.5 0.5
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal: CrystalStructure = parse_crystal(cif_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([4.212, 4.212, 4.212]),
                atol=1e-3,
            )

    def test_parse_xyz(self) -> None:
        r"""Parse XYZ file via parse_crystal.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse XYZ file via
        parse_crystal.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xyz_content: str = """2
Lattice="4.2 0.0 0.0 0.0 4.2 0.0 0.0 0.0 4.2"
Mg 0.0 0.0 0.0
O  2.1 2.1 2.1
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file: Path = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            crystal: CrystalStructure = parse_crystal(xyz_file)

            assert isinstance(crystal, CrystalStructure)
            assert crystal.cart_positions.shape[0] == 2
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([4.2, 4.2, 4.2]),
                atol=1e-6,
            )

    def test_unsupported_format(self) -> None:
        r"""Raise ValueError for unsupported format.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Raise ValueError
        for unsupported format.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file: Path = Path(tmp_dir) / "test.pdb"
            bad_file.write_text("dummy content")

            with pytest.raises(ValueError, match="Unsupported file format"):
                parse_crystal(bad_file)

    def test_file_not_found(self) -> None:
        r"""Raise FileNotFoundError for missing file.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Raise
        FileNotFoundError for missing file.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(FileNotFoundError):
            parse_crystal("/nonexistent/path.cif")

    def test_cif_xyz_equivalence(self) -> None:
        r"""Same structure from CIF and XYZ should give similar results.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Same structure
        from CIF and XYZ should give similar results.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Simple cubic MgO
        cif_content: str = """
data_MgO
_cell_length_a 4.212
_cell_length_b 4.212
_cell_length_c 4.212
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mg 0.0 0.0 0.0
O  0.5 0.5 0.5
"""

        xyz_content: str = """2
Lattice="4.212 0.0 0.0 0.0 4.212 0.0 0.0 0.0 4.212"
Mg 0.0 0.0 0.0
O  2.106 2.106 2.106
"""

        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "mgo.cif"
            xyz_file: Path = Path(tmp_dir) / "mgo.xyz"
            cif_file.write_text(cif_content)
            xyz_file.write_text(xyz_content)

            crystal_cif: Any = parse_crystal(cif_file)
            crystal_xyz: Any = parse_crystal(xyz_file)

            # Cell parameters should match
            chex.assert_trees_all_close(
                crystal_cif.cell_lengths, crystal_xyz.cell_lengths, atol=1e-3
            )
            chex.assert_trees_all_close(
                crystal_cif.cell_angles, crystal_xyz.cell_angles, atol=1e-3
            )

            # Same number of atoms
            assert (
                crystal_cif.cart_positions.shape[0]
                == crystal_xyz.cart_positions.shape[0]
            )

    def test_path_as_string(self) -> None:
        r"""Test that string paths work.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: string paths work.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.0 0.0 0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            # Pass as string, not Path
            crystal: CrystalStructure = parse_crystal(str(cif_file))

            assert isinstance(crystal, CrystalStructure)


class TestCrystalRoundtrip(chex.TestCase):
    """Test consistency between lattice and cell parameter conversions."""

    def test_orthorhombic_roundtrip(self) -> None:
        r"""Orthorhombic lattice survives roundtrip conversion.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orthorhombic
        lattice survives roundtrip conversion.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: tuple[Any, ...]
        b: tuple[Any, ...]
        c: tuple[Any, ...]
        a, b, c = 4.0, 5.0, 6.0
        lattice: Float[Array, "..."] = jnp.array(
            [[a, 0, 0], [0, b, 0], [0, 0, c]]
        )

        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        # Lengths should match
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), atol=1e-10)

        # Angles should be 90 deg
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_xyz_to_crystal_fractional_consistency(self) -> None:
        r"""Fractional coords times lattice should give Cartesian coords.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fractional coords
        times lattice should give Cartesian coords.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_crystal``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        atomic_numbers: Integer[Array, "..."] = jnp.array([6, 7])
        lattice: Float[Array, "..."] = jnp.eye(3) * 10.0

        xyz_data: Any = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal: CrystalStructure = xyz_to_crystal(xyz_data)

        # frac @ lattice should give cart
        reconstructed_cart: Any = crystal.frac_positions[:, :3] @ lattice
        chex.assert_trees_all_close(
            reconstructed_cart, crystal.cart_positions[:, :3], atol=1e-10
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
