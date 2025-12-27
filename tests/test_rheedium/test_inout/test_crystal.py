"""Tests for crystal structure parsing and conversion utilities."""

import tempfile
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout.crystal import (
    _infer_lattice_from_positions,
    lattice_to_cell_params,
    parse_crystal,
    xyz_to_crystal,
)
from rheedium.types import CrystalStructure, create_xyz_data


class TestLatticeToCellParams(chex.TestCase):
    """Test lattice vector to cell parameter conversion."""

    def test_cubic_lattice(self) -> None:
        """Cubic: a=b=c, alpha=beta=gamma=90 deg."""
        a = 4.0
        lattice = jnp.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_orthorhombic_lattice(self) -> None:
        """Orthorhombic: a != b != c, alpha=beta=gamma=90 deg."""
        lattice = jnp.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]])
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 4.0, 5.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_hexagonal_lattice(self) -> None:
        """Hexagonal: a=b != c, alpha=beta=90 deg, gamma=120 deg."""
        a = 3.0
        c = 5.0
        # a along x, b at 120 deg from a in xy plane
        lattice = jnp.array(
            [[a, 0, 0], [-a / 2, a * jnp.sqrt(3) / 2, 0], [0, 0, c]]
        )
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, c]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 120.0]), atol=1e-6
        )

    def test_triclinic_lattice(self) -> None:
        """Triclinic: all angles and lengths different."""
        # Known triclinic cell
        lattice = jnp.array(
            [[5.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.5, 6.0]]
        )
        lengths, angles = lattice_to_cell_params(lattice)

        # Verify lengths
        chex.assert_trees_all_close(lengths[0], 5.0, atol=1e-10)
        chex.assert_trees_all_close(
            lengths[1], jnp.sqrt(1 + 16), atol=1e-10
        )  # sqrt(17)
        chex.assert_trees_all_close(
            lengths[2], jnp.sqrt(0.25 + 0.25 + 36), atol=1e-10
        )  # sqrt(36.5)

        # Verify angles are not 90 deg (triclinic)
        assert not jnp.allclose(angles[0], 90.0)
        assert not jnp.allclose(angles[1], 90.0)
        assert not jnp.allclose(angles[2], 90.0)

    def test_tetragonal_lattice(self) -> None:
        """Tetragonal: a=b != c, alpha=beta=gamma=90 deg."""
        a = 4.0
        c = 6.0
        lattice = jnp.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, c]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    @parameterized.named_parameters(
        ("small_cubic", 2.0),
        ("medium_cubic", 5.43),
        ("large_cubic", 10.0),
    )
    def test_various_cell_sizes(self, a: float) -> None:
        """Test with various cell sizes."""
        lattice = jnp.eye(3) * a
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )


class TestInferLatticeFromPositions(chex.TestCase):
    """Test lattice inference from atomic positions."""

    def test_single_atom(self) -> None:
        """Single atom should create minimum extent cell."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        lattice = _infer_lattice_from_positions(positions, padding_ang=2.0)

        # Single atom: extent = 0, but minimum is 1.0
        # Final = max(0 + 2*2, 1.0) = 4.0
        expected = jnp.diag(jnp.array([4.0, 4.0, 4.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_atoms_with_extent(self) -> None:
        """Atoms spanning 0-10 A in each direction."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        lattice = _infer_lattice_from_positions(positions, padding_ang=2.0)

        # Extent = 10, + 2*2 = 14 in each direction
        expected = jnp.diag(jnp.array([14.0, 14.0, 14.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_asymmetric_extent(self) -> None:
        """Atoms with different extents in each direction."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 10.0, 15.0],
            ]
        )
        lattice = _infer_lattice_from_positions(positions, padding_ang=1.0)

        # Extents: x=5, y=10, z=15, + 2*1 = 7, 12, 17
        expected = jnp.diag(jnp.array([7.0, 12.0, 17.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)

    def test_minimum_extent_enforced(self) -> None:
        """Minimum extent of 1 A should be enforced."""
        positions = jnp.array([[5.0, 5.0, 5.0]])  # Single atom
        lattice = _infer_lattice_from_positions(positions, padding_ang=0.0)

        # No padding, extent = 0, minimum = 1.0
        expected = jnp.diag(jnp.array([1.0, 1.0, 1.0]))
        chex.assert_trees_all_close(lattice, expected, atol=1e-10)


class TestXyzToCrystal(chex.TestCase):
    """Test XYZ to CrystalStructure conversion."""

    def test_with_explicit_lattice(self) -> None:
        """XYZData with explicit cell_vectors override."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.1, 2.1, 2.1]])
        atomic_numbers = jnp.array([12, 8])  # Mg, O
        lattice = jnp.eye(3) * 4.2

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal = xyz_to_crystal(xyz_data)

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
        expected_frac = positions / 4.2
        chex.assert_trees_all_close(
            crystal.frac_positions[:, :3], expected_frac, atol=1e-10
        )

    def test_with_cell_vectors_override(self) -> None:
        """Test explicit cell_vectors parameter overrides xyz_data.lattice."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([14])  # Si
        xyz_lattice = jnp.eye(3) * 5.43
        override_lattice = jnp.eye(3) * 10.0

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=xyz_lattice,
        )

        crystal = xyz_to_crystal(xyz_data, cell_vectors=override_lattice)

        # Should use override, not xyz_data.lattice
        chex.assert_trees_all_close(
            crystal.cell_lengths, jnp.array([10.0, 10.0, 10.0]), atol=1e-6
        )

    def test_non_orthorhombic_lattice(self) -> None:
        """XYZ with non-orthorhombic (hexagonal) lattice."""
        a = 3.2
        c = 5.2
        # Hexagonal lattice vectors
        lattice = jnp.array(
            [[a, 0, 0], [-a / 2, a * jnp.sqrt(3) / 2, 0], [0, 0, c]]
        )

        positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, c / 2]])
        atomic_numbers = jnp.array([30, 8])  # Zn, O

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal = xyz_to_crystal(xyz_data)

        # Check hexagonal angles
        chex.assert_trees_all_close(crystal.cell_angles[2], 120.0, atol=1e-5)
        chex.assert_trees_all_close(
            crystal.cell_angles[:2], jnp.array([90.0, 90.0]), atol=1e-5
        )

    def test_preserves_atomic_numbers(self) -> None:
        """Atomic numbers correctly transferred to 4th column."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        atomic_numbers = jnp.array([26, 8, 14])  # Fe, O, Si

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=jnp.eye(3) * 5.0,
        )

        crystal = xyz_to_crystal(xyz_data)

        # Both frac and cart should have same atomic numbers
        expected = jnp.array([26.0, 8.0, 14.0])
        chex.assert_trees_all_close(
            crystal.frac_positions[:, 3], expected, atol=1e-10
        )
        chex.assert_trees_all_close(
            crystal.cart_positions[:, 3], expected, atol=1e-10
        )

    def test_cartesian_positions_preserved(self) -> None:
        """Cartesian positions should be preserved exactly."""
        positions = jnp.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
        atomic_numbers = jnp.array([6, 7])  # C, N
        lattice = jnp.eye(3) * 10.0

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal = xyz_to_crystal(xyz_data)

        chex.assert_trees_all_close(
            crystal.cart_positions[:, :3], positions, atol=1e-10
        )

    def test_returns_crystal_structure(self) -> None:
        """Should return CrystalStructure instance."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1])
        lattice = jnp.eye(3) * 5.0

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal = xyz_to_crystal(xyz_data)

        assert isinstance(crystal, CrystalStructure)


class TestParseCrystal(chex.TestCase):
    """Test unified crystal parser."""

    def test_parse_cif(self) -> None:
        """Parse CIF file via parse_crystal."""
        cif_content = """
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal = parse_crystal(cif_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([4.212, 4.212, 4.212]),
                atol=1e-3,
            )

    def test_parse_xyz(self) -> None:
        """Parse XYZ file via parse_crystal."""
        xyz_content = """2
Lattice="4.2 0.0 0.0 0.0 4.2 0.0 0.0 0.0 4.2"
Mg 0.0 0.0 0.0
O  2.1 2.1 2.1
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            crystal = parse_crystal(xyz_file)

            assert isinstance(crystal, CrystalStructure)
            assert crystal.cart_positions.shape[0] == 2
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([4.2, 4.2, 4.2]),
                atol=1e-6,
            )

    def test_unsupported_format(self) -> None:
        """Raise ValueError for unsupported format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "test.pdb"
            bad_file.write_text("dummy content")

            with pytest.raises(ValueError, match="Unsupported file format"):
                parse_crystal(bad_file)

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_crystal("/nonexistent/path.cif")

    def test_cif_xyz_equivalence(self) -> None:
        """Same structure from CIF and XYZ should give similar results."""
        # Simple cubic MgO
        cif_content = """
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

        xyz_content = """2
Lattice="4.212 0.0 0.0 0.0 4.212 0.0 0.0 0.0 4.212"
Mg 0.0 0.0 0.0
O  2.106 2.106 2.106
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "mgo.cif"
            xyz_file = Path(tmp_dir) / "mgo.xyz"
            cif_file.write_text(cif_content)
            xyz_file.write_text(xyz_content)

            crystal_cif = parse_crystal(cif_file)
            crystal_xyz = parse_crystal(xyz_file)

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
        """Test that string paths work."""
        cif_content = """
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            # Pass as string, not Path
            crystal = parse_crystal(str(cif_file))

            assert isinstance(crystal, CrystalStructure)


class TestCrystalRoundtrip(chex.TestCase):
    """Test consistency between lattice and cell parameter conversions."""

    def test_orthorhombic_roundtrip(self) -> None:
        """Orthorhombic lattice survives roundtrip conversion."""
        a, b, c = 4.0, 5.0, 6.0
        lattice = jnp.array([[a, 0, 0], [0, b, 0], [0, 0, c]])

        lengths, angles = lattice_to_cell_params(lattice)

        # Lengths should match
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), atol=1e-10)

        # Angles should be 90 deg
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_xyz_to_crystal_fractional_consistency(self) -> None:
        """Fractional coords times lattice should give Cartesian coords."""
        positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        atomic_numbers = jnp.array([6, 7])
        lattice = jnp.eye(3) * 10.0

        xyz_data = create_xyz_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
        )

        crystal = xyz_to_crystal(xyz_data)

        # frac @ lattice should give cart
        reconstructed_cart = crystal.frac_positions[:, :3] @ lattice
        chex.assert_trees_all_close(
            reconstructed_cart, crystal.cart_positions[:, :3], atol=1e-10
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
