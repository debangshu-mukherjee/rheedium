"""Tests for VASP POSCAR/CONTCAR file parsing."""

import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout.poscar import (
    _parse_poscar_header,
    _parse_poscar_positions,
    parse_poscar,
)
from rheedium.types import CrystalStructure


class TestParsePoscarHeader(chex.TestCase):
    """Test POSCAR header parsing."""

    def test_simple_cubic(self) -> None:
        """Parse simple cubic Si header."""
        lines = [
            "Simple cubic Si",
            "1.0",
            "5.43 0.0 0.0",
            "0.0 5.43 0.0",
            "0.0 0.0 5.43",
            "Si",
            "1",
        ]
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert scaling == 1.0
        chex.assert_trees_all_close(
            lattice,
            jnp.array([[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]]),
            atol=1e-10,
        )
        assert species == ["Si"]
        assert counts == [1]

    def test_multi_species(self) -> None:
        """Parse multi-species structure (NaCl)."""
        lines = [
            "NaCl rock salt",
            "1.0",
            "5.64 0.0 0.0",
            "0.0 5.64 0.0",
            "0.0 0.0 5.64",
            "Na Cl",
            "4 4",
        ]
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert species == ["Na", "Cl"]
        assert counts == [4, 4]

    def test_scaling_factor(self) -> None:
        """Scaling factor applied to lattice vectors."""
        lines = [
            "Scaled Si",
            "2.0",
            "2.715 0.0 0.0",
            "0.0 2.715 0.0",
            "0.0 0.0 2.715",
            "Si",
            "1",
        ]
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert scaling == 2.0
        # Lattice vectors should be doubled
        chex.assert_trees_all_close(
            lattice,
            jnp.array([[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]]),
            atol=1e-10,
        )

    def test_invalid_scaling_factor(self) -> None:
        """Invalid scaling factor raises ValueError."""
        lines = [
            "Invalid",
            "not_a_number",
            "5.43 0.0 0.0",
            "0.0 5.43 0.0",
            "0.0 0.0 5.43",
            "Si",
            "1",
        ]
        with pytest.raises(ValueError, match="scaling factor"):
            _parse_poscar_header(lines)

    def test_invalid_lattice_vector(self) -> None:
        """Invalid lattice vector raises ValueError."""
        lines = [
            "Invalid",
            "1.0",
            "5.43 0.0",  # Missing third component
            "0.0 5.43 0.0",
            "0.0 0.0 5.43",
            "Si",
            "1",
        ]
        with pytest.raises(ValueError, match="lattice vector"):
            _parse_poscar_header(lines)

    def test_mismatched_species_counts(self) -> None:
        """Mismatched species and counts raises ValueError."""
        lines = [
            "Mismatch",
            "1.0",
            "5.43 0.0 0.0",
            "0.0 5.43 0.0",
            "0.0 0.0 5.43",
            "Si O",  # 2 species
            "1",  # Only 1 count
        ]
        with pytest.raises(ValueError, match="atom counts"):
            _parse_poscar_header(lines)


class TestParsePoscarPositions(chex.TestCase):
    """Test POSCAR position parsing."""

    def test_direct_coordinates(self) -> None:
        """Parse Direct (fractional) coordinates."""
        lines = [
            "0.0 0.0 0.0",
            "0.5 0.5 0.5",
        ]
        lattice = jnp.eye(3) * 4.0
        positions = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=False, lattice=lattice
        )

        expected = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_cartesian_coordinates(self) -> None:
        """Parse Cartesian coordinates and convert to fractional."""
        lines = [
            "0.0 0.0 0.0",
            "2.0 2.0 2.0",
        ]
        lattice = jnp.eye(3) * 4.0
        positions = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=True, lattice=lattice
        )

        # 2.0 / 4.0 = 0.5 in each direction
        expected = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_selective_dynamics_ignored(self) -> None:
        """Selective dynamics flags (T/F) are ignored."""
        lines = [
            "0.0 0.0 0.0 T T T",
            "0.5 0.5 0.5 F F F",
        ]
        lattice = jnp.eye(3) * 4.0
        positions = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=False, lattice=lattice
        )

        expected = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_insufficient_atoms(self) -> None:
        """Requesting more atoms than available raises ValueError."""
        lines = [
            "0.0 0.0 0.0",
        ]
        lattice = jnp.eye(3) * 4.0
        with pytest.raises(ValueError, match="file ends"):
            _parse_poscar_positions(
                lines, start_idx=0, n_atoms=2, is_cartesian=False, lattice=lattice
            )


class TestParsePoscar(chex.TestCase):
    """Test complete POSCAR file parsing."""

    def test_simple_cubic_si(self) -> None:
        """Parse simple cubic Si POSCAR."""
        poscar_content = """Simple cubic Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Direct
  0.0  0.0  0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.43, 5.43, 5.43]),
                atol=1e-3,
            )
            chex.assert_trees_all_close(
                crystal.cell_angles,
                jnp.array([90.0, 90.0, 90.0]),
                atol=1e-3,
            )
            assert crystal.frac_positions.shape == (1, 4)
            # Si has Z=14
            chex.assert_trees_all_close(
                crystal.frac_positions[0, 3], 14.0, atol=1e-10
            )

    def test_mgo_rock_salt(self) -> None:
        """Parse MgO rock salt structure."""
        poscar_content = """MgO rock salt
1.0
  4.21  0.00  0.00
  0.00  4.21  0.00
  0.00  0.00  4.21
  Mg O
  1 1
Direct
  0.0  0.0  0.0
  0.5  0.5  0.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            assert crystal.frac_positions.shape == (2, 4)
            # Mg=12, O=8
            chex.assert_trees_all_close(
                crystal.frac_positions[:, 3],
                jnp.array([12.0, 8.0]),
                atol=1e-10,
            )

    def test_cartesian_mode(self) -> None:
        """Parse POSCAR with Cartesian coordinates."""
        poscar_content = """Cartesian Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  2
Cartesian
  0.0    0.0    0.0
  2.715  2.715  2.715
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            # 2.715 / 5.43 = 0.5
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                atol=1e-3,
            )

    def test_selective_dynamics(self) -> None:
        """Parse POSCAR with selective dynamics."""
        poscar_content = """Selective dynamics Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  2
Selective dynamics
Direct
  0.0  0.0  0.0  T T T
  0.5  0.5  0.5  F F F
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            assert crystal.frac_positions.shape == (2, 4)
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                atol=1e-10,
            )

    def test_scaling_factor(self) -> None:
        """Scaling factor applied correctly."""
        poscar_content = """Scaled Si
2.0
  2.715  0.00  0.00
  0.00  2.715  0.00
  0.00  0.00  2.715
  Si
  1
Direct
  0.0  0.0  0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            # 2.715 * 2.0 = 5.43
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.43, 5.43, 5.43]),
                atol=1e-3,
            )

    def test_non_orthogonal_cell(self) -> None:
        """Parse POSCAR with non-orthogonal (hexagonal) cell."""
        a = 3.0
        c = 5.0
        # Hexagonal lattice vectors
        poscar_content = f"""Hexagonal ZnO
1.0
  {a}  0.0  0.0
  {-a/2}  {a * 0.866025}  0.0
  0.0  0.0  {c}
  Zn O
  1 1
Direct
  0.0  0.0  0.0
  0.333333  0.666667  0.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([a, a, c]),
                atol=1e-3,
            )
            # gamma should be 120 degrees
            chex.assert_trees_all_close(
                crystal.cell_angles[2], 120.0, atol=1e-1
            )

    def test_file_not_found(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_poscar("/nonexistent/path/POSCAR")

    def test_too_few_lines(self) -> None:
        """File with too few lines raises ValueError."""
        poscar_content = """Short file
1.0
5.43 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            with pytest.raises(ValueError, match="at least"):
                parse_poscar(poscar_file)

    def test_invalid_coordinate_mode(self) -> None:
        """Invalid coordinate mode raises ValueError."""
        poscar_content = """Invalid mode
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Invalid
  0.0  0.0  0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            with pytest.raises(ValueError, match="Direct.*Cartesian"):
                parse_poscar(poscar_file)

    def test_string_path(self) -> None:
        """Accept string path as well as Path object."""
        poscar_content = """Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Direct
  0.0  0.0  0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            # Pass as string instead of Path
            crystal = parse_poscar(str(poscar_file))
            assert isinstance(crystal, CrystalStructure)

    @parameterized.named_parameters(
        ("silicon", "Si", 14),
        ("magnesium", "Mg", 12),
        ("oxygen", "O", 8),
        ("iron", "Fe", 26),
        ("gold", "Au", 79),
    )
    def test_various_elements(self, element: str, expected_z: int) -> None:
        """Test parsing various element types."""
        poscar_content = f"""Element test
1.0
  4.0  0.0  0.0
  0.0  4.0  0.0
  0.0  0.0  4.0
  {element}
  1
Direct
  0.0  0.0  0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)
            chex.assert_trees_all_close(
                crystal.frac_positions[0, 3],
                float(expected_z),
                atol=1e-10,
            )


class TestPoscarRoundtrip(chex.TestCase):
    """Test POSCAR parsing consistency."""

    def test_frac_cart_consistency(self) -> None:
        """Fractional and Cartesian positions should be consistent."""
        poscar_content = """MgO
1.0
  4.21  0.00  0.00
  0.00  4.21  0.00
  0.00  0.00  4.21
  Mg O
  1 1
Direct
  0.0  0.0  0.0
  0.5  0.5  0.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            # Cartesian = frac @ lattice
            expected_cart = crystal.frac_positions[:, :3] @ jnp.diag(
                crystal.cell_lengths
            )
            chex.assert_trees_all_close(
                crystal.cart_positions[:, :3],
                expected_cart,
                atol=1e-6,
            )

    def test_atomic_numbers_preserved(self) -> None:
        """Atomic numbers match in both frac and cart positions."""
        poscar_content = """Multi-element
1.0
  4.0  0.0  0.0
  0.0  4.0  0.0
  0.0  0.0  4.0
  Fe O Si
  1 2 1
Direct
  0.0  0.0  0.0
  0.25 0.25 0.25
  0.75 0.75 0.75
  0.5  0.5  0.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal = parse_poscar(poscar_file)

            # Atomic numbers should match in both arrays
            chex.assert_trees_all_close(
                crystal.frac_positions[:, 3],
                crystal.cart_positions[:, 3],
                atol=1e-10,
            )
            # Expected: Fe=26, O=8, O=8, Si=14
            chex.assert_trees_all_close(
                crystal.frac_positions[:, 3],
                jnp.array([26.0, 8.0, 8.0, 14.0]),
                atol=1e-10,
            )
