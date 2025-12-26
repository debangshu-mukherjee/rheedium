"""Tests for XYZ file parsing and atomic data utilities."""

import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout.xyz import (
    _parse_atom_line,
    _parse_xyz_metadata,
    atomic_symbol,
    kirkland_potentials,
    parse_xyz,
)
from rheedium.types import XYZData


class TestAtomicSymbol(chex.TestCase):
    """Test atomic symbol to number conversion."""

    @parameterized.named_parameters(
        ("hydrogen", "H", 1),
        ("helium", "He", 2),
        ("carbon", "C", 6),
        ("nitrogen", "N", 7),
        ("oxygen", "O", 8),
        ("silicon", "Si", 14),
        ("iron", "Fe", 26),
        ("copper", "Cu", 29),
        ("silver", "Ag", 47),
        ("gold", "Au", 79),
    )
    def test_common_elements(self, symbol: str, expected: int) -> None:
        """Test common element symbols."""
        result = atomic_symbol(symbol)
        assert result == expected

    def test_case_insensitive(self) -> None:
        """Symbol lookup should be case insensitive."""
        assert atomic_symbol("fe") == 26
        assert atomic_symbol("FE") == 26
        assert atomic_symbol("Fe") == 26

    def test_whitespace_handling(self) -> None:
        """Whitespace should be stripped."""
        assert atomic_symbol("  Si  ") == 14
        assert atomic_symbol("\tAu\n") == 79

    def test_invalid_symbol_raises(self) -> None:
        """Invalid symbol should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            atomic_symbol("Xx")

    def test_empty_string_raises(self) -> None:
        """Empty string should raise KeyError."""
        with pytest.raises(KeyError):
            atomic_symbol("")


class TestKirklandPotentials(chex.TestCase):
    """Test Kirkland potential loading."""

    def test_returns_array(self) -> None:
        """Should return JAX array."""
        potentials = kirkland_potentials()
        assert isinstance(potentials, jnp.ndarray)

    def test_correct_shape(self) -> None:
        """Should have shape (103, 12)."""
        potentials = kirkland_potentials()
        assert potentials.shape == (103, 12)

    def test_values_finite(self) -> None:
        """All values should be finite."""
        potentials = kirkland_potentials()
        assert jnp.all(jnp.isfinite(potentials))

    def test_cached_access(self) -> None:
        """Multiple calls should return same object."""
        p1 = kirkland_potentials()
        p2 = kirkland_potentials()
        chex.assert_trees_all_close(p1, p2)


class TestParseXyzMetadata(chex.TestCase):
    """Test XYZ metadata parsing."""

    def test_lattice_extraction(self) -> None:
        """Extract lattice from extended XYZ format."""
        line = 'Lattice="4.2 0.0 0.0 0.0 4.2 0.0 0.0 0.0 4.2"'
        metadata = _parse_xyz_metadata(line)

        assert "lattice" in metadata
        expected = jnp.array([
            [4.2, 0.0, 0.0],
            [0.0, 4.2, 0.0],
            [0.0, 0.0, 4.2],
        ])
        chex.assert_trees_all_close(metadata["lattice"], expected, atol=1e-10)

    def test_stress_extraction(self) -> None:
        """Extract stress tensor from metadata."""
        line = 'stress="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"'
        metadata = _parse_xyz_metadata(line)

        assert "stress" in metadata
        assert metadata["stress"].shape == (3, 3)

    def test_energy_extraction(self) -> None:
        """Extract energy from metadata."""
        line = "energy=-123.456"
        metadata = _parse_xyz_metadata(line)

        assert "energy" in metadata
        assert metadata["energy"] == pytest.approx(-123.456)

    def test_energy_scientific_notation(self) -> None:
        """Extract energy with scientific notation."""
        line = "energy=-1.23e-4"
        metadata = _parse_xyz_metadata(line)

        assert metadata["energy"] == pytest.approx(-1.23e-4)

    def test_properties_extraction(self) -> None:
        """Extract properties descriptor."""
        line = "Properties=species:S:1:pos:R:3"
        metadata = _parse_xyz_metadata(line)

        assert "properties" in metadata
        assert len(metadata["properties"]) == 2
        assert metadata["properties"][0]["name"] == "species"
        assert metadata["properties"][1]["name"] == "pos"

    def test_empty_line(self) -> None:
        """Empty line returns empty metadata."""
        metadata = _parse_xyz_metadata("")
        assert metadata == {}

    def test_comment_only(self) -> None:
        """Plain comment returns empty metadata."""
        metadata = _parse_xyz_metadata("This is a comment")
        assert metadata == {}

    def test_invalid_lattice_raises(self) -> None:
        """Wrong number of lattice values should raise."""
        line = 'Lattice="4.2 0.0 0.0 0.0 4.2"'
        with pytest.raises(ValueError, match="9 values"):
            _parse_xyz_metadata(line)


class TestParseAtomLine(chex.TestCase):
    """Test single atom line parsing."""

    def test_standard_format(self) -> None:
        """Parse standard 4-column XYZ line."""
        parts = ["Si", "1.5", "2.5", "3.5"]
        symbol, x, y, z = _parse_atom_line(parts)

        assert symbol == "Si"
        assert x == pytest.approx(1.5)
        assert y == pytest.approx(2.5)
        assert z == pytest.approx(3.5)

    def test_extended_format(self) -> None:
        """Parse extended XYZ with extra columns."""
        parts = ["Fe", "0.0", "0.0", "0.0", "1.0"]
        symbol, x, y, z = _parse_atom_line(parts)

        assert symbol == "Fe"
        assert x == pytest.approx(0.0)

    def test_additional_columns(self) -> None:
        """Parse line with many extra columns."""
        parts = ["Au", "1.0", "2.0", "3.0", "0.5", "1.0", "0.8"]
        symbol, x, y, z = _parse_atom_line(parts)

        assert symbol == "Au"
        assert z == pytest.approx(3.0)


class TestParseXyz(chex.TestCase):
    """Test complete XYZ file parsing."""

    def test_simple_xyz(self) -> None:
        """Parse simple XYZ file."""
        xyz_content = """3
Simple H2O molecule
O  0.0 0.0 0.0
H  0.96 0.0 0.0
H  -0.24 0.93 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "water.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert isinstance(data, XYZData)
            assert data.positions.shape == (3, 3)
            assert data.atomic_numbers.shape == (3,)
            assert int(data.atomic_numbers[0]) == 8
            assert int(data.atomic_numbers[1]) == 1

    def test_extended_xyz_with_lattice(self) -> None:
        """Parse extended XYZ with lattice."""
        xyz_content = """2
Lattice="4.2 0.0 0.0 0.0 4.2 0.0 0.0 0.0 4.2"
Mg 0.0 0.0 0.0
O  2.1 2.1 2.1
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "mgo.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.lattice is not None
            chex.assert_trees_all_close(
                data.lattice[0, 0], 4.2, atol=1e-10
            )

    def test_atomic_numbers_as_symbols(self) -> None:
        """XYZ with atomic numbers instead of symbols."""
        xyz_content = """2
Test with atomic numbers
14 0.0 0.0 0.0
8 1.0 1.0 1.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert int(data.atomic_numbers[0]) == 14
            assert int(data.atomic_numbers[1]) == 8

    def test_preserves_comment(self) -> None:
        """Comment line should be preserved."""
        xyz_content = """1
This is a custom comment
Fe 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.comment == "This is a custom comment"

    def test_path_as_string(self) -> None:
        """String path should work."""
        xyz_content = """1
comment
C 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(str(xyz_file))

            assert isinstance(data, XYZData)

    def test_too_few_lines_raises(self) -> None:
        """File with fewer than 2 lines should raise."""
        xyz_content = "1"
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            with pytest.raises(ValueError, match="fewer than 2 lines"):
                parse_xyz(xyz_file)

    def test_invalid_atom_count_raises(self) -> None:
        """Non-integer atom count should raise."""
        xyz_content = """abc
comment
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            with pytest.raises(ValueError, match="number of atoms"):
                parse_xyz(xyz_file)

    def test_insufficient_atoms_raises(self) -> None:
        """Fewer atoms than declared should raise."""
        xyz_content = """3
comment
O 0.0 0.0 0.0
H 1.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            with pytest.raises(ValueError, match="Expected 3 atoms"):
                parse_xyz(xyz_file)

    def test_bad_line_format_raises(self) -> None:
        """Malformed atom line should raise."""
        xyz_content = """1
comment
O 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            with pytest.raises(ValueError, match="unexpected format"):
                parse_xyz(xyz_file)


class TestXyzDataStructure(chex.TestCase):
    """Test XYZData structure and fields."""

    def test_positions_dtype(self) -> None:
        """Positions should be float64."""
        xyz_content = """1
comment
Si 1.5 2.5 3.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.positions.dtype == jnp.float64

    def test_atomic_numbers_dtype(self) -> None:
        """Atomic numbers should be int32."""
        xyz_content = """1
comment
Fe 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.atomic_numbers.dtype == jnp.int32

    def test_optional_fields_none(self) -> None:
        """Optional fields should be None when not present."""
        xyz_content = """1
plain comment
C 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "test.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.stress is None
            assert data.energy is None
            assert data.properties is None


class TestXyzLargeFiles(chex.TestCase):
    """Test XYZ parsing with larger files."""

    def test_many_atoms(self) -> None:
        """Parse file with many atoms."""
        n_atoms = 100
        lines = [f"{n_atoms}", "Many atoms test"]
        for i in range(n_atoms):
            lines.append(f"C {i * 0.1} {i * 0.2} {i * 0.3}")

        xyz_content = "\n".join(lines)

        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_file = Path(tmp_dir) / "large.xyz"
            xyz_file.write_text(xyz_content)

            data = parse_xyz(xyz_file)

            assert data.positions.shape[0] == n_atoms
            assert data.atomic_numbers.shape[0] == n_atoms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
