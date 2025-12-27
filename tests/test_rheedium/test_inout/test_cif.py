"""Tests for CIF file parsing and symmetry expansion."""

import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout.cif import (
    _deduplicate_positions,
    _extract_cell_params,
    _extract_sym_op_from_line,
    _parse_atom_positions,
    _parse_sym_op,
    _parse_symmetry_ops,
    parse_cif,
    symmetry_expansion,
)
from rheedium.types import CrystalStructure, create_crystal_structure


class TestExtractCellParams(chex.TestCase):
    """Test cell parameter extraction from CIF text."""

    def test_standard_cif_params(self) -> None:
        """Extract standard cell parameters."""
        cif_text = """
data_test
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
"""
        a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)

        assert a == pytest.approx(5.43)
        assert b == pytest.approx(5.43)
        assert c == pytest.approx(5.43)
        assert alpha == pytest.approx(90.0)
        assert beta == pytest.approx(90.0)
        assert gamma == pytest.approx(90.0)

    def test_hexagonal_params(self) -> None:
        """Extract hexagonal cell parameters."""
        cif_text = """
_cell_length_a 3.25
_cell_length_b 3.25
_cell_length_c 5.21
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""
        a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)

        assert a == pytest.approx(3.25)
        assert c == pytest.approx(5.21)
        assert gamma == pytest.approx(120.0)

    def test_missing_param_raises(self) -> None:
        """Missing parameter should raise ValueError."""
        cif_text = """
_cell_length_a 5.43
_cell_length_b 5.43
_cell_angle_alpha 90
"""
        with pytest.raises(ValueError, match="_cell_length_c"):
            _extract_cell_params(cif_text)


class TestParseAtomPositions(chex.TestCase):
    """Test atomic position parsing from CIF lines."""

    def test_standard_atom_loop(self) -> None:
        """Parse standard atom site loop."""
        lines = [
            "loop_",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "Si 0.0 0.0 0.0",
            "Si 0.25 0.25 0.25",
        ]
        positions = _parse_atom_positions(lines)

        assert len(positions) == 2
        assert positions[0][:3] == [0.0, 0.0, 0.0]
        assert positions[1][:3] == [0.25, 0.25, 0.25]
        assert positions[0][3] == 14
        assert positions[1][3] == 14

    def test_atom_loop_with_extra_columns(self) -> None:
        """Parse atom loop with additional columns."""
        lines = [
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
            "Mg1 Mg 0.0 0.0 0.0 1.0",
            "O1 O 0.5 0.5 0.5 1.0",
        ]
        positions = _parse_atom_positions(lines)

        assert len(positions) == 2
        assert positions[0][3] == 12
        assert positions[1][3] == 8

    def test_empty_lines_return_empty(self) -> None:
        """No atom site loop should return empty list."""
        lines = [
            "_cell_length_a 5.0",
            "_cell_length_b 5.0",
        ]
        positions = _parse_atom_positions(lines)

        assert positions == []


class TestParseSymmetryOps(chex.TestCase):
    """Test symmetry operation parsing."""

    def test_quoted_symmetry_ops(self) -> None:
        """Parse quoted symmetry operations."""
        lines = [
            "loop_",
            "_symmetry_equiv_pos_as_xyz",
            "'x,y,z'",
            "'-x,-y,z'",
            "'x,-y,-z'",
        ]
        sym_ops = _parse_symmetry_ops(lines)

        assert len(sym_ops) == 3
        assert "x,y,z" in sym_ops
        assert "-x,-y,z" in sym_ops

    def test_default_identity_op(self) -> None:
        """Default to identity when no symmetry ops found."""
        lines = ["_cell_length_a 5.0"]
        sym_ops = _parse_symmetry_ops(lines)

        assert sym_ops == ["x,y,z"]


class TestExtractSymOpFromLine(chex.TestCase):
    """Test symmetry operation extraction from single line."""

    def test_single_quoted(self) -> None:
        """Extract from single-quoted line."""
        op = _extract_sym_op_from_line("'x,y,z'", [])

        assert op == "x,y,z"

    def test_double_quoted(self) -> None:
        """Extract from double-quoted line."""
        op = _extract_sym_op_from_line('"x,-y,z"', [])

        assert op == "x,-y,z"

    def test_with_xyz_column(self) -> None:
        """Extract when xyz column is specified."""
        columns = ["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"]
        op = _extract_sym_op_from_line("1 'x,y,z'", columns)

        assert op == "x,y,z"


class TestParseSymOp(chex.TestCase):
    """Test symmetry operation parsing into callable."""

    def test_identity(self) -> None:
        """Identity operation x,y,z."""
        op = _parse_sym_op("x,y,z")
        pos = jnp.array([0.25, 0.5, 0.75])
        result = op(pos)

        chex.assert_trees_all_close(result, pos, atol=1e-10)

    def test_inversion(self) -> None:
        """Inversion operation -x,-y,-z."""
        op = _parse_sym_op("-x,-y,-z")
        pos = jnp.array([0.25, 0.5, 0.75])
        result = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([-0.25, -0.5, -0.75]), atol=1e-10
        )

    def test_with_translation(self) -> None:
        """Operation with translation x+1/2,y,z."""
        op = _parse_sym_op("x+1/2,y,z")
        pos = jnp.array([0.0, 0.0, 0.0])
        result = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([0.5, 0.0, 0.0]), atol=1e-10
        )

    def test_cubic_symmetry(self) -> None:
        """Face-centered cubic symmetry operation."""
        op = _parse_sym_op("-y,x,z")
        pos = jnp.array([0.25, 0.5, 0.75])
        result = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([-0.5, 0.25, 0.75]), atol=1e-10
        )


class TestDeduplicatePositions(chex.TestCase):
    """Test position deduplication."""

    def test_removes_duplicates(self) -> None:
        """Duplicate positions within tolerance are removed."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [0.001, 0.001, 0.001, 14.0],
                [1.0, 1.0, 1.0, 8.0],
            ]
        )
        unique = _deduplicate_positions(positions, tol=0.1)

        assert unique.shape[0] == 2

    def test_keeps_distinct_atoms(self) -> None:
        """Distinct positions are preserved."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [2.0, 2.0, 2.0, 8.0],
                [4.0, 4.0, 4.0, 14.0],
            ]
        )
        unique = _deduplicate_positions(positions, tol=0.5)

        assert unique.shape[0] == 3


class TestSymmetryExpansion(chex.TestCase):
    """Test symmetry expansion of crystal structures."""

    def test_identity_expansion(self) -> None:
        """Identity operation should not change structure."""
        frac_positions = jnp.array([[0.0, 0.0, 0.0, 14.0]])
        cart_positions = jnp.array([[0.0, 0.0, 0.0, 14.0]])
        cell_lengths = jnp.array([5.43, 5.43, 5.43])
        cell_angles = jnp.array([90.0, 90.0, 90.0])

        crystal = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        expanded = symmetry_expansion(crystal, ["x,y,z"], tolerance=0.5)

        assert expanded.frac_positions.shape[0] == 1

    def test_inversion_expansion(self) -> None:
        """Inversion should double atoms not at origin."""
        frac_positions = jnp.array([[0.25, 0.25, 0.25, 14.0]])
        cart_positions = jnp.array([[1.36, 1.36, 1.36, 14.0]])
        cell_lengths = jnp.array([5.43, 5.43, 5.43])
        cell_angles = jnp.array([90.0, 90.0, 90.0])

        crystal = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        expanded = symmetry_expansion(
            crystal, ["x,y,z", "-x,-y,-z"], tolerance=0.5
        )

        assert expanded.frac_positions.shape[0] == 2

    def test_fcc_expansion(self) -> None:
        """FCC symmetry should generate 4 atoms from 1."""
        frac_positions = jnp.array([[0.0, 0.0, 0.0, 29.0]])
        cart_positions = jnp.array([[0.0, 0.0, 0.0, 29.0]])
        cell_lengths = jnp.array([3.61, 3.61, 3.61])
        cell_angles = jnp.array([90.0, 90.0, 90.0])

        crystal = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        fcc_ops = [
            "x,y,z",
            "x+1/2,y+1/2,z",
            "x+1/2,y,z+1/2",
            "x,y+1/2,z+1/2",
        ]
        expanded = symmetry_expansion(crystal, fcc_ops, tolerance=0.5)

        assert expanded.frac_positions.shape[0] == 4


class TestParseCif(chex.TestCase):
    """Test complete CIF file parsing."""

    def test_simple_cif(self) -> None:
        """Parse simple cubic CIF file."""
        cif_content = """
data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na 0.0 0.0 0.0
Cl 0.5 0.5 0.5
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "nacl.cif"
            cif_file.write_text(cif_content)

            crystal = parse_cif(cif_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.64, 5.64, 5.64]),
                atol=1e-3,
            )

    def test_cif_with_symmetry(self) -> None:
        """Parse CIF with symmetry operations."""
        cif_content = """
data_Si
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
'-x,-y,z'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.125 0.125 0.125
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "si.cif"
            cif_file.write_text(cif_content)

            crystal = parse_cif(cif_file)

            assert crystal.frac_positions.shape[0] >= 1

    def test_file_not_found(self) -> None:
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_cif("/nonexistent/path.cif")

    def test_wrong_extension(self) -> None:
        """Wrong extension should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "test.xyz"
            bad_file.write_text("dummy")

            with pytest.raises(ValueError, match=".cif extension"):
                parse_cif(bad_file)

    def test_no_atoms_raises(self) -> None:
        """CIF without atoms should raise ValueError."""
        cif_content = """
data_empty
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "empty.cif"
            cif_file.write_text(cif_content)

            with pytest.raises(ValueError, match="No atomic positions"):
                parse_cif(cif_file)

    def test_path_as_string(self) -> None:
        """String path should work."""
        cif_content = """
data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal = parse_cif(str(cif_file))

            assert isinstance(crystal, CrystalStructure)


class TestCifAtomicNumbers(chex.TestCase):
    """Test atomic number assignment in CIF parsing."""

    @parameterized.named_parameters(
        ("hydrogen", "H", 1),
        ("carbon", "C", 6),
        ("oxygen", "O", 8),
        ("silicon", "Si", 14),
        ("iron", "Fe", 26),
        ("gold", "Au", 79),
    )
    def test_element_atomic_numbers(
        self, element: str, expected_z: int
    ) -> None:
        """Verify correct atomic numbers for various elements."""
        cif_content = f"""
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
{element} 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal = parse_cif(cif_file)

            assert crystal.frac_positions[0, 3] == expected_z


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
