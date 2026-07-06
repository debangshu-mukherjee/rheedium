"""Tests for CIF file parsing and symmetry expansion."""

import tempfile
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

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
from rheedium.types.crystal_types import (
    CrystalStructure,
    create_crystal_structure,
)


class TestExtractCellParams(chex.TestCase):
    """Test cell parameter extraction from CIF text.

    :see: :func:`~rheedium.inout.cif._extract_cell_params`
    """

    def test_standard_cif_params(self) -> None:
        r"""Extract standard cell parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract standard
        cell parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_text: str = """
data_test
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
"""
        a: float
        b: float
        c: float
        alpha: float
        beta: float
        gamma: float
        a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)

        assert a == pytest.approx(5.43)
        assert b == pytest.approx(5.43)
        assert c == pytest.approx(5.43)
        assert alpha == pytest.approx(90.0)
        assert beta == pytest.approx(90.0)
        assert gamma == pytest.approx(90.0)

    def test_hexagonal_params(self) -> None:
        r"""Extract hexagonal cell parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract hexagonal
        cell parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_text: str = """
_cell_length_a 3.25
_cell_length_b 3.25
_cell_length_c 5.21
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""
        a: float
        b: float
        c: float
        alpha: float
        beta: float
        gamma: float
        a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)

        assert a == pytest.approx(3.25)
        assert c == pytest.approx(5.21)
        assert gamma == pytest.approx(120.0)

    def test_missing_param_raises(self) -> None:
        r"""Missing parameter should raise ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing parameter
        should raise ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_text: str = """
_cell_length_a 5.43
_cell_length_b 5.43
_cell_angle_alpha 90
"""
        with pytest.raises(ValueError, match="_cell_length_c"):
            _extract_cell_params(cif_text)


class TestParseAtomPositions(chex.TestCase):
    """Test atomic position parsing from CIF lines.

    :see: :func:`~rheedium.inout.cif._parse_atom_positions`
    """

    def test_standard_atom_loop(self) -> None:
        r"""Parse standard atom site loop.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse standard
        atom site loop.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "loop_",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "Si 0.0 0.0 0.0",
            "Si 0.25 0.25 0.25",
        ]
        positions: Float[Array, "..."] = _parse_atom_positions(lines)

        assert len(positions) == 2
        assert positions[0][:3] == [0.0, 0.0, 0.0]
        assert positions[1][:3] == [0.25, 0.25, 0.25]
        assert positions[0][3] == 14
        assert positions[1][3] == 14

    def test_atom_loop_with_extra_columns(self) -> None:
        r"""Parse atom loop with additional columns.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse atom loop
        with additional columns.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
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
        positions: Float[Array, "..."] = _parse_atom_positions(lines)

        assert len(positions) == 2
        assert positions[0][3] == 12
        assert positions[1][3] == 8

    def test_empty_lines_return_empty(self) -> None:
        r"""No atom site loop should return empty list.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No atom site loop
        should return empty list.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "_cell_length_a 5.0",
            "_cell_length_b 5.0",
        ]
        positions: Float[Array, "..."] = _parse_atom_positions(lines)

        assert positions == []


class TestParseSymmetryOps(chex.TestCase):
    """Test symmetry operation parsing.

    :see: :func:`~rheedium.inout.cif._parse_symmetry_ops`
    """

    def test_quoted_symmetry_ops(self) -> None:
        r"""Parse quoted symmetry operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse quoted
        symmetry operations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "loop_",
            "_symmetry_equiv_pos_as_xyz",
            "'x,y,z'",
            "'-x,-y,z'",
            "'x,-y,-z'",
        ]
        sym_ops: Any = _parse_symmetry_ops(lines)

        assert len(sym_ops) == 3
        assert "x,y,z" in sym_ops
        assert "-x,-y,z" in sym_ops

    def test_default_identity_op(self) -> None:
        r"""Default to identity when no symmetry ops found.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Default to
        identity when no symmetry ops found.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = ["_cell_length_a 5.0"]
        sym_ops: Any = _parse_symmetry_ops(lines)

        assert sym_ops == ["x,y,z"]

    def test_modern_space_group_symop_loop(self) -> None:
        """Parse modern IUCr space-group symmetry operation loops."""
        lines: list[str] = [
            "loop_",
            "_space_group_symop_id",
            "_space_group_symop_operation_xyz",
            "1 'x,y,z'",
            "2 'x+1/2,y+1/2,z'",
            "3 'x+1/2,y,z+1/2'",
            "4 'x,y+1/2,z+1/2'",
        ]
        sym_ops: Any = _parse_symmetry_ops(lines)

        assert sym_ops == [
            "x,y,z",
            "x+1/2,y+1/2,z",
            "x+1/2,y,z+1/2",
            "x,y+1/2,z+1/2",
        ]


class TestExtractSymOpFromLine(chex.TestCase):
    """Test symmetry operation extraction from single line.

    :see: :func:`~rheedium.inout.cif._extract_sym_op_from_line`
    """

    def test_single_quoted(self) -> None:
        r"""Extract from single-quoted line.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract from
        single-quoted line.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _extract_sym_op_from_line("'x,y,z'", [])

        assert op == "x,y,z"

    def test_double_quoted(self) -> None:
        r"""Extract from double-quoted line.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract from
        double-quoted line.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _extract_sym_op_from_line('"x,-y,z"', [])

        assert op == "x,-y,z"

    def test_with_xyz_column(self) -> None:
        r"""Extract when xyz column is specified.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract when xyz
        column is specified.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        columns: list[str] = [
            "_symmetry_equiv_pos_site_id",
            "_symmetry_equiv_pos_as_xyz",
        ]
        op: Any = _extract_sym_op_from_line("1 'x,y,z'", columns)

        assert op == "x,y,z"


class TestParseSymOp(chex.TestCase):
    """Test symmetry operation parsing into callable.

    :see: :func:`~rheedium.inout.cif._parse_sym_op`
    """

    def test_identity(self) -> None:
        r"""Identity operation x,y,z.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Identity operation
        x,y,z.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _parse_sym_op("x,y,z")
        pos: Float[Array, "..."] = jnp.array([0.25, 0.5, 0.75])
        result: Float[Array, "..."] = op(pos)

        chex.assert_trees_all_close(result, pos, atol=1e-10)

    def test_inversion(self) -> None:
        r"""Inversion operation -x,-y,-z.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Inversion
        operation -x,-y,-z.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _parse_sym_op("-x,-y,-z")
        pos: Float[Array, "..."] = jnp.array([0.25, 0.5, 0.75])
        result: Float[Array, "..."] = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([-0.25, -0.5, -0.75]), atol=1e-10
        )

    def test_with_translation(self) -> None:
        r"""Operation with translation x+1/2,y,z.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Operation with
        translation x+1/2,y,z.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _parse_sym_op("x+1/2,y,z")
        pos: Float[Array, "..."] = jnp.array([0.0, 0.0, 0.0])
        result: Float[Array, "..."] = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([0.5, 0.0, 0.0]), atol=1e-10
        )

    def test_cubic_symmetry(self) -> None:
        r"""Face-centered cubic symmetry operation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Face-centered
        cubic symmetry operation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        op: Any = _parse_sym_op("-y,x,z")
        pos: Float[Array, "..."] = jnp.array([0.25, 0.5, 0.75])
        result: Float[Array, "..."] = op(pos)

        chex.assert_trees_all_close(
            result, jnp.array([-0.5, 0.25, 0.75]), atol=1e-10
        )


class TestDeduplicatePositions(chex.TestCase):
    """Test position deduplication.

    :see: :func:`~rheedium.inout.cif._deduplicate_positions`
    """

    def test_removes_duplicates(self) -> None:
        r"""Duplicate positions within tolerance are removed.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Duplicate
        positions within tolerance are removed.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [0.001, 0.001, 0.001, 14.0],
                [1.0, 1.0, 1.0, 8.0],
            ]
        )
        unique: Any = _deduplicate_positions(positions, tol=0.1)

        assert unique.shape[0] == 2

    def test_keeps_distinct_atoms(self) -> None:
        r"""Distinct positions are preserved.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Distinct positions
        are preserved.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [2.0, 2.0, 2.0, 8.0],
                [4.0, 4.0, 4.0, 14.0],
            ]
        )
        unique: Any = _deduplicate_positions(positions, tol=0.5)

        assert unique.shape[0] == 3


class TestSymmetryExpansion(chex.TestCase):
    """Test symmetry expansion of crystal structures.

    :see: :func:`~rheedium.inout.symmetry_expansion`
    """

    def test_identity_expansion(self) -> None:
        r"""Identity operation should not change structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Identity operation
        should not change structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 14.0]]
        )
        cart_positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 14.0]]
        )
        cell_lengths: Float[Array, "..."] = jnp.array([5.43, 5.43, 5.43])
        cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        expanded: Any = symmetry_expansion(crystal, ["x,y,z"], tolerance=0.5)

        assert expanded.frac_positions.shape[0] == 1

    def test_inversion_expansion(self) -> None:
        r"""Inversion should double atoms not at origin.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Inversion should
        double atoms not at origin.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "..."] = jnp.array(
            [[0.25, 0.25, 0.25, 14.0]]
        )
        cart_positions: Float[Array, "..."] = jnp.array(
            [[1.3575, 1.3575, 1.3575, 14.0]]
        )
        cell_lengths: Float[Array, "..."] = jnp.array([5.43, 5.43, 5.43])
        cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        expanded: Any = symmetry_expansion(
            crystal, ["x,y,z", "-x,-y,-z"], tolerance=0.5
        )

        assert expanded.frac_positions.shape[0] == 2

    def test_fcc_expansion(self) -> None:
        r"""FCC symmetry should generate 4 atoms from 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FCC symmetry
        should generate 4 atoms from 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 29.0]]
        )
        cart_positions: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 29.0]]
        )
        cell_lengths: Float[Array, "..."] = jnp.array([3.61, 3.61, 3.61])
        cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        fcc_ops: list[str] = [
            "x,y,z",
            "x+1/2,y+1/2,z",
            "x+1/2,y,z+1/2",
            "x,y+1/2,z+1/2",
        ]
        expanded: Any = symmetry_expansion(crystal, fcc_ops, tolerance=0.5)

        assert expanded.frac_positions.shape[0] == 4


class TestParseCif(chex.TestCase):
    """Test complete CIF file parsing.

    :see: :func:`~rheedium.inout.parse_cif`
    """

    def test_simple_cif(self) -> None:
        r"""Parse simple cubic CIF file.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse simple cubic
        CIF file.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "nacl.cif"
            cif_file.write_text(cif_content)

            crystal: CrystalStructure = parse_cif(cif_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.64, 5.64, 5.64]),
                atol=1e-3,
            )

    def test_cif_with_symmetry(self) -> None:
        r"""Parse CIF with symmetry operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse CIF with
        symmetry operations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "si.cif"
            cif_file.write_text(cif_content)

            crystal: CrystalStructure = parse_cif(cif_file)

            assert crystal.frac_positions.shape[0] >= 1

    def test_modern_symop_cif_matches_legacy_loop(self) -> None:
        """Modern and legacy CIF symmetry tags expand the same fcc basis."""
        base_cif: str = """
data_Cu
_cell_length_a 3.61
_cell_length_b 3.61
_cell_length_c 3.61
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
{symmetry_headers}
  1 'x,y,z'
  2 'x+1/2,y+1/2,z'
  3 'x+1/2,y,z+1/2'
  4 'x,y+1/2,z+1/2'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu 0.125 0.125 0.125
"""
        modern_headers = "\n".join(
            [
                "_space_group_symop_id",
                "_space_group_symop_operation_xyz",
            ]
        )
        legacy_headers = "\n".join(
            [
                "_symmetry_equiv_pos_site_id",
                "_symmetry_equiv_pos_as_xyz",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            modern_file: Path = Path(tmp_dir) / "cu_modern.cif"
            legacy_file: Path = Path(tmp_dir) / "cu_legacy.cif"
            modern_file.write_text(
                base_cif.format(symmetry_headers=modern_headers)
            )
            legacy_file.write_text(
                base_cif.format(symmetry_headers=legacy_headers)
            )

            modern = parse_cif(modern_file)
            legacy = parse_cif(legacy_file)

        assert modern.frac_positions.shape[0] == 4
        chex.assert_trees_all_close(
            jnp.sort(modern.frac_positions[:, :3], axis=0),
            jnp.sort(legacy.frac_positions[:, :3], axis=0),
            atol=1e-8,
        )

    def test_non_p1_without_operator_loop_warns(self) -> None:
        """Declared non-P1 space groups without ops warn before P1 fallback."""
        cif_content: str = """
data_Cu
_space_group_name_H-M 'F m -3 m'
_cell_length_a 3.61
_cell_length_b 3.61
_cell_length_c 3.61
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu 0.0 0.0 0.0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "cu_missing_ops.cif"
            cif_file.write_text(cif_content)

            with pytest.warns(
                UserWarning,
                match="space group declared but no operator loop",
            ):
                crystal = parse_cif(cif_file)

        assert crystal.frac_positions.shape[0] == 1

    def test_file_not_found(self) -> None:
        r"""Missing file should raise FileNotFoundError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing file
        should raise FileNotFoundError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(FileNotFoundError):
            parse_cif("/nonexistent/path.cif")

    def test_wrong_extension(self) -> None:
        r"""Wrong extension should raise ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Wrong extension
        should raise ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file: Path = Path(tmp_dir) / "test.xyz"
            bad_file.write_text("dummy")

            with pytest.raises(ValueError, match=".cif extension"):
                parse_cif(bad_file)

    def test_no_atoms_raises(self) -> None:
        r"""CIF without atoms should raise ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CIF without atoms
        should raise ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
data_empty
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "empty.cif"
            cif_file.write_text(cif_content)

            with pytest.raises(ValueError, match="No atomic positions"):
                parse_cif(cif_file)

    def test_non_utf8_file_raises_runtime_error(self) -> None:
        r"""A present but non-UTF-8 CIF file should raise RuntimeError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A present but
        non-UTF-8 CIF file should raise RuntimeError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "binary.cif"
            cif_file.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")

            with pytest.raises(RuntimeError, match="Failed to read CIF"):
                parse_cif(cif_file)

    def test_path_as_string(self) -> None:
        r"""String path should work.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: String path should
        work.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: str = """
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal: CrystalStructure = parse_cif(str(cif_file))

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
        r"""Verify correct atomic numbers for various elements.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: correct atomic
        numbers for various elements.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``element``,
        ``expected_z``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_cif``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cif_content: Any = f"""
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            cif_file: Path = Path(tmp_dir) / "test.cif"
            cif_file.write_text(cif_content)

            crystal: CrystalStructure = parse_cif(cif_file)

            assert crystal.frac_positions[0, 3] == expected_z


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
