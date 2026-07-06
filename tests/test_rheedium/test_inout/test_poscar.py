"""Tests for VASP POSCAR/CONTCAR file parsing."""

import tempfile
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout.poscar import (
    _parse_poscar_header,
    _parse_poscar_positions,
    parse_poscar,
)
from rheedium.types.crystal_types import CrystalStructure


class TestParsePoscarHeader(chex.TestCase):
    """Test POSCAR header parsing.

    :see: :func:`~rheedium.inout.poscar._parse_poscar_header`
    """

    def test_simple_cubic(self) -> None:
        r"""Parse simple cubic Si header.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse simple cubic
        Si header.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "Simple cubic Si",
            "1.0",
            "5.43 0.0 0.0",
            "0.0 5.43 0.0",
            "0.0 0.0 5.43",
            "Si",
            "1",
        ]
        scaling: Any
        lattice: Any
        species: Any
        counts: Any
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
        r"""Parse multi-species structure (NaCl).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse
        multi-species structure (NaCl).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "NaCl rock salt",
            "1.0",
            "5.64 0.0 0.0",
            "0.0 5.64 0.0",
            "0.0 0.0 5.64",
            "Na Cl",
            "4 4",
        ]
        scaling: Any
        lattice: Any
        species: Any
        counts: Any
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert species == ["Na", "Cl"]
        assert counts == [4, 4]

    def test_scaling_factor(self) -> None:
        r"""Scaling factor applied to lattice vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Scaling factor
        applied to lattice vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "Scaled Si",
            "2.0",
            "2.715 0.0 0.0",
            "0.0 2.715 0.0",
            "0.0 0.0 2.715",
            "Si",
            "1",
        ]
        scaling: Any
        lattice: Any
        species: Any
        counts: Any
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert scaling == 2.0
        # Lattice vectors should be doubled
        chex.assert_trees_all_close(
            lattice,
            jnp.array([[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]]),
            atol=1e-10,
        )

    def test_invalid_scaling_factor(self) -> None:
        r"""Invalid scaling factor raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid scaling
        factor raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
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
        r"""Invalid lattice vector raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid lattice
        vector raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
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
        r"""Mismatched species and counts raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mismatched species
        and counts raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
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

    def test_negative_scale_is_target_volume(self) -> None:
        r"""Negative scale is a target cell volume (VASP semantics).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a negative
        universal scale of -160.103007 on a unit lattice resolves to the
        linear factor (160.103007)**(1/3) = 5.43, matching ASE's parsing
        of the same POSCAR (cell lengths [5.43, 5.43, 5.43] Angstroms).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The expected lattice literals are the ASE ``read(format="vasp")``
        ground-truth values for the identical file contents.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "Si target volume",
            "-160.103007",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "0.0 0.0 1.0",
            "Si",
            "1",
        ]
        scaling: Any
        lattice: Any
        species: Any
        counts: Any
        scaling, lattice, species, counts = _parse_poscar_header(lines)

        assert scaling == pytest.approx(5.43, abs=1e-9)
        chex.assert_trees_all_close(
            lattice,
            jnp.array([[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]]),
            atol=1e-9,
        )

    def test_three_scale_factors_rejected(self) -> None:
        r"""A scale line with three values raises an accurate error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a POSCAR scale
        line with three values (per-axis lattice scale factors) is rejected
        with a message that names the actual unsupported feature instead of
        a generic parse failure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "Three scales",
            "2.0 2.0 2.0",
            "2.715 0.0 0.0",
            "0.0 2.715 0.0",
            "0.0 0.0 2.715",
            "Si",
            "1",
        ]
        with pytest.raises(
            ValueError,
            match="three lattice scale factors are not supported",
        ):
            _parse_poscar_header(lines)

    def test_negative_scale_left_handed_lattice_rejected(self) -> None:
        r"""Negative scale with non-positive determinant raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a negative
        (target volume) scale cannot be resolved for a left-handed raw
        lattice whose determinant is not positive, so the parser raises
        instead of producing a complex or NaN linear factor.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "Left-handed",
            "-160.103007",
            "1.0 0.0 0.0",
            "0.0 0.0 1.0",
            "0.0 1.0 0.0",
            "Si",
            "1",
        ]
        with pytest.raises(ValueError, match="positive determinant"):
            _parse_poscar_header(lines)


class TestParsePoscarPositions(chex.TestCase):
    """Test POSCAR position parsing.

    :see: :func:`~rheedium.inout.poscar._parse_poscar_positions`
    """

    def test_direct_coordinates(self) -> None:
        r"""Parse Direct (fractional) coordinates.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse Direct
        (fractional) coordinates.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "0.0 0.0 0.0",
            "0.5 0.5 0.5",
        ]
        lattice: Float[Array, "..."] = jnp.eye(3) * 4.0
        positions: Float[Array, "..."] = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=False, lattice=lattice
        )

        expected: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_cartesian_coordinates(self) -> None:
        r"""Parse Cartesian coordinates and convert to fractional.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse Cartesian
        coordinates and convert to fractional.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "0.0 0.0 0.0",
            "2.0 2.0 2.0",
        ]
        lattice: Float[Array, "..."] = jnp.eye(3) * 4.0
        positions: Float[Array, "..."] = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=True, lattice=lattice
        )

        # 2.0 / 4.0 = 0.5 in each direction
        expected: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_cartesian_coordinates_scaled(self) -> None:
        r"""Cartesian coordinate lines are multiplied by the scale factor.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: in Cartesian
        mode the raw coordinate values are multiplied by the resolved
        universal scale factor before conversion to fractional coordinates,
        matching VASP/ASE semantics.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "0.0 0.0 0.0",
            "1.3575 1.3575 1.3575",
        ]
        # Scaled lattice (raw 2.715 cubic times scale 2.0)
        lattice: Float[Array, "..."] = jnp.eye(3) * 5.43
        positions: Float[Array, "..."] = _parse_poscar_positions(
            lines,
            start_idx=0,
            n_atoms=2,
            is_cartesian=True,
            lattice=lattice,
            scale_factor=2.0,
        )

        # 1.3575 * 2.0 / 5.43 = 0.5 in each direction
        expected: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_selective_dynamics_ignored(self) -> None:
        r"""Selective dynamics flags (T/F) are ignored.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Selective dynamics
        flags (T/F) are ignored.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "0.0 0.0 0.0 T T T",
            "0.5 0.5 0.5 F F F",
        ]
        lattice: Float[Array, "..."] = jnp.eye(3) * 4.0
        positions: Float[Array, "..."] = _parse_poscar_positions(
            lines, start_idx=0, n_atoms=2, is_cartesian=False, lattice=lattice
        )

        expected: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )
        chex.assert_trees_all_close(positions, expected, atol=1e-10)

    def test_insufficient_atoms(self) -> None:
        r"""Requesting more atoms than available raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Requesting more
        atoms than available raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lines: list[str] = [
            "0.0 0.0 0.0",
        ]
        lattice: Float[Array, "..."] = jnp.eye(3) * 4.0
        with pytest.raises(ValueError, match="file ends"):
            _parse_poscar_positions(
                lines,
                start_idx=0,
                n_atoms=2,
                is_cartesian=False,
                lattice=lattice,
            )


class TestParsePoscar(chex.TestCase):
    """Test complete POSCAR file parsing.

    :see: :func:`~rheedium.inout.parse_poscar`
    """

    def test_simple_cubic_si(self) -> None:
        r"""Parse simple cubic Si POSCAR.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse simple cubic
        Si POSCAR.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Simple cubic Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Direct
  0.0  0.0  0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

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
        r"""Parse MgO rock salt structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse MgO rock
        salt structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """MgO rock salt
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            assert crystal.frac_positions.shape == (2, 4)
            # Mg=12, O=8
            chex.assert_trees_all_close(
                crystal.frac_positions[:, 3],
                jnp.array([12.0, 8.0]),
                atol=1e-10,
            )

    def test_cartesian_mode(self) -> None:
        r"""Parse POSCAR with Cartesian coordinates.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse POSCAR with
        Cartesian coordinates.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Cartesian Si
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            # 2.715 / 5.43 = 0.5
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                atol=1e-3,
            )

    def test_negative_scale_target_volume(self) -> None:
        r"""Negative scale POSCAR resolves to the target-volume cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a POSCAR with
        universal scale -160.103007 and a unit lattice parses to a cubic
        5.43 Angstrom cell. The expected literals [5.43, 5.43, 5.43] are
        the ASE ``read(format="vasp")`` cell lengths for the identical
        file, cross-checking the VASP negative-scale (target volume)
        convention.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Si negative volume scale
-160.103007
  1.0 0.0 0.0
  0.0 1.0 0.0
  0.0 0.0 1.0
  Si
  1
Direct
  0.5 0.5 0.5
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            # ASE ground truth: cubic 5.43 A cell (160.103007 ** (1/3))
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.43, 5.43, 5.43]),
                atol=1e-8,
            )
            chex.assert_trees_all_close(
                crystal.cell_angles,
                jnp.array([90.0, 90.0, 90.0]),
                atol=1e-8,
            )
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.5, 0.5, 0.5]]),
                atol=1e-10,
            )

    def test_cartesian_mode_with_scale(self) -> None:
        r"""Scaled Cartesian POSCAR positions convert to correct fractions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a POSCAR with
        universal scale 2.0, raw lattice 2.715 Angstrom cubic, and a
        Cartesian coordinate line (1.3575, 1.3575, 1.3575) parses to
        fractional coordinates (0.5, 0.5, 0.5) in a 5.43 Angstrom cell.
        The expected literals match ASE ``read(format="vasp")`` for the
        identical file.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Si cartesian scaled
2.0
  2.715 0.0 0.0
  0.0 2.715 0.0
  0.0 0.0 2.715
  Si
  1
Cartesian
  1.3575 1.3575 1.3575
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            # ASE ground truth: 5.43 A cell, frac (0.5, 0.5, 0.5)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.43, 5.43, 5.43]),
                atol=1e-10,
            )
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.5, 0.5, 0.5]]),
                atol=1e-10,
            )

    def test_selective_dynamics(self) -> None:
        r"""Parse POSCAR with selective dynamics.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse POSCAR with
        selective dynamics.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Selective dynamics Si
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            assert crystal.frac_positions.shape == (2, 4)
            chex.assert_trees_all_close(
                crystal.frac_positions[:, :3],
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                atol=1e-10,
            )

    def test_scaling_factor(self) -> None:
        r"""Scaling factor applied correctly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Scaling factor
        applied correctly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Scaled Si
2.0
  2.715  0.00  0.00
  0.00  2.715  0.00
  0.00  0.00  2.715
  Si
  1
Direct
  0.0  0.0  0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            # 2.715 * 2.0 = 5.43
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([5.43, 5.43, 5.43]),
                atol=1e-3,
            )

    def test_non_orthogonal_cell(self) -> None:
        r"""Parse POSCAR with non-orthogonal (hexagonal) cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse POSCAR with
        non-orthogonal (hexagonal) cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 3.0
        c: float = 5.0
        # Hexagonal lattice vectors
        poscar_content: Any = f"""Hexagonal ZnO
1.0
  {a}  0.0  0.0
  {-a / 2}  {a * 0.866025}  0.0
  0.0  0.0  {c}
  Zn O
  1 1
Direct
  0.0  0.0  0.0
  0.333333  0.666667  0.5
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

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
        r"""Missing file raises FileNotFoundError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing file
        raises FileNotFoundError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_poscar("/nonexistent/path/POSCAR")

    def test_non_utf8_file_raises_runtime_error(self) -> None:
        r"""A present but non-UTF-8 POSCAR file raises RuntimeError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A present but
        non-UTF-8 POSCAR file raises RuntimeError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")

            with pytest.raises(RuntimeError, match="Failed to read POSCAR"):
                parse_poscar(poscar_file)

    def test_too_few_lines(self) -> None:
        r"""File with too few lines raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: File with too few
        lines raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Short file
1.0
5.43 0.0 0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            with pytest.raises(ValueError, match="at least"):
                parse_poscar(poscar_file)

    def test_invalid_coordinate_mode(self) -> None:
        r"""Invalid coordinate mode raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid coordinate
        mode raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Invalid mode
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Invalid
  0.0  0.0  0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            with pytest.raises(ValueError, match="Direct.*Cartesian"):
                parse_poscar(poscar_file)

    def test_string_path(self) -> None:
        r"""Accept string path as well as Path object.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Accept string path
        as well as Path object.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Si
1.0
  5.43  0.00  0.00
  0.00  5.43  0.00
  0.00  0.00  5.43
  Si
  1
Direct
  0.0  0.0  0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            # Pass as string instead of Path
            crystal: CrystalStructure = parse_poscar(str(poscar_file))
            assert isinstance(crystal, CrystalStructure)

    @parameterized.named_parameters(
        ("silicon", "Si", 14),
        ("magnesium", "Mg", 12),
        ("oxygen", "O", 8),
        ("iron", "Fe", 26),
        ("gold", "Au", 79),
    )
    def test_various_elements(self, element: str, expected_z: int) -> None:
        r"""Test parsing various element types.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: parsing various
        element types.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``element``,
        ``expected_z``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: Any = f"""Element test
1.0
  4.0  0.0  0.0
  0.0  4.0  0.0
  0.0  0.0  4.0
  {element}
  1
Direct
  0.0  0.0  0.0
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)
            chex.assert_trees_all_close(
                crystal.frac_positions[0, 3],
                float(expected_z),
                atol=1e-10,
            )


class TestPoscarRoundtrip(chex.TestCase):
    """Test POSCAR parsing consistency."""

    def test_frac_cart_consistency(self) -> None:
        r"""Fractional and Cartesian positions should be consistent.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fractional and
        Cartesian positions should be consistent.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """MgO
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

            expected_cart: Float[Array, "..."] = crystal.frac_positions[
                :, :3
            ] @ jnp.diag(crystal.cell_lengths)
            chex.assert_trees_all_close(
                crystal.cart_positions[:, :3],
                expected_cart,
                atol=1e-6,
            )

    def test_atomic_numbers_preserved(self) -> None:
        r"""Atomic numbers match in both frac and cart positions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Atomic numbers
        match in both frac and cart positions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_poscar``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        poscar_content: str = """Multi-element
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
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            poscar_file: Path = Path(tmp_dir) / "POSCAR"
            poscar_file.write_text(poscar_content)

            crystal: CrystalStructure = parse_poscar(poscar_file)

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
