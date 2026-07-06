"""Tests for VASP vasprun.xml file parsing."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from rheedium.inout.vaspxml import (
    _extract_energy,
    _extract_forces,
    _extract_stress,
    _extract_structure_block,
    _get_species_list,
    parse_vaspxml,
    parse_vaspxml_trajectory,
)
from rheedium.types.crystal_types import CrystalStructure, XYZData

# Sample vasprun.xml content for testing
SIMPLE_VASPXML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
  <atominfo>
    <atoms>2</atoms>
    <array name="atoms">
      <set>
        <rc><c>Mg</c><c>1</c></rc>
        <rc><c>O </c><c>2</c></rc>
      </set>
    </array>
  </atominfo>
  <structure name="initialpos">
    <crystal>
      <varray name="basis">
        <v>4.21 0.0 0.0</v>
        <v>0.0 4.21 0.0</v>
        <v>0.0 0.0 4.21</v>
      </varray>
    </crystal>
    <varray name="positions">
      <v>0.0 0.0 0.0</v>
      <v>0.5 0.5 0.5</v>
    </varray>
  </structure>
  <calculation>
    <structure>
      <crystal>
        <varray name="basis">
          <v>4.21 0.0 0.0</v>
          <v>0.0 4.21 0.0</v>
          <v>0.0 0.0 4.21</v>
        </varray>
      </crystal>
      <varray name="positions">
        <v>0.0 0.0 0.0</v>
        <v>0.5 0.5 0.5</v>
      </varray>
    </structure>
    <varray name="forces">
      <v>0.001 0.002 0.003</v>
      <v>-0.001 -0.002 -0.003</v>
    </varray>
    <varray name="stress">
      <v>1.0 0.1 0.2</v>
      <v>0.1 2.0 0.3</v>
      <v>0.2 0.3 3.0</v>
    </varray>
    <energy>
      <i name="e_fr_energy">-12.34567890</i>
      <i name="e_0_energy">-12.34500000</i>
    </energy>
  </calculation>
</modeling>
"""

TRAJECTORY_VASPXML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
  <atominfo>
    <atoms>2</atoms>
    <array name="atoms">
      <set>
        <rc><c>Si</c><c>1</c></rc>
        <rc><c>Si</c><c>1</c></rc>
      </set>
    </array>
  </atominfo>
  <calculation>
    <structure>
      <crystal>
        <varray name="basis">
          <v>5.43 0.0 0.0</v>
          <v>0.0 5.43 0.0</v>
          <v>0.0 0.0 5.43</v>
        </varray>
      </crystal>
      <varray name="positions">
        <v>0.0 0.0 0.0</v>
        <v>0.25 0.25 0.25</v>
      </varray>
    </structure>
    <energy>
      <i name="e_fr_energy">-10.0</i>
    </energy>
  </calculation>
  <calculation>
    <structure>
      <crystal>
        <varray name="basis">
          <v>5.40 0.0 0.0</v>
          <v>0.0 5.40 0.0</v>
          <v>0.0 0.0 5.40</v>
        </varray>
      </crystal>
      <varray name="positions">
        <v>0.0 0.0 0.0</v>
        <v>0.25 0.25 0.25</v>
      </varray>
    </structure>
    <energy>
      <i name="e_fr_energy">-11.0</i>
    </energy>
  </calculation>
  <calculation>
    <structure>
      <crystal>
        <varray name="basis">
          <v>5.38 0.0 0.0</v>
          <v>0.0 5.38 0.0</v>
          <v>0.0 0.0 5.38</v>
        </varray>
      </crystal>
      <varray name="positions">
        <v>0.0 0.0 0.0</v>
        <v>0.25 0.25 0.25</v>
      </varray>
    </structure>
    <energy>
      <i name="e_fr_energy">-12.0</i>
    </energy>
  </calculation>
</modeling>
"""


class TestGetSpeciesList(chex.TestCase):
    """Test species list extraction from atominfo.

    :see: :func:`~rheedium.inout.vaspxml._get_species_list`
    """

    def test_simple_species(self) -> None:
        r"""Extract species from simple atominfo.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract species
        from simple atominfo.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        root: Any = ET.fromstring(SIMPLE_VASPXML)
        species: Any = _get_species_list(root)

        assert species == ["Mg", "O"]

    def test_missing_atominfo(self) -> None:
        r"""Missing atominfo raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing atominfo
        raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xml_content: str = """<modeling></modeling>"""
        root: Any = ET.fromstring(xml_content)

        with pytest.raises(ValueError, match="atominfo"):
            _get_species_list(root)


class TestExtractStructureBlock(chex.TestCase):
    """Test structure block extraction.

    :see: :func:`~rheedium.inout.vaspxml._extract_structure_block`
    """

    def test_extract_lattice_positions(self) -> None:
        r"""Extract lattice and positions from structure element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract lattice
        and positions from structure element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        root: Any = ET.fromstring(SIMPLE_VASPXML)
        structure: Any = root.find(".//structure[@name='initialpos']")
        lattice: Any
        positions: Float[Array, "..."]
        lattice, positions = _extract_structure_block(structure)

        chex.assert_trees_all_close(
            lattice,
            jnp.array([[4.21, 0.0, 0.0], [0.0, 4.21, 0.0], [0.0, 0.0, 4.21]]),
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            positions,
            jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            atol=1e-10,
        )


class TestExtractForces(chex.TestCase):
    """Test forces extraction.

    :see: :func:`~rheedium.inout.vaspxml._extract_forces`
    """

    def test_extract_forces(self) -> None:
        r"""Extract forces from calculation element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract forces
        from calculation element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        root: Any = ET.fromstring(SIMPLE_VASPXML)
        calculation: Any = root.find(".//calculation")
        forces: Any = _extract_forces(calculation)

        assert forces is not None
        chex.assert_trees_all_close(
            forces,
            jnp.array([[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]]),
            atol=1e-10,
        )

    def test_missing_forces(self) -> None:
        r"""Missing forces returns None.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing forces
        returns None.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xml_content: str = """<calculation></calculation>"""
        calculation: Any = ET.fromstring(xml_content)
        forces: Any = _extract_forces(calculation)

        assert forces is None


class TestExtractStress(chex.TestCase):
    """Test stress tensor extraction.

    :see: :func:`~rheedium.inout.vaspxml._extract_stress`
    """

    def test_extract_stress(self) -> None:
        r"""Extract stress tensor from calculation element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract stress
        tensor from calculation element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        root: Any = ET.fromstring(SIMPLE_VASPXML)
        calculation: Any = root.find(".//calculation")
        stress: Any = _extract_stress(calculation)

        assert stress is not None
        expected: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.3],
                [0.2, 0.3, 3.0],
            ]
        )
        chex.assert_trees_all_close(stress, expected, atol=1e-10)

    def test_missing_stress(self) -> None:
        r"""Missing stress returns None.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing stress
        returns None.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xml_content: str = """<calculation></calculation>"""
        calculation: Any = ET.fromstring(xml_content)
        stress: Any = _extract_stress(calculation)

        assert stress is None


class TestExtractEnergy(chex.TestCase):
    """Test energy extraction.

    :see: :func:`~rheedium.inout.vaspxml._extract_energy`
    """

    def test_extract_energy(self) -> None:
        r"""Extract energy from calculation element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Extract energy
        from calculation element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        root: Any = ET.fromstring(SIMPLE_VASPXML)
        calculation: Any = root.find(".//calculation")
        energy: Any = _extract_energy(calculation)

        assert energy is not None
        chex.assert_trees_all_close(energy, -12.34567890, atol=1e-6)

    def test_missing_energy(self) -> None:
        r"""Missing energy returns None.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Missing energy
        returns None.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xml_content: str = """<calculation></calculation>"""
        calculation: Any = ET.fromstring(xml_content)
        energy: Any = _extract_energy(calculation)

        assert energy is None


class TestParseVaspxml(chex.TestCase):
    """Test complete vasprun.xml parsing.

    :see: :func:`~rheedium.inout.parse_vaspxml`
    """

    def test_parse_crystal_structure(self) -> None:
        r"""Parse vasprun.xml to CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse vasprun.xml
        to CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal: CrystalStructure = parse_vaspxml(xml_file)

            assert isinstance(crystal, CrystalStructure)
            chex.assert_trees_all_close(
                crystal.cell_lengths,
                jnp.array([4.21, 4.21, 4.21]),
                atol=1e-3,
            )
            chex.assert_trees_all_close(
                crystal.cell_angles,
                jnp.array([90.0, 90.0, 90.0]),
                atol=1e-3,
            )
            assert crystal.frac_positions.shape == (2, 4)
            # Mg=12, O=8
            chex.assert_trees_all_close(
                crystal.frac_positions[:, 3],
                jnp.array([12.0, 8.0]),
                atol=1e-10,
            )

    def test_parse_with_forces(self) -> None:
        r"""Parse vasprun.xml to XYZData with forces.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse vasprun.xml
        to XYZData with forces.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            xyz_data: Any = parse_vaspxml(xml_file, include_forces=True)

            assert isinstance(xyz_data, XYZData)
            assert xyz_data.positions.shape == (2, 3)
            assert xyz_data.energy is not None
            chex.assert_trees_all_close(
                xyz_data.energy, -12.34567890, atol=1e-6
            )
            assert xyz_data.stress is not None
            assert xyz_data.lattice is not None
            assert xyz_data.forces is not None
            chex.assert_trees_all_close(
                xyz_data.forces,
                jnp.array([[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]]),
                atol=1e-12,
            )
            assert xyz_data.properties == [
                {"name": "forces", "type": "R", "count": 3}
            ]

    def test_forces_absent_when_not_requested(self) -> None:
        r"""Trajectory parsed with include_forces=False stores no forces.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: when forces
        are not requested, ``XYZData.forces`` is None and the properties
        metadata does not advertise a forces column, even though the
        vasprun.xml fixture contains a forces varray.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(
                xml_file, include_forces=False
            )

            assert all(xyz.forces is None for xyz in trajectory)
            assert all(xyz.properties is None for xyz in trajectory)

    def test_trajectory_forces_populated(self) -> None:
        r"""Trajectory parsed with include_forces=True stores force values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the trajectory
        variant populates ``XYZData.forces`` with the exact values from the
        vasprun.xml forces varray, and the properties metadata advertises
        the forces column only because forces are actually present.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(
                xml_file, include_forces=True
            )

            assert len(trajectory) == 1
            assert trajectory[0].forces is not None
            chex.assert_trees_all_close(
                trajectory[0].forces,
                jnp.array([[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]]),
                atol=1e-12,
            )
            assert trajectory[0].properties == [
                {"name": "forces", "type": "R", "count": 3}
            ]

    def test_parse_specific_step(self) -> None:
        r"""Parse specific ionic step.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse specific
        ionic step.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            # Parse step 0
            xyz_0: Any = parse_vaspxml(xml_file, step=0, include_forces=True)
            chex.assert_trees_all_close(xyz_0.energy, -10.0, atol=1e-6)

            # Parse step 1
            xyz_1: Any = parse_vaspxml(xml_file, step=1, include_forces=True)
            chex.assert_trees_all_close(xyz_1.energy, -11.0, atol=1e-6)

            # Parse last step (step=-1)
            xyz_last: Any = parse_vaspxml(
                xml_file, step=-1, include_forces=True
            )
            chex.assert_trees_all_close(xyz_last.energy, -12.0, atol=1e-6)

    def test_step_out_of_range(self) -> None:
        r"""Out of range step raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Out of range step
        raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            with pytest.raises(ValueError, match="out of range"):
                parse_vaspxml(xml_file, step=100)

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
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(FileNotFoundError):
            parse_vaspxml("/nonexistent/vasprun.xml")

    def test_invalid_xml(self) -> None:
        r"""Invalid XML raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid XML raises
        ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text("not valid xml <<<<")

            with pytest.raises(ValueError, match="Invalid XML"):
                parse_vaspxml(xml_file)

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
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal: CrystalStructure = parse_vaspxml(str(xml_file))
            assert isinstance(crystal, CrystalStructure)


class TestParseVaspxmlTrajectory(chex.TestCase):
    """Test trajectory parsing from vasprun.xml.

    :see: :func:`~rheedium.inout.parse_vaspxml_trajectory`
    """

    def test_parse_trajectory(self) -> None:
        r"""Parse full trajectory from vasprun.xml.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Parse full
        trajectory from vasprun.xml.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(xml_file)

            assert len(trajectory) == 3
            assert all(isinstance(xyz, XYZData) for xyz in trajectory)

    def test_trajectory_energies(self) -> None:
        r"""Energies are extracted for each step.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Energies are
        extracted for each step.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(xml_file)

            energies: Any = [xyz.energy for xyz in trajectory]
            chex.assert_trees_all_close(
                jnp.array(energies),
                jnp.array([-10.0, -11.0, -12.0]),
                atol=1e-6,
            )

    def test_trajectory_lattice_changes(self) -> None:
        r"""Lattice can change during relaxation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Lattice can change
        during relaxation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(xml_file)

            # Cell shrinks during relaxation
            lattice_0: Any = trajectory[0].lattice
            lattice_2: Any = trajectory[2].lattice
            assert lattice_0 is not None
            assert lattice_2 is not None
            cell_a_0: Float[Array, "..."] = jnp.linalg.norm(lattice_0[0])
            cell_a_2: Float[Array, "..."] = jnp.linalg.norm(lattice_2[0])

            assert cell_a_0 > cell_a_2  # Cell shrinks

    def test_trajectory_without_forces(self) -> None:
        r"""Trajectory without forces has None for metadata.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Trajectory without
        forces has None for metadata.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(
                xml_file, include_forces=False
            )

            # Energy and forces should be None when include_forces=False
            assert all(xyz.energy is None for xyz in trajectory)
            assert all(xyz.forces is None for xyz in trajectory)

    def test_trajectory_file_not_found(self) -> None:
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
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(FileNotFoundError):
            parse_vaspxml_trajectory("/nonexistent/vasprun.xml")

    def test_no_calculation_steps(self) -> None:
        r"""XML without calculation steps raises ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XML without
        calculation steps raises ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        xml_content: str = """<?xml version="1.0"?>
<modeling>
  <atominfo>
    <array name="atoms">
      <set><rc><c>Si</c><c>1</c></rc></set>
    </array>
  </atominfo>
</modeling>
"""
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(xml_content)

            with pytest.raises(ValueError, match="no calculation steps"):
                parse_vaspxml_trajectory(xml_file)

    def test_atomic_numbers_preserved(self) -> None:
        r"""Atomic numbers are consistent across trajectory.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Atomic numbers are
        consistent across trajectory.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory: Any = parse_vaspxml_trajectory(xml_file)

            # All steps should have same atomic numbers (Si=14)
            xyz: Any
            for xyz in trajectory:
                chex.assert_trees_all_close(
                    xyz.atomic_numbers,
                    jnp.array([14, 14]),
                    atol=1e-10,
                )


class TestVaspxmlRoundtrip(chex.TestCase):
    """Test vasprun.xml parsing consistency."""

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
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal: CrystalStructure = parse_vaspxml(xml_file)

            lattice: Float[Array, "..."] = jnp.diag(crystal.cell_lengths)
            expected_cart: Float[Array, "..."] = (
                crystal.frac_positions[:, :3] @ lattice
            )
            chex.assert_trees_all_close(
                crystal.cart_positions[:, :3],
                expected_cart,
                atol=1e-6,
            )

    def test_xyz_lattice_matches_crystal(self) -> None:
        r"""XYZData lattice should match CrystalStructure cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XYZData lattice
        should match CrystalStructure cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_vaspxml``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file: Path = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal: CrystalStructure = parse_vaspxml(
                xml_file, include_forces=False
            )
            xyz_data: Any = parse_vaspxml(xml_file, include_forces=True)

            # Lattice from XYZData should give same cell lengths
            from rheedium.inout import lattice_to_cell_params

            lengths: Float[Array, "..."]
            angles: Float[Array, "..."]
            lengths, angles = lattice_to_cell_params(xyz_data.lattice)

            chex.assert_trees_all_close(
                lengths, crystal.cell_lengths, atol=1e-6
            )
            chex.assert_trees_all_close(angles, crystal.cell_angles, atol=1e-6)
