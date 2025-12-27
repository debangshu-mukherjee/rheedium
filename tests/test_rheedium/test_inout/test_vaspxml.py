"""Tests for VASP vasprun.xml file parsing."""

import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest

from rheedium.inout.vaspxml import (
    _extract_energy,
    _extract_forces,
    _extract_stress,
    _extract_structure_block,
    _get_species_list,
    parse_vaspxml,
    parse_vaspxml_trajectory,
)
from rheedium.types import CrystalStructure, XYZData
import xml.etree.ElementTree as ET


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
    """Test species list extraction from atominfo."""

    def test_simple_species(self) -> None:
        """Extract species from simple atominfo."""
        root = ET.fromstring(SIMPLE_VASPXML)
        species = _get_species_list(root)

        assert species == ["Mg", "O"]

    def test_missing_atominfo(self) -> None:
        """Missing atominfo raises ValueError."""
        xml_content = """<modeling></modeling>"""
        root = ET.fromstring(xml_content)

        with pytest.raises(ValueError, match="atominfo"):
            _get_species_list(root)


class TestExtractStructureBlock(chex.TestCase):
    """Test structure block extraction."""

    def test_extract_lattice_positions(self) -> None:
        """Extract lattice and positions from structure element."""
        root = ET.fromstring(SIMPLE_VASPXML)
        structure = root.find(".//structure[@name='initialpos']")
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
    """Test forces extraction."""

    def test_extract_forces(self) -> None:
        """Extract forces from calculation element."""
        root = ET.fromstring(SIMPLE_VASPXML)
        calculation = root.find(".//calculation")
        forces = _extract_forces(calculation)

        assert forces is not None
        chex.assert_trees_all_close(
            forces,
            jnp.array([[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]]),
            atol=1e-10,
        )

    def test_missing_forces(self) -> None:
        """Missing forces returns None."""
        xml_content = """<calculation></calculation>"""
        calculation = ET.fromstring(xml_content)
        forces = _extract_forces(calculation)

        assert forces is None


class TestExtractStress(chex.TestCase):
    """Test stress tensor extraction."""

    def test_extract_stress(self) -> None:
        """Extract stress tensor from calculation element."""
        root = ET.fromstring(SIMPLE_VASPXML)
        calculation = root.find(".//calculation")
        stress = _extract_stress(calculation)

        assert stress is not None
        expected = jnp.array([
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0],
        ])
        chex.assert_trees_all_close(stress, expected, atol=1e-10)

    def test_missing_stress(self) -> None:
        """Missing stress returns None."""
        xml_content = """<calculation></calculation>"""
        calculation = ET.fromstring(xml_content)
        stress = _extract_stress(calculation)

        assert stress is None


class TestExtractEnergy(chex.TestCase):
    """Test energy extraction."""

    def test_extract_energy(self) -> None:
        """Extract energy from calculation element."""
        root = ET.fromstring(SIMPLE_VASPXML)
        calculation = root.find(".//calculation")
        energy = _extract_energy(calculation)

        assert energy is not None
        chex.assert_trees_all_close(energy, -12.34567890, atol=1e-6)

    def test_missing_energy(self) -> None:
        """Missing energy returns None."""
        xml_content = """<calculation></calculation>"""
        calculation = ET.fromstring(xml_content)
        energy = _extract_energy(calculation)

        assert energy is None


class TestParseVaspxml(chex.TestCase):
    """Test complete vasprun.xml parsing."""

    def test_parse_crystal_structure(self) -> None:
        """Parse vasprun.xml to CrystalStructure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal = parse_vaspxml(xml_file)

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
        """Parse vasprun.xml to XYZData with forces."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            xyz_data = parse_vaspxml(xml_file, include_forces=True)

            assert isinstance(xyz_data, XYZData)
            assert xyz_data.positions.shape == (2, 3)
            assert xyz_data.energy is not None
            chex.assert_trees_all_close(xyz_data.energy, -12.34567890, atol=1e-6)
            assert xyz_data.stress is not None
            assert xyz_data.lattice is not None

    def test_parse_specific_step(self) -> None:
        """Parse specific ionic step."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            # Parse step 0
            xyz_0 = parse_vaspxml(xml_file, step=0, include_forces=True)
            chex.assert_trees_all_close(xyz_0.energy, -10.0, atol=1e-6)

            # Parse step 1
            xyz_1 = parse_vaspxml(xml_file, step=1, include_forces=True)
            chex.assert_trees_all_close(xyz_1.energy, -11.0, atol=1e-6)

            # Parse last step (step=-1)
            xyz_last = parse_vaspxml(xml_file, step=-1, include_forces=True)
            chex.assert_trees_all_close(xyz_last.energy, -12.0, atol=1e-6)

    def test_step_out_of_range(self) -> None:
        """Out of range step raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            with pytest.raises(ValueError, match="out of range"):
                parse_vaspxml(xml_file, step=100)

    def test_file_not_found(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_vaspxml("/nonexistent/vasprun.xml")

    def test_invalid_xml(self) -> None:
        """Invalid XML raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text("not valid xml <<<<")

            with pytest.raises(ValueError, match="Invalid XML"):
                parse_vaspxml(xml_file)

    def test_string_path(self) -> None:
        """Accept string path as well as Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal = parse_vaspxml(str(xml_file))
            assert isinstance(crystal, CrystalStructure)


class TestParseVaspxmlTrajectory(chex.TestCase):
    """Test trajectory parsing from vasprun.xml."""

    def test_parse_trajectory(self) -> None:
        """Parse full trajectory from vasprun.xml."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory = parse_vaspxml_trajectory(xml_file)

            assert len(trajectory) == 3
            assert all(isinstance(xyz, XYZData) for xyz in trajectory)

    def test_trajectory_energies(self) -> None:
        """Energies are extracted for each step."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory = parse_vaspxml_trajectory(xml_file)

            energies = [xyz.energy for xyz in trajectory]
            chex.assert_trees_all_close(
                jnp.array(energies),
                jnp.array([-10.0, -11.0, -12.0]),
                atol=1e-6,
            )

    def test_trajectory_lattice_changes(self) -> None:
        """Lattice can change during relaxation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory = parse_vaspxml_trajectory(xml_file)

            # Cell shrinks during relaxation
            cell_a_0 = jnp.linalg.norm(trajectory[0].lattice[0])
            cell_a_2 = jnp.linalg.norm(trajectory[2].lattice[0])

            assert cell_a_0 > cell_a_2  # Cell shrinks

    def test_trajectory_without_forces(self) -> None:
        """Trajectory without forces has None for metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory = parse_vaspxml_trajectory(
                xml_file, include_forces=False
            )

            # Energy should be None when include_forces=False
            assert all(xyz.energy is None for xyz in trajectory)

    def test_trajectory_file_not_found(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_vaspxml_trajectory("/nonexistent/vasprun.xml")

    def test_no_calculation_steps(self) -> None:
        """XML without calculation steps raises ValueError."""
        xml_content = """<?xml version="1.0"?>
<modeling>
  <atominfo>
    <array name="atoms">
      <set><rc><c>Si</c><c>1</c></rc></set>
    </array>
  </atominfo>
</modeling>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(xml_content)

            with pytest.raises(ValueError, match="no calculation steps"):
                parse_vaspxml_trajectory(xml_file)

    def test_atomic_numbers_preserved(self) -> None:
        """Atomic numbers are consistent across trajectory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(TRAJECTORY_VASPXML)

            trajectory = parse_vaspxml_trajectory(xml_file)

            # All steps should have same atomic numbers (Si=14)
            for xyz in trajectory:
                chex.assert_trees_all_close(
                    xyz.atomic_numbers,
                    jnp.array([14, 14]),
                    atol=1e-10,
                )


class TestVaspxmlRoundtrip(chex.TestCase):
    """Test vasprun.xml parsing consistency."""

    def test_frac_cart_consistency(self) -> None:
        """Fractional and Cartesian positions should be consistent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal = parse_vaspxml(xml_file)

            # cart = frac @ lattice
            lattice = jnp.diag(crystal.cell_lengths)
            expected_cart = crystal.frac_positions[:, :3] @ lattice
            chex.assert_trees_all_close(
                crystal.cart_positions[:, :3],
                expected_cart,
                atol=1e-6,
            )

    def test_xyz_lattice_matches_crystal(self) -> None:
        """XYZData lattice should match CrystalStructure cell."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_file = Path(tmp_dir) / "vasprun.xml"
            xml_file.write_text(SIMPLE_VASPXML)

            crystal = parse_vaspxml(xml_file, include_forces=False)
            xyz_data = parse_vaspxml(xml_file, include_forces=True)

            # Lattice from XYZData should give same cell lengths
            from rheedium.inout.crystal import lattice_to_cell_params
            lengths, angles = lattice_to_cell_params(xyz_data.lattice)

            chex.assert_trees_all_close(
                lengths, crystal.cell_lengths, atol=1e-6
            )
            chex.assert_trees_all_close(
                angles, crystal.cell_angles, atol=1e-6
            )
