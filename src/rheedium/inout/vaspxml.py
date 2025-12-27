"""VASP vasprun.xml parsing utilities.

Extended Summary
----------------
This module provides utilities for parsing VASP vasprun.xml files, which
contain detailed output from VASP calculations including crystal structures,
forces, energies, and stress tensors. Supports both single structure
extraction and full MD/relaxation trajectory parsing.

Routine Listings
----------------
parse_vaspxml : function
    Parse single structure from vasprun.xml with optional metadata.
parse_vaspxml_trajectory : function
    Parse full MD/relaxation trajectory from vasprun.xml.
_extract_structure_block : function, internal
    Extract lattice, positions, and species from XML structure element.
_extract_forces : function, internal
    Extract forces array from XML element.
_extract_stress : function, internal
    Extract stress tensor from XML element.
_extract_energy : function, internal
    Extract total energy from XML element.
_get_species_list : function, internal
    Get element species list from atominfo section.

Notes
-----
Internal functions prefixed with underscore are not part of the public API.
All returned data structures are JAX-compatible arrays.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    XYZData,
    create_crystal_structure,
    create_xyz_data,
)

from .crystal import lattice_to_cell_params
from .xyz import atomic_symbol


@beartype
def _get_species_list(root: ET.Element) -> List[str]:
    """Get element species list from atominfo section.

    Parses the atominfo/array section of vasprun.xml to extract the
    element symbol for each atom in order.

    Parameters
    ----------
    root : ET.Element
        Root element of the parsed vasprun.xml.

    Returns
    -------
    species : List[str]
        Element symbols for each atom in order.

    Raises
    ------
    ValueError
        If atominfo section is missing or malformed.
    """
    atominfo = root.find(".//atominfo")
    if atominfo is None:
        raise ValueError("Invalid vasprun.xml: missing <atominfo> section")

    atoms_array = atominfo.find(".//array[@name='atoms']")
    if atoms_array is None:
        raise ValueError(
            "Invalid vasprun.xml: missing <array name='atoms'> in atominfo"
        )

    species: List[str] = []
    for rc in atoms_array.findall(".//rc"):
        c_elements = rc.findall("c")
        if c_elements:
            element = c_elements[0].text
            if element is not None:
                species.append(element.strip())

    if not species:
        raise ValueError("Invalid vasprun.xml: no atoms found in atominfo")

    return species


@beartype
def _extract_structure_block(
    structure_elem: ET.Element,
) -> Tuple[Float[Array, "3 3"], Float[Array, "n_atoms 3"]]:
    """Extract lattice and positions from XML structure element.

    Parses a <structure> element from vasprun.xml to extract the
    lattice vectors and fractional atomic positions.

    Parameters
    ----------
    structure_elem : ET.Element
        A <structure> element from vasprun.xml.

    Returns
    -------
    lattice : Float[Array, "3 3"]
        Lattice vectors as rows in Angstroms.
    positions : Float[Array, "n_atoms 3"]
        Fractional atomic positions.

    Raises
    ------
    ValueError
        If required elements are missing or malformed.
    """
    crystal = structure_elem.find("crystal")
    if crystal is None:
        raise ValueError("Invalid structure: missing <crystal> element")

    basis = crystal.find(".//varray[@name='basis']")
    if basis is None:
        raise ValueError("Invalid structure: missing lattice basis")

    lattice_rows: List[List[float]] = []
    for v in basis.findall("v"):
        if v.text is not None:
            row = [float(x) for x in v.text.split()]
            lattice_rows.append(row)

    if len(lattice_rows) != 3:
        raise ValueError(
            f"Invalid lattice: expected 3 vectors, got {len(lattice_rows)}"
        )

    lattice: Float[Array, "3 3"] = jnp.array(lattice_rows, dtype=jnp.float64)

    positions_elem = structure_elem.find(".//varray[@name='positions']")
    if positions_elem is None:
        raise ValueError("Invalid structure: missing positions")

    position_rows: List[List[float]] = []
    for v in positions_elem.findall("v"):
        if v.text is not None:
            row = [float(x) for x in v.text.split()]
            position_rows.append(row)

    positions: Float[Array, "n_atoms 3"] = jnp.array(
        position_rows, dtype=jnp.float64
    )

    return lattice, positions


@beartype
def _extract_forces(
    calculation_elem: ET.Element,
) -> Optional[Float[Array, "n_atoms 3"]]:
    """Extract forces array from calculation element.

    Parses the forces varray from a <calculation> element.

    Parameters
    ----------
    calculation_elem : ET.Element
        A <calculation> element from vasprun.xml.

    Returns
    -------
    forces : Optional[Float[Array, "n_atoms 3"]]
        Forces in eV/Angstrom, or None if not found.
    """
    forces_elem = calculation_elem.find(".//varray[@name='forces']")
    if forces_elem is None:
        return None

    force_rows: List[List[float]] = []
    for v in forces_elem.findall("v"):
        if v.text is not None:
            row = [float(x) for x in v.text.split()]
            force_rows.append(row)

    if not force_rows:
        return None

    forces: Float[Array, "n_atoms 3"] = jnp.array(force_rows, dtype=jnp.float64)
    return forces


@beartype
def _extract_stress(
    calculation_elem: ET.Element,
) -> Optional[Float[Array, "3 3"]]:
    """Extract stress tensor from calculation element.

    Parses the stress varray from a <calculation> element.

    Parameters
    ----------
    calculation_elem : ET.Element
        A <calculation> element from vasprun.xml.

    Returns
    -------
    stress : Optional[Float[Array, "3 3"]]
        Stress tensor in kBar, or None if not found.
    """
    stress_elem = calculation_elem.find(".//varray[@name='stress']")
    if stress_elem is None:
        return None

    stress_rows: List[List[float]] = []
    for v in stress_elem.findall("v"):
        if v.text is not None:
            row = [float(x) for x in v.text.split()]
            stress_rows.append(row)

    if len(stress_rows) != 3:
        return None

    stress: Float[Array, "3 3"] = jnp.array(stress_rows, dtype=jnp.float64)
    return stress


@beartype
def _extract_energy(
    calculation_elem: ET.Element,
) -> Optional[float]:
    """Extract total energy from calculation element.

    Parses the energy section from a <calculation> element, looking
    for the free energy (e_fr_energy).

    Parameters
    ----------
    calculation_elem : ET.Element
        A <calculation> element from vasprun.xml.

    Returns
    -------
    energy : Optional[float]
        Total energy in eV, or None if not found.
    """
    energy_elem = calculation_elem.find(".//energy")
    if energy_elem is None:
        return None

    e_fr = energy_elem.find(".//i[@name='e_fr_energy']")
    if e_fr is not None and e_fr.text is not None:
        return float(e_fr.text.strip())

    e_0 = energy_elem.find(".//i[@name='e_0_energy']")
    if e_0 is not None and e_0.text is not None:
        return float(e_0.text.strip())

    return None


@jaxtyped(typechecker=beartype)
def parse_vaspxml(
    file_path: Union[str, Path],
    step: int = -1,
    include_forces: bool = False,
) -> Union[CrystalStructure, XYZData]:
    """Parse single structure from vasprun.xml.

    Reads a VASP vasprun.xml file and extracts a single structure,
    optionally including forces, energy, and stress metadata.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to vasprun.xml file.
    step : int, optional
        Ionic step to extract. Use -1 for final structure, 0 for initial,
        or any positive integer for a specific step. Default: -1
    include_forces : bool, optional
        If True, return XYZData with forces/energy/stress metadata.
        If False, return CrystalStructure only. Default: False

    Returns
    -------
    structure : Union[CrystalStructure, XYZData]
        Parsed structure. Returns XYZData if include_forces=True,
        otherwise returns CrystalStructure.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If XML parsing fails, step index is out of range, or required
        elements are missing.

    Notes
    -----
    The function extracts:

    - Lattice vectors from <structure><crystal><varray name="basis">
    - Positions from <structure><varray name="positions">
    - Species from <atominfo><array name="atoms">
    - Energy from <calculation><energy><i name="e_fr_energy">
    - Forces from <calculation><varray name="forces">
    - Stress from <calculation><varray name="stress">

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_vaspxml("vasprun.xml")
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)

    >>> xyz_data = rh.inout.parse_vaspxml("vasprun.xml", include_forces=True)
    >>> xyz_data.energy
    -123.456
    """
    path: Path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"vasprun.xml file not found: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as err:
        raise ValueError(f"Invalid XML in vasprun.xml: {err}") from err

    species_list: List[str] = _get_species_list(root)
    n_atoms: int = len(species_list)

    atomic_numbers_list: List[int] = [atomic_symbol(sp) for sp in species_list]
    atomic_numbers: Int[Array, "n_atoms"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )

    calculations = root.findall(".//calculation")

    if not calculations:
        structure = root.find(".//structure[@name='initialpos']")
        if structure is None:
            structure = root.find(".//structure")
        if structure is None:
            raise ValueError("Invalid vasprun.xml: no structure found")

        lattice, frac_positions = _extract_structure_block(structure)
        energy = None
        forces = None
        stress = None
    else:
        if step == -1:
            calc_idx = len(calculations) - 1
        elif step < 0:
            calc_idx = len(calculations) + step
        else:
            calc_idx = step

        if calc_idx < 0 or calc_idx >= len(calculations):
            raise ValueError(
                f"Step {step} out of range. Available steps: 0-{len(calculations)-1}"
            )

        calculation = calculations[calc_idx]
        structure = calculation.find("structure")
        if structure is None:
            raise ValueError(
                f"Invalid vasprun.xml: no structure in calculation step {calc_idx}"
            )

        lattice, frac_positions = _extract_structure_block(structure)
        energy = _extract_energy(calculation)
        forces = _extract_forces(calculation)
        stress = _extract_stress(calculation)

    cart_positions: Float[Array, "n_atoms 3"] = frac_positions @ lattice

    if include_forces:
        properties = None
        if forces is not None:
            properties = [
                {"name": "forces", "type": "R", "count": 3}
            ]

        return create_xyz_data(
            positions=cart_positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=properties,
            comment=f"From vasprun.xml step {step}",
        )
    else:
        cell_lengths, cell_angles = lattice_to_cell_params(lattice)

        frac_positions_4: Float[Array, "n_atoms 4"] = jnp.column_stack(
            [frac_positions, atomic_numbers.astype(jnp.float64)]
        )
        cart_positions_4: Float[Array, "n_atoms 4"] = jnp.column_stack(
            [cart_positions, atomic_numbers.astype(jnp.float64)]
        )

        return create_crystal_structure(
            frac_positions=frac_positions_4,
            cart_positions=cart_positions_4,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )


@jaxtyped(typechecker=beartype)
def parse_vaspxml_trajectory(
    file_path: Union[str, Path],
    include_forces: bool = True,
) -> List[XYZData]:
    """Parse full MD/relaxation trajectory from vasprun.xml.

    Reads a VASP vasprun.xml file and extracts all ionic steps as a
    trajectory, with optional forces, energy, and stress per step.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to vasprun.xml file.
    include_forces : bool, optional
        Include forces, energy, and stress per step. Default: True

    Returns
    -------
    trajectory : List[XYZData]
        List of XYZData for each ionic step, preserving forces/energy/stress
        metadata when available.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If XML parsing fails or no calculation steps are found.

    Notes
    -----
    For large trajectories, memory usage scales with the number of steps.
    Each XYZData contains the full structure and optional metadata arrays.

    Examples
    --------
    >>> import rheedium as rh
    >>> trajectory = rh.inout.parse_vaspxml_trajectory("vasprun.xml")
    >>> len(trajectory)
    100
    >>> trajectory[0].energy
    -123.456
    >>> trajectory[-1].energy
    -125.789
    """
    path: Path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"vasprun.xml file not found: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as err:
        raise ValueError(f"Invalid XML in vasprun.xml: {err}") from err

    species_list: List[str] = _get_species_list(root)
    atomic_numbers_list: List[int] = [atomic_symbol(sp) for sp in species_list]
    atomic_numbers: Int[Array, "n_atoms"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )

    calculations = root.findall(".//calculation")

    if not calculations:
        raise ValueError(
            "Invalid vasprun.xml: no calculation steps found for trajectory"
        )

    trajectory: List[XYZData] = []

    for step_idx, calculation in enumerate(calculations):
        structure = calculation.find("structure")
        if structure is None:
            continue

        lattice, frac_positions = _extract_structure_block(structure)
        cart_positions: Float[Array, "n_atoms 3"] = frac_positions @ lattice

        energy: Optional[float] = None
        forces: Optional[Float[Array, "n_atoms 3"]] = None
        stress: Optional[Float[Array, "3 3"]] = None

        if include_forces:
            energy = _extract_energy(calculation)
            forces = _extract_forces(calculation)
            stress = _extract_stress(calculation)

        properties = None
        if forces is not None:
            properties = [{"name": "forces", "type": "R", "count": 3}]

        xyz_data = create_xyz_data(
            positions=cart_positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=properties,
            comment=f"vasprun.xml step {step_idx}",
        )
        trajectory.append(xyz_data)

    return trajectory
