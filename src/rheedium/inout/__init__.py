"""Data input/output utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for reading and writing various file formats
used in crystallography and RHEED simulations, including CIF files, XYZ files,
VASP POSCAR/CONTCAR files, and vasprun.xml for crystal structures. It also
provides interoperability with ASE and pymatgen libraries.

Routine Listings
----------------
:func:`atomic_symbol`
    Returns atomic number for given atomic symbol string.
:func:`from_ase`
    Convert ASE Atoms to CrystalStructure.
:func:`from_pymatgen`
    Convert pymatgen Structure to CrystalStructure.
:func:`kirkland_potentials`
    Loads Kirkland scattering factors from CSV file.
:func:`lattice_to_cell_params`
    Convert 3x3 lattice vectors to crystallographic cell parameters.
:func:`parse_cif`
    Parse a CIF file into a JAX-compatible CrystalStructure.
:func:`parse_crystal`
    Parse CIF, XYZ, or POSCAR file into simulation-ready CrystalStructure.
:func:`parse_poscar`
    Parse VASP POSCAR/CONTCAR file into CrystalStructure.
:func:`parse_vaspxml`
    Parse vasprun.xml for structure with optional metadata.
:func:`parse_vaspxml_trajectory`
    Parse full trajectory from vasprun.xml.
:func:`parse_xyz`
    Parses XYZ files and returns atoms with element symbols and 3D coordinates.
:func:`symmetry_expansion`
    Apply symmetry operations to expand fractional positions.
:func:`to_ase`
    Convert CrystalStructure to ASE Atoms.
:func:`to_pymatgen`
    Convert CrystalStructure to pymatgen Structure.
:func:`xyz_to_crystal`
    Convert XYZData to CrystalStructure for simulation.

Notes
-----
All parsing functions return JAX-compatible data structures suitable for
automatic differentiation and GPU acceleration.

Optional dependencies (ASE, pymatgen) are imported lazily and will raise
ImportError with installation instructions if not available.
"""

from .cif import parse_cif, symmetry_expansion
from .crystal import lattice_to_cell_params, parse_crystal, xyz_to_crystal
from .interop import from_ase, from_pymatgen, to_ase, to_pymatgen
from .poscar import parse_poscar
from .vaspxml import parse_vaspxml, parse_vaspxml_trajectory
from .xyz import atomic_symbol, kirkland_potentials, parse_xyz

__all__ = [
    "atomic_symbol",
    "from_ase",
    "from_pymatgen",
    "kirkland_potentials",
    "lattice_to_cell_params",
    "parse_cif",
    "parse_crystal",
    "parse_poscar",
    "parse_vaspxml",
    "parse_vaspxml_trajectory",
    "parse_xyz",
    "symmetry_expansion",
    "to_ase",
    "to_pymatgen",
    "xyz_to_crystal",
]
