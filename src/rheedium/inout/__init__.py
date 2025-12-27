"""Data input/output utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for reading and writing various file formats
used in crystallography and RHEED simulations, including CIF files, XYZ files,
VASP POSCAR/CONTCAR files, and vasprun.xml for crystal structures. It also
provides interoperability with ASE and pymatgen libraries.

Routine Listings
----------------
atomic_symbol : function
    Returns atomic number for given atomic symbol string.
from_ase : function
    Convert ASE Atoms to CrystalStructure.
from_pymatgen : function
    Convert pymatgen Structure to CrystalStructure.
kirkland_potentials : function
    Loads Kirkland scattering factors from CSV file.
lattice_to_cell_params : function
    Convert 3x3 lattice vectors to crystallographic cell parameters.
parse_cif : function
    Parse a CIF file into a JAX-compatible CrystalStructure.
parse_crystal : function
    Parse CIF, XYZ, or POSCAR file into simulation-ready CrystalStructure.
parse_poscar : function
    Parse VASP POSCAR/CONTCAR file into CrystalStructure.
parse_vaspxml : function
    Parse vasprun.xml for structure with optional metadata.
parse_vaspxml_trajectory : function
    Parse full trajectory from vasprun.xml.
parse_xyz : function
    Parses XYZ files and returns atoms with element symbols and 3D coordinates.
symmetry_expansion : function
    Apply symmetry operations to expand fractional positions and remove
    duplicates.
to_ase : function
    Convert CrystalStructure to ASE Atoms.
to_pymatgen : function
    Convert CrystalStructure to pymatgen Structure.
xyz_to_crystal : function
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
