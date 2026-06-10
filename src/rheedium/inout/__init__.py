"""Data input/output utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for reading and writing various file formats
used in crystallography and RHEED simulations, including CIF files, XYZ files,
VASP POSCAR/CONTCAR files, and vasprun.xml for crystal structures. It also
provides interoperability with ASE and pymatgen libraries.

Routine Listings
----------------
:func:`atomic_masses`
    Return preloaded atomic masses as JAX array.
:class:`FrameMetadata`
    Per-frame metadata extracted from TIFF tags.
:func:`atomic_symbol`
    Returns atomic number for given atomic symbol string.
:func:`debye_temperatures`
    Return preloaded Debye temperatures as JAX array.
:func:`detect_beam_center`
    Locate the specular spot position automatically.
:func:`extract_frame_metadata`
    Extract exposure time, timestamp, and description from a TIFF page.
:func:`from_ase`
    Convert ASE Atoms to CrystalStructure.
:func:`from_pymatgen`
    Convert pymatgen Structure to CrystalStructure.
:func:`kirkland_potentials`
    Loads Kirkland scattering factors from CSV file.
:func:`load_from_h5`
    Load one or more rheedium PyTrees from an HDF5 file.
:func:`lobato_potentials`
    Loads Lobato-van Dyck scattering factor parameters
    from CSV file.
:func:`lattice_to_cell_params`
    Convert 3x3 lattice vectors to crystallographic cell parameters.
:func:`load_tiff_as_rheed_image`
    Load a single TIFF frame and return a RHEEDImage PyTree.
:func:`load_tiff_sequence`
    Load ordered TIFF stack into a JAX array.
:func:`normalize_sequence`
    Background subtraction, flat-field correction, and normalization.
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
:func:`save_to_h5`
    Save one or more rheedium PyTrees to an HDF5 file.
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
from .crystal import parse_crystal, xyz_to_crystal
from .hdf5 import load_from_h5, save_to_h5
from .interop import from_ase, from_pymatgen, to_ase, to_pymatgen
from .lattice import lattice_to_cell_params
from .poscar import parse_poscar
from .tiff import (
    FrameMetadata,
    detect_beam_center,
    extract_frame_metadata,
    load_tiff_as_rheed_image,
    load_tiff_sequence,
    normalize_sequence,
)
from .vaspxml import parse_vaspxml, parse_vaspxml_trajectory
from .xyz import (
    atomic_masses,
    atomic_symbol,
    debye_temperatures,
    kirkland_potentials,
    lobato_potentials,
    parse_xyz,
)

__all__: list[str] = [
    "atomic_masses",
    "atomic_symbol",
    "debye_temperatures",
    "detect_beam_center",
    "extract_frame_metadata",
    "FrameMetadata",
    "from_ase",
    "from_pymatgen",
    "kirkland_potentials",
    "lobato_potentials",
    "lattice_to_cell_params",
    "load_from_h5",
    "load_tiff_as_rheed_image",
    "load_tiff_sequence",
    "normalize_sequence",
    "parse_cif",
    "parse_crystal",
    "parse_poscar",
    "parse_vaspxml",
    "parse_vaspxml_trajectory",
    "parse_xyz",
    "save_to_h5",
    "symmetry_expansion",
    "to_ase",
    "to_pymatgen",
    "xyz_to_crystal",
]
