"""Data input/output utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for reading and writing various file formats
used in crystallography and RHEED simulations, including CIF files for crystal
structures and XYZ files for atomic coordinates.

Routine Listings
----------------
atomic_symbol : function
    Returns atomic number for given atomic symbol string
kirkland_potentials : function
    Loads Kirkland scattering factors from CSV file
parse_cif : function
    Parse a CIF file into a JAX-compatible CrystalStructure
parse_xyz : function
    Parses an XYZ file and returns atoms with element symbols and 3D coordinates
symmetry_expansion : function
    Apply symmetry operations to expand fractional positions and remove duplicates

Notes
-----
All parsing functions return JAX-compatible data structures suitable for
automatic differentiation and GPU acceleration.
"""

from .cif import parse_cif, symmetry_expansion
from .xyz import atomic_symbol, kirkland_potentials, parse_xyz

__all__ = [
    "atomic_symbol",
    "kirkland_potentials",
    "parse_cif",
    "parse_xyz",
    "symmetry_expansion",
]
