"""
Module: inout
-------------
Data input/output utilities for RHEED simulation.

Functions
---------
- `parse_cif`:
    Parse a CIF file into a JAX-compatible CrystalStructure
- `symmetry_expansion`:
    Apply symmetry operations to expand fractional positions and remove duplicates
- `atomic_symbol`:
    Returns atomic number for given atomic symbol string.
- `kirkland_potentials`:
    Loads Kirkland scattering factors from CSV file.
- `parse_xyz`:
    Parses an XYZ file and returns a list of atoms with their
    element symbols and 3D coordinates.
"""

from .cif import parse_cif, symmetry_expansion
from .xyz import atomic_symbol, kirkland_potentials, parse_xyz

__all__ = [
    "atomic_symbol",
    "kirkland_potentials",
    "parse_xyz",
    "parse_cif",
    "symmetry_expansion",
]
