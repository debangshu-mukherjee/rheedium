"""
Module: inout
-------------
Data input/output utilities for RHEED simulation.

Functions
---------
- From `data_io.py` submodule:
    - `load_atomic_numbers`:
        Load the atomic numbers mapping from a JSON file
    - `parse_cif`:
        Parse a CIF file into a JAX-compatible CrystalStructure
    - `symmetry_expansion`:
        Apply symmetry operations to expand fractional positions and remove duplicates
"""

from .data_io import load_atomic_numbers, parse_cif, symmetry_expansion

__all__ = [
    "load_atomic_numbers",
    "parse_cif",
    "symmetry_expansion",
]
