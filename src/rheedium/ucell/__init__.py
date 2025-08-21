"""
Module: ucell
-------------
Unit cell and crystallographic utilities for RHEED simulation.

Functions
---------
- `angle_in_degrees`:
    Calculate angle in degrees between two vectors
- `compute_lengths_angles`:
    Extract lattice parameters from cell vectors
- `parse_cif_and_scrape`:
    Parse CIF file and filter atoms within penetration depth
- `reciprocal_unitcell`:
    Calculate reciprocal unit cell from direct cell vectors
- `reciprocal_uc_angles`:
    Calculate reciprocal lattice parameters from direct parameters
- `get_unit_cell_matrix`:
    Build transformation matrix from lattice parameters
- `build_cell_vectors`:
    Convert lattice parameters to Cartesian cell vectors
- `generate_reciprocal_points`:
    Generate reciprocal lattice points for given Miller indices
- `atom_scraper`:
    Filter atoms within specified depth from surface along zone axis
- `bessel_k0`:
    Modified Bessel function of second kind, order 0
- `bessel_k1`:
    Modified Bessel function of second kind, order 1
- `bessel_kv`:
    Modified Bessel function of second kind, arbitrary order
"""

from .bessel import bessel_kv
from .helper import (angle_in_degrees, compute_lengths_angles,
                     parse_cif_and_scrape)
from .unitcell import (atom_scraper, build_cell_vectors,
                       generate_reciprocal_points, get_unit_cell_matrix,
                       reciprocal_uc_angles, reciprocal_unitcell)

__all__ = [
    "angle_in_degrees",
    "compute_lengths_angles",
    "parse_cif_and_scrape",
    "reciprocal_unitcell",
    "reciprocal_uc_angles",
    "get_unit_cell_matrix",
    "build_cell_vectors",
    "generate_reciprocal_points",
    "atom_scraper",
    "bessel_kv",
]
