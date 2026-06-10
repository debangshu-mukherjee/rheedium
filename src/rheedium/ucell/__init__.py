"""Unit cell and crystallographic utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for crystallographic calculations including
unit cell transformations, reciprocal space operations, and specialized
mathematical functions like Bessel functions used in scattering calculations.

Routine Listings
----------------
:func:`angle_in_degrees`
    Calculate angle in degrees between two vectors.
:func:`atom_scraper`
    Filter atoms within specified depth from surface along zone axis.
:func:`build_cell_vectors`
    Convert lattice parameters to Cartesian cell vectors.
:func:`compute_lengths_angles`
    Extract lattice parameters from cell vectors.
:func:`generate_reciprocal_points`
    Generate reciprocal lattice points for given Miller indices.
:func:`get_unit_cell_matrix`
    Build transformation matrix from lattice parameters.
:func:`miller_to_reciprocal`
    Convert Miller indices to reciprocal lattice basis vectors.
:func:`parse_cif_and_scrape`
    Parse CIF file and filter atoms within penetration depth.
:func:`reciprocal_lattice_vectors`
    Generate reciprocal lattice basis vectors b₁, b₂, b₃.
:func:`reciprocal_unitcell`
    Calculate reciprocal unit cell from direct cell vectors.
"""

from .helper import (
    angle_in_degrees,
    compute_lengths_angles,
    parse_cif_and_scrape,
)
from .unitcell import (
    atom_scraper,
    build_cell_vectors,
    bulk_to_slice,
    generate_reciprocal_points,
    get_unit_cell_matrix,
    miller_to_reciprocal,
    reciprocal_lattice_vectors,
    reciprocal_unitcell,
)

__all__: list[str] = [
    "angle_in_degrees",
    "atom_scraper",
    "build_cell_vectors",
    "bulk_to_slice",
    "compute_lengths_angles",
    "generate_reciprocal_points",
    "get_unit_cell_matrix",
    "miller_to_reciprocal",
    "parse_cif_and_scrape",
    "reciprocal_lattice_vectors",
    "reciprocal_unitcell",
]
