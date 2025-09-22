"""Unit cell and crystallographic utilities for RHEED simulation.

Extended Summary
----------------
This module provides functions for crystallographic calculations including
unit cell transformations, reciprocal space operations, and specialized
mathematical functions like Bessel functions used in scattering calculations.

Routine Listings
----------------
angle_in_degrees : function
    Calculate angle in degrees between two vectors
atom_scraper : function
    Filter atoms within specified depth from surface along zone axis
bessel_kv : function
    Modified Bessel function of second kind, arbitrary order
build_cell_vectors : function
    Convert lattice parameters to Cartesian cell vectors
compute_lengths_angles : function
    Extract lattice parameters from cell vectors
generate_reciprocal_points : function
    Generate reciprocal lattice points for given Miller indices
get_unit_cell_matrix : function
    Build transformation matrix from lattice parameters
miller_to_reciprocal : function
    Convert Miller indices to reciprocal lattice basis vectors
parse_cif_and_scrape : function
    Parse CIF file and filter atoms within penetration depth
reciprocal_lattice_vectors : function
    Generate reciprocal lattice basis vectors b₁, b₂, b₃
reciprocal_unitcell : function
    Calculate reciprocal unit cell from direct cell vectors
"""

from .bessel import bessel_kv
from .helper import (
    angle_in_degrees,
    compute_lengths_angles,
    parse_cif_and_scrape,
)
from .unitcell import (
    atom_scraper,
    build_cell_vectors,
    generate_reciprocal_points,
    get_unit_cell_matrix,
    miller_to_reciprocal,
    reciprocal_lattice_vectors,
    reciprocal_unitcell,
)

__all__ = [
    "angle_in_degrees",
    "atom_scraper",
    "bessel_kv",
    "build_cell_vectors",
    "compute_lengths_angles",
    "generate_reciprocal_points",
    "get_unit_cell_matrix",
    "miller_to_reciprocal",
    "parse_cif_and_scrape",
    "reciprocal_lattice_vectors",
    "reciprocal_unitcell",
]
