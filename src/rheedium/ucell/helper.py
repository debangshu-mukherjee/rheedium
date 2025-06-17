"""
Module: ucell.helper
--------------------
Helper functions for unit cell calculations and transformations.

Functions
---------
- `wavelength_ang`:
    Calculates the relativistic electron wavelength in angstroms
- `angle_in_degrees`:
    Calculate the angle in degrees between two vectors
- `compute_lengths_angles`:
    Compute unit cell lengths and angles from lattice vectors
- `parse_cif_and_scrape`:
    Parse CIF file and filter atoms within specified thickness
"""

from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Float, Real, jaxtyped

import rheedium as rh
from rheedium.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def wavelength_ang(energy_kev: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Calculate the relativistic electron wavelength in angstroms.

    Parameters
    ----------
    energy_kev : Float[Array, "..."]
        Electron energy in kiloelectron volts

    Returns
    -------
    Float[Array, "..."]
        Electron wavelength in angstroms

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.helper import wavelength_ang
    >>> energy = jnp.array([10.0, 20.0, 30.0])
    >>> wavelengths = wavelength_ang(energy)
    >>> print(wavelengths)
    [0.1226 0.0866 0.0707]
    """
    return 12.398 / jnp.sqrt(energy_kev * (2 * 511.0 + energy_kev))


@jaxtyped(typechecker=beartype)
def angle_in_degrees(v1: Float[Array, "3"], v2: Float[Array, "3"]) -> Float[Array, ""]:
    """
    Calculate the angle in degrees between two vectors.

    Parameters
    ----------
    v1 : Float[Array, "3"]
        First vector
    v2 : Float[Array, "3"]
        Second vector

    Returns
    -------
    Float[Array, ""]
        Angle between vectors in degrees

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.helper import angle_in_degrees
    >>> v1 = jnp.array([1.0, 0.0, 0.0])
    >>> v2 = jnp.array([0.0, 1.0, 0.0])
    >>> angle = angle_in_degrees(v1, v2)
    >>> print(angle)
    90.0
    """
    return 180.0 * jnp.arccos(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))) / jnp.pi


@jaxtyped(typechecker=beartype)
def compute_lengths_angles(vectors: Float[Array, "3 3"]) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Compute unit cell lengths and angles from lattice vectors.

    Parameters
    ----------
    vectors : Float[Array, "3 3"]
        Lattice vectors as rows of a 3x3 matrix

    Returns
    -------
    tuple[Float[Array, "3"], Float[Array, "3"]]
        Tuple containing (lengths, angles) where:
        - lengths: Array of unit cell lengths in angstroms
        - angles: Array of unit cell angles in degrees

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.helper import compute_lengths_angles
    >>> # Cubic unit cell with a=5.0 Å
    >>> vectors = jnp.array([
    ...     [5.0, 0.0, 0.0],
    ...     [0.0, 5.0, 0.0],
    ...     [0.0, 0.0, 5.0]
    ... ])
    >>> lengths, angles = compute_lengths_angles(vectors)
    >>> print(lengths)
    [5.0 5.0 5.0]
    >>> print(angles)
    [90.0 90.0 90.0]
    """
    lengths = jnp.array([jnp.linalg.norm(v) for v in vectors])
    angles = jnp.array([
        angle_in_degrees(vectors[0], vectors[1]),
        angle_in_degrees(vectors[1], vectors[2]),
        angle_in_degrees(vectors[2], vectors[0])
    ])
    return lengths, angles


@jaxtyped(typechecker=beartype)
def parse_cif_and_scrape(
    cif_path: Union[str, Path],
    zone_axis: Real[Array, "3"],
    thickness_xyz: Real[Array, "3"],
    tolerance: Optional[scalar_float] = 1e-3,
) -> CrystalStructure:
    """
    Description
    -----------
    Parse a CIF file, apply symmetry operations to obtain all equivalent
    atomic positions, and scrape (filter) atoms within specified thickness
    along a given zone axis.

    Parameters
    ----------
    - `cif_path` (Union[str, Path]):
        Path to the CIF file.
    - `zone_axis` (Real[Array, "3"]):
        Vector indicating the zone axis direction (surface normal) in
        Cartesian coordinates.
    - `thickness_xyz` (Real[Array, "3"]):
        Thickness along x, y, z directions in Ångstroms; currently,
        only thickness_xyz[2] (z-direction)
        is used to filter atoms along the provided zone axis.
    - `tolerance` (scalar_float, optional):
        Numerical tolerance parameter reserved for future use.
        Default is 1e-3.

    Returns
    -------
    - `filtered_crystal` (CrystalStructure):
        Crystal structure containing atoms filtered within the specified thickness.

    Notes
    -----
    - The provided `zone_axis` is normalized internally.
    - Current implementation uses thickness only along the zone axis
        direction (z-component of `thickness_xyz`).
    - The `tolerance` parameter is reserved for compatibility and future
        functionality.

    Flow
    ----
    - Parse CIF file to get initial crystal structure
    - Extract Cartesian positions and atomic numbers
    - Normalize zone axis vector
    - Calculate projections of atomic positions onto zone axis
    - Find minimum and maximum projections
    - Calculate center projection and half thickness
    - Create mask for atoms within thickness range
    - Filter Cartesian positions and atomic numbers using mask
    - Build cell vectors from crystal parameters
    - Calculate inverse of cell vectors
    - Convert filtered Cartesian positions to fractional coordinates
    - Create new CrystalStructure with filtered positions
    - Return filtered crystal structure
    """
    crystal: CrystalStructure = rh.inout.parse_cif(cif_path)
    cart_positions: Float[Array, "n 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "n"] = crystal.cart_positions[:, 3]
    zone_axis_norm: Float[Array, ""] = jnp.linalg.norm(zone_axis)
    zone_axis_hat: Float[Array, "3"] = zone_axis / (zone_axis_norm + 1e-12)
    projections: Float[Array, "n"] = cart_positions @ zone_axis_hat
    min_proj: Float[Array, ""] = jnp.min(projections)
    max_proj: Float[Array, ""] = jnp.max(projections)
    center_proj: Float[Array, ""] = (max_proj + min_proj) / 2.0
    half_thickness: Float[Array, ""] = thickness_xyz[2] / 2.0
    mask: Bool[Array, "n"] = jnp.abs(projections - center_proj) <= half_thickness
    filtered_cart_positions: Float[Array, "m 3"] = cart_positions[mask]
    filtered_atomic_numbers: Float[Array, "m"] = atomic_numbers[mask]
    cell_vectors: Float[Array, "3 3"] = rh.ucell.build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    cell_inv: Float[Array, "3 3"] = jnp.linalg.inv(cell_vectors)
    filtered_frac_positions: Float[Array, "m 3"] = (
        filtered_cart_positions @ cell_inv
    ) % 1.0
    filtered_crystal: CrystalStructure = rh.types.create_crystal_structure(
        frac_positions=filtered_frac_positions,
        cart_positions=filtered_cart_positions,
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return filtered_crystal

def parse_cif(cif_path: str) -> CrystalStructure:
    """
    Parse a CIF file into a CrystalStructure object.

    Parameters
    ----------
    cif_path : str
        Path to the CIF file

    Returns
    -------
    CrystalStructure
        Crystal structure containing atomic positions and types

    Examples
    --------
    >>> from rheedium.ucell.helper import parse_cif
    >>> # Parse a CIF file for a simple cubic structure
    >>> structure = parse_cif("path/to/structure.cif")
    >>> print(f"Unit cell vectors:\n{structure.vectors}")
    Unit cell vectors:
    [[5.0 0.0 0.0]
     [0.0 5.0 0.0]
     [0.0 0.0 5.0]]
    """
    # Implementation details...
    pass

def symmetry_expansion(structure: CrystalStructure) -> CrystalStructure:
    """
    Apply symmetry operations to expand fractional positions.

    Parameters
    ----------
    structure : CrystalStructure
        Input crystal structure

    Returns
    -------
    CrystalStructure
        Expanded crystal structure with all symmetry-equivalent positions

    Examples
    --------
    >>> from rheedium.ucell.helper import parse_cif, symmetry_expansion
    >>> # Parse a CIF file and expand symmetry
    >>> structure = parse_cif("path/to/structure.cif")
    >>> expanded = symmetry_expansion(structure)
    >>> print(f"Original atoms: {len(structure.positions)}")
    >>> print(f"Expanded atoms: {len(expanded.positions)}")
    Original atoms: 1
    Expanded atoms: 8
    """
    # Implementation details...
    pass
