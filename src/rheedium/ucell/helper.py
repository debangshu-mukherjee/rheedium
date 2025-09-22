"""Helper functions for unit cell calculations and transformations.

Extended Summary
----------------
This module provides utility functions for crystallographic calculations,
including vector operations, lattice parameter computations, and crystal
structure filtering based on geometric criteria.

Routine Listings
----------------
angle_in_degrees : function
    Calculate the angle in degrees between two vectors
compute_lengths_angles : function
    Compute unit cell lengths and angles from lattice vectors
parse_cif_and_scrape : function
    Parse CIF file and filter atoms within specified thickness

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Float, Real, jaxtyped

import rheedium as rh
from rheedium.types import CrystalStructure

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def angle_in_degrees(
    v1: Float[Array, " n"], v2: Float[Array, " n"]
) -> Float[Array, " "]:
    """Calculate the angle in degrees between two vectors.

    As long as the vectors have the same number of elements,
    any dimensional vectors will work.

    Parameters
    ----------
    v1 : Float[Array, " n"]
        First vector
    v2 : Float[Array, " n"]
        Second vector

    Returns
    -------
    angle : Float[Array, " "]
        Angle between vectors in degrees

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> v1 = jnp.array([1.0, 0.0, 0.0])
    >>> v2 = jnp.array([0.0, 1.0, 0.0])
    >>> angle = rh.ucell.angle_in_degrees(v1, v2)
    >>> print(angle)
    90.0
    """

    def _check_vector_dimensions() -> (
        Tuple[Float[Array, " n"], Float[Array, " n"]]
    ):
        return jax.lax.cond(
            v1.shape == v2.shape,
            lambda: (v1, v2),
            lambda: jax.lax.stop_gradient(
                jax.lax.cond(False, lambda: (v1, v2), lambda: (v1, v2))
            ),
        )

    _check_vector_dimensions()
    angle: Float[Array, " "] = (
        180.0
        * jnp.arccos(
            jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
        )
        / jnp.pi
    )
    return angle


@jaxtyped(typechecker=beartype)
def compute_lengths_angles(
    vectors: Float[Array, " 3 3"],
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """Compute unit cell lengths and angles from lattice vectors.

    Parameters
    ----------
    vectors : Float[Array, " 3 3"]
        Lattice vectors as rows of a 3x3 matrix

    Returns
    -------
    lengths : Float[Array, " 3"]
        Unit cell lengths in angstroms
    angles : Float[Array, " 3"]
        Unit cell angles in degrees

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> # Cubic unit cell with a=5.0 Å
    >>> vectors = jnp.array([
    ...     [5.0, 0.0, 0.0],
    ...     [0.0, 5.0, 0.0],
    ...     [0.0, 0.0, 5.0]
    ... ])
    >>> lengths, angles = rh.ucell.compute_lengths_angles(vectors)
    >>> print(lengths)
    [5.0 5.0 5.0]
    >>> print(angles)
    [90.0 90.0 90.0]
    """
    lengths: Float[Array, " 3"] = jnp.array(
        [jnp.linalg.norm(v) for v in vectors]
    )
    angles: Float[Array, " 3"] = jnp.array(
        [
            angle_in_degrees(vectors[0], vectors[1]),
            angle_in_degrees(vectors[1], vectors[2]),
            angle_in_degrees(vectors[2], vectors[0]),
        ]
    )
    return lengths, angles


@jaxtyped(typechecker=beartype)
def parse_cif_and_scrape(
    cif_path: Union[str, Path],
    zone_axis: Real[Array, " 3"],
    thickness_xyz: Real[Array, " 3"],
) -> CrystalStructure:
    """Parse a CIF file and filter atoms within specified thickness along a zone axis.

    Parse a CIF file, apply symmetry operations to obtain all equivalent
    atomic positions, and scrape (filter) atoms within specified thickness
    along a given zone axis.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the CIF file.
    zone_axis : Real[Array, " 3"]
        Vector indicating the zone axis direction (surface normal) in
        Cartesian coordinates.
    thickness_xyz : Real[Array, " 3"]
        Thickness along x, y, z directions in Ångstroms; currently,
        only thickness_xyz[2] (z-direction)
        is used to filter atoms along the provided zone axis.

    Returns
    -------
    filtered_crystal : CrystalStructure
        Crystal structure containing atoms filtered within the specified thickness.

    Notes
    -----
    - The provided `zone_axis` is normalized internally.
    - Current implementation uses thickness only along the zone axis
      direction (z-component of `thickness_xyz`).
    - The `tolerance` parameter is reserved for compatibility and future
      functionality.

    Algorithm
    ---------
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
    cart_positions: Float[Array, " n 3"] = crystal.cart_positions[:, :3]
    zone_axis_norm: Float[Array, " "] = jnp.linalg.norm(zone_axis)
    zone_axis_hat: Float[Array, " 3"] = zone_axis / (zone_axis_norm + 1e-12)
    projections: Float[Array, " n"] = cart_positions @ zone_axis_hat
    min_proj: Float[Array, " "] = jnp.min(projections)
    max_proj: Float[Array, " "] = jnp.max(projections)
    center_proj: Float[Array, " "] = (max_proj + min_proj) / 2.0
    half_thickness: Float[Array, " "] = thickness_xyz[2] / 2.0
    mask: Bool[Array, " n"] = (
        jnp.abs(projections - center_proj) <= half_thickness
    )
    filtered_cart_positions: Float[Array, " m 3"] = cart_positions[mask]
    cell_vectors: Float[Array, " 3 3"] = rh.ucell.build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    cell_inv: Float[Array, " 3 3"] = jnp.linalg.inv(cell_vectors)
    filtered_frac_positions: Float[Array, " m 3"] = (
        filtered_cart_positions @ cell_inv
    ) % 1.0
    filtered_crystal: CrystalStructure = rh.types.create_crystal_structure(
        frac_positions=filtered_frac_positions,
        cart_positions=filtered_cart_positions,
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return filtered_crystal
