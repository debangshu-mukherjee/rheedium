from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Float, jaxtyped

from rheedium import *
from rheedium.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def wavelength_ang(voltage_kV: scalar_num) -> Float[Array, ""]:
    """
    Description
    -----------
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `voltage_kV` (scalar_num):
        The microscope accelerating voltage in kilo
        electronVolts

    Returns
    -------
    - `in_angstroms (Float[Array, ""]):
        The electron wavelength in angstroms

    Flow
    ----
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: Float[Array, ""] = jnp.float64(9.109383e-31)
    e: Float[Array, ""] = jnp.float64(1.602177e-19)
    c: Float[Array, ""] = jnp.float64(299792458.0)
    h: Float[Array, ""] = jnp.float64(6.62607e-34)
    voltage: Float[Array, ""] = jnp.multiply(jnp.float64(voltage_kV), jnp.float64(1000))
    eV = jnp.multiply(e, voltage)
    numerator: Float[Array, ""] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, ""] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, ""] = jnp.sqrt(numerator / denominator)
    in_angstroms: Float[Array, ""] = 1e10 * wavelength_meters
    return in_angstroms


@jaxtyped(typechecker=beartype)
def angle_in_degrees(u: Float[Array, "c"], v: Float[Array, "c"]) -> Float[Array, ""]:
    """
    Description
    -----------
    Calculate the angle in degrees between two vectors u and v.

    Parameters
    ----------
    - `u` (Float[Array, "c"]):
        The first vector.
    - `v` (Float[Array, "c"]):
        The second vector.

    Returns
    -------
    - `cos_val_degrees` (Float[Array, ""]):
        The angle in degrees between u and v.

    Flow
    ----
    - Calculate the dot product of u and v
    - Calculate the cosine of the angle
    - Convert the cosine to degrees
    """
    dot_uv: Float[Array, ""] = jnp.dot(u, v)
    cos_val: Float[Array, ""] = dot_uv / (
        jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-32
    )
    cos_val_clamped: Float[Array, ""] = jnp.clip(cos_val, -1.0, 1.0)
    cos_val_degrees: Float[Array, ""] = (jnp.arccos(cos_val_clamped) * 180.0) / jnp.pi
    return cos_val_degrees


@jaxtyped(typechecker=beartype)
def compute_lengths_angles(
    cell_vectors: Float[Array, "3 3"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Description
    -----------
    Given a (3, 3) array of lattice vectors (a_vec, b_vec, c_vec),
    compute (a, b, c) in Å and (alpha, beta, gamma) in degrees.

    Parameters
    ----------
    - `cell_vectors` (Float[Array, "3 3"]):
        The 3x3 array of lattice vectors in Cartesian coordinates.

    Returns
    -------
    - `lengths` (Float[Array, "3"]):
        The lengths of the lattice vectors (a, b, c) in Å.
    - `angles` (Float[Array, "3"]):
        The angles between the lattice vectors (alpha, beta, gamma) in degrees.
    """
    a_vec: Float[Array, "3"] = cell_vectors[0]
    b_vec: Float[Array, "3"] = cell_vectors[1]
    c_vec: Float[Array, "3"] = cell_vectors[2]
    a_len: Float[Array, ""] = jnp.linalg.norm(a_vec)
    b_len: Float[Array, ""] = jnp.linalg.norm(b_vec)
    c_len: Float[Array, ""] = jnp.linalg.norm(c_vec)
    alpha: Float[Array, ""] = ucell.angle_in_degrees(b_vec, c_vec)
    beta: Float[Array, ""] = ucell.angle_in_degrees(a_vec, c_vec)
    gamma: Float[Array, ""] = ucell.angle_in_degrees(a_vec, b_vec)
    lengths: Float[Array, "3"] = jnp.array([a_len, b_len, c_len])
    angles: Float[Array, "3"] = jnp.array([alpha, beta, gamma])
    return (lengths, angles)


@jaxtyped(typechecker=beartype)
def parse_cif_and_scrape(
    cif_path: Union[str, Path],
    zone_axis: Float[Array, "3"],
    thickness_xyz: Float[Array, "3"],
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
    - `zone_axis` (Float[Array, "3"]):
        Vector indicating the zone axis direction (surface normal) in
        Cartesian coordinates.
    - `thickness_xyz` (Float[Array, "3"]):
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
    """
    crystal: CrystalStructure = inout.parse_cif(cif_path)
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
    cell_vectors: Float[Array, "3 3"] = ucell.build_cell_vectors(
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
    filtered_crystal: CrystalStructure = CrystalStructure(
        frac_positions=jnp.column_stack(
            [filtered_frac_positions, filtered_atomic_numbers]
        ),
        cart_positions=jnp.column_stack(
            [filtered_cart_positions, filtered_atomic_numbers]
        ),
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return filtered_crystal
