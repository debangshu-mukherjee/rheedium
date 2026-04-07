"""Surface slab construction and reconstruction utilities.

Extended Summary
----------------
Provides functions for building surface slabs from bulk crystals,
applying m x n surface reconstructions, and adding adsorbate layers.
These are the building blocks for constructing physically realistic
surface models that go beyond ideal bulk termination.

Routine Listings
----------------
:func:`create_surface_slab`
    Construct a surface slab from a bulk crystal by cutting along
    a specified Miller plane with vacuum gap.
:func:`apply_surface_reconstruction`
    Expand the in-plane unit cell and displace surface-layer atoms
    to model an m x n surface reconstruction.
:func:`add_adsorbate_layer`
    Add an adsorbate layer with fractional coverage using the
    virtual crystal approximation.

Notes
-----
All functions return ``CrystalStructure`` PyTrees and are compatible
with JAX transformations. Coordinate transformations use Rodrigues'
rotation formula implemented with pure JAX operations to maintain
differentiability of continuous parameters (displacements, coverage).
"""

import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, Int, Num, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors, reciprocal_lattice_vectors


@jaxtyped(typechecker=beartype)
def create_surface_slab(
    bulk_crystal: CrystalStructure,
    surface_normal_miller: Int[Array, "3"],
    slab_thickness_angstrom: scalar_float,
    vacuum_gap_angstrom: scalar_float,
) -> CrystalStructure:
    """Construct a surface slab from a bulk crystal.

    Parameters
    ----------
    bulk_crystal : CrystalStructure
        Bulk crystal with lattice parameters and atomic basis.
    surface_normal_miller : Int[Array, "3"]
        Miller indices [h, k, l] of the surface normal direction.
    slab_thickness_angstrom : scalar_float
        Total slab thickness in Angstroms.
    vacuum_gap_angstrom : scalar_float
        Vacuum layer thickness in Angstroms above the slab.

    Returns
    -------
    slab : CrystalStructure
        Slab with modified c parameter (slab + vacuum) and all
        atomic positions in the new coordinate system.

    Notes
    -----
    1. Compute rotation matrix R that maps the reciprocal lattice
       vector corresponding to [h, k, l] onto [0, 0, 1] using
       Rodrigues' formula.
    2. Apply R to lattice vectors to get surface-oriented cell.
    3. Expand unit cell along z until depth >= slab_thickness.
    4. Filter atoms: keep only those with 0 <= z <= slab_thickness.
    5. Extend c lattice vector by vacuum_gap.
    6. Recompute fractional coordinates in the new cell.
    7. Return new CrystalStructure with surface-oriented basis.
    """
    surface_normal_miller: Int[Array, "3"] = jnp.asarray(
        surface_normal_miller, dtype=jnp.int32
    )
    slab_thickness_angstrom: Float[Array, ""] = jnp.asarray(
        slab_thickness_angstrom, dtype=jnp.float64
    )
    vacuum_gap_angstrom: Float[Array, ""] = jnp.asarray(
        vacuum_gap_angstrom, dtype=jnp.float64
    )

    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *bulk_crystal.cell_lengths, *bulk_crystal.cell_angles
    )

    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *bulk_crystal.cell_lengths,
        *bulk_crystal.cell_angles,
        in_degrees=True,
    )

    hkl_cart: Float[Array, "3"] = (
        surface_normal_miller[0] * recip_vecs[0]
        + surface_normal_miller[1] * recip_vecs[1]
        + surface_normal_miller[2] * recip_vecs[2]
    )
    hkl_cart_norm: Float[Array, ""] = jnp.linalg.norm(hkl_cart)
    hkl_norm: Float[Array, "3"] = hkl_cart / hkl_cart_norm

    z_axis: Float[Array, "3"] = jnp.array([0.0, 0.0, 1.0])
    rot_axis: Float[Array, "3"] = jnp.cross(hkl_norm, z_axis)
    rot_axis_norm: Float[Array, ""] = jnp.linalg.norm(rot_axis)

    cos_angle: Float[Array, ""] = jnp.dot(hkl_norm, z_axis)
    angle: Float[Array, ""] = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))

    def _aligned_matrix() -> Float[Array, "3 3"]:
        return jnp.eye(3)

    def _rotation_matrix() -> Float[Array, "3 3"]:
        k: Float[Array, "3"] = rot_axis / (rot_axis_norm + 1e-10)
        skew: Float[Array, "3 3"] = jnp.array(
            [
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ]
        )
        rot: Float[Array, "3 3"] = (
            jnp.eye(3)
            + jnp.sin(angle) * skew
            + (1 - jnp.cos(angle)) * (skew @ skew)
        )
        return rot

    _rot_threshold: float = 1e-6
    rotation_matrix: Float[Array, "3 3"] = lax.cond(
        rot_axis_norm < _rot_threshold,
        _aligned_matrix,
        _rotation_matrix,
    )

    rotated_cell_vecs: Float[Array, "3 3"] = cell_vecs @ rotation_matrix.T

    cell_z_proj: Float[Array, ""] = jnp.abs(rotated_cell_vecs[2, 2])
    nz: int = int(jnp.ceil(slab_thickness_angstrom / (cell_z_proj + 1e-10)))
    nz = max(nz, 3)

    positions_xyz: Float[Array, "N 3"] = bulk_crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = bulk_crystal.cart_positions[:, 3]
    rotated_positions: Float[Array, "N 3"] = positions_xyz @ rotation_matrix.T

    n_replicas: int = 2 * nz + 1
    iz_range: Float[Array, "R"] = jnp.arange(-nz, nz + 1, dtype=jnp.float64)
    translations: Float[Array, "R 3"] = (
        iz_range[:, None] * rotated_cell_vecs[2][None, :]
    )
    tiled_positions: Float[Array, "R N 3"] = (
        rotated_positions[None, :, :] + translations[:, None, :]
    )
    supercell_positions: Float[Array, "M 3"] = tiled_positions.reshape(-1, 3)
    supercell_atomic_nums: Float[Array, "M"] = jnp.tile(
        atomic_numbers, n_replicas
    )

    z_min: Float[Array, ""] = supercell_positions[:, 2].min()
    centered_z: Float[Array, "M"] = supercell_positions[:, 2] - z_min

    z_mask: Bool[Array, "M"] = jnp.logical_and(
        centered_z >= 0.0,
        centered_z <= slab_thickness_angstrom,
    )
    filtered_positions: Float[Array, "K 3"] = supercell_positions[z_mask]
    z_bottom: Float[Array, ""] = filtered_positions[:, 2].min()
    filtered_positions: Float[Array, "K 3"] = filtered_positions.at[:, 2].set(
        filtered_positions[:, 2] - z_bottom
    )
    filtered_atomic_nums: Float[Array, "K"] = supercell_atomic_nums[z_mask]

    total_c: Float[Array, ""] = slab_thickness_angstrom + vacuum_gap_angstrom

    a_surf: Float[Array, ""] = jnp.linalg.norm(rotated_cell_vecs[0])
    b_surf: Float[Array, ""] = jnp.linalg.norm(rotated_cell_vecs[1])

    ab_dot: Float[Array, ""] = jnp.dot(
        rotated_cell_vecs[0], rotated_cell_vecs[1]
    )
    cos_gamma: Float[Array, ""] = ab_dot / (a_surf * b_surf + 1e-10)
    gamma_deg: Float[Array, ""] = jnp.degrees(
        jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))
    )

    new_cell_lengths: Float[Array, "3"] = jnp.array([a_surf, b_surf, total_c])
    new_cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, gamma_deg])

    new_cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *new_cell_lengths, *new_cell_angles
    )
    inv_cell: Float[Array, "3 3"] = jnp.linalg.inv(new_cell_vecs)
    frac_positions: Float[Array, "K 3"] = filtered_positions @ inv_cell.T

    cart_with_z: Float[Array, "K 4"] = jnp.column_stack(
        [filtered_positions, filtered_atomic_nums]
    )
    frac_with_z: Float[Array, "K 4"] = jnp.column_stack(
        [frac_positions, filtered_atomic_nums]
    )

    return create_crystal_structure(
        frac_positions=frac_with_z,
        cart_positions=cart_with_z,
        cell_lengths=new_cell_lengths,
        cell_angles=new_cell_angles,
    )


@jaxtyped(typechecker=beartype)
def apply_surface_reconstruction(
    slab: CrystalStructure,
    reconstruction_matrix: Int[Array, "2 2"],
    surface_layer_depth_angstrom: scalar_float,
    atomic_displacements: Float[Array, "N_surface_atoms 3"],
) -> CrystalStructure:
    """Apply an m x n surface reconstruction to a slab.

    Parameters
    ----------
    slab : CrystalStructure
        Base surface slab (unreconstructed).
    reconstruction_matrix : Int[Array, "2 2"]
        2x2 integer matrix relating new surface cell to bulk cell.
        For Si(111) 7x7: ``[[7,0],[0,7]]``.
        For GaAs(001) 2x4: ``[[2,0],[0,4]]``.
    surface_layer_depth_angstrom : scalar_float
        Depth from top surface that defines the reconstructed region.
    atomic_displacements : Float[Array, "N_surface_atoms 3"]
        Displacement vectors [dx, dy, dz] in Angstroms for each atom
        in the reconstructed surface layer.

    Returns
    -------
    reconstructed_slab : CrystalStructure
        Slab with expanded in-plane cell and displaced surface atoms.

    Notes
    -----
    1. Identify surface-layer atoms: those within
       ``surface_layer_depth_angstrom`` of the top surface.
    2. Expand the in-plane cell by the reconstruction matrix:
       ``a1_new = m11*a1 + m12*a2``,
       ``a2_new = m21*a1 + m22*a2``.
    3. Replicate atoms to fill the expanded cell.
    4. Apply displacements to the surface-layer atoms.
    5. Recompute fractional coordinates in the new cell.
    """
    reconstruction_matrix: Int[Array, "2 2"] = jnp.asarray(
        reconstruction_matrix, dtype=jnp.int32
    )
    surface_layer_depth_angstrom: Float[Array, ""] = jnp.asarray(
        surface_layer_depth_angstrom, dtype=jnp.float64
    )

    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *slab.cell_lengths, *slab.cell_angles
    )

    m11: int = int(reconstruction_matrix[0, 0])
    m12: int = int(reconstruction_matrix[0, 1])
    m21: int = int(reconstruction_matrix[1, 0])
    m22: int = int(reconstruction_matrix[1, 1])

    new_a_vec: Float[Array, "3"] = m11 * cell_vecs[0] + m12 * cell_vecs[1]
    new_b_vec: Float[Array, "3"] = m21 * cell_vecs[0] + m22 * cell_vecs[1]
    new_c_vec: Float[Array, "3"] = cell_vecs[2]

    positions_xyz: Float[Array, "N 3"] = slab.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = slab.cart_positions[:, 3]

    nx_range: int = max(abs(m11), abs(m21)) + 1
    ny_range: int = max(abs(m12), abs(m22)) + 1
    n_replicas: int = nx_range * ny_range

    ix_vals: Float[Array, "Rx"] = jnp.arange(nx_range, dtype=jnp.float64)
    iy_vals: Float[Array, "Ry"] = jnp.arange(ny_range, dtype=jnp.float64)
    ix_grid: Float[Array, "Rx Ry"] = jnp.repeat(
        ix_vals[:, None], ny_range, axis=1
    )
    iy_grid: Float[Array, "Rx Ry"] = jnp.repeat(
        iy_vals[None, :], nx_range, axis=0
    )
    ix_flat: Float[Array, "R"] = ix_grid.ravel()
    iy_flat: Float[Array, "R"] = iy_grid.ravel()
    translations: Float[Array, "R 3"] = (
        ix_flat[:, None] * cell_vecs[0][None, :]
        + iy_flat[:, None] * cell_vecs[1][None, :]
    )
    tiled_positions: Float[Array, "R N 3"] = (
        positions_xyz[None, :, :] + translations[:, None, :]
    )
    supercell_positions: Float[Array, "M 3"] = tiled_positions.reshape(-1, 3)
    supercell_atomic_nums: Float[Array, "M"] = jnp.tile(
        atomic_numbers, n_replicas
    )

    new_cell_vecs_mat: Float[Array, "3 3"] = jnp.stack(
        [new_a_vec, new_b_vec, new_c_vec], axis=0
    )
    inv_new_cell: Float[Array, "3 3"] = jnp.linalg.inv(new_cell_vecs_mat)
    frac_in_new: Float[Array, "M 3"] = supercell_positions @ inv_new_cell.T

    in_cell: Bool[Array, "M"] = jnp.all(
        (frac_in_new >= -1e-6) & (frac_in_new < 1.0 - 1e-6),
        axis=1,
    )
    filtered_positions: Float[Array, "K 3"] = supercell_positions[in_cell]
    filtered_atomic_nums: Float[Array, "K"] = supercell_atomic_nums[in_cell]

    z_max: Float[Array, ""] = filtered_positions[:, 2].max()
    z_threshold: Float[Array, ""] = z_max - surface_layer_depth_angstrom
    is_surface: Bool[Array, "K"] = filtered_positions[:, 2] >= z_threshold

    n_surface: int = int(jnp.sum(is_surface))
    n_disp: int = atomic_displacements.shape[0]
    n_apply: int = min(n_surface, n_disp)

    surface_indices: Int[Array, "n_surface"] = jnp.where(
        is_surface, size=n_surface
    )[0]
    displaced_positions: Float[Array, "K 3"] = filtered_positions.at[
        surface_indices[:n_apply]
    ].add(atomic_displacements[:n_apply])

    new_a_len: Float[Array, ""] = jnp.linalg.norm(new_a_vec)
    new_b_len: Float[Array, ""] = jnp.linalg.norm(new_b_vec)
    new_c_len: Float[Array, ""] = jnp.linalg.norm(new_c_vec)

    cos_alpha: Float[Array, ""] = jnp.dot(new_b_vec, new_c_vec) / (
        new_b_len * new_c_len + 1e-10
    )
    cos_beta: Float[Array, ""] = jnp.dot(new_a_vec, new_c_vec) / (
        new_a_len * new_c_len + 1e-10
    )
    cos_gamma: Float[Array, ""] = jnp.dot(new_a_vec, new_b_vec) / (
        new_a_len * new_b_len + 1e-10
    )

    new_cell_lengths: Float[Array, "3"] = jnp.array(
        [new_a_len, new_b_len, new_c_len]
    )
    cos_angles: Float[Array, "3"] = jnp.array([cos_alpha, cos_beta, cos_gamma])
    new_cell_angles: Float[Array, "3"] = jnp.degrees(
        jnp.arccos(jnp.clip(cos_angles, -1.0, 1.0))
    )

    new_frac: Float[Array, "K 3"] = displaced_positions @ inv_new_cell.T
    cart_with_z: Float[Array, "K 4"] = jnp.column_stack(
        [displaced_positions, filtered_atomic_nums]
    )
    frac_with_z: Float[Array, "K 4"] = jnp.column_stack(
        [new_frac, filtered_atomic_nums]
    )

    return create_crystal_structure(
        frac_positions=frac_with_z,
        cart_positions=cart_with_z,
        cell_lengths=new_cell_lengths,
        cell_angles=new_cell_angles,
    )


@jaxtyped(typechecker=beartype)
def add_adsorbate_layer(
    slab: CrystalStructure,
    adsorbate_positions_fractional: Float[Array, "N_ads 3"],
    adsorbate_atomic_numbers: Num[Array, "N_ads"],
    coverage_fraction: scalar_float,
) -> CrystalStructure:
    """Add an adsorbate layer to a surface slab.

    Parameters
    ----------
    slab : CrystalStructure
        Base surface slab.
    adsorbate_positions_fractional : Float[Array, "N_ads 3"]
        Fractional coordinates of adsorbate atoms in the slab cell.
    adsorbate_atomic_numbers : Num[Array, "N_ads"]
        Atomic number of each adsorbate atom.
    coverage_fraction : scalar_float
        Fractional monolayer coverage in [0, 1]. Weights the
        adsorbate scattering contribution via the virtual crystal
        approximation.

    Returns
    -------
    decorated_slab : CrystalStructure
        Slab with adsorbates appended. Adsorbate atomic numbers are
        stored as ``Z * coverage_fraction`` to weight their form
        factor contribution.

    Notes
    -----
    1. Convert adsorbate fractional positions to Cartesian using
       the slab cell vectors.
    2. Scale adsorbate atomic numbers by ``coverage_fraction`` to
       implement the virtual crystal approximation.
    3. Concatenate slab and adsorbate positions.
    4. Return new CrystalStructure with combined atoms.

    The virtual crystal approximation is appropriate for RHEED
    because the beam illuminates macroscopic areas and samples a
    statistical average over many unit cells.
    """
    coverage_fraction: Float[Array, ""] = jnp.asarray(
        coverage_fraction, dtype=jnp.float64
    )
    adsorbate_atomic_numbers: Float[Array, "N_ads"] = jnp.asarray(
        adsorbate_atomic_numbers, dtype=jnp.float64
    )

    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *slab.cell_lengths, *slab.cell_angles
    )

    ads_cart: Float[Array, "N_ads 3"] = (
        adsorbate_positions_fractional @ cell_vecs
    )

    weighted_z: Float[Array, "N_ads"] = (
        adsorbate_atomic_numbers * coverage_fraction
    )

    ads_cart_with_z: Float[Array, "N_ads 4"] = jnp.column_stack(
        [ads_cart, weighted_z]
    )
    ads_frac_with_z: Float[Array, "N_ads 4"] = jnp.column_stack(
        [adsorbate_positions_fractional, weighted_z]
    )

    new_cart: Num[Array, "M 4"] = jnp.concatenate(
        [slab.cart_positions, ads_cart_with_z], axis=0
    )
    new_frac: Num[Array, "M 4"] = jnp.concatenate(
        [slab.frac_positions, ads_frac_with_z], axis=0
    )

    return create_crystal_structure(
        frac_positions=new_frac,
        cart_positions=new_cart,
        cell_lengths=slab.cell_lengths,
        cell_angles=slab.cell_angles,
    )


__all__: list[str] = [
    "add_adsorbate_layer",
    "apply_surface_reconstruction",
    "create_surface_slab",
]
