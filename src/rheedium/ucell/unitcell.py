import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Float, Num, jaxtyped

import rheedium as rh
from rheedium.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def reciprocal_unitcell(unitcell: Num[Array, "3 3"]) -> Float[Array, "3 3"]:
    """
    Description
    -----------
    Calculate the reciprocal cell of a unit cell.

    Parameters
    ----------
    - `unitcell` (Num[Array, "3 3"]):
        The unit cell.

    Returns
    -------
    - `reciprocal_cell` (Float[Array, "3 3"]):
        The reciprocal cell.

    Flow
    ----
    - Calculate the reciprocal cell
    - Check if the matrix is well-conditioned
    - If not, replace the values with NaN
    """
    condition_number = jnp.linalg.cond(unitcell)
    is_well_conditioned = condition_number < 1e10
    reciprocal_cell_uncond: Float[Array, "3 3"] = (
        2 * jnp.pi * jnp.transpose(jnp.linalg.inv(unitcell))
    )
    reciprocal_cell: Float[Array, "3 3"] = jnp.where(
        is_well_conditioned,
        reciprocal_cell_uncond,
        jnp.full_like(reciprocal_cell_uncond, 0.0),
    )
    return reciprocal_cell


@jaxtyped(typechecker=beartype)
def reciprocal_uc_angles(
    unitcell_abc: Num[Array, "3"],
    unitcell_angles: Num[Array, "3"],
    in_degrees: Optional[bool] = True,
    out_degrees: Optional[bool] = False,
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Description
    -----------
    Calculate the reciprocal unit cell when the sides (a, b, c) and
    the angles (alpha, beta, gamma) are given.

    Parameters
    ----------
    - `unitcell_abc` (Num[Array, "3"]):
        The sides of the unit cell.
    - `unitcell_angles` (Num[Array, "3"]):
        The angles of the unit cell.
    - `in_degrees` (bool | None):
        Whether the angles are in degrees or radians.
        If None, it will be assumed that the angles are
        in degrees.
        Default is True.
    - `out_degrees` (bool | None):
        Whether the angles should be in degrees or radians.
        If None, it will be assumed that the angles should
        be in radians.
        Default is False.

    Returns
    -------
    - `reciprocal_abc` (Float[Array, "3"]):
        The sides of the reciprocal unit cell.
    - `reciprocal_angles` (Float[Array, "3"]):
        The angles of the reciprocal unit cell.

    Flow
    ----
    - Convert the angles to radians if they are in degrees
    - Calculate the cos and sin values of the angles
    - Calculate the volume factor of the unit cell
    - Calculate the unit cell volume
    - Calculate the reciprocal lattice parameters
    - Calculate the reciprocal angles
    - Convert the angles to degrees if they are in radians
    """
    if in_degrees:
        unitcell_angles = jnp.radians(unitcell_angles)
    cos_angles: Float[Array, "3"] = jnp.cos(unitcell_angles)
    sin_angles: Float[Array, "3"] = jnp.sin(unitcell_angles)
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )
    volume: Float[Array, ""] = jnp.prod(unitcell_abc) * volume_factor
    reciprocal_abc: Float[Array, "3"] = (
        jnp.array(
            [
                unitcell_abc[1] * unitcell_abc[2] * sin_angles[0],
                unitcell_abc[2] * unitcell_abc[0] * sin_angles[1],
                unitcell_abc[0] * unitcell_abc[1] * sin_angles[2],
            ]
        )
        / volume
    )
    reciprocal_angles = jnp.arccos(
        (cos_angles[:, None] * cos_angles[None, :] - cos_angles[None, :])
        / (sin_angles[:, None] * sin_angles[None, :])
    )
    reciprocal_angles: Float[Array, "3"] = jnp.array(
        [reciprocal_angles[1, 2], reciprocal_angles[2, 0], reciprocal_angles[0, 1]]
    )
    if out_degrees:
        reciprocal_angles = jnp.degrees(reciprocal_angles)
    return (reciprocal_abc, reciprocal_angles)


@jaxtyped(typechecker=beartype)
def get_unit_cell_matrix(
    unitcell_abc: Num[Array, "3"],
    unitcell_angles: Num[Array, "3"],
    in_degrees: Optional[bool] = True,
) -> Float[Array, "3 3"]:
    """
    Description
    -----------
    Calculate the transformation matrix for a unit cell using JAX.

    Parameters
    ----------
    - `unitcell_abc` (Num[Array, "3"]):
        Length of the unit cell edges (a, b, c) in Angstroms.
    - `unitcell_angles` (Num[Array, "3"]):
        Angles between the edges (alpha, beta, gamma) in degrees or radians.
    - `in_degrees` (bool | None):
        Whether the angles are in degrees or radians.
        Default is True.

    Returns
    -------
    - `matrix` (Float[Array, "3 3"]):
        3x3 transformation matrix

    Flow
    ----
    - Convert angles to radians if needed
    - Calculate trigonometric values
    - Calculate volume factor
    - Create the transformation matrix
    """
    angles_rad: Num[Array, "3"]
    if in_degrees:
        angles_rad = jnp.radians(unitcell_angles)
    else:
        angles_rad = unitcell_angles
    cos_angles: Float[Array, "3"] = jnp.cos(angles_rad)
    sin_angles: Float[Array, "3"] = jnp.sin(angles_rad)
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )
    matrix: Float[Array, "3 3"] = jnp.zeros(shape=(3, 3), dtype=jnp.float64)
    matrix = matrix.at[0, 0].set(unitcell_abc[0])
    matrix = matrix.at[0, 1].set(unitcell_abc[1] * cos_angles[2])
    matrix = matrix.at[0, 2].set(unitcell_abc[2] * cos_angles[1])
    matrix = matrix.at[1, 1].set(unitcell_abc[1] * sin_angles[2])
    matrix = matrix.at[1, 2].set(
        unitcell_abc[2]
        * (cos_angles[0] - cos_angles[1] * cos_angles[2])
        / sin_angles[2]
    )
    matrix = matrix.at[2, 2].set(unitcell_abc[2] * volume_factor / sin_angles[2])
    return matrix


@jaxtyped(typechecker=beartype)
def build_cell_vectors(
    a: scalar_num,
    b: scalar_num,
    c: scalar_num,
    alpha_deg: scalar_num,
    beta_deg: scalar_num,
    gamma_deg: scalar_num,
) -> Float[Array, "3 3"]:
    """
    Description
    -----------
    Convert (a, b, c, alpha, beta, gamma) into a 3x3 set of lattice vectors
    in Cartesian coordinates, using the standard crystallographic convention:

    - alpha = angle(b, c)
    - beta  = angle(a, c)
    - gamma = angle(a, b)

    Angles are in degrees.

    Parameters
    ----------
    - `a` (scalar_num):
        Length of the a-vector in Å
    - `b` (scalar_num):
        Length of the b-vector in Å
    - `c` (scalar_num):
        Length of the c-vector in Å
    - `alpha_deg` (scalar_num):
        Angle between b and c in degrees
    - `beta_deg` (scalar_num):
        Angle between a and c in degrees
    - `gamma_deg` (scalar_num):
        Angle between a and b in degrees

    Returns
    -------
    - `cell_vectors` (Float[Array, "3 3"]):
        The 3x3 array of lattice vectors in Cartesian coordinates.
        * cell_vectors[0] = a-vector
        * cell_vectors[1] = b-vector
        * cell_vectors[2] = c-vector

    Flow
    ----
    - Convert angles to radians
    - Calculate the a-vector along x
    - Calculate the b-vector in the x-y plane
    - Calculate the c-vector in full 3D
    - Stack the vectors to form the cell_vectors array
    """
    alpha: Float[Array, ""] = (alpha_deg * jnp.pi) / 180.0
    beta: Float[Array, ""] = (beta_deg * jnp.pi) / 180.0
    gamma: Float[Array, ""] = (gamma_deg * jnp.pi) / 180.0
    a_vec: Float[Array, "3"] = jnp.array([a, 0.0, 0.0])
    b_x: Float[Array, ""] = b * jnp.cos(gamma)
    b_y: Float[Array, ""] = b * jnp.sin(gamma)
    b_vec: Float[Array, "3"] = jnp.array([b_x, b_y, 0.0])
    c_x: Float[Array, ""] = c * jnp.cos(beta)
    c_y: Float[Array, ""] = c * (
        (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma)
    )
    c_z_sq: Float[Array, ""] = (c**2) - (c_x**2) - (c_y**2)
    c_z: Float[Array, ""] = jnp.sqrt(jnp.clip(c_z_sq, a_min=0.0))
    c_vec: Float[Array, "3"] = jnp.array([c_x, c_y, c_z])
    cell_vectors: Float[Array, "3 3"] = jnp.stack([a_vec, b_vec, c_vec], axis=0)
    return cell_vectors


@jaxtyped(typechecker=beartype)
def generate_reciprocal_points(
    crystal: CrystalStructure,
    hmax: Optional[scalar_int] = 3,
    kmax: Optional[scalar_int] = 3,
    lmax: Optional[scalar_int] = 1,
    in_degrees: Optional[bool] = True,
) -> Float[Array, "M 3"]:
    """
    Description
    -----------
    Generate a set of reciprocal-lattice vectors
    G_{hkl} = h a* + k b* + l c*
    for integer h, k, l in [-hmax..hmax], [-kmax..kmax], [-lmax..lmax].

    Utilizes `reciprocal_uc_angles` to find the reciprocal cell parameters
    from the direct (a,b,c,alpha,beta,gamma). Then constructs the
    (3x3) reciprocal-cell vectors via `build_cell_vectors`
    and forms the linear combinations.

    Parameters
    ----------
    - `crystal` (CrystalStructure)
        A NamedTuple containing cell_lengths and cell_angles (in degrees by default).
    - `hmax` (Optional[Int[Array, ""]]):
        Bounds on h. Default is 3.
    - `kmax` (Optional[Int[Array, ""]]):
        Bounds on k. Default is 3.
    - `lmax` (Optional[Int[Array, ""]]):
        Bounds on l. Default is 1.
    - `in_degrees` (Optional[bool]):
        If True, interpret the crystal.cell_angles as degrees.

    Returns
    -------
    - `Gs` (Float[Array, "M 3"]):
        The set of reciprocal-lattice vectors in inverse angstroms.
    """
    abc: Num[Array, "3"] = crystal.cell_lengths
    angles: Num[Array, "3"] = crystal.cell_angles
    rec_abc: Float[Array, "3"]
    rec_angles: Float[Array, "3"]
    rec_abc, rec_angles = rh.ucell.reciprocal_uc_angles(
        unitcell_abc=abc,
        unitcell_angles=angles,
        in_degrees=in_degrees,
        out_degrees=False,
    )
    rec_vectors: Float[Array, "3 3"] = rh.ucell.build_cell_vectors(
        rec_abc, rec_angles, in_degrees=False
    )
    a_star: Float[Array, "3"] = rec_vectors[0]
    b_star: Float[Array, "3"] = rec_vectors[1]
    c_star: Float[Array, "3"] = rec_vectors[2]
    hs: Num[Array, "n_h"] = jnp.arange(-hmax, hmax + 1)
    ks: Num[Array, "n_k"] = jnp.arange(-kmax, kmax + 1)
    ls: Num[Array, "n_l"] = jnp.arange(-lmax, lmax + 1)
    H: Num[Array, "n_h n_k n_l"]
    K: Num[Array, "n_h n_k n_l"]
    L: Num[Array, "n_h n_k n_l"]
    H, K, L = jnp.meshgrid(hs, ks, ls, indexing="ij")
    hkl: Num[Array, "M 3"] = jnp.stack([H.ravel(), K.ravel(), L.ravel()], axis=-1)

    def single_G(hkl_1d):
        h_ = hkl_1d[0]
        k_ = hkl_1d[1]
        l_ = hkl_1d[2]
        return (h_ * a_star) + (k_ * b_star) + (l_ * c_star)

    Gs: Float[Array, "M 3"] = jax.vmap(single_G)(hkl)
    return Gs


@jaxtyped(typechecker=beartype)
def atom_scraper(
    crystal: CrystalStructure,
    zone_axis: Num[Array, "3"],
    penetration_depth: Optional[scalar_num] = 0.0,
    eps: Optional[scalar_float] = 1e-3,
    max_atoms: Optional[scalar_int] = None,
) -> CrystalStructure:
    """
    Robustly filter atoms in `crystal` within `penetration_depth` from the top surface
    (along `zone_axis`). If `penetration_depth == 0.0`, only the topmost layer is returned.

    Parameters
    ----------
    - `crystal` (CrystalStructure):
        The input crystal structure.
    - `zone_axis` (Num[Array, "3"]):
        Surface normal axis in Cartesian space.
    - `penetration_depth` (scalar_num, optional):
        Thickness from the top layer to retain.
        Default: 0.0 (only top layer).
    - `eps` (scalar_float, optional):
        Numerical tolerance for identifying top layer atoms.
        Default: 1e-3.
    - `max_atoms` (scalar_int, optional):
        Maximum atoms to include. If None, defaults to the number of atoms in the input.

    Returns
    -------
    - `filtered_crystal` (CrystalStructure):
        New CrystalStructure with filtered atoms and adjusted cell.
    """
    orig_cell_vectors: Float[Array, "3 3"] = rh.ucell.build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    zone_axis_norm: Float[Array, ""] = jnp.linalg.norm(zone_axis)
    zone_axis_hat: Float[Array, "3"] = zone_axis / (zone_axis_norm + 1e-32)
    cart_xyz: Float[Array, "n 3"] = crystal.cart_positions[:, :3]
    dot_vals: Float[Array, "n"] = jnp.einsum("ij,j->i", cart_xyz, zone_axis_hat)
    d_max: Float[Array, ""] = jnp.max(dot_vals)
    dist_from_top: Float[Array, "n"] = d_max - dot_vals
    positive_distances = dist_from_top[dist_from_top > 1e-8]
    adaptive_eps = jnp.where(
        positive_distances.size > 0,
        jnp.maximum(eps, 2 * jnp.min(positive_distances)),
        eps,
    )
    is_top_layer_mode: Bool[Array, ""] = jnp.isclose(
        penetration_depth, jnp.asarray(0.0), atol=1e-8
    )
    mask: Bool[Array, "n"] = jnp.where(
        is_top_layer_mode,
        dist_from_top <= adaptive_eps,
        dist_from_top <= penetration_depth,
    )

    def gather_valid_positions(
        positions: Float[Array, "n 4"], gather_mask: Bool[Array, "n"]
    ) -> Float[Array, "m 4"]:
        return positions[gather_mask]

    filtered_frac: Float[Array, "m 4"] = gather_valid_positions(
        crystal.frac_positions, mask
    )
    filtered_cart: Float[Array, "m 4"] = gather_valid_positions(
        crystal.cart_positions, mask
    )
    original_height: Float[Array, ""] = jnp.max(dot_vals) - jnp.min(dot_vals)
    new_height: Float[Array, ""] = jnp.where(
        is_top_layer_mode, adaptive_eps, jnp.minimum(penetration_depth, original_height)
    )

    def scale_vector(
        vec: Float[Array, "3"],
        zone_axis_hat: Float[Array, "3"],
        old_height: Float[Array, ""],
        new_height: Float[Array, ""],
    ) -> Float[Array, "3"]:
        proj_mag: Float[Array, ""] = jnp.dot(vec, zone_axis_hat)
        parallel_comp: Float[Array, "3"] = proj_mag * zone_axis_hat
        perp_comp: Float[Array, "3"] = vec - parallel_comp
        scale_factor: Float[Array, ""] = jnp.where(
            old_height < 1e-32, 1.0, new_height / old_height
        )
        scaled_parallel: Float[Array, "3"] = scale_factor * parallel_comp
        return scaled_parallel + perp_comp

    def scale_if_needed(
        vec: Float[Array, "3"],
        zone_axis_hat: Float[Array, "3"],
        original_height: Float[Array, ""],
        new_height: Float[Array, ""],
    ) -> Float[Array, "3"]:
        needs_scaling: Bool[Array, ""] = jnp.abs(jnp.dot(vec, zone_axis_hat)) > 1e-8
        scaled: Float[Array, "3"] = scale_vector(
            vec, zone_axis_hat, original_height, new_height
        )
        return jnp.where(needs_scaling, scaled, vec)

    scaled_vectors = jnp.stack(
        [
            scale_if_needed(
                orig_cell_vectors[i], zone_axis_hat, original_height, new_height
            )
            for i in range(3)
        ]
    )
    new_lengths, new_angles = rh.ucell.compute_lengths_angles(scaled_vectors)
    filtered_crystal = CrystalStructure(
        frac_positions=filtered_frac,
        cart_positions=filtered_cart,
        cell_lengths=new_lengths,
        cell_angles=new_angles,
    )
    return filtered_crystal
