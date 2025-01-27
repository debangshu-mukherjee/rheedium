import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, Num, jaxtyped

from rheedium import io, uc

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
    # Optional: Check that matrix is well-conditioned
    condition_number = jnp.linalg.cond(unitcell)
    is_well_conditioned = condition_number < 1e10  # threshold can be adjusted

    # Calculate reciprocal cell
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
    # Convert to radians if the angles are in degrees
    if in_degrees:
        unitcell_angles = jnp.radians(unitcell_angles)

    # Calculate cos and sin values of the angles
    cos_angles: Float[Array, "3"] = jnp.cos(unitcell_angles)
    sin_angles: Float[Array, "3"] = jnp.sin(unitcell_angles)

    # Calculate the volume factor of the unit cell
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )

    # Calculate unit cell volume
    volume: Float[Array, ""] = jnp.prod(unitcell_abc) * volume_factor

    # Calculate reciprocal lattice parameters
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

    # Calculate reciprocal angles
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
    """
    # Convert to radians if needed
    angles_rad: Num[Array, "3"]
    if in_degrees:
        angles_rad = jnp.radians(unitcell_angles)
    else:
        angles_rad = unitcell_angles

    # Calculate trigonometric values
    cos_angles: Float[Array, "3"] = jnp.cos(angles_rad)
    sin_angles: Float[Array, "3"] = jnp.sin(angles_rad)

    # Calculate volume factor
    volume_factor: Float[Array, ""] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )

    # Create the transformation matrix
    matrix: Float[Array, "3 3"] = jnp.zeros(shape=(3, 3), dtype=jnp.float64)

    # Update matrix elements
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
    a: Float[Array, ""],
    b: Float[Array, ""],
    c: Float[Array, ""],
    alpha_deg: Float[Array, ""],
    beta_deg: Float[Array, ""],
    gamma_deg: Float[Array, ""],
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
    - `a` (Float[Array, ""]):
        Length of the a-vector in Å
    - `b` (Float[Array, ""]):
        Length of the b-vector in Å
    - `c` (Float[Array, ""]):
        Length of the c-vector in Å
    - `alpha_deg` (Float[Array, ""]):
        Angle between b and c in degrees
    - `beta_deg` (Float[Array, ""]):
        Angle between a and c in degrees
    - `gamma_deg` (Float[Array, ""]):
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

    # Vector a along x
    a_vec: Float[Array, "3"] = jnp.array([a, 0.0, 0.0])

    # Vector b in the x-y plane
    b_x: Float[Array, ""] = b * jnp.cos(gamma)
    b_y: Float[Array, ""] = b * jnp.sin(gamma)
    b_vec: Float[Array, "3"] = jnp.array([b_x, b_y, 0.0])

    # Vector c in full 3D
    c_x: Float[Array, ""] = c * jnp.cos(beta)
    # The expression for c_y uses the fact that cos(alpha) = (b·c)/(|b||c|)
    # combined with the known c_x and gamma.  This is a standard formula:
    c_y: Float[Array, ""] = c * (
        (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma)
    )
    # Finally, c_z by Pythagoras to ensure the correct |c|
    c_z_sq: Float[Array, ""] = (c**2) - (c_x**2) - (c_y**2)
    # clip to avoid negative sqrt from floating error
    c_z: Float[Array, ""] = jnp.sqrt(jnp.clip(c_z_sq, a_min=0.0))

    c_vec: Float[Array, "3"] = jnp.array([c_x, c_y, c_z])

    cell_vectors: Float[Array, "3 3"] = jnp.stack([a_vec, b_vec, c_vec], axis=0)
    return cell_vectors


@jaxtyped(typechecker=beartype)
def atom_scraper(
    crystal: io.CrystalStructure,
    zone_axis: Float[Array, "3"],
    penetration_depth: Optional[Float[Array, ""]] = jnp.asarray(0.0),
    eps: Optional[Float[Array, ""]] = jnp.asarray(1e-8),
    max_atoms: Optional[Int[Array, ""]] = jnp.asarray(0),
) -> io.CrystalStructure:
    """
    Description
    ------------
    Filters atoms in `crystal` so only those within `penetration_depth`
    from the top surface (along `zone_axis`) are returned.
    If `penetration_depth == 0.0`, only the topmost layer is returned.
    Adjusts the unit cell dimension along zone_axis to match the
    penetration depth.

    Parameters
    ----------
    - `crystal` (CrystalStructure):
        The input crystal structure
    - `zone_axis` (Float[Array, "3"]):
        The reference axis (surface normal) in Cartesian space.
    - `penetration_depth` (Optional[Float[Array, ""]]):
        Thickness (in Å) from the top layer to retain.
    - `eps` (Optional[Float[Array, ""]]):
        Tolerance for identifying top layer atoms.
    - `max_atoms` (Optional[Int[Array, ""]]):
        Maximum number of atoms to handle. Used for static shapes.
        If None, uses the length of input positions.

    Returns
    -------
    - `filtered_crystal` (CrystalStructure):
        A new CrystalStructure containing only the filtered atoms and
        adjusted cell.
    """
    # Handle max_atoms for static shapes
    if max_atoms == 0:
        max_atoms = crystal.cart_positions.shape[0]

    # Normalize zone_axis
    zone_axis_norm: Float[Array, ""] = jnp.linalg.norm(zone_axis)
    zone_axis_hat: Float[Array, "3"] = zone_axis / (zone_axis_norm + 1e-32)

    # Extract xyz coordinates
    cart_xyz: Float[Array, "n 3"] = crystal.cart_positions[:, :3]

    # Compute dot products
    dot_vals: Float[Array, "n"] = jnp.einsum("ij,j->i", cart_xyz, zone_axis_hat)
    d_max: Float[Array, ""] = jnp.max(dot_vals)
    d_min: Float[Array, ""] = jnp.min(dot_vals)
    dist_from_top: Float[Array, "n"] = d_max - dot_vals

    is_top_layer_mode: Bool[Array, ""] = jnp.isclose(
        penetration_depth, jnp.asarray(0.0), atol=1e-8
    )

    # Create mask
    mask: Bool[Array, "n"] = jnp.where(
        is_top_layer_mode,
        dist_from_top <= eps,
        dist_from_top <= penetration_depth,
    )

    # Convert boolean mask to indices
    indices: Int[Array, "k"] = jnp.where(mask)[0]

    # Pad indices to fixed length
    padded_indices: Int[Array, "max_n"] = jnp.pad(
        indices, (0, max_atoms - indices.shape[0]), mode="constant", constant_values=-1
    )

    # Use gather with masked updates for fixed-shape output
    def gather_positions(positions: Float[Array, "n 4"]) -> Float[Array, "max_n 4"]:
        gathered: Float[Array, "max_n 4"] = jnp.zeros(
            (max_atoms,) + positions.shape[1:]
        )
        valid_mask: Bool[Array, "max_n"] = padded_indices >= 0
        return jax.lax.select(valid_mask[:, None], positions[padded_indices], gathered)

    filtered_frac: Float[Array, "max_n 4"] = gather_positions(crystal.frac_positions)
    filtered_cart: Float[Array, "max_n 4"] = gather_positions(crystal.cart_positions)

    # Calculate the new cell dimension along zone_axis
    original_height: Float[Array, ""] = d_max - d_min
    new_height: Float[Array, ""] = jnp.where(
        is_top_layer_mode,
        eps,  # Minimal thickness for top layer
        jnp.minimum(penetration_depth, original_height),
    )

    # Project cell vectors onto zone_axis and perpendicular components
    cell_vectors: Float[Array, "3 3"] = jnp.array(
        [
            [crystal.cell_lengths[0], 0, 0],  # a
            [0, crystal.cell_lengths[1], 0],  # b
            [0, 0, crystal.cell_lengths[2]],  # c
        ]
    )

    # For each cell vector, calculate its component along zone_axis
    projections: Float[Array, "3"] = jnp.einsum("ij,j->i", cell_vectors, zone_axis_hat)

    # Scale the cell vectors along zone_axis direction
    scale_factors: Float[Array, "3"] = jnp.where(
        jnp.abs(projections) > eps, new_height / (jnp.abs(projections) + 1e-32), 1.0
    )

    # Apply scaling to cell lengths
    new_cell_lengths: Float[Array, "3"] = crystal.cell_lengths * scale_factors

    filtered_crystal = io.CrystalStructure(
        frac_positions=filtered_frac,
        cart_positions=filtered_cart,
        cell_lengths=new_cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return filtered_crystal
