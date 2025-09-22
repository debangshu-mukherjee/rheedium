"""Functions for unit cell calculations and transformations.

Extended Summary
----------------
This module provides functions for crystallographic unit cell operations
including reciprocal space calculations, lattice transformations, and atom
filtering for specific zones and thicknesses.

Routine Listings
----------------
reciprocal_unitcell : function
    Calculate reciprocal unit cell angles from direct cell angles
get_unit_cell_matrix : function
    Build transformation matrix between direct and reciprocal space
build_cell_vectors : function
    Construct unit cell vectors from lengths and angles
compute_lengths_angles : function
    Compute unit cell lengths and angles from lattice vectors
generate_reciprocal_points : function
    Generate reciprocal lattice points for given hkl ranges
atom_scraper : function
    Filter atoms within specified thickness along zone axis
reciprocal_lattice_vectors : function
    Generate reciprocal lattice basis vectors b₁, b₂, b₃
miller_to_reciprocal : function
    Convert Miller indices to reciprocal lattice basis vectors

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Int, Num, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
    scalar_int,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def reciprocal_unitcell(
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
    alpha: scalar_float,
    beta: scalar_float,
    gamma: scalar_float,
    in_degrees: bool = True,
    out_degrees: bool = True,
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """
    Calculate reciprocal unit cell parameters from direct cell parameters.

    Description
    -----------
    Computes reciprocal lattice parameters (a*, b*, c*, α*, β*, γ*) from
    direct lattice parameters using crystallographic relationships.

    Parameters
    ----------
    a : scalar_float
        Direct cell length a in Angstroms
    b : scalar_float
        Direct cell length b in Angstroms
    c : scalar_float
        Direct cell length c in Angstroms
    alpha : scalar_float
        Direct cell angle α (between b and c axes)
    beta : scalar_float
        Direct cell angle β (between a and c axes)
    gamma : scalar_float
        Direct cell angle γ (between a and b axes)
    in_degrees : bool
        If True, input angles are in degrees. Default: True
    out_degrees : bool
        If True, output angles are in degrees. Default: True

    Returns
    -------
    reciprocal_lengths : Float[Array, " 3"]
        Reciprocal cell lengths [a*, b*, c*] in 1/Angstroms
    reciprocal_angles : Float[Array, " 3"]
        Reciprocal cell angles [α*, β*, γ*] in degrees or radians

    Flow
    ----
    - Convert input angles to radians if needed
    - Calculate unit cell volume using triple product formula
    - Compute reciprocal lengths using volume relationships
    - Calculate reciprocal angles using crystallographic formulas
    - Convert output angles to degrees if requested
    """
    alpha_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(alpha), alpha
    )
    beta_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(beta), beta
    )
    gamma_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(gamma), gamma
    )
    cos_alpha: Float[Array, " "] = jnp.cos(alpha_rad)
    cos_beta: Float[Array, " "] = jnp.cos(beta_rad)
    cos_gamma: Float[Array, " "] = jnp.cos(gamma_rad)
    sin_alpha: Float[Array, " "] = jnp.sin(alpha_rad)
    sin_beta: Float[Array, " "] = jnp.sin(beta_rad)
    sin_gamma: Float[Array, " "] = jnp.sin(gamma_rad)
    volume_squared: Float[Array, " "] = (
        1.0
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    )
    volume: Float[Array, " "] = a * b * c * jnp.sqrt(volume_squared)
    a_star: Float[Array, " "] = 2.0 * jnp.pi * b * c * sin_alpha / volume
    b_star: Float[Array, " "] = 2.0 * jnp.pi * a * c * sin_beta / volume
    c_star: Float[Array, " "] = 2.0 * jnp.pi * a * b * sin_gamma / volume
    cos_alpha_star: Float[Array, " "] = (cos_beta * cos_gamma - cos_alpha) / (
        sin_beta * sin_gamma
    )
    cos_beta_star: Float[Array, " "] = (cos_alpha * cos_gamma - cos_beta) / (
        sin_alpha * sin_gamma
    )
    cos_gamma_star: Float[Array, " "] = (cos_alpha * cos_beta - cos_gamma) / (
        sin_alpha * sin_beta
    )
    alpha_star_rad: Float[Array, " "] = jnp.arccos(
        jnp.clip(cos_alpha_star, -1.0, 1.0)
    )
    beta_star_rad: Float[Array, " "] = jnp.arccos(
        jnp.clip(cos_beta_star, -1.0, 1.0)
    )
    gamma_star_rad: Float[Array, " "] = jnp.arccos(
        jnp.clip(cos_gamma_star, -1.0, 1.0)
    )
    alpha_star: Float[Array, " "] = jnp.where(
        out_degrees, jnp.rad2deg(alpha_star_rad), alpha_star_rad
    )
    beta_star: Float[Array, " "] = jnp.where(
        out_degrees, jnp.rad2deg(beta_star_rad), beta_star_rad
    )
    gamma_star: Float[Array, " "] = jnp.where(
        out_degrees, jnp.rad2deg(gamma_star_rad), gamma_star_rad
    )

    reciprocal_lengths: Float[Array, " 3"] = jnp.array(
        [a_star, b_star, c_star]
    )
    reciprocal_angles: Float[Array, " 3"] = jnp.array(
        [alpha_star, beta_star, gamma_star]
    )

    return reciprocal_lengths, reciprocal_angles


@jaxtyped(typechecker=beartype)
def get_unit_cell_matrix(
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
    alpha: scalar_float,
    beta: scalar_float,
    gamma: scalar_float,
) -> Float[Array, " 3 3"]:
    r"""Build transformation matrix between direct and reciprocal space.

    Parameters
    ----------
    a, b, c : scalar_float
        Direct cell lengths in angstroms.
    alpha, beta, gamma : scalar_float
        Direct cell angles in degrees.

    Returns
    -------
    Float[Array, " 3 3"]
        Transformation matrix from direct to reciprocal space.

    Algorithm
    ---------
    - Calculate cell volume from lattice parameters
    - Calculate reciprocal lengths
    - Calculate transformation matrix elements
    - Return 3x3 transformation matrix

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Get transformation matrix for a cubic cell
    >>> matrix = get_unit_cell_matrix(
    ...     a=3.0, b=3.0, c=3.0,  # 3 Å cubic cell
    ...     alpha=90.0, beta=90.0, gamma=90.0
    ... )
    >>> print(f"Transformation matrix:\n{matrix}")
    >>>
    >>> # Transform a direct space vector to reciprocal space
    >>> direct_vec = jnp.array([1.0, 0.0, 0.0])
    >>> recip_vec = direct_vec @ matrix
    >>> print(f"Reciprocal vector: {recip_vec}")
    """
    alpha_rad: Float[Array, " "] = jnp.radians(alpha)
    beta_rad: Float[Array, " "] = jnp.radians(beta)
    gamma_rad: Float[Array, " "] = jnp.radians(gamma)
    cos_angles: Float[Array, " 3"] = jnp.array(
        [jnp.cos(alpha_rad), jnp.cos(beta_rad), jnp.cos(gamma_rad)]
    )
    sin_angles: Float[Array, " 3"] = jnp.array(
        [jnp.sin(alpha_rad), jnp.sin(beta_rad), jnp.sin(gamma_rad)]
    )
    volume_factor: Float[Array, " "] = jnp.sqrt(
        1 - jnp.sum(jnp.square(cos_angles)) + (2 * jnp.prod(cos_angles))
    )
    matrix: Float[Array, " 3 3"] = jnp.zeros(shape=(3, 3), dtype=jnp.float64)
    matrix = matrix.at[0, 0].set(a)
    matrix = matrix.at[0, 1].set(b * cos_angles[2])
    matrix = matrix.at[0, 2].set(c * cos_angles[1])
    matrix = matrix.at[1, 1].set(b * sin_angles[2])
    matrix = matrix.at[1, 2].set(
        c * (cos_angles[0] - cos_angles[1] * cos_angles[2]) / sin_angles[2]
    )
    matrix_assigned: Float[Array, " 3 3"] = matrix.at[2, 2].set(
        c * volume_factor / sin_angles[2]
    )
    return matrix_assigned


@jaxtyped(typechecker=beartype)
def build_cell_vectors(
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
    alpha: scalar_float,
    beta: scalar_float,
    gamma: scalar_float,
) -> Float[Array, " 3 3"]:
    r"""Construct unit cell vectors from lengths and angles.

    Parameters
    ----------
    a, b, c : scalar_float
        Direct cell lengths in angstroms.
    alpha, beta, gamma : scalar_float
        Direct cell angles in degrees.

    Returns
    -------
    Float[Array, " 3 3"]
        Unit cell vectors as rows of 3x3 matrix.

    Algorithm
    ---------
    - Convert angles to radians
    - Calculate cosines of angles
    - Build first vector along x-axis
    - Build second vector in x-y plane
    - Build third vector using all angles
    - Return 3x3 matrix of vectors

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Build vectors for a cubic cell
    >>> vectors = build_cell_vectors(
    ...     a=3.0, b=3.0, c=3.0,  # 3 Å cubic cell
    ...     alpha=90.0, beta=90.0, gamma=90.0
    ... )
    >>> print(f"Cell vectors:\n{vectors}")
    >>>
    >>> # Calculate cell volume
    >>> volume = jnp.linalg.det(vectors)
    >>> print(f"Cell volume: {volume}")
    """
    alpha_rad: Float[Array, " "] = jnp.radians(alpha)
    beta_rad: Float[Array, " "] = jnp.radians(beta)
    gamma_rad: Float[Array, " "] = jnp.radians(gamma)
    a_vec: Float[Array, " 3"] = jnp.array([a, 0.0, 0.0])
    b_x: Float[Array, " "] = b * jnp.cos(gamma_rad)
    b_y: Float[Array, " "] = b * jnp.sin(gamma_rad)
    b_vec: Float[Array, " 3"] = jnp.array([b_x, b_y, 0.0])
    c_x: Float[Array, " "] = c * jnp.cos(beta_rad)
    c_y: Float[Array, " "] = c * (
        (jnp.cos(alpha_rad) - jnp.cos(beta_rad) * jnp.cos(gamma_rad))
        / jnp.sin(gamma_rad)
    )
    c_z_sq: Float[Array, " "] = (c**2) - (c_x**2) - (c_y**2)
    c_z: Float[Array, " "] = jnp.sqrt(jnp.clip(c_z_sq, a_min=0.0))
    c_vec: Float[Array, " 3"] = jnp.array([c_x, c_y, c_z])
    cell_vectors: Float[Array, " 3 3"] = jnp.stack(
        [a_vec, b_vec, c_vec], axis=0
    )
    return cell_vectors


@jaxtyped(typechecker=beartype)
def compute_lengths_angles(
    vectors: Float[Array, " 3 3"],
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """Compute unit cell lengths and angles from lattice vectors.

    Parameters
    ----------
    vectors : Float[Array, " 3 3"]
        Unit cell vectors as rows of 3x3 matrix.

    Returns
    -------
    lengths : Float[Array, " 3"]
        Unit cell lengths [a, b, c] in angstroms
    angles : Float[Array, " 3"]
        Unit cell angles [α, β, γ] in degrees

    Algorithm
    ---------
    - Calculate lengths of each vector
    - Calculate angle between b and c vectors (α)
    - Calculate angle between a and c vectors (β)
    - Calculate angle between a and b vectors (γ)
    - Convert angles to degrees
    - Return lengths and angles

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Create some cell vectors
    >>> vectors = jnp.array([
    ...     [3.0, 0.0, 0.0],  # a vector
    ...     [0.0, 3.0, 0.0],  # b vector
    ...     [0.0, 0.0, 3.0]   # c vector
    ... ])
    >>>
    >>> # Compute lengths and angles
    >>> lengths, angles = compute_lengths_angles(vectors)
    >>> print(f"Cell lengths: {lengths}")
    >>> print(f"Cell angles: {angles}")
    """
    lengths: Float[Array, " 3"] = jnp.linalg.norm(vectors, axis=1)

    a_vec: Float[Array, " 3"] = vectors[0]
    b_vec: Float[Array, " 3"] = vectors[1]
    c_vec: Float[Array, " 3"] = vectors[2]

    cos_alpha: Float[Array, " "] = jnp.dot(b_vec, c_vec) / (
        lengths[1] * lengths[2]
    )
    cos_beta: Float[Array, " "] = jnp.dot(a_vec, c_vec) / (
        lengths[0] * lengths[2]
    )
    cos_gamma: Float[Array, " "] = jnp.dot(a_vec, b_vec) / (
        lengths[0] * lengths[1]
    )

    alpha_rad: Float[Array, " "] = jnp.arccos(jnp.clip(cos_alpha, -1.0, 1.0))
    beta_rad: Float[Array, " "] = jnp.arccos(jnp.clip(cos_beta, -1.0, 1.0))
    gamma_rad: Float[Array, " "] = jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))

    angles: Float[Array, " 3"] = jnp.array(
        [jnp.rad2deg(alpha_rad), jnp.rad2deg(beta_rad), jnp.rad2deg(gamma_rad)]
    )

    return lengths, angles


@jaxtyped(typechecker=beartype)
def generate_reciprocal_points(
    crystal: CrystalStructure,
    hmax: scalar_int,
    kmax: scalar_int,
    lmax: scalar_int,
    in_degrees: bool = True,
) -> Float[Array, " M 3"]:
    r"""Generate reciprocal-lattice vectors based on the crystal structure.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to generate points for.
    hmax, kmax, lmax : scalar_int
        Maximum h, k, l indices to generate.
    in_degrees : bool, optional
        Whether to use degrees for angles. Default: True.

    Returns
    -------
    Float[Array, " M 3"]
        Reciprocal lattice vectors in 1/angstroms.

    Algorithm
    ---------
    - Get cell parameters from crystal structure
    - Generate reciprocal lattice vectors directly from direct cell
    - Generate h, k, l indices
    - Transform indices to reciprocal space using miller_to_reciprocal
    - Return reciprocal vectors

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Load crystal structure from CIF
    >>> crystal = parse_cif("path/to/crystal.cif")
    >>>
    >>> # Generate reciprocal points up to (2,2,1)
    >>> G_vectors = generate_reciprocal_points(
    ...     crystal=crystal,
    ...     hmax=2,
    ...     kmax=2,
    ...     lmax=1
    ... )
    >>> print(f"Number of G vectors: {len(G_vectors)}")
    >>> print(f"First few G vectors:\n{G_vectors[:5]}")
    """
    abc: Num[Array, " 3"] = crystal.cell_lengths
    angles: Num[Array, " 3"] = crystal.cell_angles

    a: Float[Array, " "] = abc[0]
    b: Float[Array, " "] = abc[1]
    c: Float[Array, " "] = abc[2]
    alpha: Float[Array, " "] = angles[0]
    beta: Float[Array, " "] = angles[1]
    gamma: Float[Array, " "] = angles[2]

    rec_vectors: Float[Array, " 3 3"] = reciprocal_lattice_vectors(
        a, b, c, alpha, beta, gamma, in_degrees=in_degrees
    )

    hs: Int[Array, " n_h"] = jnp.arange(-hmax, hmax + 1)
    ks: Int[Array, " n_k"] = jnp.arange(-kmax, kmax + 1)
    ls: Int[Array, " n_l"] = jnp.arange(-lmax, lmax + 1)

    hh: Int[Array, " n_h n_k n_l"]
    kk: Int[Array, " n_h n_k n_l"]
    ll: Int[Array, " n_h n_k n_l"]
    hh, kk, ll = jnp.meshgrid(hs, ks, ls, indexing="ij")

    hkl: Int[Array, " M 3"] = jnp.stack(
        [hh.ravel(), kk.ravel(), ll.ravel()], axis=-1
    )

    g_vectors: Float[Array, " M 3"] = miller_to_reciprocal(hkl, rec_vectors)

    return g_vectors


@jaxtyped(typechecker=beartype)
def atom_scraper(
    crystal: CrystalStructure,
    zone_axis: Float[Array, " 3"],
    thickness: Float[Array, " 3"],
) -> CrystalStructure:
    """Filter atoms within specified thickness along zone axis.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to filter.
    zone_axis : Float[Array, " 3"]
        Zone axis direction.
    thickness : Float[Array, " 3"]
        Thickness in each direction.

    Returns
    -------
    filtered_crystal : CrystalStructure
        Filtered crystal structure.

    Algorithm
    ---------
    - Normalize zone axis
    - Calculate distances along zone axis
    - Filter atoms within thickness
    - Create new crystal structure with filtered atoms
    - Return filtered crystal

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Load crystal structure
    >>> crystal = parse_cif("path/to/crystal.cif")
    >>>
    >>> # Filter atoms within 12 Å along [111] direction
    >>> filtered = atom_scraper(
    ...     crystal=crystal,
    ...     zone_axis=jnp.array([1.0, 1.0, 1.0]),
    ...     thickness=jnp.array([12.0, 12.0, 12.0])
    ... )
    >>> print(f"Original atoms: {len(crystal.frac_positions)}")
    >>> print(f"Filtered atoms: {len(filtered.frac_positions)}")
    """
    orig_cell_vectors: Float[Array, " 3 3"] = build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    zone_axis_norm: Float[Array, " "] = jnp.linalg.norm(zone_axis)
    zone_axis_hat: Float[Array, " 3"] = zone_axis / (zone_axis_norm + 1e-32)
    cart_xyz: Float[Array, " n 3"] = crystal.cart_positions[:, :3]
    dot_vals: Float[Array, " n"] = jnp.einsum(
        "ij,j->i", cart_xyz, zone_axis_hat
    )
    d_max: Float[Array, " "] = jnp.max(dot_vals)
    dist_from_top: Float[Array, " n"] = d_max - dot_vals
    distance_cutoff: Float[Array, " "] = 1e-8
    positive_distances: Float[Array, " m"] = dist_from_top[
        dist_from_top > distance_cutoff
    ]
    adaptive_eps: Float[Array, " "] = jnp.where(
        positive_distances.size > 0,
        jnp.maximum(1e-3, 2 * jnp.min(positive_distances)),
        1e-3,
    )
    is_top_layer_mode: Bool[Array, " "] = jnp.isclose(
        thickness, jnp.asarray(0.0), atol=1e-8
    )
    mask: Bool[Array, " n"] = jnp.where(
        is_top_layer_mode,
        dist_from_top <= adaptive_eps,
        dist_from_top <= thickness,
    )

    def _gather_valid_positions(
        positions: Float[Array, " n 4"], gather_mask: Bool[Array, " n"]
    ) -> Float[Array, " m 4"]:
        return positions[gather_mask]

    filtered_frac: Float[Array, " m 4"] = _gather_valid_positions(
        crystal.frac_positions, mask
    )
    filtered_cart: Float[Array, " m 4"] = _gather_valid_positions(
        crystal.cart_positions, mask
    )
    original_height: Float[Array, " "] = jnp.max(dot_vals) - jnp.min(dot_vals)
    new_height: Float[Array, " "] = jnp.where(
        is_top_layer_mode,
        adaptive_eps,
        jnp.minimum(thickness, original_height),
    )

    def _scale_vector(
        vec: Float[Array, " 3"],
        zone_axis_hat: Float[Array, " 3"],
        old_height: Float[Array, " "],
        new_height: Float[Array, " "],
    ) -> Float[Array, " 3"]:
        height_cutoff: Float[Array, " "] = 1e-32
        proj_mag: Float[Array, " "] = jnp.dot(vec, zone_axis_hat)
        parallel_comp: Float[Array, " 3"] = proj_mag * zone_axis_hat
        perp_comp: Float[Array, " 3"] = vec - parallel_comp
        scale_factor: Float[Array, " "] = jnp.where(
            old_height < height_cutoff, 1.0, new_height / old_height
        )
        scaled_parallel: Float[Array, " 3"] = scale_factor * parallel_comp
        return scaled_parallel + perp_comp

    def _scale_if_needed(
        vec: Float[Array, " 3"],
        zone_axis_hat: Float[Array, " 3"],
        original_height: Float[Array, " "],
        new_height: Float[Array, " "],
    ) -> Float[Array, " 3"]:
        needs_scaling: Bool[Array, " "] = (
            jnp.abs(jnp.dot(vec, zone_axis_hat)) > distance_cutoff
        )
        scaled: Float[Array, " 3"] = _scale_vector(
            vec, zone_axis_hat, original_height, new_height
        )
        return jnp.where(needs_scaling, scaled, vec)

    scaled_vectors: Float[Array, " 3 3"] = jnp.stack(
        [
            _scale_if_needed(
                orig_cell_vectors[i],
                zone_axis_hat,
                original_height,
                new_height,
            )
            for i in range(3)
        ]
    )
    new_lengths: Float[Array, " 3"]
    new_angles: Float[Array, " 3"]
    new_lengths, new_angles = compute_lengths_angles(scaled_vectors)
    filtered_crystal: CrystalStructure = create_crystal_structure(
        frac_positions=filtered_frac,
        cart_positions=filtered_cart,
        cell_lengths=new_lengths,
        cell_angles=new_angles,
    )
    return filtered_crystal


@jaxtyped(typechecker=beartype)
def reciprocal_lattice_vectors(
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
    alpha: scalar_float,
    beta: scalar_float,
    gamma: scalar_float,
    in_degrees: bool = True,
) -> Float[Array, " 3 3"]:
    """Generate reciprocal lattice basis vectors b₁, b₂, b₃.

    Description
    -----------
    Computes the three reciprocal lattice basis vectors from direct lattice
    parameters using the crystallographic relationships:
    b₁ = 2π(a₂ × a₃)/(a₁ · (a₂ × a₃))
    b₂ = 2π(a₃ × a₁)/(a₁ · (a₂ × a₃))
    b₃ = 2π(a₁ × a₂)/(a₁ · (a₂ × a₃))

    Parameters
    ----------
    a : scalar_float
        Direct cell length a in Angstroms
    b : scalar_float
        Direct cell length b in Angstroms
    c : scalar_float
        Direct cell length c in Angstroms
    alpha : scalar_float
        Direct cell angle α (between b and c axes)
    beta : scalar_float
        Direct cell angle β (between a and c axes)
    gamma : scalar_float
        Direct cell angle γ (between a and b axes)
    in_degrees : bool
        If True, input angles are in degrees. Default: True

    Returns
    -------
    reciprocal_vectors : Float[Array, " 3 3"]
        Reciprocal lattice vectors as rows of 3x3 matrix in 1/Angstroms.
        Each row is a reciprocal basis vector [b₁, b₂, b₃]

    Flow
    ----
    - Convert angles to radians if needed
    - Build direct lattice vectors using build_cell_vectors
    - Extract individual direct vectors a₁, a₂, a₃
    - Calculate unit cell volume using triple product
    - Compute cross products for each reciprocal vector
    - Scale by 2π/volume to get final reciprocal vectors
    - Stack vectors into 3x3 matrix
    """
    alpha_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(alpha), alpha
    )
    beta_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(beta), beta
    )
    gamma_rad: Float[Array, " "] = jnp.where(
        in_degrees, jnp.deg2rad(gamma), gamma
    )

    direct_vectors: Float[Array, " 3 3"] = build_cell_vectors(
        a,
        b,
        c,
        jnp.rad2deg(alpha_rad),
        jnp.rad2deg(beta_rad),
        jnp.rad2deg(gamma_rad),
    )

    a_vec: Float[Array, " 3"] = direct_vectors[0]
    b_vec: Float[Array, " 3"] = direct_vectors[1]
    c_vec: Float[Array, " 3"] = direct_vectors[2]

    cross_b_c: Float[Array, " 3"] = jnp.cross(b_vec, c_vec)
    cross_c_a: Float[Array, " 3"] = jnp.cross(c_vec, a_vec)
    cross_a_b: Float[Array, " 3"] = jnp.cross(a_vec, b_vec)

    volume: Float[Array, " "] = jnp.dot(a_vec, cross_b_c)

    two_pi: Float[Array, " "] = 2.0 * jnp.pi
    scale_factor: Float[Array, " "] = two_pi / volume

    b1_vec: Float[Array, " 3"] = scale_factor * cross_b_c
    b2_vec: Float[Array, " 3"] = scale_factor * cross_c_a
    b3_vec: Float[Array, " 3"] = scale_factor * cross_a_b

    reciprocal_vectors: Float[Array, " 3 3"] = jnp.stack(
        [b1_vec, b2_vec, b3_vec], axis=0
    )

    return reciprocal_vectors


@jaxtyped(typechecker=beartype)
def miller_to_reciprocal(
    hkl: Int[Array, "... 3"],
    reciprocal_vectors: Float[Array, " 3 3"],
) -> Float[Array, "... 3"]:
    """Convert Miller indices to reciprocal space vectors.

    Description
    -----------
    Transforms Miller indices (h,k,l) to reciprocal space vectors G
    using the reciprocal lattice basis vectors. Each reciprocal vector
    is computed as G = h*b₁ + k*b₂ + l*b₃ where b₁, b₂, b₃ are the
    reciprocal lattice basis vectors.

    Parameters
    ----------
    hkl : Int[Array, "... 3"]
        Miller indices with shape (..., 3) where the last dimension
        contains [h, k, l] values. Can be a single set of indices or
        a batch of multiple indices.
    reciprocal_vectors : Float[Array, " 3 3"]
        Reciprocal lattice basis vectors as rows of 3x3 matrix in
        1/Angstroms, as returned by reciprocal_lattice_vectors function

    Returns
    -------
    g_vectors : Float[Array, "... 3"]
        Reciprocal space vectors in 1/Angstroms with same batch shape
        as input hkl indices

    Flow
    ----
    - Cast Miller indices to float for computation
    - Extract reciprocal basis vectors b₁, b₂, b₃
    - Extract h, k, l components from input
    - Compute linear combination h*b₁ + k*b₂ + l*b₃
    - Use einsum for efficient batched computation
    """
    hkl_float: Float[Array, "... 3"] = jnp.asarray(hkl, dtype=jnp.float64)

    b1_vec: Float[Array, " 3"] = reciprocal_vectors[0]
    b2_vec: Float[Array, " 3"] = reciprocal_vectors[1]
    b3_vec: Float[Array, " 3"] = reciprocal_vectors[2]

    h_component: Float[Array, " ..."] = hkl_float[..., 0]
    k_component: Float[Array, " ..."] = hkl_float[..., 1]
    l_component: Float[Array, " ..."] = hkl_float[..., 2]

    h_contribution: Float[Array, "... 3"] = (
        h_component[..., jnp.newaxis] * b1_vec
    )
    k_contribution: Float[Array, " ... 3"] = (
        k_component[..., jnp.newaxis] * b2_vec
    )
    l_contribution: Float[Array, " ... 3"] = (
        l_component[..., jnp.newaxis] * b3_vec
    )

    g_vectors: Float[Array, " ... 3"] = (
        h_contribution + k_contribution + l_contribution
    )

    return g_vectors
