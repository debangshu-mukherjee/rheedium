"""Differentiable surface modifiers for RHEED forward models.

Extended Summary
----------------
This module provides both structure-space and pattern-space surface
modifiers. Structure-space APIs perturb an ideal slab in a
differentiable way before simulation, while pattern-space APIs model
reduced effective behavior such as step-induced rod splitting or mixed
surface domains.

Routine Listings
----------------
:func:`apply_step_edge_field`
    Apply a smooth periodic step-edge displacement field to the top
    surface of a slab.
:func:`apply_surface_occupancy_field`
    Attenuate the effective atomic numbers of atoms near the surface.
:func:`apply_surface_displacement_field`
    Apply per-atom displacement vectors to the top surface region.
:func:`step_edge_to_distribution`
    Convert step-edge population metadata into a generic Distribution.
:func:`twin_wall_to_distribution`
    Convert twin-wall population metadata into a generic Distribution.
:func:`vicinal_surface_step_splitting`
    Compute CTR intensity modification due to a periodic step array.
:func:`incoherent_domain_average`
    Compute the incoherently averaged RHEED pattern from multiple
    independent surface domains.

Notes
-----
All public APIs are implemented with pure JAX operations. The
structure-space modifiers keep the cell fixed and update only the
position or effective-occupancy fields in the returned
``CrystalStructure``.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    Distribution,
    ReductionMode,
    create_crystal_structure,
    create_distribution,
    reduction_mode_from_coherence_length,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors

_DEFAULT_STEP_DIRECTION: Float[Array, "2"] = jnp.array([1.0, 0.0])


def _surface_depth_weights(
    slab: CrystalStructure,
    surface_layer_depth_angstrom: scalar_float,
    transition_sharpness: scalar_float,
) -> Float[Array, "N_atoms"]:
    """Return a differentiable gate that selects the top surface region."""
    top_surface_z: Float[Array, ""] = jnp.max(slab.cart_positions[:, 2])
    depth_from_surface: Float[Array, "N_atoms"] = (
        top_surface_z - slab.cart_positions[:, 2]
    )
    return jax.nn.sigmoid(
        jnp.asarray(transition_sharpness, dtype=jnp.float64)
        * (
            jnp.asarray(surface_layer_depth_angstrom, dtype=jnp.float64)
            - depth_from_surface
        )
    )


def _rebuild_surface_structure(
    slab: CrystalStructure,
    cart_xyz: Float[Array, "N_atoms 3"],
    effective_atomic_numbers: Float[Array, "N_atoms"],
) -> CrystalStructure:
    """Recompute fractional coordinates after a surface-space update."""
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        *slab.cell_lengths, *slab.cell_angles
    )
    inv_cell_vectors: Float[Array, "3 3"] = jnp.linalg.inv(cell_vectors)
    frac_xyz: Float[Array, "N_atoms 3"] = cart_xyz @ inv_cell_vectors.T
    frac_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [frac_xyz, effective_atomic_numbers]
    )
    cart_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [cart_xyz, effective_atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=slab.cell_lengths,
        cell_angles=slab.cell_angles,
    )


@jaxtyped(typechecker=beartype)
def apply_step_edge_field(
    slab: CrystalStructure,
    step_height_angstrom: scalar_float,
    terrace_width_angstrom: scalar_float,
    surface_layer_depth_angstrom: scalar_float,
    step_direction_xy: Float[Array, "2"] = _DEFAULT_STEP_DIRECTION,
    edge_sharpness: scalar_float = 12.0,
    transition_sharpness: scalar_float = 40.0,
) -> CrystalStructure:
    """Apply a smooth periodic step-edge field to a surface slab.

    :see: :class:`~.test_surface_modifier.TestApplyStepEdgeField`

    Parameters
    ----------
    slab : CrystalStructure
        Input surface slab.
    step_height_angstrom : scalar_float
        Peak-to-peak height difference between neighboring terraces.
    terrace_width_angstrom : scalar_float
        Mean terrace width along the in-plane step progression
        direction.
    surface_layer_depth_angstrom : scalar_float
        Depth of the surface region affected by the step-edge field.
    step_direction_xy : Float[Array, "2"], optional
        In-plane direction along which terraces repeat. Default:
        ``[1, 0]``.
    edge_sharpness : scalar_float, optional
        Controls how step-like the periodic terrace profile is. Default:
        12.0
    transition_sharpness : scalar_float, optional
        Controls how sharply the modifier is restricted to the top
        surface region. Default: 40.0

    Returns
    -------
    step_modified_slab : CrystalStructure
        Slab with the same cell and effective atomic numbers, but with a
        smooth terrace-height modulation applied near the surface.

    Notes
    -----
    1. **Build a soft surface gate** --
       Identify the top surface region with a sigmoid depth gate.
    2. **Project onto the terrace direction** --
       Use the in-plane atomic coordinates and the normalized
       ``step_direction_xy`` vector.
    3. **Generate a smooth periodic step profile** --
       Apply ``0.5 * tanh(edge_sharpness * sin(phase))`` so adjacent
       terraces differ smoothly by approximately one step height.
    4. **Apply a vertical displacement** --
       Multiply the profile by the surface gate and add it only to the
       Cartesian ``z`` coordinate.

    The modifier preserves atom count and cell geometry, which keeps the
    API shape-stable under ``jax.jit`` and ``jax.vmap``.
    """
    surface_weights: Float[Array, "N_atoms"] = _surface_depth_weights(
        slab=slab,
        surface_layer_depth_angstrom=surface_layer_depth_angstrom,
        transition_sharpness=transition_sharpness,
    )
    in_plane_direction: Float[Array, "2"] = jnp.asarray(
        step_direction_xy, dtype=jnp.float64
    )
    normalized_direction: Float[Array, "2"] = in_plane_direction / (
        jnp.linalg.norm(in_plane_direction) + 1e-10
    )
    projected_coordinate: Float[Array, "N_atoms"] = jnp.dot(
        slab.cart_positions[:, :2], normalized_direction
    )
    phase: Float[Array, "N_atoms"] = (
        2.0
        * jnp.pi
        * projected_coordinate
        / (jnp.asarray(terrace_width_angstrom, dtype=jnp.float64) + 1e-10)
    )
    terrace_profile: Float[Array, "N_atoms"] = 0.5 * jnp.tanh(
        jnp.asarray(edge_sharpness, dtype=jnp.float64) * jnp.sin(phase)
    )
    z_shift: Float[Array, "N_atoms"] = (
        jnp.asarray(step_height_angstrom, dtype=jnp.float64)
        * surface_weights
        * terrace_profile
    )
    displaced_cart_xyz: Float[Array, "N_atoms 3"] = (
        slab.cart_positions[:, :3].at[:, 2].add(z_shift)
    )
    return _rebuild_surface_structure(
        slab=slab,
        cart_xyz=displaced_cart_xyz,
        effective_atomic_numbers=slab.cart_positions[:, 3],
    )


@jaxtyped(typechecker=beartype)
def apply_surface_occupancy_field(
    slab: CrystalStructure,
    surface_layer_depth_angstrom: scalar_float,
    site_occupancies: Float[Array, "N_atoms"],
    transition_sharpness: scalar_float = 40.0,
) -> CrystalStructure:
    """Apply a continuous occupancy field to the top surface region.

    :see: :class:`~.test_surface_modifier.TestApplySurfaceOccupancyField`

    Parameters
    ----------
    slab : CrystalStructure
        Input surface slab.
    surface_layer_depth_angstrom : scalar_float
        Depth of the surface region affected by the occupancy field.
    site_occupancies : Float[Array, "N_atoms"]
        Per-atom occupancy values. Values are clipped to ``[0, 1]`` and
        are applied progressively to atoms near the surface.
    transition_sharpness : scalar_float, optional
        Sharpness of the depth gate used to localize the effect to the
        top surface region. Default: 40.0

    Returns
    -------
    occupancy_modified_slab : CrystalStructure
        Slab with unchanged coordinates and cell, but with surface-local
        effective atomic numbers scaled by the occupancy field.

    Notes
    -----
    1. **Build a soft surface gate** --
       Compute a sigmoid depth weight from the topmost atomic layer.
    2. **Clip occupancies** --
       Restrict the supplied per-atom occupancies to ``[0, 1]``.
    3. **Blend interior and surface behavior** --
       Interior atoms retain unit occupancy while top-surface atoms move
       toward the supplied values.
    4. **Return a new slab** --
       Preserve geometry and update only the effective atomic-number
       column.
    """
    surface_weights: Float[Array, "N_atoms"] = _surface_depth_weights(
        slab=slab,
        surface_layer_depth_angstrom=surface_layer_depth_angstrom,
        transition_sharpness=transition_sharpness,
    )
    clipped_occupancies: Float[Array, "N_atoms"] = jnp.clip(
        jnp.asarray(site_occupancies, dtype=jnp.float64), 0.0, 1.0
    )
    occupancy_scale: Float[Array, "N_atoms"] = 1.0 + surface_weights * (
        clipped_occupancies - 1.0
    )
    effective_atomic_numbers: Float[Array, "N_atoms"] = (
        slab.cart_positions[:, 3] * occupancy_scale
    )
    return _rebuild_surface_structure(
        slab=slab,
        cart_xyz=slab.cart_positions[:, :3],
        effective_atomic_numbers=effective_atomic_numbers,
    )


@jaxtyped(typechecker=beartype)
def apply_surface_displacement_field(
    slab: CrystalStructure,
    surface_layer_depth_angstrom: scalar_float,
    atomic_displacements: Float[Array, "N_atoms 3"],
    transition_sharpness: scalar_float = 40.0,
) -> CrystalStructure:
    """Apply per-atom displacement vectors to the top surface region.

    :see: :class:`~.test_surface_modifier.TestApplySurfaceDisplacementField`

    Parameters
    ----------
    slab : CrystalStructure
        Input surface slab.
    surface_layer_depth_angstrom : scalar_float
        Depth of the surface region affected by the displacement field.
    atomic_displacements : Float[Array, "N_atoms 3"]
        Per-atom displacement vectors in Angstroms.
    transition_sharpness : scalar_float, optional
        Sharpness of the depth gate used to localize the effect to the
        top surface region. Default: 40.0

    Returns
    -------
    displaced_slab : CrystalStructure
        Slab with displaced Cartesian coordinates and recomputed
        fractional coordinates.

    Notes
    -----
    1. **Build a soft surface gate** --
       Compute a differentiable depth weight for each atom.
    2. **Apply the vector field** --
       Multiply each displacement vector by the surface gate so interior
       atoms remain unchanged.
    3. **Recompute fractional coordinates** --
       Convert the updated Cartesian positions back into the fixed slab
       cell.
    """
    surface_weights: Float[Array, "N_atoms"] = _surface_depth_weights(
        slab=slab,
        surface_layer_depth_angstrom=surface_layer_depth_angstrom,
        transition_sharpness=transition_sharpness,
    )
    displaced_cart_xyz: Float[Array, "N_atoms 3"] = slab.cart_positions[
        :, :3
    ] + surface_weights[:, None] * jnp.asarray(
        atomic_displacements, dtype=jnp.float64
    )
    return _rebuild_surface_structure(
        slab=slab,
        cart_xyz=displaced_cart_xyz,
        effective_atomic_numbers=slab.cart_positions[:, 3],
    )


@jaxtyped(typechecker=beartype)
def twin_wall_to_distribution(
    twin_angles_deg: Float[Array, "N"],
    wall_positions_angstrom: Float[Array, "M"],
    twin_fractions: Float[Array, "K"],
    twin_spacing_angstrom: scalar_float,
    coherence_length_angstrom: scalar_float,
    axis_id: str | None = "twins",
) -> Distribution:
    """Convert twin-wall metadata to a generic Distribution.

    :see: :class:`~.test_surface_modifier.TestTwinWallToDistribution`

    Parameters
    ----------
    twin_angles_deg : Float[Array, "N"]
        Twin rotation or shear-angle samples in degrees.
    wall_positions_angstrom : Float[Array, "M"]
        Representative twin-wall positions in Angstroms.
    twin_fractions : Float[Array, "K"]
        Non-negative population weights for each twin sample.
    twin_spacing_angstrom : scalar_float
        Characteristic spacing between twin walls in Angstroms.
    coherence_length_angstrom : scalar_float
        Beam-mode transverse coherence length in Angstroms.
    axis_id : str | None, optional
        Optional static label for the returned distribution. Default:
        ``"twins"``.

    Returns
    -------
    distribution : Distribution
        Generic distribution with samples
        ``[twin_angle_deg, wall_position_angstrom]`` and a coherence-derived
        reduction mode.

    Notes
    -----
    1. Validate one-to-one twin angle, position, and fraction arrays.
    2. Choose coherent reduction when wall spacing fits inside the coherent
       footprint.
    3. Delegate probability validation and normalization to
       :func:`create_distribution`.
    """
    twin_angles: Float[Array, "N"] = jnp.asarray(
        twin_angles_deg,
        dtype=jnp.float64,
    )
    wall_positions: Float[Array, "M"] = jnp.asarray(
        wall_positions_angstrom,
        dtype=jnp.float64,
    )
    fractions: Float[Array, "K"] = jnp.asarray(
        twin_fractions,
        dtype=jnp.float64,
    )
    if twin_angles.ndim != 1:
        raise ValueError("twin_angles_deg must have shape (N,)")
    if wall_positions.ndim != 1:
        raise ValueError("wall_positions_angstrom must have shape (N,)")
    if fractions.ndim != 1:
        raise ValueError("twin_fractions must have shape (N,)")
    if twin_angles.shape[0] != wall_positions.shape[0]:
        raise ValueError(
            "twin_angles_deg and wall_positions_angstrom must share length"
        )
    if twin_angles.shape[0] != fractions.shape[0]:
        raise ValueError(
            "twin_angles_deg and twin_fractions must share length"
        )

    samples: Float[Array, "N 2"] = jnp.stack(
        [twin_angles, wall_positions],
        axis=-1,
    )
    reduction: ReductionMode = reduction_mode_from_coherence_length(
        feature_length_angstrom=twin_spacing_angstrom,
        coherence_length_angstrom=coherence_length_angstrom,
    )
    return create_distribution(
        samples=samples,
        weights=fractions,
        reduction=reduction,
        axis_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def step_edge_to_distribution(
    step_heights_angstrom: Float[Array, "N"],
    terrace_widths_angstrom: Float[Array, "M"],
    line_azimuths_deg: Float[Array, "K"],
    step_fractions: Float[Array, "L"],
    coherence_length_angstrom: scalar_float,
    regular: bool = True,
    axis_id: str | None = "steps",
) -> Distribution:
    """Convert step-edge metadata to a generic Distribution.

    :see: :class:`~.test_surface_modifier.TestStepEdgeToDistribution`

    Parameters
    ----------
    step_heights_angstrom : Float[Array, "N"]
        Step-height samples in Angstroms.
    terrace_widths_angstrom : Float[Array, "M"]
        Terrace-width samples in Angstroms.
    line_azimuths_deg : Float[Array, "K"]
        Step-line azimuth samples in degrees.
    step_fractions : Float[Array, "L"]
        Non-negative population weights for each step sample.
    coherence_length_angstrom : scalar_float
        Beam-mode transverse coherence length in Angstroms.
    regular : bool, optional
        If True, choose coherent reduction when the terrace period is inside
        the coherence length. If False, random steps reduce incoherently.
        Default: True.
    axis_id : str | None, optional
        Optional static label for the returned distribution. Default:
        ``"steps"``.

    Returns
    -------
    distribution : Distribution
        Generic distribution with samples
        ``[step_height_angstrom, terrace_width_angstrom, line_azimuth_deg]``.

    Notes
    -----
    1. Validate one-to-one step metadata arrays.
    2. Encode regular terraces as coherent only when the mean terrace width
       fits inside the coherent footprint.
    3. Encode random step populations as incoherent mixtures.
    """
    heights: Float[Array, "N"] = jnp.asarray(
        step_heights_angstrom,
        dtype=jnp.float64,
    )
    widths: Float[Array, "M"] = jnp.asarray(
        terrace_widths_angstrom,
        dtype=jnp.float64,
    )
    azimuths: Float[Array, "K"] = jnp.asarray(
        line_azimuths_deg,
        dtype=jnp.float64,
    )
    fractions: Float[Array, "L"] = jnp.asarray(
        step_fractions,
        dtype=jnp.float64,
    )
    if heights.ndim != 1:
        raise ValueError("step_heights_angstrom must have shape (N,)")
    if widths.ndim != 1:
        raise ValueError("terrace_widths_angstrom must have shape (N,)")
    if azimuths.ndim != 1:
        raise ValueError("line_azimuths_deg must have shape (N,)")
    if fractions.ndim != 1:
        raise ValueError("step_fractions must have shape (N,)")
    if heights.shape[0] != widths.shape[0]:
        raise ValueError(
            "step_heights_angstrom and terrace_widths_angstrom must share "
            "length"
        )
    if heights.shape[0] != azimuths.shape[0]:
        raise ValueError(
            "step_heights_angstrom and line_azimuths_deg must share length"
        )
    if heights.shape[0] != fractions.shape[0]:
        raise ValueError(
            "step_heights_angstrom and step_fractions must share length"
        )

    samples: Float[Array, "N 3"] = jnp.stack(
        [heights, widths, azimuths],
        axis=-1,
    )
    reduction: ReductionMode = ReductionMode.INCOHERENT
    if regular:
        mean_terrace_width: float = float(jnp.mean(widths))
        reduction = reduction_mode_from_coherence_length(
            feature_length_angstrom=mean_terrace_width,
            coherence_length_angstrom=coherence_length_angstrom,
        )
    return create_distribution(
        samples=samples,
        weights=fractions,
        reduction=reduction,
        axis_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def vicinal_surface_step_splitting(
    hk_index: Int[Array, "2"],
    step_height_angstrom: scalar_float,
    terrace_width_angstrom: scalar_float,
    q_z: Float[Array, "N_qz"],
) -> Float[Array, "N_qz"]:
    """Compute CTR intensity modification from periodic steps.

    :see: :class:`~.test_surface_modifier.TestVicinalSurfaceStepSplitting`

    Parameters
    ----------
    hk_index : Int[Array, "2"]
        In-plane Miller indices ``(h, k)`` of the rod.
    step_height_angstrom : scalar_float
        Single step height in Angstroms.
    terrace_width_angstrom : scalar_float
        Mean terrace width in Angstroms.
    q_z : Float[Array, "N_qz"]
        Perpendicular momentum transfer in inverse Angstroms.

    Returns
    -------
    step_modified_intensity : Float[Array, "N_qz"]
        CTR intensity profile modified by step interference and
        normalized to unit maximum.

    Notes
    -----
    1. **Build the phase difference** --
       Compute ``delta_phi = q_z * step_height``.
    2. **Use an Airy-style interference form** --
       Model the regular terrace array as
       ``1 / (1 + F * sin(delta_phi / 2)**2)`` with
       ``F = 4 * (w / d)**2``.
    3. **Normalize** --
       Divide by the maximum value so the result peaks at one.
    """
    del hk_index  # Reserved for future rod-dependent step models.

    step_height_angstrom = jnp.asarray(step_height_angstrom, dtype=jnp.float64)
    terrace_width_angstrom = jnp.asarray(
        terrace_width_angstrom, dtype=jnp.float64
    )

    delta_phi: Float[Array, "N_qz"] = q_z * step_height_angstrom
    sin_sq: Float[Array, "N_qz"] = jnp.sin(delta_phi / 2.0) ** 2
    finesse: Float[Array, ""] = (
        4.0 * (terrace_width_angstrom / (step_height_angstrom + 1e-10)) ** 2
    )
    intensity: Float[Array, "N_qz"] = 1.0 / (1.0 + finesse * sin_sq)
    return intensity / (jnp.max(intensity) + 1e-10)


@jaxtyped(typechecker=beartype)
def incoherent_domain_average(
    domain_patterns: Float[Array, "N_domains H W"],
    domain_volume_fractions: Float[Array, "N_domains"],
) -> Float[Array, "H W"]:
    """Compute an incoherently averaged RHEED pattern from domains.

    :see: :class:`~.test_surface_modifier.TestIncoherentDomainAverage`

    Parameters
    ----------
    domain_patterns : Float[Array, "N_domains H W"]
        Independent pattern for each domain orientation or termination.
    domain_volume_fractions : Float[Array, "N_domains"]
        Domain fractions. They are renormalized internally.

    Returns
    -------
    mixed_pattern : Float[Array, "H W"]
        Intensity-weighted average ``sum_i(f_i * I_i)``.

    Notes
    -----
    1. **Normalize fractions** --
       Divide by the total domain weight.
    2. **Broadcast and sum** --
       Reshape to ``(N_domains, 1, 1)`` and sum over the domain axis.

    Domains scatter independently in this reduced model, so intensities
    add incoherently rather than amplitudes.
    """
    fraction_sum: Float[Array, ""] = jnp.sum(domain_volume_fractions)
    normalized_fractions: Float[Array, "N_domains"] = (
        domain_volume_fractions / (fraction_sum + 1e-10)
    )
    weights: Float[Array, "N_domains 1 1"] = normalized_fractions[
        :, None, None
    ]
    return jnp.sum(weights * domain_patterns, axis=0)


__all__: list[str] = [
    "apply_step_edge_field",
    "apply_surface_displacement_field",
    "apply_surface_occupancy_field",
    "incoherent_domain_average",
    "step_edge_to_distribution",
    "twin_wall_to_distribution",
    "vicinal_surface_step_splitting",
]
