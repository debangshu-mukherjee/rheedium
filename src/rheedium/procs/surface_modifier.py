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
:func:`apply_vicinal_staircase`
    Apply a smooth monotonic differentiable vicinal staircase height
    field along an in-plane direction.
:func:`apply_surface_occupancy_field`
    Attenuate the site occupancies of atoms near the surface.
:func:`apply_surface_displacement_field`
    Apply per-atom displacement vectors to the top surface region.
:func:`apply_twin_wall_field`
    Apply a smooth twin-wall rotation field to the top surface.
:func:`bind_step_edge_distribution`
    Bind step-edge samples to per-sample structure builders.
:func:`bind_twin_wall_distribution`
    Bind twin-wall samples to per-sample structure builders.
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
position or first-class ``occupancies`` fields in the returned
``CrystalStructure``; atomic numbers stay integral.

R5 return type: structure-space modifiers return ``CrystalStructure``;
statistical step/twin population producers return ``Distribution``; bind
helpers return closures that apply sub-coherence disorder to a per-sample
structure before the Layer-1 reducer evaluates the detector kernel.
Pattern-space averaging helpers return detector-image arrays as compatibility
front ends over the shared reducer.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Complex, Float, Int, jaxtyped

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
_DEFAULT_TWIN_NORMAL: Float[Array, "2"] = jnp.array([1.0, 0.0])


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


def _slab_occupancies(slab: CrystalStructure) -> Float[Array, "N_atoms"]:
    """Return the slab's occupancies, defaulting to fully occupied."""
    if slab.occupancies is None:
        return jnp.ones(slab.cart_positions.shape[0], dtype=jnp.float64)
    return jnp.asarray(slab.occupancies, dtype=jnp.float64)


def _rebuild_surface_structure(
    slab: CrystalStructure,
    cart_xyz: Float[Array, "N_atoms 3"],
    atomic_numbers: Float[Array, "N_atoms"],
    occupancies: Float[Array, "N_atoms"] | None = None,
) -> CrystalStructure:
    """Recompute fractional coordinates after a surface-space update.

    When ``occupancies`` is None, the slab's own occupancies are
    carried through unchanged (ones when the slab has none).
    """
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        *slab.cell_lengths, *slab.cell_angles
    )
    inv_cell_vectors: Float[Array, "3 3"] = jnp.linalg.inv(cell_vectors)
    frac_xyz: Float[Array, "N_atoms 3"] = cart_xyz @ inv_cell_vectors
    frac_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [frac_xyz, atomic_numbers]
    )
    cart_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [cart_xyz, atomic_numbers]
    )
    new_occupancies: Float[Array, "N_atoms"] = (
        _slab_occupancies(slab) if occupancies is None else occupancies
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=slab.cell_lengths,
        cell_angles=slab.cell_angles,
        occupancies=new_occupancies,
    )


@jaxtyped(typechecker=beartype)
def _unit_vector_from_azimuth_deg(
    azimuth_deg: scalar_float,
) -> Float[Array, "2"]:
    """Return an in-plane unit vector from an azimuth in degrees."""
    azimuth_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(azimuth_deg, dtype=jnp.float64)
    )
    return jnp.array([jnp.cos(azimuth_rad), jnp.sin(azimuth_rad)])


@jaxtyped(typechecker=beartype)
def apply_step_edge_field(
    slab: CrystalStructure,
    step_height_angstrom: scalar_float,
    corrugation_period_ang: scalar_float,
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
    corrugation_period_ang : scalar_float
        Spatial period of the two-level ``tanh`` corrugation along the
        step-progression direction. This is a *full period* of the
        up/down terrace pattern, so each terrace (one plateau) spans half
        of ``corrugation_period_ang``.
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
    in_plane_norm: Float[Array, ""] = jnp.linalg.norm(in_plane_direction)
    safe_in_plane_norm: Float[Array, ""] = jnp.where(
        in_plane_norm > 0.0, in_plane_norm, 1.0
    )
    normalized_direction: Float[Array, "2"] = jnp.where(
        in_plane_norm > 0.0,
        in_plane_direction / safe_in_plane_norm,
        in_plane_direction,
    )
    projected_coordinate: Float[Array, "N_atoms"] = jnp.dot(
        slab.cart_positions[:, :2], normalized_direction
    )
    phase: Float[Array, "N_atoms"] = (
        2.0
        * jnp.pi
        * projected_coordinate
        / (jnp.asarray(corrugation_period_ang, dtype=jnp.float64) + 1e-10)
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
        atomic_numbers=slab.cart_positions[:, 3],
    )


@jaxtyped(typechecker=beartype)
def apply_vicinal_staircase(
    surface: CrystalStructure,
    terrace_width_ang: scalar_float,
    step_height_ang: scalar_float,
    step_direction_xy: Float[Array, "2"] = _DEFAULT_STEP_DIRECTION,
) -> CrystalStructure:
    r"""Apply a smooth monotonic vicinal staircase height field.

    :see: :class:`~.test_surface_modifier.TestApplyVicinalStaircase`

    Parameters
    ----------
    surface : CrystalStructure
        Input surface slab.
    terrace_width_ang : scalar_float
        Terrace width ``w`` in Angstroms (one full period of the
        staircase corresponds to one terrace-and-step).
    step_height_ang : scalar_float
        Step height ``s`` in Angstroms (descent per terrace).
    step_direction_xy : Float[Array, "2"], optional
        In-plane direction of the staircase descent. Default ``[1, 0]``.

    Returns
    -------
    staircased : CrystalStructure
        Slab whose Cartesian ``z`` is offset by the staircase height
        field.

    Notes
    -----
    1. **Height profile** -- with ``x`` the in-plane coordinate along
       ``step_direction_xy``, the displacement is the smooth, monotonic,
       differentiable staircase

       .. math::

          h(x) = -s \left[ \frac{x}{w}
                 - \frac{\sin(2\pi x / w)}{2\pi} \right].

    2. **Monotonic descent** -- the derivative
       ``h'(x) = -(s/w)(1 - cos(2\pi x / w)) <= 0`` is flat mid-terrace
       and steepest at each step, so the surface never rises.
    3. **Exact averages** -- the mean slope over any whole number of
       terraces is exactly ``-s/w`` and the height drop over five
       terraces is exactly ``5 s``; ``h`` is differentiable in ``w`` and
       ``s`` everywhere, so it composes with ``jax.grad``.
    """
    direction: Float[Array, "2"] = jnp.asarray(
        step_direction_xy, dtype=jnp.float64
    )
    direction_norm: Float[Array, ""] = jnp.linalg.norm(direction)
    safe_norm: Float[Array, ""] = jnp.where(
        direction_norm > 0.0, direction_norm, 1.0
    )
    unit_direction: Float[Array, "2"] = jnp.where(
        direction_norm > 0.0, direction / safe_norm, direction
    )
    projected: Float[Array, "N_atoms"] = jnp.dot(
        surface.cart_positions[:, :2], unit_direction
    )
    width: Float[Array, ""] = jnp.asarray(terrace_width_ang, dtype=jnp.float64)
    height: Float[Array, ""] = jnp.asarray(step_height_ang, dtype=jnp.float64)
    phase: Float[Array, "N_atoms"] = 2.0 * jnp.pi * projected / width
    z_shift: Float[Array, "N_atoms"] = -height * (
        projected / width - jnp.sin(phase) / (2.0 * jnp.pi)
    )
    displaced_cart_xyz: Float[Array, "N_atoms 3"] = (
        surface.cart_positions[:, :3].at[:, 2].add(z_shift)
    )
    return _rebuild_surface_structure(
        slab=surface,
        cart_xyz=displaced_cart_xyz,
        atomic_numbers=surface.cart_positions[:, 3],
    )


@jaxtyped(typechecker=beartype)
def apply_twin_wall_field(
    slab: CrystalStructure,
    twin_angle_deg: scalar_float,
    wall_position_angstrom: scalar_float,
    surface_layer_depth_angstrom: scalar_float,
    wall_normal_xy: Float[Array, "2"] = _DEFAULT_TWIN_NORMAL,
    wall_width_angstrom: scalar_float = 1.0,
    transition_sharpness: scalar_float = 40.0,
) -> CrystalStructure:
    """Apply a smooth twin-wall rotation field to a surface slab.

    :see: :class:`~.test_surface_modifier.TestApplyTwinWallField`

    Parameters
    ----------
    slab : CrystalStructure
        Input surface slab.
    twin_angle_deg : scalar_float
        Relative in-plane twin rotation angle in degrees across the wall.
    wall_position_angstrom : scalar_float
        Signed wall position along ``wall_normal_xy`` in Angstroms.
    surface_layer_depth_angstrom : scalar_float
        Depth of the surface region affected by the twin field.
    wall_normal_xy : Float[Array, "2"], optional
        In-plane normal direction of the twin wall. Default: ``[1, 0]``.
    wall_width_angstrom : scalar_float, optional
        Smooth transition width across the wall in Angstroms. Default: 1.0.
    transition_sharpness : scalar_float, optional
        Sharpness of the depth gate selecting the top surface. Default: 40.0.

    Returns
    -------
    twin_modified_slab : CrystalStructure
        Slab with a smooth, shape-stable in-plane twin rotation near the
        surface.

    Notes
    -----
    1. Build a surface depth gate.
    2. Build a smooth wall-side coordinate using ``tanh``.
    3. Rotate surface atoms about the slab in-plane centroid by a signed
       fraction of ``twin_angle_deg``.
    4. Rebuild fractional coordinates in the original cell.
    """
    surface_weights: Float[Array, "N_atoms"] = _surface_depth_weights(
        slab=slab,
        surface_layer_depth_angstrom=surface_layer_depth_angstrom,
        transition_sharpness=transition_sharpness,
    )
    normal: Float[Array, "2"] = jnp.asarray(wall_normal_xy, dtype=jnp.float64)
    normalized_normal: Float[Array, "2"] = normal / (
        jnp.linalg.norm(normal) + 1e-10
    )
    wall_coordinate: Float[Array, "N_atoms"] = jnp.dot(
        slab.cart_positions[:, :2], normalized_normal
    ) - jnp.asarray(wall_position_angstrom, dtype=jnp.float64)
    width: Float[Array, ""] = jnp.maximum(
        jnp.asarray(wall_width_angstrom, dtype=jnp.float64),
        1e-10,
    )
    side_weight: Float[Array, "N_atoms"] = jnp.tanh(wall_coordinate / width)
    local_angle_rad: Float[Array, "N_atoms"] = (
        surface_weights
        * side_weight
        * 0.5
        * jnp.deg2rad(jnp.asarray(twin_angle_deg, dtype=jnp.float64))
    )
    centroid_xy: Float[Array, "2"] = jnp.mean(
        slab.cart_positions[:, :2],
        axis=0,
    )
    relative_xy: Float[Array, "N_atoms 2"] = (
        slab.cart_positions[:, :2] - centroid_xy
    )
    cos_angle: Float[Array, "N_atoms"] = jnp.cos(local_angle_rad)
    sin_angle: Float[Array, "N_atoms"] = jnp.sin(local_angle_rad)
    rotated_x: Float[Array, "N_atoms"] = (
        cos_angle * relative_xy[:, 0] - sin_angle * relative_xy[:, 1]
    )
    rotated_y: Float[Array, "N_atoms"] = (
        sin_angle * relative_xy[:, 0] + cos_angle * relative_xy[:, 1]
    )
    rotated_xy: Float[Array, "N_atoms 2"] = (
        jnp.stack([rotated_x, rotated_y], axis=-1) + centroid_xy
    )
    displaced_cart_xyz: Float[Array, "N_atoms 3"] = (
        slab.cart_positions[:, :3].at[:, :2].set(rotated_xy)
    )
    return _rebuild_surface_structure(
        slab=slab,
        cart_xyz=displaced_cart_xyz,
        atomic_numbers=slab.cart_positions[:, 3],
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
        Slab with unchanged coordinates, cell, and atomic numbers, but
        with surface-local site ``occupancies`` scaled by the occupancy
        field.

    Notes
    -----
    1. **Build a soft surface gate** --
       Compute a sigmoid depth weight from the topmost atomic layer.
    2. **Clip occupancies** --
       Restrict the supplied per-atom occupancies to ``[0, 1]``.
    3. **Blend interior and surface behavior** --
       Interior atoms retain their existing occupancy while top-surface
       atoms move toward the supplied values; the resulting scale
       multiplies the slab's occupancies (ones when absent).
    4. **Return a new slab** --
       Preserve geometry and atomic numbers; only the first-class
       ``occupancies`` field changes, which every simulation kernel
       applies as a per-atom form-factor weight.
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
    new_occupancies: Float[Array, "N_atoms"] = (
        _slab_occupancies(slab) * occupancy_scale
    )
    return _rebuild_surface_structure(
        slab=slab,
        cart_xyz=slab.cart_positions[:, :3],
        atomic_numbers=slab.cart_positions[:, 3],
        occupancies=new_occupancies,
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
        atomic_numbers=slab.cart_positions[:, 3],
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
def bind_twin_wall_distribution(
    slab: CrystalStructure,
    surface_layer_depth_angstrom: scalar_float,
    wall_normal_xy: Float[Array, "2"] = _DEFAULT_TWIN_NORMAL,
    wall_width_angstrom: scalar_float = 1.0,
    transition_sharpness: scalar_float = 40.0,
) -> Callable[[Float[Array, "2"]], CrystalStructure]:
    """Bind twin-wall samples to a structure-building closure.

    :see: :class:`~.test_surface_modifier.TestBindTwinWallDistribution`

    Parameters
    ----------
    slab : CrystalStructure
        Reference surface slab.
    surface_layer_depth_angstrom : scalar_float
        Depth affected by each twin sample.
    wall_normal_xy : Float[Array, "2"], optional
        Twin-wall normal direction. Default: ``[1, 0]``.
    wall_width_angstrom : scalar_float, optional
        Smooth transition width. Default: 1.0.
    transition_sharpness : scalar_float, optional
        Surface depth-gate sharpness. Default: 40.0.

    Returns
    -------
    builder : Callable[[Float[Array, "2"]], CrystalStructure]
        Closure mapping ``[twin_angle_deg, wall_position_angstrom]`` samples
        to modified slabs.

    Notes
    -----
    The returned closure is intended to run inside a Layer-1 amplitude closure,
    so defect samples can modify the structure before the coherent kernel call.
    """

    def _builder(sample: Float[Array, "2"]) -> CrystalStructure:
        return apply_twin_wall_field(
            slab=slab,
            twin_angle_deg=sample[0],
            wall_position_angstrom=sample[1],
            surface_layer_depth_angstrom=surface_layer_depth_angstrom,
            wall_normal_xy=wall_normal_xy,
            wall_width_angstrom=wall_width_angstrom,
            transition_sharpness=transition_sharpness,
        )

    return _builder


@jaxtyped(typechecker=beartype)
def bind_step_edge_distribution(
    slab: CrystalStructure,
    surface_layer_depth_angstrom: scalar_float,
    edge_sharpness: scalar_float = 12.0,
    transition_sharpness: scalar_float = 40.0,
) -> Callable[[Float[Array, "3"]], CrystalStructure]:
    """Bind step-edge samples to a structure-building closure.

    :see: :class:`~.test_surface_modifier.TestBindStepEdgeDistribution`

    Parameters
    ----------
    slab : CrystalStructure
        Reference surface slab.
    surface_layer_depth_angstrom : scalar_float
        Depth affected by each step sample.
    edge_sharpness : scalar_float, optional
        Smoothness of the periodic step profile. Default: 12.0.
    transition_sharpness : scalar_float, optional
        Surface depth-gate sharpness. Default: 40.0.

    Returns
    -------
    builder : Callable[[Float[Array, "3"]], CrystalStructure]
        Closure mapping
        ``[step_height_angstrom, terrace_width_angstrom, line_azimuth_deg]``
        samples to modified slabs.

    Notes
    -----
    The line azimuth sample is converted to the step-progression direction
    consumed by :func:`apply_step_edge_field`.
    """

    def _builder(sample: Float[Array, "3"]) -> CrystalStructure:
        return apply_step_edge_field(
            slab=slab,
            step_height_angstrom=sample[0],
            corrugation_period_ang=sample[1],
            surface_layer_depth_angstrom=surface_layer_depth_angstrom,
            step_direction_xy=_unit_vector_from_azimuth_deg(sample[2]),
            edge_sharpness=edge_sharpness,
            transition_sharpness=transition_sharpness,
        )

    return _builder


@jaxtyped(typechecker=beartype)
def vicinal_surface_step_splitting(
    step_height_angstrom: scalar_float,
    terrace_width_angstrom: scalar_float,
    q_z: Float[Array, "N_qz"],
) -> Float[Array, "N_qz"]:
    """Compute CTR intensity modification from periodic steps.

    :see: :class:`~.test_surface_modifier.TestVicinalSurfaceStepSplitting`

    Parameters
    ----------
    step_height_angstrom : scalar_float
        Single step height ``d`` in Angstroms.
    terrace_width_angstrom : scalar_float
        Mean terrace width ``w`` in Angstroms.
    q_z : Float[Array, "N_qz"]
        Perpendicular momentum transfer in inverse Angstroms.

    Returns
    -------
    step_modified_intensity : Float[Array, "N_qz"]
        CTR intensity profile modified by step interference, normalized
        by the analytic zero-splitting (in-phase) peak so the result is
        independent of the ``q_z`` sampling grid.

    Notes
    -----
    1. **Build the phase difference** --
       Compute ``delta_phi = q_z * d``.
    2. **Airy-style interference form** --
       Model the regular terrace array as
       ``1 / (1 + F * sin(delta_phi / 2)**2)``.
    3. **Grid-independent normalization** --
       The analytic in-phase (zero-splitting) peak is ``1`` (at
       ``sin(delta_phi / 2) = 0``), so the profile is already normalized
       to that closed-form maximum rather than to the discrete
       per-grid maximum. Two different ``q_z`` grids therefore return the
       same value at any shared ``q_z``.

    The ``hk_index`` parameter of earlier versions was unused and has been
    removed (red-team finding N12): this reduced model has no in-plane rod
    dependence. The finesse ``F = 4 (w / d)**2`` is an *uncalibrated
    heuristic* controlling peak sharpness; it is not fit to a dynamical
    reference and should be treated as qualitative.
    """
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
    return intensity


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
    The pattern bank is bound to an incoherent generic
    :class:`~rheedium.types.Distribution` and reduced by the shared Layer-1
    reducer. Domains scatter independently in this reduced model, so
    intensities add incoherently rather than amplitudes.
    """
    clipped_fractions: Float[Array, "N_domains"] = jnp.clip(
        jnp.asarray(domain_volume_fractions, dtype=jnp.float64),
        0.0,
        None,
    )
    fraction_sum: Float[Array, ""] = jnp.sum(clipped_fractions)
    has_positive_weight: Float[Array, ""] = jnp.where(
        fraction_sum > 0.0,
        1.0,
        0.0,
    )
    safe_fractions: Float[Array, "N_domains"] = jnp.where(
        fraction_sum > 0.0,
        clipped_fractions,
        jnp.ones_like(clipped_fractions),
    )
    sample_indices: Float[Array, "N_domains 1"] = jnp.arange(
        domain_patterns.shape[0],
        dtype=jnp.float64,
    )[:, None]
    distribution: Distribution = create_distribution(
        samples=sample_indices,
        weights=safe_fractions,
        reduction=ReductionMode.INCOHERENT,
        axis_id="surface_domain_pattern",
    )

    def _domain_amplitude(
        sample: Float[Array, "1"],
    ) -> Complex[Array, "H W"]:
        pattern_index: Int[Array, ""] = sample[0].astype(jnp.int32)
        pattern: Float[Array, "H W"] = domain_patterns[pattern_index]
        return jnp.sqrt(jnp.maximum(pattern, 0.0)).astype(jnp.complex128)

    from rheedium.simul.beam_averaging import apply_distributions

    mixed_pattern: Float[Array, "H W"] = apply_distributions(
        (distribution,),
        _domain_amplitude,
    )
    return has_positive_weight * mixed_pattern


__all__: list[str] = [
    "apply_step_edge_field",
    "apply_surface_displacement_field",
    "apply_surface_occupancy_field",
    "apply_twin_wall_field",
    "apply_vicinal_staircase",
    "bind_step_edge_distribution",
    "bind_twin_wall_distribution",
    "incoherent_domain_average",
    "step_edge_to_distribution",
    "twin_wall_to_distribution",
    "vicinal_surface_step_splitting",
]
