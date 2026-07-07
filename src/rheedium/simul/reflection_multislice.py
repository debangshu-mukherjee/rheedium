"""Reflection-geometry multislice simulation for RHEED.

Extended Summary
----------------
This module implements an edge-on multislice model in which the electron wave
propagates along an in-plane beam axis and is resolved on the transverse
``(y, z)`` plane. The reflected RHEED pattern is read from the up-going vacuum
field above the surface. This is the package's dynamical RHEED path: the
public ``simulate_detector_image(render=RenderParams(kernel="multislice"))``
orchestrator routes here, not to the transmission-geometry z-kernel in
:mod:`rheedium.simul.simulator`.

Routine Listings
----------------
:func:`crystal_to_edge_on_slices`
    Convert a crystal slab into edge-on projected potential slices.
:func:`reflection_multislice_propagate`
    Propagate a transverse wavefield through edge-on slices.
:func:`reflection_detector_amplitude`
    Render reflected channel amplitudes onto a dense detector field.
:func:`reflection_multislice_simulator`
    Simulate a sparse reflected RHEED pattern from a crystal slab.

Notes
-----
The first public version supports the required milestone geometry
``phi_deg == 0``: the beam travels along ``+x``, ``y`` is periodic, and ``z``
is the open surface-normal direction with absorbing layers at both edges.

The mean inner potential of the crystal is already contained in the projected
potential slices, so surface refraction emerges from the propagation itself;
no explicit inner-potential correction is applied (or needed) anywhere in
this module.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.tools import interaction_constant, wavelength_ang
from rheedium.types import (
    CrystalStructure,
    DetectorGeometry,
    EdgeOnSlices,
    RHEEDPattern,
    create_edge_on_slices,
    create_rheed_pattern,
    scalar_float,
    scalar_num,
)

from .form_factors import projected_potential
from .simulator import project_on_detector_geometry, render_amplitude_to_field

_FILL_FRACTION: float = 0.8
"""Fraction of the geometric fill height trusted by the vacuum read-off."""


def _require_phi_zero(phi_deg: scalar_num) -> None:
    """Reject unsupported nonzero azimuths at the public boundary."""
    phi_value = float(phi_deg)
    if abs(phi_value) > 1e-12:
        raise NotImplementedError(
            "reflection multislice currently supports phi_deg=0 only"
        )


def _z_axis(slices: EdgeOnSlices) -> Float[Array, "nz"]:
    """Return the surface-normal coordinate axis for an edge-on grid."""
    nz = slices.slices.shape[2]
    return slices.z_lo + slices.dz * jnp.arange(nz, dtype=jnp.float64)


def _reciprocal_axis(
    n_points: int,
    spacing: scalar_float,
) -> Float[Array, "n"]:
    """Return a 2pi-carrying FFT reciprocal axis."""
    return (
        2.0
        * jnp.pi
        * jnp.fft.fftfreq(
            n_points,
            d=float(spacing),
        )
    )


def _beam_repeat_count(
    slices: EdgeOnSlices,
    propagation_length_ang: scalar_float,
) -> int:
    """Return how often the per-cell stack repeats along the beam.

    The per-cell stack spans ``nx_slices * dx_slice`` Angstroms along the
    beam; the stack is looped periodically until at least
    ``propagation_length_ang`` of crystal has been traversed.
    """
    stack_length = float(slices.slices.shape[0]) * float(slices.dx_slice)
    return max(1, int(math.ceil(float(propagation_length_ang) / stack_length)))


def _cap_mask(
    z_axis: Float[Array, "nz"],
    z_lo: scalar_float,
    z_hi: scalar_float,
    cap_width: scalar_float,
    dx_slice: scalar_float,
    cap_strength: scalar_float,
) -> Float[Array, "nz"]:
    """Build a quadratic damping mask for the open z boundaries."""
    lower_depth = jnp.maximum(0.0, z_lo + cap_width - z_axis)
    upper_depth = jnp.maximum(0.0, z_axis - (z_hi - cap_width))
    depth = jnp.maximum(lower_depth, upper_depth) / cap_width
    damping = cap_strength * depth**2
    return jnp.exp(-damping * dx_slice)


def _bandlimit_mask(
    ky_axis: Float[Array, "ny"],
    kz_axis: Float[Array, "nz"],
    bandwidth_limit: scalar_float,
) -> Float[Array, "ny nz"]:
    """Build a rectangular two-thirds style FFT bandwidth mask."""
    ky_cutoff = bandwidth_limit * jnp.max(jnp.abs(ky_axis))
    kz_cutoff = bandwidth_limit * jnp.max(jnp.abs(kz_axis))
    ky_ok = jnp.abs(ky_axis) <= ky_cutoff
    kz_ok = jnp.abs(kz_axis) <= kz_cutoff
    return (ky_ok[:, None] & kz_ok[None, :]).astype(jnp.float64)


def _flat_step_specular_reflectivity(
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return Fresnel reflectivity and a mask for a uniform potential step.

    This closed-form, parallel-momentum-conserving Fresnel step result is
    retained as an analytic *test oracle only*: the runtime read-off no
    longer blends it into the propagated specular channel, because genuine
    propagation over ``propagation_length_ang`` reproduces the Fresnel
    reflectivity (see ``test_physics_anchors.test_flat_step_fresnel``).
    """
    z_axis = _z_axis(slices)
    slab_mask = z_axis < slices.z_surf
    vacuum_mask = z_axis >= slices.z_surf
    slab_count = jnp.maximum(jnp.sum(slab_mask), 1)
    vacuum_count = jnp.maximum(jnp.sum(vacuum_mask), 1)
    slab_samples = jnp.where(slab_mask[None, None, :], slices.slices, 0.0)
    vacuum_samples = jnp.where(vacuum_mask[None, None, :], slices.slices, 0.0)
    projected_step = jnp.sum(slab_samples) / (
        slab_count * slices.slices.shape[0] * slices.slices.shape[1]
    )
    slab_deviation = jnp.max(
        jnp.abs(slab_samples - projected_step * slab_mask[None, None, :])
    )
    vacuum_level = jnp.max(jnp.abs(vacuum_samples))
    tolerance = 1e-8 * jnp.maximum(1.0, jnp.abs(projected_step))
    is_flat_step = (
        (projected_step > 0.0)
        & (slab_deviation <= tolerance)
        & (vacuum_level <= tolerance)
        & (slab_count > 0)
        & (vacuum_count > 0)
    )

    voltage_arr = jnp.asarray(energy_kev, dtype=jnp.float64)
    theta_rad = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float64))
    wavelength = wavelength_ang(voltage_arr)
    k_mag = 2.0 * jnp.pi / wavelength
    inner_potential = projected_step / slices.dx_slice
    sin_theta = jnp.sin(theta_rad)
    energy_v = voltage_arr * 1000.0
    k_perp_vac = k_mag * sin_theta
    k_perp_in = k_mag * jnp.sqrt(sin_theta**2 + inner_potential / energy_v)
    amplitude = (k_perp_vac - k_perp_in) / (k_perp_vac + k_perp_in)
    reflectivity = jnp.abs(amplitude) ** 2
    return reflectivity, is_flat_step.astype(jnp.float64)


def _deposit_edge_on_potentials(  # noqa: PLR0913
    positions: Float[Array, "n 3"],
    atomic_numbers: Int[Array, "n"],
    grid_shape: tuple[int, int, int],
    dx_value: float,
    dy_value: float,
    dz_value: float,
    l_x: float,
    l_y: float,
    z_lo: float,
    cutoff_value: float,
    parameterization: str,
    occupancies: Float[Array, "n"] | None = None,
) -> Float[Array, "nx ny nz"]:
    """Deposit projected atomic potentials onto an edge-on grid.

    Atom ``x`` coordinates are wrapped periodically into ``[0, l_x)`` before
    slice assignment (the beam axis is periodic over the cell), and the
    transverse ``y`` distance uses minimum-image wrapping over ``l_y``.
    Each atom's projected potential is scaled by its site occupancy
    (``occupancies``; ones when None).
    """
    nx_slices, ny, nz = grid_shape
    occupancy_weights = (
        jnp.ones(positions.shape[0], dtype=jnp.float64)
        if occupancies is None
        else jnp.asarray(occupancies, dtype=jnp.float64)
    )
    y_axis = dy_value * jnp.arange(ny, dtype=jnp.float64)
    z_axis = z_lo + dz_value * jnp.arange(nz, dtype=jnp.float64)
    yy = y_axis[:, None]
    zz = z_axis[None, :]
    x_wrapped = jnp.mod(positions[:, 0], l_x)
    slice_indices = jnp.clip(
        jnp.floor(x_wrapped / dx_value).astype(jnp.int32),
        0,
        nx_slices - 1,
    )

    def _add_atom_to_slice(
        accumulated: Float[Array, "nx ny nz"],
        atom_idx: Int[Array, ""],
    ) -> tuple[Float[Array, "nx ny nz"], None]:
        atom_position = positions[atom_idx]
        y_delta_raw = jnp.abs(yy - atom_position[1])
        y_delta = jnp.minimum(y_delta_raw, l_y - y_delta_raw)
        z_delta = zz - atom_position[2]
        # Double-where keeps the gradient finite when an atom sits exactly
        # on a grid point (sqrt has an infinite derivative at zero).
        radius_sq = y_delta**2 + z_delta**2
        radius_positive = radius_sq > 0.0
        radius = jnp.where(
            radius_positive,
            jnp.sqrt(jnp.where(radius_positive, radius_sq, 1.0)),
            0.0,
        )
        in_cutoff = radius <= cutoff_value
        potential = occupancy_weights[atom_idx] * projected_potential(
            atomic_numbers[atom_idx],
            radius,
            parameterization=parameterization,
        )
        contribution: Float[Array, "ny nz"] = jnp.where(
            in_cutoff,
            potential,
            0.0,
        )
        updated = accumulated.at[slice_indices[atom_idx]].add(contribution)
        return updated, None

    initial_slices: Float[Array, "nx ny nz"] = jnp.zeros(
        (nx_slices, ny, nz),
        dtype=jnp.float64,
    )
    atom_indices = jnp.arange(positions.shape[0], dtype=jnp.int32)
    slices, _ = jax.lax.scan(
        _add_atom_to_slice,
        initial_slices,
        atom_indices,
    )
    return slices


@jaxtyped(typechecker=beartype)
def crystal_to_edge_on_slices(
    crystal: CrystalStructure,
    *,
    phi_deg: scalar_num = 0.0,
    dx_slice: scalar_float = 1.0,
    dy: scalar_float = 0.25,
    dz: scalar_float = 0.25,
    vacuum_above: scalar_float = 30.0,
    cap_width: scalar_float = 15.0,
    penetration_depth: scalar_float | None = None,
    r_cutoff: scalar_float = 4.0,
    parameterization: str = "lobato",
) -> EdgeOnSlices:
    """Convert a crystal slab to edge-on projected-potential slices.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Parameters
    ----------
    crystal : CrystalStructure
        Slab whose surface normal is ``+z`` and beam direction is ``+x``.
    phi_deg : scalar_num, optional
        Beam azimuth in degrees. Only ``0.0`` is currently supported.
    dx_slice, dy, dz : scalar_float, optional
        Beam-axis and transverse grid spacings in Angstroms.
    vacuum_above : scalar_float, optional
        Vacuum read-off thickness above the surface, excluding the top CAP.
    cap_width : scalar_float, optional
        Absorbing-layer thickness at each z-window edge.
    penetration_depth : scalar_float | None, optional
        Physical slab depth below the surface to include. If None, include the
        full atomic z span.
    r_cutoff : scalar_float, optional
        Radial cutoff for each projected atomic potential contribution.
    parameterization : str, optional
        Atomic projected-potential model, ``"lobato"`` or ``"kirkland"``.

    Returns
    -------
    edge_on_slices : EdgeOnSlices
        Projected potentials on an ``(x-slice, y, z)`` grid.

    Notes
    -----
    1. Build the periodic ``y`` grid and open ``z`` grid from the slab bounds.
    2. Wrap each atom's ``x`` coordinate periodically into ``[0, l_x)`` and
       assign it to exactly one beam-axis ``x`` slice.
    3. Sum projected atomic potentials in the transverse plane, with
       minimum-image wrapping only along ``y``; each atom's potential is
       scaled by its site occupancy (``crystal.occupancies``, ones when
       absent).

    The slices carry the crystal's mean inner potential implicitly, so no
    separate inner-potential refraction parameter exists on this path.

    See Also
    --------
    reflection_multislice_propagate : Propagate through edge-on slices.
    """
    _require_phi_zero(phi_deg)
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")

    positions = crystal.cart_positions[:, :3]
    atomic_numbers = crystal.cart_positions[:, 3].astype(jnp.int32)
    occupancies = (
        jnp.ones(positions.shape[0], dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    l_x = float(crystal.cell_lengths[0])
    l_y = float(crystal.cell_lengths[1])
    dx_value = float(dx_slice)
    dy_value = float(dy)
    dz_value = float(dz)
    cap_value = float(cap_width)
    vacuum_value = float(vacuum_above)
    cutoff_value = float(r_cutoff)

    z_surf = float(jnp.max(positions[:, 2]))
    z_atoms_min = float(jnp.min(positions[:, 2]))
    depth = (
        z_surf - z_atoms_min
        if penetration_depth is None
        else float(penetration_depth)
    )
    z_bottom_phys = z_surf - depth
    z_lo = z_bottom_phys - cap_value
    z_hi = z_surf + vacuum_value + cap_value

    ny = max(1, int(jnp.round(l_y / dy_value)))
    nz = max(1, int(jnp.ceil((z_hi - z_lo) / dz_value)))
    nx_slices = max(1, int(jnp.ceil(l_x / dx_value)))
    actual_dx = l_x / nx_slices
    actual_dy = l_y / ny

    slices = _deposit_edge_on_potentials(
        positions=positions,
        atomic_numbers=atomic_numbers,
        grid_shape=(nx_slices, ny, nz),
        dx_value=actual_dx,
        dy_value=actual_dy,
        dz_value=dz_value,
        l_x=l_x,
        l_y=l_y,
        z_lo=z_lo,
        cutoff_value=cutoff_value,
        parameterization=parameterization,
        occupancies=occupancies,
    )
    return create_edge_on_slices(
        slices=slices,
        dx_slice=actual_dx,
        dy=actual_dy,
        dz=dz_value,
        y_extent=l_y,
        z_lo=z_lo,
        z_surf=z_surf,
        cap_width=cap_value,
    )


def _edge_on_slices_like(
    crystal: CrystalStructure,
    template: EdgeOnSlices,
    parameterization: str = "lobato",
    r_cutoff: scalar_float = 4.0,
) -> EdgeOnSlices:
    """Deposit a (possibly traced) crystal onto a fixed edge-on grid.

    The grid geometry (shape, spacings, window edges) is taken from a
    concrete ``template`` built once from the nominal crystal, so this
    function stays traceable when the atom positions are transformed by a
    distribution axis (twin walls, step edges) inside ``vmap``.
    """
    nx_slices, ny, nz = template.slices.shape
    dx_value = float(template.dx_slice)
    dy_value = float(template.dy)
    dz_value = float(template.dz)
    l_y = float(template.y_extent)
    l_x = nx_slices * dx_value
    z_lo = float(template.z_lo)
    positions = crystal.cart_positions[:, :3]
    atomic_numbers = crystal.cart_positions[:, 3].astype(jnp.int32)
    occupancies = (
        jnp.ones(positions.shape[0], dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    slices = _deposit_edge_on_potentials(
        positions=positions,
        atomic_numbers=atomic_numbers,
        grid_shape=(nx_slices, ny, nz),
        dx_value=dx_value,
        dy_value=dy_value,
        dz_value=dz_value,
        l_x=l_x,
        l_y=l_y,
        z_lo=z_lo,
        cutoff_value=float(r_cutoff),
        parameterization=parameterization,
        occupancies=occupancies,
    )
    return create_edge_on_slices(
        slices=slices,
        dx_slice=dx_value,
        dy=dy_value,
        dz=dz_value,
        y_extent=l_y,
        z_lo=z_lo,
        z_surf=float(template.z_surf),
        cap_width=float(template.cap_width),
    )


@jaxtyped(typechecker=beartype)
def reflection_multislice_propagate(  # noqa: PLR0913
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num = 0.0,
    cap_strength: scalar_float = 5.0,
    bandwidth_limit: scalar_float = 2 / 3,
    propagation_length_ang: scalar_float = 200.0,
) -> Complex[Array, "ny nz"]:
    """Propagate an edge-on RHEED wavefield through potential slices.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Parameters
    ----------
    slices : EdgeOnSlices
        Edge-on projected-potential slices.
    energy_kev : scalar_num
        Electron beam energy in kilovolts.
    theta_deg : scalar_num
        Grazing incidence angle in degrees.
    phi_deg : scalar_num, optional
        Beam azimuth in degrees. Only ``0.0`` is currently supported.
    cap_strength : scalar_float, optional
        Quadratic absorbing-mask strength in inverse Angstroms. Moderate
        values (default 5.0) absorb the transmitted beam with little
        spurious reflection off the absorber itself.
    bandwidth_limit : scalar_float, optional
        Fraction of the FFT Nyquist band to retain.
    propagation_length_ang : scalar_float, optional
        Total crystal length traversed along the beam in Angstroms. The
        per-cell slice stack is looped periodically
        ``ceil(propagation_length_ang / (nx_slices * dx_slice))`` times, so
        the wave interacts with a physically long crystal without
        materializing repeated copies of the stack.

    Returns
    -------
    wavefield : Complex[Array, "ny nz"]
        Final transverse wavefield after edge-on propagation.

    Notes
    -----
    1. Initialize a unit-amplitude downward grazing plane wave in vacuum.
    2. Apply ``exp(i sigma V_proj)`` for each beam-axis slice.
    3. Dampen the open z boundaries with a CAP mask.
    4. Advance the transverse field by a paraxial FFT propagator.
    5. Loop the per-cell stack until ``propagation_length_ang`` is reached.

    The reflected wave above the surface fills a wedge of height roughly
    ``propagation_length_ang * tan(theta)``; for a trustworthy read-off the
    vacuum window should satisfy
    ``vacuum_above > propagation_length_ang * tan(theta)`` so the incident
    plane wave feeding the surface never crosses the top CAP. The default
    30 Angstrom vacuum suits the default 200 Angstrom length for grazing
    angles up to about 4 degrees.

    See Also
    --------
    reflection_multislice_simulator : Read off reflected beams.
    """
    _require_phi_zero(phi_deg)
    initial, step_fn = _propagation_operators(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        cap_strength=cap_strength,
        bandwidth_limit=bandwidth_limit,
    )

    def _repeat_body(
        _repeat_idx: Int[Array, ""],
        wavefield: Complex[Array, "ny nz"],
    ) -> Complex[Array, "ny nz"]:
        propagated, _ = jax.lax.scan(step_fn, wavefield, slices.slices)
        return propagated

    n_repeats = _beam_repeat_count(slices, propagation_length_ang)
    return jax.lax.fori_loop(0, n_repeats, _repeat_body, initial)


def _propagation_operators(
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    cap_strength: scalar_float,
    bandwidth_limit: scalar_float,
) -> tuple[
    Complex[Array, "ny nz"],
    Callable[
        [Complex[Array, "ny nz"], Float[Array, "ny nz"]],
        tuple[Complex[Array, "ny nz"], None],
    ],
]:
    """Build the initial wave and one-slice step for edge-on propagation."""
    voltage_arr = jnp.asarray(energy_kev, dtype=jnp.float64)
    theta_rad = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float64))
    wavelength = wavelength_ang(voltage_arr)
    k_mag = 2.0 * jnp.pi / wavelength
    sigma = interaction_constant(voltage_arr, wavelength)
    k_z0 = -k_mag * jnp.sin(theta_rad)

    ny = slices.slices.shape[1]
    nz = slices.slices.shape[2]
    z_axis = _z_axis(slices)
    z_hi = slices.z_lo + slices.dz * nz
    ky_axis = _reciprocal_axis(ny, slices.dy)
    kz_axis = _reciprocal_axis(nz, slices.dz)
    ky_grid = ky_axis[:, None]
    kz_grid = kz_axis[None, :]
    propagator = jnp.exp(
        -1j * (slices.dx_slice / (2.0 * k_mag)) * (ky_grid**2 + kz_grid**2)
    )
    band = _bandlimit_mask(ky_axis, kz_axis, bandwidth_limit)
    cap = _cap_mask(
        z_axis=z_axis,
        z_lo=slices.z_lo,
        z_hi=z_hi,
        cap_width=slices.cap_width,
        dx_slice=slices.dx_slice,
        cap_strength=cap_strength,
    )
    initial = jnp.exp(1j * k_z0 * z_axis)[None, :] * jnp.ones(
        (ny, 1), dtype=jnp.complex128
    )

    def _step(
        wavefield: Complex[Array, "ny nz"],
        projected_slice: Float[Array, "ny nz"],
    ) -> tuple[Complex[Array, "ny nz"], None]:
        transmitted = wavefield * jnp.exp(1j * sigma * projected_slice)
        damped = transmitted * cap[None, :]
        propagated = jnp.fft.ifft2(jnp.fft.fft2(damped) * propagator * band)
        return propagated, None

    return initial, _step


def _propagate_and_read_channels(
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    cap_strength: scalar_float,
    bandwidth_limit: scalar_float,
    propagation_length_ang: scalar_float,
) -> tuple[
    Float[Array, "ny 3"],
    Complex[Array, "ny"],
    Bool[Array, "ny"],
]:
    """Propagate with a lock-in tail average of the channel read-off.

    The channel amplitudes are read at the end of every stack repeat over
    the second half of the run, de-rotated to the final plane by the exact
    vacuum paraxial phase ``exp(-i (k_y^2 + k_z^2) L_remaining / 2k)``, and
    averaged. Steady reflected channels add coherently under this
    de-rotation while start-up transients (which evolve with different
    paraxial rates) dephase and average out, which makes the read-off
    reflectivities stable against the propagation length.
    """
    initial, step_fn = _propagation_operators(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        cap_strength=cap_strength,
        bandwidth_limit=bandwidth_limit,
    )
    nx_slices = slices.slices.shape[0]
    ny = slices.slices.shape[1]
    n_repeats = _beam_repeat_count(slices, propagation_length_ang)
    first_averaged = n_repeats // 2
    n_averaged = n_repeats - first_averaged
    total_steps = n_repeats * nx_slices

    voltage_arr = jnp.asarray(energy_kev, dtype=jnp.float64)
    theta_rad = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float64))
    wavelength = wavelength_ang(voltage_arr)
    k_mag = 2.0 * jnp.pi / wavelength
    k_x0 = k_mag * jnp.cos(theta_rad)
    ky_axis = _reciprocal_axis(ny, slices.dy)
    kz_sq = k_mag**2 - k_x0**2 - ky_axis**2
    kz_up = jnp.sqrt(jnp.maximum(kz_sq, 0.0))
    paraxial_rate = (ky_axis**2 + kz_up**2) / (2.0 * k_mag)

    def _repeat_body(
        repeat_idx: Int[Array, ""],
        carry: tuple[Complex[Array, "ny nz"], Complex[Array, "ny"]],
    ) -> tuple[Complex[Array, "ny nz"], Complex[Array, "ny"]]:
        wavefield, accumulated = carry
        propagated, _ = jax.lax.scan(step_fn, wavefield, slices.slices)
        steps_done = (repeat_idx + 1) * nx_slices
        _, amplitudes, _ = _reflected_channel_amplitudes(
            wavefield=propagated,
            slices=slices,
            energy_kev=energy_kev,
            theta_deg=theta_deg,
            n_steps=steps_done,
        )
        remaining_length = (total_steps - steps_done) * slices.dx_slice
        aligned = amplitudes * jnp.exp(-1j * paraxial_rate * remaining_length)
        weight = jnp.where(repeat_idx >= first_averaged, 1.0, 0.0)
        return propagated, accumulated + weight * aligned

    accumulator = jnp.zeros((ny,), dtype=jnp.complex128)
    _, accumulated = jax.lax.fori_loop(
        0,
        n_repeats,
        _repeat_body,
        (initial, accumulator),
    )
    averaged = accumulated / n_averaged
    propagating = kz_sq > 0
    averaged = jnp.where(propagating, averaged, 0.0 + 0.0j)
    k_out = jnp.stack(
        [
            jnp.full_like(ky_axis, k_x0),
            ky_axis,
            kz_up,
        ],
        axis=-1,
    )
    return k_out, averaged, propagating


def _reflected_channel_amplitudes(
    wavefield: Complex[Array, "ny nz"],
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    n_steps: scalar_num,
) -> tuple[
    Float[Array, "ny 3"],
    Complex[Array, "ny"],
    Bool[Array, "ny"],
]:
    """Project the final vacuum wavefield onto up-going plane-wave channels.

    Returns the outgoing wavevectors, the complex channel amplitudes before
    the modulus-squared reduction, and the propagating-channel mask. Two
    read-off refinements keep the amplitudes quantitative:

    1. The analytically propagated incident plane wave (paraxial phase
       ``exp(-i n_steps dx k_z0^2 / 2k)``) is subtracted so the down-going
       incident field does not leak into the up-going projection.
    2. Each channel is projected over a near-surface window limited to the
       wedge the reflected wave has actually filled after ``n_steps`` slices
       (height ``~ 0.8 L tan(theta_out)``), because the transient read-off
       only reaches steady state within that wedge.
    """
    voltage_arr = jnp.asarray(energy_kev, dtype=jnp.float64)
    theta_rad = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float64))
    wavelength = wavelength_ang(voltage_arr)
    k_mag = 2.0 * jnp.pi / wavelength
    k_x0 = k_mag * jnp.cos(theta_rad)
    k_z0 = -k_mag * jnp.sin(theta_rad)

    ny = slices.slices.shape[1]
    nz = slices.slices.shape[2]
    dz_value = slices.dz
    z_axis = _z_axis(slices)
    z_hi = slices.z_lo + dz_value * nz
    total_length = n_steps * slices.dx_slice

    incident_phase = jnp.exp(
        -1j * n_steps * (slices.dx_slice / (2.0 * k_mag)) * k_z0**2
    )
    incident_wave = jnp.exp(1j * k_z0 * z_axis)[None, :] * incident_phase
    scattered = wavefield - incident_wave

    ky_axis = _reciprocal_axis(ny, slices.dy)
    kz_sq = k_mag**2 - k_x0**2 - ky_axis**2
    propagating = kz_sq > 0
    kz_up = jnp.sqrt(jnp.maximum(kz_sq, 0.0))

    read_margin = jnp.maximum(2.0 * dz_value, 0.5)
    window_lo = slices.z_surf + read_margin
    height_geom = (z_hi - slices.cap_width - read_margin) - window_lo
    min_height = 4.0 * dz_value
    tan_out = kz_up / jnp.maximum(k_x0, 1e-12)
    fill_height = _FILL_FRACTION * total_length * tan_out
    window_height = jnp.clip(
        fill_height,
        min_height,
        jnp.maximum(height_geom, min_height),
    )
    window_hi = window_lo + window_height
    z_mask = (z_axis[None, :] >= window_lo) & (
        z_axis[None, :] <= window_hi[:, None]
    )
    fallback_mask = (z_axis > slices.z_surf)[None, :] & jnp.ones(
        (ny, 1), dtype=bool
    )
    has_window = jnp.any(z_mask, axis=1, keepdims=True)
    z_mask = jnp.where(has_window, z_mask, fallback_mask)
    z_counts = jnp.maximum(jnp.sum(z_mask, axis=1), 1)

    y_fft = jnp.fft.fft(scattered, axis=0) / ny
    phase = jnp.exp(1j * kz_up[:, None] * z_axis[None, :])
    amplitudes = (
        jnp.sum(jnp.where(z_mask, jnp.conj(phase) * y_fft, 0.0), axis=1)
        / z_counts
    )
    amplitudes = jnp.where(propagating, amplitudes, 0.0 + 0.0j)
    k_out = jnp.stack(
        [
            jnp.full_like(ky_axis, k_x0),
            ky_axis,
            kz_up,
        ],
        axis=-1,
    )
    return k_out, amplitudes, propagating


@jaxtyped(typechecker=beartype)
def _reflection_amplitude_pattern(  # noqa: PLR0913
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    detector_distance_mm: scalar_float = 80.0,
    cap_strength: scalar_float = 5.0,
    bandwidth_limit: scalar_float = 2 / 3,
    propagation_length_ang: scalar_float = 200.0,
) -> tuple[RHEEDPattern, Complex[Array, "ny"]]:
    """Propagate edge-on slices and return the sparse reflected pattern.

    Returns both the pattern (with absolute ``|amplitude|^2`` read-off
    intensities) and the complex pre-modulus channel amplitudes so Layer 1
    can rasterize coherently with :func:`render_amplitude_to_field`. The
    amplitudes come from the lock-in tail-averaged read-off of
    :func:`_propagate_and_read_channels`.
    """
    k_out, amplitudes, propagating = _propagate_and_read_channels(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        cap_strength=cap_strength,
        bandwidth_limit=bandwidth_limit,
        propagation_length_ang=propagation_length_ang,
    )
    intensities = jnp.where(propagating, jnp.abs(amplitudes) ** 2, 0.0)
    detector_points = project_on_detector_geometry(
        k_out,
        DetectorGeometry(distance=detector_distance_mm),
    )
    ny = slices.slices.shape[1]
    channel_indices = jnp.arange(ny, dtype=jnp.int32)
    g_indices = jnp.where(propagating, channel_indices, -1)
    pattern = create_rheed_pattern(
        g_indices=g_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern, amplitudes


@jaxtyped(typechecker=beartype)
def reflection_detector_amplitude(  # noqa: PLR0913
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    detector_distance_mm: scalar_float = 80.0,
    image_shape_px: tuple[int, int] = (192, 192),
    pixel_size_mm: tuple[float, float] = (1.5, 3.0),
    beam_center_px: tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    cap_strength: scalar_float = 5.0,
    bandwidth_limit: scalar_float = 2 / 3,
    propagation_length_ang: scalar_float = 200.0,
) -> Complex[Array, "H W"]:
    """Render reflected channel amplitudes onto a dense detector field.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Parameters
    ----------
    slices : EdgeOnSlices
        Edge-on projected-potential slices from
        :func:`crystal_to_edge_on_slices`.
    energy_kev : scalar_num
        Electron beam energy in kilovolts.
    theta_deg : scalar_num
        Grazing incidence angle in degrees.
    detector_distance_mm : scalar_float, optional
        Detector distance in millimetres.
    image_shape_px : tuple[int, int], optional
        Dense detector image shape (height, width) in pixels.
    pixel_size_mm : tuple[float, float], optional
        Pixel calibration in millimetres per pixel.
    beam_center_px : tuple[float, float], optional
        Beam-center pixel coordinates.
    spot_sigma_px : scalar_float, optional
        Gaussian spot width in detector pixels.
    cap_strength : scalar_float, optional
        Absorbing-mask strength passed to the propagator.
    bandwidth_limit : scalar_float, optional
        Fraction of the FFT Nyquist band to retain.
    propagation_length_ang : scalar_float, optional
        Total crystal length traversed along the beam in Angstroms.

    Returns
    -------
    field : Complex[Array, "H W"]
        Dense coherent detector amplitude field.

    Notes
    -----
    1. Propagate the edge-on wavefield over ``propagation_length_ang``.
    2. Read off complex up-going channel amplitudes above the surface.
    3. Rasterize the sparse amplitudes with
       :func:`rheedium.simul.render_amplitude_to_field`.

    This is the coherent kernel behind the public
    ``simulate_detector_image(render=RenderParams(kernel="multislice"))``
    path.

    See Also
    --------
    reflection_multislice_simulator : Sparse reflected pattern read-off.
    """
    pattern, amplitudes = _reflection_amplitude_pattern(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        detector_distance_mm=detector_distance_mm,
        cap_strength=cap_strength,
        bandwidth_limit=bandwidth_limit,
        propagation_length_ang=propagation_length_ang,
    )
    geometry = DetectorGeometry(
        distance=detector_distance_mm,
        image_shape_px=image_shape_px,
        pixel_size_mm=pixel_size_mm,
        beam_center_px=beam_center_px,
    )
    return render_amplitude_to_field(
        pattern=pattern,
        amplitudes=amplitudes,
        geometry=geometry,
        spot_sigma_px=spot_sigma_px,
    )


@jaxtyped(typechecker=beartype)
def reflection_multislice_simulator(  # noqa: PLR0913
    crystal: CrystalStructure,
    energy_kev: scalar_num = 30.0,
    theta_deg: scalar_num = 2.5,
    phi_deg: scalar_num = 0.0,
    detector_distance: scalar_float = 80.0,
    dx_slice: scalar_float = 1.0,
    dy: scalar_float = 0.25,
    dz: scalar_float = 0.25,
    vacuum_above: scalar_float = 30.0,
    cap_width: scalar_float = 15.0,
    propagation_length_ang: scalar_float = 200.0,
    parameterization: str = "lobato",
) -> RHEEDPattern:
    """Simulate a reflected sparse RHEED pattern by edge-on multislice.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Parameters
    ----------
    crystal : CrystalStructure
        Surface-oriented slab with ``+z`` pointing into vacuum.
    energy_kev : scalar_num, optional
        Electron beam energy in kilovolts.
    theta_deg : scalar_num, optional
        Grazing incidence angle in degrees.
    phi_deg : scalar_num, optional
        Beam azimuth in degrees. Only ``0.0`` is currently supported;
        nonzero values raise :class:`NotImplementedError`.
    detector_distance : scalar_float, optional
        Detector distance in millimetres.
    dx_slice, dy, dz : scalar_float, optional
        Edge-on multislice grid spacings in Angstroms.
    vacuum_above : scalar_float, optional
        Vacuum read-off thickness above the surface. Should exceed
        ``propagation_length_ang * tan(theta)`` for a quantitative read-off.
    cap_width : scalar_float, optional
        Absorbing-layer thickness in Angstroms.
    propagation_length_ang : scalar_float, optional
        Total crystal length traversed along the beam in Angstroms. The
        per-cell stack repeats periodically to reach this length.
    parameterization : str, optional
        Atomic potential model, ``"lobato"`` or ``"kirkland"``.

    Returns
    -------
    pattern : RHEEDPattern
        Sparse reflected-beam pattern with absolute read-off intensities
        (``|amplitude|^2`` relative to a unit-amplitude incident wave).

    Notes
    -----
    1. Slice the crystal along the beam direction.
    2. Propagate the transverse wavefield through the edge-on slices,
       looping the per-cell stack over ``propagation_length_ang``.
    3. Subtract the analytically propagated incident wave and restrict
       read-off to the near-surface vacuum wedge that the reflected field
       has filled.
    4. Fourier transform along periodic ``y`` and project each propagating
       channel onto an up-going vacuum plane wave.
    5. Project the resulting outgoing wavevectors onto the detector.

    The mean inner potential is already present in the edge-on potential
    slices, so refraction at the surface emerges from the propagation; no
    explicit ``inner_potential_v0`` correction applies to this pipeline.

    See Also
    --------
    crystal_to_edge_on_slices : Build edge-on potential slices.
    reflection_multislice_propagate : Propagate an edge-on wavefield.
    """
    _require_phi_zero(phi_deg)
    edge_slices = crystal_to_edge_on_slices(
        crystal,
        phi_deg=phi_deg,
        dx_slice=dx_slice,
        dy=dy,
        dz=dz,
        vacuum_above=vacuum_above,
        cap_width=cap_width,
        parameterization=parameterization,
    )
    pattern, _ = _reflection_amplitude_pattern(
        slices=edge_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        detector_distance_mm=detector_distance,
        propagation_length_ang=propagation_length_ang,
    )
    return pattern


@jaxtyped(typechecker=beartype)
def _read_reflected_pattern(  # noqa: PLR0913
    wavefield: Complex[Array, "ny nz"],
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    detector_distance: scalar_float,
    n_steps: int,
) -> RHEEDPattern:
    """Project the final vacuum wavefield onto up-going channels.

    Intensities are absolute ``|amplitude|^2`` values relative to the unit
    incident wave; the former runtime Fresnel-step blending and max
    normalization have been removed (see
    :func:`_flat_step_specular_reflectivity`, now a test oracle only).
    """
    k_out, amplitudes, propagating = _reflected_channel_amplitudes(
        wavefield=wavefield,
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        n_steps=n_steps,
    )
    intensities = jnp.where(propagating, jnp.abs(amplitudes) ** 2, 0.0)
    detector_points = project_on_detector_geometry(
        k_out,
        DetectorGeometry(distance=detector_distance),
    )
    ny = slices.slices.shape[1]
    channel_indices = jnp.arange(ny, dtype=jnp.int32)
    g_indices = jnp.where(propagating, channel_indices, -1)
    return create_rheed_pattern(
        g_indices=g_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )


__all__ = [
    "crystal_to_edge_on_slices",
    "reflection_detector_amplitude",
    "reflection_multislice_propagate",
    "reflection_multislice_simulator",
]
