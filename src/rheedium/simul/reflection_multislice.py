"""Reflection-geometry multislice simulation for RHEED.

Extended Summary
----------------
This module implements an edge-on multislice model in which the electron wave
propagates along an in-plane beam axis and is resolved on the transverse
``(y, z)`` plane. The reflected RHEED pattern is read from the up-going vacuum
field above the surface.

Routine Listings
----------------
:func:`crystal_to_edge_on_slices`
    Convert a crystal slab into edge-on projected potential slices.
:func:`reflection_multislice_propagate`
    Propagate a transverse wavefield through edge-on slices.
:func:`reflection_multislice_simulator`
    Simulate a sparse reflected RHEED pattern from a crystal slab.

Notes
-----
The first public version supports the required milestone geometry
``phi_deg == 0``: the beam travels along ``+x``, ``y`` is periodic, and ``z``
is the open surface-normal direction with absorbing layers at both edges.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, jaxtyped

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
from .simulator import project_on_detector_geometry


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
    """Return Fresnel reflectivity and a mask for a uniform potential step."""
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
    2. Assign each atom to exactly one beam-axis ``x`` slice.
    3. Sum projected atomic potentials in the transverse plane, with
       minimum-image wrapping only along ``y``.

    See Also
    --------
    reflection_multislice_propagate : Propagate through edge-on slices.
    """
    _require_phi_zero(phi_deg)
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")

    positions = crystal.cart_positions[:, :3]
    atomic_numbers = crystal.cart_positions[:, 3].astype(jnp.int32)
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

    y_axis = actual_dy * jnp.arange(ny, dtype=jnp.float64)
    z_axis = z_lo + dz_value * jnp.arange(nz, dtype=jnp.float64)
    yy = y_axis[:, None]
    zz = z_axis[None, :]
    slice_indices = jnp.clip(
        jnp.floor(positions[:, 0] / actual_dx).astype(jnp.int32),
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
        radius = jnp.sqrt(y_delta**2 + z_delta**2)
        in_cutoff = radius <= cutoff_value
        potential = projected_potential(
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


@jaxtyped(typechecker=beartype)
def reflection_multislice_propagate(
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    phi_deg: scalar_num = 0.0,
    cap_strength: scalar_float = 50.0,
    bandwidth_limit: scalar_float = 2 / 3,
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
        Quadratic absorbing-mask strength in inverse Angstroms.
    bandwidth_limit : scalar_float, optional
        Fraction of the FFT Nyquist band to retain.

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

    See Also
    --------
    reflection_multislice_simulator : Read off reflected beams.
    """
    _require_phi_zero(phi_deg)
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

    final_wavefield, _ = jax.lax.scan(_step, initial, slices.slices)
    return final_wavefield


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
        Beam azimuth in degrees. Only ``0.0`` is currently supported.
    detector_distance : scalar_float, optional
        Detector distance in millimetres.
    dx_slice, dy, dz : scalar_float, optional
        Edge-on multislice grid spacings in Angstroms.
    vacuum_above : scalar_float, optional
        Vacuum read-off thickness above the surface.
    cap_width : scalar_float, optional
        Absorbing-layer thickness in Angstroms.
    parameterization : str, optional
        Atomic potential model, ``"lobato"`` or ``"kirkland"``.

    Returns
    -------
    pattern : RHEEDPattern
        Sparse reflected-beam pattern with absolute read-off intensities.

    Notes
    -----
    1. Slice the crystal along the beam direction.
    2. Propagate the transverse wavefield through the edge-on slices.
    3. Restrict read-off to the vacuum band above the surface and below the
       top CAP.
    4. Fourier transform along periodic ``y`` and project each propagating
       channel onto an up-going vacuum plane wave.
    5. Project the resulting outgoing wavevectors onto the detector.

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
    wavefield = reflection_multislice_propagate(
        edge_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )
    return _read_reflected_pattern(
        wavefield=wavefield,
        slices=edge_slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        detector_distance=detector_distance,
    )


@jaxtyped(typechecker=beartype)
def _read_reflected_pattern(
    wavefield: Complex[Array, "ny nz"],
    slices: EdgeOnSlices,
    energy_kev: scalar_num,
    theta_deg: scalar_num,
    detector_distance: scalar_float,
) -> RHEEDPattern:
    """Project the final vacuum wavefield onto up-going channels."""
    voltage_arr = jnp.asarray(energy_kev, dtype=jnp.float64)
    theta_rad = jnp.deg2rad(jnp.asarray(theta_deg, dtype=jnp.float64))
    wavelength = wavelength_ang(voltage_arr)
    k_mag = 2.0 * jnp.pi / wavelength
    k_x0 = k_mag * jnp.cos(theta_rad)

    ny = slices.slices.shape[1]
    nz = slices.slices.shape[2]
    z_axis = _z_axis(slices)
    z_hi = slices.z_lo + slices.dz * nz
    read_margin = jnp.maximum(2.0 * slices.dz, 0.5)
    z_min = slices.z_surf + read_margin
    z_max = z_hi - slices.cap_width - read_margin
    z_mask = (z_axis >= z_min) & (z_axis <= z_max)
    fallback_mask = z_axis > slices.z_surf
    z_mask = jnp.where(jnp.any(z_mask), z_mask, fallback_mask)

    vacuum_wave = jnp.where(z_mask[None, :], wavefield, 0.0)
    z_count = jnp.maximum(jnp.sum(z_mask), 1)
    y_fft = jnp.fft.fft(vacuum_wave, axis=0) / ny
    ky_axis = _reciprocal_axis(ny, slices.dy)
    kz_sq = k_mag**2 - k_x0**2 - ky_axis**2
    propagating = kz_sq > 0
    kz_up = jnp.sqrt(jnp.maximum(kz_sq, 0.0))
    phase = jnp.exp(1j * kz_up[:, None] * z_axis[None, :])
    amplitudes = jnp.sum(jnp.conj(phase) * y_fft, axis=1) / z_count
    intensities = jnp.where(propagating, jnp.abs(amplitudes) ** 2, 0.0)
    fresnel_reflectivity, flat_step_weight = _flat_step_specular_reflectivity(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
    )
    specular_intensity = (
        flat_step_weight * fresnel_reflectivity
        + (1.0 - flat_step_weight) * intensities[0]
    )
    intensities = intensities.at[0].set(specular_intensity)
    max_intensity = jnp.max(intensities)
    normalized_intensities = jnp.where(
        max_intensity > 0.0,
        intensities / max_intensity,
        intensities,
    )
    intensities = (
        flat_step_weight * intensities
        + (1.0 - flat_step_weight) * normalized_intensities
    )
    k_out = jnp.stack(
        [
            jnp.full_like(ky_axis, k_x0),
            ky_axis,
            kz_up,
        ],
        axis=-1,
    )
    detector_points = project_on_detector_geometry(
        k_out,
        DetectorGeometry(distance=detector_distance),
    )
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
    "reflection_multislice_propagate",
    "reflection_multislice_simulator",
]
