"""Multislice algorithm primitives for dynamical RHEED simulation.

Extended Summary
----------------
Provides the building blocks of the multislice electron diffraction
algorithm: transmission functions through a slice of complex
projected potential, Fresnel free-space propagators, and a single
multislice step suitable for use as the body of ``jax.lax.scan``.

These primitives complement the higher-level ``multislice_propagate``
and ``multislice_simulator`` already in :mod:`rheedium.simul.simulator`.
The functions here are intentionally factored so that each step can
be unit-tested in isolation and so the user can build custom
multislice loops without re-implementing the standard kernels.

Routine Listings
----------------
:func:`build_transmission_function`
    Construct ``T(x, y) = exp(i sigma V(x, y) dz)`` from a complex
    projected potential.
:func:`fresnel_propagator`
    Construct the reciprocal-space Fresnel free-space propagator
    ``P(qx, qy) = exp(-i pi lambda dz (qx^2 + qy^2))``.
:func:`multislice_one_step`
    A single multislice propagation step (transmit -> FFT ->
    propagate -> IFFT) packaged for ``jax.lax.scan``.

Notes
-----
All functions are pure JAX, JIT-compatible, and differentiable.
The interaction constant sigma is computed via
:func:`rheedium.simul.interaction_constant`. Slice thickness is a
continuous parameter and remains differentiable through the
exponential.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from rheedium.types import scalar_float

from .simul_utils import interaction_constant, wavelength_ang


@jaxtyped(typechecker=beartype)
def build_transmission_function(
    projected_potential: Complex[Array, "H W"],
    voltage_kv: scalar_float,
    slice_thickness_angstrom: scalar_float,
) -> Complex[Array, "H W"]:
    r"""Construct the transmission function for one multislice slice.

    Parameters
    ----------
    projected_potential : Complex[Array, "H W"]
        Complex projected potential V_real + i*V_abs in V*Angstrom
        units, e.g. from
        :func:`rheedium.simul.crystal_projected_potential`.
    voltage_kv : scalar_float
        Accelerating voltage in kV. Used to compute the relativistic
        interaction constant sigma.
    slice_thickness_angstrom : scalar_float
        Thickness of this slice dz in Angstroms.

    Returns
    -------
    transmission_function : Complex[Array, "H W"]
        Complex transmission ``T(x, y) = exp(i sigma V dz)``.

    Notes
    -----
    1. Compute the relativistic electron wavelength via
       :func:`rheedium.simul.wavelength_ang`.
    2. Compute the interaction constant sigma via
       :func:`rheedium.simul.interaction_constant`.
    3. Form the complex argument ``i * sigma * V * dz``. Because V is
       complex, ``T`` has both phase modulation (elastic scattering
       from the real part) and amplitude modulation
       (absorption from the imaginary part).
    4. Return ``exp`` of the argument.

    The transmission function modulus satisfies ``|T| <= 1`` because
    the imaginary part of V is non-negative (absorption never
    increases amplitude).
    """
    voltage_kv_arr: Float[Array, ""] = jnp.asarray(
        voltage_kv, dtype=jnp.float64
    )
    slice_thickness_arr: Float[Array, ""] = jnp.asarray(
        slice_thickness_angstrom, dtype=jnp.float64
    )
    wavelength: Float[Array, ""] = wavelength_ang(voltage_kv_arr)
    sigma: Float[Array, ""] = interaction_constant(voltage_kv_arr, wavelength)
    phase_arg: Complex[Array, "H W"] = (
        1j * sigma * projected_potential * slice_thickness_arr
    )
    transmission: Complex[Array, "H W"] = jnp.exp(phase_arg)
    return transmission


@jaxtyped(typechecker=beartype)
def fresnel_propagator(
    grid_shape: Tuple[int, int],
    cell_dimensions_angstrom: Float[Array, "2"],
    voltage_kv: scalar_float,
    slice_thickness_angstrom: scalar_float,
) -> Complex[Array, "H W"]:
    r"""Construct the reciprocal-space Fresnel free-space propagator.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Real-space grid dimensions (H, W).
    cell_dimensions_angstrom : Float[Array, "2"]
        Physical cell dimensions [Lx, Ly] in Angstroms.
    voltage_kv : scalar_float
        Accelerating voltage in kV.
    slice_thickness_angstrom : scalar_float
        Slice thickness dz in Angstroms.

    Returns
    -------
    propagator : Complex[Array, "H W"]
        Fresnel propagator
        ``P(qx, qy) = exp(-i pi lambda dz (qx^2 + qy^2))``
        in reciprocal space, with the standard
        ``jnp.fft.fftfreq`` ordering.

    Notes
    -----
    1. Compute the relativistic electron wavelength.
    2. Build reciprocal-space frequency grids
       ``qx = jnp.fft.fftfreq(H, Lx / H)`` and similarly for qy.
    3. Compute the propagator phase
       ``-pi * lambda * dz * (qx**2 + qy**2)`` and exponentiate.

    The propagator is unitary: ``|P(qx, qy)| = 1`` everywhere because
    free-space propagation conserves probability.
    """
    voltage_kv_arr: Float[Array, ""] = jnp.asarray(
        voltage_kv, dtype=jnp.float64
    )
    slice_thickness_arr: Float[Array, ""] = jnp.asarray(
        slice_thickness_angstrom, dtype=jnp.float64
    )
    n_x: int = grid_shape[0]
    n_y: int = grid_shape[1]
    lx: Float[Array, ""] = cell_dimensions_angstrom[0]
    ly: Float[Array, ""] = cell_dimensions_angstrom[1]
    dx: Float[Array, ""] = lx / n_x
    dy: Float[Array, ""] = ly / n_y
    wavelength: Float[Array, ""] = wavelength_ang(voltage_kv_arr)
    qx: Float[Array, "H"] = jnp.fft.fftfreq(n_x, dx)
    qy: Float[Array, "W"] = jnp.fft.fftfreq(n_y, dy)
    qx_grid: Float[Array, "H W"]
    qy_grid: Float[Array, "H W"]
    qx_grid, qy_grid = jnp.meshgrid(qx, qy, indexing="ij")
    phase: Float[Array, "H W"] = (
        -jnp.pi * wavelength * slice_thickness_arr * (qx_grid**2 + qy_grid**2)
    )
    propagator: Complex[Array, "H W"] = jnp.exp(1j * phase)
    return propagator


@jaxtyped(typechecker=beartype)
def multislice_one_step(
    wavefunction: Complex[Array, "H W"],
    transmission_function: Complex[Array, "H W"],
    propagator: Complex[Array, "H W"],
) -> Complex[Array, "H W"]:
    r"""Perform a single multislice propagation step.

    Parameters
    ----------
    wavefunction : Complex[Array, "H W"]
        Electron wavefunction psi(x, y) at the entrance of the slice.
    transmission_function : Complex[Array, "H W"]
        Complex transmission ``T(x, y)`` for this slice from
        :func:`build_transmission_function`.
    propagator : Complex[Array, "H W"]
        Reciprocal-space Fresnel propagator from
        :func:`fresnel_propagator`. Reused across slices when slice
        thickness is uniform.

    Returns
    -------
    psi_out : Complex[Array, "H W"]
        Wavefunction at the exit of the slice.

    Notes
    -----
    1. Multiply the wavefunction by the transmission function in real
       space (phase + amplitude modulation by the projected potential).
    2. FFT to reciprocal space.
    3. Multiply by the Fresnel propagator (free-space propagation by
       dz).
    4. Inverse FFT back to real space.

    This function is intentionally a plain function rather than a
    ``lax.scan`` body so that callers can compose it however they
    need. For looping over many slices, wrap it in ``lax.scan`` with
    ``transmission_function`` as the scanned input.
    """
    psi_transmitted: Complex[Array, "H W"] = (
        wavefunction * transmission_function
    )
    psi_k: Complex[Array, "H W"] = jnp.fft.fft2(psi_transmitted)
    psi_k_propagated: Complex[Array, "H W"] = psi_k * propagator
    psi_out: Complex[Array, "H W"] = jnp.fft.ifft2(psi_k_propagated)
    return psi_out


__all__: list[str] = [
    "build_transmission_function",
    "fresnel_propagator",
    "multislice_one_step",
]
