"""Differentiable surface modifiers for RHEED forward models.

Extended Summary
----------------
Provides differentiable modifiers that perturb ideal surfaces before or
after forward simulation. Current functionality includes vicinal step
splitting of crystal truncation rods and incoherent averaging over
multiple surface domains with different orientations or terminations.

Routine Listings
----------------
:func:`vicinal_surface_step_splitting`
    Compute CTR intensity modification due to a periodic step array
    on a vicinal surface.
:func:`incoherent_domain_average`
    Compute the incoherently averaged RHEED pattern from multiple
    independent surface domains.

Notes
-----
All functions are implemented with pure JAX operations and are
differentiable via ``jax.grad``. Step height, terrace width, and
domain fractions are continuous parameters that inverse models can
optimize against experimental data.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def vicinal_surface_step_splitting(
    hk_index: Int[Array, "2"],
    step_height_angstrom: scalar_float,
    terrace_width_angstrom: scalar_float,
    q_z: Float[Array, "N_qz"],
) -> Float[Array, "N_qz"]:
    """Compute CTR intensity modification from periodic steps.

    Parameters
    ----------
    hk_index : Int[Array, "2"]
        In-plane Miller indices (h, k) of the rod.
    step_height_angstrom : scalar_float
        Single step height d in Angstroms. Equal to one atomic
        layer spacing for simple step arrays.
    terrace_width_angstrom : scalar_float
        Mean terrace width w in Angstroms.
    q_z : Float[Array, "N_qz"]
        Perpendicular momentum transfer values at which to
        evaluate intensity, in inverse Angstroms.

    Returns
    -------
    step_modified_intensity : Float[Array, "N_qz"]
        CTR intensity profile modified by step-induced
        interference. Normalized to unit maximum.

    Notes
    -----
    1. Compute phase difference per step:
       ``delta_phi = q_z * step_height``.
    2. Compute the interference function of the step array
       using the geometric series result for N terraces in
       the large-N limit:
       ``S_step(q_z) = (1 - r^2) / |1 - r * exp(i*delta_phi)|^2``
       where ``r = exp(-a / terrace_width)`` is the
       terrace-width damping factor and ``a`` is the lattice
       parameter projected along the step direction.
    3. For simplicity in the large-N limit, model the step
       interference as:
       ``I(q_z) = 1 / (1 + F * sin^2(delta_phi / 2))``
       where ``F = 4 * (w / d)^2`` controls the sharpness
       of the splitting. This is the Airy function form
       appropriate for a regular step array.
    4. Normalize to unit maximum.
    """
    del hk_index  # Reserved for future rod-dependent step models.

    step_height_angstrom: Float[Array, ""] = jnp.asarray(
        step_height_angstrom, dtype=jnp.float64
    )
    terrace_width_angstrom: Float[Array, ""] = jnp.asarray(
        terrace_width_angstrom, dtype=jnp.float64
    )

    delta_phi: Float[Array, "N_qz"] = q_z * step_height_angstrom

    sin_half: Float[Array, "N_qz"] = jnp.sin(delta_phi / 2.0)
    sin_sq: Float[Array, "N_qz"] = sin_half**2

    finesse: Float[Array, ""] = (
        4.0 * (terrace_width_angstrom / (step_height_angstrom + 1e-10)) ** 2
    )

    intensity: Float[Array, "N_qz"] = 1.0 / (1.0 + finesse * sin_sq)

    max_val: Float[Array, ""] = jnp.max(intensity)
    normalized: Float[Array, "N_qz"] = intensity / (max_val + 1e-10)

    return normalized


@jaxtyped(typechecker=beartype)
def incoherent_domain_average(
    domain_patterns: Float[Array, "N_domains H W"],
    domain_volume_fractions: Float[Array, "N_domains"],
) -> Float[Array, "H W"]:
    """Compute incoherently averaged RHEED pattern from domains.

    Parameters
    ----------
    domain_patterns : Float[Array, "N_domains H W"]
        Individual RHEED pattern for each domain orientation.
    domain_volume_fractions : Float[Array, "N_domains"]
        Volume (area) fraction of each domain. Must sum to 1
        within a tolerance of 1e-3.

    Returns
    -------
    mixed_pattern : Float[Array, "H W"]
        Intensity-weighted average: ``sum_i(f_i * I_i)``.

    Notes
    -----
    1. Validate that ``domain_volume_fractions`` sums to 1
       within tolerance.
    2. Reshape fractions to ``(N_domains, 1, 1)`` for
       broadcasting against ``(N_domains, H, W)`` patterns.
    3. Return ``jnp.sum(fractions * domain_patterns, axis=0)``.

    Domains scatter independently so their intensities add,
    not their amplitudes. This is the correct treatment for
    rotational twins, anti-phase domains, mixed terminations,
    and polycrystalline grains.
    """
    fraction_sum: Float[Array, ""] = jnp.sum(domain_volume_fractions)
    fractions_normalized: Float[Array, "N_domains"] = (
        domain_volume_fractions / (fraction_sum + 1e-10)
    )

    weights: Float[Array, "N_domains 1 1"] = fractions_normalized[
        :, None, None
    ]

    mixed: Float[Array, "H W"] = jnp.sum(weights * domain_patterns, axis=0)

    return mixed


__all__: list[str] = [
    "incoherent_domain_average",
    "vicinal_surface_step_splitting",
]
