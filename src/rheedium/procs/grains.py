"""Differentiable grain and domain distribution models.

Extended Summary
----------------
This module provides continuous grain-population models for forward
RHEED calculations. The first public APIs operate on already simulated
patterns and combine them with differentiable grain fractions or
misorientation distributions.

Routine Listings
----------------
:func:`grain_distribution_average`
    Incoherently average a bank of domain patterns with continuous grain
    fractions.
:func:`apply_misorientation_distribution`
    Apply a Gaussian misorientation distribution to explicit orientation
    samples and average the corresponding patterns.

Notes
-----
These functions deliberately model grain populations in pattern space
first. Exact atomistic grain-boundary builders can be layered on top of
this later, but continuous orientation populations are the first public
inverse-facing contract.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def grain_distribution_average(
    domain_patterns: Float[Array, "N_grains H W"],
    grain_volume_fractions: Float[Array, "N_grains"],
) -> Float[Array, "H W"]:
    """Average domain patterns with continuous grain fractions.

    Parameters
    ----------
    domain_patterns : Float[Array, "N_grains H W"]
        Pattern bank for each grain or domain population.
    grain_volume_fractions : Float[Array, "N_grains"]
        Non-negative grain fractions. Values are clipped at zero and
        renormalized internally.

    Returns
    -------
    mixed_pattern : Float[Array, "H W"]
        Incoherently averaged pattern.

    Notes
    -----
    1. **Clip fractions** --
       Negative grain fractions are set to zero.
    2. **Normalize** --
       Divide by the total weight so the fractions sum to one whenever
       any weight is present.
    3. **Average intensities** --
       Compute the weighted sum over the grain axis.

    This is an intensity-space mixture model, so intensities add
    incoherently rather than amplitudes.
    """
    clipped_fractions: Float[Array, "N_grains"] = jnp.clip(
        jnp.asarray(grain_volume_fractions, dtype=jnp.float64), 0.0, None
    )
    normalization: Float[Array, ""] = jnp.sum(clipped_fractions)
    normalized_fractions: Float[Array, "N_grains"] = clipped_fractions / (
        normalization + 1e-10
    )
    weights: Float[Array, "N_grains 1 1"] = normalized_fractions[:, None, None]
    return jnp.sum(weights * domain_patterns, axis=0)


@jaxtyped(typechecker=beartype)
def apply_misorientation_distribution(
    oriented_patterns: Float[Array, "N_orientations H W"],
    orientation_angles_deg: Float[Array, "N_orientations"],
    orientation_base_weights: Float[Array, "N_orientations"],
    mean_angle_deg: scalar_float,
    angular_width_deg: scalar_float,
) -> Float[Array, "H W"]:
    """Average explicit orientation samples with a smooth angle density.

    Parameters
    ----------
    oriented_patterns : Float[Array, "N_orientations H W"]
        Simulated or measured pattern for each explicit orientation
        sample.
    orientation_angles_deg : Float[Array, "N_orientations"]
        Angle assigned to each pattern sample in degrees.
    orientation_base_weights : Float[Array, "N_orientations"]
        Baseline population weight for each orientation sample before
        applying the smooth misorientation envelope.
    mean_angle_deg : scalar_float
        Center of the misorientation distribution in degrees.
    angular_width_deg : scalar_float
        Gaussian width of the misorientation distribution in degrees.

    Returns
    -------
    misorientation_averaged_pattern : Float[Array, "H W"]
        Pattern averaged over the explicit orientation bank.

    Notes
    -----
    1. **Compute angular offsets** --
       Form ``(theta - theta_mean) / sigma`` for each explicit
       orientation sample.
    2. **Build a smooth envelope** --
       Apply a Gaussian weight ``exp(-0.5 * offset**2)``.
    3. **Combine with baseline weights** --
       Multiply the Gaussian envelope by the non-negative baseline
       orientation weights.
    4. **Average patterns** --
       Pass the combined weights into
       :func:`grain_distribution_average`.

    This lets inverse models optimize the center and width of an
    orientation distribution while keeping the orientation support set
    explicit and deterministic.
    """
    width: Float[Array, ""] = jnp.maximum(
        jnp.asarray(angular_width_deg, dtype=jnp.float64), 1e-8
    )
    mean_angle: Float[Array, ""] = jnp.asarray(
        mean_angle_deg, dtype=jnp.float64
    )
    angle_offsets: Float[Array, "N_orientations"] = (
        jnp.asarray(orientation_angles_deg, dtype=jnp.float64) - mean_angle
    ) / width
    gaussian_envelope: Float[Array, "N_orientations"] = jnp.exp(
        -0.5 * angle_offsets**2
    )
    combined_weights: Float[Array, "N_orientations"] = (
        jnp.asarray(orientation_base_weights, dtype=jnp.float64)
        * gaussian_envelope
    )
    return grain_distribution_average(
        domain_patterns=oriented_patterns,
        grain_volume_fractions=combined_weights,
    )


__all__: list[str] = [
    "apply_misorientation_distribution",
    "grain_distribution_average",
]
