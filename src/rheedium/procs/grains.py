"""Differentiable grain and domain distribution models.

Extended Summary
----------------
This module provides continuous grain-population models for forward
RHEED calculations. The first public APIs operate on already simulated
patterns and combine them with differentiable grain fractions or
misorientation distributions.

Routine Listings
----------------
:func:`grain_population_to_distribution`
    Convert grain orientation/size/fraction metadata into a generic
    incoherent Distribution.
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

R5 return type: ``grain_population_to_distribution`` is the statistical
producer and returns ``Distribution``. The pattern-space averaging helpers
return detector-image arrays because they are compatibility front ends over the
shared Layer-1 reducer, not structure builders.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import (
    Distribution,
    ReductionMode,
    create_distribution,
    scalar_float,
)


@jaxtyped(typechecker=beartype)
def grain_population_to_distribution(
    orientation_angles_deg: Float[Array, "N"],
    grain_sizes_angstrom: Float[Array, "M"],
    grain_volume_fractions: Float[Array, "K"],
    axis_id: str | None = "grains",
) -> Distribution:
    """Convert grain population metadata to a generic Distribution.

    :see: :class:`~.test_grains.TestGrainPopulationToDistribution`

    Parameters
    ----------
    orientation_angles_deg : Float[Array, "N"]
        Grain/domain azimuth samples in degrees, relative to the nominal
        crystal orientation.
    grain_sizes_angstrom : Float[Array, "M"]
        Characteristic grain sizes in Angstroms, aligned one-to-one with
        ``orientation_angles_deg``.
    grain_volume_fractions : Float[Array, "K"]
        Non-negative grain population fractions. Values are normalized by the
        generic distribution factory.
    axis_id : str | None, optional
        Optional static label for the returned ensemble axis. Default:
        ``"grains"``.

    Returns
    -------
    distribution : Distribution
        Incoherent grain distribution with samples
        ``[orientation_angle_deg, grain_size_angstrom]``.

    Notes
    -----
    1. Convert grain orientations, sizes, and fractions to ``float64`` arrays.
    2. Validate one-dimensional, one-to-one grain metadata.
    3. Reject non-positive grain sizes via ``eqx.error_if``.
    4. Delegate probability normalization to :func:`create_distribution`.

    See Also
    --------
    grain_distribution_average : Pattern-space grain intensity mixture.
    """
    orientation_angles: Float[Array, "N"] = jnp.asarray(
        orientation_angles_deg,
        dtype=jnp.float64,
    )
    grain_sizes: Float[Array, "M"] = jnp.asarray(
        grain_sizes_angstrom,
        dtype=jnp.float64,
    )
    grain_fractions: Float[Array, "K"] = jnp.asarray(
        grain_volume_fractions,
        dtype=jnp.float64,
    )
    if orientation_angles.ndim != 1:
        raise ValueError("orientation_angles_deg must have shape (N,)")
    if grain_sizes.ndim != 1:
        raise ValueError("grain_sizes_angstrom must have shape (N,)")
    if grain_fractions.ndim != 1:
        raise ValueError("grain_volume_fractions must have shape (N,)")
    if orientation_angles.shape[0] != grain_sizes.shape[0]:
        raise ValueError(
            "orientation_angles_deg and grain_sizes_angstrom must share length"
        )
    if orientation_angles.shape[0] != grain_fractions.shape[0]:
        raise ValueError(
            "orientation_angles_deg and grain_volume_fractions must share "
            "length"
        )

    checked_sizes: Float[Array, "N"] = eqx.error_if(
        grain_sizes,
        jnp.any(grain_sizes <= 0.0),
        "grain_sizes_angstrom must be positive",
    )
    samples: Float[Array, "N 2"] = jnp.stack(
        [orientation_angles, checked_sizes],
        axis=-1,
    )
    return create_distribution(
        samples=samples,
        weights=grain_fractions,
        reduction=ReductionMode.INCOHERENT,
        axis_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def grain_distribution_average(
    domain_patterns: Float[Array, "N_grains H W"],
    grain_volume_fractions: Float[Array, "N_grains"],
) -> Float[Array, "H W"]:
    """Average domain patterns with continuous grain fractions.

    :see: :class:`~.test_grains.TestGrainDistributionAverage`

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
    The pattern bank is bound to an incoherent generic
    :class:`~rheedium.types.Distribution` and reduced by the shared Layer-1
    reducer. This is an intensity-space mixture model, so intensities add
    incoherently rather than amplitudes.
    """
    clipped_fractions: Float[Array, "N_grains"] = jnp.clip(
        jnp.asarray(grain_volume_fractions, dtype=jnp.float64), 0.0, None
    )
    fraction_sum: Float[Array, ""] = jnp.sum(clipped_fractions)
    has_positive_weight: Float[Array, ""] = jnp.where(
        fraction_sum > 0.0,
        1.0,
        0.0,
    )
    safe_fractions: Float[Array, "N_grains"] = jnp.where(
        fraction_sum > 0.0,
        clipped_fractions,
        jnp.ones_like(clipped_fractions),
    )
    sample_indices: Float[Array, "N_grains 1"] = jnp.arange(
        domain_patterns.shape[0],
        dtype=jnp.float64,
    )[:, None]
    distribution: Distribution = create_distribution(
        samples=sample_indices,
        weights=safe_fractions,
        reduction=ReductionMode.INCOHERENT,
        axis_id="grain_pattern",
    )

    def _domain_intensity(
        sample: Float[Array, "1"],
    ) -> Float[Array, "H W"]:
        pattern_index: Int[Array, ""] = sample[0].astype(jnp.int32)
        return domain_patterns[pattern_index]

    from rheedium.simul.beam_averaging import apply_distribution_intensity

    mixed_pattern: Float[Array, "H W"] = apply_distribution_intensity(
        distribution,
        _domain_intensity,
    )
    return has_positive_weight * mixed_pattern


@jaxtyped(typechecker=beartype)
def apply_misorientation_distribution(
    oriented_patterns: Float[Array, "N_orientations H W"],
    orientation_angles_deg: Float[Array, "N_orientations"],
    orientation_base_weights: Float[Array, "N_orientations"],
    mean_angle_deg: scalar_float,
    angular_width_deg: scalar_float,
) -> Float[Array, "H W"]:
    """Average explicit orientation samples with a smooth angle density.

    :see: :class:`~.test_grains.TestApplyMisorientationDistribution`

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
    "grain_population_to_distribution",
]
