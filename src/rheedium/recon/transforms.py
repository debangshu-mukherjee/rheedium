r"""Bijective parameter transforms for differentiable reconstruction.

Extended Summary
----------------
This module provides small, composable transforms between unconstrained
optimizer coordinates and physically constrained parameters. Reconstruction
solvers can therefore run in smooth Euclidean space while positivity, bounded
intervals, occupancies, and probability-simplex constraints are enforced by
construction.

Routine Listings
----------------
:func:`positive_from_unconstrained`
    Map unconstrained values to strictly positive physical values.
:func:`unconstrained_from_positive`
    Map strictly positive physical values back to unconstrained coordinates.
:func:`bounded_from_unconstrained`
    Map unconstrained values into a closed finite interval.
:func:`unconstrained_from_bounded`
    Map bounded physical values back to unconstrained coordinates.
:func:`fractional_from_unconstrained`
    Map unconstrained values to unit-cell fractional coordinates.
:func:`unconstrained_from_fractional`
    Map fractional coordinates back to unconstrained coordinates.
:func:`lattice_from_unconstrained`
    Map unconstrained lengths and angles to lattice parameters.
:func:`unconstrained_from_lattice`
    Map lattice parameters back to unconstrained coordinates.
:func:`wyckoff_fractional_from_unconstrained`
    Map independent unconstrained coordinates to constrained Wyckoff positions.
:func:`simplex_from_unconstrained`
    Map unconstrained logits onto a probability simplex.
:func:`unconstrained_from_simplex`
    Map simplex weights to centered unconstrained logits.
:func:`ordered_bounded_from_unconstrained`
    Map unconstrained values to ordered points inside a finite interval.
:func:`unconstrained_from_ordered_bounded`
    Map ordered bounded points back to unconstrained increment logits.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float

_EPS: float = 1e-12


@jaxtyped(typechecker=beartype)
def positive_from_unconstrained(
    unconstrained: Float[Array, "..."],
    minimum: scalar_float = 0.0,
) -> Float[Array, "..."]:
    """Map unconstrained values to strictly positive physical values.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained : Float[Array, "..."]
        Unconstrained optimizer coordinates.
    minimum : scalar_float, optional
        Lower asymptote of the positive transform. Default: 0.0

    Returns
    -------
    positive : Float[Array, "..."]
        Strictly positive values offset by ``minimum``.

    Notes
    -----
    1. Apply the numerically stable JAX softplus transform.
    2. Add the requested lower asymptote.
    """
    positive: Float[Array, "..."] = jax.nn.softplus(unconstrained) + minimum
    return positive


@jaxtyped(typechecker=beartype)
def unconstrained_from_positive(
    positive: Float[Array, "..."],
    minimum: scalar_float = 0.0,
) -> Float[Array, "..."]:
    """Map strictly positive physical values back to unconstrained coordinates.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    positive : Float[Array, "..."]
        Physical values above ``minimum``.
    minimum : scalar_float, optional
        Lower asymptote used by :func:`positive_from_unconstrained`.
        Default: 0.0

    Returns
    -------
    unconstrained : Float[Array, "..."]
        Unconstrained coordinates whose softplus image recovers ``positive``.

    Notes
    -----
    1. Subtract the lower asymptote.
    2. Apply a stable inverse softplus expression.
    """
    shifted: Float[Array, "..."] = jnp.maximum(
        positive - minimum,
        _EPS,
    )
    unconstrained: Float[Array, "..."] = shifted + jnp.log1p(
        -jnp.exp(-shifted)
    )
    return unconstrained


@jaxtyped(typechecker=beartype)
def bounded_from_unconstrained(
    unconstrained: Float[Array, "..."],
    lower: scalar_float,
    upper: scalar_float,
) -> Float[Array, "..."]:
    """Map unconstrained values into a closed finite interval.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained : Float[Array, "..."]
        Unconstrained optimizer coordinates.
    lower : scalar_float
        Lower interval endpoint.
    upper : scalar_float
        Upper interval endpoint.

    Returns
    -------
    bounded : Float[Array, "..."]
        Values in ``[lower, upper]``.

    Notes
    -----
    1. Evaluate a logistic sigmoid in unconstrained space.
    2. Affinely scale it to the requested physical interval.
    """
    width: scalar_float = upper - lower
    bounded: Float[Array, "..."] = lower + width * jax.nn.sigmoid(
        unconstrained
    )
    return bounded


@jaxtyped(typechecker=beartype)
def unconstrained_from_bounded(
    bounded: Float[Array, "..."],
    lower: scalar_float,
    upper: scalar_float,
) -> Float[Array, "..."]:
    """Map bounded physical values back to unconstrained coordinates.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    bounded : Float[Array, "..."]
        Physical values inside ``[lower, upper]``.
    lower : scalar_float
        Lower interval endpoint.
    upper : scalar_float
        Upper interval endpoint.

    Returns
    -------
    unconstrained : Float[Array, "..."]
        Logit coordinates whose sigmoid image recovers ``bounded``.

    Notes
    -----
    1. Normalize the physical value to the unit interval.
    2. Clip away exact endpoints for finite inverse logits.
    3. Apply the logit transform.
    """
    width: scalar_float = upper - lower
    probability: Float[Array, "..."] = jnp.clip(
        (bounded - lower) / width,
        _EPS,
        1.0 - _EPS,
    )
    unconstrained: Float[Array, "..."] = jnp.log(probability) - jnp.log1p(
        -probability
    )
    return unconstrained


@jaxtyped(typechecker=beartype)
def fractional_from_unconstrained(
    unconstrained: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Map unconstrained values to unit-cell fractional coordinates.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained : Float[Array, "..."]
        Unconstrained optimizer coordinates.

    Returns
    -------
    fractional : Float[Array, "..."]
        Fractional coordinates in ``[0, 1]``.

    Notes
    -----
    1. Reuse the logistic sigmoid as the smooth unit-interval bijector.
    2. Return coordinates suitable for fractional atomic positions,
       occupancies, and symmetry-internal parameters.
    """
    fractional: Float[Array, "..."] = jax.nn.sigmoid(unconstrained)
    return fractional


@jaxtyped(typechecker=beartype)
def unconstrained_from_fractional(
    fractional: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Map fractional coordinates back to unconstrained coordinates.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    fractional : Float[Array, "..."]
        Fractional coordinates in the unit interval.

    Returns
    -------
    unconstrained : Float[Array, "..."]
        Logit coordinates whose sigmoid image recovers ``fractional``.

    Notes
    -----
    1. Clip exact unit-cell boundaries away from singular logits.
    2. Apply the same stable logit expression used by bounded transforms.
    """
    clipped: Float[Array, "..."] = jnp.clip(
        fractional,
        _EPS,
        1.0 - _EPS,
    )
    unconstrained: Float[Array, "..."] = jnp.log(clipped) - jnp.log1p(-clipped)
    return unconstrained


@jaxtyped(typechecker=beartype)
def lattice_from_unconstrained(
    unconstrained_lengths: Float[Array, "3"],
    unconstrained_angles: Float[Array, "3"],
    minimum_length: scalar_float = 0.0,
    minimum_angle_deg: scalar_float = 1.0,
    maximum_angle_deg: scalar_float = 179.0,
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Map unconstrained lengths and angles to lattice parameters.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained_lengths : Float[Array, "3"]
        Unconstrained optimizer coordinates for ``a, b, c``.
    unconstrained_angles : Float[Array, "3"]
        Unconstrained optimizer coordinates for ``alpha, beta, gamma``.
    minimum_length : scalar_float, optional
        Lower asymptote for lattice lengths. Default: 0.0
    minimum_angle_deg : scalar_float, optional
        Lower angle bound in degrees. Default: 1.0
    maximum_angle_deg : scalar_float, optional
        Upper angle bound in degrees. Default: 179.0

    Returns
    -------
    lengths, angles : Tuple[Float[Array, "3"], Float[Array, "3"]]
        Positive cell lengths and bounded cell angles in degrees.

    Notes
    -----
    1. Use softplus for strictly positive cell lengths.
    2. Use the bounded sigmoid map for crystallographic angles.
    """
    lengths: Float[Array, "3"] = positive_from_unconstrained(
        unconstrained_lengths,
        minimum=minimum_length,
    )
    angles: Float[Array, "3"] = bounded_from_unconstrained(
        unconstrained_angles,
        minimum_angle_deg,
        maximum_angle_deg,
    )
    return lengths, angles


@jaxtyped(typechecker=beartype)
def unconstrained_from_lattice(
    lengths: Float[Array, "3"],
    angles: Float[Array, "3"],
    minimum_length: scalar_float = 0.0,
    minimum_angle_deg: scalar_float = 1.0,
    maximum_angle_deg: scalar_float = 179.0,
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Map lattice parameters back to unconstrained coordinates.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    lengths : Float[Array, "3"]
        Positive cell lengths.
    angles : Float[Array, "3"]
        Cell angles in degrees.
    minimum_length : scalar_float, optional
        Lower asymptote used by :func:`lattice_from_unconstrained`.
        Default: 0.0
    minimum_angle_deg : scalar_float, optional
        Lower angle bound in degrees. Default: 1.0
    maximum_angle_deg : scalar_float, optional
        Upper angle bound in degrees. Default: 179.0

    Returns
    -------
    unconstrained_lengths, unconstrained_angles
        Optimizer coordinates whose forward image recovers the lattice.

    Notes
    -----
    1. Invert the length softplus transform.
    2. Invert the bounded angle transform.
    """
    unconstrained_lengths: Float[Array, "3"] = unconstrained_from_positive(
        lengths,
        minimum=minimum_length,
    )
    unconstrained_angles: Float[Array, "3"] = unconstrained_from_bounded(
        angles,
        minimum_angle_deg,
        maximum_angle_deg,
    )
    return unconstrained_lengths, unconstrained_angles


@jaxtyped(typechecker=beartype)
def wyckoff_fractional_from_unconstrained(
    unconstrained: Float[Array, "D"],
    basis: Float[Array, "M D"],
    offset: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Map independent coordinates to constrained Wyckoff positions.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained : Float[Array, "D"]
        Independent unconstrained Wyckoff degrees of freedom.
    basis : Float[Array, "M D"]
        Linear map from independent fractional degrees of freedom to the full
        constrained coordinate vector.
    offset : Float[Array, "M"]
        Fixed fractional offset for the Wyckoff orbit.

    Returns
    -------
    fractional : Float[Array, "M"]
        Unit-cell wrapped constrained fractional coordinates.

    Notes
    -----
    1. Map independent degrees to unit-interval fractional coordinates.
    2. Apply the affine Wyckoff constraint.
    3. Wrap into the unit cell with an almost-everywhere differentiable modulo.
    """
    independent_fractional: Float[Array, "D"] = fractional_from_unconstrained(
        unconstrained
    )
    raw_fractional: Float[Array, "M"] = offset + basis @ independent_fractional
    fractional: Float[Array, "M"] = raw_fractional - jnp.floor(raw_fractional)
    return fractional


@jaxtyped(typechecker=beartype)
def simplex_from_unconstrained(
    logits: Float[Array, "N"],
) -> Float[Array, "N"]:
    """Map unconstrained logits onto a probability simplex.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    logits : Float[Array, "N"]
        Unconstrained log-probability coordinates.

    Returns
    -------
    weights : Float[Array, "N"]
        Positive probability weights summing to one.

    Notes
    -----
    1. Use JAX's stable softmax implementation.
    2. Return the normalized weights directly.
    """
    weights: Float[Array, "N"] = jax.nn.softmax(logits)
    return weights


@jaxtyped(typechecker=beartype)
def unconstrained_from_simplex(
    weights: Float[Array, "N"],
) -> Float[Array, "N"]:
    """Map simplex weights to centered unconstrained logits.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    weights : Float[Array, "N"]
        Non-negative probability weights. They are normalized before
        inversion.

    Returns
    -------
    logits : Float[Array, "N"]
        Centered logits whose softmax recovers the normalized weights.

    Notes
    -----
    1. Clip and normalize weights onto the simplex.
    2. Take logs and remove the mean logit to fix the softmax gauge.
    3. The mathematical inverse is differentiable only in the strict simplex
       interior. At a zero-weight boundary it requires ``log(0)`` and is
       legitimately non-differentiable; the numerical clip keeps the forward
       result finite but does not define a supported boundary gradient.
    """
    clipped: Float[Array, "N"] = jnp.clip(weights, _EPS, None)
    normalized: Float[Array, "N"] = clipped / jnp.sum(clipped)
    raw_logits: Float[Array, "N"] = jnp.log(normalized)
    logits: Float[Array, "N"] = raw_logits - jnp.mean(raw_logits)
    return logits


@jaxtyped(typechecker=beartype)
def ordered_bounded_from_unconstrained(
    z: Float[Array, "N"],
    lower: scalar_float,
    upper: scalar_float,
) -> Float[Array, "N"]:
    """Map unconstrained values to ordered points inside a finite interval.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    z : Float[Array, "N"]
        Unconstrained segment logits.
    lower : scalar_float
        Lower interval endpoint.
    upper : scalar_float
        Upper interval endpoint.

    Returns
    -------
    ordered : Float[Array, "N"]
        Strictly increasing points inside ``(lower, upper)``.

    Notes
    -----
    1. Convert logits to positive simplex increments.
    2. Use their cumulative sum as ordered fractional coordinates.
    3. Affinely scale the fractions to the physical interval.
    """
    logits: Float[Array, "N_plus_1"] = jnp.concatenate(
        [z, jnp.zeros((1,), dtype=z.dtype)]
    )
    increments: Float[Array, "N_plus_1"] = jax.nn.softmax(logits)
    cumulative: Float[Array, "N"] = jnp.cumsum(increments)[:-1]
    width: scalar_float = upper - lower
    ordered: Float[Array, "N"] = lower + width * cumulative
    return ordered


@jaxtyped(typechecker=beartype)
def unconstrained_from_ordered_bounded(
    x: Float[Array, "N"],
    lower: scalar_float,
    upper: scalar_float,
) -> Float[Array, "N"]:
    """Map ordered finite-interval points back to unconstrained logits.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    x : Float[Array, "N"]
        Strictly increasing physical points inside ``(lower, upper)``.
    lower : scalar_float
        Lower interval endpoint.
    upper : scalar_float
        Upper interval endpoint.

    Returns
    -------
    unconstrained : Float[Array, "N"]
        Increment logits whose forward transform recovers ``x``.
    """
    u: Float[Array, "N"] = (x - lower) / (upper - lower)
    inc: Float[Array, "N"] = jnp.diff(
        jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), u])
    )
    last: Float[Array, ""] = 1.0 - u[-1]
    unconstrained: Float[Array, "N"] = jnp.log(inc / last)
    return unconstrained


__all__: list[str] = [
    "bounded_from_unconstrained",
    "fractional_from_unconstrained",
    "lattice_from_unconstrained",
    "ordered_bounded_from_unconstrained",
    "positive_from_unconstrained",
    "simplex_from_unconstrained",
    "unconstrained_from_bounded",
    "unconstrained_from_fractional",
    "unconstrained_from_lattice",
    "unconstrained_from_ordered_bounded",
    "unconstrained_from_positive",
    "unconstrained_from_simplex",
    "wyckoff_fractional_from_unconstrained",
]
