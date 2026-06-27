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
:func:`simplex_from_unconstrained`
    Map unconstrained logits onto a probability simplex.
:func:`unconstrained_from_simplex`
    Map simplex weights to centered unconstrained logits.
:func:`ordered_bounded_from_unconstrained`
    Map unconstrained values to ordered points inside a finite interval.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
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
    """
    clipped: Float[Array, "N"] = jnp.clip(weights, _EPS, None)
    normalized: Float[Array, "N"] = clipped / jnp.sum(clipped)
    raw_logits: Float[Array, "N"] = jnp.log(normalized)
    logits: Float[Array, "N"] = raw_logits - jnp.mean(raw_logits)
    return logits


@jaxtyped(typechecker=beartype)
def ordered_bounded_from_unconstrained(
    unconstrained: Float[Array, "N"],
    lower: scalar_float,
    upper: scalar_float,
) -> Float[Array, "N"]:
    """Map unconstrained values to ordered points inside a finite interval.

    :see: :class:`~.test_transforms.TestReconTransforms`

    Parameters
    ----------
    unconstrained : Float[Array, "N"]
        Unconstrained segment logits.
    lower : scalar_float
        Lower interval endpoint.
    upper : scalar_float
        Upper interval endpoint.

    Returns
    -------
    ordered : Float[Array, "N"]
        Monotonically non-decreasing points in ``[lower, upper]``.

    Notes
    -----
    1. Convert logits to positive simplex increments.
    2. Use their cumulative sum as ordered fractional coordinates.
    3. Affinely scale the fractions to the physical interval.
    """
    increments: Float[Array, "N"] = simplex_from_unconstrained(unconstrained)
    cumulative: Float[Array, "N"] = jnp.cumsum(increments)
    width: scalar_float = upper - lower
    ordered: Float[Array, "N"] = lower + width * cumulative
    return ordered


__all__: list[str] = [
    "bounded_from_unconstrained",
    "ordered_bounded_from_unconstrained",
    "positive_from_unconstrained",
    "simplex_from_unconstrained",
    "unconstrained_from_bounded",
    "unconstrained_from_positive",
    "unconstrained_from_simplex",
]
