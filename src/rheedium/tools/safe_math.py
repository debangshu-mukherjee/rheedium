r"""Gradient-safe elementary math for differentiable simulation paths.

Several elementary operations have well-defined forward values but
undefined or unbounded gradients at domain boundaries: ``sqrt`` and vector
norms at zero, ``arccos`` at :math:`\pm 1`, and quotients with vanishing
denominators. Under ``jax.grad`` the non-selected branch of a bare
``jnp.where`` is still differentiated, so guarding the forward value alone
leaks ``NaN``/``inf`` into gradients (the classic double-``where``
problem). The helpers here centralize the double-``where`` protection so
call sites stay readable and the chosen boundary subgradients are
documented in exactly one place.

Boundary conventions (deliberate, repository-wide):

* ``safe_sqrt``/``safe_norm`` return a **zero** subgradient at the
  boundary. The true derivative diverges there; a zero subgradient keeps
  optimizers stationary instead of poisoning the whole gradient with
  ``NaN``.
* ``safe_arccos`` likewise has a **zero** subgradient at the domain edges
  ``+/-1``: an angle at exactly 0 or pi is at a physical extremum, so a
  stationary subgradient is both safe and meaningful.
* ``safe_divide`` floors the denominator magnitude sign-preservingly
  (never with an additive offset, which flips sign for small negative
  denominators).

Routines
--------
:func:`safe_sqrt`
    Square root with a finite (zero) gradient at zero.
:func:`safe_norm`
    Vector norm with a finite (zero) gradient at the zero vector.
:func:`safe_arccos`
    Inverse cosine with finite gradients at the domain edges.
:func:`safe_divide`
    Quotient with a sign-preserving denominator floor.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float

__all__ = [
    "safe_arccos",
    "safe_divide",
    "safe_norm",
    "safe_sqrt",
]


@jaxtyped(typechecker=beartype)
def safe_sqrt(
    value: Float[Array, "..."],
    eps: scalar_float = 1e-24,
) -> Float[Array, "..."]:
    """Square root whose gradient is finite (zero) at ``value <= 0``.

    :see: :class:`~.test_safe_math.TestSafeSqrt`

    Parameters
    ----------
    value : Float[Array, "..."]
        Radicand. Values at or below zero return 0.0.
    eps : scalar_float, optional
        Threshold below which the input is treated as zero.
        Default: 1e-24

    Returns
    -------
    Float[Array, "..."]
        ``sqrt(value)`` where ``value > eps`` and 0.0 elsewhere, with the
        non-selected branch never differentiated (double-``where``).

    Notes
    -----
    The true derivative of ``sqrt`` diverges at zero; this helper commits
    to a zero subgradient there so gradient descent treats the boundary as
    stationary rather than propagating ``NaN``.
    """
    positive: Float[Array, "..."] = value > eps
    guarded: Float[Array, "..."] = jnp.where(positive, value, 1.0)
    return jnp.where(positive, jnp.sqrt(guarded), 0.0)


@jaxtyped(typechecker=beartype)
def safe_norm(
    vector: Float[Array, "... D"],
    axis: int = -1,
    eps: scalar_float = 1e-24,
) -> Float[Array, "..."]:
    """Euclidean norm whose gradient is finite (zero) at the zero vector.

    :see: :class:`~.test_safe_math.TestSafeNorm`

    Parameters
    ----------
    vector : Float[Array, "... D"]
        Vector(s) to measure.
    axis : int, optional
        Axis holding the vector components. Default: -1
    eps : scalar_float, optional
        Squared-norm threshold below which the vector is treated as zero.
        Default: 1e-24

    Returns
    -------
    Float[Array, "..."]
        ``|vector|`` with a zero subgradient at the zero vector.

    Notes
    -----
    Implemented as ``safe_sqrt(sum(v * v))`` so both the square root and
    the sum are protected by the same double-``where``.
    """
    squared: Float[Array, "..."] = jnp.sum(vector * vector, axis=axis)
    return safe_sqrt(squared, eps=eps)


@jaxtyped(typechecker=beartype)
def safe_arccos(
    cosine: Float[Array, "..."],
    eps: scalar_float = 1e-12,
) -> Float[Array, "..."]:
    """Inverse cosine with finite gradients at ``cosine = +/-1``.

    :see: :class:`~.test_safe_math.TestSafeArccos`

    Parameters
    ----------
    cosine : Float[Array, "..."]
        Cosine values; anything outside ``[-1, 1]`` (from floating-point
        noise) is clipped.
    eps : scalar_float, optional
        Width of the edge band treated as saturated. Default: 1e-12

    Returns
    -------
    Float[Array, "..."]
        ``arccos(cosine)`` with an exact forward value on ``[-1, 1]`` and
        a zero subgradient inside the edge bands ``|cosine| >= 1 - eps``
        (instead of the true divergent derivative): an angle at exactly
        0 or pi sits at a physical extremum, so a stationary subgradient
        keeps optimizers finite without pushing past the boundary.
    """
    clipped: Float[Array, "..."] = jnp.clip(cosine, -1.0, 1.0)
    interior: Float[Array, "..."] = jnp.clip(clipped, -1.0 + eps, 1.0 - eps)
    # Exact forward value, gradient evaluated at the interior point: the
    # stop_gradient term carries the (constant) forward correction without
    # contributing the divergent edge derivative.
    gradient_path: Float[Array, "..."] = jnp.arccos(interior)
    forward_correction: Float[Array, "..."] = jax.lax.stop_gradient(
        jnp.arccos(clipped) - gradient_path
    )
    return gradient_path + forward_correction


@jaxtyped(typechecker=beartype)
def safe_divide(
    numerator: Float[Array, "..."],
    denominator: Float[Array, "..."],
    eps: scalar_float = 1e-12,
) -> Float[Array, "..."]:
    """Quotient with a sign-preserving floor on ``|denominator|``.

    :see: :class:`~.test_safe_math.TestSafeDivide`

    Parameters
    ----------
    numerator : Float[Array, "..."]
        Dividend.
    denominator : Float[Array, "..."]
        Divisor; magnitudes below ``eps`` are floored to ``eps`` while
        keeping the divisor's sign (zero is treated as positive).
    eps : scalar_float, optional
        Minimum divisor magnitude. Default: 1e-12

    Returns
    -------
    Float[Array, "..."]
        ``numerator / denominator`` with bounded value and gradient.

    Notes
    -----
    An additive offset (``den + eps``) flips the result's sign for small
    negative denominators and still explodes at ``den = -eps``; the
    sign-preserving floor does neither.
    """
    sign: Float[Array, "..."] = jnp.where(denominator < 0.0, -1.0, 1.0)
    magnitude: Float[Array, "..."] = jnp.maximum(
        jnp.abs(denominator),
        eps,
    )
    return numerator / (sign * magnitude)
