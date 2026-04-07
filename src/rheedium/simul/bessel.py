"""Modified Bessel functions of the second kind for JAX.

Extended Summary
----------------
Provides differentiable, JIT-compatible implementations of the modified
Bessel functions K_0(x) and K_1(x) in pure JAX. These are required by
the Lobato-van Dyck projected potential parameterization, which
expresses the projected atomic potential in terms of K_0 and K_1.

JAX does not include K_0 or K_1 natively. The implementations here use
polynomial approximations split at x = 2: a logarithmic series for
small x and an asymptotic expansion for large x, following Abramowitz
and Stegun (1964) sections 9.8.1--9.8.8. Accuracy is better than
1 × 10⁻⁷ relative error across the full domain x > 0.

Routine Listings
----------------
:func:`bessel_k0`
    Modified Bessel function K_0(x).
:func:`bessel_k1`
    Modified Bessel function K_1(x).

Notes
-----
Both functions support arbitrary batch dimensions, ``jax.jit``,
``jax.grad``, and ``jax.vmap``. Input must satisfy x > 0; behaviour
at x = 0 is undefined (K_0 and K_1 diverge there).

Adapted from the janssen project's ``utils/bessel.py`` module, which
implements the full K_v(x) for arbitrary order v. Here only the
integer orders 0 and 1 are needed.

References
----------
.. [1] Abramowitz, M. and Stegun, I.A. (1964). *Handbook of Mathematical
   Functions*, National Bureau of Standards, §9.8.
.. [2] Cephes Mathematical Library, Stephen L. Moshier.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Final
from jaxtyping import Array, Float, jaxtyped

SAFE_X: Final[float] = 2.0


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_k0(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Modify Bessel function of the second kind, order zero.

    Parameters
    ----------
    x : Float[Array, "..."]
        Positive real input. Must satisfy x > 0.

    Returns
    -------
    k0 : Float[Array, "..."]
        Values of K_0(x).

    Notes
    -----
    Uses a two-region polynomial approximation:

    For 0 < x <= 2, the series expansion

    .. math::

        K_0(x) = -\ln(x/2)\,I_0(x) + P(t),
        \qquad t = (x/2)^2

    where :math:`I_0` is obtained from ``jax.lax.bessel_i0e`` and
    :math:`P` is a degree-6 polynomial with coefficients from [1]_.

    For x > 2, the asymptotic expansion

    .. math::

        K_0(x) = \frac{e^{-x}}{\sqrt{x}}\,Q(2/x)

    where :math:`Q` is a degree-6 polynomial.

    1. **Clamp input** --
       Ensure x > 0 for numerical safety.
    2. **Small-x branch** --
       Logarithmic series with :math:`I_0` from
       ``jax.lax.bessel_i0e``.
    3. **Large-x branch** --
       Asymptotic polynomial times
       :math:`e^{-x}/\\sqrt{x}`.
    4. **Select branch** --
       ``jnp.where`` at x = 2 boundary.

    References
    ----------
    .. [1] Abramowitz and Stegun, §9.8.1--9.8.2.
    """
    x_safe: Float[Array, "..."] = jnp.maximum(x, 1e-20)

    t_small: Float[Array, "..."] = jnp.square(x_safe / 2.0)
    p_coeffs: Float[Array, "7"] = jnp.array(
        [
            -0.57721566,
            0.42278420,
            0.23069756,
            0.03488590,
            0.00262698,
            0.00010750,
            0.00000740,
        ],
        dtype=jnp.float64,
    )
    powers_small: Float[Array, "... 7"] = jnp.power(
        t_small[..., jnp.newaxis],
        jnp.arange(7, dtype=jnp.float64),
    )
    poly_small: Float[Array, "..."] = jnp.sum(p_coeffs * powers_small, axis=-1)
    i0e_val: Float[Array, "..."] = jax.lax.bessel_i0e(x_safe)
    i0_val: Float[Array, "..."] = i0e_val * jnp.exp(jnp.abs(x_safe))
    log_term: Float[Array, "..."] = -jnp.log(x_safe / 2.0) * i0_val
    k0_small: Float[Array, "..."] = log_term + poly_small

    q_coeffs: Float[Array, "7"] = jnp.array(
        [
            1.25331414,
            -0.07832358,
            0.02189568,
            -0.01062446,
            0.00587872,
            -0.00251540,
            0.00053208,
        ],
        dtype=jnp.float64,
    )
    inv_x: Float[Array, "..."] = 2.0 / x_safe
    powers_large: Float[Array, "... 7"] = jnp.power(
        inv_x[..., jnp.newaxis],
        jnp.arange(7, dtype=jnp.float64),
    )
    poly_large: Float[Array, "..."] = jnp.sum(q_coeffs * powers_large, axis=-1)
    k0_large: Float[Array, "..."] = (
        jnp.exp(-x_safe) / jnp.sqrt(x_safe) * poly_large
    )

    k0_val: Float[Array, "..."] = jnp.where(
        x_safe <= SAFE_X, k0_small, k0_large
    )
    return k0_val


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_k1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Modify Bessel function of the second kind, order one.

    Parameters
    ----------
    x : Float[Array, "..."]
        Positive real input. Must satisfy x > 0.

    Returns
    -------
    k1 : Float[Array, "..."]
        Values of K_1(x).

    Notes
    -----
    Uses a two-region polynomial approximation:

    For 0 < x <= 2, the series expansion

    .. math::

        K_1(x) = \ln(x/2)\,I_1(x) + \frac{1}{x}\,P(t),
        \qquad t = (x/2)^2

    where :math:`I_1` is obtained from ``jax.lax.bessel_i1e`` and
    :math:`P` is a degree-6 polynomial.

    For x > 2, the asymptotic expansion

    .. math::

        K_1(x) = \frac{e^{-x}}{\sqrt{x}}\,Q(2/x)

    1. **Clamp input** --
       Ensure x > 0 for numerical safety.
    2. **Small-x branch** --
       Logarithmic series with :math:`I_1` from
       ``jax.lax.bessel_i1e``.
    3. **Large-x branch** --
       Asymptotic polynomial times
       :math:`e^{-x}/\\sqrt{x}`.
    4. **Select branch** --
       ``jnp.where`` at x = 2 boundary.

    References
    ----------
    .. [1] Abramowitz and Stegun, §9.8.3--9.8.4.
    """
    x_safe: Float[Array, "..."] = jnp.maximum(x, 1e-20)

    t_small: Float[Array, "..."] = jnp.square(x_safe / 2.0)
    p_coeffs: Float[Array, "7"] = jnp.array(
        [
            1.0,
            0.15443144,
            -0.67278579,
            -0.18156897,
            -0.01919402,
            -0.00110404,
            -0.00004686,
        ],
        dtype=jnp.float64,
    )
    powers_small: Float[Array, "... 7"] = jnp.power(
        t_small[..., jnp.newaxis],
        jnp.arange(7, dtype=jnp.float64),
    )
    poly_small: Float[Array, "..."] = jnp.sum(p_coeffs * powers_small, axis=-1)
    i1e_val: Float[Array, "..."] = jax.lax.bessel_i1e(x_safe)
    i1_val: Float[Array, "..."] = i1e_val * jnp.exp(jnp.abs(x_safe))
    log_term: Float[Array, "..."] = jnp.log(x_safe / 2.0) * i1_val
    k1_small: Float[Array, "..."] = log_term + (1.0 / x_safe) * poly_small

    q_coeffs: Float[Array, "7"] = jnp.array(
        [
            1.25331414,
            0.23498619,
            -0.03655620,
            0.01504268,
            -0.00780353,
            0.00325614,
            -0.00068245,
        ],
        dtype=jnp.float64,
    )
    inv_x: Float[Array, "..."] = 2.0 / x_safe
    powers_large: Float[Array, "... 7"] = jnp.power(
        inv_x[..., jnp.newaxis],
        jnp.arange(7, dtype=jnp.float64),
    )
    poly_large: Float[Array, "..."] = jnp.sum(q_coeffs * powers_large, axis=-1)
    k1_large: Float[Array, "..."] = (
        jnp.exp(-x_safe) / jnp.sqrt(x_safe) * poly_large
    )

    k1_val: Float[Array, "..."] = jnp.where(
        x_safe <= SAFE_X, k1_small, k1_large
    )
    return k1_val


__all__: list[str] = [
    "bessel_k0",
    "bessel_k1",
]
