"""Special-function kernels shared across the rheedium package.

This module centralizes the JAX-compatible Bessel implementations used by
the simulation and crystallography packages so domain modules can depend on a
single numerical toolbox.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Final, Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.types import scalar_float

SAFE_X: Final[float] = 2.0


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_k0(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute modified Bessel function of the second kind, order zero."""
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

    return jnp.where(x_safe <= SAFE_X, k0_small, k0_large)


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_k1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute modified Bessel function of the second kind, order one."""
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

    return jnp.where(x_safe <= SAFE_X, k1_small, k1_large)


@jaxtyped(typechecker=beartype)
def _bessel_iv_series(
    v_order: scalar_float,
    x_val: Float[Array, "..."],
    dtype: jnp.dtype,
) -> Float[Array, "..."]:
    """Compute I_v(x) using series expansion for Bessel function."""
    x_half: Float[Array, "..."] = x_val / 2.0
    x_half_v: Float[Array, "..."] = jnp.power(x_half, v_order)
    x2_quarter: Float[Array, "..."] = (x_val * x_val) / 4.0
    max_terms: int = 20
    k_arr: Float[Array, "20"] = jnp.arange(max_terms, dtype=dtype)
    gamma_v_plus_1: Float[Array, ""] = jax.scipy.special.gamma(v_order + 1)
    gamma_terms: Float[Array, "20"] = jax.scipy.special.gamma(
        k_arr + v_order + 1
    )
    factorial_terms: Float[Array, "20"] = jax.scipy.special.factorial(k_arr)
    powers: Float[Array, "... 20"] = jnp.power(
        x2_quarter[..., jnp.newaxis], k_arr
    )
    series_terms: Float[Array, "... 20"] = powers / (
        factorial_terms * gamma_terms / gamma_v_plus_1
    )
    return x_half_v / gamma_v_plus_1 * jnp.sum(series_terms, axis=-1)


@jaxtyped(typechecker=beartype)
def _bessel_k0_series(
    x: Float[Array, "..."],
    dtype: jnp.dtype,
) -> Float[Array, "..."]:
    """Compute K_0(x) using series expansion."""
    i0: Float[Array, "..."] = jax.scipy.special.i0(x)
    coeffs: Float[Array, "7"] = jnp.array(
        [
            -0.57721566,
            0.42278420,
            0.23069756,
            0.03488590,
            0.00262698,
            0.00010750,
            0.00000740,
        ],
        dtype=dtype,
    )
    x2: Float[Array, "..."] = (x * x) / 4.0
    powers: Float[Array, "... 7"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(7)
    )
    poly: Float[Array, "..."] = jnp.sum(coeffs * powers, axis=-1)
    log_term: Float[Array, "..."] = -jnp.log(x / 2.0) * i0
    return log_term + poly


@jaxtyped(typechecker=beartype)
def _bessel_kn_recurrence(
    n: Int[Array, ""],
    x: Float[Array, "..."],
    k0: Float[Array, "..."],
    k1: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute K_n(x) using recurrence relation."""

    def _compute_kn() -> Float[Array, "..."]:
        init: Tuple[Float[Array, "..."], Float[Array, "..."]] = (k0, k1)
        max_n: int = 20
        indices: Float[Array, "19"] = jnp.arange(1, max_n, dtype=jnp.float32)

        def masked_step(
            carry: Tuple[Float[Array, "..."], Float[Array, "..."]],
            i: Float[Array, ""],
        ) -> Tuple[
            Tuple[Float[Array, "..."], Float[Array, "..."]],
            Float[Array, "..."],
        ]:
            k_prev2: Float[Array, "..."]
            k_prev1: Float[Array, "..."]
            k_prev2, k_prev1 = carry
            mask: Bool[Array, ""] = i < n
            two_i_over_x: Float[Array, "..."] = 2.0 * i / x
            k_curr: Float[Array, "..."] = two_i_over_x * k_prev1 + k_prev2
            k_curr = jnp.where(mask, k_curr, k_prev1)
            return (k_prev1, k_curr), k_curr

        carry: Tuple[Float[Array, "..."], Float[Array, "..."]]
        carry, _ = jax.lax.scan(masked_step, init, indices)
        return carry[1]

    return jnp.where(n == 0, k0, jnp.where(n == 1, k1, _compute_kn()))


@jaxtyped(typechecker=beartype)
def _bessel_kv_small_non_integer(
    v: scalar_float,
    x: Float[Array, "..."],
    dtype: jnp.dtype,
) -> Float[Array, "..."]:
    """Compute K_v(x) for small x and non-integer v."""
    error_bound: Float[Array, ""] = jnp.asarray(1e-10)
    iv_pos: Float[Array, "..."] = _bessel_iv_series(v, x, dtype)
    iv_neg: Float[Array, "..."] = _bessel_iv_series(-v, x, dtype)
    sin_piv: Float[Array, ""] = jnp.sin(jnp.pi * v)
    pi_over_2sin: Float[Array, ""] = jnp.pi / (2.0 * sin_piv)
    iv_diff: Float[Array, "..."] = iv_neg - iv_pos
    return jnp.where(
        jnp.abs(sin_piv) > error_bound, pi_over_2sin * iv_diff, 0.0
    )


@jaxtyped(typechecker=beartype)
def _bessel_kv_small_integer(
    v: Float[Array, ""],
    x: Float[Array, "..."],
    dtype: jnp.dtype,
) -> Float[Array, "..."]:
    """Compute K_v(x) for small x and integer v."""
    v_int: Float[Array, ""] = jnp.round(v)
    n: Int[Array, ""] = jnp.abs(v_int).astype(jnp.int32)

    k0: Float[Array, "..."] = _bessel_k0_series(x, dtype)

    i1: Float[Array, "..."] = jax.scipy.special.i1(x)
    k1_coeffs: Float[Array, "5"] = jnp.array(
        [1.0, -0.5, 0.0625, -0.03125, 0.0234375], dtype=dtype
    )
    x2: Float[Array, "..."] = (x * x) / 4.0
    k1_powers: Float[Array, "... 5"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(5)
    )
    k1_poly: Float[Array, "..."] = jnp.sum(k1_coeffs * k1_powers, axis=-1)
    log_i1_term: Float[Array, "..."] = -jnp.log(x / 2.0) * i1
    k1: Float[Array, "..."] = log_i1_term + k1_poly / x

    kn_result: Float[Array, "..."] = _bessel_kn_recurrence(n, x, k0, k1)
    return jnp.where(v >= 0, kn_result, kn_result)


@jaxtyped(typechecker=beartype)
def _bessel_kv_large(
    v: scalar_float,
    x: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Asymptotic expansion for K_v(x) for large x."""
    sqrt_term: Float[Array, "..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_term: Float[Array, "..."] = jnp.exp(-x)

    v2: Float[Array, ""] = v * v
    four_v2: Float[Array, ""] = 4.0 * v2
    a0: Float[Array, ""] = 1.0
    a1: Float[Array, ""] = (four_v2 - 1.0) / 8.0
    a2: Float[Array, ""] = (four_v2 - 1.0) * (four_v2 - 9.0) / (2.0 * 64.0)
    a3: Float[Array, ""] = (
        (four_v2 - 1.0) * (four_v2 - 9.0) * (four_v2 - 25.0) / (6.0 * 512.0)
    )
    a4: Float[Array, ""] = (
        (four_v2 - 1.0)
        * (four_v2 - 9.0)
        * (four_v2 - 25.0)
        * (four_v2 - 49.0)
        / (24.0 * 4096.0)
    )

    z: Float[Array, "..."] = 1.0 / x
    poly: Float[Array, "..."] = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))
    return sqrt_term * exp_term * poly


@jaxtyped(typechecker=beartype)
def _bessel_k_half(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute special case K_{1/2}(x) = sqrt(pi/(2x)) * exp(-x)."""
    sqrt_pi_over_2x: Float[Array, "..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_neg_x: Float[Array, "..."] = jnp.exp(-x)
    return sqrt_pi_over_2x * exp_neg_x


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_kv(v: scalar_float, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute the modified Bessel function of the second kind K_v(x)."""
    v = jnp.asarray(v)
    x = jnp.asarray(x)
    dtype: jnp.dtype = x.dtype

    v_int: Float[Array, ""] = jnp.round(v)
    epsilon_tolerance: float = 1e-10
    is_integer: Bool[Array, ""] = jnp.abs(v - v_int) < epsilon_tolerance

    small_x_non_int: Float[Array, "..."] = _bessel_kv_small_non_integer(
        v, x, dtype
    )
    small_x_int: Float[Array, "..."] = _bessel_kv_small_integer(v, x, dtype)
    small_x_vals: Float[Array, "..."] = jnp.where(
        is_integer, small_x_int, small_x_non_int
    )

    large_x_vals: Float[Array, "..."] = _bessel_kv_large(v, x)

    small_x_threshold: float = 2.0
    general_result: Float[Array, "..."] = jnp.where(
        x <= small_x_threshold, small_x_vals, large_x_vals
    )

    k_half_vals: Float[Array, "..."] = _bessel_k_half(x)
    is_half: Bool[Array, ""] = jnp.abs(v - 0.5) < epsilon_tolerance
    return jnp.where(is_half, k_half_vals, general_result)


__all__: list[str] = [
    "_bessel_iv_series",
    "_bessel_k0_series",
    "_bessel_k_half",
    "_bessel_kn_recurrence",
    "_bessel_kv_large",
    "_bessel_kv_small_integer",
    "_bessel_kv_small_non_integer",
    "bessel_k0",
    "bessel_k1",
    "bessel_kv",
]
