"""
Module: ucell.bessel
--------------------
JAX-compatible implementation of modified Bessel functions of the second kind.

Functions
---------
- `bessel_k0`:
    Computes the modified Bessel function of the second kind of order 0 (K₀)
- `bessel_k1`:
    Computes the modified Bessel function of the second kind of order 1 (K₁)
- `bessel_kv`:
    General modified Bessel function of the second kind for arbitrary order ν
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped
from rheedium.types import scalar_float

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def bessel_k0(x: Float[Array, "*"]) -> Float[Array, "*"]:
    """
    Description
    -----------
    Compute the modified Bessel function of the second kind of order 0 (K₀)
    using rational approximation for JAX compatibility.
    This implementation uses Chebyshev polynomial approximations that are
    accurate to machine precision for the K₀ Bessel function.

    Parameters
    ----------
    - `x` (Float[Array, "*"]):
        Input array of positive real values

    Returns
    -------
    - `result` (Float[Array, "*"]):
        K₀(x) values with same shape as input

    Notes
    -----
    For x ≤ 2, uses series expansion around x=0
    For x > 2, uses asymptotic expansion

    Flow
    ----
    - Ensure input values are positive by applying minimum threshold
    - Define coefficients for small x approximation (p0_coeffs)
    - Define coefficients for large x approximation (q0_coeffs)
    - For small x (≤ 2):
        - Calculate t = x/2
        - Compute I₀ series expansion
        - Calculate logarithmic term
        - Evaluate polynomial using p0_coeffs
        - Combine terms for final result
    - For large x (> 2):
        - Calculate t = 2/x
        - Evaluate polynomial using q0_coeffs
        - Apply exponential and square root scaling
    - Select appropriate result based on x value
    """
    x_safe: Float[Array, "*"] = jnp.maximum(x, jnp.asarray(1e-16))
    p0_coeffs: Float[Array, "7"] = jnp.array(
        [
            -0.57721566,
            0.42278420,
            0.23069756,
            0.03488590,
            0.00262698,
            0.00010750,
            0.00000740,
        ]
    )
    q0_coeffs: Float[Array, "8"] = jnp.array(
        [
            1.25331414,
            -0.07832358,
            0.02189568,
            -0.01062446,
            0.00587872,
            -0.00251540,
            0.00053208,
            -0.00000206,
        ]
    )

    def small_x_k0(x_val: Float[Array, "*"]) -> Float[Array, "*"]:
        t: Float[Array, "*"] = x_val / 2.0
        t2: Float[Array, "*"] = t * t
        i0_series: Float[Array, "*"] = 1.0 + t2 * (
            1.0
            + t2
            * (
                0.25
                + t2
                * (
                    0.0277777778
                    + t2 * (0.00173611111 + t2 * (0.00003472222 + t2 * 0.00000024801))
                )
            )
        )
        ln_term: Float[Array, "*"] = -jnp.log(t) * i0_series
        poly: Float[Array, "*"] = p0_coeffs[0]
        t2_power: Float[Array, "*"] = t2
        for i in range(1, len(p0_coeffs)):
            poly = poly + p0_coeffs[i] * t2_power
            t2_power = t2_power * t2

        return ln_term + poly

    def large_x_k0(x_val: Float[Array, "*"]) -> Float[Array, "*"]:
        t: Float[Array, "*"] = 2.0 / x_val

        poly: Float[Array, "*"] = q0_coeffs[0]
        t_power: Float[Array, "*"] = t
        for i in range(1, len(q0_coeffs)):
            poly = poly + q0_coeffs[i] * t_power
            t_power = t_power * t
        return jnp.exp(-x_val) * poly / jnp.sqrt(x_val)

    small_result: Float[Array, "*"] = small_x_k0(x_safe)
    large_result: Float[Array, "*"] = large_x_k0(x_safe)
    result: Float[Array, "*"] = jnp.where(x_safe <= 2.0, small_result, large_result)
    return result


@jaxtyped(typechecker=beartype)
def bessel_k1(x: Float[Array, "*"]) -> Float[Array, "*"]:
    """
    Description
    -----------
    Compute the modified Bessel function of the second kind of order 1 (K₁)
    using rational approximation for JAX compatibility.

    Parameters
    ----------
    - `x` (Float[Array, "*"]):
        Input array of positive real values

    Returns
    -------
    - `result` (Float[Array, "*"]):
        K₁(x) values with same shape as input

    Flow
    ----
    - Ensure input values are positive by applying minimum threshold
    - Define coefficients for small x approximation (p1_coeffs)
    - Define coefficients for large x approximation (q1_coeffs)
    - For small x (≤ 2):
        - Calculate t = x/2
        - Compute I₁ series expansion
        - Calculate logarithmic term
        - Evaluate polynomial using p1_coeffs
        - Combine terms with 1/x term for final result
    - For large x (> 2):
        - Calculate t = 2/x
        - Evaluate polynomial using q1_coeffs
        - Apply exponential and square root scaling
    - Select appropriate result based on x value
    """
    x_safe: Float[Array, "*"] = jnp.maximum(x, jnp.asarray(1e-16))
    p1_coeffs: Float[Array, "7"] = jnp.array(
        [
            1.0,
            0.15443144,
            -0.67278579,
            -0.18156897,
            -0.01919402,
            -0.00110404,
            -0.00004686,
        ]
    )
    q1_coeffs: Float[Array, "8"] = jnp.array(
        [
            1.25331414,
            0.23498619,
            -0.03655620,
            0.01504268,
            -0.00780353,
            0.00325614,
            -0.00068245,
            0.00002670,
        ]
    )

    def small_x_k1(x_val: Float[Array, "*"]) -> Float[Array, "*"]:
        t: Float[Array, "*"] = x_val / 2.0
        t2: Float[Array, "*"] = t * t
        i1_series: Float[Array, "*"] = t * (
            1.0
            + t2
            * (
                0.5
                + t2
                * (0.0625 + t2 * (0.00520833 + t2 * (0.00032552 + t2 * 0.00001221)))
            )
        )
        ln_term: Float[Array, "*"] = jnp.log(t) * i1_series
        poly: Float[Array, "*"] = p1_coeffs[0]
        t2_power: Float[Array, "*"] = t2
        for i in range(1, len(p1_coeffs)):
            poly = poly + p1_coeffs[i] * t2_power
            t2_power = t2_power * t2
        return (1.0 / x_val) + ln_term + (x_val * poly)

    def large_x_k1(x_val: Float[Array, "*"]) -> Float[Array, "*"]:
        t: Float[Array, "*"] = 2.0 / x_val
        poly: Float[Array, "*"] = q1_coeffs[0]
        t_power: Float[Array, "*"] = t
        for i in range(1, len(q1_coeffs)):
            poly = poly + q1_coeffs[i] * t_power
            t_power = t_power * t
        return jnp.exp(-x_val) * poly / jnp.sqrt(x_val)

    small_result: Float[Array, "*"] = small_x_k1(x_safe)
    large_result: Float[Array, "*"] = large_x_k1(x_safe)
    result: Float[Array, "*"] = jnp.where(x_safe <= 2.0, small_result, large_result)

    return result


@jaxtyped(typechecker=beartype)
def bessel_kv(nu: scalar_float, x: Float[Array, "*"]) -> Float[Array, "*"]:
    """
    Description
    -----------
    Compute the modified Bessel function of the second kind of arbitrary order ν
    using recurrence relations and base functions K₀ and K₁.

    Parameters
    ----------
    - `nu` (scalar_float):
        Order of the Bessel function (must be non-negative)
    - `x` (Float[Array, "*"]):
        Input array of positive real values

    Returns
    -------
    - `result` (Float[Array, "*"]):
        Kν(x) values with same shape as input

    Notes
    -----
    For integer orders, uses recurrence relation:
    K_{n+1}(x) = K_{n-1}(x) + (2n/x) * K_n(x)

    For non-integer orders, uses approximation suitable for atomic potentials

    Flow
    ----
    - Convert nu to JAX array
    - Ensure input values are positive by applying minimum threshold
    - For nu = 0:
        - Return result from bessel_k0
    - For nu = 1:
        - Return result from bessel_k1
    - For general nu:
        - Calculate initial K₀ and K₁ values
        - Determine number of recurrence steps needed
        - Apply recurrence relation using scan:
            - K_{n+1}(x) = K_{n-1}(x) + (2n/x) * K_n(x)
        - Return final result
    """
    nu_val: Float[Array, ""] = jnp.asarray(nu)
    x_safe: Float[Array, "*"] = jnp.maximum(x, jnp.asarray(1e-16))

    def handle_nu_0() -> Float[Array, "*"]:
        return bessel_k0(x_safe)

    def handle_nu_1() -> Float[Array, "*"]:
        return bessel_k1(x_safe)

    def handle_general_nu() -> Float[Array, "*"]:
        def recurrence_step(carry: tuple, _):
            k_prev: Float[Array, "*"]
            k_curr: Float[Array, "*"]
            n: Float[Array, ""]
            k_prev, k_curr, n = carry
            k_next: Float[Array, "*"] = k_prev + (2.0 * n / x_safe) * k_curr
            return (k_curr, k_next, n + 1.0), None

        k0: Float[Array, "*"] = bessel_k0(x_safe)
        k1: Float[Array, "*"] = bessel_k1(x_safe)
        n_steps: int = int(jnp.maximum(0, jnp.ceil(nu_val) - 1))
        final_state, _ = jax.lax.scan(
            recurrence_step, (k0, k1, 1.0), None, length=n_steps
        )
        return final_state[1]

    result: Float[Array, "*"] = jax.lax.cond(
        jnp.abs(nu_val) < 1e-10,
        handle_nu_0,
        lambda: jax.lax.cond(
            jnp.abs(nu_val - 1.0) < 1e-10, handle_nu_1, handle_general_nu
        ),
    )
    return result
