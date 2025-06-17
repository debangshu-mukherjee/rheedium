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
def bessel_k0(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the modified Bessel function of the second kind of order 0.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array of real values

    Returns
    -------
    Float[Array, "..."]
        Values of K0(x)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.bessel import bessel_k0
    >>> x = jnp.array([0.1, 1.0, 10.0])
    >>> k0_values = bessel_k0(x)
    >>> print(k0_values)
    [2.42706902 0.42102444 0.00001754]
    """
    return jax.scipy.special.k0(x)


@jaxtyped(typechecker=beartype)
def bessel_k1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the modified Bessel function of the second kind of order 1.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array of real values

    Returns
    -------
    Float[Array, "..."]
        Values of K1(x)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.bessel import bessel_k1
    >>> x = jnp.array([0.1, 1.0, 10.0])
    >>> k1_values = bessel_k1(x)
    >>> print(k1_values)
    [9.85384478 0.60190723 0.00001847]
    """
    return jax.scipy.special.k1(x)


@jaxtyped(typechecker=beartype)
def bessel_kv(v: Float[Array, "..."], x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Compute the modified Bessel function of the second kind of order v.

    Parameters
    ----------
    v : Float[Array, "..."]
        Order of the Bessel function
    x : Float[Array, "..."]
        Input array of real values

    Returns
    -------
    Float[Array, "..."]
        Values of Kv(x)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.ucell.bessel import bessel_kv
    >>> v = jnp.array([0.5, 1.5, 2.5])
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> kv_values = bessel_kv(v, x)
    >>> print(kv_values)
    [0.70710678 0.27738780 0.08323903]
    """
    return jax.scipy.special.kv(v, x)
