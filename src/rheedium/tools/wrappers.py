"""Function wrappers for JAX compatibility.

Extended Summary
----------------
Provides decorator-style wrappers that bridge compatibility gaps
between JAX and external tools. The primary use case is ensuring
that functions decorated with ``beartype`` + ``jaxtyping`` accept
inputs from tools that produce numpy arrays instead of JAX arrays
(e.g. ``jax.test_util.check_grads``).

Routine Listings
----------------
:func:`jax_safe`
    Wrap a function to convert positional args to JAX arrays.

Notes
-----
These wrappers are intentionally minimal and composable. They do
not modify the return value or keyword arguments of the wrapped
function.
"""

from collections.abc import Callable

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any
from jaxtyping import jaxtyped


@jaxtyped(typechecker=beartype)
def jax_safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to convert positional args to JAX arrays.

    :see: :class:`~.test_wrappers.TestJaxSafe`

    Parameters
    ----------
    fn : Callable[..., Any]
        Function whose scalar positional arguments should be
        converted via ``jnp.asarray`` before dispatch.

    Returns
    -------
    wrapper : Callable[..., Any]
        Wrapped function that calls ``jnp.asarray`` on each
        positional argument.

    Notes
    -----
    1. Iterate over all positional arguments passed to ``fn``.
    2. Call ``jnp.asarray`` on each, converting numpy scalars
       and arrays to their JAX equivalents.
    3. Forward the converted arguments to ``fn`` and return
       the result unchanged.

    This is required when using ``jax.test_util.check_grads``,
    which perturbs inputs via numpy arithmetic. The perturbed
    values are numpy scalars (``f64[](numpy)``) that fail
    beartype's ``Float[Array, '']`` checks. Wrapping the
    function under test with ``jax_safe`` resolves this.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function after converting positional args."""
        converted_args: tuple[Any, ...] = tuple(jnp.asarray(a) for a in args)
        result: Any = fn(*converted_args, **kwargs)
        return result

    wrapped: Callable[..., Any] = wrapper
    return wrapped


__all__: list[str] = [
    "jax_safe",
]
