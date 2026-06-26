"""Test suite for JAX compatibility wrappers.

Extended Summary
----------------
Validates helper wrappers that adapt external numerical inputs to the
JAX array contracts expected by typed Rheedium functions.
"""

from collections.abc import Callable

import chex
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.tools.wrappers import jax_safe


class TestJaxSafe(chex.TestCase):
    """Validate :func:`~rheedium.tools.wrappers.jax_safe`.

    :see: :func:`~rheedium.tools.jax_safe`
    """

    def test_converts_numpy_scalar_to_jax_scalar(self) -> None:
        r"""Wrapped functions accept NumPy scalar inputs as JAX scalars.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Wrapped functions
        accept NumPy scalar inputs as JAX scalars.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_wrappers``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """

        @jaxtyped(typechecker=beartype)
        def _double_scalar(value: Float[Array, ""]) -> Float[Array, ""]:
            """Double one scalar after jaxtyping validation."""
            doubled: Float[Array, ""] = value * 2.0
            return doubled

        wrapped: Callable[..., Float[Array, ""]] = jax_safe(_double_scalar)
        doubled: Float[Array, ""] = wrapped(np.float64(3.0))

        chex.assert_trees_all_close(doubled, jnp.asarray(6.0))
