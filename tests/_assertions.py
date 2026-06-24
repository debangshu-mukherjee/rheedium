"""Typed Chex assertion helpers for tests."""

from collections.abc import Callable

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped

from rheedium.types import CrystalStructure

from ._types import (
    JaxF32Matrix,
    JaxF32Tensor3,
    JaxF32Vector,
    NpF32Matrix,
    NpF32Tensor3,
    NpF32Vector,
)


def assert_rejects(
    fn: Callable[..., object],
    *args: object,
    match: str | None = None,
    **kwargs: object,
) -> None:
    """Assert a call rejects eagerly and under ``eqx.filter_jit``.

    Use ``eqx.filter_jit``, not bare ``jax.jit``: under ``jax.jit`` the
    ``error_if`` callback still fires and still raises, but the message is an
    uninformative Equinox blob, so a ``match=`` regex would not match.
    ``filter_jit`` surfaces the real message string.
    """
    with pytest.raises(Exception, match=match):
        fn(*args, **kwargs)

    with pytest.raises(Exception, match=match):
        eqx.filter_jit(lambda: fn(*args, **kwargs))()


@jaxtyped(typechecker=beartype)
def assert_jax_f32_vector(
    x: JaxF32Vector,
    *,
    length: int,
) -> None:
    """Assert a JAX array is a finite float32 vector."""
    chex.assert_shape(x, (length,))
    chex.assert_type(x, jnp.float32)
    chex.assert_tree_all_finite(x)


@jaxtyped(typechecker=beartype)
def assert_jax_f32_matrix(
    x: JaxF32Matrix,
    *,
    rows: int,
    cols: int,
) -> None:
    """Assert a JAX array is a finite float32 matrix."""
    chex.assert_shape(x, (rows, cols))
    chex.assert_type(x, jnp.float32)
    chex.assert_tree_all_finite(x)


@jaxtyped(typechecker=beartype)
def assert_jax_f32_tensor3(
    x: JaxF32Tensor3,
    *,
    shape: tuple[int, int, int],
) -> None:
    """Assert a JAX array is a finite float32 rank-3 tensor."""
    chex.assert_shape(x, shape)
    chex.assert_type(x, jnp.float32)
    chex.assert_tree_all_finite(x)


@jaxtyped(typechecker=beartype)
def assert_np_f32_vector(
    x: NpF32Vector,
    *,
    length: int,
) -> None:
    """Assert a NumPy array is a finite float32 vector."""
    chex.assert_shape(x, (length,))
    chex.assert_type(x, np.float32)
    chex.assert_tree_all_finite(x)


@jaxtyped(typechecker=beartype)
def assert_np_f32_matrix(
    x: NpF32Matrix,
    *,
    rows: int,
    cols: int,
) -> None:
    """Assert a NumPy array is a finite float32 matrix."""
    chex.assert_shape(x, (rows, cols))
    chex.assert_type(x, np.float32)
    chex.assert_tree_all_finite(x)


@jaxtyped(typechecker=beartype)
def assert_np_f32_tensor3(
    x: NpF32Tensor3,
    *,
    shape: tuple[int, int, int],
) -> None:
    """Assert a NumPy array is a finite float32 rank-3 tensor."""
    chex.assert_shape(x, shape)
    chex.assert_type(x, np.float32)
    chex.assert_tree_all_finite(x)


def assert_crystal_structure_arrays(
    crystal: CrystalStructure,
    *,
    n_atoms: int,
) -> None:
    """Assert a crystal has the expected finite array field shapes."""
    chex.assert_shape(crystal.frac_positions, (n_atoms, 4))
    chex.assert_shape(crystal.cart_positions, (n_atoms, 4))
    chex.assert_shape(crystal.cell_lengths, (3,))
    chex.assert_shape(crystal.cell_angles, (3,))
    chex.assert_tree_all_finite(crystal.frac_positions)
    chex.assert_tree_all_finite(crystal.cart_positions)
    chex.assert_tree_all_finite(crystal.cell_lengths)
    chex.assert_tree_all_finite(crystal.cell_angles)
