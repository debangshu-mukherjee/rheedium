"""Typed factories for test data.

Extended Summary
----------------
Provides shared, typed fixture builders used by multiple test modules so
array construction stays consistent with the runtime jaxtyping checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from rheedium.types import CrystalStructure
from rheedium.types.crystal_types import create_crystal_structure

from ._types import (
    JaxF32Matrix,
    JaxF32Tensor3,
    JaxF32Vector,
    Key,
    NpF32Matrix,
    NpF32Tensor3,
    NpF32Vector,
)


def make_key(seed: int = 0) -> Key:
    """Create a deterministic JAX random key."""
    key: Key = jax.random.key(seed)
    return key


@jaxtyped(typechecker=beartype)
def make_jax_f32_vector(length: int, *, fill: float = 0.0) -> JaxF32Vector:
    """Create a float32 JAX vector with the requested length."""
    vector: JaxF32Vector = jnp.full((length,), fill, dtype=jnp.float32)
    return vector


@jaxtyped(typechecker=beartype)
def make_jax_f32_matrix(
    rows: int,
    cols: int,
    *,
    fill: float = 0.0,
) -> JaxF32Matrix:
    """Create a float32 JAX matrix with the requested shape."""
    matrix: JaxF32Matrix = jnp.full((rows, cols), fill, dtype=jnp.float32)
    return matrix


@jaxtyped(typechecker=beartype)
def make_jax_f32_tensor3(
    x: int,
    y: int,
    z: int,
    *,
    fill: float = 0.0,
) -> JaxF32Tensor3:
    """Create a float32 JAX rank-3 tensor with the requested shape."""
    tensor: JaxF32Tensor3 = jnp.full((x, y, z), fill, dtype=jnp.float32)
    return tensor


@jaxtyped(typechecker=beartype)
def make_np_f32_vector(length: int, *, fill: float = 0.0) -> NpF32Vector:
    """Create a float32 NumPy vector with the requested length."""
    vector: NpF32Vector = np.full((length,), fill, dtype=np.float32)
    return vector


@jaxtyped(typechecker=beartype)
def make_np_f32_matrix(
    rows: int,
    cols: int,
    *,
    fill: float = 0.0,
) -> NpF32Matrix:
    """Create a float32 NumPy matrix with the requested shape."""
    matrix: NpF32Matrix = np.full((rows, cols), fill, dtype=np.float32)
    return matrix


@jaxtyped(typechecker=beartype)
def make_np_f32_tensor3(
    x: int,
    y: int,
    z: int,
    *,
    fill: float = 0.0,
) -> NpF32Tensor3:
    """Create a float32 NumPy rank-3 tensor with the requested shape."""
    tensor: NpF32Tensor3 = np.full((x, y, z), fill, dtype=np.float32)
    return tensor


def make_si_crystal_2atom() -> CrystalStructure:
    """Create a two-atom silicon crystal for fast simulation tests."""
    a_si: float = 5.431
    frac_coords: JaxF32Matrix = jnp.array(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        dtype=jnp.float32,
    )
    cart_coords: JaxF32Matrix = frac_coords * a_si
    atomic_numbers: JaxF32Vector = jnp.full((2,), 14.0, dtype=jnp.float32)
    frac_positions: JaxF32Matrix = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_positions: JaxF32Matrix = jnp.column_stack(
        [cart_coords, atomic_numbers]
    )
    cell_lengths: JaxF32Vector = jnp.array(
        [a_si, a_si, a_si],
        dtype=jnp.float32,
    )
    cell_angles: JaxF32Vector = jnp.array(
        [90.0, 90.0, 90.0],
        dtype=jnp.float32,
    )
    crystal: CrystalStructure = create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    return crystal
