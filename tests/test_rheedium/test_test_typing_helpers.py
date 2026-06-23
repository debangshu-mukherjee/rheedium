"""Smoke tests for shared typed test helpers."""

import chex
import jax.numpy as jnp

from rheedium.types import CrystalStructure

from .._assertions import (
    assert_crystal_structure_arrays,
    assert_jax_f32_matrix,
    assert_jax_f32_tensor3,
    assert_jax_f32_vector,
    assert_np_f32_matrix,
    assert_np_f32_tensor3,
    assert_np_f32_vector,
)
from .._factories import (
    make_jax_f32_matrix,
    make_jax_f32_tensor3,
    make_jax_f32_vector,
    make_key,
    make_np_f32_matrix,
    make_np_f32_tensor3,
    make_np_f32_vector,
    make_si_crystal_2atom,
)
from .._types import (
    JaxF32Matrix,
    JaxF32Tensor3,
    JaxF32Vector,
    Key,
    NpF32Matrix,
    NpF32Tensor3,
    NpF32Vector,
)


def test_typed_array_factories_and_assertions() -> None:
    """Typed factories return arrays accepted by typed Chex helpers."""
    key: Key = make_key(123)
    jax_vector: JaxF32Vector = make_jax_f32_vector(4, fill=1.0)
    jax_matrix: JaxF32Matrix = make_jax_f32_matrix(2, 3, fill=2.0)
    jax_tensor: JaxF32Tensor3 = make_jax_f32_tensor3(2, 3, 4, fill=3.0)
    np_vector: NpF32Vector = make_np_f32_vector(4, fill=1.0)
    np_matrix: NpF32Matrix = make_np_f32_matrix(2, 3, fill=2.0)
    np_tensor: NpF32Tensor3 = make_np_f32_tensor3(2, 3, 4, fill=3.0)

    assert_jax_f32_vector(jax_vector, length=4)
    assert_jax_f32_matrix(jax_matrix, rows=2, cols=3)
    assert_jax_f32_tensor3(jax_tensor, shape=(2, 3, 4))
    assert_np_f32_vector(np_vector, length=4)
    assert_np_f32_matrix(np_matrix, rows=2, cols=3)
    assert_np_f32_tensor3(np_tensor, shape=(2, 3, 4))
    chex.assert_shape(key, ())
    chex.assert_trees_all_close(jax_matrix[0, 0], jnp.asarray(2.0))


def test_typed_crystal_factory_and_assertion() -> None:
    """Project-specific typed factories produce valid crystal structures."""
    crystal: CrystalStructure = make_si_crystal_2atom()

    assert_crystal_structure_arrays(crystal, n_atoms=2)
