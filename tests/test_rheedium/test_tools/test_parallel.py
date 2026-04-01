"""Test suite for tools/parallel.py sharding utilities.

This module tests the shard_array function for distributing JAX arrays
across devices, verifying shape preservation, sharding specs, data
integrity, and edge cases.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from rheedium.tools.parallel import shard_array


class TestShardArray(chex.TestCase):
    def test_output_shape_matches_input(self) -> None:
        """Sharded array must preserve the original shape."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (8, 4))

    def test_output_values_match_input(self) -> None:
        """Sharding must not alter array values."""
        arr = jnp.arange(12.0).reshape(3, 4)
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_single_int_shard_axis(self) -> None:
        """Passing a single int for shard_axes should work."""
        arr = jnp.ones((6, 3))
        result = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)
        chex.assert_shape(result, (6, 3))

    def test_list_shard_axes(self) -> None:
        """Passing a list of axes should work."""
        arr = jnp.ones((6, 4))
        result = shard_array(arr, shard_axes=[0])
        chex.assert_shape(result, (6, 4))
        chex.assert_trees_all_close(result, arr)

    def test_shard_second_axis(self) -> None:
        """Sharding along axis 1 should produce correct spec."""
        arr = jnp.ones((3, 8))
        result = shard_array(arr, shard_axes=1)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] == "devices"

    def test_shard_first_axis_spec(self) -> None:
        """Sharding along axis 0 should produce correct spec."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=0)
        spec = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None

    def test_skip_axis_with_minus_one(self) -> None:
        """Using -1 should skip sharding (all axes None)."""
        arr = jnp.ones((4, 4))
        result = shard_array(arr, shard_axes=-1)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_skip_axis_in_list(self) -> None:
        """A list containing -1 should skip that entry."""
        arr = jnp.ones((4, 4))
        result = shard_array(arr, shard_axes=[-1])
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_explicit_devices(self) -> None:
        """Passing explicit devices should use them."""
        devices = jax.devices()
        arr = jnp.ones((8,))
        result = shard_array(arr, shard_axes=0, devices=devices)
        chex.assert_shape(result, (8,))
        chex.assert_trees_all_close(result, arr)

    def test_1d_array(self) -> None:
        """Sharding a 1D array should work."""
        arr = jnp.arange(10.0)
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (10,))
        chex.assert_trees_all_close(result, arr)

    def test_3d_array(self) -> None:
        """Sharding a 3D array along axis 0 should work."""
        arr = jnp.ones((4, 3, 2))
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (4, 3, 2))
        spec = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None
        assert spec[2] is None

    def test_axis_beyond_ndim_ignored(self) -> None:
        """An axis index >= ndim should be silently ignored."""
        arr = jnp.ones((4, 3))
        result = shard_array(arr, shard_axes=5)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None
        chex.assert_trees_all_close(result, arr)

    def test_float64_preserved(self) -> None:
        """Float64 dtype should be preserved through sharding."""
        arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        result = shard_array(arr, shard_axes=0)
        assert result.dtype == jnp.float64
        chex.assert_trees_all_close(result, arr)

    def test_complex_dtype(self) -> None:
        """Complex arrays should shard correctly."""
        arr = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_integer_array(self) -> None:
        """Integer arrays should shard correctly."""
        arr = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_sharding_is_named_sharding(self) -> None:
        """Result should use NamedSharding."""
        arr = jnp.ones((4, 4))
        result = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)
