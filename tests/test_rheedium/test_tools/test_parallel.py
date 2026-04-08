"""Test suite for tools/parallel.py sharding utilities.

This module tests the shard_array function for distributing JAX arrays
across devices.  With ``XLA_FLAGS=--xla_force_host_platform_device_count=8``
set in conftest, ``jax.devices()`` returns 8 virtual CPU devices, enabling
meaningful multi-device and ``pmap`` tests via ``chex.variants``.
"""

import chex
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from rheedium.tools.parallel import shard_array


class TestShardArray(chex.TestCase):
    """Core shard_array tests with 8-device-divisible shapes."""

    def test_output_shape_matches_input(self):
        """Sharded array must preserve the original shape."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (8, 4))

    def test_output_values_match_input(self):
        """Sharding must not alter array values."""
        arr = jnp.arange(24.0).reshape(8, 3)
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_single_int_shard_axis(self):
        """Passing a single int for shard_axes should work."""
        arr = jnp.ones((8, 3))
        result = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)
        chex.assert_shape(result, (8, 3))

    def test_list_shard_axes(self):
        """Passing a list of axes should work."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=[0])
        chex.assert_shape(result, (8, 4))
        chex.assert_trees_all_close(result, arr)

    def test_shard_second_axis(self):
        """Sharding along axis 1 should produce correct spec."""
        arr = jnp.ones((3, 8))
        result = shard_array(arr, shard_axes=1)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] == "devices"

    def test_shard_first_axis_spec(self):
        """Sharding along axis 0 should produce correct spec."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=0)
        spec = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None

    def test_skip_axis_with_minus_one(self):
        """Using -1 should skip sharding (all axes None)."""
        arr = jnp.ones((8, 8))
        result = shard_array(arr, shard_axes=-1)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_skip_axis_in_list(self):
        """A list containing -1 should skip that entry."""
        arr = jnp.ones((8, 8))
        result = shard_array(arr, shard_axes=[-1])
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_explicit_devices(self):
        """Passing explicit devices should use them."""
        devices = jax.devices()
        arr = jnp.ones((8,))
        result = shard_array(arr, shard_axes=0, devices=devices)
        chex.assert_shape(result, (8,))
        chex.assert_trees_all_close(result, arr)

    def test_1d_array(self):
        """Sharding a 1D array should work."""
        arr = jnp.arange(16.0)
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (16,))
        chex.assert_trees_all_close(result, arr)

    def test_3d_array(self):
        """Sharding a 3D array along axis 0 should work."""
        arr = jnp.ones((8, 3, 2))
        result = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (8, 3, 2))
        spec = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None
        assert spec[2] is None

    def test_axis_beyond_ndim_ignored(self):
        """An axis index >= ndim should be silently ignored."""
        arr = jnp.ones((8, 3))
        result = shard_array(arr, shard_axes=5)
        spec = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None
        chex.assert_trees_all_close(result, arr)

    def test_float64_preserved(self):
        """Float64 dtype should be preserved through sharding."""
        arr = jnp.arange(8.0, dtype=jnp.float64)
        result = shard_array(arr, shard_axes=0)
        assert result.dtype == jnp.float64
        chex.assert_trees_all_close(result, arr)

    def test_complex_dtype(self):
        """Complex arrays should shard correctly."""
        arr = jnp.arange(8.0) + 1j * jnp.arange(8.0, 16.0)
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_integer_array(self):
        """Integer arrays should shard correctly."""
        arr = jnp.arange(8, dtype=jnp.int32)
        result = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_sharding_is_named_sharding(self):
        """Result should use NamedSharding."""
        arr = jnp.ones((8, 4))
        result = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)


class TestShardArrayMultiDevice(chex.TestCase):
    """Tests that verify real multi-device distribution."""

    def test_eight_devices_available(self):
        """Conftest must expose 8 virtual CPU devices."""
        assert len(jax.devices()) == 8

    def test_sharded_across_all_devices(self):
        """Array sharded on axis 0 should span all 8 devices."""
        arr = jnp.arange(24.0).reshape(8, 3)
        result = shard_array(arr, shard_axes=0)
        device_set = result.sharding.device_set
        assert len(device_set) == 8

    def test_subset_devices(self):
        """Sharding across a subset of devices should work."""
        devices = jax.devices()[:4]
        arr = jnp.arange(12.0).reshape(4, 3)
        result = shard_array(arr, shard_axes=0, devices=devices)
        assert len(result.sharding.device_set) == 4
        chex.assert_trees_all_close(result, arr)

    def test_shard_values_roundtrip(self):
        """Data gathered from sharded array must match original."""
        arr = jnp.arange(16.0).reshape(8, 2)
        result = shard_array(arr, shard_axes=0)
        gathered = jax.device_get(result)
        chex.assert_trees_all_close(gathered, arr)


class TestPmapCompatibility(chex.TestCase):
    """Test pmap execution across 8 virtual devices."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_on_sharded_array(self):
        """JIT-compiled function should work on sharded data."""

        @self.variant
        def add_one(x):
            return x + 1.0

        arr = jnp.arange(8.0)
        sharded = shard_array(arr, shard_axes=0)
        result = add_one(sharded)
        expected = jnp.arange(1.0, 9.0)
        chex.assert_trees_all_close(result, expected)

    def test_pmap_elementwise(self):
        """pmap should distribute elementwise ops across devices."""
        fn = jax.pmap(lambda x: x**2)
        arr = jnp.arange(8.0).reshape(8, 1)
        result = fn(arr)
        expected = arr**2
        chex.assert_trees_all_close(result, expected)

    def test_pmap_reduction(self):
        """pmap with inner sum should reduce per-device slices."""
        fn = jax.pmap(lambda x: jnp.sum(x))
        arr = jnp.ones((8, 4))
        result = fn(arr)
        expected = jnp.full((8,), 4.0)
        chex.assert_trees_all_close(result, expected)

    def test_pmap_preserves_dtype(self):
        """pmap should preserve float64 dtype."""
        fn = jax.pmap(lambda x: x * 2.0)
        arr = jnp.ones((8, 2), dtype=jnp.float64)
        result = fn(arr)
        assert result.dtype == jnp.float64
        chex.assert_trees_all_close(result, arr * 2.0)

    def test_pmap_trig_identity(self):
        """pmap should handle sin^2 + cos^2 = 1 across devices."""
        fn = jax.pmap(lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2)
        arr = jnp.linspace(0, 6.28, 8 * 4).reshape(8, 4)
        result = fn(arr)
        expected = jnp.ones_like(arr)
        chex.assert_trees_all_close(result, expected, atol=1e-6)

    def test_pmap_with_broadcast(self):
        """pmap should handle scalar broadcast."""
        fn = jax.pmap(lambda x: x + 10.0)
        arr = jnp.arange(8.0).reshape(8, 1)
        result = fn(arr)
        expected = arr + 10.0
        chex.assert_trees_all_close(result, expected)
