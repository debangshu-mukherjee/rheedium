"""Test suite for tools/parallel.py sharding utilities.

This module tests the shard_array function for distributing JAX arrays
across devices.  With ``XLA_FLAGS=--xla_force_host_platform_device_count=8``
set in conftest, ``jax.devices()`` returns 8 virtual CPU devices, enabling
meaningful multi-device and ``pmap`` tests via ``chex.variants``.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Complex, Float, Integer

from rheedium.tools.parallel import shard_array


class TestShardArray(chex.TestCase):
    """Core shard_array tests with 8-device-divisible shapes."""

    def test_output_shape_matches_input(self) -> None:
        """Sharded array must preserve the original shape."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 4))
        result: Float[Array, "devices cols"] = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (8, 4))

    def test_output_values_match_input(self) -> None:
        """Sharding must not alter array values."""
        arr: Float[Array, "devices coords"] = jnp.arange(24.0).reshape(8, 3)
        result: Float[Array, "devices coords"] = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_single_int_shard_axis(self) -> None:
        """Passing a single int for shard_axes should work."""
        arr: Float[Array, "devices coords"] = jnp.ones((8, 3))
        result: Float[Array, "devices coords"] = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)
        chex.assert_shape(result, (8, 3))

    def test_list_shard_axes(self) -> None:
        """Passing a list of axes should work."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 4))
        result: Float[Array, "devices cols"] = shard_array(arr, shard_axes=[0])
        chex.assert_shape(result, (8, 4))
        chex.assert_trees_all_close(result, arr)

    def test_shard_second_axis(self) -> None:
        """Sharding along axis 1 should produce correct spec."""
        arr: Float[Array, "rows devices"] = jnp.ones((3, 8))
        result: Float[Array, "rows devices"] = shard_array(arr, shard_axes=1)
        spec: Any = result.sharding.spec
        assert spec[0] is None
        assert spec[1] == "devices"

    def test_shard_first_axis_spec(self) -> None:
        """Sharding along axis 0 should produce correct spec."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 4))
        result: Float[Array, "devices cols"] = shard_array(arr, shard_axes=0)
        spec: Any = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None

    def test_skip_axis_with_minus_one(self) -> None:
        """Using -1 should skip sharding (all axes None)."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 8))
        result: Float[Array, "devices cols"] = shard_array(arr, shard_axes=-1)
        spec: Any = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_skip_axis_in_list(self) -> None:
        """A list containing -1 should skip that entry."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 8))
        result: Float[Array, "devices cols"] = shard_array(
            arr, shard_axes=[-1]
        )
        spec: Any = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None

    def test_explicit_devices(self) -> None:
        """Passing explicit devices should use them."""
        devices: Any = jax.devices()
        arr: Float[Array, "devices"] = jnp.ones((8,))
        result: Float[Array, "devices"] = shard_array(
            arr, shard_axes=0, devices=devices
        )
        chex.assert_shape(result, (8,))
        chex.assert_trees_all_close(result, arr)

    def test_1d_array(self) -> None:
        """Sharding a 1D array should work."""
        arr: Float[Array, "points"] = jnp.arange(16.0)
        result: Float[Array, "points"] = shard_array(arr, shard_axes=0)
        chex.assert_shape(result, (16,))
        chex.assert_trees_all_close(result, arr)

    def test_3d_array(self) -> None:
        """Sharding a 3D array along axis 0 should work."""
        arr: Float[Array, "devices rows cols"] = jnp.ones((8, 3, 2))
        result: Float[Array, "devices rows cols"] = shard_array(
            arr, shard_axes=0
        )
        chex.assert_shape(result, (8, 3, 2))
        spec: Any = result.sharding.spec
        assert spec[0] == "devices"
        assert spec[1] is None
        assert spec[2] is None

    def test_axis_beyond_ndim_ignored(self) -> None:
        """An axis index >= ndim should be silently ignored."""
        arr: Float[Array, "devices coords"] = jnp.ones((8, 3))
        result: Float[Array, "devices coords"] = shard_array(arr, shard_axes=5)
        spec: Any = result.sharding.spec
        assert spec[0] is None
        assert spec[1] is None
        chex.assert_trees_all_close(result, arr)

    def test_float64_preserved(self) -> None:
        """Float64 dtype should be preserved through sharding."""
        arr: Float[Array, "devices"] = jnp.arange(8.0, dtype=jnp.float64)
        result: Float[Array, "devices"] = shard_array(arr, shard_axes=0)
        assert result.dtype == jnp.float64
        chex.assert_trees_all_close(result, arr)

    def test_complex_dtype(self) -> None:
        """Complex arrays should shard correctly."""
        arr: Complex[Array, "devices"] = jnp.arange(8.0) + 1j * jnp.arange(
            8.0, 16.0
        )
        result: Complex[Array, "devices"] = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_integer_array(self) -> None:
        """Integer arrays should shard correctly."""
        arr: Integer[Array, "devices"] = jnp.arange(8, dtype=jnp.int32)
        result: Integer[Array, "devices"] = shard_array(arr, shard_axes=0)
        chex.assert_trees_all_close(result, arr)

    def test_sharding_is_named_sharding(self) -> None:
        """Result should use NamedSharding."""
        arr: Float[Array, "devices cols"] = jnp.ones((8, 4))
        result: Float[Array, "devices cols"] = shard_array(arr, shard_axes=0)
        assert isinstance(result.sharding, NamedSharding)


class TestShardArrayMultiDevice(chex.TestCase):
    """Tests that verify real multi-device distribution."""

    def test_eight_devices_available(self) -> None:
        """Conftest must expose 8 virtual CPU devices."""
        assert len(jax.devices()) == 8

    def test_sharded_across_all_devices(self) -> None:
        """Array sharded on axis 0 should span all 8 devices."""
        arr: Float[Array, "devices coords"] = jnp.arange(24.0).reshape(8, 3)
        result: Float[Array, "devices coords"] = shard_array(arr, shard_axes=0)
        device_set: Any = result.sharding.device_set
        assert len(device_set) == 8

    def test_subset_devices(self) -> None:
        """Sharding across a subset of devices should work."""
        devices: Any = jax.devices()[:4]
        arr: Float[Array, "devices coords"] = jnp.arange(12.0).reshape(4, 3)
        result: Float[Array, "devices coords"] = shard_array(
            arr, shard_axes=0, devices=devices
        )
        assert len(result.sharding.device_set) == 4
        chex.assert_trees_all_close(result, arr)

    def test_shard_values_roundtrip(self) -> None:
        """Data gathered from sharded array must match original."""
        arr: Float[Array, "devices coords"] = jnp.arange(16.0).reshape(8, 2)
        result: Float[Array, "devices coords"] = shard_array(arr, shard_axes=0)
        gathered: Float[Array, "devices coords"] = jax.device_get(result)
        chex.assert_trees_all_close(gathered, arr)


class TestPmapCompatibility(chex.TestCase):
    """Test pmap execution across 8 virtual devices."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_on_sharded_array(self) -> None:
        """JIT-compiled function should work on sharded data."""

        @self.variant
        def add_one(x: Float[Array, "8"]) -> Float[Array, "8"]:
            return x + 1.0

        arr: Float[Array, "8"] = jnp.arange(8.0)
        sharded: Float[Array, "8"] = shard_array(arr, shard_axes=0)
        result: Float[Array, "8"] = add_one(sharded)
        expected: Float[Array, "8"] = jnp.arange(1.0, 9.0)
        chex.assert_trees_all_close(result, expected)

    def test_pmap_elementwise(self) -> None:
        """Pmap should distribute elementwise ops across devices."""
        fn: Callable[
            [Float[Array, "devices cols"]], Float[Array, "devices cols"]
        ] = jax.pmap(lambda x: x**2)
        arr: Float[Array, "devices cols"] = jnp.arange(8.0).reshape(8, 1)
        result: Float[Array, "devices cols"] = fn(arr)
        expected: Float[Array, "devices cols"] = arr**2
        chex.assert_trees_all_close(result, expected)

    def test_pmap_reduction(self) -> None:
        """Pmap with inner sum should reduce per-device slices."""
        fn: Callable[
            [Float[Array, "devices cols"]], Float[Array, "devices"]
        ] = jax.pmap(lambda x: jnp.sum(x))
        arr: Float[Array, "devices cols"] = jnp.ones((8, 4))
        result: Float[Array, "devices"] = fn(arr)
        expected: Float[Array, "devices"] = jnp.full((8,), 4.0)
        chex.assert_trees_all_close(result, expected)

    def test_pmap_preserves_dtype(self) -> None:
        """Pmap should preserve float64 dtype."""
        fn: Callable[
            [Float[Array, "devices cols"]], Float[Array, "devices cols"]
        ] = jax.pmap(lambda x: x * 2.0)
        arr: Float[Array, "devices cols"] = jnp.ones((8, 2), dtype=jnp.float64)
        result: Float[Array, "devices cols"] = fn(arr)
        assert result.dtype == jnp.float64
        chex.assert_trees_all_close(result, arr * 2.0)

    def test_pmap_trig_identity(self) -> None:
        """Pmap should handle sin^2 + cos^2 = 1 across devices."""
        fn: Callable[
            [Float[Array, "devices cols"]], Float[Array, "devices cols"]
        ] = jax.pmap(lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2)
        arr: Float[Array, "devices cols"] = jnp.linspace(
            0, 6.28, 8 * 4
        ).reshape(8, 4)
        result: Float[Array, "devices cols"] = fn(arr)
        expected: Float[Array, "devices cols"] = jnp.ones_like(arr)
        chex.assert_trees_all_close(result, expected, atol=1e-6)

    def test_pmap_with_broadcast(self) -> None:
        """Pmap should handle scalar broadcast."""
        fn: Callable[
            [Float[Array, "devices cols"]], Float[Array, "devices cols"]
        ] = jax.pmap(lambda x: x + 10.0)
        arr: Float[Array, "devices cols"] = jnp.arange(8.0).reshape(8, 1)
        result: Float[Array, "devices cols"] = fn(arr)
        expected: Float[Array, "devices cols"] = arr + 10.0
        chex.assert_trees_all_close(result, expected)
