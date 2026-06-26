"""Test suite for tools/parallel.py sharding utilities.

This module tests the shard_array and distribute_batched utilities for
distributing JAX arrays and batched computations across devices.  With
``XLA_FLAGS=--xla_force_host_platform_device_count=8`` set in conftest,
``jax.devices()`` returns 8 virtual CPU devices, enabling meaningful
multi-device and ``pmap`` tests via ``chex.variants``.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.sharding import NamedSharding
from jaxtyping import Array, Complex, Float, Integer

from rheedium.simul.sweeps import simulate_detector_image_sweep
from rheedium.tools.parallel import distribute_batched, shard_array
from rheedium.types import (
    BeamSpec,
    DetectorGeometry,
    RenderParams,
    SurfaceCTRParams,
)

from ..._factories import make_si_crystal_2atom


class TestShardArray(chex.TestCase):
    """Core shard_array tests with 8-device-divisible shapes.

    :see: :func:`~rheedium.tools.shard_array`
    """

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


def _batched_double(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
    """Double every row, written as a vmap to mimic a batched simulator."""
    return jax.vmap(lambda row: row * 2.0)(x)


def _batched_outer(x: Float[Array, "N"]) -> Float[Array, "N 2 3"]:
    """Map each scalar to a (2, 3) block, exercising trailing batch dims."""
    return jax.vmap(lambda value: jnp.full((2, 3), value))(x)


class TestDistributeBatched(chex.TestCase):
    """Tests for data-parallel execution of batched callables.

    :see: :func:`~rheedium.tools.distribute_batched`
    """

    def test_matches_serial_divisible(self) -> None:
        """Divisible batch must match the serial vmap result exactly."""
        arr: Float[Array, "N D"] = jnp.arange(32.0).reshape(8, 4)
        result: Float[Array, "N D"] = distribute_batched(_batched_double, arr)
        chex.assert_shape(result, (8, 4))
        chex.assert_trees_all_close(result, _batched_double(arr))

    @parameterized.parameters(1, 3, 5, 7, 13)
    def test_non_divisible_batch(self, n_rows: int) -> None:
        """Non-divisible batch returns the unpadded, correct result."""
        arr: Float[Array, "N D"] = jnp.arange(float(n_rows * 4)).reshape(
            n_rows, 4
        )
        result: Float[Array, "N D"] = distribute_batched(_batched_double, arr)
        chex.assert_shape(result, (n_rows, 4))
        chex.assert_trees_all_close(result, _batched_double(arr))

    def test_padding_value_does_not_leak(self) -> None:
        """A non-zero pad value must not affect the returned rows."""
        arr: Float[Array, "N D"] = jnp.arange(20.0).reshape(5, 4)
        result: Float[Array, "N D"] = distribute_batched(
            _batched_double, arr, pad_value=999.0
        )
        chex.assert_trees_all_close(result, _batched_double(arr))

    def test_trailing_batch_dims(self) -> None:
        """Outputs with extra trailing axes keep their full shape."""
        arr: Float[Array, "N"] = jnp.arange(6.0)
        result: Float[Array, "N 2 3"] = distribute_batched(_batched_outer, arr)
        chex.assert_shape(result, (6, 2, 3))
        chex.assert_trees_all_close(result, _batched_outer(arr))

    def test_divisible_output_sharded_across_devices(self) -> None:
        """A device-multiple batch stays sharded across the whole mesh."""
        arr: Float[Array, "N D"] = jnp.arange(32.0).reshape(8, 4)
        result: Float[Array, "N D"] = distribute_batched(_batched_double, arr)
        assert isinstance(result.sharding, NamedSharding)
        assert len(result.sharding.device_set) == 8

    def test_subset_devices(self) -> None:
        """Distributing across an explicit device subset must work."""
        devices: Any = jax.devices()[:4]
        arr: Float[Array, "N D"] = jnp.arange(16.0).reshape(4, 4)
        result: Float[Array, "N D"] = distribute_batched(
            _batched_double, arr, devices=devices
        )
        chex.assert_trees_all_close(result, _batched_double(arr))
        assert len(result.sharding.device_set) == 4

    def test_single_device(self) -> None:
        """Distributing across a single device degrades to replication."""
        devices: Any = jax.devices()[:1]
        arr: Float[Array, "N D"] = jnp.arange(12.0).reshape(3, 4)
        result: Float[Array, "N D"] = distribute_batched(
            _batched_double, arr, devices=devices
        )
        chex.assert_trees_all_close(result, _batched_double(arr))

    def test_float64_preserved(self) -> None:
        """Float64 dtype must survive the distribute round-trip."""
        arr: Float[Array, "N D"] = jnp.arange(32.0, dtype=jnp.float64).reshape(
            8, 4
        )
        result: Float[Array, "N D"] = distribute_batched(_batched_double, arr)
        assert result.dtype == jnp.float64

    def test_real_phi_sweep_matches_serial(self) -> None:
        """Distributing a real phi sweep matches the direct sweep result."""
        crystal: Any = make_si_crystal_2atom()
        phi_bank: Float[Array, "N"] = jnp.linspace(0.0, 30.0, 6)
        sweep_kwargs: dict[str, Any] = {
            "beam": BeamSpec(
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            ),
            "surface": SurfaceCTRParams(hmax=0, kmax=0),
            "detector": DetectorGeometry(
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                psf_sigma_pixels=0.0,
            ),
            "render": RenderParams(
                n_angular_samples=1,
                n_energy_samples=1,
                render_ctrs_as_streaks=False,
            ),
        }

        def _run_phi_sweep(bank: Float[Array, "N"]) -> Float[Array, "N H W"]:
            """Run one phi-angle detector sweep for a bank of samples."""
            images: Float[Array, "N H W"] = simulate_detector_image_sweep(
                crystal=crystal,
                axis=("phi_deg", bank),
                **sweep_kwargs,
            )
            return images

        distributed: Float[Array, "N H W"] = distribute_batched(
            _run_phi_sweep, phi_bank
        )
        serial: Float[Array, "N H W"] = _run_phi_sweep(phi_bank)
        chex.assert_shape(distributed, serial.shape)
        chex.assert_trees_all_close(distributed, serial, atol=1e-6)
