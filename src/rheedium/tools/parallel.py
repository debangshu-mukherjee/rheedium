"""Parallel processing utilities for distributed RHEED simulations.

Extended Summary
----------------
Provides utilities for sharding arrays across multiple devices
for parallel processing and distributed computing in RHEED
simulation workflows. All functions are JAX-compatible and
support automatic differentiation.

Routine Listings
----------------
:func:`shard_array`
    Shard an array across specified axes and devices for
    parallel processing.
:func:`distribute_batched`
    Run an already-batched callable data-parallel across a
    one-dimensional device mesh.

Notes
-----
This module is designed for distributed computing scenarios
where large arrays need to be processed across multiple
devices. The sharding utilities work with JAX's device mesh
system and can be used with various JAX transformations
including ``jit``, ``grad``, and ``vmap``.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Num, jaxtyped


def shard_array(
    input_array: Num[Array, " ..."],
    shard_axes: int | list[int] | tuple[int, ...],
    devices: list[jax.Device] | tuple[jax.Device, ...] | None = None,
) -> Num[Array, " ..."]:
    """Shard an array across specified axes and devices.

    Extended Summary
    ----------------
    Distributes an array across multiple devices for parallel
    processing by creating a device mesh and applying
    appropriate partitioning based on the specified axes.

    :see: :class:`~.test_parallel.TestShardArray`

    Parameters
    ----------
    input_array : Array
        The input array to be sharded.
    shard_axes : int | Sequence[int]
        The axis or axes to shard along. Use ``-1`` (or a
        sequence containing ``-1``) to skip sharding along
        that axis.
    devices : Sequence[jax.Device], optional
        The devices to shard across. If ``None``, all
        available devices are used.

    Returns
    -------
    sharded_array : Array
        The array distributed across the specified devices.
    """
    if devices is None:
        devices = jax.devices()
    if isinstance(shard_axes, int):
        shard_axes = [shard_axes]
    num_devices: int = len(devices)
    mesh: jax.sharding.Mesh = jax.make_mesh(
        (num_devices,),
        ("devices",),
    )
    pspec_list: list[str | None] = [None] * input_array.ndim
    for ax in shard_axes:
        if ax != -1 and ax < input_array.ndim:
            pspec_list[ax] = "devices"
    pspec: PartitionSpec = PartitionSpec(*pspec_list)
    sharding: NamedSharding = NamedSharding(mesh, pspec)
    with mesh:
        return jax.device_put(input_array, sharding)


@jaxtyped(typechecker=beartype)
def distribute_batched(
    batched_fn: Callable[[Num[Array, " ..."]], Num[Array, " ..."]],
    batch_array: Num[Array, "N ..."],
    devices: list[jax.Device] | tuple[jax.Device, ...] | None = None,
    pad_value: float = 0.0,
) -> Num[Array, "N ..."]:
    """Run a batched callable data-parallel across a 1-D device mesh.

    Extended Summary
    ----------------
    Distributes an embarrassingly parallel batch -- such as a
    ``jax.vmap``-wrapped parameter sweep -- across every available
    device. ``batched_fn`` is compiled with ``jax.jit`` and explicit
    ``in_shardings``/``out_shardings`` that shard the leading (batch)
    axis across a one-dimensional device mesh, so XLA's SPMD
    partitioner splits the work across devices and auto-partitions all
    intermediates. Any arrays the callable closes over (for example a
    :class:`~rheedium.types.CrystalStructure`) are replicated, not
    sharded.

    Explicit boundary shardings are used rather than committing the
    input with :func:`shard_array`: pinning the input and output specs
    lets the partitioner propagate sharding cleanly through complex
    simulators whose intermediates gain extra axes (e.g. beam-averaging
    quadrature stacks), which bare input commitment does not.

    The batch length need not be a multiple of the device count: the
    leading axis is padded up to the next multiple before sharding and
    the padding is trimmed from the result, so coverage is never
    silently dropped.

    :see: :class:`~.test_parallel.TestDistributeBatched`

    Parameters
    ----------
    batched_fn : Callable
        A function mapping one batched array (leading axis = batch) to
        one batched array. Typically a ``jax.vmap`` of a per-sample
        simulator, or a one-argument closure over a ``*_sweep`` helper.
        Values it closes over are replicated across devices.
    batch_array : Num[Array, "N ..."]
        The batched argument whose leading axis is distributed. ``N``
        may be any positive length.
    devices : Sequence[jax.Device], optional
        The devices to distribute across. If ``None``, all available
        devices from :func:`jax.devices` are used.
    pad_value : float, optional
        Fill value used to pad the leading axis up to a device
        multiple. Padded rows are discarded before returning, so the
        value only affects discarded computation. Default: ``0.0``.

    Returns
    -------
    result : Num[Array, "N ..."]
        The stacked per-sample outputs, with the same leading length
        ``N`` as ``batch_array`` and the trailing shape produced by
        ``batched_fn``. When ``N`` is a multiple of the device count
        the result keeps its sharding across the mesh; otherwise the
        padded result is gathered and returned as the unpadded slice
        (slicing a sharded axis to a non-multiple length is not
        supported on device).

    Notes
    -----
    1. Resolve the device list and compute the padding needed to make
       the leading axis divisible by the device count.
    2. Pad the leading axis with ``pad_value`` when required.
    3. Build a 1-D device mesh and a ``NamedSharding`` that shards the
       leading axis and replicates the rest.
    4. Compile ``batched_fn`` with ``jax.jit`` using that sharding as
       both ``in_shardings`` and ``out_shardings`` and run it inside the
       mesh context; XLA shards the input on entry and distributes the
       computation.
    5. When padding was added, gather the result across the sharded
       axis and trim the padded rows; otherwise return the sharded
       result unchanged.

    See Also
    --------
    shard_array : Shard an array across devices along chosen axes.
    """
    if devices is None:
        devices = jax.devices()
    num_devices: int = len(devices)
    original_length: int = batch_array.shape[0]
    pad_count: int = (-original_length) % num_devices
    if pad_count > 0:
        pad_width: list[tuple[int, int]] = [(0, pad_count)] + [(0, 0)] * (
            batch_array.ndim - 1
        )
        padded_array: Num[Array, " ..."] = jnp.pad(
            batch_array,
            pad_width,
            constant_values=pad_value,
        )
    else:
        padded_array = batch_array
    mesh: jax.sharding.Mesh = jax.make_mesh(
        (num_devices,),
        ("devices",),
        devices=tuple(devices),
    )
    batch_sharding: NamedSharding = NamedSharding(
        mesh,
        PartitionSpec("devices"),
    )
    compiled_fn: Callable[[Num[Array, " ..."]], Num[Array, " ..."]] = jax.jit(
        batched_fn,
        in_shardings=batch_sharding,
        out_shardings=batch_sharding,
    )
    with mesh:
        padded_result: Num[Array, " ..."] = compiled_fn(padded_array)
    if pad_count > 0:
        gathered_result: Num[Array, " ..."] = jnp.asarray(
            jax.device_get(padded_result)
        )
        return gathered_result[:original_length]
    return padded_result


__all__: list[str] = [
    "distribute_batched",
    "shard_array",
]
