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

Notes
-----
This module is designed for distributed computing scenarios
where large arrays need to be processed across multiple
devices. The sharding utilities work with JAX's device mesh
system and can be used with various JAX transformations
including ``jit``, ``grad``, and ``vmap``.
"""

from collections.abc import Sequence

import jax
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Num


def shard_array(
    input_array: Num[Array, " ..."],
    shard_axes: int | Sequence[int],
    devices: Sequence[jax.Device] | None = None,
) -> Num[Array, " ..."]:
    """Shard an array across specified axes and devices.

    Extended Summary
    ----------------
    Distributes an array across multiple devices for parallel
    processing by creating a device mesh and applying
    appropriate partitioning based on the specified axes.

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


__all__: list[str] = [
    "shard_array",
]
