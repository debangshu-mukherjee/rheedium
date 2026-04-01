"""Utility tools for RHEED simulations.

Extended Summary
----------------
This package contains utilities for parallel processing and
distributed computing in RHEED simulation workflows. All
functions are JAX-compatible and support automatic
differentiation.

Routine Listings
----------------
:func:`shard_array`
    Shard an array across specified axes and devices for
    parallel processing.

Notes
-----
For multi-node distributed execution, set the environment
variable ``RHEEDIUM_DISTRIBUTED=1`` before launching with
``srun`` or equivalent. An optional
``RHEEDIUM_COORDINATOR_ADDRESS`` environment variable
overrides automatic SLURM coordinator detection.
"""

from .parallel import shard_array

__all__: list[str] = [
    "shard_array",
]
