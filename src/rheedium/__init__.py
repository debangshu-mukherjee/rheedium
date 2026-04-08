"""JAX-based RHEED simulation and analysis package.

Extended Summary
----------------
Rheedium provides a comprehensive suite of tools for simulating and analyzing
Reflection High-Energy Electron Diffraction (RHEED) patterns. Built on JAX,
it offers differentiable simulations suitable for optimization and machine
learning applications in materials science and surface physics. Distributed
computing is supported through device mesh parallelism.

Routine Listings
----------------
:mod:`audit`
    Benchmarking and realism-audit utilities for detector images.
:mod:`inout`
    Data input/output operations for crystal structures and RHEED images.
:mod:`plots`
    Visualization tools for RHEED patterns and crystal structures.
:mod:`procs`
    Differentiable procedural models and preprocessing utilities.
:mod:`recon`
    Inverse-problem utilities for inferring structure from RHEED data.
:mod:`simul`
    RHEED pattern simulation using kinematic diffraction theory.
:mod:`tools`
    Utility tools for parallel processing and distributed computing.
:mod:`types`
    Custom type definitions and data structures for JAX compatibility.
:mod:`ucell`
    Unit cell and crystallographic computation utilities.

Examples
--------
>>> import rheedium as rh
>>> crystal = rh.inout.parse_cif("structure.cif")
>>> pattern = rh.simul.simulate_rheed_pattern(crystal)
>>> rh.plots.plot_rheed(pattern)

Notes
-----
All computations are JAX-compatible and support automatic differentiation
for gradient-based optimization of crystal structures and simulation
parameters.

Multi-node distributed execution is supported via
``jax.distributed.initialize()``. To enable, set the environment variable
``RHEEDIUM_DISTRIBUTED=1`` before launching with ``srun`` or equivalent.
An optional ``RHEEDIUM_COORDINATOR_ADDRESS`` environment variable overrides
automatic SLURM coordinator detection.
"""

import os
from importlib.metadata import version

# Enable multi-threaded CPU execution for JAX
# must be set before importing JAX
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

# Enable 64-bit precision in JAX must be set before importing submodules
import jax  # noqa: E402

if (
    os.environ.get("RHEEDIUM_DISTRIBUTED", "0") == "1"
    and int(os.environ.get("SLURM_NTASKS", "1")) > 1
):
    coordinator_address: str | None = os.environ.get(
        "RHEEDIUM_COORDINATOR_ADDRESS"
    )
    if coordinator_address is not None:
        jax.distributed.initialize(coordinator_address=coordinator_address)
    else:
        jax.distributed.initialize()

jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402, I001
    audit,
    inout,
    plots,
    procs,
    recon,
    simul,
    tools,
    types,
    ucell,
)

__version__: str = version("rheedium")

__all__: list[str] = [
    "audit",
    "inout",
    "plots",
    "procs",
    "recon",
    "simul",
    "tools",
    "types",
    "ucell",
]
