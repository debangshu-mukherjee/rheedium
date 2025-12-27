"""JAX-based RHEED simulation and analysis package.

Extended Summary
----------------
Rheedium provides a comprehensive suite of tools for simulating and analyzing
Reflection High-Energy Electron Diffraction (RHEED) patterns. Built on JAX,
it offers differentiable simulations suitable for optimization and machine
learning applications in materials science and surface physics.

Routine Listings
----------------
inout : module
    Data input/output operations for crystal structures and RHEED images
plots : module
    Visualization tools for RHEED patterns and crystal structures
recon : module
    Surface reconstruction analysis and modeling utilities
simul : module
    RHEED pattern simulation using kinematic diffraction theory
types : module
    Custom type definitions and data structures for JAX compatibility
ucell : module
    Unit cell and crystallographic computation utilities

Examples
--------
>>> import rheedium as rh
>>> crystal = rh.inout.parse_cif("structure.cif")
>>> pattern = rh.simul.simulate_rheed_pattern(crystal)
>>> rh.plots.plot_rheed(pattern)

Notes
-----
All computations are JAX-compatible and support automatic differentiation
for gradient-based optimization of crystal structures and simulation parameters.
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

jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402, I001
    inout,
    plots,
    recon,
    simul,
    types,
    ucell,
)

__version__: str = version("rheedium")

__all__ = [
    "inout",
    "plots",
    "recon",
    "simul",
    "types",
    "ucell",
]
