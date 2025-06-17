"""
Package: rheedium
-----------------
JAX-based RHEED (Reflection High-Energy Electron Diffraction) simulation and analysis.

Submodules
----------
- `inout`:
    Data input/output operations for crystal structures and RHEED images
- `plots`:
    Visualization tools for RHEED patterns and crystal structures
- `recon`:
    Surface reconstruction analysis and modeling utilities
- `simul`:
    RHEED pattern simulation using kinematic diffraction theory
- `types`:
    Custom type definitions and data structures for JAX compatibility
- `ucell`:
    Unit cell and crystallographic computation utilities

Usage
-----
Import the package and access submodules:
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> pattern = rh.simul.simulate_rheed_pattern(crystal)
    >>> rh.plots.plot_rheed(pattern)
"""

from . import inout, plots, recon, simul, types, ucell

__all__ = [
    "inout",
    "plots",
    "recon",
    "simul",
    "types",
    "ucell",
]
