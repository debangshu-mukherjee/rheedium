"""
=========================================================

RHEEDIUM Package (:mod:`rheedium`)

=========================================================

This is the root of the rheedium package, containing submodules for:
- Data I/O (`io`)
- Simulations (`sim`)
- Unit cell computations (`uc`)
- Custom types (`types`)

Each submodule can be directly accessed after importing rheedium.
"""

from . import io, recon, sim, types, uc
