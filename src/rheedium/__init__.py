"""
=========================================================

RHEEDIUM Package (:mod:`rheedium`)

=========================================================

This is the root of the rheedium package, containing submodules for:
- Data I/O (`inout`)
- Plotting (`plots`)
- Reconstructions (`recon`)
- Simulations (`simul`)
- Unit cell computations (`ucell`)
- Custom types (`types`)

Each submodule can be directly accessed after importing rheedium.
"""

from . import inout, plots, recon, simul, types, ucell
