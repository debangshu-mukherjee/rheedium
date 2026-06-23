# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Atomic Visualizer
#
# This notebook demonstrates the ASE-backed atomic viewer added for quick
# inspection of small cells and large MD slabs. In a notebook front end,
# `backend="auto"` prefers nglview when installed and otherwise falls back to
# ASE's x3d HTML viewer.

# %%
from pathlib import Path

import rheedium as rh
from rheedium.types import SurfaceConfig


def find_repo_root() -> Path:
    """Find the repository root when launched from VS Code or a shell."""
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents]
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    for candidate in candidates:
        if (candidate / "tests" / "test_data" / "bi2se3").exists():
            return candidate
    raise RuntimeError("Could not find tests/test_data/bi2se3.")


repo_root = find_repo_root()
bi2se3_data = repo_root / "tests" / "test_data" / "bi2se3"

# %% [markdown]
# ## Hexagonal Bi2Se3 Cell
#
# Tile the conventional CIF cell so the layered structure is easier to inspect.

# %%
bi2se3_cell = rh.inout.parse_cif(bi2se3_data / "Bi2Se3.cif")
rh.plots.view_atoms_interactive(
    bi2se3_cell,
    supercell=(2, 2, 1),
    backend="auto",
)

# %% [markdown]
# ## MD Slab Surface
#
# The surface-highlight mask uses the same `identify_surface_atoms` machinery
# as the RHEED simulator, so the displayed surface matches the simulated one.

# %%
slab = rh.inout.parse_crystal(bi2se3_data / "500K.final.xyz")
rh.plots.view_atoms_interactive(
    slab,
    highlight_surface=True,
    surface_config=SurfaceConfig(method="height", height_fraction=0.3),
    beam_direction=(1.0, 0.0, -0.05),
    backend="auto",
)
