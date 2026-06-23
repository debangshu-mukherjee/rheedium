# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -trusted
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Atomic Visualizer
#
# Rheedium includes an ASE-backed viewer for quickly inspecting the crystal
# structures that later feed the RHEED simulator. This tutorial shows the two
# main workflows the viewer was designed for:
#
# - tiling a small crystallographic cell so the layered structure is easy to
#   recognize;
# - inspecting a large molecular-dynamics slab while highlighting the surface
#   atoms that the RHEED calculation will treat as exposed.
#
# The public entry point is `rh.plots.view_atoms_interactive`. In a notebook
# frontend, `backend="auto"` prefers `nglview` when it is installed and falls
# back to ASE's x3d HTML viewer otherwise.

# %% [markdown]
# ## Setup
#
# The tutorial data live in `tests/test_data/bi2se3`, so the helper below finds
# the repository root whether this notebook is launched from VS Code, Jupyter,
# or a terminal.

# %%
from pathlib import Path
import importlib.util
import os

from IPython.display import display
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import rheedium as rh
from rheedium.types import SurfaceConfig


def find_repo_root() -> Path:
    """Find the repository root when launched from a notebook or shell."""
    candidates: list[Path] = [Path.cwd(), *Path.cwd().parents]
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])

    for candidate in candidates:
        if (candidate / "tests" / "test_data" / "bi2se3").exists():
            return candidate
    raise RuntimeError("Could not find tests/test_data/bi2se3.")


repo_root = find_repo_root()
bi2se3_data = repo_root / "tests" / "test_data" / "bi2se3"
print(f"Repository root: {repo_root}")
print(f"Bi2Se3 data:     {bi2se3_data}")

# %% [markdown]
# ## Load the Example Structures
#
# `Bi2Se3.cif` is a compact crystallographic description. The `500K.final.xyz`
# file is a much larger MD slab with periodic cell metadata. Both parse into the
# same `CrystalStructure` type, so the visualizer does not need separate code
# paths for CIF, POSCAR, or extended XYZ input.

# %%
bi2se3_cell = rh.inout.parse_cif(bi2se3_data / "Bi2Se3.cif")
slab_500k = rh.inout.parse_crystal(bi2se3_data / "500K.final.xyz")


def summarize_crystal(label: str, crystal: rh.types.CrystalStructure) -> None:
    """Print a compact one-line summary of a loaded structure."""
    n_atoms = int(crystal.cart_positions.shape[0])
    a, b, c = [float(value) for value in crystal.cell_lengths]
    alpha, beta, gamma = [float(value) for value in crystal.cell_angles]
    print(
        f"{label:<16} atoms={n_atoms:>5d}  "
        f"cell=({a:6.2f}, {b:6.2f}, {c:6.2f}) Angstrom  "
        f"angles=({alpha:5.1f}, {beta:5.1f}, {gamma:5.1f}) deg"
    )


summarize_crystal("Bi2Se3 CIF", bi2se3_cell)
summarize_crystal("500 K slab", slab_500k)

# %% [markdown]
# ## Backend Selection
#
# The interactive function exposes a single `backend` argument:
#
# - `"auto"`: prefer `nglview` if it is importable, otherwise use x3d;
# - `"ngl"`: require `nglview`;
# - `"x3d"`: force ASE's inline HTML viewer.
#
# Use `"auto"` for normal notebook work. If you want a portable HTML fallback
# for a small structure, set `VISUALIZER_BACKEND = "x3d"`.

# %%
VISUALIZER_BACKEND = os.environ.get("RHEEDIUM_VISUALIZER_BACKEND", "auto")
has_nglview = importlib.util.find_spec("nglview") is not None

print(f"nglview available: {has_nglview}")
print(f"requested backend: {VISUALIZER_BACKEND!r}")

# %% [markdown]
# ## Inspect the Data Path
#
# Internally, `view_atoms_interactive` converts `CrystalStructure` to
# `ase.Atoms`, optionally repeats the cell, then hands the result to ASE's
# notebook viewer. You can use the same conversion directly when debugging file
# loading or cell geometry.

# %%
ase_cell = rh.inout.to_ase(bi2se3_cell)
tiled_cell = ase_cell.repeat((2, 2, 1))

print(ase_cell)
print(tiled_cell)
print(f"base atoms:  {len(ase_cell)}")
print(f"tiled atoms: {len(tiled_cell)}")

# %% [markdown]
# ## Static Preview of the Conventional Cell
#
# A quick matplotlib preview is useful in static documentation and test logs.
# The interactive widget in the next cell uses the same loaded structure, but
# it adds rotation, zoom, and optional surface or beam annotations.

# %%
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
rh.plots.view_atoms(
    bi2se3_cell,
    elev=12.0,
    azim=35.0,
    atom_scale=1.4,
    ax=ax,
)
ax.set_title("Bi2Se3 conventional cell")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interactive Cell Viewer
#
# The CIF cell is small, so we tile it as a `2 x 2 x 1` supercell. This makes
# the layered Bi2Se3 motif easier to see without changing the original
# `CrystalStructure`.

# %%
cell_view = rh.plots.view_atoms_interactive(
    bi2se3_cell,
    supercell=(2, 2, 1),
    show_cell=True,
    backend=VISUALIZER_BACKEND,
)
display(cell_view)


# %% [markdown]
# The returned object is the notebook widget or HTML object, with Rheedium
# metadata attached for inspection. That metadata is handy when you want to
# confirm which backend was selected or how many atoms were passed to ASE.


# %%
def describe_viewer(label: str, viewer) -> None:
    """Print the Rheedium metadata attached to an interactive viewer."""
    atoms = viewer.rheedium_atoms
    surface_mask = viewer.rheedium_surface_mask
    backend = viewer.rheedium_backend
    beam_direction = viewer.rheedium_beam_direction

    print(f"{label}:")
    print(f"  backend:        {backend}")
    print(f"  atoms rendered: {len(atoms)}")
    print(f"  cell repeats:   {atoms.info.get('rheedium_supercell')}")
    print(f"  beam direction: {beam_direction}")
    if surface_mask is None:
        print("  surface atoms:  not requested")
    else:
        print(f"  surface atoms:  {int(np.sum(surface_mask))}")


describe_viewer("Tiled Bi2Se3 cell", cell_view)

# %% [markdown]
# ## Compute the Same Surface Mask Used by the Viewer
#
# Surface highlighting calls `rh.types.identify_surface_atoms`, the same
# configurable helper used by the RHEED simulation layer. Here we run it
# explicitly so the surface-selection rule is visible before rendering the slab.

# %%
surface_config = SurfaceConfig(method="height", height_fraction=0.3)
slab_atoms = rh.inout.to_ase(slab_500k)
slab_positions = np.asarray(slab_atoms.get_positions(), dtype=float)
surface_mask = np.asarray(
    rh.types.identify_surface_atoms(
        jnp.asarray(slab_positions),
        surface_config,
    ),
    dtype=bool,
)

z_positions = slab_positions[:, 2]
surface_z = z_positions[surface_mask]
bulk_z = z_positions[~surface_mask]

print(f"surface method:      {surface_config.method}")
print(f"height fraction:     {float(surface_config.height_fraction):.2f}")
print(f"surface atom count:  {int(surface_mask.sum())} / {len(surface_mask)}")
print(
    f"surface z range:     {surface_z.min():.2f} to {surface_z.max():.2f} Angstrom"
)
print(
    f"non-surface z range: {bulk_z.min():.2f} to {bulk_z.max():.2f} Angstrom"
)

# %% [markdown]
# ## Interactive MD Slab Viewer
#
# For large slabs, leave `supercell=(1, 1, 1)` and use the surface mask instead
# of tiling. Surface atoms are emphasized in orange. The optional
# `beam_direction` marker gives a visual reminder of the grazing-incidence
# RHEED geometry; here it points mostly along `+x` with a small downward
# component.

# %%
slab_view = rh.plots.view_atoms_interactive(
    slab_500k,
    highlight_surface=True,
    surface_config=surface_config,
    beam_direction=(1.0, 0.0, -0.05),
    backend=VISUALIZER_BACKEND,
)
display(slab_view)

# %%
describe_viewer("500 K MD slab", slab_view)

# %% [markdown]
# ## Force the Portable x3d Fallback
#
# This smaller example forces `backend="x3d"`. It is useful when you want an
# inline HTML representation without relying on `nglview` widgets. For very
# large slabs, `nglview` is usually smoother.

# %%
x3d_cell_view = rh.plots.view_atoms_interactive(
    bi2se3_cell,
    supercell=(1, 1, 1),
    show_cell=True,
    backend="x3d",
)
display(x3d_cell_view)
describe_viewer("x3d Bi2Se3 cell", x3d_cell_view)

# %% [markdown]
# ## Practical Notes
#
# - Use `supercell=(2, 2, 1)` or similar for small cells where the chemistry is
#   hard to see in one unit cell.
# - Keep MD slabs at the default supercell. They are already large, and tiling
#   them can produce unwieldy widgets.
# - Use `highlight_surface=True` with the same `SurfaceConfig` you plan to pass
#   to the simulator. This keeps visual inspection and RHEED simulation aligned.
# - Set `backend="x3d"` when you need a no-widget fallback, or `backend="ngl"`
#   when you want to fail fast if `nglview` is not installed.
