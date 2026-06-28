# Rheedium [(Documentation)](https://rheedium.readthedocs.io/en/latest/)

[![PyPI Downloads](https://static.pepy.tech/badge/rheedium)](https://pepy.tech/projects/rheedium)
[![License](https://img.shields.io/pypi/l/rheedium.svg)](https://github.com/debangshu-mukherjee/rheedium/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/rheedium.svg)](https://pypi.python.org/pypi/rheedium)
[![Python Versions](https://img.shields.io/pypi/pyversions/rheedium.svg)](https://pypi.python.org/pypi/rheedium)
[![Tests](https://github.com/debangshu-mukherjee/rheedium/actions/workflows/test.yml/badge.svg)](https://github.com/debangshu-mukherjee/rheedium/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/rheedium/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/rheedium)
[![Documentation Status](https://readthedocs.org/projects/rheedium/badge/?version=latest)](https://rheedium.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14757399.svg)](https://doi.org/10.5281/zenodo.14757399)
[![Ruff](https://img.shields.io/badge/lint%20and%20format-ruff-D7FF64?logo=ruff&logoColor=1D1D1D)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![jax_badge](https://tinyurl.com/mucknrvu)](https://docs.jax.dev/)
[![Lines of Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/.github/badges/loc.json)](https://github.com/debangshu-mukherjee/rheedium)

<p align="center">
  <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/_static/readme/rheed-sweep-gallery.png" alt="Simulated RHEED detector images for SrTiO3, MgO, and Bi2Se3" width="900">
</p>

## Overview

Rheedium is a JAX-based computational framework for simulating **RHEED**
(Reflection High-Energy Electron Diffraction) patterns, with full automatic
differentiation and GPU acceleration. Because every simulation step is a
differentiable JAX function, you can run it forward (crystal → pattern) *or*
backward (experimental pattern → crystal/instrument parameters) by gradient
descent — the same code powers both simulation and reconstruction.

<p align="center">
  <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/grazing_incidence_geometry.png" alt="Grazing-incidence RHEED geometry: electron beam striking a crystal surface at a shallow angle" width="600">
  <br>
  <em>RHEED geometry: a high-energy beam grazes the surface and diffracts onto the detector.</em>
</p>

> **New here? Read this README top to bottom first.** The
> [Mental model](#mental-model) and [Repository map](#repository-map) sections
> are enough to know *where* any piece of functionality lives and *why* it is
> shaped the way it is. Then jump to
> [Tests, tutorials, and docs](#tests-tutorials-and-docs) for a worked example.

## Install

```bash
pip install rheedium            # from PyPI
```

```bash
git clone git@github.com:debangshu-mukherjee/rheedium.git
cd rheedium
uv sync --extra dev             # full dev environment (recommended for hacking)
```

GPU users: install the `cuda` extra (`pip install "rheedium[cuda]"`) on Linux
x86-64. Everything below works identically on CPU.

## Quick start

```python
import rheedium as rh

# 1. Load a crystal from CIF / XYZ / POSCAR (format auto-detected)
crystal = rh.inout.parse_crystal("tests/test_data/SrTiO3.cif")

# 2. Simulate a detector image end-to-end (structure -> pattern -> image)
image = rh.simul.simulate_detector_image(
    crystal,
    voltage_kv=20.0,   # beam energy
    theta_deg=2.0,     # grazing incidence angle
    hmax=5, kmax=5,    # reciprocal-lattice range
)

# 3. Display it with a phosphor-screen colormap
rh.plots.plot_rheed(image)
```

For the lower-level pattern object (sparse reflections + detector coordinates,
before rasterization) use `rh.simul.ewald_simulator(crystal, ...)`, which
returns a `RHEEDPattern`. `simulate_detector_image` is the full kinematic
pipeline that wraps it and renders a dense image.

## Mental model

RHEED is grazing-incidence electron diffraction off a crystal *surface*. The
physics — and this codebase — follow one pipeline. Read this once and the
package layout becomes obvious:

<p align="center">
  <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/data_flow_diagram.png" alt="Data flow through Rheedium: crystal structure to reciprocal lattice to Ewald construction to detector image" width="720">
</p>

```
   CIF / XYZ / POSCAR              inout/        parse_crystal()
            │
            ▼
   CrystalStructure  ──────────────  types/      JAX PyTree (positions, cell)
            │
   (optional surface work)           procs/      slabs, reconstructions,
            │                                     defects, grains, library
            ▼
   reciprocal lattice + Ewald sphere  ucell/      build_cell_vectors(),
            │                          simul/      reciprocal lattice, Ewald
            ▼
   reflections × structure factors    simul/      form factors, Debye-Waller,
   intersected with the Ewald sphere              CTRs, finite domains
            │
   instrument broadening              simul/      divergence, energy spread,
            │                                     detector PSF
            ▼
   RHEEDPattern  ──► detector image   simul/      project + rasterize
            │
            ▼
   plot / compare / fit               plots/  recon/  audit/
```

- **Forward** (simulate): walk the pipeline top to bottom. Entry point
  `simul.simulate_detector_image` / `simul.ewald_simulator`.
- **Inverse** (reconstruct): `recon/` differentiates the forward pipeline and
  fits crystal/instrument/orientation parameters to an experimental image.
- **Validate**: `audit/` checks the simulator against physics invariants
  (Friedel symmetry, elastic closure, form-factor monotonicity, …) and against
  stored reference images.

### Why the code is shaped this way (key design choices)

- **JAX everywhere.** Every numerical function is pure, traceable, and
  `jit`/`grad`/`vmap`-friendly. This is what makes reconstruction (`recon/`)
  possible at all — the forward simulator *is* the model you optimize through.
- **PyTrees for all data structures.** `CrystalStructure`, `RHEEDPattern`,
  `RHEEDImage`, beams, and distributions (in `types/`) are registered JAX
  PyTrees. They flow through `jit`/`grad` and shard across devices unchanged.
  Construct them with the `create_*` factory functions, not raw constructors.

  <p align="center">
    <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/pytree_hierarchy.png" alt="PyTree hierarchy of Rheedium data structures" width="560">
  </p>

- **Runtime-checked array shapes.** Functions are annotated with
  [`jaxtyping`](https://docs.kidger.site/jaxtyping/) shapes (e.g.
  `Float[Array, "n 3"]`) and enforced by `beartype` during tests. A wrong-shape
  array fails loudly at the call site instead of producing silent garbage.
- **64-bit by default.** `jax_enable_x64` is set at import (in
  [src/rheedium/__init__.py](https://github.com/debangshu-mukherjee/rheedium/blob/main/src/rheedium/__init__.py)) — diffraction
  geometry needs the precision.
- **Namespace sub-packages, flat public API.** Access everything as
  `rh.<subpackage>.<function>` (e.g. `rh.inout.parse_crystal`). Each
  subpackage's `__init__.py` defines its public surface via `__all__`.
- **Optional distributed execution.** `rh.init_distributed()` wraps multi-host
  JAX setup; opt in with `RHEEDIUM_DISTRIBUTED=1` for SLURM batch jobs. No-op
  on a single machine.

### The physics, visualized

Each stage of the pipeline has a physical knob you can inspect and plot
(`rh.plots.*`). These figures come straight from the package:

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/ewald_sphere_3d_perspective.png" width="330"><br>
      <em>Ewald construction (<code>simul.ewald</code>)</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/form_factor_curves.png" width="330"><br>
      <em>Atomic form factors (<code>simul.form_factors</code>)</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/ctr_intensity_profile.png" width="330"><br>
      <em>Crystal truncation rods (<code>simul.surface_rods</code>)</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/docs/source/guides/figures/mgo_kinematic_rheed.png" width="330"><br>
      <em>Simulated RHEED — MgO(001) (<code>simul.simulate_detector_image</code>)</em>
    </td>
  </tr>
</table>

## Repository map

```
src/rheedium/
├── types/    PyTree data structures + physical constants   (the vocabulary)
├── ucell/    unit-cell & reciprocal-space crystallography
├── inout/    parsing & I/O (CIF/XYZ/POSCAR/VASP, TIFF, HDF5, ASE/pymatgen)
├── simul/    the RHEED simulator (kinematic, multislice, instrument effects)
├── procs/    differentiable surface models (slabs, reconstructions, defects)
├── recon/    inverse problems (fit structure/params from experimental data)
├── plots/    visualization (patterns + physics diagrams)
├── tools/    shared numerical kernels (Bessel, quadrature, wavelength, sharding)
└── audit/    physics-invariant checks & reference-image benchmarking
```

### What lives where

**`types/`** — the data vocabulary the whole package speaks. Define a structure
or beam here, pass it everywhere.
- `crystal_types.py`, `beam_types.py`, `rheed_types.py` — the core PyTrees.
- `distributions.py` — orientation/size distributions for ensemble averaging.
- `constants.py`, `custom_types.py` — physical constants and scalar/array aliases.
- Entry points: `create_crystal_structure`, `create_electron_beam`,
  `create_rheed_pattern`, `create_rheed_image`, `create_ewald_data`.

**`ucell/`** — crystallography: lattice parameters ↔ Cartesian vectors,
reciprocal space, surface slicing.
- Entry points: `build_cell_vectors`, `reciprocal_unitcell`,
  `generate_reciprocal_points`, `atom_scraper`, `bulk_to_slice`,
  `miller_to_reciprocal`.

**`inout/`** — get structures and images in and out.
- `crystal.py` (`parse_crystal` — auto-detect), `cif.py`, `poscar.py`,
  `vaspxml.py`, `xyz.py` — structure parsers.
- `tiff.py` — load experimental detector frames (`load_tiff_as_rheed_image`,
  `detect_beam_center`).
- `hdf5.py` — serialize PyTrees (`save_to_h5`, `load_from_h5`).
- `interop.py` — `from_ase`/`to_ase`, `from_pymatgen`/`to_pymatgen`.

**`simul/`** — the heart of the package; the forward simulator.
- `simulator.py` — high-level orchestrators: `simulate_detector_image`,
  `ewald_simulator`, `multislice_simulator`, plus detector projection/rendering.
- `ewald.py`, `kinematic.py` — Ewald-sphere construction and kinematic spots.
- `form_factors.py` — Kirkland/Lobato atomic form factors + Debye-Waller.
- `surface_rods.py`, `finite_domain.py` — crystal truncation rods and
  finite-domain broadening (the surface-sensitivity physics).
- `multislice.py`, `potential.py` — dynamical multislice primitives.
- `beam_averaging.py` — instrument broadening (divergence, energy spread, PSF).
- `sweeps.py` — batched simulators over angle/energy/orientation grids.

**`procs/`** — differentiable procedural *surface* models (apply before simulating).
- `surface_builder.py` (`create_surface_slab`, `apply_surface_reconstruction`,
  `add_adsorbate_layer`), `surface_modifier.py` (steps, occupancy/displacement
  fields), `crystal_defects.py`, `grains.py`, `preprocessing.py` (experimental
  image cleanup), `library.py` (ready-made surfaces: Si(111)-7×7, GaAs(001)-2×4,
  SrTiO₃, MgO, …).

**`recon/`** — fit parameters to data by differentiating the simulator.
- `optimizers.py` (`gauss_newton_reconstruction`, `adam_reconstruction`),
  `losses.py`, `orientation.py` (orientation fitting + Fisher-information
  uncertainty).

**`plots/`** — `figuring.py` (`plot_rheed`, phosphor colormap) and `diagrams.py`
(publication physics figures: Ewald sphere, form factors, CTR profiles, …).

**`tools/`** — shared math kernels used across `simul/`: `special.py` (Bessel
functions), `quadrature.py` (Gauss-Hermite), `simul_utils.py` (`wavelength_ang`,
`incident_wavevector`, `interaction_constant`), `parallel.py` (device sharding),
`wrappers.py` (`jax_safe`).

**`audit/`** — `invariants.py` (`run_default_invariants` — physics checks, no
fixtures needed), `metrics.py` (image-space accuracy metrics),
`reference_benchmark.py` (regression against stored reference images).

## Common workflows

```python
import rheedium as rh

# --- Build and simulate a reconstructed surface ---
crystal = rh.inout.parse_crystal("structure.cif")
slab    = rh.procs.create_surface_slab(crystal, miller_indices=[1, 1, 1])
recon7  = rh.procs.apply_surface_reconstruction(slab, m=7, n=7)
image   = rh.simul.simulate_detector_image(recon7, voltage_kv=20.0, theta_deg=2.0)

# --- Or start from a pre-built surface in the library ---
si = rh.procs.si111_7x7()

# --- Reconstruct: fit a structure to an experimental image ---
exp    = rh.inout.load_tiff_as_rheed_image("rheed.tif")
result = rh.recon.adam_reconstruction(crystal_init, exp.data, ...)

# --- Validate the simulator against physics ---
rh.audit.run_default_invariants()
```

## Tests, tutorials, and docs

- **Tutorials** — runnable, narrated examples in
  [tutorials/](https://github.com/debangshu-mukherjee/rheedium/tree/main/tutorials)
  (paired `.ipynb` + jupytext `.py`), e.g.
  [01_kinematic_SrTiO3.py](tutorials/01_kinematic_SrTiO3.py). Start here for a
  worked end-to-end run. Also rendered in the
  [online tutorials](https://rheedium.readthedocs.io/en/latest/tutorials/index.html).
- **Tests** —
  [tests/](https://github.com/debangshu-mukherjee/rheedium/tree/main/tests)
  mirrors `src/` one-to-one
  (`tests/test_rheedium/test_<subpkg>/test_<module>.py`). To understand any
  function's contract, read its test. Property-based tests use `hypothesis`;
  shape contracts are enforced via `jaxtyping` + `beartype`. The suite is also
  rendered as a
  [Testing & Validation reference](https://rheedium.readthedocs.io/en/latest/tests/index.html).
- **Guides & API** — conceptual
  [guides](https://rheedium.readthedocs.io/en/latest/guides/index.html) (Ewald
  sphere, form factors, CTRs, PyTree architecture, …) and the full
  [API reference](https://rheedium.readthedocs.io/en/latest/api/index.html) on
  Read the Docs.

## Development

```bash
uv sync --extra dev
```

Recommended local validation before pushing (matches CI):

```bash
uv run ruff check src/ tests/        # lint
uv run ruff format --check src/ tests/   # format
uv run ty check src                  # static type check
uv run pytest -v                     # tests (pytest-xdist runs them in parallel)
```

See
[CONTRIBUTING.md](https://github.com/debangshu-mukherjee/rheedium/blob/main/CONTRIBUTING.md)
for conventions (numpydoc docstrings, jaxtyping idioms, the tests-mirror-src
layout, and the type-checker configuration rationale).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/debangshu-mukherjee/rheedium/blob/main/LICENSE) file for details.

## Citation

If you use Rheedium in your research, please cite:

```bibtex
@software{rheedium_software,
  title={Rheedium: High-Performance RHEED Pattern Simulation},
  author={Mukherjee, Debangshu},
  year={2025},
  url={https://github.com/debangshu-mukherjee/rheedium},
  version={2025.10.05},
  doi={10.5281/zenodo.14757400},
}
```
