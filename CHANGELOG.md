# Changelog

All notable changes to **rheedium** are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).
Each entry summarizes the commits that landed for that version bump in
`pyproject.toml`; dates are the date the version was committed.

## [Unreleased]

- Multislice simulator and Lobató potentials are now the **default** scattering
  model.

## [2026.6.6] - 2026-06-23

- Publish to PyPI via `uv publish` driven entirely from `pyproject.toml`.

## [2026.6.5] - 2026-06-23

- Parallelized simulation routines; new visualizer, temperature-dependence
  notebooks, and sweep functions.
- Migrated tutorials from Marimo back to Jupyter/IPython notebooks.
- Equinox-based runtime error handling (`eqx.error_if` / `checkify`), worked out
  in detail.
- Extensive jaxtyping annotations across the test suite (bare `Array` → specific
  jaxtyping types); fixed invariant-typing bugs.
- Cheap JIT wins; all GitHub runners moved to Node 24.
- Removed `MANIFEST.in` (unnecessary under the uv build backend); README update.

## [2026.6.1] - 2026-06-11

- Vectorized parameter sweeps, with sweep tutorials/notebooks and docs.
- Fully typed test suite: `None` return annotations, typed tests, ANN lint
  violations cleared, and external test linking added to function docstrings.
- File-input error paths now tested; updated test fixtures.

## [2026.4.13] - 2026-04-15

- Orientation reconstruction support.
- `Distribution` types representing an incoherent sum over orientation
  probabilities.
- Fixed STO simulation; orientation annotations; Marimo notebook updates.

## [2026.4.10] - 2026-04-12

- Completed migration to the Astral stack (ruff / uv).
- New `procs` submodule; `simul_utils` and Bessel functions relocated into the
  `tools` submodule.
- Multislice implementation; HDF5 support; inverse modelling in the `recon`
  submodule.
- Audit submodule for reference quantification; physics invariants and updates.

## [2026.4.7] - 2026-04-07

- Lobató–van Dyck atomic-potential values, codes, and tests; corrected helper
  potential.
- Switched notebooks from `.ipynb` to Marimo (added `marimo` dependency).
- Vectorized code; recon test fixtures; generated test data.

## [2026.4.6] - 2026-04-06

- Expanded the `recon` library; more tests.
- CI runs the test suite with `-n auto` / 8 cores; dropped Python 3.15.

## [2026.4.3] - 2026-04-03

- `jax_safe` is now a first-class wrapper.
- TIFF image reader → `RHEEDImage`.
- Separate constants file; conftest dynamic memory handling for tests;
  rationalized `test_simul` directory structure; gradient tests.

## [2026.4.2] - 2026-04-02

- Experimental-data preprocessing; beam averaging.
- CIF types and extensive type hints throughout the codebase.
- Gradient and imaging tests; latest-Python support; removed deprecated
  functions; fixed circular imports.

## [2026.4.1] - 2026-04-01

- Improved docstrings.

## [2025.10.10] - 2025-12-31

- More RHEED pattern figures and guides; SVG figures replacing ASCII art.
- Ewald cleanup; fixed atom types, `__init__`, and test warnings; docs now link
  to GitHub.

## [2025.10.09] - 2025-12-29

- Atom viewer and atom-plotting utilities.
- Updated CIF readers; added bismuth selenide test data.

## [2025.10.08] - 2025-12-29

- Finite-domain physics upgrade.
- Documentation overhaul: figures in docs, a jaxtyped patch plus mock imports in
  `conf.py` so functions render correctly, absolute URLs, and math fonts.
- Removed redundant per-file x64 setup; black formatting; automated LOC badge.

## [2025.10.07] - 2025-12-27

- Better Ewald sphere handling; improved POSCAR/VASP support and more data
  loaders.
- Removed deprecated functions; fixed failing tests; consolidated README.

## [2025.10.05] - 2025-12-26

- Common crystal reader handling both CIF and XYZ files.

## [2025.10.03] - 2025-12-22

- Ewald sphere builder; kinematic CTR with extinction; reached meaningful
  simulation results.

## [2025.10.02] - 2025-12-22

- Streak-rendering work; renamed `kinematic.py` → `paper_kinematic.py` and
  cleaned the kinematic code.
- CIF fix; notebooks rendered in the docs.

## [2025.09.30] - 2025-11-21

- Simulator-function updates; multislice work; MgO kinematic; beam-error fix.
- Updated testing style; Ruff cleanup.

## [2025.09.10] - 2025-11-02

- `create_surface_rods` surface-rods function (tested).
- Separated the CUDA install — GPU JAX only via the `dev`/`cuda` extras, CPU JAX
  everywhere else.
- Docs rebuilt in the Janssen style with Flow/Algo formatting.

## [2025.08.31] - 2025-09-23

- Added `form_factors.py` with tests; new `unitcell.py` functions and bug fixes.
- Switched to NumPy docstring style; code coverage with a coverage token.

## [2025.06.25] - 2025-08-21

- Tests for the `types` submodule; PyTrees follow numpydoc.
- readthedocs build-environment and doc-workflow updates; black + isort.

## [2025.06.23] - 2025-08-20

- Switched to numpydoc; type hints for internal variables and private functions.
- Mock imports for jaxtyping/beartype during the doc build.

## [2025.06.21] - 2025-08-20

- Started the XYZ file reader.
- Preloaded atomic symbols and Kirkland polynomials.
- Modern documentation; stopped tracking the data folder.

## [2025.06.19] - 2025-06-23

- Moved wavelength calculations from the unit-cell module to the simulation
  module.
- Fixed circular imports and missing `__init__.py` imports.

## [2025.06.17] - 2025-06-20

- `crystal_potential` for generating multislice potentials; `atomic_potential`
  now accepts arbitrary grids and arbitrary atom centering, computing potentials
  only for unique atoms and shifting with `fftshift`.
- JAX-compliant validation via `jax.lax.cond` (replacing Python `if`/`else`) in
  the factory functions.
- Added license; longer, more detailed README.

## [2025.06.16] - 2025-06-17

- Major release: RHEED simulation built on atomic potentials.
- Kirkland potentials (CSV + potential function) and Bessel functions.
- Factory functions for RHEED types and for loading data.
- Clean, explicit namespaces — each submodule imports only what it needs.
- Cross-referenced docstrings with example usage and Flow sections.

## [2025.04.02] - 2025-04-10

- Refined atom scraper; `parse` and `scrape` functions.
- CIF parser handles symmetry operators and symmetry expansion, and now reads
  *all* atoms (previously only the first line).
- New `plots` submodule, separating pure-JAX submodules from external-facing
  ones (`inout`, `plots`).

## [2025.04.01] - 2025-04-03

- Fixed the CIF parser; added a CIF file reader.
- New PyTree for experimental images; started the `recon` module (takes
  experimental data + a CIF file and returns the correct angle).
- All submodule names normalized to 5 characters; type-import fixes.

## [2025.03.31] - 2025-04-01

- Fixed import error; explicit `types` submodule; fixed the atom-scraper code.

## [2025.01.29] - 2025-01-30

- Fixed gather-positions error; doc updates.

## [2025.01.28.2] - 2025-01-28

- Docs-module updates.

## [2025.01.28v2] - 2025-01-28

- Began simulation testing; doc updates; README badges.
- Removed the envs folder — environment management is handled by uv.

## [2025.01.28] - 2025-01-28

- Basic RHEED simulator function complete; simulation submodule started.
- Rebuilt the atom-scraper function; custom docstring style.
- Project reorganized into submodules.

## [2025.01.14] - 2025-01-14

- Switched to the `typing` module from beartype; I/O tests; documentation
  groundwork.

## [2024.12.23] - 2024-12-23

- CIF reader; phosphor-screen colormap.
- Switched to uv; upgraded `num_type`.
- Type stubs, badges folder, and more tests.

## [2024.11.15] - 2024-11-15

- Progressed the unit-cell module.

## [2024.11.05] - 2024-11-05

- Initial package structuring and README.
