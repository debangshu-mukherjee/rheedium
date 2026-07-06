# Changelog

All notable changes to **rheedium** are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).
Each entry summarizes the commits that landed for that version bump in
`pyproject.toml`; dates are the date the version was committed.

## [Unreleased]

### Red-team remediation

- Standardized the binary Ewald condition on one absolute tolerance
  (WP4.1/M7): `simul.ewald_allowed_reflections` and
  `simul.find_kinematic_reflections` (and the
  `simul.kinematic_spot_simulator` pass-through) now take
  `tolerance_inv_ang: scalar_float | None = None`, the Ewald-shell
  half-thickness in 1/Angstrom, replacing the old `tolerance` knob that
  meant a 5% fractional error in one function and an absolute 0.05
  1/Angstrom in the other. `None` derives
  `3 * compute_shell_sigma(k, energy_spread_frac=1e-4,
  beam_divergence_rad=1e-3)`; at 20 kV the default admits
  |dk| < ~0.22 1/Angstrom instead of the old 3.66 1/Angstrom.
- Padded `-1` slots from `ewald_allowed_reflections` (both modes) and
  `find_kinematic_reflections` now return exactly zero `k_out` rows and
  zero intensities instead of gathering the last live reflection through
  JAX's negative-index semantics (WP4.2/M8); summed intensities no longer
  contain phantom copies.
- `SurfaceConfig` gained `method="none"` (all-False surface mask) and it
  is now the default in `ewald_simulator`,
  `compute_kinematic_intensities_with_ctrs`, and the internal
  `_ewald_amplitude_pattern` when `surface_config=None` (WP4.3/M10):
  bulk unit cells repeated by the CTR factor have no surface atoms, so
  the previous silent 2x mean-square-displacement enhancement of the top
  30% of every cell is gone; slab-based callers must opt in explicitly.
  `compute_kinematic_intensities_with_ctrs`'s `surface_fraction`
  parameter is retained but ignored unless a height-based config is
  passed explicitly.
- Periodic real-space grids in `simul.crystal_projected_potential` and
  `simul.sliced_crystal_to_projected_potential_slices` now use
  `arange(n) * (L / n)` instead of `linspace(0, L, n)` (WP4.4/M12), so
  the sample spacing matches the `fftfreq(n, L/n)` propagator lattice
  and the periodic boundary column is no longer double-weighted.

- Fixed VASP POSCAR scale semantics (WP3.3/M14): a negative universal scale
  is now interpreted as a target cell volume
  (`factor = (-scale / det(raw_lattice)) ** (1/3)`, requiring a positive
  determinant), Cartesian-mode coordinate lines are multiplied by the
  resolved scale factor before fractional conversion, and a scale line with
  three values is rejected with an accurate "three lattice scale factors are
  not supported" message. `_parse_poscar_positions` gained a
  `scale_factor` keyword (default 1.0).
- `rheedium.types.XYZData` gained a real `forces` PyTree field
  (`Float[Array, "N 3"] | None`, default `None`) and `create_xyz_data`
  accepts a `forces` keyword (WP3.4/M15). `inout.parse_vaspxml` /
  `parse_vaspxml_trajectory` now actually store parsed VASP forces instead
  of only advertising them in `properties`; HDF5 serialization round-trips
  the new field. The `properties` forces descriptor is still emitted only
  when forces are present.
- Extended-XYZ metadata parsing anchors the `energy=` regex with a
  `(?<![A-Za-z_])` boundary so `free_energy=` no longer shadows the
  standalone `energy` key (WP3.5/N18).
- `create_xyz_data(lattice=None)` now stores `None` instead of fabricating
  an identity lattice, and `inout.xyz_to_crystal` uses an explicit
  `lattice is None` check for the bounding-box fallback, so a genuine
  1 Angstrom cubic cell survives ingestion un-replaced (WP3.5/N15).
  `XYZData.properties` and `XYZData.comment` now default to `None`.
- TIFF fixes (WP3.6/N20): `load_tiff_sequence(sort_by="timestamp")` now
  deterministically falls back to lexicographic name order when *any*
  frame timestamp is NaN (documented); `detect_beam_center` smooths with an
  edge-padded separable Gaussian convolution instead of circular FFT
  convolution, so spots near an image edge no longer bleed to the opposite
  edge (edge-spot centroid error drops from ~1.2 px to ~0.03 px); docstring
  references to the nonexistent `DetectorGeometry.center_pixel` now point
  to `DetectorGeometry.beam_center_px`.

- Changed `ucell.atom_scraper` to take scalar `thickness_ang` measured along
  the normalized zone axis, replacing the previous vector `thickness` argument.
  The returned scraped structure is rebuilt to satisfy the canonical
  `frac @ cell == cart` frame contract.
- Added the foundational `recon` optimization API from the optimistix/optax
  plan: constrained latent transforms, richer differentiable losses, the
  general `rheedium.types.ReconProblem` + `recon.solve`/`multistart` surface,
  incoherent distribution weight reconstruction, the base-object distribution
  library builder,
  Fisher/Laplace uncertainty helpers, and `recipe_deviation` reports.
  The legacy hand-rolled `ReconstructionResult`, `gauss_newton_*`,
  `adam_*`, and `adagrad_*` optimizer APIs were removed with no shim; migrate
  reconstruction callers to `rheedium.types.ReconProblem` plus `recon.solve`.
  HDF5 serialization now registers `rheedium.types.ReconResult`.
- Centralized the remaining structured carriers under `rheedium.types`:
  recon problem/result/spec/UQ/report carriers and axis-spec constructors,
  TIFF `FrameMetadata`, and detector-kernel axis update tuples. The consuming
  `recon`, `inout`, and `procs` subpackages no longer export those carriers.
- Completed the generalized recon UQ gate: `uncertainty.py` now exposes
  blackjax NUTS posterior sampling, R-hat/ESS diagnostics, credible intervals,
  and Laplace/Fisher inverse-mass warm starts; `rheedium.types.PosteriorSamples`
  owns the sample container. Tests cover empirical covariance calibration,
  multimodal posterior retention, orientation Fisher regression, and free-form
  distribution bands.
- Completed the K3 recon robustness gate: `multistart` can now generate seeded
  random starts from a template latent, scalar minimization results report the
  optimized objective in `ReconResult.loss`, and tests cover reproducibility,
  planted local-minimum escape, and bracketed-init faster refinement.
- Completed the K5 recipe-deviation contract: reports now default to K4
  Laplace covariance when uncertainty is not supplied, carry covariance and
  per-parameter sigma metadata, export a schema-validated automaton payload, and
  include calibrated matched/mismatched recipe tests.
- Completed the K6 recon hardening gate: `solve`/`multistart` can enable the
  persistent XLA compilation cache before inversion scans, `fit_geometry_beam`
  is exported as the fixed-crystal geometry/beam convenience wrapper,
  `fit_orientation_weights` now uses the shared `ReconProblem`/`solve` path
  instead of a bespoke Adam loop, and the inverse API is Routine-Listed,
  documented, typed, and frozen by tests.
- Multislice simulator and Lobató potentials are now the **default** scattering
  model.
- Began detector-geometry rationalization: `DetectorGeometry` now carries dense
  detector calibration (`image_shape_px`, `pixel_size_mm`, `beam_center_px`) in
  addition to projection distance/tilt/PSF fields, and sparse render helpers now
  consume that carrier directly.
  Migration:
  ```python
  # Before
  image = rh.simul.render_pattern_to_image(
      pattern,
      image_shape_px=(192, 192),
      pixel_size_mm=(1.5, 3.0),
      beam_center_px=(96.0, 8.0),
      spot_sigma_px=1.4,
  )
  extent = rh.simul.detector_extent_mm(
      image_shape_px=(192, 192),
      pixel_size_mm=(1.5, 3.0),
      beam_center_px=(96.0, 8.0),
  )

  # After
  detector = rh.types.DetectorGeometry(
      image_shape_px=(192, 192),
      pixel_size_mm=(1.5, 3.0),
      beam_center_px=(96.0, 8.0),
  )
  image = rh.simul.render_pattern_to_image(
      pattern,
      geometry=detector,
      spot_sigma_px=1.4,
  )
  extent = rh.simul.detector_extent_mm(detector)
  ```
- Completed the R1/RG1 detector-projection cleanup: production sparse-pattern
  call sites now route projection through `DetectorGeometry` and
  `project_on_detector_geometry`; the raw-distance `project_on_detector`
  helper was removed with no shim.
  Migration:
  ```python
  # Before
  points = rh.simul.project_on_detector(k_out, detector_distance)

  # After
  detector = rh.types.DetectorGeometry(distance=detector_distance)
  points = rh.simul.project_on_detector_geometry(k_out, detector)
  ```
- Removed the orphan `coherence_envelope` beam helper from the public surface;
  partial coherence is represented through beam-mode distributions, and
  `SizeDistribution` is consumed by the finite-domain intensity bridge.
- Completed the R2/RG2 averaging collapse: pattern-space grain/domain mixers,
  `instrument_broadened_pattern`, and `integrate_over_orientation` now build
  generic `Distribution` axes and reduce through the shared Layer-1 reducers.
  The standalone `ewald_simulator_with_orientation_distribution` sparse wrapper
  was removed; route orientation ensembles through
  `simulate_detector_image(render=RenderParams(orientation_distribution=...))`.
- Completed the R4/RG4 detector-image carrier cut: `simulate_detector_image`
  now consumes `BeamSpec`, `SurfaceCTRParams`, `DetectorGeometry`, and
  `RenderParams`; the old scalar keyword surface was removed with no shim.
  The detector sweep API collapsed to `simulate_detector_image_sweep` and
  `simulate_detector_image_grid`, and affected tutorials/notebooks/generator
  scripts were migrated.
- Completed the R5/RG5 naming and procs-contract cut: public beam energy names
  use `energy_kev`, generated guide-figure scripts no longer call the retired
  `voltage_kv`/`voltage_range_kv` keywords, and degree-to-radian incidence
  conversion is centralized at the detector-kernel boundary. The `procs`
  modules now document the structure-builder / distribution-producer /
  sub-coherence-modifier return split.
- Completed the R6/RG6 module reorganization: `types.distributions` is now a
  subpackage split into base, beam, orientation, and size modules with the old
  public import path preserved. The transient `rheedium.simul.layer0` /
  `rheedium.simul.layer1` compatibility modules were removed with no shim;
  import simulator-owned kernels from `rheedium.simul` or
  `rheedium.simul.simulator`.
- Removed compatibility forwarding exports for `lattice_to_cell_params` from
  `rheedium.inout.crystal` and `DetectorGeometry` from
  `rheedium.types.rheed_types`; import them from `rheedium.inout.lattice` (or
  `rheedium.inout`) and `rheedium.types.detector` (or `rheedium.types`).
- Began W7 naming cleanup: code and tests now use `energy_kev` for beam energy
  in keV, deleting the old `voltage_kv` name from the checked code surface.
  Tutorials and source docs now use the same `energy_kev` spelling; internal
  `voltage_v` locals remain only where calculations are explicitly in volts.

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
