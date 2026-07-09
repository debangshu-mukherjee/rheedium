# Changelog

All notable changes to **rheedium** are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).
Each entry summarizes the commits that landed for that version bump in
`pyproject.toml`; dates are the date the version was committed.

## [Unreleased]

### Red-team remediation

- Final integration (Phase 12). Regenerated the synthetic reference-image
  fixtures (`mgo001`, `srtio3_001`) on the fully-fixed pipeline and
  un-`xfail`ed the two benchmark regression tests (WP8.11/N23), which now
  pin pixel output (`atol=1e-12`) and self-consistent benchmark metrics
  (NCC = 1, all geometry errors = 0). All twelve `run_default_invariants`
  checks pass, including the new Phase-8 physics invariants.
- Corrected the Ewald construction diagrams (Phase 10, WP10.1/WP10.2, M19).
  `plot_ewald_sphere_2d` now draws `k_out` from the sphere center
  `C = −k_in` to the specular rod–sphere intersection `P = (0, 2k sinθ)`
  (asserted on the sphere to 1e-9, `|P − C| = k`) and `k_in` from `C` to
  the reciprocal origin; `plot_ewald_sphere_3d` meshes the near-origin
  belt (`v ∈ [π/2 − 0.15, π/2 + 0.15]`) instead of the north-pole cap, so
  the drawn surface actually intersects the drawn rods (min distance to
  the origin rod < 0.5 Å⁻¹, was ≈ 45).
- Hardened the data-parallel harness during integration.
  `tools.distribute_batched` now edge-replicates padding rows by default
  (`pad_value=None`) so the rows appended to reach a device multiple are
  always valid inputs to runtime checks — fixing an 8-device all-reduce
  deadlock where the new `wavelength_ang` guard fired on zero-padded sweep
  rows; an explicit float still selects legacy constant padding.
- Fixed the tools and types layer (Phase 9). `tools.special.bessel_kv`
  (WP9.1/M28) was rebuilt — integer orders via stable upward recurrence
  (static cap 50, `eqx.error_if` beyond), half-integers via exact closed
  forms, non-integers via an 8-term asymptotic series — matching SciPy to
  `<1e-6` (integer), `<1e-12` (half-integer) and `<1e-3` (non-integer
  midrange), with `bessel_kv(0, x) == bessel_k0(x)`. The exponential size
  distribution (WP9.2/M29a) discretizes the truncated exponential by
  equal-probability bin centroids (`E[L]` within 0.5 %, skewness within
  10 % of 2). `shard_array` (WP9.3/M29b) honors its `devices` argument via
  `jax.make_mesh(..., devices=...)`. `SurfaceConfig` (WP9.4/N14) is now an
  `eqx.Module` with `explicit_mask` as an array leaf and scalar fields
  `static=True`, and `SurfaceCTRParams.surface_config` is a pytree child,
  so equal-but-fresh masks no longer force recompilation.
  `discretize_orientation` (WP9.5/N16) drops the ignored `n_sigma_range`
  parameter and `create_orientation_distribution`'s docstring matches its
  raise-on-negative-FWHM behavior. The harness (WP9.6/N17, N25) re-execs
  on `--unchecked` with `RHEEDIUM_DISABLE_RUNTIME_CHECKS=1` (loop-guarded)
  and `emit` writes strict JSON (`allow_nan=False`) with a non-finite →
  `null` sanitizer. Small-tool sweep (WP9.7/N25): `wavelength_ang` rejects
  non-positive energy, `jax_safe` forwards `**kwargs`, XLA-flag dedup is
  token-wise, and `crystal_types.__all__` exports `EdgeOnSlices` /
  `KirklandParameters` and their factories.
- Made the reconstruction and audit layers trustworthy (Phase 8).
  Incoherent orientation averaging (WP8.1/M22) sums intensities directly
  through the new `apply_distribution_intensity` (no `√I` complex
  round-trip), so gradients are finite at exactly-zero pixels and
  `fit_orientation_weights` no longer raises. `laplace_inverse_mass_matrix`
  (WP8.2/M23) returns the posterior covariance (blackjax convention), not
  precision. `multistart` (WP8.3/M24) guards `nanargmin` (all-NaN → first
  result, `converged=False`, warning) and `converged` (WP8.4/M25) reflects
  solver success only, with an explicit opt-in `loss_atol`. The
  ordered-bounded transform (WP8.5/M26) gained a correct inverse
  (`∂x_N/∂z_i ≠ 0`, `x[-1] < upper` strictly). Orientation residuals
  (WP8.6/M17) are mask-weighted (`Σ(mask·r²)/Σmask`). `sample_posterior`
  (WP8.7/N21) only takes the 2-D `(chains, params)` path for
  `jax.Array`, never pytrees. `reconstruct_incoherent_weights` (WP8.8/N24)
  uses projected-gradient NNLS (residual strictly below clip-renormalize)
  and `_split_r_hat` returns NaN for `n_draws < 4`, gating on ESS. Audit
  invariants (WP8.9/M27) call the production structure factor for the
  Friedel check on non-centrosymmetric GaAs, add independent legs to the
  elastic closure, and register five new invariants (Bethe sum rule,
  Hankel pair, slab density, triclinic frame contract, k∥ conservation).
  Metric guards (WP8.10/N8, N22) handle non-positive peaks, all-masked
  peaks (nan-aware), and constant images, and scale the recon NCC-loss
  `ε` by the image second moment so the flat-image gradient no longer
  explodes. The reference benchmark (WP8.11/N23) uses the tensor-product
  incoherent average over `θ × E`.
- Rebuilt the surface layer (Phase 7 Wave 2). `create_surface_slab`
  (WP7.1/C5) now cuts the true `(hkl)` surface: it reuses
  `reorient_to_zone_axis` and reduces the in-plane cell to its primitive
  surface mesh, so Si(111) returns `|a| = |b| = 3.840 Å`, `γ = 120°`
  (previously a non-primitive/rotated cell). It accepts either
  `n_layers` (exact integer repeats, takes precedence) or
  `slab_thickness_angstrom`, tiles in-plane only via `in_plane_repeat`,
  and wraps fractional coordinates into `[0, 1)` (fixes N6) so the
  returned `a`/`b` are genuine lattice translations.
- `apply_surface_reconstruction` (WP7.4/M18, N6, N7) now tiles over the
  signed bounding box of the supercell corners, tests membership in
  half-open `[0, 1)` supercell fractional coordinates with corner-snap
  dedup (exactly `|det M| · n_basis` atoms for any integer matrix,
  including shears), orders atoms by `(z, y, x)`, and raises
  `ValueError` on a displacement/surface-atom count mismatch instead of
  silently truncating. `atomic_displacements` is now optional
  (`None` = expand only).
- Fixed the structure library (WP7.3/C7). GaAs now uses the correct
  8-atom zincblende basis (`(001)` layer spacing `a/4 = 1.4133 Å`,
  alternating Ga/As). Every reconstructed surface is built from the true
  base mesh, expanded with `apply_surface_reconstruction`, and decorated
  by the new `place_adatoms` helper (adatoms placed by height above the
  top atomic layer, not a cell fraction): `si100_2x1` (symmetric 2.35 Å
  dimers by displacement), `si111_7x7` (7×7 supercell + 12 adatoms,
  documented adatoms-only simplified DAS), `gaas001_2x4` (2 As dimers +
  missing-dimer trench), and `srtio3_001_2x1` (new `termination`
  argument, double-layer TiO2 unit). Fractional-order rods are now
  present in each reconstruction and vanish for the ideal slab.
- Vicinal step models (WP7.5/N5, WP7.6/N12). `apply_step_edge_field`
  renames its period argument to `corrugation_period_ang` (each terrace
  is half a period); a new `apply_vicinal_staircase` applies the smooth,
  monotonic, differentiable staircase
  `h(x) = −s [x/w − sin(2πx/w)/(2π)]` (mean slope `−s/w`, drop `5s` over
  five terraces). `vicinal_surface_step_splitting` drops the unused
  `hk_index`, normalizes by the analytic zero-splitting peak
  (grid-independent), and documents the `4(w/d)²` finesse as an
  uncalibrated heuristic.
- Site occupancy is now a first-class float field through the entire
  simulation stack (WP7.2/C6). `CrystalStructure`, `SlicedCrystal`, and
  `EwaldData` carry an optional `occupancies` array (`None` = fully
  occupied; factories validate the [0, 1] range), and every scattering
  kernel multiplies each atom's form factor or projected potential by
  its occupancy: `_compute_structure_factor_single`, `build_ewald_data`,
  both `ewald_allowed_reflections` modes (including the continuous
  rod-intersection path via `rod_base_intensities`),
  `simple_structure_factor`, `surface_structure_factor` and the CTR
  entry points, the `ewald_simulator` /
  `compute_kinematic_intensities_with_ctrs` / `_ewald_amplitude_pattern`
  kernels, `crystal_projected_potential` (new optional `occupancies`
  argument), `sliced_crystal_to_projected_potential_slices`, and
  `crystal_to_edge_on_slices`. A half-occupied Si site now scatters as
  exactly `0.5 * f_Si(q)` (previously it scattered as nitrogen via the
  truncated "effective Z" encoding), occupancy 0 contributes exactly
  zero, and gradients w.r.t. occupancies are finite and nonzero.
  Producers keep Z integral and set occupancies instead:
  `apply_vacancy_field` scales occupancies (a 1% Si vacancy stays
  silicon, never aluminum), `apply_interstitial_field` stores candidate
  occupancies, `apply_antisite_field` now returns **two co-located
  sites** per input site (host at `occ*(1-x)`, substitute at `occ*x`,
  doubling N) so the site scatters as `(1-x) f_A + x f_B`,
  `apply_surface_occupancy_field` scales surface occupancies, and
  `add_adsorbate_layer` stores `coverage_fraction` as the adsorbate
  occupancy. Carriers preserve the column: `atom_scraper`,
  `bulk_to_slice`, `reorient_to_zone_axis` (replicas inherit their
  source atom's occupancy), `create_surface_slab`,
  `apply_surface_reconstruction`, `parse_cif_and_scrape`, displacement
  perturbations, and HDF5 round trips. Interop maps occupancies to
  `atoms.info["occupancies"]` for ASE and to species dictionaries for
  pymatgen; `from_pymatgen` now expands disordered sites into co-located
  atoms instead of raising.

- Unified the CTR model on the semi-infinite truncation rod
  (WP5.1/M4, M6, N2): `simul.calculate_ctr_intensity` /
  `simul.calculate_ctr_amplitude` now take continuous `l_values` instead
  of Cartesian `q_z`, build `q = h b1 + k b2 + l b3` from the full
  reciprocal basis (no hard-coded Cartesian-z rod), and multiply the
  single-cell structure factor by the truncation factor
  `1 / (1 - exp(-2 pi i l) exp(-eps))` (amplitude) /
  `1 / (1 - 2 e^-eps cos(2 pi l) + e^-2eps)` (intensity). New public
  helpers `ctr_truncation_amplitude` / `ctr_truncation_intensity`. The
  per-layer attenuation `layer_attenuation` (eps, default 0.01) replaces
  `ctr_regularization`, which is retained as a deprecated alias
  (`eps = 2 asinh(sqrt(reg)/2)` preserves the Bragg cap and emits a
  `DeprecationWarning`). `ewald_simulator` / `kinematic_amplitude` /
  `SurfaceCTRParams` gained `layer_attenuation`; `ctr_power` and
  `roughness_power` are documented as non-physical diagnostic knobs and
  both now default to 1.0 (previously `roughness_power=0.25`).
  `compute_kinematic_intensities_with_ctrs` no longer adds an integrated
  CTR intensity on top of the Bragg `|F|^2` (the double count is gone):
  `ctr_mixing_mode="rod"` (new default) applies
  `|F(q)|^2 * T(l) * exp(-q_z^2 sigma^2)` as one model to near-integer
  (h, k) reflections, `"none"` returns bare `|F|^2`, and
  `"coherent"`/`"incoherent"` are deprecated aliases of `"rod"`.
- One roughness convention (WP5.2/M5): `simul.roughness_damping` is the
  single exponent definition and returns the **amplitude** factor
  `exp(-q_z^2 sigma^2 / 2)`; amplitude paths multiply by it once and
  every intensity path multiplies by its square `exp(-q_z^2 sigma^2)`
  (previously three inconsistent conventions under-damped by 2x, 4x, and
  8x in the exponent). The `plots.plot_roughness_damping` figure now
  plots the intensity factor.
- Renamed `detector_acceptance` to `detector_acceptance_inv_ang` in
  `integrated_rod_intensity`, `integrated_ctr_amplitude`, and
  `compute_kinematic_intensities_with_ctrs` (WP5.3/M20): it is the
  Gaussian sigma of the q_z acceptance window in 1/Angstrom and was
  never an angle in radians.
- Finite-domain overlap now operates on (h, k) rods, not discrete
  (h, k, l) points (WP5.4/M9, N11): new `simul.rod_domain_overlap`
  solves the rod-sphere quadratic per rod, evaluates real intersections
  at the continuous `l*` with unit envelope weight (composing with the
  WP5.1 truncation shape via a 7-point Gauss-Hermite l-window with
  `sigma_l = sigma_z / |b3|`), and weights lateral misses by
  `exp(-d^2 / (2 sigma_lat^2 + 2 sigma_shell^2))` with the closed-form
  miss distance `d = sqrt(-disc) / (2 sqrt(a))`.
  `finite_domain_intensities` and the `domain_extent_ang` branch of
  `ewald_allowed_reflections` are rewired to the rod API (each rod is
  represented by the grid point whose integer l is nearest `l*`; both
  gained a `layer_attenuation` keyword). `extent_to_rod_sigma` now
  returns all three widths `[sx, sy, sz]` with the sinc^2-matched
  constant `0.886/2.355 * 2 pi / L ~ 2.364/L` (the old
  `2 pi / (L sqrt(2 pi)) ~ 2.507/L` was ~6% too wide, and the z-extent
  previously never entered the model). The legacy point-based Gaussian
  survives privately as `finite_domain._point_domain_overlap`;
  `rod_ewald_overlap` is removed from the public API.
  `find_ctr_ewald_intersection` moved from `simul.simulator` to
  `simul.finite_domain` (still re-exported from both).
- Consistent acceptance-window normalization (WP5.5/N3):
  `integrated_rod_intensity` and `integrated_ctr_amplitude` both return
  the window-weighted mean (`sum(w x) / sum(w)`), so
  `|integrated_amplitude|^2 ~ integrated_intensity` on a smooth rod
  (previously the intensity version trapezoid-integrated the weighted
  integrand without normalizing, making the two paths differ by an
  arbitrary scale).

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
