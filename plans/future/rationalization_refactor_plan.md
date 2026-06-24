# Codebase Rationalization & Refactor Plan

Scope: `rheedium` — after the
[distribution_framework_plan.md](plans/partial/distribution_framework_plan.md) lands,
harvest the simplification it makes possible and pay down the structural debt it
exposes. This is **not** gratuitous churn: every item either deletes redundancy,
wires a dangling piece, or collapses a duplicated path — and every item must
**preserve the Core Principle** (CONTRIBUTING.md → *Invertible Modularity*): no
refactor may weld a differentiable seam shut.

Status: **proposed** — sequenced *after* (and partly interleaved with) the
framework phases. Assumes the three-layer architecture (amplitude kernels →
`Distribution` integrator → producers) exists.

Guard on every workstream below: **tests stay green, public results unchanged
(or migrated behind a deprecation), and `jax.grad` still flows end-to-end.** A
refactor that breaks differentiability has failed even if tests pass.

---

## 1. Why now

The framework turns four structural facts into cleanup opportunities:

1. **Four near-identical ensemble integrators** become one `apply`/`apply_all`.
2. **Dangling code** (`coherence_envelope`, `SizeDistribution`) gets wired or
   removed — no more orphans.
3. **24-argument signatures** (`simulate_detector_image` and its seven sweep
   clones) collapse once parameters are bundled into PyTree config objects.
4. **Two parallel detector-projection paths** unify on one `DetectorGeometry`.

Doing these *with* the framework rather than after keeps the diff coherent and
prevents a second migration.

---

## 2. Workstreams

### W1 — Collapse the ensemble integrators (highest payoff)

One reduction operator replaces all hand-rolled incoherent sums:

- `integrate_over_orientation` ([distributions.py:541](src/rheedium/types/distributions.py#L541))
  → thin wrapper over `apply`.
- `grains.grain_distribution_average`, `grains.apply_misorientation_distribution`
  ([procs/grains.py](src/rheedium/procs/grains.py)) → producers + `apply`.
- `surface_modifier.incoherent_domain_average` → `apply`.
- `ewald_simulator_with_orientation_distribution`
  ([simulator.py:948](src/rheedium/simul/simulator.py#L948)) and the nested
  orientation `vmap`s inside `simulate_detector_image`
  ([simulator.py:1420-1500](src/rheedium/simul/simulator.py#L1420)) → an
  `OrientationDistribution` passed to Layer 1.
- `instrument_broadened_pattern`'s angular+energy quadrature
  ([beam_averaging.py:409](src/rheedium/simul/beam_averaging.py#L409)) → a
  `BeamModeDistribution`.

Net: ~5 bespoke averaging code paths → 1. Each call site becomes "build a
`Distribution`, call `apply`."

### W2 — Wire or delete dangling code

- `coherence_envelope` ([beam_averaging.py:225](src/rheedium/simul/beam_averaging.py#L225))
  — never called; superseded by explicit beam-mode spread. **Delete** (or demote
  to a documented single-mode shortcut). Removing it also closes the
  double-counting risk.
- `SizeDistribution` ([distributions.py:150](src/rheedium/types/distributions.py#L150))
  — type with no consumer. **Wire** to `finite_domain` rod broadening as a
  size producer, or delete if the producer subsumes it.

### W3 — Parameter-object rationalization (biggest readability win)

`simulate_detector_image` takes **24 arguments**
([simulator.py:1273](src/rheedium/simul/simulator.py#L1273)); the seven
`*_sweep` functions ([sweeps.py](src/rheedium/simul/sweeps.py)) each duplicate
the full passthrough; the `generate_*_sweeps` scripts thread ~20-key settings
dicts. Bundle into PyTree carriers (all `eqx.Module`, all differentiable):

- `DetectorGeometry` — `image_shape_px`, `pixel_size_mm`, `beam_center_px`,
  `detector_distance_mm` (also the W4 unifier).
- `BeamSpec` — energy, divergence, spread, coherence (reuse/extend the existing
  `ElectronBeam` rather than inventing a parallel type).
- `SurfaceCTRParams` — `ctr_regularization`, `ctr_power`, `roughness_power`,
  `surface_roughness`.
- `RenderParams` — `spot_sigma_px`, `psf_sigma_pixels`, `render_ctrs_as_streaks`.

Signature collapses to roughly
`simulate_detector_image(crystal, detector, beam, distribution=TRIVIAL,
ctr=..., render=...)` — ~6 args. Keeps `noqa: PLR0913` suppressions from
spreading and makes the public API legible. **Honors the principle**: these are
inert data carriers, not reductions.

### W4 — Unify detector projection

`project_on_detector`, `project_on_detector_geometry`, `detector_extent_mm`,
`render_pattern_to_image`, `_render_ctr_streaks_to_image`
([simulator.py:117-1196](src/rheedium/simul/simulator.py#L117)) carry
overlapping geometry logic. Route all through the single `DetectorGeometry`
carrier so the kinematic and (future) multislice kernels cannot drift, and the
intensity/amplitude render variants share one coordinate mapping.

### W5 — `procs` return-type split

Per the framework's §4.2, make the contract explicit and consistent:

- **Structure builders** (`surface_builder.*`, `library.*`) → `CrystalStructure`.
- **Statistical/defect modifiers** (`grains`, twin/step/texture producers) →
  `Distribution`.
- **Sub-coherence disorder** (`crystal_defects.apply_*_field`) → structure
  modifiers (analytic coherent-average / VCA limit), unchanged.

Document the trichotomy in each `procs` module's `Notes`.

### W6 — Sweeps collapse

With W3's config objects + `distribute_batched`, the seven `simulate_detector_image_*_sweep`
functions ([sweeps.py](src/rheedium/simul/sweeps.py)) — each ~60 lines of
passthrough — reduce to one or two generic helpers: a parameter axis (or a
`Distribution`) + the bundled config. The `generate_*_sweeps.py` scripts and
their hand-rolled chunking ([generate_bi2se3_sweeps.py:19](tutorials/sweeps/generate_bi2se3_sweeps.py#L19))
simplify to a config + `distribute_batched` call.

### W7 — Naming & units consistency pass

- Energy: `voltage_kv` (simulator) vs `energy_kev` (`ElectronBeam`) — pick one,
  alias the other through deprecation.
- Angles: `theta_deg`/`phi_deg` (public) vs `polar`/`azimuth_rad` (internal) —
  standardize the boundary conversion in one place.
- Keep scientific single-letter symbols (`G`, `T`, `dE_E`) where they mirror the
  physics, per CONTRIBUTING.

### W8 — Module reorganization

- `types/distributions.py` (713 lines + the new `Distribution` base + beam
  modes) → split into a `types/distributions/` subpackage (base, orientation,
  size, beam, defects) with a re-exporting `__init__`.
- `simul`: separate the amplitude kernels (Layer 0) from the integrator
  (Layer 1) into clearly named modules so the layering is visible in the tree.

---

## 3. Deprecation & compatibility policy

`rheedium` is **published on PyPI** (CalVer, currently 2026.6.x). Public API
changes need a migration path:

- Keep removed/renamed public symbols as **thin deprecated shims** for at least
  one CalVer cycle, emitting `DeprecationWarning` with the replacement.
- The framework's `simulate_detector_image_instrument(...)` wrapper already
  preserves the legacy default-broadened behavior — old call sites migrate by
  name, not by rewrite.
- Record every break + migration in `CHANGELOG.md` (and the
  `Backwards Compatibility` section of CONTRIBUTING).
- Tutorials and `generate_*_sweeps` scripts updated in the same PR that lands
  each workstream, so the canonical examples never lag the API.

---

## 4. Sequencing (interleave with framework phases)

| After framework phase | Refactor workstream |
|-----------------------|---------------------|
| Phase 1 (Layer 0+1, trivial default) | W4 (detector unify), W3 start (`DetectorGeometry`) |
| Phase 2 (retrofit orientation/size) | W1 (collapse integrators), W2 (dangling code) |
| Phase 3 (beam modes) | W3 finish (`BeamSpec`), W7 (naming) |
| Phase 4 (defect producers) | W5 (`procs` split), W6 (sweeps collapse) |
| Any time | W8 (module reorg) — mechanical, low-risk |

Each row is an independently reviewable PR that leaves the suite green.

---

## 5. Guardrails (the principle, operationalized)

Every refactor PR must demonstrate:

1. **Differentiability preserved** — a `jax.grad` smoke test through the touched
   path returns finite gradients (add to the PR's tests if not already covered).
2. **No new premature reduction** — no `|·|²`, hard threshold, discrete swap, or
   data-dependent Python branch introduced inside a kernel or carrier. Reviewer
   checks this explicitly (it is the silent failure mode).
3. **Results unchanged or migrated** — numerical regression vs the pre-refactor
   output (`chex.assert_trees_all_close`), or a documented deprecation.
4. **Net complexity down** — LOC and/or argument-count reduced, or a dangling
   symbol removed; `pygount` LOC and `interrogate` coverage tracked.

---

## 6. Risks

- **Big-bang temptation.** Doing all workstreams at once produces an
  unreviewable diff. Mitigation: the §4 per-phase sequencing; one workstream per
  PR.
- **Hidden behavioral coupling.** The 24-arg signature hides which params
  interact (e.g. `surface_roughness` feeds both CTR and roughness damping).
  Mitigation: W3 introduces carriers *additively* first (accept both old kwargs
  and new objects), then removes kwargs after call sites migrate.
- **Differentiability regressions from "cleanup".** Replacing a `jnp.where` with
  a Python `if` during tidy-up silently welds a seam. Mitigation: guardrail #1/#2
  are mandatory review items, not optional.
- **PyPI users on the old API.** Mitigation: §3 deprecation shims + CHANGELOG;
  no silent renames.

---

## 7. Outcome

When complete: one amplitude contract, one `Distribution` reduction, one detector
geometry, one set of config carriers, and a `procs` layer with a clear
three-way return contract — with the dangling orphans gone and the seven sweep
clones collapsed. Fewer lines, far fewer arguments, no orphan code, and the
invertibility that motivated the framework preserved at every seam.
