# Codebase Rationalization & Refactor Plan

Scope: `rheedium` — after the
[distribution_framework_plan.md](plans/partial/distribution_framework_plan.md) lands,
harvest the simplification it makes possible and pay down the structural debt it
exposes. This is **not** gratuitous churn: every item either deletes redundancy,
wires a dangling piece, or collapses a duplicated path — and every item must
**preserve the Core Principle** (CONTRIBUTING.md → *Invertible Modularity*): no
refactor may weld a differentiable seam shut.

Status: **proposed** — gated, sequenced **strictly after** the
distribution-framework plan completes (no longer interleaved). Entry gate **R0**:
the framework's six phases have landed and the suite is green. Assumes the
three-layer architecture (amplitude kernels → `Distribution` integrator →
producers) is fully in place. **Roadmap position:** second of four — framework →
*this* → [recon (optax inversion)](plans/future/recon_optax_plan.md) →
[automatons](plans/future/automatons_plan.md). The recon solver and the
automatons are both written against the rationalized API this plan produces, so
neither starts until it completes.

Guard on every workstream below: **tests stay green, public results unchanged
(or migrated behind a deprecation), and `jax.grad` still flows end-to-end.** A
refactor that breaks differentiability has failed even if tests pass.

---

## 1. Why — and why after the framework

The framework turns four structural facts into cleanup opportunities:

1. **Four near-identical ensemble integrators** become one `apply`/`apply_all`.
2. **Dangling code** (`coherence_envelope`, `SizeDistribution`) gets wired or
   removed — no more orphans.
3. **24-argument signatures** (`simulate_detector_image` and its seven sweep
   clones) collapse once parameters are bundled into PyTree config objects.
4. **Two parallel detector-projection paths** unify on one `DetectorGeometry`.

Doing these **after** the framework — not interleaved — is the deliberate
sequencing: the framework is a large, in-flight migration (the simulator is
mid-inversion), and layering structural refactors on a half-migrated `simulate_
detector_image` would be churn-on-churn against a moving baseline. Once the
framework lands, its enablers exist (`apply`/`apply_all`, the `DetectorGeometry`
slot, the instrument wrapper, producers returning `Distribution`s), so each
cleanup below becomes a mechanical harvest against a *stable* tree, reviewable in
isolation.

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

## 4. Gated phases (sequenced after the framework)

The workstreams of §2 are scheduled as a strict gated sequence. Each phase is
one independently reviewable PR that ends at a **gate**: an objectively checkable
set of conditions that must all hold before the next phase starts.

### Entry — Gate R0 (the precondition for *any* work below)

The distribution-framework plan is **complete**: all six framework phases landed,
the suite is green, and `simulate_detector_image` is the Layer-1 integrator with
`kernel=` / `distribution=` first-class. **Reconcile overlap first** — the
framework's own backlog (the unified `DetectorGeometry`, retiring
`coherence_envelope`) may already be done; if so, R1/R3 shrink to verification
rather than work. Until R0 holds this plan does not start.

### Universal gate (applies to *every* phase, on top of its specific gate)

The §5 guardrails, as pass/fail commands:

- `jax.grad` through the touched path returns **finite gradients**;
- **results unchanged**: `chex.assert_trees_all_close` vs the pre-refactor output,
  *or* a documented `DeprecationWarning` shim;
- **no new premature reduction**: no `|·|²`, hard threshold, discrete swap, or
  data-dependent Python branch added inside a kernel or carrier (reviewer-verified
  — the silent failure mode);
- **net complexity down**: `pygount` LOC and/or argument-count reduced, or a
  dangling symbol removed;
- suite green; `ty` / `ruff` clean.

### Phase R1 — Detector unification (foundational)  ·  W4 + `DetectorGeometry` half of W3

*Tasks:* route `project_on_detector`, `project_on_detector_geometry`,
`detector_extent_mm`, `render_pattern_to_image`, `_render_ctr_streaks_to_image`
through one `DetectorGeometry` carrier (the `types/detector.py` the framework
slot-defined); intensity and amplitude render variants share one coordinate map.

**Gate RG1:** kinematic and (future) multislice paths assert *identical* detector
extents from the shared carrier; pixelwise regression vs pre-refactor images;
universal gate.

### Phase R2 — Collapse the ensemble integrators (highest payoff)  ·  W1

*Tasks:* `integrate_over_orientation`, `grains.grain_distribution_average` /
`apply_misorientation_distribution`, `surface_modifier.incoherent_domain_average`,
`ewald_simulator_with_orientation_distribution` + the nested orientation `vmap`s,
and `instrument_broadened_pattern`'s quadrature → all become "build a
`Distribution`, call `apply` / `apply_all`."

**Gate RG2:** each retired path reproduces its pre-refactor output to tolerance;
the count of bespoke averaging functions drops to the one reducer (asserted by an
inventory/grep-guard test); universal gate.

### Phase R3 — Retire dangling code  ·  W2

*Tasks:* delete `coherence_envelope` (or demote to a documented single-mode
shortcut), closing the double-counting risk; confirm `SizeDistribution` is fully
consumed by the framework's `finite_domain` producer and delete any residual dead
path.

**Gate RG3:** no unreferenced public symbol remains in the touched modules
(import-graph check); every removal carries a deprecation shim (§3); universal
gate.

### Phase R4 — Parameter-object rationalization + sweeps collapse  ·  W3 (finish) + W6

*Tasks:* bundle the 24-arg `simulate_detector_image` into `DetectorGeometry`,
`BeamSpec` (extend `ElectronBeam`), `SurfaceCTRParams`, `RenderParams` → a ~6-arg
signature; collapse the seven `*_sweep` clones + `generate_*_sweeps` chunking onto
one or two generic helpers (config + axis/`Distribution` + `distribute_batched`).
Introduce carriers **additively first** (accept old kwargs *and* new objects),
migrate call sites, then deprecate the kwargs.

**Gate RG4:** public signature arg-count drops below the set threshold; the seven
sweep functions reduce to ≤2; the old-kwargs path still works behind a
`DeprecationWarning`; tutorials + `generate_*` scripts updated in the same PR;
universal gate.

### Phase R5 — `procs` return-type split + naming/units  ·  W5 + W7

*Tasks:* make the procs trichotomy explicit and documented — structure builders →
`CrystalStructure`, statistical/defect modifiers → `Distribution`, sub-coherence
disorder → structure modifier; standardize the `voltage_kv`/`energy_kev` and
`theta_deg`/`polar_rad` boundary conversions in one place, aliasing the other
through deprecation.

**Gate RG5:** every `procs` public function's return type matches the trichotomy
(documented in each module's `Notes`); one canonical energy/angle unit at the
public boundary with deprecated aliases; universal gate.

### Phase R6 — Module reorganization (mechanical, low-risk)  ·  W8

*Tasks:* split `types/distributions.py` into a `types/distributions/` subpackage
(base, orientation, size, beam, defects) with a re-exporting `__init__`; separate
the Layer-0 amplitude kernels from the Layer-1 integrator into clearly named
`simul` modules so the layering is visible in the tree.

**Gate RG6:** public import paths unchanged (re-exports preserve
`rheedium.types.*` / `rheedium.simul.*`); a no-op import-parity test; universal
gate.

### Gate summary

| Gate | Pass condition (+ the universal gate) |
|------|----------------------------------------|
| **R0** | framework complete + green; overlap with framework backlog reconciled |
| **RG1** | one `DetectorGeometry`; identical extents; pixelwise regression |
| **RG2** | ~5 averaging paths → 1 reducer; per-path regression |
| **RG3** | no dangling public symbol; deprecation shims for removals |
| **RG4** | ≤6-arg signature, ≤2 sweep fns; old kwargs deprecated, not broken |
| **RG5** | `procs` trichotomy enforced; one unit at the boundary |
| **RG6** | module reorg with unchanged public import paths |

R1–R3 are the structural debt the framework most exposes (do first); R4–R5 are
the readability/ergonomics wins; R6 is cosmetic and may land any time after R0.

---

## 5. Guardrails (the principle, operationalized) — the universal gate

These four are the **universal gate** referenced by every phase in §4; every
refactor PR must demonstrate them in addition to its phase-specific gate:

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
