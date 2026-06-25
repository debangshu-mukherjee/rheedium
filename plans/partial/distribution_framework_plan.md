# Unified Distribution Framework for RHEED Simulation

Scope: `rheedium` — invert the simulation architecture into three layers:
**(0)** base coherent amplitude kernels, **(1)** a single `Distribution`
integrator (the "current simulator"), and **(2)** physics producers (beam
multimodality, orientation, size, twins, steps, grains) that each emit a
`Distribution`. One `Distribution` PyTree, bolted onto any simulation, that
auto-`vmap`s and reduces — incoherently *or* coherently — subsumes
multimodal beams, statistical ensembles, and defects under one differentiable,
parallel contract.

Status: **partially implemented.** Phases 1–3 are substantially in; Phase 4
(defect producers) and Phase 5 (`multislice` kernel selection) are started;
Phase 6 / recon K0 is gated green. The architectural inversion (§1) **has
landed for the kinematic kernel and opt-in multislice** —
`simulate_detector_image` is now the thin Layer-1 integrator (build
distributions → bind selected kernel → `apply_distributions` → PSF/normalize).
Remaining gaps are higher-fidelity defect diffraction, automatic
`PotentialSlices` producers for structure-changing multislice axes, and legacy
Gaussian-quadrature cleanup.

> **Keystone — the critical path.** This is the root of the whole roadmap:
> [rationalization](plans/future/rationalization_refactor_plan.md) R0,
> [recon](plans/future/recon_optimization_plan.md) K0, and
> [automatons](plans/future/automatons_plan.md) A0 all gate transitively on it.
> The single load-bearing item is the **Phase 6 end-to-end differentiability
> guarantee** — `jax.grad` finite through `simulate_detector_image` w.r.t. every
> producer latent — because it is *literally* recon's entry gate **K0**. It is
> **locked today**: grads are live for GSM `β`, twin population fraction, and
> grain/domain size through the public detector integrator. This completes K0 and
> unblocks the downstream chain; it did **not** require the producer-owned
> bind-module split, detector type split, or deeper defect-fidelity work.

### Done — Phase 1 (Layer 0 + Layer 1 core)

- Layer-1 `Distribution` PyTree + `ReductionMode` (COHERENT/INCOHERENT),
  `create_trivial_distribution` / `TRIVIAL_DISTRIBUTION`, and the
  `apply_distribution` / `apply_distributions` reduction + nested-composition
  helpers (`src/rheedium/simul/beam_averaging.py`).
- Layer-0 complex amplitude path: `_ewald_amplitude_pattern` /
  `kinematic_amplitude` (complex Ewald amplitudes), `render_amplitude_to_field`,
  and `render_ctr_amplitude_to_field` for complex CTR streaks
  (`src/rheedium/simul/simulator.py`).
- `simulate_detector_image(..., kernel="kinematic" | "multislice")` — the first
  public Layer-0 kernel selector. Multislice requires explicit
  `potential_slices`; kinematic uses `crystal` directly.
- Generic `distribution=` routes through a **central kinematic bind registry**
  rather than assuming every sample is an azimuth: beam-like, orientation,
  trivial, grain-orientation, twin-wall, and step-edge axes have explicit binds;
  `size` and grain-size axes bind finite-domain rod broadening into the detector
  integrator.

### Done — Phase 2 (retrofit existing producers)

- Adapters `orientation_to_distribution` / `size_to_distribution`;
  `integrate_over_orientation` is now a thin `apply_distribution` wrapper; the
  spot-rendered `simulate_detector_image(..., distribution=...)` Layer-1 entry
  point.
- `SizeDistribution` is wired into finite-domain physics through
  `finite_domain_intensities_for_size_distribution`, which uses the generic size
  producer and incoherently averages per-size rod-overlap intensities.
- The no-distribution path now uses the same Layer-1 reducer for both the trivial
  identity case and the legacy public instrument widths — converting
  `angular_divergence_mrad` / `energy_spread_ev` into an incoherent `Distribution`
  over `(delta_theta_rad, delta_phi_rad, delta_energy_ev)` instead of the old
  angular+energy broadening helper (spot or CTR default images).

### Done — Phase 3 (beam modes, GSM)

- `BeamModeDistribution`, `create_gaussian_schell_beam`, `create_coherent_beam`,
  `decompose_beam_modes`, and `decompose_beam_modes_static` emit a generic
  incoherent `Distribution` over `(delta_theta_rad, delta_phi_rad,
  delta_energy_ev)` with variance-matched anisotropic GSM samples.
- `beam_modes_from_electron_beam`, `create_field_emission_beam`, and
  `create_thermionic_beam` bridge the existing `ElectronBeam` metadata and source
  presets to the GSM producer.
- `simulate_detector_image_instrument` consumes beam modes through the Layer-1
  reducer; `simulate_detector_image(..., beam_modes=...)` exposes the explicit
  path; beam modes and orientation compose through `apply_distributions` in
  `simulate_detector_image(..., beam_modes=..., orientation_distribution=...)`;
  both generic `distribution=` and explicit `beam_modes=` can render CTR streaks.

### Started — Phase 4 (defect producers)

- `grain_population_to_distribution` converts grain orientation / size / fraction
  metadata into an incoherent generic `Distribution` (tested to match
  `grain_distribution_average` mixture semantics).
- `reduction_mode_from_coherence_length` — the first static coherence-threshold
  reducer.
- `twin_wall_to_distribution` / `step_edge_to_distribution` convert twin-wall and
  step-edge metadata into `Distribution`s, with coherent/incoherent reduction
  selected from feature size, coherence length, and regular-vs-random step
  semantics.
- `apply_twin_wall_field` / `apply_step_edge_field` are bound by
  `bind_twin_wall_distribution` / `bind_step_edge_distribution`, so twin/step
  samples build modified `CrystalStructure`s inside a Layer-1 amplitude closure;
  the public detector-image path has end-to-end tests for twin, step, and grain
  binds.

### Started — Phase 5 (multislice slot)

- `multislice_amplitude` returns `FFT(exit_wave)` before modulus-squared, and
  `multislice_simulator` consumes that amplitude before its legacy sparse-pattern
  intensity reduction.
- `multislice_detector_amplitude` projects that complex reciprocal-space field
  onto the dense detector field, and `simulate_detector_image(...,
  kernel="multislice", potential_slices=...)` routes it through the same
  Distribution reducer, detector PSF, and normalization as the kinematic kernel.
  Beam-like and orientation-style axes bind to multislice; structure-changing
  axes fail loudly until a `PotentialSlices` producer exists for them.

### Tests & exports

All of the above is exported through `rheedium.types` / `rheedium.simul` and
covered by tests: distribution validation, reduction algebra, composition,
amplitude parity, sparse relative phase, real-kernel coherent interference,
trivial→intensity, simulator distribution identity / manual Layer-1 parity,
defect detector-image distinguishability, multislice public-kernel selection and
direct-field parity, size-distribution finite-domain parity, beam-mode
normalization / variance / coherent-limit, ElectronBeam/preset bridge,
instrument-wrapper Layer-1 parity, main-simulator beam-mode parity,
beam×orientation composition parity, and CTR amplitude-renderer parity.

### Not yet done

- Add automatic `CrystalStructure` / defect-sample → `PotentialSlices` producer
  binds so structure-changing axes can run under `kernel="multislice"` instead
  of requiring precomputed `PotentialSlices`.
- Retire the remaining orientation+CTR angular+energy Gaussian quadrature path
  in the rationalization track rather than duplicating that cleanup here.

### Remaining work to completion

The keystone (**Phase 6 / recon K0**) is **green**, so the roadmap is already
unblocked. What remains to mark *this plan* complete is small and well-scoped —
two framework tasks plus one delegated cleanup:

- **F1 — Multislice producer polymorphism** *(the only substantive item)*. Add the
  `CrystalStructure` / defect-sample → `PotentialSlices` producer bind so
  structure-changing axes (twins, steps, grain morphology, size) run under
  `kernel="multislice"` instead of raising
  ([distribution_binds.py:178](src/rheedium/procs/distribution_binds.py#L178)).
  **Gate FG1:** each structure-changing axis produces a *distinguishable* and
  *differentiable* multislice detector image (no raise), mirroring the kinematic
  distinguishability + grad tests. *(Medium–Large.)*
- **F2 — Detector-contract verification** *(mostly done)*. `DetectorGeometry` is
  split into `types/detector.py` and both kernels already project through
  `project_on_detector_geometry` (§2.4). What remains is the **standing
  regression**: **Gate FG2** — a test asserting kinematic and multislice yield
  *identical* detector extents from the shared carrier (so they cannot drift).
  Tilted/curved dense rendering is explicitly a future geometry-depth item, not a
  gate. *(Small — a test, not a refactor.)*
- **Delegated — legacy quadrature.** Retiring `instrument_broadened_pattern` /
  `gauss_hermite_nodes_weights` is owned by
  [rationalization R2](plans/future/rationalization_refactor_plan.md); not a
  framework task.

**Definition of done:** FG1 + FG2 green (K0 already is); the legacy quadrature is
retired by rationalization. Higher-fidelity defect physics is explicitly **out of
scope** — see
[defect_diffraction_fidelity_plan.md](plans/future/defect_diffraction_fidelity_plan.md).

### Housekeeping

- `coherence_envelope` has been removed from `beam_averaging` and the public
  `rheedium.simul` export surface.
- Moved from `plans/future/` to `plans/partial/` on landing the Phase-1 slice.

### Relationship to other plans

- **Subsumes** the earlier mixed-state beam decomposition: its GSM mode math,
  broadening taxonomy, and beam/sample ensemble now live here as producers (§4.1)
  and Layer-1 mechanics (§3, §5).
- **Builds on**
  [parallel_sweeps_plan.md](plans/implemented/parallel_sweeps_plan.md): the product
  ensemble is exactly the
  [`tools.distribute_batched`](src/rheedium/tools/parallel.py) batch axis.
- **Roadmap position** — first of four, each downstream plan gated on this one
  completing:
  1. *this* (distribution framework)
  2. [rationalization refactor](plans/future/rationalization_refactor_plan.md)
  3. [recon (inversion)](plans/future/recon_optimization_plan.md)
  4. [automatons](plans/future/automatons_plan.md)
- **Physics-depth follow-up (not a gate)** —
  [defect_diffraction_fidelity_plan.md](plans/future/defect_diffraction_fidelity_plan.md)
  takes the deferred high-fidelity defect/coherence physics (fine-twin satellites,
  step-terrace splitting, displacement-fringe validation). It builds on this
  plan's coherent reduction but **nothing on the gated chain depends on it**, so it
  can land in parallel or later.

### Decisions locked

1. **(a) Default is *trivial-sharp*** — a single coherent pattern, no hidden
   broadening; the **kinematic** amplitude kernel ships in v1 with **multislice**
   as a defined Layer-0 slot.
2. **(b) Forward-only ownership** — this plan owns the *forward* model and its
   differentiability **only**. Every inverse / reconstruction capability (fitting
   latents, reconstructing probability distributions, recipe-deviation,
   uncertainty) lives in the **`recon`** module, specified by
   [recon_optimization_plan.md](plans/future/recon_optimization_plan.md); the
   framework's sole inverse-related obligation is to keep `jax.grad` flowing
   end-to-end (Phase 6).

---

## 1. Architectural inversion (the forward model)

**Disambiguation.** "Inversion" in this plan means *flipping the forward
architecture* — pushing the modulus-squared down to Layer 0 so kernels return
amplitude, and collapsing the simulator into a thin `apply(distribution, kernel)`
integrator. It does **not** mean the inverse problem (recovering parameters from a
pattern); that — fitting, reconstruction, recipe-deviation, UQ — lives in
[`recon`](plans/future/recon_optimization_plan.md), per decision (b).

Today `simulate_detector_image` bakes the modulus-squared into the sparse
reflection list, then scatters instrument effects (`n_angular_samples`,
`n_energy_samples`, `orientation_distribution`) through nested ad-hoc `vmap`s.
The former orphan `coherence_envelope` has been removed rather than migrated.
Statistical physics lives in four near-identical
but disconnected places — `integrate_over_orientation`, `grains`,
`SizeDistribution` (dangling), and the proposed `BeamModeDistribution`.

Invert it:

```
Layer 0  base coherent kernels — always complex, return AMPLITUDE on a shared
         detector field:
             kinematic_amplitude(crystal, geom, E)  -> Complex[H, W]
             multislice_amplitude(crystal, geom, E) -> Complex[H, W]   (slot)

Layer 1  the integrator (== the current simulator):
             simulate_detector_image(crystal, ..., distribution=TRIVIAL,
                                     kernel=kinematic)
               -> apply(distribution, bound_kernel)   # vmap + reduce
               -> |·|² / weighted sum per reduction
               -> detector PSF -> normalize -> intensity

Layer 2  physics producers -> Distribution (or a composition of them):
             beam modes | orientation | size | twins | steps | grains
```

**Identity element.** The default `Distribution` is *unitary*: one sample, the
identity perturbation, weight 1.0. At `N=1` the coherent and incoherent
reductions coincide (`|1·A|² = 1·|A|²`), so the default unambiguously reproduces
a single coherent pattern. Richer physics swaps in a richer `Distribution`;
Layer 1 never changes.

---

## 2. Layer 0 — base coherent amplitude kernels

Both kernels are **already complex internally**; this layer exposes a stage that
exists by removing a premature `|·|²`.

### 2.1 `kinematic_amplitude`

- `compute_kinematic_intensities_with_ctrs`
  ([simulator.py:334](src/rheedium/simul/simulator.py#L334)) computes
  `Complex[M]` per-reflection contributions and *then* squares
  ([simulator.py:498](src/rheedium/simul/simulator.py#L498)). Expose the complex
  field: return amplitude + phase per reflection, not intensity.
- `ewald_simulator` gains/【or is mirrored by】a complex sibling returning a
  `RHEEDPattern`-like structure carrying **complex** per-reflection amplitude.
- Current status: `_ewald_amplitude_pattern` mirrors the sparse Ewald geometry,
  carries the complex structure factor through CTR and roughness amplitude
  scaling, and normalizes amplitudes so `|A|²` matches `ewald_simulator`
  intensities. `kinematic_amplitude` now renders these complex sparse
  amplitudes directly; tests assert both intensity parity and non-trivial
  imaginary phase.

### 2.2 Complex render-to-field (the one genuinely new kernel)

`render_pattern_to_image` ([simulator.py:1048](src/rheedium/simul/simulator.py#L1048))
splats *intensities*. Add `render_amplitude_to_field` that deposits
`A·exp(iφ)` onto the dense `H×W` **complex** grid, so that coherently-summed
samples interfere only where reflections overlap (physically correct: large
relative rotations don't interfere; sub-coherence displacements do). A matching
complex variant of `_render_ctr_streaks_to_image` for streak amplitudes.

Relative phase is what matters; fix a single phase origin (beam/sample origin)
shared by all samples so cross-sample phases are consistent.

### 2.3 `multislice_amplitude` (Layer-0 slot)

`multislice_propagate` already carries a complex exit wavefunction; the pattern
is `|FFT(ψ)|²`. `multislice_amplitude` returns the complex diffraction field
before the modulus. This slot is now implemented as `FFT(exit_wave)` and is
used by both `multislice_simulator` before that legacy path applies `|·|²` and
`multislice_detector_amplitude`, which projects the complex reciprocal-space
grid to the dense detector field. `simulate_detector_image(kernel="multislice",
potential_slices=...)` now selects this path. The remaining gap is not kernel
selection; it is producer binding for structure-changing multislice samples,
because those need a real `CrystalStructure` / defect sample →
`PotentialSlices` conversion rather than a direct `CrystalStructure` bind.

### 2.4 Shared detector contract

Both kernels now render onto the same dense `(H, W, pixel_size_mm,
beam_center_px)` field so Layer 1 can reduce either selected kernel. Kinematic
maps `k_out → detector` via Ewald geometry; multislice maps a `q`-grid →
detector and rasterizes the complex field with the same amplitude renderer.
`DetectorGeometry` lives in `types/detector.py`; kinematic and multislice sparse
pattern builders both project through `project_on_detector_geometry`, and the
public detector-image path binds distance and PSF through that carrier for both
selected kernels. Full tilted/curved dense-image rendering remains a future
geometry-depth improvement rather than a framework gate.

---

## 3. Layer 1 — the `Distribution` integrator

### 3.1 Base type

```python
class Distribution(eqx.Module):
    samples: Float[Array, "N ..."]     # latent parameter values per sample
    weights: Float[Array, "N"]         # real, >= 0
    reduction: ReductionMode           # static: COHERENT | INCOHERENT
    axis_id: Optional[str] = eqx.field(static=True, default=None)
```

- **`reduction` is the §6 switch.** It may be *computed* (not hardcoded) by
  comparing the axis's characteristic length to the beam per-mode coherence
  length `ℓ_c` — making the coherent/incoherent partition physics, not a user
  choice. The first implementation is
  `reduction_mode_from_coherence_length(feature_length_angstrom,
  coherence_length_angstrom)`, which returns a static `ReductionMode` for
  producers that know their characteristic length eagerly.
- `TRIVIAL = Distribution(samples=zeros(1, …), weights=ones(1),
  reduction=INCOHERENT)` — the identity element.
- Concrete distributions (Orientation, Size, BeamMode, TwinWall, StepEdge,
  Grain) are factories that emit this base type, or thin subclasses adding
  static metadata + a `.discretize()` that yields `(samples, weights)`.

### 3.2 The bind/closure contract

A sample is not always a scalar the kernel eats directly. Beam modes perturb
`(θ,φ,E)`; orientation perturbs `φ`; size perturbs rod width; **twins/steps need
a modified `CrystalStructure` per sample** (a `procs` builder runs *inside* the
closure). So a producer supplies a **bound closure** mapping a sample to a kernel
call:

```python
dist.bind(kernel, crystal, geom) -> (sample -> Complex[H, W])
```

This generalizes exactly what `integrate_over_orientation`'s `simulate_fn`
already is ([distributions.py:541](src/rheedium/types/distributions.py#L541)).
Current implementation status: the reducer is generic, and `Distribution.bind`
now owns the public entry point for turning one distribution axis into a bound
sample closure. Concrete axis semantics live in
`rheedium.procs.distribution_binds`, which returns kernel-local update records
for kinematic and multislice detector kernels. Those producer-owned binders
prevent silent mis-binds and have public-path coverage for beam-like,
orientation/trivial, grain-orientation, grain-size, size, twin, and step axes.

### 3.3 Reduction and composition

```python
def apply(dist, bound_amp_fn) -> Float[Array, "H W"]:
    A = jax.vmap(bound_amp_fn)(dist.samples)          # Complex[N, H, W]
    if dist.reduction is COHERENT:
        return jnp.abs(jnp.einsum("n,nhw->hw", dist.weights, A)) ** 2
    return jnp.einsum("n,nhw->hw", dist.weights, jnp.abs(A) ** 2)
```

Coherent = sum amplitudes **then** square; incoherent = square **then** sum.
For multiple bolted-on distributions the reduction **nests** — coherent axes
collapse *inside* the modulus, incoherent axes *outside*:

```
I = Σ_incoh w_k | Σ_coh c_j A(sample_j, sample_k) |²
```

`apply_all([d1, d2, …], bound)` partitions axes by `reduction`, runs the
coherent product inside `|·|²`, the incoherent product outside.

### 3.4 The integrator == current simulator

`simulate_detector_image` keeps its role but its body becomes:

1. normalize public inputs into ordered `Distribution` axes,
2. bind the selected coherent kernel to the composed sample contract,
3. `image = apply_all(distributions, bound)`,
4. `detector_psf_convolve` + normalize.

Current status: the public path follows this spine for the kinematic kernel
(default instrument widths, explicit beam modes, orientations, generic
distributions, grains, twins, and steps) and for the opt-in multislice kernel
when `potential_slices` are supplied. Each axis is bound through
`Distribution.bind(...)`; concrete per-axis semantics are producer-owned in
`rheedium.procs.distribution_binds` and return kernel-local update records.

> **Caveat — kernel-agnostic Layer 1 is now real but incomplete.** The same
> reducer/PSF spine can select `"kinematic"` or `"multislice"`; multislice shares
> beam-like and orientation-style binds and has public direct-field parity tests.
> What is still incomplete is producer polymorphism: structure-changing axes
> (`twins`, `steps`, grain morphology, and `size`) cannot yet generate or mutate
> `PotentialSlices`, so they intentionally fail under `kernel="multislice"`
> instead of silently mis-binding.

`distribution=TRIVIAL` ⇒ a single coherent pattern's intensity — the (a)
default. **No** `coherence_envelope` here: partial coherence is produced
explicitly by beam-mode samples (see §5).

### 3.5 Parallelism

The full product ensemble (beam ⊗ orientation ⊗ size ⊗ defect ⊗ energy) is the
`distribute_batched` batch axis. Large or high-dimensional ensembles use
quasi-Monte-Carlo (Sobol) sampling of the joint distribution with per-axis
stratification instead of a dense tensor product; each joint sample is one
coherent kernel call. No new parallel machinery.

---

## 4. Layer 2 — physics producers

Each returns a `Distribution` (+ `bind`). Existing scattered machinery is
retrofitted onto the base type; nothing is reinvented.

| Producer | Sample (latent) | Reduction | Reuses |
|----------|-----------------|-----------|--------|
| **Beam modes** (GSM) | `(δθ, δφ, δE)` tilts/energy | incoherent (occupations `λ_n`) | new `decompose_beam_modes` |
| **Orientation** | `δφ` azimuth | incoherent | retrofit `OrientationDistribution` |
| **Size / finite domain** | domain size `L` | incoherent (rod width) | wire dangling `SizeDistribution` → `finite_domain` |
| **Twin walls** | twin angle / wall positions | **computed** by `L` vs `ℓ_c` | `OrientationDistribution` (coarse) / builder (fine → satellites) |
| **Step edges** | `(h, terrace width, line azimuth)` | computed; regular→coherent, random→incoherent | `apply_step_edge_field`, `vicinal_surface_step_splitting` |
| **Grains** | orientation + size per grain | incoherent | `grains.grain_distribution_average` |
| **Sub-coherence disorder** (vacancy/DW/displacement) | — | analytic coherent-average (VCA) | `crystal_defects.*` — stays a structure modifier |

### 4.1 Beam modes (Gaussian Schell-model)

The beam producer is the most physics-heavy and is the authoritative home for
the mixed-state treatment. A real electron beam is a **partially coherent
source** — a statistical mixture of coherent wavefields, i.e. a mixed quantum
state with density operator `ρ = Σ_n λ_n |ψ_n⟩⟨ψ_n|`, where `{ψ_n}` are
orthogonal **coherent modes** and `λ_n ≥ 0`, `Σ λ_n = 1` their occupations (Wolf
coherent-mode decomposition; Starikov & Wolf 1982). The measured pattern is the
**incoherent sum of per-mode coherent intensities**:

```
I(detector) = Σ_n λ_n | A[ψ_n] |²
```

where `A` is the coherent RHEED amplitude kernel (one Layer-0 call). In this
framework that is exactly `apply(beam_dist, bound_kernel)` with
`reduction = INCOHERENT`; the producer's only job is to emit
`(samples, weights) = ((δθ, δφ, δE)_n, λ_n)`. The mode count is set by the
source's phase-space volume / emittance — i.e. how incoherent it is.

**Why RHEED needs many modes.** For a Gaussian Schell-model (GSM) source — the
standard analytic model of partial coherence — the eigenvalues form a geometric
series `λ_n = (1−β) βⁿ` with decay ratio `β ∈ (0,1)` fixed by the ratio of
coherence length to source size. The *effective* mode count is the participation
ratio `N_eff ≈ 1/(1−β)`:

- **Cold/Schottky FEG** (ΔE ≈ 0.3–0.7 eV, reduced brightness ~10⁸–10⁹): `β`
  small, `N_eff` a few — yet mixed-state ptychography still carries 8–10 modes.
- **Thermionic W/LaB₆ (RHEED)** (ΔE ≈ 1–3 eV, brightness ~10⁴–10⁶): `β` near 1,
  `N_eff` large. So 8–10 modes is a **floor**, not a ceiling.

**RHEED anisotropy.** Grazing incidence at angle θ foreshortens the footprint by
`1/sin θ` (≈20–30× at 2°), so the projected source size — and thus `β`, `N_eff`,
and the modal angular widths — differ sharply **in-plane vs out-of-plane** of
scattering. The mode structure is intrinsically anisotropic in a way TEM's is
not; **azimuth varies** (today's `instrument_broadened_pattern` holds it at the
nominal value, giving no out-of-plane divergence and no anisotropy).

#### `BeamModeDistribution` producer

An `eqx.Module` storing the **physical** GSM parameters (modes derived on
demand, exactly as `OrientationDistribution` stores peaks and derives
quadrature):

```python
class BeamModeDistribution(eqx.Module):
    beta_in_plane: Float[Array, ""]           # GSM decay ratio, scattering plane
    beta_out_of_plane: Float[Array, ""]       # perpendicular
    divergence_in_plane_rad: Float[Array, ""]       # 1σ angular divergence/axis
    divergence_out_of_plane_rad: Float[Array, ""]
    energy_spread_ev: Float[Array, ""]        # longitudinal mixed state
    distribution_id: Optional[str] = eqx.field(static=True, default=None)
```

Parameterizing by `(β, total divergence)` rather than raw µm/wavelength (a) sits
directly on the existing `(θ, φ, E)` kernel interface, (b) reduces exactly to
today's Gaussian model in the incoherent limit, and (c) keeps the absolute
source-size → wavelength → grazing-projection chain as an optional second
parameterization (the `ElectronBeam` bridge). `β` carries the coherence; total
divergence carries the scale. Validate with `eqx.error_if` (β∈[0,1),
divergences ≥ 0, spread ≥ 0) per the two-tier validation pattern.

Factories (parallel to `create_*_orientation`): `create_gaussian_schell_beam`
(general anisotropic GSM), `create_coherent_beam` (β→0, single transverse mode),
`create_thermionic_beam` and `create_field_emission_beam` (presets setting
`β`/anisotropy from gun type + incidence angle — W/LaB₆ vs Schottky), and
`beam_modes_from_electron_beam(beam, θ)` — the bridge wiring the dormant
`ElectronBeam` coherence fields
([beam_types.py:101](src/rheedium/types/beam_types.py#L101)) into the simulator
at last.

#### `decompose_beam_modes` — the producer's `.discretize()`

Analytic, fully differentiable; yields the `Distribution` `(samples, weights)`:

1. **GSM eigenvalues per transverse axis.** `λ_n = (1−β)βⁿ`, `n = 0…N−1`,
   truncated when cumulative mass ≥ `1 − weight_tol` (with an `n_modes_per_axis`
   cap) — the heavy tail the Gaussian quadrature lacks.
2. **Modal angular offsets.** Mode `n` of a GSM is a Hermite–Gauss function
   whose angular-spectrum width grows with `n`. Place each axis's samples at
   Hermite–Gauss-like nodes scaled so the **occupation-weighted variance equals
   the specified total divergence²** — the constraint that makes the expansion
   reduce to the Gaussian model (incoherent limit) and to a single sharp beam
   (coherent limit).
3. **2D anisotropic tensor product.** Outer-product the in-plane and
   out-of-plane mode sets → `(δθ, δφ)` offsets with weights `λ_n·λ_m`. This is
   where azimuth finally varies and where anisotropy enters.
4. **Longitudinal (energy) modes.** Energy spread is genuinely incoherent; keep
   Gauss–Hermite energy nodes (today's treatment is correct) folded in as `δE_k`
   with weights `w_k`, reframed as longitudinal modes for unification.
5. **Flatten** `(transverse ⊗ energy)` → `N` samples; renormalize weights to 1.

Provide `decompose_beam_modes_static` (Python-branch truncation outside JIT),
mirroring `discretize_orientation_static` — the tolerance-pruned eager path vs
the fixed-N JIT path (§9).

The incoherent mode sum is rigorous in the far-field kinematic limit:
mutually-incoherent source coherence-cells illuminate the crystal as coherent
tilted plane-wave bundles whose detector intensities add — so each mode is one
coherent Layer-0 call and the modes sum with `λ_n`. Energy spread is a
longitudinal incoherent sub-axis. This composes cleanly with the orientation and
defect producers: modes × orientations is the nested incoherent sum of §3.3.
`simulate_detector_image(..., beam_modes=..., orientation_distribution=...)`
now exercises this composition path directly for the kinematic amplitude kernel.

### 4.2 procs returns Distributions, not CrystalStructures — with one split

- **Pure structure builders** (`create_surface_slab`,
  `apply_surface_reconstruction`, 7×7 library) → still return `CrystalStructure`
  (a 7×7 is a different cell, not a distribution over 1×1).
- **Statistical / defect modifiers** (grains, twins, steps, size, texture) →
  return a `Distribution` over latent parameters whose bind closure may call a
  builder per sample. Grain, twin-wall, and step-edge producers now emit generic
  distributions; `simulate_detector_image(..., distribution=...)` binds grain
  orientation samples through the kinematic kernel and twin/step samples through
  modified `CrystalStructure`s. Grain size is still metadata for future
  finite-domain coupling rather than an active detector-image parameter.
- **Sub-coherence disorder** (`apply_vacancy_field` VCA occupancy, Debye–Waller,
  displacement fields) → the **analytic coherent-average limit**; modifies the
  structure factor in closed form, stays a structure modifier. This is the
  `L ≪ ℓ_c` branch done without sampling.

---

## 5. The coherent/incoherent partition is set by the beam

The reduction flag is physics: a sample feature averages coherently or
incoherently depending on whether it is **smaller or larger than a mode's
coherent footprint** (`ℓ_c`, projected by grazing geometry).

- **Larger than `ℓ_c`** (grains, separated islands, distinct domain
  orientations, coarse twins, random steps) → **incoherent** sum.
- **Smaller than `ℓ_c`** (atomic roughness, intermixing, thermal motion, fine
  twins) → **coherent** average → structure-factor modification (Debye–Waller,
  `roughness_damping`, `finite_domain`, fine-twin satellites).

So beam modes are a **prerequisite**: their per-mode `ℓ_c` *defines* each defect
axis's `reduction`. Twins and steps live on both sides depending on density.

### 5.1 Broadening taxonomy — what the framework removes

| Mechanism | Origin | Fate |
|-----------|--------|------|
| Angular-divergence GH quadrature | beam coherence | **Replaced** by beam-mode `Distribution` |
| `coherence_envelope` (retired orphan) | beam coherence | **Removed** from code/export surface (double-counted modal spread) |
| Energy-spread GH quadrature | beam coherence | **Kept** as a longitudinal sub-axis |
| `spot_sigma_px` | numerical | **Demoted** to anti-aliasing |
| Detector PSF | detector | **Kept** — a genuine convolution |
| `finite_domain`, `roughness_damping`, Debye–Waller | sample (sub-`ℓ_c`) | **Kept** — structure factor |

A Gaussian convolution is shift-invariant; real divergence blur is shift-variant
and anisotropic. The modal sum produces position-dependent blur a single kernel
cannot — strictly more expressive, not a re-weighting.

---

## 6. Backward compatibility

- Default (a): `simulate_detector_image(crystal)` with `distribution=TRIVIAL`
  returns a single sharp coherent pattern (matches raw `ewald_simulator`
  intensity, modulo PSF/normalize).
- Today's broadened behavior moves to a thin convenience wrapper
  `simulate_detector_image_instrument(...)` that builds the default instrument
  `BeamModeDistribution` (angular+energy) and calls Layer 1 — so existing
  callers/tutorials keep their results via a one-line shim.
- `ewald_simulator`, the orientation path, CTR rendering, and the diffraction
  math are untouched at the physics level; the intensity sims are re-expressed
  as `apply(TRIVIAL, kinematic_amplitude)` internally.

---

## 7. Phasing

1. **Phase 1 — Layer 0 (kinematic) + Layer 1 + trivial default.**
   `kinematic_amplitude`, `render_amplitude_to_field`, `DetectorGeometry`,
   `Distribution` base + `apply`/`apply_all` + `TRIVIAL`. Prove
   `apply(TRIVIAL, kinematic_amplitude)` reproduces today's single pattern.
2. **Phase 2 — retrofit existing producers.** `OrientationDistribution` and
   `SizeDistribution` onto the base type; `integrate_over_orientation` becomes a
   thin `apply`. `SizeDistribution` and grain-size samples now bind
   finite-domain broadening through the detector integrator.
3. **Phase 3 — beam modes (GSM).** `BeamModeDistribution` producer; instrument
   convenience wrapper; the first coherence-length `reduction` machinery; and
   beam×orientation composition through Layer 1.
4. **Phase 4 — defect producers.** Twins, steps, grains/texture as
   Distributions; the coherent fine-twin/satellite path exercises a coherent
   reduction end-to-end. Grain, twin-wall, and step-edge producers now exist;
   twin/step bind closures now build modified structures per sample. Fine-twin
   satellite physics and higher-fidelity step diffraction are **scoped out** to
   [defect_diffraction_fidelity_plan.md](plans/future/defect_diffraction_fidelity_plan.md)
   (a parallel physics-depth track, not a roadmap gate).
5. **Phase 5 — `multislice_amplitude`** fills the Layer-0 slot. The complex
   amplitude slot exists and is used by `multislice_simulator`; high-level
   detector geometry unification remains.
6. **Phase 6 — differentiability guarantee (the inverse *solver* lives in
   `recon`).** The framework's only inverse-related obligation is to keep the
   forward model differentiable end-to-end w.r.t. every producer latent —
   `jax.grad` through `apply` / `simulate_detector_image` finite and live
   (test 9). Current gates cover public gradients w.r.t. GSM `β`, twin
   population fraction, and grain/domain size. The inverse-problem *solver*
   itself — optax optimization, loss design, multistart, uncertainty,
   distribution reconstruction, recipe-deviation — is **not built here**; it
   belongs to the `recon` module and is specified by
   [recon_optimization_plan.md](plans/future/recon_optimization_plan.md), whose
   entry gate **K0** is exactly this differentiability guarantee.

Phases 1–2 are independently shippable and fully CPU-testable; later phases are
additive.

---

## 8. Testing

Per CONTRIBUTING (`chex`, parameterized, 8-virtual-device harness,
`@jaxtyped(typechecker=beartype)`, `:see:` cross-refs). Backbone tests:

1. **Identity:** `apply(TRIVIAL, kinematic_amplitude)` == legacy
   `simulate_detector_image` single pattern (`chex.assert_trees_all_close`).
2. **Reduction algebra:** `apply` coherent vs incoherent on a hand-built 2-sample
   case matches `|w₀A₀+w₁A₁|²` vs `w₀|A₀|²+w₁|A₁|²` exactly.
3. **N=1 coincidence:** both reductions agree at one sample.
4. **Amplitude kernel:** `|kinematic_amplitude|²` == legacy intensity, pixelwise.
5. **Coherent interference:** real `kinematic_amplitude` fields cancel under a
   coherent opposite-phase axis while the same samples remain bright under
   incoherent reduction. Full displacement-fringe validation is **scoped out** to
   [defect_diffraction_fidelity_plan.md](plans/future/defect_diffraction_fidelity_plan.md).
6. **Beam-mode limits:** β→0 → 1 mode → coherent pattern; incoherent isotropic
   limit reproduces legacy angular-Gaussian broadening (regression guard for
   removing the Gaussian).
7. **Composition:** `apply_all([beam, orientation])` == manual nested reduction.
   Covered both at reducer level and through
   `simulate_detector_image(..., beam_modes=..., orientation_distribution=...)`.
8. **Parallel:** ensemble via `distribute_batched` == serial.
9. **Differentiability:** `jax.grad` of public detector images w.r.t. β, twin
   density, and grain/domain size is finite and non-zero. Multislice amplitude
   gradients through potential scale are covered separately; the inverse solver
   lives in `recon`.
10. **Anisotropy:** in-plane ≠ out-of-plane streak FWHM under anisotropic β.

---

## 9. Risks

- **Amplitude path correctness.** Premature `|·|²` removal must be exact;
  guarded by test 4 (pixelwise equality of `|amplitude|²` vs legacy intensity).
  The sparse helper is additionally guarded against legacy drift by
  `test_ewald_amplitude_pattern_matches_intensity_simulator`, and
  `test_kinematic_amplitude_carries_nontrivial_phase` asserts non-zero relative
  phase between sparse reflections, not only a dense-field imaginary smoke
  signal.
- **Phase-reference consistency** across samples in coherent reduction → fix one
  shared origin; test 5 falsifies a wrong reference.
- **Defect bind no-ops** → public twin, step, and grain distributions must change
  detector images relative to the undefected crystal, not merely match a manual
  binding parity calculation.
- **Kernel detector drift** (kinematic vs multislice) → one `DetectorGeometry`
  carrier; assert identical extents.
- **Double-counting coherence** → `coherence_envelope` is retired; the beam
  `Distribution` is the sole source of partial-coherence broadening in the
  framework path.
- **Ensemble blow-up** → Sobol joint sampling + `distribute_batched`; `log` any
  axis truncated before its tolerance (no silent caps).
- **Static vs traced mode/sample counts** → tolerance-pruned eager path +
  fixed-N JIT path, mirroring `discretize_orientation` vs `_static`.
- **Backward-compat surface** → the instrument convenience wrapper preserves the
  current default-broadened results; tutorials updated to call it.

---

## 10. Diff surface

| File | Change |
|------|--------|
| `src/rheedium/simul/simulator.py` | `kinematic_amplitude`, `render_amplitude_to_field`, `render_ctr_amplitude_to_field`, `multislice_amplitude`, `multislice_detector_amplitude`; `simulate_detector_image` → Layer-1 kernel selector for kinematic and opt-in multislice; consumes producer-owned bind updates; beam×orientation composition; `_instrument` kinematic convenience wrapper |
| `src/rheedium/simul/multislice.py` | Multislice primitives remain here; `multislice_amplitude` currently lives in `simulator.py` beside `multislice_propagate` / `multislice_simulator` |
| `src/rheedium/types/distributions.py` | `Distribution` base + `ReductionMode` + `TRIVIAL`; retrofit `OrientationDistribution`/`SizeDistribution`; `BeamModeDistribution` + GSM factories; `reduction_mode_from_coherence_length` |
| `src/rheedium/types/detector.py` | `DetectorGeometry` carrier and detector helper accessors shared by kinematic and multislice detector paths |
| `src/rheedium/types/rheed_types.py` | RHEED image/pattern/sliced-crystal/surface types; compatibility re-export of `DetectorGeometry` |
| `src/rheedium/simul/beam_averaging.py` | `apply`/`apply_all` integrator + `decompose_beam_modes`; `coherence_envelope` removed; legacy angular/energy quadrature remains outside the default framework path |
| `src/rheedium/procs/{grains,surface_modifier,crystal_defects}.py` | grain/twin-wall/step-edge producers return `Distribution`; twin/step bind helpers build modified structures per sample; sub-coherence disorder remains analytic structure modification |
| `src/rheedium/procs/distribution_binds.py` | Producer-owned kinematic and multislice axis bind semantics for beam/orientation/grain/size/twin/step axes |
| `src/rheedium/{simul,types,procs}/__init__.py` | exports + Routine Listings |
| `tests/.../{test_simulator,test_distributions,test_beam_averaging,test_surface_modifier,test_grains}.py` | identity, reduction-algebra, interference, limits, composition, parallel, grad |
| `tutorials/` | trivial vs instrument default; mode-count convergence; defect-density demo |

The diffraction physics is preserved; the change is structural — one amplitude
contract at the bottom, one `Distribution` reduction in the middle, and every
multimodal/statistical/defect effect expressed as a producer on top.
