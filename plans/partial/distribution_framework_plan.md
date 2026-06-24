# Unified Distribution Framework for RHEED Simulation

Scope: `rheedium` — invert the simulation architecture into three layers:
**(0)** base coherent amplitude kernels, **(1)** a single `Distribution`
integrator (the "current simulator"), and **(2)** physics producers (beam
multimodality, orientation, size, twins, steps, grains) that each emit a
`Distribution`. One `Distribution` PyTree, bolted onto any simulation, that
auto-`vmap`s and reduces — incoherently *or* coherently — subsumes
multimodal beams, statistical ensembles, and defects under one differentiable,
parallel contract.

Status: **partially implemented** — Phases 1–2 landed and Phase 3 foundations
started (commits `264a722`…`6044682`); Phases 4–6 pending. Implemented and tested:
the Layer-1 `Distribution` PyTree + `ReductionMode` (COHERENT/INCOHERENT),
`create_trivial_distribution` / `TRIVIAL_DISTRIBUTION`, the
`apply_distribution` / `apply_distributions` reduction + nested-composition
helpers (`src/rheedium/simul/beam_averaging.py`), the Layer-0
`render_amplitude_to_field` complex renderer and a conservative
`kinematic_amplitude` bridge (`src/rheedium/simul/simulator.py`), Phase-2
*adapters* `orientation_to_distribution` / `size_to_distribution`,
`integrate_over_orientation` as a thin `apply_distribution` wrapper, and the
spot-rendered `simulate_detector_image(..., distribution=...)` Layer-1 entry
point. `SizeDistribution` is now wired into finite-domain physics through
`finite_domain_intensities_for_size_distribution`, which uses the generic size
producer and incoherently averages per-size rod-overlap intensities. The
spot-rendered no-distribution path now uses the same Layer-1 reducer for both
the trivial identity case and the legacy public instrument widths, converting
`angular_divergence_mrad` / `energy_spread_ev` into an incoherent
`Distribution` over `(delta_theta_rad, delta_phi_rad, delta_energy_ev)` rather
than calling the old angular+energy broadening helper. `simulate_detector_image(
..., kernel="kinematic")` exposes the first public Layer-0 kernel selector. Phase 3
beam-mode foundations are started: `BeamModeDistribution`,
`create_gaussian_schell_beam`, `create_coherent_beam`, `decompose_beam_modes`,
and `decompose_beam_modes_static` now emit a generic incoherent `Distribution` over
`(delta_theta_rad, delta_phi_rad, delta_energy_ev)` with variance-matched
anisotropic GSM samples. `beam_modes_from_electron_beam`,
`create_field_emission_beam`, and `create_thermionic_beam` connect the existing
`ElectronBeam` metadata and source presets to the GSM producer.
`simulate_detector_image_instrument` now consumes beam modes through the Layer-1
reducer and binds `(delta_theta_rad, delta_phi_rad, delta_energy_ev)` into the
kinematic amplitude kernel. All exported through `rheedium.types` /
`rheedium.simul` with tests (distribution validation, reduction algebra,
composition, amplitude parity, coherent interference, trivial→intensity,
simulator distribution identity / manual Layer-1 parity, size-distribution
finite-domain parity, beam-mode normalization / variance / coherent-limit
checks, ElectronBeam/preset bridge checks, and instrument-wrapper Layer-1
parity).

**Not yet done:** the full core inversion — `simulate_detector_image` still has
legacy CTR orchestration paths; `distribution=` is currently limited to the
spot-rendered coherent-amplitude path because CTR streaks do not yet expose a
complex amplitude renderer, and `kernel=` currently supports only the kinematic
kernel. Also pending: the unified `DetectorGeometry` carrier (plan wanted a new
`types/detector.py`); completing the Phase 3 default-path replacement for CTR
streaks and explicit beam-mode defaults; Phase 4 defect producers; Phase 5
`multislice_amplitude`; Phase 6 inverse problem; and retiring
`coherence_envelope` / the remaining CTR-only angular+energy Gaussian
quadrature path. Moved from `plans/future/` to `plans/partial/` on landing the
Phase-1 slice.

Subsumes the earlier mixed-state
beam decomposition: its GSM mode math, broadening taxonomy, and beam/sample
ensemble now live here as producers (§4.1) and Layer-1 mechanics (§3, §5).
Builds on
[parallel_sweeps_plan.md](plans/implemented/parallel_sweeps_plan.md): the product ensemble
is exactly the [`tools.distribute_batched`](src/rheedium/tools/parallel.py) batch
axis.

Decisions locked: **(a)** the default distribution is *trivial-sharp* (single
coherent pattern, no hidden broadening); **kinematic** amplitude kernel ships in
v1 with **multislice** as a defined Layer-0 slot.

---

## 1. The inversion

Today `simulate_detector_image` bakes the modulus-squared into the sparse
reflection list, then scatters instrument effects (`n_angular_samples`,
`n_energy_samples`, `orientation_distribution`, the orphan `coherence_envelope`)
through nested ad-hoc `vmap`s. Statistical physics lives in four near-identical
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

### 2.2 Complex render-to-field (the one genuinely new kernel)

`render_pattern_to_image` ([simulator.py:1048](src/rheedium/simul/simulator.py#L1048))
splats *intensities*. Add `render_amplitude_to_field` that deposits
`A·exp(iφ)` onto the dense `H×W` **complex** grid, so that coherently-summed
samples interfere only where reflections overlap (physically correct: large
relative rotations don't interfere; sub-coherence displacements do). A matching
complex variant of `_render_ctr_streaks_to_image` for streak amplitudes.

Relative phase is what matters; fix a single phase origin (beam/sample origin)
shared by all samples so cross-sample phases are consistent.

### 2.3 `multislice_amplitude` (Layer-0 slot, v2)

`multislice_propagate` already carries a complex exit wavefunction; the pattern
is `|FFT(ψ)|²`. `multislice_amplitude` returns the complex diffraction field
before the modulus. Filled after v1; Layers 1–2 are kernel-agnostic so nothing
above changes.

### 2.4 Shared detector contract

Both kernels must render onto the **same** `(H, W, pixel_size_mm,
beam_center_px)` field so Layer 1 is kernel-agnostic. Kinematic maps
`k_out → detector` via Ewald geometry; multislice maps a `q`-grid → detector.
Unify on `project_on_detector_geometry` + `detector_extent_mm`
([simulator.py:164](src/rheedium/simul/simulator.py#L164),
[simulator.py:1197](src/rheedium/simul/simulator.py#L1197)). Define one
`DetectorGeometry` carrier so the two paths cannot drift.

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
  choice.
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

1. choose kernel (kinematic default; multislice opt-in),
2. `bound = distribution.bind(kernel, crystal, detector_geom)`,
3. `image = apply_all(distribution, bound)`,
4. `detector_psf_convolve` + normalize.

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

### 4.2 procs returns Distributions, not CrystalStructures — with one split

- **Pure structure builders** (`create_surface_slab`,
  `apply_surface_reconstruction`, 7×7 library) → still return `CrystalStructure`
  (a 7×7 is a different cell, not a distribution over 1×1).
- **Statistical / defect modifiers** (grains, twins, steps, size, texture) →
  return a `Distribution` over latent parameters whose `bind` closure may call a
  builder per sample.
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
| `coherence_envelope` (orphan) | beam coherence | **Removed** (double-counts modal spread) |
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
   thin `apply`. Wire `SizeDistribution → finite_domain`.
3. **Phase 3 — beam modes (GSM).** `BeamModeDistribution` producer; instrument
   convenience wrapper; the coherence-length `reduction` machinery.
4. **Phase 4 — defect producers.** Twins, steps, grains/texture as
   Distributions; the coherent fine-twin/satellite path exercises a coherent
   reduction end-to-end.
5. **Phase 5 — `multislice_amplitude`** fills the Layer-0 slot.
6. **Phase 6 — inverse problem.** Differentiate through `apply` to fit beam
   coherence, twin density, grain size jointly from measured RHEED data in
   `recon`.

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
5. **Coherent interference:** two displaced identical reflections within `ℓ_c`
   produce fringes; outside `ℓ_c` (incoherent) they sum flat.
6. **Beam-mode limits:** β→0 → 1 mode → coherent pattern; incoherent isotropic
   limit reproduces legacy angular-Gaussian broadening (regression guard for
   removing the Gaussian).
7. **Composition:** `apply_all([beam, orientation])` == manual nested reduction.
8. **Parallel:** ensemble via `distribute_batched` == serial.
9. **Differentiability:** `jax.grad` of pattern w.r.t. β, twin density, grain
   size is finite.
10. **Anisotropy:** in-plane ≠ out-of-plane streak FWHM under anisotropic β.

---

## 9. Risks

- **Amplitude path correctness.** Premature `|·|²` removal must be exact;
  guarded by test 4 (pixelwise equality of `|amplitude|²` vs legacy intensity).
- **Phase-reference consistency** across samples in coherent reduction → fix one
  shared origin; test 5 falsifies a wrong reference.
- **Kernel detector drift** (kinematic vs multislice) → one `DetectorGeometry`
  carrier; assert identical extents.
- **Double-counting coherence** → no `coherence_envelope` in Layer 1; the beam
  `Distribution` is the sole source of partial-coherence broadening.
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
| `src/rheedium/simul/simulator.py` | `kinematic_amplitude`, `render_amplitude_to_field`; `simulate_detector_image` → Layer-1 integrator; `_instrument` convenience wrapper |
| `src/rheedium/simul/multislice.py` | `multislice_amplitude` (Phase 5) |
| `src/rheedium/types/distributions.py` | `Distribution` base + `ReductionMode` + `TRIVIAL`; retrofit `OrientationDistribution`/`SizeDistribution`; `BeamModeDistribution` + GSM factories |
| `src/rheedium/types/` (new `detector.py`) | `DetectorGeometry` carrier |
| `src/rheedium/simul/beam_averaging.py` | `apply`/`apply_all` integrator + `decompose_beam_modes`; retire angular/energy quadrature + `coherence_envelope` from the default path |
| `src/rheedium/procs/{grains,surface_modifier,crystal_defects}.py` | twin/step/grain/texture producers return `Distribution`; structure builders unchanged |
| `src/rheedium/{simul,types,procs}/__init__.py` | exports + Routine Listings |
| `tests/.../{test_simulator,test_distributions,test_beam_averaging,test_surface_modifier,test_grains}.py` | identity, reduction-algebra, interference, limits, composition, parallel, grad |
| `tutorials/` | trivial vs instrument default; mode-count convergence; defect-density demo |

The diffraction physics is preserved; the change is structural — one amplitude
contract at the bottom, one `Distribution` reduction in the middle, and every
multimodal/statistical/defect effect expressed as a producer on top.
