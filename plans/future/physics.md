# RHEED Physics A+ — Stratified, Differentiable, Self-Auditing Forward Physics

Scope: `rheedium` — elevate the RHEED forward model from *parameters → image* into a
**physically stratified, differentiable, self-auditing engine** that, for every
simulated or fitted pattern, can answer five questions:

1. What RHEED physics generated this image?
2. Which approximation level was used?
3. Which physical parameters are identifiable?
4. Where did information become aliased or lost?
5. What next RHEED measurement would reduce the ambiguity?

The target pipeline is not `parameters → image` but:

```text
surface / beam / defects / detector
  → complex RHEED amplitude
  → coherent / incoherent / mixed-state reduction
  → detector image
  → information ledger → null-space genealogy → closure certificate
  → next-measurement recommendation
```

Status: **future / gated — and partially pre-satisfied.** Much of the lower physics
stack already exists from the completed roadmap (framework → rationalization →
recon), so this plan is **not** greenfield: its job is to (a) *formalize* the
honesty/provenance layer over what exists, (b) add the genuinely-missing
grazing-incidence physics (inner potential, refraction, absorption, RHEED-specific
multislice), and (c) make identifiability a first-class output. See §3 for the
done/partial/new reconciliation — five of sixteen gates are already largely
landed.

**Roadmap position:** a **foundational physics-depth track**, parallel to (and the
umbrella over) the
[defect & coherence fidelity plan](defect_diffraction_fidelity_plan.md), which it
owns as gate P6. It builds on the completed
[framework](../implemented/distribution_framework_plan.md) →
[rationalization](../implemented/rationalization_refactor_plan.md) →
[recon](../implemented/recon_optimization_plan.md) chain and feeds the
[automatons](../implemented/automatons_plan.md) (the ledger + experiment-design gates
P12–P15 are what Loop C's `recipe_deviation` and tie-breaking-experiment actions
consume). Like the defect plan, it is a physics-fidelity track, **not** a blocking
roadmap gate for the automatons.

Guard on every gate (inherited): **`jax.grad` flows end-to-end through the touched
path (finite gradients — the invertibility principle, CONTRIBUTING *Invertible
Modularity*); results are reproducible (seeded); tests stay green; `ty`/`ruff`
clean; full `@jaxtyped(typechecker=beartype)`.** Correct RHEED physics outranks
code elegance: a clean-but-wrong simulator is scientifically useless.

**Conventions this plan inherits (CONTRIBUTING):**

- **All proposed PyTrees live in `rheedium.types`** — `RHEEDPhysicsMode`,
  `RHEEDSimulationResult`, `SurfaceFrame`, `RHEEDInformationLedger`, and every other
  `eqx.Module` below is defined under `src/rheedium/types/` (e.g. a new
  `types/physics_types.py` / `types/surface_types.py`), **not** beside the kernel
  that returns it (*Custom Types and PyTrees*).
- **Reuse, don't redefine.** Where a function already exists
  (`debye_waller_factor`, `kirkland_form_factor`, `decompose_beam_modes`,
  `project_on_detector_geometry`, …) this plan *extends* it, never re-creates a
  parallel copy (*Export once, from the module that owns it*).
- **One canonical unit at the public boundary** — beam energy is `energy_kev`,
  incidence is `theta_deg`/`phi_deg` (rationalization R5); no `voltage_kv`.

---

## 1. Non-negotiable principle

Every public simulation result must declare its physics honestly:

```text
physics mode · assumptions · omissions · validation level
differentiable parameters · known degeneracies
```

A+ means: physically stratified, amplitude-preserving, surface-aware,
coherence-aware, instrument-aware, dynamical-ready, experiment-validated,
identifiability-reporting. No stage may silently collapse amplitude into intensity
before the correct physical reduction (this is the framework's core invariant,
already enforced for the existing kernels).

---

## 2. Core pipeline and the result PyTree

The canonical forward model:

```text
CrystalStructure → SurfaceFrame → SurfaceSlab / SurfaceRodSet
  → RHEEDScatteringAmplitude → CoherenceReduction → DetectorImage
  → InformationLedger
```

The result carrier (all in `rheedium.types`):

```python
class RHEEDSimulationResult(eqx.Module):
    image: Float[Array, "ny nx"]
    intensity: Float[Array, "..."]
    amplitude: Complex[Array, "..."]
    physics_mode: RHEEDPhysicsMode
    states: RHEEDSimulationStates          # crystal · surface_frame · rod_set · scattering · coherence · detector
    information: RHEEDInformationLedger     # optional in fast mode; mandatory in paper-quality inversion mode
    provenance: RHEEDSimulationProvenance
```

`RHEEDInformationLedger` (gates P12–P15) holds the per-stage information reports,
the three identifiability levels (local / practical / epistemic), the null-space
genealogy, the closure report, and the experiment-design report.

---

## 3. Reconciliation — what already exists (read this first)

This plan's lower half is largely **already built** by the completed roadmap. Each
gate below is tagged **✅ landed · ◑ partial · ☐ new**; landed/partial gates become
*verify-and-formalize*, not build.

| Gate | Physics | Status | Where it lives today |
|---|---|---|---|
| **P0** physics-mode honesty | ☐ new | — (the provenance/mode layer does not exist) |
| **P1** explicit surface frame | ◑ partial | orientation/azimuth/grazing handled across `ucell` + `simul` + the *arbitrary-directions* guide; a formal `SurfaceFrame` PyTree + invariants is new |
| **P2** Ewald–CTR geometry | ✅ landed | `simul/ewald.py` (`build_ewald_data`, `ewald_allowed_reflections`), `make_ewald_sphere`, `find_ctr_ewald_intersection`, `ewald_simulator`, `project_on_detector_geometry` |
| **P3** complex-amplitude contract + reducers | ✅ landed | the **distribution framework** — `ReductionMode` (COHERENT/INCOHERENT) in `types/distributions/base.py`; amplitude kernels (`kinematic_amplitude`, `multislice_amplitude`) in `simul`; mixed-state via `BeamModeDistribution` |
| **P4** slab / CTR discipline | ◑ partial | analytic CTR + roughness in `simul/surface_rods.py`; explicit slab via `multislice`/`finite_domain`; the *one-declared-mechanism* flags + no-double-count discipline is new |
| **P5** atomic scattering + thermal/disorder | ◑ mostly landed | `simul/form_factors.py` (`kirkland_form_factor`, `lobato_form_factor`, `debye_waller_factor`, `get_mean_square_displacement`, occupancy); **anisotropic DW + surface-disorder factor** are new |
| **P6** surface non-ideality / defect signatures | ◑ owned elsewhere | finite-domain broadening already in `simul/finite_domain.py` (`SizeDistribution`); twins/steps/mosaic/fringes are the [defect & coherence fidelity plan](defect_diffraction_fidelity_plan.md) (D1–D4) — **that plan is gate P6** |
| **P7** partial coherence / GSM mixed-state | ✅ landed | `BeamModeDistribution` + `decompose_beam_modes` (Gaussian–Schell modes), reduced through the framework's mixed-state path |
| **P8** detector + image formation | ◑ mostly landed | `DetectorGeometry` (`types/detector.py`), `project_on_detector_geometry`, `render_pattern_to_image`, `detector_psf_convolve` (rationalization R1); **noise / gain / saturation / mask** for synthetic *experimental* frames are new |
| **P9** inner potential / refraction / absorption | ☐ new | **the real gap for quantitative intensity** — nothing today |
| **P10** RHEED-specific multislice | ◑ partial | `simul/reflection_multislice.py` exists; grazing refraction, inner-potential entry, absorbing boundaries, depth weighting are new |
| **P11** experimental benchmark suite | ☐ new | `rheedium.audit` has invariant scaffolding; real-frame fitting is new |
| **P12–P15** information ledger / genealogy / closure / design | ◑ foundation in recon | recon §2.3 (identifiability; the `argmax` tie-breaking experiment = expected information gain) + `fisher_information_from_residual` / `laplace_uncertainty` / `sample_posterior` are the foundation; **formalizing them into ledger PyTrees** is new |

**Net:** P2/P3/P7 are done; P5/P8 are mostly done; P1/P4/P10/P12–P15 build on real
foundations; **P0/P9/P11 are the genuinely greenfield physics.** P6 is delegated to
the defect plan (no duplication). The honest scope of *new* work is the
honesty/provenance layer, grazing electron optics, RHEED-true multislice,
experimental benchmarking, and the information ledger.

---

## 4. Physics-mode taxonomy (P0)

Every simulator pathway carries an explicit `RHEEDPhysicsMode` (in
`rheedium.types`), so no output is ever called simply "the RHEED simulation"
without its mode attached.

Modes: `geometry_only`, `kinematic_bulk_points`, `kinematic_surface_rods`,
`kinematic_slab_explicit`, `kinematic_ctr_analytic`,
`kinematic_ctr_phenomenological`, `partial_coherence_mixed_state`,
`instrument_convolved`, `inner_potential_corrected`, `absorptive_kinematic`,
`rheed_multislice`, `experimental_hybrid`.

```python
class RHEEDPhysicsMode(eqx.Module):
    name: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    omissions: tuple[str, ...] = eqx.field(static=True)
    elastic_only: bool = eqx.field(static=True)
    single_scattering: bool = eqx.field(static=True)
    dynamical: bool = eqx.field(static=True)
    uses_surface_rods: bool = eqx.field(static=True)
    uses_explicit_slab: bool = eqx.field(static=True)
    uses_analytic_ctr: bool = eqx.field(static=True)
    uses_phenomenological_ctr: bool = eqx.field(static=True)
    uses_partial_coherence: bool = eqx.field(static=True)
    uses_inner_potential: bool = eqx.field(static=True)
    uses_absorption: bool = eqx.field(static=True)
    uses_detector_psf: bool = eqx.field(static=True)
    validation_level: str = eqx.field(static=True)   # the §7 ladder
```

---

## 5. Gated phases

Each gate ends at an objectively checkable exit criterion. Tags carry the §3
reconciliation status; landed/partial gates are *verify/extend*, not *build*.

### P0 — Physics-mode honesty ☐ new

*Tasks:* add `RHEEDPhysicsMode` + `RHEEDSimulationProvenance` (in `rheedium.types`);
attach a mode to every existing simulator output; separate phenomenological /
analytic / explicit-slab / dynamical modes; store assumptions + omissions +
validation level on every result.
*Exit:* every simulator result declares mode, assumptions, omissions; a
phenomenological CTR cannot be selected in quantitative mode; dynamical mode cannot
be claimed unless the multislice path ran. A user can tell exactly what physics
generated an image.

### P1 — Explicit RHEED surface frame ◑ partial

RHEED is surface-defined: surface normal, azimuth, grazing angle, and the beam /
detector frames must be explicit, not implied by `c*`.

```python
class SurfaceFrame(eqx.Module):                 # rheedium.types
    surface_normal_crystal: Float[Array, "3"]
    azimuth_crystal: Float[Array, "3"]
    surface_x_crystal: Float[Array, "3"]
    surface_y_crystal: Float[Array, "3"]
    surface_z_crystal: Float[Array, "3"]
    crystal_to_surface: Float[Array, "3 3"]
    surface_to_crystal: Float[Array, "3 3"]
    reciprocal_surface_x: Float[Array, "3"]
    reciprocal_surface_y: Float[Array, "3"]
    reciprocal_surface_z: Float[Array, "3"]
    grazing_angle_rad: Float[Array, ""]
    azimuth_angle_rad: Float[Array, ""]
```

*Functions* (reuse `ucell` basis helpers where they exist): `build_surface_frame`,
`validate_surface_frame`, `crystal_to_surface_vectors`,
`surface_to_crystal_vectors`, `surface_to_lab_vectors`, `surface_reciprocal_basis`,
`incident_wavevector_surface` / `_lab`.
*Invariants:* orthonormal surface basis; `crystal_to_surface @ surface_to_crystal =
I`; `aᵢ·bⱼ = 2πδᵢⱼ`; beam direction has the requested grazing angle; azimuth
projects correctly.
*Tests:* cubic (001)/(110)/(111), SrTiO₃ (001), Si (001), GaAs (001), a triclinic
cell, 90° azimuth rotation, round-trip transforms.
*Exit:* no RHEED kernel assumes `c*` is the rod direction unless the `SurfaceFrame`
makes it so.

### P2 — Ewald–CTR geometry ✅ landed (verify + formalize)

`q(l) = G∥ + l·b_⊥`, `k_out = k_in + q(l)`, `|k_out| = |k_in|` — already implemented
(`ewald.py`, `ewald_simulator`, `find_ctr_ewald_intersection`,
`project_on_detector_geometry`).
*Remaining:* route the existing rod/Ewald path through the P1 `SurfaceFrame` (so
geometry is frame-derived, not `c*`-assumed) and add the standing geometric
regressions below.
*Tests:* `|k_out|−|k_in|` within tolerance; specular at `G∥=0`; beam-energy scan
flattens Ewald curvature; grazing scan shifts intersections smoothly; azimuth scan
rotates the pattern; symmetry-equivalent rods map correctly.
*Exit:* the geometric pattern is defensible independent of intensity modeling.

### P3 — Complex-amplitude contract ✅ landed

Already the framework's core invariant: kernels return `Complex[Array, ...]`;
intensity appears only through explicit reducers — `coherent: I=|Σ Aᵢ|²`,
`incoherent: I=Σ wᵢ|Aᵢ|²`, `mixed-state: I=Σ_m w_m|Σᵢ A_{m,i}|²` (`ReductionMode`,
`BeamModeDistribution`).
*Remaining:* none structural — keep the phase/interference regressions (global phase
invariance, relative-phase interference, single-mode = coherent, many-mode →
incoherent) as standing tests, and assert the "no premature `|·|²` in a kernel"
guard for any **new** kernel this plan adds (P9/P10).
*Exit:* the code knows where physics switches from amplitude to intensity (it does).

### P4 — Slab / CTR discipline ◑ partial

Three CTR mechanisms must not double-count: **explicit slab**
`A_hk(q_z)=Σⱼ occⱼ fⱼ(q) e^{−Wⱼ} e^{iG∥·r∥} e^{iq_z zⱼ}`; **analytic CTR**
`A_hk=F_hk·T(q_z)`; **phenomenological CTR** (`1/sin²(πl)`-style, *quantitative-mode
forbidden*). The models exist (`surface_rods.py`, `multislice`/`finite_domain`); the
gate adds the `uses_explicit_slab` / `uses_analytic_ctr` /
`uses_phenomenological_ctr` flags (exactly one true) and the no-double-count rule.
*Tests:* N-layer slab fringe spacing ∝ 1/N; large-N → sharp rod modulation; analytic
CTR ≈ explicit slab in the ideal limit; top-layer removal changes rod amplitude;
phenomenological CTR unavailable in quantitative mode.
*Exit:* CTR physics comes from exactly one declared mechanism.

### P5 — Atomic scattering & thermal/disorder ◑ mostly landed

`kirkland_form_factor` / `lobato_form_factor`, `debye_waller_factor`, occupancy, and
`get_mean_square_displacement` already exist (`form_factors.py`). *Remaining (new):*
`anisotropic_debye_waller_factor` and `surface_disorder_factor` (surface-enhanced
disorder), wired to the existing form-factor path (reuse, don't fork).
*Tests:* form factors fall with |q|; DW suppresses high-q more than low-q; occupancy
scales linearly; zero occupancy removes the atom; surface disorder damps
surface-sensitive features.
*Exit:* atomic intensity physics is complete, with anisotropic + surface terms.

### P6 — Surface non-ideality & defect signatures ◑ owned by the defect plan

This gate **is** the
[defect & coherence fidelity plan](defect_diffraction_fidelity_plan.md) — its single
source of truth, to avoid duplication. That plan owns fine-twin satellites (D1),
step-terrace/vicinal splitting (D2), coherent displacement-fringe validation (D3),
and `blackjax` sampling producers for implicit/Boltzmann ensembles (D4); finite
lateral-domain broadening already ships (`finite_domain.py` / `SizeDistribution`).
*Remaining here:* nothing duplicated — P6 references the defect plan; this gate's
exit criterion is **satisfied when the defect plan's DG1–DG4 close** (plus the
already-landed finite-domain broadening). The signature library (finite domain →
reciprocal broadening; regular steps → satellites at 2π/D; random steps → diffuse;
mosaicity → angular broadening; reconstruction → fractional-order rods; twins →
secondary rods; strain → streak bending) is specified there.
*Exit:* surface-defect claims are tied to characteristic reciprocal-space
signatures, owned by one plan.

### P7 — Partial coherence / mixed-state beam ✅ landed

Gaussian–Schell mixed-state source already implemented (`BeamModeDistribution`,
`decompose_beam_modes`): transverse coherence length, source angular spread, mode
weights, coherent/incoherent limits, reduced through the framework's mixed-state
path. Beam coherence is modeled *before* intensity reduction and is distinct from
detector PSF.
*Remaining:* `grazing_footprint_anisotropy` (the grazing-incidence footprint
distortion) as an extension of the existing GSM path; keep the "coherence length ≠
PSF, separable only in multi-measurement" regression (this is also a P13 degeneracy).
*Exit:* partial coherence is modeled before intensity reduction (it is).

### P8 — Detector & experimental image formation ◑ mostly landed

`DetectorGeometry` + `project_on_detector_geometry` + `render_pattern_to_image` +
`detector_psf_convolve` already cover screen geometry, distance, pixel size,
rasterization, and PSF (rationalization R1). *Remaining (new), for synthetic
experimental frames:* `apply_gain_offset`, `apply_noise_model` (Poisson + read),
saturation, `apply_mask` (beamstop), background, composed as
`simulate_experimental_frame` over the existing render path.
*Tests:* normalized PSF preserves total intensity; supersampling converges; noise
mean/variance correct; masking preserves unmasked pixels; gain/exposure separable
only with calibration; geometry perturbations shift the image predictably.
*Exit:* synthetic images compare to real RHEED frames with known metadata.

### P9 — Inner potential, refraction, absorption ☐ new (the quantitative-intensity gate)

At grazing incidence, refraction and the inner potential strongly affect normal
momentum, penetration depth, and intensity — **currently unmodeled.**
*Functions (new, in `simul`):* `electron_interaction_constant`,
`inner_potential_corrected_wavevector`, `refracted_grazing_angle`,
`penetration_depth`, `inelastic_attenuation`, `absorptive_layer_weight` (real +
imaginary inner potential, critical-angle behavior, depth weighting).
*Tests:* zero inner potential recovers vacuum geometry; real V₀ changes the internal
normal wavevector; imaginary V₀ reduces deep-layer contribution; penetration depth
grows with grazing angle; absorption never increases elastic intensity; **V₀ aliases
with grazing-angle calibration** (feed this to P13).
*Exit:* quantitative RHEED intensity claims require this gate.

### P10 — RHEED-specific multislice ◑ partial

`reflection_multislice.py` exists; full RHEED multislice adds surface-oriented
slicing, a grazing incident wave, the vacuum/crystal interface with P9 refraction,
absorptive potential, long path lengths, absorbing boundaries, exit-wave extraction,
and detector projection.
*Functions:* `build_surface_slices`, `initialize_grazing_incident_wave`,
`apply_surface_refraction`, extend `reflection_multislice_propagate`,
`extract_exit_wave`, `project_exit_wave_to_detector`.
*Tests:* zero potential → undeviated beam; weak-potential limit → kinematic result;
slice-thickness + grid-size convergence; thickness changes intensities nonlinearly;
absorption damps multiple scattering; comparison to a trusted external benchmark.
*Exit:* dynamical RHEED claims require this gate.

### P11 — Experimental benchmark suite ☐ new

Tiered validation: **T1 analytic** (simple cubic, known rods/slab/domain envelope);
**T2 synthetic materials** (Si, SrTiO₃, MgO/GaAs at known orientation/`energy_kev`/
detector geometry); **T3 external reference** (published pattern / trusted simulator
/ rocking curve); **T4 real experiment** (material, orientation, `energy_kev`,
grazing, azimuth, detector distance, calibration, temperature, growth state, raw +
background images). Metrics: spot/streak position error, rod FWHM error, intensity
rank correlation, normalized + background-subtracted residual, symmetry residual,
parameter/calibration recovery error. (Build on `rheedium.audit`.)
*Exit:* at least one measured RHEED frame is fit with stated residuals and stated
missing physics.

### P12–P15 — The RHEED information ledger ◑ foundation in recon

These four gates formalize what recon already does (identifiability with bands,
Fisher/Laplace/posterior UQ, and the §2.3 tie-breaking experiment = expected
information gain) into first-class output PyTrees in `rheedium.types`. They consume
recon's `fisher_information_from_residual` / `laplace_uncertainty` /
`sample_posterior` and residual Jacobians rather than re-deriving them.

- **P12 — stage information.** Per-stage `RHEEDStageInformation` (local rank,
  singular values, condition number, null/sloppy/sensitive directions,
  preserved/lost/amplified/suppressed scores) for crystal → surface frame → slab/rods
  → amplitude → coherence → detector → image. Audit levels: none / local / spectral /
  closure / design. *Exit:* a simulation reports which RHEED parameters are locally
  identifiable.
- **P13 — null-space genealogy.** `RHEEDNullDirectionGenealogy` — not just *that* a
  direction is degenerate but *where* the degeneracy entered (first-loss stage,
  strongest-alias stage, survival/amplification by stage) + candidate breaking
  measurements. Degeneracy library: roughness↔coherence length, domain
  size↔beam divergence, Debye–Waller↔absorption, inner potential↔grazing
  calibration, detector PSF↔finite-domain broadening, gain↔scattering strength,
  background↔diffuse scattering. *Exit:* "unidentifiable **because** this stage
  aliased these mechanisms."
- **P14 — closure certificate.** Decompose `r = y−f(θ*)` via the local Jacobian into
  `r_explainable ∈ Range(J)` and `r_unexplained ∈ Range(J)^⊥`; `RHEEDClosureReport`
  carries the residuals, closure score, per-stage scores, and ranked
  likely-missing-physics (reconstruction rods, steps, mosaicity, detector geometry,
  inner potential, absorption, dynamical scattering, diffuse background, wrong
  termination). *Exit:* distinguish "bad parameter fit" from "missing RHEED physics."
- **P15 — next-measurement design.** `RHEEDExperimentDesignReport` ranks candidate
  measurements (new grazing angle / azimuth / `energy_kev` / rocking curve / temp
  series / coherence or detector calibration) by expected rank gain, information
  gain, condition improvement, and closure reduction — the differentiable
  realization of recon §2.3's `argmax_Δ E[D_KL]`. *Exit:* the package recommends a
  physically meaningful next RHEED measurement. **This is the report Loop C's
  tie-breaking action ([automatons](../implemented/automatons_plan.md)) consumes.**

---

## 6. Validation ladder

Every output gets a level: **L0** visual · **L1** analytic invariant · **L2**
synthetic ground-truth recovered · **L3** external simulator/literature agreement ·
**L4** real experimental agreement · **L5** inverse identifiability quantified ·
**L6** closure + next-measurement validated. Minimum paper claims: geometry L2,
kinematic intensity L3, surface-defect L2/L3, coherence L2/L3, instrument L4
preferred, dynamical L3, experimental inversion L4/L5, physics-assaying L5/L6.

---

## 7. Implementation phases (new + extend work only)

Landed gates (P2/P3/P7, most of P5/P8) need only the standing regressions; the
build effort concentrates on the new/partial gates.

1. **Honesty layer** — `RHEEDPhysicsMode` + `RHEEDSimulationResult` +
   `RHEEDSimulationProvenance`; attach to existing simulators (P0).
2. **Surface frame** — `SurfaceFrame`; route P2 geometry through it (P1; verify P2).
3. **Scattering completeness** — anisotropic + surface-disorder factors; CTR
   one-mechanism discipline (P5, P4).
4. **Defect fidelity** — execute the [defect plan](defect_diffraction_fidelity_plan.md)
   DG1–DG4 (P6).
5. **Coherence + experimental detector** — grazing footprint anisotropy; noise / gain
   / mask / `simulate_experimental_frame` (P7 extend, P8).
6. **Grazing electron optics** — inner potential, refraction, absorption (P9).
7. **Dynamical** — RHEED-specific multislice (P10).
8. **Benchmarks** — analytic → synthetic → external → real (P11).
9. **Information ledger** — stage information, genealogy, closure, design, on recon's
   UQ foundation (P12–P15).

---

## 8. Diff surface

| Path | Change |
|---|---|
| `src/rheedium/types/physics_types.py` *(new)* | `RHEEDPhysicsMode`, `RHEEDSimulationResult`, `RHEEDSimulationStates`, `RHEEDSimulationProvenance`, `RHEEDInformationLedger`, `RHEEDStageInformation`, `RHEEDNullDirectionGenealogy`, `RHEEDClosureReport`, `RHEEDExperimentDesignReport` + constructors (all PyTrees centralized here per the types rule) |
| `src/rheedium/types/surface_types.py` *(new)* | `SurfaceFrame` + `build_surface_frame` / validators |
| `src/rheedium/simul/surface_frame.py` *(new)* | frame-derived geometry; route existing Ewald/rod path through `SurfaceFrame` (P1/P2) |
| `src/rheedium/simul/form_factors.py` | **extend** — `anisotropic_debye_waller_factor`, `surface_disorder_factor` (P5) |
| `src/rheedium/simul/surface_rods.py` | CTR one-mechanism flags + no-double-count discipline (P4) |
| `src/rheedium/simul/grazing_optics.py` *(new)* | inner potential / refraction / absorption (P9) |
| `src/rheedium/simul/reflection_multislice.py` | **extend** — grazing wave, surface refraction, absorbing boundaries, exit-wave (P10) |
| `src/rheedium/simul/detector.py` *(detector image-formation extensions)* | noise / gain / mask / `simulate_experimental_frame` (P8) |
| `src/rheedium/audit/` | experimental benchmark tiers + metrics (P11) |
| `src/rheedium/recon/` | reuse `fisher_information_from_residual` / `laplace_uncertainty` / `sample_posterior` for the ledger; no re-derivation (P12–P15) |
| `defect_diffraction_fidelity_plan.md` | **is gate P6** — referenced, not duplicated |
| `tests/test_rheedium/...` | per-gate physics/invariant/identifiability tests at the §6 validation levels |

The lower physics stack stays where it already lives; this plan adds the
honesty layer, the grazing-incidence physics, RHEED-true multislice, benchmarks,
and the ledger — and centralizes every new PyTree in `rheedium.types`.

---

## 9. Paper-readiness and claim boundaries

**Strong methods paper:** P0–P5, P7 complete; P8 mostly; P11 ≥1 real/literature
benchmark; P12 local/spectral identifiability on a synthetic case. **Strong RHEED
physics paper:** P0–P9 complete; P11 real benchmark; P12–P14 complete; one
degeneracy/closure demonstration. **A+ engine:** P0–P15 complete; real experimental
inversion; validated inner potential/refraction, partial coherence, defect
signatures, dynamical path; closure residual identifies omitted physics; next
measurement breaks a demonstrated degeneracy.

Hard boundaries: no quantitative-intensity claim without P9; no dynamical claim
without P10; no defect-physics claim without P6 (defect plan); no partial-coherence
claim without P7; no experiment-facing inversion without P11; no identifiability
claim without P12; no missing-physics claim without P14; no next-experiment-design
claim without P15.

---

## 10. Killer demonstration

A single RHEED frame cannot distinguish roughness, finite domain size, beam
divergence, and transverse coherence. `rheedium` fits the frame (recon), reports a
sloppy/null direction (P12), **localizes the degeneracy to the
coherence/image-formation stages** (P13), flags that no parameter explains part of
the residual (P14), and **recommends a second grazing angle or coherence
calibration** (P15) — the action the autonomous lab's Loop C
([automatons](../implemented/automatons_plan.md)) then takes. After the second
measurement the null space shrinks and the parameters separate.

This is what makes `rheedium` not merely a differentiable RHEED simulator but a
**RHEED information engine**: it produces, fits, and *audits* a pattern — reporting
the approximation level, the amplitude pathway, the surface geometry, the scattering
mechanism, the coherence and detector models, the identifiable and null directions,
the stage where information was lost, the residual no parameter can explain, the
likely missing physics, and the next measurement that would help.
