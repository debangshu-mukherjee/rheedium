# Paper Outline — Differentiable RHEED: a unified forward + inverse engine in JAX

Scope: the methods/software paper for `rheedium` — the differentiable RHEED engine
(forward simulation + inverse reconstruction). This is a self-contained,
publishable methods paper and the citable foundation any later application work
builds on.

This outline is derived from the plans and is **honest about status** — the paper
is gated on roadmap milestones that have not all landed yet (see
[§ Readiness](#readiness--what-must-land-before-submission)). It is a writing
target, not a claim that the results exist today.

Source plans:
[distribution framework](plans/partial/distribution_framework_plan.md) ·
[recon (inversion)](plans/future/recon_optimization_plan.md) ·
[rationalization](plans/future/rationalization_refactor_plan.md) ·
[audit/formal verification](plans/future/audit_formal_verification_plan.md) ·
[hypothesis testing](plans/future/hypothesis_testing_plan.md) ·
[parallel sweeps](plans/implemented/parallel_sweeps_plan.md) ·
[multislice](plans/implemented/reflection_multislice_plan.md)

---

## Working titles

1. *Differentiable RHEED: end-to-end gradients from crystal structure to detector
   image, and back*
2. *`rheedium`: a JAX framework for forward simulation and inverse reconstruction
   of RHEED patterns*
3. *One amplitude contract, one reduction: a differentiable distribution
   framework for partially-coherent electron diffraction*

## Target venue

**Target: *npj Computational Materials*.** Fallbacks: *Digital Discovery* (RSC),
*Journal of Applied Crystallography*, *Ultramicroscopy*.

### Editorial framing (npj Comp Mater)

- **Main text carries the physics, the mathematics, and the capabilities** — the
  diffraction physics (complex amplitude, partial-coherence mixed-state beam, the
  coherent/incoherent partition), the inverse-problem mathematics
  (differentiability, reparameterization, uncertainty, identifiability), and what
  the engine can *do* (forward validation, structure/orientation/beam recovery,
  probability-distribution reconstruction).
- **Architecture and implementation are Supplementary.** The three-layer
  inversion, the `Distribution` PyTree / bind contract, the AOT/compilation-cache
  engineering, the typing/validation discipline, and the rigor ladder are *how the
  capability is delivered*, not the scientific claim — they go to the SI, refer­enced
  from a short Methods section.
- **Figure budget: 10 main figures + 10–20 supplementary figures.** Main figures
  are physics/maths/capability; supplementary figures are architecture, extended
  validation, and engineering.

---

## One-sentence thesis

RHEED simulation can be written so that a single complex-amplitude kernel and a
single `Distribution` reduction express every multimodal, statistical, and defect
effect *and* keep `jax.grad` flowing end-to-end — making the inverse problem
(structure / orientation / beam / probability-distribution recovery) a direct
gradient-based fit rather than a bespoke optimizer.

## Contributions (main-text claim list — physics, maths, capability)

1. **A complex-amplitude kinematic theory of RHEED that preserves relative phase.**
   Per-reflection structure factor `F(G) = Σⱼ fⱼ exp(iG·rⱼ)` carried as a complex
   field through CTR and roughness scaling, so coherently-overlapping reflections
   interfere with the correct relative phase — the physics a phaseless
   intensity-only simulator cannot represent. [framework §2, maths]
2. **A mixed-state (Gaussian–Schell) partial-coherence beam model for RHEED.** Wolf
   coherent-mode decomposition with geometric GSM eigenvalues `λₙ = (1−β)βⁿ`, made
   *anisotropic* by grazing incidence (`1/sin θ` footprint foreshortening, in-plane
   vs out-of-plane) — strictly more expressive than a shift-invariant Gaussian blur,
   and the correct physics of a real electron source. [framework §4.1, physics]
3. **A physically-derived coherent/incoherent partition.** Whether a feature
   averages coherently (inside `|·|²`) or incoherently (outside) follows from its
   size relative to the per-mode coherence length `ℓ_c` — a derived physical
   criterion, not a modelling switch — mapping defect classes (grains, twins,
   steps, finite domains) to their diffraction signatures. [framework §5, physics]
4. **The inverse problem as a direct gradient fit, with quantified uncertainty.**
   Because the forward map is differentiable end-to-end, recovering
   structure / orientation / beam reduces to least-squares on the residual
   (Gauss–Newton / Levenberg–Marquardt), with the Gauss–Newton `JᵀJ` giving a
   Fisher/Laplace covariance — error bars, not point estimates. The maths: a
   reparameterize-don't-project bijection keeps the fit unconstrained and smooth.
   [recon §1–§2, maths + capability]
5. **Probability-distribution reconstruction with credible bands.** Recover a
   *distribution* over a latent (orientation spread, grain-size, beam coherence) —
   parametric or free-form — including a convex maximum-entropy / NNLS fast-path for
   the incoherent case (`I = Σₙ wₙ|Aₙ|²` is linear in the weights). A capability no
   point-estimate fitter offers. [recon §2.1, maths + capability]
6. **An identifiability theory for RHEED inversion.** The `material model →
   distribution → pattern` hierarchy: the distribution is recoverable (bounded null
   space), but the generating *mechanism* is many-to-one — distinct physics can
   yield the same distribution. We state what the data can and cannot determine —
   the honest scientific boundary of inverse RHEED. [recon §2.3, maths]
7. **Validated forward + inverse capability.** Quantitative agreement against
   reference simulations (kinematic and multislice) and ≥1 experimental pattern;
   recovery of known structure/orientation/beam/distributions from synthetic ground
   truth; throughput sufficient for high-volume and near-real-time use. [results]

*Architecture and implementation (the three-layer inversion, the `Distribution`
PyTree + bind contract, AOT/compilation-cache engineering, the typing/validation
discipline, and the proof-vs-sampled rigor ladder) are the **means** to the above
and are documented in the **Supplementary Information**, summarized in Methods —
see [framework §1–§4](plans/partial/distribution_framework_plan.md), recon K6,
[audit](plans/future/audit_formal_verification_plan.md), and
[hypothesis](plans/future/hypothesis_testing_plan.md).*

## Figures — 10 main + 10–20 supplementary

### Main figures (physics · maths · capability) — exactly 10

1. **RHEED forward physics overview.** Grazing-incidence geometry, the Ewald
   construction, reciprocal-rod ↔ detector mapping — the physical picture the rest
   of the paper inverts.
2. **Complex amplitude & coherent interference.** Per-reflection structure-factor
   phase; two reflections within vs beyond `ℓ_c` → fringes vs flat sum; why phase
   (not just intensity) is physical.
3. **Mixed-state Gaussian–Schell beam.** Mode spectrum `λₙ=(1−β)βⁿ`; effective
   mode count vs `β`; partial coherence reconstructed from modes vs the
   single-mode limit.
4. **RHEED beam anisotropy.** Grazing `1/sin θ` foreshortening → in-plane vs
   out-of-plane divergence and streak FWHM; the anisotropy a Gaussian blur cannot
   produce.
5. **Coherent/incoherent partition from `ℓ_c`.** Feature-size-vs-`ℓ_c` regime map;
   grains/twins/steps/finite domains → their distinct pattern signatures.
6. **Forward validation.** Simulated vs reference (kinematic **and** multislice)
   patterns; per-reflection amplitude/intensity parity; agreement metrics.
7. **Differentiable inverse — geometry & beam.** Recover orientation + beam from a
   CIF + pattern; loss-vs-iteration; multistart resolving the symmetry orbit;
   covariance ellipses (uncertainty, not point estimate).
8. **Inversion on experimental RHEED.** Recover orientation/beam/structure from a
   *measured* frame; fit overlay + residual — the real-data capability.
9. **Probability-distribution reconstruction.** Planted vs recovered distribution
   (orientation spread / grain-size / beam coherence) with credible band; convex
   incoherent fast-path vs nonlinear coherent.
10. **Identifiability.** Two distinct mechanisms → the same distribution; the
    `model → distribution → pattern` hierarchy; what the data can and cannot
    determine.

### Supplementary figures (architecture · engineering · extended validation) — 10–20

- **S1 Three-layer architecture** — Layer 0 amplitude → Layer 1 `apply` → Layer 2
  producers.
- **S2 `Distribution` PyTree + reduction algebra** — coherent vs incoherent
  reducer; the `N=1` identity.
- **S3 Bind/closure contract** — sample → kernel call; the registry; structure-
  rebuilding vs angular-perturbation binds.
- **S4 Kernel-agnostic integrator** — kinematic vs multislice through the *same*
  spine onto one detector geometry; parity.
- **S5 GSM derivation detail** — Hermite–Gauss mode nodes; variance-matching to the
  Gaussian limit; truncation/heavy tail.
- **S6 Reparameterization bijectors** — positivity/bounded/simplex maps; smoothness
  vs projected gradients.
- **S7 Uncertainty calibration** — Gauss–Newton/Laplace covariance vs empirical
  spread; PSD/coverage check.
- **S8 Multistart basins** — symmetry-orbit landscape; cold vs bracketed starts.
- **S9 Convex incoherent fast-path** — NNLS/max-entropy weights vs nonlinear
  `solve`; timing.
- **S10 Compilation cache** — cold vs warm compile timing; SIGILL/machine-feature
  portability note.
- **S11 AOT StableHLO export** — symbolic atom count; FFT non-polymorphism; bucketed
  grids; cross-process reuse.
- **S12 Batched scaling** — `distribute_batched` throughput vs devices.
- **S13 Validation discipline** — jaxtyping + beartype two-tier checks; a caught
  malformed input.
- **S14 Rigor ladder** — `audit` invariants labelled proof vs sampled-within-
  tolerance.
- **S15 Property-based tests** — `hypothesis` round-trip/invariant coverage; a
  shrunk counterexample.
- **S16 Convergence studies** — mode-count and multislice-grid convergence.
- **S17 Additional experimental examples** — further measured patterns / inversions.
- **S18 Reproducibility build graph** — `rheedium_paper` one-command figure→PDF DAG.
- *(S19–S20 reserve)* — extra ablations / robustness (noise, partial detector,
  background) as needed to fill 10–20.

## Section skeleton

### Main text (physics · maths · capability)

1. **Introduction** — RHEED's information content; why forward sims are siloed and
   inverse analysis is manual; the differentiable opportunity.
2. **RHEED forward physics** — kinematic complex amplitude and the structure-factor
   phase; CTR/roughness; the multislice (dynamical) kernel as an alternative;
   partial coherence as a mixed quantum state and the anisotropic Gaussian–Schell
   beam; the coherence-length-derived coherent/incoherent partition. *(Figs 1–5)*
3. **The differentiable inverse problem** — end-to-end differentiability of the
   forward map; inversion as Gauss–Newton/LM least-squares; reparameterization;
   Fisher/Laplace uncertainty; probability-distribution reconstruction; the
   identifiability hierarchy. *(maths-led)*
4. **Results** — forward validation vs reference + experiment; geometry/beam and
   structure recovery (synthetic + experimental); distribution reconstruction with
   bands; an identifiability case study; throughput/real-time capability.
   *(Figs 6–10)*
5. **Discussion** — scope and limits (kinematic vs dynamical; what identifiability
   forbids); relation to mixed-state ptychography; outlook (autonomous/real-time
   use).
6. **Methods** — concise: the engine in one paragraph (forward kernels →
   `Distribution` integrator → differentiable inverse), pointing to the SI for the
   architecture, solver, and implementation; data sources; reproducibility
   (pinned version, seeds, DOI).
7. **Code & data availability** — PyPI release + GitHub (MIT), pinned version, the
   archived `rheedium_paper` reproducibility repo (DOI).

### Supplementary Information (architecture · implementation · extended validation)

- **SI-A Architecture** — the three-layer inversion (push `|·|²` to Layer 0, thin
  `apply(distribution, kernel)` integrator, producers emitting one `Distribution`);
  the kernel-agnostic detector contract. *(Figs S1–S4)*
- **SI-B The `Distribution` contract** — reduction algebra, nested composition, the
  bind/closure registry. *(Figs S2–S3)*
- **SI-C Beam-mode mathematics** — full GSM derivation, mode placement, energy
  sub-axis. *(Fig S5)*
- **SI-D Inverse solver internals** — `optimistix`/`optax` stack, bijectors,
  multistart, UQ derivation. *(Figs S6–S9)*
- **SI-E Performance & deployment** — compilation cache, AOT StableHLO export,
  batched scaling. *(Figs S10–S12)*
- **SI-F Correctness & reproducibility** — typing/validation discipline, the
  proof-vs-sampled rigor ladder, property-based tests, convergence, the
  reproducibility repo. *(Figs S13–S18)*

---

## Readiness — what must land before submission

| Component | Hard prerequisites | Current status |
|---|---|---|
| **Forward half** | framework Phases 1–6 **complete**: the §6 differentiability guarantee, plus a `PotentialSlices` producer bind for structure-changing multislice axes and the size/finite-domain detector bind | mostly landed — integrator inverted; **multislice now selectable via `kernel=`**; interference + defect-distinguishability tests exist; differentiability gate **partially locked** (β + twin-fraction grads live, **grain-size a strict `xfail`** pending the size-into-integrator work); remaining: `Distribution.bind` contract, size bind, multislice structure-axis producers |
| **Inverse half** | `recon` K0–KG6: solver, distribution reconstruction, identifiability, UQ on synthetic ground truth | specified, **not built** (gated after rationalization) |
| **Rigor** | `audit` proof/sampled labelling + `hypothesis` round-trip/invariant tests | planned |
| **Evidence** | ≥1 experimental RHEED dataset inverted end-to-end | to source |

**Recommended writing order**

1. Finish framework Phase 6 — **close the grain-size differentiability `xfail`**
   (wire `size` into the integrator) so the end-to-end grad guarantee holds; then
   draft the *forward* results. (Multislice-in-`kernel=` already landed.)
2. Land `recon` (the inverse solver + distribution reconstruction + UQ) →
   complete the *inverse* results.
3. Add the rigor pass (`audit` labelling + `hypothesis` tests) → **submit**.

### Cross-cutting assets to prepare early

- A **reproducibility statement**: pinned `rheedium` version, archived DOI
  (Zenodo), deterministic seeds, `jax_backend` recorded per result.
- A **synthetic ground-truth benchmark suite** (known CIF + orientation + beam →
  pattern) for inverse validation.
- ≥1 **experimental RHEED dataset** with enough metadata to invert — the single
  biggest credibility lever.

---

## Companion reproducibility repo — `rheedium_paper`

A **separate Git repository** (not part of the `rheedium` package) that holds the
entire manuscript and regenerates it from scratch on any machine. It applies the
automatons' PEP 723 single-dependency philosophy to the paper itself: every figure
is a self-contained, version-pinned script, the data ships with the repo, and the
whole thing builds with one command.

**Principles**

1. **Self-contained & pinned.** Each figure is a standalone **PEP 723** Python
   file whose sole pinned dependency is the released engine —
   `dependencies = ["rheedium==<release>"]` — so a download + `uv run` reproduces
   the figure byte-for-byte, no environment setup. Plotting extras (matplotlib
   etc.) arrive transitively with `rheedium`, keeping the dependency list minimal.
2. **One file per figure.** `figures/fig01_architecture.py`,
   `figures/fig05_inverse_geometry.py`, … each `uv run`-runnable on its own,
   emitting a deterministic artifact (PDF/PNG/SVG) into `figures/out/`. Mirrors the
   automaton template — typed, seeded, machine-runnable.
3. **Data travels with the repo.** Input data (synthetic ground-truth benchmark,
   the ≥1 experimental RHEED dataset, reference patterns) lives in `data/`,
   checksummed; large files via Git LFS or a checksum-verified fetch script — never
   an unpinned remote pull.
4. **`.tex` sources included.** The full manuscript (`paper.tex`, `refs.bib`,
   class/style files) lives in the repo; figures are included from `figures/out/`.
5. **One-command build.** A top-level driver (`Makefile` / `build.py`) runs every
   `figures/*.py` then compiles the PDF — `make` (or `uv run build.py`) reproduces
   the entire manuscript end-to-end.
6. **Reproducibility metadata.** Deterministic seeds, the pinned `rheedium`
   version, and `jax_backend` recorded per figure (the same contract the
   automatons already emit); the repo is archived to a DOI (Zenodo) at submission.

**Layout sketch**

```
rheedium_paper/
  paper.tex  refs.bib  <class/style files>
  figures/
    fig01_architecture.py        # PEP 723: deps = ["rheedium==<release>"]
    fig05_inverse_geometry.py
    ...
    out/                         # generated artifacts (git-ignored or committed)
  data/                          # inputs + checksums (LFS / fetch script)
  build.py | Makefile            # run all figures -> compile PDF
  README.md                      # `uv run build.py` and you have the paper
```

**Relationship to the rest of the ecosystem**

- Same PEP 723 / single-`rheedium`-dependency contract as
  [automatons](plans/future/automatons_plan.md); a figure script may simply *be* a
  thin caller of the public API (or reuse an automaton's emitted result JSON /
  artifacts), so figure code and experiment code don't diverge.
- Pins to the **same release** the paper cites, so the figures and the claimed
  results can never drift from the version under review.
- Deferred until the paper's results exist; tracked here so the figure scripts are
  written reproducibly *as the results land*, not reconstructed afterward.

**Role at submission — the Code & Data Availability artifact**

`rheedium_paper` *is* the repo submitted under the journal's **Code and Data
Availability** statement (mirrored to a Zenodo DOI). The reproducibility chain is
fully public and permanent, with no "available on request" gap:

1. **Engine** — `rheedium` is open source on **PyPI + GitHub** (MIT), so the
   pinned `rheedium==<release>` resolves for anyone, anywhere, indefinitely.
2. **Paper repo** — `rheedium_paper` carries the figure scripts, data, and `.tex`,
   archived to a DOI at submission (citable, immutable).
3. **One command** — a reviewer or reader runs `make` / `uv run build.py` and
   regenerates every figure and the PDF from the pinned engine + shipped data.

So the availability statement names two concrete, downloadable artifacts (the PyPI
release / GitHub tag, and the archived paper repo) rather than a promise.

---

## Open framing decisions

1. **~~Venue tone~~ — settled.** Target *npj Computational Materials*; main text =
   physics/maths/capability, architecture/implementation → SI; 10 main + 10–20 supp
   figures. (See [Editorial framing](#editorial-framing-npj-comp-mater).)
2. **Kinematic-vs-dynamical honesty.** State the scope plainly. Multislice is now
   selectable via `kernel=` (a real dynamical *forward* kernel), but the *inverse*
   results are kinematic-first — do not imply a validated dynamical inversion until
   one exists.
3. **Inverse-results bar.** Synthetic ground truth alone (faster, weaker) vs
   synthetic + ≥1 experimental inversion (stronger, dataset-gated). Recommend the
   latter for a top-tier methods venue.
4. **Distribution-reconstruction emphasis.** Headline the identifiability story
   (novel, distinctive) vs treat it as one result among several — affects title
   and abstract framing.
