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

## Target venues (in preference order)

- *npj Computational Materials* — methods + open differentiable-physics software.
- *Digital Discovery* (RSC) — differentiable/ML-for-characterization framing.
- Fallbacks: *Journal of Applied Crystallography*, *Computer Physics
  Communications* (software), *Ultramicroscopy* (electron-diffraction physics).

---

## One-sentence thesis

RHEED simulation can be written so that a single complex-amplitude kernel and a
single `Distribution` reduction express every multimodal, statistical, and defect
effect *and* keep `jax.grad` flowing end-to-end — making the inverse problem
(structure / orientation / beam / probability-distribution recovery) a direct
gradient-based fit rather than a bespoke optimizer.

## Contributions (the claim list)

1. **An architectural inversion of the forward model.** Push `|·|²` to the bottom
   (Layer-0 complex amplitude kernels), collapse the simulator to a thin
   `apply(distribution, kernel)` integrator (Layer 1), and express beam modes /
   orientation / size / twins / steps / grains as producers emitting one
   `Distribution` PyTree (Layer 2). [framework §1–§4]
2. **A physically-grounded coherent/incoherent partition.** The reduction mode is
   *computed* from the feature length vs the beam per-mode coherence length
   `ℓ_c` — coherent inside `|·|²`, incoherent outside — not a user switch.
   [framework §5]
3. **A mixed-state (Gaussian–Schell) beam model for RHEED.** Wolf coherent-mode
   decomposition with geometric GSM eigenvalues, made *anisotropic* by grazing
   incidence (`1/sin θ` footprint foreshortening) — strictly more expressive than
   a shift-invariant Gaussian blur. [framework §4.1]
4. **A general differentiable inverse solver (`recon`).** `optimistix`
   Levenberg–Marquardt / Gauss–Newton as the primary solver with `optax` as the
   gradient/schedule layer; reparameterize-don't-project bijectors; Gauss–Newton
   `JᵀJ` reused for Fisher/Laplace uncertainty. [recon §1–§2, K1–K4]
5. **Probability-distribution reconstruction + an identifiability analysis.**
   Recover a *distribution* (orientation spread, grain-size, beam coherence) —
   parametric or free-form — with credible bands; the convex incoherent fast-path
   (NNLS over `{|Aₙ|²}`); and the `model → distribution → pattern` identifiability
   hierarchy (invert the distribution; the generating *mechanism* is many-to-one).
   [recon §2.1–§2.3]
6. **Differentiable-programming engineering for physics.** Persistent compilation
   cache, AOT StableHLO export (`jax.export`; symbolic atom count; FFT
   non-polymorphism caveat), an opt-in unchecked fast path, and a typed two-tier
   (jaxtyping + beartype) validation discipline. [framework §3.5, recon K6, tools]
7. **A rigor ladder for trust.** `audit` invariants that *label* proof vs
   sampled-within-tolerance, plus property-based (`hypothesis`) round-trip and
   invariant tests — so every claim states what is guaranteed.
   [audit plan, hypothesis plan]

## Figure plan

1. **Architecture** — the three-layer inversion (Layer 0 amplitude → Layer 1
   `apply` → Layer 2 producers); the identity element at `N=1`.
2. **Coherent vs incoherent** — two displaced reflections within vs beyond `ℓ_c`:
   fringes vs flat sum; the kernel-level interference test.
3. **Mixed-state beam** — GSM mode count vs `β`; anisotropic in-plane vs
   out-of-plane streak FWHM; vs the legacy isotropic-Gaussian limit.
4. **Forward validation** — simulated patterns vs reference (kinematic +
   multislice), per-reflection amplitude/intensity parity.
5. **Inverse — geometry/beam** — recover orientation + beam params from a CIF +
   pattern; multistart resolving the symmetry orbit; covariance ellipses.
6. **Inverse — distribution reconstruction** — planted vs recovered distribution
   shape with credible band; convex incoherent fast-path vs nonlinear coherent.
7. **Identifiability** — two distinct mechanisms producing the same distribution;
   what the data can and cannot separate.
8. **Performance** — compile-cache warm/cold, AOT export, batched sweep scaling
   (`distribute_batched`); inversion wall-clock.

## Section skeleton

1. Introduction — RHEED's information content; why forward sims are siloed and
   inverse analysis is manual; the differentiable opportunity.
2. The forward model — Layers 0/1/2; the `Distribution` contract; the
   coherent/incoherent partition; the GSM beam.
3. The inverse problem — `recon` solver; reparameterization; UQ; distribution
   reconstruction; identifiability.
4. Implementation — JAX/equinox; AOT + cache; validation discipline; the rigor
   ladder.
5. Results — forward validation, inverse recovery on synthetic ground truth, and
   ≥1 experimental pattern; performance.
6. Discussion — scope/limits (kinematic vs dynamical; what identifiability
   forbids); relation to ptychography mixed-state methods.
7. Code & data availability — PyPI release, pinned version, archived DOI.

---

## Readiness — what must land before submission

| Component | Hard prerequisites | Current status |
|---|---|---|
| **Forward half** | framework Phases 1–6 **complete**: kernel-polymorphic integrator (multislice in `kernel=`), unified `DetectorGeometry`, the §6 differentiability guarantee | substantially landed — kinematic path **is inverted**; interference + defect-distinguishability tests now exist; remaining: kernel polymorphism, multislice-in-`kernel=`, `DetectorGeometry` split |
| **Inverse half** | `recon` K0–KG6: solver, distribution reconstruction, identifiability, UQ on synthetic ground truth | specified, **not built** (gated after rationalization) |
| **Rigor** | `audit` proof/sampled labelling + `hypothesis` round-trip/invariant tests | planned |
| **Evidence** | ≥1 experimental RHEED dataset inverted end-to-end | to source |

**Recommended writing order**

1. Finish framework Phases 1–6 (kernel polymorphism + multislice in `kernel=` +
   the differentiability guarantee) → draft the *forward* results.
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

1. **Venue tone.** Lean *npj Comp Mater* (physics/software) vs *Digital Discovery*
   (ML-for-characterization) — changes emphasis (physics validation vs
   differentiable-inversion novelty).
2. **Kinematic-only honesty.** State the kinematic-vs-dynamical scope plainly;
   multislice is a defined Layer-0 slot, not a finished dynamical inverse — do not
   imply dynamical inversion.
3. **Inverse-results bar.** Synthetic ground truth alone (faster, weaker) vs
   synthetic + ≥1 experimental inversion (stronger, dataset-gated). Recommend the
   latter for a top-tier methods venue.
4. **Distribution-reconstruction emphasis.** Headline the identifiability story
   (novel, distinctive) vs treat it as one result among several — affects title
   and abstract framing.
