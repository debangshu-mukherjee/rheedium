# Differentiable RHEED Inversion with `optimistix` + `optax` — the `recon` solver

Scope: `rheedium.recon` — turn the inverse problem into a **general,
`optimistix`-based differentiable solver** (with `optax` as the
gradient-transform / schedule layer it composes with): given a measured RHEED
pattern (or set), recover the
latent parameters that produced it — structure / composition / thickness, beam
coherence (GSM `β`), defect statistics (twin density, grain size, orientation
spread), and instrument nuisances — **and, crucially, the full probability
distributions over those latents** (the orientation spread, the grain-size
distribution, the beam coherence), not merely point values — by gradient descent
through the differentiable forward model. **Reconstructing the distributions
themselves is a key goal** (§2.1), and it is the natural endpoint of a forward
model that is *literally a function of a `Distribution`*. This is the realization
of the framework's Phase-6 "inverse problem" sketch and the autonomous lab's
**Loop C** (the differentiable payoff); the automatons' `invert_structure` /
`recipe_deviation` import this module's API.

Status: **complete** — KG1-KG6 closed. **Roadmap position:** third of four —
[framework](../implemented/distribution_framework_plan.md) →
[rationalization](../implemented/rationalization_refactor_plan.md) → *this* →
[automatons](../future/automatons_plan.md). Entry gate **K0** is satisfied by
the completed rationalization refactor (⇒ the forward model differentiates
end-to-end against a clean, stable API). The KG1-KG6 implementation is complete in
`recon`: transforms, richer losses, lattice/Wyckoff constraints, finite-gradient
guards, the `ReconProblem`/`solve` surface, multistart, the incoherent
distribution library path, crystal-backed displacement-axis reconstruction,
parametric spread fitting, seeded basin-escape robustness, generalized
Fisher/Laplace UQ and the `blackjax` posterior sampler, schema-validated
`recipe_deviation` reports with default K4 covariance, cache-aware inversion,
the geometry/beam convenience wrapper, orientation fitting through the shared
solver, docs, and a frozen automaton import surface. The hand-rolled
`recon/optimizers.py` surface has been deleted outright with no shim; migration
is pinned by `solve` regression tests and the `CHANGELOG.md` note. This plan
**supersedes** the framework's Phase-6 line and now records the closed single
implementation.

**Post-completion refinement (types centralization).** Per the CONTRIBUTING
*Custom Types and PyTrees* (all types in `rheedium.types`) and *export-once* rules,
the recon PyTrees and their constructors — `ReconProblem`, `ReconResult`,
`DistributionAxisSpec`, `LaplaceUncertainty`, `PosteriorSamples`,
`RecipeDeviationReport`, `OrientationFitResult`, plus `create_distribution_axis_spec`
/ `create_crystal_displacement_axis_spec` — were moved out of the `recon`
submodules into `src/rheedium/types/recon_types.py`. The `recon` modules now import
them from `rheedium.types`; `recon/__init__` exports the inverse-problem
*functions* (not the type classes), and a guard test asserts `rheedium.recon` no
longer carries the carriers. No behavior change — `ty`/`ruff` clean, suite green.

Guard on every phase: **`jax.grad` flows end-to-end, tests stay green, results are
reproducible (seeded), and solver-migrated functions are verified to preserve the
old numerical behavior by regression test before the hand-rolled path is deleted —
no shim.** Per the rationalization plan's zero-legacy policy, the retired
optimizers are **deleted outright** (no `DeprecationWarning`, no alias); the only
migration surface is a `CHANGELOG.md` note. Recon is the one consumer that makes
differentiability pay off — a welded seam anywhere upstream is caught here.

---

## 1. Why optimistix + optax, and what already exists

`recon` used to hand-roll its optimization:

- `optimizers.py` — bespoke `adam_optimize`, `adagrad_optimize`,
  `gauss_newton_least_squares` (+ `*_reconstruction` wrappers,
  `ReconstructionResult`) — now deleted with no shim after the KG2 regression
  guard.
- `orientation.py` — originally a full orientation-weight fitter with its own
  Adam (`_adam_update`, `_OrientationAdamState`); now migrated so
  `fit_orientation_weights` is a wrapper over the shared
  `ReconProblem`/`solve` stack while preserving `orientation_loss`,
  `compute_fisher_information`, and `estimate_weight_uncertainty`.
- `losses.py` — `weighted_image_residual`, `weighted_mean_squared_error` (+
  `checked_*`).

This is real machinery, but the legacy path is (a) **hand-rolled**
(re-implementing Adam and Gauss–Newton is maintenance the ecosystem already
solved), (b) **orientation-specific** (the only old end-to-end fitter is
`fit_orientation_weights`), and (c) missing the composable extras —
trust-region / line-search, quasi-Newton, learning-rate schedules, gradient
clipping, multi-start — a non-convex RHEED inverse problem needs. The new
foundation (`transforms.py`, extended `losses.py`, `solve.py`, `uncertainty.py`,
`deviation.py`) is the **single source of truth**. The legacy `optimizers.py`
retirement is complete, and `orientation.py` no longer carries a second Adam
implementation.

The RHEED inverse is, at its core, a **nonlinear least-squares** fit of a forward
model to a measured pattern, whose answer is reported as a **posterior**. Three
libraries split cleanly along that grain — **all now core dependencies**:

- **`optimistix` owns the solve** — it is the JAX nonlinear-solver library
  (Levenberg–Marquardt, Gauss–Newton, Dogleg least-squares; BFGS / NonlinearCG
  minimisation; root-find; fixed-point), with convergence criteria, line search
  and trust region built in. It is the natural replacement for the hand-rolled
  `gauss_newton_least_squares`, it is from the **equinox family** (rheedium already
  depends on `equinox`, so PyTree interop is seamless), and its Gauss–Newton/LM
  step forms the residual Jacobian `JᵀJ` — the **Laplace approximation that
  warm-starts** the posterior sampler.
- **`optax` owns the gradient layer** — composable `GradientTransformation`s,
  LR schedules, clipping; the descent `optimistix.minimise` can wrap; and the home
  for the stochastic / high-dimensional and free-form-simplex (entropic) paths
  where a first-order method fits better than LM.
- **`blackjax` owns explicit posterior sampling** — HMC / NUTS over the
  differentiable log-posterior `∇ log p(θ│I)` the forward model already provides.
  The sampler (`sample_posterior`) and diagnostics are built and validated (KG4).
  The frozen public reporting policy is Laplace/Fisher by default for fast bands
  and z-scores, with blackjax sampling available when a caller needs the full
  posterior. The Laplace `JᵀJ` also remains the sampler's warm-start / proposal
  mass. It is pure JAX (jit/vmap-able), so chains `vmap` across the multistart
  inits and share the compilation cache.

The plan therefore *rationalizes and generalizes* recon rather than greenfielding
it: keep the good ideas (reparameterization, Fisher UQ), swap the hand-rolled
Gauss–Newton/Adam for `optimistix`/`optax`, and lift the orientation-only fitter
to an arbitrary latent space.

---

## 2. Architecture

```
ReconProblem (eqx.Module; owner: `rheedium.types`)
    forward:   latent -> Distribution/sim -> Float[H, W]   (the differentiable kernel)
    measured:  Float[H, W]   (or a set)
    transform: unconstrained latent <-> physical params (bijectors)
    loss:      (sim, measured) -> scalar   (data fidelity + priors)

solve(problem, solver, init) -> ReconResult (owner: `rheedium.types`)
    - optimistix.least_squares / minimise  (LM · Gauss-Newton · BFGS)
        · residual = loss ∘ forward ∘ transform ; JᵀJ from the LM/GN step
        · convergence / line-search / trust-region handled by optimistix
        · optax GradientTransformation as the descent where a first-order
          step fits (free-form weights, stochastic / high-dim)
ReconResult: fitted params, loss history, convergence, covariance (UQ = JᵀJ)

multistart(problem, k_inits) -> best ReconResult
reconstruct_distribution(measured, base, axis_spec) -> (Distribution, band)
recipe_deviation(problem, intended) -> per-param gap + z-score + severity
```

Three layers, each independently testable: **(1)** a reparameterization +
differentiable-loss foundation, **(2)** an `optimistix` solver core (LM /
Gauss–Newton, with `optax` as the gradient layer), **(3)** robustness
(multistart), uncertainty, and the recipe-deviation contract on top.

### 2.1 Distribution reconstruction — the key capability

Because the forward model is **literally a function of a `Distribution`** —
`I = apply(Distribution(samples, weights), kernel)` — inverting it recovers the
*probability distribution* that produced the data, not just a point estimate.
This is the deepest capability of the framework + recon combination, in three
senses:

1. **Parametric** — each producer *is* a parameterized distribution
   (`OrientationDistribution` peaks+spreads, `SizeDistribution` lognormal params,
   `BeamModeDistribution` GSM `β`/divergence). Fitting its parameters (the solver
   core) *is* reconstructing that physical distribution. `fit_orientation_weights`
   already does this for orientation; the plan generalizes it to every producer.
2. **Free-form (non-parametric)** — fix a latent grid as the `samples` and fit the
   `weights` **simplex** directly (the reparameterization layer's `softmax`
   bijector) under an entropy / smoothness / sparsity prior, recovering an
   *arbitrary* distribution shape with no family assumption. This needs **no new
   solver** — it is `solve` over the weight vector.
3. **Posterior (a distribution over the answer) — available when needed.** The
   differentiable model yields `p(distribution │ data)`, and recon can sample it
   with `blackjax` NUTS, warm-started from the `optimistix` Laplace/Fisher mode,
   with chains `vmap`-ed across multistart inits so a multimodal posterior (e.g.
   the symmetry orbit) is captured rather than collapsed to one Gaussian. The
   frozen default output for fast public bands and recipe z-scores is
   Laplace/Fisher; callers opt into `sample_posterior` when they need full
   posterior samples.

**The `reduction` flag makes this tractable** — and cleanly bifurcated:

- **Incoherent** distributions reduce as `I = Σ_n w_n |A(sample_n)|²` — **linear in
  the weights**. Reconstructing them is a **convex non-negative linear inverse
  problem** (NNLS / maximum-entropy deconvolution) over the per-sample intensity
  library `{|A(sample_n)|²}`: well-posed with regularization, fast, unique up to
  the operator's null space. Covers orientation, size, grains, beam modes — a
  dedicated convex fast-path, not gradient descent.
- **Coherent** distributions reduce as `I = |Σ_n w_n A(sample_n)|²` — **nonlinear
  in the weights**: gradient descent, non-convex, needs multistart.

**Identifiability (the honest caveat).** The forward operator is a smoothing
average; distinct distributions can produce near-identical patterns (incoherent
sums discard phase). Reconstruction recovers the distribution *only up to the
resolution the data supports* (the operator's effective rank). The entropy/
smoothness prior fills the null space with the least-committal answer, and the
posterior (sense 3) reports *which* features of the distribution are actually
constrained — so recon returns the **identifiable distribution with error bands**,
never an over-confident shape.

#### From a base crystal + a measured pattern — the convex library route

The most direct, user-facing form of this capability: **pass a base crystal and a
measured pattern, get the distribution.** The base crystal plus a chosen
perturbation axis — Debye–Waller *B* / RMS displacement, orientation spread, domain
size, strain, or even a detector-plane shift grid — defines the per-sample
amplitude library `{Aₙ}`; `|Aₙ|²` is the incoherent intensity library. This
generic path is split between type ownership and solver behavior:
`rheedium.types.DistributionAxisSpec` describes the latent samples plus
perturbation/forward functions,
`build_incoherent_intensity_library(base, axis_spec)` builds `{|Aₙ|²}`,
`reconstruct_incoherent_weights(library, measured)` solves the convex weights, and
`reconstruct_distribution(measured, base, axis_spec) -> (Distribution, band)`
chains builder → convex solver → Fisher/Laplace weight band.

The remaining work is not the generic inverse machinery; it is the
physics-specific library builders and validation fixtures that instantiate
`axis_spec` for real crystals and simulator carriers: Debye–Waller *B* / RMS
displacement, orientation spread, domain size, strain, and detector-plane shift
grids.

This *is* a **deconvolution**: `I_measured = Σₙ wₙ |A(perturbationₙ)|²` is the base
intensity mixed by `w`, so the recovered `w` is the **mixing distribution — the
"convolution function" relating the base to the measured pattern.** Concretely,
*pristine vs higher-temperature Bi₂Se₃*: base = pristine, axis = thermal
displacement amplitude (Debye–Waller *B*), and `w(B)` is the distribution over
thermal displacement that turns pristine into hot. (It generalizes to *relate any
two patterns* against a shared base — `relate_patterns(base, I_a, I_b, axis)` is
just `reconstruct_distribution` run against each.)

**Physics caveat — temperature is not a shift-invariant convolution.** Debye–Waller
is a *q-dependent multiplicative damping* `exp(−M(q))` (it suppresses high-angle
reflections), plus thermal expansion (peak shifts) and a diffuse background — so
`I_hot = I_pristine ⊛ K` holds only approximately. Two honest, *same-solver*
choices for the axis: **(a) a physical latent** (DW-*B* / displacement / disorder)
→ `w` is a *meaningful physical distribution*; **(b) a detector-plane shift grid**
→ `w` is a *literal but phenomenological* blur kernel `K`. Pick (a) for physics,
(b) for a descriptor. Either way the identifiability caveat above holds: `w` is the
distribution that *explains the pattern*, recovered only up to the operator's null
space — not a unique statement of what physically happened (§2.3).

### 2.2 Canonical inverse problems (what you actually ask it)

The one `solve` instantiates a ladder of concrete problems, in increasing
difficulty and decreasing well-posedness — each the same `ReconProblem` with a
different *active* latent subset (the rest fixed or supplied as a prior):

1. **Geometry + beam from a known structure** — *given a measured RHEED pattern
   and a CIF, recover the orientation and beam parameters that generated it.*
   Structure is fixed from the CIF; fit the ~8 continuous latents — incidence
   `(θ, φ)` + azimuth/tilt and the beam `(energy, divergence_in/out, GSM
   β_in/out, energy spread)` as a `BeamModeDistribution`. **The most well-posed
   and fastest case** (few parameters, structure known) — the natural alignment +
   beam-metrology problem for the autonomous lab. The one wrinkle is **symmetry**:
   point-group-equivalent orientations are exact global optima, so seed
   `multistart` (K3) with the symmetry orbit. Convenience entry
   `fit_geometry_beam(crystal, measured) -> (orientation, BeamModeDistribution,
   covariance)`. **The canonical first milestone** and the KG2/KG3 exemplar.
2. **Distribution reconstruction** (§2.1) — recover the orientation spread, the
   grain-size distribution, or the beam coherence (parametric family or free-form
   weights), optionally jointly with the geometry of (1).
3. **Structure inversion** — fit atomic positions / composition / thickness
   (hardest; non-convex, seeded by the Loop-B `.xyz` bracket).
4. **Recipe deviation** (K5) — any of the above against an intended recipe → the
   per-parameter gap with significance.

The solver, uncertainty, and reporting are shared across all four; they differ
only in which latents are free.

### 2.3 Model identifiability — distribution vs mechanism

Inversion runs along a two-step chain, and identifiability fails *differently* at
each step:

```
material model ──(generative)──▶ distribution P(latent) ──(apply)──▶ pattern I
```

- **`I → P` (what recon does, §2.1).** Ill-posed but *quantifiably* so — bounded
  by the forward operator's effective rank; recovered with credible bands, and the
  UQ (K4) reports which features of `P` are actually constrained.
- **`P → M` (which material model generated the distribution).** Fundamentally
  **many-to-one**, and *not* from noise: `P` is a sufficient statistic for the
  *pattern*, not for the *mechanism*. RHEED sees a marginal / ensemble
  distribution, and distinct generative processes routinely share a marginal — a
  lognormal grain-size `P` is consistent with coalescence *or* a different
  nucleation law; an orientation spread with discrete mosaic blocks *or* a
  continuous misorientation field. The data alone cannot separate them.

So recon's honest output is **never a forced unique mechanism**: it reports the
recovered `P̂` (with bands) plus the **equivalence class of mechanisms consistent
with it** — a model posterior `p(M │ I)` that is *flat over the degenerate set*.
Asserting one mechanism where the posterior is flat is exactly the over-confidence
the error-bar discipline exists to prevent.

**Resolution — design the tie-breaking experiment.** Non-identifiability becomes
an *experiment-design objective* in the closed loop: if models `M_a`, `M_b` are
degenerate at the current condition but predict different distribution-*evolution*
under a perturbation Δ (temperature, flux, time), choose the Δ that maximizes the
predicted divergence — `argmax_Δ E[D_KL(P_a(Δ) ‖ P_b(Δ))]`, the expected
information gain. The differentiable forward model makes that gradient-computable,
so the agent does not merely *report* "cannot distinguish `M_a` from `M_b`" — it
**acts to break the tie** (the automatons' Loop C owns this action).

**Architecture.** A material model is a *generator of `Distribution`s* (a `procs`
hyper-producer: mechanism → `Distribution`). recon inverts the last two arrows
(`I → P`); the `P → M` arrow — mapping a recovered `Distribution` back to the
mechanism that built it — is the open, generically-ill-posed step. The framework's
gift is that `P` is a first-class, comparable object, so model comparison is a
well-defined question ("does `M_i`'s generated `Distribution` match `P̂`?") even
when its answer is "several do."

---

## 3. Gated phases

Each phase is one reviewable PR ending at a **gate**: an objectively checkable set
of conditions that must all hold before the next phase starts.

### Entry — Gate K0 (precondition for *any* work below)

The [rationalization refactor](../implemented/rationalization_refactor_plan.md) is
**complete** (its R0–R6 done), so the forward model (`simulate_detector_image` /
`apply`) differentiates end-to-end against the rationalized API (config carriers,
one reduction). A `jax.grad` of a forward pattern w.r.t. a producer's latent
params is finite (this is the framework's invertibility property, now relied on).
Until K0 holds this plan does not start.

### Universal gate (applies to *every* phase, on top of its specific gate)

- `jax.grad` / `jax.value_and_grad` through the touched path returns **finite
  gradients** (the silent failure mode is a welded seam upstream);
- **reproducible**: a seeded run reproduces its result bit-for-bit;
- **migration-safe, zero-legacy**: a replacement is pinned to the old result to
  tolerance by a **regression test**, then the hand-rolled path is **deleted in the
  same PR** — no `DeprecationWarning`, no shim, no alias; the rename is recorded in
  `CHANGELOG.md` (the refactor plan's §3 zero-legacy policy). No two optimizer
  implementations ever ship together;
- suite green; `ty` / `ruff` clean; full `@jaxtyped(typechecker=beartype)`.

### Phase K1 — dependencies + reparameterization & loss foundation

*Tasks:*
- **Implemented:** `optimistix`, `optax`, and `blackjax` are core dependencies
  (`pyproject.toml`), and local import/smoke coverage exists through the recon
  solver tests.
- **Implemented:** `recon/transforms.py` — a **general** latent↔physical bijector layer
  (generalize `orientation.py`'s softmax/softplus): positivity (`softplus`),
  bounded `[a,b]` (`sigmoid`-scaled), simplex/occupancy (`softmax`), and
  lattice/Wyckoff constraints — so the solver optimizes in unconstrained space and
  the physical bounds are enforced *by construction* (no projected-gradient hacks).
- **Implemented:** `recon/losses.py` extended into a **differentiable loss library**: pixelwise
  `L2`/Huber, **log-intensity** (RHEED spans orders of magnitude), normalized
  cross-correlation (scale-invariant), with **analytic scale/background
  marginalization** so intensity calibration is not a fit parameter; composable
  priors/regularizers (bounds, smoothness, defect sparsity, and **maximum-entropy
  / smoothness priors over `Distribution.weights`** for the free-form distribution
  reconstruction of §2.1 — the regularizer that fills the ill-posed null space).
- **Implemented:** real-carrier finite-gradient checks now exercise a bounded
  latent through `CrystalStructure` displacement, the public `ReconProblem` loss
  surface, and differentiable forward intensities. Lattice/fractional/Wyckoff
  transforms are covered for round-trip behavior where bijective and finite
  gradients where constrained.

**Gate KG1: closed.** Every loss and every transform is differentiable (finite grad);
each bijector round-trips (`physical = fwd(inv(physical))` to tolerance); a planted
loss has its minimum at the true parameters (1-D sanity sweep); `optimistix` and
`optax` both run a step on a toy objective; universal gate.

### Phase K2 — the `optimistix` solver core (`solve`)

*Tasks:*
- **Implemented:** `types/recon_types.py` owns `ReconProblem` (eqx.Module) and
  `ReconResult`; `recon/solve.py` owns `solve(problem, solver, init) ->
  ReconResult`. The default solver is **`optimistix.LevenbergMarquardt` /
  `GaussNewton`** for the least-squares residual (it owns convergence, line
  search, trust region); `optimistix.BFGS` for general minimisation; and an
  **`optax` first-order path** (`optax.chain` of `clip_by_global_norm` + scheduled
  `adamw`) for the stochastic / high-dim / free-form-weight cases — wrapped via
  `optimistix.minimise(..., optax_solver)` so it shares the same `solve` surface.
  Bounded `max_steps` gives the JIT-friendly fixed-iteration "rapid" path; eager
  early-stopping is the other mode.
- **Implemented (legacy retirement, zero-cruft):** KG2's regression guard pins
  `solve` to the old deterministic linear fixtures, then
  `adam_optimize` / `adagrad_optimize` / `gauss_newton_least_squares`, their
  `*_reconstruction` wrappers, and `ReconstructionResult` were deleted
  outright — no shim, no forwarding alias — and dropped from
  `recon/__init__` exports. `CHANGELOG.md` records the migration to
  `ReconProblem` + `solve`. `optimistix` LM/GN is the replacement; there is no
  second optimizer.
- **Implemented foundation:** **Distribution reconstruction (§2.1).** `solve` over `Distribution.weights`
  (with the K1 simplex bijector + entropy prior) recovers a *free-form* shape. The
  **convex fast-path** for incoherent distributions — `I = Σ_n w_n |A_n|²`, linear
  in the weights, solved as NNLS / maximum-entropy over `{|A_n|²}` — exists as
  `reconstruct_incoherent_weights`; `types/recon_types.py` owns
  `DistributionAxisSpec`, while `build_incoherent_intensity_library` and
  `reconstruct_distribution(measured, base, axis_spec) -> (Distribution, band)`
  provide the generic "base object + pattern → distribution" route.
- **Implemented:** `types/recon_types.py:create_crystal_displacement_axis_spec`
  instantiates a physics-specific `DistributionAxisSpec` for real
  `CrystalStructure` carriers.
  K2 tests now cover planted physical-axis weight recovery, parametric spread
  recovery via `solve`, fixed-step vs longer-solve equivalence, the warmed
  wall-clock budget, and hand-rolled optimizer retirement guards. The coherent
  case stays on nonlinear `solve` plus multistart.

**Gate KG2: closed.** On synthetic data simulated from a *known* structure,
`solve` (LM/GN) recovers the parameters to tolerance with a monotone loss;
the bounded fixed-step result == the eager result; the new path reproduces the
retired hand-rolled optimizers' results to tolerance (regression guard); a
**wall-clock budget** assertion on the reference problem (the "rapid" claim);
**a planted free-form distribution shape is recovered** — incoherent weights via
the convex path, a parametric spread (e.g. a lognormal size distribution) via
`solve` — to tolerance; a **base-crystal self-consistency** check —
`reconstruct_distribution(measured, base_crystal, axis)` recovers a *planted*
distribution over a physical axis (e.g. hot Bi₂Se₃ simulated at a known
Debye–Waller *B*; the recovered `w(B)` peaks at the true *B*) to tolerance;
universal gate.

### Phase K3 — Robustness: multistart + bracket-then-refine

*Tasks:*
- **Implemented:** `multistart(problem, initial_latents)` accepts a leading start
  axis, runs the common `solve` surface, and returns the best result by final
  loss. For K3 it also accepts a template latent plus `key`/`n_starts` to generate
  reproducible random starts around the template.
- **Implemented:** planted local-minimum escape regression: a finite-budget
  AdamW single start remains in the higher-loss basin, while multistart selects
  the lower-loss basin.
- **Implemented:** Loop-B → Loop-C handoff fixture: a coarse bracketed start for
  a nonlinear least-squares problem converges in strictly fewer reported solver
  steps than a cold start.

**Gate KG3: closed.** `multistart` escapes a *planted* local minimum that a single start
provably falls into (the canonical case: a symmetry-equivalent orientation in the
§2.2-#1 geometry+beam problem); a bracketed init converges in strictly fewer steps
than a cold start; reproducible across seeds; universal gate.

### Phase K4 — Uncertainty quantification (generalized)

*Tasks:* **Implemented:** generalized `orientation.py`'s
`compute_fisher_information` /
`estimate_weight_uncertainty` to **arbitrary** latent parameters —
Gauss–Newton / Fisher covariance from the residual Jacobian. `uncertainty.py`
provides `fisher_information_from_residual`, `covariance_from_fisher`, and
`laplace_uncertainty`; `reconstruct_distribution` returns a linearized weight
band for the recovered distribution. KG4 now adds the posterior-first layer:
`rheedium.types.PosteriorSamples`, `sample_posterior` (blackjax NUTS),
`posterior_from_samples`, and `laplace_inverse_mass_matrix`. NUTS runs over a
differentiable log-posterior, can be warm-started from the Laplace/Fisher `JᵀJ`
precision, and treats rows of a flattened initial-position array as K3
multistart chains so multimodal posteriors are represented rather than collapsed.
Diagnostics include R-hat, ESS, acceptance rate, posterior covariance, and
equal-tailed credible intervals. Tests calibrate Laplace covariance against the
empirical spread of many noisy synthetic realizations and reduce the generic
Fisher helper to the existing orientation UQ by regression. The Laplace Gaussian
stays as the warm-start / fast approximation and the frozen default for public
bands/z-scores; `sample_posterior` remains the explicit full-posterior path when
a caller needs non-Gaussian or multimodal uncertainty.

**Gate KG4: closed.** The **`blackjax` sampler converges** (R-hat ≈ 1, adequate ESS) and
recovers a multimodal posterior where one exists; the **posterior credible band
matches the empirical** parameter spread over many noisy synthetic realizations to
tolerance; the Laplace warm-start covariance is finite + positive semi-definite and
agrees with the posterior in the unimodal case; it reduces to the existing
orientation UQ by regression; on a free-form reconstruction the **posterior** band
on the recovered shape is calibrated; universal gate.

### Phase K5 — the recipe-deviation contract (automaton-facing)

*Tasks:* **Implemented:**
`recipe_deviation(problem, intended_recipe, initial_latent) -> report` inverts the
measured pattern, then computes the **per-parameter gap** between the LLM agent's
intended recipe and the inverted reality. It uses supplied uncertainty when
provided and otherwise defaults to K4 Laplace covariance in the fitted physical
parameter basis. The report carries covariance, one-sigma denominators,
uncertainty source, thresholds, and flattened parameter labels. The frozen
automaton payload is committed as `recipe_deviation_report.schema.json` and
served by `recipe_deviation_report_payload`,
`recipe_deviation_report_schema`, and `validate_recipe_deviation_report`. Tests
calibrate deliberately mismatched and matched noisy recipes. This is the exact
object the automatons' `recipe_deviation.py` emits and `invert_structure.py`
builds on.

**Gate KG5: closed.** On a synthetic pattern grown from a structure *deliberately
mismatched* from a stated intended recipe, the report flags the right parameters
with correct sign and magnitude and a **calibrated** significance (a matched
recipe yields z-scores within noise); the report validates against a committed
schema; universal gate.

### Phase K6 — Speed, docs, API freeze

*Tasks:* **Implemented:** `solve` and `multistart` accept
`compilation_cache_dir` / `compilation_cache_per_arch` and route that request
through `tools.enable_compilation_cache` before the optimizer scan lowers; the
warm-cache path is covered by a direct solve-level cache regression and the
reference wall-clock test. `fit_geometry_beam` is exported as the §2.2
geometry+beam convenience wrapper, returning fitted orientation, fitted
`BeamModeDistribution`, and Laplace covariance for the physical pytree.
`fit_orientation_weights` now builds a `ReconProblem` and calls the shared
`solve` path; the bespoke `_OrientationAdamState` / `_adam_update` stack is
gone. The "Differentiable inversion with optimistix" guide is under
`docs/source/guides/`, linked from the guides index and Sphinx toctree. The
automaton-facing inverse API is exported from `recon/__init__`, Routine-Listed,
PEP-561 typed, and frozen by `test_api_freeze.py`; that test is included in the
Test Reference.

**Gate KG6: closed.** Docs build (`make html`) succeeds; the reference inversion
meets the warm-cache wall-clock budget; the public inverse API is exported,
typed, Routine-Listed, and covered by a frozen import-surface test; universal
gate.

### Gate summary

| Gate | Pass condition (+ universal gate) |
|------|------------------------------------|
| **K0** | rationalization complete; forward model differentiates end-to-end |
| **KG1** | **closed** — optimistix + optax import; bijectors round-trip; losses differentiable; planted-loss minimum correct; lattice/Wyckoff finite-grad guards |
| **KG2** | **closed** — `solve` (LM/GN) recovers a known structure; fixed-step == eager; new path == retired hand-rolled (regression); within wall-clock budget |
| **KG3** | **closed** — multistart escapes a planted local min; bracketed init converges faster; seeded random starts are reproducible |
| **KG4** | **closed** — blackjax NUTS + R-hat/ESS + calibration + orientation regression + free-form band |
| **KG5** | **closed** — recipe-deviation flags the right params with calibrated z-scores; payload validates against committed schema |
| **KG6** | **closed** — docs build; budget met warm-cache; inverse API exported + frozen |

**Current checkpoint:** KG1-KG6 are closed. The constrained transform/loss
foundation, `optimistix`/`optax` solver core, distribution reconstruction paths,
physical crystal-backed fixture, regression guards, wall-clock check,
zero-legacy optimizer retirement, seeded multistart robustness, bracketed-refine
fixture, generalized posterior UQ, schema-validated recipe-deviation contract,
cache-aware solve path, geometry/beam convenience wrapper, orientation migration,
docs guide, and frozen automaton import surface have landed.

K1–K2 are the foundation (replace hand-rolled with optimistix/optax, prove
recovery); K3–K5
are what the autonomous lab actually needs (robustness, uncertainty, the control
signal); K6 hardens and freezes the surface for the automatons.

---

## 4. Loss & conditioning notes (the physics)

- **Dynamic range.** RHEED intensities span orders of magnitude; a raw-L2 loss is
  dominated by the specular peak. Default to **log-intensity or NCC**, with
  raw-L2 available; document the choice per problem.
- **Scale/background are nuisances, not unknowns.** Marginalize them analytically
  (closed-form optimal scale + affine background per evaluation) so the optimizer
  never wastes capacity on calibration — keeps the fit identifiable.
- **Reparameterize, don't project.** Bounds enforced by bijector (K1) keep the
  optimization unconstrained and smooth; projected gradients introduce
  non-differentiable kinks (a principle violation).
- **Non-convexity is real.** The forward map is many-to-one near symmetry; K3's
  multistart + the Loop-B discrete bracket are the mitigation, not a better local
  optimizer.

---

## 5. Risks

- **solver migration regressions.** Mitigation: KG2's regression guard pins the
  optimistix/optax path to the retired hand-rolled results **in the same PR that
  deletes them** (no shim window — the guard is the safety net, not a shim).
- **Local minima / identifiability.** Mitigation: K3 multistart + bracket-then-
  refine; report multimodality (K4 covariance flags flat directions).
- **UQ validity.** Gauss–Newton/Laplace assumes a near-minimum, locally-Gaussian
  posterior; document the assumption and the regime where it holds; KG4 calibrates
  it empirically rather than trusting it blindly.
- **Speed vs the growth cadence.** Mitigation: JIT-scan fixed-N loop + compilation
  cache + optional AOT (K6); the wall-clock budget is a gate, not an aspiration.
- **New dependencies.** `optimistix`, `optax`, and `blackjax` are all pure-JAX and
  lightweight (rheedium already ships `equinox`); the `automatons-smoke`
  ephemeral-env check (downstream) catches any packaging surprise.
- **Posterior-sampling cost.** MCMC runs the forward model per leapfrog step.
  Mitigation: warm-start from the Laplace mode (few adaptation steps), short NUTS
  chains `vmap`-ed across the multistart inits, and the compilation cache. A VI
  or AOT hot-loop artifact can still be added later without changing the frozen
  public API.
- **Differentiability regression from "cleanup".** Mitigation: the universal gate's
  finite-grad check is mandatory on every PR.

---

## 6. Diff surface

| Path | Change |
|------|--------|
| `pyproject.toml` | `optimistix` + `optax` + `blackjax` core deps (done; CPU + CUDA resolve) |
| `src/rheedium/recon/transforms.py` | present — general bijector / reparameterization layer including lattice, fractional, and Wyckoff constraints |
| `src/rheedium/types/recon_types.py` | present — `ReconProblem`, `ReconResult`, `DistributionAxisSpec`, `LaplaceUncertainty`, `PosteriorSamples`, `RecipeDeviationReport`, `OrientationFitResult`, and reconstruction-specific type constructors |
| `src/rheedium/recon/solve.py` | present — cache-aware `solve` (`optimistix` LM/GN, BFGS, optax AdamW descent), seeded/cache-aware `multistart`, `fit_geometry_beam`, `build_incoherent_intensity_library`, `reconstruct_incoherent_weights`, and `reconstruct_distribution` |
| `src/rheedium/recon/uncertainty.py` | present — Fisher/Laplace covariance from residual Jacobians; `blackjax` posterior sampling (`sample_posterior`), R-hat/ESS diagnostics, credible intervals, Laplace inverse-mass warm start, empirical calibration, orientation regression, and free-form band guard |
| `src/rheedium/recon/deviation.py` | present — `recipe_deviation` with default K4 Laplace covariance, frozen automaton payload, schema loading/validation, and calibrated matched/mismatched tests |
| `src/rheedium/recon/optimizers.py` | deleted — hand-rolled Adam/Adagrad/Gauss-Newton + `*_reconstruction` + `ReconstructionResult` retired with no shim; `optimistix` LM/GN is the only optimizer |
| `src/rheedium/recon/orientation.py` | present — `fit_orientation_weights` is migrated onto the shared `ReconProblem`/`solve` path; bespoke `_OrientationAdamState`/`_adam_update` deleted; orientation loss and Fisher helpers remain as public compatibility and regression surfaces |
| `src/rheedium/recon/losses.py` | present — log-intensity, NCC, analytic scale/background marginalization, priors |
| `src/rheedium/recon/__init__.py` | present — inverse behavior API exports + Routine Listings frozen for automatons; type carriers are exported from `rheedium.types`; legacy optimizer exports removed |
| `tests/.../test_recon/*` | present — focused synthetic coverage for transforms, losses, solve, cache-aware inversion, `fit_geometry_beam`, distribution reconstruction, posterior UQ, schema-validated recipe deviation, API freeze, recovery-of-known-physical-params, fixed-step==eager, seeded multistart, local-minimum escape, bracketed-refine speedup, solver==hand-rolled regression, orientation migration, and wall-clock budget |
| `docs/source/guides/` | present — "Differentiable inversion with optimistix" guide linked from the guide index and Sphinx toctree |

---

## 7. Outcome

When complete: `recon` is a **general differentiable inverse-problem solver** —
one `optimistix`-backed `solve` (LM/Gauss–Newton, `optax` for the gradient layer)
over any latent space, multistart for robustness,
posterior credible bands throughout (`blackjax`), and a frozen `recipe_deviation` contract — replacing three
hand-rolled optimizers and the orientation-only fitter. It answers the full ladder
of §2.2: **orientation + beam from a CIF + pattern** (the fast, well-posed
alignment case), **probability-distribution reconstruction** (§2.1 — the spread,
size distribution, or beam coherence, parametric or free-form, *with credible
bands*), structure inversion, and recipe deviation. This is the engine the
automatons' Loop C calls to answer, in real time and with error bars, *what is
actually being grown and how far it is from intent* — the scientific payoff of the
entire differentiable architecture.
