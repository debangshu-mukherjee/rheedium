# Differentiable RHEED Inversion with `optax` — the `recon` solver

Scope: `rheedium.recon` — turn the inverse problem into a **general, optax-based
differentiable solver**: given a measured RHEED pattern (or set), recover the
latent parameters that produced it — structure / composition / thickness, beam
coherence (GSM `β`), defect statistics (twin density, grain size, orientation
spread), and instrument nuisances — by gradient descent through the differentiable
forward model. This is the realization of the framework's Phase-6 "inverse problem"
sketch and the autonomous lab's **Loop C** (the differentiable payoff); the
automatons' `invert_structure` / `recipe_deviation` import this module's API.

Status: **proposed** — gated. **Roadmap position:** third of four —
[framework](plans/partial/distribution_framework_plan.md) →
[rationalization](plans/future/rationalization_refactor_plan.md) → *this* →
[automatons](plans/future/automatons_plan.md). Entry gate **K0**: the
rationalization refactor is complete (⇒ the forward model differentiates
end-to-end against a clean, stable API). It **supersedes** the framework's
Phase-6 line and **replaces** the existing hand-rolled `recon` optimizers.

Guard on every phase: **`jax.grad` flows end-to-end, tests stay green, results are
reproducible (seeded), and optax-migrated functions preserve the old numerical
behavior behind a deprecation shim.** Recon is the one consumer that makes
differentiability pay off — a welded seam anywhere upstream is caught here.

---

## 1. Why optax, and what already exists

`recon` today hand-rolls its optimization:

- `optimizers.py` — bespoke `adam_optimize`, `adagrad_optimize`,
  `gauss_newton_least_squares` (+ `*_reconstruction` wrappers, `ReconstructionResult`).
- `orientation.py` — a full orientation-weight fitter with its own Adam
  (`_adam_update`, `_OrientationAdamState`), softmax/softplus reparameterization,
  `orientation_loss`, `compute_fisher_information`, `estimate_weight_uncertainty`.
- `losses.py` — `weighted_image_residual`, `weighted_mean_squared_error` (+
  `checked_*`).

This is real machinery, but it is (a) **hand-rolled** (re-implementing Adam is
maintenance the ecosystem already solved), (b) **orientation-specific** (the only
end-to-end fitter is `fit_orientation_weights`), and (c) missing the composable
extras — learning-rate schedules, gradient clipping, quasi-Newton (L-BFGS),
multi-start — that a non-convex RHEED inverse problem needs.

`optax` is the JAX-native answer: composable `GradientTransformation`s, schedules,
clipping, and a deep optimizer set (`adam`/`adamw`/`lbfgs`/`sgd`), all pure-JAX and
differentiation-friendly. **Adding it is the central move of this plan.** optax is
a lightweight, pure-Python+JAX dependency (no new native code).

The plan therefore *rationalizes and generalizes* recon rather than greenfielding
it: keep the good ideas (reparameterization, Fisher UQ), swap the hand-rolled
optimizers for optax, and lift the orientation-only fitter to an arbitrary latent
space.

---

## 2. Architecture

```
ReconProblem (eqx.Module)
    forward:   latent -> Distribution/sim -> Float[H, W]   (the differentiable kernel)
    measured:  Float[H, W]   (or a set)
    transform: unconstrained latent <-> physical params (bijectors)
    loss:      (sim, measured) -> scalar   (data fidelity + priors)

solve(problem, optimizer, n_steps, init) -> ReconResult
    - jax.value_and_grad(loss ∘ forward ∘ transform)
    - optax optimizer.update / apply_updates
    - lax.scan (fixed-N, JIT, the "rapid" path) | eager (early-stop) path
ReconResult: fitted params, loss history, convergence, covariance (UQ)

multistart(problem, k_inits) -> best ReconResult     (vmap / distribute_batched)
recipe_deviation(problem, intended) -> per-param gap + z-score + severity
```

Three layers, each independently testable: **(1)** a reparameterization +
differentiable-loss foundation, **(2)** an optax solver core, **(3)** robustness
(multistart), uncertainty, and the recipe-deviation contract on top.

---

## 3. Gated phases

Each phase is one reviewable PR ending at a **gate**: an objectively checkable set
of conditions that must all hold before the next phase starts.

### Entry — Gate K0 (precondition for *any* work below)

The [rationalization refactor](plans/future/rationalization_refactor_plan.md) is
**complete** (its R0–R6 done), so the forward model (`simulate_detector_image` /
`apply`) differentiates end-to-end against the rationalized API (config carriers,
one reduction). A `jax.grad` of a forward pattern w.r.t. a producer's latent
params is finite (this is the framework's invertibility property, now relied on).
Until K0 holds this plan does not start.

### Universal gate (applies to *every* phase, on top of its specific gate)

- `jax.grad` / `jax.value_and_grad` through the touched path returns **finite
  gradients** (the silent failure mode is a welded seam upstream);
- **reproducible**: a seeded run reproduces its result bit-for-bit;
- **migration-safe**: any optax replacement of a hand-rolled optimizer reproduces
  the old result to tolerance, or ships behind a `DeprecationWarning` shim (the
  refactor plan's §3 policy);
- suite green; `ty` / `ruff` clean; full `@jaxtyped(typechecker=beartype)`.

### Phase K1 — optax dependency + reparameterization & loss foundation

*Tasks:*
- Add `optax` to `pyproject.toml` dependencies; confirm it resolves CPU + CUDA.
- `recon/transforms.py` — a **general** latent↔physical bijector layer
  (generalize `orientation.py`'s softmax/softplus): positivity (`softplus`),
  bounded `[a,b]` (`sigmoid`-scaled), simplex/occupancy (`softmax`), and
  lattice/Wyckoff constraints — so optax optimizes in unconstrained space and the
  physical bounds are enforced *by construction* (no projected-gradient hacks).
- `recon/losses.py` extended into a **differentiable loss library**: pixelwise
  `L2`/Huber, **log-intensity** (RHEED spans orders of magnitude), normalized
  cross-correlation (scale-invariant), with **analytic scale/background
  marginalization** so intensity calibration is not a fit parameter; composable
  priors/regularizers (bounds, smoothness, defect sparsity).

**Gate KG1:** every loss and every transform is differentiable (finite grad);
each bijector round-trips (`physical = fwd(inv(physical))` to tolerance); a planted
loss has its minimum at the true parameters (1-D sanity sweep); `optax.adam` runs
a step on a toy objective; universal gate.

### Phase K2 — the optax solver core (`solve`)

*Tasks:*
- `recon/solve.py` — `ReconProblem` (eqx.Module) + `solve(problem, optimizer,
  n_steps, init) -> ReconResult`. The step is `optax` (`optax.chain` of
  `clip_by_global_norm` + a scheduled `adamw`, with an `lbfgs` option for the
  well-conditioned endgame); a **JIT-compiled `lax.scan`** fixed-N loop is the
  rapid path, with an eager early-stopping variant (mirrors
  `discretize_orientation` vs `_static`).
- **Replace** `adam_optimize` / `adagrad_optimize` / `gauss_newton_least_squares`
  with optax-backed equivalents; keep the public names as deprecated shims that
  forward to `solve`.

**Gate KG2:** on synthetic data simulated from a *known* structure,
`solve` recovers the parameters to tolerance with a monotone (scheduled) loss;
the JIT-scan result == the eager result; the optax path reproduces the retired
hand-rolled optimizers' results to tolerance (regression guard); a **wall-clock
budget** assertion on the reference problem (the "rapid" claim); universal gate.

### Phase K3 — Robustness: multistart + bracket-then-refine

*Tasks:*
- `multistart(problem, k_inits, key)` — `vmap` (or `distribute_batched`) over `k`
  random inits, return the best by final loss; the non-convexity guard.
- **Loop-B → Loop-C handoff:** consume a coarse candidate from the automatons'
  `screen_xyz_ensemble` (or a `.xyz` prior) as an init, so the discrete bracket
  seeds the gradient refine.

**Gate KG3:** `multistart` escapes a *planted* local minimum that a single start
provably falls into; a bracketed init converges in strictly fewer steps than a
cold start; reproducible across seeds; universal gate.

### Phase K4 — Uncertainty quantification (generalized)

*Tasks:* generalize `orientation.py`'s `compute_fisher_information` /
`estimate_weight_uncertainty` to **arbitrary** latent parameters —
Gauss–Newton / Fisher covariance from the residual Jacobian (`jax.jacfwd`/`jacrev`),
a Laplace approximation at the optimum, per-parameter error bars + the correlation
matrix. UQ is what lets the agent say *what is being grown* **with confidence**,
not just a point estimate.

**Gate KG4:** the recovered covariance matches the **empirical** parameter spread
over many noisy synthetic realizations to tolerance; the covariance is finite and
positive semi-definite; it reduces to the existing orientation UQ on that
sub-problem (regression); universal gate.

### Phase K5 — the recipe-deviation contract (automaton-facing)

*Tasks:* `recipe_deviation(problem, intended_recipe) -> report`: invert the
measured pattern, then compute the **per-parameter gap** between the LLM agent's
intended recipe and the inverted reality, each gap normalized to a **z-score** via
the K4 covariance, with a severity flag. This is the exact object the automatons'
`recipe_deviation.py` emits and `invert_structure.py` builds on — **freeze its
shape here** so the downstream plan pins against it.

**Gate KG5:** on a synthetic pattern grown from a structure *deliberately
mismatched* from a stated intended recipe, the report flags the right parameters
with correct sign and magnitude and a **calibrated** significance (a matched
recipe yields z-scores within noise); the report validates against a committed
schema; universal gate.

### Phase K6 — Speed, docs, API freeze

*Tasks:* wire the inversion's `solve`-scan through `tools.enable_compilation_cache`
and (optionally) export the **fixed-iteration grad step** via `tools.export_forward`
for a deployable inversion; a "Differentiable inversion with optax" guide under
`docs/source/guides/`; export the inverse API (`ReconProblem`, `solve`,
`multistart`, `recipe_deviation`, the UQ helpers) via `recon/__init__` `__all__`
+ Routine Listings, and **freeze it** as the automatons' import surface.

**Gate KG6:** docs build (`make html`); a reference inversion meets the wall-clock
budget warm-cache; the public inverse API is exported, typed, and Routine-Listed;
universal gate.

### Gate summary

| Gate | Pass condition (+ universal gate) |
|------|------------------------------------|
| **K0** | rationalization complete; forward model differentiates end-to-end |
| **KG1** | optax added; bijectors round-trip; losses differentiable; planted-loss minimum correct |
| **KG2** | `solve` recovers a known structure; JIT-scan == eager; optax == retired hand-rolled (regression); within wall-clock budget |
| **KG3** | multistart escapes a planted local min; bracketed init converges faster; reproducible |
| **KG4** | recovered covariance matches empirical spread; PSD; reduces to orientation UQ |
| **KG5** | recipe-deviation flags the right params with calibrated z-scores; schema-validated |
| **KG6** | docs build; budget met warm-cache; inverse API exported + frozen |

K1–K2 are the foundation (replace hand-rolled with optax, prove recovery); K3–K5
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

- **optax migration regressions.** Mitigation: KG2's regression guard pins the
  optax path to the retired hand-rolled results before the shims are removed.
- **Local minima / identifiability.** Mitigation: K3 multistart + bracket-then-
  refine; report multimodality (K4 covariance flags flat directions).
- **UQ validity.** Gauss–Newton/Laplace assumes a near-minimum, locally-Gaussian
  posterior; document the assumption and the regime where it holds; KG4 calibrates
  it empirically rather than trusting it blindly.
- **Speed vs the growth cadence.** Mitigation: JIT-scan fixed-N loop + compilation
  cache + optional AOT (K6); the wall-clock budget is a gate, not an aspiration.
- **New dependency.** optax is pure-JAX, lightweight, and already a transitive peer
  of the stack; the `automatons-smoke` ephemeral-env check (downstream) catches any
  packaging surprise.
- **Differentiability regression from "cleanup".** Mitigation: the universal gate's
  finite-grad check is mandatory on every PR.

---

## 6. Diff surface

| Path | Change |
|------|--------|
| `pyproject.toml` | add `optax` to dependencies (CPU + CUDA resolve) |
| `src/rheedium/recon/transforms.py` | **new** — general bijector / reparameterization layer |
| `src/rheedium/recon/solve.py` | **new** — `ReconProblem`, `solve` (optax + `lax.scan`), `multistart` |
| `src/rheedium/recon/uncertainty.py` | **new** — generalized Gauss–Newton/Fisher covariance, Laplace UQ |
| `src/rheedium/recon/deviation.py` | **new** — `recipe_deviation` (the automaton-facing control signal) |
| `src/rheedium/recon/optimizers.py` | retire hand-rolled Adam/Adagrad/Gauss-Newton → optax-backed shims (deprecated) |
| `src/rheedium/recon/orientation.py` | refit onto `solve`/`transforms`/`uncertainty`; keep `fit_orientation_weights` as a thin wrapper |
| `src/rheedium/recon/losses.py` | extend: log-intensity, NCC, analytic scale/background marginalization, priors |
| `src/rheedium/recon/__init__.py` | exports + Routine Listings; freeze the inverse API |
| `tests/.../test_recon/*` | recovery-of-known-params, JIT-scan==eager, optax==hand-rolled regression, multistart-escape, UQ calibration, recipe-deviation calibration, grad-finite |
| `docs/source/guides/` | "Differentiable inversion with optax" guide |

---

## 7. Outcome

When complete: `recon` is a **general differentiable inverse-problem solver** —
one optax-backed `solve` over any latent space, multistart for robustness,
calibrated uncertainty, and a frozen `recipe_deviation` contract — replacing three
hand-rolled optimizers and the orientation-only fitter. This is the engine the
automatons' Loop C calls to answer, in real time and with error bars, *what is
actually being grown and how far it is from intent* — the scientific payoff of the
entire differentiable architecture.
