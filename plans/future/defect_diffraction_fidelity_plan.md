# High-Fidelity Defect & Coherence Diffraction Physics

Scope: `rheedium` — add the **physically-faithful diffraction signatures of
defects and partial coherence** that the
[distribution framework](plans/implemented/distribution_framework_plan.md) ships as
*smooth structure modifiers*. The three fidelity items scoped **out** of the
framework plan (its "Tier 3"), plus a fourth, related forward-physics capability:

1. **Fine-twin satellites** — coherent twin-domain superlattices that produce
   satellite reflections around the fundamentals, not just a blended pattern.
2. **Step-terrace / vicinal diffraction** — regular step trains that split streaks
   (RHEED beam splitting), beyond the current smooth step modifier.
3. **Full coherent displacement-fringe validation** — geometric interference
   fringes from two displaced reflections within `ℓ_c`, upgrading today's
   opposite-phase cancellation check to true fringe physics.
4. **Sampling producers for implicit / Boltzmann ensembles** — `blackjax`-sampled
   `Distribution`s (anharmonic thermal displacement, correlated disorder) for
   physical densities with **no closed form**, where the framework's analytic
   producers (GSM, Gaussian divergence, lognormal size, harmonic Debye–Waller)
   stop. The forward-side mirror of recon's posterior sampling.

This is a **physics-depth track, not a roadmap gate.** It builds on the
framework's coherent reduction + `Distribution`/bind contract (which already
exist), but **nothing downstream depends on it**: the gated chain
(rationalization → [recon](../implemented/recon_optimization_plan.md) →
[automatons](plans/partial/automatons_plan.md)) needs only the framework's
differentiability guarantee, which this plan does not touch. It runs in parallel
to — or after — the roadmap, purely to raise simulation fidelity.

Status: **proposed.** Scoped out of
[distribution_framework_plan.md](plans/implemented/distribution_framework_plan.md) by
decision (the framework treats these defects as smooth structure modifiers that
*change* the pattern but do not reproduce satellites / streak splitting). Not on
the critical path.

**This plan is gate P6** ("surface non-ideality & defect signatures") of the
umbrella [RHEED physics A+ plan](physics.md): it is the single source of truth for
the twin / step / mosaic / strain / reconstruction signatures, so physics.md
references it rather than duplicating them. Its DG1–DG4 closing *is* P6's exit
criterion.

---

## 1. Why a separate plan

The framework's job is the **architecture** (Layer-0 complex amplitude → Layer-1
`Distribution` reducer → Layer-2 producers) and the **coherent/incoherent
partition**. It delivers, end to end, that a twin/step/grain distribution *changes*
the detector image and stays differentiable. What it deliberately does **not**
deliver is the *characteristic fine structure* of those defects:

- A coherent twin lamella array is a superlattice → **satellite peaks** at
  `G ± n·q_twin`. The framework's twin bind builds a modified `CrystalStructure`
  and reduces smoothly; it does not place satellites.
- A regular vicinal step train coherently splits each streak into a doublet/
  multiplet at a spacing set by the terrace width. The framework's step modifier
  is smooth, not split.
- Coherent interference is *tested* only as opposite-phase cancellation, not as
  geometric fringes at the correct spacing.

These are real, publishable physics (the kind a referee will look for), but they
are **orthogonal to the framework's structural thesis** and much heavier (they
touch the diffraction math, not the plumbing). Hence a dedicated plan.

---

## 2. What it adds

### D1 — Fine-twin satellites (the coherent sub-`ℓ_c` twin path)

- Model a fine-twin domain array as a coherent superstructure (period `q_twin`
  set by lamella spacing) so the structure factor acquires satellite terms around
  each fundamental.
- Route through the framework's **coherent** reduction (twins below `ℓ_c` reduce
  coherently — §5 of the framework); the producer emits the satellite-bearing
  structure, the integrator does the rest.
- Validate satellite **position** (vs `q_twin`) and **relative intensity** (vs
  twin fraction) against analytic/textbook expectations.

### D2 — Step-terrace / vicinal diffraction

- Coherent regular-step train → streak splitting; couple terrace width → split
  spacing and step height → phase, extending `vicinal_surface_step_splitting`
  beyond the smooth modifier.
- Keep the **regular → coherent / random → incoherent** partition the framework
  already computes; this plan supplies the coherent-branch fine structure.
- Validate split spacing vs terrace width and the regular↔random crossover.

### D3 — Full coherent displacement-fringe validation

- Upgrade framework test 5: two displaced identical reflections within `ℓ_c`
  produce **fringes at the predicted spacing**; outside `ℓ_c` they sum flat.
- Becomes the rigorous coherence-physics regression for the whole stack.

### D4 — Sampling producers for implicit / Boltzmann ensembles (`blackjax`)

The framework's analytic producers (GSM beam modes, Gaussian divergence, lognormal
size, **harmonic** Debye–Waller) have closed forms — generate them by quadrature or
direct reparameterized sampling; that is exact, cheap, and convergence-free, and it
**stays as-is**. D4 adds a **sampling producer** *only* for physical densities with
no closed form, known up to a normalizing constant:

- **Anharmonic thermal displacement** — `u ~ exp(−E(u)/kT)` for a non-quadratic
  potential: the *non-Gaussian* generalization of Debye–Waller (soft-mode /
  large-amplitude regimes), where the harmonic Gaussian no longer holds.
- **Correlated disorder / short-range order** — vacancy or occupancy
  configurations drawn from a **Gibbs/Boltzmann distribution over a lattice
  Hamiltonian** (no analytic marginal).

**Mechanism.** `blackjax` (HMC/NUTS) samples the unnormalized density; the draws
become a `Distribution(samples, weights = 1/N, reduction = INCOHERENT)` that flows
through the **existing** Layer-1 reducer — no new integrator, no parallel
simulator. This is the **forward-side mirror of recon's posterior sampling** (same
`blackjax` core dependency, same "sample a density" pattern); **no new dependency**
since recon already makes `blackjax` core.

**Differentiability rule (decisive — it is the framework's invertibility
principle).** A sampled producer must not silently weld a seam:

- **Reparameterize where possible** — sample fixed base noise, transform by the
  physical params `θ`; gradients flow through the transform (and this is usually
  *not even MCMC*, just `jax.random` + a map). **Prefer this always.**
- **MCMC only for truly implicit densities**, and then document **per axis** that
  the producer is differentiable only via reparam / score-function — and where it
  is **not** differentiable w.r.t. `θ`, **recon cannot invert through that axis**;
  that limitation is stated, not hidden.
- The producer is **seeded** (reproducibility guard); the sample count controls
  variance, and the forward image is sample-dependent (fine for generation/
  augmentation, flagged for any deterministic regression).

---

## 3. Constraints (inherited from the framework)

- **Differentiability preserved** — every addition keeps `jax.grad` flowing
  (CONTRIBUTING *Invertible Modularity*); satellite/split structure must come from
  differentiable structure-factor terms, not a discrete branch. For the D4 sampling
  producer this means **reparameterize-where-possible** (gradients flow through the
  transform); a genuinely non-differentiable sampled axis is permitted only when
  the density is irreducibly implicit, and is **documented as non-invertible** so
  recon never silently fails through it.
- **Reuse, don't reinvent** — satellites/splits are produced *inside* the existing
  `Distribution` + coherent-reduction machinery; no parallel simulator.
- **No new premature reduction** — the silent failure mode; reviewer-checked.

---

## 4. Gated phases (sketch)

Each phase ends at an objectively checkable gate (a command, not an opinion).

- **D0 — entry.** The framework's coherent reduction + bind contract are stable
  (framework Phases 1–5 landed; coherent reduction exercised end-to-end).
- **DG1 — satellites.** A fine-twin distribution produces satellite reflections at
  the analytic `q_twin` positions with intensity tracking twin fraction;
  differentiable; regression vs an analytic superlattice.
- **DG2 — step splitting.** A regular vicinal step train splits a streak at the
  terrace-width-predicted spacing; the regular↔random crossover reproduces the
  coherent↔incoherent transition; differentiable.
- **DG3 — fringe validation.** The displacement-fringe test asserts correct fringe
  spacing within `ℓ_c` and flat sum beyond; wired as a standing regression.
- **DG4 — sampling producer.** A `blackjax` sampling producer (a) **reduces to the
  analytic producer in the closed-form limit** — the anharmonic-thermal sampler
  matches the harmonic Debye–Waller Gaussian as the potential → quadratic
  (regression); (b) recovers the correct **non-Gaussian** displacement distribution
  from a planted anharmonic potential (samples match a reference); (c) is
  **seeded → reproducible**; (d) the reparameterized path returns **finite
  gradients** w.r.t. `θ` where claimed, and any non-differentiable axis is
  documented as recon-non-invertible.

---

## 5. Outcome

When complete, `rheedium` reproduces the *characteristic* diffraction fine
structure of twins and vicinal steps — satellites and streak splitting at the
right positions and intensities — not merely a blended change, and can **generate
distributions for implicit/Boltzmann ensembles** (anharmonic thermal, correlated
disorder) by `blackjax` sampling — the forward-side mirror of recon's posterior
sampling — all through the framework's differentiable `Distribution` contract. This
is fidelity for its own sake (and for referees), decoupled from the roadmap so it
can land whenever the physics is ready.
