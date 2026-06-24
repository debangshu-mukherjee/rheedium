# Plan: Transition `rheedium.audit` Toward Formal Verification

Scope: `src/rheedium/audit/` (`invariants.py`, `reference_types.py`,
`metrics.py`, `reference_benchmark.py`). Today each `check_*` states a physical
law, samples a fixed `jnp.linspace` grid, and reports a worst-case `residual`
vs `tolerance` as an `InvariantResult`. That is **specification-based numerical
testing**. This plan pushes each invariant **as far up the rigor ladder as its
mathematics allows** — and, crucially, makes the report state *which* level of
rigor each result carries.

## 1. The honest ceiling

Full formal verification of floating-point physics kernels (multislice
propagation, optimizers, the simulated intensities) is **not** practical. The
goal is not "prove everything" — it is:

1. **Promote** every invariant that *is* an exact identity or a structurally
   provable inequality from *sampled* to *proven*.
2. **Verify** the discrete/combinatorial machinery (symmetry ops, Miller grids,
   reconstruction matrices) by exhaustion or SMT.
3. **Honestly label** the irreducibly-numerical checks as *validated within
   tolerance over generated inputs* — not proven — and make that visible in the
   report.

The deliverable is a module whose output distinguishes **proof** from
**sampling**, so a reader knows exactly what is guaranteed.

## 2. The rigor ladder

| Tier | Method | Guarantee | Tooling |
|---|---|---|---|
| T1 Exact identity | symbolic derivation + machine-ε residual | holds ∀ inputs (exact arithmetic) | SymPy |
| T2 Structural inequality | prove from finite coefficient/structure conditions | holds ∀ inputs | SymPy / Z3 |
| T3 Discrete property | exhaustive enumeration or SMT | holds ∀ inputs in a finite domain | Z3 / itertools |
| T4 Numerical bound | property-based testing + interval bounds | holds over *sampled* inputs | Hypothesis |
| T5 (stretch) machine-checked | proof assistant | holds ∀, machine-audited | Lean/Coq |

## 3. Reclassify the six existing invariants

| Current check | Tier | Why / how to promote |
|---|---|---|
| `check_friedel_law_structure_factor` | **T1** | `F(-G) = conj(F(G))` is *exact* for real atom positions ⇒ `|F(-G)|² = |F(G)|²` identically. Replace the loose tolerance with a machine-ε residual; add a SymPy derivation as the proof artifact. |
| `check_elastic_closure_ewald` | **T1** | Elastic scattering preserves `|k|` by construction. Exact geometric identity ⇒ residual ~1e-12, SymPy-checked. |
| `check_wavelength_relativistic_consistency` | **T1** | `λ(V)` is a closed form; verify the algebraic identity symbolically and that it is strictly monotone ↓ in `V` (sign of dλ/dV proven, not sampled). |
| `check_form_factor_positivity` | **T2** | Kirkland/Lobato `f(q)=Σ aᵢ/(s²+bᵢ)+Σ cᵢ·exp(−dᵢs²)`. If every `aᵢ,bᵢ,cᵢ,dᵢ>0` then `f(q)>0` ∀q — a **finite** check over the coefficient table *proves* the inequality for all q. Replace q-sampling with: (a) assert table positivity, (b) Z3/SymPy confirm the implication. |
| `check_form_factor_monotonic_decrease` | **T2** | Each term is monotonically decreasing in q for q>0 (Lorentzian and Gaussian both); `df/dq<0` provable termwise from the same positive coefficients. Z3 over `q∈[0,qmax]` or symbolic sign argument. |
| `check_form_factor_kirkland_lobato_close` | **T4** | Two independent empirical fits being numerically close is **not** an identity. Keep as a bounded property test (Hypothesis over q, Z) with a stated, justified tolerance; label `kind="sampled"`. |

Net: **3 invariants become exact proofs, 2 become structural proofs, 1 stays
(honestly) sampled.**

## 4. Extend `InvariantResult` to carry rigor

```python
from typing import Literal

@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    residual: float
    tolerance: float
    units: str
    detail: str
    kind: Literal["proof", "exhaustive", "smt", "sampled"] = "sampled"
    # optional: a reference to the proof artifact (SymPy expr id, Z3 model,
    # enumerated domain size) for the report.
```

`run_default_invariants()` then reports, e.g., "5/6 proven, 1/6 sampled" — the
single most valuable output of this transition.

## 5. Verify the discrete surface (T3) — new checks

These live *outside* `audit` today but are exactly what SMT/exhaustion can
*prove*. Add `audit` checks for them:

- **CIF symmetry-op parsing** (`inout/cif.py::_parse_sym_op`): for every op in a
  space group, the parsed 3×3+translation matrix matches the spec; group
  closure (composition stays in the group); determinant ±1. Exhaustive over the
  finite op list per group; or Z3 for the parser's algebra.
- **Miller-index grid generation** (`ucell`): generated `(h,k,l)` sets are
  complete and duplicate-free over the requested bounds. Exhaustive.
- **Surface-reconstruction matrices** (`procs`): integer, invertible over ℤ,
  determinant = expected cell multiplier. Z3 over the integer matrices.
- **Probability simplex** (`distributions`): orientation/size weights are
  non-negative and sum to 1 after normalization — a refinement contract,
  provable from the normalization code (T2) and enforceable at runtime (§6).

## 6. Refinement contracts as lightweight verification (T2-ish, runtime)

jaxtyping already enforces shape/dtype at boundaries (now via the pytest hook).
Extend toward **value-range refinements** as runtime-checked contracts — the
closest practical thing to verified preconditions/postconditions:

- cell angles ∈ (0, 180), cell lengths > 0 (the `create_*` factories already
  check these — formalize them as documented pre/postconditions and assert at
  every construction).
- probability weights on the simplex; intensities ≥ 0.

These don't *prove* the math but turn invariants into always-on enforced
contracts rather than occasional tests.

## 7. Tooling

- **SymPy** (new `audit`/dev dep): derive and machine-check the T1 identities
  (Friedel, elastic closure, wavelength) and the T2 sign arguments. Commit the
  derivations as `audit/_proofs/*.py` that *assert* the symbolic result, so they
  run in CI.
- **Z3 (`z3-solver`)** (new dev dep): T2 inequalities over coefficient tables
  and T3 discrete/integer properties. Each Z3 check asserts `unsat` of the
  negation (i.e. proves the property) and records the result `kind="smt"`.
- **`itertools`/numpy** exhaustion: T3 finite domains.
- **Hypothesis**: T4 numerical bounds (shares `tests/_strategies.py` with the
  property-testing plan).
- **Lean/Coq (stretch, optional)**: a machine-checked proof of the *algebraic
  core* (e.g. Friedel Hermitian symmetry) committed under `audit/_proofs/lean/`.
  High effort; do only if a reviewer wants an auditable proof object.

## 8. What stays out of scope (document explicitly)

- Multislice propagation, optimizer convergence, simulated detector images:
  floating-point + iterative ⇒ **property-tested only** (T4). State this in the
  module docstring so no one mistakes "audited" for "proven".

## 9. Rollout (PR-sized)

1. **PR 1**: add `kind` to `InvariantResult`; make `run_default_invariants`
   report the proof/sampled split. (Pure plumbing; no new deps.)
2. **PR 2 — T1 promotions**: SymPy-backed exact proofs for Friedel, elastic
   closure, wavelength; tighten residuals to machine-ε; `audit/_proofs/`.
3. **PR 3 — T2 promotions**: prove form-factor positivity & monotonicity from
   coefficient-table positivity (Z3/SymPy); replace q-sampling.
4. **PR 4 — T3 discrete**: symmetry-op parsing, Miller grids, reconstruction
   matrices via exhaustion/Z3 as new `audit` checks.
5. **PR 5 — refinement contracts**: formalize factory pre/postconditions;
   probability-simplex enforcement.
6. **PR 6 — T4 honesty**: convert the remaining sampled checks to Hypothesis,
   labeled `kind="sampled"`; document the out-of-scope kernels.
7. **PR 7 — CI**: a `formal-verification` job runs the SymPy/Z3 proofs and fails
   if any *proof* regresses (distinct from the numerical test job).

## 10. Success criteria
- `run_default_invariants()` reports each result's `kind`, and ≥ 5 of the
  current 6 are `proof`/`smt`/`exhaustive` rather than `sampled`.
- SymPy/Z3 proof artifacts run in CI and fail on regression.
- Discrete machinery (symmetry ops, Miller grids, reconstruction matrices) has
  exhaustive/SMT verification.
- The module docstring states precisely what is *proven* vs *validated*, ending
  the "everything is audited" ambiguity.
