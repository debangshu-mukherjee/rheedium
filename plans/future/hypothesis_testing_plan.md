# Plan: Property-Based Testing with Hypothesis

Scope: `rheedium` (this repo). `hypothesis>=6.155.2` is already a `test`
dependency but is **not yet imported anywhere**. This plan introduces it as a
second testing layer alongside the existing example-based `chex`/`parameterized`
tests — it does not replace them.

## 1. Goal and mental model

Example tests assert `f(specific_input) == specific_output`. Property tests
assert `for all valid x: P(f(x))` and let Hypothesis *generate* `x` (random +
boundary cases) and *shrink* any failure to a minimal counterexample.

The win for a differentiable-physics package: the input space (crystals, beams,
q-grids, images) is large and structured, and the correctness conditions are
mostly **physical invariants and algebraic round-trips**, not hand-picked
output values. Those are exactly what property testing expresses well.

Keep the existing suite. Add property tests where a *property* is clearer or
stronger than an example.

## 2. Layered targets (highest leverage first)

### Layer A — Round-trip / inverse properties (cheapest, highest value)
Pure structural identities; no physics tolerance needed.

- **PyTree flatten/unflatten** for every `eqx.Module`:
  `tree_unflatten(treedef, leaves) == original` (values + static fields).
- **HDF5 round-trip**: `load_from_h5(save_to_h5(x)) == x` for every registered
  type (`CrystalStructure`, `XYZData`, `RHEEDPattern`, …, `ReconResult`).
  Generalizes `test_hdf5.py::_assert_round_trip_equal`.
- **Parser round-trips**: `parse_xyz(write_xyz(data)) ≈ data`,
  `parse_poscar`/`parse_cif`/`parse_vaspxml` where a writer exists; and
  cross-format consistency (`from_ase(to_ase(c)) ≈ c`,
  `from_pymatgen(to_pymatgen(c)) ≈ c`).

### Layer B — Validation/contract properties (factories)
The `create_*` factories validate inputs. Property-test the contract:

- Valid generated inputs → factory succeeds and **preserves** shapes/dtypes/values.
- Invalid generated inputs (wrong shape, NaN, negative cell length, angle ∉
  (0,180)) → raises (a `TypeCheckError`/`ValueError`). This hardens the
  validation surface far beyond the current fixed cases.

### Layer C — Physical invariants over *generated* inputs (the big one)
Today `rheedium.audit.invariants` checks laws on a fixed `linspace` grid. Drive
the **same** properties with Hypothesis-generated inputs:

- `f(q) > 0`, `f(q)` monotonically decreasing, `kirkland ≈ lobato` — over
  generated `q`, `atomic_number`.
- Friedel's law `|F(-G)|² == |F(G)|²` over generated `G`, structures.
- Elastic closure `|k_in| == |k_out|` over generated geometry.
- Relativistic wavelength monotonic ↓ in voltage; matches the closed form.
- Simulated `intensities ≥ 0` and finite for any valid crystal+beam.

### Layer D — Metamorphic properties (no reference output needed)
Relations between *two* runs:

- **Transform invariance**: translating all atom positions by a lattice vector,
  or applying a symmetry operation, leaves `|F(G)|²` invariant.
- **vmap == loop**: `jax.vmap(sim)(batch)` equals stacking `sim` over the batch
  (a strong correctness + shape check).
- **Determinism**: same input twice → identical output.
- **Unit/scale equivariance** where physics dictates it.

### Layer E — Differentiability properties
- `jax.grad(loss)(generated_params)` is finite (no NaN) for any valid input —
  a generated-input version of the existing `test_grad_*` cases.

## 3. Infrastructure (project-specific)

Create `tests/_strategies.py` — typed Hypothesis strategies for rheedium types
(the property-test analogue of `tests/_factories.py`):

```python
# tests/_strategies.py
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
import jax.numpy as jnp
from rheedium.types import create_crystal_structure, CrystalStructure

@st.composite
def crystals(draw, max_atoms: int = 6) -> CrystalStructure:
    n = draw(st.integers(1, max_atoms))
    frac = draw(hnp.arrays(np.float64, (n, 3),
                           elements=st.floats(0.0, 1.0, allow_nan=False)))
    z = draw(hnp.arrays(np.float64, (n, 1),
                        elements=st.integers(1, 92).map(float)))
    lengths = draw(hnp.arrays(np.float64, (3,),
                              elements=st.floats(2.0, 12.0)))
    angles = draw(hnp.arrays(np.float64, (3,),
                             elements=st.floats(60.0, 120.0)))
    frac4 = jnp.asarray(np.hstack([frac, z]))
    return create_crystal_structure(frac4, frac4, jnp.asarray(lengths),
                                    jnp.asarray(angles))
```

Provide strategies for: `crystals`, `electron_beams`, `q_values`,
`atomic_numbers`, `reciprocal_vectors`, `rheed_images`, `reflection_sets`.

### JAX / chex / conftest interactions — get these right
- **`deadline=None`**: JIT/trace warmup makes per-example timing wildly variable.
  Set `@settings(deadline=None)` (or a global profile) or Hypothesis will flag
  false "too slow" failures.
- **`max_examples` budget**: physics calls are expensive. Default 100 is often
  too many. Use `@settings(max_examples=25–50)` for simulator-level properties,
  more for cheap structural ones.
- **conftest memory leak detector** (`MEM_LEAK_THRESHOLD_GB = 1.0`): each
  property test runs N examples in one test function → N× the allocation of one
  example. Heavy generated simulations may trip the 1 GB per-test limit. Mark
  heavy property tests to allow more headroom, or cap `max_examples`, or call
  `jax.clear_caches()` inside the property.
- **float64**: tests assume x64 (`conftest`/`JAX_ENABLE_X64`); generate float64
  and keep tolerances physical.
- **xdist**: Hypothesis works under `-n auto`; pin a seed-independent
  `derandomize=False` and commit a `.hypothesis/` example DB is optional (see CI).
- **jaxtyping hook**: generated invalid inputs will raise `jaxtyping.TypeCheckError`
  / beartype violations — property tests for the "invalid → raises" contract
  should expect those exception types.

### A global Hypothesis profile
```python
# tests/conftest.py  (add)
from hypothesis import settings, HealthCheck
settings.register_profile(
    "rheedium",
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("rheedium")
```

## 4. Rollout (PR-sized steps)

1. **PR 1 — infra**: `tests/_strategies.py` (crystals, beams, q, images) +
   the Hypothesis profile in `conftest`. One trivial property test to prove
   wiring under `-n auto` and the hook.
2. **PR 2 — Layer A round-trips**: pytree + HDF5 + parser round-trips. Highest
   value, no physics tolerances, will likely surface real edge cases fast.
3. **PR 3 — Layer B factory contracts**: valid-preserves / invalid-raises.
4. **PR 4 — Layer C invariants**: rewrite the `audit` checks' *drivers* as
   Hypothesis properties (see the audit formal-verification plan — they share
   strategies). Keep the audit functions; call them from property tests.
5. **PR 5 — Layer D/E metamorphic + grad**: vmap==loop, transform invariance,
   finite gradients.
6. **PR 6 — CI**: run property tests as a separate job; commit `.hypothesis/`
   example DB *or* set `derandomize=True` for reproducibility; add a nightly
   job with large `max_examples` (`--hypothesis-profile=nightly`).

## 5. Anti-patterns / pitfalls
- Don't assert exact float equality on physics outputs — use `chex.assert_trees_all_close`
  with a justified `atol`/`rtol`, or assert sign/monotonicity/bounds.
- Don't generate unphysical inputs that the function never promises to handle
  (e.g. zero-length cells) unless testing the *reject* contract — use `assume()`
  or constrained strategies.
- Don't leave `deadline` at default — JIT warmup will cause flaky failures.
- Don't let one property test silently balloon memory — cap `max_examples` and
  clear caches for simulator-level properties.

## 6. Success criteria
- `tests/_strategies.py` exists; a `rheedium` Hypothesis profile is loaded.
- Round-trip + factory-contract + invariant + metamorphic property tests exist
  for the core types and physics functions.
- Property tests pass under `-n auto` with the jaxtyping hook and the memory
  detector, and run green in CI (reproducibly).
- At least one previously-unknown edge case found and fixed (expected outcome of
  Layer A/B).
