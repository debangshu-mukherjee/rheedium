# JIT-Compatible Runtime Validation Plan

Scope: `rheedium` — convert input validation from non-rejecting / non-jit-safe
patterns to real, JIT-compatible runtime rejection using `eqx.error_if`
(constructors) and `jax.experimental.checkify` (numerical entry points).

Status: **implemented and verified in-tree** — all phases (0–5) landed:
constructor validation migrated to `eqx.error_if` across `beam_types`,
`crystal_types`, `distributions`, and `rheed_types` (no NaN substitutions
remain); opt-in `checkify` checked variants for the simulator entry points and
recon losses; the shared `assert_rejects` helper lives in `tests/_assertions.py`;
accept-path gradient parity is covered by
`TestElectronBeamGradients.test_filter_jit_grad_matches_eager_grad`; and the
checked-entry-points guide is published. Remaining work is limited to optional,
more specific `checkify.user_checks` guards if future APIs need them. Moved from
`plans/partial/` to `plans/implemented/` on completion.

---

## 1. Problem statement

Today the package "validates" inputs in two ways, **neither of which actually
rejects an invalid value under `jax.jit`**:

### Anti-pattern A — silent NaN substitution (`beam_types.py`)

`create_electron_beam` runs each check through `lax.cond` and, on failure,
returns `jnp.full_like(x, jnp.nan)` instead of raising. 6 sites:
`src/rheedium/types/beam_types.py` (`_check_energy`, `_check_energy_spread`,
`_check_divergence`, `_check_transverse_coherence`,
`_check_longitudinal_coherence`, `_check_spot_size`).

```python
def _check_energy() -> scalar_float:
    valid = jnp.logical_and(energy_kev >= 5.0, energy_kev <= 100.0)
    return lax.cond(valid, lambda: energy_kev,
                    lambda: jnp.full_like(energy_kev, jnp.nan))
```

The invalid case is **executed**, not rejected: the caller silently receives a
NaN-poisoned `ElectronBeam` and the error only surfaces (maybe) many ops later.
The tests encode this as the contract:

```python
# tests/test_rheedium/test_types/test_beam_types.py:111
def test_energy_too_low(self):
    """Energy below 5 keV should produce NaN."""
    beam = create_electron_beam(energy_kev=3.0, ...)
    assert jnp.isnan(beam.energy_kev)      # asserts substitution, not rejection
```

### Anti-pattern B — eager `raise` on traced values (`crystal_types.py`)

Value checks raise Python exceptions off traced arrays. 9 raise sites in
`crystal_types.py`; the **value** ones (not the static shape ones) are the
problem:

```python
# src/rheedium/types/crystal_types.py:1009
if not jnp.all(jnp.isfinite(positions)):           # concretizes a traced bool
    raise ValueError("positions contain non-finite values")
if not jnp.all(atomic_numbers >= 0):
    raise ValueError("atomic_numbers must be non-negative")
```

This works **eagerly** but raises `ConcretizationTypeError` the moment the
constructor is traced inside `jit`/`vmap`/`grad`. So it is both a latent crash
and not a real guarantee.

### Goal

Turn validation tests from *"executed the invalid case"* into *"the invalid
case is actually rejected"* — under both eager and JIT execution — using the
documented JIT-able mechanisms:

- **Equinox `eqx.error_if`** — raises based on runtime values, including under
  JIT (host-callback / checkify-backed).
- **JAX `jax.experimental.checkify`** — the functionalized mechanism for
  JIT-able runtime checks at function boundaries.

Both are available in this environment: equinox **0.13.8** (`eqx.error_if`
present), `jax.experimental.checkify` importable.

---

## 2. Mechanism comparison

| | `eqx.error_if` | `checkify` |
|---|---|---|
| Call style | `x = eqx.error_if(x, pred, msg)` inline | `checkify.check(pred, msg)`; wrap fn with `checkify.checkify` |
| Signature impact | none (drop-in) | caller must run `err, out = checkified(...)` then `err.throw()` |
| Under jit/vmap/grad | yes (host callback) | yes (functional, threaded error) |
| Cost | host callback per check at runtime | ~free in-graph; cost paid at boundary |
| Caller burden | none | must thread/throw the error |
| Disable switch | `EQX_ON_ERROR=raise|nan|breakpoint` | choose `errors=` set / skip wrapping |
| Best for | **constructors / validation boundary** | **numerical entry points & hot paths** |

Key correctness notes:
- `eqx.error_if` returns a **new** value carrying the check as a data
  dependency. You **must use the returned array** downstream, else the check is
  dead-code-eliminated and never runs.
- `checkify.check` is a no-op until the function is transformed with
  `checkify.checkify(...)`. Errors are accumulated functionally and only raised
  when the caller calls `err.throw()`. It also offers automatic
  `nan_checks` / `div_checks` / `index_checks` sets.
- Both compose with `grad` (checks are non-differentiable pass-throughs) and
  `vmap` (a batched predicate rejects if **any** element is invalid).

### Recommendation (the split)

- **Constructors** (`create_*`) → `eqx.error_if`. They are the validation
  boundary, already return `eqx.Module`/PyTrees, are usually called once at
  setup, and need no signature change. This is the high-value, low-churn move.
- **Numerical entry points** (`ewald_simulator`, `simulate_detector_image`,
  multislice, recon loss) → optional `checkify` wrapping for `nan`/`div`/
  finite guards the **caller opts into**, so inner jit hot loops are not
  burdened with host callbacks by default.

Only **value** checks move. **Static shape/rank checks stay as plain Python
`if` + `raise ValueError`** — shapes are known at trace time, so those are
already correct and JIT-safe.

### Differentiability (core package goal)

Differentiability is a first-class requirement for rheedium (it is a
*differentiable* RHEED simulator), so any validation mechanism must be
gradient-transparent on the **accept** path. Both candidates are — verified
empirically (equinox 0.13.8 / jax 0.9.2):

| Mechanism | `grad` (accept path) | `jit`+`grad` | `vmap` | Invalid case |
|---|---|---|---|---|
| `lax.cond` (current) | `6.0` ✓ | `6.0` ✓ | ✓ | returns **NaN**, grad `0.0` — silent |
| `eqx.error_if` (target) | `6.0` ✓ | `6.0` ✓ | ✓ | **raises** under jit+grad — rejected |

Why `error_if` is the *better* choice for a differentiable package, not just an
equal one:

- **Accept path is a pure identity** (`∂/∂x = 1`). It adds zero arithmetic to
  the differentiated graph — the gradient is exactly that of the unguarded
  function. (`lax.cond` is also differentiable, but it is genuine branching
  that the autodiff/partial-eval machinery must carry.)
- **No second branch ⇒ no double-branch NaN-gradient hazard.** Under `vmap`,
  `lax.cond` lowers to `select` and executes **both** branches; if a branch
  ever contains an `x`-dependent unsafe op (`1/x`, `sqrt`, `log`), `vmap(grad)`
  leaks NaN into *valid* lanes. The current NaN branch (`full_like(x, nan)`) is
  constant in `x` so it is safe today, but it is a latent trap the moment a
  value-dependent branch is added. `error_if` has no second branch and sidesteps
  the entire class.
- **Reject path stops cleanly.** On invalid input there is no fabricated value
  and therefore no (meaningless) gradient — the program raises instead of
  silently differentiating a NaN. This is exactly the "actually rejected"
  semantics this plan is for.

Mandatory differentiability coverage (add to the test migration): for at least
one migrated constructor, assert `jax.grad`/`jax.value_and_grad` through a
**valid** build returns finite gradients, and that `eqx.filter_jit(grad(...))`
agrees with the eager gradient. This locks in that switching from `lax.cond` to
`error_if` did not perturb gradients on the accept path.

---

## 3. Inventory (what gets touched)

Public constructors (validation boundary):

```
beam_types.py     create_electron_beam            ← Anti-pattern A (6 NaN sites)
crystal_types.py  create_crystal_structure        ← Anti-pattern B
                  create_ewald_data               ← audit (negative-value tests exist)
                  create_kirkland_parameters
                  create_potential_slices
                  create_xyz_data                 ← Anti-pattern B (finite/sign)
distributions.py  create_orientation_distribution, create_discrete_orientation,
                  create_gaussian_orientation, create_mixed_orientation,
                  create_lognormal_size           ← audit
rheed_types.py    create_rheed_pattern, create_rheed_image, create_sliced_crystal
                                                  ← audit
```

Counts found: `beam_types.py` = 6 `jnp.full_like(..., nan)` substitutions;
`crystal_types.py` = 9 `raise` sites (subset are value checks to migrate, rest
are static shape checks to keep).

Tests that encode the wrong contract (sample):
- `test_beam_types.py` — ~10 `assert jnp.isnan(...)` "should produce NaN" cases.
- `test_crystal_types.py:304-329` — `test_ewald_data_negative_*` cases.

---

## 4. Target patterns

### 4a. Constructor value check → `eqx.error_if`

Before (`beam_types.py`):
```python
def _check_energy() -> scalar_float:
    valid = jnp.logical_and(energy_kev >= 5.0, energy_kev <= 100.0)
    return lax.cond(valid, lambda: energy_kev,
                    lambda: jnp.full_like(energy_kev, jnp.nan))
```

After:
```python
def _check_energy() -> scalar_float:
    # reject (raise) instead of poisoning with NaN; jit-safe
    return eqx.error_if(
        energy_kev,
        jnp.logical_or(energy_kev < 5.0, energy_kev > 100.0),
        "energy_kev must be in [5, 100] keV",
    )
```
The returned (checked) value flows into the final `ElectronBeam(...)`, so the
check is not DCE'd.

Before (`crystal_types.py`, value check):
```python
if not jnp.all(jnp.isfinite(positions)):
    raise ValueError("positions contain non-finite values")
```
After:
```python
positions = eqx.error_if(
    positions, jnp.any(~jnp.isfinite(positions)),
    "positions contain non-finite values",
)
```
Keep the **shape** checks exactly as they are (static, JIT-safe):
```python
if positions.shape[1] != max_dims:           # unchanged — trace-time known
    raise ValueError("positions must have shape (N, 3)")
```

### 4b. Numerical entry point → opt-in `checkify`

Expose a checked variant rather than forcing callbacks into the hot path:
```python
from jax.experimental import checkify

def ewald_simulator(...):                     # unchanged hot path
    ...

# opt-in, caller controls
checked_ewald = checkify.checkify(
    ewald_simulator, errors=checkify.nan_checks | checkify.div_checks
)
err, pattern = jax.jit(checked_ewald)(crystal, ...)
err.throw()                                   # raises on NaN / division by zero
```
Document this in the checked-entry-points guide; do not change the default
signature. `user_checks` are intentionally excluded from shared raw numerical
paths because bare `checkify.check(...)` calls break standard `jax.jit`/`grad`
unless the call is functionalized by a checked wrapper.

---

## 5. Test migration

`assert jnp.isnan(...)` → assert the construction is **rejected**, eager and
jitted. Add a shared helper:

```python
# tests/conftest.py (or a test util)
import equinox as eqx
import pytest

def assert_rejects(fn, *args, match=None, **kwargs):
    """The call must raise both eagerly and under jit.

    Use ``eqx.filter_jit``, NOT bare ``jax.jit``: under ``jax.jit`` the
    ``error_if`` callback still fires and still raises, but the message is an
    uninformative Equinox blob ("wrap your program with equinox.filter_jit"),
    so a ``match=`` regex would not match. ``filter_jit`` surfaces the real
    message string.
    """
    with pytest.raises(Exception, match=match):
        fn(*args, **kwargs)                       # eager
    with pytest.raises(Exception, match=match):
        eqx.filter_jit(lambda: fn(*args, **kwargs))()   # jitted
```

Before:
```python
def test_energy_too_low(self):
    beam = create_electron_beam(energy_kev=3.0, ...)
    assert jnp.isnan(beam.energy_kev)
```
After:
```python
def test_energy_too_low(self):
    assert_rejects(create_electron_beam, energy_kev=3.0, ...,
                   match="energy_kev must be in")
```
Keep one positive test per constructor asserting a valid build still works and
is finite (`chex.assert_tree_all_finite`).

Note on exception type (confirmed empirically, equinox 0.13.8 / jax 0.9.2):
- Eager: raises `equinox._errors._EquinoxRuntimeError` (with the real message).
- Under `eqx.filter_jit`: raises `jax.errors.JaxRuntimeError` wrapping the
  Equinox message — `match=` works.
- Under bare `jax.jit`: still raises (`JaxRuntimeError`) but the message is the
  uninformative blob — `match=` will NOT match.

Catch broadly (`pytest.raises(Exception, match=...)`) via the helper rather than
pinning these classes; both are private/backend-ish.

---

## 6. Risks & edge cases

1. **DCE of unused checks** — must thread the value returned by `error_if`
   into the constructed object. Add a test that a jitted invalid build raises
   (proves the check survived tracing).
2. **Behavior switch** `EQX_ON_ERROR` — default `raise`. Tests must run in
   `raise` mode. Decide whether production should default to `raise` or `nan`
   (recommended: `raise`; document in CONTRIBUTING).
3. **Performance** — host-callback checks cost per call; array-wide `isfinite`
   over large arrays is the heaviest. Constructors are setup-time so this is
   fine; do **not** push `error_if` into per-step hot loops — use checkify
   there.
4. **Removing NaN substitution is a behavior change** — anything downstream
   that currently tolerates/relies on NaN beams must be updated. Grep for
   consumers that special-case NaN before deleting the substitution.
5. **`grad`/`vmap`** — checks pass through grad; under vmap a batched predicate
   rejects if any lane is invalid. Add a vmap rejection test for at least one
   constructor. (See the Differentiability section — grad-safety is verified.)
6. **checkify functionalization** — `checkify.check` does nothing unless the
   function is wrapped; never rely on a bare `check` call. Entry-point wrappers
   must `err.throw()`.
7. **Complementary, not redundant, with jaxtyping/beartype** — jaxtyping +
   beartype already enforce shape/dtype at runtime; `error_if`/`checkify` add
   **value** semantics (sign, finiteness, range). Keep both.
8. **Equinox pin** — relies on `eqx.error_if` (present in 0.13.8). Note the
   minimum in `pyproject.toml` if not already constrained tightly enough.
9. **`error_if` needs `eqx.filter_jit` for usable errors** — under bare
   `jax.jit` the check still fires and still raises, but the message is the
   uninformative blob ("wrap your program with equinox.filter_jit"), so a
   `match=` regex won't match. Use `eqx.filter_jit` for any jitted call you want
   a readable / matchable error from. Constructors are usually called eagerly,
   so this mostly affects jitted *tests* (hence the helper uses `filter_jit`).
10. **`error_if` is a host callback (`pure_callback`)** — implications:
    (a) the checked value must be consumed, else DCE drops the check (see #1);
    (b) per-call runtime cost (see #3); (c) callbacks are unsupported in some
    lowering contexts — AOT / `jax.export`, and certain sharded / multi-device
    or `pmap` settings — where Equinox's `EQX_ON_ERROR=nan` fallback (or
    `checkify`) is required. **Differentiation is unaffected**: the callback
    lives in the primal trace; the value path is a clean identity passthrough.

---

## 7. Phased rollout

- **Phase 0 — policy + scaffolding** — DONE
  - Decide the split (error_if for constructors, checkify opt-in for sims).
  - Add `assert_rejects` test helper. Decide `EQX_ON_ERROR` default; document
    in CONTRIBUTING "Coding Standards".
- **Phase 1 — `beam_types.create_electron_beam`** — DONE
  - Replace the 6 NaN substitutions with `eqx.error_if`.
  - Migrate `test_beam_types.py`: NaN asserts → `assert_rejects` (eager + jit).
  - Confirm concrete exception classes; tighten helper.
- **Phase 2 — `crystal_types`** — DONE
  - Convert value checks (finite/sign/range) to `error_if`; keep static shape
    checks as Python `if`. Cover `create_crystal_structure`, `create_xyz_data`,
    `create_ewald_data` (+ the negative-value tests at 304-329).
- **Phase 3 — `distributions` + `rheed_types` constructors** — DONE
  - Audit each `create_*`; apply the same value-vs-shape split.
- **Phase 4 — numerical entry points (optional, opt-in)** — DONE
  - Implemented: `checked_ewald_simulator`, `checked_simulate_detector_image`,
    `checked_multislice_propagate`, `checked_multislice_simulator` in
    `simul/simulator.py` via `checkify.checkify(fn, errors=_CHECKIFY_ERRORS)`,
    exported from `simul`. Raw signatures unchanged; smoke-tested under
    `jax.jit`; usage + caveats documented in a comment block above them.
  - Implemented: `checked_weighted_image_residual` and
    `checked_weighted_mean_squared_error` in `recon/losses.py` with the same
    opt-in `checkify` contract, exported from `recon`.
  - **Finding — `user_checks` dropped.** Error set is now
    `nan_checks | div_checks` (was `| user_checks`). A bare `checkify.check`
    inside a *shared* simulator raises *"Cannot abstractly evaluate a
    checkify.check which was not functionalized"* under plain `jax.jit` /
    `jax.grad` — it would break the raw, differentiable call path. So
    `user_checks` is meaningless here and was removed; only re-enable it with
    checks living in code reached **exclusively** via the checkify wrappers.
  - **Differentiability caveat (documented):** checked variants return
    `(err, out)` and are **not** drop-in differentiable; differentiate the raw
    simulators and use `checked_*` for validation/debug runs only.
  - Remaining (optional): a dedicated checkify-only function if specific
    `checkify.check` value guards are wanted in the future.
- **Phase 5 — docs + verification** — DONE
  - Checked-vs-standard numerical entry point guide added under the canonical
    `docs/source/guides` tree.
  - Verify: `uv run pytest` green; explicit jitted-rejection tests added;
    `uv run ty check tests` clean; `uv run ruff check src tests` clean.

---

## 8. Closed decisions

1. **Scope**: migrate all public constructor validation boundaries named in the
   inventory, not just the beam pilot.
2. **Entry-point checkify**: include simulator entry points and reconstruction
   loss helpers as opt-in checked variants.
3. **`EQX_ON_ERROR` default**: rely on Equinox's default `raise` behavior in
   tests and normal use.
4. **Exception contract in tests**: use broad `pytest.raises(Exception,
   match=...)` through the shared helper for constructor rejection tests, since
   backend exception classes are private or backend-dependent.
5. **NaN substitution**: remove it from validation boundaries; invalid inputs
   now reject instead of manufacturing NaN-valued domain objects.
