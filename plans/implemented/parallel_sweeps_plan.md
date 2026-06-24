# Parallelizing RHEED Simulations — Minimal-Code Plan

Scope: `rheedium` — distribute the existing parameter-sweep simulations across
multiple JAX devices (multi-GPU on one host, or multi-host via SLURM) with the
**smallest possible code addition**, reusing the infrastructure already present
(`tools/parallel.shard_array`, `init_distributed`, the `jax.vmap` sweep
wrappers, and `eqx.filter_jit`).

Status: **implemented** — `tools.distribute_batched` added with tests
(`TestDistributeBatched`). One detail changed during implementation: instead of
committing the input via `shard_array` + `eqx.filter_jit`, the helper compiles
`batched_fn` with `jax.jit` and explicit `in_shardings`/`out_shardings`. Bare
input commitment let GSPMD raise an empty-abstract-mesh error when propagating
sharding through the simulator's beam-averaging quadrature stacks (a 4-D
intermediate); pinning the boundary specs fixes it. The optional §3.2 sweep
passthrough was not added — the closure form proved clean enough.

---

## 1. Problem statement

The simulation core produces one detector image per parameter set:

- `simul.simulate_detector_image(...) -> Float[Array, "H W"]`
  ([simulator.py:1273](src/rheedium/simul/simulator.py#L1273)).

The sweep layer batches over a parameter bank with `jax.vmap`
([sweeps.py](src/rheedium/simul/sweeps.py)):

- `simulate_detector_image_phi_sweep`, `_theta_sweep`, `_energy_sweep`,
  `_orientation_sweep`, `_roughness_sweep` → `Float[Array, "N H W"]`
- `simulate_detector_image_all_sweep` / `_parameter_grid` (nested vmap) →
  `Float[Array, "N_phi N_theta N_voltage H W"]`

**The gap:** `jax.vmap` *vectorizes* the batch but executes it on a **single
device**. None of the sweep functions touch more than one GPU/host. The only
distribution primitive in the package — `tools/parallel.shard_array`
([parallel.py:30](src/rheedium/tools/parallel.py#L30)) — is fully tested
([test_parallel.py](tests/test_rheedium/test_tools/test_parallel.py)) but **is
never called by any simulation code**. Multi-host bootstrap (`init_distributed`,
[\_\_init\_\_.py](src/rheedium/__init__.py)) is wired but, again, nothing
downstream consumes the resulting device mesh.

Today's scaling workaround is a manual Python `for`-loop that slices the bank
into chunks and runs them **sequentially** to bound memory
(`_chunked_phi_bank` in
[generate_bi2se3_sweeps.py:19](tutorials/sweeps/generate_bi2se3_sweeps.py#L19)) —
the opposite of parallel.

The sweeps are **embarrassingly parallel**: every image is independent. This is
the ideal target for data parallelism.

---

## 2. Why the minimal approach is "shard the batch axis under `jit`"

JAX offers three ways to use multiple devices. We want the one that adds the
least code:

| Mechanism | New code required | Composes with existing `vmap` sweeps? |
|-----------|-------------------|----------------------------------------|
| `jax.pmap` | Rewrite each sweep; leading axis must equal device count; nested `vmap`/`pmap` is awkward | Poorly — would replace, not reuse |
| `shard_map` | Write an explicit per-shard function for every sweep | No — more code per sweep |
| **`jit` + sharded inputs (GSPMD / automatic SPMD)** | **One thin wrapper; sweeps untouched** | **Yes — wraps the existing callable** |

The third path is the canonical "data parallelism with minimal code" idiom:

1. Build a 1-D device mesh (reuse the mesh logic already in `shard_array`).
2. Shard the **leading (batch) axis** of the parameter bank across the mesh;
   replicate everything else (`crystal`, scalar params).
3. Run the **existing** `jax.vmap`-wrapped sweep under `eqx.filter_jit`. XLA's
   SPMD partitioner sees the sharded input and **automatically** splits the
   batched program across devices — no per-device code.
4. Gather the result (or keep it sharded for a downstream sharded reduction).

This works because the sweeps are *already* `vmap`-ed pure functions with a
fixed `(H, W)` output, so they are equally `jit`-traceable. The `crystal`
argument is an `eqx.Module` PyTree closed over inside `_simulate_one` (not a
vmap axis), so it replicates naturally. `eqx.filter_jit` is the right JIT
boundary — it is already used elsewhere
([surface_rods.py:696](src/rheedium/simul/surface_rods.py#L696),
[orientation.py:804](src/rheedium/recon/orientation.py#L804)) and it handles the
static Python args (`image_shape_px`, `n_angular_samples`, the
`CrystalStructure` static fields) that plain `jax.jit` would choke on.

---

## 3. Design — one new helper, zero changes to the sweeps

### 3.1 New function: `tools/parallel.distribute_batched`

Add a single generic wrapper to
[parallel.py](src/rheedium/tools/parallel.py). It takes any already-`vmap`-ed
callable plus its batched argument(s) and runs it data-parallel across the mesh.

```python
def distribute_batched(
    batched_fn,                       # e.g. a *_sweep function, or jax.vmap(f)
    batch_arg,                        # the leading-axis array to shard (the bank)
    *,
    devices=None,                     # default: jax.devices()
    pad_value=0.0,                    # value used to pad to a device multiple
) -> Array:
    """Run a batched callable data-parallel over a 1-D device mesh.

    Pads the bank to a multiple of len(devices), shards axis 0 across the
    mesh via `shard_array`, runs `eqx.filter_jit(batched_fn)` so XLA's SPMD
    partitioner distributes the work, then trims the padding.
    """
```

Implementation outline (~30–40 lines, reusing `shard_array`):

1. `devices = devices or jax.devices()`; `n = len(devices)`.
2. `orig = batch_arg.shape[0]`; pad axis 0 up to `ceil(orig/n)*n` with
   `pad_value` (sharding a 1-D mesh requires the batch length be divisible by
   `n`). Padding, not truncation, so coverage is never silently dropped.
3. `sharded = shard_array(padded_bank, shard_axes=0, devices=devices)` —
   **reuses the existing, tested utility**.
4. `out = eqx.filter_jit(batched_fn)(sharded)` inside the mesh context. XLA
   shards the output's leading axis to match; `out_shardings` is inferred.
5. Return `out[:orig]` (trim padding). Optionally `jax.device_get` to gather,
   or leave sharded for a downstream sharded reduction.

`batched_fn` is a one-argument closure over the bank — callers pass
`lambda bank: simulate_detector_image_phi_sweep(crystal, bank, **settings)`,
so **the sweep functions in `sweeps.py` need no edits at all**.

### 3.2 Optional thin ergonomic layer (only if desired)

If we want a turnkey call rather than asking users to write the closure, add one
keyword to the sweep signatures — `devices: Sequence[jax.Device] | None = None`
— and, when non-`None`, route the final `jax.vmap(_simulate_one)(bank)` through
`distribute_batched`. That is a 2-line change per sweep (guard + delegate) and
is **optional**; the closure form in §3.1 already covers every case with zero
sweep edits. Recommended: ship §3.1 first, add §3.2 only if the closure proves
clumsy in the tutorials.

### 3.3 What stays untouched

- `simulate_detector_image` and all physics kernels — unchanged.
- `sweeps.py` — unchanged (closure form) or +2 lines/function (optional form).
- `init_distributed` / multi-host bootstrap — already correct; the mesh built
  from `jax.devices()` spans all hosts once `init_distributed()` has run, so the
  **same** `distribute_batched` works single-host multi-GPU *and* multi-host
  SLURM with no branching.

---

## 4. Implementation steps

1. **`distribute_batched`** in
   [parallel.py](src/rheedium/tools/parallel.py): implement per §3.1; reuse
   `shard_array`; import `equinox as eqx`. Add to `__all__`.
2. **Export** it from
   [tools/\_\_init\_\_.py](src/rheedium/tools/__init__.py) `__all__` and the
   Routine Listings docstring (mirror the `shard_array` entry).
3. **Padding helper** (inline or tiny private `_pad_to_multiple`): pad axis 0,
   return `(padded, original_len)`. Keep it in `parallel.py`.
4. **(Optional, §3.2)** add the `devices=` keyword to the five 1-D sweeps and
   the grid sweep; guard-and-delegate to `distribute_batched`.
5. **Tutorial migration (demonstration of value):** replace the sequential
   `_chunked_phi_bank` loop in
   [generate_bi2se3_sweeps.py](tutorials/sweeps/generate_bi2se3_sweeps.py) with
   a single `distribute_batched` call. This both documents the API and converts
   the existing sequential chunking into real parallelism. Repeat for the STO
   and MgO sweep generators if desired.

---

## 5. Testing

Reuse the existing 8-virtual-CPU-device harness — `conftest` already sets
`XLA_FLAGS=--xla_force_host_platform_device_count=8`
([test_parallel.py](tests/test_rheedium/test_tools/test_parallel.py) header), so
multi-device behavior is testable on CPU in CI with no GPU.

Add to `test_parallel.py` (or a sibling `test_distribute.py`):

1. **Correctness vs. serial:** `distribute_batched(vmapped_f, bank)` equals the
   plain `jax.vmap`/serial result to tolerance (`chex.assert_trees_all_close`).
   Use a cheap analytic `f` first, then a small real
   `simulate_detector_image_phi_sweep` closure on a tiny crystal fixture.
2. **Non-divisible batch:** `N` not a multiple of 8 (e.g. 5, 13) → output shape
   is exactly `N` (padding trimmed) and values match serial.
3. **Output sharding:** result's leading axis is sharded across all 8 devices
   (assert `len(result.sharding.device_set) == 8` before gather).
4. **Replication:** confirm the crystal/scalars are not required to be sharded
   (closure captures them; no shape error).
5. **Dtype/precision:** float64 preserved (mirrors existing
   `test_float64_preserved`).

Run: `uv run pytest tests/test_rheedium/test_tools/ -q`.

---

## 6. Risks & mitigations

- **Batch not divisible by device count** → handled by pad-then-trim (§3.1.2);
  never truncate. Padding cost is at most `n_devices - 1` extra images.
- **Per-image memory × (batch/devices):** sharding *reduces* per-device batch vs
  today's single-device vmap, so memory improves. For very large grids, combine
  with the existing chunking loop (chunk outer, shard inner).
- **Static-arg retracing:** `eqx.filter_jit` treats `image_shape_px`,
  sample counts, and `CrystalStructure` static fields as static — each distinct
  shape/config compiles once, then caches. Same trade-off the package already
  accepts at its JIT boundaries.
- **Multislice is out of scope:** `multislice_simulator` and
  `sliced_crystal_to_projected_potential_slices` size outputs from data and are
  **not** `jit`-compilable as written (see the JAX-transformability note in
  [simul/\_\_init\_\_.py:116](src/rheedium/simul/__init__.py#L116)). Per-image
  data-parallel sweeping of multislice would need fixed grid sizes first; treat
  as a separate follow-up, not part of this minimal change.
- **`pmap` legacy tests:** the existing `TestPmapCompatibility` tests stay as-is
  — they validate the primitive, not the sweep path; no conflict.

---

## 7. Summary of the diff surface

| File | Change | Size |
|------|--------|------|
| `src/rheedium/tools/parallel.py` | add `distribute_batched` (+ `_pad_to_multiple`) | ~40 lines |
| `src/rheedium/tools/__init__.py` | export + docstring entry | ~3 lines |
| `src/rheedium/simul/sweeps.py` | *(optional §3.2)* `devices=` passthrough | ~2 lines × 6 |
| `tests/.../test_parallel.py` | data-parallel correctness tests | ~5 tests |
| `tutorials/sweeps/generate_*_sweeps.py` | swap sequential chunk loop → one call | demo only |

Net new **production** code: effectively **one function**. Everything else —
the mesh, the sharding, the multi-host bootstrap, the batched sweeps, the JIT
boundary — already exists and is reused.
