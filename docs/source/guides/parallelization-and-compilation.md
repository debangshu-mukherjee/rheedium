# Parallelization and Compilation

Rheedium runs on JAX, so the forward models compile through XLA and parallelize
across devices with the same machinery as any JAX program. This guide covers
the three practical levers: **caching compilation** so you pay for it once,
**distributing batched work** across devices, and **exporting** compiled
artifacts for distribution. For *which* transforms (`grad`/`vmap`/`jit`) apply
to *which* functions, see [JAX Transformability](jax-transformability.md); this
guide assumes that and focuses on making compilation and parallelism fast and
reproducible.

## The compilation model in one paragraph

XLA specializes every compiled executable on the **shapes and dtypes** it is
traced with, recompiling whenever a new shape signature appears. In rheedium the
genuinely variable dimensions are the **atom count** (per crystal), the
**reflection count** (`hmax`/`kmax`, held static so the grid is fixed), and the
**detector / FFT grid** (`H × W`). Everything below follows from that: caching
saves a recompile *per shape*, export with symbolic shapes lets one artifact
span a *range* of atom counts, and the FFT in the multislice path forces a
*concrete* grid.

## Persistent compilation cache

By default JAX recompiles from scratch in every fresh process. The XLA
persistent cache writes compiled executables to disk so a later run — a new
worker, a CI job, the next cell in a notebook — loads them instead.

It is **opt-in** and must be configured **before the first compilation**. The
simplest way is the environment trigger, read at import:

```bash
# Use the default location (~/.cache/rheedium/xla), namespaced per architecture
export RHEEDIUM_COMPILATION_CACHE=1

# ...or point it somewhere explicit
export RHEEDIUM_CACHE_DIR=/scratch/$USER/rheedium-xla
```

```python
import rheedium as rh   # cache is enabled here, before anything compiles
```

For interactive sessions, call it directly before your first simulation:

```python
from rheedium.tools import enable_compilation_cache

enable_compilation_cache()                       # default location, per-arch
enable_compilation_cache("/scratch/me/xla")      # explicit root
```

What it does and does not do:

- **Per-shape, not universal.** Each new shape signature still compiles once,
  then every later process reuses it. A warm process skips the compile entirely
  (roughly halving end-to-end time for the small kinematic kernel, and saving
  far more for the heavier multislice graph).
- **Not portable.** Cached executables are tied to the backend (CPU ≠ CUDA) and
  the jaxlib version. Treat the cache as a local accelerator, not a shippable
  binary — for distribution use [export](#exporting-compiled-artifacts).
- **Architecture-namespaced by default.** XLA:CPU code is specialized on the
  host CPU feature set (`avx512`, …). A cache shared between cluster nodes with
  different CPUs could hand a worker an executable its host cannot run
  (risking `SIGILL`). `enable_compilation_cache(per_arch=True)` (the default)
  appends an OS/machine/CPU-flags tag to the directory so heterogeneous nodes
  never cross executables. Keep it on for shared filesystems.

```python
# Only cache executables that took real time to build; cap tiny entries.
enable_compilation_cache(min_compile_time_secs=1.0)
```

## Parallelizing across devices

The forward models are **embarrassingly parallel over a batch axis** — an
azimuthal scan, an orientation distribution, a parameter grid. The natural batch
sources are the [`rheedium.simul`](../api/index.rst) sweep helpers
(`simulate_detector_image_phi_sweep`, `..._orientation_sweep`,
`..._parameter_grid`, …), which produce one batched array.

### `distribute_batched` — data-parallel sweeps

`distribute_batched` runs an already-batched callable across every available
device. It compiles the function with `jax.jit` and explicit in/out shardings
that split the **leading (batch) axis** across a 1-D device mesh; XLA's SPMD
partitioner distributes the work and auto-partitions intermediates. Arrays the
callable closes over (e.g. a `CrystalStructure`) are **replicated**, not
sharded.

```python
import jax
from rheedium.tools import distribute_batched
from rheedium.simul import ewald_simulator

# A per-sample simulator vmapped over a stack of azimuths, then distributed.
phis = jnp.linspace(0.0, 90.0, 256)

def one(phi):
    return ewald_simulator(crystal, phi_deg=phi).intensities

batched = jax.vmap(one)
images = distribute_batched(batched, phis)   # sharded over all devices
```

The batch length need **not** be a multiple of the device count: the leading
axis is padded up to the next multiple before sharding and trimmed afterward, so
coverage is never silently dropped (the padded rows use `pad_value`, default
`0.0`, and are discarded).

### `shard_array` — placing an array across devices

For finer control, `shard_array` distributes a chosen axis (or axes) of an array
across a device mesh:

```python
from rheedium.tools import shard_array

sharded = shard_array(big_batch, shard_axes=0)   # shard the leading axis
sharded = shard_array(big_batch, shard_axes=-1)  # -1 = do not shard
```

### Testing parallel code locally

On a single CPU you can still exercise the multi-device paths by asking XLA for
several virtual devices **before importing JAX** — this is exactly what the test
harness does:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python my_sweep.py
```

### Multi-host (SLURM) execution

For multi-node jobs, initialize the cluster once on every rank. Set
`RHEEDIUM_DISTRIBUTED=1` before launching under `srun` and the package calls
`jax.distributed.initialize` on import; an optional
`RHEEDIUM_COORDINATOR_ADDRESS` overrides automatic SLURM coordinator detection.
For interactive multi-host sessions, call `rheedium.init_distributed(force=True)`
directly. Initialization is guarded (idempotent, degrades to a warning) but is a
**collective** operation — reach it from all ranks, not from a transitive
single-rank import.

## Exporting compiled artifacts

To ship a compiled forward model — so downstream users pay no compilation cost —
use `jax.export` through the `rheedium.tools` helpers, which produce a portable
StableHLO artifact. Two properties of the models shape what is possible.

**The kinematic / Ewald path is shape-polymorphic in the atom count.** Because
the reflection grid is sized statically from `hmax`/`kmax`, a single artifact can
serve any crystal size via a symbolic dimension:

```python
import jax, jax.numpy as jnp
from rheedium.tools import (
    export_forward, serialize_exported, deserialize_exported,
)

# Array-only wrapper; static options (hmax/kmax/parameterization) closed over.
def forward(frac, cart, cell_lengths, cell_angles, voltage, theta, phi):
    crystal = CrystalStructure(frac, cart, cell_lengths, cell_angles)
    return ewald_simulator(
        crystal, voltage_kv=voltage, theta_deg=theta, phi_deg=phi,
        hmax=5, kmax=5, parameterization="lobato",
    ).intensities

(n,) = jax.export.symbolic_shape("n")            # symbolic atom count
sds = lambda shp: jax.ShapeDtypeStruct(shp, jnp.float64)
exported = export_forward(
    forward,
    sds((n, 4)), sds((n, 4)), sds((3,)), sds((3,)), sds(()), sds(()), sds(()),
)

blob = serialize_exported(exported)              # bytes -> write to disk
reloaded = deserialize_exported(blob)            # later / elsewhere
images = reloaded.call(frac, cart, lengths, angles, v, th, ph)  # any N
```

**Runtime checks block export.** Functions carrying `equinox.error_if` checks —
the factory validators and the [`checked_*` simulators](checked-numerical-entry-points.md)
— lower to host callbacks that cannot be serialized. Because `equinox` reads
`EQX_ON_ERROR` at import and never re-reads it, the checks can only be disabled
**before importing rheedium**. The package exposes an opt-in env knob for this
(checks stay on by default):

```bash
# rheedium sets EQX_ON_ERROR=off ahead of the first equinox import
RHEEDIUM_DISABLE_RUNTIME_CHECKS=1 python export_my_model.py

# ...or set the equinox variable yourself; an explicit value always wins
EQX_ON_ERROR=nan python export_my_model.py
```

Only use this for trusted, pre-validated data — it disables input validation
globally, so a bad crystal that would have raised now passes through. If you
forget, `export_forward` detects the host-callback failure and raises
`ExportError` with this guidance, so it is an explicit error, not a silent one.

**The multislice path cannot be shape-polymorphic.** Its FFT requires a
*concrete* transform size — XLA rejects a symbolic grid. Export one artifact per
detector grid and dispatch with `bucketize_grid`, padding the input up to the
nearest pre-exported bucket:

```python
from rheedium.tools import bucketize_grid

bucket = bucketize_grid(height, width, buckets=(256, 512, 1024))
# -> e.g. (512, 512): pad the input to `bucket` and call that artifact.
```

`export_forward` translates a symbolic-FFT attempt into an `ExportError`
pointing here, rather than surfacing a raw XLA lowering error.

## Rule of thumb

- Re-running the same shapes across processes? **Enable the persistent cache**
  (`RHEEDIUM_COMPILATION_CACHE=1`); keep `per_arch` on for shared filesystems.
- Sweeping a parameter across many devices? Build the batch with a `*_sweep`
  helper (or `jax.vmap`) and hand it to **`distribute_batched`**.
- Multi-node job? Set **`RHEEDIUM_DISTRIBUTED=1`** and reach `init_distributed`
  on every rank.
- Shipping a compiled model? **`export_forward`** with a symbolic atom count for
  the kinematic path; bucketed concrete grids (`bucketize_grid`) for multislice;
  set `EQX_ON_ERROR=nan` before import.
