# JAX Transformability

Rheedium is a JAX framework, but JAX's three transforms (`grad`, `vmap`,
`jit`) are **not** uniformly available across every function. This guide
states exactly what is supported where, so you can structure code without
surprises.

## Summary matrix

| Layer | `grad` | `vmap` | `jit` |
|---|:---:|:---:|:---:|
| Numerical kernels — form factors, projected potentials, structure factors (fixed input shapes) | ✅ | ✅ | ✅ |
| String-mode simulators — `compute_kinematic_intensities_with_ctrs`, `projected_potential`, surface-rod profiles | ✅ | ✅ | ✅ *(string arg must be static — see below)* |
| Grid/reflection builders — `sliced_crystal_to_projected_potential_slices`, `multislice_simulator` | ✅ † | ⚠️ | ❌ *(data-dependent output shapes)* |
| Construction factories — `create_crystal_structure`, `create_xyz_data`, … | n/a | n/a | ❌ *(by design: Python validation)* |

† `grad` flows through the grid builders **with respect to continuous
parameters** (atom positions, temperature, calibrations). The grid
*dimensions* themselves are treated as concrete Python values, so you cannot
differentiate the output shape — only the values on a fixed grid.

## `grad` and `vmap`: supported throughout

Differentiability is the headline feature and it holds for the parameters you
actually optimize. Gradients flow from a loss through the Ewald/CTR pipeline
back to atom positions, cell parameters, beam energy, and orientation weights
(this is what the `rheedium.recon` module relies on). `vmap` works for
parameter sweeps — azimuthal scans, orientation distributions, batched
structures.

## `jit`: three cases

### 1. Fixed-shape kernels — just works

Functions whose output shape is fixed by their array inputs `jit` directly:

```python
import jax
from rheedium.simul import atomic_scattering_factor

f = jax.jit(atomic_scattering_factor)
```

### 2. String-mode simulators — make the string static

Some simulators select behaviour with a Python `str` argument
(`ctr_mixing_mode`, `parameterization`, `profile_type`). A traced string can't
drive a Python `if`, so the string must be a **static** argument. Two ways:

```python
import jax, equinox as eqx
from rheedium.simul.simulator import compute_kinematic_intensities_with_ctrs

# Option A: stdlib JAX — name the static argument explicitly.
jitted = jax.jit(
    compute_kinematic_intensities_with_ctrs,
    static_argnames=("ctr_mixing_mode",),
)

# Option B: equinox (a rheedium dependency) — non-array args are made
# static automatically, so no static_argnames bookkeeping is needed.
jitted = eqx.filter_jit(compute_kinematic_intensities_with_ctrs)
```

Both produce a fully compiled function. The output shape is set by the input
reflection array, so nothing else blocks compilation.

### 3. Grid / reflection builders — not jittable as written

The multislice potential builder sizes its grid from continuous geometry:

```python
nx = int(jnp.ceil(x_extent / pixel_size))   # concretizes a value -> shape
```

and the Ewald construction keeps a *data-dependent number* of reflections.
JAX requires static output shapes under `jit`, so these top-level builders —
`sliced_crystal_to_projected_potential_slices`, `multislice_simulator`, and the
reflection-selecting Ewald path — **cannot** be `jit`-compiled directly. Run
them eagerly (they are still GPU-accelerated and differentiable w.r.t.
continuous inputs), and `jit` the fixed-shape kernels they feed. If you need a
fully compiled pipeline, fix the grid dimensions / reflection count up front
(pass them as static integers, or pad to a maximum) so the shapes become
static.

## Rule of thumb

- Optimizing parameters? Use `grad`/`vmap` freely — supported everywhere.
- Compiling a hot kernel with fixed array shapes? `jit` it directly.
- Compiling a simulator with a `str` mode? Mark the string static
  (`static_argnames=...` or `eqx.filter_jit`).
- A function that builds a grid or selects reflections from data? Keep it
  eager, or fix its sizes to make it `jit`-able.
