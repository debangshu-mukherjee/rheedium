# Checked vs Standard Numerical Entry Points

Rheedium exposes two styles of numerical entry point for selected
workflows:

- **standard functions**, such as `ewald_simulator`
- **checked functions**, such as `checked_ewald_simulator` or
  `checked_weighted_mean_squared_error`

They compute the same numerical result on valid inputs, but they have different
error-reporting contracts.

## Standard Functions

Use the standard function for normal simulation work:

```python
from rheedium.simul import ewald_simulator

pattern = ewald_simulator(
    crystal=crystal,
    voltage_kv=20.0,
    theta_deg=2.0,
    phi_deg=0.0,
)
```

The standard function returns the simulation output directly:

```python
RHEEDPattern
```

This is the preferred form for production simulations, reconstruction
objectives, optimization loops, and performance-sensitive code. It keeps the
hot path clean and avoids threading error state through every caller.

## Checked Functions

Use the checked function when you want JAX-functional numerical error checking:

```python
import equinox as eqx
from rheedium.simul import checked_ewald_simulator

err, pattern = eqx.filter_jit(checked_ewald_simulator)(
    crystal=crystal,
    voltage_kv=20.0,
    theta_deg=2.0,
    phi_deg=0.0,
)
err.throw()
```

The checked function returns:

```python
tuple[checkify.Error, RHEEDPattern]
```

The first value, `err`, accumulates numerical errors discovered by
`jax.experimental.checkify`. Calling `err.throw()` raises if a checked numerical
error occurred. If no error occurred, `err.throw()` is a no-op.

## What The Checked Version Checks

The checked entry points use:

```python
checkify.nan_checks | checkify.div_checks
```

This means they can detect:

- NaNs produced inside the checked computation
- division-by-zero events detected by JAX checkify

Both are automatic: they instrument the existing array operations and need no
changes to the simulator code.

### Why not `user_checks`?

`checkify.user_checks` is intentionally **not** enabled. It only does anything
when the running code contains explicit `checkify.check(...)` calls, and such a
call cannot be added to the shared simulators: a bare `checkify.check` inside a
function raises *"Cannot abstractly evaluate a checkify.check which was not
functionalized"* under plain `jax.jit` or `jax.grad`. That would break the raw,
differentiable call path (`ewald_simulator` and friends) used everywhere else.
Only enable `user_checks` alongside checks that live in code reached
exclusively through a `checkify.checkify` wrapper.

These checks are functionalized by JAX. They are not ordinary Python exceptions
raised directly at the point of failure. Instead, the error is returned as data:

```python
err, output = checked_function(...)
err.throw()
```

This makes the mechanism compatible with JAX transformations such as `jax.jit`.

## Available Checked Entry Points

The checked simulator variants are exported from `rheedium.simul`:

| Standard | Checked |
|----------|---------|
| `ewald_simulator` | `checked_ewald_simulator` |
| `simulate_detector_image` | `checked_simulate_detector_image` |
| `multislice_propagate` | `checked_multislice_propagate` |
| `multislice_simulator` | `checked_multislice_simulator` |

The checked reconstruction-loss variants are exported from `rheedium.recon`:

| Standard | Checked |
|----------|---------|
| `weighted_image_residual` | `checked_weighted_image_residual` |
| `weighted_mean_squared_error` | `checked_weighted_mean_squared_error` |

## When To Use The Standard Version

Use the standard version when:

- you are running normal simulations
- you are inside a tight optimization loop
- you are evaluating reconstruction losses in a gradient calculation
- you already validated inputs at constructor boundaries
- you want the simplest return type
- you want the same direct API used throughout most examples

Example:

```python
image = simulate_detector_image(
    crystal=crystal,
    voltage_kv=20.0,
    theta_deg=2.0,
)
```

## When To Use The Checked Version

Use the checked version when:

- you are debugging a new model or workflow
- you suspect NaNs are being introduced inside a numerical kernel
- you want CI coverage that catches numerical failures at simulation boundaries
- you are validating a new differentiable reconstruction or fitting pipeline
- you need a JIT-compatible way to surface numerical errors

Example:

```python
import equinox as eqx
from rheedium.simul import checked_multislice_propagate

err, exit_wave = eqx.filter_jit(checked_multislice_propagate)(
    potential_slices,
    20.0,
    2.0,
)
err.throw()
```

For reconstruction losses:

```python
import equinox as eqx
from rheedium.recon import checked_weighted_mean_squared_error

err, loss = eqx.filter_jit(checked_weighted_mean_squared_error)(
    simulated_image,
    experimental_image,
    weight_map,
)
err.throw()
```

## Relationship To Constructor Validation

Constructor validation and checked numerical entry points solve different
problems.

Constructors use `equinox.error_if` to reject invalid data at validation
boundaries:

```python
beam = create_electron_beam(energy_kev=20.0)
```

Invalid constructor inputs are rejected eagerly and under JIT. For example,
negative beam energies, negative calibration values, and non-finite array
values should fail before they enter the simulation pipeline.

Checked numerical entry points are for failures produced during computation:

```python
err, pattern = eqx.filter_jit(checked_ewald_simulator)(...)
err.throw()
```

They are useful for detecting numerical instability even when the inputs were
valid.

## Differentiability

Differentiability is a core rheedium goal, and the **standard** functions are
the differentiable path. They are ordinary JAX functions and the default choice
for gradient-based workflows.

The checked functions wrap those same functions with `checkify.checkify`. On
valid inputs the numerical output is identical, and on invalid numerical states
the checked version returns an error object so the caller can raise deliberately
with `err.throw()`.

However, a checked function returns an `(err, out)` **tuple**, so it is **not**
drop-in differentiable: `jax.grad(checked_ewald_simulator)` will not work the
way `jax.grad` over a loss built on the standard `ewald_simulator` does.
Differentiate the standard simulators and standard losses; reach for the
checked variants in validation and debugging runs, not inside a gradient
computation.

## Practical Rule

Use this rule of thumb:

| Situation | Use |
|-----------|-----|
| routine simulation | standard function |
| optimization or reconstruction hot path | standard function |
| differentiating a reconstruction loss | standard function |
| debugging NaNs | checked function |
| CI numerical stability test | checked function |
| validating a new simulation workflow | checked function |
| checking input shapes, ranges, or finite values | constructor validation |

The checked functions are safety rails you opt into at numerical boundaries.
They are not replacements for the standard API.
