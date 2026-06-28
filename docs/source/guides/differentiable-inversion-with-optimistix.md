# Differentiable Inversion With Optimistix

`rheedium.recon` exposes one shared inverse-problem surface for detector
alignment, distribution reconstruction, structure fitting, uncertainty
quantification, and recipe-deviation reporting. The core object is
`rheedium.types.ReconProblem`: a differentiable forward model, measured data, an optional
latent-to-physical transform, and the residual/loss functions used by the
solver.

The public import surface intended for downstream automatons is:

```python
from rheedium.recon import (
    fit_geometry_beam,
    multistart,
    recipe_deviation,
    reconstruct_distribution,
    sample_posterior,
    solve,
)
from rheedium.types import ReconProblem
```

## Cache Before Solving

JAX compiles the optimizer scan for the traced pytree shapes it sees. For
production inversions, enable the persistent compilation cache before the first
solve in the process:

```python
from rheedium.tools import enable_compilation_cache

enable_compilation_cache("/scratch/$USER/rheedium-xla")
```

The `solve` and `multistart` entry points also accept `compilation_cache_dir`
for workflows that want the inversion call itself to enable the cache before
lowering the optimizer executable:

```python
result = solve(
    problem,
    initial_latent,
    max_steps=64,
    compilation_cache_dir="/scratch/$USER/rheedium-xla",
)
```

Use fixed detector shapes and fixed static options for hot loops. New shapes,
new static arguments, or a different pytree structure still compile once before
the persistent cache can reuse the executable.

## Define a Problem

Separate unconstrained optimizer coordinates from physical parameters. Put
bounds, positivity, simplex constraints, and lattice constraints in
`transform`; keep the forward model written against physical values.

```python
import jax.numpy as jnp

from rheedium.recon import positive_from_unconstrained, solve
from rheedium.types import ReconProblem


def transform(latent):
    return {
        "thickness": positive_from_unconstrained(latent[0], minimum=0.0),
        "roughness": positive_from_unconstrained(latent[1], minimum=0.0),
    }


def forward(params):
    return simulator(
        thickness=params["thickness"],
        roughness=params["roughness"],
    )


problem = ReconProblem(
    forward=forward,
    measured=measured_image,
    transform=transform,
)

result = solve(
    problem,
    initial_latent=jnp.zeros(2),
    mode="least_squares",
    max_steps=64,
)
```

`result.params` is the fitted physical pytree, `result.latent_params` is the
final unconstrained coordinate pytree, and `result.loss` is the final scalar
objective.

## Seed Multiple Basins

Use `multistart` when symmetry, detector ambiguity, or a nonconvex structure
model creates multiple plausible basins. Starts can be supplied explicitly or
generated reproducibly around a template latent.

```python
best = multistart(
    problem,
    initial_latent_template,
    key=random_key,
    n_starts=16,
    random_scale=0.5,
    max_steps=64,
)
```

For orientation problems, seed starts with the known point-group orbit when the
crystal has exact symmetry-equivalent optima.

## Fit Geometry And Beam

`fit_geometry_beam` is a convenience wrapper for the well-posed case where the
crystal is fixed and the lab supplies a calibrated detector forward closure.
The transform returns `(orientation, beam_modes)`; the forward closure receives
`(crystal, orientation, beam_modes)`.

```python
from rheedium.recon import fit_geometry_beam
from rheedium.types import BeamModeDistribution


def transform(latent):
    orientation = latent[:2]
    beam_modes = BeamModeDistribution(
        beta_in_plane=0.0,
        beta_out_of_plane=0.0,
        divergence_in_plane_rad=latent[2],
        divergence_out_of_plane_rad=latent[3],
        energy_spread_ev=latent[4],
        distribution_id="fit",
    )
    return orientation, beam_modes


orientation, beam_modes, covariance = fit_geometry_beam(
    crystal=known_crystal,
    measured=measured_image,
    forward=detector_forward,
    initial_latent=initial_guess,
    transform=transform,
    compilation_cache_dir="/scratch/$USER/rheedium-xla",
)
```

The returned covariance is the Laplace covariance flattened over the physical
`(orientation, beam_modes)` pytree.

## Quantify Uncertainty

Use the local Fisher/Laplace helpers for fast Gaussian bands around a fitted
solution, or `sample_posterior` when the posterior is non-Gaussian or
multimodal.

```python
from rheedium.recon import laplace_uncertainty, sample_posterior


def residual_from_physical(params):
    return forward(params) - measured_image


uncertainty = laplace_uncertainty(
    residual_from_physical,
    result.params,
    noise_variance=noise_variance,
)

posterior = sample_posterior(
    log_posterior,
    result.params,
    key=random_key,
    n_chains=4,
    n_samples=512,
)
```

`recipe_deviation` uses the same uncertainty machinery. If no uncertainty is
supplied, it estimates a Laplace covariance in the fitted physical-parameter
basis and emits the schema-validated automaton payload through
`recipe_deviation_report_payload`.

## Exporting A Hot Loop

Forward simulators should be exported with `rheedium.tools.export_forward` when
they need a portable StableHLO artifact. For inversion, keep the deployed loop
fixed-shape, enable the persistent cache before the first solve, and export only
closed-over pure functions whose static options are already frozen.
