# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Average a small detector ensemble through rheedium distributions.

The automaton builds a generic weighted ``Distribution`` over synthetic
orientation-like samples, applies it with the simulator distribution reducer,
and writes the averaged detector image plus effective-count diagnostics. Smoke
mode is intentionally tiny but exercises the same coherent/incoherent reducer
used by beam, size, and orientation ensembles.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Complex, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import Distribution, ReductionMode


def _array_param(
    values: list[Any],
    *,
    name: str,
) -> Float[Array, "N"]:
    """Convert a JSON list parameter to a one-dimensional float array."""
    array: Float[Array, "N"] = jnp.asarray(values, dtype=jnp.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _normalise_weights(weights: Float[Array, "N"]) -> Float[Array, "N"]:
    """Return a finite non-negative probability simplex."""
    clipped: Float[Array, "N"] = jnp.clip(weights, 0.0, None)
    total: Float[Array, ""] = jnp.sum(clipped)
    if float(total) <= 0.0:
        raise ValueError("weights must contain positive mass")
    return clipped / total


@jaxtyped(typechecker=beartype)
def _coordinate_grid(
    image_size: int,
) -> tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """Return normalized image-coordinate grids."""
    axis: Float[Array, "S"] = jnp.linspace(-1.0, 1.0, image_size)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return yy, xx


def _amplitude_kernel(
    image_size: int,
) -> Any:
    """Create a complex amplitude kernel controlled by one sample value."""
    yy, xx = _coordinate_grid(image_size)

    @jaxtyped(typechecker=beartype)
    def _kernel(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
        phi_scale: Float[Array, ""] = sample[0] / 30.0
        first = jnp.exp(
            -((xx - 0.35 * phi_scale) ** 2 + (yy + 0.25) ** 2) / 0.045
        )
        second = 0.55 * jnp.exp(
            -((xx + 0.45 * phi_scale) ** 2 + (yy - 0.35) ** 2) / 0.07
        )
        phase = jnp.exp(1j * phi_scale * (2.0 * xx + yy))
        amplitude: Complex[Array, "H W"] = (first + second) * phase
        return amplitude

    return _kernel


@experiment(
    name="ensemble-average",
    params=[
        Param(
            "samples_deg",
            list,
            default=[-12.0, 0.0, 9.0, 18.0],
            help="Synthetic orientation-like samples in degrees.",
            example=[-12.0, 0.0, 9.0, 18.0],
        ),
        Param(
            "weights",
            list,
            default=[0.15, 0.35, 0.30, 0.20],
            help="Non-negative sample weights.",
            example=[0.15, 0.35, 0.30, 0.20],
        ),
        Param(
            "reduction",
            str,
            default="incoherent",
            help="Distribution reduction mode.",
            choices=("incoherent", "coherent"),
        ),
        Param("image_size", int, default=64, help="Square detector size."),
    ],
    returns={
        "metrics": {
            "mode_count": {"type": "integer"},
            "effective_count": {"type": "number"},
            "max_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": ["ensemble_summary", "ensemble_arrays", "ensemble_image"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Apply a generic ensemble distribution and write diagnostics."""
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )
    samples: Float[Array, "N"] = _array_param(
        args.samples_deg,
        name="samples_deg",
    )
    weights: Float[Array, "N"] = _normalise_weights(
        _array_param(args.weights, name="weights")
    )
    if samples.shape != weights.shape:
        raise ValueError("samples_deg and weights must have the same length")
    distribution: Distribution = rh.types.create_distribution(
        samples=samples[:, None],
        weights=weights,
        reduction=ReductionMode(args.reduction),
        axis_id="synthetic_orientation",
    )
    averaged: Float[Array, "H W"] = rh.simul.apply_distribution(
        distribution,
        _amplitude_kernel(image_size),
    )
    mode_count: int = int(jnp.sum(distribution.weights > 1e-12))
    effective_count: float = float(1.0 / jnp.sum(distribution.weights**2))
    summary: dict[str, Any] = {
        "axis_id": distribution.axis_id,
        "reduction": distribution.reduction.value,
        "samples_deg": np.asarray(samples).tolist(),
        "weights": np.asarray(distribution.weights).tolist(),
        "mode_count": mode_count,
        "effective_count": effective_count,
    }

    summary_artifact = ctx.save_json(
        "ensemble_average.json",
        summary,
        role="ensemble_summary",
    )
    array_artifact = ctx.save_array(
        "ensemble_average.npz",
        {
            "samples_deg": np.asarray(samples),
            "weights": np.asarray(distribution.weights),
            "average": np.asarray(averaged),
        },
        role="ensemble_arrays",
    )
    image_artifact = ctx.save_image(
        "ensemble_average.png",
        averaged,
        cmap="phosphor",
        role="ensemble_image",
    )
    metrics: dict[str, Any] = {
        "mode_count": mode_count,
        "effective_count": effective_count,
        "max_intensity": float(jnp.max(averaged)),
        "integrated_intensity": float(jnp.sum(averaged)),
    }
    return {
        "metrics": metrics,
        "artifacts": [summary_artifact, array_artifact, image_artifact],
        "ensemble": summary,
    }


if __name__ == "__main__":
    main()
