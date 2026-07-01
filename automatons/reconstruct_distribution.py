# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.10"]
# ///
"""Recover an incoherent distribution with rheedium.recon.

The automaton wraps :func:`rheedium.recon.reconstruct_distribution` around a
small three-template detector library. Smoke mode plants a known probability
simplex, mixes the library intensities, reconstructs the distribution weights,
and writes a JSON distribution artifact plus the recovered uncertainty band.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import Distribution, DistributionAxisSpec


@jaxtyped(typechecker=beartype)
def _amplitude_templates() -> Float[Array, "samples rows cols"]:
    """Return independent synthetic amplitude templates."""
    return jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 1.0]],
        ],
        dtype=jnp.float64,
    )


def _one_hot_perturbation(
    base_templates: Float[Array, "samples rows cols"],
    sample: Float[Array, "samples"],
) -> Float[Array, "rows cols"]:
    """Select one amplitude template from a one-hot sample coordinate."""
    amplitude: Float[Array, "rows cols"] = jnp.einsum(
        "n,nhw->hw",
        sample,
        base_templates,
    )
    return amplitude


def _identity_forward(
    amplitude: Float[Array, "rows cols"],
) -> Float[Array, "rows cols"]:
    """Return a synthetic coherent amplitude image unchanged."""
    return amplitude


def _load_image(path: str) -> Float[Array, "rows cols"]:
    """Load a two-dimensional measured image from ``.npy`` or ``.npz``."""
    artifact = Path(path)
    if artifact.suffix == ".npy":
        raw = np.load(artifact)
    elif artifact.suffix == ".npz":
        with np.load(artifact) as data:
            key: str = "image" if "image" in data.files else data.files[0]
            raw = data[key]
    else:
        raise ValueError("measured_array must be a .npy or .npz file")
    image: Float[Array, "rows cols"] = jnp.asarray(raw, dtype=jnp.float64)
    if image.ndim != 2:
        raise ValueError(f"measured image must be 2D; got {image.shape}")
    return image


def _simplex_param(
    values: list[Any],
    *,
    size: int,
    name: str,
) -> Float[Array, "samples"]:
    """Convert a JSON list parameter to a normalized non-negative simplex."""
    raw: Float[Array, "samples"] = jnp.asarray(values, dtype=jnp.float64)
    if raw.shape != (size,):
        raise ValueError(f"{name} must have length {size}")
    clipped: Float[Array, "samples"] = jnp.clip(raw, 0.0, None)
    total: Float[Array, ""] = jnp.sum(clipped)
    if float(total) <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return clipped / total


@experiment(
    name="reconstruct-distribution",
    params=[
        Param(
            "measured_array",
            str,
            default="",
            help="Optional measured detector image as .npy or .npz.",
            example="measured_distribution.npz",
        ),
        Param(
            "target_weights",
            list,
            default=[0.2, 0.5, 0.3],
            help="Synthetic planted weights for the three-sample fixture.",
            example=[0.2, 0.5, 0.3],
        ),
        Param(
            "ridge",
            float,
            default=1e-12,
            help="Tikhonov ridge for weight reconstruction.",
            bounds=(0.0, 1.0),
        ),
        Param(
            "noise_variance",
            float,
            default=0.05,
            help="Per-pixel noise variance for the one-sigma band.",
            bounds=(1e-12, 10.0),
        ),
    ],
    returns={
        "metrics": {
            "weight_l1_error": {"type": "number"},
            "max_band": {"type": "number"},
            "effective_weight_count": {"type": "number"},
        },
        "artifacts": {"roles": ["distribution", "distribution_arrays"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Recover the planted incoherent distribution fixture."""
    samples: Float[Array, "samples sample_dim"] = jnp.eye(
        3,
        dtype=jnp.float64,
    )
    axis_spec: DistributionAxisSpec = rh.types.create_distribution_axis_spec(
        samples=samples,
        perturbation_fn=_one_hot_perturbation,
        forward_model=_identity_forward,
        output_kind="amplitude",
        axis_id="synthetic_axis",
    )
    base_templates: Float[Array, "samples rows cols"] = _amplitude_templates()
    intensity_library: Float[Array, "samples rows cols"] = (
        rh.recon.build_incoherent_intensity_library(
            base_object=base_templates,
            axis_spec=axis_spec,
        )
    )
    target_weights: Float[Array, "samples"] = _simplex_param(
        args.target_weights,
        size=3,
        name="target_weights",
    )
    if args.measured_array:
        measured: Float[Array, "rows cols"] = _load_image(args.measured_array)
    else:
        measured = jnp.einsum(
            "n,nhw->hw",
            target_weights,
            intensity_library,
        )

    distribution: Distribution
    band: Float[Array, "samples"]
    distribution, band = rh.recon.reconstruct_distribution(
        measured_image=measured,
        base_object=base_templates,
        axis_spec=axis_spec,
        ridge=args.ridge,
        noise_variance=args.noise_variance,
    )

    weights: Float[Array, "samples"] = distribution.weights
    weight_l1_error: float = float(jnp.sum(jnp.abs(weights - target_weights)))
    effective_weight_count: float = float(1.0 / jnp.sum(weights**2))
    payload: dict[str, Any] = {
        "axis_id": distribution.axis_id,
        "samples": np.asarray(distribution.samples).tolist(),
        "weights": np.asarray(weights).tolist(),
        "target_weights": np.asarray(target_weights).tolist(),
        "band": np.asarray(band).tolist(),
    }
    distribution_artifact = ctx.save_json(
        "reconstructed_distribution.json",
        payload,
        role="distribution",
    )
    array_artifact = ctx.save_array(
        "reconstructed_distribution.npz",
        {
            "samples": np.asarray(distribution.samples),
            "weights": np.asarray(weights),
            "target_weights": np.asarray(target_weights),
            "band": np.asarray(band),
            "measured": np.asarray(measured),
            "intensity_library": np.asarray(intensity_library),
        },
        role="distribution_arrays",
    )
    metrics: dict[str, Any] = {
        "weight_l1_error": weight_l1_error,
        "max_band": float(jnp.max(band)),
        "effective_weight_count": effective_weight_count,
        "best_sample_index": int(jnp.argmax(weights)),
    }
    return {
        "metrics": metrics,
        "artifacts": [distribution_artifact, array_artifact],
        "distribution": payload,
    }


if __name__ == "__main__":
    main()
