# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Emit a monotone detector-ensemble convergence diagnostic.

The automaton evaluates a controlled mode-count sequence through
``rheedium.simul.apply_distribution`` and compares each approximate detector
image to a zero-perturbation reference. The synthetic amplitude kernel is built
so residuals decrease with mode count, making the smoke gate a deterministic
guard for convergence-curve artifact plumbing.
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


def _mode_counts(values: list[Any]) -> list[int]:
    """Convert a JSON list into a positive increasing mode-count sequence."""
    counts: list[int] = [int(value) for value in values]
    if not counts:
        raise ValueError("mode_counts must contain at least one value")
    if any(count <= 0 for count in counts):
        raise ValueError("mode_counts must be positive")
    if counts != sorted(counts):
        raise ValueError("mode_counts must be sorted ascending")
    return counts


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
    """Create a complex amplitude kernel for convergence estimates."""
    yy, xx = _coordinate_grid(image_size)
    base = jnp.exp(-((xx**2) + (yy + 0.15) ** 2) / 0.08)
    detail = 0.35 * jnp.exp(-(((xx - 0.35) ** 2) + ((yy - 0.25) ** 2)) / 0.05)

    @jaxtyped(typechecker=beartype)
    def _kernel(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
        scale: Float[Array, ""] = sample[0]
        phase = jnp.exp(1j * scale * (xx - yy))
        amplitude: Complex[Array, "H W"] = (base + scale * detail) * phase
        return amplitude

    return _kernel


def _estimate_image(
    *,
    n_modes: int,
    image_size: int,
) -> Float[Array, "H W"]:
    """Return one mode-count approximation with a shrinking perturbation."""
    scale: float = 1.0 / float(n_modes)
    distribution: Distribution = rh.types.create_distribution(
        samples=jnp.full((n_modes, 1), scale, dtype=jnp.float64),
        weights=jnp.ones(n_modes, dtype=jnp.float64) / n_modes,
        reduction=ReductionMode.INCOHERENT,
        axis_id="mode_count_convergence",
    )
    image: Float[Array, "H W"] = rh.simul.apply_distribution(
        distribution,
        _amplitude_kernel(image_size),
    )
    return image


def _reference_image(image_size: int) -> Float[Array, "H W"]:
    """Return the zero-perturbation reference image."""
    distribution: Distribution = rh.types.create_distribution(
        samples=jnp.zeros((1, 1), dtype=jnp.float64),
        weights=jnp.ones(1, dtype=jnp.float64),
        reduction=ReductionMode.INCOHERENT,
        axis_id="mode_count_reference",
    )
    image: Float[Array, "H W"] = rh.simul.apply_distribution(
        distribution,
        _amplitude_kernel(image_size),
    )
    return image


@experiment(
    name="convergence-study",
    params=[
        Param(
            "mode_counts",
            list,
            default=[1, 2, 4, 8],
            help="Increasing mode-count sequence.",
            example=[1, 2, 4, 8],
        ),
        Param("image_size", int, default=64, help="Square detector size."),
    ],
    returns={
        "metrics": {
            "n_mode_counts": {"type": "integer"},
            "residual_monotone": {"type": "boolean"},
            "final_residual": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "convergence_summary",
                "convergence_arrays",
                "convergence_curve",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run a deterministic mode-count convergence study."""
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )
    counts: list[int] = _mode_counts(args.mode_counts)
    if args.smoke:
        counts = counts[:4]
    reference: Float[Array, "H W"] = _reference_image(image_size)
    images: list[Float[Array, "H W"]] = [
        _estimate_image(n_modes=count, image_size=image_size)
        for count in counts
    ]
    residuals_np = np.asarray(
        [
            float(jnp.linalg.norm(image - reference))
            / float(jnp.linalg.norm(reference))
            for image in images
        ],
        dtype=np.float64,
    )
    residual_monotone: bool = bool(np.all(np.diff(residuals_np) <= 1e-12))
    rows: list[dict[str, float | int]] = [
        {"mode_count": count, "relative_residual": float(residual)}
        for count, residual in zip(counts, residuals_np, strict=True)
    ]
    image_stack = np.stack([np.asarray(image) for image in images], axis=0)
    curve_image = residuals_np[None, :]

    summary_artifact = ctx.save_json(
        "convergence_study.json",
        {"rows": rows, "residual_monotone": residual_monotone},
        role="convergence_summary",
    )
    array_artifact = ctx.save_array(
        "convergence_study.npz",
        {
            "mode_counts": np.asarray(counts, dtype=np.int64),
            "relative_residual": residuals_np,
            "images": image_stack,
            "reference": np.asarray(reference),
        },
        role="convergence_arrays",
    )
    curve_artifact = ctx.save_image(
        "convergence_curve.png",
        curve_image,
        cmap="phosphor",
        role="convergence_curve",
    )
    preview_artifact = ctx.save_image(
        "convergence_final.png",
        images[-1],
        cmap="phosphor",
        role="convergence_preview",
    )
    metrics: dict[str, Any] = {
        "n_mode_counts": len(counts),
        "residual_monotone": residual_monotone,
        "initial_residual": float(residuals_np[0]),
        "final_residual": float(residuals_np[-1]),
        "residual_ratio": float(residuals_np[-1] / residuals_np[0]),
    }
    return {
        "metrics": metrics,
        "artifacts": [
            summary_artifact,
            array_artifact,
            curve_artifact,
            preview_artifact,
        ],
        "convergence": rows,
    }


if __name__ == "__main__":
    main()
