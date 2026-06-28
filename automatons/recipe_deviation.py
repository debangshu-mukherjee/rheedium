# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Report recipe deviation with rheedium.recon.

The automaton wraps :func:`rheedium.recon.recipe_deviation` around a compact
linear inverse fixture. Smoke mode plants a measured "actual" recipe, compares
it with a deliberately mismatched intended recipe, and emits the frozen
recipe-deviation report payload that downstream control logic can consume.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any
from jaxtyping import Array, Float

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import RecipeDeviationReport, ReconProblem

_DESIGN: Float[Array, "pixels params"] = jnp.asarray(
    [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
    dtype=jnp.float64,
)


def _forward(params: Float[Array, "params"]) -> Float[Array, "pixels"]:
    """Map recipe parameters to synthetic measurements."""
    pixels: Float[Array, "pixels"] = _DESIGN @ params
    return pixels


def _array_param(
    values: list[Any], *, size: int, name: str
) -> Float[Array, "n"]:
    """Convert a JSON list parameter to a fixed-length float array."""
    array: Float[Array, "n"] = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have length {size}")
    return array


def _load_array(path: str) -> Float[Array, "pixels"]:
    """Load a one-dimensional detector vector from ``.npy`` or ``.npz``."""
    artifact = Path(path)
    if artifact.suffix == ".npy":
        raw = np.load(artifact)
    elif artifact.suffix == ".npz":
        with np.load(artifact) as data:
            key: str = (
                "measured" if "measured" in data.files else data.files[0]
            )
            raw = data[key]
    else:
        raise ValueError("measured_array must be a .npy or .npz file")
    array: Float[Array, "pixels"] = jnp.asarray(raw, dtype=jnp.float64)
    if array.ndim != 1:
        raise ValueError(f"measured array must be 1D; got {array.shape}")
    return array


@experiment(
    name="recipe-deviation",
    params=[
        Param(
            "measured_array",
            str,
            default="",
            help="Optional measured detector vector as .npy or .npz.",
            example="measured_recipe.npz",
        ),
        Param(
            "true_params",
            list,
            default=[1.5, -0.5],
            help="Synthetic actual recipe parameters.",
            example=[1.5, -0.5],
        ),
        Param(
            "intended_params",
            list,
            default=[1.0, -0.5],
            help="Intended recipe parameters.",
            example=[1.0, -0.5],
        ),
        Param(
            "initial_latent",
            list,
            default=[0.0, 0.0],
            help="Initial inverse-solver latent vector.",
            example=[0.0, 0.0],
        ),
        Param(
            "parameter_uncertainty",
            list,
            default=[0.1, 0.1],
            help="Per-parameter one-sigma uncertainty.",
            example=[0.1, 0.1],
        ),
        Param("max_steps", int, default=32, help="Maximum solver steps."),
        Param(
            "warning_z",
            float,
            default=2.0,
            help="Warning threshold in absolute z-score.",
        ),
        Param(
            "critical_z",
            float,
            default=3.0,
            help="Critical threshold in absolute z-score.",
        ),
        Param(
            "sigma_floor",
            float,
            default=1e-8,
            help="Minimum uncertainty denominator.",
        ),
    ],
    returns={
        "metrics": {
            "severity": {"type": "integer"},
            "max_abs_z": {"type": "number"},
            "first_deviation": {"type": "number"},
        },
        "artifacts": {"roles": ["recipe_deviation_report", "covariance"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Solve and report the synthetic recipe deviation."""
    true_params: Float[Array, "params"] = _array_param(
        args.true_params,
        size=2,
        name="true_params",
    )
    intended_params: Float[Array, "params"] = _array_param(
        args.intended_params,
        size=2,
        name="intended_params",
    )
    initial_latent: Float[Array, "params"] = _array_param(
        args.initial_latent,
        size=2,
        name="initial_latent",
    )
    uncertainty: Float[Array, "params"] = _array_param(
        args.parameter_uncertainty,
        size=2,
        name="parameter_uncertainty",
    )
    if args.measured_array:
        measured: Float[Array, "pixels"] = _load_array(args.measured_array)
    else:
        measured = _forward(true_params)
    problem = ReconProblem(forward=_forward, measured=measured)
    report: RecipeDeviationReport = rh.recon.recipe_deviation(
        problem=problem,
        intended_params=intended_params,
        initial_latent=initial_latent,
        parameter_uncertainty=uncertainty,
        warning_z=args.warning_z,
        critical_z=args.critical_z,
        sigma_floor=args.sigma_floor,
        max_steps=args.max_steps,
    )
    payload: dict[str, Any] = rh.recon.recipe_deviation_report_payload(report)
    rh.recon.validate_recipe_deviation_report(payload)
    report_artifact = ctx.save_json(
        "recipe_deviation_report.json",
        payload,
        role="recipe_deviation_report",
    )
    covariance_artifact = ctx.save_array(
        "recipe_deviation_covariance.npz",
        {"covariance": np.asarray(report.parameter_covariance)},
        role="covariance",
    )
    first_parameter: dict[str, Any] = payload["parameters"][0]
    metrics: dict[str, Any] = {
        "severity": int(payload["severity"]),
        "max_abs_z": float(payload["max_abs_z"]),
        "first_deviation": float(first_parameter["deviation"]),
        "first_z_score": float(first_parameter["z_score"]),
        "solver_loss": float(payload["solver"]["loss"]),
        "solver_converged": bool(payload["solver"]["converged"]),
    }
    return {
        "metrics": metrics,
        "artifacts": [report_artifact, covariance_artifact],
        "recipe_deviation": payload,
    }


if __name__ == "__main__":
    main()
