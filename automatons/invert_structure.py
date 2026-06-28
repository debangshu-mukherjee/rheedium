# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Invert synthetic structure parameters with rheedium.recon.solve.

The automaton wraps the general :func:`rheedium.recon.solve` entry point around
a deterministic linear structure surrogate. The three recovered parameters act
as a compact stand-in for displacement, composition, and thickness latents, so
smoke mode proves the ReconProblem/solve handoff without paying for a full
atomic detector simulation.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype.typing import Any
from jaxtyping import Array, Float

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import ReconProblem, ReconResult

_STRUCTURE_MATRIX: Float[Array, "pixels params"] = jnp.asarray(
    [
        [1.0, 0.0, 0.5],
        [0.0, 2.0, -0.25],
        [1.0, -1.0, 0.75],
        [0.5, 0.25, 1.25],
        [-0.75, 0.5, 1.0],
    ],
    dtype=jnp.float64,
)


def _structure_forward(
    params: Float[Array, "params"],
) -> Float[Array, "pixels"]:
    """Map structure latents to synthetic detector pixels."""
    pixels: Float[Array, "pixels"] = _STRUCTURE_MATRIX @ params
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
    name="invert-structure",
    params=[
        Param(
            "measured_array",
            str,
            default="",
            help="Optional measured detector vector as .npy or .npz.",
            example="measured_structure.npz",
        ),
        Param(
            "target_params",
            list,
            default=[0.15, -0.25, 1.2],
            help="Synthetic target [displacement, composition, thickness].",
            example=[0.15, -0.25, 1.2],
        ),
        Param(
            "initial_latent",
            list,
            default=[0.0, 0.0, 0.0],
            help="Initial structure latent vector.",
            example=[0.0, 0.0, 0.0],
        ),
        Param(
            "mode",
            str,
            default="least_squares",
            help="Recon solver mode.",
            choices=("least_squares", "bfgs", "adamw"),
            example="least_squares",
        ),
        Param("max_steps", int, default=32, help="Maximum solver steps."),
        Param("rtol", float, default=1e-8, help="Relative solver tolerance."),
        Param("atol", float, default=1e-10, help="Absolute solver tolerance."),
        Param(
            "learning_rate",
            float,
            default=1e-2,
            help="Learning rate for adamw mode.",
        ),
    ],
    returns={
        "metrics": {
            "param_l2_error": {"type": "number"},
            "final_loss": {"type": "number"},
            "converged": {"type": "boolean"},
        },
        "artifacts": {"roles": ["structure_fit", "structure_arrays"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Solve the synthetic structure inverse problem."""
    target_params: Float[Array, "params"] = _array_param(
        args.target_params,
        size=3,
        name="target_params",
    )
    initial_latent: Float[Array, "params"] = _array_param(
        args.initial_latent,
        size=3,
        name="initial_latent",
    )
    if args.measured_array:
        measured: Float[Array, "pixels"] = _load_array(args.measured_array)
    else:
        measured = _structure_forward(target_params)
    problem = ReconProblem(
        forward=_structure_forward,
        measured=measured,
    )
    result: ReconResult = rh.recon.solve(
        problem=problem,
        initial_latent=initial_latent,
        mode=args.mode,
        max_steps=args.max_steps,
        rtol=args.rtol,
        atol=args.atol,
        learning_rate=args.learning_rate,
    )
    fitted: Float[Array, "params"] = jnp.asarray(
        result.params,
        dtype=jnp.float64,
    )
    param_l2_error: float = float(jnp.linalg.norm(fitted - target_params))
    payload: dict[str, Any] = {
        "parameter_names": [
            "displacement_latent",
            "composition_latent",
            "thickness_latent",
        ],
        "fitted_params": np.asarray(fitted).tolist(),
        "target_params": np.asarray(target_params).tolist(),
        "residual": np.asarray(result.residual).tolist(),
        "solver": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "loss": float(result.loss),
            "status": result.solver_status,
        },
    }
    fit_artifact = ctx.save_json(
        "structure_fit.json",
        payload,
        role="structure_fit",
    )
    array_artifact = ctx.save_array(
        "structure_fit.npz",
        {
            "fitted_params": np.asarray(fitted),
            "target_params": np.asarray(target_params),
            "measured": np.asarray(measured),
            "simulated": np.asarray(result.simulated),
            "residual": np.asarray(result.residual),
        },
        role="structure_arrays",
    )
    metrics: dict[str, Any] = {
        "param_l2_error": param_l2_error,
        "final_loss": float(result.loss),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
    }
    return {
        "metrics": metrics,
        "artifacts": [fit_artifact, array_artifact],
        "structure_fit": payload,
    }


if __name__ == "__main__":
    main()
