# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Recover discrete orientation weights from a synthetic detector pattern.

The automaton is the narrow orientation-inversion diagnostic for G6. It plants
a small orientation mixture, synthesizes a measured image with
``integrate_over_orientation``, fits the weights with
``rheedium.recon.fit_orientation_weights``, and writes the recovered weights,
loss endpoints, residual image, and a gradient-finiteness check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import OrientationDistribution, OrientationFitResult
from rheedium.types.custom_types import scalar_float


@jaxtyped(typechecker=beartype)
def _synthetic_pattern(phi_deg: scalar_float) -> Float[Array, "rows cols"]:
    """Map an orientation angle to a distinct positive detector image."""
    x: scalar_float = jnp.asarray(phi_deg, dtype=jnp.float64) / 10.0
    return jnp.asarray(
        [
            [1.0 + x, 0.5 + x**2, 0.25 + x**3],
            [1.5 + x**4, 0.1 + x**5, 0.2 + x + x**2],
        ],
        dtype=jnp.float64,
    )


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


def _weight_loss_gradient(
    angles_deg: Float[Array, "N"],
    weights: Float[Array, "N"],
    observed: Float[Array, "rows cols"],
) -> Float[Array, "N"]:
    """Differentiate the orientation loss with respect to weights."""

    def _loss(current_weights: Float[Array, "N"]) -> Float[Array, ""]:
        distribution = rh.types.create_orientation_distribution(
            angles_deg=angles_deg,
            weights=current_weights,
            mosaic_fwhm_deg=0.0,
            distribution_id="gradient_probe",
        )
        loss: Float[Array, ""] = rh.recon.orientation_loss(
            distribution=distribution,
            simulate_fn=_synthetic_pattern,
            observed_pattern=observed,
            normalize=False,
            regularization_strength=0.0,
            entropy_weight=0.0,
            n_mosaic_points=1,
        )
        return loss

    gradient: Float[Array, "N"] = jax.grad(_loss)(weights)
    return gradient


@experiment(
    name="reconstruct-orientation",
    params=[
        Param(
            "candidate_angles_deg",
            list,
            default=[0.0, 10.0, 20.0],
            help="Discrete candidate orientation support.",
            example=[0.0, 10.0, 20.0],
        ),
        Param(
            "target_weights",
            list,
            default=[0.15, 0.55, 0.30],
            help="Synthetic planted orientation weights.",
            example=[0.15, 0.55, 0.30],
        ),
        Param(
            "learning_rate",
            float,
            default=0.05,
            help="AdamW learning rate for the orientation fit.",
        ),
        Param("n_iterations", int, default=400, help="Maximum fit steps."),
        Param(
            "convergence_tol",
            float,
            default=1e-9,
            help="Absolute solver-loss tolerance.",
        ),
    ],
    returns={
        "metrics": {
            "weight_l1_error": {"type": "number"},
            "final_loss": {"type": "number"},
            "gradient_finite": {"type": "boolean"},
        },
        "artifacts": {
            "roles": [
                "orientation_fit",
                "orientation_arrays",
                "residual_image",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Fit discrete orientation weights for the synthetic diagnostic."""
    angles: Float[Array, "N"] = _array_param(
        args.candidate_angles_deg,
        name="candidate_angles_deg",
    )
    target_weights: Float[Array, "N"] = _normalise_weights(
        _array_param(args.target_weights, name="target_weights")
    )
    if angles.shape != target_weights.shape:
        raise ValueError("candidate_angles_deg and target_weights must match")
    iterations: int = (
        min(args.n_iterations, 260) if args.smoke else args.n_iterations
    )
    target_distribution: OrientationDistribution = (
        rh.types.create_discrete_orientation(
            angles_deg=angles,
            weights=target_weights,
            distribution_id="synthetic_orientation_truth",
        )
    )
    observed: Float[Array, "rows cols"] = rh.types.integrate_over_orientation(
        _synthetic_pattern,
        target_distribution,
        n_mosaic_points=1,
    )
    uniform_weights: Float[Array, "N"] = (
        jnp.ones_like(target_weights) / target_weights.shape[0]
    )
    initial_distribution = rh.types.create_discrete_orientation(
        angles_deg=angles,
        weights=uniform_weights,
        distribution_id="uniform_start",
    )
    initial_loss: Float[Array, ""] = rh.recon.orientation_loss(
        distribution=initial_distribution,
        simulate_fn=_synthetic_pattern,
        observed_pattern=observed,
        normalize=False,
        regularization_strength=0.0,
        entropy_weight=0.0,
        n_mosaic_points=1,
    )
    result: OrientationFitResult = rh.recon.fit_orientation_weights(
        observed_pattern=observed,
        simulate_fn=_synthetic_pattern,
        candidate_angles_deg=angles,
        initial_weights=uniform_weights,
        learning_rate=args.learning_rate,
        n_iterations=iterations,
        convergence_tol=args.convergence_tol,
        regularization_strength=0.0,
        entropy_weight=0.0,
        n_mosaic_points=1,
        normalize=False,
    )
    fitted_weights: Float[Array, "N"] = (
        result.fitted_distribution.discrete_weights
    )
    gradient: Float[Array, "N"] = _weight_loss_gradient(
        angles,
        fitted_weights,
        observed,
    )
    gradient_finite: bool = bool(jnp.all(jnp.isfinite(gradient)))
    weight_l1_error: float = float(
        jnp.sum(jnp.abs(fitted_weights - target_weights))
    )
    loss_curve: Float[Array, "two"] = jnp.asarray(
        [initial_loss, result.final_loss],
        dtype=jnp.float64,
    )
    fit_payload: dict[str, Any] = {
        "candidate_angles_deg": np.asarray(angles).tolist(),
        "target_weights": np.asarray(target_weights).tolist(),
        "fitted_weights": np.asarray(fitted_weights).tolist(),
        "loss_curve": np.asarray(loss_curve).tolist(),
        "gradient": np.asarray(gradient).tolist(),
        "converged": bool(result.converged),
    }

    fit_artifact = ctx.save_json(
        "reconstruct_orientation.json",
        fit_payload,
        role="orientation_fit",
    )
    array_artifact = ctx.save_array(
        "reconstruct_orientation.npz",
        {
            "candidate_angles_deg": np.asarray(angles),
            "target_weights": np.asarray(target_weights),
            "fitted_weights": np.asarray(fitted_weights),
            "loss_curve": np.asarray(loss_curve),
            "gradient": np.asarray(gradient),
            "observed": np.asarray(observed),
            "residual": np.asarray(result.residual_pattern),
        },
        role="orientation_arrays",
    )
    residual_artifact = ctx.save_image(
        "orientation_residual.png",
        jnp.abs(result.residual_pattern),
        cmap="phosphor",
        role="residual_image",
    )
    metrics: dict[str, Any] = {
        "weight_l1_error": weight_l1_error,
        "final_loss": float(result.final_loss),
        "initial_loss": float(initial_loss),
        "gradient_finite": gradient_finite,
        "n_iterations": int(result.n_iterations),
    }
    return {
        "metrics": metrics,
        "artifacts": [fit_artifact, array_artifact, residual_artifact],
        "orientation_fit": fit_payload,
    }


if __name__ == "__main__":
    main()
