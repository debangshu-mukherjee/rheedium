r"""Recipe-deviation reporting for reconstruction outputs.

Extended Summary
----------------
This module provides the automaton-facing bridge from inversion to control:
solve a reconstruction problem, compare the fitted physical parameters with an
intended recipe, normalize the gaps by supplied uncertainty, and expose a small
severity signal.

Routine Listings
----------------
:class:`RecipeDeviationReport`
    Compare fitted reconstruction parameters with an intended recipe.
:func:`recipe_deviation`
    Solve an inverse problem and report signed recipe deviations.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import scalar_float

from .solve import ReconProblem, ReconResult, solve


class RecipeDeviationReport(eqx.Module):
    """Compare fitted reconstruction parameters with an intended recipe.

    :see: :class:`~.test_deviation.TestRecipeDeviation`

    Attributes
    ----------
    result : ReconResult
        Reconstruction result used as the fitted reality estimate.
    intended_params : Any
        Intended physical parameter pytree.
    deviation : Any
        Signed fitted-minus-intended parameter gap pytree.
    z_score : Any
        Gap normalized by per-parameter uncertainty.
    max_abs_z : Float[Array, ""]
        Maximum absolute z-score across flattened parameters.
    severity : Int[Array, ""]
        Severity code: 0 for matched, 1 for warning, 2 for critical.
    """

    result: ReconResult
    intended_params: Any
    deviation: Any
    z_score: Any
    max_abs_z: Float[Array, ""]
    severity: Int[Array, ""]


def _parameter_standard_deviation(
    parameter_uncertainty: Optional[Float[Array, "..."]],
    n_params: int,
    sigma_floor: scalar_float,
) -> Float[Array, "P"]:
    """Return flattened parameter standard deviations."""
    floor: Float[Array, ""] = jnp.asarray(sigma_floor, dtype=jnp.float64)
    if parameter_uncertainty is None:
        standard_deviation: Float[Array, "P"] = jnp.ones(
            n_params,
            dtype=jnp.float64,
        )
        return jnp.maximum(standard_deviation, floor)
    uncertainty: Float[Array, "..."] = jnp.asarray(
        parameter_uncertainty,
        dtype=jnp.float64,
    )
    if uncertainty.ndim == 2:
        diagonal: Float[Array, "P"] = jnp.diag(uncertainty)
        standard_deviation = jnp.sqrt(jnp.maximum(diagonal, 0.0))
    else:
        standard_deviation = jnp.ravel(uncertainty)
    if standard_deviation.shape[0] != n_params:
        raise ValueError("parameter_uncertainty has incompatible size")
    safe_standard_deviation: Float[Array, "P"] = jnp.maximum(
        standard_deviation,
        floor,
    )
    return safe_standard_deviation


@jaxtyped(typechecker=beartype)
def recipe_deviation(  # noqa: PLR0913
    problem: ReconProblem,
    intended_params: Any,
    initial_latent: Any,
    parameter_uncertainty: Optional[Float[Array, "..."]] = None,
    warning_z: scalar_float = 2.0,
    critical_z: scalar_float = 3.0,
    sigma_floor: scalar_float = 1e-8,
    max_steps: int = 256,
    mode: str = "least_squares",
) -> RecipeDeviationReport:
    """Solve an inverse problem and report signed recipe deviations.

    :see: :class:`~.test_deviation.TestRecipeDeviation`

    Parameters
    ----------
    problem : ReconProblem
        Reconstruction problem to solve.
    intended_params : Any
        Intended physical recipe parameters. Its pytree structure must match
        the fitted physical parameter structure.
    initial_latent : Any
        Initial unconstrained solver coordinates.
    parameter_uncertainty : Optional[Float[Array, "..."]], optional
        Per-parameter one-sigma values or a covariance matrix. If omitted,
        unit standard deviations are used. Default: None
    warning_z : scalar_float, optional
        Absolute z-score threshold for warning severity. Default: 2.0
    critical_z : scalar_float, optional
        Absolute z-score threshold for critical severity. Default: 3.0
    sigma_floor : scalar_float, optional
        Minimum uncertainty denominator. Default: 1e-8
    max_steps : int, optional
        Maximum solver steps (**static**). Default: 256
    mode : str, optional
        Solver family (**static**) passed to :func:`solve`.
        Default: ``"least_squares"``.

    Returns
    -------
    report : RecipeDeviationReport
        Reconstruction result, signed parameter gaps, z-scores, and severity.

    Notes
    -----
    1. Run the common reconstruction solver.
    2. Flatten fitted and intended parameters in the intended recipe basis.
    3. Normalize their signed gap by supplied uncertainty.
    4. Promote the largest absolute z-score into a compact severity code.
    """
    result: ReconResult = solve(
        problem=problem,
        initial_latent=initial_latent,
        mode=mode,
        max_steps=max_steps,
    )
    fitted_flat: Float[Array, "P"]
    fitted_flat, _ = ravel_pytree(result.params)
    intended_flat: Float[Array, "P"]
    intended_unravel: Any
    intended_flat, intended_unravel = ravel_pytree(intended_params)
    if fitted_flat.shape != intended_flat.shape:
        raise ValueError("intended_params must match fitted parameter size")

    deviation_flat: Float[Array, "P"] = fitted_flat - intended_flat
    standard_deviation: Float[Array, "P"] = _parameter_standard_deviation(
        parameter_uncertainty=parameter_uncertainty,
        n_params=int(fitted_flat.shape[0]),
        sigma_floor=sigma_floor,
    )
    z_score_flat: Float[Array, "P"] = deviation_flat / standard_deviation
    max_abs_z: Float[Array, ""] = jnp.max(jnp.abs(z_score_flat))
    warning_code: Int[Array, ""] = jnp.asarray(1, dtype=jnp.int32)
    critical_code: Int[Array, ""] = jnp.asarray(2, dtype=jnp.int32)
    matched_code: Int[Array, ""] = jnp.asarray(0, dtype=jnp.int32)
    severity: Int[Array, ""] = jnp.where(
        max_abs_z >= critical_z,
        critical_code,
        jnp.where(max_abs_z >= warning_z, warning_code, matched_code),
    )
    deviation: Any = intended_unravel(deviation_flat)
    z_score: Any = intended_unravel(z_score_flat)
    report: RecipeDeviationReport = RecipeDeviationReport(
        result=result,
        intended_params=intended_params,
        deviation=deviation,
        z_score=z_score,
        max_abs_z=max_abs_z,
        severity=severity,
    )
    return report


__all__: list[str] = [
    "RecipeDeviationReport",
    "recipe_deviation",
]
