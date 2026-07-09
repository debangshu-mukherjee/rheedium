r"""Recipe-deviation reporting for reconstruction outputs.

Extended Summary
----------------
This module provides the automaton-facing bridge from inversion to control:
solve a reconstruction problem, compare the fitted physical parameters with an
intended recipe, normalize the gaps by supplied uncertainty, and expose a small
severity signal with a schema-validated automaton payload.

Routine Listings
----------------
:func:`recipe_deviation`
    Solve an inverse problem and report signed recipe deviations.
:func:`recipe_deviation_report_payload`
    Convert a recipe-deviation report into the frozen JSON payload.
:func:`recipe_deviation_report_schema`
    Load the committed recipe-deviation report schema.
:func:`validate_recipe_deviation_report`
    Validate a recipe-deviation report payload against the committed schema.
"""

import json
from importlib import resources

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.tools import safe_sqrt
from rheedium.types import (
    RECIPE_DEVIATION_SCHEMA_VERSION,
    LaplaceUncertainty,
    RecipeDeviationReport,
    ReconProblem,
    ReconResult,
    scalar_float,
)

from .solve import solve
from .uncertainty import laplace_uncertainty


def _path_label(path: tuple[Any, ...]) -> str:
    """Return a stable dotted label for one pytree path."""
    parts: list[str] = []
    entry: Any
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        else:
            parts.append(str(entry))
    label: str = ".".join(parts) if parts else "param"
    return label


def _parameter_labels(params: Any) -> tuple[str, ...]:
    """Return flattened scalar labels for a parameter pytree."""
    leaves_with_path: list[tuple[tuple[Any, ...], Any]] = (
        jax.tree_util.tree_flatten_with_path(params)[0]
    )
    labels: list[str] = []
    for path, leaf in leaves_with_path:
        base_label: str = _path_label(path)
        leaf_array: Array = jnp.asarray(leaf)
        if leaf_array.ndim == 0:
            labels.append(base_label)
            continue
        flat_size: int = int(leaf_array.size)
        for flat_index in range(flat_size):
            labels.append(f"{base_label}[{flat_index}]")
    return tuple(labels)


def _covariance_from_parameter_uncertainty(
    parameter_uncertainty: Float[Array, "..."],
    n_params: int,
    sigma_floor: scalar_float,
) -> Float[Array, "P P"]:
    """Return a covariance matrix from supplied sigma values or covariance."""
    uncertainty: Float[Array, "..."] = jnp.asarray(
        parameter_uncertainty,
        dtype=jnp.float64,
    )
    floor: Float[Array, ""] = jnp.asarray(sigma_floor, dtype=jnp.float64)
    if uncertainty.ndim == 2:
        if uncertainty.shape != (n_params, n_params):
            raise ValueError(
                "parameter_uncertainty covariance has wrong shape"
            )
        covariance: Float[Array, "P P"] = 0.5 * (uncertainty + uncertainty.T)
        diagonal: Float[Array, "P"] = jnp.maximum(
            jnp.diag(covariance),
            floor**2,
        )
        return covariance.at[jnp.diag_indices(n_params)].set(diagonal)
    standard_deviation: Float[Array, "P"] = jnp.ravel(uncertainty)
    if standard_deviation.shape[0] != n_params:
        raise ValueError("parameter_uncertainty has incompatible size")
    safe_standard_deviation: Float[Array, "P"] = jnp.maximum(
        standard_deviation,
        floor,
    )
    return jnp.diag(safe_standard_deviation**2)


def _parameter_standard_deviation(
    covariance: Float[Array, "P P"],
    sigma_floor: scalar_float,
) -> Float[Array, "P"]:
    """Return flattened parameter standard deviations."""
    floor: Float[Array, ""] = jnp.asarray(sigma_floor, dtype=jnp.float64)
    diagonal: Float[Array, "P"] = jnp.diag(covariance)
    standard_deviation: Float[Array, "P"] = safe_sqrt(
        jnp.maximum(diagonal, floor**2)
    )
    return standard_deviation


@jaxtyped(typechecker=beartype)
def recipe_deviation(  # noqa: PLR0913
    problem: ReconProblem,
    intended_params: Any,
    initial_latent: Any,
    parameter_uncertainty: Optional[Float[Array, "..."]] = None,
    noise_variance: scalar_float = 1.0,
    uncertainty_regularization: scalar_float = 1e-6,
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
        Per-parameter one-sigma values or a covariance matrix. If omitted, K4
        Laplace covariance is estimated from residual sensitivities in the
        fitted physical-parameter basis. Default: None
    noise_variance : scalar_float, optional
        Per-residual Gaussian noise variance for default Laplace covariance.
        Default: 1.0
    uncertainty_regularization : scalar_float, optional
        Diagonal Fisher regularization for default Laplace covariance.
        Default: 1e-6
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
    3. Normalize their signed gap by supplied or K4 Laplace uncertainty.
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
    n_params: int = int(fitted_flat.shape[0])
    covariance: Float[Array, "P P"]
    uncertainty_source: str
    if parameter_uncertainty is None:

        def residual_from_physical(physical_params: Any) -> Any:
            simulated: Any = problem.forward(physical_params)
            residual: Any = problem.residual_fn(simulated, problem.measured)
            return residual

        uncertainty: LaplaceUncertainty = laplace_uncertainty(
            residual_fn=residual_from_physical,
            params=result.params,
            noise_variance=noise_variance,
            regularization=uncertainty_regularization,
        )
        covariance = uncertainty.covariance
        uncertainty_source = "laplace"
    else:
        covariance = _covariance_from_parameter_uncertainty(
            parameter_uncertainty=parameter_uncertainty,
            n_params=n_params,
            sigma_floor=sigma_floor,
        )
        uncertainty_source = "supplied"
    standard_deviation: Float[Array, "P"] = _parameter_standard_deviation(
        covariance=covariance,
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
    parameter_standard_deviation: Any = intended_unravel(standard_deviation)
    report: RecipeDeviationReport = RecipeDeviationReport(
        result=result,
        intended_params=intended_params,
        deviation=deviation,
        z_score=z_score,
        parameter_standard_deviation=parameter_standard_deviation,
        parameter_covariance=covariance,
        max_abs_z=max_abs_z,
        severity=severity,
        uncertainty_source=uncertainty_source,
        parameter_labels=_parameter_labels(intended_params),
        warning_z=float(warning_z),
        critical_z=float(critical_z),
    )
    return report


def _flat_python_list(value: Any) -> list[float]:
    """Return a flattened list of Python floats."""
    flat_value: Float[Array, "P"] = jnp.ravel(jnp.asarray(value))
    return [float(item) for item in flat_value]


def _matrix_python_list(value: Any) -> list[list[float]]:
    """Return a nested list of Python floats for a matrix."""
    matrix: Float[Array, "P P"] = jnp.asarray(value, dtype=jnp.float64)
    return [[float(item) for item in row] for row in matrix]


@jaxtyped(typechecker=beartype)
def recipe_deviation_report_payload(
    report: RecipeDeviationReport,
) -> dict[str, Any]:
    """Convert a recipe-deviation report into the frozen JSON payload.

    :see: :class:`~.test_deviation.TestRecipeDeviationSchema`

    Parameters
    ----------
    report : RecipeDeviationReport
        Report produced by :func:`recipe_deviation`.

    Returns
    -------
    payload : dict[str, Any]
        JSON-compatible payload following the committed schema.

    Notes
    -----
    1. Flatten fitted, intended, deviation, z-score, and sigma pytrees.
    2. Pair flattened scalar values with stable parameter labels.
    3. Emit solver diagnostics, including the reconstruction ``converged``
       flag. By default that flag reflects solver success rather than a tiny
       data-scale loss, unless the solve caller explicitly opted into a
       data-units loss threshold.
    """
    fitted_flat: Float[Array, "P"]
    fitted_flat, _ = ravel_pytree(report.result.params)
    intended_flat: Float[Array, "P"]
    intended_flat, _ = ravel_pytree(report.intended_params)
    deviation_flat: Float[Array, "P"]
    deviation_flat, _ = ravel_pytree(report.deviation)
    z_score_flat: Float[Array, "P"]
    z_score_flat, _ = ravel_pytree(report.z_score)
    sigma_flat: Float[Array, "P"]
    sigma_flat, _ = ravel_pytree(report.parameter_standard_deviation)
    labels: tuple[str, ...] = report.parameter_labels
    if len(labels) != int(fitted_flat.shape[0]):
        labels = tuple(
            f"param_{index}" for index in range(fitted_flat.shape[0])
        )

    parameters: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        parameters.append(
            {
                "name": label,
                "intended": float(intended_flat[index]),
                "fitted": float(fitted_flat[index]),
                "deviation": float(deviation_flat[index]),
                "z_score": float(z_score_flat[index]),
                "standard_deviation": float(sigma_flat[index]),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "severity": int(report.severity),
        "max_abs_z": float(report.max_abs_z),
        "thresholds": {
            "warning_z": float(report.warning_z),
            "critical_z": float(report.critical_z),
        },
        "uncertainty": {
            "source": report.uncertainty_source,
            "standard_deviation": _flat_python_list(
                report.parameter_standard_deviation
            ),
            "covariance": _matrix_python_list(report.parameter_covariance),
        },
        "solver": {
            "converged": bool(report.result.converged),
            "iterations": int(report.result.iterations),
            "loss": float(report.result.loss),
            "status": report.result.solver_status,
        },
        "parameters": parameters,
    }
    return payload


@jaxtyped(typechecker=beartype)
def recipe_deviation_report_schema() -> dict[str, Any]:
    """Load the committed recipe-deviation report schema.

    :see: :class:`~.test_deviation.TestRecipeDeviationSchema`

    Returns
    -------
    schema : dict[str, Any]
        Parsed JSON schema bundled with :mod:`rheedium.recon.schemas`.

    Notes
    -----
    1. Resolve the package data resource with :mod:`importlib.resources`.
    2. Parse the committed JSON schema into a Python dictionary.
    """
    schema_resource: Any = resources.files("rheedium.recon.schemas").joinpath(
        "recipe_deviation_report.schema.json"
    )
    with schema_resource.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    return schema


@jaxtyped(typechecker=beartype)
def validate_recipe_deviation_report(payload: dict[str, Any]) -> None:
    """Validate a recipe-deviation report payload against the schema.

    :see: :class:`~.test_deviation.TestRecipeDeviationSchema`

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-compatible recipe-deviation payload to validate.

    Notes
    -----
    1. Load the committed schema and check required top-level keys.
    2. Validate the frozen schema version and severity code.
    3. Check covariance and per-parameter entry structure.
    4. Raise ``ValueError`` when the payload misses required fields, uses an
       unsupported schema version or severity code, has an inconsistent
       covariance shape, or contains incomplete parameter entries.
    """
    schema: dict[str, Any] = recipe_deviation_report_schema()
    required: list[str] = list(schema["required"])
    missing: list[str] = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Recipe-deviation payload missing keys: {missing}")
    if payload["schema_version"] != RECIPE_DEVIATION_SCHEMA_VERSION:
        raise ValueError("Unsupported recipe-deviation schema version")
    if payload["severity"] not in {0, 1, 2}:
        raise ValueError("severity must be 0, 1, or 2")
    parameters: Any = payload["parameters"]
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("parameters must be a non-empty list")
    n_params: int = len(parameters)
    covariance: Any = payload["uncertainty"]["covariance"]
    if len(covariance) != n_params:
        raise ValueError("covariance row count must match parameters")
    for row in covariance:
        if len(row) != n_params:
            raise ValueError("covariance must be square in parameter count")
    parameter_required: set[str] = set(
        schema["properties"]["parameters"]["items"]["required"]
    )
    for parameter in parameters:
        if not parameter_required.issubset(parameter):
            raise ValueError("parameter entry missing required keys")


__all__: list[str] = [
    "recipe_deviation",
    "recipe_deviation_report_payload",
    "recipe_deviation_report_schema",
    "validate_recipe_deviation_report",
]
