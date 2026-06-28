r"""Inverse recovery of orientation distributions from experimental RHEED.

Extended Summary
----------------
This module solves the inverse problem: given experimental RHEED data,
recover a probability distribution over domain orientations. The current
public API fits the discrete variant weights, together with a shared
Gaussian mosaic broadening, on a fixed candidate angle support.

The forward model is:

.. math::

    I_\mathrm{sim} = \int P(\theta)\, I(\theta)\, d\theta
    \approx \sum_i w_i\, I(\theta_i)

The inverse problem minimizes a masked image-matching loss with optional
weight regularization and entropy stabilization.

Routine Listings
----------------
:func:`orientation_loss`
    Compute loss between an observed pattern and a trial distribution.
:func:`fit_orientation_weights`
    Optimize discrete orientation weights on a fixed support set.
:func:`compute_fisher_information`
    Estimate Fisher information for the fitted weight logits.
:func:`estimate_weight_uncertainty`
    Propagate Fisher information to 1σ weight uncertainties.

Notes
-----
The inverse problem is most informative when:
1. Different orientations produce distinguishable patterns.
2. The number of fitted orientation parameters is smaller than the
   number of independent detector pixels.
3. Regularization prevents overfitting to noise.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Final, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    OrientationDistribution,
    OrientationFitResult,
    ReconProblem,
    ReconResult,
    create_orientation_distribution,
    float_jax_image,
    integrate_over_orientation,
    scalar_float,
    scalar_int,
)
from rheedium.types.recon_types import _OrientationWeightParameters

from .solve import solve

_PROBABILITY_EPS: Final[float] = 1e-10
_DEFAULT_MOSAIC_PARAM: Final[float] = -5.0
_FISHER_REGULARIZATION: Final[float] = 1e-6


@jaxtyped(typechecker=beartype)
def _normalize_pattern(
    pattern: float_jax_image,
    mask: Optional[float_jax_image] = None,
) -> float_jax_image:
    """Normalize a detector image to unit sum within an optional mask."""
    mask_array: float_jax_image
    if mask is None:
        mask_array = jnp.ones_like(pattern)
    else:
        mask_array = jnp.asarray(mask, dtype=jnp.float64)
    masked_pattern: float_jax_image = pattern * mask_array
    total_intensity: Float[Array, ""] = (
        jnp.sum(masked_pattern) + _PROBABILITY_EPS
    )
    return masked_pattern / total_intensity


@jaxtyped(typechecker=beartype)
def _normalize_probability_weights(
    weights: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Clip and renormalize weights to a valid probability simplex."""
    clipped_weights: Float[Array, "M"] = jnp.clip(
        jnp.asarray(weights, dtype=jnp.float64),
        0.0,
        None,
    )
    weight_sum: Float[Array, ""] = jnp.sum(clipped_weights)
    n_weights: int = clipped_weights.shape[0]
    uniform_weights: Float[Array, "M"] = (
        jnp.ones(n_weights, dtype=jnp.float64) / n_weights
    )
    return jax.lax.cond(
        weight_sum > 0.0,
        lambda: clipped_weights / weight_sum,
        lambda: uniform_weights,
    )


@jaxtyped(typechecker=beartype)
def _softmax_weights(weight_logits: Float[Array, "M"]) -> Float[Array, "M"]:
    """Convert unconstrained logits to normalized positive weights."""
    shifted_logits: Float[Array, "M"] = weight_logits - jnp.max(weight_logits)
    exp_logits: Float[Array, "M"] = jnp.exp(shifted_logits)
    softmax: Float[Array, "M"] = exp_logits / (
        jnp.sum(exp_logits) + _PROBABILITY_EPS
    )
    return softmax


@jaxtyped(typechecker=beartype)
def _softplus(x: Float[Array, ""]) -> Float[Array, ""]:
    """Softplus transform for non-negative scalar parameters."""
    return jnp.log1p(jnp.exp(x))


@jaxtyped(typechecker=beartype)
def _sanitize_distribution(
    distribution: OrientationDistribution,
) -> OrientationDistribution:
    """Return a numerically safe orientation distribution."""
    return create_orientation_distribution(
        angles_deg=distribution.discrete_angles_deg,
        weights=distribution.discrete_weights,
        mosaic_fwhm_deg=distribution.mosaic_fwhm_deg,
        distribution_id=distribution.distribution_id,
    )


@jaxtyped(typechecker=beartype)
def _distribution_from_parameters(
    candidate_angles_deg: Float[Array, "M"],
    params: _OrientationWeightParameters,
) -> OrientationDistribution:
    """Build a physical distribution from unconstrained optimizer params."""
    return create_orientation_distribution(
        angles_deg=candidate_angles_deg,
        weights=_softmax_weights(params.weight_logits),
        mosaic_fwhm_deg=_softplus(params.mosaic_param),
        distribution_id=None,
    )


@jaxtyped(typechecker=beartype)
def _simulate_distribution_pattern(
    simulate_fn: Callable[[scalar_float], float_jax_image],
    distribution: OrientationDistribution,
    n_mosaic_points: scalar_int,
) -> float_jax_image:
    """Evaluate the forward model for one orientation distribution."""
    return integrate_over_orientation(
        simulate_fn,
        _sanitize_distribution(distribution),
        n_mosaic_points=n_mosaic_points,
    )


@jaxtyped(typechecker=beartype)
def _prepare_pattern_for_loss(
    pattern: float_jax_image,
    mask: Optional[float_jax_image] = None,
    normalize: bool = True,
) -> float_jax_image:
    """Apply masking and optional normalization to a detector pattern."""
    if normalize:
        return _normalize_pattern(pattern, mask)
    if mask is None:
        return jnp.asarray(pattern, dtype=jnp.float64)
    return jnp.asarray(pattern, dtype=jnp.float64) * jnp.asarray(
        mask,
        dtype=jnp.float64,
    )


@jaxtyped(typechecker=beartype)
def orientation_loss(
    distribution: OrientationDistribution,
    simulate_fn: Callable[[scalar_float], float_jax_image],
    observed_pattern: float_jax_image,
    mask: Optional[float_jax_image] = None,
    normalize: bool = True,
    regularization_strength: scalar_float = 0.0,
    entropy_weight: scalar_float = 0.0,
    reference_weights: Optional[Float[Array, "M"]] = None,
    n_mosaic_points: scalar_int = 7,
) -> Float[Array, ""]:
    """Compute loss between observed and simulated patterns.

    Description
    -----------
    Computes a masked mean-squared error between an observed detector
    image and the incoherently averaged forward model implied by an
    :class:`OrientationDistribution`.

    :see: :class:`~.test_orientation.TestOrientationLoss`

    Parameters
    ----------
    distribution : OrientationDistribution
        Trial orientation distribution.
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Forward simulation function mapping ``phi_deg`` to a detector
        pattern.
    observed_pattern : Float[Array, "H W"]
        Experimental or synthetic detector image to match.
    mask : Optional[Float[Array, "H W"]], optional
        Pixel mask for loss computation. Default: all pixels.
    normalize : bool, optional
        If True, normalize both patterns before comparison. Default: True
    regularization_strength : scalar_float, optional
        L2 penalty on deviations from ``reference_weights``. If
        ``reference_weights`` is omitted, a uniform distribution is used.
        Default: 0.0
    entropy_weight : scalar_float, optional
        Weight on the entropy bonus ``-H(w)``. Positive values discourage
        collapse onto a single orientation. Default: 0.0
    reference_weights : Optional[Float[Array, "M"]], optional
        Reference weight vector for the L2 penalty. Default: uniform
        weights on the candidate support.
    n_mosaic_points : scalar_int, optional
        Quadrature points for mosaic broadening. Default: 7

    Returns
    -------
    loss : Float[Array, ""]
        Scalar objective value.
    """
    sanitized_distribution: OrientationDistribution = _sanitize_distribution(
        distribution
    )
    simulated_pattern: float_jax_image = _simulate_distribution_pattern(
        simulate_fn,
        sanitized_distribution,
        n_mosaic_points=n_mosaic_points,
    )

    prepared_observed: float_jax_image = _prepare_pattern_for_loss(
        observed_pattern,
        mask=mask,
        normalize=normalize,
    )
    prepared_simulated: float_jax_image = _prepare_pattern_for_loss(
        simulated_pattern,
        mask=mask,
        normalize=normalize,
    )

    mask_array: float_jax_image
    if mask is None:
        mask_array = jnp.ones_like(observed_pattern)
    else:
        mask_array = jnp.asarray(mask, dtype=jnp.float64)
    n_pixels: Float[Array, ""] = jnp.sum(mask_array) + _PROBABILITY_EPS
    mse: Float[Array, ""] = (
        jnp.sum(jnp.square(prepared_observed - prepared_simulated)) / n_pixels
    )

    weights: Float[Array, "M"] = sanitized_distribution.discrete_weights
    target_weights: Float[Array, "M"]
    if reference_weights is None:
        n_orientations: int = weights.shape[0]
        target_weights = jnp.ones(n_orientations, dtype=jnp.float64) / (
            n_orientations
        )
    else:
        target_weights = _normalize_probability_weights(reference_weights)

    l2_penalty: Float[Array, ""] = jnp.sum(
        jnp.square(weights - target_weights)
    )
    entropy: Float[Array, ""] = -jnp.sum(
        weights * jnp.log(weights + _PROBABILITY_EPS)
    )
    return (
        mse + regularization_strength * l2_penalty - entropy_weight * entropy
    )


@jaxtyped(typechecker=beartype)
def fit_orientation_weights(  # noqa: PLR0913, PLR0915
    observed_pattern: float_jax_image,
    simulate_fn: Callable[[scalar_float], float_jax_image],
    candidate_angles_deg: Float[Array, "M"],
    mask: Optional[float_jax_image] = None,
    initial_weights: Optional[Float[Array, "M"]] = None,
    learning_rate: scalar_float = 0.1,
    n_iterations: scalar_int = 500,
    convergence_tol: scalar_float = 1e-6,
    regularization_strength: scalar_float = 1e-4,
    entropy_weight: scalar_float = 1e-3,
    n_mosaic_points: scalar_int = 7,
    normalize: bool = True,
    verbose: bool = False,
) -> OrientationFitResult:
    """Optimize orientation weights to match an observed pattern.

    Description
    -----------
    Given a fixed candidate angle support, optimize the discrete
    orientation weights together with a shared Gaussian mosaic width.
    The internal optimization is carried out in the shared
    :class:`ReconProblem` / :func:`solve` stack using unconstrained
    ``(weight_logits, mosaic_param)`` coordinates.

    :see: :class:`~.test_orientation.TestOrientationFitting`

    Parameters
    ----------
    observed_pattern : Float[Array, "H W"]
        Experimental RHEED pattern to match.
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Forward simulation function mapping ``phi_deg`` to a detector
        pattern.
    candidate_angles_deg : Float[Array, "M"]
        Candidate orientation angles in degrees.
    mask : Optional[Float[Array, "H W"]], optional
        Pixel mask for loss computation. Default: all pixels.
    initial_weights : Optional[Float[Array, "M"]], optional
        Initial weight guess. Default: uniform weights.
    learning_rate : scalar_float, optional
        AdamW learning rate used by the shared solver. Default: 0.1
    n_iterations : scalar_int, optional
        Maximum optimization steps. Default: 500
    convergence_tol : scalar_float, optional
        Absolute solver-loss tolerance. Default: 1e-6
    regularization_strength : scalar_float, optional
        L2 regularization on weight deviations from uniform.
        Default: 1e-4
    entropy_weight : scalar_float, optional
        Entropy regularization. Default: 1e-3
    n_mosaic_points : scalar_int, optional
        Quadrature points for mosaic broadening. Default: 7
    normalize : bool, optional
        If True, normalize both observed and simulated patterns before
        comparison. Default: True
    verbose : bool, optional
        Print the final fitted weights and loss. Default: False

    Returns
    -------
    result : OrientationFitResult
        Fitting result with recovered distribution and diagnostics.
    """
    candidate_angles: Float[Array, "M"] = jnp.atleast_1d(
        jnp.asarray(candidate_angles_deg, dtype=jnp.float64)
    )
    n_angles: int = candidate_angles.shape[0]
    initial_weight_array: Float[Array, "M"]
    if initial_weights is None:
        initial_weight_array = jnp.ones(n_angles, dtype=jnp.float64) / n_angles
    else:
        initial_weight_array = _normalize_probability_weights(initial_weights)
    initial_params: _OrientationWeightParameters = (
        _OrientationWeightParameters(
            weight_logits=jnp.log(initial_weight_array + _PROBABILITY_EPS),
            mosaic_param=jnp.asarray(
                _DEFAULT_MOSAIC_PARAM,
                dtype=jnp.float64,
            ),
        )
    )

    n_iterations_int: int = int(n_iterations)

    def transform(
        params: _OrientationWeightParameters,
    ) -> OrientationDistribution:
        distribution: OrientationDistribution = _distribution_from_parameters(
            candidate_angles,
            params,
        )
        return distribution

    def forward(
        distribution: OrientationDistribution,
    ) -> Tuple[float_jax_image, OrientationDistribution]:
        simulated_pattern: float_jax_image = _simulate_distribution_pattern(
            simulate_fn,
            distribution,
            n_mosaic_points=n_mosaic_points,
        )
        simulated: Tuple[float_jax_image, OrientationDistribution] = (
            simulated_pattern,
            distribution,
        )
        return simulated

    def objective_from_simulated(
        simulated: Tuple[float_jax_image, OrientationDistribution],
        measured: float_jax_image,
    ) -> Float[Array, ""]:
        simulated_pattern: float_jax_image
        distribution: OrientationDistribution
        simulated_pattern, distribution = simulated
        prepared_observed: float_jax_image = _prepare_pattern_for_loss(
            measured,
            mask=mask,
            normalize=normalize,
        )
        prepared_simulated: float_jax_image = _prepare_pattern_for_loss(
            simulated_pattern,
            mask=mask,
            normalize=normalize,
        )
        mask_array: float_jax_image
        if mask is None:
            mask_array = jnp.ones_like(measured)
        else:
            mask_array = jnp.asarray(mask, dtype=jnp.float64)
        n_pixels: Float[Array, ""] = jnp.sum(mask_array) + _PROBABILITY_EPS
        mse: Float[Array, ""] = (
            jnp.sum(jnp.square(prepared_observed - prepared_simulated))
            / n_pixels
        )
        weights: Float[Array, "M"] = distribution.discrete_weights
        target_weights: Float[Array, "M"] = (
            jnp.ones(weights.shape[0], dtype=jnp.float64) / weights.shape[0]
        )
        l2_penalty: Float[Array, ""] = jnp.sum(
            jnp.square(weights - target_weights)
        )
        entropy: Float[Array, ""] = -jnp.sum(
            weights * jnp.log(weights + _PROBABILITY_EPS)
        )
        loss: Float[Array, ""] = (
            mse
            + regularization_strength * l2_penalty
            - entropy_weight * entropy
        )
        return loss

    def residual_fn(
        simulated: Tuple[float_jax_image, OrientationDistribution],
        measured: float_jax_image,
    ) -> Tuple[float_jax_image, Float[Array, "M"]]:
        simulated_pattern: float_jax_image
        distribution: OrientationDistribution
        simulated_pattern, distribution = simulated
        prepared_observed: float_jax_image = _prepare_pattern_for_loss(
            measured,
            mask=mask,
            normalize=normalize,
        )
        prepared_simulated: float_jax_image = _prepare_pattern_for_loss(
            simulated_pattern,
            mask=mask,
            normalize=normalize,
        )
        weights: Float[Array, "M"] = distribution.discrete_weights
        target_weights: Float[Array, "M"] = (
            jnp.ones(weights.shape[0], dtype=jnp.float64) / weights.shape[0]
        )
        regularization_scale: Float[Array, ""] = jnp.sqrt(
            jnp.maximum(
                jnp.asarray(regularization_strength, dtype=jnp.float64),
                0.0,
            )
        )
        regularization_residual: Float[Array, "M"] = regularization_scale * (
            weights - target_weights
        )
        residual: Tuple[float_jax_image, Float[Array, "M"]] = (
            prepared_simulated - prepared_observed,
            regularization_residual,
        )
        return residual

    problem: ReconProblem = ReconProblem(
        forward=forward,
        measured=jnp.asarray(observed_pattern, dtype=jnp.float64),
        transform=transform,
        residual_fn=residual_fn,
        loss_fn=objective_from_simulated,
    )
    solve_result: ReconResult = solve(
        problem=problem,
        initial_latent=initial_params,
        mode="adamw",
        max_steps=n_iterations_int,
        atol=convergence_tol,
        learning_rate=learning_rate,
    )
    fitted_distribution: OrientationDistribution = solve_result.params
    simulated_output: Any = solve_result.simulated
    simulated_pattern: float_jax_image = simulated_output[0]
    residual_pattern: float_jax_image = (
        jnp.asarray(observed_pattern, dtype=jnp.float64) - simulated_pattern
    )
    completed_steps_int: int = int(solve_result.iterations)
    loss_history: Float[Array, "N_steps"] = jnp.asarray(
        [solve_result.loss],
        dtype=jnp.float64,
    )
    if verbose:
        fitted_weights: list[str] = [
            f"{float(weight):.3f}"
            for weight in fitted_distribution.discrete_weights
        ]
        print(
            "Orientation fit: "
            f"loss={float(solve_result.loss):.6f} "
            f"weights={fitted_weights} "
            f"mosaic_fwhm_deg={float(fitted_distribution.mosaic_fwhm_deg):.4f}"
        )

    result: OrientationFitResult = OrientationFitResult(
        fitted_distribution=fitted_distribution,
        final_loss=solve_result.loss,
        loss_history=loss_history,
        converged=bool(solve_result.converged),
        n_iterations=completed_steps_int,
        residual_pattern=residual_pattern,
    )
    return result


@jaxtyped(typechecker=beartype)
def compute_fisher_information(
    simulate_fn: Callable[[scalar_float], float_jax_image],
    distribution: OrientationDistribution,
    noise_variance: scalar_float = 1.0,
    mask: Optional[float_jax_image] = None,
    normalize: bool = True,
    n_mosaic_points: scalar_int = 7,
) -> Float[Array, "M M"]:
    """Compute Fisher information for the discrete weight logits.

    Description
    -----------
    This function treats the discrete orientation weights as a softmax of
    unconstrained logits, holds the orientation angles and mosaic width
    fixed, and computes the Gaussian-noise Fisher information matrix for
    those logits.

    :see: :class:`~.test_orientation.TestOrientationUncertainty`

    Parameters
    ----------
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Forward simulation function.
    distribution : OrientationDistribution
        Distribution about which the local Fisher information is computed.
    noise_variance : scalar_float, optional
        Assumed per-pixel Gaussian noise variance. Default: 1.0
    mask : Optional[Float[Array, "H W"]], optional
        Optional detector mask. Default: all pixels.
    normalize : bool, optional
        If True, compute Fisher information on normalized detector
        patterns. Default: True
    n_mosaic_points : scalar_int, optional
        Quadrature points for mosaic broadening. Default: 7

    Returns
    -------
    fisher : Float[Array, "M M"]
        Fisher information matrix for the ``M`` discrete weight logits.
    """
    sanitized_distribution: OrientationDistribution = _sanitize_distribution(
        distribution
    )
    initial_logits: Float[Array, "M"] = jnp.log(
        sanitized_distribution.discrete_weights + _PROBABILITY_EPS
    )

    def flattened_pattern_from_logits(
        weight_logits: Float[Array, "M"],
    ) -> Float[Array, "P"]:
        current_distribution: OrientationDistribution = (
            create_orientation_distribution(
                angles_deg=sanitized_distribution.discrete_angles_deg,
                weights=_softmax_weights(weight_logits),
                mosaic_fwhm_deg=sanitized_distribution.mosaic_fwhm_deg,
                distribution_id=sanitized_distribution.distribution_id,
            )
        )
        pattern: float_jax_image = _simulate_distribution_pattern(
            simulate_fn,
            current_distribution,
            n_mosaic_points=n_mosaic_points,
        )
        prepared_pattern: float_jax_image = _prepare_pattern_for_loss(
            pattern,
            mask=mask,
            normalize=normalize,
        )
        return prepared_pattern.reshape(-1)

    jacobian: Float[Array, "P M"] = jax.jacrev(flattened_pattern_from_logits)(
        initial_logits
    )
    safe_noise_variance: Float[Array, ""] = jnp.maximum(
        jnp.asarray(noise_variance, dtype=jnp.float64),
        _PROBABILITY_EPS,
    )
    return (jacobian.T @ jacobian) / safe_noise_variance


@jaxtyped(typechecker=beartype)
def estimate_weight_uncertainty(
    result: OrientationFitResult,
    simulate_fn: Callable[[scalar_float], float_jax_image],
    noise_variance: scalar_float = 1.0,
    mask: Optional[float_jax_image] = None,
    normalize: bool = True,
    n_mosaic_points: scalar_int = 7,
) -> Float[Array, "M"]:
    """Estimate 1σ uncertainties on the fitted discrete weights.

    Description
    -----------
    Fisher information is computed in the unconstrained softmax-logit
    parameterization and then propagated to the physical weights with the
    Jacobian of the softmax map.

    :see: :class:`~.test_orientation.TestOrientationUncertainty`

    Parameters
    ----------
    result : OrientationFitResult
        Fitting result from :func:`fit_orientation_weights`.
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Forward simulation function.
    noise_variance : scalar_float, optional
        Assumed per-pixel Gaussian noise variance. Default: 1.0
    mask : Optional[Float[Array, "H W"]], optional
        Optional detector mask. Default: all pixels.
    normalize : bool, optional
        If True, compute uncertainties on normalized detector patterns.
        Default: True
    n_mosaic_points : scalar_int, optional
        Quadrature points for mosaic broadening. Default: 7

    Returns
    -------
    uncertainties : Float[Array, "M"]
        Approximate 1σ uncertainty for each discrete orientation weight.

    Notes
    -----
    The returned uncertainties are conditional on the fitted orientation
    angles and mosaic broadening being held fixed.
    """
    fitted_distribution: OrientationDistribution = _sanitize_distribution(
        result.fitted_distribution
    )
    fisher_logits: Float[Array, "M M"] = compute_fisher_information(
        simulate_fn,
        fitted_distribution,
        noise_variance=noise_variance,
        mask=mask,
        normalize=normalize,
        n_mosaic_points=n_mosaic_points,
    )
    n_weights: int = fisher_logits.shape[0]
    fisher_regularized: Float[Array, "M M"] = fisher_logits + (
        _FISHER_REGULARIZATION * jnp.eye(n_weights, dtype=fisher_logits.dtype)
    )
    covariance_logits: Float[Array, "M M"] = jnp.linalg.inv(fisher_regularized)

    weights: Float[Array, "M"] = fitted_distribution.discrete_weights
    softmax_jacobian: Float[Array, "M M"] = (
        jnp.diag(weights) - weights[:, None] * weights[None, :]
    )
    covariance_weights: Float[Array, "M M"] = (
        softmax_jacobian @ covariance_logits @ softmax_jacobian.T
    )
    variances: Float[Array, "M"] = jnp.maximum(
        jnp.diag(covariance_weights),
        0.0,
    )
    return jnp.sqrt(variances)


__all__: list[str] = [
    "compute_fisher_information",
    "estimate_weight_uncertainty",
    "fit_orientation_weights",
    "orientation_loss",
]
