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
:class:`OrientationFitResult`
    Container for orientation-fitting outputs and diagnostics.
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
from beartype.typing import Callable, NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.types import (
    OrientationDistribution,
    integrate_over_orientation,
    scalar_float,
    scalar_int,
)

_PROBABILITY_EPS: float = 1e-10
_DEFAULT_MOSAIC_PARAM: float = -5.0
_FISHER_REGULARIZATION: float = 1e-6
_ADAM_BETA1: float = 0.9
_ADAM_BETA2: float = 0.999
_ADAM_EPSILON: float = 1e-8


class _OrientationWeightParameters(NamedTuple):
    """Internal unconstrained parameterization for weight fitting."""

    weight_logits: Float[Array, "M"]
    mosaic_param: Float[Array, ""]


class _OrientationAdamState(NamedTuple):
    """Internal Adam optimizer state for orientation fitting."""

    first_logits: Float[Array, "M"]
    second_logits: Float[Array, "M"]
    first_mosaic: Float[Array, ""]
    second_mosaic: Float[Array, ""]


@register_pytree_node_class
class OrientationFitResult(NamedTuple):
    """Results from orientation distribution fitting.

    Attributes
    ----------
    fitted_distribution : OrientationDistribution
        Recovered orientation distribution.
    final_loss : Float[Array, ""]
        Final scalar loss value.
    loss_history : Float[Array, "N_steps"]
        Loss value after each optimizer step.
    converged : bool
        Whether the optimizer met its stopping tolerance.
    n_iterations : int
        Number of recorded optimization iterations.
    residual_pattern : Float[Array, "H W"]
        Difference ``I_observed - I_fitted`` for diagnostics.
    """

    fitted_distribution: OrientationDistribution
    final_loss: Float[Array, ""]
    loss_history: Float[Array, "N_steps"]
    converged: bool
    n_iterations: int
    residual_pattern: Float[Array, "H W"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            OrientationDistribution,
            Float[Array, ""],
            Float[Array, "N_steps"],
            Float[Array, "H W"],
        ],
        Tuple[bool, int],
    ]:
        """Flatten for JAX PyTree support."""
        return (
            (
                self.fitted_distribution,
                self.final_loss,
                self.loss_history,
                self.residual_pattern,
            ),
            (self.converged, self.n_iterations),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[bool, int],
        children: Tuple[
            OrientationDistribution,
            Float[Array, ""],
            Float[Array, "N_steps"],
            Float[Array, "H W"],
        ],
    ) -> "OrientationFitResult":
        """Unflatten from a JAX PyTree."""
        return cls(
            fitted_distribution=children[0],
            final_loss=children[1],
            loss_history=children[2],
            converged=aux_data[0],
            n_iterations=aux_data[1],
            residual_pattern=children[3],
        )


@jaxtyped(typechecker=beartype)
def _normalize_pattern(
    pattern: Float[Array, "H W"],
    mask: Optional[Float[Array, "H W"]] = None,
) -> Float[Array, "H W"]:
    """Normalize a detector image to unit sum within an optional mask."""
    mask_array: Float[Array, "H W"]
    if mask is None:
        mask_array = jnp.ones_like(pattern)
    else:
        mask_array = jnp.asarray(mask, dtype=jnp.float64)
    masked_pattern: Float[Array, "H W"] = pattern * mask_array
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
    return exp_logits / (jnp.sum(exp_logits) + _PROBABILITY_EPS)


@jaxtyped(typechecker=beartype)
def _softplus(x: Float[Array, ""]) -> Float[Array, ""]:
    """Softplus transform for non-negative scalar parameters."""
    return jnp.log1p(jnp.exp(x))


@jaxtyped(typechecker=beartype)
def _sanitize_distribution(
    distribution: OrientationDistribution,
) -> OrientationDistribution:
    """Return a numerically safe orientation distribution."""
    return OrientationDistribution(
        discrete_angles_deg=jnp.asarray(
            distribution.discrete_angles_deg,
            dtype=jnp.float64,
        ),
        discrete_weights=_normalize_probability_weights(
            distribution.discrete_weights
        ),
        mosaic_fwhm_deg=jnp.maximum(
            jnp.asarray(distribution.mosaic_fwhm_deg, dtype=jnp.float64),
            0.0,
        ),
        distribution_id=distribution.distribution_id,
    )


@jaxtyped(typechecker=beartype)
def _distribution_from_parameters(
    candidate_angles_deg: Float[Array, "M"],
    params: _OrientationWeightParameters,
) -> OrientationDistribution:
    """Build a physical distribution from unconstrained optimizer params."""
    return OrientationDistribution(
        discrete_angles_deg=jnp.asarray(
            candidate_angles_deg,
            dtype=jnp.float64,
        ),
        discrete_weights=_softmax_weights(params.weight_logits),
        mosaic_fwhm_deg=_softplus(params.mosaic_param),
        distribution_id=None,
    )


@jaxtyped(typechecker=beartype)
def _simulate_distribution_pattern(
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    distribution: OrientationDistribution,
    n_mosaic_points: scalar_int,
) -> Float[Array, "H W"]:
    """Evaluate the forward model for one orientation distribution."""
    return integrate_over_orientation(
        simulate_fn,
        _sanitize_distribution(distribution),
        n_mosaic_points=n_mosaic_points,
    )


@jaxtyped(typechecker=beartype)
def _parameter_norm(params: _OrientationWeightParameters) -> Float[Array, ""]:
    """Compute the Euclidean norm of the fit parameter pytree."""
    logit_norm_sq: Float[Array, ""] = jnp.real(
        jnp.vdot(params.weight_logits, params.weight_logits)
    )
    mosaic_norm_sq: Float[Array, ""] = jnp.square(params.mosaic_param)
    return jnp.sqrt(logit_norm_sq + mosaic_norm_sq)


@jaxtyped(typechecker=beartype)
def _adam_update(
    params: _OrientationWeightParameters,
    gradients: _OrientationWeightParameters,
    optimizer_state: _OrientationAdamState,
    learning_rate: scalar_float,
    iteration: scalar_int,
) -> Tuple[
    _OrientationWeightParameters,
    _OrientationAdamState,
    Float[Array, ""],
]:
    """Apply one Adam step to the orientation fit parameters."""
    first_logits: Float[Array, "M"] = (
        _ADAM_BETA1 * optimizer_state.first_logits
        + (1.0 - _ADAM_BETA1) * gradients.weight_logits
    )
    second_logits: Float[Array, "M"] = (
        _ADAM_BETA2 * optimizer_state.second_logits
        + (1.0 - _ADAM_BETA2) * gradients.weight_logits**2
    )
    first_mosaic: Float[Array, ""] = (
        _ADAM_BETA1 * optimizer_state.first_mosaic
        + (1.0 - _ADAM_BETA1) * gradients.mosaic_param
    )
    second_mosaic: Float[Array, ""] = (
        _ADAM_BETA2 * optimizer_state.second_mosaic
        + (1.0 - _ADAM_BETA2) * gradients.mosaic_param**2
    )

    first_bias_correction: Float[Array, ""] = 1.0 - _ADAM_BETA1**iteration
    second_bias_correction: Float[Array, ""] = 1.0 - _ADAM_BETA2**iteration
    first_logits_hat: Float[Array, "M"] = first_logits / first_bias_correction
    second_logits_hat: Float[Array, "M"] = (
        second_logits / second_bias_correction
    )
    first_mosaic_hat: Float[Array, ""] = first_mosaic / first_bias_correction
    second_mosaic_hat: Float[Array, ""] = (
        second_mosaic / second_bias_correction
    )

    logit_step: Float[Array, "M"] = (
        -learning_rate
        * first_logits_hat
        / (jnp.sqrt(second_logits_hat) + _ADAM_EPSILON)
    )
    mosaic_step: Float[Array, ""] = (
        -learning_rate
        * first_mosaic_hat
        / (jnp.sqrt(second_mosaic_hat) + _ADAM_EPSILON)
    )
    updated_params = _OrientationWeightParameters(
        weight_logits=params.weight_logits + logit_step,
        mosaic_param=params.mosaic_param + mosaic_step,
    )
    updated_state = _OrientationAdamState(
        first_logits=first_logits,
        second_logits=second_logits,
        first_mosaic=first_mosaic,
        second_mosaic=second_mosaic,
    )
    step_norm: Float[Array, ""] = _parameter_norm(
        _OrientationWeightParameters(
            weight_logits=logit_step,
            mosaic_param=mosaic_step,
        )
    )
    return updated_params, updated_state, step_norm


@jaxtyped(typechecker=beartype)
def _prepare_pattern_for_loss(
    pattern: Float[Array, "H W"],
    mask: Optional[Float[Array, "H W"]] = None,
    normalize: bool = True,
) -> Float[Array, "H W"]:
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
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    observed_pattern: Float[Array, "H W"],
    mask: Optional[Float[Array, "H W"]] = None,
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
    simulated_pattern: Float[Array, "H W"] = _simulate_distribution_pattern(
        simulate_fn,
        sanitized_distribution,
        n_mosaic_points=n_mosaic_points,
    )

    prepared_observed: Float[Array, "H W"] = _prepare_pattern_for_loss(
        observed_pattern,
        mask=mask,
        normalize=normalize,
    )
    prepared_simulated: Float[Array, "H W"] = _prepare_pattern_for_loss(
        simulated_pattern,
        mask=mask,
        normalize=normalize,
    )

    mask_array: Float[Array, "H W"]
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
    observed_pattern: Float[Array, "H W"],
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    candidate_angles_deg: Float[Array, "M"],
    mask: Optional[Float[Array, "H W"]] = None,
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
    The internal optimization is carried out in an unconstrained space
    with a specialized Adam update on ``(weight_logits, mosaic_param)``
    and a JIT-compiled :func:`jax.lax.scan` loop.

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
        Adam learning rate. Default: 0.1
    n_iterations : scalar_int, optional
        Maximum optimization steps. Default: 500
    convergence_tol : scalar_float, optional
        Stop when the optimizer update norm falls below this tolerance.
        Default: 1e-6
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
    initial_params = _OrientationWeightParameters(
        weight_logits=jnp.log(initial_weight_array + _PROBABILITY_EPS),
        mosaic_param=jnp.asarray(_DEFAULT_MOSAIC_PARAM, dtype=jnp.float64),
    )

    def objective_fn(
        params: _OrientationWeightParameters,
    ) -> Float[Array, ""]:
        return orientation_loss(
            distribution=_distribution_from_parameters(
                candidate_angles,
                params,
            ),
            simulate_fn=simulate_fn,
            observed_pattern=observed_pattern,
            mask=mask,
            normalize=normalize,
            regularization_strength=regularization_strength,
            entropy_weight=entropy_weight,
            n_mosaic_points=n_mosaic_points,
        )

    n_iterations_int: int = int(n_iterations)

    def _run_optimizer(
        start_params: _OrientationWeightParameters,
    ) -> Tuple[
        _OrientationWeightParameters,
        Float[Array, "N_steps_full"],
        scalar_int,
        Float[Array, ""],
        Bool[Array, ""],
    ]:
        objective_and_grad_fn = jax.value_and_grad(objective_fn)
        initial_optimizer_state = _OrientationAdamState(
            first_logits=jnp.zeros_like(start_params.weight_logits),
            second_logits=jnp.zeros_like(start_params.weight_logits),
            first_mosaic=jnp.zeros_like(start_params.mosaic_param),
            second_mosaic=jnp.zeros_like(start_params.mosaic_param),
        )
        initial_loss: Float[Array, ""] = objective_fn(start_params)

        def _step(
            carry: Tuple[
                _OrientationWeightParameters,
                _OrientationAdamState,
                Bool[Array, ""],
                scalar_int,
                Float[Array, ""],
            ],
            step_index: Int[Array, ""],
        ) -> Tuple[
            Tuple[
                _OrientationWeightParameters,
                _OrientationAdamState,
                Bool[Array, ""],
                scalar_int,
                Float[Array, ""],
            ],
            Float[Array, ""],
        ]:
            params, optimizer_state, converged_flag, steps_taken, last_loss = (
                carry
            )

            def _frozen_step(
                frozen_carry: Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Bool[Array, ""],
                    scalar_int,
                    Float[Array, ""],
                ],
            ) -> Tuple[
                Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Bool[Array, ""],
                    scalar_int,
                    Float[Array, ""],
                ],
                Float[Array, ""],
            ]:
                return frozen_carry, frozen_carry[-1]

            def _active_step(
                active_carry: Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Bool[Array, ""],
                    scalar_int,
                    Float[Array, ""],
                ],
            ) -> Tuple[
                Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Bool[Array, ""],
                    scalar_int,
                    Float[Array, ""],
                ],
                Float[Array, ""],
            ]:
                active_params, active_optimizer_state, _, active_steps, _ = (
                    active_carry
                )
                objective_value: Float[Array, ""]
                gradients: _OrientationWeightParameters
                objective_value, gradients = objective_and_grad_fn(
                    active_params
                )
                gradient_norm: Float[Array, ""] = _parameter_norm(gradients)

                def _converged_on_gradient(
                    operand: Tuple[
                        _OrientationWeightParameters,
                        _OrientationAdamState,
                        Float[Array, ""],
                    ],
                ) -> Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Float[Array, ""],
                    Bool[Array, ""],
                ]:
                    grad_params, grad_optimizer_state, grad_loss = operand
                    return (
                        grad_params,
                        grad_optimizer_state,
                        grad_loss,
                        jnp.asarray(True),
                    )

                def _take_update_step(
                    operand: Tuple[
                        _OrientationWeightParameters,
                        _OrientationAdamState,
                        Float[Array, ""],
                    ],
                ) -> Tuple[
                    _OrientationWeightParameters,
                    _OrientationAdamState,
                    Float[Array, ""],
                    Bool[Array, ""],
                ]:
                    update_params, update_optimizer_state, _ = operand
                    next_params: _OrientationWeightParameters
                    next_optimizer_state: _OrientationAdamState
                    step_norm: Float[Array, ""]
                    next_params, next_optimizer_state, step_norm = (
                        _adam_update(
                            params=update_params,
                            gradients=gradients,
                            optimizer_state=update_optimizer_state,
                            learning_rate=learning_rate,
                            iteration=step_index + 1,
                        )
                    )
                    next_loss: Float[Array, ""] = objective_fn(next_params)
                    next_converged: Bool[Array, ""] = (
                        step_norm <= convergence_tol
                    )
                    return (
                        next_params,
                        next_optimizer_state,
                        next_loss,
                        next_converged,
                    )

                next_params: _OrientationWeightParameters
                next_optimizer_state: _OrientationAdamState
                recorded_loss: Float[Array, ""]
                next_converged_flag: Bool[Array, ""]
                (
                    next_params,
                    next_optimizer_state,
                    recorded_loss,
                    (next_converged_flag),
                ) = jax.lax.cond(
                    gradient_norm <= convergence_tol,
                    _converged_on_gradient,
                    _take_update_step,
                    (active_params, active_optimizer_state, objective_value),
                )
                next_steps: scalar_int = active_steps + jnp.asarray(
                    1, dtype=jnp.int32
                )
                next_carry = (
                    next_params,
                    next_optimizer_state,
                    next_converged_flag,
                    next_steps,
                    recorded_loss,
                )
                return next_carry, recorded_loss

            return jax.lax.cond(
                converged_flag,
                _frozen_step,
                _active_step,
                carry,
            )

        final_carry, loss_history_full = jax.lax.scan(
            _step,
            (
                start_params,
                initial_optimizer_state,
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                initial_loss,
            ),
            jnp.arange(n_iterations_int, dtype=jnp.int32),
        )
        final_params, _, converged_flag, steps_taken, final_loss = final_carry
        return (
            final_params,
            loss_history_full,
            steps_taken,
            final_loss,
            converged_flag,
        )

    (
        fitted_params,
        full_loss_history,
        completed_steps,
        final_loss,
        converged_flag,
    ) = jax.jit(_run_optimizer)(initial_params)
    completed_steps_int: int = int(completed_steps)
    loss_history: Float[Array, "N_steps"] = full_loss_history[
        :completed_steps_int
    ]
    fitted_distribution: OrientationDistribution = (
        _distribution_from_parameters(
            candidate_angles,
            fitted_params,
        )
    )
    simulated_pattern: Float[Array, "H W"] = _simulate_distribution_pattern(
        simulate_fn,
        fitted_distribution,
        n_mosaic_points=n_mosaic_points,
    )
    residual_pattern: Float[Array, "H W"] = (
        jnp.asarray(observed_pattern, dtype=jnp.float64) - simulated_pattern
    )
    if completed_steps_int == 0:
        final_loss = objective_fn(initial_params)

    if verbose:
        fitted_weights = [
            f"{float(weight):.3f}"
            for weight in fitted_distribution.discrete_weights
        ]
        print(
            "Orientation fit:",
            f"loss={float(final_loss):.6f}",
            f"weights={fitted_weights}",
            f"mosaic_fwhm_deg={float(fitted_distribution.mosaic_fwhm_deg):.4f}",
        )

    return OrientationFitResult(
        fitted_distribution=fitted_distribution,
        final_loss=final_loss,
        loss_history=loss_history,
        converged=bool(converged_flag),
        n_iterations=completed_steps_int,
        residual_pattern=residual_pattern,
    )


@jaxtyped(typechecker=beartype)
def compute_fisher_information(
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    distribution: OrientationDistribution,
    noise_variance: scalar_float = 1.0,
    mask: Optional[Float[Array, "H W"]] = None,
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
        current_distribution = OrientationDistribution(
            discrete_angles_deg=sanitized_distribution.discrete_angles_deg,
            discrete_weights=_softmax_weights(weight_logits),
            mosaic_fwhm_deg=sanitized_distribution.mosaic_fwhm_deg,
            distribution_id=sanitized_distribution.distribution_id,
        )
        pattern: Float[Array, "H W"] = _simulate_distribution_pattern(
            simulate_fn,
            current_distribution,
            n_mosaic_points=n_mosaic_points,
        )
        prepared_pattern: Float[Array, "H W"] = _prepare_pattern_for_loss(
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
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    noise_variance: scalar_float = 1.0,
    mask: Optional[Float[Array, "H W"]] = None,
    normalize: bool = True,
    n_mosaic_points: scalar_int = 7,
) -> Float[Array, "M"]:
    """Estimate 1σ uncertainties on the fitted discrete weights.

    Description
    -----------
    Fisher information is computed in the unconstrained softmax-logit
    parameterization and then propagated to the physical weights with the
    Jacobian of the softmax map.

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
    "OrientationFitResult",
    "compute_fisher_information",
    "estimate_weight_uncertainty",
    "fit_orientation_weights",
    "orientation_loss",
]
