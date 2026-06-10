r"""Optimization routines for inverse RHEED problems.

Extended Summary
----------------
This module provides general-purpose optimization routines for
reconstruction problems built on differentiable forward models. The
low-level solvers operate on arbitrary JAX pytrees, while the
high-level wrappers target image-matching workflows that compare a
simulated detector image against an experimental one.

Routine Listings
----------------
:class:`ReconstructionResult`
    Result container returned by all reconstruction solvers.
:func:`gauss_newton_least_squares`
    Gauss-Newton optimizer for least-squares residual functions.
:func:`adam_optimize`
    Adam optimizer for arbitrary scalar objectives.
:func:`adagrad_optimize`
    Adagrad optimizer for arbitrary scalar objectives.
:func:`gauss_newton_reconstruction`
    Image-matching reconstruction using Gauss-Newton.
:func:`adam_reconstruction`
    Image-matching reconstruction using Adam.
:func:`adagrad_reconstruction`
    Image-matching reconstruction using Adagrad.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, NamedTuple, Optional, Tuple
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.types import scalar_float

from .losses import (
    weighted_image_residual,
    weighted_mean_squared_error,
)


@register_pytree_node_class
class ReconstructionResult(NamedTuple):
    """Container for reconstruction outputs and optimization traces.

    Attributes
    ----------
    params : Any
        Final reconstructed parameter pytree.
    objective_history : Float[Array, "N"]
        Objective value after each accepted optimization step.
    gradient_norm_history : Float[Array, "N"]
        L2 norm of the gradient-like search direction at each iteration.
    step_norm_history : Float[Array, "N"]
        L2 norm of the parameter update applied at each iteration.
    iterations : Int[Array, ""]
        Number of recorded optimization iterations.
    converged : Bool[Array, ""]
        True when a convergence tolerance was met before exhausting the
        iteration budget.
    """

    params: Any
    objective_history: Float[Array, "N"]
    gradient_norm_history: Float[Array, "N"]
    step_norm_history: Float[Array, "N"]
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Any,
            Float[Array, "N"],
            Float[Array, "N"],
            Float[Array, "N"],
            Int[Array, ""],
            Bool[Array, ""],
        ],
        None,
    ]:
        """Flatten the PyTree into a tuple of leaves."""
        return (
            (
                self.params,
                self.objective_history,
                self.gradient_norm_history,
                self.step_norm_history,
                self.iterations,
                self.converged,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[
            Any,
            Float[Array, "N"],
            Float[Array, "N"],
            Float[Array, "N"],
            Int[Array, ""],
            Bool[Array, ""],
        ],
    ) -> "ReconstructionResult":
        """Unflatten the PyTree into a result instance."""
        del aux_data
        return cls(*children)


def _tree_l2_norm(tree: Any) -> scalar_float:
    """Compute the Euclidean norm of all leaves in a pytree."""
    leaves: list[Any] = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    squared_norm: scalar_float = jnp.asarray(0.0)
    for leaf in leaves:
        leaf_array: Array = jnp.asarray(leaf)
        squared_norm = squared_norm + jnp.real(
            jnp.vdot(leaf_array, leaf_array)
        )
    return jnp.sqrt(squared_norm)


def _result_from_history(
    params: Any,
    objective_history: list[scalar_float],
    gradient_norm_history: list[scalar_float],
    step_norm_history: list[scalar_float],
    converged: bool,
) -> ReconstructionResult:
    """Assemble a result object from Python-side optimization traces."""
    return ReconstructionResult(
        params=params,
        objective_history=jnp.asarray(objective_history),
        gradient_norm_history=jnp.asarray(gradient_norm_history),
        step_norm_history=jnp.asarray(step_norm_history),
        iterations=jnp.asarray(len(objective_history), dtype=jnp.int32),
        converged=jnp.asarray(converged),
    )


def _apply_postprocess(
    simulated_image: Float[Array, "H W"],
    postprocess_fn: Optional[
        Callable[[Float[Array, "H W"]], Float[Array, "H W"]]
    ],
) -> Float[Array, "H W"]:
    """Apply an optional post-processing transform to a simulated image."""
    if postprocess_fn is None:
        return simulated_image
    return postprocess_fn(simulated_image)


@jaxtyped(typechecker=beartype)
def gauss_newton_least_squares(
    initial_params: Any,
    residual_fn: Callable[[Any], Array],
    damping: scalar_float = 1e-3,
    step_scale: scalar_float = 1.0,
    max_iterations: int = 25,
    tolerance: scalar_float = 1e-6,
) -> ReconstructionResult:
    r"""Minimize a least-squares objective with Gauss-Newton iterations.

    Extended Summary
    ----------------
    This solver targets objectives of the form
    :math:`\min_\theta \|r(\theta)\|_2^2`, where ``residual_fn``
    returns the residual vector or tensor. Parameters may be any JAX
    pytree; internally they are flattened with
    :func:`jax.flatten_util.ravel_pytree`.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree.
    residual_fn : Callable[[Any], Array]
        Residual function returning any array-shaped residual.
    damping : scalar_float, optional
        Levenberg-style diagonal damping added to the normal matrix.
        Default: 1e-3
    step_scale : scalar_float, optional
        Scalar multiplier applied to the Gauss-Newton step.
        Default: 1.0
    max_iterations : int, optional
        Maximum number of iterations. Default: 25
    tolerance : scalar_float, optional
        Convergence threshold applied to the gradient norm and update
        norm. Default: 1e-6

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """
    flat_params: Float[Array, " P"]
    unravel_fn: Callable[[Float[Array, "P"]], Any]
    flat_params, unravel_fn = ravel_pytree(initial_params)

    def flat_residual_fn(
        flat_parameter_vector: Float[Array, "P"],
    ) -> Float[Array, " R"]:
        residual: Array = residual_fn(unravel_fn(flat_parameter_vector))
        return jnp.ravel(jnp.asarray(residual))

    objective_history: list[scalar_float] = []
    gradient_norm_history: list[scalar_float] = []
    step_norm_history: list[scalar_float] = []
    converged: bool = False

    for _ in range(max_iterations):
        residual_vector: Float[Array, " R"] = flat_residual_fn(flat_params)
        objective_value: scalar_float = jnp.mean(residual_vector**2)
        jacobian: Float[Array, "R P"] = jax.jacrev(flat_residual_fn)(
            flat_params
        )
        gradient_vector: Float[Array, " P"] = jacobian.T @ residual_vector
        gradient_norm: scalar_float = jnp.linalg.norm(gradient_vector)

        if bool(gradient_norm <= tolerance):
            objective_history.append(objective_value)
            gradient_norm_history.append(gradient_norm)
            step_norm_history.append(jnp.asarray(0.0))
            converged = True
            break

        normal_matrix: Float[Array, "P P"] = jacobian.T @ jacobian
        identity: Float[Array, "P P"] = jnp.eye(
            normal_matrix.shape[0], dtype=normal_matrix.dtype
        )
        step: Float[Array, " P"] = -step_scale * jnp.linalg.solve(
            normal_matrix + damping * identity,
            gradient_vector,
        )
        step_norm: scalar_float = jnp.linalg.norm(step)
        flat_params = flat_params + step

        updated_residual: Float[Array, " R"] = flat_residual_fn(flat_params)
        updated_objective: scalar_float = jnp.mean(updated_residual**2)
        objective_history.append(updated_objective)
        gradient_norm_history.append(gradient_norm)
        step_norm_history.append(step_norm)

        if bool(step_norm <= tolerance):
            converged = True
            break

    return _result_from_history(
        params=unravel_fn(flat_params),
        objective_history=objective_history,
        gradient_norm_history=gradient_norm_history,
        step_norm_history=step_norm_history,
        converged=converged,
    )


@jaxtyped(typechecker=beartype)
def adam_optimize(
    initial_params: Any,
    objective_fn: Callable[[Any], scalar_float],
    learning_rate: scalar_float = 1e-2,
    beta1: scalar_float = 0.9,
    beta2: scalar_float = 0.999,
    epsilon: scalar_float = 1e-8,
    max_iterations: int = 250,
    tolerance: scalar_float = 1e-6,
) -> ReconstructionResult:
    r"""Minimize a scalar objective with the Adam optimizer.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree.
    objective_fn : Callable[[Any], scalar_float]
        Scalar objective to minimize.
    learning_rate : scalar_float, optional
        Adam learning rate. Default: 1e-2
    beta1 : scalar_float, optional
        Exponential decay factor for the first moment. Default: 0.9
    beta2 : scalar_float, optional
        Exponential decay factor for the second moment. Default: 0.999
    epsilon : scalar_float, optional
        Denominator stabilizer. Default: 1e-8
    max_iterations : int, optional
        Maximum number of iterations. Default: 250
    tolerance : scalar_float, optional
        Convergence threshold applied to the gradient norm and update
        norm. Default: 1e-6

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """
    params: Any = initial_params
    first_moment: Any = jax.tree_util.tree_map(jnp.zeros_like, params)
    second_moment: Any = jax.tree_util.tree_map(jnp.zeros_like, params)
    objective_history: list[scalar_float] = []
    gradient_norm_history: list[scalar_float] = []
    step_norm_history: list[scalar_float] = []
    converged: bool = False

    for iteration in range(1, max_iterations + 1):
        objective_value: scalar_float
        gradients: Any
        objective_value, gradients = jax.value_and_grad(objective_fn)(params)
        gradient_norm: scalar_float = _tree_l2_norm(gradients)

        if bool(gradient_norm <= tolerance):
            objective_history.append(objective_value)
            gradient_norm_history.append(gradient_norm)
            step_norm_history.append(jnp.asarray(0.0))
            converged = True
            break

        first_moment = jax.tree_util.tree_map(
            lambda moment, grad: beta1 * moment + (1.0 - beta1) * grad,
            first_moment,
            gradients,
        )
        second_moment = jax.tree_util.tree_map(
            lambda moment, grad: beta2 * moment + (1.0 - beta2) * grad**2,
            second_moment,
            gradients,
        )
        first_bias_correction: scalar_float = 1.0 - beta1**iteration
        second_bias_correction: scalar_float = 1.0 - beta2**iteration

        first_moment_hat: Any = jax.tree_util.tree_map(
            lambda moment, correction=first_bias_correction: (
                moment / correction
            ),
            first_moment,
        )
        second_moment_hat: Any = jax.tree_util.tree_map(
            lambda moment, correction=second_bias_correction: (
                moment / correction
            ),
            second_moment,
        )

        step: Any = jax.tree_util.tree_map(
            lambda moment, variance: (
                -learning_rate * moment / (jnp.sqrt(variance) + epsilon)
            ),
            first_moment_hat,
            second_moment_hat,
        )
        step_norm: scalar_float = _tree_l2_norm(step)
        params = jax.tree_util.tree_map(
            lambda param, update: param + update,
            params,
            step,
        )
        updated_objective: scalar_float = objective_fn(params)
        objective_history.append(updated_objective)
        gradient_norm_history.append(gradient_norm)
        step_norm_history.append(step_norm)

        if bool(step_norm <= tolerance):
            converged = True
            break

    return _result_from_history(
        params=params,
        objective_history=objective_history,
        gradient_norm_history=gradient_norm_history,
        step_norm_history=step_norm_history,
        converged=converged,
    )


@jaxtyped(typechecker=beartype)
def adagrad_optimize(
    initial_params: Any,
    objective_fn: Callable[[Any], scalar_float],
    learning_rate: scalar_float = 1e-1,
    epsilon: scalar_float = 1e-8,
    max_iterations: int = 500,
    tolerance: scalar_float = 1e-6,
) -> ReconstructionResult:
    r"""Minimize a scalar objective with the Adagrad optimizer.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree.
    objective_fn : Callable[[Any], scalar_float]
        Scalar objective to minimize.
    learning_rate : scalar_float, optional
        Adagrad base learning rate. Default: 1e-1
    epsilon : scalar_float, optional
        Denominator stabilizer. Default: 1e-8
    max_iterations : int, optional
        Maximum number of iterations. Default: 500
    tolerance : scalar_float, optional
        Convergence threshold applied to the gradient norm and update
        norm. Default: 1e-6

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """
    params: Any = initial_params
    accumulator: Any = jax.tree_util.tree_map(jnp.zeros_like, params)
    objective_history: list[scalar_float] = []
    gradient_norm_history: list[scalar_float] = []
    step_norm_history: list[scalar_float] = []
    converged: bool = False

    for _ in range(max_iterations):
        objective_value: scalar_float
        gradients: Any
        objective_value, gradients = jax.value_and_grad(objective_fn)(params)
        gradient_norm: scalar_float = _tree_l2_norm(gradients)

        if bool(gradient_norm <= tolerance):
            objective_history.append(objective_value)
            gradient_norm_history.append(gradient_norm)
            step_norm_history.append(jnp.asarray(0.0))
            converged = True
            break

        accumulator = jax.tree_util.tree_map(
            lambda state, grad: state + grad**2,
            accumulator,
            gradients,
        )
        step: Any = jax.tree_util.tree_map(
            lambda grad, state: (
                -learning_rate * grad / (jnp.sqrt(state) + epsilon)
            ),
            gradients,
            accumulator,
        )
        step_norm: scalar_float = _tree_l2_norm(step)
        params = jax.tree_util.tree_map(
            lambda param, update: param + update,
            params,
            step,
        )
        updated_objective: scalar_float = objective_fn(params)
        objective_history.append(updated_objective)
        gradient_norm_history.append(gradient_norm)
        step_norm_history.append(step_norm)

        if bool(step_norm <= tolerance):
            converged = True
            break

    return _result_from_history(
        params=params,
        objective_history=objective_history,
        gradient_norm_history=gradient_norm_history,
        step_norm_history=step_norm_history,
        converged=converged,
    )


@jaxtyped(typechecker=beartype)
def gauss_newton_reconstruction(  # noqa: PLR0913
    initial_params: Any,
    forward_model: Callable[[Any], Float[Array, "H W"]],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    postprocess_fn: Optional[
        Callable[[Float[Array, "H W"]], Float[Array, "H W"]]
    ] = None,
    damping: scalar_float = 1e-3,
    step_scale: scalar_float = 1.0,
    max_iterations: int = 25,
    tolerance: scalar_float = 1e-6,
) -> ReconstructionResult:
    r"""Reconstruct parameters by least-squares image matching.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree passed to ``forward_model``.
    forward_model : Callable[[Any], Float[Array, "H W"]]
        Differentiable simulator that maps parameters to a detector
        image.
    experimental_image : Float[Array, "H W"]
        Target detector image.
    weight_map : Float[Array, "H W"], optional
        Non-negative per-pixel weights for least-squares fitting.
    postprocess_fn : Callable[[Float[Array, "H W"]], \
            Float[Array, "H W"]], optional
        Optional transform applied to each simulated image before it is
        compared against ``experimental_image``.
    damping : scalar_float, optional
        Diagonal damping for the Gauss-Newton normal equations.
        Default: 1e-3
    step_scale : scalar_float, optional
        Scalar multiplier applied to each update. Default: 1.0
    max_iterations : int, optional
        Maximum number of iterations. Default: 25
    tolerance : scalar_float, optional
        Convergence threshold. Default: 1e-6

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """

    def residual_fn(params: Any) -> Float[Array, "H W"]:
        simulated_image: Float[Array, "H W"] = _apply_postprocess(
            forward_model(params),
            postprocess_fn,
        )
        return weighted_image_residual(
            simulated_image=simulated_image,
            experimental_image=experimental_image,
            weight_map=weight_map,
        )

    return gauss_newton_least_squares(
        initial_params=initial_params,
        residual_fn=residual_fn,
        damping=damping,
        step_scale=step_scale,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


@jaxtyped(typechecker=beartype)
def adam_reconstruction(  # noqa: PLR0913
    initial_params: Any,
    forward_model: Callable[[Any], Float[Array, "H W"]],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    postprocess_fn: Optional[
        Callable[[Float[Array, "H W"]], Float[Array, "H W"]]
    ] = None,
    learning_rate: scalar_float = 1e-2,
    beta1: scalar_float = 0.9,
    beta2: scalar_float = 0.999,
    epsilon: scalar_float = 1e-8,
    max_iterations: int = 250,
    tolerance: scalar_float = 1e-6,
    loss_fn: Callable[
        [
            Float[Array, "H W"],
            Float[Array, "H W"],
            Optional[Float[Array, "H W"]],
        ],
        scalar_float,
    ] = weighted_mean_squared_error,
) -> ReconstructionResult:
    r"""Reconstruct parameters by minimizing an image-matching loss.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree passed to ``forward_model``.
    forward_model : Callable[[Any], Float[Array, "H W"]]
        Differentiable simulator that maps parameters to a detector
        image.
    experimental_image : Float[Array, "H W"]
        Target detector image.
    weight_map : Float[Array, "H W"], optional
        Optional non-negative per-pixel weights.
    postprocess_fn : Callable[[Float[Array, "H W"]], \
            Float[Array, "H W"]], optional
        Optional transform applied to each simulated image before loss
        evaluation.
    learning_rate : scalar_float, optional
        Adam learning rate. Default: 1e-2
    beta1 : scalar_float, optional
        First-moment decay factor. Default: 0.9
    beta2 : scalar_float, optional
        Second-moment decay factor. Default: 0.999
    epsilon : scalar_float, optional
        Denominator stabilizer. Default: 1e-8
    max_iterations : int, optional
        Maximum number of iterations. Default: 250
    tolerance : scalar_float, optional
        Convergence threshold. Default: 1e-6
    loss_fn : Callable[..., scalar_float], optional
        Image loss used for optimization. Default:
        :func:`weighted_mean_squared_error`

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """

    def objective_fn(params: Any) -> scalar_float:
        simulated_image: Float[Array, "H W"] = _apply_postprocess(
            forward_model(params),
            postprocess_fn,
        )
        return loss_fn(simulated_image, experimental_image, weight_map)

    return adam_optimize(
        initial_params=initial_params,
        objective_fn=objective_fn,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


@jaxtyped(typechecker=beartype)
def adagrad_reconstruction(  # noqa: PLR0913
    initial_params: Any,
    forward_model: Callable[[Any], Float[Array, "H W"]],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    postprocess_fn: Optional[
        Callable[[Float[Array, "H W"]], Float[Array, "H W"]]
    ] = None,
    learning_rate: scalar_float = 1e-1,
    epsilon: scalar_float = 1e-8,
    max_iterations: int = 500,
    tolerance: scalar_float = 1e-6,
    loss_fn: Callable[
        [
            Float[Array, "H W"],
            Float[Array, "H W"],
            Optional[Float[Array, "H W"]],
        ],
        scalar_float,
    ] = weighted_mean_squared_error,
) -> ReconstructionResult:
    r"""Reconstruct parameters by minimizing an image-matching loss.

    Parameters
    ----------
    initial_params : Any
        Initial parameter pytree passed to ``forward_model``.
    forward_model : Callable[[Any], Float[Array, "H W"]]
        Differentiable simulator that maps parameters to a detector
        image.
    experimental_image : Float[Array, "H W"]
        Target detector image.
    weight_map : Float[Array, "H W"], optional
        Optional non-negative per-pixel weights.
    postprocess_fn : Callable[[Float[Array, "H W"]], \
            Float[Array, "H W"]], optional
        Optional transform applied to each simulated image before loss
        evaluation.
    learning_rate : scalar_float, optional
        Adagrad base learning rate. Default: 1e-1
    epsilon : scalar_float, optional
        Denominator stabilizer. Default: 1e-8
    max_iterations : int, optional
        Maximum number of iterations. Default: 500
    tolerance : scalar_float, optional
        Convergence threshold. Default: 1e-6
    loss_fn : Callable[..., scalar_float], optional
        Image loss used for optimization. Default:
        :func:`weighted_mean_squared_error`

    Returns
    -------
    result : ReconstructionResult
        Final parameters plus optimization traces.
    """

    def objective_fn(params: Any) -> scalar_float:
        simulated_image: Float[Array, "H W"] = _apply_postprocess(
            forward_model(params),
            postprocess_fn,
        )
        return loss_fn(simulated_image, experimental_image, weight_map)

    return adagrad_optimize(
        initial_params=initial_params,
        objective_fn=objective_fn,
        learning_rate=learning_rate,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


__all__: list[str] = [
    "ReconstructionResult",
    "adagrad_optimize",
    "adagrad_reconstruction",
    "adam_optimize",
    "adam_reconstruction",
    "gauss_newton_least_squares",
    "gauss_newton_reconstruction",
]
