r"""Optimistix-backed solvers for differentiable RHEED inversion.

Extended Summary
----------------
This module defines the general reconstruction problem surface used by the new
inverse API. A :class:`ReconProblem` stores a differentiable forward model,
measured data, and the latent-to-physical transform. :func:`solve` dispatches
to ``optimistix`` least-squares or minimisation solvers, with ``optax``
available as the first-order gradient-transform layer for high-dimensional or
free-form-weight problems.

Routine Listings
----------------
:class:`DistributionAxisSpec`
    Static perturbation-axis contract for distribution reconstruction.
:class:`ReconProblem`
    Differentiable inverse problem definition for reconstruction solvers.
:class:`ReconResult`
    Result container returned by the general reconstruction solver.
:func:`create_distribution_axis_spec`
    Create a perturbation-axis specification for library reconstruction.
:func:`solve`
    Solve a reconstruction problem with optimistix or optax.
:func:`multistart`
    Run a reconstruction problem from multiple initial guesses.
:func:`build_incoherent_intensity_library`
    Build a per-sample incoherent intensity library from a base object.
:func:`reconstruct_incoherent_weights`
    Recover incoherent distribution weights from an intensity library.
:func:`reconstruct_distribution`
    Recover a distribution over a perturbation axis from a measured image.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.types import (
    Distribution,
    ReductionMode,
    create_distribution,
    scalar_float,
)

from .uncertainty import covariance_from_fisher


def _identity(value: Any) -> Any:
    """Return a value unchanged."""
    return value


def _default_residual(simulated: Any, measured: Any) -> Any:
    """Return a default residual pytree."""
    residual: Any = jax.tree_util.tree_map(
        lambda sim_leaf, meas_leaf: jnp.asarray(sim_leaf)
        - jnp.asarray(meas_leaf),
        simulated,
        measured,
    )
    return residual


def _tree_mean_square(tree: Any) -> Float[Array, ""]:
    """Return the mean square across all numeric leaves in a pytree."""
    leaves: list[Any] = jax.tree_util.tree_leaves(tree)
    if not leaves:
        loss: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
        return loss
    total: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    count: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    for leaf in leaves:
        leaf_array: Array = jnp.asarray(leaf)
        total = total + jnp.sum(jnp.square(leaf_array))
        count = count + jnp.asarray(leaf_array.size, dtype=jnp.float64)
    loss: Float[Array, ""] = total / jnp.maximum(count, 1.0)
    return loss


def _default_loss(simulated: Any, measured: Any) -> Float[Array, ""]:
    """Return the default mean-squared data loss."""
    residual: Any = _default_residual(simulated, measured)
    loss: Float[Array, ""] = _tree_mean_square(residual)
    return loss


def _result_message(result: Any) -> str:
    """Convert an optimistix result enum to a readable status string."""
    is_successful: bool = bool(result == optx.RESULTS.successful)
    if is_successful:
        message: str = "successful"
        return message
    message: str = str(optx.RESULTS[result])
    return message


class DistributionAxisSpec(eqx.Module):
    """Static perturbation-axis contract for distribution reconstruction.

    :see: :class:`~.test_solve.TestReconDistributionReconstruction`

    Attributes
    ----------
    samples : Float[Array, "N D"]
        Latent-axis sample coordinates used as rows in the recovered
        distribution.
    perturbation_fn : Callable[[Any, Float[Array, "D"]], Any]
        Static function that maps ``(base_object, sample)`` to a perturbed
        physical object.
    forward_model : Callable[[Any], Array]
        Static differentiable model that maps the perturbed object to either a
        coherent amplitude image or an intensity image.
    output_kind : str
        Static forward-output interpretation: ``"amplitude"`` or
        ``"intensity"``.
    axis_id : Optional[str]
        Optional static label attached to the recovered
        :class:`~rheedium.types.Distribution`.
    """

    samples: Float[Array, "N D"]
    perturbation_fn: Callable[[Any, Float[Array, "D"]], Any] = eqx.field(
        static=True
    )
    forward_model: Callable[[Any], Array] = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="amplitude")
    axis_id: Optional[str] = eqx.field(static=True, default=None)


class ReconProblem(eqx.Module):
    """Differentiable inverse problem definition for reconstruction solvers.

    :see: :class:`~.test_solve.TestReconSolve`

    Attributes
    ----------
    forward : Callable[[Any], Any]
        Differentiable forward model accepting physical parameters.
    measured : Any
        Measured target data pytree.
    transform : Callable[[Any], Any]
        Static map from unconstrained latent parameters to physical
        parameters.
    residual_fn : Callable[[Any, Any], Any]
        Static residual builder comparing simulated and measured data.
    loss_fn : Callable[[Any, Any], Float[Array, ""]]
        Static scalar loss for minimisation modes.
    """

    forward: Callable[[Any], Any] = eqx.field(static=True)
    measured: Any
    transform: Callable[[Any], Any] = eqx.field(
        static=True,
        default=_identity,
    )
    residual_fn: Callable[[Any, Any], Any] = eqx.field(
        static=True,
        default=_default_residual,
    )
    loss_fn: Callable[[Any, Any], Float[Array, ""]] = eqx.field(
        static=True,
        default=_default_loss,
    )

    def physical_from_latent(self, latent_params: Any) -> Any:
        """Map latent optimizer coordinates to physical parameters."""
        physical_params: Any = self.transform(latent_params)
        return physical_params

    def simulate(self, latent_params: Any) -> Any:
        """Evaluate the forward model from latent optimizer coordinates."""
        physical_params: Any = self.physical_from_latent(latent_params)
        simulated: Any = self.forward(physical_params)
        return simulated

    def residual_from_latent(self, latent_params: Any) -> Any:
        """Evaluate residuals from latent optimizer coordinates."""
        simulated: Any = self.simulate(latent_params)
        residual: Any = self.residual_fn(simulated, self.measured)
        return residual

    def loss_from_latent(self, latent_params: Any) -> Float[Array, ""]:
        """Evaluate scalar loss from latent optimizer coordinates."""
        simulated: Any = self.simulate(latent_params)
        loss: Float[Array, ""] = self.loss_fn(simulated, self.measured)
        return loss


class ReconResult(eqx.Module):
    """Result container returned by the general reconstruction solver.

    :see: :class:`~.test_solve.TestReconSolve`

    Attributes
    ----------
    params : Any
        Final physical parameter pytree.
    latent_params : Any
        Final unconstrained optimizer-coordinate pytree.
    simulated : Any
        Forward-model output at the final parameters.
    residual : Any
        Residual pytree at the final parameters.
    loss : Float[Array, ""]
        Final scalar data loss.
    iterations : Int[Array, ""]
        Number of nonlinear solver steps reported by ``optimistix``.
    converged : Bool[Array, ""]
        True when the solver result is successful or the final loss is below
        the absolute tolerance.
    solver_status : str
        Static human-readable solver status.
    """

    params: Any
    latent_params: Any
    simulated: Any
    residual: Any
    loss: Float[Array, ""]
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]
    solver_status: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_distribution_axis_spec(
    samples: Float[Array, "N D"],
    perturbation_fn: Callable[[Any, Float[Array, "D"]], Any],
    forward_model: Callable[[Any], Array],
    output_kind: str = "amplitude",
    axis_id: Optional[str] = None,
) -> DistributionAxisSpec:
    """Create a perturbation-axis specification for library reconstruction.

    :see: :class:`~.test_solve.TestReconDistributionReconstruction`

    Parameters
    ----------
    samples : Float[Array, "N D"]
        Two-dimensional latent-axis sample coordinates.
    perturbation_fn : Callable[[Any, Float[Array, "D"]], Any]
        Static function mapping ``(base_object, sample)`` to a perturbed
        physical object.
    forward_model : Callable[[Any], Array]
        Static differentiable forward model for one perturbed object.
    output_kind : str, optional
        Static forward-output interpretation, either ``"amplitude"`` or
        ``"intensity"``. Default: ``"amplitude"``
    axis_id : Optional[str], optional
        Optional static axis label stored on the recovered distribution.

    Returns
    -------
    axis_spec : DistributionAxisSpec
        Validated perturbation-axis specification.

    Notes
    -----
    1. Convert samples to ``float64``.
    2. Validate the static sample rank and supported output interpretation.
    3. Store callables and metadata as static PyTree fields.
    """
    sample_array: Float[Array, "N D"] = jnp.asarray(
        samples,
        dtype=jnp.float64,
    )
    if sample_array.ndim != 2:
        raise ValueError("samples must have shape (N, D)")
    if sample_array.shape[0] <= 0:
        raise ValueError("samples must contain at least one row")
    if output_kind not in {"amplitude", "intensity"}:
        raise ValueError("output_kind must be 'amplitude' or 'intensity'")
    axis_spec: DistributionAxisSpec = DistributionAxisSpec(
        samples=sample_array,
        perturbation_fn=perturbation_fn,
        forward_model=forward_model,
        output_kind=output_kind,
        axis_id=axis_id,
    )
    return axis_spec


@jaxtyped(typechecker=beartype)
def solve(  # noqa: PLR0913
    problem: ReconProblem,
    initial_latent: Any,
    solver: Optional[Any] = None,
    mode: str = "least_squares",
    max_steps: int = 256,
    rtol: scalar_float = 1e-6,
    atol: scalar_float = 1e-6,
    learning_rate: scalar_float = 1e-2,
    clip_norm: scalar_float = 1.0,
    throw: bool = False,
) -> ReconResult:
    """Solve a reconstruction problem with optimistix or optax.

    :see: :class:`~.test_solve.TestReconSolve`

    Parameters
    ----------
    problem : ReconProblem
        Reconstruction problem containing forward model, measured data,
        transform, residual, and loss functions.
    initial_latent : Any
        Initial unconstrained optimizer-coordinate pytree.
    solver : Optional[Any], optional
        Explicit ``optimistix`` solver instance. If omitted, a suitable solver
        is chosen from ``mode``.
    mode : str, optional
        Solver family (**static**): ``"least_squares"``, ``"bfgs"``, or
        ``"adamw"``. Default: ``"least_squares"``.
    max_steps : int, optional
        Maximum nonlinear solver steps (**static**). Default: 256
    rtol : scalar_float, optional
        Relative convergence tolerance. Default: 1e-6
    atol : scalar_float, optional
        Absolute convergence tolerance. Default: 1e-6
    learning_rate : scalar_float, optional
        Learning rate used by the optax ``"adamw"`` path. Default: 1e-2
    clip_norm : scalar_float, optional
        Global gradient clipping norm for the optax path. Default: 1.0
    throw : bool, optional
        If True, let ``optimistix`` raise on solver failure. Default: False

    Returns
    -------
    result : ReconResult
        Fitted physical parameters and final reconstruction diagnostics.

    Notes
    -----
    1. Least-squares mode minimizes the problem residual with Levenberg-
       Marquardt by default.
    2. BFGS mode minimizes the scalar ``problem.loss_fn``.
    3. AdamW mode wraps an ``optax`` gradient transformation in
       :class:`optimistix.OptaxMinimiser`.
    """
    if mode == "least_squares":
        active_solver: Any = solver
        if active_solver is None:
            active_solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)

        def residual_objective(latent_params: Any, args: Any) -> Any:
            del args
            residual: Any = problem.residual_from_latent(latent_params)
            return residual

        solution: optx.Solution = optx.least_squares(
            residual_objective,
            active_solver,
            initial_latent,
            max_steps=max_steps,
            throw=throw,
        )
    else:
        active_minimiser: Any = solver
        if active_minimiser is None and mode == "bfgs":
            active_minimiser = optx.BFGS(rtol=rtol, atol=atol)
        elif active_minimiser is None and mode == "adamw":
            gradient_transform: optax.GradientTransformation = optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adamw(learning_rate),
            )
            active_minimiser = optx.OptaxMinimiser(
                gradient_transform,
                rtol=rtol,
                atol=atol,
            )
        elif active_minimiser is None:
            raise ValueError(
                "mode must be 'least_squares', 'bfgs', or 'adamw'"
            )

        def scalar_objective(
            latent_params: Any,
            args: Any,
        ) -> Float[Array, ""]:
            del args
            loss: Float[Array, ""] = problem.loss_from_latent(latent_params)
            return loss

        solution = optx.minimise(
            scalar_objective,
            active_minimiser,
            initial_latent,
            max_steps=max_steps,
            throw=throw,
        )

    final_latent: Any = solution.value
    final_params: Any = problem.physical_from_latent(final_latent)
    final_simulated: Any = problem.forward(final_params)
    final_residual: Any = problem.residual_fn(
        final_simulated,
        problem.measured,
    )
    final_loss: Float[Array, ""] = _tree_mean_square(final_residual)
    iterations: Int[Array, ""] = jnp.asarray(
        solution.stats.get("num_steps", max_steps),
        dtype=jnp.int32,
    )
    solver_successful: Bool[Array, ""] = jnp.asarray(
        solution.result == optx.RESULTS.successful
    )
    loss_successful: Bool[Array, ""] = jnp.asarray(final_loss <= atol)
    converged: Bool[Array, ""] = jnp.logical_or(
        solver_successful,
        loss_successful,
    )
    result: ReconResult = ReconResult(
        params=final_params,
        latent_params=final_latent,
        simulated=final_simulated,
        residual=final_residual,
        loss=final_loss,
        iterations=iterations,
        converged=converged,
        solver_status=_result_message(solution.result),
    )
    return result


@jaxtyped(typechecker=beartype)
def multistart(
    problem: ReconProblem,
    initial_latents: Any,
    solver: Optional[Any] = None,
    mode: str = "least_squares",
    max_steps: int = 256,
    rtol: scalar_float = 1e-6,
    atol: scalar_float = 1e-6,
    learning_rate: scalar_float = 1e-2,
    clip_norm: scalar_float = 1.0,
) -> ReconResult:
    """Run a reconstruction problem from multiple initial guesses.

    :see: :class:`~.test_solve.TestReconSolve`

    Parameters
    ----------
    problem : ReconProblem
        Reconstruction problem to solve.
    initial_latents : Any
        Pytree whose leaves have a leading ``K`` start dimension.
    solver : Optional[Any], optional
        Explicit ``optimistix`` solver instance. Default: chosen by ``mode``.
    mode : str, optional
        Solver family (**static**) passed to :func:`solve`.
        Default: ``"least_squares"``.
    max_steps : int, optional
        Maximum nonlinear solver steps (**static**). Default: 256
    rtol : scalar_float, optional
        Relative convergence tolerance. Default: 1e-6
    atol : scalar_float, optional
        Absolute convergence tolerance. Default: 1e-6
    learning_rate : scalar_float, optional
        Learning rate used by the optax path. Default: 1e-2
    clip_norm : scalar_float, optional
        Global gradient clipping norm for the optax path. Default: 1.0

    Returns
    -------
    best_result : ReconResult
        Result with the lowest final loss across starts.

    Notes
    -----
    1. Slice the leading start axis from every latent pytree leaf.
    2. Run the common ``solve`` surface for each start.
    3. Select the lowest-loss result deterministically.
    """
    leaves: list[Any] = jax.tree_util.tree_leaves(initial_latents)
    if not leaves:
        raise ValueError("initial_latents must contain at least one leaf")
    n_starts: int = int(jnp.asarray(leaves[0]).shape[0])
    if n_starts <= 0:
        raise ValueError("initial_latents must contain at least one start")

    results: list[ReconResult] = []
    for start_index in range(n_starts):
        start_latent: Any = jax.tree_util.tree_map(
            lambda leaf, idx=start_index: leaf[idx],
            initial_latents,
        )
        start_result: ReconResult = solve(
            problem=problem,
            initial_latent=start_latent,
            solver=solver,
            mode=mode,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            learning_rate=learning_rate,
            clip_norm=clip_norm,
        )
        results.append(start_result)

    losses: Float[Array, "K"] = jnp.asarray(
        [result.loss for result in results]
    )
    best_index: int = int(jnp.argmin(losses))
    best_result: ReconResult = results[best_index]
    return best_result


@jaxtyped(typechecker=beartype)
def build_incoherent_intensity_library(
    base_object: Any,
    axis_spec: DistributionAxisSpec,
) -> Float[Array, "N H W"]:
    """Build a per-sample incoherent intensity library from a base object.

    :see: :class:`~.test_solve.TestReconDistributionReconstruction`

    Parameters
    ----------
    base_object : Any
        Base crystal, structure, detector pattern carrier, or other physical
        object perturbed along ``axis_spec.samples``.
    axis_spec : DistributionAxisSpec
        Perturbation-axis specification containing samples, perturbation
        function, forward model, output interpretation, and axis label.

    Returns
    -------
    intensity_library : Float[Array, "N H W"]
        Per-sample intensity images suitable for
        :func:`reconstruct_incoherent_weights`.

    Notes
    -----
    1. Vectorize over latent-axis samples with ``jax.vmap``.
    2. Perturb the base object at each sample.
    3. Interpret the forward output as amplitude via ``|A|^2`` or as an
       already-formed intensity image.
    """

    def sample_intensity(sample: Float[Array, "D"]) -> Float[Array, "H W"]:
        perturbed_object: Any = axis_spec.perturbation_fn(
            base_object,
            sample,
        )
        forward_output: Array = jnp.asarray(
            axis_spec.forward_model(perturbed_object)
        )
        if axis_spec.output_kind == "amplitude":
            intensity: Float[Array, "H W"] = jnp.real(
                jnp.conj(forward_output) * forward_output
            )
            return intensity
        intensity: Float[Array, "H W"] = jnp.real(forward_output)
        return intensity

    intensity_library: Float[Array, "N H W"] = jax.vmap(sample_intensity)(
        axis_spec.samples
    )
    return intensity_library


@jaxtyped(typechecker=beartype)
def reconstruct_incoherent_weights(
    intensity_library: Float[Array, "N H W"],
    measured_image: Float[Array, "H W"],
    ridge: scalar_float = 1e-10,
) -> Float[Array, "N"]:
    """Recover incoherent distribution weights from an intensity library.

    :see: :class:`~.test_solve.TestReconSolve`

    Parameters
    ----------
    intensity_library : Float[Array, "N H W"]
        Per-sample intensity images :math:`|A_n|^2`.
    measured_image : Float[Array, "H W"]
        Measured image generated by an incoherent weighted sum.
    ridge : scalar_float, optional
        Diagonal Tikhonov regularization for the normal equations.
        Default: 1e-10

    Returns
    -------
    weights : Float[Array, "N"]
        Non-negative normalized distribution weights.

    Notes
    -----
    1. Flatten the intensity library into a pixel-by-sample design matrix.
    2. Solve the regularized normal equations.
    3. Project tiny negative numerical excursions to zero and renormalize.
    """
    n_samples: int = intensity_library.shape[0]
    design: Float[Array, "P N"] = jnp.reshape(
        jnp.moveaxis(intensity_library, 0, -1),
        (-1, n_samples),
    )
    target: Float[Array, "P"] = jnp.ravel(measured_image)
    normal_matrix: Float[Array, "N N"] = design.T @ design
    identity: Float[Array, "N N"] = jnp.eye(
        n_samples,
        dtype=normal_matrix.dtype,
    )
    rhs: Float[Array, "N"] = design.T @ target
    raw_weights: Float[Array, "N"] = jnp.linalg.solve(
        normal_matrix + ridge * identity,
        rhs,
    )
    nonnegative_weights: Float[Array, "N"] = jnp.clip(
        raw_weights,
        0.0,
        None,
    )
    total_weight: Float[Array, ""] = jnp.sum(nonnegative_weights)
    uniform_weights: Float[Array, "N"] = (
        jnp.ones(n_samples, dtype=nonnegative_weights.dtype) / n_samples
    )
    weights: Float[Array, "N"] = jax.lax.cond(
        total_weight > 0.0,
        lambda: nonnegative_weights / total_weight,
        lambda: uniform_weights,
    )
    return weights


@jaxtyped(typechecker=beartype)
def reconstruct_distribution(
    measured_image: Float[Array, "H W"],
    base_object: Any,
    axis_spec: DistributionAxisSpec,
    ridge: scalar_float = 1e-10,
    noise_variance: scalar_float = 1.0,
) -> Tuple[Distribution, Float[Array, "N"]]:
    """Recover a distribution over a perturbation axis from a measured image.

    :see: :class:`~.test_solve.TestReconDistributionReconstruction`

    Parameters
    ----------
    measured_image : Float[Array, "H W"]
        Measured detector image to explain as an incoherent mixture.
    base_object : Any
        Base crystal, structure, detector pattern carrier, or other physical
        object from which the per-axis library is generated.
    axis_spec : DistributionAxisSpec
        Perturbation-axis specification for building the intensity library.
    ridge : scalar_float, optional
        Diagonal Tikhonov regularization for the linear solve. Default: 1e-10
    noise_variance : scalar_float, optional
        Per-pixel Gaussian noise variance used for the returned weight band.
        Default: 1.0

    Returns
    -------
    distribution : Distribution
        Recovered incoherent distribution over ``axis_spec.samples``.
    band : Float[Array, "N"]
        One-sigma linearized uncertainty band for the recovered weights.

    Notes
    -----
    1. Build the per-sample incoherent intensity library from the base object.
    2. Solve the convex linear weight reconstruction problem.
    3. Package the recovered simplex weights as a
       :class:`~rheedium.types.Distribution`.
    4. Estimate a local Fisher/Laplace band in weight coordinates.
    """
    intensity_library: Float[Array, "N H W"] = (
        build_incoherent_intensity_library(
            base_object=base_object,
            axis_spec=axis_spec,
        )
    )
    weights: Float[Array, "N"] = reconstruct_incoherent_weights(
        intensity_library=intensity_library,
        measured_image=measured_image,
        ridge=ridge,
    )
    distribution: Distribution = create_distribution(
        samples=axis_spec.samples,
        weights=weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id=axis_spec.axis_id,
    )
    n_samples: int = intensity_library.shape[0]
    design: Float[Array, "P N"] = jnp.reshape(
        jnp.moveaxis(intensity_library, 0, -1),
        (-1, n_samples),
    )
    safe_noise_variance: Float[Array, ""] = jnp.maximum(
        jnp.asarray(noise_variance, dtype=design.dtype),
        jnp.asarray(1e-12, dtype=design.dtype),
    )
    fisher_information: Float[Array, "N N"] = (
        design.T @ design
    ) / safe_noise_variance
    covariance: Float[Array, "N N"] = covariance_from_fisher(
        fisher_information=fisher_information,
        regularization=ridge,
    )
    band: Float[Array, "N"] = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0))
    return distribution, band


__all__: list[str] = [
    "DistributionAxisSpec",
    "ReconProblem",
    "ReconResult",
    "build_incoherent_intensity_library",
    "create_distribution_axis_spec",
    "multistart",
    "reconstruct_distribution",
    "reconstruct_incoherent_weights",
    "solve",
]
