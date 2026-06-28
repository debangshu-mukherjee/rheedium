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
:func:`solve`
    Solve a reconstruction problem with optimistix or optax.
:func:`multistart`
    Run a reconstruction problem from multiple initial guesses.
:func:`fit_geometry_beam`
    Fit orientation and beam-mode parameters for a fixed crystal.
:func:`build_incoherent_intensity_library`
    Build a per-sample incoherent intensity library from a base object.
:func:`reconstruct_incoherent_weights`
    Recover incoherent distribution weights from an intensity library.
:func:`reconstruct_distribution`
    Recover a distribution over a perturbation axis from a measured image.
"""

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.tools.caching import enable_compilation_cache
from rheedium.types import (
    BeamModeDistribution,
    Distribution,
    DistributionAxisSpec,
    LaplaceUncertainty,
    ReconProblem,
    ReconResult,
    ReductionMode,
    create_distribution,
    scalar_float,
)
from rheedium.types.recon_types import (
    _default_loss,
    _default_residual,
    _tree_mean_square,
)

from .uncertainty import covariance_from_fisher, laplace_uncertainty


def _n_multistart_latents(initial_latents: Any) -> int:
    """Return and validate the leading start count for batched latents."""
    leaves: list[Any] = jax.tree_util.tree_leaves(initial_latents)
    if not leaves:
        raise ValueError("initial_latents must contain at least one leaf")
    n_starts: int = int(jnp.asarray(leaves[0]).shape[0])
    if n_starts <= 0:
        raise ValueError("initial_latents must contain at least one start")
    for leaf in leaves[1:]:
        if int(jnp.asarray(leaf).shape[0]) != n_starts:
            raise ValueError(
                "all initial_latents leaves must share a leading start axis"
            )
    return n_starts


def _random_multistart_latents(
    template_latent: Any,
    key: Array,
    n_starts: int,
    random_scale: scalar_float,
    include_initial: bool,
) -> Any:
    """Generate seeded random latent starts around a template pytree."""
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    n_random: int = n_starts - 1 if include_initial else n_starts
    if n_random < 0:
        raise ValueError("n_starts must be at least one")

    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree_util.tree_flatten(template_latent)
    if not leaves:
        raise ValueError("template_latent must contain at least one leaf")
    keys: Array = jax.random.split(key, len(leaves))
    scale: Float[Array, ""] = jnp.asarray(random_scale, dtype=jnp.float64)
    batched_leaves: list[Array] = []
    for leaf, leaf_key in zip(leaves, keys, strict=True):
        leaf_array: Array = jnp.asarray(leaf)
        dtype: Any = (
            leaf_array.dtype
            if jnp.issubdtype(leaf_array.dtype, jnp.floating)
            else jnp.float64
        )
        center: Array = jnp.asarray(leaf_array, dtype=dtype)
        random_shape: tuple[int, ...] = (n_random, *center.shape)
        noise: Array = scale * jax.random.normal(
            leaf_key,
            random_shape,
            dtype=dtype,
        )
        random_starts: Array = center[None, ...] + noise
        if include_initial:
            batched_leaf: Array = jnp.concatenate(
                [center[None, ...], random_starts],
                axis=0,
            )
        else:
            batched_leaf = random_starts
        batched_leaves.append(batched_leaf)
    generated_latents: Any = treedef.unflatten(batched_leaves)
    return generated_latents


def _result_message(result: Any) -> str:
    """Convert an optimistix result enum to a readable status string."""
    is_successful: bool = bool(result == optx.RESULTS.successful)
    if is_successful:
        message: str = "successful"
        return message
    message: str = str(optx.RESULTS[result])
    return message


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
    compilation_cache_dir: Optional[str] = None,
    compilation_cache_per_arch: bool = True,
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
    compilation_cache_dir : Optional[str], optional
        Persistent XLA compilation-cache root for this inversion. If supplied,
        :func:`rheedium.tools.enable_compilation_cache` is called before the
        solver scan is compiled. Default: None
    compilation_cache_per_arch : bool, optional
        If True, namespace ``compilation_cache_dir`` by CPU/GPU architecture
        when enabling the cache. Default: True

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
    4. When requested, persistent compilation cache configuration happens
       before any optimizer executable is lowered.
    """
    if compilation_cache_dir is not None:
        enable_compilation_cache(
            compilation_cache_dir,
            per_arch=compilation_cache_per_arch,
        )

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
    if mode == "least_squares":
        final_loss: Float[Array, ""] = _tree_mean_square(final_residual)
    else:
        final_loss = problem.loss_fn(final_simulated, problem.measured)
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
def multistart(  # noqa: PLR0913
    problem: ReconProblem,
    initial_latents: Any,
    solver: Optional[Any] = None,
    mode: str = "least_squares",
    max_steps: int = 256,
    rtol: scalar_float = 1e-6,
    atol: scalar_float = 1e-6,
    learning_rate: scalar_float = 1e-2,
    clip_norm: scalar_float = 1.0,
    key: Optional[Array] = None,
    n_starts: Optional[int] = None,
    random_scale: scalar_float = 1.0,
    include_initial: bool = True,
    compilation_cache_dir: Optional[str] = None,
    compilation_cache_per_arch: bool = True,
) -> ReconResult:
    """Run a reconstruction problem from multiple initial guesses.

    :see: :class:`~.test_solve.TestReconSolve`

    Parameters
    ----------
    problem : ReconProblem
        Reconstruction problem to solve.
    initial_latents : Any
        Pytree whose leaves have a leading ``K`` start dimension. If ``key``
        and ``n_starts`` are supplied, this is instead treated as a template
        latent used to generate seeded random starts.
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
    key : Optional[Array], optional
        PRNG key used to generate random starts around ``initial_latents``.
        If omitted, ``initial_latents`` is interpreted as an already-batched
        start pytree. Default: None
    n_starts : Optional[int], optional
        Total number of starts to generate when ``key`` is provided. Default:
        None
    random_scale : scalar_float, optional
        Standard deviation of the normal random perturbations applied to the
        template latent when generating starts. Default: 1.0
    include_initial : bool, optional
        If True, prepend the template latent as the first generated start.
        Default: True
    compilation_cache_dir : Optional[str], optional
        Persistent XLA compilation-cache root passed to each solve. Default:
        None
    compilation_cache_per_arch : bool, optional
        If True, namespace ``compilation_cache_dir`` by architecture when
        enabling the cache. Default: True

    Returns
    -------
    best_result : ReconResult
        Result with the lowest final loss across starts.

    Notes
    -----
    1. Optionally generate seeded starts around a template latent.
    2. Slice the leading start axis from every latent pytree leaf.
    3. Run the common ``solve`` surface for each start.
    4. Select the lowest-loss result deterministically.
    """
    active_initial_latents: Any = initial_latents
    if key is not None:
        if n_starts is None:
            raise ValueError("n_starts is required when key is provided")
        active_initial_latents = _random_multistart_latents(
            template_latent=initial_latents,
            key=key,
            n_starts=n_starts,
            random_scale=random_scale,
            include_initial=include_initial,
        )
    elif n_starts is not None:
        raise ValueError("key is required when n_starts is provided")

    n_active_starts: int = _n_multistart_latents(active_initial_latents)

    results: list[ReconResult] = []
    for start_index in range(n_active_starts):
        start_latent: Any = jax.tree_util.tree_map(
            lambda leaf, idx=start_index: leaf[idx],
            active_initial_latents,
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
            compilation_cache_dir=compilation_cache_dir,
            compilation_cache_per_arch=compilation_cache_per_arch,
        )
        results.append(start_result)

    losses: Float[Array, "K"] = jnp.asarray(
        [result.loss for result in results]
    )
    best_index: int = int(jnp.argmin(losses))
    best_result: ReconResult = results[best_index]
    return best_result


@jaxtyped(typechecker=beartype)
def fit_geometry_beam(  # noqa: PLR0913
    crystal: Any,
    measured: Any,
    forward: Callable[[Any, Any, BeamModeDistribution], Any],
    initial_latent: Any,
    transform: Callable[[Any], Tuple[Any, BeamModeDistribution]],
    residual_fn: Callable[[Any, Any], Any] = _default_residual,
    loss_fn: Callable[[Any, Any], Float[Array, ""]] = _default_loss,
    solver: Optional[Any] = None,
    mode: str = "least_squares",
    max_steps: int = 256,
    rtol: scalar_float = 1e-6,
    atol: scalar_float = 1e-6,
    learning_rate: scalar_float = 1e-2,
    clip_norm: scalar_float = 1.0,
    noise_variance: scalar_float = 1.0,
    uncertainty_regularization: scalar_float = 1e-6,
    compilation_cache_dir: Optional[str] = None,
    compilation_cache_per_arch: bool = True,
) -> Tuple[Any, BeamModeDistribution, Float[Array, "P P"]]:
    """Fit orientation and beam-mode parameters for a fixed crystal.

    :see: :class:`~.test_solve.TestGeometryBeamFit`

    Parameters
    ----------
    crystal : Any
        Fixed known structure or structure-like carrier supplied to
        ``forward``.
    measured : Any
        Measured detector data to match.
    forward : Callable[[Any, Any, BeamModeDistribution], Any]
        Static calibrated forward model with signature
        ``(crystal, orientation, beam_modes) -> simulated``.
    initial_latent : Any
        Initial unconstrained optimizer-coordinate pytree.
    transform : Callable[[Any], Tuple[Any, BeamModeDistribution]]
        Static map from latent coordinates to
        ``(orientation, beam_modes)`` physical parameters.
    residual_fn : Callable[[Any, Any], Any], optional
        Residual builder comparing simulated and measured data. Default:
        elementwise subtraction.
    loss_fn : Callable[[Any, Any], Float[Array, ""]], optional
        Scalar loss used by minimization modes. Default: mean squared error.
    solver : Optional[Any], optional
        Explicit optimistix solver or minimizer. Default: chosen by ``mode``.
    mode : str, optional
        Solver family (**static**) passed to :func:`solve`. Default:
        ``"least_squares"``.
    max_steps : int, optional
        Maximum solver steps (**static**). Default: 256
    rtol : scalar_float, optional
        Relative convergence tolerance. Default: 1e-6
    atol : scalar_float, optional
        Absolute convergence tolerance. Default: 1e-6
    learning_rate : scalar_float, optional
        Learning rate for the optax path. Default: 1e-2
    clip_norm : scalar_float, optional
        Global gradient clipping norm for the optax path. Default: 1.0
    noise_variance : scalar_float, optional
        Per-residual Gaussian noise variance for Laplace covariance.
        Default: 1.0
    uncertainty_regularization : scalar_float, optional
        Diagonal Fisher regularization for covariance. Default: 1e-6
    compilation_cache_dir : Optional[str], optional
        Persistent XLA compilation-cache root passed to :func:`solve`.
        Default: None
    compilation_cache_per_arch : bool, optional
        If True, namespace ``compilation_cache_dir`` by architecture.
        Default: True

    Returns
    -------
    orientation : Any
        Fitted orientation parameters returned by ``transform``.
    beam_modes : BeamModeDistribution
        Fitted beam-mode distribution returned by ``transform``.
    covariance : Float[Array, "P P"]
        Flattened Laplace covariance for ``(orientation, beam_modes)``.

    Notes
    -----
    1. Close the known crystal over a calibrated geometry/beam forward model.
    2. Solve through the common :class:`ReconProblem` surface.
    3. Estimate local covariance in the fitted physical basis.
    """

    def forward_from_physical(
        physical_params: Tuple[Any, BeamModeDistribution],
    ) -> Any:
        orientation: Any
        beam_modes: BeamModeDistribution
        orientation, beam_modes = physical_params
        simulated: Any = forward(crystal, orientation, beam_modes)
        return simulated

    problem: ReconProblem = ReconProblem(
        forward=forward_from_physical,
        measured=measured,
        transform=transform,
        residual_fn=residual_fn,
        loss_fn=loss_fn,
    )
    result: ReconResult = solve(
        problem=problem,
        initial_latent=initial_latent,
        solver=solver,
        mode=mode,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        compilation_cache_dir=compilation_cache_dir,
        compilation_cache_per_arch=compilation_cache_per_arch,
    )

    def residual_from_physical(
        physical_params: Tuple[Any, BeamModeDistribution],
    ) -> Any:
        simulated: Any = forward_from_physical(physical_params)
        residual: Any = residual_fn(simulated, measured)
        return residual

    uncertainty: LaplaceUncertainty = laplace_uncertainty(
        residual_fn=residual_from_physical,
        params=result.params,
        noise_variance=noise_variance,
        regularization=uncertainty_regularization,
    )
    orientation: Any
    beam_modes: BeamModeDistribution
    orientation, beam_modes = result.params
    covariance: Float[Array, "P P"] = uncertainty.covariance
    return orientation, beam_modes, covariance


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
    "build_incoherent_intensity_library",
    "fit_geometry_beam",
    "multistart",
    "reconstruct_distribution",
    "reconstruct_incoherent_weights",
    "solve",
]
