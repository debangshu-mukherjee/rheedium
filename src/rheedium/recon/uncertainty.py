r"""Fisher and Laplace uncertainty helpers for reconstruction problems.

Extended Summary
----------------
This module generalizes the orientation-specific Fisher information workflow
to arbitrary latent pytrees. Residual functions are flattened with JAX's pytree
utilities, differentiated with ``jax.jacrev``, and converted into
Gauss-Newton/Fisher matrices for local covariance estimates.
The same module also provides the posterior-sampling layer used by
reconstruction UQ: ``blackjax`` NUTS chains are summarized into credible
intervals, R-hat, and effective sample-size diagnostics.

Routine Listings
----------------
:func:`fisher_information_from_residual`
    Compute a Gauss-Newton/Fisher matrix from a residual function.
:func:`covariance_from_fisher`
    Regularize and invert a Fisher information matrix.
:func:`laplace_uncertainty`
    Build a local Laplace uncertainty estimate from residual sensitivities.
:func:`laplace_inverse_mass_matrix`
    Build a blackjax inverse-mass warm start from Laplace covariance.
:func:`posterior_from_samples`
    Summarize posterior samples with diagnostics and credible intervals.
:func:`sample_posterior`
    Draw blackjax NUTS samples from a differentiable log posterior.
"""

import blackjax
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, jaxtyped

from rheedium.types import LaplaceUncertainty, PosteriorSamples, scalar_float

_EPS: float = 1e-12


def _identity(value: Any) -> Any:
    """Return a value unchanged."""
    return value


def _as_chain_initial_positions(
    initial_position: Any,
) -> tuple[Float[Array, "C P"], Callable[[Float[Array, "P"]], Any]]:
    """Flatten initial positions and detect a leading chain axis."""
    if isinstance(initial_position, jax.Array) and initial_position.ndim == 2:
        chain_positions: Float[Array, "C P"] = jnp.asarray(
            initial_position,
            dtype=jnp.float64,
        )
        return chain_positions, _identity

    flat_position: Float[Array, "P"]
    unravel_fn: Callable[[Float[Array, "P"]], Any]
    flat_position, unravel_fn = ravel_pytree(initial_position)
    chain_positions = flat_position[None, :]
    return chain_positions, unravel_fn


def _split_r_hat(samples: Float[Array, "C S P"]) -> Float[Array, "P"]:
    """Compute split-chain R-hat for flattened samples."""
    n_chains: int = samples.shape[0]
    n_draws: int = samples.shape[1]
    if n_draws < 4:
        return jnp.full((samples.shape[2],), jnp.nan, dtype=samples.dtype)
    usable_draws: int = max((n_draws // 2) * 2, 2)
    trimmed: Float[Array, "C S2 P"] = samples[:, :usable_draws, :]
    half_draws: int = usable_draws // 2
    split_samples: Float[Array, "C2 H P"] = jnp.reshape(
        trimmed,
        (n_chains * 2, half_draws, samples.shape[2]),
    )
    chain_means: Float[Array, "C2 P"] = jnp.mean(split_samples, axis=1)
    chain_variances: Float[Array, "C2 P"] = jnp.var(
        split_samples,
        axis=1,
        ddof=1,
    )
    within_variance: Float[Array, "P"] = jnp.mean(chain_variances, axis=0)
    between_variance: Float[Array, "P"] = half_draws * jnp.var(
        chain_means,
        axis=0,
        ddof=1,
    )
    variance_hat: Float[Array, "P"] = (
        half_draws - 1.0
    ) / half_draws * within_variance + between_variance / half_draws
    ratio: Float[Array, "P"] = variance_hat / jnp.maximum(
        within_variance,
        _EPS,
    )
    r_hat: Float[Array, "P"] = jnp.where(
        within_variance > _EPS,
        jnp.sqrt(jnp.maximum(ratio, 0.0)),
        jnp.ones_like(within_variance),
    )
    return r_hat


def _effective_sample_size(
    samples: Float[Array, "C S P"],
) -> Float[Array, "P"]:
    """Estimate effective sample size from positive autocorrelations."""
    n_chains: int = samples.shape[0]
    n_draws: int = samples.shape[1]
    n_params: int = samples.shape[2]
    max_lag: int = max(min(n_draws - 1, 64), 1)
    centered: Float[Array, "C S P"] = samples - jnp.mean(
        samples,
        axis=1,
        keepdims=True,
    )
    variance: Float[Array, "C P"] = jnp.mean(centered**2, axis=1)
    autocorrelation_values: list[Float[Array, "P"]] = []
    for lag in range(1, max_lag + 1):
        numerator: Float[Array, "C P"] = jnp.mean(
            centered[:, :-lag, :] * centered[:, lag:, :],
            axis=1,
        )
        chain_autocorrelation: Float[Array, "C P"] = numerator / jnp.maximum(
            variance,
            _EPS,
        )
        autocorrelation: Float[Array, "P"] = jnp.mean(
            chain_autocorrelation,
            axis=0,
        )
        autocorrelation_values.append(autocorrelation)

    autocorrelations: Float[Array, "L P"] = jnp.stack(
        autocorrelation_values,
        axis=0,
    )
    positive_autocorrelations: Float[Array, "L P"] = jnp.clip(
        autocorrelations,
        0.0,
        None,
    )
    autocorrelation_time: Float[Array, "P"] = 1.0 + 2.0 * jnp.sum(
        positive_autocorrelations,
        axis=0,
    )
    total_draws: Float[Array, ""] = jnp.asarray(
        n_chains * n_draws,
        dtype=samples.dtype,
    )
    ess: Float[Array, "P"] = jnp.minimum(
        total_draws / jnp.maximum(autocorrelation_time, _EPS),
        jnp.full((n_params,), total_draws, dtype=samples.dtype),
    )
    return ess


def _sample_covariance(
    flat_samples: Float[Array, "N P"],
) -> Float[Array, "P P"]:
    """Return an unbiased empirical covariance matrix."""
    n_samples: int = flat_samples.shape[0]
    centered: Float[Array, "N P"] = flat_samples - jnp.mean(
        flat_samples,
        axis=0,
        keepdims=True,
    )
    denominator: Float[Array, ""] = jnp.maximum(
        jnp.asarray(n_samples - 1, dtype=flat_samples.dtype),
        1.0,
    )
    covariance: Float[Array, "P P"] = centered.T @ centered / denominator
    return covariance


def _run_nuts_chain(  # noqa: PLR0913
    flat_log_probability: Callable[[Float[Array, "P"]], Float[Array, ""]],
    start_position: Float[Array, "P"],
    chain_key: Array,
    num_samples: int,
    num_warmup: int,
    step_size: scalar_float,
    inverse_mass_matrix: Float[Array, "..."],
    adapt: bool,
    target_acceptance_rate: scalar_float,
) -> tuple[
    Float[Array, "S P"],
    Float[Array, "S"],
    Float[Array, ""],
]:
    """Run one blackjax NUTS chain and return samples and acceptance."""
    warmup_key: Array
    sample_key: Array
    warmup_key, sample_key = jax.random.split(chain_key)
    if adapt and num_warmup > 0:
        is_diagonal: bool = inverse_mass_matrix.ndim == 1
        adaptation: Any = blackjax.window_adaptation(
            blackjax.nuts,
            flat_log_probability,
            is_mass_matrix_diagonal=is_diagonal,
            initial_step_size=float(step_size),
            target_acceptance_rate=float(target_acceptance_rate),
            progress_bar=False,
        )
        adaptation_result: Any
        adaptation_result, _adaptation_info = adaptation.run(
            warmup_key,
            start_position,
            num_steps=num_warmup,
        )
        parameters: dict[str, Any] = adaptation_result.parameters
        initial_state: Any = adaptation_result.state
    else:
        parameters = {
            "step_size": jnp.asarray(step_size, dtype=start_position.dtype),
            "inverse_mass_matrix": inverse_mass_matrix,
        }
        initial_algorithm: Any = blackjax.nuts(
            flat_log_probability,
            **parameters,
        )
        initial_state = initial_algorithm.init(start_position)

    algorithm: Any = blackjax.nuts(flat_log_probability, **parameters)
    sample_keys: Array = jax.random.split(sample_key, num_samples)

    def one_step(state: Any, subkey: Array) -> tuple[Any, tuple[Any, ...]]:
        next_state: Any
        info: Any
        next_state, info = algorithm.step(subkey, state)
        sample_tuple: tuple[Any, ...] = (
            next_state.position,
            next_state.logdensity,
            info.acceptance_rate,
        )
        return next_state, sample_tuple

    _final_state: Any
    positions: Float[Array, "S P"]
    log_probabilities: Float[Array, "S"]
    acceptance_rates: Float[Array, "S"]
    (
        _final_state,
        (
            positions,
            log_probabilities,
            acceptance_rates,
        ),
    ) = jax.lax.scan(one_step, initial_state, sample_keys)
    mean_acceptance_rate: Float[Array, ""] = jnp.mean(acceptance_rates)
    return positions, log_probabilities, mean_acceptance_rate


@jaxtyped(typechecker=beartype)
def fisher_information_from_residual(
    residual_fn: Callable[[Any], Any],
    params: Any,
    noise_variance: scalar_float = 1.0,
) -> Float[Array, "P P"]:
    r"""Compute a Gauss-Newton/Fisher matrix from a residual function.

    :see: :class:`~.test_uncertainty.TestReconUncertainty`

    Parameters
    ----------
    residual_fn : Callable[[Any], Any]
        Function mapping a parameter pytree to a residual pytree.
    params : Any
        Parameter pytree at which the local sensitivity is evaluated.
    noise_variance : scalar_float, optional
        Per-residual Gaussian noise variance. Default: 1.0

    Returns
    -------
    fisher_information : Float[Array, "P P"]
        Flattened Gauss-Newton/Fisher matrix.

    Notes
    -----
    1. Flatten the parameter pytree into a vector.
    2. Differentiate the flattened residual vector with respect to that vector.
    3. Form :math:`J^T J / \sigma^2`.
    """
    flat_params: Float[Array, "P"]
    unravel_fn: Callable[[Float[Array, "P"]], Any]
    flat_params, unravel_fn = ravel_pytree(params)

    def flat_residual_fn(
        flat_parameter_vector: Float[Array, "P"],
    ) -> Float[Array, "R"]:
        residual_tree: Any = residual_fn(unravel_fn(flat_parameter_vector))
        residual_leaves: list[Any] = jax.tree_util.tree_leaves(residual_tree)
        if not residual_leaves:
            residual: Float[Array, "R"] = jnp.zeros(
                (0,),
                dtype=flat_parameter_vector.dtype,
            )
            return residual
        residual_parts: list[Float[Array, "R_i"]] = [
            jnp.ravel(jnp.asarray(leaf)) for leaf in residual_leaves
        ]
        residual: Float[Array, "R"] = jnp.concatenate(residual_parts)
        return residual

    jacobian: Float[Array, "R P"] = jax.jacrev(flat_residual_fn)(flat_params)
    safe_noise_variance: Float[Array, ""] = jnp.maximum(
        jnp.asarray(noise_variance, dtype=flat_params.dtype),
        jnp.asarray(1e-12, dtype=flat_params.dtype),
    )
    fisher_information: Float[Array, "P P"] = (
        jacobian.T @ jacobian
    ) / safe_noise_variance
    return fisher_information


@jaxtyped(typechecker=beartype)
def covariance_from_fisher(
    fisher_information: Float[Array, "P P"],
    regularization: scalar_float = 1e-6,
) -> Float[Array, "P P"]:
    """Regularize and invert a Fisher information matrix.

    :see: :class:`~.test_uncertainty.TestReconUncertainty`

    Parameters
    ----------
    fisher_information : Float[Array, "P P"]
        Symmetric positive semi-definite Fisher matrix.
    regularization : scalar_float, optional
        Diagonal regularization added before inversion. Default: 1e-6

    Returns
    -------
    covariance : Float[Array, "P P"]
        Regularized covariance estimate.

    Notes
    -----
    1. Add a small positive diagonal term.
    2. Use the Moore-Penrose inverse for rank-deficient problems.
    3. Symmetrize the result to remove numerical skew.
    """
    n_params: int = fisher_information.shape[0]
    identity: Float[Array, "P P"] = jnp.eye(
        n_params,
        dtype=fisher_information.dtype,
    )
    regularized: Float[Array, "P P"] = (
        fisher_information + regularization * identity
    )
    raw_covariance: Float[Array, "P P"] = jnp.linalg.pinv(regularized)
    covariance: Float[Array, "P P"] = 0.5 * (raw_covariance + raw_covariance.T)
    return covariance


@jaxtyped(typechecker=beartype)
def laplace_uncertainty(
    residual_fn: Callable[[Any], Any],
    params: Any,
    noise_variance: scalar_float = 1.0,
    regularization: scalar_float = 1e-6,
) -> LaplaceUncertainty:
    """Build a local Laplace uncertainty estimate from residual sensitivities.

    :see: :class:`~.test_uncertainty.TestReconUncertainty`

    Parameters
    ----------
    residual_fn : Callable[[Any], Any]
        Function mapping a parameter pytree to a residual pytree.
    params : Any
        Parameter pytree at which to estimate local uncertainty.
    noise_variance : scalar_float, optional
        Per-residual Gaussian noise variance. Default: 1.0
    regularization : scalar_float, optional
        Diagonal Fisher regularization before inversion. Default: 1e-6

    Returns
    -------
    uncertainty : LaplaceUncertainty
        Fisher matrix, covariance, one-sigma errors, and correlations.

    Notes
    -----
    1. Compute the local Fisher matrix from residual sensitivities.
    2. Invert it with diagonal regularization.
    3. Normalize covariance into a correlation matrix.
    """
    fisher_information: Float[Array, "P P"] = fisher_information_from_residual(
        residual_fn=residual_fn,
        params=params,
        noise_variance=noise_variance,
    )
    covariance: Float[Array, "P P"] = covariance_from_fisher(
        fisher_information=fisher_information,
        regularization=regularization,
    )
    variances: Float[Array, "P"] = jnp.maximum(jnp.diag(covariance), 0.0)
    standard_deviation: Float[Array, "P"] = jnp.sqrt(variances)
    denominator: Float[Array, "P P"] = jnp.maximum(
        standard_deviation[:, None] * standard_deviation[None, :],
        jnp.asarray(1e-12, dtype=covariance.dtype),
    )
    correlation: Float[Array, "P P"] = covariance / denominator
    uncertainty: LaplaceUncertainty = LaplaceUncertainty(
        fisher_information=fisher_information,
        covariance=covariance,
        standard_deviation=standard_deviation,
        correlation=correlation,
    )
    return uncertainty


@jaxtyped(typechecker=beartype)
def laplace_inverse_mass_matrix(
    uncertainty: LaplaceUncertainty,
    diagonal: bool = True,
    regularization: scalar_float = 1e-6,
) -> Float[Array, "..."]:
    """Build a blackjax inverse-mass warm start from Laplace covariance.

    :see: :class:`~.test_uncertainty.TestReconPosteriorUncertainty`

    Parameters
    ----------
    uncertainty : LaplaceUncertainty
        Local Laplace summary whose covariance approximates the posterior
        covariance.
    diagonal : bool, optional
        If True, return only the regularized covariance diagonal. If False,
        return the full regularized covariance matrix. Default: True
    regularization : scalar_float, optional
        Positive diagonal stabilization. Default: 1e-6

    Returns
    -------
    inverse_mass_matrix : Float[Array, "..."]
        Diagonal vector or dense matrix suitable for ``blackjax.nuts``.

    Notes
    -----
    BlackJAX's ``inverse_mass_matrix`` is the posterior covariance: window
    adaptation sets it to the Welford covariance of warmup positions.
    Therefore this helper returns the Laplace covariance, not the Fisher
    precision.
    """
    raw_covariance: Float[Array, "P P"] = uncertainty.covariance
    covariance: Float[Array, "P P"] = 0.5 * (raw_covariance + raw_covariance.T)
    n_params: int = covariance.shape[0]
    regularized_covariance: Float[Array, "P P"] = covariance + (
        regularization * jnp.eye(n_params, dtype=covariance.dtype)
    )
    if diagonal:
        return jnp.diag(regularized_covariance)
    inverse_mass_matrix: Float[Array, "P P"] = 0.5 * (
        regularized_covariance + regularized_covariance.T
    )
    return inverse_mass_matrix


@jaxtyped(typechecker=beartype)
def posterior_from_samples(
    samples: Float[Array, "C S P"],
    log_probability: Optional[Float[Array, "C S"]] = None,
    acceptance_rate: Optional[Float[Array, "C"]] = None,
    credibility: scalar_float = 0.95,
    r_hat_threshold: scalar_float = 1.1,
    min_effective_sample_size: scalar_float = 50.0,
    unravel_fn: Callable[[Float[Array, "P"]], Any] = _identity,
) -> PosteriorSamples:
    """Summarize posterior samples with diagnostics and credible intervals.

    :see: :class:`~.test_uncertainty.TestReconPosteriorUncertainty`

    Parameters
    ----------
    samples : Float[Array, "C S P"]
        Flattened samples with chain, draw, and parameter axes.
    log_probability : Optional[Float[Array, "C S"]], optional
        Log-posterior values. Default: zeros.
    acceptance_rate : Optional[Float[Array, "C"]], optional
        Mean chain acceptance rates. Default: zeros.
    credibility : scalar_float, optional
        Equal-tailed credible interval mass. Default: 0.95
    r_hat_threshold : scalar_float, optional
        Maximum allowed R-hat for ``converged``. For fewer than four draws,
        split R-hat is reported as NaN and convergence gates on ESS alone.
        Default: 1.1
    min_effective_sample_size : scalar_float, optional
        Minimum ESS for ``converged``. Default: 50.0
    unravel_fn : Callable[[Float[Array, "P"]], Any], optional
        Static map from flattened sample to original parameter pytree.

    Returns
    -------
    posterior : PosteriorSamples
        Sample summary with posterior moments, intervals, and diagnostics.
    """
    sample_array: Float[Array, "C S P"] = jnp.asarray(
        samples,
        dtype=jnp.float64,
    )
    if sample_array.ndim != 3:
        raise ValueError("samples must have shape (chains, draws, params)")
    n_chains: int = sample_array.shape[0]
    n_draws: int = sample_array.shape[1]
    n_params: int = sample_array.shape[2]
    if n_chains <= 0 or n_draws < 2 or n_params <= 0:
        raise ValueError(
            "samples must contain chains, at least 2 draws, and params"
        )

    log_probability_array: Float[Array, "C S"]
    if log_probability is None:
        log_probability_array = jnp.zeros(
            (n_chains, n_draws),
            dtype=sample_array.dtype,
        )
    else:
        log_probability_array = jnp.asarray(
            log_probability,
            dtype=sample_array.dtype,
        )

    acceptance_rate_array: Float[Array, "C"]
    if acceptance_rate is None:
        acceptance_rate_array = jnp.zeros(
            (n_chains,),
            dtype=sample_array.dtype,
        )
    else:
        acceptance_rate_array = jnp.asarray(
            acceptance_rate,
            dtype=sample_array.dtype,
        )

    flat_samples: Float[Array, "N P"] = jnp.reshape(
        sample_array,
        (-1, n_params),
    )
    mean: Float[Array, "P"] = jnp.mean(flat_samples, axis=0)
    covariance: Float[Array, "P P"] = _sample_covariance(flat_samples)
    tail_probability: Float[Array, ""] = (
        1.0 - jnp.asarray(credibility, dtype=sample_array.dtype)
    ) / 2.0
    quantiles: Float[Array, "2"] = jnp.array(
        [tail_probability, 1.0 - tail_probability],
        dtype=sample_array.dtype,
    )
    credible_interval: Float[Array, "2 P"] = jnp.quantile(
        flat_samples,
        quantiles,
        axis=0,
    )
    r_hat: Float[Array, "P"] = _split_r_hat(sample_array)
    effective_sample_size: Float[Array, "P"] = _effective_sample_size(
        sample_array
    )
    r_hat_converged: Bool[Array, ""] = jnp.logical_or(
        n_draws < 4,
        jnp.all(r_hat <= r_hat_threshold),
    )
    converged: Bool[Array, ""] = jnp.logical_and(
        r_hat_converged,
        jnp.all(effective_sample_size >= min_effective_sample_size),
    )
    posterior: PosteriorSamples = PosteriorSamples(
        samples=sample_array,
        log_probability=log_probability_array,
        acceptance_rate=acceptance_rate_array,
        mean=mean,
        covariance=covariance,
        credible_interval=credible_interval,
        r_hat=r_hat,
        effective_sample_size=effective_sample_size,
        converged=converged,
        unravel_fn=unravel_fn,
    )
    return posterior


@jaxtyped(typechecker=beartype)
def sample_posterior(  # noqa: PLR0913
    log_probability_fn: Callable[[Any], Float[Array, ""]],
    initial_position: Any,
    key: Array,
    num_samples: int = 256,
    num_warmup: int = 256,
    step_size: scalar_float = 0.1,
    inverse_mass_matrix: Optional[Float[Array, "..."]] = None,
    adapt: bool = True,
    target_acceptance_rate: scalar_float = 0.8,
    credibility: scalar_float = 0.95,
    r_hat_threshold: scalar_float = 1.1,
    min_effective_sample_size: scalar_float = 50.0,
) -> PosteriorSamples:
    """Draw blackjax NUTS samples from a differentiable log posterior.

    :see: :class:`~.test_uncertainty.TestReconPosteriorUncertainty`

    Parameters
    ----------
    log_probability_fn : Callable[[Any], Float[Array, ""]]
        Differentiable log posterior on the original parameter pytree.
    initial_position : Any
        Initial pytree, or a two-dimensional ``(chains, params)`` array of
        flattened multistart initial positions.
    key : Array
        JAX PRNG key.
    num_samples : int, optional
        Retained NUTS draws per chain. Default: 256
    num_warmup : int, optional
        Warmup/adaptation draws per chain. Default: 256
    step_size : scalar_float, optional
        Initial NUTS step size. Default: 0.1
    inverse_mass_matrix : Optional[Float[Array, "..."]], optional
        Optional Laplace/Fisher inverse-mass warm start. Default: identity.
    adapt : bool, optional
        If True, run blackjax window adaptation before sampling. Default: True
    target_acceptance_rate : scalar_float, optional
        Target acceptance rate for adaptation. Default: 0.8
    credibility : scalar_float, optional
        Equal-tailed credible interval mass. Default: 0.95
    r_hat_threshold : scalar_float, optional
        Maximum allowed R-hat for ``converged``. Default: 1.1
    min_effective_sample_size : scalar_float, optional
        Minimum ESS for ``converged``. Default: 50.0

    Returns
    -------
    posterior : PosteriorSamples
        Flattened posterior draws and diagnostics.

    Notes
    -----
    1. Flatten arbitrary parameter pytrees so blackjax samples a vector space.
    2. Treat rows of a two-dimensional initial array as independent multistart
       chains, preserving multimodal posterior structure.
    3. Warm-start NUTS from the supplied inverse mass matrix, or adapt from the
       initial step size when requested.
    """
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than one")
    if num_warmup < 0:
        raise ValueError("num_warmup must be non-negative")

    chain_positions: Float[Array, "C P"]
    unravel_fn: Callable[[Float[Array, "P"]], Any]
    chain_positions, unravel_fn = _as_chain_initial_positions(initial_position)
    n_chains: int = chain_positions.shape[0]
    n_params: int = chain_positions.shape[1]
    initial_inverse_mass: Float[Array, "..."]
    if inverse_mass_matrix is None:
        initial_inverse_mass = jnp.ones(n_params, dtype=chain_positions.dtype)
    else:
        initial_inverse_mass = jnp.asarray(
            inverse_mass_matrix,
            dtype=chain_positions.dtype,
        )

    def flat_log_probability(
        flat_position: Float[Array, "P"],
    ) -> Float[Array, ""]:
        params: Any = unravel_fn(flat_position)
        log_probability: Float[Array, ""] = log_probability_fn(params)
        return log_probability

    chain_keys: Array = jax.random.split(key, n_chains)
    samples: list[Float[Array, "S P"]] = []
    log_probabilities: list[Float[Array, "S"]] = []
    acceptance_rates: list[Float[Array, ""]] = []
    for chain_index in range(n_chains):
        chain_samples: Float[Array, "S P"]
        chain_log_probability: Float[Array, "S"]
        chain_acceptance_rate: Float[Array, ""]
        (
            chain_samples,
            chain_log_probability,
            chain_acceptance_rate,
        ) = _run_nuts_chain(
            flat_log_probability=flat_log_probability,
            start_position=chain_positions[chain_index],
            chain_key=chain_keys[chain_index],
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            inverse_mass_matrix=initial_inverse_mass,
            adapt=adapt,
            target_acceptance_rate=target_acceptance_rate,
        )
        samples.append(chain_samples)
        log_probabilities.append(chain_log_probability)
        acceptance_rates.append(chain_acceptance_rate)

    sample_array: Float[Array, "C S P"] = jnp.stack(samples, axis=0)
    log_probability_array: Float[Array, "C S"] = jnp.stack(
        log_probabilities,
        axis=0,
    )
    acceptance_rate_array: Float[Array, "C"] = jnp.stack(
        acceptance_rates,
        axis=0,
    )
    posterior: PosteriorSamples = posterior_from_samples(
        samples=sample_array,
        log_probability=log_probability_array,
        acceptance_rate=acceptance_rate_array,
        credibility=credibility,
        r_hat_threshold=r_hat_threshold,
        min_effective_sample_size=min_effective_sample_size,
        unravel_fn=unravel_fn,
    )
    return posterior


__all__: list[str] = [
    "covariance_from_fisher",
    "fisher_information_from_residual",
    "laplace_inverse_mass_matrix",
    "laplace_uncertainty",
    "posterior_from_samples",
    "sample_posterior",
]
