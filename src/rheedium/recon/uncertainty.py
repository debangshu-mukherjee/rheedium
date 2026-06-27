r"""Fisher and Laplace uncertainty helpers for reconstruction problems.

Extended Summary
----------------
This module generalizes the orientation-specific Fisher information workflow
to arbitrary latent pytrees. Residual functions are flattened with JAX's pytree
utilities, differentiated with ``jax.jacrev``, and converted into
Gauss-Newton/Fisher matrices for local covariance estimates.

Routine Listings
----------------
:class:`LaplaceUncertainty`
    Local Gaussian uncertainty estimate around a reconstruction optimum.
:func:`fisher_information_from_residual`
    Compute a Gauss-Newton/Fisher matrix from a residual function.
:func:`covariance_from_fisher`
    Regularize and invert a Fisher information matrix.
:func:`laplace_uncertainty`
    Build a local Laplace uncertainty estimate from residual sensitivities.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float


class LaplaceUncertainty(eqx.Module):
    """Local Gaussian uncertainty estimate around a reconstruction optimum.

    :see: :class:`~.test_uncertainty.TestReconUncertainty`

    Attributes
    ----------
    fisher_information : Float[Array, "P P"]
        Gauss-Newton/Fisher information matrix in flattened coordinates.
    covariance : Float[Array, "P P"]
        Regularized inverse Fisher matrix.
    standard_deviation : Float[Array, "P"]
        One-sigma uncertainty for each flattened parameter.
    correlation : Float[Array, "P P"]
        Parameter correlation matrix derived from covariance.
    """

    fisher_information: Float[Array, "P P"]
    covariance: Float[Array, "P P"]
    standard_deviation: Float[Array, "P"]
    correlation: Float[Array, "P P"]


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


__all__: list[str] = [
    "LaplaceUncertainty",
    "covariance_from_fisher",
    "fisher_information_from_residual",
    "laplace_uncertainty",
]
