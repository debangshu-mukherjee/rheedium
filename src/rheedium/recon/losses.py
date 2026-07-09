r"""Loss functions and residual builders for inverse RHEED problems.

Extended Summary
----------------
This module provides differentiable helpers for comparing simulated
detector images against experimental images. The functions are written
to stay lightweight and composable so forward models from
:mod:`rheedium.simul` and :mod:`rheedium.procs` can be optimized
through the same reconstruction routines.

Routine Listings
----------------
:func:`weighted_image_residual`
    Build a weighted least-squares residual field between two images.
:func:`weighted_mean_squared_error`
    Compute a normalized weighted mean-squared error.
:func:`l2_image_loss`
    Compute a weighted L2 image loss.
:func:`huber_image_loss`
    Compute a weighted Huber image loss.
:func:`log_intensity_loss`
    Compute a weighted log-intensity image loss.
:func:`normalized_cross_correlation_loss`
    Compute a scale-invariant normalized cross-correlation loss.
:func:`affine_intensity_marginalization`
    Find the optimal affine intensity calibration analytically.
:func:`affine_marginalized_residual`
    Build a residual after analytic scale/background marginalization.
:func:`entropy_prior`
    Compute a maximum-entropy prior for simplex weights.
:func:`smoothness_prior`
    Compute a nearest-neighbor smoothness prior.
:func:`sparsity_prior`
    Compute an L1 sparsity prior.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax.experimental import checkify
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def weighted_image_residual(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
) -> Float[Array, "H W"]:
    r"""Build a weighted least-squares residual field.

    :see: :class:`~.test_losses.TestWeightedLosses`

    Extended Summary
    ----------------
    The returned residual is designed for Gauss-Newton and other
    least-squares solvers. When a ``weight_map`` is provided it is
    interpreted as a non-negative per-pixel reliability weight:

    .. math::

        r_{ij} = \sqrt{w_{ij}}\,(I^{\text{sim}}_{ij} - I^{\text{exp}}_{ij})

    This convention ensures that minimizing :math:`\sum r_{ij}^2`
    matches the standard weighted least-squares objective.

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Values of zero exclude pixels from
        the fit. Negative entries are clipped to zero.

    Returns
    -------
    residual : Float[Array, "H W"]
        Weighted residual image with the same shape as the inputs.
    """
    residual: Float[Array, "H W"] = simulated_image - experimental_image
    if weight_map is None:
        return residual
    clipped_weight_map: Float[Array, "H W"] = jnp.maximum(weight_map, 0.0)
    return jnp.sqrt(clipped_weight_map) * residual


@jaxtyped(typechecker=beartype)
def weighted_mean_squared_error(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
) -> scalar_float:
    r"""Compute a normalized weighted mean-squared error.

    :see: :class:`~.test_losses.TestWeightedLosses`

    Extended Summary
    ----------------
    Without weights this reduces to the ordinary mean-squared error.
    With weights it computes:

    .. math::

        \mathrm{WMSE} =
        \frac{\sum_{ij} w_{ij}(I^{\text{sim}}_{ij} - I^{\text{exp}}_{ij})^2}
             {\max\left(\sum_{ij} w_{ij}, 10^{-12}\right)}

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Negative entries are clipped to
        zero before normalization.

    Returns
    -------
    loss : scalar_float
        Weighted mean-squared error.
    """
    squared_error: Float[Array, "H W"] = (
        simulated_image - experimental_image
    ) ** 2
    if weight_map is None:
        return jnp.mean(squared_error)
    clipped_weight_map: Float[Array, "H W"] = jnp.maximum(weight_map, 0.0)
    normalization: scalar_float = jnp.maximum(
        jnp.sum(clipped_weight_map), 1e-12
    )
    return jnp.sum(clipped_weight_map * squared_error) / normalization


@jaxtyped(typechecker=beartype)
def l2_image_loss(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
) -> scalar_float:
    """Compute a weighted L2 image loss.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.

    Returns
    -------
    loss : scalar_float
        Normalized weighted L2 loss.
    """
    loss: scalar_float = weighted_mean_squared_error(
        simulated_image=simulated_image,
        experimental_image=experimental_image,
        weight_map=weight_map,
    )
    return loss


@jaxtyped(typechecker=beartype)
def huber_image_loss(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    delta: scalar_float = 1.0,
) -> scalar_float:
    """Compute a weighted Huber image loss.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.
    delta : scalar_float, optional
        Huber transition scale. Default: 1.0

    Returns
    -------
    loss : scalar_float
        Normalized weighted Huber loss.

    Notes
    -----
    1. Use a quadratic penalty for residuals below ``delta``.
    2. Use a linear penalty for larger residuals.
    3. Normalize by the active weight sum.
    """
    residual: Float[Array, "H W"] = simulated_image - experimental_image
    abs_residual: Float[Array, "H W"] = jnp.abs(residual)
    quadratic: Float[Array, "H W"] = jnp.minimum(abs_residual, delta)
    linear: Float[Array, "H W"] = abs_residual - quadratic
    huber: Float[Array, "H W"] = 0.5 * quadratic**2 + delta * linear
    if weight_map is None:
        loss: scalar_float = jnp.mean(huber)
        return loss
    clipped_weight_map: Float[Array, "H W"] = jnp.maximum(weight_map, 0.0)
    normalization: scalar_float = jnp.maximum(
        jnp.sum(clipped_weight_map),
        1e-12,
    )
    loss: scalar_float = jnp.sum(clipped_weight_map * huber) / normalization
    return loss


@jaxtyped(typechecker=beartype)
def log_intensity_loss(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    epsilon: scalar_float = 1e-9,
) -> scalar_float:
    """Compute a weighted log-intensity image loss.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.
    epsilon : scalar_float, optional
        Positive floor before taking logarithms. Default: 1e-9

    Returns
    -------
    loss : scalar_float
        Weighted mean-squared log-intensity residual.

    Notes
    -----
    1. Clip intensities to a non-negative floor.
    2. Compare logarithms so peak and background regions share influence.
    """
    simulated_log: Float[Array, "H W"] = jnp.log(
        jnp.maximum(simulated_image, 0.0) + epsilon
    )
    experimental_log: Float[Array, "H W"] = jnp.log(
        jnp.maximum(experimental_image, 0.0) + epsilon
    )
    loss: scalar_float = weighted_mean_squared_error(
        simulated_image=simulated_log,
        experimental_image=experimental_log,
        weight_map=weight_map,
    )
    return loss


@jaxtyped(typechecker=beartype)
def normalized_cross_correlation_loss(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    epsilon: scalar_float = 1e-12,
) -> scalar_float:
    """Compute a scale-invariant normalized cross-correlation loss.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.
    epsilon : scalar_float, optional
        Denominator stabilizer. Default: 1e-12

    Returns
    -------
    loss : scalar_float
        ``1 - NCC``; zero is a perfect affine-intensity match.
    """
    scaled_epsilon: scalar_float = (
        epsilon
        * 0.5
        * (jnp.mean(simulated_image**2) + jnp.mean(experimental_image**2))
        + 1e-30
    )
    weights: Float[Array, "H W"]
    if weight_map is None:
        weights = jnp.ones_like(simulated_image)
    else:
        weights = jnp.maximum(weight_map, 0.0)
    normalization: scalar_float = jnp.maximum(
        jnp.sum(weights),
        scaled_epsilon,
    )
    simulated_mean: scalar_float = (
        jnp.sum(weights * simulated_image) / normalization
    )
    experimental_mean: scalar_float = (
        jnp.sum(weights * experimental_image) / normalization
    )
    simulated_centered: Float[Array, "H W"] = simulated_image - simulated_mean
    experimental_centered: Float[Array, "H W"] = (
        experimental_image - experimental_mean
    )
    numerator: scalar_float = jnp.sum(
        weights * simulated_centered * experimental_centered
    )
    simulated_norm: scalar_float = jnp.sum(weights * simulated_centered**2)
    experimental_norm: scalar_float = jnp.sum(
        weights * experimental_centered**2
    )
    denominator: scalar_float = jnp.sqrt(
        simulated_norm * experimental_norm + scaled_epsilon
    )
    correlation: scalar_float = numerator / denominator
    loss: scalar_float = 1.0 - correlation
    return loss


@jaxtyped(typechecker=beartype)
def affine_intensity_marginalization(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    fit_background: bool = True,
    ridge: scalar_float = 1e-12,
) -> Tuple[scalar_float, scalar_float]:
    """Find the optimal affine intensity calibration analytically.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image before calibration.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.
    fit_background : bool, optional
        Whether to fit an additive background (**static**). Default: True
    ridge : scalar_float, optional
        Diagonal regularization for the tiny normal system. Default: 1e-12

    Returns
    -------
    scale : scalar_float
        Optimal multiplicative scale.
    background : scalar_float
        Optimal additive background. Zero when ``fit_background`` is False.

    Notes
    -----
    1. Build the weighted least-squares normal equations for
       ``experimental ~= scale * simulated + background``.
    2. Solve the one- or two-parameter system analytically.
    """
    weights: Float[Array, "H W"]
    if weight_map is None:
        weights = jnp.ones_like(simulated_image)
    else:
        weights = jnp.maximum(weight_map, 0.0)
    weighted_sum: scalar_float = jnp.sum(weights)
    xx: scalar_float = jnp.sum(weights * simulated_image * simulated_image)
    xy: scalar_float = jnp.sum(weights * simulated_image * experimental_image)
    if not fit_background:
        scale: scalar_float = xy / jnp.maximum(xx + ridge, ridge)
        background: scalar_float = jnp.asarray(0.0, dtype=scale.dtype)
        return scale, background

    x_sum: scalar_float = jnp.sum(weights * simulated_image)
    y_sum: scalar_float = jnp.sum(weights * experimental_image)
    normal_matrix: Float[Array, "2 2"] = jnp.array(
        [[xx, x_sum], [x_sum, weighted_sum]],
        dtype=jnp.result_type(simulated_image, experimental_image),
    )
    rhs: Float[Array, "2"] = jnp.array(
        [xy, y_sum],
        dtype=normal_matrix.dtype,
    )
    identity: Float[Array, "2 2"] = jnp.eye(2, dtype=normal_matrix.dtype)
    solution: Float[Array, "2"] = jnp.linalg.solve(
        normal_matrix + ridge * identity,
        rhs,
    )
    scale = solution[0]
    background = solution[1]
    return scale, background


@jaxtyped(typechecker=beartype)
def affine_marginalized_residual(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
    fit_background: bool = True,
) -> Float[Array, "H W"]:
    """Build a residual after analytic scale/background marginalization.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    simulated_image : Float[Array, "H W"]
        Simulated detector image before calibration.
    experimental_image : Float[Array, "H W"]
        Experimental target image.
    weight_map : Float[Array, "H W"], optional
        Non-negative pixel weights. Default: all pixels.
    fit_background : bool, optional
        Whether to fit an additive background (**static**). Default: True

    Returns
    -------
    residual : Float[Array, "H W"]
        Weighted residual after optimal affine calibration.
    """
    scale: scalar_float
    background: scalar_float
    scale, background = affine_intensity_marginalization(
        simulated_image=simulated_image,
        experimental_image=experimental_image,
        weight_map=weight_map,
        fit_background=fit_background,
    )
    calibrated: Float[Array, "H W"] = scale * simulated_image + background
    residual: Float[Array, "H W"] = weighted_image_residual(
        simulated_image=calibrated,
        experimental_image=experimental_image,
        weight_map=weight_map,
    )
    return residual


@jaxtyped(typechecker=beartype)
def entropy_prior(
    weights: Float[Array, "N"],
    strength: scalar_float = 1.0,
    epsilon: scalar_float = 1e-12,
) -> scalar_float:
    """Compute a maximum-entropy prior for simplex weights.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    weights : Float[Array, "N"]
        Non-negative probability weights. They are normalized internally.
    strength : scalar_float, optional
        Prior strength. Default: 1.0
    epsilon : scalar_float, optional
        Positive probability floor. Default: 1e-12

    Returns
    -------
    loss : scalar_float
        Negative entropy penalty, minimized by diffuse distributions.
    """
    clipped: Float[Array, "N"] = jnp.clip(weights, epsilon, None)
    normalized: Float[Array, "N"] = clipped / jnp.sum(clipped)
    loss: scalar_float = strength * jnp.sum(normalized * jnp.log(normalized))
    return loss


@jaxtyped(typechecker=beartype)
def smoothness_prior(
    values: Float[Array, "N"],
    strength: scalar_float = 1.0,
) -> scalar_float:
    """Compute a nearest-neighbor smoothness prior.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    values : Float[Array, "N"]
        One-dimensional latent profile or distribution weights.
    strength : scalar_float, optional
        Prior strength. Default: 1.0

    Returns
    -------
    loss : scalar_float
        Mean-squared first-difference penalty.
    """
    if values.shape[0] < 2:
        loss: scalar_float = jnp.asarray(0.0, dtype=values.dtype)
        return loss
    differences: Float[Array, "N_minus_1"] = jnp.diff(values)
    loss: scalar_float = strength * jnp.mean(differences**2)
    return loss


@jaxtyped(typechecker=beartype)
def sparsity_prior(
    values: Float[Array, "N"],
    strength: scalar_float = 1.0,
) -> scalar_float:
    """Compute an L1 sparsity prior.

    :see: :class:`~.test_losses.TestDifferentiableLosses`

    Parameters
    ----------
    values : Float[Array, "N"]
        Latent vector to regularize.
    strength : scalar_float, optional
        Prior strength. Default: 1.0

    Returns
    -------
    loss : scalar_float
        L1 penalty scaled by ``strength``.
    """
    loss: scalar_float = strength * jnp.sum(jnp.abs(values))
    return loss


# --------------------------------------------------------------------------
# Opt-in runtime-checked loss variants
# --------------------------------------------------------------------------
# These wrappers mirror the simulator checked entry points: they keep the raw
# losses as the differentiable default and provide checkify-instrumented
# variants for debugging or CI numerical validation.
_CHECKIFY_ERRORS = checkify.nan_checks | checkify.div_checks
checked_weighted_image_residual = checkify.checkify(
    weighted_image_residual,
    errors=_CHECKIFY_ERRORS,
)
checked_weighted_mean_squared_error = checkify.checkify(
    weighted_mean_squared_error,
    errors=_CHECKIFY_ERRORS,
)


__all__: list[str] = [
    "affine_intensity_marginalization",
    "affine_marginalized_residual",
    "checked_weighted_image_residual",
    "checked_weighted_mean_squared_error",
    "entropy_prior",
    "huber_image_loss",
    "l2_image_loss",
    "log_intensity_loss",
    "normalized_cross_correlation_loss",
    "smoothness_prior",
    "sparsity_prior",
    "weighted_image_residual",
    "weighted_mean_squared_error",
]
