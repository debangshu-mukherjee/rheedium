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
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def weighted_image_residual(
    simulated_image: Float[Array, "H W"],
    experimental_image: Float[Array, "H W"],
    weight_map: Optional[Float[Array, "H W"]] = None,
) -> Float[Array, "H W"]:
    r"""Build a weighted least-squares residual field.

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


__all__: list[str] = [
    "weighted_image_residual",
    "weighted_mean_squared_error",
]
