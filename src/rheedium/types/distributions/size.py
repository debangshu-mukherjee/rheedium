"""Domain-size distributions and generic size producers."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.tools import gauss_hermite_nodes_weights

from ..custom_types import scalar_float, scalar_int
from .base import (
    Distribution,
    ReductionMode,
    _normalize_probability_weights,
    create_distribution,
)


class SizeDistribution(eqx.Module):
    """Probability distribution over coherent domain sizes.

    Extended Summary
    ----------------
    Models the statistical distribution of lateral coherent domain
    sizes on the illuminated surface. Domain size determines rod
    broadening via σ_rod = 2π / (L × √(2π)).

    Attributes
    ----------
    distribution_type : str
        Type of distribution: "lognormal", "gaussian", "exponential",
        "delta". Lognormal is most physical for nucleation/coalescence.
    mean_ang : Float[Array, ""]
        Mean domain size in Ångstroms.
    sigma_ang : Float[Array, ""]
        Standard deviation in Ångstroms. For lognormal, this is the
        underlying normal distribution's σ.
    min_size_ang : Float[Array, ""]
        Minimum size cutoff in Ångstroms. Avoids unphysical small
        domains. Typical: 5-20 Å.
    max_size_ang : Float[Array, ""]
        Maximum size cutoff in Ångstroms. Computational truncation.
        Typical: 500-2000 Å.

    Notes
    -----
    The distribution affects RHEED patterns through rod broadening:
    smaller domains → broader rods → more diffuse streaks.

    For "delta" distribution, all domains have exactly mean_ang size.
    """

    distribution_type: str = eqx.field(static=True)
    mean_ang: Float[Array, ""]
    sigma_ang: Float[Array, ""]
    min_size_ang: Float[Array, ""]
    max_size_ang: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def discretize_size_distribution(
    dist: SizeDistribution,
    n_points: scalar_int = 7,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Convert SizeDistribution to quadrature sizes and weights.

    :see: :class:`~.test_distributions.TestSizeProducer`

    Parameters
    ----------
    dist : SizeDistribution
        Domain-size probability distribution.
    n_points : scalar_int, optional
        Number of Gauss-Hermite quadrature points for finite-width
        distributions. Default: 7.

    Returns
    -------
    sizes_ang : Float[Array, "N"]
        Domain-size samples in Angstroms.
    weights : Float[Array, "N"]
        Normalized non-negative probability weights.

    Notes
    -----
    1. Delta or zero-width distributions collapse to one mean-size sample.
    2. Lognormal distributions use moment-matched log-space quadrature.
    3. Exponential distributions use equal-probability bins on the truncated
       support, represented by each bin's inverse-CDF average.
    4. Gaussian weights are normalized after support clipping.
    """
    mean_ang: Float[Array, ""] = dist.mean_ang
    sigma_ang: Float[Array, ""] = dist.sigma_ang
    clipped_mean: Float[Array, ""] = jnp.clip(
        mean_ang,
        dist.min_size_ang,
        dist.max_size_ang,
    )
    if dist.distribution_type == "delta":
        sizes: Float[Array, "1"] = jnp.atleast_1d(clipped_mean)
        weights: Float[Array, "1"] = jnp.ones((1,), dtype=jnp.float64)
        return sizes, weights

    if dist.distribution_type == "exponential":
        mean: Float[Array, ""] = dist.mean_ang
        lower_cdf: Float[Array, ""] = 1.0 - jnp.exp(-dist.min_size_ang / mean)
        upper_cdf: Float[Array, ""] = 1.0 - jnp.exp(-dist.max_size_ang / mean)
        cdf_width: Float[Array, ""] = upper_cdf - lower_cdf
        bin_edges: Float[Array, "M"] = (
            jnp.arange(n_points + 1, dtype=jnp.float64) / n_points
        )
        quantile_edges: Float[Array, "M"] = lower_cdf + (bin_edges * cdf_width)
        survival_edges: Float[Array, "M"] = 1.0 - quantile_edges

        def antiderivative(y: Float[Array, "M"]) -> Float[Array, "M"]:
            return jnp.where(y > 0.0, -y * jnp.log(y) + y, 0.0)

        integral_edges: Float[Array, "M"] = antiderivative(survival_edges)
        bin_integrals: Float[Array, "N"] = (
            integral_edges[:-1] - integral_edges[1:]
        )
        sizes = mean * n_points * bin_integrals / cdf_width
        weights = jnp.full((n_points,), 1.0 / n_points, dtype=jnp.float64)
        return sizes, weights

    nodes: Float[Array, "N"]
    quad_weights: Float[Array, "N"]
    nodes, quad_weights = gauss_hermite_nodes_weights(n_points)
    sqrt2: Float[Array, ""] = jnp.sqrt(jnp.array(2.0, dtype=jnp.float64))
    if dist.distribution_type == "lognormal":
        variance_ratio: Float[Array, ""] = (sigma_ang / mean_ang) ** 2
        sigma_log: Float[Array, ""] = jnp.sqrt(jnp.log1p(variance_ratio))
        mu_log: Float[Array, ""] = jnp.log(mean_ang) - 0.5 * sigma_log**2
        raw_sizes: Float[Array, "N"] = jnp.exp(
            mu_log + sqrt2 * sigma_log * nodes
        )
    else:
        raw_sizes = mean_ang + sqrt2 * sigma_ang * nodes
    sizes = jnp.clip(raw_sizes, dist.min_size_ang, dist.max_size_ang)
    weights = _normalize_probability_weights(quad_weights)
    return sizes, weights


@jaxtyped(typechecker=beartype)
def size_to_distribution(
    dist: SizeDistribution,
    n_points: scalar_int = 7,
) -> Distribution:
    """Convert size samples to a generic incoherent Distribution.

    :see: :class:`~.test_distributions.TestSizeProducer`

    Parameters
    ----------
    dist : SizeDistribution
        Domain-size probability distribution.
    n_points : scalar_int, optional
        Quadrature point count for finite-width distributions. Default: 7.

    Returns
    -------
    distribution : Distribution
        Generic distribution with one size-in-Angstrom sample coordinate per
        row and incoherent reduction.

    Notes
    -----
    1. Discretize the size distribution.
    2. Store sizes as one-column latent samples.
    3. Use incoherent reduction for domain-size ensembles.
    """
    sizes_ang: Float[Array, "N"]
    weights: Float[Array, "N"]
    sizes_ang, weights = discretize_size_distribution(
        dist,
        n_points=n_points,
    )
    return create_distribution(
        samples=sizes_ang[:, None],
        weights=weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id="size",
    )


@jaxtyped(typechecker=beartype)
def create_lognormal_size(
    mean_ang: scalar_float = 100.0,
    sigma_ang: scalar_float = 30.0,
    min_size_ang: scalar_float = 10.0,
    max_size_ang: scalar_float = 500.0,
) -> SizeDistribution:
    """Create lognormal domain size distribution.

    Parameters
    ----------
    mean_ang : scalar_float, optional
        Mean domain size in Ångstroms. Default: 100.0
    sigma_ang : scalar_float, optional
        Standard deviation in Ångstroms. Default: 30.0
    min_size_ang : scalar_float, optional
        Minimum size cutoff. Default: 10.0 Å
    max_size_ang : scalar_float, optional
        Maximum size cutoff. Default: 500.0 Å

    Returns
    -------
    dist : SizeDistribution
        Lognormal size distribution.

    Notes
    -----
    Lognormal is most physical for domain sizes arising from
    nucleation and coalescence processes. The mode (peak) of
    the distribution is at exp(μ - σ²) where μ, σ are the
    underlying normal parameters.
    """
    mean_arr: Float[Array, ""] = jnp.asarray(mean_ang, dtype=jnp.float64)
    sigma_arr: Float[Array, ""] = jnp.asarray(sigma_ang, dtype=jnp.float64)
    min_size_arr: Float[Array, ""] = jnp.asarray(
        min_size_ang, dtype=jnp.float64
    )
    max_size_arr: Float[Array, ""] = jnp.asarray(
        max_size_ang, dtype=jnp.float64
    )
    checked_mean: Float[Array, ""] = eqx.error_if(
        mean_arr,
        ~jnp.isfinite(mean_arr),
        "mean_ang must be finite",
    )
    checked_mean = eqx.error_if(
        checked_mean,
        checked_mean <= 0.0,
        "mean_ang must be positive",
    )
    checked_sigma: Float[Array, ""] = eqx.error_if(
        sigma_arr,
        ~jnp.isfinite(sigma_arr),
        "sigma_ang must be finite",
    )
    checked_sigma = eqx.error_if(
        checked_sigma,
        checked_sigma < 0.0,
        "sigma_ang must be non-negative",
    )
    checked_min_size: Float[Array, ""] = eqx.error_if(
        min_size_arr,
        ~jnp.isfinite(min_size_arr),
        "min_size_ang must be finite",
    )
    checked_min_size = eqx.error_if(
        checked_min_size,
        checked_min_size <= 0.0,
        "min_size_ang must be positive",
    )
    checked_max_size: Float[Array, ""] = eqx.error_if(
        max_size_arr,
        ~jnp.isfinite(max_size_arr),
        "max_size_ang must be finite",
    )
    checked_max_size = eqx.error_if(
        checked_max_size,
        checked_max_size <= checked_min_size,
        "max_size_ang must be greater than min_size_ang",
    )
    return SizeDistribution(
        distribution_type="lognormal",
        mean_ang=checked_mean,
        sigma_ang=checked_sigma,
        min_size_ang=checked_min_size,
        max_size_ang=checked_max_size,
    )


__all__: list[str] = [
    "SizeDistribution",
    "create_lognormal_size",
    "discretize_size_distribution",
    "size_to_distribution",
]
