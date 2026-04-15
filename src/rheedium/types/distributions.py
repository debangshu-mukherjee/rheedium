"""Probability distribution types for statistical RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible probability distribution types for
modeling surface statistics in RHEED. The large beam footprint (mm² at
grazing incidence) samples a statistical ensemble of domains - we specify
distributions rather than individual domains.

Routine Listings
----------------
:class:`OrientationDistribution`
    Probability distribution over domain azimuthal orientations.
:class:`SizeDistribution`
    Probability distribution over coherent domain sizes.
:func:`create_orientation_distribution`
    Canonical factory for orientation distributions.
:func:`create_discrete_orientation`
    Factory for discrete rotational variants (e.g., ±33.7° domains).
:func:`create_gaussian_orientation`
    Factory for continuous Gaussian mosaic spread.
:func:`create_mixed_orientation`
    Factory for discrete variants with mosaic broadening.
:func:`discretize_orientation`
    Convert OrientationDistribution to quadrature points and weights.
:func:`integrate_over_orientation`
    Compute incoherent intensity sum over orientation distribution.

Notes
-----
All distribution types are PyTrees supporting JAX transformations.
Integration uses Gauss-Hermite quadrature for continuous distributions
and exact summation for discrete variants.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Final, NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, jaxtyped

from rheedium.tools import gauss_hermite_nodes_weights

from .custom_types import float_jax_image, scalar_float, scalar_int

_ZERO_MOSAIC_FWHM_DEG: Final[float] = 1e-6


@jaxtyped(typechecker=beartype)
def _normalize_probability_weights(
    weights: Float[Array, "M"],
) -> Float[Array, "M"]:
    """Clip to a valid probability simplex with uniform fallback."""
    clipped_weights: Float[Array, "M"] = jnp.clip(
        jnp.asarray(weights, dtype=jnp.float64),
        0.0,
        None,
    )
    weight_sum: Float[Array, ""] = jnp.sum(clipped_weights)
    uniform_weights: Float[Array, "M"] = (
        jnp.ones_like(clipped_weights) / (clipped_weights.shape[0])
    )
    normalized_weights: Float[Array, "M"] = jax.lax.cond(
        weight_sum > 0.0,
        lambda: clipped_weights / weight_sum,
        lambda: uniform_weights,
    )
    return normalized_weights


@register_pytree_node_class
class OrientationDistribution(NamedTuple):
    r"""Probability distribution over domain azimuthal orientations.

    Extended Summary
    ----------------
    Models the statistical distribution of in-plane domain rotations
    on the illuminated surface. Supports discrete variants (e.g.,
    rotational twins), continuous mosaic spread, or combinations.

    The total intensity is computed as an incoherent sum:

    .. math::

        I(G) = \\int P(\\theta) \\, |F(G, \\theta)|^2 \\, d\\theta

    For discrete variants this becomes:

    .. math::

        I(G) = \\sum_i w_i \\, |F(G, \\theta_i)|^2

    Attributes
    ----------
    discrete_angles_deg : Float[Array, "M"]
        Azimuthal rotation angles for discrete variants in degrees.
        For continuous-only distributions, use a single-element array
        with the center angle.
    discrete_weights : Float[Array, "M"]
        Probability weights for each discrete angle. Normalized
        internally to sum to 1.0. Must be non-negative.
    mosaic_fwhm_deg : Float[Array, ""]
        Full-width at half-maximum of Gaussian mosaic spread around
        each discrete angle, in degrees. Set to 0.0 for sharp
        discrete variants with no mosaic broadening.
    distribution_id : Optional[str]
        Optional identifier for the distribution (e.g., "sqrt13_R33.7").

    Notes
    -----
    The distribution is parameterized to handle three common cases:

    1. **Discrete variants only** (mosaic_fwhm_deg = 0):
       Sharp peaks at specified angles. Example: √13×√13 R±33.7°
       reconstruction with two domains.

    2. **Continuous mosaic only** (single angle, mosaic_fwhm_deg > 0):
       Gaussian spread around a central orientation. Models strain
       relaxation or polycrystalline texture.

    3. **Mixed** (multiple angles, mosaic_fwhm_deg > 0):
       Each discrete variant is broadened by the mosaic spread.
       Most realistic for real surfaces.

    Examples
    --------
    >>> # Two rotational variants at ±33.7°
    >>> dist = OrientationDistribution(
    ...     discrete_angles_deg=jnp.array([33.7, -33.7]),
    ...     discrete_weights=jnp.array([0.5, 0.5]),
    ...     mosaic_fwhm_deg=jnp.array(0.0),
    ... )

    >>> # Gaussian mosaic spread of 0.5° FWHM
    >>> dist = OrientationDistribution(
    ...     discrete_angles_deg=jnp.array([0.0]),
    ...     discrete_weights=jnp.array([1.0]),
    ...     mosaic_fwhm_deg=jnp.array(0.5),
    ... )
    """

    discrete_angles_deg: Float[Array, "M"]
    discrete_weights: Float[Array, "M"]
    mosaic_fwhm_deg: Float[Array, ""]
    distribution_id: Optional[str] = None

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, "M"], Float[Array, "M"], Float[Array, ""]],
        Optional[str],
    ]:
        """Flatten for JAX PyTree."""
        return (
            (
                self.discrete_angles_deg,
                self.discrete_weights,
                self.mosaic_fwhm_deg,
            ),
            self.distribution_id,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Optional[str],
        children: Tuple[
            Float[Array, "M"], Float[Array, "M"], Float[Array, ""]
        ],
    ) -> "OrientationDistribution":
        """Unflatten from JAX PyTree."""
        return cls(
            discrete_angles_deg=children[0],
            discrete_weights=children[1],
            mosaic_fwhm_deg=children[2],
            distribution_id=aux_data,
        )


@register_pytree_node_class
class SizeDistribution(NamedTuple):
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

    distribution_type: str
    mean_ang: Float[Array, ""]
    sigma_ang: Float[Array, ""]
    min_size_ang: Float[Array, ""]
    max_size_ang: Float[Array, ""]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
        str,
    ]:
        """Flatten for JAX PyTree."""
        return (
            (
                self.mean_ang,
                self.sigma_ang,
                self.min_size_ang,
                self.max_size_ang,
            ),
            self.distribution_type,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: str,
        children: Tuple[
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ) -> "SizeDistribution":
        """Unflatten from JAX PyTree."""
        return cls(
            distribution_type=aux_data,
            mean_ang=children[0],
            sigma_ang=children[1],
            min_size_ang=children[2],
            max_size_ang=children[3],
        )


@jaxtyped(typechecker=beartype)
def create_orientation_distribution(
    angles_deg: Float[Array, "M"],
    weights: Optional[Float[Array, "M"]] = None,
    mosaic_fwhm_deg: scalar_float = 0.0,
    distribution_id: Optional[str] = None,
) -> OrientationDistribution:
    """Create an OrientationDistribution with normalized JAX arrays.

    Parameters
    ----------
    angles_deg : Float[Array, "M"]
        Rotation angles for each supported orientation in degrees.
    weights : Optional[Float[Array, "M"]], optional
        Probability weights for each angle. Default: equal weights.
    mosaic_fwhm_deg : scalar_float, optional
        Gaussian mosaic broadening FWHM in degrees. Negative values are
        clamped to 0.0. Default: 0.0
    distribution_id : Optional[str], optional
        Identifier for the distribution.

    Returns
    -------
    dist : OrientationDistribution
        Orientation distribution with normalized weights and a
        non-negative mosaic width.
    """
    angles_arr: Float[Array, "M"] = jnp.atleast_1d(
        jnp.asarray(angles_deg, dtype=jnp.float64)
    )
    n_angles: int = angles_arr.shape[0]
    weights_arr: Float[Array, "M"]
    if weights is None:
        weights_arr = jnp.ones(n_angles, dtype=jnp.float64) / n_angles
    else:
        weights_arr = _normalize_probability_weights(weights)
    mosaic_fwhm_arr: Float[Array, ""] = jnp.maximum(
        jnp.asarray(mosaic_fwhm_deg, dtype=jnp.float64),
        0.0,
    )
    return OrientationDistribution(
        discrete_angles_deg=angles_arr,
        discrete_weights=weights_arr,
        mosaic_fwhm_deg=mosaic_fwhm_arr,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_discrete_orientation(
    angles_deg: Float[Array, "M"],
    weights: Optional[Float[Array, "M"]] = None,
    distribution_id: Optional[str] = None,
) -> OrientationDistribution:
    """Create discrete orientation distribution for rotational variants.

    Parameters
    ----------
    angles_deg : Float[Array, "M"]
        Rotation angles for each variant in degrees.
    weights : Optional[Float[Array, "M"]], optional
        Probability weights. Default: equal weights (1/M each).
    distribution_id : Optional[str], optional
        Identifier for the distribution.

    Returns
    -------
    dist : OrientationDistribution
        Discrete orientation distribution with no mosaic spread.

    Examples
    --------
    >>> # √13×√13 R±33.7° reconstruction
    >>> dist = create_discrete_orientation(
    ...     angles_deg=jnp.array([33.7, -33.7]),
    ...     weights=jnp.array([0.5, 0.5]),
    ...     distribution_id="sqrt13_R33.7",
    ... )

    >>> # 4-fold symmetric variants
    >>> dist = create_discrete_orientation(
    ...     angles_deg=jnp.array([0.0, 90.0, 180.0, 270.0]),
    ... )
    """
    return create_orientation_distribution(
        angles_deg=angles_deg,
        weights=weights,
        mosaic_fwhm_deg=0.0,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_gaussian_orientation(
    center_deg: scalar_float = 0.0,
    fwhm_deg: scalar_float = 0.5,
    distribution_id: Optional[str] = None,
) -> OrientationDistribution:
    """Create Gaussian mosaic spread orientation distribution.

    Parameters
    ----------
    center_deg : scalar_float, optional
        Center of the distribution in degrees. Default: 0.0
    fwhm_deg : scalar_float, optional
        Full-width at half-maximum in degrees. Default: 0.5
    distribution_id : Optional[str], optional
        Identifier for the distribution.

    Returns
    -------
    dist : OrientationDistribution
        Continuous Gaussian orientation distribution.

    Notes
    -----
    FWHM relates to Gaussian σ by: FWHM = 2√(2 ln 2) × σ ≈ 2.355 σ
    """
    center_arr: Float[Array, "1"] = jnp.atleast_1d(
        jnp.asarray(center_deg, dtype=jnp.float64)
    )
    return create_orientation_distribution(
        angles_deg=center_arr,
        weights=None,
        mosaic_fwhm_deg=fwhm_deg,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_mixed_orientation(
    angles_deg: Float[Array, "M"],
    weights: Optional[Float[Array, "M"]] = None,
    mosaic_fwhm_deg: scalar_float = 0.2,
    distribution_id: Optional[str] = None,
) -> OrientationDistribution:
    """Create mixed distribution with discrete variants and mosaic spread.

    Parameters
    ----------
    angles_deg : Float[Array, "M"]
        Rotation angles for discrete variants in degrees.
    weights : Optional[Float[Array, "M"]], optional
        Probability weights for variants. Default: equal weights.
    mosaic_fwhm_deg : scalar_float, optional
        Mosaic FWHM around each variant in degrees. Default: 0.2
    distribution_id : Optional[str], optional
        Identifier for the distribution.

    Returns
    -------
    dist : OrientationDistribution
        Mixed discrete + continuous orientation distribution.

    Notes
    -----
    Each discrete variant peak is broadened by a Gaussian with the
    specified FWHM. This is the most realistic model for real surfaces.
    """
    return create_orientation_distribution(
        angles_deg=angles_deg,
        weights=weights,
        mosaic_fwhm_deg=mosaic_fwhm_deg,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def _fwhm_to_sigma(fwhm: Float[Array, ""]) -> Float[Array, ""]:
    """Convert FWHM to Gaussian sigma."""
    fwhm_to_sigma_factor: Float[Array, ""] = jnp.array(
        1.0 / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0))), dtype=jnp.float64
    )
    return fwhm * fwhm_to_sigma_factor


@jaxtyped(typechecker=beartype)
def discretize_orientation(
    dist: OrientationDistribution,
    n_mosaic_points: scalar_int = 7,
    n_sigma_range: scalar_float = 3.0,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Convert OrientationDistribution to quadrature points and weights.

    Description
    -----------
    Discretizes the orientation probability distribution into a set of
    angle samples and corresponding integration weights. Uses Gauss-Hermite
    quadrature around each discrete peak, with the spread controlled by
    mosaic_fwhm_deg.

    Parameters
    ----------
    dist : OrientationDistribution
        Orientation probability distribution.
    n_mosaic_points : scalar_int, optional
        Number of Gauss-Hermite quadrature points per discrete peak
        for mosaic integration. Default: 7
    n_sigma_range : scalar_float, optional
        Number of sigma to extend mosaic sampling. Default: 3.0

    Returns
    -------
    angles_deg : Float[Array, "N"]
        Quadrature angle samples in degrees. Shape: M × n_mosaic_points
    weights : Float[Array, "N"]
        Integration weights (sum to 1.0).

    Notes
    -----
    When mosaic_fwhm_deg is very small (< 1e-6), the quadrature points
    collapse onto the discrete peaks, exactly reproducing delta-function
    behavior in the numerical quadrature.

    The total number of output points is always M × n_mosaic_points.
    """
    del n_sigma_range
    sigma_deg: Float[Array, ""] = _fwhm_to_sigma(dist.mosaic_fwhm_deg)
    sigma_effective: Float[Array, ""] = jnp.where(
        dist.mosaic_fwhm_deg < _ZERO_MOSAIC_FWHM_DEG,
        0.0,
        sigma_deg,
    )

    nodes: Float[Array, "Q"]
    quad_weights: Float[Array, "Q"]
    nodes, quad_weights = gauss_hermite_nodes_weights(n_mosaic_points)
    discrete_weights: Float[Array, "M"] = _normalize_probability_weights(
        dist.discrete_weights
    )
    sqrt2: Float[Array, ""] = jnp.sqrt(jnp.array(2.0, dtype=jnp.float64))
    angle_offsets: Float[Array, "Q"] = sqrt2 * sigma_effective * nodes
    sqrt_pi: Float[Array, ""] = jnp.sqrt(jnp.array(jnp.pi, dtype=jnp.float64))

    def _process_peak(
        carry: None,
        peak_data: Tuple[Float[Array, ""], Float[Array, ""]],
    ) -> Tuple[None, Tuple[Float[Array, "Q"], Float[Array, "Q"]]]:
        del carry
        center: Float[Array, ""] = peak_data[0]
        peak_weight: Float[Array, ""] = peak_data[1]
        peak_angles: Float[Array, "Q"] = center + angle_offsets
        combined_weights: Float[Array, "Q"] = (
            peak_weight * quad_weights / sqrt_pi
        )
        return None, (peak_angles, combined_weights)

    _, (angles_stack, weights_stack) = jax.lax.scan(
        _process_peak,
        None,
        (dist.discrete_angles_deg, discrete_weights),
    )
    all_angles: Float[Array, "M Q"] = angles_stack
    all_weights: Float[Array, "M Q"] = weights_stack
    flat_angles: Float[Array, "N"] = all_angles.ravel()
    flat_weights: Float[Array, "N"] = all_weights.ravel()
    weight_sum: Float[Array, ""] = jnp.sum(flat_weights)
    normalized_weights: Float[Array, "N"] = flat_weights / weight_sum
    return flat_angles, normalized_weights


@jaxtyped(typechecker=beartype)
def discretize_orientation_static(
    dist: OrientationDistribution,
    n_mosaic_points: scalar_int = 7,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Static-dispatch version for use outside JIT when efficiency matters.

    Description
    -----------
    When the distribution type is known at Python level (not traced),
    this version uses Python branching for efficiency: discrete-only
    distributions return M points instead of M × n_mosaic_points.

    Parameters
    ----------
    dist : OrientationDistribution
        Orientation probability distribution.
    n_mosaic_points : scalar_int, optional
        Quadrature points per peak for mosaic. Default: 7

    Returns
    -------
    angles_deg : Float[Array, "N"]
        Quadrature angle samples in degrees.
    weights : Float[Array, "N"]
        Integration weights (sum to 1.0).

    Notes
    -----
    Use this version when calling outside of JIT for efficiency.
    Use discretize_orientation inside JIT-compiled functions.
    """
    sigma_val: float = float(dist.mosaic_fwhm_deg)
    if sigma_val < _ZERO_MOSAIC_FWHM_DEG:
        normalized_weights: Float[Array, "N"] = _normalize_probability_weights(
            dist.discrete_weights
        )
        return dist.discrete_angles_deg, normalized_weights
    return discretize_orientation(dist, n_mosaic_points=n_mosaic_points)


@jaxtyped(typechecker=beartype)
def integrate_over_orientation(
    simulate_fn: Callable[[scalar_float], float_jax_image],
    orientation_dist: OrientationDistribution,
    n_mosaic_points: scalar_int = 7,
) -> float_jax_image:
    r"""Compute incoherent intensity sum over orientation distribution.

    Description
    -----------
    Integrates RHEED intensity over the orientation probability distribution
    using numerical quadrature. Each orientation sample is simulated
    independently, then intensities are summed with distribution weights.

    This implements the statistical ensemble averaging:

    .. math::

        I_{total}(G) = \\int P(\\theta) \\, I(G, \\theta) \\, d\\theta
                     \\approx \\sum_i w_i \\, I(G, \\theta_i)

    Parameters
    ----------
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Function mapping azimuthal angle (degrees) to RHEED intensity
        pattern. Must be vmappable. Signature: phi_deg → pattern.
    orientation_dist : OrientationDistribution
        Probability distribution over orientations.
    n_mosaic_points : scalar_int, optional
        Quadrature points for mosaic spread. Default: 7

    Returns
    -------
    averaged_pattern : Float[Array, "H W"]
        Incoherently averaged RHEED intensity pattern.

    Notes
    -----
    The simulate_fn should capture all other parameters (crystal structure,
    beam energy, incidence angle, etc.) via closure. Only the azimuthal
    angle varies during integration.

    For pure discrete distributions (no mosaic), this reduces to a
    weighted sum over the discrete variants.

    Examples
    --------
    >>> # Define simulation function (captures other params)
    >>> def sim_at_phi(phi_deg):
    ...     return simulate_rheed(crystal, theta=2.0, phi=phi_deg, ...)
    >>>
    >>> # Create distribution
    >>> dist = create_discrete_orientation(jnp.array([33.7, -33.7]))
    >>>
    >>> # Integrate
    >>> pattern = integrate_over_orientation(sim_at_phi, dist)
    """
    angles_deg: Float[Array, "N"]
    weights: Float[Array, "N"]
    angles_deg, weights = discretize_orientation(
        orientation_dist, n_mosaic_points=n_mosaic_points
    )
    patterns: Float[Array, "N H W"] = jax.vmap(simulate_fn)(angles_deg)
    weights_expanded: Float[Array, "N 1 1"] = weights[:, None, None]
    weighted_sum: float_jax_image = jnp.sum(
        weights_expanded * patterns, axis=0
    )
    return weighted_sum


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
    return SizeDistribution(
        distribution_type="lognormal",
        mean_ang=jnp.asarray(mean_ang, dtype=jnp.float64),
        sigma_ang=jnp.asarray(sigma_ang, dtype=jnp.float64),
        min_size_ang=jnp.asarray(min_size_ang, dtype=jnp.float64),
        max_size_ang=jnp.asarray(max_size_ang, dtype=jnp.float64),
    )


__all__: list[str] = [
    "OrientationDistribution",
    "SizeDistribution",
    "create_orientation_distribution",
    "create_discrete_orientation",
    "create_gaussian_orientation",
    "create_lognormal_size",
    "create_mixed_orientation",
    "discretize_orientation",
    "discretize_orientation_static",
    "integrate_over_orientation",
]
