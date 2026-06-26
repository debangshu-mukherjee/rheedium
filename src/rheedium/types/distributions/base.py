"""Base distribution contracts and reduction helpers."""

from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Final, Optional
from jaxtyping import Array, Float, jaxtyped

from ..custom_types import scalar_int


class ReductionMode(str, Enum):
    """Static ensemble reduction mode.

    :see: :class:`~.test_distributions.TestDistributionFactories`

    Attributes
    ----------
    COHERENT : str
        Sum weighted amplitudes before taking the modulus squared.
    INCOHERENT : str
        Sum weighted intensities after taking the modulus squared.
    """

    COHERENT = "coherent"
    INCOHERENT = "incoherent"


class Distribution(eqx.Module):
    """Generic weighted distribution over latent simulation samples.

    :see: :class:`~.test_distributions.TestDistributionFactories`

    Attributes
    ----------
    samples : Float[Array, "N D"]
        Sample coordinates for an ensemble axis. The first dimension indexes
        samples; the remaining flat coordinate dimension is interpreted by the
        closure passed to the simulation integrator.
    weights : Float[Array, "N"]
        Non-negative probability weights normalized to sum to one.
    reduction : ReductionMode
        Static coherent or incoherent reduction mode.
    axis_id : Optional[str]
        Optional static label for diagnostics and composition.
    """

    samples: Float[Array, "N D"]
    weights: Float[Array, "N"]
    reduction: ReductionMode = eqx.field(static=True)
    axis_id: Optional[str] = eqx.field(static=True, default=None)

    def bind(
        self,
        binder: Callable[["Distribution"], Callable[[Float[Array, "D"]], Any]],
    ) -> Callable[[Float[Array, "D"]], Any]:
        """Bind this axis through a kernel-specific producer binder.

        The base Distribution owns sample/weight/reduction metadata, while the
        supplied binder owns the kernel-specific interpretation of one sample
        row. This keeps the public contract polymorphic without forcing the
        pure type module to import simulator kernels.
        """
        return binder(self)


@jaxtyped(typechecker=beartype)
def create_distribution(
    samples: Float[Array, "N D"],
    weights: Float[Array, "M"],
    reduction: ReductionMode | str = ReductionMode.INCOHERENT,
    axis_id: Optional[str] = None,
) -> Distribution:
    """Create a generic Distribution with validated probability weights.

    :see: :class:`~.test_distributions.TestDistributionFactories`

    Parameters
    ----------
    samples : Float[Array, "N D"]
        Two-dimensional sample array with one row per ensemble sample.
    weights : Float[Array, "M"]
        Non-negative sample weights. Values are normalized to sum to one.
    reduction : ReductionMode | str, optional
        Ensemble reduction mode, ``"coherent"`` or ``"incoherent"``.
        Default: ``ReductionMode.INCOHERENT``.
    axis_id : Optional[str], optional
        Optional static identifier for this ensemble axis.

    Returns
    -------
    distribution : Distribution
        Validated generic distribution PyTree.

    Notes
    -----
    1. Convert samples and weights to ``float64`` JAX arrays.
    2. Validate static rank and matching leading dimensions.
    3. Validate finite samples and non-negative finite weights.
    4. Normalize weights onto the probability simplex.
    5. Store reduction and axis metadata as static PyTree fields.
    """
    samples_arr: Float[Array, "N D"] = jnp.asarray(samples, dtype=jnp.float64)
    weights_arr: Float[Array, "N"] = jnp.asarray(weights, dtype=jnp.float64)
    if samples_arr.ndim != 2:
        raise ValueError("samples must have shape (N, D)")
    if weights_arr.ndim != 1:
        raise ValueError("weights must have shape (N,)")
    if samples_arr.shape[0] <= 0:
        raise ValueError("samples must contain at least one row")
    if samples_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("samples and weights must share leading dimension")

    reduction_mode: ReductionMode = ReductionMode(reduction)
    checked_samples: Float[Array, "N D"] = eqx.error_if(
        samples_arr,
        jnp.any(~jnp.isfinite(samples_arr)),
        "samples must be finite",
    )
    checked_weights: Float[Array, "N"] = eqx.error_if(
        weights_arr,
        jnp.any(~jnp.isfinite(weights_arr)),
        "weights must be finite",
    )
    checked_weights = eqx.error_if(
        checked_weights,
        jnp.any(checked_weights < 0.0),
        "weights must be non-negative",
    )
    checked_weights = eqx.error_if(
        checked_weights,
        jnp.sum(checked_weights) <= 0.0,
        "weights must have positive total probability",
    )
    normalized_weights: Float[Array, "N"] = _normalize_probability_weights(
        checked_weights
    )
    return Distribution(
        samples=checked_samples,
        weights=normalized_weights,
        reduction=reduction_mode,
        axis_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def create_trivial_distribution(
    sample_dim: scalar_int = 1,
    reduction: ReductionMode | str = ReductionMode.INCOHERENT,
    axis_id: Optional[str] = "trivial",
) -> Distribution:
    """Create the one-sample identity distribution.

    :see: :class:`~.test_distributions.TestDistributionFactories`

    Parameters
    ----------
    sample_dim : scalar_int, optional
        Width of the zero sample vector. Default: 1.
    reduction : ReductionMode | str, optional
        Static reduction mode for the identity axis. Coherent and incoherent
        reductions coincide for one sample. Default: incoherent.
    axis_id : Optional[str], optional
        Optional static identifier. Default: ``"trivial"``.

    Returns
    -------
    distribution : Distribution
        One zero-valued sample with unit probability weight.

    Notes
    -----
    1. Create a zero sample vector of requested width.
    2. Assign unit probability weight.
    3. Delegate validation to :func:`create_distribution`.
    """
    sample_dim_int: int = int(sample_dim)
    if sample_dim_int <= 0:
        raise ValueError("sample_dim must be positive")
    samples: Float[Array, "1 D"] = jnp.zeros(
        (1, sample_dim_int), dtype=jnp.float64
    )
    weights: Float[Array, "1"] = jnp.ones((1,), dtype=jnp.float64)
    return create_distribution(
        samples=samples,
        weights=weights,
        reduction=reduction,
        axis_id=axis_id,
    )


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


TRIVIAL_DISTRIBUTION: Final[Distribution] = create_trivial_distribution()
TRIVIAL: Final[Distribution] = TRIVIAL_DISTRIBUTION


__all__: list[str] = [
    "Distribution",
    "ReductionMode",
    "TRIVIAL",
    "TRIVIAL_DISTRIBUTION",
    "create_distribution",
    "create_trivial_distribution",
]
