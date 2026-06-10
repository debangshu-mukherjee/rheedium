r"""Differentiable preprocessing of experimental RHEED images.

Extended Summary
----------------
This module provides functions for transforming raw experimental RHEED
detector images into a form suitable for comparison with simulation
output. Every operation is differentiable via ``jax.grad``, enabling
gradient-based optimization of physical parameters against experimental
data.

Key design constraint: hard boolean masks (``jnp.where`` with boolean
conditions) break gradient flow. All masking operations use soft
sigmoid-based masks instead:

.. math::

    m(d) = \sigma\bigl(s \cdot (d - t)\bigr)

where :math:`s` is the sharpness, :math:`d` is the distance field,
and :math:`t` is the threshold.

Routine Listings
----------------
:func:`soft_threshold_mask`
    Create a differentiable soft mask from a distance field.
:func:`subtract_background`
    Subtract dark frame background with non-negative clipping.
:func:`log_intensity_transform`
    Differentiable log transform for dynamic range compression.
:func:`normalize_image`
    Normalize image intensities to [0, 1] range.
:func:`preprocess_experimental`
    Full differentiable preprocessing pipeline for experimental images.

Notes
-----
All functions are compatible with ``jax.jit``, ``jax.vmap``, and
``jax.grad``. The ``preprocess_experimental`` pipeline is the
recommended entry point for preparing experimental data before
computing a loss against simulated patterns.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def soft_threshold_mask(
    distance_field: Float[Array, "H W"],
    threshold: scalar_float,
    sharpness: scalar_float = 10.0,
) -> Float[Array, "H W"]:
    r"""Create a differentiable soft mask from a distance field.

    Extended Summary
    ----------------
    Replaces hard boolean masks (which have zero gradient everywhere)
    with a smooth sigmoid transition. The mask smoothly varies from 0
    to 1 around the threshold:

    .. math::

        m(d) = \sigma\bigl(s \cdot (d - t)\bigr)
             = \frac{1}{1 + \exp\!\bigl(-s(d - t)\bigr)}

    As sharpness :math:`s \to \infty`, this converges to a hard step
    function, but remains differentiable for any finite :math:`s`.

    Parameters
    ----------
    distance_field : Float[Array, "H W"]
        Per-pixel distance or reliability metric. Pixels with values
        above the threshold are kept (mask near 1), below are
        suppressed (mask near 0).
    threshold : scalar_float
        Transition point for the soft mask.
    sharpness : scalar_float, optional
        Controls the steepness of the sigmoid transition. Higher
        values produce a sharper edge. Default: 10.0

    Returns
    -------
    mask : Float[Array, "H W"]
        Soft mask with values in (0, 1).

    Notes
    -----
    1. **Compute argument** --
       :math:`z = s \cdot (d - t)`.
    2. **Apply sigmoid** --
       :math:`m = 1 / (1 + \exp(-z))`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> dist = jnp.linspace(0.0, 1.0, 64).reshape(8, 8)
    >>> mask = rh.procs.soft_threshold_mask(
    ...     dist,
    ...     jnp.float64(0.5),
    ...     jnp.float64(20.0),
    ... )
    >>> mask.shape
    (8, 8)
    """
    argument: Float[Array, "H W"] = sharpness * (distance_field - threshold)
    mask: Float[Array, "H W"] = jax.nn.sigmoid(argument)
    return mask


@jaxtyped(typechecker=beartype)
def subtract_background(
    image: Float[Array, "H W"],
    background: Float[Array, "H W"],
) -> Float[Array, "H W"]:
    """Subtract dark frame background with non-negative clipping.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Raw detector image.
    background : Float[Array, "H W"]
        Background frame to subtract (e.g., dark frame, median
        background). Must have the same shape as ``image``.

    Returns
    -------
    subtracted : Float[Array, "H W"]
        Background-subtracted image clipped to non-negative values.

    Notes
    -----
    1. **Subtract** --
       Compute ``image - background``.
    2. **Clip** --
       ``jnp.maximum(0, result)`` to prevent negative intensities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> img = jnp.ones((8, 8)) * 100.0
    >>> bg = jnp.ones((8, 8)) * 30.0
    >>> sub = rh.procs.subtract_background(img, bg)
    >>> float(sub[0, 0])
    70.0
    """
    subtracted: Float[Array, "H W"] = jnp.maximum(image - background, 0.0)
    return subtracted


@jaxtyped(typechecker=beartype)
def log_intensity_transform(
    image: Float[Array, "H W"],
    epsilon: scalar_float = 1e-6,
) -> Float[Array, "H W"]:
    r"""Apply differentiable log transform for dynamic range compression.

    Extended Summary
    ----------------
    RHEED images often span several orders of magnitude in intensity.
    The log transform compresses this range for more stable loss
    computation:

    .. math::

        I_{\text{log}} = \log(1 + I / \epsilon)

    The regularization parameter :math:`\epsilon` prevents divergence
    at zero intensity and controls the transition between linear
    (small :math:`I`) and logarithmic (large :math:`I`) regimes.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Input image with non-negative intensities.
    epsilon : scalar_float, optional
        Regularization parameter. Smaller values give stronger
        compression. Default: 1e-6

    Returns
    -------
    transformed : Float[Array, "H W"]
        Log-transformed image.

    Notes
    -----
    1. **Compute ratio** --
       :math:`r = I / \epsilon`.
    2. **Log transform** --
       :math:`\log(1 + r)`, which is differentiable everywhere.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> img = jnp.array([[0.0, 1.0], [100.0, 10000.0]])
    >>> transformed = rh.procs.log_intensity_transform(img)
    >>> transformed.shape
    (2, 2)
    """
    transformed: Float[Array, "H W"] = jnp.log1p(image / epsilon)
    return transformed


@jaxtyped(typechecker=beartype)
def normalize_image(
    image: Float[Array, "H W"],
) -> Float[Array, "H W"]:
    r"""Normalize image intensities to [0, 1] range.

    Extended Summary
    ----------------
    Linear rescaling to unit range. Uses a small floor on the
    denominator to avoid division by zero for uniform images, while
    preserving differentiability.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Input image with arbitrary non-negative intensities.

    Returns
    -------
    normalized : Float[Array, "H W"]
        Image with intensities in [0, 1].

    Notes
    -----
    1. **Find range** --
       :math:`I_{\min} = \min(I)`, :math:`I_{\max} = \max(I)`.
    2. **Rescale** --
       :math:`I_{\text{norm}} = (I - I_{\min}) /
       \max(I_{\max} - I_{\min},\; 10^{-12})`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> img = jnp.array([[0.0, 50.0], [100.0, 200.0]])
    >>> normed = rh.procs.normalize_image(img)
    >>> float(normed.min()), float(normed.max())
    (0.0, 1.0)
    """
    i_min: scalar_float = jnp.min(image)
    i_max: scalar_float = jnp.max(image)
    denominator: scalar_float = jnp.maximum(i_max - i_min, 1e-12)
    normalized: Float[Array, "H W"] = (image - i_min) / denominator
    return normalized


@jaxtyped(typechecker=beartype)
def preprocess_experimental(
    raw_image: Float[Array, "H W"],
    background: Optional[Float[Array, "H W"]] = None,
    beam_shadow_mask: Optional[Float[Array, "H W"]] = None,
    log_scale: bool = False,
    log_epsilon: scalar_float = 1e-6,
) -> Float[Array, "H W"]:
    r"""Full differentiable preprocessing pipeline for experimental images.

    Extended Summary
    ----------------
    Transforms a raw experimental RHEED detector image into a form
    directly comparable with simulation output. Every step is
    differentiable, enabling ``jax.grad`` to flow through the entire
    preprocessing chain back to physical parameters.

    The pipeline applies: background subtraction, soft masking,
    optional log-scale compression, and normalization to [0, 1].

    Parameters
    ----------
    raw_image : Float[Array, "H W"]
        Raw detector image.
    background : Float[Array, "H W"], optional
        Background to subtract (e.g., dark frame). If ``None``, no
        subtraction is performed.
    beam_shadow_mask : Float[Array, "H W"], optional
        Soft mask with values in [0, 1] weighting pixels by
        reliability. Use :func:`soft_threshold_mask` to generate
        from a distance field. Must be a soft (sigmoid-based) mask,
        not a hard boolean mask, to preserve gradient flow. If
        ``None``, uniform weight is applied.
    log_scale : bool, optional
        If ``True``, apply :func:`log_intensity_transform` for
        dynamic range compression. Default: ``False``
    log_epsilon : scalar_float, optional
        Regularization for log transform. Only used when
        ``log_scale=True``. Default: 1e-6

    Returns
    -------
    processed : Float[Array, "H W"]
        Preprocessed image normalized to [0, 1], ready for loss
        computation against simulated patterns.

    Notes
    -----
    1. **Background subtraction** --
       If ``background`` is provided, subtract and clip to
       non-negative via :func:`subtract_background`.
    2. **Soft masking** --
       If ``beam_shadow_mask`` is provided, multiply element-wise
       to down-weight unreliable pixels.
    3. **Log transform** --
       If ``log_scale`` is ``True``, apply
       :math:`\log(1 + I / \epsilon)` via
       :func:`log_intensity_transform`.
    4. **Normalize** --
       Rescale to [0, 1] via :func:`normalize_image`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> raw = jnp.ones((64, 64)) * 500.0
    >>> bg = jnp.ones((64, 64)) * 100.0
    >>> processed = rh.procs.preprocess_experimental(
    ...     raw,
    ...     background=bg,
    ...     log_scale=True,
    ... )
    >>> processed.shape
    (64, 64)

    See Also
    --------
    soft_threshold_mask : Generate soft masks for beam shadow regions.
    """
    image: Float[Array, "H W"] = raw_image
    if background is not None:
        image = subtract_background(image, background)
    else:
        image = jnp.maximum(image, 0.0)
    if beam_shadow_mask is not None:
        image = image * beam_shadow_mask
    if log_scale:
        image = log_intensity_transform(image, log_epsilon)
    processed: Float[Array, "H W"] = normalize_image(image)
    return processed


__all__: list[str] = [
    "log_intensity_transform",
    "normalize_image",
    "preprocess_experimental",
    "soft_threshold_mask",
    "subtract_background",
]
