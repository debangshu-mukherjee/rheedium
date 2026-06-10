"""Image-space audit metrics for RHEED benchmark comparisons.

Extended Summary
----------------
This module contains lightweight quantitative measurements for detector
images. The functions are designed to answer the basic realism
questions needed by the benchmark suite: do the dominant peaks land in
the right place, are the rod spacings preserved, are the streaks the
right width, and how similar is the full image overall.

Routine Listings
----------------
:func:`dominant_peak_positions`
    Extract the strongest peak positions from an image projection.
:func:`extract_streak_profile`
    Extract a vertical or horizontal streak profile near a peak.
:func:`normalized_cross_correlation`
    Measure image similarity on a normalized scale.
:func:`peak_centroid`
    Compute the centroid of the brightest image region.
:func:`peak_centroid_error_px`
    Compare bright-region centroids between two images.
:func:`rod_spacing_error_px`
    Compare peak-to-peak spacing between two patterns.
:func:`specular_offset_px`
    Measure displacement between two anchor positions in pixels.
:func:`streak_fwhm_px`
    Measure full width at half maximum from a 1-D profile.

Notes
-----
The metrics operate on pixel-space detector images and use simple,
robust heuristics rather than domain-specific fitting. That keeps them
fast enough for regression benchmarking while still being sensitive to
obvious realism failures.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..types import scalar_float


@jaxtyped(typechecker=beartype)
def peak_centroid(
    image: Float[Array, "H W"],
    threshold_fraction: scalar_float = 0.5,
) -> Float[Array, "2"]:
    """Compute the centroid of the brightest region in an image.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Detector image with non-negative intensities.
    threshold_fraction : scalar_float, optional
        Fraction of the maximum intensity used to define the bright
        region. Default: 0.5

    Returns
    -------
    centroid_px : Float[Array, "2"]
        ``[x_px, y_px]`` centroid of the thresholded bright region.
        Falls back to the image center if the input is empty.
    """
    image = jnp.asarray(image, dtype=jnp.float64)
    image_height: int = int(image.shape[0])
    image_width: int = int(image.shape[1])
    max_intensity: Float[Array, ""] = jnp.max(image)
    threshold: Float[Array, ""] = threshold_fraction * max_intensity
    mask: Float[Array, "H W"] = (image >= threshold).astype(jnp.float64)
    weights: Float[Array, "H W"] = image * mask
    total_weight: Float[Array, ""] = jnp.sum(weights)
    safe_total_weight: Float[Array, ""] = jnp.maximum(total_weight, 1e-12)
    x_coords: Float[Array, "W"] = jnp.arange(image_width, dtype=jnp.float64)
    y_coords: Float[Array, "H"] = jnp.arange(image_height, dtype=jnp.float64)
    x_grid: Float[Array, "H W"]
    y_grid: Float[Array, "H W"]
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords, indexing="xy")
    default_center: Float[Array, "2"] = jnp.asarray(
        [
            (image_width - 1) / 2.0,
            (image_height - 1) / 2.0,
        ],
        dtype=jnp.float64,
    )

    def _weighted_center() -> Float[Array, "2"]:
        x_center: Float[Array, ""] = (
            jnp.sum(weights * x_grid) / safe_total_weight
        )
        y_center: Float[Array, ""] = (
            jnp.sum(weights * y_grid) / safe_total_weight
        )
        return jnp.stack([x_center, y_center])

    return jnp.where(total_weight > 0.0, _weighted_center(), default_center)


@jaxtyped(typechecker=beartype)
def specular_offset_px(
    reference_position_px: Float[Array, "2"],
    simulated_position_px: Float[Array, "2"],
) -> scalar_float:
    """Measure displacement between two pixel-space anchor positions.

    Parameters
    ----------
    reference_position_px : Float[Array, "2"]
        Expected detector position, typically from a reference image.
    simulated_position_px : Float[Array, "2"]
        Simulated detector position for the same feature.

    Returns
    -------
    offset_px : scalar_float
        Euclidean separation in pixels.
    """
    delta: Float[Array, "2"] = simulated_position_px - reference_position_px
    return jnp.linalg.norm(delta)


@jaxtyped(typechecker=beartype)
def peak_centroid_error_px(
    reference_image: Float[Array, "H W"],
    simulated_image: Float[Array, "H W"],
    threshold_fraction: scalar_float = 0.5,
) -> scalar_float:
    """Compare bright-region centroids between two detector images.

    Parameters
    ----------
    reference_image : Float[Array, "H W"]
        Reference detector image.
    simulated_image : Float[Array, "H W"]
        Simulated detector image.
    threshold_fraction : scalar_float, optional
        Bright-region threshold passed to :func:`peak_centroid`.
        Default: 0.5

    Returns
    -------
    error_px : scalar_float
        Euclidean distance between the two bright-region centroids.
    """
    reference_centroid: Float[Array, "2"] = peak_centroid(
        reference_image, threshold_fraction=threshold_fraction
    )
    simulated_centroid: Float[Array, "2"] = peak_centroid(
        simulated_image, threshold_fraction=threshold_fraction
    )
    return specular_offset_px(reference_centroid, simulated_centroid)


@jaxtyped(typechecker=beartype)
def normalized_cross_correlation(
    pattern_a: Float[Array, "H W"],
    pattern_b: Float[Array, "H W"],
) -> scalar_float:
    """Compute normalized cross-correlation between two detector images.

    Parameters
    ----------
    pattern_a : Float[Array, "H W"]
        First detector image.
    pattern_b : Float[Array, "H W"]
        Second detector image.

    Returns
    -------
    ncc : scalar_float
        Similarity score in ``[-1, 1]``.
    """
    pattern_a = jnp.asarray(pattern_a, dtype=jnp.float64)
    pattern_b = jnp.asarray(pattern_b, dtype=jnp.float64)
    centered_a: Float[Array, "H W"] = pattern_a - jnp.mean(pattern_a)
    centered_b: Float[Array, "H W"] = pattern_b - jnp.mean(pattern_b)
    numerator: Float[Array, ""] = jnp.sum(centered_a * centered_b)
    denominator: Float[Array, ""] = jnp.sqrt(
        jnp.sum(centered_a**2) * jnp.sum(centered_b**2)
    )
    safe_denominator: Float[Array, ""] = jnp.maximum(denominator, 1e-12)
    ncc: Float[Array, ""] = numerator / safe_denominator
    return jnp.clip(ncc, -1.0, 1.0)


@jaxtyped(typechecker=beartype)
def dominant_peak_positions(
    image: Float[Array, "H W"],
    axis: str = "horizontal",
    n_peaks: int = 3,
    min_separation_px: int = 5,
) -> Float[Array, "N"]:
    """Extract dominant peak positions from a projected detector image.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Detector image.
    axis : str, optional
        Projection axis. ``"horizontal"`` detects peaks along image
        columns, while ``"vertical"`` detects peaks along rows.
        Default: ``"horizontal"``
    n_peaks : int, optional
        Number of peaks to return. Default: 3
    min_separation_px : int, optional
        Minimum exclusion half-width applied after selecting each peak.
        Default: 5

    Returns
    -------
    positions_px : Float[Array, "N"]
        Sorted peak positions in pixels.
    """
    image = jnp.asarray(image, dtype=jnp.float64)
    if axis == "horizontal":
        profile: Float[Array, "W"] = jnp.sum(image, axis=0)
    elif axis == "vertical":
        profile = jnp.sum(image, axis=1)
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'.")

    working_profile: Float[Array, "P"] = profile
    peak_positions: list[Float[Array, ""]] = []
    profile_length: int = int(profile.shape[0])

    for _ in range(n_peaks):
        peak_index: int = int(jnp.argmax(working_profile))
        peak_positions.append(jnp.asarray(peak_index, dtype=jnp.float64))
        left_idx: int = max(0, peak_index - min_separation_px)
        right_idx: int = min(
            profile_length, peak_index + min_separation_px + 1
        )
        working_profile = working_profile.at[left_idx:right_idx].set(-jnp.inf)

    return jnp.sort(jnp.stack(peak_positions))


@jaxtyped(typechecker=beartype)
def rod_spacing_error_px(
    reference_peak_positions_px: Float[Array, "N"],
    simulated_peak_positions_px: Float[Array, "N"],
) -> scalar_float:
    """Compare inter-peak spacing between two ordered peak sets.

    Parameters
    ----------
    reference_peak_positions_px : Float[Array, "N"]
        Reference peak positions in pixels.
    simulated_peak_positions_px : Float[Array, "N"]
        Simulated peak positions in pixels.

    Returns
    -------
    spacing_error_px : scalar_float
        Root-mean-square difference between adjacent peak spacings.
    """
    reference_spacings: Float[Array, "M"] = jnp.diff(
        jnp.sort(reference_peak_positions_px)
    )
    simulated_spacings: Float[Array, "M"] = jnp.diff(
        jnp.sort(simulated_peak_positions_px)
    )
    return jnp.sqrt(jnp.mean((reference_spacings - simulated_spacings) ** 2))


@jaxtyped(typechecker=beartype)
def extract_streak_profile(
    image: Float[Array, "H W"],
    center_px: Float[Array, "2"],
    axis: str = "vertical",
    band_half_width_px: int = 2,
) -> Float[Array, "P"]:
    """Extract a 1-D streak profile near a specified detector position.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Detector image.
    center_px : Float[Array, "2"]
        ``[x_px, y_px]`` anchor position for the profile.
    axis : str, optional
        Profile direction. ``"vertical"`` averages nearby columns and
        returns a row profile, while ``"horizontal"`` averages nearby
        rows and returns a column profile. Default: ``"vertical"``
    band_half_width_px : int, optional
        Half-width of the averaging band around the anchor.
        Default: 2

    Returns
    -------
    profile : Float[Array, "P"]
        Extracted 1-D intensity profile.
    """
    image = jnp.asarray(image, dtype=jnp.float64)
    image_height: int = int(image.shape[0])
    image_width: int = int(image.shape[1])
    center_x: int = int(jnp.rint(center_px[0]))
    center_y: int = int(jnp.rint(center_px[1]))

    if axis == "vertical":
        left_idx: int = max(0, center_x - band_half_width_px)
        right_idx: int = min(image_width, center_x + band_half_width_px + 1)
        return jnp.mean(image[:, left_idx:right_idx], axis=1)

    if axis == "horizontal":
        lower_idx: int = max(0, center_y - band_half_width_px)
        upper_idx: int = min(image_height, center_y + band_half_width_px + 1)
        return jnp.mean(image[lower_idx:upper_idx, :], axis=0)

    raise ValueError("axis must be 'horizontal' or 'vertical'.")


@jaxtyped(typechecker=beartype)
def streak_fwhm_px(
    profile: Float[Array, "N"],
) -> scalar_float:
    """Measure full width at half maximum of a 1-D intensity profile.

    Parameters
    ----------
    profile : Float[Array, "N"]
        One-dimensional streak profile.

    Returns
    -------
    fwhm_px : scalar_float
        Width in pixels above half the peak intensity. Returns ``0.0``
        for empty profiles.
    """
    profile = jnp.asarray(profile, dtype=jnp.float64)
    peak_value: Float[Array, ""] = jnp.max(profile)
    half_max: Float[Array, ""] = 0.5 * peak_value
    above_half_max: Float[Array, "N"] = (profile >= half_max).astype(
        jnp.float64
    )
    indices: Float[Array, "N"] = jnp.where(
        above_half_max > 0.0,
        jnp.arange(profile.shape[0], dtype=jnp.float64),
        jnp.nan,
    )
    valid_indices: Float[Array, "N"] = indices[~jnp.isnan(indices)]
    if valid_indices.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return valid_indices[-1] - valid_indices[0] + 1.0


__all__: list[str] = [
    "dominant_peak_positions",
    "extract_streak_profile",
    "normalized_cross_correlation",
    "peak_centroid",
    "peak_centroid_error_px",
    "rod_spacing_error_px",
    "specular_offset_px",
    "streak_fwhm_px",
]
