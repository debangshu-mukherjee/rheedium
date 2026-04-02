"""Reconstruction and analysis utilities for RHEED data.

Extended Summary
----------------
This module provides tools for reconstruction algorithms and analysis
of RHEED patterns, including differentiable preprocessing of
experimental images and inverse problem solving.

Routine Listings
----------------
:func:`log_intensity_transform`
    Differentiable log transform for dynamic range compression.
:func:`normalize_image`
    Normalize image intensities to [0, 1] range.
:func:`preprocess_experimental`
    Full differentiable preprocessing pipeline for experimental images.
:func:`soft_threshold_mask`
    Create a differentiable soft mask from a distance field.
:func:`subtract_background`
    Subtract dark frame background with non-negative clipping.

Notes
-----
All preprocessing functions are differentiable via ``jax.grad``,
enabling gradient-based optimization of physical parameters against
experimental data.
"""

from .preprocessing import (
    log_intensity_transform,
    normalize_image,
    preprocess_experimental,
    soft_threshold_mask,
    subtract_background,
)

__all__: list[str] = [
    "log_intensity_transform",
    "normalize_image",
    "preprocess_experimental",
    "soft_threshold_mask",
    "subtract_background",
]
