"""Differentiable inverse problems for RHEED data.

Extended Summary
----------------
This package is reserved for inverse-modeling code that infers
structure, defects, and processing parameters from experimental RHEED
data. Forward procedural models, preprocessing steps, and
differentiable surface-construction utilities live in
:mod:`rheedium.procs`.

Routine Listings
----------------
:class:`ReconstructionResult`
    Result container returned by reconstruction solvers.
:func:`weighted_image_residual`
    Build a weighted least-squares residual field between two images.
:func:`weighted_mean_squared_error`
    Compute a normalized weighted mean-squared error.
:func:`gauss_newton_least_squares`
    Gauss-Newton optimizer for arbitrary least-squares residuals.
:func:`adam_optimize`
    Adam optimizer for arbitrary scalar objectives.
:func:`adagrad_optimize`
    Adagrad optimizer for arbitrary scalar objectives.
:func:`gauss_newton_reconstruction`
    Reconstruct parameters by least-squares image matching.
:func:`adam_reconstruction`
    Reconstruct parameters by minimizing an image-matching loss with Adam.
:func:`adagrad_reconstruction`
    Reconstruct parameters by minimizing an image-matching loss with
    Adagrad.
"""

from .losses import (
    weighted_image_residual,
    weighted_mean_squared_error,
)
from .optimizers import (
    ReconstructionResult,
    adagrad_optimize,
    adagrad_reconstruction,
    adam_optimize,
    adam_reconstruction,
    gauss_newton_least_squares,
    gauss_newton_reconstruction,
)

__all__: list[str] = [
    "ReconstructionResult",
    "adagrad_optimize",
    "adagrad_reconstruction",
    "adam_optimize",
    "adam_reconstruction",
    "gauss_newton_least_squares",
    "gauss_newton_reconstruction",
    "weighted_image_residual",
    "weighted_mean_squared_error",
]
