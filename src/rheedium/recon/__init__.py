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
:class:`OrientationFitResult`
    Result container for orientation-distribution fitting.
:func:`orientation_loss`
    Compute a masked image loss for an orientation distribution.
:func:`fit_orientation_weights`
    Recover discrete orientation weights on a fixed candidate support.
:func:`compute_fisher_information`
    Fisher information for fitted orientation-weight logits.
:func:`estimate_weight_uncertainty`
    Propagate Fisher information to 1σ weight uncertainties.
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
from .orientation import (
    OrientationFitResult,
    compute_fisher_information,
    estimate_weight_uncertainty,
    fit_orientation_weights,
    orientation_loss,
)

__all__: list[str] = [
    "ReconstructionResult",
    "OrientationFitResult",
    "adagrad_optimize",
    "adagrad_reconstruction",
    "adam_optimize",
    "adam_reconstruction",
    "compute_fisher_information",
    "estimate_weight_uncertainty",
    "fit_orientation_weights",
    "gauss_newton_least_squares",
    "gauss_newton_reconstruction",
    "orientation_loss",
    "weighted_image_residual",
    "weighted_mean_squared_error",
]
