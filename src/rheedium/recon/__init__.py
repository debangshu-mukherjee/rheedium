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
:class:`DistributionAxisSpec`
    Static perturbation-axis contract for distribution reconstruction.
:class:`ReconProblem`
    Differentiable inverse problem definition for reconstruction solvers.
:class:`ReconResult`
    Result container returned by the general reconstruction solver.
:class:`LaplaceUncertainty`
    Local Gaussian uncertainty estimate around a reconstruction optimum.
:class:`RecipeDeviationReport`
    Compare fitted reconstruction parameters with an intended recipe.
:func:`positive_from_unconstrained`
    Map unconstrained values to strictly positive physical values.
:func:`unconstrained_from_positive`
    Map strictly positive physical values back to unconstrained coordinates.
:func:`bounded_from_unconstrained`
    Map unconstrained values into a closed finite interval.
:func:`unconstrained_from_bounded`
    Map bounded physical values back to unconstrained coordinates.
:func:`fractional_from_unconstrained`
    Map unconstrained values to unit-cell fractional coordinates.
:func:`unconstrained_from_fractional`
    Map fractional coordinates back to unconstrained coordinates.
:func:`lattice_from_unconstrained`
    Map unconstrained lengths and angles to lattice parameters.
:func:`unconstrained_from_lattice`
    Map lattice parameters back to unconstrained coordinates.
:func:`wyckoff_fractional_from_unconstrained`
    Map independent unconstrained coordinates to constrained Wyckoff positions.
:func:`simplex_from_unconstrained`
    Map unconstrained logits onto a probability simplex.
:func:`unconstrained_from_simplex`
    Map simplex weights to centered unconstrained logits.
:func:`ordered_bounded_from_unconstrained`
    Map unconstrained values to ordered points inside a finite interval.
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
:func:`checked_weighted_image_residual`
    Checkify-instrumented weighted residual for numerical validation.
:func:`checked_weighted_mean_squared_error`
    Checkify-instrumented weighted MSE for numerical validation.
:func:`create_distribution_axis_spec`
    Create a perturbation-axis specification for library reconstruction.
:func:`create_crystal_displacement_axis_spec`
    Create a crystal displacement-axis specification for library
    reconstruction.
:func:`solve`
    Solve a reconstruction problem with optimistix or optax.
:func:`multistart`
    Run a reconstruction problem from multiple initial guesses.
:func:`build_incoherent_intensity_library`
    Build a per-sample incoherent intensity library from a base object.
:func:`reconstruct_incoherent_weights`
    Recover incoherent distribution weights from an intensity library.
:func:`reconstruct_distribution`
    Recover a distribution over a perturbation axis from a measured image.
:func:`fisher_information_from_residual`
    Compute a Gauss-Newton/Fisher matrix from a residual function.
:func:`covariance_from_fisher`
    Regularize and invert a Fisher information matrix.
:func:`laplace_uncertainty`
    Build a local Laplace uncertainty estimate from residual sensitivities.
:func:`recipe_deviation`
    Solve an inverse problem and report signed recipe deviations.
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
"""

from .deviation import RecipeDeviationReport, recipe_deviation
from .losses import (
    affine_intensity_marginalization,
    affine_marginalized_residual,
    checked_weighted_image_residual,
    checked_weighted_mean_squared_error,
    entropy_prior,
    huber_image_loss,
    l2_image_loss,
    log_intensity_loss,
    normalized_cross_correlation_loss,
    smoothness_prior,
    sparsity_prior,
    weighted_image_residual,
    weighted_mean_squared_error,
)
from .orientation import (
    OrientationFitResult,
    compute_fisher_information,
    estimate_weight_uncertainty,
    fit_orientation_weights,
    orientation_loss,
)
from .solve import (
    DistributionAxisSpec,
    ReconProblem,
    ReconResult,
    build_incoherent_intensity_library,
    create_crystal_displacement_axis_spec,
    create_distribution_axis_spec,
    multistart,
    reconstruct_distribution,
    reconstruct_incoherent_weights,
    solve,
)
from .transforms import (
    bounded_from_unconstrained,
    fractional_from_unconstrained,
    lattice_from_unconstrained,
    ordered_bounded_from_unconstrained,
    positive_from_unconstrained,
    simplex_from_unconstrained,
    unconstrained_from_bounded,
    unconstrained_from_fractional,
    unconstrained_from_lattice,
    unconstrained_from_positive,
    unconstrained_from_simplex,
    wyckoff_fractional_from_unconstrained,
)
from .uncertainty import (
    LaplaceUncertainty,
    covariance_from_fisher,
    fisher_information_from_residual,
    laplace_uncertainty,
)

__all__: list[str] = [
    "DistributionAxisSpec",
    "LaplaceUncertainty",
    "ReconProblem",
    "ReconResult",
    "OrientationFitResult",
    "RecipeDeviationReport",
    "affine_intensity_marginalization",
    "affine_marginalized_residual",
    "bounded_from_unconstrained",
    "build_incoherent_intensity_library",
    "checked_weighted_image_residual",
    "checked_weighted_mean_squared_error",
    "compute_fisher_information",
    "covariance_from_fisher",
    "create_crystal_displacement_axis_spec",
    "create_distribution_axis_spec",
    "entropy_prior",
    "estimate_weight_uncertainty",
    "fisher_information_from_residual",
    "fit_orientation_weights",
    "fractional_from_unconstrained",
    "huber_image_loss",
    "lattice_from_unconstrained",
    "l2_image_loss",
    "laplace_uncertainty",
    "log_intensity_loss",
    "multistart",
    "normalized_cross_correlation_loss",
    "ordered_bounded_from_unconstrained",
    "orientation_loss",
    "positive_from_unconstrained",
    "recipe_deviation",
    "reconstruct_distribution",
    "reconstruct_incoherent_weights",
    "simplex_from_unconstrained",
    "smoothness_prior",
    "solve",
    "sparsity_prior",
    "unconstrained_from_bounded",
    "unconstrained_from_fractional",
    "unconstrained_from_lattice",
    "unconstrained_from_positive",
    "unconstrained_from_simplex",
    "weighted_image_residual",
    "weighted_mean_squared_error",
    "wyckoff_fractional_from_unconstrained",
]
