"""Type carriers and constructors for reconstruction workflows.

Routine Listings
----------------
:class:`DistributionAxisSpec`
    Static perturbation-axis contract for distribution reconstruction.
:class:`LaplaceUncertainty`
    Local Gaussian uncertainty estimate around a reconstruction optimum.
:class:`OrientationFitResult`
    Result container for orientation-distribution fitting.
:class:`PosteriorSamples`
    Posterior sample container with diagnostics and credible intervals.
:class:`RecipeDeviationReport`
    Compare fitted reconstruction parameters with an intended recipe.
:class:`ReconProblem`
    Differentiable inverse problem definition for reconstruction solvers.
:class:`ReconResult`
    Result container returned by the general reconstruction solver.
:func:`create_distribution_axis_spec`
    Create a perturbation-axis specification for library reconstruction.
:func:`create_crystal_displacement_axis_spec`
    Create a crystal displacement-axis specification for library
    reconstruction.
:obj:`RECIPE_DEVIATION_SCHEMA_VERSION`
    Frozen recipe-deviation payload schema identifier.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, NamedTuple, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from .crystal_types import (
    CrystalStructure,
    _build_canonical_cell_vectors,
    create_crystal_structure,
)
from .custom_types import float_jax_image
from .distributions import OrientationDistribution

RECIPE_DEVIATION_SCHEMA_VERSION: str = "recipe-deviation-report.v1"


def _identity(value: Any) -> Any:
    """Return a value unchanged."""
    return value


def _default_residual(simulated: Any, measured: Any) -> Any:
    """Return a default residual pytree."""
    residual: Any = jax.tree_util.tree_map(
        lambda sim_leaf, meas_leaf: jnp.asarray(sim_leaf)
        - jnp.asarray(meas_leaf),
        simulated,
        measured,
    )
    return residual


def _tree_mean_square(tree: Any) -> Float[Array, ""]:
    """Return the mean square across all numeric leaves in a pytree."""
    leaves: list[Any] = jax.tree_util.tree_leaves(tree)
    if not leaves:
        loss: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
        return loss
    total: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    count: Float[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    for leaf in leaves:
        leaf_array: Array = jnp.asarray(leaf)
        total = total + jnp.sum(jnp.square(leaf_array))
        count = count + jnp.asarray(leaf_array.size, dtype=jnp.float64)
    loss: Float[Array, ""] = total / jnp.maximum(count, 1.0)
    return loss


def _default_loss(simulated: Any, measured: Any) -> Float[Array, ""]:
    """Return the default mean-squared data loss."""
    residual: Any = _default_residual(simulated, measured)
    loss: Float[Array, ""] = _tree_mean_square(residual)
    return loss


class _OrientationWeightParameters(NamedTuple):
    """Internal unconstrained parameterization for orientation fitting."""

    weight_logits: Float[Array, "M"]
    mosaic_param: Float[Array, ""]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, "M"], Float[Array, ""]],
        None,
    ]:
        """Flatten for JAX PyTree support."""
        return ((self.weight_logits, self.mosaic_param), None)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[Float[Array, "M"], Float[Array, ""]],
    ) -> "_OrientationWeightParameters":
        """Unflatten from a JAX PyTree."""
        del aux_data
        return cls(*children)


class DistributionAxisSpec(eqx.Module):
    """Static perturbation-axis contract for distribution reconstruction.

    Attributes
    ----------
    samples : Float[Array, "N D"]
        Latent-axis sample coordinates used as rows in the recovered
        distribution.
    perturbation_fn : Callable[[Any, Float[Array, "D"]], Any]
        Static function that maps ``(base_object, sample)`` to a perturbed
        physical object.
    forward_model : Callable[[Any], Array]
        Static differentiable model that maps the perturbed object to either a
        coherent amplitude image or an intensity image.
    output_kind : str
        Static forward-output interpretation: ``"amplitude"`` or
        ``"intensity"``.
    axis_id : Optional[str]
        Optional static label attached to the recovered
        :class:`~rheedium.types.Distribution`.
    """

    samples: Float[Array, "N D"]
    perturbation_fn: Callable[[Any, Float[Array, "D"]], Any] = eqx.field(
        static=True
    )
    forward_model: Callable[[Any], Array] = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="amplitude")
    axis_id: Optional[str] = eqx.field(static=True, default=None)


class ReconProblem(eqx.Module):
    """Differentiable inverse problem definition for reconstruction solvers.

    Attributes
    ----------
    forward : Callable[[Any], Any]
        Differentiable forward model accepting physical parameters.
    measured : Any
        Measured target data pytree.
    transform : Callable[[Any], Any]
        Static map from unconstrained latent parameters to physical
        parameters.
    residual_fn : Callable[[Any, Any], Any]
        Static residual builder comparing simulated and measured data.
    loss_fn : Callable[[Any, Any], Float[Array, ""]]
        Static scalar loss for minimisation modes.
    """

    forward: Callable[[Any], Any] = eqx.field(static=True)
    measured: Any
    transform: Callable[[Any], Any] = eqx.field(
        static=True,
        default=_identity,
    )
    residual_fn: Callable[[Any, Any], Any] = eqx.field(
        static=True,
        default=_default_residual,
    )
    loss_fn: Callable[[Any, Any], Float[Array, ""]] = eqx.field(
        static=True,
        default=_default_loss,
    )

    def physical_from_latent(self, latent_params: Any) -> Any:
        """Map latent optimizer coordinates to physical parameters."""
        physical_params: Any = self.transform(latent_params)
        return physical_params

    def simulate(self, latent_params: Any) -> Any:
        """Evaluate the forward model from latent optimizer coordinates."""
        physical_params: Any = self.physical_from_latent(latent_params)
        simulated: Any = self.forward(physical_params)
        return simulated

    def residual_from_latent(self, latent_params: Any) -> Any:
        """Evaluate residuals from latent optimizer coordinates."""
        simulated: Any = self.simulate(latent_params)
        residual: Any = self.residual_fn(simulated, self.measured)
        return residual

    def loss_from_latent(self, latent_params: Any) -> Float[Array, ""]:
        """Evaluate scalar loss from latent optimizer coordinates."""
        simulated: Any = self.simulate(latent_params)
        loss: Float[Array, ""] = self.loss_fn(simulated, self.measured)
        return loss


class ReconResult(eqx.Module):
    """Result container returned by the general reconstruction solver.

    Attributes
    ----------
    params : Any
        Final physical parameter pytree.
    latent_params : Any
        Final unconstrained optimizer-coordinate pytree.
    simulated : Any
        Forward-model output at the final parameters.
    residual : Any
        Residual pytree at the final parameters.
    loss : Float[Array, ""]
        Final scalar data loss.
    iterations : Int[Array, ""]
        Number of nonlinear solver steps reported by ``optimistix``.
    converged : Bool[Array, ""]
        True when the solver result is successful or the final loss is below
        the absolute tolerance.
    solver_status : str
        Static human-readable solver status.
    """

    params: Any
    latent_params: Any
    simulated: Any
    residual: Any
    loss: Float[Array, ""]
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]
    solver_status: str = eqx.field(static=True)


class LaplaceUncertainty(eqx.Module):
    """Local Gaussian uncertainty estimate around a reconstruction optimum.

    Attributes
    ----------
    fisher_information : Float[Array, "P P"]
        Gauss-Newton/Fisher information matrix in flattened coordinates.
    covariance : Float[Array, "P P"]
        Regularized inverse Fisher matrix.
    standard_deviation : Float[Array, "P"]
        One-sigma uncertainty for each flattened parameter.
    correlation : Float[Array, "P P"]
        Parameter correlation matrix derived from covariance.
    """

    fisher_information: Float[Array, "P P"]
    covariance: Float[Array, "P P"]
    standard_deviation: Float[Array, "P"]
    correlation: Float[Array, "P P"]


class PosteriorSamples(eqx.Module):
    """Posterior sample container with diagnostics and credible intervals.

    Attributes
    ----------
    samples : Float[Array, "C S P"]
        Flattened posterior samples with chain, draw, and parameter axes.
    log_probability : Float[Array, "C S"]
        Log-posterior value for each retained sample.
    acceptance_rate : Float[Array, "C"]
        Mean NUTS acceptance rate for each chain.
    mean : Float[Array, "P"]
        Posterior mean in flattened coordinates.
    covariance : Float[Array, "P P"]
        Empirical posterior covariance.
    credible_interval : Float[Array, "2 P"]
        Equal-tailed posterior interval; row 0 is lower, row 1 is upper.
    r_hat : Float[Array, "P"]
        Split-chain Gelman-Rubin diagnostic per flattened parameter.
    effective_sample_size : Float[Array, "P"]
        Autocorrelation-adjusted effective sample size estimate.
    converged : Bool[Array, ""]
        True when all R-hat values and ESS values pass the supplied thresholds.
    unravel_fn : Callable[[Float[Array, "P"]], Any]
        Static function that maps one flattened sample back to the original
        parameter pytree.
    """

    samples: Float[Array, "C S P"]
    log_probability: Float[Array, "C S"]
    acceptance_rate: Float[Array, "C"]
    mean: Float[Array, "P"]
    covariance: Float[Array, "P P"]
    credible_interval: Float[Array, "2 P"]
    r_hat: Float[Array, "P"]
    effective_sample_size: Float[Array, "P"]
    converged: Bool[Array, ""]
    unravel_fn: Callable[[Float[Array, "P"]], Any] = eqx.field(static=True)

    def mean_tree(self) -> Any:
        """Return the posterior mean reconstructed as the original pytree."""
        mean: Any = self.unravel_fn(self.mean)
        return mean


class RecipeDeviationReport(eqx.Module):
    """Compare fitted reconstruction parameters with an intended recipe.

    Attributes
    ----------
    result : ReconResult
        Reconstruction result used as the fitted reality estimate.
    intended_params : Any
        Intended physical parameter pytree.
    deviation : Any
        Signed fitted-minus-intended parameter gap pytree.
    z_score : Any
        Gap normalized by per-parameter uncertainty.
    parameter_standard_deviation : Any
        Per-parameter one-sigma values used as z-score denominators.
    parameter_covariance : Float[Array, "P P"]
        Flattened covariance matrix used for the report.
    max_abs_z : Float[Array, ""]
        Maximum absolute z-score across flattened parameters.
    severity : Int[Array, ""]
        Severity code: 0 for matched, 1 for warning, 2 for critical.
    uncertainty_source : str
        Static source label: ``"supplied"`` or ``"laplace"``.
    parameter_labels : tuple[str, ...]
        Static flattened parameter labels for automaton payloads.
    schema_version : str
        Static payload schema version.
    warning_z : float
        Static warning threshold used for severity assignment.
    critical_z : float
        Static critical threshold used for severity assignment.
    """

    result: ReconResult
    intended_params: Any
    deviation: Any
    z_score: Any
    parameter_standard_deviation: Any
    parameter_covariance: Float[Array, "P P"]
    max_abs_z: Float[Array, ""]
    severity: Int[Array, ""]
    uncertainty_source: str = eqx.field(static=True, default="laplace")
    parameter_labels: tuple[str, ...] = eqx.field(static=True, default=())
    schema_version: str = eqx.field(
        static=True,
        default=RECIPE_DEVIATION_SCHEMA_VERSION,
    )
    warning_z: float = eqx.field(static=True, default=2.0)
    critical_z: float = eqx.field(static=True, default=3.0)


class OrientationFitResult(eqx.Module):
    """Results from orientation distribution fitting.

    Attributes
    ----------
    fitted_distribution : OrientationDistribution
        Recovered orientation distribution.
    final_loss : Float[Array, ""]
        Final scalar loss value.
    loss_history : Float[Array, "N_steps"]
        Loss value after each optimizer step.
    converged : bool
        Whether the optimizer met its stopping tolerance.
    n_iterations : int
        Number of recorded optimization iterations.
    residual_pattern : float_jax_image
        Difference ``I_observed - I_fitted`` for diagnostics.
    """

    fitted_distribution: OrientationDistribution
    final_loss: Float[Array, ""]
    loss_history: Float[Array, "N_steps"]
    converged: bool = eqx.field(static=True)
    n_iterations: int = eqx.field(static=True)
    residual_pattern: float_jax_image


@jaxtyped(typechecker=beartype)
def create_distribution_axis_spec(
    samples: Float[Array, "N D"],
    perturbation_fn: Callable[[Any, Float[Array, "D"]], Any],
    forward_model: Callable[[Any], Array],
    output_kind: str = "amplitude",
    axis_id: Optional[str] = None,
) -> DistributionAxisSpec:
    """Create a perturbation-axis specification for library reconstruction.

    Parameters
    ----------
    samples : Float[Array, "N D"]
        Two-dimensional latent-axis sample coordinates.
    perturbation_fn : Callable[[Any, Float[Array, "D"]], Any]
        Static function mapping ``(base_object, sample)`` to a perturbed
        physical object.
    forward_model : Callable[[Any], Array]
        Static differentiable forward model for one perturbed object.
    output_kind : str, optional
        Static forward-output interpretation, either ``"amplitude"`` or
        ``"intensity"``. Default: ``"amplitude"``
    axis_id : Optional[str], optional
        Optional static axis label stored on the recovered distribution.

    Returns
    -------
    axis_spec : DistributionAxisSpec
        Validated perturbation-axis specification.

    Raises
    ------
    ValueError
        If samples are not a non-empty two-dimensional array or
        ``output_kind`` is unsupported.

    Notes
    -----
    1. Convert samples to ``float64``.
    2. Validate the static sample rank and supported output interpretation.
    3. Store callables and metadata as static PyTree fields.
    """
    sample_array: Float[Array, "N D"] = jnp.asarray(
        samples,
        dtype=jnp.float64,
    )
    if sample_array.ndim != 2:
        raise ValueError("samples must have shape (N, D)")
    if sample_array.shape[0] <= 0:
        raise ValueError("samples must contain at least one row")
    if output_kind not in {"amplitude", "intensity"}:
        raise ValueError("output_kind must be 'amplitude' or 'intensity'")
    axis_spec: DistributionAxisSpec = DistributionAxisSpec(
        samples=sample_array,
        perturbation_fn=perturbation_fn,
        forward_model=forward_model,
        output_kind=output_kind,
        axis_id=axis_id,
    )
    return axis_spec


@jaxtyped(typechecker=beartype)
def create_crystal_displacement_axis_spec(
    samples: Float[Array, "N D"],
    displacement_modes: Float[Array, "D A 3"],
    forward_model: Callable[[CrystalStructure], Array],
    output_kind: str = "amplitude",
    axis_id: Optional[str] = "crystal_displacement",
) -> DistributionAxisSpec:
    """Create a crystal displacement-axis specification.

    Parameters
    ----------
    samples : Float[Array, "N D"]
        Per-row displacement-mode coordinates.
    displacement_modes : Float[Array, "D A 3"]
        Cartesian displacement modes in Angstrom for each active coordinate,
        atom, and spatial axis.
    forward_model : Callable[[CrystalStructure], Array]
        Static differentiable model for one perturbed crystal.
    output_kind : str, optional
        Static forward-output interpretation, either ``"amplitude"`` or
        ``"intensity"``. Default: ``"amplitude"``
    axis_id : Optional[str], optional
        Optional static label stored on the recovered distribution.
        Default: ``"crystal_displacement"``

    Returns
    -------
    axis_spec : DistributionAxisSpec
        Perturbation-axis specification that applies displacement modes to a
        :class:`~rheedium.types.CrystalStructure` carrier.

    Raises
    ------
    ValueError
        If samples, displacement modes, or per-crystal atom counts have
        incompatible shapes.

    Notes
    -----
    1. Validate that sample coordinates and displacement modes share the same
       active-mode dimension.
    2. Apply Cartesian displacements to the crystal carrier while preserving
       atomic numbers and cell parameters.
    3. Update fractional coordinates by the orthogonal-cell length scale, which
       is the intended lightweight fixture path for Debye-Waller/RMS
       displacement axes.
    """
    sample_array: Float[Array, "N D"] = jnp.asarray(
        samples,
        dtype=jnp.float64,
    )
    mode_array: Float[Array, "D A 3"] = jnp.asarray(
        displacement_modes,
        dtype=jnp.float64,
    )
    if sample_array.ndim != 2:
        raise ValueError("samples must have shape (N, D)")
    if mode_array.ndim != 3 or mode_array.shape[2] != 3:
        raise ValueError("displacement_modes must have shape (D, A, 3)")
    if sample_array.shape[1] != mode_array.shape[0]:
        raise ValueError(
            "samples second dimension must match displacement mode count"
        )

    def perturbation_fn(
        crystal: CrystalStructure,
        sample: Float[Array, "D"],
    ) -> CrystalStructure:
        displacement: Float[Array, "A 3"] = jnp.einsum(
            "d,daz->az",
            sample,
            mode_array,
        )
        atom_count: int = int(crystal.cart_positions.shape[0])
        if displacement.shape[0] != atom_count:
            raise ValueError(
                "displacement_modes atom dimension must match crystal atoms"
            )
        cart_xyz: Float[Array, "A 3"] = (
            crystal.cart_positions[:, :3] + displacement
        )
        cell_vectors: Float[Array, "3 3"] = _build_canonical_cell_vectors(
            crystal.cell_lengths, crystal.cell_angles
        )
        frac_xyz: Float[Array, "A 3"] = crystal.frac_positions[
            :, :3
        ] + displacement @ jnp.linalg.inv(cell_vectors)
        cart_positions: Float[Array, "A 4"] = crystal.cart_positions.at[
            :, :3
        ].set(cart_xyz)
        frac_positions: Float[Array, "A 4"] = crystal.frac_positions.at[
            :, :3
        ].set(frac_xyz)
        perturbed: CrystalStructure = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=crystal.cell_lengths,
            cell_angles=crystal.cell_angles,
        )
        return perturbed

    axis_spec: DistributionAxisSpec = create_distribution_axis_spec(
        samples=sample_array,
        perturbation_fn=perturbation_fn,
        forward_model=forward_model,
        output_kind=output_kind,
        axis_id=axis_id,
    )
    return axis_spec


__all__: list[str] = [
    "DistributionAxisSpec",
    "LaplaceUncertainty",
    "OrientationFitResult",
    "PosteriorSamples",
    "RECIPE_DEVIATION_SCHEMA_VERSION",
    "RecipeDeviationReport",
    "ReconProblem",
    "ReconResult",
    "create_crystal_displacement_axis_spec",
    "create_distribution_axis_spec",
]
