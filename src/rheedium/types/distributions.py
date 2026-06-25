"""Probability distribution types for statistical RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible probability distribution types for
modeling surface statistics in RHEED. The large beam footprint (mm² at
grazing incidence) samples a statistical ensemble of domains - we specify
distributions rather than individual domains.

Routine Listings
----------------
:class:`Distribution`
    Generic weighted ensemble over latent simulation samples.
:class:`BeamModeDistribution`
    Gaussian Schell-model beam-mode source parameters.
:class:`OrientationDistribution`
    Probability distribution over domain azimuthal orientations.
:class:`ReductionMode`
    Static reduction mode for coherent or incoherent ensemble axes.
:class:`SizeDistribution`
    Probability distribution over coherent domain sizes.
:func:`create_distribution`
    Factory for generic weighted sample distributions.
:func:`create_gaussian_schell_beam`
    Factory for anisotropic Gaussian Schell-model beam modes.
:func:`create_coherent_beam`
    Factory for a single sharp coherent beam mode.
:func:`beam_modes_from_electron_beam`
    Convert ElectronBeam coherence metadata to GSM beam-mode parameters.
:func:`create_field_emission_beam`
    Preset GSM beam producer for field-emission sources.
:func:`create_thermionic_beam`
    Preset GSM beam producer for thermionic sources.
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
:func:`create_trivial_distribution`
    Factory for the identity distribution with one zero sample.
:func:`reduction_mode_from_coherence_length`
    Choose coherent/incoherent reduction from feature and coherence lengths.
:func:`discretize_size_distribution`
    Convert SizeDistribution to quadrature sizes and weights.
:func:`orientation_to_distribution`
    Convert OrientationDistribution to the generic Distribution contract.
:func:`integrate_over_orientation`
    Compute incoherent intensity sum over orientation distribution.
:func:`size_to_distribution`
    Convert SizeDistribution to the generic Distribution contract.
:obj:`TRIVIAL_DISTRIBUTION`
    Identity one-sample distribution.
:obj:`TRIVIAL`
    Short alias for ``TRIVIAL_DISTRIBUTION``.

Notes
-----
All distribution types are PyTrees supporting JAX transformations.
Integration uses Gauss-Hermite quadrature for continuous distributions
and exact summation for discrete variants.
"""

from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Final, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.tools import gauss_hermite_nodes_weights

from .beam_types import ElectronBeam, create_electron_beam
from .custom_types import float_jax_image, scalar_float, scalar_int

_ZERO_MOSAIC_FWHM_DEG: Final[float] = 1e-6
_MIN_SIN_INCIDENCE: Final[float] = 1e-6
_BETA_UPPER_MARGIN: Final[float] = 1e-12


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


class BeamModeDistribution(eqx.Module):
    """Gaussian Schell-model beam-mode source parameters.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Attributes
    ----------
    beta_in_plane : Float[Array, ""]
        Geometric occupation decay ratio for scattering-plane modes.
    beta_out_of_plane : Float[Array, ""]
        Geometric occupation decay ratio for out-of-plane modes.
    divergence_in_plane_rad : Float[Array, ""]
        Total 1-sigma angular divergence along the polar-angle axis.
    divergence_out_of_plane_rad : Float[Array, ""]
        Total 1-sigma angular divergence along the azimuthal axis.
    energy_spread_ev : Float[Array, ""]
        Longitudinal 1-sigma energy spread in electron-volts.
    distribution_id : Optional[str]
        Optional static label for diagnostics and composition.
    """

    beta_in_plane: Float[Array, ""]
    beta_out_of_plane: Float[Array, ""]
    divergence_in_plane_rad: Float[Array, ""]
    divergence_out_of_plane_rad: Float[Array, ""]
    energy_spread_ev: Float[Array, ""]
    distribution_id: Optional[str] = eqx.field(static=True, default=None)


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


@jaxtyped(typechecker=beartype)
def reduction_mode_from_coherence_length(
    feature_length_angstrom: scalar_float,
    coherence_length_angstrom: scalar_float,
) -> ReductionMode:
    """Choose a static reduction mode from a coherence-length threshold.

    :see: :class:`~.test_distributions.TestCoherenceReduction`

    Parameters
    ----------
    feature_length_angstrom : scalar_float
        Characteristic feature size in Angstroms. Features no larger than the
        coherent footprint are averaged coherently.
    coherence_length_angstrom : scalar_float
        Beam-mode transverse coherence length in Angstroms.

    Returns
    -------
    reduction_mode : ReductionMode
        ``COHERENT`` when ``feature_length_angstrom <= coherence_length`` and
        ``INCOHERENT`` otherwise.

    Notes
    -----
    1. Convert scalar inputs to Python floats for a static PyTree field.
    2. Validate positive length scales.
    3. Return the corresponding static reduction mode.
    """
    feature_length: float = float(feature_length_angstrom)
    coherence_length: float = float(coherence_length_angstrom)
    if feature_length <= 0.0:
        raise ValueError("feature_length_angstrom must be positive")
    if coherence_length <= 0.0:
        raise ValueError("coherence_length_angstrom must be positive")
    if feature_length <= coherence_length:
        return ReductionMode.COHERENT
    return ReductionMode.INCOHERENT


@jaxtyped(typechecker=beartype)
def create_gaussian_schell_beam(
    beta_in_plane: scalar_float = 0.0,
    beta_out_of_plane: scalar_float = 0.0,
    divergence_in_plane_rad: scalar_float = 0.0,
    divergence_out_of_plane_rad: scalar_float = 0.0,
    energy_spread_ev: scalar_float = 0.0,
    distribution_id: Optional[str] = None,
) -> BeamModeDistribution:
    """Create a validated Gaussian Schell-model beam-mode producer.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Parameters
    ----------
    beta_in_plane, beta_out_of_plane : scalar_float, optional
        GSM geometric occupation decay ratios. Must satisfy ``0 <= beta < 1``.
    divergence_in_plane_rad, divergence_out_of_plane_rad : scalar_float
        Total 1-sigma angular divergence per transverse axis in radians.
    energy_spread_ev : scalar_float, optional
        Longitudinal 1-sigma energy spread in electron-volts.
    distribution_id : Optional[str], optional
        Optional static identifier for the beam axis.

    Returns
    -------
    beam_modes : BeamModeDistribution
        Validated physical beam-mode parameters.
    """
    beta_in: Float[Array, ""] = jnp.asarray(beta_in_plane, dtype=jnp.float64)
    beta_out: Float[Array, ""] = jnp.asarray(
        beta_out_of_plane, dtype=jnp.float64
    )
    divergence_in: Float[Array, ""] = jnp.asarray(
        divergence_in_plane_rad,
        dtype=jnp.float64,
    )
    divergence_out: Float[Array, ""] = jnp.asarray(
        divergence_out_of_plane_rad,
        dtype=jnp.float64,
    )
    energy_spread: Float[Array, ""] = jnp.asarray(
        energy_spread_ev,
        dtype=jnp.float64,
    )
    checked_beta_in: Float[Array, ""] = eqx.error_if(
        beta_in,
        (~jnp.isfinite(beta_in)) | (beta_in < 0.0) | (beta_in >= 1.0),
        "beta_in_plane must be finite and in [0, 1)",
    )
    checked_beta_out: Float[Array, ""] = eqx.error_if(
        beta_out,
        (~jnp.isfinite(beta_out)) | (beta_out < 0.0) | (beta_out >= 1.0),
        "beta_out_of_plane must be finite and in [0, 1)",
    )
    checked_divergence_in: Float[Array, ""] = eqx.error_if(
        divergence_in,
        (~jnp.isfinite(divergence_in)) | (divergence_in < 0.0),
        "divergence_in_plane_rad must be finite and non-negative",
    )
    checked_divergence_out: Float[Array, ""] = eqx.error_if(
        divergence_out,
        (~jnp.isfinite(divergence_out)) | (divergence_out < 0.0),
        "divergence_out_of_plane_rad must be finite and non-negative",
    )
    checked_energy_spread: Float[Array, ""] = eqx.error_if(
        energy_spread,
        (~jnp.isfinite(energy_spread)) | (energy_spread < 0.0),
        "energy_spread_ev must be finite and non-negative",
    )
    return BeamModeDistribution(
        beta_in_plane=checked_beta_in,
        beta_out_of_plane=checked_beta_out,
        divergence_in_plane_rad=checked_divergence_in,
        divergence_out_of_plane_rad=checked_divergence_out,
        energy_spread_ev=checked_energy_spread,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_coherent_beam(
    energy_spread_ev: scalar_float = 0.0,
    distribution_id: Optional[str] = "coherent_beam",
) -> BeamModeDistribution:
    """Create a single sharp coherent beam-mode producer.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Parameters
    ----------
    energy_spread_ev : scalar_float, optional
        Optional longitudinal spread to keep while collapsing transverse modes.
        Default: 0.0.
    distribution_id : Optional[str], optional
        Optional static identifier. Default: ``"coherent_beam"``.

    Returns
    -------
    beam_modes : BeamModeDistribution
        Beam parameters with zero transverse modal spread.
    """
    return create_gaussian_schell_beam(
        beta_in_plane=0.0,
        beta_out_of_plane=0.0,
        divergence_in_plane_rad=0.0,
        divergence_out_of_plane_rad=0.0,
        energy_spread_ev=energy_spread_ev,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def _beta_from_projected_source(
    projected_source_um: Float[Array, ""],
    coherence_length_transverse_angstrom: Float[Array, ""],
) -> Float[Array, ""]:
    """Map projected source/coherence scale to a bounded GSM beta."""
    coherence_um: Float[Array, ""] = (
        coherence_length_transverse_angstrom / 1.0e4
    )
    raw_beta: Float[Array, ""] = projected_source_um / (
        projected_source_um + coherence_um
    )
    return jnp.clip(raw_beta, 0.0, 1.0 - _BETA_UPPER_MARGIN)


@jaxtyped(typechecker=beartype)
def beam_modes_from_electron_beam(
    beam: ElectronBeam,
    incidence_angle_deg: scalar_float = 2.0,
    distribution_id: Optional[str] = None,
) -> BeamModeDistribution:
    """Convert ElectronBeam metadata to GSM beam-mode parameters.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Parameters
    ----------
    beam : ElectronBeam
        Existing beam metadata with divergence, coherence length, and
        footprint.
    incidence_angle_deg : scalar_float, optional
        Grazing incidence angle used to project the beam footprint in the
        scattering plane. Default: 2.0.
    distribution_id : Optional[str], optional
        Optional override for the returned beam distribution label.

    Returns
    -------
    beam_modes : BeamModeDistribution
        GSM beam-mode producer derived from the existing beam specification.
    """
    incidence_rad: Float[Array, ""] = jnp.deg2rad(
        jnp.asarray(incidence_angle_deg, dtype=jnp.float64)
    )
    sin_incidence: Float[Array, ""] = jnp.maximum(
        jnp.sin(incidence_rad),
        _MIN_SIN_INCIDENCE,
    )
    projected_in_plane_um: Float[Array, ""] = beam.spot_size_um[0] / (
        sin_incidence
    )
    projected_out_of_plane_um: Float[Array, ""] = beam.spot_size_um[1]
    beta_in: Float[Array, ""] = _beta_from_projected_source(
        projected_in_plane_um,
        beam.coherence_length_transverse_angstrom,
    )
    beta_out: Float[Array, ""] = _beta_from_projected_source(
        projected_out_of_plane_um,
        beam.coherence_length_transverse_angstrom,
    )
    divergence_rad: Float[Array, ""] = beam.angular_divergence_mrad * 1.0e-3
    axis_id: Optional[str] = (
        distribution_id if distribution_id is not None else "electron_beam"
    )
    return create_gaussian_schell_beam(
        beta_in_plane=beta_in,
        beta_out_of_plane=beta_out,
        divergence_in_plane_rad=divergence_rad,
        divergence_out_of_plane_rad=divergence_rad,
        energy_spread_ev=beam.energy_spread_ev,
        distribution_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def create_field_emission_beam(
    incidence_angle_deg: scalar_float = 2.0,
    energy_kev: scalar_float = 20.0,
    energy_spread_ev: scalar_float = 0.5,
    angular_divergence_mrad: scalar_float = 0.25,
    coherence_length_transverse_angstrom: scalar_float = 1000.0,
    coherence_length_longitudinal_angstrom: scalar_float = 2000.0,
    spot_size_um: Tuple[float, float] = (50.0, 25.0),
    distribution_id: Optional[str] = "field_emission_beam",
) -> BeamModeDistribution:
    """Create a field-emission GSM beam-mode preset.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Returns
    -------
    beam_modes : BeamModeDistribution
        Beam-mode producer with relatively high coherence and low spread.
    """
    beam: ElectronBeam = create_electron_beam(
        energy_kev=energy_kev,
        energy_spread_ev=energy_spread_ev,
        angular_divergence_mrad=angular_divergence_mrad,
        coherence_length_transverse_angstrom=(
            coherence_length_transverse_angstrom
        ),
        coherence_length_longitudinal_angstrom=(
            coherence_length_longitudinal_angstrom
        ),
        spot_size_um=jnp.asarray(spot_size_um, dtype=jnp.float64),
    )
    return beam_modes_from_electron_beam(
        beam,
        incidence_angle_deg=incidence_angle_deg,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_thermionic_beam(
    incidence_angle_deg: scalar_float = 2.0,
    energy_kev: scalar_float = 20.0,
    energy_spread_ev: scalar_float = 2.0,
    angular_divergence_mrad: scalar_float = 0.8,
    coherence_length_transverse_angstrom: scalar_float = 150.0,
    coherence_length_longitudinal_angstrom: scalar_float = 500.0,
    spot_size_um: Tuple[float, float] = (250.0, 100.0),
    distribution_id: Optional[str] = "thermionic_beam",
) -> BeamModeDistribution:
    """Create a thermionic GSM beam-mode preset.

    :see: :class:`~.test_distributions.TestBeamModeDistributionFactories`

    Returns
    -------
    beam_modes : BeamModeDistribution
        Beam-mode producer with broad, highly mixed thermionic parameters.
    """
    beam: ElectronBeam = create_electron_beam(
        energy_kev=energy_kev,
        energy_spread_ev=energy_spread_ev,
        angular_divergence_mrad=angular_divergence_mrad,
        coherence_length_transverse_angstrom=(
            coherence_length_transverse_angstrom
        ),
        coherence_length_longitudinal_angstrom=(
            coherence_length_longitudinal_angstrom
        ),
        spot_size_um=jnp.asarray(spot_size_um, dtype=jnp.float64),
    )
    return beam_modes_from_electron_beam(
        beam,
        incidence_angle_deg=incidence_angle_deg,
        distribution_id=distribution_id,
    )


class OrientationDistribution(eqx.Module):
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

    :see: :class:`~.test_distributions.TestOrientationDistributionFactories`

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
    distribution_id: Optional[str] = eqx.field(static=True, default=None)


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
    if n_angles <= 0:
        raise ValueError("angles_deg must contain at least one angle")

    checked_angles: Float[Array, "M"] = eqx.error_if(
        angles_arr,
        jnp.any(~jnp.isfinite(angles_arr)),
        "angles_deg must be finite",
    )
    weights_arr: Float[Array, "M"]
    if weights is None:
        weights_arr = jnp.ones(n_angles, dtype=jnp.float64) / n_angles
    else:
        raw_weights: Float[Array, "M"] = jnp.asarray(
            weights, dtype=jnp.float64
        )
        if raw_weights.shape != angles_arr.shape:
            raise ValueError("weights must have the same shape as angles_deg")
        checked_weights: Float[Array, "M"] = eqx.error_if(
            raw_weights,
            jnp.any(~jnp.isfinite(raw_weights)),
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
        weights_arr = _normalize_probability_weights(checked_weights)
    mosaic_fwhm_arr: Float[Array, ""] = jnp.asarray(
        mosaic_fwhm_deg, dtype=jnp.float64
    )
    checked_mosaic_fwhm: Float[Array, ""] = eqx.error_if(
        mosaic_fwhm_arr,
        ~jnp.isfinite(mosaic_fwhm_arr),
        "mosaic_fwhm_deg must be finite",
    )
    checked_mosaic_fwhm = eqx.error_if(
        checked_mosaic_fwhm,
        checked_mosaic_fwhm < 0.0,
        "mosaic_fwhm_deg must be non-negative",
    )
    return OrientationDistribution(
        discrete_angles_deg=checked_angles,
        discrete_weights=weights_arr,
        mosaic_fwhm_deg=checked_mosaic_fwhm,
        distribution_id=distribution_id,
    )


@jaxtyped(typechecker=beartype)
def create_discrete_orientation(
    angles_deg: Float[Array, "M"],
    weights: Optional[Float[Array, "M"]] = None,
    distribution_id: Optional[str] = None,
) -> OrientationDistribution:
    """Create discrete orientation distribution for rotational variants.

    :see: :class:`~.test_distributions.TestOrientationDistributionFactories`

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

    :see: :class:`~.test_distributions.TestOrientationDistributionFactories`

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

    :see: :class:`~.test_distributions.TestOrientationDistributionFactories`

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

    :see: :class:`~.test_distributions.TestOrientationDiscretization`

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

    :see: :class:`~.test_distributions.TestOrientationDiscretization`

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
def orientation_to_distribution(
    dist: OrientationDistribution,
    n_mosaic_points: scalar_int = 7,
    base_phi_deg: scalar_float = 0.0,
    use_static_discretization: bool = False,
) -> Distribution:
    """Convert orientation samples to a generic incoherent Distribution.

    :see: :class:`~.test_distributions.TestOrientationProducer`

    Parameters
    ----------
    dist : OrientationDistribution
        Orientation probability distribution.
    n_mosaic_points : scalar_int, optional
        Quadrature points per mosaic peak. Default: 7.
    base_phi_deg : scalar_float, optional
        Base azimuth added to each orientation sample. Default: 0.0.
    use_static_discretization : bool, optional
        If True, use the Python-branching static discretizer. Default: False.

    Returns
    -------
    distribution : Distribution
        Generic distribution with one ``phi_deg`` sample coordinate per row
        and incoherent reduction.

    Notes
    -----
    1. Discretize orientation support and probability weights.
    2. Shift samples by the base azimuth.
    3. Package as a one-coordinate generic incoherent distribution.
    """
    angles_deg: Float[Array, "N"]
    weights: Float[Array, "N"]
    if use_static_discretization:
        angles_deg, weights = discretize_orientation_static(
            dist,
            n_mosaic_points=n_mosaic_points,
        )
    else:
        angles_deg, weights = discretize_orientation(
            dist,
            n_mosaic_points=n_mosaic_points,
        )
    shifted_angles: Float[Array, "N"] = angles_deg + jnp.asarray(
        base_phi_deg, dtype=jnp.float64
    )
    axis_id: str = (
        dist.distribution_id
        if dist.distribution_id is not None
        else "orientation"
    )
    return create_distribution(
        samples=shifted_angles[:, None],
        weights=weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id=axis_id,
    )


@jaxtyped(typechecker=beartype)
def integrate_over_orientation(
    simulate_fn: Callable[[scalar_float], float_jax_image],
    orientation_dist: OrientationDistribution,
    n_mosaic_points: scalar_int = 7,
) -> float_jax_image:
    r"""Compute incoherent intensity sum over orientation distribution.

    :see: :class:`~.test_distributions.TestOrientationIntegration`

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
    distribution: Distribution = orientation_to_distribution(
        orientation_dist,
        n_mosaic_points=n_mosaic_points,
    )

    def _amplitude_from_orientation(
        sample: Float[Array, "D"],
    ) -> Float[Array, "H W"]:
        intensity: Float[Array, "H W"] = simulate_fn(sample[0])
        return jnp.sqrt(jnp.maximum(intensity, 0.0))

    from rheedium.simul.beam_averaging import apply_distribution

    weighted_sum: float_jax_image = apply_distribution(
        distribution,
        _amplitude_from_orientation,
    )
    return weighted_sum


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
    3. Gaussian and exponential distributions use clipped positive support.
    4. Weights are normalized after support clipping.
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
    elif dist.distribution_type == "exponential":
        raw_sizes = jnp.maximum(0.0, mean_ang * (1.0 + sqrt2 * nodes))
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
    "BeamModeDistribution",
    "Distribution",
    "OrientationDistribution",
    "ReductionMode",
    "SizeDistribution",
    "TRIVIAL",
    "TRIVIAL_DISTRIBUTION",
    "beam_modes_from_electron_beam",
    "create_coherent_beam",
    "create_distribution",
    "create_field_emission_beam",
    "create_gaussian_schell_beam",
    "create_orientation_distribution",
    "create_discrete_orientation",
    "create_gaussian_orientation",
    "create_lognormal_size",
    "create_mixed_orientation",
    "create_thermionic_beam",
    "create_trivial_distribution",
    "discretize_orientation",
    "discretize_orientation_static",
    "discretize_size_distribution",
    "integrate_over_orientation",
    "orientation_to_distribution",
    "reduction_mode_from_coherence_length",
    "size_to_distribution",
]
