"""Beam-mode distributions and ElectronBeam adapters."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Final, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from ..beam_types import ElectronBeam, create_electron_beam
from ..custom_types import scalar_float
from .base import ReductionMode

_MIN_SIN_INCIDENCE: Final[float] = 1e-6
_BETA_UPPER_MARGIN: Final[float] = 1e-12


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


__all__: list[str] = [
    "BeamModeDistribution",
    "beam_modes_from_electron_beam",
    "create_coherent_beam",
    "create_field_emission_beam",
    "create_gaussian_schell_beam",
    "create_thermionic_beam",
    "reduction_mode_from_coherence_length",
]
