"""Parameter carriers for detector-image simulation orchestration."""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from .beam_types import ElectronBeam, create_electron_beam
from .crystal_types import PotentialSlices
from .custom_types import scalar_float, scalar_int
from .distributions import (
    BeamModeDistribution,
    Distribution,
    OrientationDistribution,
)
from .rheed_types import SurfaceConfig


class BeamSpec(ElectronBeam):
    """Electron beam plus nominal incidence and modal sampling metadata."""

    energy_spread_ev: scalar_float = 0.35
    angular_divergence_mrad: scalar_float = 0.35
    theta_deg: scalar_float = 2.0
    phi_deg: scalar_float = 0.0
    beam_modes: BeamModeDistribution | None = None
    n_beam_modes_per_axis: int = eqx.field(default=3, static=True)
    n_beam_modes_out_of_plane: int | None = eqx.field(
        default=None,
        static=True,
    )
    n_beam_energy_points: int = eqx.field(default=1, static=True)


class SurfaceCTRParams(eqx.Module):
    """CTR, roughness, and finite-domain surface parameters.

    ``layer_attenuation`` is the per-layer amplitude attenuation ε of the
    semi-infinite CTR truncation factor. ``ctr_regularization`` is a
    deprecated alias (the legacy additive constant of
    ``1/(sin^2(pi l) + reg)``); when not None it is converted downstream
    so the Bragg-peak cap matches. ``ctr_power`` and ``roughness_power``
    are non-physical diagnostic exponents; 1.0 is the physical model.
    """

    hmax: scalar_int = eqx.field(default=5, static=True)
    kmax: scalar_int = eqx.field(default=5, static=True)
    temperature: scalar_float = 300.0
    surface_roughness: scalar_float = 0.0
    layer_attenuation: scalar_float = 0.01
    ctr_regularization: scalar_float | None = None
    ctr_power: scalar_float = 1.0
    roughness_power: scalar_float = 1.0
    surface_config: SurfaceConfig | None = None
    defect_surface_layer_depth_angstrom: scalar_float = 1.0
    finite_domain_aspect_ratio: Tuple[float, float, float] = eqx.field(
        default=(1.0, 1.0, 0.5),
        static=True,
    )


class RenderParams(eqx.Module):
    """Detector rendering, kernel, and ensemble-integration parameters.

    The ``kernel="multislice"`` path runs the edge-on reflection multislice
    pipeline, controlled by the grid knobs ``dx_slice``, ``dy``, ``dz``
    (edge-on grid spacings in Angstroms), ``propagation_length_ang`` (total
    crystal length traversed along the beam), ``vacuum_above`` (vacuum
    read-off window above the surface), and ``cap_width`` (absorbing-layer
    thickness). ``potential_slices`` is deprecated and ignored for that
    kernel: edge-on slices are built from the crystal directly.
    ``inner_potential_v0`` applies only to the transmission-geometry tools
    (``multislice_propagate`` and friends); the reflection pipeline carries
    the mean inner potential inside its slices.
    """

    spot_sigma_px: scalar_float = 1.4
    n_angular_samples: int = eqx.field(default=5, static=True)
    n_energy_samples: int = eqx.field(default=5, static=True)
    n_mosaic_points: scalar_int = eqx.field(default=7, static=True)
    parameterization: str = eqx.field(default="lobato", static=True)
    render_ctrs_as_streaks: bool = eqx.field(default=True, static=True)
    kernel: str = eqx.field(default="kinematic", static=True)
    inner_potential_v0: scalar_float = 0.0
    bandwidth_limit: scalar_float = 2.0 / 3.0
    dx_slice: float = eqx.field(default=1.0, static=True)
    dy: float = eqx.field(default=0.25, static=True)
    dz: float = eqx.field(default=0.25, static=True)
    propagation_length_ang: float = eqx.field(default=200.0, static=True)
    vacuum_above: float = eqx.field(default=30.0, static=True)
    cap_width: float = eqx.field(default=15.0, static=True)
    potential_slices: PotentialSlices | None = None
    orientation_distribution: OrientationDistribution | None = None
    distribution: Distribution | None = None


@jaxtyped(typechecker=beartype)
def create_beam_spec(  # noqa: PLR0913
    energy_kev: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    phi_deg: scalar_float = 0.0,
    energy_spread_ev: scalar_float = 0.35,
    angular_divergence_mrad: scalar_float = 0.35,
    coherence_length_transverse_angstrom: scalar_float = 500.0,
    coherence_length_longitudinal_angstrom: scalar_float = 1000.0,
    spot_size_um: Float[Array, "2"] = jnp.array([100.0, 50.0]),
    beam_modes: BeamModeDistribution | None = None,
    n_beam_modes_per_axis: int = 3,
    n_beam_modes_out_of_plane: int | None = None,
    n_beam_energy_points: int = 1,
) -> BeamSpec:
    """Create a validated detector-image beam specification."""
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
        spot_size_um=spot_size_um,
    )
    return BeamSpec(
        energy_kev=beam.energy_kev,
        energy_spread_ev=beam.energy_spread_ev,
        angular_divergence_mrad=beam.angular_divergence_mrad,
        coherence_length_transverse_angstrom=(
            beam.coherence_length_transverse_angstrom
        ),
        coherence_length_longitudinal_angstrom=(
            beam.coherence_length_longitudinal_angstrom
        ),
        spot_size_um=beam.spot_size_um,
        theta_deg=jnp.asarray(theta_deg, dtype=jnp.float64),
        phi_deg=jnp.asarray(phi_deg, dtype=jnp.float64),
        beam_modes=beam_modes,
        n_beam_modes_per_axis=n_beam_modes_per_axis,
        n_beam_modes_out_of_plane=n_beam_modes_out_of_plane,
        n_beam_energy_points=n_beam_energy_points,
    )


__all__: list[str] = [
    "BeamSpec",
    "RenderParams",
    "SurfaceCTRParams",
    "create_beam_spec",
]
