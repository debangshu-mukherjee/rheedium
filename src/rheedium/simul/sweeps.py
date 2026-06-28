"""Generic vectorized detector-image sweep utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Sequence, Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    BeamSpec,
    CrystalStructure,
    DetectorGeometry,
    RenderParams,
    SurfaceCTRParams,
    scalar_float,
)

from .simulator import simulate_detector_image

SweepAxis = Tuple[str, Float[Array, "..."]]

_PHI_AXIS: str = "phi_deg"
_THETA_AXIS: str = "theta_deg"
_ENERGY_AXIS: str = "energy_kev"
_ROUGHNESS_AXIS: str = "surface_roughness"


def _default_beam(beam: BeamSpec | None) -> BeamSpec:
    """Return a concrete beam carrier."""
    if beam is None:
        return BeamSpec()
    return beam


def _default_surface(
    surface: SurfaceCTRParams | None,
) -> SurfaceCTRParams:
    """Return a concrete surface carrier."""
    if surface is None:
        return SurfaceCTRParams()
    return surface


def _update_axis(
    axis: str,
    value: scalar_float,
    beam: BeamSpec,
    surface: SurfaceCTRParams,
) -> tuple[BeamSpec, SurfaceCTRParams]:
    """Apply one scalar sweep coordinate to the relevant carrier."""
    if axis == _PHI_AXIS:
        return eqx.tree_at(lambda spec: spec.phi_deg, beam, value), surface
    if axis == _THETA_AXIS:
        return eqx.tree_at(lambda spec: spec.theta_deg, beam, value), surface
    if axis == _ENERGY_AXIS:
        return eqx.tree_at(lambda spec: spec.energy_kev, beam, value), surface
    if axis == _ROUGHNESS_AXIS:
        return beam, eqx.tree_at(
            lambda params: params.surface_roughness,
            surface,
            value,
        )
    raise ValueError(
        "axis must be one of 'phi_deg', 'theta_deg', 'energy_kev', "
        "or 'surface_roughness'"
    )


@jaxtyped(typechecker=beartype)
def simulate_detector_image_sweep(
    crystal: CrystalStructure,
    axis: SweepAxis,
    beam: BeamSpec | None = None,
    surface: SurfaceCTRParams | None = None,
    detector: DetectorGeometry | None = None,
    render: RenderParams | None = None,
) -> Float[Array, "N H W"]:
    """Simulate detector images over one named scalar carrier axis."""
    return simulate_detector_image_grid(
        crystal=crystal,
        axes=(axis,),
        beam=beam,
        surface=surface,
        detector=detector,
        render=render,
    )


@jaxtyped(typechecker=beartype)
def simulate_detector_image_grid(
    crystal: CrystalStructure,
    axes: Sequence[SweepAxis],
    beam: BeamSpec | None = None,
    surface: SurfaceCTRParams | None = None,
    detector: DetectorGeometry | None = None,
    render: RenderParams | None = None,
) -> Float[Array, "..."]:
    """Simulate an ordered Cartesian grid of detector-image sweeps."""
    beam = _default_beam(beam)
    surface = _default_surface(surface)

    if len(axes) == 0:
        return simulate_detector_image(
            crystal=crystal,
            beam=beam,
            surface=surface,
            detector=detector,
            render=render,
        )

    axis_name: str
    axis_values: Float[Array, "..."]
    axis_name, axis_values = axes[0]
    axis_bank: Float[Array, "N"] = jnp.asarray(axis_values)
    remaining_axes: Sequence[SweepAxis] = axes[1:]

    def _simulate_one(value: scalar_float) -> Float[Array, "..."]:
        sample_beam: BeamSpec
        sample_surface: SurfaceCTRParams
        sample_beam, sample_surface = _update_axis(
            axis_name,
            value,
            beam,
            surface,
        )
        return simulate_detector_image_grid(
            crystal=crystal,
            axes=remaining_axes,
            beam=sample_beam,
            surface=sample_surface,
            detector=detector,
            render=render,
        )

    return jax.vmap(_simulate_one)(axis_bank)


__all__: list[str] = [
    "simulate_detector_image_grid",
    "simulate_detector_image_sweep",
]
