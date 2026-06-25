"""Producer-owned bind helpers for generic Distribution axes.

Extended Summary
----------------
This module owns the interpretation of ``Distribution.axis_id`` sample rows.
Simulation kernels supply kernel-local context, while producer bind helpers
translate one sample into geometry, structure, or finite-domain updates.

Routine Listings
----------------
:class:`KinematicAxisUpdate`
    Per-axis update consumed by the kinematic detector kernel.
:class:`MultisliceAxisUpdate`
    Per-axis update consumed by the multislice detector kernel.
:func:`bind_kinematic_axis_distribution`
    Bind one Distribution axis to kinematic sample-update semantics.
:func:`bind_multislice_axis_distribution`
    Bind one Distribution axis to multislice sample-update semantics.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Final
from jaxtyping import Array, Float

from rheedium.types import CrystalStructure, Distribution

AZIMUTH_AXIS_IDS: Final[frozenset[str]] = frozenset(
    {"trivial", "orientation", "azimuth", "phi", "test_phi"}
)
BEAM_AXIS_IDS: Final[frozenset[str]] = frozenset(
    {"beam_modes", "coherent_beam", "legacy_instrument"}
)
STRUCTURE_AXIS_IDS: Final[frozenset[str]] = frozenset({"twins", "steps"})
GRAIN_AXIS_IDS: Final[frozenset[str]] = frozenset({"grains"})
SIZE_AXIS_IDS: Final[frozenset[str]] = frozenset({"size"})
UNSUPPORTED_AXIS_IDS: Final[frozenset[str]] = frozenset()


class KinematicAxisUpdate(NamedTuple):
    """Kernel-local updates produced by one kinematic axis sample."""

    crystal: CrystalStructure | None
    voltage_delta_kv: Any
    theta_delta_deg: Any
    phi_delta_deg: Any
    domain_size_angstrom: Any | None


class MultisliceAxisUpdate(NamedTuple):
    """Kernel-local updates produced by one multislice axis sample."""

    crystal: CrystalStructure | None
    voltage_delta_kv: Any
    theta_delta_deg: Any
    phi_delta_deg: Any
    domain_size_angstrom: Any | None


@beartype
def validate_kinematic_axis(distribution: Distribution) -> None:
    """Validate one Distribution axis for the kinematic detector kernel."""
    axis_id: str | None = distribution.axis_id
    sample_dim: int = distribution.samples.shape[1]
    if axis_id in UNSUPPORTED_AXIS_IDS:
        raise ValueError(
            f"Distribution axis {axis_id!r} has no detector-image bind yet."
        )
    if axis_id in BEAM_AXIS_IDS and sample_dim != 3:
        raise ValueError(
            f"Beam-like distribution axis {axis_id!r} requires samples "
            "with shape (N, 3)."
        )
    if axis_id == "twins" and sample_dim != 2:
        raise ValueError("Twin distributions require samples (N, 2).")
    if axis_id == "steps" and sample_dim != 3:
        raise ValueError("Step distributions require samples (N, 3).")
    if axis_id in GRAIN_AXIS_IDS and sample_dim != 2:
        raise ValueError(
            "Grain distributions require samples "
            "[orientation_angle_deg, grain_size_angstrom]."
        )
    if axis_id in SIZE_AXIS_IDS and sample_dim != 1:
        raise ValueError("Size distributions require samples (N, 1).")
    if (
        axis_id not in BEAM_AXIS_IDS
        and axis_id not in STRUCTURE_AXIS_IDS
        and axis_id not in GRAIN_AXIS_IDS
        and axis_id not in SIZE_AXIS_IDS
        and axis_id not in AZIMUTH_AXIS_IDS
        and not (axis_id is None and sample_dim == 1)
    ):
        raise ValueError(
            f"Distribution axis {axis_id!r} with sample dimension "
            f"{sample_dim} does not have a registered kinematic bind."
        )


@beartype
def bind_kinematic_axis_distribution(
    distribution: Distribution,
    twin_builder: Callable[[Float[Array, "2"]], CrystalStructure],
    step_builder: Callable[[Float[Array, "3"]], CrystalStructure],
) -> Callable[[Float[Array, "D_axis"]], KinematicAxisUpdate]:
    """Bind one Distribution axis to kinematic sample-update semantics."""
    validate_kinematic_axis(distribution)
    axis_id: str | None = distribution.axis_id

    def _axis_bound(
        axis_sample: Float[Array, "D_axis"],
    ) -> KinematicAxisUpdate:
        if axis_id in BEAM_AXIS_IDS:
            return KinematicAxisUpdate(
                crystal=None,
                voltage_delta_kv=1.0e-3 * axis_sample[2],
                theta_delta_deg=jnp.rad2deg(axis_sample[0]),
                phi_delta_deg=jnp.rad2deg(axis_sample[1]),
                domain_size_angstrom=None,
            )
        if axis_id == "twins":
            return KinematicAxisUpdate(
                crystal=twin_builder(axis_sample),
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=None,
            )
        if axis_id == "steps":
            return KinematicAxisUpdate(
                crystal=step_builder(axis_sample),
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=None,
            )
        if axis_id in GRAIN_AXIS_IDS:
            return KinematicAxisUpdate(
                crystal=None,
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=axis_sample[0],
                domain_size_angstrom=axis_sample[1],
            )
        if axis_id in SIZE_AXIS_IDS:
            return KinematicAxisUpdate(
                crystal=None,
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=axis_sample[0],
            )
        return KinematicAxisUpdate(
            crystal=None,
            voltage_delta_kv=0.0,
            theta_delta_deg=0.0,
            phi_delta_deg=axis_sample[0],
            domain_size_angstrom=None,
        )

    return _axis_bound


@beartype
def validate_multislice_axis(distribution: Distribution) -> None:
    """Validate one Distribution axis for the multislice detector kernel."""
    axis_id: str | None = distribution.axis_id
    sample_dim: int = distribution.samples.shape[1]
    if axis_id in UNSUPPORTED_AXIS_IDS:
        raise ValueError(
            f"Distribution axis {axis_id!r} has no multislice bind yet."
        )
    if axis_id in BEAM_AXIS_IDS and sample_dim != 3:
        raise ValueError(
            f"Beam-like distribution axis {axis_id!r} requires samples "
            "with shape (N, 3)."
        )
    if axis_id == "twins" and sample_dim != 2:
        raise ValueError("Twin distributions require samples (N, 2).")
    if axis_id == "steps" and sample_dim != 3:
        raise ValueError("Step distributions require samples (N, 3).")
    if axis_id in GRAIN_AXIS_IDS and sample_dim != 2:
        raise ValueError(
            "Grain distributions require samples "
            "[orientation_angle_deg, grain_size_angstrom]."
        )
    if axis_id in SIZE_AXIS_IDS and sample_dim != 1:
        raise ValueError("Size distributions require samples (N, 1).")
    if (
        axis_id not in BEAM_AXIS_IDS
        and axis_id not in STRUCTURE_AXIS_IDS
        and axis_id not in GRAIN_AXIS_IDS
        and axis_id not in SIZE_AXIS_IDS
        and axis_id not in AZIMUTH_AXIS_IDS
        and not (axis_id is None and sample_dim == 1)
    ):
        raise ValueError(
            f"Distribution axis {axis_id!r} with sample dimension "
            f"{sample_dim} does not have a registered multislice bind."
        )


@beartype
def bind_multislice_axis_distribution(
    distribution: Distribution,
    twin_builder: Callable[[Float[Array, "2"]], CrystalStructure],
    step_builder: Callable[[Float[Array, "3"]], CrystalStructure],
) -> Callable[[Float[Array, "D_axis"]], MultisliceAxisUpdate]:
    """Bind one Distribution axis to multislice sample-update semantics."""
    validate_multislice_axis(distribution)
    axis_id: str | None = distribution.axis_id

    def _axis_bound(
        axis_sample: Float[Array, "D_axis"],
    ) -> MultisliceAxisUpdate:
        if axis_id in BEAM_AXIS_IDS:
            return MultisliceAxisUpdate(
                crystal=None,
                voltage_delta_kv=1.0e-3 * axis_sample[2],
                theta_delta_deg=jnp.rad2deg(axis_sample[0]),
                phi_delta_deg=jnp.rad2deg(axis_sample[1]),
                domain_size_angstrom=None,
            )
        if axis_id == "twins":
            return MultisliceAxisUpdate(
                crystal=twin_builder(axis_sample),
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=None,
            )
        if axis_id == "steps":
            return MultisliceAxisUpdate(
                crystal=step_builder(axis_sample),
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=None,
            )
        if axis_id in GRAIN_AXIS_IDS:
            return MultisliceAxisUpdate(
                crystal=None,
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=axis_sample[0],
                domain_size_angstrom=axis_sample[1],
            )
        if axis_id in SIZE_AXIS_IDS:
            return MultisliceAxisUpdate(
                crystal=None,
                voltage_delta_kv=0.0,
                theta_delta_deg=0.0,
                phi_delta_deg=0.0,
                domain_size_angstrom=axis_sample[0],
            )
        return MultisliceAxisUpdate(
            crystal=None,
            voltage_delta_kv=0.0,
            theta_delta_deg=0.0,
            phi_delta_deg=axis_sample[0],
            domain_size_angstrom=None,
        )

    return _axis_bound


__all__: list[str] = [
    "AZIMUTH_AXIS_IDS",
    "BEAM_AXIS_IDS",
    "GRAIN_AXIS_IDS",
    "KinematicAxisUpdate",
    "MultisliceAxisUpdate",
    "SIZE_AXIS_IDS",
    "STRUCTURE_AXIS_IDS",
    "UNSUPPORTED_AXIS_IDS",
    "bind_kinematic_axis_distribution",
    "bind_multislice_axis_distribution",
    "validate_kinematic_axis",
    "validate_multislice_axis",
]
