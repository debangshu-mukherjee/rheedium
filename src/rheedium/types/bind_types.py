"""Type carriers for distribution-axis bind updates.

Routine Listings
----------------
:class:`KinematicAxisUpdate`
    Per-axis update consumed by the kinematic detector kernel.
:class:`MultisliceAxisUpdate`
    Per-axis update consumed by the multislice detector kernel.
"""

from beartype.typing import Any, NamedTuple

from .crystal_types import CrystalStructure


class KinematicAxisUpdate(NamedTuple):
    """Kernel-local updates produced by one kinematic axis sample.

    Attributes
    ----------
    crystal : CrystalStructure | None
        Optional per-sample structure replacement.
    energy_delta_kev : Any
        Beam-energy delta in keV.
    theta_delta_deg : Any
        Incidence-angle delta in degrees.
    phi_delta_deg : Any
        Azimuth-angle delta in degrees.
    domain_size_angstrom : Any | None
        Optional coherent-domain size override in Angstrom.
    """

    crystal: CrystalStructure | None
    energy_delta_kev: Any
    theta_delta_deg: Any
    phi_delta_deg: Any
    domain_size_angstrom: Any | None


class MultisliceAxisUpdate(NamedTuple):
    """Kernel-local updates produced by one multislice axis sample.

    Attributes
    ----------
    crystal : CrystalStructure | None
        Optional per-sample structure replacement.
    energy_delta_kev : Any
        Beam-energy delta in keV.
    theta_delta_deg : Any
        Incidence-angle delta in degrees.
    phi_delta_deg : Any
        Azimuth-angle delta in degrees.
    domain_size_angstrom : Any | None
        Optional coherent-domain size override in Angstrom.
    """

    crystal: CrystalStructure | None
    energy_delta_kev: Any
    theta_delta_deg: Any
    phi_delta_deg: Any
    domain_size_angstrom: Any | None


__all__: list[str] = [
    "KinematicAxisUpdate",
    "MultisliceAxisUpdate",
]
