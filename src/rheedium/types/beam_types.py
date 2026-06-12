r"""Data structures for electron beam and instrument characterization.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing the
electron beam source in RHEED simulation. The ``ElectronBeam`` PyTree
captures all physical beam parameters needed for instrument-broadened
simulations: energy, divergence, coherence lengths, and spot size.

Routine Listings
----------------
:class:`ElectronBeam`
    Complete specification of an electron beam for RHEED simulation.
:func:`create_electron_beam`
    Factory function to create ElectronBeam instances with validation.

Notes
-----
All fields are JAX-traceable scalars or arrays, enabling differentiation
through beam parameters via ``jax.grad``. The class is an Equinox module
(``eqx.Module``), so it is a registered JAX PyTree and can be passed
through ``jit``/``vmap``/``grad`` boundaries.

Coherence lengths are related to energy spread and divergence by:

- Longitudinal: :math:`L_l = \lambda^2 / \Delta\lambda`
- Transverse: :math:`L_t = \lambda / (2\pi \sigma_\theta)`

where :math:`\lambda` is the de Broglie wavelength and
:math:`\sigma_\theta` is the angular divergence.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from .custom_types import scalar_float

_MIN_ENERGY_KEV: float = 5.0
_MAX_ENERGY_KEV: float = 100.0


class ElectronBeam(eqx.Module):
    """Complete specification of an electron beam for RHEED simulation.

    This PyTree captures all physical parameters of the electron source
    needed for instrument-broadened RHEED pattern simulation. Typical
    RHEED guns have angular divergence 0.1--1 mrad, energy spread
    0.1--1 eV, and transverse coherence lengths 100--1000 Angstroms.

    :see: :class:`~.test_beam_types.TestElectronBeam`

    Attributes
    ----------
    energy_kev : scalar_float
        Nominal accelerating voltage in keV. Range: 5--100 keV.
        Default: 20.0
    energy_spread_ev : scalar_float
        1-sigma energy spread in eV. Typical: 0.1--1.0 eV.
        Controls longitudinal coherence and streak position variation.
        Default: 0.5
    angular_divergence_mrad : scalar_float
        1-sigma angular divergence in milliradians. Typical: 0.1--1.0
        mrad. Controls transverse coherence and streak width.
        Default: 0.5
    coherence_length_transverse_angstrom : scalar_float
        Transverse coherence length in Angstroms. Typical: 100--1000.
        Limits the angular range over which diffraction is coherent.
        Default: 500.0
    coherence_length_longitudinal_angstrom : scalar_float
        Longitudinal coherence length in Angstroms.
        Related to energy spread by L_l = lambda^2 / delta_lambda.
        Default: 1000.0
    spot_size_um : Float[Array, "2"]
        Beam footprint [width, height] on surface in micrometers.
        RHEED illuminates mm-scale areas; this sets the incoherent
        averaging domain. Default: [100.0, 50.0]

    Notes
    -----
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX transformations like jit,
    grad, and vmap. All continuous
    parameters (energy, spread, divergence, coherence lengths) are
    differentiable. The spot_size_um is also differentiable but rarely
    optimized in practice.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> beam = rh.types.create_electron_beam(
    ...     energy_kev=15.0,
    ...     angular_divergence_mrad=0.3,
    ...     energy_spread_ev=0.2,
    ... )
    """

    energy_kev: scalar_float = 20.0
    energy_spread_ev: scalar_float = 0.5
    angular_divergence_mrad: scalar_float = 0.5
    coherence_length_transverse_angstrom: scalar_float = 500.0
    coherence_length_longitudinal_angstrom: scalar_float = 1000.0
    spot_size_um: Float[Array, "2"] = eqx.field(
        default_factory=lambda: jnp.array([100.0, 50.0])
    )


@jaxtyped(typechecker=beartype)
def create_electron_beam(
    energy_kev: scalar_float = 20.0,
    energy_spread_ev: scalar_float = 0.5,
    angular_divergence_mrad: scalar_float = 0.5,
    coherence_length_transverse_angstrom: scalar_float = 500.0,
    coherence_length_longitudinal_angstrom: scalar_float = 1000.0,
    spot_size_um: Float[Array, "2"] = jnp.array([100.0, 50.0]),
) -> ElectronBeam:
    """Create an ElectronBeam instance with data validation.

    :see: :class:`~.test_beam_types.TestElectronBeam`

    Parameters
    ----------
    energy_kev : scalar_float
        Nominal accelerating voltage in keV. Must be in [5, 100].
        Default: 20.0
    energy_spread_ev : scalar_float
        1-sigma energy spread in eV. Must be non-negative.
        Default: 0.5
    angular_divergence_mrad : scalar_float
        1-sigma angular divergence in milliradians. Must be
        non-negative. Default: 0.5
    coherence_length_transverse_angstrom : scalar_float
        Transverse coherence length in Angstroms. Must be positive.
        Default: 500.0
    coherence_length_longitudinal_angstrom : scalar_float
        Longitudinal coherence length in Angstroms. Must be positive.
        Default: 1000.0
    spot_size_um : Float[Array, "2"]
        Beam footprint [width, height] in micrometers. Both components
        must be positive. Default: [100.0, 50.0]

    Returns
    -------
    validated_beam : ElectronBeam
        Validated ElectronBeam instance.

    Notes
    -----
    1. Cast all inputs to float64 JAX arrays.
    2. Validate energy_kev is in [5, 100] keV.
    3. Validate energy_spread_ev >= 0.
    4. Validate angular_divergence_mrad >= 0.
    5. Validate coherence lengths are positive.
    6. Validate spot_size_um components are positive.
    7. Return constructed ElectronBeam.

    Examples
    --------
    >>> import rheedium as rh
    >>>
    >>> beam = rh.types.create_electron_beam(energy_kev=15.0)
    >>> beam.energy_kev
    Array(15., dtype=float64)
    """
    energy_kev: scalar_float = jnp.asarray(energy_kev, dtype=jnp.float64)
    energy_spread_ev: scalar_float = jnp.asarray(
        energy_spread_ev, dtype=jnp.float64
    )
    angular_divergence_mrad: scalar_float = jnp.asarray(
        angular_divergence_mrad, dtype=jnp.float64
    )
    coherence_length_transverse_angstrom: scalar_float = jnp.asarray(
        coherence_length_transverse_angstrom, dtype=jnp.float64
    )
    coherence_length_longitudinal_angstrom: scalar_float = jnp.asarray(
        coherence_length_longitudinal_angstrom, dtype=jnp.float64
    )
    spot_size_um: Float[Array, "2"] = jnp.asarray(
        spot_size_um, dtype=jnp.float64
    )

    def _validate_and_create() -> ElectronBeam:
        """Validate inputs and create ElectronBeam."""

        def _check_energy() -> scalar_float:
            """Check energy_kev is in [5, 100]."""
            return eqx.error_if(
                energy_kev,
                jnp.logical_or(
                    energy_kev < _MIN_ENERGY_KEV,
                    energy_kev > _MAX_ENERGY_KEV,
                ),
                "energy_kev must be in [5, 100] keV",
            )

        def _check_energy_spread() -> scalar_float:
            """Check energy_spread_ev >= 0."""
            return eqx.error_if(
                energy_spread_ev,
                energy_spread_ev < 0.0,
                "energy_spread_ev must be non-negative",
            )

        def _check_divergence() -> scalar_float:
            """Check angular_divergence_mrad >= 0."""
            return eqx.error_if(
                angular_divergence_mrad,
                angular_divergence_mrad < 0.0,
                "angular_divergence_mrad must be non-negative",
            )

        def _check_transverse_coherence() -> scalar_float:
            """Check transverse coherence length > 0."""
            return eqx.error_if(
                coherence_length_transverse_angstrom,
                coherence_length_transverse_angstrom <= 0.0,
                "coherence_length_transverse_angstrom must be positive",
            )

        def _check_longitudinal_coherence() -> scalar_float:
            """Check longitudinal coherence length > 0."""
            return eqx.error_if(
                coherence_length_longitudinal_angstrom,
                coherence_length_longitudinal_angstrom <= 0.0,
                "coherence_length_longitudinal_angstrom must be positive",
            )

        def _check_spot_size() -> Float[Array, "2"]:
            """Check spot_size_um components are positive."""
            return eqx.error_if(
                spot_size_um,
                jnp.any(spot_size_um <= 0.0),
                "spot_size_um components must be positive",
            )

        validated_energy: scalar_float = _check_energy()
        validated_spread: scalar_float = _check_energy_spread()
        validated_divergence: scalar_float = _check_divergence()
        validated_transverse: scalar_float = _check_transverse_coherence()
        validated_longitudinal: scalar_float = _check_longitudinal_coherence()
        validated_spot: Float[Array, "2"] = _check_spot_size()

        return ElectronBeam(
            energy_kev=validated_energy,
            energy_spread_ev=validated_spread,
            angular_divergence_mrad=validated_divergence,
            coherence_length_transverse_angstrom=validated_transverse,
            coherence_length_longitudinal_angstrom=(validated_longitudinal),
            spot_size_um=validated_spot,
        )

    return _validate_and_create()


__all__: list[str] = [
    "ElectronBeam",
    "create_electron_beam",
]
