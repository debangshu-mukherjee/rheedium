"""Probability distribution types for statistical RHEED simulation.

The submodules are organized as follows:

- :mod:`base`
    Base distribution contracts and reduction helpers.
- :mod:`beam`
    Beam-mode distributions and ElectronBeam adapters.
- :mod:`orientation`
    Orientation distributions and generic orientation producers.
- :mod:`size`
    Domain-size distributions and generic size producers.

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
    Factory for discrete rotational variants.
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
"""

from .base import (
    TRIVIAL,
    TRIVIAL_DISTRIBUTION,
    Distribution,
    ReductionMode,
    create_distribution,
    create_trivial_distribution,
)
from .beam import (
    BeamModeDistribution,
    beam_modes_from_electron_beam,
    create_coherent_beam,
    create_field_emission_beam,
    create_gaussian_schell_beam,
    create_thermionic_beam,
    reduction_mode_from_coherence_length,
)
from .orientation import (
    OrientationDistribution,
    create_discrete_orientation,
    create_gaussian_orientation,
    create_mixed_orientation,
    create_orientation_distribution,
    discretize_orientation,
    discretize_orientation_static,
    integrate_over_orientation,
    orientation_to_distribution,
)
from .size import (
    SizeDistribution,
    create_lognormal_size,
    discretize_size_distribution,
    size_to_distribution,
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
