"""Custom types and data structures for RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, RHEED patterns, and other simulation data. All types are PyTrees
that support JAX transformations and automatic differentiation.

The submodules are organized as follows:

- :mod:`beam_types`
    Data structures for electron beam and instrument characterization.
- :mod:`bind_types`
    Type carriers for distribution-axis bind updates.
- :mod:`constants`
    Physical constants for RHEED simulation.
- :mod:`crystal_types`
    Data structures and factory functions for crystal structure representation.
- :mod:`custom_types`
    Custom type aliases for scalar JAX data.
- :mod:`detector`
    Detector geometry carrier and detector-grid helpers.
- :mod:`distributions`
    Probability distribution types for statistical RHEED simulation.
- :mod:`inout_types`
    Type carriers for input/output metadata.
- :mod:`recon_types`
    Type carriers and constructors for reconstruction workflows.
- :mod:`rheed_types`
    Data structures and factory functions for RHEED pattern representation.
- :mod:`simulation_params`
    Parameter carriers for detector-image simulation orchestration.

Routine Listings
----------------
:class:`CrystalStructure`
    JAX-compatible crystal structure with fractional and Cartesian coordinates.
:class:`BeamModeDistribution`
    Gaussian Schell-model beam-mode source parameters.
:class:`BeamSpec`
    Electron beam, incidence geometry, and beam-mode sampling metadata.
:class:`DetectorGeometry`
    Configuration for RHEED detector geometry (tilt, curvature, offsets).
:class:`Distribution`
    Generic weighted ensemble over latent simulation samples.
:class:`DistributionAxisSpec`
    Static perturbation-axis contract for distribution reconstruction.
:class:`ElectronBeam`
    Complete specification of an electron beam for RHEED simulation.
:class:`EwaldData`
    Angle-independent Ewald sphere data for RHEED simulation.
:class:`EdgeOnSlices`
    Edge-on projected-potential slices for reflection multislice.
:class:`FrameMetadata`
    Per-frame metadata extracted from TIFF tags.
:class:`KinematicAxisUpdate`
    Per-axis update consumed by the kinematic detector kernel.
:class:`KirklandParameters`
    Structured Kirkland coefficients for one element.
:class:`LaplaceUncertainty`
    Local Gaussian uncertainty estimate around a reconstruction optimum.
:class:`MultisliceAxisUpdate`
    Per-axis update consumed by the multislice detector kernel.
:class:`OrientationFitResult`
    Result container for orientation-distribution fitting.
:class:`OrientationDistribution`
    Probability distribution over azimuthal domain orientations.
:class:`PosteriorSamples`
    Posterior sample container with diagnostics and credible intervals.
:class:`PotentialSlices`
    JAX-compatible data structure for representing multislice potential data.
:class:`RecipeDeviationReport`
    Compare fitted reconstruction parameters with an intended recipe.
:class:`ReconProblem`
    Differentiable inverse problem definition for reconstruction solvers.
:class:`ReconResult`
    Result container returned by the general reconstruction solver.
:class:`RHEEDImage`
    Container for RHEED image data with pixel coordinates and intensity values.
:class:`RHEEDPattern`
    Container for RHEED diffraction pattern data with detector points and
    intensities.
:class:`ReductionMode`
    Static coherent or incoherent distribution reduction mode.
:class:`SlicedCrystal`
    JAX-compatible crystal structure sliced for multislice simulation.
:class:`SizeDistribution`
    Probability distribution over coherent domain sizes.
:class:`SurfaceConfig`
    Configuration for surface atom identification method and parameters.
:class:`SurfaceCTRParams`
    CTR, roughness, and finite-domain surface parameters.
:class:`RenderParams`
    Detector rendering, kernel, and ensemble-integration parameters.
:class:`XYZData`
    A PyTree for XYZ file data with atomic positions and metadata.
:func:`create_crystal_structure`
    Factory function to create CrystalStructure instances.
:func:`beam_modes_from_electron_beam`
    Convert ElectronBeam metadata to GSM beam-mode parameters.
:func:`create_crystal_displacement_axis_spec`
    Create a crystal displacement-axis specification for library
    reconstruction.
:func:`create_coherent_beam`
    Create a single sharp coherent beam-mode producer.
:func:`create_distribution_axis_spec`
    Create a perturbation-axis specification for library reconstruction.
:func:`create_electron_beam`
    Factory function to create ElectronBeam instances.
:func:`create_beam_spec`
    Factory function to create BeamSpec instances.
:func:`create_ewald_data`
    Factory function to create EwaldData instances.
:func:`create_edge_on_slices`
    Factory function to create EdgeOnSlices instances.
:func:`create_field_emission_beam`
    Create a field-emission GSM beam-mode preset.
:func:`create_orientation_distribution`
    Canonical factory for orientation distributions.
:func:`create_discrete_orientation`
    Create a sharp rotational-variant distribution.
:func:`create_distribution`
    Create a generic weighted latent-sample distribution.
:func:`detector_distance_mm`
    Extract detector distance from a DetectorGeometry carrier.
:func:`detector_extent_mm`
    Compute display extent from a DetectorGeometry carrier.
:func:`detector_psf_sigma_pixels`
    Extract detector PSF width from a DetectorGeometry carrier.
:func:`create_gaussian_orientation`
    Create a Gaussian mosaic orientation distribution.
:func:`create_gaussian_schell_beam`
    Create an anisotropic Gaussian Schell-model beam producer.
:func:`create_kirkland_parameters`
    Factory function to create KirklandParameters instances.
:func:`create_lognormal_size`
    Create a lognormal domain-size distribution.
:func:`create_mixed_orientation`
    Create discrete orientation variants with mosaic broadening.
:func:`create_potential_slices`
    Factory function to create PotentialSlices instances.
:func:`create_rheed_image`
    Factory function to create RHEEDImage instances.
:func:`create_rheed_pattern`
    Factory function to create RHEEDPattern instances.
:func:`create_sliced_crystal`
    Factory function to create SlicedCrystal instances.
:func:`create_thermionic_beam`
    Create a thermionic GSM beam-mode preset.
:func:`create_trivial_distribution`
    Create the one-sample identity distribution.
:func:`create_xyz_data`
    Factory function to create XYZData instances.
:func:`discretize_orientation`
    Convert an orientation distribution to quadrature samples.
:func:`discretize_orientation_static`
    Python-branching orientation discretization for non-JIT use.
:func:`discretize_size_distribution`
    Convert size distribution to quadrature samples.
:func:`identify_surface_atoms`
    Identify surface atoms using configurable methods.
:func:`integrate_over_orientation`
    Simulate and incoherently average over orientation samples.
:func:`orientation_to_distribution`
    Convert orientation distribution to generic distribution.
:func:`reduction_mode_from_coherence_length`
    Choose coherent/incoherent reduction from feature and coherence lengths.
:func:`size_to_distribution`
    Convert size distribution to generic distribution.
:obj:`TRIVIAL_DISTRIBUTION`
    Identity one-sample distribution.
:obj:`TRIVIAL`
    Short alias for the identity one-sample distribution.
:obj:`RECIPE_DEVIATION_SCHEMA_VERSION`
    Frozen recipe-deviation payload schema identifier.
:obj:`float_jax_image`
    Type alias for float-valued 2D JAX image arrays.
:obj:`float_np_image`
    Type alias for float-valued 2D numpy image arrays.
:obj:`int_jax_image`
    Type alias for integer-valued 2D JAX image arrays.
:obj:`int_np_image`
    Type alias for integer-valued 2D numpy image arrays.
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX scalar array).
:obj:`AMU_TO_KG`
    Atomic mass unit to kg conversion factor.
:obj:`BOLTZMANN_CONSTANT_JK`
    Boltzmann constant in J/K.
:obj:`ELECTRON_MASS_KG`
    Electron rest mass in kg.
:obj:`ELEMENTARY_CHARGE_C`
    Elementary charge in C.
:obj:`H_OVER_SQRT_2ME_ANG_VSQRT`
    Electron wavelength prefactor *h / sqrt(2 m_e e)* in Ang V^0.5.
:obj:`HBAR_JS`
    Reduced Planck constant in J s.
:obj:`M2_TO_ANG2`
    Square metres to square angstroms conversion factor.
:obj:`PLANCK_CONSTANT_JS`
    Planck constant *h* in J s.
:obj:`RELATIVISTIC_COEFF_PER_V`
    Relativistic correction coefficient *e / (2 m_e c^2)* in 1/V.
:obj:`SPEED_OF_LIGHT_MS`
    Speed of light in vacuum in m/s.

Notes
-----
Every PyTree has a corresponding factory function to create the instance. This
is because beartype does not support type checking of dataclasses.
"""

from .beam_types import (
    ElectronBeam,
    create_electron_beam,
)
from .bind_types import (
    KinematicAxisUpdate,
    MultisliceAxisUpdate,
)
from .constants import (
    AMU_TO_KG,
    BOLTZMANN_CONSTANT_JK,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    H_OVER_SQRT_2ME_ANG_VSQRT,
    HBAR_JS,
    M2_TO_ANG2,
    PLANCK_CONSTANT_JS,
    RELATIVISTIC_COEFF_PER_V,
    SPEED_OF_LIGHT_MS,
)
from .crystal_types import (
    CrystalStructure,
    EdgeOnSlices,
    EwaldData,
    KirklandParameters,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
    create_edge_on_slices,
    create_ewald_data,
    create_kirkland_parameters,
    create_potential_slices,
    create_xyz_data,
)
from .custom_types import (
    float_jax_image,
    float_np_image,
    int_jax_image,
    int_np_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)
from .detector import (
    DetectorGeometry,
    detector_beam_center_px,
    detector_distance_mm,
    detector_extent_mm,
    detector_image_shape_px,
    detector_pixel_size_mm,
    detector_psf_sigma_pixels,
)
from .distributions import (
    TRIVIAL,
    TRIVIAL_DISTRIBUTION,
    BeamModeDistribution,
    Distribution,
    OrientationDistribution,
    ReductionMode,
    SizeDistribution,
    beam_modes_from_electron_beam,
    create_coherent_beam,
    create_discrete_orientation,
    create_distribution,
    create_field_emission_beam,
    create_gaussian_orientation,
    create_gaussian_schell_beam,
    create_lognormal_size,
    create_mixed_orientation,
    create_orientation_distribution,
    create_thermionic_beam,
    create_trivial_distribution,
    discretize_orientation,
    discretize_orientation_static,
    discretize_size_distribution,
    integrate_over_orientation,
    orientation_to_distribution,
    reduction_mode_from_coherence_length,
    size_to_distribution,
)
from .inout_types import FrameMetadata
from .recon_types import (
    RECIPE_DEVIATION_SCHEMA_VERSION,
    DistributionAxisSpec,
    LaplaceUncertainty,
    OrientationFitResult,
    PosteriorSamples,
    RecipeDeviationReport,
    ReconProblem,
    ReconResult,
    create_crystal_displacement_axis_spec,
    create_distribution_axis_spec,
)
from .rheed_types import (
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
    identify_surface_atoms,
)
from .simulation_params import (
    BeamSpec,
    RenderParams,
    SurfaceCTRParams,
    create_beam_spec,
)

__all__: list[str] = [
    "AMU_TO_KG",
    "BOLTZMANN_CONSTANT_JK",
    "ELEMENTARY_CHARGE_C",
    "ELECTRON_MASS_KG",
    "H_OVER_SQRT_2ME_ANG_VSQRT",
    "HBAR_JS",
    "M2_TO_ANG2",
    "PLANCK_CONSTANT_JS",
    "RELATIVISTIC_COEFF_PER_V",
    "SPEED_OF_LIGHT_MS",
    "beam_modes_from_electron_beam",
    "BeamSpec",
    "create_crystal_displacement_axis_spec",
    "create_crystal_structure",
    "create_beam_spec",
    "create_coherent_beam",
    "create_discrete_orientation",
    "create_distribution",
    "create_distribution_axis_spec",
    "create_electron_beam",
    "create_edge_on_slices",
    "create_ewald_data",
    "create_field_emission_beam",
    "create_gaussian_orientation",
    "create_gaussian_schell_beam",
    "create_orientation_distribution",
    "create_kirkland_parameters",
    "create_lognormal_size",
    "create_mixed_orientation",
    "create_potential_slices",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "create_thermionic_beam",
    "create_trivial_distribution",
    "create_xyz_data",
    "CrystalStructure",
    "BeamModeDistribution",
    "DetectorGeometry",
    "detector_beam_center_px",
    "detector_distance_mm",
    "detector_extent_mm",
    "detector_image_shape_px",
    "detector_pixel_size_mm",
    "detector_psf_sigma_pixels",
    "Distribution",
    "DistributionAxisSpec",
    "EdgeOnSlices",
    "ElectronBeam",
    "EwaldData",
    "FrameMetadata",
    "float_jax_image",
    "float_np_image",
    "integrate_over_orientation",
    "KinematicAxisUpdate",
    "KirklandParameters",
    "LaplaceUncertainty",
    "identify_surface_atoms",
    "int_jax_image",
    "int_np_image",
    "MultisliceAxisUpdate",
    "non_jax_number",
    "discretize_orientation",
    "discretize_orientation_static",
    "discretize_size_distribution",
    "OrientationFitResult",
    "OrientationDistribution",
    "PosteriorSamples",
    "PotentialSlices",
    "RECIPE_DEVIATION_SCHEMA_VERSION",
    "RecipeDeviationReport",
    "ReconProblem",
    "ReconResult",
    "RHEEDImage",
    "RHEEDPattern",
    "ReductionMode",
    "RenderParams",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
    "scalar_num",
    "orientation_to_distribution",
    "reduction_mode_from_coherence_length",
    "size_to_distribution",
    "SlicedCrystal",
    "SizeDistribution",
    "SurfaceConfig",
    "SurfaceCTRParams",
    "TRIVIAL",
    "TRIVIAL_DISTRIBUTION",
    "XYZData",
]
