"""Custom types and data structures for RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, RHEED patterns, and other simulation data. All types are PyTrees
that support JAX transformations and automatic differentiation.

Routine Listings
----------------
:class:`CrystalStructure`
    JAX-compatible crystal structure with fractional and Cartesian coordinates.
:class:`DetectorGeometry`
    Configuration for RHEED detector geometry (tilt, curvature, offsets).
:class:`ElectronBeam`
    Complete specification of an electron beam for RHEED simulation.
:class:`EwaldData`
    Angle-independent Ewald sphere data for RHEED simulation.
:class:`KirklandParameters`
    Structured Kirkland coefficients for one element.
:class:`PotentialSlices`
    JAX-compatible data structure for representing multislice potential data.
:class:`RHEEDImage`
    Container for RHEED image data with pixel coordinates and intensity values.
:class:`RHEEDPattern`
    Container for RHEED diffraction pattern data with detector points and
    intensities.
:class:`SlicedCrystal`
    JAX-compatible crystal structure sliced for multislice simulation.
:class:`SurfaceConfig`
    Configuration for surface atom identification method and parameters.
:class:`XYZData`
    A PyTree for XYZ file data with atomic positions and metadata.
:func:`create_crystal_structure`
    Factory function to create CrystalStructure instances.
:func:`create_electron_beam`
    Factory function to create ElectronBeam instances.
:func:`create_ewald_data`
    Factory function to create EwaldData instances.
:func:`create_kirkland_parameters`
    Factory function to create KirklandParameters instances.
:func:`create_potential_slices`
    Factory function to create PotentialSlices instances.
:func:`create_rheed_image`
    Factory function to create RHEEDImage instances.
:func:`create_rheed_pattern`
    Factory function to create RHEEDPattern instances.
:func:`create_sliced_crystal`
    Factory function to create SlicedCrystal instances.
:func:`create_xyz_data`
    Factory function to create XYZData instances.
:func:`identify_surface_atoms`
    Identify surface atoms using configurable methods.
:obj:`float_image`
    Type alias for float-valued 2D image arrays.
:obj:`int_image`
    Type alias for integer-valued 2D image arrays.
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
    EwaldData,
    KirklandParameters,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
    create_ewald_data,
    create_kirkland_parameters,
    create_potential_slices,
    create_xyz_data,
)
from .custom_types import (
    float_image,
    int_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)
from .rheed_types import (
    DetectorGeometry,
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
    identify_surface_atoms,
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
    "create_crystal_structure",
    "create_electron_beam",
    "create_ewald_data",
    "create_kirkland_parameters",
    "create_potential_slices",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "create_xyz_data",
    "CrystalStructure",
    "DetectorGeometry",
    "ElectronBeam",
    "EwaldData",
    "float_image",
    "KirklandParameters",
    "identify_surface_atoms",
    "int_image",
    "non_jax_number",
    "PotentialSlices",
    "RHEEDImage",
    "RHEEDPattern",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
    "scalar_num",
    "SlicedCrystal",
    "SurfaceConfig",
    "XYZData",
]
