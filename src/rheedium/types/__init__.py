"""Custom types and data structures for RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, RHEED patterns, and other simulation data. All types are PyTrees
that support JAX transformations and automatic differentiation.

Routine Listings
----------------
:func:`bulk_to_slice`
    Convert bulk CrystalStructure to SlicedCrystal for multislice simulation.
:func:`create_crystal_structure`
    Factory function to create CrystalStructure instances.
:func:`create_ewald_data`
    Factory function to create EwaldData instances.
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
:class:`CrystalStructure`
    JAX-compatible crystal structure with fractional and Cartesian coordinates.
:class:`DetectorGeometry`
    Configuration for RHEED detector geometry (tilt, curvature, offsets).
:class:`EwaldData`
    Angle-independent Ewald sphere data for RHEED simulation.
:obj:`float_image`
    Type alias for float-valued 2D image arrays.
:obj:`int_image`
    Type alias for integer-valued 2D image arrays.
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:class:`PotentialSlices`
    JAX-compatible data structure for representing multislice potential data.
:class:`RHEEDImage`
    Container for RHEED image data with pixel coordinates and intensity values.
:class:`RHEEDPattern`
    Container for RHEED diffraction pattern data with detector points and
    intensities.
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX scalar array).
:class:`SlicedCrystal`
    JAX-compatible crystal structure sliced for multislice simulation.
:class:`SurfaceConfig`
    Configuration for surface atom identification method and parameters.
:class:`XYZData`
    A PyTree for XYZ file data with atomic positions and metadata.

Notes
-----
Every PyTree has a corresponding factory function to create the instance. This
is because beartype does not support type checking of dataclasses.
"""

from .crystal_types import (
    CrystalStructure,
    EwaldData,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
    create_ewald_data,
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
    bulk_to_slice,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
    identify_surface_atoms,
)

__all__ = [
    "bulk_to_slice",
    "create_crystal_structure",
    "create_ewald_data",
    "create_potential_slices",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "create_xyz_data",
    "CrystalStructure",
    "DetectorGeometry",
    "EwaldData",
    "float_image",
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
