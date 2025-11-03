"""Custom types and data structures for RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, RHEED patterns, and other simulation data. All types are PyTrees
that support JAX transformations and automatic differentiation.

Routine Listings
----------------
create_crystal_structure : function
    Factory function to create CrystalStructure instances
create_potential_slices : function
    Factory function to create PotentialSlices instances
create_rheed_image : function
    Factory function to create RHEEDImage instances
create_rheed_pattern : function
    Factory function to create RHEEDPattern instances
create_xyz_data : function
    Factory function to create XYZData instances
CrystalStructure : PyTree
    JAX-compatible crystal structure with fractional and Cartesian coordinates
PotentialSlices : PyTree
    JAX-compatible data structure for representing multislice potential data
RHEEDImage : PyTree
    Container for RHEED image data with pixel coordinates and intensity values
RHEEDPattern : PyTree
    Container for RHEED diffraction pattern data with detector points and
    intensities.
scalar_bool : TypeAlias
    Union type for scalar boolean values (bool or JAX scalar array)
scalar_float : TypeAlias
    Union type for scalar float values (float or JAX scalar array)
scalar_int : TypeAlias
    Union type for scalar integer values (int or JAX scalar array)
scalar_num : TypeAlias
    Union type for scalar numeric values (int, float, or JAX scalar array)
non_jax_number : TypeAlias
    Union type for non-JAX numeric values (int or float)
XYZData : PyTree
    A PyTree for XYZ file data with atomic positions and metadata

Notes
-----
Every PyTree has a corresponding factory function to create the instance. This
is because beartype does not support type checking of dataclasses.
"""

from .crystal_types import (
    CrystalStructure,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
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
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    bulk_to_slice,
    create_rheed_image,
    create_rheed_pattern,
    create_sliced_crystal,
)

__all__ = [
    "bulk_to_slice",
    "create_crystal_structure",
    "create_potential_slices",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "create_xyz_data",
    "CrystalStructure",
    "float_image",
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
    "XYZData",
]
