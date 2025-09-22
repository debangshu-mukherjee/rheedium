"""Custom types and data structures for RHEED simulation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, RHEED patterns, and other simulation data. All types are PyTrees
that support JAX transformations and automatic differentiation.

Routine Listings
----------------
CrystalStructure : class
    JAX-compatible crystal structure with fractional and Cartesian coordinates
PotentialSlices : class
    JAX-compatible data structure for representing multislice potential data
RHEEDPattern : class
    Container for RHEED diffraction pattern data with detector points and intensities
RHEEDImage : class
    Container for RHEED image data with pixel coordinates and intensity values
XYZData : class
    A PyTree for XYZ file data with atomic positions and metadata
create_crystal_structure : function
    Factory function to create CrystalStructure instances
create_potential_slices : function
    Factory function to create PotentialSlices instances
create_rheed_pattern : function
    Factory function to create RHEEDPattern instances
create_rheed_image : function
    Factory function to create RHEEDImage instances
make_xyz_data : function
    Factory function to create XYZData instances

Type Aliases
------------
- `scalar_float`:
    Union type for scalar float values (float or JAX scalar array)
- `scalar_int`:
    Union type for scalar integer values (int or JAX scalar array)
- `scalar_num`:
    Union type for scalar numeric values (int, float, or JAX scalar array)
- `non_jax_number`:
    Union type for non-JAX numeric values (int or float)
"""

from .crystal_types import (
    CrystalStructure,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
    create_potential_slices,
    make_xyz_data,
)
from .custom_types import (
    float_image,
    int_image,
    non_jax_number,
    scalar_float,
    scalar_int,
    scalar_num,
)
from .rheed_types import (
    RHEEDImage,
    RHEEDPattern,
    create_rheed_image,
    create_rheed_pattern,
)

__all__ = [
    "CrystalStructure",
    "PotentialSlices",
    "XYZData",
    "make_xyz_data",
    "create_potential_slices",
    "create_crystal_structure",
    "RHEEDPattern",
    "RHEEDImage",
    "create_rheed_pattern",
    "create_rheed_image",
    "scalar_float",
    "scalar_int",
    "scalar_num",
    "non_jax_number",
    "float_image",
    "int_image",
]
