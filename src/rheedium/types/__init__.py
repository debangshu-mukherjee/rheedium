"""
Module: types
-------------
Custom types and data structures for RHEED simulation.

Classes
-------
- From `crystal_types.py` submodule:
    - `CrystalStructure`:
        JAX-compatible crystal structure with fractional and Cartesian coordinates
- From `rheed_types.py` submodule:
    - `RHEEDPattern`:
        Container for RHEED diffraction pattern data with detector points and intensities
    - `RHEEDImage`:
        Container for RHEED image data with pixel coordinates and intensity values

Functions
---------
- From `crystal_types.py` submodule:
    - `create_crystal_structure`:
        Factory function to create CrystalStructure instances
- From `rheed_types.py` submodule:
    - `create_rheed_pattern`:
        Factory function to create RHEEDPattern instances
    - `create_rheed_image`:
        Factory function to create RHEEDImage instances

Type Aliases
------------
- From `custom_types.py` submodule:
    - `scalar_float`:
        Union type for scalar float values (float or JAX scalar array)
    - `scalar_int`:
        Union type for scalar integer values (int or JAX scalar array)
    - `scalar_num`:
        Union type for scalar numeric values (int, float, or JAX scalar array)
    - `non_jax_number`:
        Union type for non-JAX numeric values (int or float)
"""

from .crystal_types import CrystalStructure, create_crystal_structure
from .custom_types import non_jax_number, scalar_float, scalar_int, scalar_num
from .rheed_types import (RHEEDImage, RHEEDPattern, create_rheed_image,
                          create_rheed_pattern)

__all__ = [
    "CrystalStructure",
    "create_crystal_structure",
    "RHEEDPattern",
    "RHEEDImage",
    "create_rheed_pattern",
    "create_rheed_image",
    "scalar_float",
    "scalar_int",
    "scalar_num",
    "non_jax_number",
]
