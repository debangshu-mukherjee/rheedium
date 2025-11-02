"""Custom type aliases for scalar JAX data.

Extended Summary
----------------
This module defines type aliases for scalar values and arrays that are
compatible with both standard Python types and JAX arrays. These types
facilitate type checking and ensure consistency across the codebase.

Routine Listings
----------------
scalar_float : TypeAlias
    Union type for scalar float values (float or JAX scalar array)
scalar_int : TypeAlias
    Union type for scalar integer values (int or JAX scalar array)
scalar_bool : TypeAlias
    Union type for scalar boolean values (bool or JAX scalar array)
scalar_num : TypeAlias
    Union type for scalar numeric values (int, float, or JAX scalar array)
non_jax_number : TypeAlias
    Union type for non-JAX numeric values (int or float)
float_image : TypeAlias
    Type alias for 2D float array (H, W)
int_image : TypeAlias
    Type alias for 2D integer array (H, W)

Notes
-----
These type aliases are used throughout the library to ensure type safety
and compatibility with JAX transformations.
"""

from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Bool, Float, Integer, Num

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_bool: TypeAlias = Union[bool, Bool[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]
float_image: TypeAlias = Float[Array, "H W"]
int_image: TypeAlias = Integer[Array, "H W"]
