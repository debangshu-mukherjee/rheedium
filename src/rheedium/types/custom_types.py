"""
Module: types.custom_types
--------------------------
Custom Type Aliases for holding Scalar JAX data.

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
- `float_image`:
    Type alias for 2D float array (H, W)
- `int_image`:
    Type alias for 2D integer array (H, W)
"""

from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Float, Integer, Num

scalar_float: TypeAlias = Union[float, Float[Array, " "]]
scalar_int: TypeAlias = Union[int, Integer[Array, " "]]
scalar_num: TypeAlias = Union[int, float, Num[Array, " "]]
non_jax_number: TypeAlias = Union[int, float]
float_image: TypeAlias = Float[Array, " H W"]
int_image: TypeAlias = Integer[Array, " H W"]
