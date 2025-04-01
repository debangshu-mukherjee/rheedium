from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Float, Integer, Num

__all__ = ["scalar_float", "scalar_int", "scalar_num", "non_jax_number"]

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]
