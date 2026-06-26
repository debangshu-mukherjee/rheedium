"""Shared type aliases for typed JAX/NumPy tests.

Extended Summary
----------------
Collects the shape-aware jaxtyping aliases used by shared factories,
assertions, and individual test modules.
"""

from beartype.typing import TypeAlias
from jax import Array
from jaxtyping import Bool, Float, Float32, Int, PRNGKeyArray, PyTree
from numpy.typing import NDArray

Key: TypeAlias = PRNGKeyArray

JaxScalar: TypeAlias = Float[Array, ""]
JaxVector: TypeAlias = Float[Array, "x"]
JaxMatrix: TypeAlias = Float[Array, "x y"]
JaxTensor3: TypeAlias = Float[Array, "x y z"]
JaxAnyFloat: TypeAlias = Float[Array, "..."]

JaxF32Scalar: TypeAlias = Float32[Array, ""]
JaxF32Vector: TypeAlias = Float32[Array, "x"]
JaxF32Matrix: TypeAlias = Float32[Array, "x y"]
JaxF32Tensor3: TypeAlias = Float32[Array, "x y z"]
JaxF32Any: TypeAlias = Float32[Array, "..."]

NpScalar: TypeAlias = Float[NDArray, ""]
NpVector: TypeAlias = Float[NDArray, "x"]
NpMatrix: TypeAlias = Float[NDArray, "x y"]
NpTensor3: TypeAlias = Float[NDArray, "x y z"]
NpAnyFloat: TypeAlias = Float[NDArray, "..."]

NpF32Scalar: TypeAlias = Float32[NDArray, ""]
NpF32Vector: TypeAlias = Float32[NDArray, "x"]
NpF32Matrix: TypeAlias = Float32[NDArray, "x y"]
NpF32Tensor3: TypeAlias = Float32[NDArray, "x y z"]
NpF32Any: TypeAlias = Float32[NDArray, "..."]

JaxMask: TypeAlias = Bool[Array, "x"]
NpMask: TypeAlias = Bool[NDArray, "x"]

JaxLabels: TypeAlias = Int[Array, "x"]
NpLabels: TypeAlias = Int[NDArray, "x"]

JaxParams: TypeAlias = PyTree[Float[Array, "..."]]
JaxGrads: TypeAlias = PyTree[Float[Array, "..."]]
NpParams: TypeAlias = PyTree[Float[NDArray, "..."]]
