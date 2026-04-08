"""Reusable numerical and workflow tools for rheedium.

This package is the shared tool chest for the rest of the codebase:
special functions, quadrature helpers, JAX wrappers, and distributed
array utilities all live here.
"""

from .parallel import shard_array
from .quadrature import gauss_hermite_nodes_weights
from .special import bessel_k0, bessel_k1, bessel_kv
from .wrappers import jax_safe

__all__: list[str] = [
    "bessel_k0",
    "bessel_k1",
    "bessel_kv",
    "gauss_hermite_nodes_weights",
    "jax_safe",
    "shard_array",
]
