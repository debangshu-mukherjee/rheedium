from functools import partial
from typing import Any, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
# from typeguard import typechecked as typechecker
from jax import lax
from jaxtyping import Array, Complex, Float, Int, jaxtyped

jax.config.update("jax_enable_x64", True)
