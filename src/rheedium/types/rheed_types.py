import jax.numpy as jnp
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int

__all__ = ["RHEEDPattern"]

@register_pytree_node_class
class RHEEDPattern(NamedTuple):
    G_indices: Int[Array, "*"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]

    def tree_flatten(self):
        return ((self.G_indices, self.k_out, self.detector_points, self.intensities), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)