import jax.numpy as jnp
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int

__all__ = ["RHEEDPattern"]


@register_pytree_node_class
class RHEEDPattern(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for representing RHEED patterns.

    Attributes
    ----------
    - `G_indices` (Int[Array, "*"]):
        Indices of reciprocal-lattice vectors that satisfy reflection
    - `k_out` (Float[Array, "M 3"]):
        Outgoing wavevectors (in 1/Å) for those reflections
    - `detector_points` (Float[Array, "M 2"]):
        (Y, Z) coordinates on the detector plane, in Ångstroms.
    - `intensities` (Float[Array, "M"]):
        Intensities for each reflection.
    """

    G_indices: Int[Array, "*"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]

    def tree_flatten(self):
        return (
            (self.G_indices, self.k_out, self.detector_points, self.intensities),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
