import jax.numpy as jnp
from beartype.typing import NamedTuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int

from rheedium.types import scalar_float, scalar_num

__all__ = ["RHEEDPattern", "RHEEDImage"]


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


@register_pytree_node_class
class RHEEDImage(NamedTuple):
    """
    Description
    -----------
    A PyTree for representing an experimental RHEED image.

    Attributes
    ----------
    - `img_array` (Float[Array, "H W"]):
        The image in 2D array format.
    - `incoming_angle` (scalar_float):
        The angle of the incoming electron beam in degrees.
    - `calibration` (Union[Float[Array, "2"], scalar_float]):
        Calibration factor for the image, either as a 2D array or a scalar.
        If scalar, then both the X and Y axes have the same calibration.
    - `electron_wavelength` (scalar_float):
        The wavelength of the electrons in Ångstroms.
    - `detector_distance` (scalar_float):
        The distance from the sample to the detector in Ångstroms.
    """

    img_array: Float[Array, "H W"]
    incoming_angle: scalar_float
    calibration: Union[Float[Array, "2"], scalar_float]
    electron_wavelength: scalar_float
    detector_distance: scalar_num

    def tree_flatten(self):
        return (
            (
                self.img_array,
                self.incoming_angle,
                self.calibration,
                self.electron_wavelength,
                self.detector_distance,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
