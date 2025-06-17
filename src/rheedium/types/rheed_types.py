"""
Module: types.rheed_types
-------------------------
Data structures and factory functions for RHEED pattern and image representation.

Classes
-------
- `RHEEDPattern`:
    Container for RHEED diffraction pattern data with detector points and intensities
- `RHEEDImage`:
    Container for RHEED image data with pixel coordinates and intensity values

Functions
---------
- `create_rheed_pattern`:
    Factory function to create RHEEDPattern instances with data validation
- `create_rheed_image`:
    Factory function to create RHEEDImage instances with data validation
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int

from rheedium.types import scalar_float, scalar_num

__all__ = ["RHEEDPattern", "RHEEDImage", "create_rheed_pattern", "create_rheed_image"]


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


@jaxtyped(typechecker=beartype)
def create_rheed_pattern(
    G_indices: Int[Array, "*"],
    k_out: Float[Array, "M 3"],
    detector_points: Float[Array, "M 2"],
    intensities: Float[Array, "M"],
) -> RHEEDPattern:
    """
    Description
    -----------
    Factory function to create a RHEEDPattern instance with data validation.

    Parameters
    ----------
    - `G_indices` (Int[Array, "*"]):
        Indices of reciprocal-lattice vectors that satisfy reflection
    - `k_out` (Float[Array, "M 3"]):
        Outgoing wavevectors (in 1/Å) for those reflections
    - `detector_points` (Float[Array, "M 2"]):
        (Y, Z) coordinates on the detector plane, in Ångstroms
    - `intensities` (Float[Array, "M"]):
        Intensities for each reflection

    Returns
    -------
    - `pattern` (RHEEDPattern):
        Validated RHEED pattern instance

    Raises
    ------
    - ValueError:
        If array shapes are inconsistent or data is invalid

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate array shapes:
        - Check k_out has shape (M, 3)
        - Check detector_points has shape (M, 2)
        - Check intensities has shape (M,)
        - Check G_indices has length M
    - Validate data:
        - Ensure intensities are non-negative
        - Ensure k_out vectors are non-zero
        - Ensure detector points are finite
    - Create and return RHEEDPattern instance
    """
    G_indices = jnp.asarray(G_indices, dtype=jnp.int32)
    k_out = jnp.asarray(k_out, dtype=jnp.float64)
    detector_points = jnp.asarray(detector_points, dtype=jnp.float64)
    intensities = jnp.asarray(intensities, dtype=jnp.float64)

    M = k_out.shape[0]
    if k_out.shape != (M, 3):
        raise ValueError(f"k_out must have shape (M, 3), got {k_out.shape}")
    if detector_points.shape != (M, 2):
        raise ValueError(
            f"detector_points must have shape (M, 2), got {detector_points.shape}"
        )
    if intensities.shape != (M,):
        raise ValueError(f"intensities must have shape (M,), got {intensities.shape}")
    if G_indices.shape[0] != M:
        raise ValueError(f"G_indices must have length M, got {G_indices.shape[0]}")

    if jnp.any(intensities < 0):
        raise ValueError("Intensities must be non-negative")
    if jnp.any(jnp.linalg.norm(k_out, axis=1) == 0):
        raise ValueError("k_out vectors must be non-zero")
    if jnp.any(~jnp.isfinite(detector_points)):
        raise ValueError("Detector points must be finite")

    return RHEEDPattern(
        G_indices=G_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )


@jaxtyped(typechecker=beartype)
def create_rheed_image(
    img_array: Float[Array, "H W"],
    incoming_angle: scalar_float,
    calibration: Union[Float[Array, "2"], scalar_float],
    electron_wavelength: scalar_float,
    detector_distance: scalar_num,
) -> RHEEDImage:
    """
    Description
    -----------
    Factory function to create a RHEEDImage instance with data validation.

    Parameters
    ----------
    - `img_array` (Float[Array, "H W"]):
        The image in 2D array format
    - `incoming_angle` (scalar_float):
        The angle of the incoming electron beam in degrees
    - `calibration` (Union[Float[Array, "2"], scalar_float]):
        Calibration factor for the image, either as a 2D array or a scalar
    - `electron_wavelength` (scalar_float):
        The wavelength of the electrons in Ångstroms
    - `detector_distance` (scalar_num):
        The distance from the sample to the detector in Ångstroms

    Returns
    -------
    - `image` (RHEEDImage):
        Validated RHEED image instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate image array:
        - Check it's 2D
        - Ensure all values are finite
        - Ensure all values are non-negative
    - Validate parameters:
        - Check incoming_angle is between 0 and 90 degrees
        - Check electron_wavelength is positive
        - Check detector_distance is positive
    - Validate calibration:
        - If scalar, ensure it's positive
        - If array, ensure shape is (2,) and all values are positive
    - Create and return RHEEDImage instance
    """
    img_array = jnp.asarray(img_array, dtype=jnp.float64)
    incoming_angle = jnp.asarray(incoming_angle, dtype=jnp.float64)
    calibration = jnp.asarray(calibration, dtype=jnp.float64)
    electron_wavelength = jnp.asarray(electron_wavelength, dtype=jnp.float64)
    detector_distance = jnp.asarray(detector_distance, dtype=jnp.float64)

    if img_array.ndim != 2:
        raise ValueError(f"img_array must be 2D, got shape {img_array.shape}")
    if jnp.any(~jnp.isfinite(img_array)):
        raise ValueError("Image array must contain only finite values")
    if jnp.any(img_array < 0):
        raise ValueError("Image array must contain only non-negative values")

    if not (0 <= incoming_angle <= 90):
        raise ValueError("incoming_angle must be between 0 and 90 degrees")
    if electron_wavelength <= 0:
        raise ValueError("electron_wavelength must be positive")
    if detector_distance <= 0:
        raise ValueError("detector_distance must be positive")

    if calibration.ndim == 0:
        if calibration <= 0:
            raise ValueError("calibration scalar must be positive")
    else:
        if calibration.shape != (2,):
            raise ValueError(
                f"calibration array must have shape (2,), got {calibration.shape}"
            )
        if jnp.any(calibration <= 0):
            raise ValueError("calibration array must contain only positive values")

    return RHEEDImage(
        img_array=img_array,
        incoming_angle=incoming_angle,
        calibration=calibration,
        electron_wavelength=electron_wavelength,
        detector_distance=detector_distance,
    )
