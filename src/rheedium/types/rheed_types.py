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

JAX Validation Pattern
---------------------
All factory functions in this codebase follow a JAX-compatible validation pattern:

1. **Use `jax.lax.cond` for validation**: Replace Python `if` statements with `lax.cond(condition, true_fn, false_fn)`
2. **Compile-time validation**: Validation happens at JIT compilation time, not runtime
3. **Side-effect validation**: Validation functions don't return modified data, they ensure original data is valid
4. **Error handling**: Use `lax.stop_gradient(lax.cond(False, ...))` in false branches to cause compilation errors

Example Pattern:
```python
def validate_and_create():
    def check_shape():
        return lax.cond(
            data.shape == expected_shape,
            lambda: data,  # Pass through if valid
            lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))  # Fail if invalid
        )

    # Execute validations (no assignment needed)
    check_shape()
    check_values()
    check_conditions()

    # Return original data (now guaranteed valid)
    return DataStructure(data=data, ...)

return validate_and_create()
```

This pattern ensures:
- JIT compatibility
- Compile-time error detection
- Zero runtime validation overhead
- Type safety through JAX's compilation system
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, jaxtyped

from .custom_types import scalar_float, scalar_num


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

    def validate_and_create():
        M = k_out.shape[0]

        def check_k_out_shape():
            return lax.cond(
                k_out.shape == (M, 3),
                lambda: k_out,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: k_out, lambda: k_out)
                ),
            )

        def check_detector_shape():
            return lax.cond(
                detector_points.shape == (M, 2),
                lambda: detector_points,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: detector_points, lambda: detector_points)
                ),
            )

        def check_intensities_shape():
            return lax.cond(
                intensities.shape == (M,),
                lambda: intensities,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: intensities, lambda: intensities)
                ),
            )

        def check_g_indices_length():
            return lax.cond(
                G_indices.shape[0] == M,
                lambda: G_indices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: G_indices, lambda: G_indices)
                ),
            )

        def check_intensities_positive():
            return lax.cond(
                jnp.all(intensities >= 0),
                lambda: intensities,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: intensities, lambda: intensities)
                ),
            )

        # Check k_out vectors non-zero
        def check_k_out_nonzero():
            return lax.cond(
                jnp.all(jnp.linalg.norm(k_out, axis=1) > 0),
                lambda: k_out,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: k_out, lambda: k_out)
                ),
            )

        def check_detector_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(detector_points)),
                lambda: detector_points,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: detector_points, lambda: detector_points)
                ),
            )

        check_k_out_shape()
        check_detector_shape()
        check_intensities_shape()
        check_g_indices_length()

        check_intensities_positive()
        check_k_out_nonzero()
        check_detector_finite()

        return RHEEDPattern(
            G_indices=G_indices,
            k_out=k_out,
            detector_points=detector_points,
            intensities=intensities,
        )

    return validate_and_create()


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

    def validate_and_create():
        def check_2d():
            return lax.cond(
                img_array.ndim == 2,
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def check_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(img_array)),
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def check_nonnegative():
            return lax.cond(
                jnp.all(img_array >= 0),
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def check_angle():
            return lax.cond(
                jnp.logical_and(incoming_angle >= 0, incoming_angle <= 90),
                lambda: incoming_angle,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: incoming_angle, lambda: incoming_angle)
                ),
            )

        def check_wavelength():
            return lax.cond(
                electron_wavelength > 0,
                lambda: electron_wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: electron_wavelength, lambda: electron_wavelength
                    )
                ),
            )

        def check_distance():
            return lax.cond(
                detector_distance > 0,
                lambda: detector_distance,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: detector_distance, lambda: detector_distance
                    )
                ),
            )

        def check_calibration():
            def check_scalar_cal():
                return lax.cond(
                    calibration > 0,
                    lambda: calibration,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: calibration, lambda: calibration)
                    ),
                )

            def check_array_cal():
                return lax.cond(
                    jnp.logical_and(
                        calibration.shape == (2,), jnp.all(calibration > 0)
                    ),
                    lambda: calibration,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: calibration, lambda: calibration)
                    ),
                )

            return lax.cond(calibration.ndim == 0, check_scalar_cal, check_array_cal)

        check_2d()
        check_finite()
        check_nonnegative()
        check_angle()
        check_wavelength()
        check_distance()
        check_calibration()

        return RHEEDImage(
            img_array=img_array,
            incoming_angle=incoming_angle,
            calibration=calibration,
            electron_wavelength=electron_wavelength,
            detector_distance=detector_distance,
        )

    return validate_and_create()
