"""Data structures and factory functions for RHEED pattern representation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing RHEED
patterns and images. All structures follow a JAX-compatible validation pattern
that ensures data integrity at compile time.

Routine Listings
----------------
RHEEDPattern : PyTree
    Container for RHEED diffraction pattern data with detector points and
    intensities.
RHEEDImage : PyTree
    Container for RHEED image data with pixel coordinates and intensity values
create_rheed_pattern : function
    Factory function to create RHEEDPattern instances with data validation
create_rheed_image : function
    Factory function to create RHEEDImage instances with data validation

Notes
-----
JAX Validation Pattern:

1. Use `jax.lax.cond` for validation instead of Python `if` statements
2. Validation happens at JIT compilation time, not runtime
3. Validation functions don't return modified data, they ensure original data
    is valid.
4. Use `lax.stop_gradient(lax.cond(False, ...))` in false branches to cause
   compilation errors
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, jaxtyped

from .custom_types import scalar_float, scalar_num


@register_pytree_node_class
class RHEEDPattern(NamedTuple):
    """JAX-compatible RHEED diffraction pattern data structure.

    This PyTree represents a RHEED diffraction pattern containing reflection
    data including reciprocal lattice indices, outgoing wavevectors, detector
    coordinates, and intensity values for electron diffraction analysis.

    Attributes
    ----------
    G_indices : Int[Array, "N"]
        Indices of reciprocal-lattice vectors that satisfy reflection
        conditions. Variable length array of integer indices.
    k_out : Float[Array, "M 3"]
        Outgoing wavevectors in 1/Å for reflections. Shape (M, 3) where M
        is the number of reflections and each row contains [kx, ky, kz]
        components.
    detector_points : Float[Array, "M 2"]
        Detector coordinates (Y, Z) on the detector plane in Ångstroms.
        Shape (M, 2) where each row contains [y, z] coordinates.
    intensities : Float[Array, "M"]
        Intensity values for each reflection. Shape (M,) with non-negative
        intensity values.
    This class is registered as a PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. All data is immutable and
    stored in JAX arrays for efficient RHEED pattern analysis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create RHEED pattern data
    >>> G_indices = jnp.array([1, 2, 3])
    >>> k_out = jnp.array([[1.0, 0.0, 0.5], [2.0, 0.0, 1.0], [3.0, 0.0, 1.5]])
    >>> detector_points = jnp.array([[10.0, 5.0], [20.0, 10.0], [30.0, 15.0]])
    >>> intensities = jnp.array([100.0, 80.0, 60.0])
    >>> pattern = rh.types.create_rheed_pattern(
    ...     G_indices=G_indices,
    ...     k_out=k_out,
    ...     detector_points=detector_points,
    ...     intensities=intensities
    ... )
    """

    G_indices: Int[Array, "N"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Int[Array, "N"],
            Float[Array, "M 3"],
            Float[Array, "M 2"],
            Float[Array, "M"],
        ],
        None,
    ]:
        """Flatten the PyTree into a tuple of arrays."""
        return (
            (
                self.G_indices,
                self.k_out,
                self.detector_points,
                self.intensities,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[
            Int[Array, "N"],
            Float[Array, "M 3"],
            Float[Array, "M 2"],
            Float[Array, "M"],
        ],
    ) -> "RHEEDPattern":
        """Unflatten the PyTree into a RHEEDPattern instance."""
        del aux_data
        return cls(*children)


@register_pytree_node_class
class RHEEDImage(NamedTuple):
    """JAX-compatible experimental RHEED image data structure.

    This PyTree represents an experimental RHEED image with associated
    experimental parameters including beam geometry, detector calibration,
    and electron beam properties for quantitative RHEED analysis.

    Attributes
    ----------
    img_array : Float[Array, "H W"]
        2D image array with shape (height, width) containing pixel intensity
        values. Non-negative finite values.
    incoming_angle : scalar_float
        Angle of the incoming electron beam in degrees, typically between
        0 and 90 degrees for grazing incidence geometry.
    calibration : Union[Float[Array, "2"], scalar_float]
        Calibration factor for converting pixels to physical units. Either
        a scalar (same calibration for both axes) or array of shape (2,)
        with separate [x, y] calibrations in appropriate units per pixel.
    electron_wavelength : scalar_float
        Wavelength of the electrons in Ångstroms. Determines the diffraction
        geometry and resolution.
    detector_distance : scalar_float
        Distance from the sample to the detector in Ångstroms. Affects the
        angular resolution and reciprocal space mapping.

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. All data is immutable for
    functional programming patterns and efficient image processing.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create RHEED image with experimental parameters
    >>> image = jnp.ones((256, 512))  # 256x512 pixel RHEED image
    >>> rheed_img = rh.types.create_rheed_image(
    ...     img_array=image,
    ...     incoming_angle=2.0,  # 2 degree grazing angle
    ...     calibration=0.01,    # 0.01 units per pixel
    ...     electron_wavelength=0.037,  # 10 keV electrons
    ...     detector_distance=1000.0     # 1000 Å to detector
    ... )
    """

    img_array: Float[Array, "H W"]
    incoming_angle: scalar_float
    calibration: Union[Float[Array, "2"], scalar_float]
    electron_wavelength: scalar_float
    detector_distance: scalar_num

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "H W"],
            scalar_float,
            Union[Float[Array, "2"], scalar_float],
            scalar_float,
            scalar_num,
        ],
        None,
    ]:
        """Flatten the PyTree into a tuple of arrays."""
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
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[
            Float[Array, "H W"],
            scalar_float,
            Union[Float[Array, "2"], scalar_float],
            scalar_float,
            scalar_num,
        ],
    ) -> "RHEEDImage":
        """Unflatten the PyTree into a RHEEDImage instance."""
        del aux_data
        return cls(*children)


@jaxtyped(typechecker=beartype)
def create_rheed_pattern(
    g_indices: Int[Array, "N"],
    k_out: Float[Array, "M 3"],
    detector_points: Float[Array, "M 2"],
    intensities: Float[Array, "M"],
) -> RHEEDPattern:
    """Create a RHEEDPattern instance with data validation.

    Parameters
    ----------
    g_indices : Int[Array, "N"]
        Indices of reciprocal-lattice vectors that satisfy reflection.
    k_out : Float[Array, "M 3"]
        Outgoing wavevectors (in 1/Å) for those reflections.
    detector_points : Float[Array, "M 2"]
        (Y, Z) coordinates on the detector plane, in Ångstroms.
    intensities : Float[Array, "M"]
        Intensities for each reflection.

    Returns
    -------
    validated_rheed_pattern : RHEEDPattern
        Validated RHEED pattern instance.

    Notes
    -----
    - Convert inputs to JAX arrays.
    - Validate array shapes: check k_out has shape (M, 3), detector_points
      has shape (M, 2), intensities has shape (M,), and g_indices has length M
    - Validate data: ensure intensities are non-negative, k_out vectors are
      non-zero, and detector points are finite
    - Create and return RHEEDPattern instance
    """
    g_indices: Int[Array, "nn"] = jnp.asarray(g_indices, dtype=jnp.int32)
    k_out: Float[Array, "mm 3"] = jnp.asarray(k_out, dtype=jnp.float64)
    detector_points: Float[Array, "mm 2"] = jnp.asarray(
        detector_points, dtype=jnp.float64
    )
    intensities: Float[Array, "mm"] = jnp.asarray(
        intensities, dtype=jnp.float64
    )

    def _validate_and_create() -> RHEEDPattern:
        """Validate and create a RHEEDPattern instance."""
        mm: int = k_out.shape[0]

        def _check_k_out_shape() -> Float[Array, "mm 3"]:
            """Check the shape of the k_out array."""
            return lax.cond(
                k_out.shape == (mm, 3),
                lambda: k_out,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: k_out, lambda: k_out)
                ),
            )

        def _check_detector_shape() -> Float[Array, "mm 2"]:
            """Check the shape of the detector_points array."""
            return lax.cond(
                detector_points.shape == (mm, 2),
                lambda: detector_points,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: detector_points, lambda: detector_points
                    )
                ),
            )

        def _check_intensities_shape() -> Float[Array, "mm"]:
            """Check the shape of the intensities array."""
            return lax.cond(
                intensities.shape == (mm,),
                lambda: intensities,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: intensities, lambda: intensities)
                ),
            )

        def _check_g_indices_length() -> Int[Array, "nn"]:
            """Check the length of the g_indices array."""
            return lax.cond(
                g_indices.shape[0] == mm,
                lambda: g_indices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: g_indices, lambda: g_indices)
                ),
            )

        def _check_intensities_positive() -> Float[Array, "mm"]:
            """Check the intensities are positive."""
            return lax.cond(
                jnp.all(intensities >= 0),
                lambda: intensities,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: intensities, lambda: intensities)
                ),
            )

        def _check_k_out_nonzero() -> Float[Array, "mm 3"]:
            """Check the k_out vectors are non-zero."""
            return lax.cond(
                jnp.all(jnp.linalg.norm(k_out, axis=1) > 0),
                lambda: k_out,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: k_out, lambda: k_out)
                ),
            )

        def _check_detector_finite() -> Float[Array, "mm 2"]:
            """Check the detector points are finite."""
            return lax.cond(
                jnp.all(jnp.isfinite(detector_points)),
                lambda: detector_points,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: detector_points, lambda: detector_points
                    )
                ),
            )

        _check_k_out_shape()
        _check_detector_shape()
        _check_intensities_shape()
        _check_g_indices_length()
        _check_intensities_positive()
        _check_k_out_nonzero()
        _check_detector_finite()

        return RHEEDPattern(
            G_indices=g_indices,
            k_out=k_out,
            detector_points=detector_points,
            intensities=intensities,
        )

    validated_rheed_pattern: RHEEDPattern = _validate_and_create()
    return validated_rheed_pattern


@jaxtyped(typechecker=beartype)
def create_rheed_image(
    img_array: Float[Array, "H W"],
    incoming_angle: scalar_float,
    calibration: Union[Float[Array, "2"], scalar_float],
    electron_wavelength: scalar_float,
    detector_distance: scalar_num,
) -> RHEEDImage:
    """Create a RHEEDImage instance with data validation.

    Parameters
    ----------
    img_array : Float[Array, "H W"]
        The image in 2D array format.
    incoming_angle : scalar_float
        The angle of the incoming electron beam in degrees.
    calibration : Union[Float[Array, "2"], scalar_float]
        Calibration factor for the image, either as a 2D array or a scalar.
    electron_wavelength : scalar_float
        The wavelength of the electrons in Ångstroms.
    detector_distance : scalar_num
        The distance from the sample to the detector in Ångstroms.

    Returns
    -------
    validated_rheed_image : RHEEDImage
        Validated RHEED image instance.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Convert inputs to JAX arrays
    2. Validate image array: check it's 2D, all values are finite and
       non-negative.
    3. Validate parameters: check incoming_angle is between 0 and 90 degrees,
       electron_wavelength is positive, and detector_distance is positive
    4. Validate calibration: if scalar, ensure it's positive; if array, ensure
       shape is (2,) and all values are positive
    5. Create and return RHEEDImage instance
    """
    img_array: Float[Array, "H W"] = jnp.asarray(img_array, dtype=jnp.float64)
    incoming_angle: Float[Array, ""] = jnp.asarray(
        incoming_angle, dtype=jnp.float64
    )
    calibration: Union[Float[Array, "2"], Float[Array, ""]] = jnp.asarray(
        calibration, dtype=jnp.float64
    )
    electron_wavelength: Float[Array, ""] = jnp.asarray(
        electron_wavelength, dtype=jnp.float64
    )
    detector_distance: Float[Array, ""] = jnp.asarray(
        detector_distance, dtype=jnp.float64
    )
    max_angle: Int[Array, ""] = jnp.asarray(90, dtype=jnp.int32)
    image_dimensions: int = 2

    def _validate_and_create() -> RHEEDImage:
        """Validate and create a RHEEDImage instance."""

        def _check_2d() -> Float[Array, "H W"]:
            """Check the image array is 2D."""
            return lax.cond(
                img_array.ndim == image_dimensions,
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def _check_finite() -> Float[Array, "H W"]:
            """Check the image array is finite."""
            return lax.cond(
                jnp.all(jnp.isfinite(img_array)),
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def _check_nonnegative() -> Float[Array, "H W"]:
            """Check the image array is non-negative."""
            return lax.cond(
                jnp.all(img_array >= 0),
                lambda: img_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: img_array, lambda: img_array)
                ),
            )

        def _check_angle() -> Float[Array, ""]:
            """Check the incoming angle is between 0 and 90 degrees."""
            return lax.cond(
                jnp.logical_and(
                    incoming_angle >= 0, incoming_angle <= max_angle
                ),
                lambda: incoming_angle,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: incoming_angle, lambda: incoming_angle
                    )
                ),
            )

        def _check_wavelength() -> Float[Array, ""]:
            """Check the electron wavelength is positive."""
            return lax.cond(
                electron_wavelength > 0,
                lambda: electron_wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: electron_wavelength,
                        lambda: electron_wavelength,
                    )
                ),
            )

        def _check_distance() -> Float[Array, ""]:
            """Check the detector distance is positive."""
            return lax.cond(
                detector_distance > 0,
                lambda: detector_distance,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: detector_distance,
                        lambda: detector_distance,
                    )
                ),
            )

        def _check_calibration() -> (
            Union[Float[Array, "2"], Float[Array, ""]]
        ):
            """Check the calibration is positive."""

            def _check_scalar_cal() -> Float[Array, ""]:
                """Check the calibration is positive."""
                return lax.cond(
                    calibration > 0,
                    lambda: calibration,
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False, lambda: calibration, lambda: calibration
                        )
                    ),
                )

            def _check_array_cal() -> Float[Array, "2"]:
                """Check the calibration is positive."""
                return lax.cond(
                    jnp.logical_and(
                        calibration.shape == (2,), jnp.all(calibration > 0)
                    ),
                    lambda: calibration,
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False, lambda: calibration, lambda: calibration
                        )
                    ),
                )

            return lax.cond(
                calibration.ndim == 0, _check_scalar_cal, _check_array_cal
            )

        _check_2d()
        _check_finite()
        _check_nonnegative()
        _check_angle()
        _check_wavelength()
        _check_distance()
        _check_calibration()

        return RHEEDImage(
            img_array=img_array,
            incoming_angle=incoming_angle,
            calibration=calibration,
            electron_wavelength=electron_wavelength,
            detector_distance=detector_distance,
        )

    validated_rheed_image: RHEEDImage = _validate_and_create()
    return validated_rheed_image
