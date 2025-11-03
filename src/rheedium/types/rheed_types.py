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
from jaxtyping import Array, Bool, Float, Int, jaxtyped

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


@register_pytree_node_class
class SlicedCrystal(NamedTuple):
    """JAX-compatible surface-oriented crystal structure for RHEED simulation.

    This PyTree represents a crystal structure that has been sliced and oriented
    for RHEED simulations. The structure contains atoms from a surface region
    extended in x and y directions to cover a large area (>100 Å typically),
    with a specified depth perpendicular to the surface.

    Attributes
    ----------
    cart_positions : Float[Array, "N 4"]
        Cartesian coordinates of atoms in the slab with atomic numbers.
        Shape (N, 4) where each row is [x, y, z, atomic_number].
        Coordinates are in Ångstroms.
    cell_lengths : Float[Array, "3"]
        Lengths of the supercell edges [a, b, c] in Ångstroms.
        These define the periodicity of the extended surface slab.
    cell_angles : Float[Array, "3"]
        Angles between supercell edges [alpha, beta, gamma] in degrees.
        Typically [90, 90, 90] for surface slabs.
    orientation : Int[Array, "3"]
        Miller indices [h, k, l] of the surface orientation.
        Example: [1, 1, 1] for a (111) surface, [0, 0, 1] for (001).
    depth : scalar_float
        Depth of the slab perpendicular to the surface in Ångstroms.
        Atoms within this depth from the surface are included.
    x_extent : scalar_float
        Lateral extent of the slab in the x-direction in Ångstroms.
        Should be >100 Å for realistic RHEED simulation.
    y_extent : scalar_float
        Lateral extent of the slab in the y-direction in Ångstroms.
        Should be >100 Å for realistic RHEED simulation.

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX
    transformations. The structure is designed specifically for RHEED simulations
    where:
    - The z-direction is perpendicular to the surface (beam grazing angle)
    - The x-y plane is the surface plane
    - Large lateral extents (x_extent, y_extent) ensure realistic coherence
    - Limited depth models the surface sensitivity of RHEED

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create a (111) surface slab
    >>> cart_positions = jnp.array([[0.0, 0.0, 0.0, 14.0],  # Si atom
    ...                              [1.0, 1.0, 0.5, 14.0]]) # Another Si
    >>> sliced = rh.types.create_sliced_crystal(
    ...     cart_positions=cart_positions,
    ...     cell_lengths=jnp.array([150.0, 150.0, 20.0]),
    ...     cell_angles=jnp.array([90.0, 90.0, 90.0]),
    ...     orientation=jnp.array([1, 1, 1]),
    ...     depth=20.0,
    ...     x_extent=150.0,
    ...     y_extent=150.0
    ... )
    """

    cart_positions: Float[Array, "N 4"]
    cell_lengths: Float[Array, "3"]
    cell_angles: Float[Array, "3"]
    orientation: Int[Array, "3"]
    depth: scalar_float
    x_extent: scalar_float
    y_extent: scalar_float

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "N 4"],
            Float[Array, "3"],
            Float[Array, "3"],
            Int[Array, "3"],
            scalar_float,
            scalar_float,
            scalar_float,
        ],
        None,
    ]:
        """Flatten the PyTree into a tuple of arrays."""
        return (
            (
                self.cart_positions,
                self.cell_lengths,
                self.cell_angles,
                self.orientation,
                self.depth,
                self.x_extent,
                self.y_extent,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[
            Float[Array, "N 4"],
            Float[Array, "3"],
            Float[Array, "3"],
            Int[Array, "3"],
            scalar_float,
            scalar_float,
            scalar_float,
        ],
    ) -> "SlicedCrystal":
        """Unflatten the PyTree into a SlicedCrystal instance."""
        del aux_data
        return cls(*children)


@jaxtyped(typechecker=beartype)
def create_sliced_crystal(
    cart_positions: Float[Array, "N 4"],
    cell_lengths: Float[Array, "3"],
    cell_angles: Float[Array, "3"],
    orientation: Int[Array, "3"],
    depth: scalar_float,
    x_extent: scalar_float,
    y_extent: scalar_float,
) -> SlicedCrystal:
    """Create a SlicedCrystal instance with data validation.

    Parameters
    ----------
    cart_positions : Float[Array, "N 4"]
        Cartesian atomic positions with atomic numbers [x, y, z, Z].
    cell_lengths : Float[Array, "3"]
        Supercell edge lengths [a, b, c] in Ångstroms.
    cell_angles : Float[Array, "3"]
        Supercell angles [alpha, beta, gamma] in degrees.
    orientation : Int[Array, "3"]
        Miller indices [h, k, l] of surface orientation.
    depth : scalar_float
        Slab depth perpendicular to surface in Ångstroms.
    x_extent : scalar_float
        Lateral extent in x-direction in Ångstroms.
    y_extent : scalar_float
        Lateral extent in y-direction in Ångstroms.

    Returns
    -------
    validated_sliced_crystal : SlicedCrystal
        Validated SlicedCrystal instance.

    Notes
    -----
    Validation checks:
    - cart_positions shape is (N, 4) with N > 0
    - cell_lengths, cell_angles, orientation have correct shapes
    - All positions are finite
    - depth, x_extent, y_extent are positive
    - x_extent and y_extent are recommended to be >= 100 Å
    - Atomic numbers (cart_positions[:, 3]) are in valid range [1, 118]
    """
    cart_positions = jnp.asarray(cart_positions, dtype=jnp.float64)
    cell_lengths = jnp.asarray(cell_lengths, dtype=jnp.float64)
    cell_angles = jnp.asarray(cell_angles, dtype=jnp.float64)
    orientation = jnp.asarray(orientation, dtype=jnp.int32)
    depth = jnp.asarray(depth, dtype=jnp.float64)
    x_extent = jnp.asarray(x_extent, dtype=jnp.float64)
    y_extent = jnp.asarray(y_extent, dtype=jnp.float64)

    def _validate_and_create() -> SlicedCrystal:
        """Validate and create a SlicedCrystal instance."""

        def _check_positions_shape() -> Float[Array, "N 4"]:
            """Check cart_positions has shape (N, 4)."""
            n_atoms = cart_positions.shape[0]
            return lax.cond(
                jnp.logical_and(
                    cart_positions.ndim == 2,
                    jnp.logical_and(
                        cart_positions.shape[1] == 4, n_atoms > 0
                    ),
                ),
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: cart_positions, lambda: cart_positions
                    )
                ),
            )

        def _check_positions_finite() -> Float[Array, "N 4"]:
            """Check all positions are finite."""
            return lax.cond(
                jnp.all(jnp.isfinite(cart_positions)),
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: cart_positions, lambda: cart_positions
                    )
                ),
            )

        def _check_atomic_numbers() -> Float[Array, "N 4"]:
            """Check atomic numbers are in valid range [1, 118]."""
            atomic_nums = cart_positions[:, 3]
            return lax.cond(
                jnp.logical_and(
                    jnp.all(atomic_nums >= 1), jnp.all(atomic_nums <= 118)
                ),
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: cart_positions, lambda: cart_positions
                    )
                ),
            )

        def _check_cell_lengths() -> Float[Array, "3"]:
            """Check cell_lengths shape and positivity."""
            return lax.cond(
                jnp.logical_and(
                    cell_lengths.shape == (3,), jnp.all(cell_lengths > 0)
                ),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def _check_cell_angles() -> Float[Array, "3"]:
            """Check cell_angles shape and validity."""
            return lax.cond(
                jnp.logical_and(
                    cell_angles.shape == (3,),
                    jnp.logical_and(
                        jnp.all(cell_angles > 0), jnp.all(cell_angles < 180)
                    ),
                ),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        def _check_orientation() -> Int[Array, "3"]:
            """Check orientation shape."""
            return lax.cond(
                orientation.shape == (3,),
                lambda: orientation,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: orientation, lambda: orientation)
                ),
            )

        def _check_depth() -> Float[Array, ""]:
            """Check depth is positive."""
            return lax.cond(
                depth > 0,
                lambda: depth,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: depth, lambda: depth)
                ),
            )

        def _check_extents() -> Tuple[Float[Array, ""], Float[Array, ""]]:
            """Check x_extent and y_extent are positive."""
            x_valid = lax.cond(
                x_extent > 0,
                lambda: x_extent,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: x_extent, lambda: x_extent)
                ),
            )
            y_valid = lax.cond(
                y_extent > 0,
                lambda: y_extent,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: y_extent, lambda: y_extent)
                ),
            )
            return x_valid, y_valid

        _check_positions_shape()
        _check_positions_finite()
        _check_atomic_numbers()
        _check_cell_lengths()
        _check_cell_angles()
        _check_orientation()
        _check_depth()
        _check_extents()

        return SlicedCrystal(
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            orientation=orientation,
            depth=depth,
            x_extent=x_extent,
            y_extent=y_extent,
        )

    validated_sliced_crystal = _validate_and_create()
    return validated_sliced_crystal


@jaxtyped(typechecker=beartype)
def bulk_to_slice(
    bulk_crystal: "CrystalStructure",
    orientation: Int[Array, "3"],
    depth: scalar_float,
    x_extent: scalar_float = 150.0,
    y_extent: scalar_float = 150.0,
) -> SlicedCrystal:
    """Transform a bulk crystal structure into a surface-oriented slab.

    This function takes a bulk crystal structure and creates a surface slab
    oriented along the specified Miller indices. The slab is extended in the
    x and y directions to create a large surface area suitable for RHEED
    simulations, with atoms selected within a specified depth from the surface.

    Parameters
    ----------
    bulk_crystal : CrystalStructure
        Bulk crystal structure from CIF file or other source.
    orientation : Int[Array, "3"]
        Miller indices [h, k, l] defining the desired surface orientation.
        Example: [1, 1, 1] for (111) surface, [0, 0, 1] for (001).
    depth : scalar_float
        Depth of atoms to include perpendicular to surface in Ångstroms.
        Typically 10-30 Å to capture surface effects.
    x_extent : scalar_float, optional
        Lateral extent in x-direction in Ångstroms. Default: 150.0 Å
        Should be >= 100 Å for realistic RHEED simulations.
    y_extent : scalar_float, optional
        Lateral extent in y-direction in Ångstroms. Default: 150.0 Å
        Should be >= 100 Å for realistic RHEED simulations.

    Returns
    -------
    sliced_crystal : SlicedCrystal
        Surface-oriented crystal slab with transformed coordinates.

    Algorithm
    ---------
    1. Build rotation matrix to align [hkl] direction with z-axis:
       - Convert Miller indices to reciprocal lattice vector
       - Calculate rotation matrix R that maps this vector to [0, 0, 1]
    2. Transform all atomic positions using rotation matrix
    3. Create supercell by replicating atoms in x and y directions:
       - Determine number of repetitions needed to cover x_extent, y_extent
       - Generate all combinations of translations
       - Apply translations to create supercell
    4. Filter atoms within depth range: 0 <= z <= depth
    5. Center the slab so z=0 is at the bottom surface
    6. Calculate new cell parameters for the supercell
    7. Return SlicedCrystal with transformed coordinates

    Notes
    -----
    - The transformation preserves atomic types and relative positions
    - The resulting structure has z as the surface normal
    - Periodic boundary conditions apply in x and y directions
    - The depth direction (z) is typically non-periodic for surface slabs

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Load bulk structure
    >>> bulk = rh.inout.parse_cif("SrTiO3.cif")
    >>>
    >>> # Create (111) surface slab
    >>> slab = rh.types.bulk_to_slice(
    ...     bulk_crystal=bulk,
    ...     orientation=jnp.array([1, 1, 1]),
    ...     depth=20.0,
    ...     x_extent=150.0,
    ...     y_extent=150.0
    ... )
    """
    from ..ucell import (
        build_cell_vectors,
        reciprocal_lattice_vectors,
    )

    orientation = jnp.asarray(orientation, dtype=jnp.int32)
    depth = jnp.asarray(depth, dtype=jnp.float64)
    x_extent = jnp.asarray(x_extent, dtype=jnp.float64)
    y_extent = jnp.asarray(y_extent, dtype=jnp.float64)

    # Step 1: Build real-space lattice vectors
    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *bulk_crystal.cell_lengths, *bulk_crystal.cell_angles
    )

    # Step 2: Get reciprocal lattice vectors
    recip_vecs: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        *bulk_crystal.cell_lengths, *bulk_crystal.cell_angles, in_degrees=True
    )

    # Step 3: Calculate the [hkl] direction in Cartesian coordinates
    hkl_cart: Float[Array, "3"] = (
        orientation[0] * recip_vecs[0]
        + orientation[1] * recip_vecs[1]
        + orientation[2] * recip_vecs[2]
    )
    hkl_norm: Float[Array, "3"] = hkl_cart / jnp.linalg.norm(hkl_cart)

    # Step 4: Build rotation matrix to align hkl with z-axis
    # Target is [0, 0, 1]
    z_axis: Float[Array, "3"] = jnp.array([0.0, 0.0, 1.0])

    # Rotation axis: cross product of hkl_norm and z_axis
    rot_axis: Float[Array, "3"] = jnp.cross(hkl_norm, z_axis)
    rot_axis_norm: Float[Array, ""] = jnp.linalg.norm(rot_axis)

    # Rotation angle
    cos_angle: Float[Array, ""] = jnp.dot(hkl_norm, z_axis)
    angle: Float[Array, ""] = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))

    # Rodrigues' rotation formula
    # Handle special case when vectors are already aligned
    def _aligned_matrix() -> Float[Array, "3 3"]:
        return jnp.eye(3)

    def _rotation_matrix() -> Float[Array, "3 3"]:
        k: Float[Array, "3"] = rot_axis / (rot_axis_norm + 1e-10)
        K: Float[Array, "3 3"] = jnp.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ])
        R: Float[Array, "3 3"] = (
            jnp.eye(3)
            + jnp.sin(angle) * K
            + (1 - jnp.cos(angle)) * (K @ K)
        )
        return R

    rotation_matrix: Float[Array, "3 3"] = lax.cond(
        rot_axis_norm < 1e-6, _aligned_matrix, _rotation_matrix
    )

    # Step 5: Rotate atomic positions
    positions_xyz: Float[Array, "N 3"] = bulk_crystal.cart_positions[:, :3]
    rotated_positions: Float[Array, "N 3"] = positions_xyz @ rotation_matrix.T

    # Step 6: Determine supercell repetitions needed
    # Transform cell vectors
    rotated_cell_vecs: Float[Array, "3 3"] = cell_vecs @ rotation_matrix.T

    # Estimate repetitions needed in each direction
    # We need to cover x_extent and y_extent
    cell_x_proj: Float[Array, ""] = jnp.abs(rotated_cell_vecs[0, 0])
    cell_y_proj: Float[Array, ""] = jnp.abs(rotated_cell_vecs[1, 1])

    nx: int = int(jnp.ceil(x_extent / (cell_x_proj + 1e-10))) + 2
    ny: int = int(jnp.ceil(y_extent / (cell_y_proj + 1e-10))) + 2

    # Step 7: Create supercell by replicating atoms
    # Generate translation vectors
    n_atoms: int = rotated_positions.shape[0]
    atomic_numbers: Float[Array, "N"] = bulk_crystal.cart_positions[:, 3]

    # Create all translations
    all_positions = []
    for ix in range(-nx // 2, nx // 2 + 1):
        for iy in range(-ny // 2, ny // 2 + 1):
            translation: Float[Array, "3"] = (
                ix * rotated_cell_vecs[0] + iy * rotated_cell_vecs[1]
            )
            translated_pos: Float[Array, "N 3"] = (
                rotated_positions + translation[None, :]
            )
            all_positions.append(translated_pos)

    # Stack all positions
    supercell_positions: Float[Array, "M 3"] = jnp.concatenate(
        all_positions, axis=0
    )

    # Replicate atomic numbers
    supercell_atomic_nums: Float[Array, "M"] = jnp.tile(
        atomic_numbers, (nx + 1) * (ny + 1)
    )

    # Step 8: Filter by spatial extents and depth
    # Center around origin first
    x_min: Float[Array, ""] = supercell_positions[:, 0].min()
    y_min: Float[Array, ""] = supercell_positions[:, 1].min()
    z_min: Float[Array, ""] = supercell_positions[:, 2].min()

    centered_positions: Float[Array, "M 3"] = supercell_positions - jnp.array(
        [x_min, y_min, z_min]
    )

    # Filter atoms within bounds
    x_mask: Bool[Array, "M"] = jnp.logical_and(
        centered_positions[:, 0] >= 0,
        centered_positions[:, 0] <= x_extent,
    )
    y_mask: Bool[Array, "M"] = jnp.logical_and(
        centered_positions[:, 1] >= 0,
        centered_positions[:, 1] <= y_extent,
    )
    z_mask: Bool[Array, "M"] = jnp.logical_and(
        centered_positions[:, 2] >= 0,
        centered_positions[:, 2] <= depth,
    )

    combined_mask: Bool[Array, "M"] = x_mask & y_mask & z_mask

    # Select atoms within bounds
    filtered_positions: Float[Array, "K 3"] = centered_positions[
        combined_mask
    ]
    filtered_atomic_nums: Float[Array, "K"] = supercell_atomic_nums[
        combined_mask
    ]

    # Combine positions and atomic numbers
    final_positions: Float[Array, "K 4"] = jnp.column_stack(
        [filtered_positions, filtered_atomic_nums]
    )

    # Step 9: Define supercell parameters
    new_cell_lengths: Float[Array, "3"] = jnp.array(
        [x_extent, y_extent, depth]
    )
    new_cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

    # Create SlicedCrystal
    sliced_crystal: SlicedCrystal = create_sliced_crystal(
        cart_positions=final_positions,
        cell_lengths=new_cell_lengths,
        cell_angles=new_cell_angles,
        orientation=orientation,
        depth=depth,
        x_extent=x_extent,
        y_extent=y_extent,
    )

    return sliced_crystal
