"""Data structures and factory functions for RHEED pattern representation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing RHEED
patterns and images. All structures follow a JAX-compatible validation pattern
that ensures data integrity at compile time.

Routine Listings
----------------
:class:`DetectorGeometry`
    Configuration for RHEED detector geometry.
:class:`RHEEDImage`
    Container for RHEED image data with pixel coordinates
    and intensity values.
:class:`RHEEDPattern`
    Container for RHEED diffraction pattern data with
    detector points.
:class:`SlicedCrystal`
    JAX-compatible crystal structure sliced for multislice
    simulation.
:class:`SurfaceConfig`
    Configuration for surface atom identification method
    and parameters.
:func:`create_rheed_image`
    Factory function to create RHEEDImage instances with
    data validation.
:func:`create_rheed_pattern`
    Factory function to create RHEEDPattern instances with
    data validation.
:func:`create_sliced_crystal`
    Factory function to create SlicedCrystal instances with
    data validation.
:func:`identify_surface_atoms`
    Identify surface atoms using configurable methods.

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
        Detector coordinates (Y, Z) on the detector plane in mm.
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
        Distance from the sample to the detector in mm. Affects the
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
        (Y, Z) coordinates on the detector plane, in mm.
    intensities : Float[Array, "M"]
        Intensities for each reflection.

    Returns
    -------
    validated_rheed_pattern : RHEEDPattern
        Validated RHEED pattern instance.

    Notes
    -----
    - Convert inputs to JAX arrays.
    - Validate array shapes: check k_out has shape (M, 3),
      detector_points has shape (M, 2), intensities has
      shape (M,), and g_indices has length M.
    - Validate data: ensure intensities are non-negative,
      k_out vectors are non-zero, and detector points are
      finite.
    - Create and return RHEEDPattern instance.
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
        The distance from the sample to the detector in mm.

    Returns
    -------
    validated_rheed_image : RHEEDImage
        Validated RHEED image instance.

    Notes
    -----
    1. Convert inputs to JAX arrays.
    2. Validate image array: check it is 2D, all values are
       finite and non-negative.
    3. Validate parameters: check incoming_angle is between
       0 and 90 degrees, electron_wavelength is positive,
       and detector_distance is positive.
    4. Validate calibration: if scalar, ensure it is
       positive; if array, ensure shape is (2,) and all
       values are positive.
    5. Create and return RHEEDImage instance.
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

        def _check_calibration() -> Union[Float[Array, "2"], Float[Array, ""]]:
            """Check the calibration is positive."""
            # Use jnp.all to ensure scalar predicate regardless of calibration shape
            # Both branches are traced by JAX, so both predicates must be scalar
            return lax.cond(
                jnp.all(calibration > 0),
                lambda: calibration,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calibration, lambda: calibration)
                ),
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
    - Verify cart_positions shape is (N, 4) with N > 0.
    - Verify cell_lengths, cell_angles, orientation have
      correct shapes.
    - Ensure all positions are finite.
    - Ensure depth, x_extent, y_extent are positive.
    - Recommend x_extent and y_extent >= 100 Angstroms.
    - Validate atomic numbers (cart_positions[:, 3]) are in
      valid range [1, 118].
    """
    cart_positions: Float[Array, "N 4"] = jnp.asarray(
        cart_positions, dtype=jnp.float64
    )
    cell_lengths: Float[Array, "3"] = jnp.asarray(
        cell_lengths, dtype=jnp.float64
    )
    cell_angles: Float[Array, "3"] = jnp.asarray(
        cell_angles, dtype=jnp.float64
    )
    orientation: Int[Array, "3"] = jnp.asarray(orientation, dtype=jnp.int32)
    depth: Float[Array, ""] = jnp.asarray(depth, dtype=jnp.float64)
    x_extent: Float[Array, ""] = jnp.asarray(x_extent, dtype=jnp.float64)
    y_extent: Float[Array, ""] = jnp.asarray(y_extent, dtype=jnp.float64)

    def _validate_and_create() -> SlicedCrystal:
        """Validate and create a SlicedCrystal instance."""

        def _check_positions_shape() -> Float[Array, "N 4"]:
            """Check cart_positions has shape (N, 4)."""
            n_atoms: int = cart_positions.shape[0]
            return lax.cond(
                jnp.logical_and(
                    cart_positions.ndim == 2,
                    jnp.logical_and(cart_positions.shape[1] == 4, n_atoms > 0),
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
            atomic_nums: Float[Array, "N"] = cart_positions[:, 3]
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
            x_valid: Float[Array, ""] = lax.cond(
                x_extent > 0,
                lambda: x_extent,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: x_extent, lambda: x_extent)
                ),
            )
            y_valid: Float[Array, ""] = lax.cond(
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

    validated_sliced_crystal: SlicedCrystal = _validate_and_create()
    return validated_sliced_crystal


class SurfaceConfig(NamedTuple):
    """Configuration for surface atom identification.

    This NamedTuple specifies how to identify which atoms are considered
    surface atoms for enhanced Debye-Waller factors in RHEED simulations.

    Attributes
    ----------
    method : str
        Surface identification method:
        - "height": top fraction by z-coordinate (default)
        - "coordination": atoms with fewer neighbors than bulk
        - "layers": atoms in topmost N complete layers
        - "explicit": user-provided surface mask
    height_fraction : float
        For "height" method: fraction of z-range considered surface.
        Default: 0.3 (top 30% of atoms by height)
    coordination_cutoff : float
        For "coordination" method: cutoff distance in Angstroms for
        counting neighbors. Default: 3.0
    coordination_threshold : int
        For "coordination" method: atoms with fewer than this many
        neighbors are considered surface. Default: 8 (typical for FCC)
    n_layers : int
        For "layers" method: number of topmost layers considered
        surface. Default: 1
    layer_tolerance : float
        For "layers" method: tolerance for grouping atoms into layers
        in Angstroms. Default: 0.5
    explicit_mask : Bool[Array, "N"] | None
        For "explicit" method: user-provided boolean mask. Must have
        same length as number of atoms. Default: None
    """

    method: str = "height"
    height_fraction: float = 0.3
    coordination_cutoff: float = 3.0
    coordination_threshold: int = 8
    n_layers: int = 1
    layer_tolerance: float = 0.5
    explicit_mask: Bool[Array, "N"] | None = None


@jaxtyped(typechecker=beartype)
def identify_surface_atoms(
    atom_positions: Float[Array, "N 3"],
    config: SurfaceConfig = SurfaceConfig(),
) -> Bool[Array, "N"]:
    """Identify surface atoms using the specified method.

    Parameters
    ----------
    atom_positions : Float[Array, "N 3"]
        Cartesian atomic positions in Angstroms, shape (N, 3).
    config : SurfaceConfig, optional
        Surface identification configuration. Default: SurfaceConfig()
        with height-based method at 30% fraction.

    Returns
    -------
    is_surface : Bool[Array, "N"]
        Boolean mask indicating which atoms are surface atoms.

    Notes
    -----
    Available methods:

    - **height**: Uses z-coordinate threshold. Atoms in the top
      `height_fraction` of the z-range are marked as surface.

    - **coordination**: Uses neighbor counting. Atoms with fewer than
      `coordination_threshold` neighbors within `coordination_cutoff`
      distance are marked as surface. This is more physical for
      reconstructed or stepped surfaces.

    - **layers**: Identifies discrete atomic layers by z-coordinate and
      marks the topmost `n_layers` as surface. Uses `layer_tolerance`
      to group atoms into layers.

    - **explicit**: Uses user-provided mask directly from
      `config.explicit_mask`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.types import SurfaceConfig, identify_surface_atoms
    >>> positions = jnp.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    >>> # Height-based (default)
    >>> mask = identify_surface_atoms(positions)
    >>> # Coordination-based
    >>> config = SurfaceConfig(method="coordination", coordination_cutoff=2.5)
    >>> mask = identify_surface_atoms(positions, config)
    """
    n_atoms: int = atom_positions.shape[0]
    z_coords: Float[Array, "N"] = atom_positions[:, 2]

    if config.method == "explicit":
        # Use explicit mask if provided
        if config.explicit_mask is not None:
            return config.explicit_mask
        # Fallback to height method if no mask provided
        z_max: Float[Array, ""] = jnp.max(z_coords)
        z_min: Float[Array, ""] = jnp.min(z_coords)
        z_threshold: Float[Array, ""] = z_max - config.height_fraction * (
            z_max - z_min
        )
        return z_coords >= z_threshold

    elif config.method == "coordination":
        # Coordination-based: count neighbors for each atom
        # Compute pairwise distances
        diff: Float[Array, "N N 3"] = (
            atom_positions[:, None, :] - atom_positions[None, :, :]
        )
        distances: Float[Array, "N N"] = jnp.sqrt(jnp.sum(diff**2, axis=-1))

        # Count neighbors within cutoff (excluding self)
        neighbor_mask: Bool[Array, "N N"] = (
            distances < config.coordination_cutoff
        ) & (distances > 0.1)
        neighbor_counts: Int[Array, "N"] = jnp.sum(neighbor_mask, axis=-1)

        # Surface atoms have fewer neighbors than threshold
        is_surface: Bool[Array, "N"] = (
            neighbor_counts < config.coordination_threshold
        )
        return is_surface

    elif config.method == "layers":
        # Layer-based: identify discrete z-layers and mark top N
        # Sort z-coordinates to find layer boundaries
        z_sorted: Float[Array, "N"] = jnp.sort(z_coords)

        # Find layer boundaries using tolerance
        z_diffs: Float[Array, "N-1"] = jnp.diff(z_sorted)
        layer_breaks: Bool[Array, "N-1"] = z_diffs > config.layer_tolerance

        # Assign layer indices (0 = bottom, higher = top)
        layer_indices: Int[Array, "N"] = jnp.cumsum(
            jnp.concatenate(
                [
                    jnp.array([0], dtype=jnp.int32),
                    layer_breaks.astype(jnp.int32),
                ]
            ),
            dtype=jnp.int32,
        )

        # Map back to original atom order
        sort_indices: Int[Array, "N"] = jnp.argsort(z_coords)
        atom_layers: Int[Array, "N"] = jnp.zeros(n_atoms, dtype=jnp.int32)
        atom_layers: Int[Array, "N"] = atom_layers.at[sort_indices].set(
            layer_indices
        )

        # Mark top n_layers as surface
        max_layer: Int[Array, ""] = jnp.max(atom_layers)
        is_surface: Bool[Array, "N"] = atom_layers >= (
            max_layer - config.n_layers + 1
        )
        return is_surface

    else:
        # Default: height-based method
        z_max: Float[Array, ""] = jnp.max(z_coords)
        z_min: Float[Array, ""] = jnp.min(z_coords)
        z_threshold: Float[Array, ""] = z_max - config.height_fraction * (
            z_max - z_min
        )
        is_surface: Bool[Array, "N"] = z_coords >= z_threshold
        return is_surface


class DetectorGeometry(NamedTuple):
    """Configuration for RHEED detector geometry.

    This NamedTuple specifies the geometry of the detector screen for
    accurate projection of diffracted beams. Supports flat, tilted, and
    curved detector screens.

    Attributes
    ----------
    distance : float
        Perpendicular distance from sample to detector center in mm.
        Default: 100.0
    tilt_angle : float
        Tilt angle of the detector about the horizontal axis in degrees.
        Positive tilt rotates the top of the screen away from the sample.
        Default: 0.0 (vertical screen)
    curvature_radius : float
        Radius of curvature of the detector screen in mm.
        Use jnp.inf for flat screen. Default: jnp.inf (flat)
    center_offset_h : float
        Horizontal offset of detector center from beam axis in mm.
        Positive values shift the detector right. Default: 0.0
    center_offset_v : float
        Vertical offset of detector center from beam axis in mm.
        Positive values shift the detector up. Default: 0.0
    psf_sigma_pixels : float
        Point spread function 1-sigma width in pixels. Models phosphor
        grain size, camera lens blur, and CCD pixel diffusion. Typical:
        0.5--2.0 pixels. Use 0.0 to disable PSF convolution.
        Default: 1.0

    Notes
    -----
    For a standard RHEED setup:
    - The beam travels predominantly along +x (into the sample surface)
    - The detector is placed in the yz-plane at x = distance
    - Horizontal corresponds to y-direction, vertical to z-direction

    The tilt angle accounts for common experimental setups where the
    phosphor screen is not perfectly perpendicular to the nominal beam
    direction.

    For curved screens (e.g., cylindrical CCD detector arrays), the
    curvature_radius determines the cylinder radius. Points on the
    detector surface lie on this cylinder.
    """

    distance: float = 100.0
    tilt_angle: float = 0.0
    curvature_radius: float = float("inf")
    center_offset_h: float = 0.0
    center_offset_v: float = 0.0
    psf_sigma_pixels: float = 1.0


__all__: list[str] = [
    "DetectorGeometry",
    "RHEEDImage",
    "RHEEDPattern",
    "SlicedCrystal",
    "SurfaceConfig",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "identify_surface_atoms",
]
