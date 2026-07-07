"""Data structures and factory functions for RHEED pattern representation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing RHEED
patterns and images. All structures follow a JAX-compatible validation pattern
that ensures data integrity at compile time.

Routine Listings
----------------
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
Validation keeps static shape/rank checks as Python ``if`` statements and uses
``eqx.error_if`` for runtime value checks. This rejects invalid values under
JIT while preserving identity-like gradient flow on valid inputs.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Final, NamedTuple, Optional, Union
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from .custom_types import float_jax_image, scalar_float, scalar_num

_NDIM_POSITIONS: Final[int] = 2
_NCOLS_CART: Final[int] = 4
_MAX_ATOMIC_NUMBER: Final[int] = 118
_MAX_CELL_ANGLE: Final[int] = 180
_SELF_DISTANCE_TOL: Final[float] = 0.1


class RHEEDPattern(eqx.Module):
    """JAX-compatible RHEED diffraction pattern data structure.

    This PyTree represents a RHEED diffraction pattern containing reflection
    data including reciprocal lattice indices, outgoing wavevectors, detector
    coordinates, and intensity values for electron diffraction analysis.

    :see: :class:`~.test_rheed_types.TestRHEEDPattern`

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
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX
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
    ...     intensities=intensities,
    ... )
    """

    G_indices: Int[Array, "N"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]


class RHEEDImage(eqx.Module):
    """JAX-compatible experimental RHEED image data structure.

    This PyTree represents an experimental RHEED image with associated
    experimental parameters including beam geometry, detector calibration,
    and electron beam properties for quantitative RHEED analysis.

    :see: :class:`~.test_rheed_types.TestRHEEDImage`

    Attributes
    ----------
    img_array : float_jax_image
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
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX
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
    ...     calibration=0.01,  # 0.01 units per pixel
    ...     electron_wavelength=0.037,  # 10 keV electrons
    ...     detector_distance=1000.0,  # 1000 Å to detector
    ... )
    """

    img_array: float_jax_image
    incoming_angle: scalar_float
    calibration: Union[Float[Array, "2"], scalar_float]
    electron_wavelength: scalar_float
    detector_distance: scalar_num


@jaxtyped(typechecker=beartype)
def create_rheed_pattern(
    g_indices: Int[Array, "N"],
    k_out: Float[Array, "M 3"],
    detector_points: Float[Array, "M 2"],
    intensities: Float[Array, "M"],
) -> RHEEDPattern:
    """Create a RHEEDPattern instance with data validation.

    :see: :class:`~.test_rheed_types.TestRHEEDPattern`

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

        if k_out.shape != (mm, 3):
            raise ValueError("k_out must have shape (M, 3)")
        if detector_points.shape != (mm, 2):
            raise ValueError("detector_points must have shape (M, 2)")
        if intensities.shape != (mm,):
            raise ValueError("intensities must have shape (M,)")
        if g_indices.shape[0] != mm:
            raise ValueError("g_indices length must match reflections")

        checked_k_out: Float[Array, "mm 3"] = eqx.error_if(
            k_out,
            jnp.any(~jnp.isfinite(k_out)),
            "k_out contains non-finite values",
        )
        checked_k_out = eqx.error_if(
            checked_k_out,
            jnp.any(jnp.linalg.norm(checked_k_out, axis=1) <= 0),
            "k_out vectors must be non-zero",
        )
        checked_detector_points: Float[Array, "mm 2"] = eqx.error_if(
            detector_points,
            jnp.any(~jnp.isfinite(detector_points)),
            "detector_points contain non-finite values",
        )
        checked_intensities: Float[Array, "mm"] = eqx.error_if(
            intensities,
            jnp.any(intensities < 0),
            "intensities must be non-negative",
        )
        checked_intensities = eqx.error_if(
            checked_intensities,
            jnp.any(~jnp.isfinite(checked_intensities)),
            "intensities contain non-finite values",
        )

        return RHEEDPattern(
            G_indices=g_indices,
            k_out=checked_k_out,
            detector_points=checked_detector_points,
            intensities=checked_intensities,
        )

    validated_rheed_pattern: RHEEDPattern = _validate_and_create()
    return validated_rheed_pattern


@jaxtyped(typechecker=beartype)
def create_rheed_image(
    img_array: float_jax_image,
    incoming_angle: scalar_float,
    calibration: Union[Float[Array, "2"], scalar_float],
    electron_wavelength: scalar_float,
    detector_distance: scalar_num,
) -> RHEEDImage:
    """Create a RHEEDImage instance with data validation.

    :see: :class:`~.test_rheed_types.TestRHEEDImage`

    Parameters
    ----------
    img_array : float_jax_image
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
    img_array: float_jax_image = jnp.asarray(img_array, dtype=jnp.float64)
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
    image_dimensions: int = 2

    def _validate_and_create() -> RHEEDImage:
        """Validate and create a RHEEDImage instance."""
        if img_array.ndim != image_dimensions:
            raise ValueError("img_array must be 2D")
        if calibration.ndim > 1:
            raise ValueError("calibration must be scalar or shape (2,)")
        if calibration.ndim == 1 and calibration.shape != (2,):
            raise ValueError("calibration must be scalar or shape (2,)")

        checked_img_array: float_jax_image = eqx.error_if(
            img_array,
            jnp.any(~jnp.isfinite(img_array)),
            "img_array contains non-finite values",
        )
        checked_img_array = eqx.error_if(
            checked_img_array,
            jnp.any(checked_img_array < 0),
            "img_array must be non-negative",
        )
        checked_incoming_angle: Float[Array, ""] = eqx.error_if(
            incoming_angle,
            jnp.logical_or(incoming_angle < 0, incoming_angle > 90),
            "incoming_angle must be between 0 and 90 degrees",
        )
        checked_electron_wavelength: Float[Array, ""] = eqx.error_if(
            electron_wavelength,
            electron_wavelength <= 0,
            "electron_wavelength must be positive",
        )
        checked_detector_distance: Float[Array, ""] = eqx.error_if(
            detector_distance,
            detector_distance <= 0,
            "detector_distance must be positive",
        )
        checked_calibration: Union[Float[Array, "2"], Float[Array, ""]] = (
            eqx.error_if(
                calibration,
                jnp.any(calibration <= 0),
                "calibration must be positive",
            )
        )

        return RHEEDImage(
            img_array=checked_img_array,
            incoming_angle=checked_incoming_angle,
            calibration=checked_calibration,
            electron_wavelength=checked_electron_wavelength,
            detector_distance=checked_detector_distance,
        )

    validated_rheed_image: RHEEDImage = _validate_and_create()
    return validated_rheed_image


class SlicedCrystal(eqx.Module):
    """JAX-compatible surface-oriented crystal structure for RHEED simulation.

    This PyTree represents a crystal structure that has been sliced
    and oriented for RHEED simulations. The structure contains atoms
    from a surface region
    extended in x and y directions to cover a large area (>100 Å typically),
    with a specified depth perpendicular to the surface.

    :see: :class:`~.test_rheed_types.TestSlicedCrystal`

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
    occupancies : Float[Array, "N"] | None
        Per-atom site occupancies in [0, 1]. Each atom's projected
        potential is weighted by its occupancy in the multislice
        conversion. ``None`` means fully occupied (all ones).

    Notes
    -----
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible
    with JAX transformations. The structure is designed specifically
    for RHEED simulations
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
    >>> cart_positions = jnp.array(
    ...     [
    ...         [0.0, 0.0, 0.0, 14.0],  # Si atom
    ...         [1.0, 1.0, 0.5, 14.0],
    ...     ]
    ... )  # Another Si
    >>> sliced = rh.types.create_sliced_crystal(
    ...     cart_positions=cart_positions,
    ...     cell_lengths=jnp.array([150.0, 150.0, 20.0]),
    ...     cell_angles=jnp.array([90.0, 90.0, 90.0]),
    ...     orientation=jnp.array([1, 1, 1]),
    ...     depth=20.0,
    ...     x_extent=150.0,
    ...     y_extent=150.0,
    ... )
    """

    cart_positions: Float[Array, "N 4"]
    cell_lengths: Float[Array, "3"]
    cell_angles: Float[Array, "3"]
    orientation: Int[Array, "3"]
    depth: scalar_float
    x_extent: scalar_float
    y_extent: scalar_float
    occupancies: Float[Array, "N"] | None = None


@jaxtyped(typechecker=beartype)
def create_sliced_crystal(
    cart_positions: Float[Array, "N 4"],
    cell_lengths: Float[Array, "3"],
    cell_angles: Float[Array, "3"],
    orientation: Int[Array, "3"],
    depth: scalar_float,
    x_extent: scalar_float,
    y_extent: scalar_float,
    occupancies: Optional[Float[Array, "N"]] = None,
) -> SlicedCrystal:
    """Create a SlicedCrystal instance with data validation.

    :see: :class:`~.test_rheed_types.TestSlicedCrystal`

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
    occupancies : Optional[Float[Array, "N"]], optional
        Per-atom site occupancies in [0, 1]. If omitted, all sites are
        treated as fully occupied. Default: None.

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
    - Validate occupancies, when given, have shape (N,) and lie in
      [0, 1].
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
    occupancies_arr: Optional[Float[Array, "N"]] = (
        None
        if occupancies is None
        else jnp.asarray(occupancies, dtype=jnp.float64)
    )

    def _validate_and_create() -> SlicedCrystal:
        """Validate and create a SlicedCrystal instance."""
        n_atoms: int = cart_positions.shape[0]

        if cart_positions.ndim != _NDIM_POSITIONS:
            raise ValueError("cart_positions must have shape (N, 4)")
        if cart_positions.shape[1] != _NCOLS_CART:
            raise ValueError("cart_positions must have shape (N, 4)")
        if n_atoms <= 0:
            raise ValueError("cart_positions must contain at least one atom")
        if cell_lengths.shape != (3,):
            raise ValueError("cell_lengths must have shape (3,)")
        if cell_angles.shape != (3,):
            raise ValueError("cell_angles must have shape (3,)")
        if orientation.shape != (3,):
            raise ValueError("orientation must have shape (3,)")
        if occupancies_arr is not None and occupancies_arr.shape != (n_atoms,):
            raise ValueError("occupancies must have shape (N,)")

        checked_cart_positions: Float[Array, "N 4"] = eqx.error_if(
            cart_positions,
            jnp.any(~jnp.isfinite(cart_positions)),
            "cart_positions contain non-finite values",
        )
        atomic_nums: Float[Array, "N"] = checked_cart_positions[:, 3]
        checked_cart_positions = eqx.error_if(
            checked_cart_positions,
            jnp.any((atomic_nums < 1) | (atomic_nums > _MAX_ATOMIC_NUMBER)),
            "atomic numbers must be in [1, 118]",
        )
        checked_cell_lengths: Float[Array, "3"] = eqx.error_if(
            cell_lengths,
            jnp.any(cell_lengths <= 0),
            "cell_lengths must be positive",
        )
        checked_cell_angles: Float[Array, "3"] = eqx.error_if(
            cell_angles,
            jnp.any((cell_angles <= 0) | (cell_angles >= _MAX_CELL_ANGLE)),
            "cell_angles must be between 0 and 180 degrees",
        )
        checked_depth: Float[Array, ""] = eqx.error_if(
            depth,
            depth <= 0,
            "depth must be positive",
        )
        checked_x_extent: Float[Array, ""] = eqx.error_if(
            x_extent,
            x_extent <= 0,
            "x_extent must be positive",
        )
        checked_y_extent: Float[Array, ""] = eqx.error_if(
            y_extent,
            y_extent <= 0,
            "y_extent must be positive",
        )
        checked_occupancies: Optional[Float[Array, "N"]] = (
            None
            if occupancies_arr is None
            else eqx.error_if(
                occupancies_arr,
                jnp.any((occupancies_arr < 0.0) | (occupancies_arr > 1.0)),
                "occupancies must be between 0 and 1",
            )
        )

        return SlicedCrystal(
            cart_positions=checked_cart_positions,
            cell_lengths=checked_cell_lengths,
            cell_angles=checked_cell_angles,
            orientation=orientation,
            depth=checked_depth,
            x_extent=checked_x_extent,
            y_extent=checked_y_extent,
            occupancies=checked_occupancies,
        )

    validated_sliced_crystal: SlicedCrystal = _validate_and_create()
    return validated_sliced_crystal


class SurfaceConfig(NamedTuple):
    """Configuration for surface atom identification.

    This NamedTuple specifies how to identify which atoms are considered
    surface atoms for enhanced Debye-Waller factors in RHEED simulations.

    :see: :class:`~.test_rheed_types.TestSurfaceConfig`

    Attributes
    ----------
    method : str
        Surface identification method:
        - "height": top fraction by z-coordinate (default)
        - "coordination": atoms with fewer neighbors than bulk
        - "layers": atoms in topmost N complete layers
        - "explicit": user-provided surface mask
        - "none": no atom is a surface atom (all-False mask). This is
          the correct choice for bulk-cell simulators where the basis
          is implicitly repeated by a CTR factor: a repeated bulk unit
          cell has no surface atoms, so no thermal enhancement should
          be applied.
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


_DEFAULT_SURFACE_CONFIG: Final[SurfaceConfig] = SurfaceConfig()


@jaxtyped(typechecker=beartype)
def identify_surface_atoms(
    atom_positions: Float[Array, "N 3"],
    config: SurfaceConfig = _DEFAULT_SURFACE_CONFIG,
) -> Bool[Array, "N"]:
    """Identify surface atoms using the specified method.

    :see: :class:`~.test_rheed_types.TestIdentifySurfaceAtoms`

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

    - **none**: Returns an all-False mask; no atom receives surface
      thermal enhancement. Appropriate for bulk unit cells whose
      repetition is modeled by a CTR factor rather than by an
      explicit slab.

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

    if config.method == "none":
        return jnp.zeros(n_atoms, dtype=bool)

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

    if config.method == "coordination":
        # Coordination-based: count neighbors for each atom
        # Compute pairwise distances
        diff: Float[Array, "N N 3"] = (
            atom_positions[:, None, :] - atom_positions[None, :, :]
        )
        distances: Float[Array, "N N"] = jnp.sqrt(jnp.sum(diff**2, axis=-1))

        # Count neighbors within cutoff (excluding self)
        neighbor_mask: Bool[Array, "N N"] = (
            distances < config.coordination_cutoff
        ) & (distances > _SELF_DISTANCE_TOL)
        neighbor_counts: Int[Array, "N"] = jnp.sum(neighbor_mask, axis=-1)

        # Surface atoms have fewer neighbors than threshold
        is_surface: Bool[Array, "N"] = (
            neighbor_counts < config.coordination_threshold
        )
        return is_surface

    if config.method == "layers":
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

    # Default: height-based method
    z_max: Float[Array, ""] = jnp.max(z_coords)
    z_min: Float[Array, ""] = jnp.min(z_coords)
    z_threshold: Float[Array, ""] = z_max - config.height_fraction * (
        z_max - z_min
    )
    is_surface: Bool[Array, "N"] = z_coords >= z_threshold
    return is_surface


__all__: list[str] = [
    "RHEEDImage",
    "RHEEDPattern",
    "SlicedCrystal",
    "SurfaceConfig",
    "create_rheed_image",
    "create_rheed_pattern",
    "create_sliced_crystal",
    "identify_surface_atoms",
]
