"""Data structures and factory functions for crystal structure representation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, potential slices for multislice simulations, XYZ file data, and
Ewald sphere data for RHEED simulation. All structures are PyTrees that
support JAX transformations.

Routine Listings
----------------
:class:`CrystalStructure`
    JAX-compatible crystal structure with fractional and
    Cartesian coordinates.
:class:`EwaldData`
    Angle-independent Ewald sphere data for RHEED simulation.
:class:`KirklandParameters`
    Structured Kirkland coefficients for one element.
:class:`PotentialSlices`
    JAX-compatible data structure for representing multislice
    potential data.
:class:`EdgeOnSlices`
    JAX-compatible edge-on potential slices for reflection multislice.
:class:`XYZData`
    A PyTree for XYZ file data with atomic positions and
    metadata.
:func:`create_crystal_structure`
    Factory function to create CrystalStructure instances
    with data validation.
:func:`create_ewald_data`
    Factory function to create EwaldData instances with
    validation.
:func:`create_potential_slices`
    Factory function to create PotentialSlices instances with
    data validation.
:func:`create_edge_on_slices`
    Factory function to create EdgeOnSlices instances with
    data validation.
:func:`create_xyz_data`
    Factory function to create XYZData instances with data
    validation.

Notes
-----
All data structures are Equinox modules (``eqx.Module``): immutable JAX
PyTrees that support automatic differentiation. Static, non-array metadata
fields (e.g. ``XYZData.comment``) are declared with
``eqx.field(static=True)`` so they are excluded from the differentiable
leaves.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from .custom_types import scalar_float


def _build_canonical_cell_vectors(
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
) -> Float[Array, "3 3"]:
    """Build row-vector canonical cell vectors without importing ucell."""
    a: Num[Array, ""]
    b: Num[Array, ""]
    c: Num[Array, ""]
    alpha: Num[Array, ""]
    beta: Num[Array, ""]
    gamma: Num[Array, ""]
    a, b, c = cell_lengths
    alpha, beta, gamma = jnp.radians(cell_angles)
    a_vec: Float[Array, "3"] = jnp.array([a, 0.0, 0.0])
    b_vec: Float[Array, "3"] = jnp.array(
        [b * jnp.cos(gamma), b * jnp.sin(gamma), 0.0]
    )
    c_x: Num[Array, ""] = c * jnp.cos(beta)
    c_y: Num[Array, ""] = c * (
        (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma)
    )
    c_z_sq: Num[Array, ""] = c**2 - c_x**2 - c_y**2
    c_vec: Float[Array, "3"] = jnp.array(
        [c_x, c_y, jnp.sqrt(jnp.clip(c_z_sq, min=0.0))]
    )
    return jnp.stack([a_vec, b_vec, c_vec], axis=0)


class CrystalStructure(eqx.Module):
    """JAX-compatible Pytree with fractional and Cartesian coordinates.

    This PyTree represents a crystal structure containing atomic positions in
    both fractional and Cartesian coordinate systems, along with unit cell
    parameters. It's designed for efficient crystal structure calculations and
    electron diffraction simulations.

    :see: :class:`~.test_crystal_types.TestCrystalStructure`

    Attributes
    ----------
    frac_positions : Num[Array, "N 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional
        coordinates. Each row contains [x, y, z, atomic_number] where x, y, z
        are fractional coordinates in the unit cell (range [0,1]) and
        atomic_number is the integer atomic number (Z) of the element.
    cart_positions : Num[Array, "N 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian
        coordinates. Each row contains [x, y, z, atomic_number] where x, y, z
        are Cartesian coordinates in Ångstroms and atomic_number is the integer
        atomic number (Z).
    cell_lengths : Num[Array, "3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    cell_angles : Num[Array, "3"]
        Unit cell angles [α, β, γ] in degrees, where α is the angle between
        b and c, β is the angle between a and c, and γ is the angle between
        a and b.
    occupancies : Num[Array, "N"]
        Per-site scattering occupancies. Defaults to 1 for fully occupied
        sites.
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. All data is immutable and stored
    in JAX arrays for efficient computation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create crystal structure for simple cubic lattice
    >>> frac_pos = jnp.array([[0.0, 0.0, 0.0, 6]])  # Carbon atom at origin
    >>> cart_pos = jnp.array([[0.0, 0.0, 0.0, 6]])  # Same in Cartesian
    >>> cell_lengths = jnp.array([3.57, 3.57, 3.57])  # Diamond lattice
    >>> cell_angles = jnp.array([90.0, 90.0, 90.0])  # Cubic angles
    >>> crystal = rh.types.create_crystal_structure(
    ...     frac_positions=frac_pos,
    ...     cart_positions=cart_pos,
    ...     cell_lengths=cell_lengths,
    ...     cell_angles=cell_angles,
    ... )
    """

    frac_positions: Num[Array, "N 4"]
    cart_positions: Num[Array, "N 4"]
    cell_lengths: Num[Array, "3"]
    cell_angles: Num[Array, "3"]
    occupancies: Num[Array, "N"] | None = None


@jaxtyped(typechecker=beartype)
def create_crystal_structure(
    frac_positions: Num[Array, "... 4"],
    cart_positions: Num[Array, "... 4"],
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
    occupancies: Optional[Num[Array, "..."]] = None,
) -> CrystalStructure:
    """Create a CrystalStructure PyTree with data validation.

    :see: :class:`~.test_crystal_types.TestCrystalStructure`

    Parameters
    ----------
    frac_positions : Num[Array, "... 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional
        coordinates.
    cart_positions : Num[Array, "... 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian
        coordinates.
    cell_lengths : Num[Array, "3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    cell_angles : Num[Array, "3"]
        Unit cell angles [α, β, γ] in degrees.
    occupancies : Optional[Num[Array, "..."]]
        Per-site occupancies. If omitted, all sites are treated as fully
        occupied.

    Returns
    -------
    validated_crystal_structure : CrystalStructure
        A validated CrystalStructure instance.

    Notes
    -----
    - Convert all inputs to JAX arrays using jnp.asarray.
    - Validate shapes of frac_positions, cart_positions,
      cell_lengths, and cell_angles.
    - Verify number of atoms matches between frac and cart
      positions.
    - Verify atomic numbers match between frac and cart
      positions.
    - Ensure cell lengths are positive.
    - Ensure cell angles are between 0 and 180 degrees.
    - Create and return CrystalStructure instance with
      validated data.
    """
    frac_positions: Float[Array, "... 4"] = jnp.asarray(frac_positions)
    cart_positions: Num[Array, "... 4"] = jnp.asarray(cart_positions)
    cell_lengths: Num[Array, "3"] = jnp.asarray(cell_lengths)
    cell_angles: Num[Array, "3"] = jnp.asarray(cell_angles)
    if occupancies is None:
        occupancies_array: Num[Array, "..."] = jnp.ones(
            frac_positions.shape[0], dtype=frac_positions.dtype
        )
    else:
        occupancies_array = jnp.asarray(occupancies)

    def _validate_and_create() -> CrystalStructure:
        max_cols: int = 4

        if frac_positions.shape[1] != max_cols:
            raise ValueError("frac_positions must have shape (N, 4)")
        if cart_positions.shape[1] != max_cols:
            raise ValueError("cart_positions must have shape (N, 4)")
        if cell_lengths.shape != (3,):
            raise ValueError("cell_lengths must have shape (3,)")
        if cell_angles.shape != (3,):
            raise ValueError("cell_angles must have shape (3,)")
        if frac_positions.shape[0] != cart_positions.shape[0]:
            raise ValueError("frac_positions and cart_positions length differ")
        if occupancies_array.shape != (frac_positions.shape[0],):
            raise ValueError("occupancies must have shape (N,)")

        checked_frac_positions: Float[Array, "... 4"] = eqx.error_if(
            frac_positions,
            jnp.any(frac_positions[:, 3] != cart_positions[:, 3]),
            "atomic numbers must match between frac and cart positions",
        )
        checked_cell_lengths: Num[Array, "3"] = eqx.error_if(
            cell_lengths,
            jnp.any(cell_lengths <= 0),
            "cell_lengths must be positive",
        )
        checked_cell_angles: Num[Array, "3"] = eqx.error_if(
            cell_angles,
            jnp.any((cell_angles <= 0) | (cell_angles >= 180)),
            "cell_angles must be between 0 and 180 degrees",
        )
        checked_occupancies: Num[Array, "..."] = eqx.error_if(
            occupancies_array,
            jnp.any((occupancies_array < 0.0) | (occupancies_array > 1.0)),
            "occupancies must be between 0 and 1",
        )
        canonical_vectors: Float[Array, "3 3"] = _build_canonical_cell_vectors(
            checked_cell_lengths, checked_cell_angles
        )
        expected_cart: Float[Array, "... 3"] = (
            checked_frac_positions[:, :3] @ canonical_vectors
        )
        cart_delta: Float[Array, ""] = jnp.max(
            jnp.abs(expected_cart - cart_positions[:, :3])
        )
        checked_cart_positions: Num[Array, "... 4"] = eqx.error_if(
            cart_positions,
            cart_delta > 1e-6,
            "cart_positions must equal frac_positions @ canonical cell",
        )
        return CrystalStructure(
            frac_positions=checked_frac_positions,
            cart_positions=checked_cart_positions,
            cell_lengths=checked_cell_lengths,
            cell_angles=checked_cell_angles,
            occupancies=checked_occupancies,
        )

    validated_crystal_structure: CrystalStructure = _validate_and_create()
    return validated_crystal_structure


class EwaldData(eqx.Module):
    r"""Angle-independent Ewald sphere data for RHEED simulation.

    This PyTree contains pre-computed reciprocal lattice geometry and structure
    factors that depend only on crystal structure and beam voltage, not on
    beam orientation angles. This enables efficient reuse when scanning beam
    azimuth or incidence angle.

    :see: :class:`~.test_crystal_types.TestEwaldData`

    Attributes
    ----------
    wavelength_ang : Float[Array, ""]
        Relativistic electron wavelength in Ångstroms.
    k_magnitude : Float[Array, ""]
        Magnitude of electron wavevector :math:`|k| = 2\pi/\lambda`
        in 1/Ångstroms.
    sphere_radius : Float[Array, ""]
        Ewald sphere radius in 1/Ångstroms (equals k_magnitude).
    recip_vectors : Float[Array, "3 3"]
        Reciprocal lattice basis vectors [b₁, b₂, b₃] as rows.
    hkl_grid : Int[Array, "N 3"]
        Miller indices (h, k, l) for all reciprocal lattice points.
    g_vectors : Float[Array, "N 3"]
        Reciprocal lattice vectors G in 1/Ångstroms.
    g_magnitudes : Float[Array, "N"]
        Magnitudes :math:`|G|` for each reciprocal lattice vector.
    structure_factors : Complex[Array, "N"]
        Complex structure factors F(G) for each reciprocal lattice point.
    intensities : Float[Array, "N"]
        Kinematic diffraction intensities :math:`I(G) = |F(G)|^2`.

    Notes
    -----
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node for JAX compatibility. The
    structure factors include Lobato-default atomic form factors and
    Debye-Waller thermal damping.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ewald = rh.ucell.build_ewald_data(
    ...     crystal=crystal,
    ...     energy_kev=15.0,
    ...     hmax=3,
    ...     kmax=3,
    ...     lmax=2,
    ... )
    >>> f"Sphere radius: {ewald.sphere_radius:.2f} 1/Å"
    """

    wavelength_ang: Float[Array, ""]
    k_magnitude: Float[Array, ""]
    sphere_radius: Float[Array, ""]
    recip_vectors: Float[Array, "3 3"]
    hkl_grid: Int[Array, "N 3"]
    g_vectors: Float[Array, "N 3"]
    g_magnitudes: Float[Array, "N"]
    structure_factors: Complex[Array, "N"]
    intensities: Float[Array, "N"]


@jaxtyped(typechecker=beartype)
def create_ewald_data(
    wavelength_ang: Float[Array, ""],
    k_magnitude: Float[Array, ""],
    sphere_radius: Float[Array, ""],
    recip_vectors: Float[Array, "3 3"],
    hkl_grid: Int[Array, "N 3"],
    g_vectors: Float[Array, "N 3"],
    g_magnitudes: Float[Array, "N"],
    structure_factors: Complex[Array, "N"],
    intensities: Float[Array, "N"],
) -> EwaldData:
    r"""Create an EwaldData PyTree with validation.

    :see: :class:`~.test_crystal_types.TestEwaldData`

    Parameters
    ----------
    wavelength_ang : Float[Array, ""]
        Electron wavelength in Ångstroms.
    k_magnitude : Float[Array, ""]
        Wavevector magnitude :math:`|k| = 2\pi/\lambda` in 1/Ångstroms.
    sphere_radius : Float[Array, ""]
        Ewald sphere radius in 1/Ångstroms.
    recip_vectors : Float[Array, "3 3"]
        Reciprocal lattice basis vectors as 3×3 matrix.
    hkl_grid : Int[Array, "N 3"]
        Miller indices for N reciprocal lattice points.
    g_vectors : Float[Array, "N 3"]
        Reciprocal lattice vectors for N points.
    g_magnitudes : Float[Array, "N"]
        Magnitudes of N reciprocal vectors.
    structure_factors : Complex[Array, "N"]
        Complex structure factors for N points.
    intensities : Float[Array, "N"]
        Diffraction intensities for N points.

    Returns
    -------
    ewald_data : EwaldData
        Validated EwaldData PyTree instance.

    Notes
    -----
    1. **Convert dtypes** --
       float64 for real-valued fields, int32 for Miller
       indices, complex128 for structure factors.
    2. **Validate scalars** --
       Wavelength, k_magnitude, and sphere_radius must be
       positive.
    3. **Validate shapes** --
       recip_vectors is (3, 3); hkl_grid, g_vectors,
       g_magnitudes, structure_factors, and intensities
       share the same leading dimension N.
    4. **Validate values** --
       Intensities must be non-negative; all real-valued
       arrays must be finite.
    5. **Create instance** --
       Assemble validated arrays into EwaldData PyTree.
    """
    wavelength_ang = jnp.asarray(wavelength_ang, dtype=jnp.float64)
    k_magnitude = jnp.asarray(k_magnitude, dtype=jnp.float64)
    sphere_radius = jnp.asarray(sphere_radius, dtype=jnp.float64)
    recip_vectors = jnp.asarray(recip_vectors, dtype=jnp.float64)
    hkl_grid = jnp.asarray(hkl_grid, dtype=jnp.int32)
    g_vectors = jnp.asarray(g_vectors, dtype=jnp.float64)
    g_magnitudes = jnp.asarray(g_magnitudes, dtype=jnp.float64)
    structure_factors = jnp.asarray(structure_factors, dtype=jnp.complex128)
    intensities = jnp.asarray(intensities, dtype=jnp.float64)

    def _validate_and_create() -> EwaldData:
        if recip_vectors.shape != (3, 3):
            raise ValueError("recip_vectors must have shape (3, 3)")

        n_hkl: int = hkl_grid.shape[0]
        if (
            g_vectors.shape[0] != n_hkl
            or g_magnitudes.shape[0] != n_hkl
            or structure_factors.shape[0] != n_hkl
            or intensities.shape[0] != n_hkl
        ):
            raise ValueError("Ewald arrays must share leading dimension N")

        checked_wavelength: Float[Array, ""] = eqx.error_if(
            wavelength_ang,
            wavelength_ang <= 0,
            "wavelength_ang must be positive",
        )
        checked_k_magnitude: Float[Array, ""] = eqx.error_if(
            k_magnitude,
            k_magnitude <= 0,
            "k_magnitude must be positive",
        )
        checked_sphere_radius: Float[Array, ""] = eqx.error_if(
            sphere_radius,
            sphere_radius <= 0,
            "sphere_radius must be positive",
        )
        checked_g_vectors: Float[Array, "N 3"] = eqx.error_if(
            g_vectors,
            jnp.any(~jnp.isfinite(g_vectors)),
            "g_vectors contain non-finite values",
        )
        checked_g_magnitudes: Float[Array, "N"] = eqx.error_if(
            g_magnitudes,
            jnp.any(~jnp.isfinite(g_magnitudes)),
            "g_magnitudes contain non-finite values",
        )
        checked_intensities: Float[Array, "N"] = eqx.error_if(
            intensities,
            jnp.any(intensities < 0),
            "intensities must be non-negative",
        )
        checked_intensities = eqx.error_if(
            checked_intensities,
            jnp.any(~jnp.isfinite(checked_intensities)),
            "intensities contain non-finite values",
        )
        checked_recip_vectors: Float[Array, "3 3"] = eqx.error_if(
            recip_vectors,
            jnp.any(~jnp.isfinite(recip_vectors)),
            "recip_vectors contain non-finite values",
        )
        checked_wavelength = eqx.error_if(
            checked_wavelength,
            ~jnp.isfinite(checked_wavelength),
            "wavelength_ang must be finite",
        )
        checked_k_magnitude = eqx.error_if(
            checked_k_magnitude,
            ~jnp.isfinite(checked_k_magnitude),
            "k_magnitude must be finite",
        )
        checked_sphere_radius = eqx.error_if(
            checked_sphere_radius,
            ~jnp.isfinite(checked_sphere_radius),
            "sphere_radius must be finite",
        )

        return EwaldData(
            wavelength_ang=checked_wavelength,
            k_magnitude=checked_k_magnitude,
            sphere_radius=checked_sphere_radius,
            recip_vectors=checked_recip_vectors,
            hkl_grid=hkl_grid,
            g_vectors=checked_g_vectors,
            g_magnitudes=checked_g_magnitudes,
            structure_factors=structure_factors,
            intensities=checked_intensities,
        )

    validated_ewald_data: EwaldData = _validate_and_create()
    return validated_ewald_data


class KirklandParameters(eqx.Module):
    """Structured Kirkland coefficients for one element.

    This PyTree holds the three Lorentzian and three Gaussian
    amplitude/scale pairs from the Kirkland parameterization of
    electron scattering factors.

    Attributes
    ----------
    lorentzian_amplitudes : Float[Array, "3"]
        Lorentzian amplitude coefficients (a_1, a_2, a_3).
    lorentzian_scales : Float[Array, "3"]
        Lorentzian scale coefficients (b_1, b_2, b_3).
    gaussian_amplitudes : Float[Array, "3"]
        Gaussian amplitude coefficients (c_1, c_2, c_3).
    gaussian_scales : Float[Array, "3"]
        Gaussian scale coefficients (d_1, d_2, d_3).

    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible
    with JAX transformations like jit, grad, and vmap. All fields are
    JAX arrays and are stored as leaf nodes.
    """

    lorentzian_amplitudes: Float[Array, "3"]
    lorentzian_scales: Float[Array, "3"]
    gaussian_amplitudes: Float[Array, "3"]
    gaussian_scales: Float[Array, "3"]


@jaxtyped(typechecker=beartype)
def create_kirkland_parameters(
    lorentzian_amplitudes: Float[Array, "3"],
    lorentzian_scales: Float[Array, "3"],
    gaussian_amplitudes: Float[Array, "3"],
    gaussian_scales: Float[Array, "3"],
) -> KirklandParameters:
    """Create a KirklandParameters PyTree with data validation.

    Parameters
    ----------
    lorentzian_amplitudes : Float[Array, "3"]
        Lorentzian amplitude coefficients (a_1, a_2, a_3).
    lorentzian_scales : Float[Array, "3"]
        Lorentzian scale coefficients (b_1, b_2, b_3).
    gaussian_amplitudes : Float[Array, "3"]
        Gaussian amplitude coefficients (c_1, c_2, c_3).
    gaussian_scales : Float[Array, "3"]
        Gaussian scale coefficients (d_1, d_2, d_3).

    Returns
    -------
    validated_kirkland_parameters : KirklandParameters
        Validated KirklandParameters instance.

    Notes
    -----
    1. Convert inputs to JAX float64 arrays.
    2. Validate all arrays have exactly 3 elements.
    3. Ensure all values are finite.
    4. Create and return KirklandParameters instance.
    """
    lorentzian_amplitudes = jnp.asarray(
        lorentzian_amplitudes, dtype=jnp.float64
    )
    lorentzian_scales = jnp.asarray(lorentzian_scales, dtype=jnp.float64)
    gaussian_amplitudes = jnp.asarray(gaussian_amplitudes, dtype=jnp.float64)
    gaussian_scales = jnp.asarray(gaussian_scales, dtype=jnp.float64)
    n_coeffs: int = 3

    def _validate_and_create() -> KirklandParameters:
        if lorentzian_amplitudes.shape != (n_coeffs,):
            raise ValueError("lorentzian_amplitudes must have shape (3,)")
        if lorentzian_scales.shape != (n_coeffs,):
            raise ValueError("lorentzian_scales must have shape (3,)")
        if gaussian_amplitudes.shape != (n_coeffs,):
            raise ValueError("gaussian_amplitudes must have shape (3,)")
        if gaussian_scales.shape != (n_coeffs,):
            raise ValueError("gaussian_scales must have shape (3,)")

        checked_lorentzian_amplitudes: Float[Array, "3"] = eqx.error_if(
            lorentzian_amplitudes,
            jnp.any(~jnp.isfinite(lorentzian_amplitudes)),
            "lorentzian_amplitudes contain non-finite values",
        )
        checked_lorentzian_scales: Float[Array, "3"] = eqx.error_if(
            lorentzian_scales,
            jnp.any(~jnp.isfinite(lorentzian_scales)),
            "lorentzian_scales contain non-finite values",
        )
        checked_gaussian_amplitudes: Float[Array, "3"] = eqx.error_if(
            gaussian_amplitudes,
            jnp.any(~jnp.isfinite(gaussian_amplitudes)),
            "gaussian_amplitudes contain non-finite values",
        )
        checked_gaussian_scales: Float[Array, "3"] = eqx.error_if(
            gaussian_scales,
            jnp.any(~jnp.isfinite(gaussian_scales)),
            "gaussian_scales contain non-finite values",
        )
        return KirklandParameters(
            lorentzian_amplitudes=checked_lorentzian_amplitudes,
            lorentzian_scales=checked_lorentzian_scales,
            gaussian_amplitudes=checked_gaussian_amplitudes,
            gaussian_scales=checked_gaussian_scales,
        )

    validated_kirkland_parameters: KirklandParameters = _validate_and_create()
    return validated_kirkland_parameters


class PotentialSlices(eqx.Module):
    """JAX-compatible multislice potential data for electron beam propagation.

    This PyTree represents discretized potential data used in multislice
    electron diffraction calculations. It contains 3D potential slices with
    associated calibration information for accurate physical modeling.

    :see: :class:`~.test_crystal_types.TestPotentialSlices`

    Attributes
    ----------
    slices : Float[Array, "n_slices height width"]
        3D array containing projected-potential data for each slice.
        First dimension indexes slices, second and third dimensions are
        spatial coordinates. Units: Volt-Angstrom. These are projected
        slice potentials, not volumetric potentials.
    slice_thickness : scalar_float
        Thickness of each slice in Ångstroms. Determines the z-spacing
        between consecutive slices.
    x_calibration : scalar_float
        Real space calibration in the x-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.
    y_calibration : scalar_float
        Real space calibration in the y-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.
    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. The calibration metadata is
    preserved as auxiliary data while slice data can be efficiently processed.
    All data is immutable for functional programming patterns.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create potential slices for multislice calculation
    >>> slices_data = jnp.zeros((10, 64, 64))  # 10 slices, 64x64 each
    >>> potential_slices = rh.types.create_potential_slices(
    ...     slices=slices_data,
    ...     slice_thickness=2.0,  # 2 Å per slice
    ...     x_calibration=0.1,  # 0.1 Å per pixel in x
    ...     y_calibration=0.1,  # 0.1 Å per pixel in y
    ... )
    """

    slices: Float[Array, "n_slices height width"]
    slice_thickness: scalar_float
    x_calibration: scalar_float
    y_calibration: scalar_float


@jaxtyped(typechecker=beartype)
def create_potential_slices(
    slices: Float[Array, "n_slices height width"],
    slice_thickness: scalar_float,
    x_calibration: scalar_float,
    y_calibration: scalar_float,
) -> PotentialSlices:
    """Create a PotentialSlices PyTree with data validation.

    :see: :class:`~.test_crystal_types.TestPotentialSlices`

    Parameters
    ----------
    slices : Float[Array, "n_slices height width"]
        3D array containing projected-potential data in Volt-Angstrom
        for each slice.
    slice_thickness : scalar_float
        Thickness of each slice in Ångstroms.
    x_calibration : scalar_float
        Real space calibration in x-direction in Ångstroms per pixel.
    y_calibration : scalar_float
        Real space calibration in y-direction in Ångstroms per pixel.

    Returns
    -------
    validated_potential_slices : PotentialSlices
        Validated PotentialSlices instance.

    Notes
    -----
    1. Convert inputs to JAX arrays with appropriate dtypes.
    2. Validate slice array is 3D.
    3. Ensure slice thickness is positive.
    4. Ensure calibrations are positive.
    5. Check that all slice data is finite.
    6. Create and return PotentialSlices instance.
    """
    slices: Float[Array, "n_slices height width"] = jnp.asarray(
        slices, dtype=jnp.float64
    )
    slice_thickness: Float[Array, ""] = jnp.asarray(
        slice_thickness, dtype=jnp.float64
    )
    x_calibration: Float[Array, ""] = jnp.asarray(
        x_calibration, dtype=jnp.float64
    )
    y_calibration: Float[Array, ""] = jnp.asarray(
        y_calibration, dtype=jnp.float64
    )

    def _validate_and_create() -> PotentialSlices:
        max_dims: int = 3

        if slices.ndim != max_dims:
            raise ValueError("slices must be 3D")
        if slices.shape[0] <= 0:
            raise ValueError("slices must contain at least one slice")
        if slices.shape[1] <= 0 or slices.shape[2] <= 0:
            raise ValueError("slice dimensions must be positive")

        checked_slices: Float[Array, "n_slices height width"] = eqx.error_if(
            slices,
            jnp.any(~jnp.isfinite(slices)),
            "slices contain non-finite values",
        )
        checked_slice_thickness: Float[Array, ""] = eqx.error_if(
            slice_thickness,
            slice_thickness <= 0,
            "slice_thickness must be positive",
        )
        checked_x_calibration: Float[Array, ""] = eqx.error_if(
            x_calibration,
            x_calibration <= 0,
            "x_calibration must be positive",
        )
        checked_y_calibration: Float[Array, ""] = eqx.error_if(
            y_calibration,
            y_calibration <= 0,
            "y_calibration must be positive",
        )
        return PotentialSlices(
            slices=checked_slices,
            slice_thickness=checked_slice_thickness,
            x_calibration=checked_x_calibration,
            y_calibration=checked_y_calibration,
        )

    validated_potential_slices: PotentialSlices = _validate_and_create()
    return validated_potential_slices


class EdgeOnSlices(eqx.Module):
    """JAX-compatible edge-on potential slices for RHEED reflection.

    This PyTree stores projected potentials for multislice propagation along
    an in-plane beam axis. Each slice is projected along ``x`` and resolved on
    the transverse ``(y, z)`` plane, where ``z`` is the surface normal.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Attributes
    ----------
    slices : Float[Array, "nx_slices ny nz"]
        Projected potential for each beam-axis slice in Volt-Angstrom.
    dx_slice : scalar_float
        Slice thickness along the beam axis in Angstroms.
    dy : scalar_float
        Transverse in-plane grid spacing in Angstroms.
    dz : scalar_float
        Surface-normal grid spacing in Angstroms.
    y_extent : scalar_float
        Periodic transverse in-plane cell length in Angstroms.
    z_lo : scalar_float
        Lower edge of the open surface-normal simulation window.
    z_surf : scalar_float
        Surface height, the top of the atomic slab.
    cap_width : scalar_float
        Absorbing-layer thickness at both z-window edges.
    """

    slices: Float[Array, "nx_slices ny nz"]
    dx_slice: scalar_float
    dy: scalar_float
    dz: scalar_float
    y_extent: scalar_float
    z_lo: scalar_float
    z_surf: scalar_float
    cap_width: scalar_float


@jaxtyped(typechecker=beartype)
def create_edge_on_slices(
    slices: Float[Array, "nx_slices ny nz"],
    dx_slice: scalar_float,
    dy: scalar_float,
    dz: scalar_float,
    y_extent: scalar_float,
    z_lo: scalar_float,
    z_surf: scalar_float,
    cap_width: scalar_float,
) -> EdgeOnSlices:
    """Create an EdgeOnSlices PyTree with data validation.

    :see: :class:`~.test_reflection_multislice.TestReflectionMultislice`

    Parameters
    ----------
    slices : Float[Array, "nx_slices ny nz"]
        Edge-on projected potentials in Volt-Angstrom.
    dx_slice, dy, dz : scalar_float
        Beam-axis and transverse grid spacings in Angstroms.
    y_extent : scalar_float
        Periodic transverse in-plane cell length in Angstroms.
    z_lo : scalar_float
        Lower edge of the surface-normal window in Angstroms.
    z_surf : scalar_float
        Surface height in Angstroms.
    cap_width : scalar_float
        Absorbing-layer thickness in Angstroms.

    Returns
    -------
    validated_edge_on_slices : EdgeOnSlices
        Validated edge-on slice container.

    Notes
    -----
    1. Convert all numeric inputs to JAX arrays.
    2. Validate the projected-potential array rank and dimensions.
    3. Ensure spacings, extents, and CAP width are positive.
    4. Ensure all potential samples are finite.
    """
    slices = jnp.asarray(slices, dtype=jnp.float64)
    dx_slice = jnp.asarray(dx_slice, dtype=jnp.float64)
    dy = jnp.asarray(dy, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    y_extent = jnp.asarray(y_extent, dtype=jnp.float64)
    z_lo = jnp.asarray(z_lo, dtype=jnp.float64)
    z_surf = jnp.asarray(z_surf, dtype=jnp.float64)
    cap_width = jnp.asarray(cap_width, dtype=jnp.float64)

    def _validate_and_create() -> EdgeOnSlices:
        if slices.ndim != 3:
            raise ValueError("slices must be 3D")
        if slices.shape[0] <= 0:
            raise ValueError("slices must contain at least one x slice")
        if slices.shape[1] <= 0 or slices.shape[2] <= 0:
            raise ValueError("transverse slice dimensions must be positive")

        checked_slices = eqx.error_if(
            slices,
            jnp.any(~jnp.isfinite(slices)),
            "slices contain non-finite values",
        )
        checked_dx = eqx.error_if(
            dx_slice, dx_slice <= 0, "dx_slice must be positive"
        )
        checked_dy = eqx.error_if(dy, dy <= 0, "dy must be positive")
        checked_dz = eqx.error_if(dz, dz <= 0, "dz must be positive")
        checked_y_extent = eqx.error_if(
            y_extent, y_extent <= 0, "y_extent must be positive"
        )
        checked_cap_width = eqx.error_if(
            cap_width, cap_width <= 0, "cap_width must be positive"
        )
        return EdgeOnSlices(
            slices=checked_slices,
            dx_slice=checked_dx,
            dy=checked_dy,
            dz=checked_dz,
            y_extent=checked_y_extent,
            z_lo=z_lo,
            z_surf=z_surf,
            cap_width=checked_cap_width,
        )

    validated_edge_on_slices: EdgeOnSlices = _validate_and_create()
    return validated_edge_on_slices


class XYZData(eqx.Module):
    """JAX-compatible representation of parsed XYZ file data.

    This PyTree represents a complete XYZ file structure with atomic positions,
    optional lattice information, and metadata. It's designed for geometry
    parsing, simulation preparation, and machine learning data processing.

    This class is an Equinox module (``eqx.Module``) registered as a JAX
    PyTree node, making it compatible with JAX transformations like jit,
    grad, and vmap. Numerical data is stored as JAX arrays while metadata is
    preserved as auxiliary data. All data is immutable for functional
    programming patterns.

    :see: :class:`~.test_crystal_types.TestXYZData`

    Attributes
    ----------
    positions : Float[Array, "N 3"]
        Cartesian atomic positions in Ångstroms. Shape (N, 3) where N is
        the number of atoms.
    atomic_numbers : Int[Array, "N"]
        Atomic numbers (Z) corresponding to each atom. Shape (N,) with
        integer values.
    lattice : Optional[Float[Array, "3 3"]]
        Lattice vectors in Ångstroms if present in the XYZ file, otherwise
        None. Shape (3, 3) matrix where each row is a lattice vector.
        When absent, the field stays None; no placeholder lattice is
        fabricated.
    stress : Optional[Float[Array, "3 3"]]
        Symmetric stress tensor if present in the metadata, otherwise None.
        Shape (3, 3) matrix with stress components.
    energy : Optional[scalar_float]
        Total energy in eV if present in the metadata, otherwise None.
        Scalar value.
    forces : Optional[Float[Array, "N 3"]]
        Per-atom forces in eV/Ångstrom if present in the source data,
        otherwise None. Shape (N, 3) matching ``positions``.
    properties : Optional[List[Dict[str, Union[str, int]]]]
        List of per-atom properties described in the metadata, otherwise None.
    comment : Optional[str]
        The raw comment line from the XYZ file header, otherwise None.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create XYZ data for water molecule
    >>> positions = jnp.array(
    ...     [
    ...         [0.0, 0.0, 0.0],
    ...         [0.76, 0.59, 0.0],
    ...         [-0.76, 0.59, 0.0],
    ...     ]
    ... )
    >>> atomic_numbers = jnp.array([8, 1, 1])  # O, H, H
    >>> xyz_data = rh.types.create_xyz_data(
    ...     positions=positions,
    ...     atomic_numbers=atomic_numbers,
    ...     comment="Water molecule",
    ... )
    """

    positions: Float[Array, "N 3"]
    atomic_numbers: Int[Array, "N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, ""]]
    forces: Optional[Float[Array, "N 3"]] = None
    properties: Optional[List[Dict[str, Union[str, int]]]] = eqx.field(
        static=True, default=None
    )
    comment: Optional[str] = eqx.field(static=True, default=None)


@jaxtyped(typechecker=beartype)
def create_xyz_data(
    positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
    forces: Optional[Float[Array, "N 3"]] = None,
) -> XYZData:
    """Create a XYZData PyTree with runtime validation.

    :see: :class:`~.test_crystal_types.TestXYZData`

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian positions in Ångstroms.
    atomic_numbers : Int[Array, "N"]
        Atomic numbers (Z) for each atom.
    lattice : Optional[Float[Array, "3 3"]], optional
        Lattice vectors (if any). ``None`` is stored as ``None``; no
        identity-matrix placeholder is fabricated, so downstream
        consumers can distinguish "no lattice given" from a genuine
        1 Ångstrom cubic cell.
    stress : Optional[Float[Array, "3 3"]], optional
        Stress tensor (if any).
    energy : Optional[scalar_float], optional
        Total energy (if any).
    properties : Optional[List[Dict[str, Union[str, int]]]], optional
        Per-atom metadata.
    comment : Optional[str], optional
        Original XYZ comment line.
    forces : Optional[Float[Array, "N 3"]], optional
        Per-atom forces in eV/Ångstrom (if any).

    Returns
    -------
    validated_xyz_data : XYZData
        Validated PyTree structure for XYZ file contents.

    Notes
    -----
    - Convert required inputs to JAX arrays with appropriate
      dtypes: positions to float64, atomic_numbers to int32,
      lattice/stress/energy/forces to float64 if provided.
    - Execute shape validation checks: verify positions has
      shape (N, 3) and atomic_numbers has shape (N,).
    - Execute value validation checks: ensure all position
      values are finite and atomic numbers are non-negative.
    - Execute optional matrix validation checks: for lattice
      and stress tensors verify shape is (3, 3) and all
      values are finite; for forces verify shape is (N, 3)
      and all values are finite.
    - If all validations pass, create and return XYZData
      instance.
    - If any validation fails, raise ValueError with
      descriptive error message.
    """
    positions: Float[Array, "N 3"] = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers: Int[Array, "N"] = jnp.asarray(
        atomic_numbers, dtype=jnp.int32
    )
    if lattice is not None:
        lattice: Float[Array, "3 3"] = jnp.asarray(lattice, dtype=jnp.float64)

    if stress is not None:
        stress: Float[Array, "3 3"] = jnp.asarray(stress, dtype=jnp.float64)

    if energy is not None:
        energy: Float[Array, ""] = jnp.asarray(energy, dtype=jnp.float64)

    if forces is not None:
        forces: Float[Array, "N 3"] = jnp.asarray(forces, dtype=jnp.float64)

    def _validate_and_create() -> XYZData:
        nn: int = positions.shape[0]
        max_dims: int = 3

        if positions.shape[1] != max_dims:
            raise ValueError("positions must have shape (N, 3)")
        if atomic_numbers.shape[0] != nn:
            raise ValueError("atomic_numbers must have shape (N,)")
        if lattice is not None and lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if stress is not None and stress.shape != (3, 3):
            raise ValueError("stress must have shape (3, 3)")
        if forces is not None and forces.shape != (nn, max_dims):
            raise ValueError("forces must have shape (N, 3)")

        checked_positions: Float[Array, "N 3"] = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)),
            "positions contain non-finite values",
        )
        checked_atomic_numbers: Int[Array, "N"] = eqx.error_if(
            atomic_numbers,
            jnp.any(atomic_numbers < 0),
            "atomic_numbers must be non-negative",
        )
        checked_lattice: Optional[Float[Array, "3 3"]] = (
            None
            if lattice is None
            else eqx.error_if(
                lattice,
                jnp.any(~jnp.isfinite(lattice)),
                "lattice contains non-finite values",
            )
        )
        checked_stress: Optional[Float[Array, "3 3"]] = (
            None
            if stress is None
            else eqx.error_if(
                stress,
                jnp.any(~jnp.isfinite(stress)),
                "stress contains non-finite values",
            )
        )
        checked_energy: Optional[Float[Array, ""]] = (
            None
            if energy is None
            else eqx.error_if(
                energy,
                ~jnp.isfinite(energy),
                "energy must be finite",
            )
        )
        checked_forces: Optional[Float[Array, "N 3"]] = (
            None
            if forces is None
            else eqx.error_if(
                forces,
                jnp.any(~jnp.isfinite(forces)),
                "forces contain non-finite values",
            )
        )

        return XYZData(
            positions=checked_positions,
            atomic_numbers=checked_atomic_numbers,
            lattice=checked_lattice,
            stress=checked_stress,
            energy=checked_energy,
            forces=checked_forces,
            properties=properties,
            comment=comment,
        )

    validated_xyz_data: XYZData = _validate_and_create()
    return validated_xyz_data


__all__: list[str] = [
    "CrystalStructure",
    "EwaldData",
    "PotentialSlices",
    "XYZData",
    "create_crystal_structure",
    "create_ewald_data",
    "create_potential_slices",
    "create_xyz_data",
]
