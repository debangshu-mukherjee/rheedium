"""Data structures and factory functions for crystal structure representation.

Extended Summary
----------------
This module defines JAX-compatible data structures for representing crystal
structures, potential slices for multislice simulations, and XYZ file data.
All structures are PyTrees that support JAX transformations.

Routine Listings
----------------
CrystalStructure : PyTree
    JAX-compatible crystal structure with fractional and Cartesian coordinates
PotentialSlices : PyTree
    JAX-compatible data structure for representing multislice potential data
XYZData : PyTree
    A PyTree for XYZ file data with atomic positions and metadata
create_crystal_structure : function
    Factory function to create CrystalStructure instances with data validation
create_potential_slices : function
    Factory function to create PotentialSlices instances with data validation
create_xyz_data : function
    Factory function to create XYZData instances with data validation

Notes
-----
All data structures are immutable and support automatic differentiation
through JAX's PyTree mechanism.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, NamedTuple, Optional, Tuple, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, Num, jaxtyped

from .custom_types import scalar_float


@register_pytree_node_class
class CrystalStructure(NamedTuple):
    """JAX-compatible Pytree with fractional and Cartesian coordinates.

    This PyTree represents a crystal structure containing atomic positions in
    both fractional and Cartesian coordinate systems, along with unit cell
    parameters. It's designed for efficient crystal structure calculations and
    electron diffraction simulations.

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
    This class is registered as a PyTree node, making it compatible with JAX
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
    ...     cell_angles=cell_angles
    ... )
    """

    frac_positions: Num[Array, "N 4"]
    cart_positions: Num[Array, "N 4"]
    cell_lengths: Num[Array, "3"]
    cell_angles: Num[Array, "3"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "N 4"],
            Num[Array, "N 4"],
            Num[Array, "3"],
            Num[Array, "3"],
        ],
        None,
    ]:
        """Flatten the PyTree into a tuple of arrays."""
        return (
            (
                self.frac_positions,
                self.cart_positions,
                self.cell_lengths,
                self.cell_angles,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Tuple[
            Float[Array, "N 4"],
            Num[Array, "N 4"],
            Num[Array, "3"],
            Num[Array, "3"],
        ],
    ) -> "CrystalStructure":
        """Unflatten the PyTree into a CrystalStructure instance."""
        del aux_data
        return cls(*children)


@beartype
def create_crystal_structure(
    frac_positions: Num[Array, "N 4"],
    cart_positions: Num[Array, "N 4"],
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
) -> CrystalStructure:
    """Create a CrystalStructure PyTree with data validation.

    Parameters
    ----------
    frac_positions : Num[Array, "N 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional
        coordinates.
    cart_positions : Num[Array, "N 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian
        coordinates.
    cell_lengths : Num[Array, "3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    cell_angles : Num[Array, "3"]
        Unit cell angles [α, β, γ] in degrees.

    Returns
    -------
    validated_crystal_structure : CrystalStructure
        A validated CrystalStructure instance.

    Notes
    -----
    - Convert all inputs to JAX arrays using jnp.asarray.
    - Validate shapes of frac_positions, cart_positions, cell_lengths,.
      and cell_angles.
    - Verify number of atoms matches between frac and cart positions
    - Verify atomic numbers match between frac and cart positions
    - Ensure cell lengths are positive
    - Ensure cell angles are between 0 and 180 degrees
    - Create and return CrystalStructure instance with validated data
    """
    frac_positions: Float[Array, "N 4"] = jnp.asarray(frac_positions)
    cart_positions: Num[Array, "N 4"] = jnp.asarray(cart_positions)
    cell_lengths: Num[Array, "3"] = jnp.asarray(cell_lengths)
    cell_angles: Num[Array, "3"] = jnp.asarray(cell_angles)

    def _validate_and_create() -> CrystalStructure:
        max_cols: int = 4

        def _check_frac_shape() -> Float[Array, "N max_cols"]:
            return lax.cond(
                frac_positions.shape[1] == max_cols,
                lambda: frac_positions,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: frac_positions, lambda: frac_positions
                    )
                ),
            )

        def _check_cart_shape() -> Num[Array, "N max_cols"]:
            return lax.cond(
                cart_positions.shape[1] == max_cols,
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: cart_positions, lambda: cart_positions
                    )
                ),
            )

        def _check_cell_lengths_shape() -> Num[Array, "3"]:
            return lax.cond(
                cell_lengths.shape == (3,),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def _check_cell_angles_shape() -> Num[Array, "3"]:
            return lax.cond(
                cell_angles.shape == (3,),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        def _check_atom_count() -> (
            Tuple[Float[Array, "N 4"], Num[Array, "N 4"]]
        ):
            return lax.cond(
                frac_positions.shape[0] == cart_positions.shape[0],
                lambda: (frac_positions, cart_positions),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (frac_positions, cart_positions),
                        lambda: (frac_positions, cart_positions),
                    )
                ),
            )

        def _check_atomic_numbers() -> (
            Tuple[Float[Array, "N 4"], Num[Array, "N 4"]]
        ):
            return lax.cond(
                jnp.all(frac_positions[:, 3] == cart_positions[:, 3]),
                lambda: (frac_positions, cart_positions),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (frac_positions, cart_positions),
                        lambda: (frac_positions, cart_positions),
                    )
                ),
            )

        def _check_cell_lengths_positive() -> Num[Array, "3"]:
            return lax.cond(
                jnp.all(cell_lengths > 0),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def _check_cell_angles_valid() -> Num[Array, "3"]:
            min_angle: scalar_float = 0.0
            max_angle: scalar_float = 180.0
            return lax.cond(
                jnp.all(
                    jnp.logical_and(
                        cell_angles > min_angle, cell_angles < max_angle
                    )
                ),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        _check_frac_shape()
        _check_cart_shape()
        _check_cell_lengths_shape()
        _check_cell_angles_shape()
        _check_atom_count()
        _check_atomic_numbers()
        _check_cell_lengths_positive()
        _check_cell_angles_valid()
        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

    validated_crystal_structure: CrystalStructure = _validate_and_create()
    return validated_crystal_structure


@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """JAX-compatible multislice potential data for electron beam propagation.

    This PyTree represents discretized potential data used in multislice
    electron diffraction calculations. It contains 3D potential slices with
    associated calibration information for accurate physical modeling.

    Attributes
    ----------
    slices : Float[Array, "n_slices height width"]
        3D array containing potential data for each slice. First dimension
        indexes slices, second and third dimensions are spatial coordinates.
        Units: Volts or appropriate potential units.
    slice_thickness : scalar_float
        Thickness of each slice in Ångstroms. Determines the z-spacing
        between consecutive slices.
    x_calibration : scalar_float
        Real space calibration in the x-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.
    y_calibration : scalar_float
        Real space calibration in the y-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.
    This class is registered as a PyTree node, making it compatible with JAX
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
    ...     x_calibration=0.1,    # 0.1 Å per pixel in x
    ...     y_calibration=0.1     # 0.1 Å per pixel in y
    ... )
    """

    slices: Float[Array, "n_slices height width"]
    slice_thickness: scalar_float
    x_calibration: scalar_float
    y_calibration: scalar_float

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, "n_slices height width"]],
        Tuple[scalar_float, scalar_float, scalar_float],
    ]:
        """Flatten the PyTree into a tuple of arrays."""
        return (
            (self.slices,),
            (self.slice_thickness, self.x_calibration, self.y_calibration),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[scalar_float, scalar_float, scalar_float],
        children: Tuple[Float[Array, "n_slices height width"]],
    ) -> "PotentialSlices":
        """Unflatten the PyTree into a PotentialSlices instance."""
        slice_thickness: scalar_float
        x_calibration: scalar_float
        y_calibration: scalar_float
        slice_thickness, x_calibration, y_calibration = aux_data
        slices: Float[Array, "n_slices height width"] = children[0]
        return cls(
            slices=slices,
            slice_thickness=slice_thickness,
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )


@jaxtyped(typechecker=beartype)
def create_potential_slices(
    slices: Float[Array, "n_slices height width"],
    slice_thickness: scalar_float,
    x_calibration: scalar_float,
    y_calibration: scalar_float,
) -> PotentialSlices:
    """Create a PotentialSlices PyTree with data validation.

    Parameters
    ----------
    slices : Float[Array, "n_slices height width"]
        3D array containing potential data for each slice.
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
    The algorithm proceeds as follows:

    1. Convert inputs to JAX arrays with appropriate dtypes
    2. Validate slice array is 3D
    3. Ensure slice thickness is positive
    4. Ensure calibrations are positive
    5. Check that all slice data is finite
    6. Create and return PotentialSlices instance
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

        def _check_3d() -> Float[Array, "n_slices height width"]:
            return lax.cond(
                slices.ndim == max_dims,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def _check_slice_count() -> Float[Array, "n_slices height width"]:
            return lax.cond(
                slices.shape[0] > 0,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def _check_slice_dimensions() -> (
            Float[Array, "n_slices height width"]
        ):
            return lax.cond(
                jnp.logical_and(slices.shape[1] > 0, slices.shape[2] > 0),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def _check_thickness() -> Float[Array, ""]:
            return lax.cond(
                slice_thickness > 0,
                lambda: slice_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: slice_thickness, lambda: slice_thickness
                    )
                ),
            )

        def _check_x_cal() -> Float[Array, ""]:
            return lax.cond(
                x_calibration > 0,
                lambda: x_calibration,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: x_calibration, lambda: x_calibration
                    )
                ),
            )

        def _check_y_cal() -> Float[Array, ""]:
            return lax.cond(
                y_calibration > 0,
                lambda: y_calibration,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: y_calibration, lambda: y_calibration
                    )
                ),
            )

        def _check_finite() -> Float[Array, "n_slices height width"]:
            return lax.cond(
                jnp.all(jnp.isfinite(slices)),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        _check_3d()
        _check_slice_count()
        _check_slice_dimensions()
        _check_thickness()
        _check_x_cal()
        _check_y_cal()
        _check_finite()
        return PotentialSlices(
            slices=slices,
            slice_thickness=slice_thickness,
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    validated_potential_slices: PotentialSlices = _validate_and_create()
    return validated_potential_slices


@register_pytree_node_class
class XYZData(NamedTuple):
    """JAX-compatible representation of parsed XYZ file data.

    This PyTree represents a complete XYZ file structure with atomic positions,
    optional lattice information, and metadata. It's designed for geometry
    parsing, simulation preparation, and machine learning data processing.

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
    stress : Optional[Float[Array, "3 3"]]
        Symmetric stress tensor if present in the metadata, otherwise None.
        Shape (3, 3) matrix with stress components.
    energy : Optional[scalar_float]
        Total energy in eV if present in the metadata, otherwise None.
        Scalar value.
    properties : Optional[List[Dict[str, Union[str, int]]]]
        List of per-atom properties described in the metadata, otherwise None.
    comment : Optional[str]
        The raw comment line from the XYZ file header, otherwise None.
    This class is registered as a PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. Numerical data is stored as
    JAX arrays while metadata is preserved as auxiliary data. All data is
    immutable for functional programming patterns.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create XYZ data for water molecule
    >>> positions = jnp.array(
    ...     [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]
    ... )
    >>> atomic_numbers = jnp.array([8, 1, 1])  # O, H, H
    >>> xyz_data = rh.types.create_xyz_data(
    ...     positions=positions,
    ...     atomic_numbers=atomic_numbers,
    ...     comment="Water molecule"
    ... )
    """

    positions: Float[Array, "N 3"]
    atomic_numbers: Int[Array, "N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, ""]]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, "N 3"],
            Int[Array, "N"],
            Optional[Float[Array, "3 3"]],
            Optional[Float[Array, "3 3"]],
            Optional[Float[Array, ""]],
        ],
        Dict[str, Optional[List[Dict[str, Union[str, int]]]]],
    ]:
        """Flatten the PyTree into a tuple of arrays."""
        children = (
            self.positions,
            self.atomic_numbers,
            self.lattice,
            self.stress,
            self.energy,
        )
        aux_data = {
            "properties": self.properties,
            "comment": self.comment,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Dict[str, Optional[List[Dict[str, Union[str, int]]]]],
        children: Tuple[
            Float[Array, "N 3"],
            Int[Array, "N"],
            Optional[Float[Array, "3 3"]],
            Optional[Float[Array, "3 3"]],
            Optional[Float[Array, ""]],
        ],
    ) -> "XYZData":
        """Unflatten the PyTree into a XYZData instance."""
        positions: Float[Array, "N 3"]
        atomic_numbers: Int[Array, "N"]
        lattice: Optional[Float[Array, "3 3"]]
        stress: Optional[Float[Array, "3 3"]]
        energy: Optional[Float[Array, ""]]
        positions, atomic_numbers, lattice, stress, energy = children
        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=aux_data["properties"],
            comment=aux_data["comment"],
        )


@jaxtyped(typechecker=beartype)
def create_xyz_data(
    positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> XYZData:
    """Create a XYZData PyTree with runtime validation.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian positions in Ångstroms.
    atomic_numbers : Int[Array, "N"]
        Atomic numbers (Z) for each atom.
    lattice : Optional[Float[Array, "3 3"]], optional
        Lattice vectors (if any).
    stress : Optional[Float[Array, "3 3"]], optional
        Stress tensor (if any).
    energy : Optional[scalar_float], optional
        Total energy (if any).
    properties : Optional[List[Dict[str, Union[str, int]]]], optional
        Per-atom metadata.
    comment : Optional[str], optional
        Original XYZ comment line.

    Returns
    -------
    validated_xyz_data : XYZData
        Validated PyTree structure for XYZ file contents.

    Notes
    -----
    - Convert required inputs to JAX arrays with appropriate dtypes
      positions to float64, atomic_numbers to int32, lattice/stress/energy
      to float64 if provided
    - Execute shape validation checks: verify positions has shape (N, 3)
      and atomic_numbers has shape (N,)
    - Execute value validation checks: ensure all position values are finite
      and atomic numbers are non-negative
    - Execute optional matrix validation checks: for lattice and stress tensors
      verify shape is (3, 3) and all values are finite
    - If all validations pass, create and return XYZData instance
    - If any validation fails, raise ValueError with descriptive error message
    """
    positions: Float[Array, "N 3"] = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers: Int[Array, "N"] = jnp.asarray(
        atomic_numbers, dtype=jnp.int32
    )
    if lattice is not None:
        lattice: Float[Array, "3 3"] = jnp.asarray(lattice, dtype=jnp.float64)
    else:
        lattice: Float[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)

    if stress is not None:
        stress: Float[Array, "3 3"] = jnp.asarray(stress, dtype=jnp.float64)

    if energy is not None:
        energy: Float[Array, ""] = jnp.asarray(energy, dtype=jnp.float64)

    def _validate_and_create() -> XYZData:
        nn: int = positions.shape[0]
        max_dims: int = 3

        def _check_shape() -> None:
            if positions.shape[1] != max_dims:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers.shape[0] != nn:
                raise ValueError("atomic_numbers must have shape (N,)")

        def _check_finiteness() -> None:
            if not jnp.all(jnp.isfinite(positions)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def _check_optional_matrices() -> None:
            if lattice is not None:
                if lattice.shape != (3, 3):
                    raise ValueError("lattice must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(lattice)):
                    raise ValueError("lattice contains non-finite values")

            if stress is not None:
                if stress.shape != (3, 3):
                    raise ValueError("stress must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(stress)):
                    raise ValueError("stress contains non-finite values")

        _check_shape()
        _check_finiteness()
        _check_optional_matrices()

        return XYZData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=properties,
            comment=comment,
        )

    validated_xyz_data: XYZData = _validate_and_create()
    return validated_xyz_data
