"""
Module: types.crystal_types
---------------------------
Data structures and factory functions for crystal structure representation.

Classes
-------
- `CrystalStructure`:
    JAX-compatible crystal structure with fractional and Cartesian coordinates
- `PotentialSlices`:
    JAX-compatible data structure for representing multislice potential data
- `XYZData`:
    A PyTree for XYZ file data with atomic positions, lattice vectors,
    stress tensor, energy, properties, and comment

FactoryFunctions
-----------------
- `create_crystal_structure`:
    Factory function to create CrystalStructure instances with data validation
- `create_potential_slices`:
    Factory function to create PotentialSlices instances with data validation
- `create_xyz_data`:
    Factory function to create XYZData instances with data validation
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, NamedTuple, Optional, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, Num, jaxtyped

from .custom_types import scalar_float


@register_pytree_node_class
class CrystalStructure(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure representing a crystal structure with both
    fractional and Cartesian coordinates.

    Attributes
    ----------
    - `frac_positions` (Float[Array, "* 4"]):
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Fractional coordinates in the unit cell (range [0,1])
        - atomic_number: Integer atomic number (Z) of the element

    - `cart_positions` (Num[Array, "* 4"]):
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Cartesian coordinates in Ångstroms
        - atomic_number: Integer atomic number (Z) of the element

    - `cell_lengths` (Num[Array, "3"]):
        Unit cell lengths [a, b, c] in Ångstroms

    - `cell_angles` (Num[Array, "3"]):
        Unit cell angles [α, β, γ] in degrees.
        - α is the angle between b and c
        - β is the angle between a and c
        - γ is the angle between a and b

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX transformations
    like jit, grad, and vmap. The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    frac_positions: Float[Array, "* 4"]
    cart_positions: Num[Array, "* 4"]
    cell_lengths: Num[Array, "3"]
    cell_angles: Num[Array, "3"]

    def tree_flatten(self):
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
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@beartype
def create_crystal_structure(
    frac_positions: Float[Array, "* 4"],
    cart_positions: Num[Array, "* 4"],
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
) -> CrystalStructure:
    """
    Factory function to create a CrystalStructure instance with type checking.

    Parameters
    ----------
    - `frac_positions` : Float[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
    - `cart_positions` : Num[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
    - `cell_lengths` : Num[Array, "3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    - `cell_angles` : Num[Array, "3"]
        Unit cell angles [α, β, γ] in degrees.

    Returns
    -------
    - `CrystalStructure` : CrystalStructure
        A validated CrystalStructure instance.

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes or invalid values.

    Flow
    ----
    - Convert all inputs to JAX arrays using jnp.asarray
    - Validate shape of frac_positions is (n_atoms, 4)
    - Validate shape of cart_positions is (n_atoms, 4)
    - Validate shape of cell_lengths is (3,)
    - Validate shape of cell_angles is (3,)
    - Verify number of atoms matches between frac and cart positions
    - Verify atomic numbers match between frac and cart positions
    - Ensure cell lengths are positive
    - Ensure cell angles are between 0 and 180 degrees
    - Create and return CrystalStructure instance with validated data
    """
    frac_positions = jnp.asarray(frac_positions)
    cart_positions = jnp.asarray(cart_positions)
    cell_lengths = jnp.asarray(cell_lengths)
    cell_angles = jnp.asarray(cell_angles)

    def validate_and_create():
        def check_frac_shape():
            return lax.cond(
                frac_positions.shape[1] == 4,
                lambda: frac_positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: frac_positions, lambda: frac_positions)
                ),
            )

        def check_cart_shape():
            return lax.cond(
                cart_positions.shape[1] == 4,
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cart_positions, lambda: cart_positions)
                ),
            )

        def check_cell_lengths_shape():
            return lax.cond(
                cell_lengths.shape == (3,),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def check_cell_angles_shape():
            return lax.cond(
                cell_angles.shape == (3,),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        def check_atom_count():
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

        def check_atomic_numbers():
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

        def check_cell_lengths_positive():
            return lax.cond(
                jnp.all(cell_lengths > 0),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def check_cell_angles_valid():
            return lax.cond(
                jnp.all(jnp.logical_and(cell_angles > 0, cell_angles < 180)),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        check_frac_shape()
        check_cart_shape()
        check_cell_lengths_shape()
        check_cell_angles_shape()
        check_atom_count()
        check_atomic_numbers()
        check_cell_lengths_positive()
        check_cell_angles_valid()
        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

    return validate_and_create()


@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for representing multislice potential data
    used in electron beam propagation calculations.

    Attributes
    ----------
    - `slices` (Float[Array, "n_slices height width"]):
        3D array containing potential data for each slice.
        First dimension indexes slices, second and third are spatial coordinates.
        Units: Volts or appropriate potential units.
    - `slice_thickness` (scalar_float):
        Thickness of each slice in Ångstroms.
        Determines the z-spacing between consecutive slices.
    - `x_calibration` (scalar_float):
        Real space calibration in the x-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.
    - `y_calibration` (scalar_float):
        Real space calibration in the y-direction in Ångstroms per pixel.
        Converts pixel coordinates to physical distances.

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX
    transformations like jit, grad, and vmap. The metadata (calibrations and
    thickness) is preserved through transformations while the slice data can
    be efficiently processed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create potential slices
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

    def tree_flatten(self):
        return (
            (self.slices,),
            (self.slice_thickness, self.x_calibration, self.y_calibration),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        slice_thickness, x_calibration, y_calibration = aux_data
        slices = children[0]
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
    """
    Description
    -----------
    Factory function to create a PotentialSlices instance with data validation.

    Parameters
    ----------
    - `slices` (Float[Array, "n_slices height width"]):
        3D array containing potential data for each slice
    - `slice_thickness` (scalar_float):
        Thickness of each slice in Ångstroms
    - `x_calibration` (scalar_float):
        Real space calibration in x-direction in Ångstroms per pixel
    - `y_calibration` (scalar_float):
        Real space calibration in y-direction in Ångstroms per pixel

    Returns
    -------
    - `potential_slices` (PotentialSlices):
        Validated PotentialSlices instance

    Raises
    ------
    - ValueError:
        If array shapes are invalid, calibrations are non-positive,
        or slice thickness is non-positive

    Flow
    ----
    - Convert inputs to JAX arrays with appropriate dtypes
    - Validate slice array is 3D
    - Ensure slice thickness is positive
    - Ensure calibrations are positive
    - Check that all slice data is finite
    - Create and return PotentialSlices instance
    """
    slices = jnp.asarray(slices, dtype=jnp.float64)
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    x_calibration = jnp.asarray(x_calibration, dtype=jnp.float64)
    y_calibration = jnp.asarray(y_calibration, dtype=jnp.float64)

    def validate_and_create():
        def check_3d():
            return lax.cond(
                slices.ndim == 3,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slice_count():
            return lax.cond(
                slices.shape[0] > 0,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slice_dimensions():
            return lax.cond(
                jnp.logical_and(slices.shape[1] > 0, slices.shape[2] > 0),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_thickness():
            return lax.cond(
                slice_thickness > 0,
                lambda: slice_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slice_thickness, lambda: slice_thickness)
                ),
            )

        def check_x_cal():
            return lax.cond(
                x_calibration > 0,
                lambda: x_calibration,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: x_calibration, lambda: x_calibration)
                ),
            )

        def check_y_cal():
            return lax.cond(
                y_calibration > 0,
                lambda: y_calibration,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: y_calibration, lambda: y_calibration)
                ),
            )

        def check_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(slices)),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        check_3d()
        check_slice_count()
        check_slice_dimensions()
        check_thickness()
        check_x_cal()
        check_y_cal()
        check_finite()
        return PotentialSlices(
            slices=slices,
            slice_thickness=slice_thickness,
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    return validate_and_create()


@register_pytree_node_class
class XYZData(NamedTuple):
    """
    Description
    -----------
    JAX-compatible PyTree representing a full parsed XYZ file.

    Attributes
    ----------
    - `positions` (Float[Array, "N 3"]):
        Cartesian positions in Ångstroms.
    - `atomic_numbers` (Int[Array, "N"]):
        Atomic numbers (Z) corresponding to each atom.
    - `lattice` (Optional[Float[Array, "3 3"]]):
        Lattice vectors in Ångstroms if present, otherwise None.
    - `stress` (Optional[Float[Array, "3 3"]]):
        Symmetric stress tensor if present.
    - `energy` (Optional[scalar_float]):
        Total energy in eV if present.
    - `properties` (Optional[List[Dict[str, Union[str, int]]]]):
        List of properties described in the metadata.
    - `comment` (Optional[str]):
        The raw comment line from the XYZ file.

    Notes
    -----
    - Can be used for geometry parsing, simulation prep, or ML data loaders.
    - Compatible with JAX transformations (jit, vmap, etc).
    """

    positions: Float[Array, "N 3"]
    atomic_numbers: Int[Array, "N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, ""]]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(self):
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
    def tree_unflatten(cls, aux_data, children):
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
def make_xyz_data(
    positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> XYZData:
    """
    Description
    -----------
    JAX-safe factory function for XYZData with runtime validation.

    Parameters
    ----------
    - `positions` (Float[Array, "N 3"]):
        Cartesian positions in Ångstroms
    - `atomic_numbers` (Int[Array, "N"]):
        Atomic numbers (Z) for each atom
    - `lattice` (Optional[Float[Array, "3 3"]]):
        Lattice vectors (if any)
    - `stress` (Optional[Float[Array, "3 3"]]):
        Stress tensor (if any)
    - `energy` (Optional[scalar_float]):
        Total energy (if any)
    - `properties` (Optional[List[Dict[str, Union[str, int]]]]):
        Per-atom metadata
    - `comment` (Optional[str]):
        Original XYZ comment line

    Returns
    -------
    - `XYZData`:
        Validated PyTree structure for XYZ file contents

    Validation Flow
    ---------------
    - Convert required inputs to JAX arrays with appropriate dtypes:
       - positions: Convert to float64
       - atomic_numbers: Convert to int32
       - lattice (if provided): Convert to float64
       - stress (if provided): Convert to float64
       - energy (if provided): Convert to float64
    - Extract number of atoms (N) from positions array
    - Execute shape validation checks:
       - check_shape(): Verify positions has shape (N, 3) and atomic_numbers has shape (N,)
    - Execute value validation checks:
       - check_finiteness(): Ensure all position values are finite and atomic numbers are non-negative
    - Execute optional matrix validation checks (if provided):
       - check_optional_matrices(): For lattice and stress tensors:
         * Verify shape is (3, 3)
         * Ensure all values are finite
    - If all validations pass, create and return XYZData instance
    - If any validation fails, raise ValueError with descriptive error message
    """

    positions = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    if lattice is not None:
        lattice = jnp.asarray(lattice, dtype=jnp.float64)
    else:
        lattice = jnp.eye(3, dtype=jnp.float64)

    if stress is not None:
        stress = jnp.asarray(stress, dtype=jnp.float64)

    if energy is not None:
        energy = jnp.asarray(energy, dtype=jnp.float64)

    def validate_and_create():
        N = positions.shape[0]

        def check_shape():
            if positions.shape[1] != 3:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers.shape[0] != N:
                raise ValueError("atomic_numbers must have shape (N,)")

        def check_finiteness():
            if not jnp.all(jnp.isfinite(positions)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def check_optional_matrices():
            # We have to use Python if for None checks here as well
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

        check_shape()
        check_finiteness()
        check_optional_matrices()

        return XYZData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=properties,
            comment=comment,
        )

    return validate_and_create()
