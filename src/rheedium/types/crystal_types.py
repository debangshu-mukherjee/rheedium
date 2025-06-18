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

Functions
---------
- `create_crystal_structure`:
    Factory function to create CrystalStructure instances with data validation
- `create_potential_slices`:
    Factory function to create PotentialSlices instances with data validation
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Num

from rheedium.types import scalar_float


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

    if frac_positions.shape[1] != 4:
        raise ValueError("frac_positions must have shape (n_atoms, 4)")
    if cart_positions.shape[1] != 4:
        raise ValueError("cart_positions must have shape (n_atoms, 4)")
    if cell_lengths.shape != (3,):
        raise ValueError("cell_lengths must have shape (3,)")
    if cell_angles.shape != (3,):
        raise ValueError("cell_angles must have shape (3,)")

    if frac_positions.shape[0] != cart_positions.shape[0]:
        raise ValueError(
            "Number of atoms must match between frac_positions and cart_positions"
        )
    if not jnp.all(frac_positions[:, 3] == cart_positions[:, 3]):
        raise ValueError(
            "Atomic numbers must match between frac_positions and cart_positions"
        )
    if jnp.any(cell_lengths <= 0):
        raise ValueError("Cell lengths must be positive")
    if jnp.any(cell_angles <= 0) or jnp.any(cell_angles >= 180):
        raise ValueError("Cell angles must be between 0 and 180 degrees")

    return CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


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

    if slices.ndim != 3:
        raise ValueError(f"slices must be 3D array, got shape {slices.shape}")

    if slices.shape[0] == 0:
        raise ValueError("Must have at least one slice")

    if slices.shape[1] == 0 or slices.shape[2] == 0:
        raise ValueError(
            f"Each slice must have non-zero dimensions, got {slices.shape[1:3]}"
        )

    if slice_thickness <= 0:
        raise ValueError(f"slice_thickness must be positive, got {slice_thickness}")

    if x_calibration <= 0:
        raise ValueError(f"x_calibration must be positive, got {x_calibration}")

    if y_calibration <= 0:
        raise ValueError(f"y_calibration must be positive, got {y_calibration}")

    if jnp.any(~jnp.isfinite(slices)):
        raise ValueError("All slice data must be finite")

    return PotentialSlices(
        slices=slices,
        slice_thickness=slice_thickness,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
