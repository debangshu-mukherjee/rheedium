"""
Module: types.crystal_types
---------------------------
Data structures and factory functions for crystal structure representation.

Classes
-------
- `CrystalStructure`:
    JAX-compatible crystal structure with fractional and Cartesian coordinates

Functions
---------
- `create_crystal_structure`:
    Factory function to create CrystalStructure instances with data validation
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Num

__all__ = ["CrystalStructure", "create_crystal_structure"]


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
