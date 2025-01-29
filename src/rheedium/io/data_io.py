from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Num, jaxtyped
from matplotlib.colors import LinearSegmentedColormap
from pymatgen.core import Element
from pymatgen.io.cif import CifParser

jax.config.update("jax_enable_x64", True)


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
        α is the angle between b and c
        β is the angle between a and c
        γ is the angle between a and b

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


@jaxtyped(typechecker=beartype)
def parse_cif_to_jax(
    cif_path: Union[str, Path], primitive: Optional[bool] = False
) -> CrystalStructure:
    """
    Description
    -----------
    Parse a CIF file and return atomic positions as JAX array and unit cell parameters.

    Parameters
    ----------
    - `cif_path` (str | Path):
        Path to the CIF file
    - `primitive` (bool, optional):
        Whether to return the primitive unit cell.
        If False, the full cell is returned.
        Default is False.

    Returns
    -------
    CrystalStructure with:
        - `frac_positions` (Float[Array, "* 4"]):
            containing [x, y, z, atomic_number] in fractional coordinates
        - `cart_positions` (Num[Array, "* 4"]):
            containing [x, y, z, atomic_number] in Cartesian coordinates (Å)
        - `cell_lengths` (Num[Array, "3"]):
            containing (a, b, c) in Å
        - `cell_angles` (Num[Array, "3"])
            containing (alpha, beta, gamma) in degree
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")
    if path.suffix.lower() != ".cif":
        raise ValueError(f"File must have .cif extension: {path}")
    parser = CifParser(cif_path)
    structure = parser.parse_structures(primitive=primitive)[0]
    cell_lengths: Num[Array, "3"] = jnp.array(structure.lattice.abc)
    cell_angles: Num[Array, "3"] = jnp.array(structure.lattice.angles)
    if jnp.any(cell_lengths <= 0):
        raise ValueError("Cell lengths must be positive")
    if jnp.any((cell_angles <= 0) | (cell_angles >= 180)):
        raise ValueError("Cell angles must be between 0 and 180 degrees")
    frac_coords: Num[Array, "* 3"] = jnp.array(
        [[site.frac_coords[i] for i in range(3)] for site in structure.sites]
    )
    cart_coords: Num[Array, "* 3"] = jnp.array(
        [[site.coords[i] for i in range(3)] for site in structure.sites]
    )
    atomic_numbers: Num[Array, "*"] = jnp.array(
        [Element(site.specie.symbol).Z for site in structure.sites]
    )
    frac_positions: Float[Array, "* 4"] = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_positions: Num[Array, "* 4"] = jnp.column_stack([cart_coords, atomic_numbers])
    return CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


def create_phosphor_colormap(
    name: Optional[str] = "phosphor",
) -> LinearSegmentedColormap:
    """
    Description
    -----------
    Create a custom colormap that simulates a phosphor screen appearance.
    The colormap transitions from black through a bright phosphorescent green,
    with a slight white bloom at maximum intensity.

    Parameters
    ----------
    - `name` (str, optional):
        Name for the colormap.
        Default is 'phosphor'

    Returns
    -------
    - `matplotlib.colors.LinearSegmentedColormap`
        Custom phosphor screen colormap
    """
    colors: list[tuple[float, tuple[float, float, float]]] = [
        (0.0, (0.0, 0.0, 0.0)),
        (0.4, (0.0, 0.05, 0.0)),
        (0.7, (0.15, 0.85, 0.15)),
        (0.9, (0.45, 0.95, 0.45)),
        (1.0, (0.8, 1.0, 0.8)),
    ]
    positions: list[float] = [x[0] for x in colors]
    rgb_values: list[tuple[float, float, float]] = [x[1] for x in colors]
    red: list[tuple[float, float, float]] = [
        (pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values)
    ]
    green: list[tuple[float, float, float]] = [
        (pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values)
    ]
    blue: list[tuple[float, float, float]] = [
        (pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values)
    ]
    cmap = LinearSegmentedColormap(name, {"red": red, "green": green, "blue": blue})
    return cmap
