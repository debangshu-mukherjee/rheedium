import json
import re
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Float, Num, jaxtyped
from matplotlib.colors import LinearSegmentedColormap

import rheedium as rh
from rheedium.types import *

DEFAULT_ATOMIC_NUMBERS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "atomic_numbers.json"
)


@beartype
def load_atomic_numbers(path: str = str(DEFAULT_ATOMIC_NUMBERS_PATH)) -> dict[str, int]:
    """
    Description
    -----------
    Load the atomic numbers mapping from a JSON file.

    Parameters
    ----------
    - `path` (str, optional):
        Path to the atomic numbers JSON file.
        Defaults to '<project_root>/data/atomic_numbers.json'.

    Returns
    -------
    - `atomic_numbers` (dict[str, int]):
        Dictionary mapping element symbols to atomic numbers.
    """
    with open(path, "r") as f:
        atomic_numbers = json.load(f)
    return atomic_numbers


@jaxtyped(typechecker=beartype)
def parse_cif(cif_path: Union[str, Path]) -> CrystalStructure:
    """
    Description
    -----------
    Parse a CIF file into a JAX-compatible CrystalStructure.

    Parameters
    ----------
    - `cif_path` (Union[str, Path]):
        Path to the CIF file.

    Returns
    -------
    - `CrystalStructure`:
        Parsed crystal structure object with fractional and Cartesian coordinates.
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    if cif_path.suffix.lower() != ".cif":
        raise ValueError(f"File must have .cif extension: {cif_path}")

    cif_text = cif_path.read_text()
    atomic_numbers = load_atomic_numbers()

    # Extract cell parameters
    cell_params_pattern = (
        r"_cell_length_a\s+([\d\.]+)\s+"
        r"_cell_length_b\s+([\d\.]+)\s+"
        r"_cell_length_c\s+([\d\.]+)\s+"
        r"_cell_angle_alpha\s+([\d\.]+)\s+"
        r"_cell_angle_beta\s+([\d\.]+)\s+"
        r"_cell_angle_gamma\s+([\d\.]+)"
    )
    cell_match = re.search(cell_params_pattern, cif_text, re.DOTALL)
    if not cell_match:
        raise ValueError("Failed to parse cell parameters from CIF.")

    a, b, c, alpha, beta, gamma = map(float, cell_match.groups())
    cell_lengths: Num[Array, "3"] = jnp.array([a, b, c], dtype=jnp.float64)
    cell_angles: Num[Array, "3"] = jnp.array([alpha, beta, gamma], dtype=jnp.float64)

    # Find atom_site loop and columns
    loop_pattern = r"(loop_[\s\S]+?)(_atom_site_[\s\S]+?)\n\s*([^_]+?(?:\n|$))"
    loops = re.findall(loop_pattern, cif_text, re.DOTALL)

    atom_loop, columns_block, data_block = None, None, None
    for loop, columns, data in loops:
        if "_atom_site_fract_x" in columns and "_atom_site_fract_y" in columns and "_atom_site_fract_z" in columns:
            atom_loop, columns_block, data_block = loop, columns, data
            break

    if atom_loop is None:
        raise ValueError("Failed to find atom positions block in CIF.")

    # Parse column headers
    column_headers = [line.strip() for line in columns_block.strip().splitlines()]
    column_indices = {col: idx for idx, col in enumerate(column_headers)}

    required_cols = ["_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    if not all(col in column_indices for col in required_cols):
        raise ValueError("Required atomic position columns missing in CIF.")

    # Extract positions data
    positions_lines = data_block.strip().splitlines()
    frac_positions_list: List[List[float]] = []

    for line in positions_lines:
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        element_symbol = tokens[column_indices["_atom_site_type_symbol"]]
        frac_x = float(tokens[column_indices["_atom_site_fract_x"]])
        frac_y = float(tokens[column_indices["_atom_site_fract_y"]])
        frac_z = float(tokens[column_indices["_atom_site_fract_z"]])

        atomic_number = atomic_numbers.get(element_symbol)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {element_symbol}")

        frac_positions_list.append([frac_x, frac_y, frac_z, atomic_number])

    frac_positions: Float[Array, "* 4"] = jnp.array(frac_positions_list, dtype=jnp.float64)

    cell_vectors: Float[Array, "3 3"] = rh.ucell.build_cell_vectors(a, b, c, alpha, beta, gamma)
    cart_coords: Float[Array, "* 3"] = frac_positions[:, :3] @ cell_vectors
    cart_positions: Float[Array, "* 4"] = jnp.column_stack((cart_coords, frac_positions[:, 3]))

    return CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )



@beartype
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
    colors: List[
        Tuple[scalar_float, Tuple[scalar_float, scalar_float, scalar_float]]
    ] = [
        (0.0, (0.0, 0.0, 0.0)),
        (0.4, (0.0, 0.05, 0.0)),
        (0.7, (0.15, 0.85, 0.15)),
        (0.9, (0.45, 0.95, 0.45)),
        (1.0, (0.8, 1.0, 0.8)),
    ]
    positions: List[scalar_float] = [x[0] for x in colors]
    rgb_values: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        x[1] for x in colors
    ]
    red: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values)
    ]
    green: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values)
    ]
    blue: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values)
    ]
    cmap = LinearSegmentedColormap(name, {"red": red, "green": green, "blue": blue})
    return cmap
