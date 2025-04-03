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
    `CrystalStructure`:
        Parsed crystal structure object with fractional and Cartesian coordinates.

        Attributes:

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
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    if cif_path.suffix.lower() != ".cif":
        raise ValueError(f"File must have .cif extension: {cif_path}")
    cif_text = cif_path.read_text()
    atomic_numbers = load_atomic_numbers()

    def extract_param(name: str) -> float:
        match = re.search(rf"{name}\s+([0-9.]+)", cif_text)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Failed to parse {name} from CIF.")

    a = extract_param("_cell_length_a")
    b = extract_param("_cell_length_b")
    c = extract_param("_cell_length_c")
    alpha = extract_param("_cell_angle_alpha")
    beta = extract_param("_cell_angle_beta")
    gamma = extract_param("_cell_angle_gamma")
    cell_lengths: Num[Array, "3"] = jnp.array([a, b, c], dtype=jnp.float64)
    cell_angles: Num[Array, "3"] = jnp.array([alpha, beta, gamma], dtype=jnp.float64)
    lines = cif_text.splitlines()
    atom_site_columns = []
    positions_list = []
    in_atom_site_loop = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.lower().startswith("loop_"):
            in_atom_site_loop = False
            atom_site_columns = []
            continue
        if stripped_line.startswith("_atom_site_"):
            atom_site_columns.append(stripped_line)
            in_atom_site_loop = True
            continue
        if in_atom_site_loop and stripped_line and not stripped_line.startswith("_"):
            tokens = stripped_line.split()
            if len(tokens) != len(atom_site_columns):
                continue
            required_cols = [
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            ]
            if not all(col in atom_site_columns for col in required_cols):
                continue
            col_indices = {col: atom_site_columns.index(col) for col in required_cols}
            element_symbol = tokens[col_indices["_atom_site_type_symbol"]]
            frac_x = float(tokens[col_indices["_atom_site_fract_x"]])
            frac_y = float(tokens[col_indices["_atom_site_fract_y"]])
            frac_z = float(tokens[col_indices["_atom_site_fract_z"]])
            atomic_number = atomic_numbers.get(element_symbol)
            if atomic_number is None:
                raise ValueError(f"Unknown element symbol: {element_symbol}")
            positions_list.append([frac_x, frac_y, frac_z, atomic_number])
    if not positions_list:
        raise ValueError("No atomic positions found in CIF.")
    frac_positions: Float[Array, "* 4"] = jnp.array(positions_list, dtype=jnp.float64)
    cell_vectors: Float[Array, "3 3"] = rh.ucell.build_cell_vectors(
        a, b, c, alpha, beta, gamma
    )
    cart_coords: Float[Array, "* 3"] = frac_positions[:, :3] @ cell_vectors
    cart_positions: Float[Array, "* 4"] = jnp.column_stack(
        (cart_coords, frac_positions[:, 3])
    )
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
