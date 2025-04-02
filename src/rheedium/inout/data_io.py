import json
import re
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from jaxtyping import Array, Float, Num
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


@beartype
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

    cell_params_pattern = (
        r"_cell_length_a\s+([\d\.]+).*?"
        r"_cell_length_b\s+([\d\.]+).*?"
        r"_cell_length_c\s+([\d\.]+).*?"
        r"_cell_angle_alpha\s+([\d\.]+).*?"
        r"_cell_angle_beta\s+([\d\.]+).*?"
        r"_cell_angle_gamma\s+([\d\.]+)"
    )
    cell_match = re.search(cell_params_pattern, cif_text, re.DOTALL)
    if not cell_match:
        raise ValueError("Failed to parse cell parameters from CIF.")

    a, b, c, alpha, beta, gamma = map(float, cell_match.groups())
    cell_lengths: Num[Array, "3"] = jnp.array([a, b, c], dtype=jnp.float64)
    cell_angles: Num[Array, "3"] = jnp.array([alpha, beta, gamma], dtype=jnp.float64)

    if jnp.any(cell_lengths <= 0):
        raise ValueError("Cell lengths must be positive")
    if jnp.any((cell_angles <= 0) | (cell_angles >= 180)):
        raise ValueError("Cell angles must be between 0 and 180 degrees")

    positions_pattern = r"loop_([\s\S]+?)(_atom_site_[\s\S]+?)\n\n"
    positions_matches = re.findall(positions_pattern, cif_text, re.DOTALL)

    atom_site_block = None
    for _, block in positions_matches:
        if all(
            x in block
            for x in [
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            ]
        ):
            atom_site_block = block.strip()
            break

    if atom_site_block is None:
        raise ValueError("Failed to find atom positions block in CIF.")

    columns = atom_site_block.splitlines()
    column_indices = {name.strip(): idx for idx, name in enumerate(columns)}

    required_columns = [
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    if not all(col in column_indices for col in required_columns):
        raise ValueError("CIF file missing required atomic position columns.")

    positions_data_pattern = atom_site_block + r"([\s\S]+?)(?:\n\n|$)"
    positions_data_match = re.search(positions_data_pattern, cif_text, re.DOTALL)
    if not positions_data_match:
        raise ValueError("Failed to parse atomic positions data from CIF.")

    positions_block = positions_data_match.group(1).strip().split("\n")

    frac_positions_list: List[List[float]] = []
    for line in positions_block:
        if line.strip() == "":
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

    frac_positions: Float[Array, "* 4"] = jnp.array(
        frac_positions_list, dtype=jnp.float64
    )

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
