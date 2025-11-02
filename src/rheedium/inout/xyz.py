"""Read XYZ files and convert to JAX-compatible data structures.

Extended Summary
----------------
This module provides utilities for parsing XYZ format files commonly used
in computational chemistry and materials science. It also includes functions
for loading atomic data such as Kirkland scattering factors.

Routine Listings
----------------
atomic_symbol : function
    Returns atomic number for given atomic symbol string
kirkland_potentials : function
    Loads Kirkland scattering factors from CSV file
parse_xyz : function
    Parses an XYZ file and returns atoms with element symbols and 3D
    coordinates.
_load_atomic_numbers : function, internal
    Loads atomic number mapping from JSON file in manifest folder
_load_kirkland_potentials : function, internal
    Loads Kirkland scattering factors from CSV file
_parse_xyz_metadata : function, internal
    Internal function to extract metadata from the XYZ comment line

Notes
-----
Internal functions prefixed with underscore are not part of the public API.
All returned data structures are JAX-compatible arrays.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Optional, Union
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import XYZData, create_xyz_data, scalar_int

_KIRKLAND_PATH: Path = (
    Path(__file__).resolve().parent / "luggage" / "Kirkland_Potentials.csv"
)
_ATOMS_PATH: Path = (
    Path(__file__).resolve().parent / "luggage" / "atom_numbers.json"
)

jax.config.update("jax_enable_x64", True)


@beartype
def _load_atomic_numbers(
    json_path: Optional[Path] = _ATOMS_PATH,
) -> Dict[str, int]:
    """Load atomic number mapping from JSON file in manifest folder.

    Uses pathlib for OS-independent path handling.

    Parameters
    ----------
    json_path : Optional[Path]
        Custom path to JSON file, defaults to module path

    Returns
    -------
    atomic_data : Dict[str, int]
        Dictionary mapping atomic symbols to atomic numbers
    """
    file_path: Path = json_path if json_path is not None else _ATOMS_PATH
    with open(file_path, encoding="utf-8") as file:
        atomic_data: Dict[str, int] = json.load(file)
    return atomic_data


_ATOMIC_NUMBERS: Dict[str, int] = _load_atomic_numbers()


@jaxtyped(typechecker=beartype)
def atomic_symbol(symbol_string: str) -> scalar_int:
    """Return atomic number for given atomic symbol string.

    Uses preloaded atomic number mapping for fast lookup.

    Parameters
    ----------
    symbol_string : str
        Chemical symbol for the element (e.g., "H", "He", "Li")

    Returns
    -------
    atomic_number : scalar_int
        Atomic number corresponding to the symbol

    Notes
    -----
    The algorithm proceeds as follows:

    1. Validate input is string
    2. Strip whitespace and ensure proper case
    3. Look up atomic number in preloaded mapping
    4. Return atomic number as scalar integer
    """
    cleaned_symbol: str = symbol_string.strip()
    normalized_symbol: str = cleaned_symbol.capitalize()
    if normalized_symbol not in _ATOMIC_NUMBERS:
        available_symbols: str = ", ".join(sorted(_ATOMIC_NUMBERS.keys()))
        raise KeyError(
            f"Atomic symbol '{symbol_string}' not found. "
            f"Available symbols: {available_symbols}"
        )

    atomic_number: scalar_int = _ATOMIC_NUMBERS[normalized_symbol]
    return atomic_number


@jaxtyped(typechecker=beartype)
def _load_kirkland_csv(
    file_path: Optional[Path] = _KIRKLAND_PATH,
) -> Float[Array, "103 12"]:
    """Load Kirkland potential parameters from CSV file.

    Uses numpy to load CSV then converts to JAX array for performance.

    Parameters
    ----------
    file_path : Optional[Path]
        Custom path to CSV file, defaults to module path

    Returns
    -------
    kirkland_data : Float[Array, "103 12"]
        Kirkland potential parameters as JAX array
    """
    kirkland_numpy: np.ndarray = np.loadtxt(
        file_path, delimiter=",", dtype=np.float64
    )
    if kirkland_numpy.shape != (103, 12):
        raise ValueError(
            f"Expected CSV shape (103, 12), got {kirkland_numpy.shape}"
        )
    kirkland_data: Float[Array, "103 12"] = jnp.asarray(
        kirkland_numpy, dtype=jnp.float64
    )
    return kirkland_data


_KIRKLAND_POTENTIALS: Float[Array, "103 12"] = _load_kirkland_csv()


@jaxtyped(typechecker=beartype)
def kirkland_potentials() -> Float[Array, "103 12"]:
    """Return preloaded Kirkland potential parameters as JAX array.

    Data is loaded once at module import for optimal performance.

    Returns
    -------
    kirkland_potentials : Float[Array, "103 12"]
        Kirkland potential parameters for elements 1-103

    Notes
    -----
    Return preloaded JAX array from module-level cache. No file I/O operations
    for fast access.
    """
    return _KIRKLAND_POTENTIALS


@beartype
def _parse_xyz_metadata(line: str) -> Dict[str, Any]:
    """Extract metadata from the XYZ comment line.

    Parameters
    ----------
    line : str
        Second line of the XYZ file (comment/metadata).

    Returns
    -------
    metadata : dict
        Parsed metadata with optional keys: lattice, stress, energy, properties
    """
    metadata: Dict[str, Any] = {}
    num_values_in_lattice: int = 9
    lattice_match: Optional[re.Match[str]] = re.search(
        r'Lattice="([^"]+)"', line
    )
    if lattice_match:
        values: list = list(map(float, lattice_match.group(1).split()))
        if len(values) != num_values_in_lattice:
            raise ValueError("Lattice must contain 9 values")
        metadata["lattice"] = jnp.array(values, dtype=jnp.float64).reshape(
            3, 3
        )

    stress_match: Optional[re.Match[str]] = re.search(
        r'stress="([^"]+)"', line
    )
    if stress_match:
        values: list = list(map(float, stress_match.group(1).split()))
        if len(values) != num_values_in_lattice:
            raise ValueError("Stress tensor must contain 9 values")
        metadata["stress"] = jnp.array(values, dtype=jnp.float64).reshape(3, 3)

    energy_match: Optional[re.Match[str]] = re.search(
        r"energy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line
    )
    if energy_match:
        metadata["energy"] = float(energy_match.group(1))

    props_match: Optional[re.Match[str]] = re.search(
        r"Properties=([^ ]+)", line
    )
    if props_match:
        raw_props: str = props_match.group(1)
        parts: list = raw_props.split(":")
        props: list = []
        for i in range(0, len(parts), 3):
            props.append(
                {
                    "name": parts[i],
                    "type": parts[i + 1],
                    "count": int(parts[i + 2]),
                }
            )
        metadata["properties"] = props

    return metadata


@jaxtyped(typechecker=beartype)
def parse_xyz(file_path: Union[str, Path]) -> XYZData:
    """Parse an XYZ file and returns a validated XYZData PyTree.

    Supports both atomic symbols (e.g., "H", "Fe") and atomic numbers
    (e.g., "1", "26") in the first column of atom data.

    Parameters
    ----------
    file_path : str or Path
        Path to the XYZ file.

    Returns
    -------
    XYZData
        Validated JAX-compatible structure with all contents from the XYZ file.
    """
    with open(file_path, encoding="utf-8") as f:
        lines: list = f.readlines()
    min_lines: int = 2
    if len(lines) < min_lines:
        raise ValueError("Invalid XYZ file: fewer than 2 lines.")

    try:
        num_atoms: int = int(lines[0].strip())
    except ValueError as err:
        raise ValueError(
            "First line must be the number of atoms (int)."
        ) from err

    comment: str = lines[1].strip()
    metadata: Dict[str, Any] = _parse_xyz_metadata(comment)

    if len(lines) < 2 + num_atoms:
        raise ValueError(
            f"Expected {num_atoms} atoms, found only {len(lines) - 2}."
        )

    positions: list = []
    atomic_numbers: list = []
    standard_xyz_cols: int = 4
    extended_xyz_cols: int = 5

    for i in range(2, 2 + num_atoms):
        parts: list = lines[i].split()
        if len(parts) not in {4, 5, 6, 7}:
            raise ValueError(
                f"Line {i + 1} has unexpected format: {lines[i].strip()}"
            )

        if len(parts) == standard_xyz_cols:
            symbol: str
            x: str
            y: str
            z: str
            symbol, x, y, z = parts
        elif len(parts) == extended_xyz_cols:
            _: str
            symbol, x, y, z = parts
        else:
            symbol, x, y, z = parts[:4]

        positions.append([float(x), float(y), float(z)])

        # Handle both atomic symbols and atomic numbers
        try:
            # Try to parse as an integer (atomic number)
            atomic_num: int = int(symbol)
            atomic_numbers.append(atomic_num)
        except ValueError:
            # Not a number, treat as atomic symbol
            atomic_numbers.append(atomic_symbol(symbol))

    positions_arr: Float[Array, "N 3"] = jnp.array(
        positions, dtype=jnp.float64
    )
    atomic_z_arr: Int[Array, "N"] = jnp.array(atomic_numbers, dtype=jnp.int32)

    return create_xyz_data(
        positions=positions_arr,
        atomic_numbers=atomic_z_arr,
        lattice=metadata.get("lattice"),
        stress=metadata.get("stress"),
        energy=metadata.get("energy"),
        properties=metadata.get("properties"),
        comment=comment,
    )
