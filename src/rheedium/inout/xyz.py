"""Read XYZ files and convert to JAX-compatible data structures.

Extended Summary
----------------
This module provides utilities for parsing XYZ format files commonly used
in computational chemistry and materials science. It also includes functions
for loading atomic data such as Kirkland scattering factors.

Routine Listings
----------------
atomic_symbol : function
    Return atomic number for given atomic symbol string.
kirkland_potentials : function
    Return preloaded Kirkland potential parameters as JAX array.
parse_xyz : function
    Parse an XYZ file and return a validated XYZData PyTree.
_load_atomic_numbers : function, internal
    Load atomic number mapping from JSON file.
_load_kirkland_csv : function, internal
    Load Kirkland potential parameters from CSV file.
_parse_xyz_metadata : function, internal
    Extract metadata from the XYZ comment line.
_parse_atom_line : function, internal
    Parse a single atom line from XYZ file.

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
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
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
    """Load atomic number mapping from JSON file.

    Reads a JSON file containing a mapping from element symbols to their
    corresponding atomic numbers. Uses pathlib for OS-independent path
    handling.

    Parameters
    ----------
    json_path : Optional[Path], optional
        Path to JSON file containing atomic number mapping. If None,
        defaults to the bundled atom_numbers.json file in the luggage
        directory. Default: None

    Returns
    -------
    atomic_data : Dict[str, int]
        Dictionary mapping element symbols (e.g., "H", "He", "Li") to
        their atomic numbers (e.g., 1, 2, 3).

    Raises
    ------
    FileNotFoundError
        If the specified JSON file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.

    Examples
    --------
    >>> atomic_map = _load_atomic_numbers()
    >>> atomic_map["Fe"]
    26
    """
    file_path: Path = json_path if json_path is not None else _ATOMS_PATH
    with open(file_path, encoding="utf-8") as file:
        atomic_data: Dict[str, int] = json.load(file)
    return atomic_data


_ATOMIC_NUMBERS: Dict[str, int] = _load_atomic_numbers()


@jaxtyped(typechecker=beartype)
def atomic_symbol(symbol_string: str) -> scalar_int:
    """Return atomic number for given atomic symbol string.

    Converts a chemical element symbol to its corresponding atomic number
    using a preloaded mapping for fast lookup. The function is case-insensitive
    and handles whitespace.

    Parameters
    ----------
    symbol_string : str
        Chemical symbol for the element (e.g., "H", "He", "Li", "Fe").
        Case-insensitive; leading/trailing whitespace is stripped.

    Returns
    -------
    atomic_number : scalar_int
        Atomic number corresponding to the symbol (e.g., 1 for H, 26 for Fe).

    Raises
    ------
    KeyError
        If the symbol is not found in the atomic number mapping.

    Examples
    --------
    >>> atomic_symbol("H")
    1
    >>> atomic_symbol("fe")
    26
    >>> atomic_symbol("  Au  ")
    79
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

    Reads the Kirkland scattering factor parameters from a CSV file and
    converts them to a JAX array for use in electron scattering calculations.

    Parameters
    ----------
    file_path : Optional[Path], optional
        Path to CSV file containing Kirkland parameters. If None, defaults
        to the bundled Kirkland_Potentials.csv file in the luggage directory.
        Default: None

    Returns
    -------
    kirkland_data : Float[Array, "103 12"]
        Kirkland potential parameters as a JAX array with shape (103, 12).
        Rows correspond to elements 1-103 (H to Lr), columns contain the
        12 fitting parameters (a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3).

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    ValueError
        If the CSV file does not have the expected shape (103, 12).

    Examples
    --------
    >>> params = _load_kirkland_csv()
    >>> params.shape
    (103, 12)
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

    Provides access to the Kirkland electron scattering factor parameters
    for elements 1-103. Data is loaded once at module import for optimal
    performance with no file I/O on subsequent calls.

    Returns
    -------
    kirkland_potentials : Float[Array, "103 12"]
        Kirkland potential parameters with shape (103, 12). Rows correspond
        to elements 1-103 (H to Lr), columns contain the 12 fitting parameters
        (a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3) as defined in
        Kirkland's "Advanced Computing in Electron Microscopy".

    Examples
    --------
    >>> params = kirkland_potentials()
    >>> params.shape
    (103, 12)
    >>> params[25]  # Iron (Z=26, 0-indexed as 25)
    Array([...], dtype=float64)
    """
    return _KIRKLAND_POTENTIALS


@beartype
def _parse_xyz_metadata(line: str) -> Dict[str, Any]:
    """Extract metadata from the XYZ comment line.

    Parses the second line of an extended XYZ file to extract structured
    metadata including lattice vectors, stress tensor, energy, and
    property descriptors following the extended XYZ format specification.

    Parameters
    ----------
    line : str
        Second line of the XYZ file (comment/metadata line). May contain
        key=value pairs in extended XYZ format.

    Returns
    -------
    metadata : Dict[str, Any]
        Dictionary containing parsed metadata with optional keys:

        - ``lattice`` : Float[Array, "3 3"] - Unit cell vectors as rows
        - ``stress`` : Float[Array, "3 3"] - Stress tensor in Voigt notation
        - ``energy`` : float - Total energy value
        - ``properties`` : List[Dict] - Property descriptors

    Raises
    ------
    ValueError
        If Lattice or stress values are present but don't contain exactly
        9 values.

    Examples
    --------
    >>> line = 'Lattice="4.2 0 0 0 4.2 0 0 0 4.2" energy=-123.45'
    >>> meta = _parse_xyz_metadata(line)
    >>> meta["energy"]
    -123.45
    >>> meta["lattice"].shape
    (3, 3)
    """
    metadata: Dict[str, Any] = {}
    num_values_in_lattice: int = 9
    lattice_match: Optional[re.Match[str]] = re.search(
        r'Lattice="([^"]+)"', line
    )
    if lattice_match:
        values: List[float] = list(map(float, lattice_match.group(1).split()))
        if len(values) != num_values_in_lattice:
            raise ValueError("Lattice must contain 9 values")
        metadata["lattice"] = jnp.array(values, dtype=jnp.float64).reshape(
            3, 3
        )

    stress_match: Optional[re.Match[str]] = re.search(
        r'stress="([^"]+)"', line
    )
    if stress_match:
        stress_values: List[float] = list(
            map(float, stress_match.group(1).split())
        )
        if len(stress_values) != num_values_in_lattice:
            raise ValueError("Stress tensor must contain 9 values")
        metadata["stress"] = jnp.array(
            stress_values, dtype=jnp.float64
        ).reshape(3, 3)

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
        parts: List[str] = raw_props.split(":")
        props: List[Dict[str, Any]] = []
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


@beartype
def _parse_atom_line(
    parts: List[str],
) -> Tuple[str, float, float, float]:
    """Parse a single atom line from XYZ file.

    Extracts the element symbol and Cartesian coordinates from a tokenized
    atom line. Handles both standard 4-column XYZ format and extended XYZ
    format with additional columns.

    Parameters
    ----------
    parts : List[str]
        List of whitespace-separated tokens from an atom line. Must contain
        at least 4 elements: [symbol, x, y, z, ...].

    Returns
    -------
    symbol : str
        Element symbol or atomic number as string.
    x : float
        X coordinate in Angstroms.
    y : float
        Y coordinate in Angstroms.
    z : float
        Z coordinate in Angstroms.

    Examples
    --------
    >>> parts = ["Fe", "0.0", "1.5", "2.0"]
    >>> symbol, x, y, z = _parse_atom_line(parts)
    >>> symbol
    'Fe'
    >>> x, y, z
    (0.0, 1.5, 2.0)
    """
    standard_xyz_cols: int = 4
    extended_xyz_cols: int = 5

    if len(parts) == standard_xyz_cols:
        symbol, x, y, z = parts
    elif len(parts) == extended_xyz_cols:
        symbol, x, y, z = parts[:4]
    else:
        symbol, x, y, z = parts[:4]

    return symbol, float(x), float(y), float(z)


@jaxtyped(typechecker=beartype)
def parse_xyz(file_path: Union[str, Path]) -> XYZData:
    """Parse an XYZ file and return a validated XYZData PyTree.

    Reads an XYZ format file and converts it to a JAX-compatible XYZData
    structure. Supports both standard XYZ format and extended XYZ format
    with lattice vectors and other metadata.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the XYZ file. Can be a string or pathlib.Path object.

    Returns
    -------
    xyz_data : XYZData
        Validated JAX-compatible structure containing:

        - ``positions`` : Float[Array, "N 3"] - Cartesian coordinates
        - ``atomic_numbers`` : Int[Array, "N"] - Atomic numbers
        - ``lattice`` : Optional[Float[Array, "3 3"]] - Unit cell vectors
        - ``stress`` : Optional[Float[Array, "3 3"]] - Stress tensor
        - ``energy`` : Optional[float] - Total energy
        - ``properties`` : Optional[List[Dict]] - Property descriptors
        - ``comment`` : Optional[str] - Comment line content

    Raises
    ------
    ValueError
        If the file has fewer than 2 lines, the first line is not an integer,
        the file contains fewer atoms than declared, or atom lines have
        unexpected format.
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    The first column of atom data can contain either element symbols
    (e.g., "H", "Fe") or atomic numbers (e.g., "1", "26").

    Examples
    --------
    >>> xyz_data = parse_xyz("structure.xyz")
    >>> xyz_data.positions.shape
    (10, 3)
    >>> xyz_data.atomic_numbers
    Array([6, 6, 8, ...], dtype=int32)
    """
    with open(file_path, encoding="utf-8") as f:
        lines: List[str] = f.readlines()
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

    positions: List[List[float]] = []
    atomic_numbers: List[int] = []

    for i in range(2, 2 + num_atoms):
        parts: List[str] = lines[i].split()
        if len(parts) not in {4, 5, 6, 7}:
            raise ValueError(
                f"Line {i + 1} has unexpected format: {lines[i].strip()}"
            )

        symbol, x, y, z = _parse_atom_line(parts)
        positions.append([x, y, z])

        try:
            atomic_num: int = int(symbol)
            atomic_numbers.append(atomic_num)
        except ValueError:
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
