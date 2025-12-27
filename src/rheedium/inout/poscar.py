"""VASP POSCAR/CONTCAR file parsing utilities.

Extended Summary
----------------
This module provides utilities for parsing VASP POSCAR and CONTCAR files,
which are commonly used to define crystal structures in VASP calculations.
The parser supports VASP 5.x format with species names and both Direct
(fractional) and Cartesian coordinate modes.

Routine Listings
----------------
parse_poscar : function
    Parse VASP POSCAR/CONTCAR file into CrystalStructure.
_parse_poscar_header : function, internal
    Extract scaling factor, lattice vectors, species, and counts from header.
_parse_poscar_positions : function, internal
    Parse atomic positions in Direct or Cartesian format.

Notes
-----
Internal functions prefixed with underscore are not part of the public API.
All returned data structures are JAX-compatible arrays.
"""

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple, Union
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import CrystalStructure, create_crystal_structure

from .crystal import lattice_to_cell_params
from .xyz import atomic_symbol


@beartype
def _parse_poscar_header(
    lines: List[str],
) -> Tuple[float, Float[Array, "3 3"], List[str], List[int]]:
    """Extract scaling factor, lattice vectors, species, and counts from header.

    Parses the first 7 lines of a VASP 5.x POSCAR file to extract the
    universal scaling factor, lattice vectors, element species names,
    and atom counts per species.

    Parameters
    ----------
    lines : List[str]
        Lines from the POSCAR file. Must contain at least 7 lines for
        VASP 5.x format.

    Returns
    -------
    scaling : float
        Universal scaling factor applied to all lattice vectors.
    lattice : Float[Array, "3 3"]
        Lattice vectors as rows, scaled by the scaling factor.
        [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]] in Angstroms.
    species : List[str]
        Element symbols in order (e.g., ["Si", "O"]).
    counts : List[int]
        Number of atoms per species (e.g., [2, 4]).

    Raises
    ------
    ValueError
        If scaling factor cannot be parsed, lattice vectors are invalid,
        or species/counts lines are malformed.

    Examples
    --------
    >>> lines = [
    ...     "Comment line",
    ...     "1.0",
    ...     "5.43 0.0 0.0",
    ...     "0.0 5.43 0.0",
    ...     "0.0 0.0 5.43",
    ...     "Si",
    ...     "8",
    ... ]
    >>> scaling, lattice, species, counts = _parse_poscar_header(lines)
    """
    try:
        scaling: float = float(lines[1].strip())
    except (ValueError, IndexError) as err:
        raise ValueError(
            f"Invalid POSCAR: expected scaling factor on line 2, "
            f"got '{lines[1].strip() if len(lines) > 1 else 'EOF'}'"
        ) from err

    lattice_rows: List[List[float]] = []
    for i in range(2, 5):
        try:
            row_values: List[float] = [float(x) for x in lines[i].split()]
            if len(row_values) != 3:
                raise ValueError(
                    f"Invalid POSCAR: lattice vector on line {i + 1} must have "
                    f"3 components, got {len(row_values)}"
                )
            lattice_rows.append(row_values)
        except (ValueError, IndexError) as err:
            raise ValueError(
                f"Invalid POSCAR: cannot parse lattice vector on line {i + 1}"
            ) from err

    lattice: Float[Array, "3 3"] = jnp.array(lattice_rows, dtype=jnp.float64)
    lattice = lattice * scaling

    try:
        species: List[str] = lines[5].split()
        if not species:
            raise ValueError("Invalid POSCAR: empty species line (line 6)")
    except IndexError as err:
        raise ValueError("Invalid POSCAR: missing species line (line 6)") from err

    try:
        counts: List[int] = [int(x) for x in lines[6].split()]
        if len(counts) != len(species):
            raise ValueError(
                f"Invalid POSCAR: species count ({len(species)}) does not match "
                f"atom counts ({len(counts)}) on line 7"
            )
    except (ValueError, IndexError) as err:
        raise ValueError(
            f"Invalid POSCAR: cannot parse atom counts on line 7"
        ) from err

    return scaling, lattice, species, counts


@beartype
def _parse_poscar_positions(
    lines: List[str],
    start_idx: int,
    n_atoms: int,
    is_cartesian: bool,
    lattice: Float[Array, "3 3"],
) -> Float[Array, "n_atoms 3"]:
    """Parse atomic positions in Direct or Cartesian format.

    Reads atomic position lines from a POSCAR file, handling both
    Direct (fractional) and Cartesian coordinate modes. Selective
    dynamics flags (T/F) are parsed but not stored.

    Parameters
    ----------
    lines : List[str]
        Lines from the POSCAR file.
    start_idx : int
        Index of the first position line (0-based).
    n_atoms : int
        Total number of atoms to parse.
    is_cartesian : bool
        If True, coordinates are Cartesian (Angstroms).
        If False, coordinates are Direct (fractional).
    lattice : Float[Array, "3 3"]
        Lattice vectors as rows, used for Cartesian to fractional conversion.

    Returns
    -------
    frac_positions : Float[Array, "n_atoms 3"]
        Fractional coordinates for all atoms.

    Raises
    ------
    ValueError
        If position lines are missing or malformed.

    Examples
    --------
    >>> frac_pos = _parse_poscar_positions(
    ...     lines, start_idx=8, n_atoms=2, is_cartesian=False, lattice=lattice
    ... )
    """
    positions: List[List[float]] = []

    for i in range(n_atoms):
        line_idx: int = start_idx + i
        if line_idx >= len(lines):
            raise ValueError(
                f"Invalid POSCAR: expected {n_atoms} atoms, "
                f"but file ends at line {len(lines)}"
            )

        parts: List[str] = lines[line_idx].split()
        if len(parts) < 3:
            raise ValueError(
                f"Invalid POSCAR: position line {line_idx + 1} must have "
                f"at least 3 coordinates, got {len(parts)}"
            )

        try:
            coords: List[float] = [float(parts[j]) for j in range(3)]
        except ValueError as err:
            raise ValueError(
                f"Invalid POSCAR: cannot parse coordinates on line {line_idx + 1}"
            ) from err

        positions.append(coords)

    positions_arr: Float[Array, "n_atoms 3"] = jnp.array(
        positions, dtype=jnp.float64
    )

    if is_cartesian:
        lattice_inv: Float[Array, "3 3"] = jnp.linalg.inv(lattice)
        frac_positions: Float[Array, "n_atoms 3"] = positions_arr @ lattice_inv
    else:
        frac_positions = positions_arr

    return frac_positions


@jaxtyped(typechecker=beartype)
def parse_poscar(
    file_path: Union[str, Path],
) -> CrystalStructure:
    """Parse VASP POSCAR or CONTCAR file into CrystalStructure.

    Reads a VASP POSCAR or CONTCAR file and converts it to a JAX-compatible
    CrystalStructure suitable for RHEED simulation. Supports VASP 5.x format
    with species names and both Direct (fractional) and Cartesian coordinate
    modes.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to POSCAR or CONTCAR file. Can be a string or pathlib.Path object.

    Returns
    -------
    crystal : CrystalStructure
        Parsed crystal structure containing:

        - ``frac_positions`` : Fractional coordinates with atomic numbers
        - ``cart_positions`` : Cartesian coordinates with atomic numbers
        - ``cell_lengths`` : [a, b, c] in Angstroms
        - ``cell_angles`` : [alpha, beta, gamma] in degrees

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid, has fewer than 8 lines, contains
        invalid scaling factor, malformed lattice vectors, missing species
        names, or position parsing fails.

    Notes
    -----
    VASP POSCAR format (VASP 5.x):

    .. code-block:: text

        Comment line
        1.0                    # Scaling factor
          ax ay az             # Lattice vector a
          bx by bz             # Lattice vector b
          cx cy cz             # Lattice vector c
          Si O                 # Species names (VASP 5.x)
          2  4                 # Atom counts per species
        Selective dynamics     # Optional
        Direct                 # or Cartesian
          0.0 0.0 0.0 T T T   # Positions with optional selective dynamics

    The scaling factor is applied to all lattice vectors. Selective dynamics
    flags (T/F) are parsed but not stored in CrystalStructure.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_poscar("POSCAR")
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)

    >>> crystal = rh.inout.parse_poscar("CONTCAR")
    >>> crystal.frac_positions.shape
    (8, 4)
    """
    path: Path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"POSCAR file not found: {path}")

    with open(path, encoding="utf-8") as f:
        lines: List[str] = f.readlines()

    lines = [line.rstrip("\n") for line in lines]

    min_lines: int = 8
    if len(lines) < min_lines:
        raise ValueError(
            f"Invalid POSCAR: file has only {len(lines)} lines, "
            f"expected at least {min_lines}"
        )

    scaling: float
    lattice: Float[Array, "3 3"]
    species: List[str]
    counts: List[int]
    scaling, lattice, species, counts = _parse_poscar_header(lines)

    n_atoms: int = sum(counts)

    line_idx: int = 7

    # Skip optional "Selective dynamics" line
    if lines[line_idx].strip().lower().startswith("s"):
        line_idx += 1

    coord_line: str = lines[line_idx].strip().lower()
    if coord_line.startswith("d"):
        is_cartesian: bool = False
    elif coord_line.startswith(("c", "k")):
        is_cartesian = True
    else:
        raise ValueError(
            f"Invalid POSCAR: expected 'Direct' or 'Cartesian' on line "
            f"{line_idx + 1}, got '{lines[line_idx].strip()}'"
        )

    line_idx += 1

    frac_positions_3: Float[Array, "n_atoms 3"] = _parse_poscar_positions(
        lines, line_idx, n_atoms, is_cartesian, lattice
    )

    atomic_numbers_list: List[int] = []
    for sp, count in zip(species, counts, strict=True):
        z: int = atomic_symbol(sp)
        atomic_numbers_list.extend([z] * count)

    atomic_numbers: Int[Array, "n_atoms"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )

    cell_lengths: Float[Array, "3"]
    cell_angles: Float[Array, "3"]
    cell_lengths, cell_angles = lattice_to_cell_params(lattice)

    cart_positions_3: Float[Array, "n_atoms 3"] = frac_positions_3 @ lattice

    frac_positions: Float[Array, "n_atoms 4"] = jnp.column_stack(
        [frac_positions_3, atomic_numbers.astype(jnp.float64)]
    )
    cart_positions: Float[Array, "n_atoms 4"] = jnp.column_stack(
        [cart_positions_3, atomic_numbers.astype(jnp.float64)]
    )

    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
