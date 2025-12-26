"""Functions for reading and writing crystal structure data.

Extended Summary
----------------
This module provides utilities for parsing Crystallographic Information Format
(CIF) files and converting them to JAX-compatible data structures. It includes
symmetry expansion capabilities for generating complete unit cells from
asymmetric units.

Routine Listings
----------------
parse_cif : function
    Parse CIF file into JAX-compatible CrystalStructure.
symmetry_expansion : function
    Apply symmetry operations to expand fractional positions.
_extract_cell_params : function, internal
    Extract unit cell parameters from CIF text.
_parse_atom_positions : function, internal
    Parse atomic positions from CIF atom site loop.
_extract_sym_op_from_line : function, internal
    Extract symmetry operation from a single line.
_parse_symmetry_ops : function, internal
    Parse symmetry operations from CIF file.
_parse_sym_op : function, internal
    Parse a symmetry operation string into a callable function.
_apply_symmetry_ops : function, internal
    Apply symmetry operations to fractional positions.
_deduplicate_positions : function, internal
    Remove duplicate positions within tolerance.

Notes
-----
All functions return JAX-compatible arrays suitable for automatic
differentiation and GPU acceleration.
"""

import fractions
import re
from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from jaxtyping import Array, Bool, Float, Int, Num

from rheedium.inout.xyz import atomic_symbol
from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors


@beartype
def _extract_cell_params(
    cif_text: str,
) -> Tuple[float, float, float, float, float, float]:
    r"""Extract unit cell parameters from CIF text.

    Parses the CIF text to extract the six crystallographic unit cell
    parameters: three cell lengths and three cell angles.

    Parameters
    ----------
    cif_text : str
        Full text content of a CIF file.

    Returns
    -------
    a : float
        Cell length a in Angstroms.
    b : float
        Cell length b in Angstroms.
    c : float
        Cell length c in Angstroms.
    alpha : float
        Cell angle alpha in degrees.
    beta : float
        Cell angle beta in degrees.
    gamma : float
        Cell angle gamma in degrees.

    Raises
    ------
    ValueError
        If any of the required cell parameters cannot be found in the CIF text.

    Examples
    --------
    >>> cif_text = "_cell_length_a 5.43\\n_cell_length_b 5.43\\n..."
    >>> a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)
    >>> a
    5.43
    """
    def _extract_param(name: str) -> float:
        match: Optional[re.Match[str]] = re.search(
            rf"{name}\s+([0-9.]+)", cif_text
        )
        if match:
            return float(match.group(1))
        raise ValueError(f"Failed to parse {name} from CIF.")

    a: float = _extract_param("_cell_length_a")
    b: float = _extract_param("_cell_length_b")
    c: float = _extract_param("_cell_length_c")
    alpha: float = _extract_param("_cell_angle_alpha")
    beta: float = _extract_param("_cell_angle_beta")
    gamma: float = _extract_param("_cell_angle_gamma")
    return a, b, c, alpha, beta, gamma


@beartype
def _parse_atom_positions(
    lines: List[str],
) -> List[List[float]]:
    """Parse atomic positions from CIF atom site loop.

    Extracts fractional coordinates and element types from the _atom_site_
    loop section of a CIF file. Handles various CIF formats with different
    column orderings.

    Parameters
    ----------
    lines : List[str]
        List of lines from the CIF file.

    Returns
    -------
    positions_list : List[List[float]]
        List of [frac_x, frac_y, frac_z, atomic_number] for each atom.
        Returns empty list if no valid atom positions are found.

    Examples
    --------
    >>> lines = ["loop_", "_atom_site_type_symbol", "_atom_site_fract_x",
    ...          "_atom_site_fract_y", "_atom_site_fract_z", "Si 0.0 0.0 0.0"]
    >>> positions = _parse_atom_positions(lines)
    >>> positions[0]
    [0.0, 0.0, 0.0, 14]
    """
    atom_site_columns: List[str] = []
    positions_list: List[List[float]] = []
    in_atom_site_loop: bool = False

    for line in lines:
        stripped_line: str = line.strip()
        if stripped_line.lower().startswith("loop_"):
            in_atom_site_loop = False
            atom_site_columns = []
            continue
        if stripped_line.startswith("_atom_site_"):
            atom_site_columns.append(stripped_line)
            in_atom_site_loop = True
            continue
        if not in_atom_site_loop:
            continue
        if not stripped_line or stripped_line.startswith("_"):
            continue

        tokens: List[str] = stripped_line.split()
        if len(tokens) != len(atom_site_columns):
            continue

        required_cols: List[str] = [
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        if not all(col in atom_site_columns for col in required_cols):
            continue

        col_indices: Dict[str, int] = {
            col: atom_site_columns.index(col) for col in required_cols
        }
        element_symbol: str = tokens[col_indices["_atom_site_type_symbol"]]
        frac_x: float = float(tokens[col_indices["_atom_site_fract_x"]])
        frac_y: float = float(tokens[col_indices["_atom_site_fract_y"]])
        frac_z: float = float(tokens[col_indices["_atom_site_fract_z"]])
        atomic_number: float = float(atomic_symbol(element_symbol))
        positions_list.append([frac_x, frac_y, frac_z, atomic_number])

    return positions_list


@beartype
def _extract_sym_op_from_line(
    stripped_line: str,
    sym_loop_columns: List[str],
) -> Optional[str]:
    """Extract symmetry operation from a single line.

    Parses a line from the symmetry operations section of a CIF file to
    extract the symmetry operation string (e.g., "x,y,z" or "-x,-y,z+1/2").

    Parameters
    ----------
    stripped_line : str
        A stripped line from the CIF symmetry section.
    sym_loop_columns : List[str]
        List of column headers from the symmetry loop.

    Returns
    -------
    sym_op : Optional[str]
        The extracted symmetry operation string, or None if no valid
        operation could be extracted from the line.

    Examples
    --------
    >>> columns = ["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"]
    >>> op = _extract_sym_op_from_line("1 'x,y,z'", columns)
    >>> op
    'x,y,z'
    """
    if "_symmetry_equiv_pos_as_xyz" in sym_loop_columns:
        xyz_col_idx = sym_loop_columns.index("_symmetry_equiv_pos_as_xyz")
        match = re.search(r"'([^']+)'", stripped_line)
        if not match:
            match = re.search(r'"([^"]+)"', stripped_line)
        if match:
            return match.group(1).strip()
        tokens = stripped_line.split()
        if len(tokens) > xyz_col_idx:
            op = tokens[xyz_col_idx].strip("'\"")
            if "," in op:
                return op
    elif not sym_loop_columns:
        if stripped_line.startswith("'") and stripped_line.endswith("'"):
            return stripped_line.strip("'").strip()
        if stripped_line.startswith('"') and stripped_line.endswith('"'):
            return stripped_line.strip('"').strip()
    return None


@beartype
def _parse_symmetry_ops(lines: List[str]) -> List[str]:
    """Parse symmetry operations from CIF file.

    Extracts all symmetry operations from the _symmetry_equiv_pos section
    of a CIF file. Returns ["x,y,z"] (identity) if no symmetry operations
    are found.

    Parameters
    ----------
    lines : List[str]
        List of lines from the CIF file.

    Returns
    -------
    sym_ops : List[str]
        List of symmetry operation strings (e.g., ["x,y,z", "-x,-y,z"]).
        Returns ["x,y,z"] if no operations are found.

    Examples
    --------
    >>> lines = ["loop_", "_symmetry_equiv_pos_as_xyz", "'x,y,z'", "'-x,-y,z'"]
    >>> ops = _parse_symmetry_ops(lines)
    >>> ops
    ['x,y,z', '-x,-y,z']
    """
    sym_ops: List[str] = []
    sym_loop_columns: List[str] = []
    in_sym_loop: bool = False

    for line in lines:
        stripped_line: str = line.strip()
        if stripped_line.lower().startswith("loop_"):
            in_sym_loop = False
            sym_loop_columns = []
            continue
        if stripped_line.startswith("_symmetry_equiv_pos"):
            sym_loop_columns.append(stripped_line)
            in_sym_loop = True
            continue
        if not in_sym_loop or not stripped_line:
            continue
        if stripped_line.startswith("_") and not stripped_line.startswith(
            "_symmetry"
        ):
            in_sym_loop = False
            continue

        op = _extract_sym_op_from_line(stripped_line, sym_loop_columns)
        if op is not None:
            sym_ops.append(op)

    if not sym_ops:
        sym_ops = ["x,y,z"]

    return sym_ops


@beartype
def parse_cif(cif_path: Union[str, Path]) -> CrystalStructure:
    r"""Parse a CIF file into a JAX-compatible CrystalStructure.

    Reads a Crystallographic Information Format (CIF) file and converts it
    to a CrystalStructure object with symmetry expansion applied.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the CIF file.

    Returns
    -------
    expanded_crystal : CrystalStructure
        Parsed crystal structure object with fractional and Cartesian
        coordinates. Contains arrays of atomic positions in both fractional
        (range [0,1]) and Cartesian (Angstroms) coordinates, along with unit
        cell parameters (lengths in Angstroms, angles in degrees).

    Raises
    ------
    FileNotFoundError
        If the CIF file does not exist.
    ValueError
        If the file does not have .cif extension, or if required cell
        parameters or atomic positions cannot be parsed.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Validate CIF file path and extension
    2. Read CIF file content
    3. Extract unit cell parameters (cell lengths and angles)
    4. Parse atomic positions from atom site loop section
    5. Convert element symbols to atomic numbers
    6. Convert fractional to Cartesian coordinates using cell vectors
    7. Parse symmetry operations from CIF file
    8. Create initial CrystalStructure
    9. Apply symmetry operations to expand positions
    10. Return expanded crystal structure

    Examples
    --------
    >>> from rheedium.inout import parse_cif
    >>> structure = parse_cif("path/to/silicon.cif")
    >>> structure.cart_positions.shape
    (8, 4)
    """
    cif_path_obj: Path = Path(cif_path)
    if not cif_path_obj.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path_obj}")
    if cif_path_obj.suffix.lower() != ".cif":
        raise ValueError(f"File must have .cif extension: {cif_path_obj}")
    cif_text: str = cif_path_obj.read_text()

    a, b, c, alpha, beta, gamma = _extract_cell_params(cif_text)
    cell_lengths: Num[Array, "3"] = jnp.array([a, b, c], dtype=jnp.float64)
    cell_angles: Num[Array, "3"] = jnp.array(
        [alpha, beta, gamma], dtype=jnp.float64
    )

    lines: List[str] = cif_text.splitlines()
    positions_list = _parse_atom_positions(lines)

    if not positions_list:
        raise ValueError("No atomic positions found in CIF.")

    frac_positions: Float[Array, "N 4"] = jnp.array(
        positions_list, dtype=jnp.float64
    )
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        a, b, c, alpha, beta, gamma
    )
    cart_coords: Float[Array, "N 3"] = frac_positions[:, :3] @ cell_vectors
    cart_positions: Float[Array, "N 4"] = jnp.column_stack(
        (cart_coords, frac_positions[:, 3])
    )

    sym_ops = _parse_symmetry_ops(lines)

    crystal: CrystalStructure = create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    expanded_crystal: CrystalStructure = symmetry_expansion(
        crystal, sym_ops, tolerance=0.5
    )
    return expanded_crystal


@beartype
def _parse_sym_op(op_str: str) -> Callable[[Array], Array]:
    """Parse a symmetry operation string into a callable function.

    Converts a crystallographic symmetry operation string (e.g., "-x,y+1/2,z")
    into a function that transforms fractional coordinates.

    Parameters
    ----------
    op_str : str
        Symmetry operation string in CIF format (e.g., "x,y,z", "-x,-y,z+1/2").

    Returns
    -------
    op_func : Callable[[Array], Array]
        A function that takes a (3,) array of fractional coordinates and
        returns the transformed (3,) array.

    Examples
    --------
    >>> op = _parse_sym_op("-x,y,z+1/2")
    >>> import jax.numpy as jnp
    >>> result = op(jnp.array([0.25, 0.5, 0.0]))
    >>> result
    Array([-0.25, 0.5, 0.5], dtype=float64)
    """
    def _op(pos: Array) -> Array:
        replacements: Dict[str, float] = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
        }
        components: List[str] = op_str.lower().replace(" ", "").split(",")

        def _eval_comp(comp: str) -> float:
            comp = comp.replace("-", "+-")
            terms: List[str] = comp.split("+")
            total: float = 0.0
            for term in terms:
                if not term:
                    continue
                coeff: float = 1.0
                for var in ("x", "y", "z"):
                    if var in term:
                        part: str = term.split(var)[0]
                        if part == "-":
                            coeff = -1.0
                        elif part:
                            coeff = float(fractions.Fraction(part))
                        else:
                            coeff = 1.0
                        total += coeff * replacements[var]
                        break
                else:
                    total += float(fractions.Fraction(term))
            return total

        return jnp.array([_eval_comp(c) for c in components])

    return _op


@beartype
def _apply_symmetry_ops(
    frac_positions: Float[Array, "N 4"],
    sym_ops: List[str],
) -> Float[Array, "M 4"]:
    """Apply symmetry operations to fractional positions.

    Generates all symmetry-equivalent positions by applying each symmetry
    operation to each input atomic position. The resulting positions are
    wrapped to the [0, 1) range using modulo.

    Parameters
    ----------
    frac_positions : Float[Array, "N 4"]
        Array of shape (N, 4) containing 
        [frac_x, frac_y, frac_z, atomic_number] for each atom.
    sym_ops : List[str]
        List of symmetry operation strings (e.g., ["x,y,z", "-x,-y,z"]).

    Returns
    -------
    expanded_positions : Float[Array, "M 4"]
        Array of shape (M, 4) where M = N * len(sym_ops), containing all
        symmetry-equivalent positions with atomic numbers preserved.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([[0.25, 0.25, 0.0, 14.0]])
    >>> ops = ["x,y,z", "-x,-y,z"]
    >>> expanded = _apply_symmetry_ops(positions, ops)
    >>> expanded.shape
    (2, 4)
    """
    expanded_positions: List[Array] = []
    ops: List[Callable[[Array], Array]] = [_parse_sym_op(op) for op in sym_ops]

    for pos in frac_positions:
        xyz: Array = pos[:3]
        atomic_number: float = pos[3]
        for op in ops:
            new_xyz: Array = jnp.mod(op(xyz), 1.0)
            expanded_positions.append(
                jnp.concatenate([new_xyz, atomic_number[None]])
            )

    return jnp.array(expanded_positions)


@beartype
def _deduplicate_positions(
    positions_with_z: Float[Array, "n_pos 4"],
    tol: scalar_float,
) -> Float[Array, "n_unique 4"]:
    """Remove duplicate positions within tolerance.

    Iterates through positions and keeps only those that are not within
    the specified tolerance of any previously kept position. Uses JAX's
    scan for efficient sequential processing.

    Parameters
    ----------
    positions_with_z : Float[Array, "n_pos 4"]
        Array of shape (n_pos, 4) containing [x, y, z, atomic_number] for each
        position in Cartesian coordinates.
    tol : scalar_float
        Distance tolerance in Angstroms. Positions within this distance of
        an existing unique position are considered duplicates.

    Returns
    -------
    unique_positions : Float[Array, "n_unique 4"]
        Array of shape (n_unique, 4) where n_unique <= n_pos, containing only
        unique positions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([
    ...     [0.0, 0.0, 0.0, 14.0],
    ...     [0.01, 0.01, 0.01, 14.0],
    ...     [2.0, 2.0, 2.0, 8.0],
    ... ])
    >>> unique = _deduplicate_positions(positions, tol=0.1)
    >>> unique.shape[0]
    2
    """
    n_positions: int = positions_with_z.shape[0]

    def _unique_cond(
        carry: Tuple[Float[Array, "n_pos 4"], Int[Array, ""]],
        pos_z: Float[Array, "4"],
    ) -> Tuple[Tuple[Float[Array, "n_pos 4"], Int[Array, ""]], None]:
        unique: Float[Array, "n_pos 4"]
        count: Int[Array, ""]
        unique, count = carry
        pos: Float[Array, "3"] = pos_z[:3]
        diff: Float[Array, "n_pos 3"] = unique[:, :3] - pos
        dist_sq: Float[Array, "n_pos"] = jnp.sum(diff**2, axis=1)
        indices: Int[Array, "n_pos"] = jnp.arange(n_positions)
        valid_mask: Bool[Array, "n_pos"] = indices < count
        is_dup: bool = jnp.any((dist_sq < tol**2) & valid_mask)
        unique = jax.lax.cond(
            is_dup,
            lambda u: u,
            lambda u: u.at[count].set(pos_z),
            unique,
        )
        count += jnp.logical_not(is_dup)
        return (unique, count), None

    unique_init: Float[Array, "n_pos 4"] = jnp.zeros_like(positions_with_z)
    unique_init = unique_init.at[0].set(positions_with_z[0])
    count_init: int = 1
    (unique_final, final_count), _ = jax.lax.scan(
        _unique_cond, (unique_init, count_init), positions_with_z[1:]
    )
    return unique_final[:final_count]


@beartype
def symmetry_expansion(
    crystal: CrystalStructure,
    sym_ops: List[str],
    tolerance: scalar_float = 1.0,
) -> CrystalStructure:
    """Apply symmetry operations to expand fractional positions.

    Generates a complete unit cell from an asymmetric unit by applying
    crystallographic symmetry operations and removing duplicate atoms.

    Parameters
    ----------
    crystal : CrystalStructure
        The initial crystal structure with symmetry-independent positions.
    sym_ops : List[str]
        List of symmetry operations as strings from the CIF file.
        Example: ["x,y,z", "-x,-y,z", "x+1/2,y+1/2,z"]
    tolerance : scalar_float, optional
        Distance tolerance in Angstroms for duplicate atom removal.
        Atoms within this distance are considered duplicates.
        Default: 1.0 A.

    Returns
    -------
    expanded_crystal : CrystalStructure
        Symmetry-expanded crystal structure without duplicates.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Parse symmetry operations into callable functions
    2. Apply each symmetry operation to each atomic position
    3. Apply modulo 1 to keep positions within unit cell
    4. Convert expanded positions to Cartesian coordinates
    5. Remove duplicate positions within tolerance
    6. Convert unique positions back to fractional coordinates
    7. Create and return expanded CrystalStructure

    Examples
    --------
    >>> from rheedium.inout import parse_cif, symmetry_expansion
    >>> structure = parse_cif("path/to/structure.cif")
    >>> expanded = symmetry_expansion(structure, ["x,y,z", "-x,-y,z"])
    >>> expanded.frac_positions.shape[0]
    2
    """
    frac_positions = crystal.frac_positions

    expanded_positions = _apply_symmetry_ops(frac_positions, sym_ops)

    cell_vectors = build_cell_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    cart_coords = expanded_positions[:, :3] @ cell_vectors
    cart_with_z = jnp.column_stack([cart_coords, expanded_positions[:, 3]])

    unique_cart_with_z = _deduplicate_positions(cart_with_z, tolerance)

    unique_cart = unique_cart_with_z[:, :3]
    atomic_numbers = unique_cart_with_z[:, 3]
    cell_inv = jnp.linalg.inv(cell_vectors)
    unique_frac = (unique_cart @ cell_inv) % 1.0

    expanded_crystal: CrystalStructure = create_crystal_structure(
        frac_positions=jnp.column_stack([unique_frac, atomic_numbers]),
        cart_positions=jnp.column_stack([unique_cart, atomic_numbers]),
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return expanded_crystal
