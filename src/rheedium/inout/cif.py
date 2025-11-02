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
    Parse CIF file into JAX-compatible CrystalStructure
symmetry_expansion : function
    Apply symmetry operations to expand fractional positions

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
from jaxtyping import Array, Float, Int, Num, jaxtyped

from rheedium.inout.xyz import atomic_symbol
from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors


@jaxtyped(typechecker=beartype)
def parse_cif(cif_path: Union[str, Path]) -> CrystalStructure:
    r"""Parse a CIF file into a JAX-compatible CrystalStructure.

    Parameters
    ----------
    cif_path : str | Path
        Path to the CIF file.

    Returns
    -------
    expanded_crystal : CrystalStructure
        Parsed crystal structure object with fractional and Cartesian
        coordinates. Contains arrays of atomic positions in both fractional
        (range [0,1]) and Cartesian (Ångstroms) coordinates, along with unit
        cell parameters (lengths in Ångstroms, angles in degrees).

    Notes
    -----
    The algorithm proceeds as follows:

    1. Validate CIF file path and extension
    2. Read CIF file content
    3. Load atomic numbers mapping
    4. Extract unit cell parameters (cell lengths and angles)
    5. Parse atomic positions from atom site loop section
    6. Convert element symbols to atomic numbers
    7. Convert fractional to Cartesian coordinates using cell vectors
    8. Parse symmetry operations from CIF file
    9. Create initial CrystalStructure
    10. Apply symmetry operations to expand positions
    11. Return expanded crystal structure

    Examples
    --------
    >>> from rheedium.inout.data_io import parse_cif
    >>> # Parse a CIF file for silicon
    >>> structure = parse_cif("path/to/silicon.cif")
    >>> print(f"Unit cell vectors:\n{structure.vectors}")
    Unit cell vectors:
    [[5.431 0.000 0.000]
     [0.000 5.431 0.000]
     [0.000 0.000 5.431]]
    >>> print(f"Number of atoms: {len(structure.positions)}")
    Number of atoms: 8
    """
    cif_path: Path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    if cif_path.suffix.lower() != ".cif":
        raise ValueError(f"File must have .cif extension: {cif_path}")
    cif_text: str = cif_path.read_text()

    def _extract_param(name: str) -> float:
        """Extract a numerical parameter from CIF text."""
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
    cell_lengths: Num[Array, "3"] = jnp.array([a, b, c], dtype=jnp.float64)
    cell_angles: Num[Array, "3"] = jnp.array(
        [alpha, beta, gamma], dtype=jnp.float64
    )
    lines: List[str] = cif_text.splitlines()
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
        if (
            in_atom_site_loop
            and stripped_line
            and not stripped_line.startswith("_")
        ):
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
            atomic_number: int = atomic_symbol(element_symbol)
            positions_list.append([frac_x, frac_y, frac_z, atomic_number])
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
    sym_ops: List[str] = []
    lines = cif_text.splitlines()
    collect_sym_ops: bool = False
    for line in lines:
        stripped_line: str = line.strip()
        if stripped_line.startswith("_symmetry_equiv_pos_as_xyz"):
            collect_sym_ops = True
            continue
        if collect_sym_ops:
            if stripped_line.startswith("'") and stripped_line.endswith("'"):
                op_clean: str = stripped_line.strip("'").strip()
                sym_ops.append(op_clean)
            elif stripped_line.startswith('"') and stripped_line.endswith('"'):
                op_clean: str = stripped_line.strip('"').strip()
                sym_ops.append(op_clean)
            elif sym_ops:
                break

    if not sym_ops:
        sym_ops = ["x,y,z"]

    crystal: CrystalStructure = create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    expanded_crystal: CrystalStructure = symmetry_expansion(
        crystal, sym_ops, tolerance=1.0
    )
    return expanded_crystal


@jaxtyped(typechecker=beartype)
def symmetry_expansion(
    crystal: CrystalStructure,
    sym_ops: List[str],
    tolerance: scalar_float = 1.0,
) -> CrystalStructure:
    """Apply symmetry operations to expand fractional positions.

    Parameters
    ----------
    crystal : CrystalStructure
        The initial crystal structure with symmetry-independent positions.
    sym_ops : List[str]
        List of symmetry operations as strings from the CIF file.
        Example: ["x,y,z", "-x,-y,z", ...]
    tolerance : scalar_float, optional
        Distance tolerance in angstroms for duplicate atom removal.
        Default: 1.0 Å.

    Returns
    -------
    CrystalStructure
        Symmetry-expanded crystal structure without duplicates.

    Notes
    -----
    - Parse symmetry operations into functions by splitting operation strings.
      into components and creating evaluation functions for coefficients
    - Apply symmetry operations to each atomic position to generate new
      positions.
    - Apply modulo 1 to keep positions within unit cell
    - Convert expanded positions to Cartesian coordinates using cell vectors
    - Remove duplicate positions by calculating distances and keeping only
      unique positions within tolerance
    - Create and return expanded CrystalStructure

    Examples
    --------
    >>> from rheedium.inout.data_io import parse_cif, symmetry_expansion
    >>> # Parse a CIF file and expand symmetry
    >>> structure = parse_cif("path/to/structure.cif")
    >>> expanded = symmetry_expansion(structure)
    >>> print(f"Original atoms: {len(structure.positions)}")
    >>> print(f"Expanded atoms: {len(expanded.positions)}")
    Original atoms: 1
    Expanded atoms: 8
    """
    frac_positions: Float[Array, "N 4"] = crystal.frac_positions
    expanded_positions: List[Array] = []

    def _parse_sym_op(op_str: str) -> Callable[[Array], Array]:
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
                            coeff = (
                                float(fractions.Fraction(part))
                                if part
                                else 1.0
                            )
                            total += coeff * replacements[var]
                            break
                    else:
                        total += float(fractions.Fraction(term))
                return total

            return jnp.array([_eval_comp(c) for c in components])

        return _op

    ops: List[Callable[[Array], Array]] = [_parse_sym_op(op) for op in sym_ops]
    for pos in frac_positions:
        xyz: Array = pos[:3]
        atomic_number: float = pos[3]
        for op in ops:
            new_xyz: Array = jnp.mod(op(xyz), 1.0)
            expanded_positions.append(
                jnp.concatenate([new_xyz, atomic_number[None]])
            )
    expanded_positions: Float[Array, "N 4"] = jnp.array(expanded_positions)
    cart_positions: Float[Array, "N 3"] = expanded_positions[
        :, :3
    ] @ build_cell_vectors(*crystal.cell_lengths, *crystal.cell_angles)

    def _deduplicate(
        positions: Float[Array, "N 3"], tol: scalar_float
    ) -> Float[Array, "N 3"]:
        def _unique_cond(
            carry: Tuple[Float[Array, "N 3"], Int[Array, ""]],
            pos: Float[Array, "3"],
        ) -> Tuple[Tuple[Float[Array, "N 3"], Int[Array, ""]], None]:
            unique: Float[Array, "N 3"]
            count: Int[Array, ""]
            unique, count = carry
            diff: Float[Array, "N 3"] = unique - pos
            dist_sq: Float[Array, "N"] = jnp.sum(diff**2, axis=1)
            is_dup: bool = jnp.any(dist_sq < tol**2)
            unique: Float[Array, "N 3"] = jax.lax.cond(
                is_dup,
                lambda u: u,
                lambda u: u.at[count].set(pos),
                unique,
            )
            count += jnp.logical_not(is_dup)
            return (unique, count), None

        unique_init: Float[Array, "N 3"] = jnp.zeros_like(positions)
        unique_init = unique_init.at[0].set(positions[0])
        count_init: int = 1
        (unique_final, final_count), _ = jax.lax.scan(
            _unique_cond, (unique_init, count_init), positions[1:]
        )
        return unique_final[:final_count]

    unique_cart: Float[Array, "N 3"] = _deduplicate(cart_positions, tolerance)
    cell_inv: Float[Array, "3 3"] = jnp.linalg.inv(
        build_cell_vectors(*crystal.cell_lengths, *crystal.cell_angles)
    )
    unique_frac: Float[Array, "N 3"] = (unique_cart @ cell_inv) % 1.0
    atomic_numbers: Float[Array, "N"] = expanded_positions[:, 3][
        : unique_cart.shape[0]
    ]
    expanded_crystal: CrystalStructure = create_crystal_structure(
        frac_positions=jnp.column_stack([unique_frac, atomic_numbers]),
        cart_positions=jnp.column_stack([unique_cart, atomic_numbers]),
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return expanded_crystal
