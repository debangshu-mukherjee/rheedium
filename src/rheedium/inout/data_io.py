"""
Module: inout.data_io
---------------------
Functions for reading and writing crystal structure data.

Functions
---------
- `load_atomic_numbers`:
    Load atomic numbers mapping from JSON file
- `parse_cif`:
    Parse CIF file into JAX-compatible CrystalStructure
- `symmetry_expansion`:
    Apply symmetry operations to expand fractional positions
"""

import fractions
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Union
from jaxtyping import Array, Float, Num, jaxtyped

import rheedium as rh
from rheedium.types import CrystalStructure, scalar_float

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

    Flow
    ----
    - Open and read the JSON file at the specified path
    - Parse the JSON content into a dictionary
    - Return the atomic numbers mapping

    Examples
    --------
    >>> from rheedium.inout.data_io import load_atomic_numbers
    >>> atomic_numbers = load_atomic_numbers()
    >>> print(atomic_numbers["Si"])
    14
    >>> print(atomic_numbers["Au"])
    79
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

    Flow
    ----
    - Validate CIF file path and extension
    - Read CIF file content
    - Load atomic numbers mapping
    - Extract unit cell parameters:
        - Parse cell lengths (a, b, c)
        - Parse cell angles (alpha, beta, gamma)
    - Parse atomic positions:
        - Find atom site loop section
        - Extract required columns
        - Parse element symbols and fractional coordinates
        - Convert element symbols to atomic numbers
    - Convert fractional to Cartesian coordinates:
        - Build cell vectors
        - Transform coordinates
    - Parse symmetry operations:
        - Find symmetry operations section
        - Extract operation strings
    - Create initial CrystalStructure
    - Apply symmetry operations to expand positions
    - Return expanded crystal structure

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
    sym_ops = []
    lines = cif_text.splitlines()
    collect_sym_ops = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("_symmetry_equiv_pos_as_xyz"):
            collect_sym_ops = True
            continue
        if collect_sym_ops:
            if stripped_line.startswith("'") and stripped_line.endswith("'"):
                op_clean = stripped_line.strip("'").strip()
                sym_ops.append(op_clean)
            elif stripped_line.startswith('"') and stripped_line.endswith('"'):
                op_clean = stripped_line.strip('"').strip()
                sym_ops.append(op_clean)
            else:
                if sym_ops:
                    break

    if not sym_ops:
        sym_ops = ["x,y,z"]

    crystal: CrystalStructure = rh.types.create_crystal_structure(
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
    """
    Description
    -----------
    Apply symmetry operations to expand fractional positions and remove duplicates.

    Parameters
    ----------
    - `crystal` (CrystalStructure):
        The initial crystal structure with symmetry-independent positions.
    - `sym_ops` (List[str]):
        List of symmetry operations as strings from the CIF file.
        Example: ["x,y,z", "-x,-y,z", ...]
    - `tolerance` (scalar_float):
        Distance tolerance in angstroms for duplicate atom removal.
        Default: 1.0 Å.

    Returns
    -------
    - `expanded_crystal` (CrystalStructure):
        Symmetry-expanded crystal structure without duplicates.

    Flow
    ----
    - Parse symmetry operations into functions:
        - Split operation strings into components
        - Create functions to evaluate each component
        - Handle coefficients and variables
    - Apply symmetry operations:
        - For each atomic position
        - For each symmetry operation
        - Generate new positions
        - Apply modulo 1 to keep in unit cell
    - Convert expanded positions to Cartesian coordinates:
        - Build cell vectors
        - Transform coordinates
    - Remove duplicate positions:
        - Calculate distances between positions
        - Keep only unique positions within tolerance
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
    frac_positions = crystal.frac_positions
    expanded_positions = []

    def parse_sym_op(op_str: str):
        def op(pos):
            replacements = {"x": pos[0], "y": pos[1], "z": pos[2]}
            components = op_str.lower().replace(" ", "").split(",")

            def eval_comp(comp):
                comp = comp.replace("-", "+-")
                terms = comp.split("+")
                total = 0.0
                for term in terms:
                    if not term:
                        continue
                    coeff = 1.0
                    for var in ("x", "y", "z"):
                        if var in term:
                            part = term.split(var)[0]
                            coeff = float(fractions.Fraction(part)) if part else 1.0
                            total += coeff * replacements[var]
                            break
                    else:
                        total += float(fractions.Fraction(term))
                return total

            return jnp.array([eval_comp(c) for c in components])

        return op

    ops = [parse_sym_op(op) for op in sym_ops]
    for pos in frac_positions:
        xyz, atomic_number = pos[:3], pos[3]
        for op in ops:
            new_xyz = jnp.mod(op(xyz), 1.0)
            expanded_positions.append(jnp.concatenate([new_xyz, atomic_number[None]]))
    expanded_positions = jnp.array(expanded_positions)
    cart_positions = expanded_positions[:, :3] @ rh.ucell.build_cell_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )

    def deduplicate(positions, tol):
        def dist(p1, p2):
            return jnp.sqrt(jnp.sum((p1 - p2) ** 2))

        def unique_cond(carry, pos):
            unique, count = carry
            diff = unique - pos
            dist_sq = jnp.sum(diff**2, axis=1)
            is_dup = jnp.any(dist_sq < tol**2)
            unique = jax.lax.cond(
                is_dup,
                lambda u: u,
                lambda u: u.at[count].set(pos),
                unique,
            )
            count += jnp.logical_not(is_dup)
            return (unique, count), None

        unique_init = jnp.zeros_like(positions)
        unique_init = unique_init.at[0].set(positions[0])
        count_init = 1
        (unique_final, final_count), _ = jax.lax.scan(
            unique_cond, (unique_init, count_init), positions[1:]
        )
        return unique_final[:final_count]

    unique_cart = deduplicate(cart_positions, tolerance)
    cell_inv = jnp.linalg.inv(
        rh.ucell.build_cell_vectors(*crystal.cell_lengths, *crystal.cell_angles)
    )
    unique_frac = (unique_cart @ cell_inv) % 1.0
    atomic_numbers = expanded_positions[:, 3][: unique_cart.shape[0]]
    expanded_crystal: CrystalStructure = rh.types.create_crystal_structure(
        frac_positions=jnp.column_stack([unique_frac, atomic_numbers]),
        cart_positions=jnp.column_stack([unique_cart, atomic_numbers]),
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return expanded_crystal
