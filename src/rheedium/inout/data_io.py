import fractions
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Tuple, Union
from jaxtyping import Array, Float, Num, jaxtyped

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
    sym_ops_raw = re.findall(r"_symmetry_equiv_pos_as_xyz\s+'([^']+)'", cif_text)
    if not sym_ops_raw:
        sym_ops_matches = re.findall(r"_symmetry_equiv_pos_as_xyz\s+([^\n]+)", cif_text)
        sym_ops_raw = [m.strip().strip("'\"") for m in sym_ops_matches]
    sym_ops = [op.replace("'", "").replace('"', "").strip() for op in sym_ops_raw]
    if not sym_ops:
        sym_ops = ["x,y,z"]
    crystal = CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    expanded_crystal = symmetry_expansion(crystal, sym_ops, tolerance=1.0)
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
    """
    frac_positions = crystal.frac_positions
    expanded_positions = []

    def parse_sym_op(op_str: str):
        def op(pos):
            replacements = {"x": pos[0], "y": pos[1], "z": pos[2]}

            def safe_eval(comp: str) -> float:
                comp = comp.strip().lower().replace("'", "").replace('"', "")
                comp = comp.replace("-", "+-")
                terms = comp.split("+")
                result = 0.0
                for term in terms:
                    term = term.strip()
                    if not term:
                        continue
                    factor = 1.0
                    for var in ("x", "y", "z"):
                        if var in term:
                            parts = term.split(var)
                            coeff_str = parts[0].strip()
                            if coeff_str in ["", "+"]:
                                coeff = 1.0
                            elif coeff_str == "-":
                                coeff = -1.0
                            else:
                                coeff = float(fractions.Fraction(coeff_str))
                            factor *= coeff * replacements[var]
                            break
                    else:
                        factor *= float(fractions.Fraction(term))
                    result += factor
                return result

            components = op_str.strip().split(",")
            return jnp.array([safe_eval(comp) for comp in components])

        return op

    ops = [parse_sym_op(op) for op in sym_ops]
    for pos in frac_positions:
        xyz, atomic_number = pos[:3], pos[3]
        for op in ops:
            new_xyz = jnp.mod(op(xyz), 1.0)
            expanded_positions.append(jnp.concatenate([new_xyz, atomic_number[None]]))
    expanded_positions = jnp.array(expanded_positions)
    cell_vectors = rh.ucell.build_cell_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    cart_positions = expanded_positions[:, :3] @ cell_vectors

    def deduplicate_positions(
        cart: Float[Array, "N 3"],
        frac: Float[Array, "N 4"],
        tol: scalar_float,
    ) -> Tuple[Float[Array, "M 3"], Float[Array, "M 4"]]:
        N = cart.shape[0]

        def body_fn(i, carry):
            unique_cart, unique_frac, count = carry
            current_cart = cart[i]
            current_frac = frac[i]

            diff = unique_cart[:count] - current_cart
            dist_sq = jnp.sum(diff**2, axis=1)
            is_duplicate = jnp.any(dist_sq < tol**2)

            def append_atom(args):
                unique_cart, unique_frac, count = args
                unique_cart = unique_cart.at[count].set(current_cart)
                unique_frac = unique_frac.at[count].set(current_frac)
                return unique_cart, unique_frac, count + 1

            carry = jax.lax.cond(
                is_duplicate,
                lambda args: args,  # if duplicate, return unchanged
                append_atom,  # if unique, append atom
                carry,
            )
            return carry

        # Preallocate arrays to maximum possible size
        unique_cart_init = jnp.zeros_like(cart)
        unique_frac_init = jnp.zeros_like(frac)
        unique_cart_init = unique_cart_init.at[0].set(cart[0])
        unique_frac_init = unique_frac_init.at[0].set(frac[0])
        count_init = jnp.array(1, dtype=int)

        init_carry = (unique_cart_init, unique_frac_init, count_init)
        unique_cart, unique_frac, final_count = jax.lax.fori_loop(
            1, N, body_fn, init_carry
        )

        # Slice arrays to actual final size
        unique_cart_final = unique_cart[:final_count]
        unique_frac_final = unique_frac[:final_count]

        return unique_cart_final, unique_frac_final

    unique_cart, unique_frac = deduplicate_positions(
        cart_positions, expanded_positions, tolerance
    )
    expanded_crystal = CrystalStructure(
        frac_positions=unique_frac,
        cart_positions=jnp.column_stack((unique_cart, unique_frac[:, 3])),
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
    )
    return expanded_crystal
