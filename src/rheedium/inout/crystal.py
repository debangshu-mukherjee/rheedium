"""Unified crystal structure parsing and conversion utilities.

Extended Summary
----------------
This module provides a unified interface for parsing crystal structure files
(CIF and XYZ formats) into simulation-ready CrystalStructure objects. It also
provides conversion utilities between different crystal representations.

Routine Listings
----------------
lattice_to_cell_params : function
    Convert 3x3 lattice vectors to crystallographic cell parameters
parse_crystal : function
    Parse CIF or XYZ file into simulation-ready CrystalStructure
xyz_to_crystal : function
    Convert XYZData to CrystalStructure for simulation
_infer_lattice_from_positions : function, internal
    Infer orthorhombic lattice from atomic bounding box

Notes
-----
All functions return JAX-compatible data structures suitable for automatic
differentiation and GPU acceleration.
"""

import logging
from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    XYZData,
    create_crystal_structure,
    scalar_float,
)

from .cif import parse_cif
from .xyz import parse_xyz

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype)
def lattice_to_cell_params(
    lattice: Float[Array, "3 3"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    r"""Convert lattice vectors to cell lengths and angles.

    Computes crystallographic cell parameters (a, b, c, alpha, beta, gamma)
    from a 3x3 matrix of lattice vectors. This is the inverse operation of
    constructing lattice vectors from cell parameters.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Lattice vectors as rows: [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]].
        Each row represents one lattice vector (a, b, c respectively) with
        components in Cartesian coordinates (Angstroms).

    Returns
    -------
    cell_lengths : Float[Array, "3"]
        Unit cell lengths [a, b, c] in Angstroms. Computed as the Euclidean
        norm of each lattice vector.
    cell_angles : Float[Array, "3"]
        Unit cell angles [alpha, beta, gamma] in degrees.

        - ``alpha`` : angle between vectors b and c
        - ``beta`` : angle between vectors a and c
        - ``gamma`` : angle between vectors a and b

    Notes
    -----
    The conversion uses the standard crystallographic definitions:

    .. math::

        a = |\\mathbf{a}|, \\quad b = |\\mathbf{b}|, \\quad c = |\\mathbf{c}|

    .. math::

        \\cos(\\alpha) = \\frac{\\mathbf{b} \\cdot \\mathbf{c}}{bc}

    Cosine values are clipped to [-1, 1] to prevent numerical issues with
    arccos when vectors are nearly parallel or antiparallel.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheedium.inout import lattice_to_cell_params
    >>> lattice = jnp.eye(3) * 4.2
    >>> lengths, angles = lattice_to_cell_params(lattice)
    >>> lengths
    Array([4.2, 4.2, 4.2], dtype=float64)
    >>> angles
    Array([90., 90., 90.], dtype=float64)

    Non-orthogonal lattice (hexagonal):

    >>> hex_lattice = jnp.array([
    ...     [3.0, 0.0, 0.0],
    ...     [-1.5, 2.598, 0.0],
    ...     [0.0, 0.0, 5.0]
    ... ])
    >>> lengths, angles = lattice_to_cell_params(hex_lattice)
    """
    a_vec: Float[Array, "3"] = lattice[0]
    b_vec: Float[Array, "3"] = lattice[1]
    c_vec: Float[Array, "3"] = lattice[2]

    a: Float[Array, ""] = jnp.linalg.norm(a_vec)
    b: Float[Array, ""] = jnp.linalg.norm(b_vec)
    c: Float[Array, ""] = jnp.linalg.norm(c_vec)
    cell_lengths: Float[Array, "3"] = jnp.array([a, b, c])

    cos_alpha: Float[Array, ""] = jnp.dot(b_vec, c_vec) / (b * c)
    cos_beta: Float[Array, ""] = jnp.dot(a_vec, c_vec) / (a * c)
    cos_gamma: Float[Array, ""] = jnp.dot(a_vec, b_vec) / (a * b)

    cos_alpha = jnp.clip(cos_alpha, min=-1.0, max=1.0)
    cos_beta = jnp.clip(cos_beta, min=-1.0, max=1.0)
    cos_gamma = jnp.clip(cos_gamma, min=-1.0, max=1.0)

    alpha: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_alpha))
    beta: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_beta))
    gamma: Float[Array, ""] = jnp.degrees(jnp.arccos(cos_gamma))
    cell_angles: Float[Array, "3"] = jnp.array([alpha, beta, gamma])

    return cell_lengths, cell_angles


@jaxtyped(typechecker=beartype)
def _infer_lattice_from_positions(
    positions: Float[Array, "N 3"],
    padding_ang: scalar_float,
) -> Float[Array, "3 3"]:
    """Infer orthorhombic lattice from atomic bounding box.

    Computes a minimal orthorhombic (rectangular) unit cell that encompasses
    all atomic positions with specified padding. This is used as a fallback
    when XYZ files lack explicit lattice information.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian atomic positions in Angstroms with shape (N, 3) where N
        is the number of atoms. Each row contains [x, y, z] coordinates.
    padding_ang : scalar_float
        Padding added to each dimension in Angstroms. The padding is applied
        symmetrically (padding_ang on each side, so 2*padding_ang total per
        dimension).

    Returns
    -------
    lattice : Float[Array, "3 3"]
        Orthorhombic lattice vectors as a diagonal matrix. The diagonal
        elements are the cell dimensions [a, b, c] in Angstroms. Off-diagonal
        elements are zero (90-degree angles).

    Notes
    -----
    The algorithm:

    1. Computes the bounding box of all atomic positions (min/max in each
       dimension)
    2. Adds 2*padding_ang to each dimension (padding on both sides)
    3. Enforces a minimum extent of 1.0 Angstrom per dimension to avoid
       degenerate cells (e.g., for 2D slabs or linear molecules)
    4. Constructs a diagonal matrix with these extents

    This function always produces orthorhombic cells (alpha=beta=gamma=90°).
    For non-orthorhombic structures, explicit lattice vectors should be
    provided via the extended XYZ format or the cell_vectors parameter.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    >>> lattice = _infer_lattice_from_positions(positions, padding_ang=1.0)
    >>> lattice
    Array([[4., 0., 0.],
           [0., 4., 0.],
           [0., 0., 4.]], dtype=float64)
    """
    min_coords: Float[Array, "3"] = jnp.min(positions, axis=0)
    max_coords: Float[Array, "3"] = jnp.max(positions, axis=0)
    extent: Float[Array, "3"] = max_coords - min_coords + 2 * padding_ang

    min_extent: float = 1.0
    extent = jnp.maximum(extent, min_extent)

    lattice: Float[Array, "3 3"] = jnp.diag(extent)
    return lattice


@jaxtyped(typechecker=beartype)
def xyz_to_crystal(
    xyz_data: XYZData,
    cell_vectors: Optional[Float[Array, "3 3"]] = None,
    padding_ang: scalar_float = 2.0,
) -> CrystalStructure:
    """Convert XYZData to CrystalStructure for simulation.

    Transforms parsed XYZ file data into a CrystalStructure object suitable
    for RHEED simulation. Handles cell parameter determination from multiple
    sources with automatic fallback logic.

    Parameters
    ----------
    xyz_data : XYZData
        Parsed XYZ file data containing:

        - ``positions`` : Cartesian coordinates in Angstroms
        - ``atomic_numbers`` : Element atomic numbers
        - ``lattice`` : Optional lattice vectors from extended XYZ format
        - ``comment`` : Optional comment line metadata

    cell_vectors : Optional[Float[Array, "3 3"]], optional
        Override lattice vectors as rows [[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]].
        If provided, takes precedence over xyz_data.lattice. Default: None
    padding_ang : scalar_float, optional
        Padding added to inferred cell boundaries in Angstroms. Only used
        when inferring cell from atomic positions (when no lattice info
        is available). Default: 2.0

    Returns
    -------
    crystal : CrystalStructure
        Simulation-ready crystal structure containing:

        - ``frac_positions`` : Fractional coordinates with atomic numbers
        - ``cart_positions`` : Cartesian coordinates with atomic numbers
        - ``cell_lengths`` : [a, b, c] in Angstroms
        - ``cell_angles`` : [alpha, beta, gamma] in degrees

    Notes
    -----
    Cell determination priority:

    1. Explicit ``cell_vectors`` parameter (if provided)
    2. ``xyz_data.lattice`` (from extended XYZ ``Lattice=`` metadata)
    3. Inferred from atomic extent + padding (last resort)

    When inferring cell from positions:

    - Computes bounding box of all atomic positions
    - Adds ``padding_ang`` to each dimension on both sides
    - Creates orthorhombic cell (alpha=beta=gamma=90°)
    - Logs a warning to alert user of the inference

    Fractional coordinates are computed as:

    .. code-block:: python

        frac_coords = cart_coords @ inv(lattice)

    Examples
    --------
    Basic usage with extended XYZ file containing lattice:

    >>> import rheedium as rh
    >>> xyz_data = rh.inout.parse_xyz("structure.xyz")
    >>> crystal = rh.inout.xyz_to_crystal(xyz_data)

    Override lattice vectors:

    >>> import jax.numpy as jnp
    >>> custom_cell = jnp.eye(3) * 5.0
    >>> crystal = rh.inout.xyz_to_crystal(xyz_data, cell_vectors=custom_cell)

    Adjust padding for inferred cells:

    >>> crystal = rh.inout.xyz_to_crystal(xyz_data, padding_ang=3.0)
    """
    positions: Float[Array, "N 3"] = xyz_data.positions
    atomic_numbers: Int[Array, "N"] = xyz_data.atomic_numbers

    lattice: Float[Array, "3 3"]
    if cell_vectors is not None:
        lattice = jnp.asarray(cell_vectors)
    elif xyz_data.lattice is not None:
        lattice = xyz_data.lattice
        is_identity = jnp.allclose(lattice, jnp.eye(3))
        if is_identity:
            logger.warning(
                "XYZ file has no Lattice= metadata. Inferring cell from "
                "atomic positions with %.1f A padding.",
                padding_ang,
            )
            lattice = _infer_lattice_from_positions(positions, padding_ang)
    else:
        logger.warning(
            "XYZ file has no Lattice= metadata. Inferring cell from "
            "atomic positions with %.1f A padding.",
            padding_ang,
        )
        lattice = _infer_lattice_from_positions(positions, padding_ang)

    cell_lengths: Float[Array, "3"]
    cell_angles: Float[Array, "3"]
    cell_lengths, cell_angles = lattice_to_cell_params(lattice)

    lattice_inv: Float[Array, "3 3"] = jnp.linalg.inv(lattice)
    frac_coords: Float[Array, "N 3"] = positions @ lattice_inv

    frac_positions: Float[Array, "N 4"] = jnp.column_stack(
        [frac_coords, atomic_numbers.astype(jnp.float64)]
    )
    cart_positions: Float[Array, "N 4"] = jnp.column_stack(
        [positions, atomic_numbers.astype(jnp.float64)]
    )

    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


@jaxtyped(typechecker=beartype)
def parse_crystal(
    file_path: Union[str, Path],
) -> CrystalStructure:
    """Parse CIF, XYZ, or POSCAR file into simulation-ready CrystalStructure.

    Provides a unified interface for loading crystal structures from common
    file formats. Auto-detects the format based on file extension or filename
    and delegates to the appropriate parser.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to crystal structure file. Supported formats:

        - ``.cif`` : Crystallographic Information File
        - ``.xyz`` : XYZ coordinate file (standard or extended format)
        - ``POSCAR``, ``CONTCAR`` : VASP structure files (by filename)
        - ``.poscar``, ``.contcar`` : VASP structure files (by extension)

        Can be a string path or pathlib.Path object.

    Returns
    -------
    crystal : CrystalStructure
        JAX-compatible crystal structure containing:

        - ``frac_positions`` : Fractional coordinates with atomic numbers
        - ``cart_positions`` : Cartesian coordinates with atomic numbers
        - ``cell_lengths`` : [a, b, c] in Angstroms
        - ``cell_angles`` : [alpha, beta, gamma] in degrees

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If file format is not supported.
        Also raised by underlying parsers for malformed files.

    Notes
    -----
    Format-specific behavior:

    **CIF files (.cif)**:

    - Parses cell parameters from ``_cell_length_*`` and ``_cell_angle_*``
    - Reads atomic positions from ``_atom_site_*`` loops
    - Applies symmetry operations from ``_symmetry_equiv_pos_as_xyz``
    - Deduplicates overlapping atoms after symmetry expansion

    **XYZ files (.xyz)**:

    - Reads Cartesian coordinates directly
    - Uses ``Lattice=`` metadata from extended XYZ format if present
    - Falls back to inferring cell from atomic bounding box + padding
    - Converts Cartesian to fractional coordinates

    **POSCAR/CONTCAR files**:

    - Supports VASP 5.x format with species line
    - Handles both Direct (fractional) and Cartesian coordinates
    - Applies scaling factor to lattice vectors

    This function is the recommended entry point for loading crystal
    structures, as it handles format detection automatically.

    Examples
    --------
    Load a CIF file:

    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_crystal("MgO.cif")
    >>> crystal.cell_lengths
    Array([4.21, 4.21, 4.21], dtype=float64)

    Load an XYZ file:

    >>> crystal = rh.inout.parse_crystal("slab.xyz")
    >>> crystal.frac_positions.shape
    (100, 4)

    Load a POSCAR file:

    >>> crystal = rh.inout.parse_crystal("POSCAR")
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)

    Works with pathlib.Path:

    >>> from pathlib import Path
    >>> crystal = rh.inout.parse_crystal(Path("structure.cif"))
    """
    path: Path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Crystal file not found: {path}")

    suffix: str = path.suffix.lower()
    name: str = path.name.upper()

    if suffix == ".cif":
        return parse_cif(path)

    if suffix == ".xyz":
        xyz_data: XYZData = parse_xyz(path)
        return xyz_to_crystal(xyz_data)

    if name in ("POSCAR", "CONTCAR") or suffix in (".poscar", ".contcar"):
        from .poscar import parse_poscar

        return parse_poscar(path)

    supported: str = ".cif, .xyz, POSCAR, CONTCAR"
    raise ValueError(
        f"Unsupported file format '{suffix}'. Supported: {supported}"
    )
