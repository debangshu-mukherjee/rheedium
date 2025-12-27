"""External library interoperability for ASE and pymatgen.

Extended Summary
----------------
This module provides bidirectional conversion functions between rheedium's
CrystalStructure and external atomistic simulation libraries (ASE and
pymatgen). Dependencies are lazily imported to allow rheedium to function
without these optional packages installed.

Routine Listings
----------------
from_ase : function
    Convert ASE Atoms to CrystalStructure.
to_ase : function
    Convert CrystalStructure to ASE Atoms.
from_pymatgen : function
    Convert pymatgen Structure to CrystalStructure.
to_pymatgen : function
    Convert CrystalStructure to pymatgen Structure.

Notes
-----
ASE and pymatgen are optional dependencies. Functions will raise ImportError
with installation instructions if the required library is not installed.
"""

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, Int, jaxtyped

from rheedium.types import CrystalStructure, create_xyz_data
from rheedium.ucell import build_cell_vectors

from .crystal import xyz_to_crystal
from .xyz import _ATOMIC_NUMBERS


_Z_TO_SYMBOL: dict[int, str] = {v: k for k, v in _ATOMIC_NUMBERS.items()}


@jaxtyped(typechecker=beartype)
def from_ase(atoms: Any) -> CrystalStructure:
    """Convert ASE Atoms object to CrystalStructure.

    Extracts cell parameters, atomic positions, and species from an ASE
    Atoms object and creates a rheedium CrystalStructure suitable for
    RHEED simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object with cell and positions defined. The cell must
        be a valid 3D periodic cell (not degenerate).

    Returns
    -------
    crystal : CrystalStructure
        Equivalent rheedium crystal structure containing:

        - ``frac_positions`` : Fractional coordinates with atomic numbers
        - ``cart_positions`` : Cartesian coordinates with atomic numbers
        - ``cell_lengths`` : [a, b, c] in Angstroms
        - ``cell_angles`` : [alpha, beta, gamma] in degrees

    Raises
    ------
    ImportError
        If ASE is not installed.
    ValueError
        If atoms has no cell defined, cell is degenerate (volume near zero),
        or cell has fewer than 3 dimensions.

    Notes
    -----
    The conversion extracts:

    - ``atoms.get_cell()`` : Unit cell vectors
    - ``atoms.get_positions()`` : Cartesian atomic positions
    - ``atoms.get_atomic_numbers()`` : Element atomic numbers

    Periodic boundary conditions (PBC) from the ASE Atoms are not preserved
    in CrystalStructure, which assumes full 3D periodicity.

    Examples
    --------
    >>> from ase.build import bulk
    >>> import rheedium as rh
    >>> si = bulk('Si', 'diamond', a=5.43)
    >>> crystal = rh.inout.from_ase(si)
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)
    """
    try:
        from ase import Atoms
    except ImportError as e:
        raise ImportError(
            "ASE is not installed. Install with: pip install ase"
        ) from e

    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected ase.Atoms, got {type(atoms).__name__}")

    cell = atoms.get_cell()

    if cell is None or cell.rank < 3:
        raise ValueError(
            "ASE Atoms must have a valid 3D cell defined. "
            "Set cell with atoms.set_cell() or use atoms.center()."
        )

    cell_volume = abs(cell.volume)
    if cell_volume < 1e-10:
        raise ValueError(
            f"ASE Atoms cell is degenerate (volume={cell_volume:.2e}). "
            "Please define a valid unit cell."
        )

    lattice: Float[Array, "3 3"] = jnp.asarray(
        cell[:], dtype=jnp.float64
    )
    positions: Float[Array, "N 3"] = jnp.asarray(
        atoms.get_positions(), dtype=jnp.float64
    )
    atomic_numbers: Int[Array, "N"] = jnp.asarray(
        atoms.get_atomic_numbers(), dtype=jnp.int32
    )

    xyz_data = create_xyz_data(
        positions=positions,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
    )

    return xyz_to_crystal(xyz_data)


@jaxtyped(typechecker=beartype)
def to_ase(crystal: CrystalStructure) -> Any:
    """Convert CrystalStructure to ASE Atoms object.

    Creates an ASE Atoms object from a rheedium CrystalStructure,
    preserving cell parameters, positions, and atomic species.

    Parameters
    ----------
    crystal : CrystalStructure
        rheedium crystal structure to convert.

    Returns
    -------
    atoms : ase.Atoms
        Equivalent ASE Atoms object with:

        - ``cell`` : Unit cell vectors from cell_lengths and cell_angles
        - ``positions`` : Cartesian atomic coordinates
        - ``numbers`` : Atomic numbers
        - ``pbc`` : Periodic boundary conditions set to True

    Raises
    ------
    ImportError
        If ASE is not installed.

    Notes
    -----
    The created Atoms object has ``pbc=True`` (full 3D periodicity),
    matching the assumption in rheedium that crystal structures are
    periodic.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> atoms = rh.inout.to_ase(crystal)
    >>> atoms.get_cell()
    Cell([...])
    >>> atoms.write("structure.xyz")
    """
    try:
        from ase import Atoms
    except ImportError as e:
        raise ImportError(
            "ASE is not installed. Install with: pip install ase"
        ) from e

    cell: Float[Array, "3 3"] = build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )

    positions_np: np.ndarray = np.asarray(crystal.cart_positions[:, :3])
    atomic_numbers_np: np.ndarray = np.asarray(
        crystal.cart_positions[:, 3], dtype=int
    )
    cell_np: np.ndarray = np.asarray(cell)

    atoms = Atoms(
        numbers=atomic_numbers_np,
        positions=positions_np,
        cell=cell_np,
        pbc=True,
    )

    return atoms


@jaxtyped(typechecker=beartype)
def from_pymatgen(structure: Any) -> CrystalStructure:
    """Convert pymatgen Structure to CrystalStructure.

    Extracts lattice, positions, and species from a pymatgen Structure
    object and creates a rheedium CrystalStructure suitable for RHEED
    simulation.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        pymatgen Structure object to convert.

    Returns
    -------
    crystal : CrystalStructure
        Equivalent rheedium crystal structure containing:

        - ``frac_positions`` : Fractional coordinates with atomic numbers
        - ``cart_positions`` : Cartesian coordinates with atomic numbers
        - ``cell_lengths`` : [a, b, c] in Angstroms
        - ``cell_angles`` : [alpha, beta, gamma] in degrees

    Raises
    ------
    ImportError
        If pymatgen is not installed.
    TypeError
        If input is not a pymatgen Structure.

    Notes
    -----
    The conversion extracts:

    - ``structure.lattice.matrix`` : Lattice vectors
    - ``structure.cart_coords`` : Cartesian atomic positions
    - ``site.specie.Z`` : Atomic numbers for each site

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> import rheedium as rh
    >>> struct = Structure.from_file("POSCAR")
    >>> crystal = rh.inout.from_pymatgen(struct)
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)
    """
    try:
        from pymatgen.core import Structure
    except ImportError as e:
        raise ImportError(
            "pymatgen is not installed. Install with: pip install pymatgen"
        ) from e

    if not isinstance(structure, Structure):
        raise TypeError(
            f"Expected pymatgen.core.Structure, got {type(structure).__name__}"
        )

    lattice: Float[Array, "3 3"] = jnp.asarray(
        structure.lattice.matrix, dtype=jnp.float64
    )

    positions: Float[Array, "N 3"] = jnp.asarray(
        structure.cart_coords, dtype=jnp.float64
    )

    atomic_numbers_list = [site.specie.Z for site in structure]
    atomic_numbers: Int[Array, "N"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )

    xyz_data = create_xyz_data(
        positions=positions,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
    )

    return xyz_to_crystal(xyz_data)


@jaxtyped(typechecker=beartype)
def to_pymatgen(crystal: CrystalStructure) -> Any:
    """Convert CrystalStructure to pymatgen Structure.

    Creates a pymatgen Structure object from a rheedium CrystalStructure,
    preserving lattice, positions, and atomic species.

    Parameters
    ----------
    crystal : CrystalStructure
        rheedium crystal structure to convert.

    Returns
    -------
    structure : pymatgen.core.Structure
        Equivalent pymatgen Structure object.

    Raises
    ------
    ImportError
        If pymatgen is not installed.

    Notes
    -----
    The created Structure uses:

    - Lattice from cell_lengths and cell_angles (reconstructed via
      build_cell_vectors)
    - Fractional coordinates from frac_positions
    - Element species from atomic numbers

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> struct = rh.inout.to_pymatgen(crystal)
    >>> struct.to("POSCAR", "output_POSCAR")
    """
    try:
        from pymatgen.core import Lattice, Structure
    except ImportError as e:
        raise ImportError(
            "pymatgen is not installed. Install with: pip install pymatgen"
        ) from e

    cell: Float[Array, "3 3"] = build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )

    lattice = Lattice(np.asarray(cell))

    atomic_numbers = np.asarray(crystal.frac_positions[:, 3], dtype=int)
    frac_coords = np.asarray(crystal.frac_positions[:, :3])

    species = [_Z_TO_SYMBOL.get(z, f"X{z}") for z in atomic_numbers]

    structure = Structure(lattice, species, frac_coords)

    return structure
