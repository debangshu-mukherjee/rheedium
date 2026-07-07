"""External library interoperability for ASE and pymatgen.

Extended Summary
----------------
This module provides bidirectional conversion functions between rheedium's
CrystalStructure and external atomistic simulation libraries (ASE and
pymatgen). Both are core dependencies.

Routine Listings
----------------
:func:`from_ase`
    Convert ASE Atoms to CrystalStructure.
:func:`to_ase`
    Convert CrystalStructure to ASE Atoms.
:func:`from_pymatgen`
    Convert pymatgen Structure to CrystalStructure.
:func:`to_pymatgen`
    Convert CrystalStructure to pymatgen Structure.

Notes
-----
Both ASE and pymatgen are core dependencies and are always available.
"""

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure

from rheedium.types import (
    CrystalStructure,
    XYZData,
    create_crystal_structure,
    create_xyz_data,
)
from rheedium.ucell import build_cell_vectors

from .crystal import xyz_to_crystal
from .xyz import _ATOMIC_NUMBERS

_Z_TO_SYMBOL: dict[int, str] = {v: k for k, v in _ATOMIC_NUMBERS.items()}


def _with_occupancies(
    crystal: CrystalStructure,
    occupancies: Float[Array, "N"],
) -> CrystalStructure:
    """Rebuild a crystal with the given per-site occupancies attached."""
    return create_crystal_structure(
        frac_positions=crystal.frac_positions,
        cart_positions=crystal.cart_positions,
        cell_lengths=crystal.cell_lengths,
        cell_angles=crystal.cell_angles,
        occupancies=occupancies,
    )


def from_ase(atoms: Atoms) -> CrystalStructure:
    """Convert ASE Atoms object to CrystalStructure.

    Extracts cell parameters, atomic positions, and species from an ASE
    Atoms object and creates a rheedium CrystalStructure suitable for
    RHEED simulation.

    :see: :class:`~.test_interop.TestAseInterop`

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
    ValueError
        If atoms has no cell defined, cell is degenerate (volume near
        zero), or cell has fewer than 3 dimensions.

    Notes
    -----
    The conversion extracts:

    - ``atoms.get_cell()`` : Unit cell vectors
    - ``atoms.get_positions()`` : Cartesian atomic positions
    - ``atoms.get_atomic_numbers()`` : Element atomic numbers

    Periodic boundary conditions (PBC) from the ASE Atoms are not
    preserved in CrystalStructure, which assumes full 3D periodicity.

    ASE has no first-class site occupancy. If
    ``atoms.info["occupancies"]`` is present (as written by
    :func:`to_ase`), it is validated against the atom count and stored
    in ``crystal.occupancies``; otherwise all sites are fully occupied.

    1. **Validate input** --
       Check cell is 3D and non-degenerate.
    2. **Extract data** --
       Cell, positions, and atomic numbers from
       ASE Atoms object.
    3. **Convert** --
       Create XYZData and delegate to
       :func:`xyz_to_crystal`.
    4. **Attach occupancies** --
       Read ``atoms.info["occupancies"]`` when present.

    Examples
    --------
    >>> from ase.build import bulk
    >>> import rheedium as rh
    >>> si = bulk("Si", "diamond", a=5.43)
    >>> crystal = rh.inout.from_ase(si)
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected ase.Atoms, got {type(atoms).__name__}.")

    cell = atoms.get_cell()

    _min_cell_rank: int = 3
    if cell is None or cell.rank < _min_cell_rank:
        raise ValueError(
            "ASE Atoms must have a valid 3D cell defined. "
            "Set cell with atoms.set_cell() or use atoms.center()."
        )

    cell_volume: float = abs(cell.volume)
    _min_cell_volume: float = 1e-10
    if cell_volume < _min_cell_volume:
        raise ValueError(
            f"ASE Atoms cell is degenerate (volume={cell_volume:.2e}). "
            "Please define a valid unit cell."
        )

    lattice: Float[Array, "3 3"] = jnp.asarray(cell.array, dtype=jnp.float64)
    positions: Float[Array, "N 3"] = jnp.asarray(
        atoms.get_positions(), dtype=jnp.float64
    )
    atomic_numbers: Int[Array, "N"] = jnp.asarray(
        atoms.get_atomic_numbers(), dtype=jnp.int32
    )

    xyz_data: XYZData = create_xyz_data(
        positions=positions,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
    )

    crystal: CrystalStructure = xyz_to_crystal(xyz_data)
    info_occupancies = atoms.info.get("occupancies")
    if info_occupancies is not None:
        occupancies: Float[Array, "N"] = jnp.asarray(
            np.asarray(info_occupancies, dtype=np.float64)
        )
        if occupancies.shape != (positions.shape[0],):
            raise ValueError(
                "atoms.info['occupancies'] must have one entry per atom"
            )
        crystal = _with_occupancies(crystal, occupancies)
    return crystal


@jaxtyped(typechecker=beartype)
def to_ase(crystal: CrystalStructure) -> Atoms:
    """Convert CrystalStructure to ASE Atoms object.

    Creates an ASE Atoms object from a rheedium CrystalStructure,
    preserving cell parameters, positions, and atomic species.

    :see: :class:`~.test_interop.TestAseInterop`

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

    Notes
    -----
    The created Atoms object has ``pbc=True`` (full 3D periodicity),
    matching the assumption in rheedium that crystal structures are
    periodic.

    ASE has no first-class site occupancy, so the crystal's
    ``occupancies`` array is stashed in ``atoms.info["occupancies"]``;
    :func:`from_ase` reads it back for a lossless round trip.

    1. **Reconstruct cell** --
       Build cell vectors from lengths and angles.
    2. **Convert to NumPy** --
       Extract positions and atomic numbers.
    3. **Create Atoms** --
       ASE Atoms with ``pbc=True``.
    4. **Stash occupancies** --
       Store ``crystal.occupancies`` in ``atoms.info``.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> atoms = rh.inout.to_ase(crystal)
    >>> atoms.get_cell()
    Cell([...])
    >>> atoms.write("structure.xyz")
    """
    cell: Float[Array, "3 3"] = build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )

    positions_np: Float[NDArray, "N 3"] = np.asarray(
        crystal.cart_positions[:, :3]
    )
    atomic_numbers_np: Int[NDArray, "N"] = np.asarray(
        crystal.cart_positions[:, 3], dtype=int
    )
    cell_np: Float[NDArray, "3 3"] = np.asarray(cell)

    atoms: Atoms = Atoms(
        numbers=atomic_numbers_np,
        positions=positions_np,
        cell=cell_np,
        pbc=True,
    )
    if crystal.occupancies is not None:
        # ASE has no first-class site occupancy; stash the array in
        # atoms.info so from_ase can round-trip it.
        atoms.info["occupancies"] = np.asarray(
            crystal.occupancies, dtype=np.float64
        )
    return atoms


def from_pymatgen(structure: Structure) -> CrystalStructure:
    """Convert pymatgen Structure to CrystalStructure.

    Extracts lattice, positions, and species from a pymatgen Structure
    object and creates a rheedium CrystalStructure suitable for RHEED
    simulation.

    :see: :class:`~.test_interop.TestPymatgenInterop`

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

    Notes
    -----
    The conversion extracts:

    - ``structure.lattice.matrix`` : Lattice vectors
    - ``site.coords`` : Cartesian position of each site
    - ``site.species`` : Species dict of each site

    Disordered (partially occupied) sites are supported: each
    ``(element, occupancy)`` entry of a site's species dict becomes a
    co-located atom with that element's integral Z and the fractional
    occupancy stored in ``crystal.occupancies``.

    1. **Validate input** --
       Check input type.
    2. **Extract data** --
       Lattice matrix plus, per site, one atom per species-dict entry
       with its occupancy.
    3. **Convert** --
       Create XYZData and delegate to
       :func:`xyz_to_crystal`, then attach the occupancy column.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> import rheedium as rh
    >>> struct = Structure.from_file("POSCAR")
    >>> crystal = rh.inout.from_pymatgen(struct)
    >>> crystal.cell_lengths
    Array([5.43, 5.43, 5.43], dtype=float64)
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"Expected pymatgen Structure, got {type(structure).__name__}."
        )

    lattice: Float[Array, "3 3"] = jnp.asarray(
        structure.lattice.matrix, dtype=jnp.float64
    )

    positions_list: list[NDArray] = []
    atomic_numbers_list: list[int] = []
    occupancies_list: list[float] = []
    for site in structure:
        site_coords: NDArray = np.asarray(site.coords, dtype=np.float64)
        for element, occupancy in site.species.items():
            positions_list.append(site_coords)
            atomic_numbers_list.append(int(element.Z))
            occupancies_list.append(float(occupancy))

    positions: Float[Array, "N 3"] = jnp.asarray(
        np.stack(positions_list, axis=0), dtype=jnp.float64
    )
    atomic_numbers: Int[Array, "N"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )
    occupancies: Float[Array, "N"] = jnp.array(
        occupancies_list, dtype=jnp.float64
    )

    xyz_data: XYZData = create_xyz_data(
        positions=positions,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
    )

    crystal: CrystalStructure = xyz_to_crystal(xyz_data)
    return _with_occupancies(crystal, occupancies)


@jaxtyped(typechecker=beartype)
def to_pymatgen(crystal: CrystalStructure) -> Structure:
    """Convert CrystalStructure to pymatgen Structure.

    Creates a pymatgen Structure object from a rheedium CrystalStructure,
    preserving lattice, positions, and atomic species.

    :see: :class:`~.test_interop.TestPymatgenInterop`

    Parameters
    ----------
    crystal : CrystalStructure
        rheedium crystal structure to convert.

    Returns
    -------
    structure : pymatgen.core.Structure
        Equivalent pymatgen Structure object.

    Notes
    -----
    The created Structure uses:

    - Lattice from cell_lengths and cell_angles (reconstructed via
      build_cell_vectors)
    - Fractional coordinates from frac_positions
    - Element species from atomic numbers, with per-site occupancies
      mapped to pymatgen species dictionaries (``{symbol: occupancy}``)
      so partially occupied sites survive the conversion

    1. **Reconstruct cell** --
       Build cell vectors from lengths and angles.
    2. **Map species** --
       Convert atomic numbers to element symbols; when the crystal
       carries occupancies, each site becomes a ``{symbol: occupancy}``
       species dict (pymatgen's native partial-occupancy form).
    3. **Create Structure** --
       pymatgen Structure from lattice, species, and
       fractional coordinates.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> struct = rh.inout.to_pymatgen(crystal)
    >>> struct.to("POSCAR", "output_POSCAR")
    """
    cell: Float[Array, "3 3"] = build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )

    lattice_pmg: Lattice = Lattice(np.asarray(cell))

    atomic_numbers: Int[NDArray, "N"] = np.asarray(
        crystal.frac_positions[:, 3], dtype=int
    )
    frac_coords: Float[NDArray, "N 3"] = np.asarray(
        crystal.frac_positions[:, :3]
    )

    symbols: list[str] = [_Z_TO_SYMBOL.get(z, f"X{z}") for z in atomic_numbers]
    if crystal.occupancies is None:
        species: list[str] | list[dict[str, float]] = symbols
    else:
        occupancies_np: Float[NDArray, "N"] = np.asarray(
            crystal.occupancies, dtype=np.float64
        )
        species = [
            {symbol: float(occupancy)}
            for symbol, occupancy in zip(symbols, occupancies_np, strict=True)
        ]

    return Structure(lattice_pmg, species, frac_coords)


__all__: list[str] = [
    "from_ase",
    "from_pymatgen",
    "to_ase",
    "to_pymatgen",
]
