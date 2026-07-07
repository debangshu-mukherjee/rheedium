"""Pre-parameterized differentiable surface model library.

Extended Summary
----------------
Provides factory functions that return ``CrystalStructure`` slabs for
common experimentally studied surfaces. Each reconstructed surface is
built from the true ``(hkl)`` primitive mesh
(:func:`~rheedium.procs.create_surface_slab`), expanded to the
reconstruction supercell
(:func:`~rheedium.procs.apply_surface_reconstruction`),
and decorated with adatoms placed by *height above the top atomic layer*
(:func:`place_adatoms`) or by displacing existing surface atoms. All bulk
lattice constants remain overridable for thermal-expansion studies.

Routine Listings
----------------
:func:`place_adatoms`
    Append adatoms at a fixed height above the slab's top atomic layer.
:func:`si111_1x1`
    Bulk-terminated Si(111) surface slab.
:func:`si111_7x7`
    Si(111)-7x7 simplified adatoms-only DAS reconstruction.
:func:`si100_2x1`
    Si(100)-2x1 symmetric dimer row reconstruction.
:func:`gaas001_2x4`
    GaAs(001)-2x4 beta2-like As-dimer reconstruction.
:func:`mgo001_bulk_terminated`
    Bulk-terminated MgO(001) rocksalt surface.
:func:`srtio3_001_2x1`
    SrTiO3(001)-2x1 double-layer TiO2 (or SrO) reconstruction.

Notes
-----
Each library function returns a ``CrystalStructure`` that can be
passed directly to ``ewald_simulator`` or ``multislice_simulator``.
``reorient_to_zone_axis`` uses concrete Miller indices, so these builders
run eagerly (they are not traced); lattice constants remain plain floats.

R5 return type: every public library function is a structure builder and
returns ``CrystalStructure``. Statistical populations and sub-coherence bind
closures live in the dedicated producer/modifier modules.
"""

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors

from .surface_builder import (
    apply_surface_reconstruction,
    create_surface_slab,
)

# -------------------------------------------------------------------
# Crystallographic data: fractional coordinates and atomic numbers
# for each bulk unit cell.
# -------------------------------------------------------------------

# Diamond cubic Si (Fd-3m, #227): 8 atoms per cell, Z=14
_SI_DIAMOND_FRAC: Float[Array, "8 3"] = jnp.array(
    [
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
        [0.250, 0.250, 0.250],
        [0.750, 0.750, 0.250],
        [0.750, 0.250, 0.750],
        [0.250, 0.750, 0.750],
    ]
)
_SI_DIAMOND_Z: Float[Array, "8"] = jnp.full(8, 14.0)

# Zincblende GaAs (F-43m, #216): 8-atom conventional cell (4 Ga + 4 As)
_GAAS_ZB_FRAC: Float[Array, "8 3"] = jnp.array(
    [
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
        [0.250, 0.250, 0.250],
        [0.750, 0.750, 0.250],
        [0.750, 0.250, 0.750],
        [0.250, 0.750, 0.750],
    ]
)
_GAAS_ZB_Z: Float[Array, "8"] = jnp.array(
    [31.0, 31.0, 31.0, 31.0, 33.0, 33.0, 33.0, 33.0]
)

# Rocksalt MgO (Fm-3m, #225): 8 atoms (4 Mg + 4 O)
_MGO_RS_FRAC: Float[Array, "8 3"] = jnp.array(
    [
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
        [0.500, 0.000, 0.000],
        [0.000, 0.500, 0.000],
        [0.000, 0.000, 0.500],
        [0.500, 0.500, 0.500],
    ]
)
_MGO_RS_Z: Float[Array, "8"] = jnp.array(
    [12.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0, 8.0]
)

# Perovskite SrTiO3 (Pm-3m, #221): 5 atoms (1 Sr + 1 Ti + 3 O)
_STO_PV_FRAC: Float[Array, "5 3"] = jnp.array(
    [
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.500],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
    ]
)
_STO_PV_Z: Float[Array, "5"] = jnp.array([38.0, 22.0, 8.0, 8.0, 8.0])

# Si(111)-7x7 simplified DAS adatom in-plane positions (12 per cell) in
# fractional coordinates of the 7x7 surface supercell. Any placement with
# 7x7 periodicity, 12 adatoms/cell, and local rest-atom spacing is
# acceptable for the documented "adatoms-only simplified DAS" model.
_SI_7X7_ADATOM_XY: Float[Array, "12 2"] = np.array(
    [
        [1.0 / 7.0, 1.0 / 7.0],
        [3.0 / 7.0, 1.0 / 7.0],
        [5.0 / 7.0, 1.0 / 7.0],
        [0.0 / 7.0, 3.0 / 7.0],
        [2.0 / 7.0, 3.0 / 7.0],
        [4.0 / 7.0, 3.0 / 7.0],
        [1.0 / 7.0, 5.0 / 7.0],
        [3.0 / 7.0, 5.0 / 7.0],
        [5.0 / 7.0, 5.0 / 7.0],
        [0.0 / 7.0, 0.0 / 7.0],
        [2.0 / 7.0, 5.0 / 7.0],
        [4.0 / 7.0, 1.0 / 7.0],
    ]
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _build_bulk_crystal(
    frac_coords: Float[Array, "N 3"],
    atomic_numbers: Float[Array, "N"],
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
    alpha: scalar_float,
    beta: scalar_float,
    gamma: scalar_float,
) -> CrystalStructure:
    """Build a CrystalStructure from fractional coords and cell params.

    Parameters
    ----------
    frac_coords : Float[Array, "N 3"]
        Fractional coordinates of atoms in the unit cell.
    atomic_numbers : Float[Array, "N"]
        Atomic number for each atom.
    a, b, c : scalar_float
        Lattice parameters in Angstroms.
    alpha, beta, gamma : scalar_float
        Lattice angles in degrees.

    Returns
    -------
    crystal : CrystalStructure
        Bulk crystal structure.
    """
    frac_coords = np.asarray(frac_coords, dtype=np.float64)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.float64)
    cell_vecs: np.ndarray = np.asarray(
        build_cell_vectors(a, b, c, alpha, beta, gamma), dtype=np.float64
    )
    cart_coords: np.ndarray = frac_coords @ cell_vecs
    return create_crystal_structure(
        frac_positions=jnp.asarray(
            np.column_stack([frac_coords, atomic_numbers])
        ),
        cart_positions=jnp.asarray(
            np.column_stack([cart_coords, atomic_numbers])
        ),
        cell_lengths=jnp.asarray([a, b, c], dtype=jnp.float64),
        cell_angles=jnp.asarray([alpha, beta, gamma], dtype=jnp.float64),
    )


@jaxtyped(typechecker=beartype)
def place_adatoms(
    slab: CrystalStructure,
    frac_xy_in_supercell: Float[Array, "N_ad 2"],
    height_above_top_ang: scalar_float,
    z_number: scalar_float,
) -> CrystalStructure:
    """Append adatoms at a height above the slab's top atomic layer.

    :see: :class:`~.test_library.TestPlaceAdatoms`

    Parameters
    ----------
    slab : CrystalStructure
        Surface slab to decorate.
    frac_xy_in_supercell : Float[Array, "N_ad 2"]
        In-plane fractional ``(u, v)`` coordinates of each adatom in the
        slab supercell.
    height_above_top_ang : scalar_float
        Height in Angstroms placed above the current topmost atom (not a
        cell fraction), so the adatom never floats an arbitrary distance
        above the surface.
    z_number : scalar_float
        Integral atomic number of the adatom species.

    Returns
    -------
    decorated : CrystalStructure
        Slab with the adatoms appended at full occupancy.

    Notes
    -----
    1. The in-plane fractional coordinates are mapped to Cartesian with
       the slab's in-plane cell vectors.
    2. The adatom ``z`` is the current top-atom Cartesian ``z`` plus
       ``height_above_top_ang``.
    3. Fractional coordinates are recomputed so the frame contract
       ``cart = frac @ build_cell_vectors`` holds for every atom.
    """
    cell: np.ndarray = np.asarray(
        build_cell_vectors(*slab.cell_lengths, *slab.cell_angles),
        dtype=np.float64,
    )
    slab_cart: np.ndarray = np.asarray(slab.cart_positions, dtype=np.float64)
    slab_frac: np.ndarray = np.asarray(slab.frac_positions, dtype=np.float64)
    top_z: float = float(slab_cart[:, 2].max())
    fxy: np.ndarray = np.asarray(frac_xy_in_supercell, dtype=np.float64)
    n_ad: int = fxy.shape[0]

    ad_cart: np.ndarray = (
        fxy[:, 0:1] * cell[0][None, :] + fxy[:, 1:2] * cell[1][None, :]
    )
    ad_cart[:, 2] = top_z + float(height_above_top_ang)
    ad_frac: np.ndarray = ad_cart @ np.linalg.inv(cell)
    znum: np.ndarray = np.full((n_ad, 1), float(z_number))

    new_cart: np.ndarray = np.concatenate(
        [slab_cart, np.column_stack([ad_cart, znum])], axis=0
    )
    new_frac: np.ndarray = np.concatenate(
        [slab_frac, np.column_stack([ad_frac, znum])], axis=0
    )
    slab_occ: np.ndarray = (
        np.ones(slab_cart.shape[0], dtype=np.float64)
        if slab.occupancies is None
        else np.asarray(slab.occupancies, dtype=np.float64)
    )
    new_occ: np.ndarray = np.concatenate(
        [slab_occ, np.ones(n_ad, dtype=np.float64)]
    )
    return create_crystal_structure(
        frac_positions=jnp.asarray(new_frac),
        cart_positions=jnp.asarray(new_cart),
        cell_lengths=jnp.asarray(slab.cell_lengths),
        cell_angles=jnp.asarray(slab.cell_angles),
        occupancies=jnp.asarray(new_occ),
    )


def _symmetric_dimer_displacements(
    expanded: CrystalStructure,
    surface_depth_ang: float,
    bond_ang: float,
) -> np.ndarray:
    """Displacements pulling the top two surface atoms into one dimer.

    The two top-layer atoms (in the deterministic (z, y, x) order used by
    :func:`apply_surface_reconstruction`) are moved symmetrically toward
    each other along their connecting line until their separation equals
    ``bond_ang`` (a documented symmetric-dimer simplification).
    """
    cart: np.ndarray = np.asarray(expanded.cart_positions, dtype=np.float64)
    order: np.ndarray = np.lexsort((cart[:, 0], cart[:, 1], cart[:, 2]))
    cart = cart[order]
    z_max: float = float(cart[:, 2].max())
    surface_idx: np.ndarray = np.nonzero(
        cart[:, 2] >= z_max - surface_depth_ang
    )[0]
    if surface_idx.shape[0] != 2:
        raise ValueError(
            "symmetric dimer expects exactly two surface atoms, found "
            f"{surface_idx.shape[0]}"
        )
    p_a: np.ndarray = cart[surface_idx[0], :3]
    p_b: np.ndarray = cart[surface_idx[1], :3]
    sep: np.ndarray = p_b - p_a
    dist: float = float(np.linalg.norm(sep))
    unit: np.ndarray = sep / dist
    shift: float = 0.5 * (dist - bond_ang)
    displacements: np.ndarray = np.zeros((2, 3), dtype=np.float64)
    displacements[0] = unit * shift
    displacements[1] = -unit * shift
    return displacements


# -------------------------------------------------------------------
# Library functions
# -------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def si111_1x1(
    a_lattice_angstrom: scalar_float = 5.431,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct bulk-terminated Si(111) surface slab.

    :see: :class:`~.test_library.TestSi111_1x1`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.431 Angstroms at 300 K.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        Si(111)-1x1 primitive hexagonal surface mesh (|a| = |b| = a/sqrt2,
        gamma = 120 degrees).

    Notes
    -----
    1. Build diamond cubic Si unit cell with 8 atoms per cell.
    2. Cut the primitive ``(111)`` surface mesh with the requested depth.
    3. Add the vacuum gap above the surface.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        _SI_DIAMOND_FRAC,
        _SI_DIAMOND_Z,
        a_lattice_angstrom,
        a_lattice_angstrom,
        a_lattice_angstrom,
        90.0,
        90.0,
        90.0,
    )
    return create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=jnp.asarray([1, 1, 1], dtype=jnp.int32),
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )


@jaxtyped(typechecker=beartype)
def si111_7x7(
    a_lattice_angstrom: scalar_float = 5.431,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct Si(111)-7x7 simplified adatoms-only DAS slab.

    :see: :class:`~.test_library.TestSi111_7x7`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.431 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        Si(111)-7x7 supercell of the primitive hexagonal Si(111) mesh with
        12 Si adatoms per cell placed 1.5 Angstroms above the top layer.

    Notes
    -----
    1. Build the primitive Si(111)-1x1 slab.
    2. Expand it to the 7x7 supercell via
       :func:`~rheedium.procs.apply_surface_reconstruction`.
    3. Place 12 Si adatoms per 7x7 cell 1.5 Angstroms above the top layer.

    This is a documented *adatoms-only simplified DAS* model: it captures
    the 7x7 periodicity and the 12-adatom count of the true
    dimer-adatom-stacking-fault structure, but does not add the corner
    holes, dimer walls, or stacking fault of the full DAS model.
    """
    base: CrystalStructure = si111_1x1(
        a_lattice_angstrom, slab_depth_angstrom, vacuum_gap_angstrom
    )
    super7: CrystalStructure = apply_surface_reconstruction(
        slab=base,
        reconstruction_matrix=jnp.asarray([[7, 0], [0, 7]], dtype=jnp.int32),
        surface_layer_depth_angstrom=0.0,
    )
    return place_adatoms(super7, jnp.asarray(_SI_7X7_ADATOM_XY), 1.5, 14.0)


@jaxtyped(typechecker=beartype)
def si100_2x1(
    a_lattice_angstrom: scalar_float = 5.431,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct Si(100)-2x1 symmetric dimer row slab.

    :see: :class:`~.test_library.TestSi100_2x1`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.431 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        Si(100)-2x1 slab whose two top-layer atoms per cell are displaced
        toward each other into a 2.35 Angstrom symmetric dimer.

    Notes
    -----
    1. Build the primitive Si(100)-1x1 slab.
    2. Expand to the 2x1 supercell.
    3. Displace the two top-layer atoms toward each other to a 2.35
       Angstrom bond (documented symmetric-dimer simplification: real
       Si(100) dimers also buckle, which this model omits).
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        _SI_DIAMOND_FRAC,
        _SI_DIAMOND_Z,
        a_lattice_angstrom,
        a_lattice_angstrom,
        a_lattice_angstrom,
        90.0,
        90.0,
        90.0,
    )
    base: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=jnp.asarray([1, 0, 0], dtype=jnp.int32),
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )
    matrix: Array = jnp.asarray([[2, 0], [0, 1]], dtype=jnp.int32)
    surface_depth: float = 0.6
    expanded: CrystalStructure = apply_surface_reconstruction(
        slab=base,
        reconstruction_matrix=matrix,
        surface_layer_depth_angstrom=surface_depth,
    )
    displacements: np.ndarray = _symmetric_dimer_displacements(
        expanded, surface_depth, 2.35
    )
    return apply_surface_reconstruction(
        slab=base,
        reconstruction_matrix=matrix,
        surface_layer_depth_angstrom=surface_depth,
        atomic_displacements=jnp.asarray(displacements),
    )


@jaxtyped(typechecker=beartype)
def gaas001_2x4(
    a_lattice_angstrom: scalar_float = 5.6533,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct GaAs(001)-2x4 beta2-like As-dimer surface slab.

    :see: :class:`~.test_library.TestGaAs001_2x4`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.6533 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        GaAs(001)-2x4 supercell with two As dimers (bond 2.5 Angstroms)
        per cell placed 1.4 Angstroms above the complete layer, and a
        documented missing-dimer trench.

    Notes
    -----
    1. Build the fixed (unreconstructed) GaAs(001) slab (8-atom bulk cell;
       (001) layer spacing a/4 = 1.4133 Angstroms, alternating Ga/As).
    2. Expand to the 2x4 supercell.
    3. Place two As dimers (4 As atoms, 2.5 Angstrom bonds) 1.4 Angstroms
       above the top layer, leaving the third dimer row empty as the
       documented missing-dimer trench (beta2-like simplification).
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        _GAAS_ZB_FRAC,
        _GAAS_ZB_Z,
        a_lattice_angstrom,
        a_lattice_angstrom,
        a_lattice_angstrom,
        90.0,
        90.0,
        90.0,
    )
    base: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )
    super24: CrystalStructure = apply_surface_reconstruction(
        slab=base,
        reconstruction_matrix=jnp.asarray([[2, 0], [0, 4]], dtype=jnp.int32),
        surface_layer_depth_angstrom=0.0,
    )
    a_super: float = float(super24.cell_lengths[0])
    half_bond_frac: float = 0.5 * 2.5 / a_super
    # Two As dimers in adjacent quarter-rows, leaving the v=0 and v=0.75
    # rows empty as the documented missing-dimer trench. The asymmetric
    # occupancy breaks both the 1/2- (a) and 1/4-order (b) periodicities.
    dimer_rows: list[float] = [0.25, 0.5]
    as_xy: list[list[float]] = []
    for v in dimer_rows:
        as_xy.append([0.5 - half_bond_frac, v])
        as_xy.append([0.5 + half_bond_frac, v])
    return place_adatoms(
        super24, jnp.asarray(as_xy, dtype=jnp.float64), 1.4, 33.0
    )


@jaxtyped(typechecker=beartype)
def mgo001_bulk_terminated(
    a_lattice_angstrom: scalar_float = 4.211,
    slab_depth_angstrom: scalar_float = 25.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct bulk-terminated MgO(001) surface slab.

    :see: :class:`~.test_library.TestMgO001BulkTerminated`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 4.211 Angstroms at 300 K.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 25.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        MgO(001) slab with rocksalt structure and a 1x1 termination.

    Notes
    -----
    1. Build rocksalt MgO unit cell (8 atoms: 4 Mg + 4 O), Fm-3m (#225).
    2. Cut the primitive ``(001)`` mesh.
    3. Add the vacuum gap. No reconstruction is applied for 1x1.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        _MGO_RS_FRAC,
        _MGO_RS_Z,
        a_lattice_angstrom,
        a_lattice_angstrom,
        a_lattice_angstrom,
        90.0,
        90.0,
        90.0,
    )
    return create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )


@jaxtyped(typechecker=beartype)
def srtio3_001_2x1(
    a_lattice_angstrom: scalar_float = 3.905,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
    termination: Literal["TiO2", "SrO"] = "TiO2",
) -> CrystalStructure:
    """Construct SrTiO3(001)-2x1 double-layer TiO2 (or SrO) slab.

    :see: :class:`~.test_library.TestSrTiO3_001_2x1`

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 3.905 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.
    termination : {"TiO2", "SrO"}
        Which atomic plane terminates the surface. The bulk cut origin is
        shifted by half a layer for the ``"SrO"`` termination. Default
        ``"TiO2"``.

    Returns
    -------
    slab : CrystalStructure
        SrTiO3(001)-2x1 supercell with one extra TiO2 unit (1 Ti + 2 O)
        per cell added on top (double-layer TiO2 simplified model).

    Notes
    -----
    1. Build the perovskite SrTiO3 unit cell (5 atoms), Pm-3m (#221),
       shifting the cut origin to expose the requested termination.
    2. Cut the primitive ``(001)`` mesh and expand to the 2x1 supercell.
    3. Add one extra TiO2 formula unit (1 Ti + 2 O) per 2x1 cell on top at
       ~1.95 Angstroms above the surface (documented simplified
       double-layer TiO2 model; the true reconstruction also relaxes the
       subsurface layers, which this model omits).
    """
    if termination not in ("TiO2", "SrO"):
        raise ValueError("termination must be 'TiO2' or 'SrO'")
    z_shift: float = 0.5 if termination == "SrO" else 0.0
    shifted_frac: np.ndarray = np.asarray(_STO_PV_FRAC).copy()
    shifted_frac[:, 2] = np.mod(shifted_frac[:, 2] + z_shift, 1.0)
    bulk: CrystalStructure = _build_bulk_crystal(
        jnp.asarray(shifted_frac),
        _STO_PV_Z,
        a_lattice_angstrom,
        a_lattice_angstrom,
        a_lattice_angstrom,
        90.0,
        90.0,
        90.0,
    )
    base: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )
    super2: CrystalStructure = apply_surface_reconstruction(
        slab=base,
        reconstruction_matrix=jnp.asarray([[2, 0], [0, 1]], dtype=jnp.int32),
        surface_layer_depth_angstrom=0.0,
    )
    height: float = 0.5 * a_lattice_angstrom
    decorated: CrystalStructure = place_adatoms(
        super2, jnp.asarray([[0.25, 0.5]], dtype=jnp.float64), height, 22.0
    )
    return place_adatoms(
        decorated,
        jnp.asarray([[0.0, 0.5], [0.5, 0.5]], dtype=jnp.float64),
        height,
        8.0,
    )


__all__: list[str] = [
    "gaas001_2x4",
    "mgo001_bulk_terminated",
    "place_adatoms",
    "si100_2x1",
    "si111_1x1",
    "si111_7x7",
    "srtio3_001_2x1",
]
