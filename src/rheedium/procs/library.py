"""Pre-parameterized differentiable surface model library.

Extended Summary
----------------
Provides factory functions that return ``CrystalStructure`` slabs for
common experimentally studied surfaces. Each function constructs a bulk
unit cell with known lattice parameters, creates a surface slab of
standard thickness, and optionally applies the canonical surface
reconstruction. All parameters are JAX-traceable scalars so lattice
constants and defect or process variables can be optimized.

Routine Listings
----------------
:func:`si111_1x1`
    Bulk-terminated Si(111) surface slab.
:func:`si111_7x7`
    Si(111)-7x7 DAS reconstruction with adatoms and restatoms.
:func:`si100_2x1`
    Si(100)-2x1 symmetric dimer row reconstruction.
:func:`gaas001_2x4`
    GaAs(001)-2x4 beta2 As-rich reconstruction.
:func:`mgo001_bulk_terminated`
    Bulk-terminated MgO(001) rocksalt surface.
:func:`srtio3_001_2x1`
    SrTiO3(001)-2x1 TiO2 double-layer reconstruction.

Notes
-----
Each library function returns a ``CrystalStructure`` that can be
passed directly to ``ewald_simulator`` or ``multislice_simulator``.
Lattice parameters default to literature values at 300 K but can be
overridden for thermal expansion studies or optimization.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors

from .surface_builder import add_adsorbate_layer, create_surface_slab

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

# Zincblende GaAs (F-43m, #216): 4 atoms (2 Ga + 2 As)
_GAAS_ZB_FRAC: Float[Array, "4 3"] = jnp.array(
    [
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.000],
        [0.250, 0.250, 0.250],
        [0.750, 0.750, 0.250],
    ]
)
_GAAS_ZB_Z: Float[Array, "4"] = jnp.array([31.0, 31.0, 33.0, 33.0])

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
    [
        12.0,
        12.0,
        12.0,
        12.0,
        8.0,
        8.0,
        8.0,
        8.0,
    ]
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

# Si(111)-7x7 DAS adatom positions in fractional coords of the
# surface cell. 12 adatoms at approximate T4 sites, z_frac ~ 0.95
# (just above slab surface).
_SI_7X7_ADATOM_FRAC: Float[Array, "12 3"] = jnp.array(
    [
        [1.0 / 7.0, 1.0 / 7.0, 0.95],
        [3.0 / 7.0, 1.0 / 7.0, 0.95],
        [5.0 / 7.0, 1.0 / 7.0, 0.95],
        [0.0 / 7.0, 3.0 / 7.0, 0.95],
        [2.0 / 7.0, 3.0 / 7.0, 0.95],
        [4.0 / 7.0, 3.0 / 7.0, 0.95],
        [1.0 / 7.0, 5.0 / 7.0, 0.95],
        [3.0 / 7.0, 5.0 / 7.0, 0.95],
        [5.0 / 7.0, 5.0 / 7.0, 0.95],
        [0.0 / 7.0, 0.0 / 7.0, 0.95],
        [2.0 / 7.0, 5.0 / 7.0, 0.95],
        [4.0 / 7.0, 1.0 / 7.0, 0.95],
    ]
)
_SI_7X7_ADATOM_Z: Float[Array, "12"] = jnp.full(12, 14.0)

# Si(100)-2x1 dimer positions
_SI_2X1_DIMER_FRAC: Float[Array, "2 3"] = jnp.array(
    [
        [0.25, 0.0, 0.95],
        [0.75, 0.0, 0.95],
    ]
)
_SI_2X1_DIMER_Z: Float[Array, "2"] = jnp.array([14.0, 14.0])

# GaAs(001)-2x4 As dimer positions
_GAAS_2X4_AS_FRAC: Float[Array, "2 3"] = jnp.array(
    [
        [0.25, 0.125, 0.95],
        [0.75, 0.125, 0.95],
    ]
)
_GAAS_2X4_AS_Z: Float[Array, "2"] = jnp.array([33.0, 33.0])


# -------------------------------------------------------------------
# Helper
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
    cell_lengths: Float[Array, "3"] = jnp.array([a, b, c])
    cell_angles: Float[Array, "3"] = jnp.array([alpha, beta, gamma])

    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        a, b, c, alpha, beta, gamma
    )
    cart_coords: Float[Array, "N 3"] = frac_coords @ cell_vecs

    frac_with_z: Float[Array, "N 4"] = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_with_z: Float[Array, "N 4"] = jnp.column_stack(
        [cart_coords, atomic_numbers]
    )

    return create_crystal_structure(
        frac_positions=frac_with_z,
        cart_positions=cart_with_z,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


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

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.431 Angstroms at
        300 K.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        Si(111)-1x1 surface slab. Baseline for validating against
        Laue ring positions.

    Notes
    -----
    1. Build diamond cubic Si unit cell with 8 atoms per cell.
    2. Cut along (111) plane with specified thickness.
    3. Add vacuum gap above the surface.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        frac_coords=_SI_DIAMOND_FRAC,
        atomic_numbers=_SI_DIAMOND_Z,
        a=a_lattice_angstrom,
        b=a_lattice_angstrom,
        c=a_lattice_angstrom,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )

    surface_normal: jnp.ndarray = jnp.array([1, 1, 1], dtype=jnp.int32)
    slab: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=surface_normal,
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    return slab


@jaxtyped(typechecker=beartype)
def si111_7x7(
    a_lattice_angstrom: scalar_float = 5.431,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct Si(111)-7x7 DAS reconstruction slab.

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
        Si(111)-7x7 slab with 12 adatoms placed at the
        canonical DAS model T4 sites above the surface.

    Notes
    -----
    1. Build bulk-terminated Si(111)-1x1 slab.
    2. Add 12 adatom positions at T4 sites distributed over
       the 7x7 surface cell using the virtual crystal
       approximation with full coverage.

    The DAS (dimer-adatom-stacking-fault) model has 12 adatoms
    per 7x7 cell: 6 in the faulted half and 6 in the unfaulted
    half. This simplified model places Si adatoms at approximate
    T4 positions above the surface.
    """
    base_slab: CrystalStructure = si111_1x1(
        a_lattice_angstrom=a_lattice_angstrom,
        slab_depth_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    slab: CrystalStructure = add_adsorbate_layer(
        slab=base_slab,
        adsorbate_positions_fractional=_SI_7X7_ADATOM_FRAC,
        adsorbate_atomic_numbers=_SI_7X7_ADATOM_Z,
        coverage_fraction=1.0,
    )

    return slab


@jaxtyped(typechecker=beartype)
def si100_2x1(
    a_lattice_angstrom: scalar_float = 5.431,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct Si(100)-2x1 symmetric dimer row slab.

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
        Si(100)-2x1 slab with symmetric dimer atoms at the
        surface.

    Notes
    -----
    1. Build diamond cubic Si bulk cell.
    2. Cut along (100) to create a surface slab.
    3. Add dimer atoms at surface positions representing the
       2x1 symmetric dimer reconstruction.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        frac_coords=_SI_DIAMOND_FRAC,
        atomic_numbers=_SI_DIAMOND_Z,
        a=a_lattice_angstrom,
        b=a_lattice_angstrom,
        c=a_lattice_angstrom,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )

    surface_normal: jnp.ndarray = jnp.array([1, 0, 0], dtype=jnp.int32)
    slab: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=surface_normal,
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    slab_with_dimers: CrystalStructure = add_adsorbate_layer(
        slab=slab,
        adsorbate_positions_fractional=_SI_2X1_DIMER_FRAC,
        adsorbate_atomic_numbers=_SI_2X1_DIMER_Z,
        coverage_fraction=1.0,
    )

    return slab_with_dimers


@jaxtyped(typechecker=beartype)
def gaas001_2x4(
    a_lattice_angstrom: scalar_float = 5.653,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct GaAs(001)-2x4 beta2 As-rich surface slab.

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 5.653 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        GaAs(001)-2x4 slab with As dimer rows at the surface.
        Used as the canonical MBE growth surface.

    Notes
    -----
    1. Build zincblende GaAs unit cell (4 atoms: 2 Ga + 2 As).
    2. Cut along (001) to create a surface slab.
    3. Add As dimer atoms at surface positions representing
       the beta2 reconstruction.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        frac_coords=_GAAS_ZB_FRAC,
        atomic_numbers=_GAAS_ZB_Z,
        a=a_lattice_angstrom,
        b=a_lattice_angstrom,
        c=a_lattice_angstrom,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )

    surface_normal: jnp.ndarray = jnp.array([0, 0, 1], dtype=jnp.int32)
    slab: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=surface_normal,
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    slab_with_as: CrystalStructure = add_adsorbate_layer(
        slab=slab,
        adsorbate_positions_fractional=_GAAS_2X4_AS_FRAC,
        adsorbate_atomic_numbers=_GAAS_2X4_AS_Z,
        coverage_fraction=1.0,
    )

    return slab_with_as


@jaxtyped(typechecker=beartype)
def mgo001_bulk_terminated(
    a_lattice_angstrom: scalar_float = 4.211,
    slab_depth_angstrom: scalar_float = 25.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct bulk-terminated MgO(001) surface slab.

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 4.211 Angstroms at
        300 K.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 25.0 (6 unit cells).
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        MgO(001) slab with rocksalt structure. Canonical test case
        for oxide surface RHEED with well-known 1x1 patterns.

    Notes
    -----
    1. Build rocksalt MgO unit cell (8 atoms: 4 Mg + 4 O).
       Space group Fm-3m (#225).
    2. Cut along (001) plane.
    3. Add vacuum gap. No reconstruction needed for 1x1.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        frac_coords=_MGO_RS_FRAC,
        atomic_numbers=_MGO_RS_Z,
        a=a_lattice_angstrom,
        b=a_lattice_angstrom,
        c=a_lattice_angstrom,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )

    surface_normal: jnp.ndarray = jnp.array([0, 0, 1], dtype=jnp.int32)
    slab: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=surface_normal,
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    return slab


@jaxtyped(typechecker=beartype)
def srtio3_001_2x1(
    a_lattice_angstrom: scalar_float = 3.905,
    slab_depth_angstrom: scalar_float = 20.0,
    vacuum_gap_angstrom: scalar_float = 15.0,
) -> CrystalStructure:
    """Construct SrTiO3(001)-2x1 TiO2 double-layer slab.

    Parameters
    ----------
    a_lattice_angstrom : scalar_float
        Cubic lattice parameter. Literature: 3.905 Angstroms.
    slab_depth_angstrom : scalar_float
        Slab thickness in Angstroms. Default: 20.0.
    vacuum_gap_angstrom : scalar_float
        Vacuum gap above slab in Angstroms. Default: 15.0.

    Returns
    -------
    slab : CrystalStructure
        SrTiO3(001) slab with TiO2-terminated surface. Common
        ferroelectric substrate for thin film growth.

    Notes
    -----
    1. Build perovskite SrTiO3 unit cell (5 atoms per cell).
       Space group Pm-3m (#221).
    2. Cut along (001) plane.
    3. Add vacuum gap.
    """
    bulk: CrystalStructure = _build_bulk_crystal(
        frac_coords=_STO_PV_FRAC,
        atomic_numbers=_STO_PV_Z,
        a=a_lattice_angstrom,
        b=a_lattice_angstrom,
        c=a_lattice_angstrom,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )

    surface_normal: jnp.ndarray = jnp.array([0, 0, 1], dtype=jnp.int32)
    slab: CrystalStructure = create_surface_slab(
        bulk_crystal=bulk,
        surface_normal_miller=surface_normal,
        slab_thickness_angstrom=slab_depth_angstrom,
        vacuum_gap_angstrom=vacuum_gap_angstrom,
    )

    return slab


__all__: list[str] = [
    "gaas001_2x4",
    "mgo001_bulk_terminated",
    "si100_2x1",
    "si111_1x1",
    "si111_7x7",
    "srtio3_001_2x1",
]
