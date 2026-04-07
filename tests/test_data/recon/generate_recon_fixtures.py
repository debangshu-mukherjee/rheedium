# ruff: noqa: E402
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "rheedium",
#   "ase>=3.26.0",
#   "beartype",
#   "jaxtyping>=0.3.0",
#   "jax>=0.9.2",
#   "matplotlib>=3.10.0",
#   "numpy>=2.2.1",
#   "pymatgen>=2025.10.7",
#   "scipy>=1.14.1",
#   "tifffile>=2026.3.3",
# ]
# ///
"""Generate pre-built slab fixtures for recon tests.

Run once:
    uv run tests/test_data/recon/generate_recon_fixtures.py

Produces .npz files in tests/test_data/recon/ that tests load
instead of building slabs at test time.
"""

import gc
import os
import sys
from collections.abc import Callable
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_ROOT: Path = REPO_ROOT / "src"
OUT_DIR: Path = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_ROOT))

from rheedium.recon import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
    gaas001_2x4,
    mgo001_bulk_terminated,
    si100_2x1,
    si111_1x1,
    si111_7x7,
    srtio3_001_2x1,
)
from rheedium.types import CrystalStructure, create_crystal_structure
from rheedium.ucell import build_cell_vectors


def _save_crystal(path: str, crystal: CrystalStructure) -> None:
    """Save a CrystalStructure as .npz."""
    np.savez(
        path,
        frac_positions=np.array(crystal.frac_positions),
        cart_positions=np.array(crystal.cart_positions),
        cell_lengths=np.array(crystal.cell_lengths),
        cell_angles=np.array(crystal.cell_angles),
    )


def _write_fixture(name: str, crystal: CrystalStructure) -> None:
    """Persist one fixture and clear JAX caches between generations."""
    path: Path = OUT_DIR / name
    print(f"Writing {path.name}", flush=True)
    _save_crystal(str(path), crystal)
    jax.clear_caches()
    gc.collect()


def _make_cubic_crystal(a: float = 4.0) -> CrystalStructure:
    """Simple FCC-like cubic crystal."""
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    z_nums = np.full(4, 14.0)
    cell_vecs = np.array(build_cell_vectors(a, a, a, 90.0, 90.0, 90.0))
    cart_coords = frac_coords @ cell_vecs
    return create_crystal_structure(
        frac_positions=jnp.array(np.column_stack([frac_coords, z_nums])),
        cart_positions=jnp.array(np.column_stack([cart_coords, z_nums])),
        cell_lengths=jnp.array([a, a, a]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


def _make_mgo_crystal() -> CrystalStructure:
    """MgO rocksalt crystal."""
    a = 4.211
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    z_nums = np.array([12.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0, 8.0])
    cell_vecs = np.array(build_cell_vectors(a, a, a, 90.0, 90.0, 90.0))
    cart_coords = frac_coords @ cell_vecs
    return create_crystal_structure(
        frac_positions=jnp.array(np.column_stack([frac_coords, z_nums])),
        cart_positions=jnp.array(np.column_stack([cart_coords, z_nums])),
        cell_lengths=jnp.array([a, a, a]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


def _generate_surface_slab_fixtures(
    crystal: CrystalStructure,
    mgo_crystal: CrystalStructure,
) -> CrystalStructure:
    """Generate the base slab fixtures from aligned spec arrays."""
    slab_names: tuple[str, ...] = (
        "slab_001.npz",
        "slab_110.npz",
        "slab_111.npz",
        "mgo_slab.npz",
        "thin_slab.npz",
    )
    base_crystals: tuple[CrystalStructure, ...] = (
        crystal,
        crystal,
        crystal,
        mgo_crystal,
        crystal,
    )
    miller_specs: np.ndarray = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    thickness_specs: np.ndarray = np.array(
        [10.0, 10.0, 10.0, 20.0, 5.0],
        dtype=np.float64,
    )
    vacuum_specs: np.ndarray = np.full(len(slab_names), 15.0, dtype=np.float64)

    slab_001: CrystalStructure | None = None
    for name, base_crystal, miller, thickness, vacuum in zip(
        slab_names,
        base_crystals,
        miller_specs,
        thickness_specs,
        vacuum_specs,
        strict=True,
    ):
        slab: CrystalStructure = create_surface_slab(
            base_crystal,
            jnp.asarray(miller, dtype=jnp.int32),
            jnp.asarray(thickness, dtype=jnp.float64),
            jnp.asarray(vacuum, dtype=jnp.float64),
        )
        _write_fixture(name, slab)
        if name == "slab_001.npz":
            slab_001 = slab

    assert slab_001 is not None
    return slab_001


def _generate_reconstruction_fixtures(slab_001: CrystalStructure) -> None:
    """Generate reconstruction fixtures from a declarative spec list."""
    reconstruction_specs: tuple[
        tuple[str, np.ndarray, float, np.ndarray],
        ...,
    ] = (
        (
            "recon_2x2.npz",
            np.array([[2, 0], [0, 2]], dtype=np.int32),
            3.0,
            np.zeros((1, 3), dtype=np.float64),
        ),
        (
            "recon_no_disp.npz",
            np.array([[1, 0], [0, 1]], dtype=np.int32),
            3.0,
            np.zeros((1, 3), dtype=np.float64),
        ),
        (
            "recon_with_disp.npz",
            np.array([[1, 0], [0, 1]], dtype=np.int32),
            3.0,
            np.array([[0.5, 0.0, 0.0]], dtype=np.float64),
        ),
    )

    for (
        name,
        reconstruction_matrix,
        surface_depth,
        displacements,
    ) in reconstruction_specs:
        reconstructed: CrystalStructure = apply_surface_reconstruction(
            slab=slab_001,
            reconstruction_matrix=jnp.asarray(
                reconstruction_matrix, dtype=jnp.int32
            ),
            surface_layer_depth_angstrom=jnp.asarray(
                surface_depth, dtype=jnp.float64
            ),
            atomic_displacements=jnp.asarray(displacements, dtype=jnp.float64),
        )
        _write_fixture(name, reconstructed)


def _generate_adsorbate_fixtures(slab_001: CrystalStructure) -> None:
    """Generate adsorbate fixtures from a declarative spec list."""
    adsorbate_specs: tuple[
        tuple[str, np.ndarray, np.ndarray, float],
        ...,
    ] = (
        (
            "ads_full.npz",
            np.array([[0.5, 0.5, 0.95]], dtype=np.float64),
            np.array([8.0], dtype=np.float64),
            1.0,
        ),
        (
            "ads_half.npz",
            np.array([[0.5, 0.5, 0.95]], dtype=np.float64),
            np.array([8.0], dtype=np.float64),
            0.5,
        ),
        (
            "ads_multi.npz",
            np.array(
                [[0.25, 0.25, 0.95], [0.75, 0.75, 0.95]],
                dtype=np.float64,
            ),
            np.array([8.0, 8.0], dtype=np.float64),
            1.0,
        ),
        (
            "ads_zero.npz",
            np.array([[0.5, 0.5, 0.95]], dtype=np.float64),
            np.array([14.0], dtype=np.float64),
            0.0,
        ),
    )

    for name, positions, atomic_numbers, coverage in adsorbate_specs:
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab_001,
            adsorbate_positions_fractional=jnp.asarray(
                positions, dtype=jnp.float64
            ),
            adsorbate_atomic_numbers=jnp.asarray(
                atomic_numbers, dtype=jnp.float64
            ),
            coverage_fraction=jnp.asarray(coverage, dtype=jnp.float64),
        )
        _write_fixture(name, decorated)


def _generate_library_fixtures() -> None:
    """Generate library fixtures from a factory table."""
    library_specs: tuple[tuple[str, Callable[[], CrystalStructure]], ...] = (
        ("si111_1x1.npz", si111_1x1),
        (
            "si111_1x1_custom.npz",
            lambda: si111_1x1(a_lattice_angstrom=6.0),
        ),
        ("si111_7x7.npz", si111_7x7),
        ("si100_2x1.npz", si100_2x1),
        ("gaas001_2x4.npz", gaas001_2x4),
        ("mgo001.npz", mgo001_bulk_terminated),
        ("srtio3_001.npz", srtio3_001_2x1),
    )

    for name, builder in library_specs:
        _write_fixture(name, builder())


def main() -> None:
    """Generate all recon test fixtures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crystal = _make_cubic_crystal()
    mgo_crystal = _make_mgo_crystal()

    _write_fixture("cubic_crystal.npz", crystal)
    slab_001: CrystalStructure = _generate_surface_slab_fixtures(
        crystal,
        mgo_crystal,
    )
    _generate_reconstruction_fixtures(slab_001)
    _generate_adsorbate_fixtures(slab_001)
    _generate_library_fixtures()

    print(f"Generated {len(os.listdir(OUT_DIR))} fixtures in {OUT_DIR}")


if __name__ == "__main__":
    main()
