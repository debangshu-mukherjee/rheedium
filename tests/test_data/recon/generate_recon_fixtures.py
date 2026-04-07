"""Generate pre-built slab fixtures for recon tests.

Run once:
    JAX_PLATFORMS=cpu python tests/test_data/generate_recon_fixtures.py

Produces .npz files in tests/test_data/recon/ that tests load
instead of building slabs at test time.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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


def main() -> None:
    """Generate all recon test fixtures."""
    out_dir: str = os.path.join(os.path.dirname(__file__), "recon")
    os.makedirs(out_dir, exist_ok=True)

    miller_001 = jnp.array([0, 0, 1], dtype=jnp.int32)
    miller_110 = jnp.array([1, 1, 0], dtype=jnp.int32)
    miller_111 = jnp.array([1, 1, 1], dtype=jnp.int32)

    crystal = _make_cubic_crystal()
    mgo_crystal = _make_mgo_crystal()

    _save_crystal(os.path.join(out_dir, "cubic_crystal.npz"), crystal)

    slab_001 = create_surface_slab(crystal, miller_001, 10.0, 15.0)
    _save_crystal(os.path.join(out_dir, "slab_001.npz"), slab_001)

    slab_110 = create_surface_slab(crystal, miller_110, 10.0, 15.0)
    _save_crystal(os.path.join(out_dir, "slab_110.npz"), slab_110)

    slab_111 = create_surface_slab(crystal, miller_111, 10.0, 15.0)
    _save_crystal(os.path.join(out_dir, "slab_111.npz"), slab_111)

    mgo_slab = create_surface_slab(mgo_crystal, miller_001, 20.0, 15.0)
    _save_crystal(os.path.join(out_dir, "mgo_slab.npz"), mgo_slab)

    thin_slab = create_surface_slab(crystal, miller_001, 5.0, 15.0)
    _save_crystal(os.path.join(out_dir, "thin_slab.npz"), thin_slab)

    recon_2x2 = apply_surface_reconstruction(
        slab=slab_001,
        reconstruction_matrix=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
        surface_layer_depth_angstrom=3.0,
        atomic_displacements=jnp.zeros((1, 3)),
    )
    _save_crystal(os.path.join(out_dir, "recon_2x2.npz"), recon_2x2)

    identity = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
    recon_no_disp = apply_surface_reconstruction(
        slab=slab_001,
        reconstruction_matrix=identity,
        surface_layer_depth_angstrom=3.0,
        atomic_displacements=jnp.zeros((1, 3)),
    )
    _save_crystal(
        os.path.join(out_dir, "recon_no_disp.npz"),
        recon_no_disp,
    )

    recon_with_disp = apply_surface_reconstruction(
        slab=slab_001,
        reconstruction_matrix=identity,
        surface_layer_depth_angstrom=3.0,
        atomic_displacements=jnp.array([[0.5, 0.0, 0.0]]),
    )
    _save_crystal(
        os.path.join(out_dir, "recon_with_disp.npz"),
        recon_with_disp,
    )

    ads_full = add_adsorbate_layer(
        slab=slab_001,
        adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
        adsorbate_atomic_numbers=jnp.array([8.0]),
        coverage_fraction=1.0,
    )
    _save_crystal(os.path.join(out_dir, "ads_full.npz"), ads_full)

    ads_half = add_adsorbate_layer(
        slab=slab_001,
        adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
        adsorbate_atomic_numbers=jnp.array([8.0]),
        coverage_fraction=0.5,
    )
    _save_crystal(os.path.join(out_dir, "ads_half.npz"), ads_half)

    ads_multi = add_adsorbate_layer(
        slab=slab_001,
        adsorbate_positions_fractional=jnp.array(
            [[0.25, 0.25, 0.95], [0.75, 0.75, 0.95]]
        ),
        adsorbate_atomic_numbers=jnp.array([8.0, 8.0]),
        coverage_fraction=1.0,
    )
    _save_crystal(os.path.join(out_dir, "ads_multi.npz"), ads_multi)

    ads_zero = add_adsorbate_layer(
        slab=slab_001,
        adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
        adsorbate_atomic_numbers=jnp.array([14.0]),
        coverage_fraction=0.0,
    )
    _save_crystal(os.path.join(out_dir, "ads_zero.npz"), ads_zero)

    # Library slabs
    _save_crystal(os.path.join(out_dir, "si111_1x1.npz"), si111_1x1())
    _save_crystal(
        os.path.join(out_dir, "si111_1x1_custom.npz"),
        si111_1x1(a_lattice_angstrom=6.0),
    )
    _save_crystal(os.path.join(out_dir, "si111_7x7.npz"), si111_7x7())
    _save_crystal(os.path.join(out_dir, "si100_2x1.npz"), si100_2x1())
    _save_crystal(
        os.path.join(out_dir, "gaas001_2x4.npz"),
        gaas001_2x4(),
    )
    _save_crystal(
        os.path.join(out_dir, "mgo001.npz"),
        mgo001_bulk_terminated(),
    )
    _save_crystal(
        os.path.join(out_dir, "srtio3_001.npz"),
        srtio3_001_2x1(),
    )

    print(f"Generated {len(os.listdir(out_dir))} fixtures in {out_dir}")


if __name__ == "__main__":
    main()
