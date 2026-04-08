"""Test suite for recon/surface_builder.py.

Tests surface slab construction, surface reconstruction application,
and adsorbate layer addition. All slabs are loaded from pre-built
.npz fixtures in tests/test_data/recon/ and a small set of direct
unit tests exercises the branch-heavy geometry builders.
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from rheedium.recon.surface_builder import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
)
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.ucell.unitcell import build_cell_vectors

_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name):
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


def _make_cubic_bulk(a=2.0):
    """Build a minimal cubic bulk crystal for direct slab tests."""
    frac_positions = jnp.array(
        [
            [0.0, 0.0, 0.0, 14.0],
            [0.5, 0.5, 0.5, 14.0],
        ]
    )
    cell_vectors = build_cell_vectors(a, a, a, 90.0, 90.0, 90.0)
    cart_positions = jnp.column_stack(
        [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([a, a, a]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


def _make_test_slab():
    """Build a simple orthorhombic slab for direct reconstruction tests."""
    cell_vectors = build_cell_vectors(2.0, 2.0, 6.0, 90.0, 90.0, 90.0)
    cart_positions = jnp.array(
        [
            [0.2, 0.2, 0.5, 14.0],
            [0.4, 0.4, 4.2, 14.0],
            [1.2, 1.6, 5.0, 8.0],
        ]
    )
    frac_positions = jnp.column_stack(
        [
            cart_positions[:, :3] @ jnp.linalg.inv(cell_vectors).T,
            cart_positions[:, 3],
        ]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([2.0, 2.0, 6.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestSurfaceBuilderDirect(chex.TestCase):
    """Direct unit tests for branch-heavy geometry builders."""

    def test_create_surface_slab_handles_aligned_surface_normal(self):
        """(001) cuts should keep the in-plane cubic metric."""
        slab = create_surface_slab(
            _make_cubic_bulk(),
            jnp.array([0, 0, 1], dtype=jnp.int32),
            3.0,
            5.0,
        )

        np.testing.assert_allclose(
            np.asarray(slab.cell_lengths),
            np.array([2.0, 2.0, 8.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(slab.cell_angles),
            np.array([90.0, 90.0, 90.0]),
            atol=1e-6,
        )
        assert float(slab.cart_positions[:, 2].min()) >= 0.0
        assert float(slab.cart_positions[:, 2].max()) <= 3.0 + 1e-6

    def test_create_surface_slab_rotates_non_aligned_surface_normal(
        self,
    ):
        """Non-(001) cuts should still yield a finite, bounded slab."""
        slab = create_surface_slab(
            _make_cubic_bulk(),
            jnp.array([1, 1, 0], dtype=jnp.int32),
            3.0,
            5.0,
        )

        assert slab.cart_positions.shape == (5, 4)
        np.testing.assert_allclose(
            np.asarray(slab.cell_lengths),
            np.array([2.0, 2.0, 8.0]),
            atol=1e-6,
        )
        assert float(slab.cart_positions[:, 2].max()) <= 3.0 + 1e-6
        assert np.all(np.isfinite(np.asarray(slab.frac_positions)))

    def test_apply_surface_reconstruction_moves_only_requested_surface_atoms(
        self,
    ):
        """Only the first n displaced surface atoms should be updated."""
        reconstructed = apply_surface_reconstruction(
            _make_test_slab(),
            jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
            1.0,
            jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float64),
        )

        np.testing.assert_allclose(
            np.asarray(reconstructed.cart_positions),
            np.array(
                [
                    [0.2, 0.2, 0.5, 14.0],
                    [0.5, 0.6, 4.5, 14.0],
                    [1.2, 1.6, 5.0, 8.0],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(reconstructed.cell_lengths),
            np.array([2.0, 2.0, 6.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(reconstructed.cell_angles),
            np.array([90.0, 90.0, 90.0]),
            atol=1e-6,
        )

    def test_apply_surface_reconstruction_shears_the_in_plane_cell(
        self,
    ):
        """Off-diagonal reconstruction matrices should shear the cell."""
        reconstructed = apply_surface_reconstruction(
            _make_test_slab(),
            jnp.array([[1, 1], [0, 1]], dtype=jnp.int32),
            0.0,
            jnp.zeros((0, 3), dtype=jnp.float64),
        )

        assert reconstructed.cart_positions.shape == (3, 4)
        np.testing.assert_allclose(
            np.asarray(reconstructed.cell_lengths),
            np.array([np.sqrt(8.0), 2.0, 6.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(reconstructed.cell_angles),
            np.array([90.0, 90.0, 45.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(reconstructed.cart_positions[:, 2])),
            np.array([0.5, 4.2, 5.0]),
            atol=1e-6,
        )

    def test_add_adsorbate_layer_appends_weighted_cartesian_positions(
        self,
    ):
        """Adsorbates should be appended in both coordinate systems."""
        slab = _make_test_slab()
        decorated = add_adsorbate_layer(
            slab,
            jnp.array(
                [[0.25, 0.5, 0.75], [0.75, 0.25, 0.25]], dtype=jnp.float64
            ),
            jnp.array([8.0, 16.0], dtype=jnp.float64),
            0.25,
        )

        np.testing.assert_allclose(
            np.asarray(decorated.cart_positions[-2:]),
            np.array(
                [
                    [0.5, 1.0, 4.5, 2.0],
                    [1.5, 0.5, 1.5, 4.0],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.frac_positions[-2:]),
            np.array(
                [
                    [0.25, 0.5, 0.75, 2.0],
                    [0.75, 0.25, 0.25, 4.0],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.cell_lengths),
            np.asarray(slab.cell_lengths),
            atol=1e-6,
        )


class TestCreateSurfaceSlab(chex.TestCase, parameterized.TestCase):
    """Tests for create_surface_slab (validated via fixtures)."""

    def test_returns_four_columns(self):
        """Cart positions should have 4 columns [x,y,z,Z]."""
        d = _load("slab_001.npz")
        assert d["cart_positions"].shape[1] == 4

    def test_slab_c_equals_thickness_plus_vacuum(self):
        """Cell c parameter should equal slab + vacuum thickness."""
        d = _load("slab_001.npz")
        c = float(d["cell_lengths"][2])
        chex.assert_trees_all_close(c, 25.0, atol=1e-6)

    def test_slab_has_atoms(self):
        """Slab should contain at least one atom."""
        d = _load("slab_001.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_atoms_within_slab_thickness(self):
        """All atom z-coordinates should be within [0, thickness]."""
        d = _load("slab_001.npz")
        z = d["cart_positions"][:, 2]
        assert float(z.min()) >= -0.1
        assert float(z.max()) <= 10.1

    def test_atomic_numbers_preserved(self):
        """Slab atoms should have same Z as bulk crystal."""
        bulk = _load("cubic_crystal.npz")
        slab = _load("slab_001.npz")
        bulk_z = set(bulk["cart_positions"][:, 3])
        slab_z = set(slab["cart_positions"][:, 3])
        assert slab_z.issubset(bulk_z)

    def test_stoichiometry_preserved_mgo(self):
        """MgO slab should have Mg:O ratio close to 1:1."""
        d = _load("mgo_slab.npz")
        z_nums = d["cart_positions"][:, 3]
        n_mg = int(np.sum(np.abs(z_nums - 12.0) < 0.5))
        n_o = int(np.sum(np.abs(z_nums - 8.0) < 0.5))
        assert n_mg > 0
        assert n_o > 0
        ratio = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_angles_valid(self):
        """All output cell angles should be in (0, 180)."""
        d = _load("slab_001.npz")
        for angle in d["cell_angles"]:
            assert 0.0 < float(angle) < 180.0

    @parameterized.named_parameters(
        ("001", "slab_001.npz"),
        ("110", "slab_110.npz"),
        ("111", "slab_111.npz"),
    )
    def test_various_orientations(self, fname):
        """Slabs for various Miller indices should have atoms."""
        d = _load(fname)
        assert d["cart_positions"].shape[0] > 0
        assert d["cart_positions"].shape[1] == 4

    def test_thicker_slab_has_more_atoms(self):
        """A thicker slab should contain more atoms."""
        thin = _load("thin_slab.npz")
        thick = _load("slab_001.npz")
        assert thick["cart_positions"].shape[0] >= (
            thin["cart_positions"].shape[0]
        )

    def test_frac_and_cart_shapes_match(self):
        """Fractional and Cartesian arrays should match shape."""
        d = _load("slab_001.npz")
        assert d["frac_positions"].shape == d["cart_positions"].shape


class TestApplySurfaceReconstruction(chex.TestCase, parameterized.TestCase):
    """Tests for apply_surface_reconstruction (via fixtures)."""

    def test_reconstruction_has_atoms(self):
        """Reconstructed slab should have atoms."""
        d = _load("recon_2x2.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_2x2_expands_cell(self):
        """2x2 reconstruction should roughly double a and b."""
        orig = _load("slab_001.npz")
        recon = _load("recon_2x2.npz")
        chex.assert_trees_all_close(
            float(recon["cell_lengths"][0]),
            2.0 * float(orig["cell_lengths"][0]),
            atol=0.5,
        )
        chex.assert_trees_all_close(
            float(recon["cell_lengths"][1]),
            2.0 * float(orig["cell_lengths"][1]),
            atol=0.5,
        )

    def test_atom_count_scales(self):
        """2x2 recon should have more atoms than 1x1."""
        orig = _load("slab_001.npz")
        recon = _load("recon_2x2.npz")
        assert recon["cart_positions"].shape[0] > (
            orig["cart_positions"].shape[0]
        )

    def test_displacement_moves_atoms(self):
        """Non-zero displacement should change positions."""
        no_disp = _load("recon_no_disp.npz")
        with_disp = _load("recon_with_disp.npz")
        diff = float(
            np.sum(
                np.abs(
                    no_disp["cart_positions"][:, :3]
                    - with_disp["cart_positions"][:, :3]
                )
            )
        )
        assert diff > 0.1


class TestAddAdsorbateLayer(chex.TestCase, parameterized.TestCase):
    """Tests for add_adsorbate_layer (via fixtures)."""

    def test_adsorbate_increases_atom_count(self):
        """Adding adsorbates should increase total atom count."""
        orig = _load("slab_001.npz")
        ads = _load("ads_full.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 1
        )

    def test_coverage_weights_atomic_number(self):
        """Coverage 0.5 should halve the adsorbate Z value."""
        orig = _load("slab_001.npz")
        ads = _load("ads_half.npz")
        n_orig = orig["cart_positions"].shape[0]
        ads_z = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 4.0, atol=1e-6)

    def test_cell_parameters_unchanged(self):
        """Adsorbates should not change cell parameters."""
        orig = _load("slab_001.npz")
        ads = _load("ads_full.npz")
        np.testing.assert_allclose(ads["cell_lengths"], orig["cell_lengths"])
        np.testing.assert_allclose(ads["cell_angles"], orig["cell_angles"])

    def test_multiple_adsorbates(self):
        """Should handle multiple adsorbate atoms."""
        orig = _load("slab_001.npz")
        ads = _load("ads_multi.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 2
        )

    def test_zero_coverage_zeroes_z(self):
        """Coverage 0.0 should produce zero effective Z."""
        orig = _load("slab_001.npz")
        ads = _load("ads_zero.npz")
        n_orig = orig["cart_positions"].shape[0]
        ads_z = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 0.0, atol=1e-10)
