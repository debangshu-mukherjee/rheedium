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
from jaxtyping import Float
from numpy import ndarray as NDArray  # noqa: N812

from rheedium.recon.surface_builder import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
)
from rheedium.types import CrystalStructure, create_crystal_structure
from rheedium.ucell import build_cell_vectors

_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name: str) -> dict[str, object]:
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


def _make_cubic_bulk(a: float = 2.0) -> CrystalStructure:
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


def _make_test_slab() -> CrystalStructure:
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

    def test_create_surface_slab_handles_aligned_surface_normal(self) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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

    def test_returns_four_columns(self) -> None:
        """Cart positions should have 4 columns [x,y,z,Z]."""
        d: dict = _load("slab_001.npz")
        assert d["cart_positions"].shape[1] == 4

    def test_slab_c_equals_thickness_plus_vacuum(self) -> None:
        """Cell c parameter should equal slab + vacuum thickness."""
        d: dict = _load("slab_001.npz")
        c: float = float(d["cell_lengths"][2])
        chex.assert_trees_all_close(c, 25.0, atol=1e-6)

    def test_slab_has_atoms(self) -> None:
        """Slab should contain at least one atom."""
        d: dict = _load("slab_001.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_atoms_within_slab_thickness(self) -> None:
        """All atom z-coordinates should be within [0, thickness]."""
        d: dict = _load("slab_001.npz")
        z: Float[NDArray, "N"] = d["cart_positions"][:, 2]
        assert float(z.min()) >= -0.1
        assert float(z.max()) <= 10.1

    def test_atomic_numbers_preserved(self) -> None:
        """Slab atoms should have same Z as bulk crystal."""
        bulk: dict = _load("cubic_crystal.npz")
        slab: dict = _load("slab_001.npz")
        bulk_z: set = set(bulk["cart_positions"][:, 3])
        slab_z: set = set(slab["cart_positions"][:, 3])
        assert slab_z.issubset(bulk_z)

    def test_stoichiometry_preserved_mgo(self) -> None:
        """MgO slab should have Mg:O ratio close to 1:1."""
        d: dict = _load("mgo_slab.npz")
        z_nums: Float[NDArray, "N"] = d["cart_positions"][:, 3]
        n_mg: int = int(np.sum(np.abs(z_nums - 12.0) < 0.5))
        n_o: int = int(np.sum(np.abs(z_nums - 8.0) < 0.5))
        assert n_mg > 0
        assert n_o > 0
        ratio: float = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_angles_valid(self) -> None:
        """All output cell angles should be in (0, 180)."""
        d: dict = _load("slab_001.npz")
        for angle in d["cell_angles"]:
            assert 0.0 < float(angle) < 180.0

    @parameterized.named_parameters(
        ("001", "slab_001.npz"),
        ("110", "slab_110.npz"),
        ("111", "slab_111.npz"),
    )
    def test_various_orientations(self, fname: str) -> None:
        """Slabs for various Miller indices should have atoms."""
        d: dict = _load(fname)
        assert d["cart_positions"].shape[0] > 0
        assert d["cart_positions"].shape[1] == 4

    def test_thicker_slab_has_more_atoms(self) -> None:
        """A thicker slab should contain more atoms."""
        thin: dict = _load("thin_slab.npz")
        thick: dict = _load("slab_001.npz")
        assert thick["cart_positions"].shape[0] >= (
            thin["cart_positions"].shape[0]
        )

    def test_frac_and_cart_shapes_match(self) -> None:
        """Fractional and Cartesian arrays should match shape."""
        d: dict = _load("slab_001.npz")
        assert d["frac_positions"].shape == d["cart_positions"].shape


class TestApplySurfaceReconstruction(chex.TestCase, parameterized.TestCase):
    """Tests for apply_surface_reconstruction (via fixtures)."""

    def test_reconstruction_has_atoms(self) -> None:
        """Reconstructed slab should have atoms."""
        d: dict = _load("recon_2x2.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_2x2_expands_cell(self) -> None:
        """2x2 reconstruction should roughly double a and b."""
        orig: dict = _load("slab_001.npz")
        recon: dict = _load("recon_2x2.npz")
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

    def test_atom_count_scales(self) -> None:
        """2x2 recon should have more atoms than 1x1."""
        orig: dict = _load("slab_001.npz")
        recon: dict = _load("recon_2x2.npz")
        assert recon["cart_positions"].shape[0] > (
            orig["cart_positions"].shape[0]
        )

    def test_displacement_moves_atoms(self) -> None:
        """Non-zero displacement should change positions."""
        no_disp: dict = _load("recon_no_disp.npz")
        with_disp: dict = _load("recon_with_disp.npz")
        diff: float = float(
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

    def test_adsorbate_increases_atom_count(self) -> None:
        """Adding adsorbates should increase total atom count."""
        orig: dict = _load("slab_001.npz")
        ads: dict = _load("ads_full.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 1
        )

    def test_coverage_weights_atomic_number(self) -> None:
        """Coverage 0.5 should halve the adsorbate Z value."""
        orig: dict = _load("slab_001.npz")
        ads: dict = _load("ads_half.npz")
        n_orig: int = orig["cart_positions"].shape[0]
        ads_z: float = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 4.0, atol=1e-6)

    def test_cell_parameters_unchanged(self) -> None:
        """Adsorbates should not change cell parameters."""
        orig: dict = _load("slab_001.npz")
        ads: dict = _load("ads_full.npz")
        np.testing.assert_allclose(ads["cell_lengths"], orig["cell_lengths"])
        np.testing.assert_allclose(ads["cell_angles"], orig["cell_angles"])

    def test_multiple_adsorbates(self) -> None:
        """Should handle multiple adsorbate atoms."""
        orig: dict = _load("slab_001.npz")
        ads: dict = _load("ads_multi.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 2
        )

    def test_zero_coverage_zeroes_z(self) -> None:
        """Coverage 0.0 should produce zero effective Z."""
        orig: dict = _load("slab_001.npz")
        ads: dict = _load("ads_zero.npz")
        n_orig: int = orig["cart_positions"].shape[0]
        ads_z: float = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 0.0, atol=1e-10)
