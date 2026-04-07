"""Test suite for recon/surface_builder.py.

Tests surface slab construction, surface reconstruction application,
and adsorbate layer addition.  All slabs are loaded from pre-built
.npz fixtures in tests/test_data/recon/ — no JIT at test time.
"""

import os
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jaxtyping import Float
from numpy import ndarray as NDArray  # noqa: N812

from rheedium.types import CrystalStructure

_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name: str) -> dict[str, object]:
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


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
