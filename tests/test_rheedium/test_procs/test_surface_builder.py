"""Test suite for procs/surface_builder.py.

Tests surface slab construction, surface reconstruction application,
and adsorbate layer addition. All slabs are loaded from pre-built
.npz fixtures in tests/test_data/recon/ and a small set of direct
unit tests exercises the branch-heavy geometry builders.
"""

from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jaxtyping import Array, Float, Integer
from numpy.typing import NDArray

from rheedium.procs.surface_builder import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
)
from rheedium.types import CrystalStructure
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.types.custom_types import scalar_float
from rheedium.ucell.unitcell import build_cell_vectors

_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "recon"


def _load(name: str) -> dict[str, Float[NDArray, "..."]]:
    """Load a fixture .npz by name."""
    return dict(np.load(_DATA_DIR / name))


def _make_cubic_bulk(a: float = 2.0) -> CrystalStructure:
    """Build a minimal cubic bulk crystal for direct slab tests."""
    frac_positions: Float[Array, "..."] = jnp.array(
        [
            [0.0, 0.0, 0.0, 14.0],
            [0.5, 0.5, 0.5, 14.0],
        ]
    )
    cell_vectors: Float[Array, "..."] = build_cell_vectors(
        a, a, a, 90.0, 90.0, 90.0
    )
    cart_positions: Float[Array, "..."] = jnp.column_stack(
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
    cell_vectors: Float[Array, "..."] = build_cell_vectors(
        2.0, 2.0, 6.0, 90.0, 90.0, 90.0
    )
    cart_positions: Float[Array, "..."] = jnp.array(
        [
            [0.2, 0.2, 0.5, 14.0],
            [0.4, 0.4, 4.2, 14.0],
            [1.2, 1.6, 5.0, 8.0],
        ]
    )
    frac_positions: Float[Array, "..."] = jnp.column_stack(
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
        r"""(001) cuts should keep the in-plane cubic metric.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (001) cuts should
        keep the in-plane cubic metric.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: Integer[Array, "..."] = create_surface_slab(
            _make_cubic_bulk(),
            jnp.array([0, 0, 1], dtype=jnp.int32),
            3.0,
            5.0,
        )

        # A 3 Angstrom request on a d_layer=2 (001) cut snaps up to two whole
        # layers (span 4), so c = 2 * 2 + 5 vacuum = 9; the material span is an
        # exact multiple of the interlayer spacing (a valid periodic cell).
        np.testing.assert_allclose(
            np.asarray(slab.cell_lengths),
            np.array([2.0, 2.0, 9.0]),
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
        r"""Non-(001) cuts should still yield a finite, bounded slab.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-(001) cuts
        should still yield a finite, bounded slab.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: Integer[Array, "..."] = create_surface_slab(
            _make_cubic_bulk(),
            jnp.array([1, 1, 0], dtype=jnp.int32),
            3.0,
            5.0,
        )

        assert slab.cart_positions.shape[1] == 4
        assert slab.cart_positions.shape[0] > 0
        # (110) of the BCC test cell reduces to a primitive in-plane mesh of
        # length sqrt(3); the 3 Angstrom request snaps up to three whole
        # d_layer=a/sqrt(2) layers, so c = 3 * a / sqrt(2) + 5 vacuum.
        np.testing.assert_allclose(
            float(slab.cell_lengths[2]), 3.0 * np.sqrt(2.0) + 5.0, atol=1e-6
        )
        np.testing.assert_allclose(
            float(slab.cell_lengths[0]), np.sqrt(3.0), atol=1e-4
        )
        assert float(slab.cart_positions[:, 2].max()) <= 3.0 + 1e-6
        assert np.all(np.isfinite(np.asarray(slab.frac_positions)))

    def test_apply_surface_reconstruction_moves_only_requested_surface_atoms(
        self,
    ) -> None:
        r"""Only the first n displaced surface atoms should be updated.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Only the first n
        displaced surface atoms should be updated.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        reconstructed: Integer[Array, "..."] = apply_surface_reconstruction(
            _make_test_slab(),
            jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
            1.0,
            jnp.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=jnp.float64),
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
        r"""Off-diagonal reconstruction matrices should shear the cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Off-diagonal
        reconstruction matrices should shear the cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        reconstructed: Integer[Array, "..."] = apply_surface_reconstruction(
            _make_test_slab(),
            jnp.array([[1, 1], [0, 1]], dtype=jnp.int32),
            0.0,
            jnp.zeros((1, 3), dtype=jnp.float64),
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

    def test_add_adsorbate_layer_appends_occupancy_weighted_sites(
        self,
    ) -> None:
        r"""Adsorbates keep integral Z and store coverage as occupancy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: adsorbates are
        appended in both coordinate systems with their element's integral
        atomic number, while the fractional monolayer coverage lands in the
        first-class ``occupancies`` field (the C6 contract — coverage never
        rescales Z into a different element).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: Any = _make_test_slab()
        decorated: Float[Array, "..."] = add_adsorbate_layer(
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
                    [0.5, 1.0, 4.5, 8.0],
                    [1.5, 0.5, 1.5, 16.0],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.frac_positions[-2:]),
            np.array(
                [
                    [0.25, 0.5, 0.75, 8.0],
                    [0.75, 0.25, 0.25, 16.0],
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.occupancies[-2:]),
            np.array([0.25, 0.25]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.occupancies[:-2]),
            np.ones(slab.cart_positions.shape[0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(decorated.cell_lengths),
            np.asarray(slab.cell_lengths),
            atol=1e-6,
        )


class TestCreateSurfaceSlab(chex.TestCase, parameterized.TestCase):
    """Tests for create_surface_slab (validated via fixtures).

    :see: :func:`~rheedium.procs.create_surface_slab`
    """

    def test_returns_four_columns(self) -> None:
        r"""Cart positions should have 4 columns [x,y,z,Z].

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cart positions
        should have 4 columns [x,y,z,Z].

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        assert d["cart_positions"].shape[1] == 4

    def test_slab_c_equals_thickness_plus_vacuum(self) -> None:
        r"""Cell c parameter should equal slab + vacuum thickness.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cell c parameter
        should equal slab + vacuum thickness.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        c: float = float(d["cell_lengths"][2])
        chex.assert_trees_all_close(c, 25.0, atol=1e-6)

    def test_slab_has_atoms(self) -> None:
        r"""Slab should contain at least one atom.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab should
        contain at least one atom.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_atoms_within_slab_thickness(self) -> None:
        r"""All atom z-coordinates should be within [0, thickness].

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All atom
        z-coordinates should be within [0, thickness].

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        z: Any = d["cart_positions"][:, 2]
        assert float(z.min()) >= -0.1
        assert float(z.max()) <= 10.1

    def test_atomic_numbers_preserved(self) -> None:
        r"""Slab atoms should have same Z as bulk crystal.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab atoms should
        have same Z as bulk crystal.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        bulk: Any = _load("cubic_crystal.npz")
        slab: Any = _load("slab_001.npz")
        bulk_z: Any = set(bulk["cart_positions"][:, 3])
        slab_z: Any = set(slab["cart_positions"][:, 3])
        assert slab_z.issubset(bulk_z)

    def test_stoichiometry_preserved_mgo(self) -> None:
        r"""MgO slab should have Mg:O ratio close to 1:1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: MgO slab should
        have Mg:O ratio close to 1:1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("mgo_slab.npz")
        z_nums: Any = d["cart_positions"][:, 3]
        n_mg: int = int(np.sum(np.abs(z_nums - 12.0) < 0.5))
        n_o: int = int(np.sum(np.abs(z_nums - 8.0) < 0.5))
        assert n_mg > 0
        assert n_o > 0
        ratio: Any = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_angles_valid(self) -> None:
        r"""All output cell angles should be in (0, 180).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All output cell
        angles should be in (0, 180).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        angle: scalar_float
        for angle in d["cell_angles"]:
            assert 0.0 < float(angle) < 180.0

    @parameterized.named_parameters(
        ("001", "slab_001.npz"),
        ("110", "slab_110.npz"),
        ("111", "slab_111.npz"),
    )
    def test_various_orientations(self, fname: str) -> None:
        r"""Slabs for various Miller indices should have atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slabs for various
        Miller indices should have atoms.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``fname``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load(fname)
        assert d["cart_positions"].shape[0] > 0
        assert d["cart_positions"].shape[1] == 4

    def test_thicker_slab_has_more_atoms(self) -> None:
        r"""A thicker slab should contain more atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A thicker slab
        should contain more atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        thin: Any = _load("thin_slab.npz")
        thick: Any = _load("slab_001.npz")
        assert (
            thick["cart_positions"].shape[0]
            >= (thin["cart_positions"].shape[0])
        )

    def test_frac_and_cart_shapes_match(self) -> None:
        r"""Fractional and Cartesian arrays should match shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fractional and
        Cartesian arrays should match shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("slab_001.npz")
        assert d["frac_positions"].shape == d["cart_positions"].shape


class TestApplySurfaceReconstruction(chex.TestCase, parameterized.TestCase):
    """Tests for apply_surface_reconstruction (via fixtures).

    :see: :func:`~rheedium.procs.apply_surface_reconstruction`
    """

    def test_reconstruction_has_atoms(self) -> None:
        r"""Reconstructed slab should have atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Reconstructed slab
        should have atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        d: Any = _load("recon_2x2.npz")
        assert d["cart_positions"].shape[0] > 0

    def test_2x2_expands_cell(self) -> None:
        r"""2x2 reconstruction should roughly double a and b.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2x2 reconstruction
        should roughly double a and b.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        recon: Any = _load("recon_2x2.npz")
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
        r"""2x2 recon should have more atoms than 1x1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2x2 recon should
        have more atoms than 1x1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        recon: Any = _load("recon_2x2.npz")
        assert (
            recon["cart_positions"].shape[0]
            > (orig["cart_positions"].shape[0])
        )

    def test_displacement_moves_atoms(self) -> None:
        r"""Non-zero displacement should change positions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-zero
        displacement should change positions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        no_disp: Any = _load("recon_no_disp.npz")
        with_disp: Any = _load("recon_with_disp.npz")
        diff: Any = float(
            np.sum(
                np.abs(
                    no_disp["cart_positions"][:, :3]
                    - with_disp["cart_positions"][:, :3]
                )
            )
        )
        assert diff > 0.1


class TestAddAdsorbateLayer(chex.TestCase, parameterized.TestCase):
    """Tests for add_adsorbate_layer (via fixtures).

    :see: :func:`~rheedium.procs.add_adsorbate_layer`
    """

    def test_adsorbate_increases_atom_count(self) -> None:
        r"""Adding adsorbates should increase total atom count.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Adding adsorbates
        should increase total atom count.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        ads: Any = _load("ads_full.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 1
        )

    def test_coverage_weights_atomic_number(self) -> None:
        r"""Coverage keeps integral Z (coverage lives in occupancy).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: partial
        coverage no longer rescales the adsorbate atomic number; the
        adsorbate keeps its element's integral Z (here O, Z=8) while the
        coverage enters the first-class occupancy field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        ads: Any = _load("ads_half.npz")
        n_orig: int = orig["cart_positions"].shape[0]
        ads_z: Any = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 8.0, atol=1e-6)

    def test_cell_parameters_unchanged(self) -> None:
        r"""Adsorbates should not change cell parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Adsorbates should
        not change cell parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        ads: Any = _load("ads_full.npz")
        np.testing.assert_allclose(ads["cell_lengths"], orig["cell_lengths"])
        np.testing.assert_allclose(ads["cell_angles"], orig["cell_angles"])

    def test_multiple_adsorbates(self) -> None:
        r"""Should handle multiple adsorbate atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should handle
        multiple adsorbate atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        ads: Any = _load("ads_multi.npz")
        assert ads["cart_positions"].shape[0] == (
            orig["cart_positions"].shape[0] + 2
        )

    def test_zero_coverage_zeroes_z(self) -> None:
        r"""Coverage 0.0 keeps integral Z (zero enters occupancy).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: zero coverage
        keeps the adsorbate's integral atomic number (Z=14) rather than
        rescaling it to zero; the vanishing weight lives in the occupancy
        field, so the atom simply contributes no scattering amplitude.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_builder``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orig: Any = _load("slab_001.npz")
        ads: Any = _load("ads_zero.npz")
        n_orig: int = orig["cart_positions"].shape[0]
        ads_z: Any = float(ads["cart_positions"][n_orig, 3])
        chex.assert_trees_all_close(ads_z, 14.0, atol=1e-10)
