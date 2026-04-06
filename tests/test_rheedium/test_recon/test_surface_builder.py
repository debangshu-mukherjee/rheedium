"""Test suite for recon/surface_builder.py.

Tests surface slab construction, surface reconstruction application,
and adsorbate layer addition.
"""

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from rheedium.recon import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
)
from rheedium.types import CrystalStructure, create_crystal_structure
from rheedium.ucell import build_cell_vectors


def _make_cubic_crystal(
    a: float = 4.0,
    n_atoms: int = 4,
) -> CrystalStructure:
    """Create a simple FCC-like cubic crystal for testing."""
    frac_coords: np.ndarray = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )[:n_atoms]
    z_nums: np.ndarray = np.full(n_atoms, 14.0)

    cell_lengths = jnp.array([a, a, a])
    cell_angles = jnp.array([90.0, 90.0, 90.0])
    cell_vecs = build_cell_vectors(a, a, a, 90.0, 90.0, 90.0)

    cart_coords: np.ndarray = frac_coords @ np.array(cell_vecs)

    frac_pos = jnp.array(np.column_stack([frac_coords, z_nums]))
    cart_pos = jnp.array(np.column_stack([cart_coords, z_nums]))

    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


def _make_mgo_crystal(a: float = 4.211) -> CrystalStructure:
    """Create MgO rocksalt crystal for stoichiometry tests."""
    frac_coords: np.ndarray = np.array(
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
    z_nums: np.ndarray = np.array(
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

    cell_lengths = jnp.array([a, a, a])
    cell_angles = jnp.array([90.0, 90.0, 90.0])
    cell_vecs = build_cell_vectors(a, a, a, 90.0, 90.0, 90.0)

    cart_coords: np.ndarray = frac_coords @ np.array(cell_vecs)

    frac_pos = jnp.array(np.column_stack([frac_coords, z_nums]))
    cart_pos = jnp.array(np.column_stack([cart_coords, z_nums]))

    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


class TestCreateSurfaceSlab(chex.TestCase, parameterized.TestCase):
    """Tests for create_surface_slab function."""

    def test_returns_crystal_structure(self) -> None:
        """Should return a CrystalStructure instance."""
        crystal: CrystalStructure = _make_cubic_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        assert isinstance(slab, CrystalStructure)

    def test_slab_c_equals_thickness_plus_vacuum(self) -> None:
        """Cell c parameter should equal slab + vacuum thickness."""
        crystal: CrystalStructure = _make_cubic_crystal()
        thickness: float = 12.0
        vacuum: float = 15.0
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=thickness,
            vacuum_gap_angstrom=vacuum,
        )
        expected_c: float = thickness + vacuum
        chex.assert_trees_all_close(
            float(slab.cell_lengths[2]),
            expected_c,
            atol=1e-6,
        )

    def test_slab_has_atoms(self) -> None:
        """Slab should contain at least one atom."""
        crystal: CrystalStructure = _make_cubic_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        n_atoms: int = slab.cart_positions.shape[0]
        assert n_atoms > 0

    def test_atoms_within_slab_thickness(self) -> None:
        """All atom z-coordinates should be within [0, thickness]."""
        crystal: CrystalStructure = _make_cubic_crystal()
        thickness: float = 10.0
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=thickness,
            vacuum_gap_angstrom=15.0,
        )
        z_coords: jnp.ndarray = slab.cart_positions[:, 2]
        assert float(jnp.min(z_coords)) >= -0.1
        assert float(jnp.max(z_coords)) <= thickness + 0.1

    def test_atomic_numbers_preserved(self) -> None:
        """Slab atoms should have same Z as bulk crystal."""
        crystal: CrystalStructure = _make_cubic_crystal()
        bulk_z: set = set(np.array(crystal.cart_positions[:, 3]))
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        slab_z: set = set(np.array(slab.cart_positions[:, 3]))
        assert slab_z.issubset(bulk_z)

    def test_stoichiometry_preserved_mgo(self) -> None:
        """MgO slab should have Mg:O ratio close to 1:1."""
        crystal: CrystalStructure = _make_mgo_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=20.0,
            vacuum_gap_angstrom=15.0,
        )
        z_nums: np.ndarray = np.array(slab.cart_positions[:, 3])
        n_mg: int = int(np.sum(np.abs(z_nums - 12.0) < 0.5))
        n_o: int = int(np.sum(np.abs(z_nums - 8.0) < 0.5))
        assert n_mg > 0
        assert n_o > 0
        ratio: float = n_mg / (n_o + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_cell_angles_valid(self) -> None:
        """All output cell angles should be in (0, 180)."""
        crystal: CrystalStructure = _make_cubic_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        for i in range(3):
            angle: float = float(slab.cell_angles[i])
            assert 0.0 < angle < 180.0

    @parameterized.named_parameters(
        ("001", [0, 0, 1]),
        ("110", [1, 1, 0]),
        ("111", [1, 1, 1]),
    )
    def test_various_orientations(self, miller: list[int]) -> None:
        """Slab construction should work for various Miller indices."""
        crystal: CrystalStructure = _make_cubic_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array(miller, dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        assert slab.cart_positions.shape[0] > 0
        assert slab.cart_positions.shape[1] == 4

    def test_thicker_slab_has_more_atoms(self) -> None:
        """A thicker slab should contain more atoms."""
        crystal: CrystalStructure = _make_cubic_crystal()
        thin: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=5.0,
            vacuum_gap_angstrom=15.0,
        )
        thick: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=20.0,
            vacuum_gap_angstrom=15.0,
        )
        assert thick.cart_positions.shape[0] >= (thin.cart_positions.shape[0])

    def test_frac_and_cart_shapes_match(self) -> None:
        """Fractional and Cartesian position arrays should match."""
        crystal: CrystalStructure = _make_cubic_crystal()
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        chex.assert_shape(
            slab.frac_positions,
            slab.cart_positions.shape,
        )


class TestApplySurfaceReconstruction(chex.TestCase, parameterized.TestCase):
    """Tests for apply_surface_reconstruction function."""

    def _make_slab(self) -> CrystalStructure:
        """Create a simple slab for reconstruction tests."""
        crystal: CrystalStructure = _make_cubic_crystal(a=4.0)
        slab: CrystalStructure = create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )
        return slab

    def test_reconstruction_returns_crystal(self) -> None:
        """Should return a CrystalStructure."""
        slab: CrystalStructure = self._make_slab()
        n_surface: int = 1
        recon: CrystalStructure = apply_surface_reconstruction(
            slab=slab,
            reconstruction_matrix=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
            surface_layer_depth_angstrom=3.0,
            atomic_displacements=jnp.zeros((n_surface, 3)),
        )
        assert isinstance(recon, CrystalStructure)

    def test_2x2_expands_cell(self) -> None:
        """2x2 reconstruction should roughly double a and b."""
        slab: CrystalStructure = self._make_slab()
        original_a: float = float(slab.cell_lengths[0])
        original_b: float = float(slab.cell_lengths[1])
        n_surface: int = 1
        recon: CrystalStructure = apply_surface_reconstruction(
            slab=slab,
            reconstruction_matrix=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
            surface_layer_depth_angstrom=3.0,
            atomic_displacements=jnp.zeros((n_surface, 3)),
        )
        new_a: float = float(recon.cell_lengths[0])
        new_b: float = float(recon.cell_lengths[1])
        chex.assert_trees_all_close(new_a, 2.0 * original_a, atol=0.5)
        chex.assert_trees_all_close(new_b, 2.0 * original_b, atol=0.5)

    def test_atom_count_scales_with_reconstruction(self) -> None:
        """2x2 recon should have roughly 4x atoms of 1x1."""
        slab: CrystalStructure = self._make_slab()
        n_orig: int = slab.cart_positions.shape[0]
        n_surface: int = 1
        recon: CrystalStructure = apply_surface_reconstruction(
            slab=slab,
            reconstruction_matrix=jnp.array([[2, 0], [0, 2]], dtype=jnp.int32),
            surface_layer_depth_angstrom=3.0,
            atomic_displacements=jnp.zeros((n_surface, 3)),
        )
        n_recon: int = recon.cart_positions.shape[0]
        ratio: float = n_recon / (n_orig + 1e-10)
        assert ratio > 2.0

    def test_displacement_moves_atoms(self) -> None:
        """Non-zero displacement should change atom positions."""
        slab: CrystalStructure = self._make_slab()
        n_surface: int = 1
        no_disp: CrystalStructure = apply_surface_reconstruction(
            slab=slab,
            reconstruction_matrix=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
            surface_layer_depth_angstrom=3.0,
            atomic_displacements=jnp.zeros((n_surface, 3)),
        )
        with_disp: CrystalStructure = apply_surface_reconstruction(
            slab=slab,
            reconstruction_matrix=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
            surface_layer_depth_angstrom=3.0,
            atomic_displacements=jnp.array([[0.5, 0.0, 0.0]]),
        )
        diff: float = float(
            jnp.sum(
                jnp.abs(
                    no_disp.cart_positions[:, :3]
                    - with_disp.cart_positions[:, :3]
                )
            )
        )
        assert diff > 0.1


class TestAddAdsorbateLayer(chex.TestCase, parameterized.TestCase):
    """Tests for add_adsorbate_layer function."""

    def _make_slab(self) -> CrystalStructure:
        """Create a simple slab for adsorbate tests."""
        crystal: CrystalStructure = _make_cubic_crystal(a=4.0)
        return create_surface_slab(
            bulk_crystal=crystal,
            surface_normal_miller=jnp.array([0, 0, 1], dtype=jnp.int32),
            slab_thickness_angstrom=10.0,
            vacuum_gap_angstrom=15.0,
        )

    def test_adsorbate_increases_atom_count(self) -> None:
        """Adding adsorbates should increase total atom count."""
        slab: CrystalStructure = self._make_slab()
        n_orig: int = slab.cart_positions.shape[0]
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab,
            adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
            adsorbate_atomic_numbers=jnp.array([8.0]),
            coverage_fraction=1.0,
        )
        n_new: int = decorated.cart_positions.shape[0]
        assert n_new == n_orig + 1

    def test_coverage_weights_atomic_number(self) -> None:
        """Coverage 0.5 should halve the adsorbate Z value."""
        slab: CrystalStructure = self._make_slab()
        n_orig: int = slab.cart_positions.shape[0]
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab,
            adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
            adsorbate_atomic_numbers=jnp.array([8.0]),
            coverage_fraction=0.5,
        )
        ads_z: float = float(decorated.cart_positions[n_orig, 3])
        chex.assert_trees_all_close(ads_z, 4.0, atol=1e-6)

    def test_cell_parameters_unchanged(self) -> None:
        """Adding adsorbates should not change cell parameters."""
        slab: CrystalStructure = self._make_slab()
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab,
            adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
            adsorbate_atomic_numbers=jnp.array([8.0]),
            coverage_fraction=1.0,
        )
        chex.assert_trees_all_close(
            decorated.cell_lengths,
            slab.cell_lengths,
        )
        chex.assert_trees_all_close(
            decorated.cell_angles,
            slab.cell_angles,
        )

    def test_multiple_adsorbates(self) -> None:
        """Should handle multiple adsorbate atoms."""
        slab: CrystalStructure = self._make_slab()
        n_orig: int = slab.cart_positions.shape[0]
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab,
            adsorbate_positions_fractional=jnp.array(
                [
                    [0.25, 0.25, 0.95],
                    [0.75, 0.75, 0.95],
                ]
            ),
            adsorbate_atomic_numbers=jnp.array([8.0, 8.0]),
            coverage_fraction=1.0,
        )
        n_new: int = decorated.cart_positions.shape[0]
        assert n_new == n_orig + 2

    def test_zero_coverage_zeroes_z(self) -> None:
        """Coverage 0.0 should produce zero effective Z."""
        slab: CrystalStructure = self._make_slab()
        n_orig: int = slab.cart_positions.shape[0]
        decorated: CrystalStructure = add_adsorbate_layer(
            slab=slab,
            adsorbate_positions_fractional=jnp.array([[0.5, 0.5, 0.95]]),
            adsorbate_atomic_numbers=jnp.array([14.0]),
            coverage_fraction=0.0,
        )
        ads_z: float = float(decorated.cart_positions[n_orig, 3])
        chex.assert_trees_all_close(ads_z, 0.0, atol=1e-10)
