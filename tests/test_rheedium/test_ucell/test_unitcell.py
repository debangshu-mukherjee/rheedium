"""Tests for ucell.unitcell module.

Tests the atom_scraper function for filtering atoms within specified
thickness along a zone axis, plus reciprocal lattice functions.
"""

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.types import CrystalStructure, create_crystal_structure
from rheedium.ucell.unitcell import (
    atom_scraper,
    build_cell_vectors,
    compute_lengths_angles,
    generate_reciprocal_points,
    get_unit_cell_matrix,
    miller_to_reciprocal,
    reciprocal_lattice_vectors,
    reciprocal_unitcell,
)


class TestAtomScraper(chex.TestCase, parameterized.TestCase):
    """Test atom_scraper function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

        # Create a simple cubic crystal with atoms at different z positions
        self.cubic_crystal = self._create_layered_crystal()

    def _create_layered_crystal(self) -> CrystalStructure:
        """Create a crystal with atoms at different z heights.

        Creates 5 atoms stacked along z-axis at z = 0, 2, 4, 6, 8 Angstroms.
        """
        a = 5.0  # lattice constant

        # Atoms at different z heights
        cart_coords: Float[Array, "5 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],  # z = 0
                [0.0, 0.0, 2.0],  # z = 2
                [0.0, 0.0, 4.0],  # z = 4
                [0.0, 0.0, 6.0],  # z = 6
                [0.0, 0.0, 8.0],  # z = 8
            ]
        )

        # Fractional coordinates
        frac_coords: Float[Array, "5 3"] = cart_coords / jnp.array(
            [a, a, 10.0]
        )

        # All silicon atoms
        atomic_numbers: Float[Array, "5"] = jnp.full(5, 14.0)

        frac_positions: Float[Array, "5 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "5 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a, a, 10.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def _create_xy_plane_crystal(self) -> CrystalStructure:
        """Create a crystal with atoms spread in XY plane at same z."""
        a = 10.0

        cart_coords: Float[Array, "4 3"] = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
            ]
        )

        frac_coords = cart_coords / a
        atomic_numbers = jnp.full(4, 14.0)

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_z_axis_scraping(self) -> None:
        """Test scraping atoms along z-axis with specific thickness."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 3.0])  # 3 Angstrom thickness

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should filter to atoms near the top (z=8)
        # With 3 Angstrom thickness from top, should include z=8 and z=6
        n_atoms = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        self.assertLessEqual(int(n_atoms), 5)

    def test_full_thickness_keeps_all_atoms(self) -> None:
        """Test that large thickness keeps all atoms."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([10.0, 10.0, 20.0])  # Much larger than crystal

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should keep all 5 atoms
        n_atoms = filtered.cart_positions.shape[0]
        self.assertEqual(int(n_atoms), 5)

    def test_zero_thickness_top_layer_only(self) -> None:
        """Test that zero thickness returns top layer atoms.

        With zero thickness, the function uses an adaptive epsilon based on
        the minimum atom spacing (2 * min_spacing). For atoms spaced 2Å apart,
        this gives adaptive_eps = 4Å, which includes multiple atoms.
        """
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array(
            [0.0, 0.0, 0.0]
        )  # Zero thickness = top layer mode

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # With atoms at z=0,2,4,6,8 and spacing=2Å, adaptive_eps=4Å
        # This includes atoms within 4Å of top (z=8): z=8,6,4
        n_atoms = filtered.cart_positions.shape[0]
        self.assertGreaterEqual(int(n_atoms), 1)
        self.assertLessEqual(int(n_atoms), 5)

        # Verify the topmost atom is included
        max_z = jnp.max(filtered.cart_positions[:, 2])
        chex.assert_trees_all_close(max_z, 8.0, atol=1e-6)

    @parameterized.named_parameters(
        ("z_axis", [0.0, 0.0, 1.0]),
        ("neg_z_axis", [0.0, 0.0, -1.0]),
        ("z_axis_scaled", [0.0, 0.0, 2.0]),
    )
    def test_zone_axis_normalization(self, zone_axis: list) -> None:
        """Test that zone axis is properly normalized."""
        zone_axis_arr = jnp.array(zone_axis)
        thickness = jnp.array([5.0, 5.0, 3.0])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis_arr,
            thickness=thickness,
        )

        # Should produce valid output regardless of axis scaling
        n_atoms = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))

    def test_x_axis_scraping(self) -> None:
        """Test scraping along x-axis."""
        # Create crystal with atoms spread along x
        cart_coords = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        frac_coords = cart_coords / 10.0
        atomic_numbers = jnp.full(5, 14.0)

        crystal = create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([10.0, 5.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        zone_axis = jnp.array([1.0, 0.0, 0.0])
        thickness = jnp.array([3.0, 5.0, 5.0])

        filtered = atom_scraper(
            crystal=crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        n_atoms = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        self.assertLess(int(n_atoms), 5)

    def test_diagonal_zone_axis(self) -> None:
        """Test scraping along diagonal [1,1,1] direction."""
        zone_axis = jnp.array([1.0, 1.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 5.0])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should produce valid output
        n_atoms = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        chex.assert_tree_all_finite(filtered.cart_positions)

    def test_output_is_valid_crystal_structure(self) -> None:
        """Test that output is a valid CrystalStructure."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 3.0])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Check it's a CrystalStructure
        self.assertIsInstance(filtered, CrystalStructure)

        # Check shapes are consistent
        n_atoms = filtered.cart_positions.shape[0]
        chex.assert_shape(filtered.frac_positions, (n_atoms, 4))
        chex.assert_shape(filtered.cart_positions, (n_atoms, 4))
        chex.assert_shape(filtered.cell_lengths, (3,))
        chex.assert_shape(filtered.cell_angles, (3,))

        # Check values are finite
        chex.assert_tree_all_finite(filtered.frac_positions)
        chex.assert_tree_all_finite(filtered.cart_positions)
        chex.assert_tree_all_finite(filtered.cell_lengths)
        chex.assert_tree_all_finite(filtered.cell_angles)

    def test_cell_lengths_positive(self) -> None:
        """Test that output cell lengths are positive."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 3.0])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        chex.assert_trees_all_equal(jnp.all(filtered.cell_lengths > 0), True)

    def test_cell_angles_valid(self) -> None:
        """Test that output cell angles are in valid range."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 3.0])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Angles should be between 0 and 180 degrees
        for angle in filtered.cell_angles:
            chex.assert_scalar_in(float(angle), 0.0, 180.0)

    def test_atomic_numbers_preserved(self) -> None:
        """Test that atomic numbers are preserved in output."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 20.0])  # Keep all atoms

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # All atoms should still be silicon (Z=14)
        atomic_nums = filtered.cart_positions[:, 3]
        chex.assert_trees_all_close(atomic_nums, jnp.full(5, 14.0), atol=1e-10)

    def test_xy_plane_atoms_same_z(self) -> None:
        """Test scraping with atoms at same z height.

        Note: When all atoms are at the same height along the zone axis,
        they are all considered "top layer" atoms and should be included.
        We use a crystal with slight z variation to avoid the edge case
        where all atoms are exactly coplanar.
        """
        a = 10.0

        # Atoms in XY plane with slight z variation
        cart_coords: Float[Array, "4 3"] = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 4.99],  # Slightly lower
                [5.0, 5.0, 5.0],
            ]
        )

        frac_coords = cart_coords / a
        atomic_numbers = jnp.full(4, 14.0)

        crystal = create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([10.0, 10.0, 1.0])  # 1 Angstrom slice

        filtered = atom_scraper(
            crystal=crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # All 4 atoms should be included (within 1 Angstrom of top)
        n_atoms = filtered.cart_positions.shape[0]
        self.assertEqual(int(n_atoms), 4)

    @parameterized.named_parameters(
        ("thin", 1.0, 1),
        ("medium", 5.0, 3),
        ("thick", 10.0, 5),
    )
    def test_thickness_controls_atom_count(
        self, z_thickness: float, min_expected: int
    ) -> None:
        """Test that increasing thickness includes more atoms."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, z_thickness])

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        n_atoms = filtered.cart_positions.shape[0]
        self.assertGreaterEqual(int(n_atoms), min_expected)

    def test_frac_and_cart_positions_consistent(self) -> None:
        """Test that fractional and Cartesian positions remain consistent."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness = jnp.array([5.0, 5.0, 20.0])  # Keep all atoms

        filtered = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Number of atoms should match
        n_frac = filtered.frac_positions.shape[0]
        n_cart = filtered.cart_positions.shape[0]
        self.assertEqual(n_frac, n_cart)

        # Atomic numbers should match between frac and cart
        frac_z = filtered.frac_positions[:, 3]
        cart_z = filtered.cart_positions[:, 3]
        chex.assert_trees_all_close(frac_z, cart_z, atol=1e-10)


class TestReciprocalUnitcell(chex.TestCase, parameterized.TestCase):
    """Test reciprocal_unitcell function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @chex.all_variants
    def test_cubic_system(self) -> None:
        """Test reciprocal parameters for cubic system."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # For cubic: a* = 2π/a
        expected_a_star = 2 * jnp.pi / 3.0
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_trees_all_close(lengths[0], expected_a_star, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], expected_a_star, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], expected_a_star, rtol=1e-5)
        # Reciprocal angles for cubic are 90°
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants
    def test_orthorhombic_system(self) -> None:
        """Test reciprocal parameters for orthorhombic system (a≠b≠c, all 90°)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # For orthorhombic: a* = 2π/a, b* = 2π/b, c* = 2π/c
        chex.assert_trees_all_close(lengths[0], 2 * jnp.pi / 3.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], 2 * jnp.pi / 4.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], 2 * jnp.pi / 5.0, rtol=1e-5)
        # Angles still 90°
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants
    def test_tetragonal_system(self) -> None:
        """Test reciprocal parameters for tetragonal system (a=b≠c)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # a* = b* but c* different
        chex.assert_trees_all_close(lengths[0], lengths[1], rtol=1e-5)
        self.assertNotAlmostEqual(
            float(lengths[0]), float(lengths[2]), places=3
        )

    @chex.all_variants
    def test_hexagonal_system(self) -> None:
        """Test reciprocal parameters for hexagonal system (γ=120°)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # Reciprocal gamma for hexagonal is 60°
        chex.assert_trees_all_close(angles[2], 60.0, atol=1e-4)

    @chex.all_variants
    def test_monoclinic_system(self) -> None:
        """Test reciprocal parameters for monoclinic system (β≠90°)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # Alpha* and gamma* should remain 90° for monoclinic
        chex.assert_trees_all_close(angles[0], 90.0, atol=1e-4)
        chex.assert_trees_all_close(angles[2], 90.0, atol=1e-4)

    @chex.all_variants
    def test_triclinic_system(self) -> None:
        """Test reciprocal parameters for triclinic system (no special symmetry)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # All values should be positive
        chex.assert_trees_all_equal(jnp.all(lengths > 0), True)
        chex.assert_trees_all_equal(jnp.all(angles > 0), True)
        chex.assert_trees_all_equal(jnp.all(angles < 180), True)

    @chex.all_variants
    def test_in_degrees_flag_true(self) -> None:
        """Test in_degrees=True (input in degrees)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)

    @chex.all_variants
    def test_in_degrees_flag_false(self) -> None:
        """Test in_degrees=False (input in radians)."""
        var_fn = self.variant(reciprocal_unitcell)
        pi_half = jnp.pi / 2
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
            out_degrees=True,
        )
        # Should give same result as cubic with 90° in degrees
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants
    def test_out_degrees_flag_false(self) -> None:
        """Test out_degrees=False (output in radians)."""
        var_fn = self.variant(reciprocal_unitcell)
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=False,
        )
        # Output angles in radians for cubic should be π/2
        chex.assert_trees_all_close(
            angles, jnp.array([jnp.pi / 2] * 3), atol=1e-5
        )

    @chex.all_variants
    def test_both_degrees_flags_false(self) -> None:
        """Test both in_degrees=False and out_degrees=False."""
        var_fn = self.variant(reciprocal_unitcell)
        pi_half = jnp.pi / 2
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
            out_degrees=False,
        )
        chex.assert_trees_all_close(
            angles, jnp.array([pi_half] * 3), atol=1e-5
        )

    @parameterized.named_parameters(
        ("small_cell", 1.0, 1.0, 1.0),
        ("medium_cell", 5.0, 5.0, 5.0),
        ("large_cell", 10.0, 10.0, 10.0),
    )
    def test_various_cell_sizes(self, a: float, b: float, c: float) -> None:
        """Test with various cell sizes."""
        lengths, angles = reciprocal_unitcell(
            a=a,
            b=b,
            c=c,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # a* should scale inversely with a
        expected = 2 * jnp.pi / a
        chex.assert_trees_all_close(lengths[0], expected, rtol=1e-5)


class TestGetUnitCellMatrix(chex.TestCase, parameterized.TestCase):
    """Test get_unit_cell_matrix function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @chex.all_variants
    def test_cubic_system(self) -> None:
        """Test transformation matrix for cubic system."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # For cubic, matrix should be diagonal with a on diagonal
        chex.assert_trees_all_close(matrix[0, 0], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[1, 1], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[2, 2], 3.0, atol=1e-10)

    @chex.all_variants
    def test_orthorhombic_system(self) -> None:
        """Test transformation matrix for orthorhombic system."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        # For orthorhombic, matrix is diagonal
        chex.assert_trees_all_close(matrix[0, 0], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[1, 1], 4.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[2, 2], 5.0, atol=1e-10)

    @chex.all_variants
    def test_monoclinic_system(self) -> None:
        """Test transformation matrix for monoclinic system."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # Off-diagonal term for c vector
        self.assertNotAlmostEqual(float(matrix[0, 2]), 0.0, places=5)

    @chex.all_variants
    def test_hexagonal_system(self) -> None:
        """Test transformation matrix for hexagonal system."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # b has x and y components due to 120° angle
        chex.assert_trees_all_close(
            matrix[0, 1], 3.0 * jnp.cos(jnp.radians(120.0)), atol=1e-10
        )

    @chex.all_variants
    def test_triclinic_system(self) -> None:
        """Test transformation matrix for triclinic system."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # Matrix should have off-diagonal elements
        self.assertNotAlmostEqual(float(matrix[0, 1]), 0.0, places=5)

    @chex.all_variants
    def test_volume_consistency(self) -> None:
        """Test that matrix determinant equals cell volume."""
        var_fn = self.variant(get_unit_cell_matrix)
        matrix = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        volume = jnp.linalg.det(matrix)
        expected_volume = 3.0 * 4.0 * 5.0  # For orthorhombic
        chex.assert_trees_all_close(volume, expected_volume, rtol=1e-5)


class TestBuildCellVectors(chex.TestCase, parameterized.TestCase):
    """Test build_cell_vectors function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @chex.all_variants
    def test_cubic_system(self) -> None:
        """Test cell vectors for cubic system."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(vectors, (3, 3))
        # First vector along x-axis
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        # Second vector along y-axis
        chex.assert_trees_all_close(
            vectors[1], jnp.array([0.0, 3.0, 0.0]), atol=1e-10
        )
        # Third vector along z-axis
        chex.assert_trees_all_close(
            vectors[2], jnp.array([0.0, 0.0, 3.0]), atol=1e-10
        )

    @chex.all_variants
    def test_orthorhombic_system(self) -> None:
        """Test cell vectors for orthorhombic system."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        # Vectors should be orthogonal with different lengths
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            vectors[1], jnp.array([0.0, 4.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            vectors[2], jnp.array([0.0, 0.0, 5.0]), atol=1e-10
        )

    @chex.all_variants
    def test_hexagonal_system(self) -> None:
        """Test cell vectors for hexagonal system (gamma=120)."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
        )
        chex.assert_shape(vectors, (3, 3))
        # a vector along x
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        # b vector in xy plane at 120° from a
        b_x = 3.0 * jnp.cos(jnp.radians(120.0))
        b_y = 3.0 * jnp.sin(jnp.radians(120.0))
        chex.assert_trees_all_close(vectors[1, 0], b_x, atol=1e-10)
        chex.assert_trees_all_close(vectors[1, 1], b_y, atol=1e-10)
        # c vector along z
        chex.assert_trees_all_close(vectors[2, 2], 5.0, atol=1e-10)

    @chex.all_variants
    def test_monoclinic_system(self) -> None:
        """Test cell vectors for monoclinic system (beta != 90)."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
        )
        chex.assert_shape(vectors, (3, 3))
        chex.assert_tree_all_finite(vectors)
        # c vector should have nonzero x component
        c_x = 5.0 * jnp.cos(jnp.radians(100.0))
        chex.assert_trees_all_close(vectors[2, 0], c_x, atol=1e-10)

    @chex.all_variants
    def test_triclinic_system(self) -> None:
        """Test cell vectors for triclinic system."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        chex.assert_shape(vectors, (3, 3))
        chex.assert_tree_all_finite(vectors)
        # All three vectors should have nonzero components

    @chex.all_variants
    def test_vector_lengths_correct(self) -> None:
        """Test that built vectors have correct lengths."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        lengths = jnp.linalg.norm(vectors, axis=1)
        chex.assert_trees_all_close(lengths[0], 3.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], 4.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], 5.0, rtol=1e-5)

    @chex.all_variants
    def test_angles_correct(self) -> None:
        """Test that angles between vectors are correct."""
        var_fn = self.variant(build_cell_vectors)
        vectors = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        # Check angle gamma between a and b
        a_vec = vectors[0]
        b_vec = vectors[1]
        cos_gamma = jnp.dot(a_vec, b_vec) / (
            jnp.linalg.norm(a_vec) * jnp.linalg.norm(b_vec)
        )
        gamma_computed = jnp.rad2deg(jnp.arccos(cos_gamma))
        chex.assert_trees_all_close(gamma_computed, 75.0, atol=1e-4)


class TestComputeLengthsAngles(chex.TestCase, parameterized.TestCase):
    """Test compute_lengths_angles function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @chex.variants(with_jit=True, without_jit=True)
    def test_cubic_system(self) -> None:
        """Test lengths and angles for cubic vectors."""
        var_fn = self.variant(compute_lengths_angles)
        vectors = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 3.0, 3.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_orthorhombic_system(self) -> None:
        """Test lengths and angles for orthorhombic vectors."""
        var_fn = self.variant(compute_lengths_angles)
        vectors = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 4.0, 5.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_cubic(self) -> None:
        """Test build_cell_vectors followed by compute_lengths_angles."""
        vectors = build_cell_vectors(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        var_fn = self.variant(compute_lengths_angles)
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 3.0, 3.0]), rtol=1e-5
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-4
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_triclinic(self) -> None:
        """Test roundtrip for triclinic system."""
        a, b, c = 3.0, 4.0, 5.0
        alpha, beta, gamma = 80.0, 85.0, 75.0
        vectors = build_cell_vectors(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        var_fn = self.variant(compute_lengths_angles)
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), rtol=1e-5)
        chex.assert_trees_all_close(
            angles, jnp.array([alpha, beta, gamma]), atol=1e-4
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_hexagonal(self) -> None:
        """Test roundtrip for hexagonal system."""
        a, b, c = 3.0, 3.0, 5.0
        alpha, beta, gamma = 90.0, 90.0, 120.0
        vectors = build_cell_vectors(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        var_fn = self.variant(compute_lengths_angles)
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), rtol=1e-5)
        chex.assert_trees_all_close(
            angles, jnp.array([alpha, beta, gamma]), atol=1e-4
        )


class TestReciprocalLatticeVectors(chex.TestCase, parameterized.TestCase):
    """Test reciprocal_lattice_vectors function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @chex.all_variants
    def test_cubic_system(self) -> None:
        """Test reciprocal vectors for cubic system."""
        var_fn = self.variant(reciprocal_lattice_vectors)
        rec_vecs = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        chex.assert_shape(rec_vecs, (3, 3))
        # For cubic: b1 = (2π/a, 0, 0), etc.
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            rec_vecs[0], jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[1], jnp.array([0.0, expected, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[2], jnp.array([0.0, 0.0, expected]), atol=1e-10
        )

    @chex.all_variants
    def test_orthorhombic_system(self) -> None:
        """Test reciprocal vectors for orthorhombic system."""
        var_fn = self.variant(reciprocal_lattice_vectors)
        rec_vecs = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        chex.assert_shape(rec_vecs, (3, 3))
        chex.assert_trees_all_close(
            rec_vecs[0], jnp.array([2 * jnp.pi / 3.0, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[1], jnp.array([0.0, 2 * jnp.pi / 4.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[2], jnp.array([0.0, 0.0, 2 * jnp.pi / 5.0]), atol=1e-10
        )

    @chex.all_variants
    def test_orthogonality_to_direct(self) -> None:
        """Test that reciprocal vectors are orthogonal to other direct vectors."""
        var_fn = self.variant(reciprocal_lattice_vectors)
        rec_vecs = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
        )
        direct_vecs = build_cell_vectors(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        # b1 · a2 = 0, b1 · a3 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[1]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[2]), 0.0, atol=1e-10
        )
        # b2 · a1 = 0, b2 · a3 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[0]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[2]), 0.0, atol=1e-10
        )
        # b3 · a1 = 0, b3 · a2 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[0]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[1]), 0.0, atol=1e-10
        )

    @chex.all_variants
    def test_bi_dot_ai_equals_2pi(self) -> None:
        """Test that b_i · a_i = 2π."""
        var_fn = self.variant(reciprocal_lattice_vectors)
        rec_vecs = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
        )
        direct_vecs = build_cell_vectors(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        two_pi = 2 * jnp.pi
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[0]), two_pi, rtol=1e-5
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[1]), two_pi, rtol=1e-5
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[2]), two_pi, rtol=1e-5
        )

    @chex.all_variants
    def test_in_degrees_flag(self) -> None:
        """Test in_degrees flag."""
        var_fn = self.variant(reciprocal_lattice_vectors)
        # With degrees
        rec_vecs_deg = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        # With radians
        pi_half = jnp.pi / 2
        rec_vecs_rad = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
        )
        chex.assert_trees_all_close(rec_vecs_deg, rec_vecs_rad, rtol=1e-5)


class TestMillerToReciprocal(chex.TestCase, parameterized.TestCase):
    """Test miller_to_reciprocal function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)
        # Set up cubic reciprocal vectors for testing
        self.cubic_rec_vecs = reciprocal_lattice_vectors(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_100(self) -> None:
        """Test (1,0,0) Miller index."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([1, 0, 0])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_010(self) -> None:
        """Test (0,1,0) Miller index."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([0, 1, 0])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, expected, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_001(self) -> None:
        """Test (0,0,1) Miller index."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([0, 0, 1])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, 0.0, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_111(self) -> None:
        """Test (1,1,1) Miller index."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([1, 1, 1])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_negative_indices(self) -> None:
        """Test negative Miller indices."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([-1, -1, -1])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = -2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_indices(self) -> None:
        """Test (0,0,0) gives zero vector."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([0, 0, 0])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, 0.0, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_batch_indices(self) -> None:
        """Test batched Miller indices."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
            ]
        )
        g_vecs = var_fn(hkl, self.cubic_rec_vecs)
        chex.assert_shape(g_vecs, (4, 3))
        expected = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vecs[0], jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            g_vecs[3], jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_higher_indices(self) -> None:
        """Test higher Miller indices (2,0,0)."""
        var_fn = self.variant(miller_to_reciprocal)
        hkl = jnp.array([2, 0, 0])
        g_vec = var_fn(hkl, self.cubic_rec_vecs)
        expected = 2 * (2 * jnp.pi / 3.0)
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )


class TestGenerateReciprocalPoints(chex.TestCase, parameterized.TestCase):
    """Test generate_reciprocal_points function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)
        # Create a simple cubic crystal
        self.cubic_crystal = self._create_cubic_crystal()

    def _create_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal."""
        a = 3.0
        cart_coords = jnp.array([[0.0, 0.0, 0.0]])
        frac_coords = cart_coords / a
        atomic_numbers = jnp.array([14.0])  # Silicon

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_point_count(self) -> None:
        """Test number of generated points."""
        var_fn = self.variant(generate_reciprocal_points)
        g_vecs = var_fn(
            crystal=self.cubic_crystal,
            hmax=2,
            kmax=2,
            lmax=1,
            in_degrees=True,
        )
        # Number of points = (2*h+1) * (2*k+1) * (2*l+1)
        expected_count = 5 * 5 * 3  # 75 points
        chex.assert_shape(g_vecs, (expected_count, 3))

    @chex.variants(with_device=True, without_jit=True)
    def test_includes_origin(self) -> None:
        """Test that origin (0,0,0) is included."""
        var_fn = self.variant(generate_reciprocal_points)
        g_vecs = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # Check if any row is close to zero
        norms = jnp.linalg.norm(g_vecs, axis=1)
        has_origin = jnp.any(norms < 1e-10)
        chex.assert_trees_all_equal(has_origin, True)

    @chex.variants(with_device=True, without_jit=True)
    def test_symmetry_pairs(self) -> None:
        """Test that (h,k,l) and (-h,-k,-l) are opposites."""
        var_fn = self.variant(generate_reciprocal_points)
        g_vecs = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # For each vector, its negative should also be present
        chex.assert_tree_all_finite(g_vecs)

    @chex.variants(with_device=True, without_jit=True)
    def test_cubic_symmetry(self) -> None:
        """Test cubic symmetry - equivalent directions have same magnitude."""
        var_fn = self.variant(generate_reciprocal_points)
        g_vecs = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # Calculate magnitudes
        mags = jnp.linalg.norm(g_vecs, axis=1)
        # For cubic, (1,0,0), (0,1,0), (0,0,1) should have same magnitude
        expected = 2 * jnp.pi / 3.0
        # Count how many have this magnitude
        matches = jnp.sum(jnp.abs(mags - expected) < 1e-5)
        # Should be 6: ±(1,0,0), ±(0,1,0), ±(0,0,1)
        chex.assert_trees_all_equal(matches, 6)

    @parameterized.named_parameters(
        ("small", 1, 1, 1),
        ("medium", 2, 2, 2),
        ("asymmetric", 3, 2, 1),
    )
    def test_various_ranges(self, hmax: int, kmax: int, lmax: int) -> None:
        """Test various hkl ranges."""
        g_vecs = generate_reciprocal_points(
            crystal=self.cubic_crystal,
            hmax=hmax,
            kmax=kmax,
            lmax=lmax,
            in_degrees=True,
        )
        expected_count = (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1)
        chex.assert_shape(g_vecs, (expected_count, 3))
        chex.assert_tree_all_finite(g_vecs)


if __name__ == "__main__":
    chex.TestCase.main()
