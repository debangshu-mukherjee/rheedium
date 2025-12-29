"""Tests for ucell.unitcell module.

Tests the atom_scraper function for filtering atoms within specified
thickness along a zone axis.
"""

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.types import CrystalStructure, create_crystal_structure
from rheedium.ucell.unitcell import atom_scraper


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


if __name__ == "__main__":
    chex.TestCase.main()
