"""
Tests for ucell.helper module.

Tests the helper functions for unit cell calculations and transformations.
"""

import os
import tempfile

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

import rheedium as rh
from rheedium.types import CrystalStructure


class TestAngleInDegrees(chex.TestCase):
    """Test angle_in_degrees function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.rng = jax.random.PRNGKey(42)
        chex.set_n_cpu_devices(1)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("orthogonal_xy", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 90.0),
        ("orthogonal_xz", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0),
        ("orthogonal_yz", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 90.0),
        ("parallel_same", [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.0),
        ("antiparallel", [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 180.0),
        ("45_degrees", [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 45.0),
        ("60_degrees", [1.0, 0.0, 0.0], [0.5, 0.866025, 0.0], 60.0),
    )
    def test_angle_calculation(
        self, v1: list, v2: list, expected_angle: float
    ) -> None:
        """Test angle calculation between various vector pairs."""
        v1_array = jnp.array(v1)
        v2_array = jnp.array(v2)

        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)
        angle = var_angle_in_degrees(v1_array, v2_array)

        chex.assert_trees_all_close(angle, expected_angle, atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_angle_random_vectors(self) -> None:
        """Test angle calculation with random vectors."""
        key1, key2 = jax.random.split(self.rng)
        v1 = jax.random.normal(key1, (3,))
        v2 = jax.random.normal(key2, (3,))

        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)
        angle = var_angle_in_degrees(v1, v2)

        # Angle should be between 0 and 180 degrees
        chex.assert_scalar_in(float(angle), 0.0, 180.0)

        # Test symmetry: angle(v1, v2) == angle(v2, v1)
        angle_reversed = var_angle_in_degrees(v2, v1)
        chex.assert_trees_all_close(angle, angle_reversed, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_angle_normalization_invariance(self) -> None:
        """Test that angle is invariant to vector magnitude."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([1.0, 1.0, 0.0])

        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)

        # Original angle
        angle1 = var_angle_in_degrees(v1, v2)

        # Scale vectors
        angle2 = var_angle_in_degrees(2.0 * v1, 3.0 * v2)

        chex.assert_trees_all_close(angle1, angle2, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_angle_2d_vectors(self) -> None:
        """Test angle calculation with 2D vectors."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)
        angle = var_angle_in_degrees(v1, v2)

        chex.assert_trees_all_close(angle, 90.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_angle_high_dimensional(self) -> None:
        """Test angle calculation with high-dimensional vectors."""
        n_dim = 10
        key1, key2 = jax.random.split(self.rng)
        v1 = jax.random.normal(key1, (n_dim,))
        v2 = jax.random.normal(key2, (n_dim,))

        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)
        angle = var_angle_in_degrees(v1, v2)

        chex.assert_scalar_in(float(angle), 0.0, 180.0)


class TestComputeLengthsAngles(chex.TestCase):
    """Test compute_lengths_angles function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.rng = jax.random.PRNGKey(42)
        chex.set_n_cpu_devices(1)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "cubic",
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            [5.0, 5.0, 5.0],
            [90.0, 90.0, 90.0],
        ),
        (
            "orthorhombic",
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
            [3.0, 4.0, 5.0],
            [90.0, 90.0, 90.0],
        ),
        (
            "monoclinic",
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 0.0, 5.0]],
            [3.0, 4.0, 5.099019513592784],
            [90.0, 90.0, 78.69006752597979],
        ),
    )
    def test_compute_lengths_angles_known_cells(
        self, vectors: list, expected_lengths: list, expected_angles: list
    ) -> None:
        """Test length and angle computation for known unit cells."""
        vectors_array = jnp.array(vectors)

        var_compute_lengths_angles = self.variant(
            rh.ucell.compute_lengths_angles
        )
        lengths, angles = var_compute_lengths_angles(vectors_array)

        chex.assert_trees_all_close(
            lengths, jnp.array(expected_lengths), atol=1e-5
        )
        chex.assert_trees_all_close(
            angles, jnp.array(expected_angles), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_lengths_angles_random(self) -> None:
        """Test length and angle computation with random vectors."""
        vectors = jax.random.normal(self.rng, (3, 3))

        var_compute_lengths_angles = self.variant(
            rh.ucell.compute_lengths_angles
        )
        lengths, angles = var_compute_lengths_angles(vectors)

        # Check shapes
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))

        # Lengths should be positive
        chex.assert_trees_all_positive(lengths)

        # Angles should be between 0 and 180
        for angle in angles:
            chex.assert_scalar_in(float(angle), 0.0, 180.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_lengths_angles_consistency(self) -> None:
        """Test consistency with individual calculations."""
        vectors = jnp.array(
            [[3.0, 1.0, 0.5], [0.5, 4.0, 1.0], [1.0, 0.5, 5.0]]
        )

        var_compute_lengths_angles = self.variant(
            rh.ucell.compute_lengths_angles
        )
        lengths, angles = var_compute_lengths_angles(vectors)

        # Compute lengths manually
        expected_lengths = jnp.array(
            [
                jnp.linalg.norm(vectors[0]),
                jnp.linalg.norm(vectors[1]),
                jnp.linalg.norm(vectors[2]),
            ]
        )
        chex.assert_trees_all_close(lengths, expected_lengths, atol=1e-10)

        # Compute angles manually
        var_angle_in_degrees = self.variant(rh.ucell.angle_in_degrees)
        expected_angles = jnp.array(
            [
                var_angle_in_degrees(vectors[0], vectors[1]),
                var_angle_in_degrees(vectors[1], vectors[2]),
                var_angle_in_degrees(vectors[2], vectors[0]),
            ]
        )
        chex.assert_trees_all_close(angles, expected_angles, atol=1e-10)


class TestParseCifAndScrape(chex.TestCase):
    """Test parse_cif_and_scrape function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.rng = jax.random.PRNGKey(42)
        chex.set_n_cpu_devices(1)

        # Create a temporary test CIF file
        self.test_cif_content = """
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.0 0.0
Si2 Si 0.5 0.5 0.1
Si3 Si 0.0 0.0 0.2
Si4 Si 0.5 0.5 0.3
Si5 Si 0.0 0.0 0.4
Si6 Si 0.5 0.5 0.5
Si7 Si 0.0 0.0 0.6
Si8 Si 0.5 0.5 0.7
Si9 Si 0.0 0.0 0.8
Si10 Si 0.5 0.5 0.9
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cif", delete=False
        ) as self.temp_file:
            self.temp_file.write(self.test_cif_content)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        os.unlink(self.temp_file.name)
        super().tearDown()

    @chex.variants(without_jit=True, with_jit=False)
    def test_parse_cif_and_scrape_basic(self) -> None:
        """Test basic CIF parsing and atom scraping."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness_xyz = jnp.array([5.0, 5.0, 3.0])

        var_parse_cif_and_scrape = self.variant(rh.ucell.parse_cif_and_scrape)
        filtered_crystal = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis, thickness_xyz
        )

        # Check that we got a CrystalStructure
        chex.assert_type(filtered_crystal, CrystalStructure)

        # Check that atoms were filtered (should have fewer than 10)
        n_atoms = filtered_crystal.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        chex.assert_scalar_in(int(n_atoms), 1, 9)  # Between 1 and 9 atoms

    @chex.variants(without_jit=True, with_jit=False)
    @parameterized.named_parameters(
        ("z_axis", [0.0, 0.0, 1.0], [5.0, 5.0, 2.0]),
        ("x_axis", [1.0, 0.0, 0.0], [2.0, 5.0, 5.0]),
        ("y_axis", [0.0, 1.0, 0.0], [5.0, 2.0, 5.0]),
        ("diagonal", [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
    )
    def test_parse_cif_and_scrape_different_axes(
        self, zone_axis: list, thickness: list
    ) -> None:
        """Test scraping along different zone axes."""
        zone_axis_array = jnp.array(zone_axis)
        thickness_array = jnp.array(thickness)

        var_parse_cif_and_scrape = self.variant(rh.ucell.parse_cif_and_scrape)
        filtered_crystal = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis_array, thickness_array
        )

        # Check valid crystal structure
        chex.assert_type(filtered_crystal, CrystalStructure)

        # Check we have some atoms
        n_atoms = filtered_crystal.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))

        # Check cell parameters are preserved
        chex.assert_trees_all_equal(
            filtered_crystal.cell_lengths, jnp.array([5.0, 5.0, 10.0])
        )
        chex.assert_trees_all_equal(
            filtered_crystal.cell_angles, jnp.array([90.0, 90.0, 90.0])
        )

    @chex.variants(without_jit=True, with_jit=False)
    def test_parse_cif_and_scrape_multiple_thicknesses(self) -> None:
        """Test with different thickness values."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])

        var_parse_cif_and_scrape = self.variant(rh.ucell.parse_cif_and_scrape)

        # Test with different thickness values
        thickness1 = jnp.array([5.0, 5.0, 2.0])
        thickness2 = jnp.array([5.0, 5.0, 4.0])

        crystal1 = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis, thickness1
        )
        crystal2 = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis, thickness2
        )

        # Both should produce valid crystals
        chex.assert_type(crystal1, CrystalStructure)
        chex.assert_type(crystal2, CrystalStructure)

        # Crystal2 should have more or equal atoms than crystal1
        n_atoms1 = crystal1.cart_positions.shape[0]
        n_atoms2 = crystal2.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms1))
        chex.assert_scalar_positive(int(n_atoms2))
        # More thickness should include more or equal atoms
        assert int(n_atoms2) >= int(n_atoms1), (
            "More thickness should include more or equal atoms"
        )

    @chex.variants(without_jit=True, with_jit=False)
    def test_parse_cif_and_scrape_thin_slice(self) -> None:
        """Test with very thin slice thickness."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness_xyz = jnp.array([5.0, 5.0, 0.5])  # Very thin slice

        var_parse_cif_and_scrape = self.variant(rh.ucell.parse_cif_and_scrape)
        filtered_crystal = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis, thickness_xyz
        )

        # Should still have at least one atom
        n_atoms = filtered_crystal.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))

    @chex.variants(without_jit=True, with_jit=False)
    def test_parse_cif_and_scrape_thick_slice(self) -> None:
        """Test with very thick slice (should include all atoms)."""
        zone_axis = jnp.array([0.0, 0.0, 1.0])
        thickness_xyz = jnp.array([10.0, 10.0, 20.0])  # Very thick slice

        var_parse_cif_and_scrape = self.variant(rh.ucell.parse_cif_and_scrape)
        filtered_crystal = var_parse_cif_and_scrape(
            self.temp_file.name, zone_axis, thickness_xyz
        )

        # Should include all 10 atoms
        n_atoms = filtered_crystal.cart_positions.shape[0]
        assert int(n_atoms) == 10, f"Expected 10 atoms, got {n_atoms}"


if __name__ == "__main__":
    chex.TestCase.main()
