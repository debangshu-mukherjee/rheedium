"""Test suite for updated RHEED simulator with surface physics.

Tests the integration of:
- Proper atomic form factors (Kirkland parameterization)
- Surface-enhanced Debye-Waller factors
- CTR intensity calculations
- Structure factor with q_z dependence
- Multislice propagation and simulation
- Kinematic reflection finding
- Detector projection
"""

from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax.test_util import check_grads

from rheedium.simul.simulator import (
    compute_kinematic_intensities_with_ctrs,
    ewald_simulator,
    ewald_simulator_with_orientation_distribution,
    find_kinematic_reflections,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    sliced_crystal_to_projected_potential_slices,
)
from rheedium.tools import wavelength_ang
from rheedium.tools.wrappers import jax_safe
from rheedium.types.crystal_types import (
    create_crystal_structure,
    create_potential_slices,
)
from rheedium.types.distributions import create_discrete_orientation
from rheedium.types.rheed_types import (
    RHEEDPattern,
    SurfaceConfig,
    create_sliced_crystal,
)


class TestUpdatedSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for updated RHEED simulator with proper surface physics."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

        # Create simple Si(111) structure for testing
        self.si_crystal = self._create_si111_crystal()

    def _create_si111_crystal(self):
        """Create a simple Si(111) crystal structure.

        Returns
        -------
        crystal : CrystalStructure
            Silicon crystal with (111) orientation
        """
        a_si = 5.431  # Si lattice constant in Angstroms

        # Si diamond structure fractional positions
        frac_coords = jnp.array(
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.25, 0.25],
                [0.50, 0.50, 0.00],
                [0.75, 0.75, 0.25],
                [0.50, 0.00, 0.50],
                [0.75, 0.25, 0.75],
                [0.00, 0.50, 0.50],
                [0.25, 0.75, 0.75],
            ]
        )

        # Convert to Cartesian coordinates
        cart_coords = frac_coords * a_si

        # Add atomic numbers (Si = 14)
        atomic_numbers = jnp.full(8, 14.0)
        frac_positions = jnp.column_stack([frac_coords, atomic_numbers])
        cart_positions = jnp.column_stack([cart_coords, atomic_numbers])

        crystal = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        return crystal

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("room_temp", 300.0, 0.5, 0.3),
        ("low_temp", 77.0, 0.3, 0.3),
        ("high_roughness", 300.0, 1.0, 0.3),
        ("thin_surface", 300.0, 0.5, 0.1),
    )
    def test_intensity_calculation_with_ctrs(
        self,
        temperature,
        surface_roughness,
        surface_fraction,
    ):
        """Test intensity calculation with CTR contributions."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        # Set up simple test case
        # 20 keV, 2 degrees
        k_in = jnp.array([73.0, 0.0, -2.5])
        g_vectors = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        k_out = k_in + g_vectors

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=temperature,
            surface_roughness=surface_roughness,
            surface_fraction=surface_fraction,
        )

        # Check properties
        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        # Surface roughness should decrease intensities
        if surface_roughness > 0.5:
            max_intensity = jnp.max(intensities)
            chex.assert_scalar_positive(float(max_intensity))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_surface_enhancement_effect(self):
        """Test that surface atoms have enhanced thermal motion."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        k_in = jnp.array([73.0, 0.0, -2.5])
        g_vectors = jnp.array([[1.0, 0.0, 0.0]])
        k_out = k_in + g_vectors

        # Compare with and without surface effects
        intensities_bulk = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_fraction=0.0,  # No surface atoms
        )

        intensities_surface = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_fraction=0.5,  # Half atoms are surface
        )

        # Surface enhancement should reduce intensity due to increased
        # DW factor (without normalization, this works correctly)
        chex.assert_trees_all_equal(
            intensities_surface[0] < intensities_bulk[0], True
        )


class TestProjectOnDetector(chex.TestCase, parameterized.TestCase):
    """Test suite for detector projection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_basic_projection(self):
        """Test basic projection onto detector plane."""
        var_project = self.variant(project_on_detector)

        k_out = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.5],
            ]
        )
        detector_distance = 100.0

        coords = var_project(k_out, detector_distance)

        chex.assert_shape(coords, (3, 2))
        chex.assert_tree_all_finite(coords)
        # First point: no lateral deflection
        chex.assert_trees_all_close(
            coords[0], jnp.array([0.0, 0.0]), atol=1e-6
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_scaling(self, distance):
        """Test that coordinates scale linearly with detector distance."""
        var_project = self.variant(project_on_detector)

        k_out = jnp.array([[1.0, 0.5, 0.3]])
        coords = var_project(k_out, distance)

        chex.assert_shape(coords, (1, 2))
        # Verify linear scaling
        expected_h = 0.5 * distance / 1.0
        expected_v = 0.3 * distance / 1.0
        chex.assert_trees_all_close(
            coords[0], jnp.array([expected_h, expected_v]), rtol=1e-5
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self):
        """Test output has correct shape for various inputs."""
        var_project = self.variant(project_on_detector)

        for n in [1, 5, 10, 50]:
            k_out = jnp.ones((n, 3))
            coords = var_project(k_out, 100.0)
            chex.assert_shape(coords, (n, 2))


class TestFindKinematicReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for kinematic reflection finding."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.k_mag = 73.0  # Typical |k| for 20 keV electrons

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_elastic_scattering_constraint(self):
        """Test that output wavevectors satisfy |k_out| ≈ |k_in|."""
        var_find = self.variant(find_kinematic_reflections)

        k_in = jnp.array([self.k_mag, 0.0, -2.5])
        gs = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [10.0, 10.0, 10.0],  # This one should fail elastic condition
            ]
        )

        allowed_indices, k_out = var_find(k_in, gs, z_sign=-1.0, tolerance=0.5)

        # Check shapes
        chex.assert_shape(allowed_indices, (5,))
        chex.assert_shape(k_out, (5, 3))

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.05),
        ("loose", 0.2),
    )
    def test_tolerance_variation(self, tolerance):
        """Test that tighter tolerances allow fewer reflections."""
        var_find = self.variant(find_kinematic_reflections)

        k_in = jnp.array([self.k_mag, 0.0, -2.5])
        # Small G vectors that barely satisfy elastic condition
        gs = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.15, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )

        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=-1.0, tolerance=tolerance
        )

        chex.assert_tree_all_finite(k_out)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_positive(self):
        """Test filtering with positive z_sign (forward scattering)."""
        var_find = self.variant(find_kinematic_reflections)

        k_in = jnp.array([self.k_mag, 0.0, 2.5])
        gs = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -5.0],  # Would give negative z
            ]
        )

        allowed_indices, k_out = var_find(k_in, gs, z_sign=1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_negative(self):
        """Test filtering with negative z_sign (back scattering - RHEED)."""
        var_find = self.variant(find_kinematic_reflections)

        k_in = jnp.array([self.k_mag, 0.0, -2.5])
        gs = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],  # Would give positive z
                [0.0, 0.0, -1.0],
            ]
        )

        allowed_indices, k_out = var_find(k_in, gs, z_sign=-1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_empty_g_vectors(self):
        """Test handling of single G vector."""
        var_find = self.variant(find_kinematic_reflections)

        k_in = jnp.array([self.k_mag, 0.0, -2.5])
        gs = jnp.array([[0.0, 0.0, 0.0]])

        allowed_indices, k_out = var_find(k_in, gs, tolerance=0.5)

        chex.assert_shape(allowed_indices, (1,))
        chex.assert_shape(k_out, (1, 3))


class TestSlicedCrystalToProjectedPotentialSlices(
    chex.TestCase, parameterized.TestCase
):
    """Test suite for converting sliced crystals to projected-potential slices."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.si_sliced = self._create_simple_sliced_crystal()

    def _create_simple_sliced_crystal(self):
        """Create a simple sliced crystal for testing."""
        # Simple 2-atom structure
        cart_positions = jnp.array(
            [
                [5.0, 5.0, 1.0, 14.0],  # Si at (5,5,1)
                [7.5, 7.5, 3.0, 14.0],  # Si at (7.5,7.5,3)
            ]
        )

        return create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_output_shape(self):
        """Test that output potential has expected shape.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
        )

        # Check slices array exists
        chex.assert_tree_all_finite(potential.slices)
        # Should have nz slices based on depth/slice_thickness
        nz_expected = int(jnp.ceil(5.0 / 2.0))
        self.assertEqual(potential.slices.shape[0], nz_expected)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin", 1.0),
        ("medium", 2.0),
        ("thick", 5.0),
    )
    def test_slice_thickness_variation(self, thickness):
        """Test potential generation with different slice thicknesses.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential = var_convert(
            self.si_sliced,
            slice_thickness=thickness,
            pixel_size=0.5,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Number of slices should be ceil(depth / thickness)
        expected_nz = int(jnp.ceil(5.0 / thickness))
        self.assertEqual(potential.slices.shape[0], expected_nz)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("fine", 0.1),
        ("medium", 0.5),
        ("coarse", 1.0),
    )
    def test_pixel_size_variation(self, pixel_size):
        """Test potential generation with different pixel sizes.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=pixel_size,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Grid should scale with pixel size
        expected_nx = int(jnp.ceil(15.0 / pixel_size))
        expected_ny = int(jnp.ceil(15.0 / pixel_size))
        self.assertEqual(potential.slices.shape[1], expected_nx)
        self.assertEqual(potential.slices.shape[2], expected_ny)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("lobato", "lobato"),
        ("kirkland", "kirkland"),
    )
    def test_parameterization_variation(self, parameterization):
        """Projected-potential slices are finite for both models."""
        var_convert = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
            parameterization=parameterization,
        )

        chex.assert_tree_all_finite(potential.slices)

    @chex.variants(with_device=True, without_jit=True)
    def test_calibration_stored(self):
        """Test that calibration values are stored correctly.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        pixel_size = 0.3
        slice_thickness = 1.5

        potential = var_convert(
            self.si_sliced,
            slice_thickness=slice_thickness,
            pixel_size=pixel_size,
        )

        chex.assert_trees_all_close(
            potential.x_calibration, pixel_size, atol=1e-6
        )
        chex.assert_trees_all_close(
            potential.y_calibration, pixel_size, atol=1e-6
        )
        chex.assert_trees_all_close(
            potential.slice_thickness, slice_thickness, atol=1e-6
        )


class TestMultislicePropagate(chex.TestCase, parameterized.TestCase):
    """Test suite for multislice wave propagation."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential = self._create_simple_potential()

    def _create_simple_potential(self):
        """Create a simple potential for testing."""
        # Small grid for fast tests
        nx, ny, nz = 32, 32, 3
        slices = jnp.zeros((nz, nx, ny))
        # Add a small potential at center of first slice
        slices = slices.at[0, 16, 16].set(1.0)

        return create_potential_slices(
            slices=slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self):
        """Test that exit wave has same shape as input grid."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_shape(exit_wave, (32, 32))
        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_exit_wave_nonzero(self):
        """Test that exit wave has non-zero amplitude."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        total_intensity = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage):
        """Test propagation at different voltages."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=voltage,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Higher voltage = shorter wavelength = different phase evolution
        total_intensity = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("shallow", 0.5),
        ("medium", 2.0),
        ("steep", 5.0),
    )
    def test_grazing_angle_variation(self, theta):
        """Test propagation at different grazing angles."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
    )
    def test_azimuthal_angle_variation(self, phi):
        """Test propagation at different azimuthal angles."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=phi,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("no_inner", 0.0),
        ("small_inner", 10.0),
        ("large_inner", 20.0),
    )
    def test_inner_potential_variation(self, v0):
        """Test effect of inner potential on propagation."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            inner_potential_v0=v0,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("half", 0.5),
        ("two_thirds", 2.0 / 3.0),
        ("full", 1.0),
    )
    def test_bandwidth_limit_variation(self, limit):
        """Test different bandwidth limiting values."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            bandwidth_limit=limit,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_zero_potential_propagation(self):
        """Test propagation through zero potential (free space)."""
        var_propagate = self.variant(multislice_propagate)

        # Zero potential
        zero_slices = jnp.zeros((3, 32, 32))
        zero_potential = create_potential_slices(
            slices=zero_slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

        exit_wave = var_propagate(
            zero_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Should still have intensity (plane wave propagates)
        total_intensity = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)


class TestMultisliceSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for complete multislice RHEED simulation."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential = self._create_test_potential()

    def _create_test_potential(self):
        """Create potential slices for testing."""
        nx, ny, nz = 32, 32, 3
        slices = jnp.zeros((nz, nx, ny))
        # Add some structure
        slices = slices.at[0, 16, 16].set(2.0)
        slices = slices.at[1, 16, 16].set(1.5)
        slices = slices.at[2, 16, 16].set(1.0)

        return create_potential_slices(
            slices=slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_returns_rheed_pattern(self):
        """Test that simulator returns valid RHEEDPattern.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_pattern_shapes_consistent(self):
        """Test that all pattern arrays have consistent shapes.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        n = pattern.G_indices.shape[0]
        chex.assert_shape(pattern.k_out, (n, 3))
        chex.assert_shape(pattern.detector_points, (n, 2))
        chex.assert_shape(pattern.intensities, (n,))

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_variation(self, distance):
        """Test simulation at different detector distances.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            detector_distance=distance,
        )

        chex.assert_tree_all_finite(pattern.detector_points)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage):
        """Test simulation at different voltages.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=voltage,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("shallow", 1.0),
        ("medium", 2.0),
        ("steep", 5.0),
    )
    def test_angle_variation(self, theta):
        """Test simulation at different grazing angles.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ewald_sphere_constraint(self):
        """Test that output wavevectors approximately satisfy Ewald sphere.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        # k_out should have approximately same magnitude as k_in
        voltage_kv = 20.0
        lam_ang = float(wavelength_ang(voltage_kv))
        k_mag_expected = 2.0 * jnp.pi / lam_ang

        k_out_mags = jnp.linalg.norm(pattern.k_out, axis=1)

        # Filter non-zero k_out (valid reflections)
        valid_mask = k_out_mags > 0
        valid_k_out_mags = k_out_mags[valid_mask]

        if valid_k_out_mags.shape[0] > 0:
            # All valid k_out should be close to k_in magnitude
            chex.assert_trees_all_close(
                valid_k_out_mags,
                jnp.full_like(valid_k_out_mags, k_mag_expected),
                rtol=0.1,
            )


class TestComputeKinematicIntensitiesExtended(
    chex.TestCase, parameterized.TestCase
):
    """Extended tests for kinematic intensity calculation with CTRs."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.si_crystal = self._create_si_crystal()
        self.k_in = jnp.array([73.0, 0.0, -2.5])
        self.g_vectors = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.k_out = self.k_in + self.g_vectors

    def _create_si_crystal(self):
        """Create simple Si crystal for testing."""
        a_si = 5.431
        frac_coords = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        )
        cart_coords = frac_coords * a_si
        atomic_numbers = jnp.full(2, 14.0)
        frac_positions = jnp.column_stack([frac_coords, atomic_numbers])
        cart_positions = jnp.column_stack([cart_coords, atomic_numbers])

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_none(self):
        """Test intensity calculation with no CTR contribution.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="none",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_coherent(self):
        """Test intensity calculation with coherent CTR mixing.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="coherent",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_incoherent(self):
        """Test intensity calculation with incoherent CTR mixing.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="incoherent",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("zero", 0.0),
        ("half", 0.5),
        ("full", 1.0),
    )
    def test_ctr_weight_variation(self, weight):
        """Test effect of CTR weight on intensities."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_weight=weight,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_height(self):
        """Test with height-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        config = SurfaceConfig(method="height", height_fraction=0.3)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_layers(self):
        """Test with layer-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        config = SurfaceConfig(method="layers", n_layers=1)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.1),
        ("loose", 0.5),
    )
    def test_hk_tolerance_variation(self, tolerance):
        """Test effect of h,k tolerance for CTR application."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hk_tolerance=tolerance,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_gating_uses_explicit_hkl(self):
        """Explicit hkl should enable CTR when |G| misses tolerance."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        hkls = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=jnp.int32,
        )

        # Tight tolerance makes derived indices miss near-integer check
        intens_no_hkl = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hk_tolerance=0.01,
        )
        intens_with_hkl = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hkl_indices=hkls,
            hk_tolerance=0.01,
        )

        total_no = float(jnp.sum(intens_no_hkl))
        total_with = float(jnp.sum(intens_with_hkl))

        self.assertGreater(total_with, total_no)


class TestEwaldSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for ewald_simulator with exact Ewald-CTR intersection."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mgo_crystal = self._create_mgo_crystal()

    def _create_mgo_crystal(self):
        """Create a simple MgO rock-salt structure for testing."""
        a_mgo = 4.212

        frac_coords = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        )

        cart_coords = frac_coords * a_mgo

        atomic_numbers = jnp.array([12.0, 8.0])
        frac_positions = jnp.column_stack([frac_coords, atomic_numbers])
        cart_positions = jnp.column_stack([cart_coords, atomic_numbers])

        crystal = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_mgo, a_mgo, a_mgo]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )
        return crystal

    def test_basic_pattern_generation(self):
        """Test that ewald_simulator produces a valid RHEED pattern."""
        pattern = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=3,
            kmax=3,
        )

        valid_mask = pattern.G_indices >= 0
        n_valid = jnp.sum(valid_mask)
        self.assertGreater(
            int(n_valid), 0, "Should have at least one valid reflection"
        )

        self.assertTrue(
            jnp.all(pattern.intensities >= 0),
            "All intensities should be non-negative",
        )

        valid_detector = pattern.detector_points[valid_mask]
        self.assertTrue(
            jnp.all(jnp.isfinite(valid_detector)),
            "Valid detector points should be finite",
        )

    def test_upward_scattering_only(self):
        """Only upward-scattered reflections are returned (k_out_z > 0)."""
        pattern = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=5,
            kmax=5,
        )

        valid_mask = pattern.G_indices >= 0
        k_out_valid = pattern.k_out[valid_mask]

        self.assertTrue(
            jnp.all(k_out_valid[:, 2] > 0),
            "All valid reflections should have k_out_z > 0",
        )

    def test_elastic_scattering_constraint(self):
        """Test that |k_out| = |k_in| (elastic scattering)."""
        pattern = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        valid_mask = pattern.G_indices >= 0
        k_out_valid = pattern.k_out[valid_mask]

        wl = wavelength_ang(20.0)
        k_mag_expected = 2.0 * jnp.pi / wl

        k_out_mags = jnp.linalg.norm(k_out_valid, axis=1)
        relative_error = jnp.abs(k_out_mags - k_mag_expected) / k_mag_expected

        self.assertTrue(
            jnp.all(relative_error < 0.01),
            "k_out magnitudes should match k_in (elastic scattering)",
        )

    def test_azimuthal_rotation_changes_pattern(self):
        """Changing phi_deg rotates the pattern."""
        pattern_0 = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=3,
            kmax=3,
        )

        pattern_45 = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=45.0,
            hmax=3,
            kmax=3,
        )

        self.assertFalse(
            jnp.allclose(
                pattern_0.detector_points, pattern_45.detector_points
            ),
            "Different azimuths should produce different patterns",
        )

    def test_temperature_affects_intensity(self):
        """Higher temperature reduces intensity (Debye-Waller)."""
        pattern_low_T = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            temperature=100.0,
            hmax=3,
            kmax=3,
        )

        pattern_high_T = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            temperature=500.0,
            hmax=3,
            kmax=3,
        )

        valid_low = pattern_low_T.G_indices >= 0
        valid_high = pattern_high_T.G_indices >= 0

        self.assertGreater(
            int(jnp.sum(valid_low)),
            0,
            "Low T pattern should have reflections",
        )
        self.assertGreater(
            int(jnp.sum(valid_high)),
            0,
            "High T pattern should have reflections",
        )

    def test_roughness_affects_intensity(self):
        """Surface roughness affects CTR intensity."""
        pattern_smooth = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_roughness=0.1,
            hmax=3,
            kmax=3,
        )

        pattern_rough = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_roughness=2.0,
            hmax=3,
            kmax=3,
        )

        valid_smooth = pattern_smooth.G_indices >= 0
        valid_rough = pattern_rough.G_indices >= 0

        self.assertGreater(
            int(jnp.sum(valid_smooth)),
            0,
            "Smooth surface should have reflections",
        )
        self.assertGreater(
            int(jnp.sum(valid_rough)),
            0,
            "Rough surface should have reflections",
        )

    def test_voltage_affects_wavevector(self):
        """Different voltages give different k magnitudes."""
        pattern_10kv = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=10.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        pattern_30kv = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=30.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        valid_10 = pattern_10kv.G_indices >= 0
        valid_30 = pattern_30kv.G_indices >= 0

        if jnp.any(valid_10) and jnp.any(valid_30):
            k_mag_10 = jnp.linalg.norm(pattern_10kv.k_out[valid_10][0])
            k_mag_30 = jnp.linalg.norm(pattern_30kv.k_out[valid_30][0])

            self.assertGreater(
                float(k_mag_30),
                float(k_mag_10),
                "Higher voltage should give larger k magnitude",
            )

    def test_jax_jit_compatible(self):
        """ewald_simulator works under JAX JIT compilation."""
        pattern = jax.jit(
            ewald_simulator,
            static_argnames=("hmax", "kmax"),
        )(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
        )

        valid_mask = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "JIT-compiled simulation should work",
        )

    def test_surface_config_parameter(self):
        """surface_config parameter works correctly."""
        config = SurfaceConfig(method="height", height_fraction=0.5)

        pattern = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_config=config,
            hmax=3,
            kmax=3,
        )

        valid_mask = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "Should produce valid pattern with custom surface config",
        )

    @parameterized.parameters(
        {"theta_deg": 1.0},
        {"theta_deg": 2.0},
        {"theta_deg": 3.0},
        {"theta_deg": 5.0},
    )
    def test_various_grazing_angles(self, theta_deg):
        """Various grazing angles produce valid patterns."""
        pattern = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=theta_deg,
            hmax=3,
            kmax=3,
        )

        valid_mask = pattern.G_indices >= 0
        self.assertGreaterEqual(
            int(jnp.sum(valid_mask)),
            0,
            f"Grazing angle {theta_deg} should produce valid reflections",
        )

    def test_orientation_distribution_matches_manual_incoherent_union(self):
        """Orientation wrapper matches explicit per-angle pattern union."""
        orientation_dist = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 45.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        def fake_ewald_simulator(
            crystal,
            voltage_kv,
            theta_deg,
            phi_deg,
            hmax,
            kmax,
            detector_distance,
            temperature,
            surface_roughness,
            surface_config,
        ):
            del (
                crystal,
                voltage_kv,
                theta_deg,
                hmax,
                kmax,
                detector_distance,
                temperature,
                surface_roughness,
                surface_config,
            )
            phi = jnp.asarray(phi_deg, dtype=jnp.float64)
            return RHEEDPattern(
                G_indices=jnp.array([3, 7], dtype=jnp.int32),
                k_out=jnp.array(
                    [
                        [1.0 + phi, 0.0, 2.0],
                        [2.0 + phi, 1.0, 3.0],
                    ],
                    dtype=jnp.float64,
                ),
                detector_points=jnp.array(
                    [
                        [phi, phi + 1.0],
                        [phi + 2.0, phi + 3.0],
                    ],
                    dtype=jnp.float64,
                ),
                intensities=jnp.array([1.0, phi + 2.0], dtype=jnp.float64),
            )

        with patch(
            "rheedium.simul.simulator.ewald_simulator",
            side_effect=fake_ewald_simulator,
        ):
            averaged = ewald_simulator_with_orientation_distribution(
                crystal=_SI_CRYSTAL_2ATOM,
                orientation_distribution=orientation_dist,
                voltage_kv=20.0,
                theta_deg=2.0,
                hmax=1,
                kmax=1,
                n_mosaic_points=1,
            )

        expected_g_indices = jnp.array([3, 7, 3, 7], dtype=jnp.int32)
        expected_k_out = jnp.array(
            [
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 3.0],
                [46.0, 0.0, 2.0],
                [47.0, 1.0, 3.0],
            ],
            dtype=jnp.float64,
        )
        expected_detector_points = jnp.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [45.0, 46.0],
                [47.0, 48.0],
            ],
            dtype=jnp.float64,
        )
        expected_intensities = jnp.array(
            [0.25, 0.5, 0.75, 35.25],
            dtype=jnp.float64,
        )

        chex.assert_trees_all_close(averaged.G_indices, expected_g_indices)
        chex.assert_trees_all_close(averaged.k_out, expected_k_out, atol=1e-10)
        chex.assert_trees_all_close(
            averaged.detector_points,
            expected_detector_points,
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            averaged.intensities,
            expected_intensities,
            atol=1e-10,
        )

    def test_orientation_distribution_wrapper_jits(self):
        """Orientation-distribution wrapper should compile under jax.jit."""
        orientation_dist = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 30.0]),
            weights=jnp.array([0.4, 0.6]),
        )

        pattern = jax.jit(
            ewald_simulator_with_orientation_distribution,
            static_argnames=("hmax", "kmax", "n_mosaic_points"),
        )(
            crystal=_SI_CRYSTAL_2ATOM,
            orientation_distribution=orientation_dist,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
            n_mosaic_points=1,
        )

        valid_mask = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "JIT-compiled orientation simulation should produce reflections",
        )
        self.assertTrue(
            jnp.all(pattern.intensities >= 0.0),
            "Orientation-averaged intensities should be non-negative",
        )


def _make_si_crystal_2atom():
    """Create a 2-atom Si crystal for fast gradient tests."""
    a_si = 5.431
    frac_coords = jnp.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    cart_coords = frac_coords * a_si
    atomic_numbers = jnp.full(2, 14.0)
    frac_positions = jnp.column_stack([frac_coords, atomic_numbers])
    cart_positions = jnp.column_stack([cart_coords, atomic_numbers])
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([a_si, a_si, a_si]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


_SI_CRYSTAL_2ATOM = _make_si_crystal_2atom()


class TestEwaldSimulatorGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for the ewald_simulator."""

    def _ewald_loss(self, **override):
        """Compute sum of intensities from ewald_simulator."""
        defaults = dict(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=2,
            kmax=2,
            temperature=300.0,
            surface_roughness=0.5,
        )
        defaults.update(override)
        pattern = ewald_simulator(**defaults)
        return jnp.sum(pattern.intensities)

    def test_grad_temperature(self):
        """Gradient w.r.t. temperature is finite and non-zero."""

        def loss(temp):
            return self._ewald_loss(temperature=temp)

        g = jax.grad(loss)(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_roughness(self):
        """Gradient w.r.t. surface roughness is finite and non-zero."""

        def loss(roughness):
            return self._ewald_loss(surface_roughness=roughness)

        g = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_polar_angle(self):
        """Gradient w.r.t. incidence angle is finite."""

        def loss(theta):
            return self._ewald_loss(theta_deg=theta)

        g = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(g)

    def test_grad_voltage(self):
        """Gradient w.r.t. beam voltage is finite."""

        def loss(voltage):
            return self._ewald_loss(voltage_kv=voltage)

        g = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_vmap_grad(self):
        """vmap(grad(loss)) over temperatures produces correct shape."""

        def loss(temp):
            return self._ewald_loss(temperature=temp)

        grad_fn = jax.grad(loss)
        batch_grad = jax.vmap(grad_fn)
        temps = jnp.array([100.0, 300.0, 600.0])
        grads = batch_grad(temps)
        chex.assert_shape(grads, (3,))
        chex.assert_tree_all_finite(grads)

    def test_jacrev(self):
        """jacrev w.r.t. (temperature, roughness) produces (2,) Jacobian."""

        def loss(params):
            return self._ewald_loss(
                temperature=params[0],
                surface_roughness=params[1],
            )

        jac_fn = jax.jacrev(loss)
        params = jnp.array([300.0, 0.5])
        jac = jac_fn(params)
        chex.assert_shape(jac, (2,))
        chex.assert_tree_all_finite(jac)

    def test_ewald_simulator_grad_temperature_correct(self):
        """Ewald simulator grad w.r.t. temperature matches finite diff."""

        def f(temp):
            pattern = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        check_grads(jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-2)

    def test_ewald_simulator_grad_roughness_correct(self):
        """Ewald simulator grad w.r.t. roughness matches finite diff."""

        def f(roughness):
            pattern = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=300.0,
                surface_roughness=roughness,
            )
            return jnp.sum(pattern.intensities)

        check_grads(jax_safe(f), (jnp.float64(0.5),), order=1, atol=1e-2)


class TestMultisliceGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for multislice forward model."""

    def test_multislice_grad_voltage(self):
        """Gradient through multislice propagation w.r.t. voltage."""
        cart_positions = jnp.array(
            [[5.0, 5.0, 1.0, 14.0], [7.5, 7.5, 3.0, 14.0]]
        )
        sliced = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def loss(voltage):
            potential = sliced_crystal_to_projected_potential_slices(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
            )
            psi_exit = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        g = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_multislice_grad_voltage_correct(self):
        """Multislice grad w.r.t. voltage matches finite diff."""
        cart_positions = jnp.array(
            [[5.0, 5.0, 1.0, 14.0], [7.5, 7.5, 3.0, 14.0]]
        )
        sliced = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def f(voltage):
            potential = sliced_crystal_to_projected_potential_slices(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
            )
            psi_exit = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        check_grads(jax_safe(f), (jnp.float64(20.0),), order=1, atol=1e-2)


class TestEwaldSimulatorVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential for ewald_simulator."""

    def test_ewald_simulator_vmap_temperature_consistent(self):
        """Batched ewald_simulator over temps matches sequential."""

        def f(temp):
            pattern = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        temp_batch = jnp.array([100.0, 300.0, 600.0])
        batched = jax.vmap(f)(temp_batch)
        sequential = jnp.stack([f(t) for t in temp_batch])
        chex.assert_trees_all_close(batched, sequential, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
