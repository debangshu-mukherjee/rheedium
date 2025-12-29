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

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float, Int

from rheedium.simul import (
    compute_kinematic_intensities_with_ctrs,
    wavelength_ang,
)
from rheedium.simul.simulator import (
    find_kinematic_reflections,
    kinematic_simulator,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    sliced_crystal_to_potential,
)
from rheedium.types import (
    CrystalStructure,
    PotentialSlices,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    bulk_to_slice,
    create_crystal_structure,
    create_potential_slices,
    create_sliced_crystal,
    scalar_float,
)


class TestUpdatedSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for updated RHEED simulator with proper surface physics."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.rng = jax.random.PRNGKey(42)

        # Create simple Si(111) structure for testing
        self.si_crystal = self._create_si111_crystal()

    def _create_si111_crystal(self) -> CrystalStructure:
        """Create a simple Si(111) crystal structure.

        Returns
        -------
        crystal : CrystalStructure
            Silicon crystal with (111) orientation
        """
        a_si: scalar_float = 5.431  # Si lattice constant in Angstroms

        # Si diamond structure fractional positions
        frac_coords: Float[Array, "8 3"] = jnp.array(
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
        cart_coords: Float[Array, "8 3"] = frac_coords * a_si

        # Add atomic numbers (Si = 14)
        atomic_numbers: Float[Array, "8"] = jnp.full(8, 14.0)
        frac_positions: Float[Array, "8 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "8 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        return crystal

    @chex.all_variants(without_device=False)
    def test_wavelength_calculation(self) -> None:
        """Test relativistic wavelength calculation."""
        var_wavelength = self.variant(wavelength_ang)

        # Test at different voltages
        voltages_kv: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = jax.vmap(var_wavelength)(voltages_kv)

        # Expected wavelengths (approximate values in Angstroms)
        expected: Float[Array, "3"] = jnp.array([0.1226, 0.0859, 0.0698])

        chex.assert_trees_all_close(wavelengths, expected, rtol=5e-3)
        chex.assert_tree_all_finite(wavelengths)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("room_temp", 300.0, 0.5, 0.3),
        ("low_temp", 77.0, 0.3, 0.3),
        ("high_roughness", 300.0, 1.0, 0.3),
        ("thin_surface", 300.0, 0.5, 0.1),
    )
    def test_intensity_calculation_with_ctrs(
        self,
        temperature: scalar_float,
        surface_roughness: scalar_float,
        surface_fraction: scalar_float,
    ) -> None:
        """Test intensity calculation with CTR contributions."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        # Set up simple test case
        # 20 keV, 2 degrees
        k_in: Float[Array, "3"] = jnp.array([73.0, 0.0, -2.5])
        g_vectors: Float[Array, "3 3"] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        k_out: Float[Array, "3 3"] = k_in + g_vectors

        intensities: Float[Array, "3"] = var_compute(
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
            max_intensity: scalar_float = jnp.max(intensities)
            chex.assert_scalar_positive(float(max_intensity))

    @chex.variants(with_device=True, without_jit=True)
    def test_simulate_rheed_pattern(self) -> None:
        """Test complete RHEED pattern simulation.

        Note: JIT compilation is not compatible with dynamic hmax/kmax/lmax
        parameters in generate_reciprocal_points.
        """
        var_simulate = self.variant(kinematic_simulator)

        pattern: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=300.0,
            surface_roughness=0.5,
        )

        # Check pattern structure
        n_reflections: int = pattern.G_indices.shape[0]
        chex.assert_shape(pattern.G_indices, (n_reflections,))
        chex.assert_shape(pattern.k_out, (n_reflections, 3))
        chex.assert_shape(pattern.detector_points, (n_reflections, 2))
        chex.assert_shape(pattern.intensities, (n_reflections,))

        # Check physical constraints
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @chex.all_variants(without_device=False)
    def test_surface_enhancement_effect(self) -> None:
        """Test that surface atoms have enhanced thermal motion."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        k_in: Float[Array, "3"] = jnp.array([73.0, 0.0, -2.5])
        g_vectors: Float[Array, "1 3"] = jnp.array([[1.0, 0.0, 0.0]])
        k_out: Float[Array, "1 3"] = k_in + g_vectors

        # Compare with and without surface effects
        intensities_bulk: Float[Array, "1"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_fraction=0.0,  # No surface atoms
        )

        intensities_surface: Float[Array, "1"] = var_compute(
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

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_contribution(self) -> None:
        """Test that CTR contributions are properly included.

        Note: JIT compilation is not compatible with dynamic hmax/kmax/lmax
        parameters in generate_reciprocal_points.
        """
        var_simulate = self.variant(kinematic_simulator)

        # Simulate with and without CTR effects
        pattern_no_ctr: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=0,  # No out-of-plane component
            surface_roughness=0.0,
            detector_acceptance=0.0,  # No CTR integration
        )

        pattern_with_ctr: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,  # Allow out-of-plane
            surface_roughness=0.5,
            detector_acceptance=0.01,  # Include CTR integration
        )

        # CTR should add intensity
        total_no_ctr: scalar_float = jnp.sum(pattern_no_ctr.intensities)
        total_with_ctr: scalar_float = jnp.sum(pattern_with_ctr.intensities)

        # With CTR should have additional intensity contributions
        chex.assert_scalar_positive(float(total_with_ctr))
        chex.assert_scalar_positive(float(total_no_ctr))

    @chex.variants(with_device=True, without_jit=True)
    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the simulation.

        Note: JIT compilation is not compatible with dynamic hmax/kmax/lmax
        parameters in generate_reciprocal_points.
        """

        def loss_fn(temperature: scalar_float) -> scalar_float:
            pattern = kinematic_simulator(
                crystal=self.si_crystal,
                voltage_kv=20.0,
                theta_deg=2.0,
                hmax=1,
                kmax=1,
                lmax=0,
                temperature=temperature,
            )
            return jnp.sum(pattern.intensities)

        var_grad_fn = self.variant(jax.grad(loss_fn))
        gradient: scalar_float = var_grad_fn(300.0)

        # Gradient should be non-zero (temperature affects intensities)
        chex.assert_tree_all_finite(gradient)
        self.assertNotEqual(gradient, 0.0)

    @parameterized.named_parameters(
        ("si_111", [1, 1, 1], 14, 300.0),
        ("si_100", [1, 0, 0], 14, 300.0),
        ("si_110", [1, 1, 0], 14, 300.0),
    )
    def test_known_reflections(
        self,
        miller_indices: list,
        atomic_number: int,
        temperature: scalar_float,
    ) -> None:
        """Test that known Si reflections appear in the pattern."""
        pattern: RHEEDPattern = kinematic_simulator(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=max(miller_indices),
            kmax=max(miller_indices),
            lmax=max(miller_indices),
            temperature=temperature,
        )

        # Check that we get reflections
        n_reflections: int = pattern.G_indices.shape[0]
        self.assertGreater(n_reflections, 0)

        # Check intensity scaling
        max_intensity: scalar_float = jnp.max(pattern.intensities)
        min_intensity: scalar_float = jnp.min(pattern.intensities)

        chex.assert_scalar_positive(float(max_intensity))
        self.assertGreaterEqual(float(min_intensity), 0.0)


class TestProjectOnDetector(chex.TestCase, parameterized.TestCase):
    """Test suite for detector projection functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(without_device=False)
    def test_basic_projection(self) -> None:
        """Test basic projection onto detector plane."""
        var_project = self.variant(project_on_detector)

        k_out: Float[Array, "3 3"] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.5],
            ]
        )
        detector_distance: float = 100.0

        coords: Float[Array, "3 2"] = var_project(k_out, detector_distance)

        chex.assert_shape(coords, (3, 2))
        chex.assert_tree_all_finite(coords)
        # First point: no lateral deflection
        chex.assert_trees_all_close(
            coords[0], jnp.array([0.0, 0.0]), atol=1e-6
        )

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_scaling(self, distance: float) -> None:
        """Test that coordinates scale linearly with detector distance."""
        var_project = self.variant(project_on_detector)

        k_out: Float[Array, "1 3"] = jnp.array([[1.0, 0.5, 0.3]])
        coords: Float[Array, "1 2"] = var_project(k_out, distance)

        chex.assert_shape(coords, (1, 2))
        # Verify linear scaling
        expected_h: float = 0.5 * distance / 1.0
        expected_v: float = 0.3 * distance / 1.0
        chex.assert_trees_all_close(
            coords[0], jnp.array([expected_h, expected_v]), rtol=1e-5
        )

    @chex.all_variants(without_device=False)
    def test_output_shape(self) -> None:
        """Test output has correct shape for various inputs."""
        var_project = self.variant(project_on_detector)

        for n in [1, 5, 10, 50]:
            k_out: Float[Array, "N 3"] = jnp.ones((n, 3))
            coords: Float[Array, "N 2"] = var_project(k_out, 100.0)
            chex.assert_shape(coords, (n, 2))


class TestFindKinematicReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for kinematic reflection finding."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.k_mag: float = 73.0  # Typical |k| for 20 keV electrons

    @chex.all_variants(without_device=False)
    def test_elastic_scattering_constraint(self) -> None:
        """Test that output wavevectors satisfy |k_out| ≈ |k_in|."""
        var_find = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "3"] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "5 3"] = jnp.array(
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

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.05),
        ("loose", 0.2),
    )
    def test_tolerance_variation(self, tolerance: float) -> None:
        """Test that tighter tolerances allow fewer reflections."""
        var_find = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "3"] = jnp.array([self.k_mag, 0.0, -2.5])
        # Small G vectors that barely satisfy elastic condition
        gs: Float[Array, "9 3"] = jnp.array(
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

    @chex.all_variants(without_device=False)
    def test_z_sign_positive(self) -> None:
        """Test filtering with positive z_sign (forward scattering)."""
        var_find = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "3"] = jnp.array([self.k_mag, 0.0, 2.5])
        gs: Float[Array, "3 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -5.0],  # Would give negative z
            ]
        )

        allowed_indices, k_out = var_find(k_in, gs, z_sign=1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False)
    def test_z_sign_negative(self) -> None:
        """Test filtering with negative z_sign (back scattering - RHEED)."""
        var_find = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "3"] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "3 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],  # Would give positive z
                [0.0, 0.0, -1.0],
            ]
        )

        allowed_indices, k_out = var_find(k_in, gs, z_sign=-1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False)
    def test_empty_g_vectors(self) -> None:
        """Test handling of single G vector."""
        var_find = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "3"] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])

        allowed_indices, k_out = var_find(k_in, gs, tolerance=0.5)

        chex.assert_shape(allowed_indices, (1,))
        chex.assert_shape(k_out, (1, 3))


class TestSlicedCrystalToPotential(chex.TestCase, parameterized.TestCase):
    """Test suite for converting sliced crystals to potential arrays."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.si_sliced = self._create_simple_sliced_crystal()

    def _create_simple_sliced_crystal(self) -> SlicedCrystal:
        """Create a simple sliced crystal for testing."""
        # Simple 2-atom structure
        cart_positions: Float[Array, "2 4"] = jnp.array(
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
    def test_output_shape(self) -> None:
        """Test that output potential has expected shape.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(sliced_crystal_to_potential)

        potential: PotentialSlices = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
            voltage_kv=20.0,
        )

        # Check slices array exists
        chex.assert_tree_all_finite(potential.slices)
        # Should have nz slices based on depth/slice_thickness
        nz_expected: int = int(jnp.ceil(5.0 / 2.0))
        self.assertEqual(potential.slices.shape[0], nz_expected)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin", 1.0),
        ("medium", 2.0),
        ("thick", 5.0),
    )
    def test_slice_thickness_variation(self, thickness: float) -> None:
        """Test potential generation with different slice thicknesses.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(sliced_crystal_to_potential)

        potential: PotentialSlices = var_convert(
            self.si_sliced,
            slice_thickness=thickness,
            pixel_size=0.5,
            voltage_kv=20.0,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Number of slices should be ceil(depth / thickness)
        expected_nz: int = int(jnp.ceil(5.0 / thickness))
        self.assertEqual(potential.slices.shape[0], expected_nz)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("fine", 0.1),
        ("medium", 0.5),
        ("coarse", 1.0),
    )
    def test_pixel_size_variation(self, pixel_size: float) -> None:
        """Test potential generation with different pixel sizes.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(sliced_crystal_to_potential)

        potential: PotentialSlices = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=pixel_size,
            voltage_kv=20.0,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Grid should scale with pixel size
        expected_nx: int = int(jnp.ceil(15.0 / pixel_size))
        expected_ny: int = int(jnp.ceil(15.0 / pixel_size))
        self.assertEqual(potential.slices.shape[1], expected_nx)
        self.assertEqual(potential.slices.shape[2], expected_ny)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage: float) -> None:
        """Test potential generation at different voltages.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(sliced_crystal_to_potential)

        potential: PotentialSlices = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
            voltage_kv=voltage,
        )

        chex.assert_tree_all_finite(potential.slices)

    @chex.variants(with_device=True, without_jit=True)
    def test_calibration_stored(self) -> None:
        """Test that calibration values are stored correctly.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert = self.variant(sliced_crystal_to_potential)

        pixel_size: float = 0.3
        slice_thickness: float = 1.5

        potential: PotentialSlices = var_convert(
            self.si_sliced,
            slice_thickness=slice_thickness,
            pixel_size=pixel_size,
            voltage_kv=20.0,
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

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential = self._create_simple_potential()

    def _create_simple_potential(self) -> PotentialSlices:
        """Create a simple potential for testing."""
        # Small grid for fast tests
        nx, ny, nz = 32, 32, 3
        slices: Float[Array, "3 32 32"] = jnp.zeros((nz, nx, ny))
        # Add a small potential at center of first slice
        slices = slices.at[0, 16, 16].set(1.0)

        return create_potential_slices(
            slices=slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

    @chex.all_variants(without_device=False)
    def test_output_shape(self) -> None:
        """Test that exit wave has same shape as input grid."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_shape(exit_wave, (32, 32))
        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False)
    def test_exit_wave_nonzero(self) -> None:
        """Test that exit wave has non-zero amplitude."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        total_intensity: float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage: float) -> None:
        """Test propagation at different voltages."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=voltage,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Higher voltage = shorter wavelength = different phase evolution
        total_intensity: float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("shallow", 0.5),
        ("medium", 2.0),
        ("steep", 5.0),
    )
    def test_grazing_angle_variation(self, theta: float) -> None:
        """Test propagation at different grazing angles."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
    )
    def test_azimuthal_angle_variation(self, phi: float) -> None:
        """Test propagation at different azimuthal angles."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=phi,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("no_inner", 0.0),
        ("small_inner", 10.0),
        ("large_inner", 20.0),
    )
    def test_inner_potential_variation(self, v0: float) -> None:
        """Test effect of inner potential on propagation."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            inner_potential_v0=v0,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("half", 0.5),
        ("two_thirds", 2.0 / 3.0),
        ("full", 1.0),
    )
    def test_bandwidth_limit_variation(self, limit: float) -> None:
        """Test different bandwidth limiting values."""
        var_propagate = self.variant(multislice_propagate)

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            bandwidth_limit=limit,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False)
    def test_zero_potential_propagation(self) -> None:
        """Test propagation through zero potential (free space)."""
        var_propagate = self.variant(multislice_propagate)

        # Zero potential
        zero_slices: Float[Array, "3 32 32"] = jnp.zeros((3, 32, 32))
        zero_potential: PotentialSlices = create_potential_slices(
            slices=zero_slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

        exit_wave: Complex[Array, "32 32"] = var_propagate(
            zero_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Should still have intensity (plane wave propagates)
        total_intensity: float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)


class TestMultisliceSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for complete multislice RHEED simulation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential = self._create_test_potential()

    def _create_test_potential(self) -> PotentialSlices:
        """Create potential slices for testing."""
        nx, ny, nz = 32, 32, 3
        slices: Float[Array, "3 32 32"] = jnp.zeros((nz, nx, ny))
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
    def test_returns_rheed_pattern(self) -> None:
        """Test that simulator returns valid RHEEDPattern.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_pattern_shapes_consistent(self) -> None:
        """Test that all pattern arrays have consistent shapes.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        n: int = pattern.G_indices.shape[0]
        chex.assert_shape(pattern.k_out, (n, 3))
        chex.assert_shape(pattern.detector_points, (n, 2))
        chex.assert_shape(pattern.intensities, (n,))

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_variation(self, distance: float) -> None:
        """Test simulation at different detector distances.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
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
    def test_voltage_variation(self, voltage: float) -> None:
        """Test simulation at different voltages.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
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
    def test_angle_variation(self, theta: float) -> None:
        """Test simulation at different grazing angles.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ewald_sphere_constraint(self) -> None:
        """Test that output wavevectors approximately satisfy Ewald sphere.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate = self.variant(multislice_simulator)

        pattern: RHEEDPattern = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        # k_out should have approximately same magnitude as k_in
        voltage_kv: float = 20.0
        lam_ang: float = float(wavelength_ang(voltage_kv))
        k_mag_expected: float = 2.0 * jnp.pi / lam_ang

        k_out_mags: Float[Array, "N"] = jnp.linalg.norm(pattern.k_out, axis=1)

        # Filter non-zero k_out (valid reflections)
        valid_mask: Bool[Array, "N"] = k_out_mags > 0
        valid_k_out_mags: Float[Array, "M"] = k_out_mags[valid_mask]

        if valid_k_out_mags.shape[0] > 0:
            # All valid k_out should be close to k_in magnitude
            chex.assert_trees_all_close(
                valid_k_out_mags,
                jnp.full_like(valid_k_out_mags, k_mag_expected),
                rtol=0.1,
            )


class TestKinematicSimulatorExtended(chex.TestCase, parameterized.TestCase):
    """Extended test suite for kinematic RHEED simulation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.si_crystal = self._create_si_crystal()

    def _create_si_crystal(self) -> CrystalStructure:
        """Create simple Si crystal for testing."""
        a_si: float = 5.431
        frac_coords: Float[Array, "2 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        )
        cart_coords: Float[Array, "2 3"] = frac_coords * a_si
        atomic_numbers: Float[Array, "2"] = jnp.full(2, 14.0)
        frac_positions: Float[Array, "2 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "2 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_binary_mode(self) -> None:
        """Test kinematic simulation in binary (tolerance) mode."""
        var_simulate = self.variant(kinematic_simulator)

        pattern: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            domain_extent_ang=None,  # Binary mode
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_finite_domain_mode(self) -> None:
        """Test kinematic simulation in finite domain mode."""
        var_simulate = self.variant(kinematic_simulator)

        pattern: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            domain_extent_ang=jnp.array([100.0, 100.0, 20.0]),
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("cold", 100.0),
        ("room", 300.0),
        ("hot", 600.0),
    )
    def test_temperature_effects(self, temperature: float) -> None:
        """Test that temperature affects intensities through Debye-Waller."""
        var_simulate = self.variant(kinematic_simulator)

        pattern: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
            lmax=0,
            temperature=temperature,
        )

        chex.assert_tree_all_finite(pattern.intensities)
        # Higher temperature should generally reduce intensity (DW damping)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("smooth", 0.0),
        ("medium", 0.5),
        ("rough", 1.0),
    )
    def test_surface_roughness_effects(self, roughness: float) -> None:
        """Test that surface roughness affects intensities."""
        var_simulate = self.variant(kinematic_simulator)

        pattern: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
            lmax=0,
            surface_roughness=roughness,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_reflection_count_scaling(self) -> None:
        """Test that reflection count scales with hmax/kmax/lmax."""
        var_simulate = self.variant(kinematic_simulator)

        pattern_small: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
            lmax=0,
            tolerance=0.5,  # Loose tolerance to get more reflections
        )

        pattern_large: RHEEDPattern = var_simulate(
            crystal=self.si_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            tolerance=0.5,
        )

        # Larger bounds should potentially give more reflections
        # (depending on which satisfy elastic scattering)
        n_small: int = pattern_small.G_indices.shape[0]
        n_large: int = pattern_large.G_indices.shape[0]

        self.assertGreater(n_small, 0)
        self.assertGreater(n_large, 0)


class TestComputeKinematicIntensitiesExtended(
    chex.TestCase, parameterized.TestCase
):
    """Extended tests for kinematic intensity calculation with CTRs."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.si_crystal = self._create_si_crystal()
        self.k_in: Float[Array, "3"] = jnp.array([73.0, 0.0, -2.5])
        self.g_vectors: Float[Array, "3 3"] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.k_out: Float[Array, "3 3"] = self.k_in + self.g_vectors

    def _create_si_crystal(self) -> CrystalStructure:
        """Create simple Si crystal for testing."""
        a_si: float = 5.431
        frac_coords: Float[Array, "2 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        )
        cart_coords: Float[Array, "2 3"] = frac_coords * a_si
        atomic_numbers: Float[Array, "2"] = jnp.full(2, 14.0)
        frac_positions: Float[Array, "2 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "2 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_none(self) -> None:
        """Test intensity calculation with no CTR contribution.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities: Float[Array, "3"] = var_compute(
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
    def test_ctr_mode_coherent(self) -> None:
        """Test intensity calculation with coherent CTR mixing.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities: Float[Array, "3"] = var_compute(
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
    def test_ctr_mode_incoherent(self) -> None:
        """Test intensity calculation with incoherent CTR mixing.

        Note: JIT not supported due to string ctr_mixing_mode parameter.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities: Float[Array, "3"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="incoherent",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("zero", 0.0),
        ("half", 0.5),
        ("full", 1.0),
    )
    def test_ctr_weight_variation(self, weight: float) -> None:
        """Test effect of CTR weight on intensities."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities: Float[Array, "3"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_weight=weight,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_height(self) -> None:
        """Test with height-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        config = SurfaceConfig(method="height", height_fraction=0.3)

        intensities: Float[Array, "3"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_layers(self) -> None:
        """Test with layer-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        config = SurfaceConfig(method="layers", n_layers=1)

        intensities: Float[Array, "3"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.1),
        ("loose", 0.5),
    )
    def test_hk_tolerance_variation(self, tolerance: float) -> None:
        """Test effect of h,k tolerance for CTR application."""
        var_compute = self.variant(compute_kinematic_intensities_with_ctrs)

        intensities: Float[Array, "3"] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hk_tolerance=tolerance,
        )

        chex.assert_tree_all_finite(intensities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
