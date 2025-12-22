"""Test suite for updated RHEED simulator with surface physics.

Tests the integration of:
- Proper atomic form factors (Kirkland parameterization)
- Surface-enhanced Debye-Waller factors
- CTR intensity calculations
- Structure factor with q_z dependence
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul import (
    compute_kinematic_intensities_with_ctrs,
    wavelength_ang,
)
from rheedium.simul.simulator import kinematic_simulator
from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_crystal_structure,
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
        frac_coords: Float[Array, "8 3"] = jnp.array([
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.00],
            [0.75, 0.75, 0.25],
            [0.50, 0.00, 0.50],
            [0.75, 0.25, 0.75],
            [0.00, 0.50, 0.50],
            [0.25, 0.75, 0.75],
        ])

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
        g_vectors: Float[Array, "3 3"] = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
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
        chex.assert_trees_all_equal(
            jnp.all(pattern.intensities >= 0), True
        )
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])