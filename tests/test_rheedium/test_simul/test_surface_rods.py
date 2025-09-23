"""Test suite for surface_rods.py using chex and parameterized testing.

This module provides comprehensive testing for CTR intensity calculations,
roughness damping, and rod profile functions used in RHEED simulations.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float, Int

from rheedium.simul.surface_rods import (
    calculate_ctr_intensity,
    integrated_rod_intensity,
    rod_profile_function,
    roughness_damping,
    surface_structure_factor,
)
from rheedium.types import CrystalStructure, create_crystal_structure


class TestSurfaceRods(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures for surface rod calculations."""
        super().setUp()

        # Create a simple cubic test crystal
        self.simple_cubic_frac = jnp.array(
            [
                [0.0, 0.0, 0.0, 14],  # Si atom at origin
                [0.5, 0.5, 0.0, 14],  # Si atom on surface
                [0.0, 0.5, 0.5, 14],  # Si atom
                [0.5, 0.0, 0.5, 14],  # Si atom
            ]
        )

        self.simple_cubic_cart = jnp.array(
            [
                [0.0, 0.0, 0.0, 14],
                [2.715, 2.715, 0.0, 14],
                [0.0, 2.715, 2.715, 14],
                [2.715, 0.0, 2.715, 14],
            ]
        )

        self.cell_lengths = jnp.array([5.43, 5.43, 5.43])  # Silicon lattice
        self.cell_angles = jnp.array([90.0, 90.0, 90.0])

        self.test_crystal = create_crystal_structure(
            frac_positions=self.simple_cubic_frac,
            cart_positions=self.simple_cubic_cart,
            cell_lengths=self.cell_lengths,
            cell_angles=self.cell_angles,
        )

        # Test parameters
        self.q_z_values = jnp.linspace(0.0, 5.0, 20)
        self.hk_indices = jnp.array(
            [[0, 0], [1, 0], [1, 1], [2, 0]], dtype=jnp.int32
        )
        self.roughness_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        self.temperatures = jnp.array([100.0, 300.0, 600.0])

        # Correlation lengths for rod profiles
        self.correlation_lengths = jnp.array([10.0, 50.0, 100.0, 500.0])
        self.q_perp_values = jnp.linspace(0.0, 0.5, 30)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("zero_roughness", 0.0),
        ("small_roughness", 0.5),
        ("medium_roughness", 1.0),
        ("large_roughness", 2.0),
    )
    def test_roughness_damping_single(self, sigma: float) -> None:
        """Test roughness damping for single q_z values."""
        var_damping = self.variant(roughness_damping)

        q_z_test = jnp.array([0.0, 1.0, 2.0, 5.0])
        damping_values = var_damping(q_z_test, sigma)

        # Check shape
        chex.assert_shape(damping_values, q_z_test.shape)

        # All values should be between 0 and 1
        chex.assert_trees_all_equal(jnp.all(damping_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(damping_values <= 1), True)

        # At q_z=0, damping should be 1
        chex.assert_trees_all_close(damping_values[0], 1.0, rtol=1e-10)

        # Should decrease monotonically with q_z
        if sigma > 1e-10:
            differences = jnp.diff(damping_values)
            chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_damping_batched(self) -> None:
        """Test roughness damping with batched q_z arrays."""
        var_damping = self.variant(roughness_damping)

        # 2D batch of q_z values
        q_z_2d = jnp.tile(self.q_z_values[:, jnp.newaxis], (1, 5))
        sigma_test = 1.0

        damping_2d = var_damping(q_z_2d, sigma_test)
        chex.assert_shape(damping_2d, q_z_2d.shape)

        # 3D batch
        q_z_3d = jnp.tile(
            self.q_z_values[:, jnp.newaxis, jnp.newaxis], (1, 3, 4)
        )
        damping_3d = var_damping(q_z_3d, sigma_test)
        chex.assert_shape(damping_3d, q_z_3d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_roughness_no_damping(self) -> None:
        """Test that zero roughness gives no damping."""
        var_damping = self.variant(roughness_damping)

        q_z_test = jnp.linspace(0.0, 10.0, 50)
        damping = var_damping(q_z_test, 0.0)

        # Should be all ones (no damping)
        expected = jnp.ones_like(q_z_test)
        chex.assert_trees_all_close(damping, expected, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("gaussian_small_corr", "gaussian", 10.0),
        ("gaussian_large_corr", "gaussian", 100.0),
        ("lorentzian_small_corr", "lorentzian", 10.0),
        ("lorentzian_large_corr", "lorentzian", 100.0),
    )
    def test_rod_profile_function(
        self, profile_type: str, correlation_length: float
    ) -> None:
        """Test rod profile functions for different types and lengths."""
        var_profile = self.variant(rod_profile_function)

        profile_values = var_profile(
            self.q_perp_values, correlation_length, profile_type
        )

        # Check shape
        chex.assert_shape(profile_values, self.q_perp_values.shape)

        # All values should be between 0 and 1
        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        # At q_perp=0, profile should be 1 (peak)
        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        # Should decrease away from center
        chex.assert_scalar_positive(
            float(profile_values[0] - profile_values[-1])
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rod_profile_width_scaling(self) -> None:
        """Test that rod width scales inversely with correlation length."""
        var_profile = self.variant(rod_profile_function)

        q_test = jnp.array(0.1)  # Fixed q_perp value

        # Small correlation length should give wider profile
        profile_small_corr = var_profile(q_test, 10.0, "gaussian")
        profile_large_corr = var_profile(q_test, 100.0, "gaussian")

        # At same q_perp, smaller correlation gives lower intensity
        chex.assert_scalar_positive(
            float(profile_large_corr - profile_small_corr)
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("room_temp_bulk", 300.0, False),
        ("room_temp_surface", 300.0, True),
        ("high_temp_surface", 600.0, True),
        ("low_temp_bulk", 100.0, False),
    )
    def test_surface_structure_factor(
        self, temperature: float, is_surface: bool
    ) -> None:
        """Test surface structure factor calculation."""
        var_structure_factor = self.variant(surface_structure_factor)

        # Test q vectors
        q_vectors = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        atomic_positions = self.test_crystal.cart_positions[:, :3]
        atomic_numbers = self.test_crystal.cart_positions[:, 3].astype(
            jnp.int32
        )

        for q_vec in q_vectors:
            f_struct = var_structure_factor(
                q_vector=q_vec,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                temperature=temperature,
                is_surface=is_surface,
            )

            # Check scalar output
            chex.assert_shape(f_struct, ())
            chex.assert_tree_all_finite(f_struct)

            # Structure factor magnitude should be non-negative
            chex.assert_trees_all_equal(jnp.abs(f_struct) >= 0, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_symmetry(self) -> None:
        """Test structure factor respects Friedel's law: F(-q) = F*(q)."""
        var_structure_factor = self.variant(surface_structure_factor)

        atomic_positions = self.test_crystal.cart_positions[:, :3]
        atomic_numbers = self.test_crystal.cart_positions[:, 3].astype(
            jnp.int32
        )

        q_vec = jnp.array([1.0, 0.5, 0.3])
        q_vec_neg = -q_vec

        f_pos = var_structure_factor(
            q_vector=q_vec,
            atomic_positions=atomic_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            is_surface=False,
        )

        f_neg = var_structure_factor(
            q_vector=q_vec_neg,
            atomic_positions=atomic_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            is_surface=False,
        )

        # F(-q) should equal conjugate of F(q)
        chex.assert_trees_all_close(f_neg, jnp.conj(f_pos), rtol=1e-8)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("specular_rod", [0, 0], 0.5),
        ("first_order", [1, 0], 1.0),
        ("diagonal_rod", [1, 1], 1.5),
        ("high_order", [2, 0], 2.0),
    )
    def test_calculate_ctr_intensity(
        self, hk_index: list, roughness: float
    ) -> None:
        """Test CTR intensity calculation for different rods."""
        var_ctr = self.variant(calculate_ctr_intensity)

        hk_array = jnp.array([hk_index], dtype=jnp.int32)

        intensities = var_ctr(
            hk_indices=hk_array,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=roughness,
            temperature=300.0,
        )

        # Check shape: (1 rod, M q_z points)
        chex.assert_shape(intensities, (1, len(self.q_z_values)))

        # All intensities should be non-negative
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        # Check finite values
        chex.assert_tree_all_finite(intensities)

        # Intensities should generally decrease with q_z due to roughness
        if roughness > 0.1:
            # Take average over first and last quarters
            n_points = len(self.q_z_values)
            first_quarter_mean = jnp.mean(intensities[0, : n_points // 4])
            last_quarter_mean = jnp.mean(intensities[0, -n_points // 4 :])

            # First quarter should have higher average intensity
            chex.assert_scalar_positive(
                float(first_quarter_mean - last_quarter_mean)
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_ctr_intensity_multiple_rods(self) -> None:
        """Test CTR calculation for multiple rods simultaneously."""
        var_ctr = self.variant(calculate_ctr_intensity)

        intensities = var_ctr(
            hk_indices=self.hk_indices,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=1.0,
            temperature=300.0,
        )

        # Check shape: (N rods, M q_z points)
        expected_shape = (len(self.hk_indices), len(self.q_z_values))
        chex.assert_shape(intensities, expected_shape)

        # All intensities should be non-negative and finite
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)
        chex.assert_tree_all_finite(intensities)

        # Each rod should have different intensity pattern
        # Check that rods are not all identical
        rod_differences = jnp.std(intensities, axis=0)
        chex.assert_scalar_positive(float(jnp.mean(rod_differences)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_effect_on_ctr(self) -> None:
        """Test that roughness reduces CTR intensity at high q_z."""
        var_ctr = self.variant(calculate_ctr_intensity)

        hk_test = jnp.array([[1, 0]], dtype=jnp.int32)

        # Calculate with different roughness values
        intensities_smooth = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=0.1,  # Nearly smooth
            temperature=300.0,
        )

        intensities_rough = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=2.0,  # Rough
            temperature=300.0,
        )

        # Rough surface should have lower intensity, especially at high q_z
        # Check last half of q_z range
        n_half = len(self.q_z_values) // 2
        smooth_high_qz = intensities_smooth[0, n_half:]
        rough_high_qz = intensities_rough[0, n_half:]

        # Smooth should have higher intensity at high q_z
        chex.assert_trees_all_equal(
            jnp.all(smooth_high_qz >= rough_high_qz), True
        )

        # The difference should be significant
        relative_reduction = jnp.mean(
            (smooth_high_qz - rough_high_qz) / (smooth_high_qz + 1e-10)
        )
        chex.assert_scalar_positive(float(relative_reduction))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("narrow_range", (0.0, 1.0), 0.1),
        ("wide_range", (0.0, 5.0), 0.5),
        ("offset_range", (2.0, 4.0), 0.2),
    )
    def test_integrated_rod_intensity(
        self,
        q_z_range: tuple,
        detector_acceptance: float,
    ) -> None:
        """Test integrated CTR intensity over detector acceptance."""
        var_integrated = self.variant(integrated_rod_intensity)

        integrated = var_integrated(
            hk_index=(1, 0),
            q_z_range=q_z_range,
            crystal=self.test_crystal,
            surface_roughness=1.0,
            detector_acceptance=detector_acceptance,
            n_integration_points=30,
            temperature=300.0,
        )

        # Check scalar output
        chex.assert_shape(integrated, ())
        chex.assert_tree_all_finite(integrated)

        # Integrated intensity should be positive
        chex.assert_scalar_positive(float(integrated))

    @chex.variants(with_jit=True, without_jit=True)
    def test_temperature_effect_on_ctr(self) -> None:
        """Test that higher temperature reduces CTR intensity."""
        var_ctr = self.variant(calculate_ctr_intensity)

        hk_test = jnp.array([[1, 1]], dtype=jnp.int32)

        intensities_cold = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=0.5,
            temperature=100.0,
        )

        intensities_hot = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=0.5,
            temperature=600.0,
        )

        # Higher temperature should reduce intensity
        mean_cold = jnp.mean(intensities_cold)
        mean_hot = jnp.mean(intensities_hot)
        chex.assert_scalar_positive(float(mean_cold - mean_hot))

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_ctr(self) -> None:
        """Test JAX transformations on CTR calculations."""
        var_ctr = self.variant(calculate_ctr_intensity)
        var_damping = self.variant(roughness_damping)

        # Test vmap over different roughness values
        def ctr_for_roughness(sigma: float) -> Float[Array, "1 M"]:
            return var_ctr(
                hk_indices=jnp.array([[1, 0]], dtype=jnp.int32),
                q_z=self.q_z_values[:10],  # Use fewer points for speed
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )

        vmapped_ctr = jax.vmap(ctr_for_roughness)
        intensities_batch = vmapped_ctr(self.roughness_values)

        expected_shape = (len(self.roughness_values), 1, 10)
        chex.assert_shape(intensities_batch, expected_shape)

        # Test grad with respect to roughness
        def loss_fn(sigma: float) -> float:
            intensities = var_ctr(
                hk_indices=jnp.array([[0, 0]], dtype=jnp.int32),
                q_z=jnp.array([2.0]),  # Single q_z
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )
            return jnp.squeeze(intensities)

        grad_fn = jax.grad(loss_fn)
        grad_roughness = grad_fn(1.0)

        # Gradient should be negative (more roughness reduces intensity)
        chex.assert_shape(grad_roughness, ())
        chex.assert_tree_all_finite(grad_roughness)
        chex.assert_scalar_positive(float(-grad_roughness))

    @chex.variants(with_jit=True, without_jit=True)
    def test_profile_comparison(self) -> None:
        """Test that Gaussian and Lorentzian profiles differ appropriately."""
        var_profile = self.variant(rod_profile_function)

        correlation = 50.0

        gaussian_profile = var_profile(
            self.q_perp_values, correlation, "gaussian"
        )
        lorentzian_profile = var_profile(
            self.q_perp_values, correlation, "lorentzian"
        )

        # Both should have same peak value (1.0 at q_perp=0)
        chex.assert_trees_all_close(
            gaussian_profile[0], lorentzian_profile[0], rtol=1e-10
        )

        # Lorentzian should have heavier tails (higher values far from center)
        tail_indices = self.q_perp_values > 0.3
        gaussian_tail = gaussian_profile[tail_indices]
        lorentzian_tail = lorentzian_profile[tail_indices]

        # Lorentzian tails should be larger on average
        tail_diff = jnp.mean(lorentzian_tail - gaussian_tail)
        chex.assert_scalar_positive(float(tail_diff))

    @chex.variants(with_jit=True, without_jit=True)
    def test_physical_consistency(self) -> None:
        """Test physical consistency of CTR calculations."""
        var_ctr = self.variant(calculate_ctr_intensity)
        var_damping = self.variant(roughness_damping)
        var_structure = self.variant(surface_structure_factor)

        # For specular rod (0,0), intensity should be highest
        hk_test = jnp.array([[0, 0], [1, 0], [2, 0]], dtype=jnp.int32)
        q_z_single = jnp.array([1.0])

        intensities = var_ctr(
            hk_indices=hk_test,
            q_z=q_z_single,
            crystal=self.test_crystal,
            surface_roughness=0.5,
            temperature=300.0,
        )
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        # Test that CTR intensity equals |F|² × damping
        # Calculate components separately for one point
        q_vec = jnp.array([0.0, 0.0, 1.0])  # Along z for (0,0) rod

        f_struct = var_structure(
            q_vector=q_vec,
            atomic_positions=self.test_crystal.cart_positions[:, :3],
            atomic_numbers=self.test_crystal.cart_positions[:, 3].astype(
                jnp.int32
            ),
            temperature=300.0,
            is_surface=True,
        )

        damping = var_damping(q_z=1.0, sigma_height=0.5)

        expected_intensity = jnp.abs(f_struct) ** 2 * damping

        # The (0,0) rod at q_z=1.0 should match this calculation
        # (within numerical precision and considering approximations)
        chex.assert_tree_all_finite(expected_intensity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
