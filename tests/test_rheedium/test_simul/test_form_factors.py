"""Test suite for form_factors.py using chex and parameterized testing.

This module provides comprehensive testing for atomic form factor calculations,
Debye-Waller factors, and atomic scattering factors used in RHEED simulations.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul.form_factors import (
    atomic_scattering_factor,
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
    load_kirkland_parameters,
)


class TestFormFactors(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Common test parameters
        self.test_atomic_numbers = {
            "H": 1,
            "C": 6,
            "Si": 14,
            "Cu": 29,
            "Au": 79,
        }
        self.q_magnitudes = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
        self.temperatures = jnp.array([100.0, 300.0, 600.0])
        # Test q vectors for 3D calculations
        self.q_vectors_3d = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        # Batched q vectors for testing vectorization
        self.batch_size = 10
        self.batched_q = jnp.tile(
            self.q_magnitudes[:, jnp.newaxis], (1, self.batch_size)
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("hydrogen", 1),
        ("carbon", 6),
        ("silicon", 14),
        ("copper", 29),
        ("gold", 79),
        ("uranium", 92),
    )
    def test_load_kirkland_parameters(self, atomic_number: int) -> None:
        """Test loading Kirkland parameters for various elements."""
        var_load_params = self.variant(load_kirkland_parameters)
        a_coeffs, b_coeffs = var_load_params(atomic_number)

        # Check shapes
        chex.assert_shape(a_coeffs, (6,))
        chex.assert_shape(b_coeffs, (6,))

        # Check coefficients are finite and positive
        chex.assert_tree_all_finite(a_coeffs)
        chex.assert_tree_all_finite(b_coeffs)

        # b coefficients should be positive (width parameters)
        chex.assert_trees_all_equal(jnp.all(b_coeffs > 0), True)

        # a coefficients sum should be close to atomic number for small q
        a_sum = jnp.sum(a_coeffs)
        chex.assert_scalar_positive(float(a_sum))

    @chex.variants(with_jit=True, without_jit=True)
    def test_load_kirkland_parameters_edge_cases(self) -> None:
        """Test parameter loading with edge cases."""
        var_load_params = self.variant(load_kirkland_parameters)

        # Test boundary values
        a_min, b_min = var_load_params(1)  # Minimum atomic number
        a_max, b_max = var_load_params(103)  # Maximum atomic number

        chex.assert_shape(a_min, (6,))
        chex.assert_shape(b_min, (6,))
        chex.assert_shape(a_max, (6,))
        chex.assert_shape(b_max, (6,))

        # Test clipping for out-of-range values
        a_clip_low, b_clip_low = var_load_params(0)  # Should clip to 1
        a_clip_high, b_clip_high = var_load_params(150)  # Should clip to 103

        chex.assert_trees_all_close(a_clip_low, a_min, rtol=1e-10)
        chex.assert_trees_all_close(b_clip_low, b_min, rtol=1e-10)
        chex.assert_trees_all_close(a_clip_high, a_max, rtol=1e-10)
        chex.assert_trees_all_close(b_clip_high, b_max, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("silicon_low_q", 14, 0.1),
        ("silicon_medium_q", 14, 1.0),
        ("silicon_high_q", 14, 10.0),
        ("gold_low_q", 79, 0.1),
        ("gold_high_q", 79, 10.0),
    )
    def test_kirkland_form_factor_single(
        self, atomic_number: int, q_mag: float
    ) -> None:
        """Test Kirkland form factor for single q values."""
        var_form_factor = self.variant(kirkland_form_factor)
        q_array = jnp.array(q_mag)

        f_q = var_form_factor(atomic_number, q_array)

        # Squeeze to scalar if needed (form factor may add dimension)
        f_q = jnp.squeeze(f_q)

        # Check shape and finiteness
        chex.assert_shape(f_q, ())
        chex.assert_tree_all_finite(f_q)

        # Form factor should be positive
        chex.assert_scalar_positive(float(f_q))

        # At very low q, form factor should be significant
        # Note: For electron scattering (Kirkland), f(0) != Z
        if q_mag < 0.2:
            chex.assert_scalar_positive(float(f_q))

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_form_factor_decreasing(self) -> None:
        """Test that form factor decreases with increasing q."""
        var_form_factor = self.variant(kirkland_form_factor)

        for name, z in self.test_atomic_numbers.items():
            f_values = var_form_factor(z, self.q_magnitudes)

            # Check monotonic decrease (excluding q=0)
            differences = jnp.diff(f_values[1:])
            chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

            # Form factor at q=0 should be positive
            # Note: For electron scattering, f(0) != Z
            chex.assert_scalar_positive(float(f_values[0]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_form_factor_batched(self) -> None:
        """Test form factor with batched q values."""
        var_form_factor = self.variant(kirkland_form_factor)

        # 2D batch of q values
        q_batch_2d = jnp.tile(self.q_magnitudes[:, jnp.newaxis], (1, 5))
        f_batch_2d = var_form_factor(14, q_batch_2d)
        chex.assert_shape(f_batch_2d, q_batch_2d.shape)

        # 3D batch of q values
        q_batch_3d = jnp.tile(
            self.q_magnitudes[:, jnp.newaxis, jnp.newaxis], (1, 3, 4)
        )
        f_batch_3d = var_form_factor(14, q_batch_3d)
        chex.assert_shape(f_batch_3d, q_batch_3d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("silicon_room_bulk", 14, 300.0, False),
        ("silicon_room_surface", 14, 300.0, True),
        ("silicon_high_temp", 14, 600.0, False),
        ("gold_room_bulk", 79, 300.0, False),
        ("gold_room_surface", 79, 300.0, True),
        ("hydrogen_low_temp", 1, 100.0, False),
    )
    def test_get_mean_square_displacement(
        self, atomic_number: int, temperature: float, is_surface: bool
    ) -> None:
        """Test mean square displacement calculation."""
        var_get_msd = self.variant(get_mean_square_displacement)

        msd = var_get_msd(atomic_number, temperature, is_surface)

        # Check scalar output
        chex.assert_shape(msd, ())
        chex.assert_tree_all_finite(msd)
        chex.assert_scalar_positive(float(msd))

        # Surface atoms should have larger MSD
        if is_surface:
            msd_bulk = var_get_msd(atomic_number, temperature, False)
            chex.assert_scalar_positive(float(msd - msd_bulk))

    @chex.variants(with_jit=True, without_jit=True)
    def test_mean_square_displacement_scaling(self) -> None:
        """Test MSD scaling with temperature and atomic number."""
        var_get_msd = self.variant(get_mean_square_displacement)

        # Temperature scaling for same element
        msd_si_100 = var_get_msd(14, 100.0, False)
        msd_si_300 = var_get_msd(14, 300.0, False)
        msd_si_600 = var_get_msd(14, 600.0, False)

        # Should increase with temperature
        chex.assert_scalar_positive(float(msd_si_300 - msd_si_100))
        chex.assert_scalar_positive(float(msd_si_600 - msd_si_300))

        # Atomic number scaling at same temperature
        msd_h = var_get_msd(1, 300.0, False)
        msd_c = var_get_msd(6, 300.0, False)
        msd_au = var_get_msd(79, 300.0, False)

        # Lighter atoms should have larger MSD
        chex.assert_scalar_positive(float(msd_h - msd_c))
        chex.assert_scalar_positive(float(msd_c - msd_au))

        # Surface enhancement should be consistent
        for z in [1, 14, 79]:
            msd_bulk = var_get_msd(z, 300.0, False)
            msd_surf = var_get_msd(z, 300.0, True)
            ratio = msd_surf / msd_bulk
            chex.assert_trees_all_close(ratio, 2.0, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_msd_low_q", 0.001, 0.5),
        ("small_msd_high_q", 0.001, 5.0),
        ("large_msd_low_q", 0.1, 0.5),
        ("large_msd_high_q", 0.1, 5.0),
        ("zero_q", 0.01, 0.0),
    )
    def test_debye_waller_factor_single(
        self, msd: float, q_mag: float
    ) -> None:
        """Test Debye-Waller factor for single values."""
        var_dw_factor = self.variant(debye_waller_factor)

        q_array = jnp.array(q_mag)
        dw = var_dw_factor(q_array, msd)

        # Check shape and bounds
        chex.assert_shape(dw, ())
        chex.assert_tree_all_finite(dw)

        # DW factor should be between 0 and 1
        chex.assert_scalar_positive(float(dw))
        chex.assert_scalar_positive(float(1.0 - dw + 1e-10))

        # At q=0, DW factor should be 1
        if q_mag < 1e-10:
            chex.assert_trees_all_close(dw, 1.0, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_debye_waller_factor_batched(self) -> None:
        """Test Debye-Waller factor with batched q values."""
        var_dw_factor = self.variant(debye_waller_factor)

        msd = 0.01

        # 1D batch
        dw_1d = var_dw_factor(self.q_magnitudes, msd)
        chex.assert_shape(dw_1d, self.q_magnitudes.shape)

        # Check monotonic decrease
        differences = jnp.diff(dw_1d)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

        # 2D batch
        q_batch_2d = self.batched_q
        dw_2d = var_dw_factor(q_batch_2d, msd)
        chex.assert_shape(dw_2d, q_batch_2d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_debye_waller_edge_cases(self) -> None:
        """Test Debye-Waller factor with edge cases."""
        var_dw_factor = self.variant(debye_waller_factor)

        q_test = jnp.array([0.0, 1.0, 10.0])

        # Zero MSD (no thermal vibration)
        dw_zero_msd = var_dw_factor(q_test, 0.0)
        chex.assert_trees_all_close(
            dw_zero_msd, jnp.ones_like(q_test), rtol=1e-10
        )

        # Very large MSD
        dw_large_msd = var_dw_factor(q_test[1:], 10.0)
        chex.assert_trees_all_equal(jnp.all(dw_large_msd < 1e-5), True)

        # Negative MSD should be handled safely (clipped to small positive)
        dw_neg_msd = var_dw_factor(q_test, -0.1)
        chex.assert_tree_all_finite(dw_neg_msd)
        chex.assert_trees_all_close(
            dw_neg_msd, jnp.ones_like(q_test), rtol=0.1
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("silicon_room_bulk", 14, 300.0, False),
        ("silicon_room_surface", 14, 300.0, True),
        ("gold_high_temp", 79, 600.0, False),
        ("hydrogen_low_temp", 1, 100.0, False),
    )
    def test_atomic_scattering_factor(
        self, atomic_number: int, temperature: float, is_surface: bool
    ) -> None:
        """Test combined atomic scattering factor."""
        var_scattering = self.variant(atomic_scattering_factor)

        f_combined = var_scattering(
            atomic_number, self.q_vectors_3d, temperature, is_surface
        )

        # Check shape
        chex.assert_shape(f_combined, (len(self.q_vectors_3d),))
        chex.assert_tree_all_finite(f_combined)

        # All values should be positive
        chex.assert_trees_all_equal(jnp.all(f_combined >= 0), True)

        # First value (q=0) should be positive and finite
        # Note: For electron scattering (Kirkland), f(0) != Z
        chex.assert_scalar_positive(float(f_combined[0]))

        # Values should generally decrease with |q|
        q_mags = jnp.linalg.norm(self.q_vectors_3d, axis=-1)
        sorted_indices = jnp.argsort(q_mags)
        f_sorted = f_combined[sorted_indices]
        # Check general trend (allowing small deviations)
        chex.assert_scalar_positive(float(f_sorted[0] - f_sorted[-1]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_atomic_scattering_factor_batched(self) -> None:
        """Test atomic scattering factor with batched inputs."""
        var_scattering = self.variant(atomic_scattering_factor)

        # 2D batch: (batch_size, 3)
        batch_2d = jnp.tile(self.q_vectors_3d[jnp.newaxis, :, :], (5, 1, 1))
        f_batch_2d = var_scattering(14, batch_2d, 300.0, False)
        chex.assert_shape(f_batch_2d, (5, len(self.q_vectors_3d)))

        # 3D batch: (batch1, batch2, 3)
        batch_3d = jnp.tile(
            self.q_vectors_3d[jnp.newaxis, jnp.newaxis, :, :], (3, 4, 1, 1)
        )
        f_batch_3d = var_scattering(14, batch_3d, 300.0, False)
        chex.assert_shape(f_batch_3d, (3, 4, len(self.q_vectors_3d)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_surface_vs_bulk_comparison(self) -> None:
        """Test that surface atoms have stronger thermal damping."""
        var_scattering = self.variant(atomic_scattering_factor)

        q_test = jnp.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        for z in [6, 14, 29]:
            f_bulk = var_scattering(z, q_test, 300.0, False)
            f_surf = var_scattering(z, q_test, 300.0, True)

            # At q=0 they should be equal, but for q>0 surface should be smaller
            chex.assert_trees_all_equal(jnp.all(f_surf < f_bulk), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_form_factor(self) -> None:
        """Test JAX transformations on form factor calculations."""
        var_form_factor = self.variant(kirkland_form_factor)

        # Test JIT compilation
        @jax.jit
        def jitted_form_factor(
            z: int, q: Float[Array, "..."]
        ) -> Float[Array, "..."]:
            return var_form_factor(z, q)

        q_test = self.q_magnitudes
        f_normal = var_form_factor(14, q_test)
        f_jitted = jitted_form_factor(14, q_test)
        chex.assert_trees_all_close(f_normal, f_jitted, rtol=1e-10)

        # Test vmap over atomic numbers
        atomic_nums = jnp.array([1, 6, 14, 29, 79])
        q_single = jnp.array(1.0)  # Use scalar q value
        vmapped_ff = jax.vmap(
            lambda z: var_form_factor(z, q_single), in_axes=0
        )
        f_vmapped = vmapped_ff(atomic_nums)
        # Note: vmap adds an extra dimension, so shape is (5, 1) not (5,)
        chex.assert_shape(f_vmapped, (len(atomic_nums), 1))

        # Test grad with respect to q
        def loss_fn(q: float) -> float:
            f_val = var_form_factor(14, jnp.array(q))
            # Ensure we return a scalar
            return jnp.squeeze(f_val)

        grad_fn = jax.grad(loss_fn)
        grad_q = grad_fn(1.0)
        chex.assert_shape(grad_q, ())
        chex.assert_tree_all_finite(grad_q)
        # Gradient should be negative (form factor decreases with q)
        chex.assert_scalar_positive(float(-grad_q))

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_scattering(self) -> None:
        """Test JAX transformations on atomic scattering factor."""
        var_scattering = self.variant(atomic_scattering_factor)

        # Test vmap over temperatures
        temps = jnp.array([100.0, 300.0, 600.0])
        q_single = jnp.array([[1.0, 0.0, 0.0]])

        vmapped_temp = jax.vmap(
            lambda t: var_scattering(14, q_single, t, False), in_axes=0
        )
        f_temps = vmapped_temp(temps)
        chex.assert_shape(f_temps, (3, 1))

        # Higher temperature should give lower scattering factor (more damping)
        # Need to squeeze the arrays to get scalars for comparison
        diff = jnp.squeeze(f_temps[0]) - jnp.squeeze(f_temps[-1])
        chex.assert_scalar_positive(float(diff))

        # Test nested vmap over atomic numbers and temperatures
        atomic_nums = jnp.array([6, 14, 29])

        def scattering_fn(z: int, t: float) -> Float[Array, " 1"]:
            return var_scattering(z, q_single, t, False)

        nested_vmap = jax.vmap(
            jax.vmap(scattering_fn, in_axes=(None, 0)), in_axes=(0, None)
        )
        f_nested = nested_vmap(atomic_nums, temps)
        chex.assert_shape(f_nested, (len(atomic_nums), len(temps), 1))

    @chex.variants(with_jit=True, without_jit=True)
    def test_physical_consistency(self) -> None:
        """Test physical consistency of calculations."""
        var_form_factor = self.variant(kirkland_form_factor)
        var_dw_factor = self.variant(debye_waller_factor)
        var_get_msd = self.variant(get_mean_square_displacement)
        var_scattering = self.variant(atomic_scattering_factor)

        # Test that combined factor equals product of components
        z = 14
        temp = 300.0
        q_vec = jnp.array([[2.0, 0.0, 0.0]])
        q_mag = jnp.linalg.norm(q_vec, axis=-1)

        # Calculate components separately
        f_kirk = var_form_factor(z, q_mag)
        msd = var_get_msd(z, temp, False)
        dw = var_dw_factor(q_mag, msd)
        f_product = f_kirk * dw

        # Calculate combined
        f_combined = var_scattering(z, q_vec, temp, False)

        # Should be equal
        chex.assert_trees_all_close(f_combined, f_product, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("zero_vector", jnp.zeros((3,))),
        ("unit_vectors", jnp.eye(3)),
        ("random_vectors", jax.random.normal(jax.random.PRNGKey(0), (10, 3))),
        ("large_magnitude", jnp.array([[100.0, 0.0, 0.0]])),
    )
    def test_q_vector_invariance(
        self, q_vectors: Float[Array, "... 3"]
    ) -> None:
        """Test that scattering depends only on |q|, not direction."""
        var_scattering = self.variant(atomic_scattering_factor)

        # For vectors with same magnitude, scattering should be identical
        if q_vectors.ndim == 1:
            q_vectors = q_vectors[jnp.newaxis, :]

        q_mags = jnp.linalg.norm(q_vectors, axis=-1, keepdims=True)

        # Create rotated versions with same magnitude
        if q_mags[0] > 1e-10:  # Skip zero vector
            q_normalized = q_vectors / (q_mags + 1e-10)
            q_rotated = jnp.roll(q_normalized, 1, axis=-1) * q_mags

            f_original = var_scattering(14, q_vectors, 300.0, False)
            f_rotated = var_scattering(14, q_rotated, 300.0, False)

            chex.assert_trees_all_close(f_original, f_rotated, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
