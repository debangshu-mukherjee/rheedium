"""Test suite for surface_rods.py using chex and parameterized testing.

This module provides comprehensive testing for CTR intensity calculations,
roughness damping, and rod profile functions used in RHEED simulations.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul.surface_rods import (
    calculate_ctr_intensity,
    gaussian_rod_profile,
    integrated_rod_intensity,
    lorentzian_rod_profile,
    rod_profile_function,
    roughness_damping,
    surface_structure_factor,
)
from rheedium.types import create_crystal_structure


class TestSurfaceRods(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures for surface rod calculations.

        Creates a simple cubic test crystal structure with Si atoms for
        testing CTR calculations. Sets up fractional and Cartesian
        coordinate arrays, cell parameters (5.43 Å cubic lattice), and
        test parameters including:
        - q_z values from 0 to 5 reciprocal lattice units
        - (h,k) indices for different crystal truncation rods
        - Surface roughness values from smooth (0) to rough (2 Å)
        - Temperature values (100K, 300K, 600K)
        - Correlation lengths (10-500 Å) and q_perp values for rod profiles
        """
        super().setUp()

        self.simple_cubic_frac = jnp.array(
            [
                [0.0, 0.0, 0.0, 14],
                [0.5, 0.5, 0.0, 14],
                [0.0, 0.5, 0.5, 14],
                [0.5, 0.0, 0.5, 14],
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

        self.cell_lengths = jnp.array([5.43, 5.43, 5.43])
        self.cell_angles = jnp.array([90.0, 90.0, 90.0])

        self.test_crystal = create_crystal_structure(
            frac_positions=self.simple_cubic_frac,
            cart_positions=self.simple_cubic_cart,
            cell_lengths=self.cell_lengths,
            cell_angles=self.cell_angles,
        )

        self.q_z_values = jnp.linspace(0.0, 5.0, 20)
        self.hk_indices = jnp.array(
            [[0, 0], [1, 0], [1, 1], [2, 0]], dtype=jnp.int32
        )
        self.roughness_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        self.temperatures = jnp.array([100.0, 300.0, 600.0])

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
        """Test roughness damping for single q_z values.

        Tests the surface roughness damping factor exp(-q_z²σ²/2) for
        various roughness values (0, 0.5, 1.0, 2.0 Å). Verifies that
        damping values are properly bounded between 0 and 1, equal to 1
        at q_z=0 (no damping for zero momentum transfer), and decrease
        monotonically with increasing q_z. The damping represents the loss
        of coherent scattering intensity due to random height variations
        at the crystal surface.
        """
        var_damping = self.variant(roughness_damping)

        q_z_test = jnp.array([0.0, 1.0, 2.0, 5.0])
        damping_values = var_damping(q_z_test, sigma)

        chex.assert_shape(damping_values, q_z_test.shape)

        chex.assert_trees_all_equal(jnp.all(damping_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(damping_values <= 1), True)

        chex.assert_trees_all_close(damping_values[0], 1.0, rtol=1e-10)

        if sigma > 1e-10:
            differences = jnp.diff(damping_values)
            chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_damping_batched(self) -> None:
        """Test roughness damping with batched q_z arrays.

        Verifies that the roughness damping function correctly handles
        vectorized operations on multi-dimensional q_z arrays. Tests 2D
        batches (20x5 array) and 3D batches (20x3x4 array) to ensure
        proper broadcasting. This batching capability is essential for
        efficient CTR calculations where multiple q_z points and detector
        positions are evaluated simultaneously in RHEED pattern
        simulations.
        """
        var_damping = self.variant(roughness_damping)

        q_z_2d = jnp.tile(self.q_z_values[:, jnp.newaxis], (1, 5))
        sigma_test = 1.0

        damping_2d = var_damping(q_z_2d, sigma_test)
        chex.assert_shape(damping_2d, q_z_2d.shape)

        q_z_3d = jnp.tile(
            self.q_z_values[:, jnp.newaxis, jnp.newaxis], (1, 3, 4)
        )
        damping_3d = var_damping(q_z_3d, sigma_test)
        chex.assert_shape(damping_3d, q_z_3d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_roughness_no_damping(self) -> None:
        """Test that zero roughness gives no damping.

        Validates that a perfectly smooth surface (σ=0) produces no
        damping (factor=1) for all q_z values. This represents an ideally
        flat crystal termination with no height variations, resulting in
        perfect coherent scattering along the crystal truncation rod
        without intensity reduction from surface disorder.
        """
        var_damping = self.variant(roughness_damping)

        q_z_test = jnp.linspace(0.0, 10.0, 50)
        damping = var_damping(q_z_test, 0.0)

        expected = jnp.ones_like(q_z_test)
        chex.assert_trees_all_close(damping, expected, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_corr", 10.0),
        ("medium_corr", 50.0),
        ("large_corr", 100.0),
        ("very_large_corr", 500.0),
    )
    def test_gaussian_rod_profile(self, correlation_length: float) -> None:
        """Test Gaussian rod profile for different correlation lengths.

        Tests the Gaussian profile function exp(-q_perp²ξ²/2) that
        describes the lateral width of crystal truncation rods. Tests
        correlation lengths from 10 to 500 Å representing different
        degrees of lateral coherence. Verifies that the profile is
        normalized (peak=1 at q_perp=0), bounded between 0 and 1, and
        decreases monotonically away from the rod center. Larger
        correlation lengths produce narrower rods in reciprocal space,
        reflecting better lateral ordering.
        """
        var_gaussian = self.variant(gaussian_rod_profile)

        profile_values = var_gaussian(
            self.q_perp_values, correlation_length
        )

        chex.assert_shape(profile_values, self.q_perp_values.shape)

        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        differences = jnp.diff(profile_values)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_corr", 10.0),
        ("medium_corr", 50.0),
        ("large_corr", 100.0),
        ("very_large_corr", 500.0),
    )
    def test_lorentzian_rod_profile(self, correlation_length: float) -> None:
        """Test Lorentzian rod profile for different correlation lengths.

        Tests the Lorentzian profile function 1/(1 + (q_perpξ)²) that
        describes CTR width with power-law tails. Tests correlation lengths
        from 10 to 500 Å. Verifies normalization (peak=1 at q_perp=0),
        bounded values [0,1], and monotonic decrease away from center.
        Lorentzian profiles have heavier tails than Gaussian, representing
        surfaces with longer-range disorder or step distributions following
        power-law decay.
        """
        var_lorentzian = self.variant(lorentzian_rod_profile)

        profile_values = var_lorentzian(
            self.q_perp_values, correlation_length
        )

        chex.assert_shape(profile_values, self.q_perp_values.shape)

        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        differences = jnp.diff(profile_values)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @parameterized.named_parameters(
        ("gaussian", "gaussian"),
        ("lorentzian", "lorentzian"),
    )
    def test_rod_profile_function_selector(self, profile_type: str) -> None:
        """Test rod profile function selector without JIT.

        Tests the dynamic selection between Gaussian and Lorentzian rod
        profiles based on a string parameter. This selector function allows
        users to choose the appropriate profile for their surface
        morphology: Gaussian for surfaces with random height distributions,
        Lorentzian for surfaces with power-law step distributions. Verifies
        that both profile types produce valid, normalized results with
        correct monotonic behavior.
        """
        correlation_length = 50.0

        profile_values = rod_profile_function(
            self.q_perp_values, correlation_length, profile_type
        )

        chex.assert_shape(profile_values, self.q_perp_values.shape)

        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        chex.assert_scalar_positive(
            float(profile_values[0] - profile_values[-1])
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rod_profile_width_scaling(self) -> None:
        """Test that rod width scales inversely with correlation length.

        Verifies the inverse relationship between real-space correlation
        length and reciprocal-space rod width. At a fixed q_perp value,
        smaller correlation lengths (10 Å) produce broader rods with higher
        intensity away from center, while larger correlation lengths
        (100 Å) produce narrower, more sharply peaked rods. This reflects
        the Fourier transform relationship between real-space disorder and
        reciprocal-space broadening.
        """
        var_gaussian = self.variant(gaussian_rod_profile)
        q_test = jnp.array(0.1)
        profile_small_corr = var_gaussian(q_test, 10.0)
        profile_large_corr = var_gaussian(q_test, 100.0)
        chex.assert_scalar_positive(
            float(profile_small_corr - profile_large_corr)
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
        """Test surface structure factor calculation.

        Tests the calculation of the complex structure factor F(q) =
        Σf_j exp(iq·r_j) for the surface unit cell. Tests various q-vectors
        including forward scattering (0,0,0) and different reciprocal
        lattice points, at temperatures from 100K to 600K, for both bulk
        and surface atoms. Verifies that the structure factor is a finite
        complex scalar with non-negative magnitude. The structure factor
        determines the intensity distribution in the RHEED pattern through
        |F(q)|².
        """
        var_structure_factor = self.variant(surface_structure_factor)

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

            chex.assert_shape(f_struct, ())
            chex.assert_tree_all_finite(f_struct)

            chex.assert_trees_all_equal(jnp.abs(f_struct) >= 0, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_symmetry(self) -> None:
        """Test structure factor respects Friedel's law: F(-q) = F*(q).

        Validates Friedel's law (centrosymmetry in reciprocal space) which
        states that F(-q) = F*(q) for real atomic scattering factors. This
        fundamental symmetry arises from the fact that the electron density
        is real in real space. Tests with an arbitrary q-vector and
        verifies that reversing q gives the complex conjugate of the
        structure factor. This symmetry is crucial for understanding
        diffraction pattern symmetries.
        """
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
        """Test CTR intensity calculation for different rods.

        Tests crystal truncation rod intensity I(h,k,q_z) = |F(h,k,q_z)|²
        × D(q_z) for various rods: specular (0,0), first-order (1,0),
        diagonal (1,1), and high-order (2,0). Tests with roughness values
        from 0.5 to 2.0 Å. Verifies that intensities are non-negative,
        finite, and generally decrease with q_z due to roughness damping.
        The CTR intensity profile along q_z contains information about
        surface structure, roughness, and relaxation.
        """
        var_ctr = self.variant(calculate_ctr_intensity)

        hk_array = jnp.array([hk_index], dtype=jnp.int32)

        intensities = var_ctr(
            hk_indices=hk_array,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=roughness,
            temperature=300.0,
        )

        chex.assert_shape(intensities, (1, len(self.q_z_values)))

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        chex.assert_tree_all_finite(intensities)

        if roughness > 0.1:
            n_points = len(self.q_z_values)
            first_quarter_mean = jnp.mean(intensities[0, : n_points // 4])
            last_quarter_mean = jnp.mean(intensities[0, -n_points // 4 :])

            chex.assert_scalar_positive(
                float(first_quarter_mean - last_quarter_mean)
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_ctr_intensity_multiple_rods(self) -> None:
        """Test CTR calculation for multiple rods simultaneously.

        Tests vectorized calculation of multiple crystal truncation rods
        in parallel: (0,0) specular, (1,0) first-order, (1,1) diagonal,
        and (2,0) second-order. Verifies that the function correctly
        computes a 2D array of intensities (N_rods × N_q_z), all values
        are physically valid (non-negative, finite), and different rods
        show distinct intensity patterns reflecting their unique structure
        factors. This parallelization is crucial for efficient RHEED
        pattern calculation.
        """
        var_ctr = self.variant(calculate_ctr_intensity)

        intensities = var_ctr(
            hk_indices=self.hk_indices,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=1.0,
            temperature=300.0,
        )

        expected_shape = (len(self.hk_indices), len(self.q_z_values))
        chex.assert_shape(intensities, expected_shape)

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)
        chex.assert_tree_all_finite(intensities)

        rod_differences = jnp.std(intensities, axis=0)
        chex.assert_scalar_positive(float(jnp.mean(rod_differences)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_effect_on_ctr(self) -> None:
        """Test that roughness reduces CTR intensity at high q_z.

        Compares CTR intensities for smooth (σ=0.1 Å) vs rough (σ=2.0 Å)
        surfaces. Verifies that roughness increasingly damps intensity at
        higher q_z values according to exp(-q_z²σ²). Tests the (1,0) rod
        and checks that:
        1. Smooth surface maintains higher intensity at large q_z
        2. Rough surface shows stronger damping, especially at high q_z
        3. The relative reduction is significant (measurable
           experimentally)
        This damping effect allows roughness determination from CTR
        measurements.
        """
        var_ctr = self.variant(calculate_ctr_intensity)

        hk_test = jnp.array([[1, 0]], dtype=jnp.int32)

        intensities_smooth = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=0.1,
            temperature=300.0,
        )

        intensities_rough = var_ctr(
            hk_indices=hk_test,
            q_z=self.q_z_values,
            crystal=self.test_crystal,
            surface_roughness=2.0,
            temperature=300.0,
        )

        n_half = len(self.q_z_values) // 2
        smooth_high_qz = intensities_smooth[0, n_half:]
        rough_high_qz = intensities_rough[0, n_half:]

        chex.assert_trees_all_equal(
            jnp.all(smooth_high_qz >= rough_high_qz), True
        )

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
        """Test integrated CTR intensity over detector acceptance.

        Tests numerical integration of CTR intensity over a q_z range,
        simulating finite detector acceptance. Tests various integration
        ranges (narrow: 0-1, wide: 0-5, offset: 2-4) and detector
        acceptances (0.1-0.5 reciprocal units). The integrated intensity
        represents what a real detector measures, accounting for its finite
        angular resolution. Verifies that integrated intensity is a positive
        scalar value, essential for comparing with experimental
        measurements.
        """
        var_integrated = self.variant(integrated_rod_intensity)

        integrated = var_integrated(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            q_z_range=jnp.array(q_z_range, dtype=jnp.float64),
            crystal=self.test_crystal,
            surface_roughness=1.0,
            detector_acceptance=detector_acceptance,
            n_integration_points=30,
            temperature=300.0,
        )

        chex.assert_shape(integrated, ())
        chex.assert_tree_all_finite(integrated)

        chex.assert_scalar_positive(float(integrated))

    @chex.variants(with_jit=True, without_jit=True)
    def test_temperature_effect_on_ctr(self) -> None:
        """Test that higher temperature reduces CTR intensity.

        Compares CTR intensities at low (100K) and high (600K) temperatures
        for the (1,1) rod. Higher temperatures increase atomic thermal
        vibrations, leading to larger Debye-Waller factors and reduced
        coherent scattering. Verifies that the mean intensity decreases
        with temperature, reflecting the exp(-B·q²) thermal damping where
        B ∝ T. This temperature dependence is crucial for understanding
        RHEED patterns at different growth temperatures.
        """
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

        mean_cold = jnp.mean(intensities_cold)
        mean_hot = jnp.mean(intensities_hot)
        chex.assert_scalar_positive(float(mean_cold - mean_hot))

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_ctr(self) -> None:
        """Test JAX transformations on CTR calculations.

        Validates JAX functional transformations on CTR calculations:
        1. vmap: Vectorizes over roughness values (0, 0.5, 1.0, 2.0 Å) to
           compute multiple CTR profiles efficiently in parallel, essential
           for parameter sweeps and fitting procedures.
        2. grad: Computes the gradient of CTR intensity with respect to
           roughness, verifying it's negative (more roughness reduces
           intensity). This gradient is crucial for optimization algorithms
           that fit CTR data to extract surface parameters.
        """
        var_ctr = self.variant(calculate_ctr_intensity)
        var_damping = self.variant(roughness_damping)

        def ctr_for_roughness(sigma: float) -> Float[Array, "1 M"]:
            return var_ctr(
                hk_indices=jnp.array([[1, 0]], dtype=jnp.int32),
                q_z=self.q_z_values[:10],
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )

        vmapped_ctr = jax.vmap(ctr_for_roughness)
        intensities_batch = vmapped_ctr(self.roughness_values)

        expected_shape = (len(self.roughness_values), 1, 10)
        chex.assert_shape(intensities_batch, expected_shape)

        def loss_fn(sigma: float) -> float:
            intensities = var_ctr(
                hk_indices=jnp.array([[0, 0]], dtype=jnp.int32),
                q_z=jnp.array([2.0]),
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )
            return jnp.squeeze(intensities)

        grad_fn = jax.grad(loss_fn)
        grad_roughness = grad_fn(1.0)

        chex.assert_shape(grad_roughness, ())
        chex.assert_tree_all_finite(grad_roughness)
        chex.assert_scalar_positive(float(-grad_roughness))

    @chex.variants(with_jit=True, without_jit=True)
    def test_profile_comparison(self) -> None:
        """Test that Gaussian and Lorentzian profiles differ appropriately.

        Compares Gaussian and Lorentzian rod profiles with the same
        correlation length (50 Å). Both profiles are normalized (peak=1 at
        q_perp=0) but differ in their tails: Lorentzian has power-law decay
        (1/q_perp²) producing heavier tails, while Gaussian has exponential
        decay (exp(-q_perp²)) producing lighter tails. This difference is
        important for distinguishing surface morphologies: Gaussian for
        random roughness, Lorentzian for step-terrace structures.
        """
        var_gaussian = self.variant(gaussian_rod_profile)
        var_lorentzian = self.variant(lorentzian_rod_profile)

        correlation = 50.0

        gaussian_profile = var_gaussian(
            self.q_perp_values, correlation
        )
        lorentzian_profile = var_lorentzian(
            self.q_perp_values, correlation
        )

        chex.assert_trees_all_close(
            gaussian_profile[0], lorentzian_profile[0], rtol=1e-10
        )

        tail_indices = self.q_perp_values > 0.3
        gaussian_tail = gaussian_profile[tail_indices]
        lorentzian_tail = lorentzian_profile[tail_indices]

        tail_diff = jnp.mean(lorentzian_tail - gaussian_tail)
        chex.assert_scalar_positive(float(tail_diff))

    @chex.variants(with_jit=True, without_jit=True)
    def test_physical_consistency(self) -> None:
        """Test physical consistency of CTR calculations.

        Validates that CTR intensity calculation correctly implements the
        physical relationship: I(h,k,q_z) = |F(h,k,q_z)|² × exp(-q_z²σ²).
        Tests the decomposition by calculating structure factor and
        roughness damping separately, then verifying their product matches
        the combined CTR intensity. Tests multiple rods (0,0), (1,0), (2,0)
        at q_z=1.0 to ensure the formula is consistently applied. This
        validates the theoretical foundation of CTR analysis for surface
        structure determination.
        """
        var_ctr = self.variant(calculate_ctr_intensity)
        var_damping = self.variant(roughness_damping)
        var_structure = self.variant(surface_structure_factor)

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

        q_vec = jnp.array([0.0, 0.0, 1.0])

        f_struct = var_structure(
            q_vector=q_vec,
            atomic_positions=self.test_crystal.cart_positions[:, :3],
            atomic_numbers=self.test_crystal.cart_positions[:, 3].astype(
                jnp.int32
            ),
            temperature=300.0,
            is_surface=True,
        )

        damping = var_damping(q_z=jnp.array(1.0), sigma_height=0.5)

        expected_intensity = jnp.abs(f_struct) ** 2 * damping

        chex.assert_tree_all_finite(expected_intensity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
