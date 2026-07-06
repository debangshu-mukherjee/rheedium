"""Test suite for surface_rods.py using chex and parameterized testing.

This module provides comprehensive testing for CTR intensity calculations,
roughness damping, and rod profile functions used in RHEED simulations.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax.test_util import check_grads
from jaxtyping import Array, Bool, Float, Int, Integer

from rheedium.simul.surface_rods import (
    calculate_ctr_intensity,
    ctr_truncation_amplitude,
    ctr_truncation_intensity,
    gaussian_rod_profile,
    integrated_ctr_amplitude,
    integrated_rod_intensity,
    lorentzian_rod_profile,
    rod_profile_function,
    roughness_damping,
    surface_structure_factor,
)
from rheedium.tools.wrappers import jax_safe
from rheedium.types import CrystalStructure
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.types.custom_types import scalar_float
from rheedium.ucell import reciprocal_lattice_vectors


class TestSurfaceRods(chex.TestCase, parameterized.TestCase):
    """Tests for surface crystal-truncation-rod calculations.

    :see: :func:`~rheedium.simul.calculate_ctr_amplitude`
    :see: :func:`~rheedium.simul.calculate_ctr_intensity`
    :see: :func:`~rheedium.simul.gaussian_rod_profile`
    :see: :func:`~rheedium.simul.integrated_rod_intensity`
    :see: :func:`~rheedium.simul.lorentzian_rod_profile`
    :see: :func:`~rheedium.simul.rod_profile_function`
    :see: :func:`~rheedium.simul.roughness_damping`
    :see: :func:`~rheedium.simul.surface_structure_factor`
    """

    def setUp(self) -> None:
        """Set up test fixtures for surface rod calculations.

        Creates a simple cubic test crystal structure with Si atoms for
        testing CTR calculations. Sets up fractional and Cartesian
        coordinate arrays, cell parameters (5.43 Å cubic lattice), and
        test parameters including:
        - Continuous l values from 0 to 5 along the rods
        - (h,k) indices for different crystal truncation rods
        - Surface roughness values from smooth (0) to rough (2 Å)
        - Temperature values (100K, 300K, 600K)
        - Correlation lengths (10-500 Å) and q_perp values for rod profiles
        """
        super().setUp()

        self.simple_cubic_frac: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 14],
                [0.5, 0.5, 0.0, 14],
                [0.0, 0.5, 0.5, 14],
                [0.5, 0.0, 0.5, 14],
            ]
        )

        self.simple_cubic_cart: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 14],
                [2.715, 2.715, 0.0, 14],
                [0.0, 2.715, 2.715, 14],
                [2.715, 0.0, 2.715, 14],
            ]
        )

        self.cell_lengths: Float[Array, "..."] = jnp.array([5.43, 5.43, 5.43])
        self.cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

        self.test_crystal: CrystalStructure = create_crystal_structure(
            frac_positions=self.simple_cubic_frac,
            cart_positions=self.simple_cubic_cart,
            cell_lengths=self.cell_lengths,
            cell_angles=self.cell_angles,
        )

        self.l_values: Float[Array, "..."] = jnp.linspace(0.0, 5.0, 20)
        self.hk_indices: Integer[Array, "..."] = jnp.array(
            [[0, 0], [1, 0], [1, 1], [2, 0]], dtype=jnp.int32
        )
        self.roughness_values: Float[Array, "..."] = jnp.array(
            [0.0, 0.5, 1.0, 2.0]
        )
        self.temperatures: Float[Array, "..."] = jnp.array(
            [100.0, 300.0, 600.0]
        )

        self.correlation_lengths: Float[Array, "..."] = jnp.array(
            [10.0, 50.0, 100.0, 500.0]
        )
        self.q_perp_values: Float[Array, "..."] = jnp.linspace(0.0, 0.5, 30)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("zero_roughness", 0.0),
        ("small_roughness", 0.5),
        ("medium_roughness", 1.0),
        ("large_roughness", 2.0),
    )
    def test_roughness_damping_single(self, sigma: float) -> None:
        r"""Test roughness damping for single q_z values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roughness damping
        for single q_z values. Existing context from the original test prose:
        Tests the surface roughness damping factor exp(-q_z²σ²/2) for various
        roughness values (0, 0.5, 1.0, 2.0 Å). Verifies that damping values are
        properly bounded between 0 and 1, equal to 1 at q_z=0 (no damping for
        zero momentum transfer), and decrease monotonically with increasing
        q_z. The damping represents the loss of coherent scattering intensity
        due to random height variations at the crystal surface.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``sigma``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_damping: Callable[..., Any] = self.variant(roughness_damping)

        q_z_test: Float[Array, "..."] = jnp.array([0.0, 1.0, 2.0, 5.0])
        damping_values: Any = var_damping(q_z_test, sigma)

        chex.assert_shape(damping_values, q_z_test.shape)

        chex.assert_trees_all_equal(jnp.all(damping_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(damping_values <= 1), True)

        chex.assert_trees_all_close(damping_values[0], 1.0, rtol=1e-10)

        if sigma > 1e-10:
            differences: Float[Array, "..."] = jnp.diff(damping_values)
            chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_damping_batched(self) -> None:
        r"""Test roughness damping with batched q_z arrays.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roughness damping
        with batched q_z arrays. Existing context from the original test prose:
        Verifies that the roughness damping function correctly handles
        vectorized operations on multi-dimensional q_z arrays. Tests 2D batches
        (20x5 array) and 3D batches (20x3x4 array) to ensure proper
        broadcasting. This batching capability is essential for efficient CTR
        calculations where multiple q_z points and detector positions are
        evaluated simultaneously in RHEED pattern simulations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_damping: Callable[..., Any] = self.variant(roughness_damping)

        q_z_2d: Float[Array, "..."] = jnp.tile(
            self.l_values[:, jnp.newaxis], (1, 5)
        )
        sigma_test: float = 1.0

        damping_2d: Any = var_damping(q_z_2d, sigma_test)
        chex.assert_shape(damping_2d, q_z_2d.shape)

        q_z_3d: Float[Array, "..."] = jnp.tile(
            self.l_values[:, jnp.newaxis, jnp.newaxis], (1, 3, 4)
        )
        damping_3d: Any = var_damping(q_z_3d, sigma_test)
        chex.assert_shape(damping_3d, q_z_3d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_roughness_no_damping(self) -> None:
        r"""Test that zero roughness gives no damping.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: zero roughness
        gives no damping. Existing context from the original test prose:
        Validates that a perfectly smooth surface (σ=0) produces no damping
        (factor=1) for all q_z values. This represents an ideally flat crystal
        termination with no height variations, resulting in perfect coherent
        scattering along the crystal truncation rod without intensity reduction
        from surface disorder.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_damping: Callable[..., Any] = self.variant(roughness_damping)

        q_z_test: Float[Array, "..."] = jnp.linspace(0.0, 10.0, 50)
        damping: Any = var_damping(q_z_test, 0.0)

        expected: Float[Array, "..."] = jnp.ones_like(q_z_test)
        chex.assert_trees_all_close(damping, expected, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_corr", 10.0),
        ("medium_corr", 50.0),
        ("large_corr", 100.0),
        ("very_large_corr", 500.0),
    )
    def test_gaussian_rod_profile(self, correlation_length: float) -> None:
        r"""Test Gaussian rod profile for different correlation lengths.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gaussian rod
        profile for different correlation lengths. Existing context from the
        original test prose: Tests the Gaussian profile function
        exp(-q_perp²ξ²/2) that describes the lateral width of crystal
        truncation rods. Tests correlation lengths from 10 to 500 Å
        representing different degrees of lateral coherence. Verifies that the
        profile is normalized (peak=1 at q_perp=0), bounded between 0 and 1,
        and decreases monotonically away from the rod center. Larger
        correlation lengths produce narrower rods in reciprocal space,
        reflecting better lateral ordering.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``correlation_length``, so the documented behavior is checked across
        the cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_gaussian: Callable[..., Any] = self.variant(gaussian_rod_profile)

        profile_values: Float[Array, "..."] = var_gaussian(
            self.q_perp_values, correlation_length
        )

        chex.assert_shape(profile_values, self.q_perp_values.shape)

        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        differences: Float[Array, "..."] = jnp.diff(profile_values)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_corr", 10.0),
        ("medium_corr", 50.0),
        ("large_corr", 100.0),
        ("very_large_corr", 500.0),
    )
    def test_lorentzian_rod_profile(self, correlation_length: float) -> None:
        r"""Test Lorentzian rod profile for different correlation lengths.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Lorentzian rod
        profile for different correlation lengths. Existing context from the
        original test prose: Tests the Lorentzian profile function 1/(1 +
        (q_perpξ)²) that describes CTR width with power-law tails. Tests
        correlation lengths from 10 to 500 Å. Verifies normalization (peak=1 at
        q_perp=0), bounded values [0,1], and monotonic decrease away from
        center. Lorentzian profiles have heavier tails than Gaussian,
        representing surfaces with longer-range disorder or step distributions
        following power-law decay.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``correlation_length``, so the documented behavior is checked across
        the cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_lorentzian: Callable[..., Any] = self.variant(
            lorentzian_rod_profile
        )

        profile_values: Float[Array, "..."] = var_lorentzian(
            self.q_perp_values, correlation_length
        )

        chex.assert_shape(profile_values, self.q_perp_values.shape)

        chex.assert_trees_all_equal(jnp.all(profile_values >= 0), True)
        chex.assert_trees_all_equal(jnp.all(profile_values <= 1), True)

        chex.assert_trees_all_close(profile_values[0], 1.0, rtol=1e-10)

        differences: Float[Array, "..."] = jnp.diff(profile_values)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    @parameterized.named_parameters(
        ("gaussian", "gaussian"),
        ("lorentzian", "lorentzian"),
    )
    def test_rod_profile_function_selector(self, profile_type: str) -> None:
        r"""Test rod profile function selector without JIT.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: rod profile
        function selector without JIT. Existing context from the original test
        prose: Tests the dynamic selection between Gaussian and Lorentzian rod
        profiles based on a string parameter. This selector function allows
        users to choose the appropriate profile for their surface morphology:
        Gaussian for surfaces with random height distributions, Lorentzian for
        surfaces with power-law step distributions. Verifies that both profile
        types produce valid, normalized results with correct monotonic
        behavior.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``profile_type``, so the documented behavior is checked across the
        cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        correlation_length: float = 50.0

        profile_values: Float[Array, "..."] = rod_profile_function(
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
        r"""Test that rod width scales inversely with correlation length.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: rod width scales
        inversely with correlation length. Existing context from the original
        test prose: Verifies the inverse relationship between real-space
        correlation length and reciprocal-space rod width. At a fixed q_perp
        value, smaller correlation lengths (10 Å) produce broader rods with
        higher intensity away from center, while larger correlation lengths
        (100 Å) produce narrower, more sharply peaked rods. This reflects the
        Fourier transform relationship between real-space disorder and
        reciprocal-space broadening.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_gaussian: Callable[..., Any] = self.variant(gaussian_rod_profile)
        q_test: scalar_float = jnp.array(0.1)
        profile_small_corr: Float[Array, "..."] = var_gaussian(q_test, 10.0)
        profile_large_corr: Float[Array, "..."] = var_gaussian(q_test, 100.0)
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
        r"""Test surface structure factor calculation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: surface structure
        factor calculation. Existing context from the original test prose:
        Tests the calculation of the complex structure factor F(q) = Σf_j
        exp(iq·r_j) for the surface unit cell. Tests various q-vectors
        including forward scattering (0,0,0) and different reciprocal lattice
        points, at temperatures from 100K to 600K, for both bulk and surface
        atoms. Verifies that the structure factor is a finite complex scalar
        with non-negative magnitude. The structure factor determines the
        intensity distribution in the RHEED pattern through \|F(q)\|².

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``temperature``, ``is_surface``, so the documented behavior is checked
        across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_structure_factor: Callable[..., Any] = self.variant(
            surface_structure_factor
        )

        q_vectors: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        atomic_positions: Float[Array, "..."] = (
            self.test_crystal.cart_positions[:, :3]
        )
        atomic_numbers: Float[Array, "..."] = self.test_crystal.cart_positions[
            :, 3
        ].astype(jnp.int32)
        # Convert scalar is_surface to per-atom mask
        n_atoms: int = atomic_positions.shape[0]
        is_surface_atom: Bool[Array, "..."] = jnp.full((n_atoms,), is_surface)

        q_vec: Float[Array, "..."]
        for q_vec in q_vectors:
            f_struct: Any = var_structure_factor(
                q_vector=q_vec,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                temperature=temperature,
                is_surface_atom=is_surface_atom,
            )

            chex.assert_shape(f_struct, ())
            chex.assert_tree_all_finite(f_struct)

            chex.assert_trees_all_equal(jnp.abs(f_struct) >= 0, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_symmetry(self) -> None:
        r"""Test structure factor respects Friedel's law: F(-q) = F*(q).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        respects Friedel's law: F(-q) = F*(q). Existing context from the
        original test prose: Validates Friedel's law (centrosymmetry in
        reciprocal space) which states that F(-q) = F*(q) for real atomic
        scattering factors. This fundamental symmetry arises from the fact that
        the electron density is real in real space. Tests with an arbitrary
        q-vector and verifies that reversing q gives the complex conjugate of
        the structure factor. This symmetry is crucial for understanding
        diffraction pattern symmetries.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_structure_factor: Callable[..., Any] = self.variant(
            surface_structure_factor
        )

        atomic_positions: Float[Array, "..."] = (
            self.test_crystal.cart_positions[:, :3]
        )
        atomic_numbers: Float[Array, "..."] = self.test_crystal.cart_positions[
            :, 3
        ].astype(jnp.int32)
        # All bulk atoms (no surface enhancement)
        n_atoms: int = atomic_positions.shape[0]
        is_surface_atom: Bool[Array, "..."] = jnp.full((n_atoms,), False)

        q_vec: Float[Array, "..."] = jnp.array([1.0, 0.5, 0.3])
        q_vec_neg: Any = -q_vec

        f_pos: Any = var_structure_factor(
            q_vector=q_vec,
            atomic_positions=atomic_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            is_surface_atom=is_surface_atom,
        )

        f_neg: Any = var_structure_factor(
            q_vector=q_vec_neg,
            atomic_positions=atomic_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            is_surface_atom=is_surface_atom,
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
        self, hk_index: list[int], roughness: float
    ) -> None:
        r"""Test CTR intensity calculation for different rods.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR intensity
        calculation for different rods. Existing context from the original test
        prose: Tests crystal truncation rod intensity I(h,k,q_z) =
        \|F(h,k,q_z)\|² × D(q_z) for various rods: specular (0,0), first-order
        (1,0), diagonal (1,1), and high-order (2,0). Tests with roughness
        values from 0.5 to 2.0 Å. Verifies that intensities are non-negative,
        finite, and generally decrease with q_z due to roughness damping. The
        CTR intensity profile along q_z contains information about surface
        structure, roughness, and relaxation.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``hk_index``,
        ``roughness``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)

        hk_array: Integer[Array, "..."] = jnp.array(
            [hk_index], dtype=jnp.int32
        )

        intensities: Float[Array, "..."] = var_ctr(
            hk_indices=hk_array,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=roughness,
            temperature=300.0,
        )

        chex.assert_shape(intensities, (1, len(self.l_values)))

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        chex.assert_tree_all_finite(intensities)

        if roughness > 0.1:
            n_points: Float[Array, "..."] = len(self.l_values)
            first_quarter_mean: scalar_float = jnp.mean(
                intensities[0, : n_points // 4]
            )
            last_quarter_mean: scalar_float = jnp.mean(
                intensities[0, -n_points // 4 :]
            )

            chex.assert_scalar_positive(
                float(first_quarter_mean - last_quarter_mean)
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_ctr_intensity_multiple_rods(self) -> None:
        r"""Test CTR calculation for multiple rods simultaneously.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR calculation
        for multiple rods simultaneously. Existing context from the original
        test prose: Tests vectorized calculation of multiple crystal truncation
        rods in parallel: (0,0) specular, (1,0) first-order, (1,1) diagonal,
        and (2,0) second-order. Verifies that the function correctly computes a
        2D array of intensities (N_rods × N_q_z), all values are physically
        valid (non-negative, finite), and different rods show distinct
        intensity patterns reflecting their unique structure factors. This
        parallelization is crucial for efficient RHEED pattern calculation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)

        intensities: Float[Array, "..."] = var_ctr(
            hk_indices=self.hk_indices,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=1.0,
            temperature=300.0,
        )

        expected_shape: tuple[Any, ...] = (
            len(self.hk_indices),
            len(self.l_values),
        )
        chex.assert_shape(intensities, expected_shape)

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)
        chex.assert_tree_all_finite(intensities)

        rod_differences: scalar_float = jnp.std(intensities, axis=0)
        chex.assert_scalar_positive(float(jnp.mean(rod_differences)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_roughness_effect_on_ctr(self) -> None:
        r"""Test that roughness reduces CTR intensity at high q_z.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roughness reduces
        CTR intensity at high q_z. Existing context from the original test
        prose: Compares CTR intensities for smooth (σ=0.1 Å) vs rough (σ=2.0 Å)
        surfaces. Verifies that roughness increasingly damps intensity at
        higher q_z values according to exp(-q_z²σ²). Tests the (1,0) rod and
        checks that: 1. Smooth surface maintains higher intensity at large q_z
        2. Rough surface shows stronger damping, especially at high q_z 3. The
        relative reduction is significant (measurable experimentally) This
        damping effect allows roughness determination from CTR measurements.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)

        hk_test: Integer[Array, "..."] = jnp.array([[1, 0]], dtype=jnp.int32)

        intensities_smooth: Float[Array, "..."] = var_ctr(
            hk_indices=hk_test,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=0.1,
            temperature=300.0,
        )

        intensities_rough: Float[Array, "..."] = var_ctr(
            hk_indices=hk_test,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=2.0,
            temperature=300.0,
        )

        n_half: int = len(self.l_values) // 2
        smooth_high_qz: Any = intensities_smooth[0, n_half:]
        rough_high_qz: Any = intensities_rough[0, n_half:]

        chex.assert_trees_all_equal(
            jnp.all(smooth_high_qz >= rough_high_qz), True
        )

        relative_reduction: scalar_float = jnp.mean(
            (smooth_high_qz - rough_high_qz) / (smooth_high_qz + 1e-10)
        )
        chex.assert_scalar_positive(float(relative_reduction))

    @parameterized.named_parameters(
        ("narrow_range", (0.0, 1.0), 0.1),
        ("wide_range", (0.0, 5.0), 0.5),
        ("offset_range", (2.0, 4.0), 0.2),
    )
    def test_integrated_rod_intensity(
        self,
        q_z_range: tuple[float, float],
        detector_acceptance_inv_ang: float,
    ) -> None:
        r"""Test integrated CTR intensity over detector acceptance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: integrated CTR
        intensity over detector acceptance. Existing context from the original
        test prose: Tests numerical integration of CTR intensity over a q_z
        range, simulating finite detector acceptance. Tests various integration
        ranges (narrow: 0-1, wide: 0-5, offset: 2-4) and detector acceptance
        window widths (Gaussian sigma of 0.1-0.5 inverse Angstroms in q_z).
        The window-weighted mean intensity represents what a real detector
        pixel measures, accounting for its finite q_z acceptance.
        Verifies that integrated intensity is a positive scalar value,
        essential for comparing with experimental measurements. Note: This
        function has built-in JIT with static_argnames for
        n_integration_points, so we call it directly instead of using
        self.variant() to avoid double-JIT issues.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``q_z_range``, ``detector_acceptance_inv_ang``, so the documented
        behavior is checked across the cases supplied by pytest, Chex,
        Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        integrated: Integer[Array, "..."] = integrated_rod_intensity(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            q_z_range=jnp.array(q_z_range, dtype=jnp.float64),
            crystal=self.test_crystal,
            surface_roughness=1.0,
            detector_acceptance_inv_ang=detector_acceptance_inv_ang,
            n_integration_points=30,
            temperature=300.0,
        )

        chex.assert_shape(integrated, ())
        chex.assert_tree_all_finite(integrated)

        chex.assert_scalar_positive(float(integrated))

    @chex.variants(with_jit=True, without_jit=True)
    def test_temperature_effect_on_ctr(self) -> None:
        r"""Test that higher temperature reduces CTR intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: higher temperature
        reduces CTR intensity. Existing context from the original test prose:
        Compares CTR intensities at low (100K) and high (600K) temperatures for
        the (1,1) rod. Higher temperatures increase atomic thermal vibrations,
        leading to larger Debye-Waller factors and reduced coherent scattering.
        Verifies that the mean intensity decreases with temperature, reflecting
        the exp(-B·q²) thermal damping where B ∝ T. This temperature dependence
        is crucial for understanding RHEED patterns at different growth
        temperatures.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)

        hk_test: Integer[Array, "..."] = jnp.array([[1, 1]], dtype=jnp.int32)

        intensities_cold: Float[Array, "..."] = var_ctr(
            hk_indices=hk_test,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=0.5,
            temperature=100.0,
        )

        intensities_hot: Float[Array, "..."] = var_ctr(
            hk_indices=hk_test,
            l_values=self.l_values,
            crystal=self.test_crystal,
            surface_roughness=0.5,
            temperature=600.0,
        )

        mean_cold: scalar_float = jnp.mean(intensities_cold)
        mean_hot: scalar_float = jnp.mean(intensities_hot)
        chex.assert_scalar_positive(float(mean_cold - mean_hot))

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_ctr(self) -> None:
        r"""Test JAX transformations on CTR calculations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JAX
        transformations on CTR calculations. Existing context from the original
        test prose: Validates JAX functional transformations on CTR
        calculations: 1. vmap: Vectorizes over roughness values (0, 0.5, 1.0,
        2.0 Å) to compute multiple CTR profiles efficiently in parallel,
        essential for parameter sweeps and fitting procedures. 2. grad:
        Computes the gradient of CTR intensity with respect to roughness,
        verifying it's negative (more roughness reduces intensity). This
        gradient is crucial for optimization algorithms that fit CTR data to
        extract surface parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, vectorization, protecting
        JAX transform compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)

        def ctr_for_roughness(sigma: scalar_float) -> Float[Array, "1 10"]:
            return var_ctr(
                hk_indices=jnp.array([[1, 0]], dtype=jnp.int32),
                l_values=self.l_values[:10],
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )

        vmapped_ctr: Callable[[Float[Array, "R"]], Float[Array, "R 1 10"]] = (
            jax.vmap(ctr_for_roughness)
        )
        intensities_batch: Float[Array, "R 1 10"] = vmapped_ctr(
            self.roughness_values
        )

        expected_shape: tuple[int, int, int] = (
            len(self.roughness_values),
            1,
            10,
        )
        chex.assert_shape(intensities_batch, expected_shape)

        def loss_fn(sigma: float) -> scalar_float:
            intensities: Float[Array, "1 1"] = var_ctr(
                hk_indices=jnp.array([[0, 0]], dtype=jnp.int32),
                l_values=jnp.array([2.0]),
                crystal=self.test_crystal,
                surface_roughness=sigma,
                temperature=300.0,
            )
            return jnp.squeeze(intensities)

        grad_fn: Callable[[scalar_float], scalar_float] = jax.grad(loss_fn)
        grad_roughness: scalar_float = grad_fn(1.0)

        chex.assert_shape(grad_roughness, ())
        chex.assert_tree_all_finite(grad_roughness)
        chex.assert_scalar_positive(float(-grad_roughness))

    @chex.variants(with_jit=True, without_jit=True)
    def test_profile_comparison(self) -> None:
        r"""Test that Gaussian and Lorentzian profiles differ appropriately.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gaussian and
        Lorentzian profiles differ appropriately. Existing context from the
        original test prose: Compares Gaussian and Lorentzian rod profiles with
        the same correlation length (50 Å). Both profiles are normalized
        (peak=1 at q_perp=0) but differ in their tails: Lorentzian has
        power-law decay (1/q_perp²) producing heavier tails, while Gaussian has
        exponential decay (exp(-q_perp²)) producing lighter tails. This
        difference is important for distinguishing surface morphologies:
        Gaussian for random roughness, Lorentzian for step-terrace structures.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_gaussian: Callable[..., Any] = self.variant(gaussian_rod_profile)
        var_lorentzian: Callable[..., Any] = self.variant(
            lorentzian_rod_profile
        )

        correlation: float = 50.0

        gaussian_profile: Float[Array, "..."] = var_gaussian(
            self.q_perp_values, correlation
        )
        lorentzian_profile: Float[Array, "..."] = var_lorentzian(
            self.q_perp_values, correlation
        )

        chex.assert_trees_all_close(
            gaussian_profile[0], lorentzian_profile[0], rtol=1e-10
        )

        tail_indices: Any = self.q_perp_values > 0.3
        gaussian_tail: Any = gaussian_profile[tail_indices]
        lorentzian_tail: Any = lorentzian_profile[tail_indices]

        tail_diff: scalar_float = jnp.mean(lorentzian_tail - gaussian_tail)
        chex.assert_scalar_positive(float(tail_diff))

    @chex.variants(with_jit=True, without_jit=True)
    def test_physical_consistency(self) -> None:
        r"""Test the unified truncation-rod decomposition of CTR intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the CTR
        intensity is exactly the unified semi-infinite truncation-rod
        model I(h,k,l) = \|F_cell(q)\|² × T(l) × exp(-q_z²σ²), with q =
        h·b1 + k·b2 + l·b3, T(l) the truncation intensity factor, and the
        roughness damping applied as the squared amplitude factor. The
        decomposition is rebuilt from surface_structure_factor,
        ctr_truncation_intensity, and roughness_damping and compared to
        calculate_ctr_intensity on the (0,0), (1,0), and (2,0) rods at
        l = 0.37 (an arbitrary off-Bragg point). The retired basis-only
        formula \|F\|² × exp(-q_z²σ²/2) had no truncation factor and only
        half the roughness exponent.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ctr: Callable[..., Any] = self.variant(calculate_ctr_intensity)
        var_damping: Callable[..., Any] = self.variant(roughness_damping)
        var_structure: Callable[..., Any] = self.variant(
            surface_structure_factor
        )

        hk_test: Integer[Array, "..."] = jnp.array(
            [[0, 0], [1, 0], [2, 0]], dtype=jnp.int32
        )
        l_single: Float[Array, "..."] = jnp.array([0.37])
        roughness: float = 0.5
        epsilon: float = 0.01

        intensities: Float[Array, "..."] = var_ctr(
            hk_indices=hk_test,
            l_values=l_single,
            crystal=self.test_crystal,
            surface_roughness=roughness,
            layer_attenuation=epsilon,
            temperature=300.0,
        )
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        recip: Float[Array, "3 3"] = reciprocal_lattice_vectors(
            *self.test_crystal.cell_lengths,
            *self.test_crystal.cell_angles,
        )
        n_atoms: int = self.test_crystal.cart_positions.shape[0]
        is_surface_atom: Bool[Array, "..."] = jnp.zeros(
            n_atoms, dtype=jnp.bool_
        )
        truncation: Float[Array, "..."] = ctr_truncation_intensity(
            l_values=l_single[0], layer_attenuation=epsilon
        )
        for rod_index in range(3):
            hk: Integer[Array, "..."] = hk_test[rod_index]
            q_vec: Float[Array, "..."] = (
                jnp.float64(hk[0]) * recip[0]
                + jnp.float64(hk[1]) * recip[1]
                + l_single[0] * recip[2]
            )
            f_struct: Any = var_structure(
                q_vector=q_vec,
                atomic_positions=self.test_crystal.cart_positions[:, :3],
                atomic_numbers=self.test_crystal.cart_positions[:, 3].astype(
                    jnp.int32
                ),
                temperature=300.0,
                is_surface_atom=is_surface_atom,
            )
            damping: Float[Array, "..."] = var_damping(
                q_z=q_vec[2], sigma_height=roughness
            )
            expected_intensity: Float[Array, "..."] = (
                jnp.abs(f_struct) ** 2 * truncation * damping**2
            )
            chex.assert_trees_all_close(
                intensities[rod_index, 0], expected_intensity, rtol=1e-10
            )


def _make_si_crystal() -> CrystalStructure:
    """Create a 2-atom Si crystal for gradient tests."""
    a_si: float = 5.431
    frac_coords: Float[Array, "..."] = jnp.array(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    )
    cart_coords: Float[Array, "..."] = frac_coords * a_si
    atomic_numbers: Float[Array, "..."] = jnp.full(2, 14.0)
    frac_positions: Float[Array, "..."] = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_positions: Float[Array, "..."] = jnp.column_stack(
        [cart_coords, atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([a_si, a_si, a_si]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


_SI_CRYSTAL = _make_si_crystal()


class TestCTRIntensityGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for CTR intensity."""

    def test_ctr_intensity_grad_roughness(self) -> None:
        r"""CTR intensity gradient w.r.t. roughness is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR intensity
        gradient w.r.t. roughness is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        l_values: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def loss(roughness: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    l_values=l_values,
                    crystal=_SI_CRYSTAL,
                    surface_roughness=roughness,
                    temperature=300.0,
                )
            )

        g: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_ctr_intensity_grad_temperature(self) -> None:
        r"""CTR intensity gradient w.r.t. temperature is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR intensity
        gradient w.r.t. temperature is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        l_values: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def loss(temp: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    l_values=l_values,
                    crystal=_SI_CRYSTAL,
                    surface_roughness=0.5,
                    temperature=temp,
                )
            )

        g: scalar_float = jax.grad(loss)(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)

    def test_ctr_intensity_grad_roughness_correct(self) -> None:
        r"""CTR intensity grad w.r.t. roughness matches finite diff.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR intensity grad
        w.r.t. roughness matches finite diff.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        l_values: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(roughness: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    l_values=l_values,
                    crystal=_SI_CRYSTAL,
                    surface_roughness=roughness,
                    temperature=300.0,
                )
            )

        check_grads(jax_safe(f), (jnp.float64(0.5),), order=1, atol=1e-3)

    def test_ctr_intensity_grad_temperature_correct(self) -> None:
        r"""CTR intensity grad w.r.t. temperature matches finite diff.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR intensity grad
        w.r.t. temperature matches finite diff.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        l_values: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(temp: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    l_values=l_values,
                    crystal=_SI_CRYSTAL,
                    surface_roughness=0.5,
                    temperature=temp,
                )
            )

        check_grads(jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-3)


class TestCTRVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential evaluation for CTR intensity."""

    def test_ctr_intensity_vmap_consistent(self) -> None:
        r"""Batched CTR intensity matches sequential evaluation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Batched CTR
        intensity matches sequential evaluation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        l_values: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(roughness: scalar_float) -> Float[Array, "1 5"]:
            return calculate_ctr_intensity(
                hk_indices=hk_indices,
                l_values=l_values,
                crystal=_SI_CRYSTAL,
                surface_roughness=roughness,
                temperature=300.0,
            )

        roughness_batch: Float[Array, "3"] = jnp.array([0.1, 0.5, 1.0])
        batched: Float[Array, "roughness one qz"] = jax.vmap(f)(
            roughness_batch
        )
        sequential: Float[Array, "..."] = jnp.stack(
            [f(r) for r in roughness_batch]
        )
        chex.assert_trees_all_close(batched, sequential, atol=1e-6)


class TestCtrTruncationFactors(chex.TestCase, parameterized.TestCase):
    """Tests for the semi-infinite CTR truncation factors.

    :see: :func:`~rheedium.simul.ctr_truncation_amplitude`
    :see: :func:`~rheedium.simul.ctr_truncation_intensity`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_amplitude_squared_equals_intensity(self) -> None:
        r"""The squared truncation amplitude equals the intensity factor.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the complex
        truncation amplitude 1/(1 - exp(-2*pi*i*l) exp(-eps)) and the
        intensity factor 1/(1 - 2 exp(-eps) cos(2*pi*l) + exp(-2*eps))
        satisfy \|amplitude\|^2 == intensity across Bragg, anti-Bragg,
        and generic l values.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.

        It runs through the Chex variant wrapper where present, so the
        same assertion covers both transformed and untransformed JAX
        execution paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_amp: Callable[..., Any] = self.variant(ctr_truncation_amplitude)
        var_int: Callable[..., Any] = self.variant(ctr_truncation_intensity)

        l_values: Float[Array, "..."] = jnp.linspace(-1.3, 2.7, 201)
        amplitude: Any = var_amp(l_values, 0.02)
        intensity: Any = var_int(l_values, 0.02)
        chex.assert_trees_all_close(
            jnp.abs(amplitude) ** 2, intensity, rtol=1e-12
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_bragg_cap_and_anti_bragg_value(self) -> None:
        r"""Truncation intensity caps at Bragg and is ~1/4 at anti-Bragg.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: at integer l
        the truncation intensity equals the exact cap 1/(1-exp(-eps))^2
        (the small-eps form 1/(4 sinh^2(eps/2)) up to exp(-eps)), and at
        half-integer l it equals 1/(1+exp(-eps))^2 which approaches 1/4
        as eps goes to zero, matching the 1/(4 sin^2(pi l)) semi-infinite
        limit.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.

        It runs through the Chex variant wrapper where present, so the
        same assertion covers both transformed and untransformed JAX
        execution paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_int: Callable[..., Any] = self.variant(ctr_truncation_intensity)

        epsilon: float = 0.01
        at_bragg: Any = var_int(jnp.array([0.0, 1.0, 2.0]), epsilon)
        expected_cap: Float[Array, ""] = 1.0 / (1.0 - jnp.exp(-epsilon)) ** 2
        chex.assert_trees_all_close(
            at_bragg, jnp.full(3, expected_cap), rtol=1e-9
        )

        at_anti_bragg: Any = var_int(jnp.array([0.5, 1.5]), epsilon)
        expected_anti: Float[Array, ""] = 1.0 / (1.0 + jnp.exp(-epsilon)) ** 2
        chex.assert_trees_all_close(
            at_anti_bragg, jnp.full(2, expected_anti), rtol=1e-9
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_matches_four_sin_squared_off_bragg(self) -> None:
        r"""Off-Bragg truncation intensity matches 1/(4 sin^2(pi l)).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: away from
        Bragg points the small-attenuation truncation intensity converges
        to the classic semi-infinite crystal truncation rod shape
        1/(4 sin^2(pi l)) within O(eps).

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.

        It runs through the Chex variant wrapper where present, so the
        same assertion covers both transformed and untransformed JAX
        execution paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_int: Callable[..., Any] = self.variant(ctr_truncation_intensity)

        l_values: Float[Array, "..."] = jnp.array([0.1, 0.25, 0.5, 0.8])
        intensity: Any = var_int(l_values, 1e-6)
        reference: Float[Array, "..."] = 1.0 / (
            4.0 * jnp.sin(jnp.pi * l_values) ** 2
        )
        chex.assert_trees_all_close(intensity, reference, rtol=1e-4)


class TestIntegratedWindowConsistency(chex.TestCase, parameterized.TestCase):
    """Window-normalization consistency of the integrated rod quantities.

    :see: :func:`~rheedium.simul.integrated_ctr_amplitude`
    :see: :func:`~rheedium.simul.integrated_rod_intensity`
    """

    def test_amplitude_squared_matches_intensity_on_smooth_rod(self) -> None:
        r"""\|integrated amplitude\|^2 tracks integrated intensity to 2%.

        Extended Summary
        ----------------
        WP5.5 acceptance: both integrated_rod_intensity and
        integrated_ctr_amplitude return the acceptance-window-weighted
        mean (sum(w x)/sum(w)), so on a slowly varying rod segment the
        squared modulus of the mean amplitude approximates the mean
        intensity within 2 percent. The retired implementation trapezoid-
        integrated the weighted intensity without normalizing, making the
        coherent and incoherent paths differ by an arbitrary scale.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case, using a narrow acceptance window on an off-Bragg rod
        segment where the CTR varies slowly.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_index: Int[Array, "2"] = jnp.array([1, 0], dtype=jnp.int32)
        q_z_range: Float[Array, "2"] = jnp.array([0.45, 0.55])
        acceptance_inv_ang: float = 0.02

        integrated_intensity: Any = integrated_rod_intensity(
            hk_index=hk_index,
            q_z_range=q_z_range,
            crystal=_SI_CRYSTAL,
            surface_roughness=0.5,
            detector_acceptance_inv_ang=acceptance_inv_ang,
            n_integration_points=101,
        )
        integrated_amplitude: Any = integrated_ctr_amplitude(
            hk_index=hk_index,
            q_z_range=q_z_range,
            crystal=_SI_CRYSTAL,
            surface_roughness=0.5,
            detector_acceptance_inv_ang=acceptance_inv_ang,
            n_integration_points=101,
        )
        ratio: float = float(
            jnp.abs(integrated_amplitude) ** 2 / integrated_intensity
        )
        self.assertLess(abs(ratio - 1.0), 0.02)

    def test_window_mean_of_constant_rod_is_identity(self) -> None:
        r"""The window-weighted mean of a constant integrand is itself.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: because both
        integrated quantities are normalized means (sum(w x)/sum(w)), a
        vanishing acceptance window width or a flat rod cannot rescale
        the result; the mean intensity lies between the minimum and
        maximum of the pointwise CTR intensity over the sampled range.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.

        The result is checked with direct unittest or Chex assertions
        against min/max bounds of the pointwise intensities.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_surface_rods``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        hk_index: Int[Array, "2"] = jnp.array([1, 0], dtype=jnp.int32)
        q_z_range: Float[Array, "2"] = jnp.array([0.9, 1.3])

        integrated: Any = integrated_rod_intensity(
            hk_index=hk_index,
            q_z_range=q_z_range,
            crystal=_SI_CRYSTAL,
            surface_roughness=0.5,
            detector_acceptance_inv_ang=0.1,
            n_integration_points=64,
        )
        recip: Float[Array, "3 3"] = reciprocal_lattice_vectors(
            *_SI_CRYSTAL.cell_lengths,
            *_SI_CRYSTAL.cell_angles,
        )
        b3_z: Any = recip[2, 2]
        q_z_values: Float[Array, "64"] = jnp.linspace(0.9, 1.3, 64)
        l_values: Float[Array, "64"] = q_z_values / b3_z
        pointwise: Any = calculate_ctr_intensity(
            hk_indices=hk_index[None, :],
            l_values=l_values,
            crystal=_SI_CRYSTAL,
            surface_roughness=0.5,
        )[0]
        self.assertGreaterEqual(float(integrated), float(jnp.min(pointwise)))
        self.assertLessEqual(float(integrated), float(jnp.max(pointwise)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
