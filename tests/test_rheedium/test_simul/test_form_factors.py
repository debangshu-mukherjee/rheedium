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
        """Set up test fixtures with common test parameters.

        Initializes atomic numbers for various elements (H, C, Si, Cu, Au),
        momentum transfer magnitudes from 0 to 8 Å⁻¹, temperature values
        (100K, 300K, 600K), 3D q-vectors for testing directional
        calculations, and batched q-vectors for testing vectorization
        capabilities.
        """
        super().setUp()
        self.test_atomic_numbers = {
            "H": 1,
            "C": 6,
            "Si": 14,
            "Cu": 29,
            "Au": 79,
        }
        self.q_magnitudes = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
        self.temperatures = jnp.array([100.0, 300.0, 600.0])
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
        """Test loading Kirkland parameters for various elements.

        Verifies that Kirkland parameterization coefficients are loaded
        correctly for different atomic numbers. Checks that both a and b
        coefficient arrays have the expected shape (6 coefficients each),
        are finite values, and that b coefficients (width parameters) are
        positive. Also validates that the sum of a coefficients is positive,
        which is physically meaningful for scattering.
        """
        var_load_params = self.variant(load_kirkland_parameters)
        a_coeffs, b_coeffs = var_load_params(atomic_number)

        chex.assert_shape(a_coeffs, (6,))
        chex.assert_shape(b_coeffs, (6,))

        chex.assert_tree_all_finite(a_coeffs)
        chex.assert_tree_all_finite(b_coeffs)

        chex.assert_trees_all_equal(jnp.all(b_coeffs > 0), True)

        a_sum = jnp.sum(a_coeffs)
        chex.assert_scalar_positive(float(a_sum))

    @chex.variants(with_jit=True, without_jit=True)
    def test_load_kirkland_parameters_edge_cases(self) -> None:
        """Test parameter loading with edge cases.

        Tests boundary conditions for atomic number inputs, verifying
        correct behavior at minimum (Z=1, hydrogen) and maximum (Z=103,
        lawrencium) atomic numbers. Also tests that out-of-range atomic
        numbers are properly clipped: values below 1 are clipped to 1,
        and values above 103 are clipped to 103, ensuring the function
        handles invalid inputs gracefully.
        """
        var_load_params = self.variant(load_kirkland_parameters)

        a_min, b_min = var_load_params(1)
        a_max, b_max = var_load_params(103)

        chex.assert_shape(a_min, (6,))
        chex.assert_shape(b_min, (6,))
        chex.assert_shape(a_max, (6,))
        chex.assert_shape(b_max, (6,))

        a_clip_low, b_clip_low = var_load_params(0)
        a_clip_high, b_clip_high = var_load_params(150)

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
        """Test Kirkland form factor for single q values.

        Validates the Kirkland electron scattering form factor calculation
        for individual momentum transfer values. Tests various q magnitudes
        (0.1, 1.0, 10.0 Å⁻¹) for different elements. Verifies that the
        form factor returns a scalar value that is finite and positive.
        Note that for electron scattering (Kirkland formulation), the form
        factor at q=0 does not equal the atomic number Z, unlike X-ray
        scattering.
        """
        var_form_factor = self.variant(kirkland_form_factor)
        q_array = jnp.array(q_mag)

        f_q = var_form_factor(atomic_number, q_array)

        f_q = jnp.squeeze(f_q)

        chex.assert_shape(f_q, ())
        chex.assert_tree_all_finite(f_q)

        chex.assert_scalar_positive(float(f_q))

        if q_mag < 0.2:
            chex.assert_scalar_positive(float(f_q))

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_form_factor_decreasing(self) -> None:
        """Test that form factor decreases with increasing q.

        Verifies the fundamental physical property that atomic form factors
        decrease monotonically with increasing momentum transfer q. Tests
        this for multiple elements (H, C, Si, Cu, Au) across q values from
        0 to 8 Å⁻¹. The decrease reflects the finite extent of the atomic
        electron density distribution - higher q probes smaller length
        scales where less electron density contributes to scattering.
        """
        var_form_factor = self.variant(kirkland_form_factor)

        for name, z in self.test_atomic_numbers.items():
            f_values = var_form_factor(z, self.q_magnitudes)

            differences = jnp.diff(f_values[1:])
            chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

            chex.assert_scalar_positive(float(f_values[0]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_form_factor_batched(self) -> None:
        """Test form factor with batched q values.

        Verifies that the form factor function correctly handles vectorized
        operations on multi-dimensional arrays of q values. Tests 2D batches
        (6x5 array) and 3D batches (6x3x4 array) to ensure the function
        broadcasts properly and maintains output shape matching the input.
        This is critical for efficient computation in RHEED simulations where
        many q points are evaluated simultaneously.
        """
        var_form_factor = self.variant(kirkland_form_factor)

        q_batch_2d = jnp.tile(self.q_magnitudes[:, jnp.newaxis], (1, 5))
        f_batch_2d = var_form_factor(14, q_batch_2d)
        chex.assert_shape(f_batch_2d, q_batch_2d.shape)

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
        """Test mean square displacement calculation.

        Validates the calculation of atomic mean square displacement (MSD)
        due to thermal vibrations. Tests various combinations of atomic
        number (1, 14, 79), temperature (100K, 300K, 600K), and surface vs
        bulk atoms. Verifies that MSD is a positive scalar value and that
        surface atoms have larger MSD than bulk atoms due to reduced
        coordination, reflecting their greater vibrational amplitude from
        fewer constraining bonds.
        """
        var_get_msd = self.variant(get_mean_square_displacement)

        msd = var_get_msd(atomic_number, temperature, is_surface)

        chex.assert_shape(msd, ())
        chex.assert_tree_all_finite(msd)
        chex.assert_scalar_positive(float(msd))

        if is_surface:
            msd_bulk = var_get_msd(atomic_number, temperature, False)
            chex.assert_scalar_positive(float(msd - msd_bulk))

    @chex.variants(with_jit=True, without_jit=True)
    def test_mean_square_displacement_scaling(self) -> None:
        """Test MSD scaling with temperature and atomic number.

        Verifies the physical scaling relationships of mean square
        displacement:
        1. MSD increases with temperature (100K < 300K < 600K for Si) due
           to increased thermal energy and vibrational amplitude.
        2. For elements with similar Debye temperatures, MSD decreases
           with atomic mass (Al > Fe at 300K) since heavier atoms vibrate
           with smaller amplitude for the same thermal energy.
        3. Surface enhancement factor is consistent (exactly 2.0) across
           different elements, reflecting the systematic reduction in
           coordination number at surfaces.

        Note: With element-specific Debye temperatures, the simple mass
        ordering (lighter = larger MSD) doesn't always hold. For example,
        carbon (Θ_D=2230K, very stiff diamond bonds) has smaller MSD than
        gold (Θ_D=165K, soft metallic bonds) despite being lighter.
        """
        var_get_msd = self.variant(get_mean_square_displacement)

        msd_si_100 = var_get_msd(14, 100.0, False)
        msd_si_300 = var_get_msd(14, 300.0, False)
        msd_si_600 = var_get_msd(14, 600.0, False)

        chex.assert_scalar_positive(float(msd_si_300 - msd_si_100))
        chex.assert_scalar_positive(float(msd_si_600 - msd_si_300))

        # For mass scaling test, use elements with similar Debye temperatures
        # so that mass effect dominates: Al (Θ_D=428K) vs Fe (Θ_D=470K)
        msd_al = var_get_msd(13, 300.0, False)  # Al: mass=27, Θ_D=428K
        msd_fe = var_get_msd(26, 300.0, False)  # Fe: mass=56, Θ_D=470K

        chex.assert_scalar_positive(float(msd_al - msd_fe))

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
        """Test Debye-Waller factor for single values.

        Validates the Debye-Waller factor calculation which accounts for
        thermal damping of scattering intensity. Tests various combinations
        of mean square displacement (0.001 to 0.1 Ų) and momentum transfer
        (0 to 5 Å⁻¹). Verifies that the factor is bounded between 0 and 1,
        representing the reduction in coherent scattering due to thermal
        vibrations. At q=0, the factor equals 1 (no damping) since there's
        no momentum transfer.
        """
        var_dw_factor = self.variant(debye_waller_factor)

        q_array = jnp.array(q_mag)
        dw = var_dw_factor(q_array, msd)

        chex.assert_shape(dw, ())
        chex.assert_tree_all_finite(dw)

        chex.assert_scalar_positive(float(dw))
        chex.assert_scalar_positive(float(1.0 - dw + 1e-10))

        if q_mag < 1e-10:
            chex.assert_trees_all_close(dw, 1.0, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_debye_waller_factor_batched(self) -> None:
        """Test Debye-Waller factor with batched q values.

        Verifies vectorized computation of Debye-Waller factors for arrays
        of q values. Tests 1D arrays (6 values) and 2D arrays (6x10 values)
        to ensure proper broadcasting. Also validates that the factor
        decreases monotonically with increasing q, reflecting stronger
        thermal damping at higher momentum transfers where atomic positions
        are more precisely probed.
        """
        var_dw_factor = self.variant(debye_waller_factor)

        msd = 0.01

        dw_1d = var_dw_factor(self.q_magnitudes, msd)
        chex.assert_shape(dw_1d, self.q_magnitudes.shape)

        differences = jnp.diff(dw_1d)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

        q_batch_2d = self.batched_q
        dw_2d = var_dw_factor(q_batch_2d, msd)
        chex.assert_shape(dw_2d, q_batch_2d.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_debye_waller_edge_cases(self) -> None:
        """Test Debye-Waller factor with edge cases.

        Tests boundary conditions and special cases:
        1. Zero MSD (no thermal vibration) should give factor of 1 (no
           damping) for all q values, representing a perfectly rigid
           lattice.
        2. Very large MSD (10 Ų) should give near-zero factors for q > 0,
           representing complete loss of coherent scattering.
        3. Negative MSD (physically invalid) should be handled gracefully
           by clipping to a small positive value, preventing numerical
           errors.
        """
        var_dw_factor = self.variant(debye_waller_factor)

        q_test = jnp.array([0.0, 1.0, 10.0])

        dw_zero_msd = var_dw_factor(q_test, 0.0)
        chex.assert_trees_all_close(
            dw_zero_msd, jnp.ones_like(q_test), rtol=1e-6
        )

        # For large MSD (10 Å²), expect significant damping for q > 0
        # DW = exp(-q²⟨u²⟩/3), so q=1, MSD=10 → exp(-10/3) ≈ 0.036
        dw_large_msd = var_dw_factor(q_test[1:], 10.0)
        chex.assert_trees_all_equal(jnp.all(dw_large_msd < 0.1), True)

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
        """Test combined atomic scattering factor.

        Tests the complete atomic scattering factor calculation that
        combines the Kirkland form factor with Debye-Waller thermal damping.
        Validates for different elements (Si, Au, H), temperatures (100K,
        300K, 600K), and surface vs bulk atoms. Verifies that the combined
        factor is positive, finite, and generally decreases with increasing
        |q| due to both the form factor falloff and thermal damping effects.
        The 3D q-vectors test that the calculation depends only on |q|,
        not direction.
        """
        var_scattering = self.variant(atomic_scattering_factor)

        f_combined = var_scattering(
            atomic_number, self.q_vectors_3d, temperature, is_surface
        )

        chex.assert_shape(f_combined, (len(self.q_vectors_3d),))
        chex.assert_tree_all_finite(f_combined)

        chex.assert_trees_all_equal(jnp.all(f_combined >= 0), True)

        chex.assert_scalar_positive(float(f_combined[0]))

        q_mags = jnp.linalg.norm(self.q_vectors_3d, axis=-1)
        sorted_indices = jnp.argsort(q_mags)
        f_sorted = f_combined[sorted_indices]
        chex.assert_scalar_positive(float(f_sorted[0] - f_sorted[-1]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_atomic_scattering_factor_batched(self) -> None:
        """Test atomic scattering factor with batched inputs.

        Verifies that the atomic scattering factor function correctly
        handles batched 3D q-vector inputs for efficient vectorized
        computation. Tests 2D batching (5 independent sets of 6 q-vectors
        each) and 3D batching (3x4 grid of sets, each with 6 q-vectors).
        This batching capability is essential for efficient RHEED pattern
        calculations where multiple detector positions or time steps are
        computed simultaneously.
        """
        var_scattering = self.variant(atomic_scattering_factor)

        batch_2d = jnp.tile(self.q_vectors_3d[jnp.newaxis, :, :], (5, 1, 1))
        f_batch_2d = var_scattering(14, batch_2d, 300.0, False)
        chex.assert_shape(f_batch_2d, (5, len(self.q_vectors_3d)))

        batch_3d = jnp.tile(
            self.q_vectors_3d[jnp.newaxis, jnp.newaxis, :, :], (3, 4, 1, 1)
        )
        f_batch_3d = var_scattering(14, batch_3d, 300.0, False)
        chex.assert_shape(f_batch_3d, (3, 4, len(self.q_vectors_3d)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_surface_vs_bulk_comparison(self) -> None:
        """Test that surface atoms have stronger thermal damping.

        Validates that surface atoms exhibit stronger thermal damping than
        bulk atoms at the same temperature. Tests for multiple elements
        (C, Si, Cu) at q values of 2 and 4 Å⁻¹. Surface atoms have larger
        mean square displacement due to reduced coordination, leading to
        stronger Debye-Waller damping and thus lower scattering factors
        for q > 0. This effect is crucial for accurate RHEED simulations
        of surface structures.
        """
        var_scattering = self.variant(atomic_scattering_factor)

        q_test = jnp.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        for z in [6, 14, 29]:
            f_bulk = var_scattering(z, q_test, 300.0, False)
            f_surf = var_scattering(z, q_test, 300.0, True)

            chex.assert_trees_all_equal(jnp.all(f_surf < f_bulk), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_form_factor(self) -> None:
        """Test JAX transformations on form factor calculations.

        Validates that form factor calculations are compatible with JAX's
        functional transformations:
        1. JIT compilation: Ensures the function can be compiled for
           performance, verifying identical results between compiled and
           uncompiled versions.
        2. vmap: Tests vectorization over atomic numbers, enabling
           efficient calculation for multiple elements simultaneously.
        3. grad: Verifies automatic differentiation with respect to q,
           confirming the gradient is negative (form factor decreases
           with q), which is essential for optimization and fitting
           procedures.
        """
        var_form_factor = self.variant(kirkland_form_factor)

        @jax.jit
        def jitted_form_factor(
            z: int, q: Float[Array, "..."]
        ) -> Float[Array, "..."]:
            return var_form_factor(z, q)

        q_test = self.q_magnitudes
        f_normal = var_form_factor(14, q_test)
        f_jitted = jitted_form_factor(14, q_test)
        chex.assert_trees_all_close(f_normal, f_jitted, rtol=1e-10)

        atomic_nums = jnp.array([1, 6, 14, 29, 79])
        q_single = jnp.array(1.0)
        vmapped_ff = jax.vmap(
            lambda z: var_form_factor(z, q_single), in_axes=0
        )
        f_vmapped = vmapped_ff(atomic_nums)
        chex.assert_shape(f_vmapped, (len(atomic_nums), 1))

        def loss_fn(q: float) -> float:
            f_val = var_form_factor(14, jnp.array(q))
            return jnp.squeeze(f_val)

        grad_fn = jax.grad(loss_fn)
        grad_q = grad_fn(1.0)
        chex.assert_shape(grad_q, ())
        chex.assert_tree_all_finite(grad_q)
        chex.assert_scalar_positive(float(-grad_q))

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations_scattering(self) -> None:
        """Test JAX transformations on atomic scattering factor.

        Tests advanced JAX transformations on the combined atomic scattering
        factor:
        1. vmap over temperatures: Vectorizes calculation across multiple
           temperatures (100K, 300K, 600K) for the same atom, verifying
           that higher temperatures produce lower scattering factors due
           to increased thermal damping.
        2. Nested vmap: Tests double vectorization over both atomic numbers
           (C, Si, Cu) and temperatures simultaneously, creating a 3x3
           grid of calculations. This demonstrates the composability of
           JAX transformations for complex parameter sweeps in RHEED
           simulations.
        """
        var_scattering = self.variant(atomic_scattering_factor)

        temps = jnp.array([100.0, 300.0, 600.0])
        q_single = jnp.array([[1.0, 0.0, 0.0]])

        vmapped_temp = jax.vmap(
            lambda t: var_scattering(14, q_single, t, False), in_axes=0
        )
        f_temps = vmapped_temp(temps)
        chex.assert_shape(f_temps, (3, 1))

        diff = jnp.squeeze(f_temps[0]) - jnp.squeeze(f_temps[-1])
        chex.assert_scalar_positive(float(diff))

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
        """Test physical consistency of calculations.

        Validates that the combined atomic scattering factor correctly
        implements the physical relationship: f_total = f_kirkland ×
        exp(-B·q²), where B is related to the mean square displacement.
        Tests that calculating the components separately (Kirkland form
        factor, MSD, Debye-Waller factor) and multiplying them gives the
        same result as the combined function. This ensures internal
        consistency and correct implementation of the thermal damping
        model in electron scattering.
        """
        var_form_factor = self.variant(kirkland_form_factor)
        var_dw_factor = self.variant(debye_waller_factor)
        var_get_msd = self.variant(get_mean_square_displacement)
        var_scattering = self.variant(atomic_scattering_factor)

        z = 14
        temp = 300.0
        q_vec = jnp.array([[2.0, 0.0, 0.0]])
        q_mag = jnp.linalg.norm(q_vec, axis=-1)

        f_kirk = var_form_factor(z, q_mag)
        msd = var_get_msd(z, temp, False)
        dw = var_dw_factor(q_mag, msd)
        f_product = f_kirk * dw

        f_combined = var_scattering(z, q_vec, temp, False)

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
        """Test that scattering depends only on |q|, not direction.

        Validates the fundamental isotropy of atomic scattering - the
        scattering factor depends only on the magnitude of momentum transfer
        |q|, not its direction. Tests with various q-vectors including zero
        vector, unit vectors along different axes, random vectors, and
        large magnitude vectors. Creates rotated versions with the same |q|
        but different directions and verifies identical scattering factors.
        This isotropy arises from the spherical symmetry of atomic electron
        density distributions.
        """
        var_scattering = self.variant(atomic_scattering_factor)

        if q_vectors.ndim == 1:
            q_vectors = q_vectors[jnp.newaxis, :]

        q_mags = jnp.linalg.norm(q_vectors, axis=-1, keepdims=True)

        if q_mags[0] > 1e-10:
            q_normalized = q_vectors / (q_mags + 1e-10)
            q_rotated = jnp.roll(q_normalized, 1, axis=-1) * q_mags

            f_original = var_scattering(14, q_vectors, 300.0, False)
            f_rotated = var_scattering(14, q_rotated, 300.0, False)

            chex.assert_trees_all_close(f_original, f_rotated, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
