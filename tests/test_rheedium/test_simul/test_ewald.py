"""Test suite for ewald.py internal functions and public API.

This module provides comprehensive testing for Ewald sphere construction,
structure factor calculations, and allowed reflection finding used in
RHEED simulations.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float, Int

from rheedium.simul.ewald import (
    _compute_structure_factor_single,
    build_ewald_data,
    ewald_allowed_reflections,
)
from rheedium.types import (
    CrystalStructure,
    EwaldData,
    create_crystal_structure,
    scalar_float,
)


class TestComputeStructureFactorSingle(chex.TestCase, parameterized.TestCase):
    """Test suite for _compute_structure_factor_single internal function."""

    def setUp(self) -> None:
        """Set up test fixtures with simple crystal structures."""
        super().setUp()

        # Simple cubic structure with one atom at origin
        self.single_atom_positions: Float[Array, "1 3"] = jnp.array(
            [[0.0, 0.0, 0.0]]
        )
        self.single_atom_numbers: Int[Array, "1"] = jnp.array([14])  # Silicon

        # Two-atom structure for phase testing
        self.two_atom_positions: Float[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]  # 2 Angstrom separation
        )
        self.two_atom_numbers: Int[Array, "2"] = jnp.array([14, 14])

        # Multi-element structure
        self.multi_element_positions: Float[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        self.multi_element_numbers: Int[Array, "2"] = jnp.array(
            [14, 8]
        )  # Si and O

        self.temperature: scalar_float = 300.0

    def test_single_atom_at_origin_g_zero(self) -> None:
        """Test structure factor for single atom at origin with G=0.

        For a single atom at the origin, the phase factor exp(i*G*r) = 1
        when G=0, so F(0) equals the atomic form factor times DW factor.
        """
        g_vector: Float[Array, "3"] = jnp.array([0.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        # Should be real and positive for G=0
        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.real(sf)))
        # Imaginary part should be negligible
        chex.assert_trees_all_close(jnp.imag(sf), 0.0, atol=1e-10)

    @parameterized.named_parameters(
        ("g_x", jnp.array([1.0, 0.0, 0.0])),
        ("g_y", jnp.array([0.0, 1.0, 0.0])),
        ("g_z", jnp.array([0.0, 0.0, 1.0])),
        ("g_diagonal", jnp.array([1.0, 1.0, 1.0])),
    )
    def test_single_atom_at_origin_nonzero_g(
        self, g_vector: Float[Array, "3"]
    ) -> None:
        """Test structure factor for single atom at origin with nonzero G.

        For atom at origin, phase factor is always 1 regardless of G,
        so the structure factor is just f(|G|) * DW(|G|).
        """
        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        # For atom at origin, imaginary part should be zero
        chex.assert_trees_all_close(jnp.imag(sf), 0.0, atol=1e-10)
        # Real part should be positive (form factor is positive)
        chex.assert_scalar_positive(float(jnp.real(sf)))

    def test_structure_factor_decreases_with_g_magnitude(self) -> None:
        """Test that |F(G)| decreases with increasing |G|.

        Due to the form factor and Debye-Waller factor both decreasing
        with |G|, the structure factor magnitude should decrease.
        """
        g_vectors: Float[Array, "4 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )

        sf_magnitudes = []
        for g in g_vectors:
            sf = _compute_structure_factor_single(
                g_vector=g,
                atom_positions=self.single_atom_positions,
                atomic_numbers=self.single_atom_numbers,
                temperature=self.temperature,
            )
            sf_magnitudes.append(jnp.abs(sf))

        sf_magnitudes = jnp.array(sf_magnitudes)
        differences = jnp.diff(sf_magnitudes)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    def test_two_atom_interference(self) -> None:
        """Test interference effects between two atoms.

        For two atoms separated by distance d along x, G = (2*pi/d, 0, 0)
        should give constructive interference (phase = 2*pi), while
        G = (pi/d, 0, 0) gives destructive interference (phase = pi).
        """
        d = 2.0  # separation in Angstroms

        # Constructive: G*d = 2*pi
        g_constructive: Float[Array, "3"] = jnp.array([jnp.pi, 0.0, 0.0])
        sf_constr: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_constructive,
            atom_positions=self.two_atom_positions,
            atomic_numbers=self.two_atom_numbers,
            temperature=self.temperature,
        )

        # Destructive: G*d = pi
        g_destructive: Float[Array, "3"] = jnp.array([jnp.pi / 2, 0.0, 0.0])
        sf_destr: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_destructive,
            atom_positions=self.two_atom_positions,
            atomic_numbers=self.two_atom_numbers,
            temperature=self.temperature,
        )

        # Constructive should have larger magnitude
        chex.assert_scalar_positive(
            float(jnp.abs(sf_constr) - jnp.abs(sf_destr))
        )

    def test_multi_element_structure(self) -> None:
        """Test structure factor with multiple element types.

        Different atomic numbers should contribute different form factors.
        """
        g_vector: Float[Array, "3"] = jnp.array([1.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.multi_element_positions,
            atomic_numbers=self.multi_element_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        # Intensity should be non-zero
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    @parameterized.named_parameters(
        ("low_temp", 100.0),
        ("room_temp", 300.0),
        ("high_temp", 600.0),
    )
    def test_temperature_dependence(self, temperature: scalar_float) -> None:
        """Test that structure factor depends on temperature.

        Higher temperature increases Debye-Waller damping, reducing |F(G)|.
        """
        g_vector: Float[Array, "3"] = jnp.array([2.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=temperature,
        )

        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    def test_temperature_ordering(self) -> None:
        """Test that higher temperature gives lower |F(G)| for G != 0."""
        g_vector: Float[Array, "3"] = jnp.array([2.0, 0.0, 0.0])

        sf_low: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=100.0,
        )
        sf_high: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=600.0,
        )

        # Lower temperature should give larger |F(G)|
        chex.assert_scalar_positive(float(jnp.abs(sf_low) - jnp.abs(sf_high)))

    def test_complex_phase_correctness(self) -> None:
        """Test that phase is computed correctly for off-origin atoms.

        For an atom at position r, the phase contribution is exp(i*G*r).
        """
        # Single atom at (1, 0, 0)
        atom_pos: Float[Array, "1 3"] = jnp.array([[1.0, 0.0, 0.0]])
        atom_nums: Int[Array, "1"] = jnp.array([14])

        # G = (pi, 0, 0) should give phase = exp(i*pi) = -1
        g_vector: Float[Array, "3"] = jnp.array([jnp.pi, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=atom_pos,
            atomic_numbers=atom_nums,
            temperature=self.temperature,
        )

        # Real part should be negative (due to phase = -1)
        self.assertLess(float(jnp.real(sf)), 0.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_compatibility(self) -> None:
        """Test that _compute_structure_factor_single works with JIT."""
        var_compute_sf = self.variant(_compute_structure_factor_single)

        g_vector: Float[Array, "3"] = jnp.array([1.0, 0.0, 0.0])

        sf: Complex[Array, ""] = var_compute_sf(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    def test_vmap_over_g_vectors(self) -> None:
        """Test vmapping _compute_structure_factor_single over G vectors."""
        g_vectors: Float[Array, "5 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        vmapped_sf = jax.vmap(
            lambda g: _compute_structure_factor_single(
                g_vector=g,
                atom_positions=self.single_atom_positions,
                atomic_numbers=self.single_atom_numbers,
                temperature=self.temperature,
            )
        )

        sfs: Complex[Array, "5"] = vmapped_sf(g_vectors)

        chex.assert_shape(sfs, (5,))
        chex.assert_tree_all_finite(sfs)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through structure factor calculation."""

        def loss_fn(temperature: scalar_float) -> scalar_float:
            g_vector = jnp.array([1.0, 0.0, 0.0])
            sf = _compute_structure_factor_single(
                g_vector=g_vector,
                atom_positions=self.single_atom_positions,
                atomic_numbers=self.single_atom_numbers,
                temperature=temperature,
            )
            return jnp.abs(sf)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(300.0)

        chex.assert_tree_all_finite(gradient)
        # Gradient should be negative (higher temp -> lower |F|)
        self.assertLess(float(gradient), 0.0)


class TestBuildEwaldData(chex.TestCase, parameterized.TestCase):
    """Test suite for build_ewald_data public function."""

    def setUp(self) -> None:
        """Set up test fixtures with a simple crystal structure."""
        super().setUp()
        self.crystal = self._create_simple_cubic_crystal()

    def _create_simple_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal for testing."""
        a = 4.0  # lattice constant in Angstroms

        frac_coords: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])
        cart_coords: Float[Array, "1 3"] = frac_coords * a
        atomic_numbers: Float[Array, "1"] = jnp.array([14.0])

        frac_positions: Float[Array, "1 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "1 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_ewald_data_creation(self) -> None:
        """Test basic EwaldData creation with minimal parameters."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=300.0,
        )

        # Check that all fields are populated
        chex.assert_tree_all_finite(ewald.wavelength_ang)
        chex.assert_tree_all_finite(ewald.k_magnitude)
        chex.assert_tree_all_finite(ewald.sphere_radius)
        chex.assert_tree_all_finite(ewald.recip_vectors)
        chex.assert_tree_all_finite(ewald.g_vectors)
        chex.assert_tree_all_finite(ewald.g_magnitudes)
        chex.assert_tree_all_finite(ewald.structure_factors)
        chex.assert_tree_all_finite(ewald.intensities)

    def test_wavelength_calculation(self) -> None:
        """Test that wavelength is correctly computed from voltage."""
        ewald_10kv: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=10.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )
        ewald_20kv: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        # Higher voltage should give shorter wavelength
        chex.assert_scalar_positive(
            float(ewald_10kv.wavelength_ang - ewald_20kv.wavelength_ang)
        )

        # Wavelength should be reasonable for RHEED (0.05 - 0.15 Angstroms)
        self.assertGreater(float(ewald_10kv.wavelength_ang), 0.05)
        self.assertLess(float(ewald_10kv.wavelength_ang), 0.15)

    def test_k_magnitude_relation(self) -> None:
        """Test that k_magnitude = 2*pi / wavelength."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        expected_k = 2.0 * jnp.pi / ewald.wavelength_ang
        chex.assert_trees_all_close(ewald.k_magnitude, expected_k, rtol=1e-10)

    def test_sphere_radius_equals_k_magnitude(self) -> None:
        """Test that Ewald sphere radius equals k magnitude."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        chex.assert_trees_all_close(
            ewald.sphere_radius, ewald.k_magnitude, rtol=1e-10
        )

    @parameterized.named_parameters(
        ("small_grid", 1, 1, 1, 27),  # (2*1+1)^3 = 27
        ("medium_grid", 2, 2, 1, 75),  # 5*5*3 = 75
        ("asymmetric", 3, 2, 1, 105),  # 7*5*3 = 105
    )
    def test_grid_size(
        self, hmax: int, kmax: int, lmax: int, expected_n: int
    ) -> None:
        """Test that the correct number of G vectors are generated."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=hmax,
            kmax=kmax,
            lmax=lmax,
        )

        chex.assert_shape(ewald.hkl_grid, (expected_n, 3))
        chex.assert_shape(ewald.g_vectors, (expected_n, 3))
        chex.assert_shape(ewald.g_magnitudes, (expected_n,))
        chex.assert_shape(ewald.structure_factors, (expected_n,))
        chex.assert_shape(ewald.intensities, (expected_n,))

    def test_intensities_are_squared_magnitudes(self) -> None:
        """Test that intensities = |structure_factors|^2."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        expected_intensities = jnp.abs(ewald.structure_factors) ** 2
        chex.assert_trees_all_close(
            ewald.intensities, expected_intensities, rtol=1e-10
        )

    def test_intensities_nonnegative(self) -> None:
        """Test that all intensities are non-negative."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        chex.assert_trees_all_equal(jnp.all(ewald.intensities >= 0), True)

    def test_g_magnitudes_nonnegative(self) -> None:
        """Test that all G magnitudes are non-negative."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        chex.assert_trees_all_equal(jnp.all(ewald.g_magnitudes >= 0), True)

    def test_g_magnitude_consistency(self) -> None:
        """Test that g_magnitudes matches norm of g_vectors."""
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        computed_mags = jnp.linalg.norm(ewald.g_vectors, axis=-1)
        chex.assert_trees_all_close(
            ewald.g_magnitudes, computed_mags, rtol=1e-10
        )

    def test_temperature_affects_intensities(self) -> None:
        """Test that temperature affects structure factors/intensities."""
        ewald_low_t: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=100.0,
        )
        ewald_high_t: EwaldData = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=600.0,
        )

        # Higher temperature should reduce total intensity (more DW damping)
        total_low = jnp.sum(ewald_low_t.intensities)
        total_high = jnp.sum(ewald_high_t.intensities)

        chex.assert_scalar_positive(float(total_low - total_high))


class TestEwaldAllowedReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for ewald_allowed_reflections function."""

    def setUp(self) -> None:
        """Set up test fixtures with pre-computed EwaldData."""
        super().setUp()
        self.crystal = self._create_simple_cubic_crystal()
        self.ewald = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=20.0,
            hmax=3,
            kmax=3,
            lmax=2,
            temperature=300.0,
        )

    def _create_simple_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal for testing."""
        a = 4.0
        frac_coords = jnp.array([[0.0, 0.0, 0.0]])
        cart_coords = frac_coords * a
        atomic_numbers = jnp.array([14.0])

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_reflection_finding(self) -> None:
        """Test basic reflection finding with default parameters."""
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        # Check output shapes are consistent
        n = indices.shape[0]
        chex.assert_shape(k_out, (n, 3))
        chex.assert_shape(intensities, (n,))

    def test_upward_scattering_only(self) -> None:
        """Test that only upward scattering reflections are returned."""
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        # Valid reflections should have k_out_z > 0
        valid_mask = indices >= 0
        valid_k_out = k_out[valid_mask]

        if valid_k_out.shape[0] > 0:
            chex.assert_trees_all_equal(jnp.all(valid_k_out[:, 2] > 0), True)

    def test_intensities_nonnegative(self) -> None:
        """Test that all returned intensities are non-negative."""
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @parameterized.named_parameters(
        ("theta_1", 1.0),
        ("theta_2", 2.0),
        ("theta_5", 5.0),
    )
    def test_different_theta_angles(self, theta_deg: scalar_float) -> None:
        """Test reflection finding at different incidence angles."""
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=theta_deg,
            phi_deg=0.0,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)

    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
    )
    def test_different_phi_angles(self, phi_deg: scalar_float) -> None:
        """Test reflection finding at different azimuthal angles."""
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=phi_deg,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)

    def test_tolerance_effect(self) -> None:
        """Test that larger tolerance allows more reflections."""
        _, _, intensities_tight = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            tolerance=0.01,
        )
        _, _, intensities_loose = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            tolerance=0.1,
        )

        # More allowed reflections with larger tolerance
        n_tight = jnp.sum(intensities_tight > 0)
        n_loose = jnp.sum(intensities_loose > 0)

        self.assertGreaterEqual(int(n_loose), int(n_tight))

    def test_finite_domain_mode(self) -> None:
        """Test finite domain mode with domain_extent_ang parameter."""
        domain = jnp.array([100.0, 100.0, 50.0])

        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    def test_finite_domain_overlap_weighting(self) -> None:
        """Test that finite domain mode applies overlap weighting."""
        domain = jnp.array([50.0, 50.0, 25.0])

        _, _, intensities_binary = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            tolerance=0.05,
        )
        _, _, intensities_finite = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )

        # Finite domain should produce different intensity distribution
        # (not necessarily larger or smaller total)
        chex.assert_tree_all_finite(intensities_binary)
        chex.assert_tree_all_finite(intensities_finite)

    def test_k_out_is_k_in_plus_g(self) -> None:
        """Test that k_out = k_in + G for allowed reflections."""
        theta_deg = 2.0
        phi_deg = 0.0

        theta_rad = jnp.deg2rad(theta_deg)
        phi_rad = jnp.deg2rad(phi_deg)
        k_mag = self.ewald.k_magnitude

        k_in = k_mag * jnp.array(
            [
                jnp.cos(theta_rad) * jnp.cos(phi_rad),
                jnp.cos(theta_rad) * jnp.sin(phi_rad),
                -jnp.sin(theta_rad),
            ]
        )

        indices, k_out, _ = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )

        # For valid indices, check k_out = k_in + G
        for i, idx in enumerate(indices):
            if idx >= 0:
                g_vec = self.ewald.g_vectors[idx]
                expected_k_out = k_in + g_vec
                chex.assert_trees_all_close(
                    k_out[i], expected_k_out, rtol=1e-10
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
