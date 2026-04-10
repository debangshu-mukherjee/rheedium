"""Test suite for RHEED simulation utility functions.

Tests for wavelength_ang, incident_wavevector, and
interaction_constant — the shared physical computations used
across multiple simulation modules.
"""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.test_util import check_grads

from rheedium.tools.simul_utils import (
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.tools.wrappers import jax_safe


class TestWavelengthAng(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic electron wavelength calculation."""

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_known_voltages(self):
        """Relativistic wavelength matches reference values."""
        var_wavelength = self.variant(wavelength_ang)

        voltages_kv = jnp.array([10.0, 20.0, 30.0])
        wavelengths = jax.vmap(var_wavelength)(voltages_kv)

        expected = jnp.array([0.1226, 0.0859, 0.0698])

        chex.assert_trees_all_close(wavelengths, expected, rtol=5e-3)
        chex.assert_tree_all_finite(wavelengths)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_higher_voltage_shorter_wavelength(self):
        """Higher voltage produces shorter wavelength."""
        var_wavelength = self.variant(wavelength_ang)

        lam_10 = var_wavelength(jnp.float64(10.0))
        lam_30 = var_wavelength(jnp.float64(30.0))

        assert float(lam_30) < float(lam_10)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_positive_output(self):
        """Wavelength is always positive."""
        var_wavelength = self.variant(wavelength_ang)

        voltages = jnp.array([5.0, 10.0, 20.0, 50.0, 100.0])
        wavelengths = jax.vmap(var_wavelength)(voltages)

        chex.assert_trees_all_equal(jnp.all(wavelengths > 0), True)

    def test_scalar_input(self):
        """Accepts scalar float input."""
        lam = wavelength_ang(20.0)
        chex.assert_tree_all_finite(lam)

    def test_array_broadcast(self):
        """Handles batched array input."""
        voltages = jnp.array([10.0, 20.0, 30.0])
        wavelengths = wavelength_ang(voltages)
        chex.assert_shape(wavelengths, (3,))
        chex.assert_tree_all_finite(wavelengths)


class TestIncidentWavevector(chex.TestCase, parameterized.TestCase):
    """Tests for incident wavevector calculation."""

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self):
        """Returns a 3-component vector."""
        var_k = self.variant(incident_wavevector)

        k_in = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )
        chex.assert_shape(k_in, (3,))
        chex.assert_tree_all_finite(k_in)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_magnitude_equals_2pi_over_lambda(self):
        """Wavevector magnitude equals 2*pi/lambda."""
        var_k = self.variant(incident_wavevector)

        lam = 0.0859
        k_in = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
        )

        k_mag = jnp.linalg.norm(k_in)
        expected_mag = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_negative_z_component(self):
        """k_z is negative (beam enters from above the surface)."""
        var_k = self.variant(incident_wavevector)

        k_in = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )

        assert float(k_in[2]) < 0, "k_z should be negative"

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_grazing_angle_controls_z(self):
        """Steeper grazing angle gives larger |k_z|."""
        var_k = self.variant(incident_wavevector)

        lam = 0.0859
        k_shallow = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(1.0),
        )
        k_steep = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(5.0),
        )

        assert float(jnp.abs(k_steep[2])) > float(jnp.abs(k_shallow[2]))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_phi_zero_no_y_component(self):
        """At phi=0, k_y should be zero (beam along x)."""
        var_k = self.variant(incident_wavevector)

        k_in = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(0.0),
        )

        chex.assert_trees_all_close(k_in[1], 0.0, atol=1e-10)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_phi_90_no_x_component(self):
        """At phi=90, k_x should be zero (beam along y)."""
        var_k = self.variant(incident_wavevector)

        k_in = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(90.0),
        )

        chex.assert_trees_all_close(k_in[0], 0.0, atol=1e-10)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
        ("phi_180", 180.0),
    )
    def test_magnitude_invariant_under_phi(self, phi):
        """Wavevector magnitude is independent of azimuthal angle."""
        var_k = self.variant(incident_wavevector)

        lam = 0.0859
        k_in = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(phi),
        )

        k_mag = jnp.linalg.norm(k_in)
        expected_mag = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)


class TestInteractionConstant(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic interaction constant."""

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_known_values(self):
        """Interaction constant matches reference values."""
        var_sigma = self.variant(interaction_constant)

        voltages_kv = jnp.array([10.0, 20.0, 30.0])
        wavelengths = jax.vmap(wavelength_ang)(voltages_kv)
        sigmas = jax.vmap(var_sigma)(voltages_kv, wavelengths)

        expected = jnp.array([0.0025738, 0.0018283, 0.0014993])

        chex.assert_trees_all_close(sigmas, expected, rtol=5e-4, atol=5e-6)
        chex.assert_tree_all_finite(sigmas)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_positive_output(self):
        """Interaction constant is always positive."""
        var_sigma = self.variant(interaction_constant)

        lam = wavelength_ang(jnp.float64(20.0))
        sigma = var_sigma(jnp.float64(20.0), lam)

        assert float(sigma) > 0

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_decreases_with_voltage(self):
        """Higher voltage gives smaller interaction constant."""
        var_sigma = self.variant(interaction_constant)

        lam_10 = wavelength_ang(jnp.float64(10.0))
        lam_30 = wavelength_ang(jnp.float64(30.0))

        sigma_10 = var_sigma(jnp.float64(10.0), lam_10)
        sigma_30 = var_sigma(jnp.float64(30.0), lam_30)

        assert float(sigma_10) > float(sigma_30)


class TestSimulUtilsGradientCorrectness(chex.TestCase, parameterized.TestCase):
    """Verify analytical gradients match finite differences."""

    def test_wavelength_grad_correct(self):
        """Relativistic wavelength grad matches finite diff to 2nd order."""

        def f(voltage):
            return wavelength_ang(voltage)

        check_grads(jax_safe(f), (jnp.float64(20.0),), order=2, atol=1e-4)


class TestSimulUtilsVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential for utility functions."""

    def test_wavelength_vmap_consistent(self):
        """Batched wavelength matches sequential evaluation."""
        voltages = jnp.array([10.0, 20.0, 30.0, 50.0])
        batched = jax.vmap(wavelength_ang)(voltages)
        sequential = jnp.stack([wavelength_ang(v) for v in voltages])
        chex.assert_trees_all_close(batched, sequential, atol=1e-8)
