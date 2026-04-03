"""Test suite for RHEED simulation utility functions.

Tests for wavelength_ang, incident_wavevector, and
interaction_constant — the shared physical computations used
across multiple simulation modules.
"""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul import (
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.types import scalar_float


class TestWavelengthAng(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic electron wavelength calculation."""

    @chex.all_variants(without_device=False)
    def test_known_voltages(self) -> None:
        """Relativistic wavelength matches reference values."""
        var_wavelength = self.variant(wavelength_ang)

        voltages_kv: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = jax.vmap(var_wavelength)(voltages_kv)

        expected: Float[Array, "3"] = jnp.array([0.1226, 0.0859, 0.0698])

        chex.assert_trees_all_close(wavelengths, expected, rtol=5e-3)
        chex.assert_tree_all_finite(wavelengths)

    @chex.all_variants(without_device=False)
    def test_higher_voltage_shorter_wavelength(self) -> None:
        """Higher voltage produces shorter wavelength."""
        var_wavelength = self.variant(wavelength_ang)

        lam_10: Float[Array, ""] = var_wavelength(jnp.float64(10.0))
        lam_30: Float[Array, ""] = var_wavelength(jnp.float64(30.0))

        assert float(lam_30) < float(lam_10)

    @chex.all_variants(without_device=False)
    def test_positive_output(self) -> None:
        """Wavelength is always positive."""
        var_wavelength = self.variant(wavelength_ang)

        voltages: Float[Array, "5"] = jnp.array([5.0, 10.0, 20.0, 50.0, 100.0])
        wavelengths: Float[Array, "5"] = jax.vmap(var_wavelength)(voltages)

        chex.assert_trees_all_equal(jnp.all(wavelengths > 0), True)

    def test_scalar_input(self) -> None:
        """Accepts scalar float input."""
        lam: Float[Array, ""] = wavelength_ang(20.0)
        chex.assert_tree_all_finite(lam)

    def test_array_broadcast(self) -> None:
        """Handles batched array input."""
        voltages: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = wavelength_ang(voltages)
        chex.assert_shape(wavelengths, (3,))
        chex.assert_tree_all_finite(wavelengths)


class TestIncidentWavevector(chex.TestCase, parameterized.TestCase):
    """Tests for incident wavevector calculation."""

    @chex.all_variants(without_device=False)
    def test_output_shape(self) -> None:
        """Returns a 3-component vector."""
        var_k = self.variant(incident_wavevector)

        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )
        chex.assert_shape(k_in, (3,))
        chex.assert_tree_all_finite(k_in)

    @chex.all_variants(without_device=False)
    def test_magnitude_equals_2pi_over_lambda(self) -> None:
        """Wavevector magnitude equals 2*pi/lambda."""
        var_k = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
        )

        k_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
        expected_mag: float = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)

    @chex.all_variants(without_device=False)
    def test_negative_z_component(self) -> None:
        """k_z is negative (beam enters from above the surface)."""
        var_k = self.variant(incident_wavevector)

        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )

        assert float(k_in[2]) < 0, "k_z should be negative"

    @chex.all_variants(without_device=False)
    def test_grazing_angle_controls_z(self) -> None:
        """Steeper grazing angle gives larger |k_z|."""
        var_k = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_shallow: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(1.0),
        )
        k_steep: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(5.0),
        )

        assert float(jnp.abs(k_steep[2])) > float(jnp.abs(k_shallow[2]))

    @chex.all_variants(without_device=False)
    def test_phi_zero_no_y_component(self) -> None:
        """At phi=0, k_y should be zero (beam along x)."""
        var_k = self.variant(incident_wavevector)

        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(0.0),
        )

        chex.assert_trees_all_close(k_in[1], 0.0, atol=1e-10)

    @chex.all_variants(without_device=False)
    def test_phi_90_no_x_component(self) -> None:
        """At phi=90, k_x should be zero (beam along y)."""
        var_k = self.variant(incident_wavevector)

        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(90.0),
        )

        chex.assert_trees_all_close(k_in[0], 0.0, atol=1e-10)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
        ("phi_180", 180.0),
    )
    def test_magnitude_invariant_under_phi(self, phi: float) -> None:
        """Wavevector magnitude is independent of azimuthal angle."""
        var_k = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_in: Float[Array, "3"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(phi),
        )

        k_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
        expected_mag: float = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)


class TestInteractionConstant(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic interaction constant."""

    @chex.all_variants(without_device=False)
    def test_known_values(self) -> None:
        """Interaction constant matches reference values."""
        var_sigma = self.variant(interaction_constant)

        voltages_kv: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = jax.vmap(wavelength_ang)(voltages_kv)
        sigmas: Float[Array, "3"] = jax.vmap(var_sigma)(
            voltages_kv, wavelengths
        )

        expected: Float[Array, "3"] = jnp.array(
            [0.0025738, 0.0018283, 0.0014993]
        )

        chex.assert_trees_all_close(sigmas, expected, rtol=5e-4, atol=5e-6)
        chex.assert_tree_all_finite(sigmas)

    @chex.all_variants(without_device=False)
    def test_positive_output(self) -> None:
        """Interaction constant is always positive."""
        var_sigma = self.variant(interaction_constant)

        lam: Float[Array, ""] = wavelength_ang(jnp.float64(20.0))
        sigma: Float[Array, ""] = var_sigma(jnp.float64(20.0), lam)

        assert float(sigma) > 0

    @chex.all_variants(without_device=False)
    def test_decreases_with_voltage(self) -> None:
        """Higher voltage gives smaller interaction constant."""
        var_sigma = self.variant(interaction_constant)

        lam_10: Float[Array, ""] = wavelength_ang(jnp.float64(10.0))
        lam_30: Float[Array, ""] = wavelength_ang(jnp.float64(30.0))

        sigma_10: Float[Array, ""] = var_sigma(jnp.float64(10.0), lam_10)
        sigma_30: Float[Array, ""] = var_sigma(jnp.float64(30.0), lam_30)

        assert float(sigma_10) > float(sigma_30)
