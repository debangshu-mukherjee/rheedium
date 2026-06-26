"""Test suite for RHEED simulation utility functions.

Tests for wavelength_ang, incident_wavevector, and
interaction_constant — the shared physical computations used
across multiple simulation modules.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.test_util import check_grads
from jaxtyping import Array, Float

from rheedium.tools.simul_utils import (
    incidence_angles_to_radians,
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from rheedium.tools.wrappers import jax_safe
from rheedium.types.custom_types import scalar_float


class TestWavelengthAng(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic electron wavelength calculation.

    :see: :func:`~rheedium.tools.wavelength_ang`
    """

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_known_voltages(self) -> None:
        """Relativistic wavelength matches reference values."""
        var_wavelength: Callable[..., Any] = self.variant(wavelength_ang)

        voltages_kv: Float[Array, "..."] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "voltages"] = jax.vmap(var_wavelength)(
            voltages_kv
        )

        expected: Float[Array, "..."] = jnp.array([0.1226, 0.0859, 0.0698])

        chex.assert_trees_all_close(wavelengths, expected, rtol=5e-3)
        chex.assert_tree_all_finite(wavelengths)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_higher_voltage_shorter_wavelength(self) -> None:
        """Higher voltage produces shorter wavelength."""
        var_wavelength: Callable[..., Any] = self.variant(wavelength_ang)

        lam_10: scalar_float = var_wavelength(jnp.float64(10.0))
        lam_30: scalar_float = var_wavelength(jnp.float64(30.0))

        assert float(lam_30) < float(lam_10)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_positive_output(self) -> None:
        """Wavelength is always positive."""
        var_wavelength: Callable[..., Any] = self.variant(wavelength_ang)

        voltages: Float[Array, "..."] = jnp.array(
            [5.0, 10.0, 20.0, 50.0, 100.0]
        )
        wavelengths: Float[Array, "voltages"] = jax.vmap(var_wavelength)(
            voltages
        )

        chex.assert_trees_all_equal(jnp.all(wavelengths > 0), True)

    def test_scalar_input(self) -> None:
        """Accepts scalar float input."""
        lam: scalar_float = wavelength_ang(20.0)
        chex.assert_tree_all_finite(lam)

    def test_array_broadcast(self) -> None:
        """Handles batched array input."""
        voltages: Float[Array, "..."] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "voltages"] = wavelength_ang(voltages)
        chex.assert_shape(wavelengths, (3,))
        chex.assert_tree_all_finite(wavelengths)


class TestIncidenceAnglesToRadians(chex.TestCase, parameterized.TestCase):
    """Tests for public-to-internal incidence angle conversion.

    :see: :func:`~rheedium.tools.incidence_angles_to_radians`
    """

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_converts_public_degrees_to_internal_radians(self) -> None:
        """Convert theta/phi degree inputs to polar/azimuth radians."""
        var_angles: Callable[..., Any] = self.variant(
            incidence_angles_to_radians
        )

        polar_angle_rad: Float[Array, ""]
        azimuth_angle_rad: Float[Array, ""]
        polar_angle_rad, azimuth_angle_rad = var_angles(
            jnp.float64(2.0),
            jnp.float64(45.0),
        )

        chex.assert_trees_all_close(
            polar_angle_rad,
            jnp.deg2rad(jnp.float64(2.0)),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            azimuth_angle_rad,
            jnp.deg2rad(jnp.float64(45.0)),
            atol=1e-12,
        )

    def test_default_phi_converts_to_zero_azimuth(self) -> None:
        """Default phi produces a zero internal azimuth angle."""
        polar_angle_rad: Float[Array, ""]
        azimuth_angle_rad: Float[Array, ""]
        polar_angle_rad, azimuth_angle_rad = incidence_angles_to_radians(
            jnp.float64(3.0)
        )

        chex.assert_trees_all_close(
            polar_angle_rad,
            jnp.deg2rad(jnp.float64(3.0)),
            atol=1e-12,
        )
        chex.assert_trees_all_close(azimuth_angle_rad, 0.0, atol=1e-12)


class TestIncidentWavevector(chex.TestCase, parameterized.TestCase):
    """Tests for incident wavevector calculation.

    :see: :func:`~rheedium.tools.incident_wavevector`
    """

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self) -> None:
        """Returns a 3-component vector."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        k_in: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )
        chex.assert_shape(k_in, (3,))
        chex.assert_tree_all_finite(k_in)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_magnitude_equals_2pi_over_lambda(self) -> None:
        """Wavevector magnitude equals 2*pi/lambda."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_in: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
        )

        k_mag: scalar_float = jnp.linalg.norm(k_in)
        expected_mag: scalar_float = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_negative_z_component(self) -> None:
        """k_z is negative (beam enters from above the surface)."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        k_in: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
        )

        assert float(k_in[2]) < 0, "k_z should be negative"

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_grazing_angle_controls_z(self) -> None:
        r"""Steeper grazing angle gives larger \|k_z\|."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_shallow: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(1.0),
        )
        k_steep: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(5.0),
        )

        assert float(jnp.abs(k_steep[2])) > float(jnp.abs(k_shallow[2]))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_phi_zero_no_y_component(self) -> None:
        """At phi=0, k_y should be zero (beam along x)."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        k_in: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(0.0859),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(0.0),
        )

        chex.assert_trees_all_close(k_in[1], 0.0, atol=1e-10)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_phi_90_no_x_component(self) -> None:
        """At phi=90, k_x should be zero (beam along y)."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        k_in: Float[Array, "three"] = var_k(
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
    def test_magnitude_invariant_under_phi(self, phi: float) -> None:
        """Wavevector magnitude is independent of azimuthal angle."""
        var_k: Callable[..., Any] = self.variant(incident_wavevector)

        lam: float = 0.0859
        k_in: Float[Array, "three"] = var_k(
            lam_ang=jnp.float64(lam),
            theta_deg=jnp.float64(2.0),
            phi_deg=jnp.float64(phi),
        )

        k_mag: scalar_float = jnp.linalg.norm(k_in)
        expected_mag: scalar_float = 2.0 * jnp.pi / lam

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)


class TestInteractionConstant(chex.TestCase, parameterized.TestCase):
    """Tests for relativistic interaction constant.

    :see: :func:`~rheedium.tools.interaction_constant`
    """

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_known_values(self) -> None:
        """Interaction constant matches reference values."""
        var_sigma: Callable[..., Any] = self.variant(interaction_constant)

        voltages_kv: Float[Array, "..."] = jnp.array(
            [10.0, 20.0, 30.0, 100.0, 300.0]
        )
        wavelengths: Float[Array, "voltages"] = jax.vmap(wavelength_ang)(
            voltages_kv
        )
        sigmas: Float[Array, "voltages"] = jax.vmap(var_sigma)(
            voltages_kv, wavelengths
        )

        expected: Float[Array, "..."] = jnp.array(
            [
                0.0025990366192425313,
                0.0018640613383894631,
                0.0015432749751989857,
                0.000924398840308211,
                0.0006526182578014016,
            ]
        )

        chex.assert_trees_all_close(sigmas, expected, rtol=5e-7, atol=5e-8)
        chex.assert_tree_all_finite(sigmas)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_positive_output(self) -> None:
        """Interaction constant is always positive."""
        var_sigma: Callable[..., Any] = self.variant(interaction_constant)

        lam: scalar_float = wavelength_ang(jnp.float64(20.0))
        sigma: scalar_float = var_sigma(jnp.float64(20.0), lam)

        assert float(sigma) > 0

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_decreases_with_voltage(self) -> None:
        """Higher voltage gives smaller interaction constant."""
        var_sigma: Callable[..., Any] = self.variant(interaction_constant)

        lam_10: scalar_float = wavelength_ang(jnp.float64(10.0))
        lam_30: scalar_float = wavelength_ang(jnp.float64(30.0))

        sigma_10: scalar_float = var_sigma(jnp.float64(10.0), lam_10)
        sigma_30: scalar_float = var_sigma(jnp.float64(30.0), lam_30)

        assert float(sigma_10) > float(sigma_30)


class TestSimulUtilsGradientCorrectness(chex.TestCase, parameterized.TestCase):
    """Verify analytical gradients match finite differences."""

    def test_wavelength_grad_correct(self) -> None:
        """Relativistic wavelength grad matches finite diff to 2nd order."""

        def f(voltage: scalar_float) -> scalar_float:
            return wavelength_ang(voltage)

        check_grads(jax_safe(f), (jnp.float64(20.0),), order=2, atol=1e-4)


class TestSimulUtilsVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential for utility functions."""

    def test_wavelength_vmap_consistent(self) -> None:
        """Batched wavelength matches sequential evaluation."""
        voltages: Float[Array, "..."] = jnp.array([10.0, 20.0, 30.0, 50.0])
        batched: Float[Array, "voltages"] = jax.vmap(wavelength_ang)(voltages)
        sequential: Float[Array, "..."] = jnp.stack(
            [wavelength_ang(v) for v in voltages]
        )
        chex.assert_trees_all_close(batched, sequential, atol=1e-8)
