import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

jax.config.update("jax_enable_x64", True)

# Import your functions here
from rheedium.unitcell import wavelength_ang

if __name__ == "__main__":
    pytest.main([__file__])


class test_wavelength_ang(chex.TestCase):
    @chex.all_variants
    @parameterized.parameters(
        {"test_kV": 200, "expected_wavelength": 0.02508},
        {"test_kV": 1000, "expected_wavelength": 0.008719185412913083},
        {"test_kV": 0.001, "expected_wavelength": 12.2642524552},
        {"test_kV": 300, "expected_wavelength": 0.0196874863882},
    )
    def test_voltage_values(self, test_kV, expected_wavelength):
        var_wavelength_ang = self.variant(wavelength_ang)
        # voltage_kV = 200.0
        # expected_wavelength = 0.02508  # Expected value based on known physics
        result = var_wavelength_ang(test_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-6
        ), f"Expected {expected_wavelength}, but got {result}"

    # Check for precision and rounding errors
    @chex.all_variants
    def test_precision_and_rounding_errors(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = 150.0
        expected_wavelength = 0.02957  # Expected value based on known physics
        result = var_wavelength_ang(voltage_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-5
        ), f"Expected {expected_wavelength}, but got {result}"

    # Ensure function returns a Float Array
    @chex.all_variants
    def test_returns_float(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = 200.0
        result = var_wavelength_ang(voltage_kV)
        assert isinstance(
            result, Float[Array, "*"]
        ), "Expected the function to return a float"

    # Test whether array inputs work
    @chex.all_variants
    def test_array_input(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltages = jnp.array([100, 200, 300, 400], dtype=jnp.float64)
        results = var_wavelength_ang(voltages)
        expected = jnp.array([0.03701436, 0.02507934, 0.01968749, 0.01643943])
        assert jnp.allclose(results, expected, atol=1e-5)
