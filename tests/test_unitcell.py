import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Import your functions here
from rheedium.unitcell import reciprocal_unitcell, wavelength_ang

if __name__ == "__main__":
    pytest.main([__file__])


class test_wavelength_ang(chex.TestCase):
    @chex.all_variants
    @parameterized.parameters(
        {"test_kV": 200.0, "expected_wavelength": 0.02508},
        {"test_kV": 1000.0, "expected_wavelength": 0.008719185412913083},
        {"test_kV": 0.001, "expected_wavelength": 12.2642524552},
        {"test_kV": 300.0, "expected_wavelength": 0.0196874863882},
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


class test_reciprocal_unitcell(chex.TestCase):
    @chex.all_variants
    @parameterized.parameters(
        {
            "test_cell": jnp.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
            "expected_reciprocal": jnp.array(
                [[1.25663706, 0.0, 0.0], [0.0, 1.25663706, 0.0], [0.0, 0.0, 1.25663706]]
            ),  # 2π/5 on diagonal
        },
        {
            "test_cell": jnp.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]),
            "expected_reciprocal": jnp.array(
                [[2.0944271, 0.0, 0.0], [0.0, 1.5708203, 0.0], [0.0, 0.0, 1.2566371]]
            ),  # 2π/[3,4,5]
        },
    )
    def test_known_cells(self, test_cell, expected_reciprocal):
        var_reciprocal_unitcell = self.variant(reciprocal_unitcell)
        result = var_reciprocal_unitcell(test_cell)
        assert jnp.allclose(
            result, expected_reciprocal, atol=1e-6
        ), f"Expected {expected_reciprocal}, but got {result}"

    # Test for ill-conditioned matrix
    @chex.all_variants
    def test_ill_conditioned_matrix(self):
        var_reciprocal_unitcell = self.variant(reciprocal_unitcell)
        ill_conditioned = jnp.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0 + 1e-8, 1.0], [1.0, 1.0, 1.0 + 1e-8]]
        )
        result = var_reciprocal_unitcell(ill_conditioned)
        assert jnp.allclose(
            result, 0.0, atol=1e-6
        ), "Expected zero values for ill-conditioned matrix"

    # Test crystallographic properties
    @chex.all_variants
    def test_crystallographic_properties(self):
        var_reciprocal_unitcell = self.variant(reciprocal_unitcell)
        unit_cell = jnp.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        reciprocal = var_reciprocal_unitcell(unit_cell)

        # Test orthogonality relations (a* ⊥ b, etc.)
        for i in range(3):
            for j in range(3):
                if i != j:
                    dot_product = jnp.dot(unit_cell[i], reciprocal[j])
                    assert (
                        jnp.abs(dot_product) < 1e-10
                    ), f"Non-zero dot product found: {dot_product}"
                else:
                    dot_product = jnp.dot(unit_cell[i], reciprocal[j])
                    assert (
                        jnp.abs(dot_product - 2 * jnp.pi) < 1e-10
                    ), f"Incorrect self dot product: {dot_product}"

    # Ensure function returns a Float Array with correct shape
    @chex.all_variants
    def test_returns_float_array_3x3(self):
        var_reciprocal_unitcell = self.variant(reciprocal_unitcell)
        unit_cell = jnp.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        result = var_reciprocal_unitcell(unit_cell)
        assert isinstance(
            result, jax.Array
        ), "Expected the function to return a JAX Array"
        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"

    # Test volume conservation
    @chex.all_variants
    def test_volume_conservation(self):
        var_reciprocal_unitcell = self.variant(reciprocal_unitcell)
        unit_cell = jnp.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        reciprocal = var_reciprocal_unitcell(unit_cell)

        direct_volume = jnp.abs(jnp.linalg.det(unit_cell))
        reciprocal_volume = jnp.abs(jnp.linalg.det(reciprocal))
        product = direct_volume * reciprocal_volume
        expected = (2 * jnp.pi) ** 3

        assert jnp.isclose(
            product, expected, atol=1e-6
        ), f"Volume conservation violated. Expected {expected}, got {product}"
