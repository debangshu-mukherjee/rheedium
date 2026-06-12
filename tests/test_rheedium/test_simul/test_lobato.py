"""Test suite for Lobato-van Dyck scattering factor parameterization.

Validates the Lobato-van Dyck (2014) form factors and projected potentials
against known physical constraints and cross-checks with the existing
Kirkland parameterization.
"""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout.xyz import lobato_potentials
from rheedium.simul.form_factors import (
    kirkland_form_factor,
    kirkland_projected_potential,
    load_lobato_parameters,
    lobato_form_factor,
    lobato_projected_potential,
    projected_potential,
)
from rheedium.types.custom_types import scalar_float


class TestLoadLobatoPotentials(chex.TestCase):
    """Test CSV loading and parameter extraction."""

    def test_csv_shape(self) -> None:
        """Lobato CSV loads as (103, 10) array."""
        params: Float[Array, "103 10"] = lobato_potentials()
        chex.assert_shape(params, (103, 10))

    def test_csv_finite(self) -> None:
        """All parameters are finite."""
        params: Float[Array, "103 10"] = lobato_potentials()
        chex.assert_tree_all_finite(params)

    def test_csv_positive(self) -> None:
        """All a_i and b_i coefficients are positive."""
        params: Float[Array, "103 10"] = lobato_potentials()
        assert jnp.all(params > 0)

    def test_load_lobato_parameters_hydrogen(self) -> None:
        """Hydrogen parameters match CSV row 1."""
        a: Float[Array, "5"]
        b: Float[Array, "5"]
        a, b = load_lobato_parameters(1)
        chex.assert_shape(a, (5,))
        chex.assert_shape(b, (5,))
        chex.assert_tree_all_finite(a)
        chex.assert_tree_all_finite(b)
        chex.assert_trees_all_close(a[0], jnp.array(0.0349), rtol=1e-3)

    def test_load_lobato_parameters_silicon(self) -> None:
        """Silicon (Z=14) parameters match CSV."""
        a: Float[Array, "5"]
        _b: Float[Array, "5"]
        a, _b = load_lobato_parameters(14)
        chex.assert_trees_all_close(a[0], jnp.array(0.2519), rtol=1e-3)


class TestLobatoFormFactor(chex.TestCase, parameterized.TestCase):
    """Test lobato_form_factor physical constraints."""

    @parameterized.named_parameters(
        ("H", 1),
        ("C", 6),
        ("Si", 14),
        ("Cu", 29),
        ("Au", 79),
    )
    def test_zero_angle_limit(self, z: int) -> None:
        """f_e(0) = sum_i 2*a_i (Mott-Bethe zero-angle limit)."""
        a: Float[Array, "5"]
        _b: Float[Array, "5"]
        a, _b = load_lobato_parameters(z)
        expected_f0: scalar_float = jnp.sum(2.0 * a)
        computed_f0: scalar_float = lobato_form_factor(z, jnp.array([0.0]))[0]
        chex.assert_trees_all_close(computed_f0, expected_f0, rtol=1e-6)

    @parameterized.named_parameters(
        ("H", 1),
        ("C", 6),
        ("Si", 14),
        ("Cu", 29),
        ("Au", 79),
    )
    def test_nonnegative(self, z: int) -> None:
        """f_e(q) >= 0 for all q >= 0."""
        q: Float[Array, "300"] = jnp.linspace(0.0, 30.0, 300)
        fe: Float[Array, "300"] = lobato_form_factor(z, q)
        chex.assert_tree_all_finite(fe)
        assert jnp.all(fe >= 0)

    def test_monotone_decreasing(self) -> None:
        """f_e(q) is monotonically decreasing for q > 0 (Si)."""
        q: Float[Array, "300"] = jnp.linspace(0.1, 30.0, 300)
        fe: Float[Array, "300"] = lobato_form_factor(14, q)
        diffs: Float[Array, "299"] = jnp.diff(fe)
        assert jnp.all(diffs <= 0)

    def test_high_q_asymptotic(self) -> None:
        """f_e decays as q^{-2} at large q (Bethe limit)."""
        q_high: Float[Array, "2"] = jnp.array([40.0, 80.0])
        fe: Float[Array, "2"] = lobato_form_factor(14, q_high)
        ratio: scalar_float = fe[1] / fe[0]
        expected_ratio: float = (40.0 / 80.0) ** 2
        chex.assert_trees_all_close(ratio, expected_ratio, rtol=0.05)

    def test_batch_shapes(self) -> None:
        """Handles arbitrary batch dimensions."""
        q_1d: Float[Array, "50"] = jnp.linspace(0.0, 10.0, 50)
        q_2d: Float[Array, "4 8"] = jnp.ones((4, 8)) * 2.0
        chex.assert_shape(lobato_form_factor(14, q_1d), (50,))
        chex.assert_shape(lobato_form_factor(14, q_2d), (4, 8))

    @parameterized.named_parameters(
        ("H", 1),
        ("Si", 14),
        ("Au", 79),
    )
    def test_vs_kirkland_same_order_of_magnitude(self, z: int) -> None:
        """Lobato and Kirkland f_e(0) within an order of magnitude.

        The Kirkland implementation in rheedium uses a simplified
        Gaussian-only parameterization with different parameter
        conventions, so close agreement is not expected. This test
        verifies both are in the same ballpark.
        """
        q_zero: Float[Array, "1"] = jnp.array([0.0])
        fe_lobato: scalar_float = lobato_form_factor(z, q_zero)[0]
        fe_kirkland: scalar_float = kirkland_form_factor(z, q_zero)[0]
        ratio: scalar_float = fe_lobato / fe_kirkland
        assert 0.1 < ratio < 10.0

    def test_gradient_finite(self) -> None:
        """jax.grad through lobato_form_factor is finite."""

        def loss(q: Float[Array, "N"]) -> scalar_float:
            return jnp.sum(lobato_form_factor(14, q))

        grad_fn: Callable[[Float[Array, "3"]], Float[Array, "3"]] = jax.grad(
            loss
        )
        grad_val: Float[Array, "3"] = grad_fn(jnp.array([1.0, 2.0, 3.0]))
        chex.assert_tree_all_finite(grad_val)

    def test_heavier_element_larger_f0(self) -> None:
        """Heavier elements have larger f_e(0)."""
        f0_h: scalar_float = lobato_form_factor(1, jnp.array([0.0]))[0]
        f0_si: scalar_float = lobato_form_factor(14, jnp.array([0.0]))[0]
        f0_au: scalar_float = lobato_form_factor(79, jnp.array([0.0]))[0]
        assert f0_au > f0_si > f0_h


class TestLobatoProjectedPotential(chex.TestCase, parameterized.TestCase):
    """Test lobato_projected_potential physical constraints."""

    @parameterized.named_parameters(
        ("H", 1),
        ("C", 6),
        ("Si", 14),
        ("Cu", 29),
        ("Au", 79),
    )
    def test_positive_for_r_gt_zero(self, z: int) -> None:
        """V_z(r) > 0 for r > 0 (attractive potential)."""
        r: Float[Array, "100"] = jnp.linspace(0.01, 5.0, 100)
        vz: Float[Array, "100"] = lobato_projected_potential(z, r)
        chex.assert_tree_all_finite(vz)
        assert jnp.all(vz > 0)

    def test_monotone_decreasing(self) -> None:
        """V_z(r) is monotonically decreasing with r (Si)."""
        r: Float[Array, "200"] = jnp.linspace(0.02, 5.0, 200)
        vz: Float[Array, "200"] = lobato_projected_potential(14, r)
        diffs: Float[Array, "199"] = jnp.diff(vz)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self) -> None:
        """Handles arbitrary batch dimensions."""
        r_1d: Float[Array, "50"] = jnp.linspace(0.01, 3.0, 50)
        r_2d: Float[Array, "4 8"] = jnp.ones((4, 8)) * 1.0
        chex.assert_shape(lobato_projected_potential(14, r_1d), (50,))
        chex.assert_shape(lobato_projected_potential(14, r_2d), (4, 8))

    def test_gradient_finite(self) -> None:
        """jax.grad through lobato_projected_potential is finite."""

        def loss(r: Float[Array, "N"]) -> scalar_float:
            return jnp.sum(lobato_projected_potential(14, r))

        grad_fn: Callable[[Float[Array, "3"]], Float[Array, "3"]] = jax.grad(
            loss
        )
        grad_val: Float[Array, "3"] = grad_fn(jnp.array([0.5, 1.0, 2.0]))
        chex.assert_tree_all_finite(grad_val)

    def test_silicon_potential_at_1ang(self) -> None:
        """Si V_z(1 Å) is in the expected physical range.

        For Si at r = 1 Å, the projected potential should be on the
        order of tens of V·Å (typical for Z ~ 14 atoms).
        """
        vz: scalar_float = lobato_projected_potential(14, jnp.array([1.0]))[0]
        assert 1.0 < vz < 500.0

    def test_heavier_element_larger_potential(self) -> None:
        """Heavier elements produce larger projected potential."""
        r: Float[Array, "1"] = jnp.array([0.5])
        vz_h: scalar_float = lobato_projected_potential(1, r)[0]
        vz_si: scalar_float = lobato_projected_potential(14, r)[0]
        vz_au: scalar_float = lobato_projected_potential(79, r)[0]
        assert vz_au > vz_si > vz_h

    def test_jit_matches_eager(self) -> None:
        """JIT-compiled output matches eager evaluation."""
        r: Float[Array, "20"] = jnp.linspace(0.1, 3.0, 20)
        eager: Float[Array, "20"] = lobato_projected_potential(14, r)
        jitted: Float[Array, "20"] = jax.jit(
            lobato_projected_potential, static_argnums=0
        )(14, r)
        chex.assert_trees_all_close(eager, jitted, rtol=1e-10)


class TestParameterizationSwitch(chex.TestCase):
    """Test that Lobato and Kirkland produce same-shape output."""

    def test_form_factor_same_shape(self) -> None:
        """Both parameterizations produce identical output shapes."""
        q: Float[Array, "50"] = jnp.linspace(0.0, 10.0, 50)
        fe_l: Float[Array, "50"] = lobato_form_factor(14, q)
        fe_k: Float[Array, "50"] = kirkland_form_factor(14, q)
        chex.assert_shape(fe_l, fe_k.shape)

    def test_projected_potential_same_shape(self) -> None:
        """Both parameterizations produce identical output shapes."""
        r: Float[Array, "50"] = jnp.linspace(0.01, 3.0, 50)
        vz_l: Float[Array, "50"] = lobato_projected_potential(14, r)
        vz_k: Float[Array, "50"] = kirkland_projected_potential(14, r)
        chex.assert_shape(vz_l, vz_k.shape)


class TestProjectedPotentialDispatch(chex.TestCase):
    """Test the projected_potential dispatcher."""

    def test_lobato_default(self) -> None:
        """Default parameterization is Lobato."""
        r: Float[Array, "20"] = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch: Float[Array, "20"] = projected_potential(14, r)
        vz_direct: Float[Array, "20"] = lobato_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_lobato_explicit(self) -> None:
        """Explicit 'lobato' matches direct call."""
        r: Float[Array, "20"] = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch: Float[Array, "20"] = projected_potential(14, r, "lobato")
        vz_direct: Float[Array, "20"] = lobato_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_kirkland_explicit(self) -> None:
        """Explicit 'kirkland' matches direct call."""
        r: Float[Array, "20"] = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch: Float[Array, "20"] = projected_potential(
            14, r, "kirkland"
        )
        vz_direct: Float[Array, "20"] = kirkland_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_lobato_differs_from_kirkland(self) -> None:
        """Lobato and Kirkland branches produce different results."""
        r: Float[Array, "20"] = jnp.linspace(0.1, 3.0, 20)
        vz_lobato: Float[Array, "20"] = projected_potential(14, r, "lobato")
        vz_kirkland: Float[Array, "20"] = projected_potential(
            14, r, "kirkland"
        )
        assert not jnp.allclose(vz_lobato, vz_kirkland)

    def test_gradient_through_dispatch(self) -> None:
        """jax.grad works through the lax.cond dispatch."""

        def loss(r: Float[Array, "N"]) -> scalar_float:
            return jnp.sum(projected_potential(14, r, "lobato"))

        grad_fn: Callable[[Float[Array, "3"]], Float[Array, "3"]] = jax.grad(
            loss
        )
        grad_val: Float[Array, "3"] = grad_fn(jnp.array([0.5, 1.0, 2.0]))
        chex.assert_tree_all_finite(grad_val)


if __name__ == "__main__":
    pytest.main([__file__])
