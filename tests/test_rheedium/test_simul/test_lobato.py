"""Test suite for Lobato-van Dyck scattering factor parameterization.

Validates the Lobato-van Dyck (2014) form factors and projected potentials
against known physical constraints and cross-checks with the existing
Kirkland parameterization.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout.xyz import lobato_potentials
from rheedium.simul.form_factors import (
    kirkland_form_factor,
    kirkland_projected_potential,
    load_lobato_parameters,
    lobato_form_factor,
    lobato_projected_potential,
    projected_potential,
)


class TestLoadLobatoPotentials(chex.TestCase):
    """Test CSV loading and parameter extraction."""

    def test_csv_shape(self):
        """Lobato CSV loads as (103, 10) array."""
        params = lobato_potentials()
        chex.assert_shape(params, (103, 10))

    def test_csv_finite(self):
        """All parameters are finite."""
        params = lobato_potentials()
        chex.assert_tree_all_finite(params)

    def test_csv_positive(self):
        """All a_i and b_i coefficients are positive."""
        params = lobato_potentials()
        assert jnp.all(params > 0)

    def test_load_lobato_parameters_hydrogen(self):
        """Hydrogen parameters match CSV row 1."""
        a, b = load_lobato_parameters(1)
        chex.assert_shape(a, (5,))
        chex.assert_shape(b, (5,))
        chex.assert_tree_all_finite(a)
        chex.assert_tree_all_finite(b)
        chex.assert_trees_all_close(a[0], jnp.array(0.0349), rtol=1e-3)

    def test_load_lobato_parameters_silicon(self):
        """Silicon (Z=14) parameters match CSV."""
        a, b = load_lobato_parameters(14)
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
    def test_zero_angle_limit(self, z):
        """f_e(0) = sum_i 2*a_i (Mott-Bethe zero-angle limit)."""
        a, _ = load_lobato_parameters(z)
        expected_f0 = jnp.sum(2.0 * a)
        computed_f0 = lobato_form_factor(z, jnp.array([0.0]))[0]
        chex.assert_trees_all_close(computed_f0, expected_f0, rtol=1e-6)

    @parameterized.named_parameters(
        ("H", 1),
        ("C", 6),
        ("Si", 14),
        ("Cu", 29),
        ("Au", 79),
    )
    def test_nonnegative(self, z):
        """f_e(q) >= 0 for all q >= 0."""
        q = jnp.linspace(0.0, 30.0, 300)
        fe = lobato_form_factor(z, q)
        chex.assert_tree_all_finite(fe)
        assert jnp.all(fe >= 0)

    def test_monotone_decreasing(self):
        """f_e(q) is monotonically decreasing for q > 0 (Si)."""
        q = jnp.linspace(0.1, 30.0, 300)
        fe = lobato_form_factor(14, q)
        diffs = jnp.diff(fe)
        assert jnp.all(diffs <= 0)

    def test_high_q_asymptotic(self):
        """f_e decays as q^{-2} at large q (Bethe limit)."""
        q_high = jnp.array([40.0, 80.0])
        fe = lobato_form_factor(14, q_high)
        ratio = fe[1] / fe[0]
        expected_ratio = (40.0 / 80.0) ** 2
        chex.assert_trees_all_close(ratio, expected_ratio, rtol=0.05)

    def test_batch_shapes(self):
        """Handles arbitrary batch dimensions."""
        q_1d = jnp.linspace(0.0, 10.0, 50)
        q_2d = jnp.ones((4, 8)) * 2.0
        chex.assert_shape(lobato_form_factor(14, q_1d), (50,))
        chex.assert_shape(lobato_form_factor(14, q_2d), (4, 8))

    @parameterized.named_parameters(
        ("H", 1),
        ("Si", 14),
        ("Au", 79),
    )
    def test_vs_kirkland_same_order_of_magnitude(self, z):
        """Lobato and Kirkland f_e(0) within an order of magnitude.

        The Kirkland implementation in rheedium uses a simplified
        Gaussian-only parameterization with different parameter
        conventions, so close agreement is not expected. This test
        verifies both are in the same ballpark.
        """
        q_zero = jnp.array([0.0])
        fe_lobato = lobato_form_factor(z, q_zero)[0]
        fe_kirkland = kirkland_form_factor(z, q_zero)[0]
        ratio = fe_lobato / fe_kirkland
        assert 0.1 < ratio < 10.0

    def test_gradient_finite(self):
        """jax.grad through lobato_form_factor is finite."""

        def loss(q):
            return jnp.sum(lobato_form_factor(14, q))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(jnp.array([1.0, 2.0, 3.0]))
        chex.assert_tree_all_finite(grad_val)

    def test_heavier_element_larger_f0(self):
        """Heavier elements have larger f_e(0)."""
        f0_h = lobato_form_factor(1, jnp.array([0.0]))[0]
        f0_si = lobato_form_factor(14, jnp.array([0.0]))[0]
        f0_au = lobato_form_factor(79, jnp.array([0.0]))[0]
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
    def test_positive_for_r_gt_zero(self, z):
        """V_z(r) > 0 for r > 0 (attractive potential)."""
        r = jnp.linspace(0.01, 5.0, 100)
        vz = lobato_projected_potential(z, r)
        chex.assert_tree_all_finite(vz)
        assert jnp.all(vz > 0)

    def test_monotone_decreasing(self):
        """V_z(r) is monotonically decreasing with r (Si)."""
        r = jnp.linspace(0.02, 5.0, 200)
        vz = lobato_projected_potential(14, r)
        diffs = jnp.diff(vz)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self):
        """Handles arbitrary batch dimensions."""
        r_1d = jnp.linspace(0.01, 3.0, 50)
        r_2d = jnp.ones((4, 8)) * 1.0
        chex.assert_shape(lobato_projected_potential(14, r_1d), (50,))
        chex.assert_shape(lobato_projected_potential(14, r_2d), (4, 8))

    def test_gradient_finite(self):
        """jax.grad through lobato_projected_potential is finite."""

        def loss(r):
            return jnp.sum(lobato_projected_potential(14, r))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(jnp.array([0.5, 1.0, 2.0]))
        chex.assert_tree_all_finite(grad_val)

    def test_silicon_potential_at_1ang(self):
        """Si V_z(1 Å) is in the expected physical range.

        For Si at r = 1 Å, the projected potential should be on the
        order of tens of V·Å (typical for Z ~ 14 atoms).
        """
        vz = lobato_projected_potential(14, jnp.array([1.0]))[0]
        assert 1.0 < vz < 500.0

    def test_heavier_element_larger_potential(self):
        """Heavier elements produce larger projected potential."""
        r = jnp.array([0.5])
        vz_h = lobato_projected_potential(1, r)[0]
        vz_si = lobato_projected_potential(14, r)[0]
        vz_au = lobato_projected_potential(79, r)[0]
        assert vz_au > vz_si > vz_h

    def test_jit_matches_eager(self):
        """JIT-compiled output matches eager evaluation."""
        r = jnp.linspace(0.1, 3.0, 20)
        eager = lobato_projected_potential(14, r)
        jitted = jax.jit(lobato_projected_potential, static_argnums=0)(14, r)
        chex.assert_trees_all_close(eager, jitted, rtol=1e-10)


class TestParameterizationSwitch(chex.TestCase):
    """Test that Lobato and Kirkland produce same-shape output."""

    def test_form_factor_same_shape(self):
        """Both parameterizations produce identical output shapes."""
        q = jnp.linspace(0.0, 10.0, 50)
        fe_l = lobato_form_factor(14, q)
        fe_k = kirkland_form_factor(14, q)
        chex.assert_shape(fe_l, fe_k.shape)

    def test_projected_potential_same_shape(self):
        """Both parameterizations produce identical output shapes."""
        r = jnp.linspace(0.01, 3.0, 50)
        vz_l = lobato_projected_potential(14, r)
        vz_k = kirkland_projected_potential(14, r)
        chex.assert_shape(vz_l, vz_k.shape)


class TestProjectedPotentialDispatch(chex.TestCase):
    """Test the projected_potential dispatcher."""

    def test_lobato_default(self):
        """Default parameterization is Lobato."""
        r = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch = projected_potential(14, r)
        vz_direct = lobato_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_lobato_explicit(self):
        """Explicit 'lobato' matches direct call."""
        r = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch = projected_potential(14, r, "lobato")
        vz_direct = lobato_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_kirkland_explicit(self):
        """Explicit 'kirkland' matches direct call."""
        r = jnp.linspace(0.1, 3.0, 20)
        vz_dispatch = projected_potential(14, r, "kirkland")
        vz_direct = kirkland_projected_potential(14, r)
        chex.assert_trees_all_close(vz_dispatch, vz_direct, rtol=1e-12)

    def test_lobato_differs_from_kirkland(self):
        """Lobato and Kirkland branches produce different results."""
        r = jnp.linspace(0.1, 3.0, 20)
        vz_lobato = projected_potential(14, r, "lobato")
        vz_kirkland = projected_potential(14, r, "kirkland")
        assert not jnp.allclose(vz_lobato, vz_kirkland)

    def test_gradient_through_dispatch(self):
        """jax.grad works through the lax.cond dispatch."""

        def loss(r):
            return jnp.sum(projected_potential(14, r, "lobato"))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(jnp.array([0.5, 1.0, 2.0]))
        chex.assert_tree_all_finite(grad_val)


if __name__ == "__main__":
    pytest.main([__file__])
