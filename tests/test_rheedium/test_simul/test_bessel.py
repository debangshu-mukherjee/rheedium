"""Test suite for modified Bessel functions K_0 and K_1.

Tests validate the polynomial approximations in simul/bessel.py against
known reference values from scipy.special, and verify JAX transformation
compatibility (JIT, grad, vmap).
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul.bessel import bessel_k0, bessel_k1


class TestBesselK0(chex.TestCase, parameterized.TestCase):
    """Test bessel_k0 against scipy reference values."""

    @parameterized.named_parameters(
        ("x_0p1", 0.1, 2.4270690247),
        ("x_0p5", 0.5, 0.9244190712),
        ("x_1p0", 1.0, 0.4210244382),
        ("x_2p0", 2.0, 0.1138938727),
        ("x_5p0", 5.0, 0.0036910983),
        ("x_10p0", 10.0, 0.0000177801),
    )
    def test_known_values(self, x_val, expected):
        """K_0 matches scipy.special.k0 at representative points."""
        result = bessel_k0(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self):
        """K_0(x) > 0 for all x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k0(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self):
        """K_0(x) is strictly decreasing for x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k0(x)
        diffs = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self):
        """Preserves arbitrary batch dimensions."""
        x_1d = jnp.linspace(0.1, 5.0, 50)
        x_2d = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k0(x_1d), (50,))
        chex.assert_shape(bessel_k0(x_2d), (8, 16))

    def test_jit_matches_eager(self):
        """JIT-compiled output matches eager evaluation."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        eager = bessel_k0(x)
        jitted = jax.jit(bessel_k0)(x)
        chex.assert_trees_all_close(eager, jitted, rtol=1e-12)

    def test_gradient_finite(self):
        """Gradient of K_0 w.r.t. x is finite and non-zero."""
        grad_fn = jax.grad(lambda x: jnp.sum(bessel_k0(x)))
        grad_val = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_gradient_equals_negative_k1(self):
        """dK_0/dx = -K_1(x) identity."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        grad_fn = jax.vmap(jax.grad(lambda xi: bessel_k0(xi[None])[0]))
        grad_vals = grad_fn(x)
        neg_k1_vals = -bessel_k1(x)
        chex.assert_trees_all_close(grad_vals, neg_k1_vals, rtol=1e-4)

    def test_vmap(self):
        """vmap over batch dimension works correctly."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        vmapped = jax.vmap(lambda xi: bessel_k0(xi[None])[0])
        result = vmapped(x)
        expected = bessel_k0(x)
        chex.assert_trees_all_close(result, expected, rtol=1e-12)


class TestBesselK1(chex.TestCase, parameterized.TestCase):
    """Test bessel_k1 against scipy reference values."""

    @parameterized.named_parameters(
        ("x_0p1", 0.1, 9.8538447809),
        ("x_0p5", 0.5, 1.6564411200),
        ("x_1p0", 1.0, 0.6019072302),
        ("x_2p0", 2.0, 0.1398658818),
        ("x_5p0", 5.0, 0.0040446134),
        ("x_10p0", 10.0, 0.0000186488),
    )
    def test_known_values(self, x_val, expected):
        """K_1 matches scipy.special.k1 at representative points."""
        result = bessel_k1(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self):
        """K_1(x) > 0 for all x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k1(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self):
        """K_1(x) is strictly decreasing for x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k1(x)
        diffs = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self):
        """Preserves arbitrary batch dimensions."""
        x_1d = jnp.linspace(0.1, 5.0, 50)
        x_2d = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k1(x_1d), (50,))
        chex.assert_shape(bessel_k1(x_2d), (8, 16))

    def test_gradient_finite(self):
        """Gradient of K_1 w.r.t. x is finite and non-zero."""
        grad_fn = jax.grad(lambda x: jnp.sum(bessel_k1(x)))
        grad_val = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_k1_greater_than_k0(self):
        """K_1(x) > K_0(x) for all x > 0."""
        x = jnp.linspace(0.01, 10.0, 100)
        assert jnp.all(bessel_k1(x) > bessel_k0(x))


class TestBesselBranchTransition(chex.TestCase):
    """Test continuity at x=2 where polynomial branches switch."""

    def test_k0_continuity_at_branch(self):
        """K_0 is continuous across the x=2 branch point.

        Points 0.002 apart on a steeply decaying function, so we
        check that the relative jump is consistent with the local
        slope rather than a branch discontinuity.
        """
        x_below = jnp.array(1.999)
        x_above = jnp.array(2.001)
        k0_below = bessel_k0(x_below)
        k0_above = bessel_k0(x_above)
        chex.assert_trees_all_close(k0_below, k0_above, rtol=5e-3)

    def test_k1_continuity_at_branch(self):
        """K_1 is continuous across the x=2 branch point."""
        x_below = jnp.array(1.999)
        x_above = jnp.array(2.001)
        k1_below = bessel_k1(x_below)
        k1_above = bessel_k1(x_above)
        chex.assert_trees_all_close(k1_below, k1_above, rtol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__])
