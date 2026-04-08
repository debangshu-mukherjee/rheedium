"""Tests for shared special functions in rheedium.tools."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from scipy.special import iv as scipy_iv
from scipy.special import kv as scipy_kv

from rheedium.tools import bessel_k0, bessel_k1, bessel_kv
from rheedium.tools.special import (
    _bessel_iv_series,
    _bessel_k0_series,
    _bessel_k_half,
    _bessel_kn_recurrence,
    _bessel_kv_large,
    _bessel_kv_small_integer,
    _bessel_kv_small_non_integer,
)


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
    def test_known_values(self, x_val: float, expected: float) -> None:
        """K_0 matches scipy.special.k0 at representative points."""
        result = bessel_k0(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self) -> None:
        """K_0(x) > 0 for all x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k0(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self) -> None:
        """K_0(x) is strictly decreasing for x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k0(x)
        diffs = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self) -> None:
        """Preserves arbitrary batch dimensions."""
        x_1d = jnp.linspace(0.1, 5.0, 50)
        x_2d = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k0(x_1d), (50,))
        chex.assert_shape(bessel_k0(x_2d), (8, 16))

    def test_jit_matches_eager(self) -> None:
        """JIT-compiled output matches eager evaluation."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        eager = bessel_k0(x)
        jitted = jax.jit(bessel_k0)(x)
        chex.assert_trees_all_close(eager, jitted, rtol=1e-12)

    def test_gradient_finite(self) -> None:
        """Gradient of K_0 w.r.t. x is finite and non-zero."""
        grad_fn = jax.grad(lambda x: jnp.sum(bessel_k0(x)))
        grad_val = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_gradient_equals_negative_k1(self) -> None:
        """dK_0/dx = -K_1(x)."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        grad_fn = jax.vmap(jax.grad(lambda xi: bessel_k0(xi[None])[0]))
        grad_vals = grad_fn(x)
        neg_k1_vals = -bessel_k1(x)
        chex.assert_trees_all_close(grad_vals, neg_k1_vals, rtol=1e-4)

    def test_vmap(self) -> None:
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
    def test_known_values(self, x_val: float, expected: float) -> None:
        """K_1 matches scipy.special.k1 at representative points."""
        result = bessel_k1(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self) -> None:
        """K_1(x) > 0 for all x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k1(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self) -> None:
        """K_1(x) is strictly decreasing for x > 0."""
        x = jnp.linspace(0.01, 20.0, 200)
        result = bessel_k1(x)
        diffs = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self) -> None:
        """Preserves arbitrary batch dimensions."""
        x_1d = jnp.linspace(0.1, 5.0, 50)
        x_2d = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k1(x_1d), (50,))
        chex.assert_shape(bessel_k1(x_2d), (8, 16))

    def test_gradient_finite(self) -> None:
        """Gradient of K_1 w.r.t. x is finite and non-zero."""
        grad_fn = jax.grad(lambda x: jnp.sum(bessel_k1(x)))
        grad_val = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_k1_greater_than_k0(self) -> None:
        """K_1(x) > K_0(x) for all x > 0."""
        x = jnp.linspace(0.01, 10.0, 100)
        assert jnp.all(bessel_k1(x) > bessel_k0(x))


class TestBesselBranchTransition(chex.TestCase):
    """Test continuity at x=2 where polynomial branches switch."""

    def test_k0_continuity_at_branch(self) -> None:
        """K_0 is continuous across the x=2 branch point."""
        x_below = jnp.array(1.999)
        x_above = jnp.array(2.001)
        k0_below = bessel_k0(x_below)
        k0_above = bessel_k0(x_above)
        chex.assert_trees_all_close(k0_below, k0_above, rtol=5e-3)

    def test_k1_continuity_at_branch(self) -> None:
        """K_1 is continuous across the x=2 branch point."""
        x_below = jnp.array(1.999)
        x_above = jnp.array(2.001)
        k1_below = bessel_k1(x_below)
        k1_above = bessel_k1(x_above)
        chex.assert_trees_all_close(k1_below, k1_above, rtol=5e-3)


class TestBesselIvSeries(chex.TestCase):
    """Test _bessel_iv_series function."""

    @parameterized.named_parameters(
        ("v0_x0p5", 0.0, 0.5),
        ("v0_x1", 0.0, 1.0),
        ("v0_x2", 0.0, 2.0),
        ("v1_x0p5", 1.0, 0.5),
        ("v1_x1", 1.0, 1.0),
        ("v1_x2", 1.0, 2.0),
        ("v2_x1", 2.0, 1.0),
        ("v0p5_x1", 0.5, 1.0),
        ("v0p25_x0p5", 0.25, 0.5),
    )
    def test_iv_series_against_scipy(self, v: float, x: float) -> None:
        """I_v(x) series expansion matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_iv_series(v, x_arr, x_arr.dtype)
        expected = scipy_iv(v, x)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-6)

    def test_iv_series_vectorized(self) -> None:
        """I_v(x) supports vectorized input."""
        x_arr = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)
        result = _bessel_iv_series(1.0, x_arr, x_arr.dtype)
        expected = scipy_iv(1.0, np.array([0.5, 1.0, 1.5, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-6)

    def test_iv_series_negative_order(self) -> None:
        """Negative orders work for the K_v path."""
        x_arr = jnp.array([1.0], dtype=jnp.float64)
        result = _bessel_iv_series(-0.5, x_arr, x_arr.dtype)
        expected = scipy_iv(-0.5, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-5)

    def test_iv_series_output_shape(self) -> None:
        """Output shape matches input shape."""
        x_arr = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64)
        result = _bessel_iv_series(0.0, x_arr, x_arr.dtype)
        chex.assert_shape(result, (2, 2))


class TestBesselK0Series(chex.TestCase):
    """Test _bessel_k0_series function."""

    @parameterized.named_parameters(
        ("x0p5", 0.5, 0.9244190712276656),
        ("x1", 1.0, 0.42102443824070834),
        ("x1p5", 1.5, 0.21380556264752564),
        ("x2", 2.0, 0.11389387274953341),
    )
    def test_k0_series_against_scipy(self, x: float, expected: float) -> None:
        """K_0(x) series expansion matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_k0_series_vectorized(self) -> None:
        """K_0(x) supports vectorized input."""
        x_arr = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        expected = scipy_kv(0, np.array([0.5, 1.0, 1.5, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_k0_series_output_shape(self) -> None:
        """Output shape matches input shape."""
        x_arr = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_shape(result, (2, 2))

    def test_k0_series_positive_values(self) -> None:
        """K_0(x) stays positive for x > 0."""
        x_arr = jnp.array([0.1, 0.5, 1.0, 2.0], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        assert jnp.all(result > 0)


class TestBesselKnRecurrence(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kn_recurrence function."""

    def test_kn_recurrence_n0(self) -> None:
        """n=0 returns K_0."""
        x = jnp.array([1.0], dtype=jnp.float64)
        k0 = jnp.array([0.42102443824070834], dtype=jnp.float64)
        k1 = jnp.array([0.6019072301972346], dtype=jnp.float64)
        result = _bessel_kn_recurrence(
            jnp.array(0, dtype=jnp.int32), x, k0, k1
        )
        chex.assert_trees_all_close(result, k0, atol=1e-10)

    def test_kn_recurrence_n1(self) -> None:
        """n=1 returns K_1."""
        x = jnp.array([1.0], dtype=jnp.float64)
        k0 = jnp.array([0.42102443824070834], dtype=jnp.float64)
        k1 = jnp.array([0.6019072301972346], dtype=jnp.float64)
        result = _bessel_kn_recurrence(
            jnp.array(1, dtype=jnp.int32), x, k0, k1
        )
        chex.assert_trees_all_close(result, k1, atol=1e-10)

    @parameterized.named_parameters(
        ("n2_x0p5", 2, 0.5, 7.550183551240869),
        ("n2_x1", 2, 1.0, 1.6248388986351774),
        ("n2_x2", 2, 2.0, 0.2537597545660559),
        ("n3_x1", 3, 1.0, 7.101262824737945),
        ("n4_x1", 4, 1.0, 44.23241584706284),
    )
    def test_kn_recurrence_higher_orders(
        self, n: int, x: float, expected: float
    ) -> None:
        """Higher-order recurrence matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        k0 = jnp.array([scipy_kv(0, x)], dtype=jnp.float64)
        k1 = jnp.array([scipy_kv(1, x)], dtype=jnp.float64)
        result = _bessel_kn_recurrence(
            jnp.array(n, dtype=jnp.int32), x_arr, k0, k1
        )
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kn_recurrence_vectorized(self) -> None:
        """Vectorized x input works."""
        x_arr = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
        k0 = jnp.array(
            scipy_kv(0, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        k1 = jnp.array(
            scipy_kv(1, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        result = _bessel_kn_recurrence(
            jnp.array(2, dtype=jnp.int32), x_arr, k0, k1
        )
        expected = scipy_kv(2, np.array([0.5, 1.0, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)


class TestBesselKvSmallNonInteger(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_small_non_integer function."""

    @parameterized.named_parameters(
        ("v0p25_x0p5", 0.25, 0.5, 0.9603163249318826),
        ("v0p75_x0p5", 0.75, 0.5, 1.2917498162179113),
        ("v1p5_x0p5", 1.5, 0.5, 3.225142810499761),
        ("v1p5_x1", 1.5, 1.0, 0.9221370088957893),
    )
    def test_kv_small_non_integer(
        self, v: float, x: float, expected: float
    ) -> None:
        """Small-x non-integer branch matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_kv_small_non_integer(v, x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    def test_kv_small_non_integer_vectorized(self) -> None:
        """Vectorized non-integer input works."""
        x_arr = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        result = _bessel_kv_small_non_integer(0.25, x_arr, x_arr.dtype)
        expected = scipy_kv(0.25, np.array([0.5, 1.0, 1.5]))
        chex.assert_trees_all_close(result, expected, rtol=5e-2)


class TestBesselKvSmallInteger(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_small_integer function."""

    @parameterized.named_parameters(
        ("v0_x0p5", 0.0, 0.5, 0.9244190712276656),
        ("v0_x1", 0.0, 1.0, 0.42102443824070834),
        ("v0_x2", 0.0, 2.0, 0.11389387274953341),
    )
    def test_kv_small_integer_v0(
        self, v: float, x: float, expected: float
    ) -> None:
        """K_0(x) is the most accurate small-x integer branch."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_kv_small_integer(
            jnp.array(v, dtype=jnp.float64), x_arr, x_arr.dtype
        )
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_kv_small_integer_positive(self) -> None:
        """K_v stays positive on the tested domain."""
        x_arr = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        for v_val in [0.0, 1.0, 2.0]:
            result = _bessel_kv_small_integer(
                jnp.array(v_val, dtype=jnp.float64), x_arr, x_arr.dtype
            )
            assert jnp.all(result > 0)


class TestBesselKvLarge(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_large function."""

    @parameterized.named_parameters(
        ("v0_x5", 0.0, 5.0, 0.0036910983340425942),
        ("v0_x10", 0.0, 10.0, 1.7780062316167652e-05),
        ("v1_x5", 1.0, 5.0, 0.004044613445452164),
        ("v1_x10", 1.0, 10.0, 1.8648773453825586e-05),
        ("v2_x5", 2.0, 5.0, 0.00530894371222346),
        ("v0p5_x5", 0.5, 5.0, 0.0037766133746428825),
    )
    def test_kv_large_x(self, v: float, x: float, expected: float) -> None:
        """Asymptotic branch matches scipy at large x."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_kv_large(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kv_large_vectorized(self) -> None:
        """Vectorized large-x input works."""
        x_arr = jnp.array([5.0, 10.0, 15.0], dtype=jnp.float64)
        result = _bessel_kv_large(0.0, x_arr)
        expected = scipy_kv(0.0, np.array([5.0, 10.0, 15.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_kv_large_decays_exponentially(self) -> None:
        """Large-x branch decays monotonically."""
        x_arr = jnp.array([5.0, 10.0, 20.0, 50.0], dtype=jnp.float64)
        result = _bessel_kv_large(0.0, x_arr)
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1]


class TestBesselKHalf(chex.TestCase, parameterized.TestCase):
    """Test _bessel_k_half function."""

    @parameterized.named_parameters(
        ("x0p5", 0.5, 1.0750476034999203),
        ("x1", 1.0, 0.4610685044478946),
        ("x2", 2.0, 0.11993777196806146),
        ("x5", 5.0, 0.0037766133746428825),
    )
    def test_k_half_against_scipy(self, x: float, expected: float) -> None:
        """K_{1/2}(x) matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-10)

    def test_k_half_formula(self) -> None:
        """Closed form matches the direct expression."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        expected = jnp.sqrt(jnp.pi / (2.0 * x_arr)) * jnp.exp(-x_arr)
        chex.assert_trees_all_close(result, expected, atol=1e-14)

    def test_k_half_vectorized(self) -> None:
        """Vectorized input works."""
        x_arr = jnp.array([[0.5, 1.0], [2.0, 5.0]], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        expected = jnp.sqrt(jnp.pi / (2.0 * x_arr)) * jnp.exp(-x_arr)
        chex.assert_shape(result, (2, 2))
        chex.assert_trees_all_close(result, expected, atol=1e-14)


class TestBesselKv(chex.TestCase, parameterized.TestCase):
    """Test bessel_kv main function."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("v0_x0p5", 0.0, 0.5, 0.9244190712276656),
        ("v0_x1", 0.0, 1.0, 0.42102443824070834),
        ("v0_x2", 0.0, 2.0, 0.11389387274953341),
        ("v0_x5", 0.0, 5.0, 0.0036910983340425942),
        ("v1_x5", 1.0, 5.0, 0.004044613445452164),
        ("v2_x5", 2.0, 5.0, 0.00530894371222346),
    )
    def test_bessel_kv_integer_orders(
        self, v: float, x: float, expected: float
    ) -> None:
        """Integer-order K_v(x) matches scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = self.variant(bessel_kv)(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("v0p5_x0p5", 0.5, 0.5, 1.0750476034999203),
        ("v0p5_x1", 0.5, 1.0, 0.4610685044478946),
        ("v0p5_x2", 0.5, 2.0, 0.11993777196806146),
        ("v0p5_x5", 0.5, 5.0, 0.0037766133746428825),
    )
    def test_bessel_kv_half_order(
        self, v: float, x: float, expected: float
    ) -> None:
        """Half-order special case works."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = self.variant(bessel_kv)(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("v0p25_x0p5", 0.25, 0.5, 0.9603163249318826),
        ("v0p25_x2", 0.25, 2.0, 0.11537827684084918),
        ("v0p75_x0p5", 0.75, 0.5, 1.2917498162179113),
        ("v0p75_x2", 0.75, 2.0, 0.12790297862917527),
        ("v1p5_x0p5", 1.5, 0.5, 3.225142810499761),
        ("v1p5_x2", 1.5, 2.0, 0.1799066579520922),
    )
    def test_bessel_kv_non_integer_orders(
        self, v: float, x: float, expected: float
    ) -> None:
        """Non-integer K_v(x) matches scipy within approximation tolerance."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = self.variant(bessel_kv)(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_vectorized(self) -> None:
        """Vectorized x input works."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        result = self.variant(bessel_kv)(0.0, x_arr)
        expected = scipy_kv(0.0, np.array([0.5, 1.0, 2.0, 5.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_2d_input(self) -> None:
        """2D array input works."""
        x_arr = jnp.array([[2.0, 3.0], [5.0, 10.0]], dtype=jnp.float64)
        result = self.variant(bessel_kv)(0.0, x_arr)
        expected = scipy_kv(0.0, np.array([[2.0, 3.0], [5.0, 10.0]]))
        chex.assert_shape(result, (2, 2))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_positive_values(self) -> None:
        """K_v(x) stays positive for positive x."""
        x_arr = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
        for v in [0.0, 0.5, 1.0, 2.0]:
            result = self.variant(bessel_kv)(v, x_arr)
            assert jnp.all(result > 0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_monotonic_decay(self) -> None:
        """K_v(x) decays monotonically for fixed v."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
        for v in [0.0, 0.5, 1.0, 2.0]:
            result = self.variant(bessel_kv)(v, x_arr)
            for i in range(len(result) - 1):
                assert result[i] > result[i + 1]

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_order_relation(self) -> None:
        """K_0(x) < K_1(x) at small x."""
        x_arr = jnp.array([0.5], dtype=jnp.float64)
        k0 = self.variant(bessel_kv)(0.0, x_arr)[0]
        k1 = self.variant(bessel_kv)(1.0, x_arr)[0]
        assert k0 < k1

    def test_bessel_kv_jit_compilation(self) -> None:
        """bessel_kv can be JIT compiled."""
        x_arr = jnp.array([1.0], dtype=jnp.float64)
        result = jax.jit(bessel_kv, static_argnums=(0,))(0.0, x_arr)
        expected = scipy_kv(0, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_bessel_kv_jit_works(self) -> None:
        """JIT works across representative orders."""
        x_arr = jnp.array([5.0, 10.0], dtype=jnp.float64)
        jit_bessel = jax.jit(bessel_kv, static_argnums=(0,))
        for v in [0.0, 0.5, 1.0]:
            result = jit_bessel(v, x_arr)
            expected = scipy_kv(v, np.array([5.0, 10.0]))
            chex.assert_trees_all_close(result, expected, rtol=1e-3)


class TestBesselKvEdgeCases(chex.TestCase):
    """Test edge cases and numerical stability."""

    def test_bessel_kv_small_x_boundary(self) -> None:
        """Behavior near the small/large branch boundary stays stable."""
        x_near_boundary = jnp.array([1.9, 2.0, 2.1], dtype=jnp.float64)
        result = bessel_kv(0.0, x_near_boundary)
        expected = scipy_kv(0, np.array([1.9, 2.0, 2.1]))
        chex.assert_trees_all_close(result, expected, rtol=5e-3)

    def test_bessel_kv_large_x(self) -> None:
        """Behavior for large x values stays accurate."""
        x_large = jnp.array([10.0, 20.0, 50.0], dtype=jnp.float64)
        result = bessel_kv(0.0, x_large)
        expected = scipy_kv(0, np.array([10.0, 20.0, 50.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    def test_bessel_kv_higher_integer_order(self) -> None:
        """Higher integer orders work in the large-x regime."""
        x_arr = jnp.array([5.0, 10.0], dtype=jnp.float64)
        for n in [3, 4, 5]:
            result = bessel_kv(float(n), x_arr)
            expected = scipy_kv(n, np.array([5.0, 10.0]))
            chex.assert_trees_all_close(result, expected, rtol=1e-2)
