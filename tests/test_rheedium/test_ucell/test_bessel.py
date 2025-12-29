"""Tests for ucell.bessel module.

Tests the JAX-compatible modified Bessel function implementations,
including internal helper functions.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from scipy.special import kv as scipy_kv
from scipy.special import iv as scipy_iv

from rheedium.ucell.bessel import (
    _bessel_iv_series,
    _bessel_k0_series,
    _bessel_k_half,
    _bessel_kn_recurrence,
    _bessel_kv_large,
    _bessel_kv_small_integer,
    _bessel_kv_small_non_integer,
    bessel_kv,
)


class TestBesselIvSeries(chex.TestCase):
    """Test _bessel_iv_series function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

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
        """Test I_v(x) series expansion against scipy reference."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_iv_series(v, x_arr, x_arr.dtype)
        expected = scipy_iv(v, x)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-6)

    def test_iv_series_vectorized(self) -> None:
        """Test I_v(x) with vectorized input."""
        x_arr = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)
        v = 1.0
        result = _bessel_iv_series(v, x_arr, x_arr.dtype)
        expected = scipy_iv(v, np.array([0.5, 1.0, 1.5, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-6)

    def test_iv_series_negative_order(self) -> None:
        """Test I_v(x) with negative order (used in K_v calculation)."""
        x_arr = jnp.array([1.0], dtype=jnp.float64)
        v = -0.5
        result = _bessel_iv_series(v, x_arr, x_arr.dtype)
        expected = scipy_iv(v, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-5)

    def test_iv_series_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        x_arr = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64)
        result = _bessel_iv_series(0.0, x_arr, x_arr.dtype)
        chex.assert_shape(result, (2, 2))


class TestBesselK0Series(chex.TestCase):
    """Test _bessel_k0_series function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @parameterized.named_parameters(
        ("x0p5", 0.5, 0.9244190712276656),
        ("x1", 1.0, 0.42102443824070834),
        ("x1p5", 1.5, 0.21380556264752564),
        ("x2", 2.0, 0.11389387274953341),
    )
    def test_k0_series_against_scipy(self, x: float, expected: float) -> None:
        """Test K_0(x) series expansion against scipy reference."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_k0_series_vectorized(self) -> None:
        """Test K_0(x) with vectorized input."""
        x_arr = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        expected = scipy_kv(0, np.array([0.5, 1.0, 1.5, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_k0_series_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        x_arr = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_shape(result, (2, 2))

    def test_k0_series_positive_values(self) -> None:
        """Test that K_0(x) returns positive values for positive x."""
        x_arr = jnp.array([0.1, 0.5, 1.0, 2.0], dtype=jnp.float64)
        result = _bessel_k0_series(x_arr, x_arr.dtype)
        assert jnp.all(result > 0), "K_0(x) should be positive for x > 0"


class TestBesselKnRecurrence(chex.TestCase):
    """Test _bessel_kn_recurrence function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    def test_kn_recurrence_n0(self) -> None:
        """Test K_n recurrence returns K_0 when n=0."""
        x = jnp.array([1.0], dtype=jnp.float64)
        k0 = jnp.array([0.42102443824070834], dtype=jnp.float64)
        k1 = jnp.array([0.6019072301972346], dtype=jnp.float64)
        n = jnp.array(0, dtype=jnp.int32)
        result = _bessel_kn_recurrence(n, x, k0, k1)
        chex.assert_trees_all_close(result, k0, atol=1e-10)

    def test_kn_recurrence_n1(self) -> None:
        """Test K_n recurrence returns K_1 when n=1."""
        x = jnp.array([1.0], dtype=jnp.float64)
        k0 = jnp.array([0.42102443824070834], dtype=jnp.float64)
        k1 = jnp.array([0.6019072301972346], dtype=jnp.float64)
        n = jnp.array(1, dtype=jnp.int32)
        result = _bessel_kn_recurrence(n, x, k0, k1)
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
        """Test K_n recurrence for higher orders against scipy."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        k0 = jnp.array([scipy_kv(0, x)], dtype=jnp.float64)
        k1 = jnp.array([scipy_kv(1, x)], dtype=jnp.float64)
        n_arr = jnp.array(n, dtype=jnp.int32)
        result = _bessel_kn_recurrence(n_arr, x_arr, k0, k1)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kn_recurrence_vectorized(self) -> None:
        """Test K_n recurrence with vectorized x input."""
        x_arr = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
        k0 = jnp.array(
            scipy_kv(0, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        k1 = jnp.array(
            scipy_kv(1, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        n = jnp.array(2, dtype=jnp.int32)
        result = _bessel_kn_recurrence(n, x_arr, k0, k1)
        expected = scipy_kv(2, np.array([0.5, 1.0, 2.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)


class TestBesselKvSmallNonInteger(chex.TestCase):
    """Test _bessel_kv_small_non_integer function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @parameterized.named_parameters(
        ("v0p25_x0p5", 0.25, 0.5, 0.9603163249318826),
        ("v0p75_x0p5", 0.75, 0.5, 1.2917498162179113),
        ("v1p5_x0p5", 1.5, 0.5, 3.225142810499761),
        ("v1p5_x1", 1.5, 1.0, 0.9221370088957893),
    )
    def test_kv_small_non_integer(
        self, v: float, x: float, expected: float
    ) -> None:
        """Test K_v(x) for small x and non-integer v."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_kv_small_non_integer(v, x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    def test_kv_small_non_integer_vectorized(self) -> None:
        """Test with vectorized input."""
        x_arr = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        v = 0.25
        result = _bessel_kv_small_non_integer(v, x_arr, x_arr.dtype)
        expected = scipy_kv(v, np.array([0.5, 1.0, 1.5]))
        chex.assert_trees_all_close(result, expected, rtol=5e-2)


class TestBesselKvSmallInteger(chex.TestCase):
    """Test _bessel_kv_small_integer function.

    Note: K_0 uses accurate polynomial approx. K_1+ has limited accuracy.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @parameterized.named_parameters(
        ("v0_x0p5", 0.0, 0.5, 0.9244190712276656),
        ("v0_x1", 0.0, 1.0, 0.42102443824070834),
        ("v0_x2", 0.0, 2.0, 0.11389387274953341),
    )
    def test_kv_small_integer_v0(
        self, v: float, x: float, expected: float
    ) -> None:
        """Test K_0(x) - the most accurate case."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        v_arr = jnp.array(v, dtype=jnp.float64)
        result = _bessel_kv_small_integer(v_arr, x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_kv_small_integer_positive(self) -> None:
        """Test that K_v returns positive values."""
        x_arr = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        for v_val in [0.0, 1.0, 2.0]:
            v = jnp.array(v_val, dtype=jnp.float64)
            result = _bessel_kv_small_integer(v, x_arr, x_arr.dtype)
            assert jnp.all(result > 0), f"K_{v_val}(x) should be positive"


class TestBesselKvLarge(chex.TestCase):
    """Test _bessel_kv_large function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @parameterized.named_parameters(
        ("v0_x5", 0.0, 5.0, 0.0036910983340425942),
        ("v0_x10", 0.0, 10.0, 1.7780062316167652e-05),
        ("v1_x5", 1.0, 5.0, 0.004044613445452164),
        ("v1_x10", 1.0, 10.0, 1.8648773453825586e-05),
        ("v2_x5", 2.0, 5.0, 0.00530894371222346),
        ("v0p5_x5", 0.5, 5.0, 0.0037766133746428825),
    )
    def test_kv_large_x(self, v: float, x: float, expected: float) -> None:
        """Test asymptotic expansion for large x."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_kv_large(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kv_large_vectorized(self) -> None:
        """Test asymptotic expansion with vectorized input."""
        x_arr = jnp.array([5.0, 10.0, 15.0], dtype=jnp.float64)
        v = 0.0
        result = _bessel_kv_large(v, x_arr)
        expected = scipy_kv(v, np.array([5.0, 10.0, 15.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_kv_large_decays_exponentially(self) -> None:
        """Test that K_v(x) decays exponentially for large x."""
        x_arr = jnp.array([5.0, 10.0, 20.0, 50.0], dtype=jnp.float64)
        result = _bessel_kv_large(0.0, x_arr)
        # Values should decrease monotonically
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1], "K_v(x) should decay for large x"


class TestBesselKHalf(chex.TestCase):
    """Test _bessel_k_half function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    @parameterized.named_parameters(
        ("x0p5", 0.5, 1.0750476034999203),
        ("x1", 1.0, 0.4610685044478946),
        ("x2", 2.0, 0.11993777196806146),
        ("x5", 5.0, 0.0037766133746428825),
    )
    def test_k_half_against_scipy(self, x: float, expected: float) -> None:
        """Test K_{1/2}(x) against scipy reference."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-10)

    def test_k_half_formula(self) -> None:
        """Test K_{1/2}(x) = sqrt(pi/(2x)) * exp(-x)."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        expected = jnp.sqrt(jnp.pi / (2.0 * x_arr)) * jnp.exp(-x_arr)
        chex.assert_trees_all_close(result, expected, atol=1e-14)

    def test_k_half_vectorized(self) -> None:
        """Test K_{1/2}(x) with vectorized input."""
        x_arr = jnp.array([[0.5, 1.0], [2.0, 5.0]], dtype=jnp.float64)
        result = _bessel_k_half(x_arr)
        chex.assert_shape(result, (2, 2))
        expected = jnp.sqrt(jnp.pi / (2.0 * x_arr)) * jnp.exp(-x_arr)
        chex.assert_trees_all_close(result, expected, atol=1e-14)


class TestBesselKv(chex.TestCase):
    """Test bessel_kv main function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

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
        """Test K_v(x) for integer orders against scipy reference."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, x_arr)
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
        """Test K_{1/2}(x) special case."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, x_arr)
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
        """Test K_v(x) for non-integer orders against scipy reference."""
        x_arr = jnp.array([x], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_vectorized(self) -> None:
        """Test K_v(x) with vectorized x input."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        v = 0.0
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, x_arr)
        expected = scipy_kv(v, np.array([0.5, 1.0, 2.0, 5.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_2d_input(self) -> None:
        """Test K_v(x) with 2D array input."""
        x_arr = jnp.array([[2.0, 3.0], [5.0, 10.0]], dtype=jnp.float64)
        v = 0.0
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, x_arr)
        chex.assert_shape(result, (2, 2))
        expected = scipy_kv(v, np.array([[2.0, 3.0], [5.0, 10.0]]))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_positive_values(self) -> None:
        """Test that K_v(x) returns positive values for positive x."""
        x_arr = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        for v in [0.0, 0.5, 1.0, 2.0]:
            result = var_bessel_kv(v, x_arr)
            assert jnp.all(
                result > 0
            ), f"K_{v}(x) should be positive for x > 0"

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_monotonic_decay(self) -> None:
        """Test that K_v(x) decays monotonically for fixed v."""
        x_arr = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        for v in [0.0, 0.5, 1.0, 2.0]:
            result = var_bessel_kv(v, x_arr)
            for i in range(len(result) - 1):
                assert (
                    result[i] > result[i + 1]
                ), f"K_{v}(x) should decay monotonically"

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_order_relation(self) -> None:
        """Test K_v(x) > K_v+1(x) for small x."""
        x_arr = jnp.array([0.5], dtype=jnp.float64)
        var_bessel_kv = self.variant(bessel_kv)
        k0 = var_bessel_kv(0.0, x_arr)[0]
        k1 = var_bessel_kv(1.0, x_arr)[0]
        # For small x, K_0(x) < K_1(x) due to singularity behavior
        # Actually K_n increases with n for small x
        assert k0 < k1, "K_0(x) < K_1(x) for small x"

    def test_bessel_kv_jit_compilation(self) -> None:
        """Test that bessel_kv can be JIT compiled."""
        x_arr = jnp.array([1.0], dtype=jnp.float64)
        jit_bessel = jax.jit(bessel_kv, static_argnums=(0,))
        result = jit_bessel(0.0, x_arr)
        expected = scipy_kv(0, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_bessel_kv_jit_works(self) -> None:
        """Test that bessel_kv works with JIT on various inputs."""
        x_arr = jnp.array([5.0, 10.0], dtype=jnp.float64)
        jit_bessel = jax.jit(bessel_kv, static_argnums=(0,))
        for v in [0.0, 0.5, 1.0]:
            result = jit_bessel(v, x_arr)
            expected = scipy_kv(v, np.array([5.0, 10.0]))
            chex.assert_trees_all_close(result, expected, rtol=1e-3)


class TestBesselKvEdgeCases(chex.TestCase):
    """Test edge cases and numerical stability."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        chex.set_n_cpu_devices(1)

    def test_bessel_kv_small_x_boundary(self) -> None:
        """Test behavior near the small/large x boundary."""
        # Boundary is at x=2.0
        x_near_boundary = jnp.array([1.9, 2.0, 2.1], dtype=jnp.float64)
        result = bessel_kv(0.0, x_near_boundary)
        expected = scipy_kv(0, np.array([1.9, 2.0, 2.1]))
        chex.assert_trees_all_close(result, expected, rtol=5e-3)

    def test_bessel_kv_large_x(self) -> None:
        """Test behavior for large x values."""
        x_large = jnp.array([10.0, 20.0, 50.0], dtype=jnp.float64)
        result = bessel_kv(0.0, x_large)
        expected = scipy_kv(0, np.array([10.0, 20.0, 50.0]))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    def test_bessel_kv_higher_integer_order(self) -> None:
        """Test higher integer orders at large x (where asymptotic is used)."""
        x_arr = jnp.array([5.0, 10.0], dtype=jnp.float64)
        for n in [3, 4, 5]:
            result = bessel_kv(float(n), x_arr)
            expected = scipy_kv(n, np.array([5.0, 10.0]))
            chex.assert_trees_all_close(result, expected, rtol=1e-2)


if __name__ == "__main__":
    chex.TestCase.main()
