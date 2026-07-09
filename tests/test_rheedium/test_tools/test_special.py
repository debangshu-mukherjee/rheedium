"""Tests for shared special functions in rheedium.tools."""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jaxtyping import Array, Float, Integer
from numpy.typing import NDArray
from scipy.special import iv as scipy_iv
from scipy.special import kv as scipy_kv

from rheedium.tools.special import (
    _bessel_iv_series,
    _bessel_k0_series,
    _bessel_k_half,
    _bessel_kn_recurrence,
    _bessel_kv_large,
    _bessel_kv_small_integer,
    _bessel_kv_small_non_integer,
    bessel_k0,
    bessel_k1,
    bessel_kv,
)
from rheedium.types.custom_types import scalar_float


def _relative_error(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return elementwise relative error against nonzero SciPy references."""
    return np.abs((actual - expected) / expected)


class TestBesselK0(chex.TestCase, parameterized.TestCase):
    """Test bessel_k0 against scipy reference values.

    :see: :func:`~rheedium.tools.bessel_k0`
    """

    @parameterized.named_parameters(
        ("x_0p1", 0.1, 2.4270690247),
        ("x_0p5", 0.5, 0.9244190712),
        ("x_1p0", 1.0, 0.4210244382),
        ("x_2p0", 2.0, 0.1138938727),
        ("x_5p0", 5.0, 0.0036910983),
        ("x_10p0", 10.0, 0.0000177801),
    )
    def test_known_values(self, x_val: float, expected: float) -> None:
        r"""K_0 matches scipy.special.k0 at representative points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0 matches
        scipy.special.k0 at representative points.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``x_val``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        result: Float[Array, "..."] = bessel_k0(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self) -> None:
        r"""K_0(x) > 0 for all x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) > 0 for all
        x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.linspace(0.01, 20.0, 200)
        result: Float[Array, "..."] = bessel_k0(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self) -> None:
        r"""K_0(x) is strictly decreasing for x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) is strictly
        decreasing for x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.linspace(0.01, 20.0, 200)
        result: Float[Array, "..."] = bessel_k0(x)
        diffs: Float[Array, "..."] = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self) -> None:
        r"""Preserves arbitrary batch dimensions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Preserves
        arbitrary batch dimensions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_1d: Float[Array, "..."] = jnp.linspace(0.1, 5.0, 50)
        x_2d: Float[Array, "..."] = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k0(x_1d), (50,))
        chex.assert_shape(bessel_k0(x_2d), (8, 16))

    def test_jit_matches_eager(self) -> None:
        r"""JIT-compiled output matches eager evaluation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT-compiled
        output matches eager evaluation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.array([0.5, 1.0, 2.0, 5.0])
        eager: Any = bessel_k0(x)
        jitted: Callable[..., Any] = jax.jit(bessel_k0)(x)
        chex.assert_trees_all_close(eager, jitted, rtol=1e-12)

    def test_gradient_finite(self) -> None:
        r"""Gradient of K_0 w.r.t. x is finite and non-zero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient of K_0
        w.r.t. x is finite and non-zero.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        grad_fn: Callable[[scalar_float], scalar_float] = jax.grad(
            lambda x: jnp.sum(bessel_k0(x))
        )
        grad_val: scalar_float = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_gradient_equals_negative_k1(self) -> None:
        r"""dK_0/dx = -K_1(x).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: dK_0/dx = -K_1(x).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, vectorization, protecting
        JAX transform compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.array([0.5, 1.0, 2.0, 5.0])
        grad_fn: Callable[[Float[Array, "points"]], Float[Array, "points"]] = (
            jax.vmap(jax.grad(lambda xi: bessel_k0(xi[None])[0]))
        )
        grad_vals: Float[Array, "points"] = grad_fn(x)
        neg_k1_vals: Float[Array, "points"] = -bessel_k1(x)
        chex.assert_trees_all_close(grad_vals, neg_k1_vals, rtol=1e-4)

    def test_vmap(self) -> None:
        r"""Vmap over batch dimension works correctly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vmap over batch
        dimension works correctly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.array([0.5, 1.0, 2.0, 5.0])
        vmapped: Callable[[Float[Array, "points"]], Float[Array, "points"]] = (
            jax.vmap(lambda xi: bessel_k0(xi[None])[0])
        )
        result: Float[Array, "points"] = vmapped(x)
        expected: Float[Array, "points"] = bessel_k0(x)
        chex.assert_trees_all_close(result, expected, rtol=1e-12)


class TestBesselK1(chex.TestCase, parameterized.TestCase):
    """Test bessel_k1 against scipy reference values.

    :see: :func:`~rheedium.tools.bessel_k1`
    """

    @parameterized.named_parameters(
        ("x_0p1", 0.1, 9.8538447809),
        ("x_0p5", 0.5, 1.6564411200),
        ("x_1p0", 1.0, 0.6019072302),
        ("x_2p0", 2.0, 0.1398658818),
        ("x_5p0", 5.0, 0.0040446134),
        ("x_10p0", 10.0, 0.0000186488),
    )
    def test_known_values(self, x_val: float, expected: float) -> None:
        r"""K_1 matches scipy.special.k1 at representative points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_1 matches
        scipy.special.k1 at representative points.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``x_val``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        result: Float[Array, "..."] = bessel_k1(jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-5)

    def test_positive_output(self) -> None:
        r"""K_1(x) > 0 for all x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_1(x) > 0 for all
        x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.linspace(0.01, 20.0, 200)
        result: Float[Array, "..."] = bessel_k1(x)
        chex.assert_tree_all_finite(result)
        assert jnp.all(result > 0)

    def test_monotone_decreasing(self) -> None:
        r"""K_1(x) is strictly decreasing for x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_1(x) is strictly
        decreasing for x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.linspace(0.01, 20.0, 200)
        result: Float[Array, "..."] = bessel_k1(x)
        diffs: Float[Array, "..."] = jnp.diff(result)
        assert jnp.all(diffs < 0)

    def test_batch_shapes(self) -> None:
        r"""Preserves arbitrary batch dimensions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Preserves
        arbitrary batch dimensions.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_1d: Float[Array, "..."] = jnp.linspace(0.1, 5.0, 50)
        x_2d: Float[Array, "..."] = jnp.ones((8, 16)) * 2.0
        chex.assert_shape(bessel_k1(x_1d), (50,))
        chex.assert_shape(bessel_k1(x_2d), (8, 16))

    def test_gradient_finite(self) -> None:
        r"""Gradient of K_1 w.r.t. x is finite and non-zero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient of K_1
        w.r.t. x is finite and non-zero.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        grad_fn: Callable[[scalar_float], scalar_float] = jax.grad(
            lambda x: jnp.sum(bessel_k1(x))
        )
        grad_val: scalar_float = grad_fn(jnp.array(1.0))
        chex.assert_tree_all_finite(grad_val)
        assert jnp.abs(grad_val) > 0

    def test_k1_greater_than_k0(self) -> None:
        r"""K_1(x) > K_0(x) for all x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_1(x) > K_0(x)
        for all x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.linspace(0.01, 10.0, 100)
        assert jnp.all(bessel_k1(x) > bessel_k0(x))


class TestBesselBranchTransition(chex.TestCase):
    """Test continuity at x=2 where polynomial branches switch."""

    def test_k0_continuity_at_branch(self) -> None:
        r"""K_0 is continuous across the x=2 branch point.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0 is continuous
        across the x=2 branch point.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_below: scalar_float = jnp.array(1.999)
        x_above: scalar_float = jnp.array(2.001)
        k0_below: Any = bessel_k0(x_below)
        k0_above: Any = bessel_k0(x_above)
        chex.assert_trees_all_close(k0_below, k0_above, rtol=5e-3)

    def test_k1_continuity_at_branch(self) -> None:
        r"""K_1 is continuous across the x=2 branch point.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_1 is continuous
        across the x=2 branch point.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_below: scalar_float = jnp.array(1.999)
        x_above: scalar_float = jnp.array(2.001)
        k1_below: Any = bessel_k1(x_below)
        k1_above: Any = bessel_k1(x_above)
        chex.assert_trees_all_close(k1_below, k1_above, rtol=5e-3)


class TestBesselIvSeries(chex.TestCase):
    """Test _bessel_iv_series function.

    :see: :func:`~rheedium.tools.special._bessel_iv_series`
    """

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
        r"""I_v(x) series expansion matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: I_v(x) series
        expansion matches scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_iv_series(v, x_arr, x_arr.dtype)
        expected: Float[Array, "..."] = scipy_iv(v, x)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-6)

    def test_iv_series_vectorized(self) -> None:
        r"""I_v(x) supports vectorized input.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: I_v(x) supports
        vectorized input.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 1.5, 2.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_iv_series(
            1.0, x_arr, x_arr.dtype
        )
        expected: Float[NDArray, "..."] = scipy_iv(
            1.0, np.array([0.5, 1.0, 1.5, 2.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-6)

    def test_iv_series_negative_order(self) -> None:
        r"""Negative orders work for the K_v path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Negative orders
        work for the K_v path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([1.0], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_iv_series(
            -0.5, x_arr, x_arr.dtype
        )
        expected: Float[Array, "..."] = scipy_iv(-0.5, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-5)

    def test_iv_series_output_shape(self) -> None:
        r"""Output shape matches input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_iv_series(
            0.0, x_arr, x_arr.dtype
        )
        chex.assert_shape(result, (2, 2))


class TestBesselK0Series(chex.TestCase):
    """Test _bessel_k0_series function.

    :see: :func:`~rheedium.tools.special._bessel_k0_series`
    """

    @parameterized.named_parameters(
        ("x0p5", 0.5, 0.9244190712276656),
        ("x1", 1.0, 0.42102443824070834),
        ("x1p5", 1.5, 0.21380556264752564),
        ("x2", 2.0, 0.11389387274953341),
    )
    def test_k0_series_against_scipy(self, x: float, expected: float) -> None:
        r"""K_0(x) series expansion matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) series
        expansion matches scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_k0_series_vectorized(self) -> None:
        r"""K_0(x) supports vectorized input.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) supports
        vectorized input.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 1.5, 2.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_k0_series(x_arr, x_arr.dtype)
        expected: Float[NDArray, "..."] = scipy_kv(
            0, np.array([0.5, 1.0, 1.5, 2.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_k0_series_output_shape(self) -> None:
        r"""Output shape matches input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_k0_series(x_arr, x_arr.dtype)
        chex.assert_shape(result, (2, 2))

    def test_k0_series_positive_values(self) -> None:
        r"""K_0(x) stays positive for x > 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) stays
        positive for x > 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.1, 0.5, 1.0, 2.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_k0_series(x_arr, x_arr.dtype)
        assert jnp.all(result > 0)


class TestBesselKnRecurrence(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kn_recurrence function.

    :see: :func:`~rheedium.tools.special._bessel_kn_recurrence`
    """

    def test_kn_recurrence_n0(self) -> None:
        r"""n=0 returns K_0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: n=0 returns K_0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.array([1.0], dtype=jnp.float64)
        k0: Float[Array, "..."] = jnp.array(
            [0.42102443824070834], dtype=jnp.float64
        )
        k1: Float[Array, "..."] = jnp.array(
            [0.6019072301972346], dtype=jnp.float64
        )
        result: Integer[Array, "..."] = _bessel_kn_recurrence(
            jnp.array(0, dtype=jnp.int32), x, k0, k1
        )
        chex.assert_trees_all_close(result, k0, atol=1e-10)

    def test_kn_recurrence_n1(self) -> None:
        r"""n=1 returns K_1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: n=1 returns K_1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x: Float[Array, "..."] = jnp.array([1.0], dtype=jnp.float64)
        k0: Float[Array, "..."] = jnp.array(
            [0.42102443824070834], dtype=jnp.float64
        )
        k1: Float[Array, "..."] = jnp.array(
            [0.6019072301972346], dtype=jnp.float64
        )
        result: Integer[Array, "..."] = _bessel_kn_recurrence(
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
        r"""Higher-order recurrence matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Higher-order
        recurrence matches scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``n``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        k0: Float[Array, "..."] = jnp.array(
            [scipy_kv(0, x)], dtype=jnp.float64
        )
        k1: Float[Array, "..."] = jnp.array(
            [scipy_kv(1, x)], dtype=jnp.float64
        )
        result: Integer[Array, "..."] = _bessel_kn_recurrence(
            jnp.array(n, dtype=jnp.int32), x_arr, k0, k1
        )
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kn_recurrence_vectorized(self) -> None:
        r"""Vectorized x input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vectorized x input
        works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 2.0], dtype=jnp.float64
        )
        k0: Float[Array, "..."] = jnp.array(
            scipy_kv(0, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        k1: Float[Array, "..."] = jnp.array(
            scipy_kv(1, np.array([0.5, 1.0, 2.0])), dtype=jnp.float64
        )
        result: Integer[Array, "..."] = _bessel_kn_recurrence(
            jnp.array(2, dtype=jnp.int32), x_arr, k0, k1
        )
        expected: Float[NDArray, "..."] = scipy_kv(
            2, np.array([0.5, 1.0, 2.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-4)


class TestBesselKvSmallNonInteger(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_small_non_integer function.

    :see: :func:`~rheedium.tools.special._bessel_kv_small_non_integer`
    """

    @parameterized.named_parameters(
        ("v0p25_x0p5", 0.25, 0.5, 0.9603163249318826),
        ("v0p75_x0p5", 0.75, 0.5, 1.2917498162179113),
        ("v1p5_x0p5", 1.5, 0.5, 3.225142810499761),
        ("v1p5_x1", 1.5, 1.0, 0.9221370088957893),
    )
    def test_kv_small_non_integer(
        self, v: float, x: float, expected: float
    ) -> None:
        r"""Small-x non-integer branch matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Small-x
        non-integer branch matches scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_kv_small_non_integer(
            v, x_arr, x_arr.dtype
        )
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    def test_kv_small_non_integer_vectorized(self) -> None:
        r"""Vectorized non-integer input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vectorized
        non-integer input works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 1.5], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_kv_small_non_integer(
            0.25, x_arr, x_arr.dtype
        )
        expected: Float[NDArray, "..."] = scipy_kv(
            0.25, np.array([0.5, 1.0, 1.5])
        )
        chex.assert_trees_all_close(result, expected, rtol=5e-2)


class TestBesselKvSmallInteger(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_small_integer function.

    :see: :func:`~rheedium.tools.special._bessel_kv_small_integer`
    """

    @parameterized.named_parameters(
        ("v0_x0p5", 0.0, 0.5, 0.9244190712276656),
        ("v0_x1", 0.0, 1.0, 0.42102443824070834),
        ("v0_x2", 0.0, 2.0, 0.11389387274953341),
    )
    def test_kv_small_integer_v0(
        self, v: float, x: float, expected: float
    ) -> None:
        r"""K_0(x) is the most accurate small-x integer branch.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) is the most
        accurate small-x integer branch.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_kv_small_integer(
            jnp.array(v, dtype=jnp.float64), x_arr, x_arr.dtype
        )
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_kv_small_integer_positive(self) -> None:
        r"""K_v stays positive on the tested domain.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_v stays positive
        on the tested domain.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 1.5], dtype=jnp.float64
        )
        v_val: scalar_float
        for v_val in [0.0, 1.0, 2.0]:
            result: Float[Array, "..."] = _bessel_kv_small_integer(
                jnp.array(v_val, dtype=jnp.float64), x_arr, x_arr.dtype
            )
            assert jnp.all(result > 0)


class TestBesselKvLarge(chex.TestCase, parameterized.TestCase):
    """Test _bessel_kv_large function.

    :see: :func:`~rheedium.tools.special._bessel_kv_large`
    """

    @parameterized.named_parameters(
        ("v0_x5", 0.0, 5.0, 0.0036910983340425942),
        ("v0_x10", 0.0, 10.0, 1.7780062316167652e-05),
        ("v1_x5", 1.0, 5.0, 0.004044613445452164),
        ("v1_x10", 1.0, 10.0, 1.8648773453825586e-05),
        ("v2_x5", 2.0, 5.0, 0.00530894371222346),
        ("v0p5_x5", 0.5, 5.0, 0.0037766133746428825),
    )
    def test_kv_large_x(self, v: float, x: float, expected: float) -> None:
        r"""Asymptotic branch matches scipy at large x.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Asymptotic branch
        matches scipy at large x.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_kv_large(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-4)

    def test_kv_large_vectorized(self) -> None:
        r"""Vectorized large-x input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vectorized large-x
        input works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [5.0, 10.0, 15.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_kv_large(0.0, x_arr)
        expected: Float[NDArray, "..."] = scipy_kv(
            0.0, np.array([5.0, 10.0, 15.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    def test_kv_large_decays_exponentially(self) -> None:
        r"""Large-x branch decays monotonically.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Large-x branch
        decays monotonically.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [5.0, 10.0, 20.0, 50.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_kv_large(0.0, x_arr)
        i: int
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1]


class TestBesselKHalf(chex.TestCase, parameterized.TestCase):
    """Test _bessel_k_half function.

    :see: :func:`~rheedium.tools.special._bessel_k_half`
    """

    @parameterized.named_parameters(
        ("x0p5", 0.5, 1.0750476034999203),
        ("x1", 1.0, 0.4610685044478946),
        ("x2", 2.0, 0.11993777196806146),
        ("x5", 5.0, 0.0037766133746428825),
    )
    def test_k_half_against_scipy(self, x: float, expected: float) -> None:
        r"""K_{1/2}(x) matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_{1/2}(x) matches
        scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Float[Array, "..."] = _bessel_k_half(x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-10)

    def test_k_half_formula(self) -> None:
        r"""Closed form matches the direct expression.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Closed form
        matches the direct expression.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 2.0, 5.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_k_half(x_arr)
        expected: Float[Array, "..."] = jnp.sqrt(
            jnp.pi / (2.0 * x_arr)
        ) * jnp.exp(-x_arr)
        chex.assert_trees_all_close(result, expected, atol=1e-14)

    def test_k_half_vectorized(self) -> None:
        r"""Vectorized input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vectorized input
        works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [[0.5, 1.0], [2.0, 5.0]], dtype=jnp.float64
        )
        result: Float[Array, "..."] = _bessel_k_half(x_arr)
        expected: Float[Array, "..."] = jnp.sqrt(
            jnp.pi / (2.0 * x_arr)
        ) * jnp.exp(-x_arr)
        chex.assert_shape(result, (2, 2))
        chex.assert_trees_all_close(result, expected, atol=1e-14)


class TestBesselKv(chex.TestCase, parameterized.TestCase):
    """Test bessel_kv main function.

    :see: :func:`~rheedium.tools.bessel_kv`
    """

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
        r"""Integer-order K_v(x) matches scipy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Integer-order
        K_v(x) matches scipy.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Callable[..., Any] = self.variant(bessel_kv)(v, x_arr)
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
        r"""Half-order special case works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Half-order special
        case works.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Callable[..., Any] = self.variant(bessel_kv)(v, x_arr)
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
        r"""Non-integer K_v(x) matches scipy within approximation tolerance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-integer K_v(x)
        matches scipy within approximation tolerance.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v``, ``x``,
        ``expected``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([x], dtype=jnp.float64)
        result: Callable[..., Any] = self.variant(bessel_kv)(v, x_arr)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_vectorized(self) -> None:
        r"""Vectorized x input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Vectorized x input
        works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 2.0, 5.0], dtype=jnp.float64
        )
        result: Callable[..., Any] = self.variant(bessel_kv)(0.0, x_arr)
        expected: Float[NDArray, "..."] = scipy_kv(
            0.0, np.array([0.5, 1.0, 2.0, 5.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_2d_input(self) -> None:
        r"""2D array input works.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2D array input
        works.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [[2.0, 3.0], [5.0, 10.0]], dtype=jnp.float64
        )
        result: Callable[..., Any] = self.variant(bessel_kv)(0.0, x_arr)
        expected: Float[NDArray, "..."] = scipy_kv(
            0.0, np.array([[2.0, 3.0], [5.0, 10.0]])
        )
        chex.assert_shape(result, (2, 2))
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_positive_values(self) -> None:
        r"""K_v(x) matches scipy for representative positive inputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_v(x) matches
        scipy for representative positive inputs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64
        )
        v: scalar_float
        for v in [0.0, 0.5, 1.0, 2.0]:
            result: Callable[..., Any] = self.variant(bessel_kv)(v, x_arr)
            expected: Float[NDArray, "..."] = scipy_kv(v, np.asarray(x_arr))
            chex.assert_trees_all_close(result, expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_monotonic_decay(self) -> None:
        r"""K_v(x) decays monotonically for fixed v.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_v(x) decays
        monotonically for fixed v.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array(
            [0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64
        )
        v: scalar_float
        for v in [0.0, 0.5, 1.0, 2.0]:
            result: Callable[..., Any] = self.variant(bessel_kv)(v, x_arr)
            i: int
            for i in range(len(result) - 1):
                assert result[i] > result[i + 1]

    @chex.variants(with_jit=True, without_jit=True)
    def test_bessel_kv_order_relation(self) -> None:
        r"""K_0(x) < K_1(x) at small x.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0(x) < K_1(x) at
        small x.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([0.5], dtype=jnp.float64)
        k0: Callable[..., Any] = self.variant(bessel_kv)(0.0, x_arr)[0]
        k1: Callable[..., Any] = self.variant(bessel_kv)(1.0, x_arr)[0]
        assert k0 < k1

    def test_bessel_kv_jit_compilation(self) -> None:
        r"""bessel_kv can be JIT compiled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: bessel_kv can be
        JIT compiled.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([1.0], dtype=jnp.float64)
        result: Callable[..., Any] = jax.jit(bessel_kv, static_argnums=(0,))(
            0.0, x_arr
        )
        expected: Float[Array, "..."] = scipy_kv(0, 1.0)
        chex.assert_trees_all_close(result[0], expected, rtol=1e-3)

    def test_bessel_kv_jit_works(self) -> None:
        r"""JIT works across representative orders.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT works across
        representative orders.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([5.0, 10.0], dtype=jnp.float64)
        jit_bessel: Callable[..., Any] = jax.jit(
            bessel_kv, static_argnums=(0,)
        )
        v: scalar_float
        for v in [0.0, 0.5, 1.0]:
            result: Float[Array, "..."] = jit_bessel(v, x_arr)
            expected: Float[NDArray, "..."] = scipy_kv(
                v, np.array([5.0, 10.0])
            )
            chex.assert_trees_all_close(result, expected, rtol=1e-3)

    def test_integer_orders_match_scipy_on_log_grid(self) -> None:
        r"""Integer K_v uses accurate K0/K1 recurrence seeds.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Integer K_v
        uses accurate K0/K1 recurrence seeds.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        x_grid: NDArray[np.float64] = np.logspace(-3.0, 2.0, 200)
        x_arr: Float[Array, "points"] = jnp.asarray(x_grid, dtype=jnp.float64)

        for order in (1, 2, 3, 5, 10, 20, 25):
            actual: NDArray[np.float64] = np.asarray(
                bessel_kv(float(order), x_arr)
            )
            expected: NDArray[np.float64] = scipy_kv(order, x_grid)
            max_rel: float = float(np.max(_relative_error(actual, expected)))
            assert max_rel < 1e-6

    def test_half_integer_orders_match_scipy_on_log_grid(self) -> None:
        r"""Half-integer K_v uses exact closed-form polynomials.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Half-
        integer K_v uses exact closed-form polynomials.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        x_grid: NDArray[np.float64] = np.logspace(-3.0, 2.0, 200)
        x_arr: Float[Array, "points"] = jnp.asarray(x_grid, dtype=jnp.float64)

        for order in (0.5, 1.5, 2.5):
            actual: NDArray[np.float64] = np.asarray(bessel_kv(order, x_arr))
            expected: NDArray[np.float64] = scipy_kv(order, x_grid)
            max_rel: float = float(np.max(_relative_error(actual, expected)))
            assert max_rel < 1e-12

    def test_non_integer_orders_match_scipy_with_documented_midrange(
        self,
    ) -> None:
        r"""Non-integer K_v respects the documented 2 < x < 10 bound.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-integer
        K_v respects the documented 2 < x < 10 bound.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        x_grid: NDArray[np.float64] = np.logspace(-3.0, 2.0, 240)
        x_arr: Float[Array, "points"] = jnp.asarray(x_grid, dtype=jnp.float64)
        midrange: NDArray[np.bool_] = (x_grid > 2.0) & (x_grid < 10.0)

        for order in (0.3, 1.7):
            actual: NDArray[np.float64] = np.asarray(bessel_kv(order, x_arr))
            expected: NDArray[np.float64] = scipy_kv(order, x_grid)
            rel_err: NDArray[np.float64] = _relative_error(actual, expected)
            assert float(np.max(rel_err[midrange])) < 1e-3
            assert float(np.max(rel_err[~midrange])) < 1e-6

    def test_order_zero_delegates_to_k0_kernel(self) -> None:
        r"""K_0 through bessel_kv equals the public K0 kernel.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K_0 through
        bessel_kv equals the public K0 kernel.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        x_arr: Float[Array, "points"] = jnp.asarray(
            [2.01, 3.0, 5.0],
            dtype=jnp.float64,
        )
        chex.assert_trees_all_close(
            bessel_kv(0.0, x_arr),
            bessel_k0(x_arr),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_nonpositive_domain_returns_inf(self) -> None:
        r"""K functions return inf at the real-domain singularity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: K functions
        return inf at the real-domain singularity.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        x_arr: Float[Array, "points"] = jnp.asarray(
            [0.0, -1.0],
            dtype=jnp.float64,
        )
        assert jnp.all(jnp.isinf(bessel_k0(x_arr)))
        assert jnp.all(jnp.isinf(bessel_k1(x_arr)))
        assert jnp.all(jnp.isinf(bessel_kv(0.3, x_arr)))


class TestBesselKvEdgeCases(chex.TestCase):
    """Test edge cases and numerical stability."""

    def test_bessel_kv_small_x_boundary(self) -> None:
        r"""Behavior near the small/large branch boundary stays stable.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Behavior near the
        small/large branch boundary stays stable.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_near_boundary: Float[Array, "..."] = jnp.array(
            [1.9, 2.0, 2.1], dtype=jnp.float64
        )
        result: Float[Array, "..."] = bessel_kv(0.0, x_near_boundary)
        expected: Float[NDArray, "..."] = scipy_kv(
            0, np.array([1.9, 2.0, 2.1])
        )
        chex.assert_trees_all_close(result, expected, rtol=5e-3)

    def test_bessel_kv_large_x(self) -> None:
        r"""Behavior for large x values stays accurate.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Behavior for large
        x values stays accurate.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_large: Float[Array, "..."] = jnp.array(
            [10.0, 20.0, 50.0], dtype=jnp.float64
        )
        result: Float[Array, "..."] = bessel_kv(0.0, x_large)
        expected: Float[NDArray, "..."] = scipy_kv(
            0, np.array([10.0, 20.0, 50.0])
        )
        chex.assert_trees_all_close(result, expected, rtol=1e-3)

    def test_bessel_kv_higher_integer_order(self) -> None:
        r"""Higher integer orders work in the large-x regime.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Higher integer
        orders work in the large-x regime.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_special``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        x_arr: Float[Array, "..."] = jnp.array([5.0, 10.0], dtype=jnp.float64)
        n: int
        for n in [3, 4, 5]:
            result: Float[Array, "..."] = bessel_kv(float(n), x_arr)
            expected: Float[NDArray, "..."] = scipy_kv(
                n, np.array([5.0, 10.0])
            )
            chex.assert_trees_all_close(result, expected, rtol=1e-2)
