"""Tests for :mod:`rheedium.tools.safe_math`.

Extended Summary
----------------
Pins the repository-wide boundary-gradient conventions of the safe math
helpers: exact forward values, exact gradients in the interior, and zero
subgradients at the singular boundaries (never ``NaN``/``inf``).

Notes
-----
Every gradient assertion runs ``jax.grad`` directly at the boundary the
helper protects, because the classic double-``where`` failure only shows
up in the backward pass.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rheedium.tools import safe_arccos, safe_divide, safe_norm, safe_sqrt

jax.config.update("jax_enable_x64", True)


class TestSafeSqrt(chex.TestCase):
    """Tests for safe_sqrt boundary and interior behavior.

    :see: :func:`~rheedium.tools.safe_sqrt`
    """

    def test_interior_matches_sqrt_value_and_gradient(self) -> None:
        r"""Interior values and gradients match the bare square root.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: away from
        zero the helper is exactly ``jnp.sqrt`` in value and derivative.

        Notes
        -----
        It compares against closed-form values (``sqrt(4) = 2``,
        ``d/dx sqrt(x) = 1/(2 sqrt(x))``) rather than rheedium output.
        """
        chex.assert_trees_all_close(safe_sqrt(jnp.asarray(4.0)), 2.0)
        gradient = float(jax.grad(safe_sqrt)(jnp.asarray(4.0)))
        chex.assert_trees_all_close(gradient, 0.25)

    def test_zero_input_gives_zero_value_and_zero_gradient(self) -> None:
        r"""The zero boundary returns value 0.0 with subgradient 0.0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: at and below
        zero the forward value is 0.0 and the subgradient is exactly 0.0,
        never ``NaN``.

        Notes
        -----
        This is the double-``where`` regression the module exists to
        prevent; ``jax.grad`` of a naive ``where``-guarded sqrt is NaN.
        """
        for value in (0.0, -1.0):
            chex.assert_trees_all_close(safe_sqrt(jnp.asarray(value)), 0.0)
            gradient = float(jax.grad(safe_sqrt)(jnp.asarray(value)))
            chex.assert_trees_all_close(gradient, 0.0)


class TestSafeNorm(chex.TestCase):
    """Tests for safe_norm boundary and interior behavior.

    :see: :func:`~rheedium.tools.safe_norm`
    """

    def test_interior_matches_linalg_norm(self) -> None:
        r"""Nonzero vectors reproduce the Euclidean norm and its gradient.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: for a 3-4-5
        triangle the norm is 5 and the gradient is the unit vector.

        Notes
        -----
        Anchored on the closed-form gradient ``v / |v|``.
        """
        vector = jnp.asarray([3.0, 4.0])
        chex.assert_trees_all_close(safe_norm(vector), 5.0)
        gradient = np.asarray(jax.grad(safe_norm)(vector))
        np.testing.assert_allclose(gradient, [0.6, 0.8], rtol=1e-12)

    def test_zero_vector_gives_zero_value_and_zero_gradient(self) -> None:
        r"""The zero vector returns norm 0.0 with an all-zero subgradient.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the zero
        vector produces value 0.0 and subgradient (0, 0, 0), never NaN.

        Notes
        -----
        ``jnp.linalg.norm`` at the zero vector has a NaN gradient; this
        pins the helper's finite replacement convention.
        """
        zero = jnp.zeros(3)
        chex.assert_trees_all_close(safe_norm(zero), 0.0)
        gradient = np.asarray(jax.grad(safe_norm)(zero))
        np.testing.assert_array_equal(gradient, np.zeros(3))


class TestSafeArccos(chex.TestCase):
    """Tests for safe_arccos boundary and interior behavior.

    :see: :func:`~rheedium.tools.safe_arccos`
    """

    def test_interior_matches_arccos(self) -> None:
        r"""Interior cosines reproduce arccos values and gradients.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: at
        ``cos = 0`` the angle is pi/2 and the derivative is exactly -1.

        Notes
        -----
        Anchored on the closed form ``d/dx arccos(x) = -1/sqrt(1-x^2)``.
        """
        chex.assert_trees_all_close(
            safe_arccos(jnp.asarray(0.0)), jnp.pi / 2.0
        )
        gradient = float(jax.grad(safe_arccos)(jnp.asarray(0.0)))
        chex.assert_trees_all_close(gradient, -1.0)

    def test_edges_exact_forward_zero_subgradient(self) -> None:
        r"""The domain edges give exact angles with zero subgradients.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: at
        ``cos = +/-1`` (and slightly beyond, from floating-point noise)
        the forward angle is exactly 0 or pi and the subgradient is 0.0.

        Notes
        -----
        The true derivative diverges at the edges; the zero subgradient is
        the documented convention because the angle sits at a physical
        extremum there.
        """
        for cosine, angle in ((1.0, 0.0), (-1.0, jnp.pi), (1.0 + 1e-15, 0.0)):
            chex.assert_trees_all_close(
                safe_arccos(jnp.asarray(cosine)), angle
            )
            gradient = float(jax.grad(safe_arccos)(jnp.asarray(cosine)))
            assert np.isfinite(gradient)
            chex.assert_trees_all_close(gradient, 0.0)


class TestSafeDivide(chex.TestCase):
    """Tests for safe_divide boundary and interior behavior.

    :see: :func:`~rheedium.tools.safe_divide`
    """

    def test_interior_matches_division(self) -> None:
        r"""Ordinary denominators reproduce the exact quotient.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 6/3 = 2 with
        the exact quotient-rule gradient in the denominator.

        Notes
        -----
        Anchored on ``d/dd (n/d) = -n/d^2 = -2/3`` at n=6, d=3.
        """
        result = safe_divide(jnp.asarray(6.0), jnp.asarray(3.0))
        chex.assert_trees_all_close(result, 2.0)
        gradient = float(
            jax.grad(lambda d: safe_divide(jnp.asarray(6.0), d))(
                jnp.asarray(3.0)
            )
        )
        chex.assert_trees_all_close(gradient, -6.0 / 9.0)

    def test_small_denominators_are_sign_preserving_and_finite(self) -> None:
        r"""Tiny denominators are floored without flipping sign.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: denominators
        of 0.0 and -1e-15 produce +1/eps and -1/eps respectively, and the
        gradient at a zero denominator is finite.

        Notes
        -----
        An additive ``den + eps`` offset would return a positive result
        for the negative denominator; the sign-preserving floor must not.
        """
        up = float(safe_divide(jnp.asarray(1.0), jnp.asarray(0.0)))
        down = float(safe_divide(jnp.asarray(1.0), jnp.asarray(-1e-15)))
        assert up > 0.0
        assert down < 0.0
        chex.assert_trees_all_close(up, -down)
        gradient = float(
            jax.grad(lambda d: safe_divide(jnp.asarray(1.0), d))(
                jnp.asarray(0.0)
            )
        )
        assert np.isfinite(gradient)
