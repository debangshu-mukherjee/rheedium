"""Tests for lattice vector conversion utilities."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout.lattice import lattice_to_cell_params


class TestLatticeToCellParams(chex.TestCase):
    """Test lattice vector to cell parameter conversion.

    :see: :func:`~rheedium.inout.lattice_to_cell_params`
    """

    def test_cubic_lattice(self) -> None:
        r"""Cubic: a=b=c, alpha=beta=gamma=90 deg.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cubic: a=b=c,
        alpha=beta=gamma=90 deg.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 4.0
        lattice: Float[Array, "..."] = jnp.array(
            [[a, 0, 0], [0, a, 0], [0, 0, a]]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_orthorhombic_lattice(self) -> None:
        r"""Orthorhombic: a != b != c, alpha=beta=gamma=90 deg.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orthorhombic: a !=
        b != c, alpha=beta=gamma=90 deg.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lattice: Float[Array, "..."] = jnp.array(
            [[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 4.0, 5.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_hexagonal_lattice(self) -> None:
        r"""Hexagonal: a=b != c, alpha=beta=90 deg, gamma=120 deg.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Hexagonal: a=b !=
        c, alpha=beta=90 deg, gamma=120 deg.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 3.0
        c: float = 5.0
        lattice: Float[Array, "..."] = jnp.array(
            [[a, 0, 0], [-a / 2, a * jnp.sqrt(3) / 2, 0], [0, 0, c]]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, c]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 120.0]), atol=1e-6
        )

    def test_triclinic_lattice(self) -> None:
        r"""Triclinic: all angles and lengths different.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Triclinic: all
        angles and lengths different.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lattice: Float[Array, "..."] = jnp.array(
            [[5.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.5, 6.0]]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths[0], 5.0, atol=1e-10)
        chex.assert_trees_all_close(lengths[1], jnp.sqrt(1 + 16), atol=1e-10)
        chex.assert_trees_all_close(
            lengths[2], jnp.sqrt(0.25 + 0.25 + 36), atol=1e-10
        )

        assert not jnp.allclose(angles[0], 90.0)
        assert not jnp.allclose(angles[1], 90.0)
        assert not jnp.allclose(angles[2], 90.0)

    def test_tetragonal_lattice(self) -> None:
        r"""Tetragonal: a=b != c, alpha=beta=gamma=90 deg.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Tetragonal: a=b !=
        c, alpha=beta=gamma=90 deg.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 4.0
        c: float = 6.0
        lattice: Float[Array, "..."] = jnp.array(
            [[a, 0, 0], [0, a, 0], [0, 0, c]]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, c]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    @parameterized.named_parameters(
        ("small_cubic", 2.0),
        ("medium_cubic", 5.43),
        ("large_cubic", 10.0),
    )
    def test_various_cell_sizes(self, a: float) -> None:
        r"""Test with various cell sizes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with various cell
        sizes.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``a``, so the
        documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_lattice``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lattice: Float[Array, "..."] = jnp.eye(3) * a
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )
