"""Tests for lattice vector conversion utilities."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized

from rheedium.inout.lattice import lattice_to_cell_params


class TestLatticeToCellParams(chex.TestCase):
    """Test lattice vector to cell parameter conversion."""

    def test_cubic_lattice(self):
        """Cubic: a=b=c, alpha=beta=gamma=90 deg."""
        a = 4.0
        lattice = jnp.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_orthorhombic_lattice(self):
        """Orthorhombic: a != b != c, alpha=beta=gamma=90 deg."""
        lattice = jnp.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]])
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 4.0, 5.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )

    def test_hexagonal_lattice(self):
        """Hexagonal: a=b != c, alpha=beta=90 deg, gamma=120 deg."""
        a = 3.0
        c = 5.0
        lattice = jnp.array(
            [[a, 0, 0], [-a / 2, a * jnp.sqrt(3) / 2, 0], [0, 0, c]]
        )
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, c]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 120.0]), atol=1e-6
        )

    def test_triclinic_lattice(self):
        """Triclinic: all angles and lengths different."""
        lattice = jnp.array(
            [[5.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.5, 6.0]]
        )
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths[0], 5.0, atol=1e-10)
        chex.assert_trees_all_close(lengths[1], jnp.sqrt(1 + 16), atol=1e-10)
        chex.assert_trees_all_close(
            lengths[2], jnp.sqrt(0.25 + 0.25 + 36), atol=1e-10
        )

        assert not jnp.allclose(angles[0], 90.0)
        assert not jnp.allclose(angles[1], 90.0)
        assert not jnp.allclose(angles[2], 90.0)

    def test_tetragonal_lattice(self):
        """Tetragonal: a=b != c, alpha=beta=gamma=90 deg."""
        a = 4.0
        c = 6.0
        lattice = jnp.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
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
    def test_various_cell_sizes(self, a):
        """Test with various cell sizes."""
        lattice = jnp.eye(3) * a
        lengths, angles = lattice_to_cell_params(lattice)

        chex.assert_trees_all_close(lengths, jnp.array([a, a, a]), atol=1e-10)
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-10
        )
