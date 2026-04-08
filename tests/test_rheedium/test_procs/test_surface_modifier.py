"""Test suite for procs/surface_modifier.py.

Tests vicinal surface step splitting and incoherent domain averaging.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from rheedium.procs.surface_modifier import (
    apply_step_edge_field,
    apply_surface_displacement_field,
    apply_surface_occupancy_field,
    incoherent_domain_average,
    vicinal_surface_step_splitting,
)
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.ucell.unitcell import build_cell_vectors


def _make_test_slab():
    """Build a small orthorhombic slab with two surface atoms."""
    cell_vectors = build_cell_vectors(2.0, 2.0, 6.0, 90.0, 90.0, 90.0)
    cart_positions = jnp.array(
        [
            [0.2, 0.2, 0.5, 14.0],
            [0.5, 0.4, 4.8, 14.0],
            [1.5, 1.6, 4.8, 8.0],
        ]
    )
    frac_positions = jnp.column_stack(
        [
            cart_positions[:, :3] @ jnp.linalg.inv(cell_vectors).T,
            cart_positions[:, 3],
        ]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([2.0, 2.0, 6.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestVicinalSurfaceStepSplitting(chex.TestCase, parameterized.TestCase):
    """Tests for vicinal_surface_step_splitting function."""

    def test_output_shape_matches_input(self):
        """Output should have same shape as q_z input."""
        q_z = jnp.linspace(0.0, 10.0, 100)
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        chex.assert_shape(result, (100,))

    def test_output_nonnegative(self):
        """All intensity values should be >= 0."""
        q_z = jnp.linspace(0.0, 10.0, 200)
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(jnp.min(result)) >= -1e-10

    def test_output_bounded_by_one(self):
        """Normalized intensity should not exceed 1."""
        q_z = jnp.linspace(0.0, 10.0, 200)
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(jnp.max(result)) <= 1.0 + 1e-10

    def test_antiphase_condition_dip(self):
        """At q_z * d = pi, intensity should show a minimum."""
        step_height = 2.0
        q_at_pi = jnp.pi / step_height
        q_z = jnp.array([0.0, q_at_pi])
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=step_height,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(result[1]) < float(result[0])

    def test_in_phase_condition_peak(self):
        """At q_z * d = 2*pi, intensity should be at maximum."""
        step_height = 2.0
        q_at_2pi = 2.0 * jnp.pi / step_height
        q_z = jnp.array([q_at_2pi])
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=step_height,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        chex.assert_trees_all_close(float(result[0]), 1.0, atol=1e-6)

    def test_wider_terraces_sharper_peaks(self):
        """Wider terraces should produce sharper (narrower) peaks."""
        q_z = jnp.linspace(0.0, 10.0, 1000)
        narrow = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=2.0,
            terrace_width_angstrom=20.0,
            q_z=q_z,
        )
        wide = vicinal_surface_step_splitting(
            hk_index=jnp.array([1, 0], dtype=jnp.int32),
            step_height_angstrom=2.0,
            terrace_width_angstrom=100.0,
            q_z=q_z,
        )
        narrow_mean = float(jnp.mean(narrow))
        wide_mean = float(jnp.mean(wide))
        assert wide_mean < narrow_mean

    def test_no_nan_or_inf(self):
        """Output should be finite everywhere."""
        q_z = jnp.linspace(0.0, 20.0, 500)
        result = vicinal_surface_step_splitting(
            hk_index=jnp.array([0, 0], dtype=jnp.int32),
            step_height_angstrom=3.0,
            terrace_width_angstrom=30.0,
            q_z=q_z,
        )
        chex.assert_tree_all_finite(result)


class TestApplySurfaceOccupancyField(chex.TestCase):
    """Tests for apply_surface_occupancy_field."""

    def test_scales_only_surface_region_atomic_numbers(self):
        slab = _make_test_slab()
        modified = apply_surface_occupancy_field(
            slab,
            0.8,
            jnp.array([0.1, 0.5, 0.25]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 7.0, 2.0]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.asarray(slab.cart_positions[:, :3]),
            atol=1e-6,
        )

    def test_clips_surface_occupancies(self):
        slab = _make_test_slab()
        modified = apply_surface_occupancy_field(
            slab,
            0.8,
            jnp.array([0.1, 2.0, -1.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 14.0, 0.0]),
            atol=1e-5,
        )

    def test_grad_flows_through_surface_occupancy(self):
        slab = _make_test_slab()

        def objective(occupancy):
            return jnp.sum(
                apply_surface_occupancy_field(
                    slab,
                    0.8,
                    jnp.array([1.0, occupancy, 1.0]),
                ).cart_positions[:, 3]
            )

        grad_value = jax.grad(objective)(0.5)
        chex.assert_trees_all_close(float(grad_value), 14.0, atol=1e-4)

    def test_jit_compiles(self):
        slab = _make_test_slab()
        compiled = jax.jit(
            lambda occupancy: apply_surface_occupancy_field(
                slab,
                0.8,
                jnp.array([1.0, occupancy, 1.0]),
            ).cart_positions[:, 3]
        )

        result = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 7.0, 8.0]),
            atol=1e-5,
        )

    def test_vmap_supports_batched_surface_occupancies(self):
        slab = _make_test_slab()

        def top_layer_weight(occupancy):
            return apply_surface_occupancy_field(
                slab,
                0.8,
                jnp.array([1.0, occupancy, 1.0]),
            ).cart_positions[1, 3]

        result = jax.vmap(top_layer_weight)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.0, 7.0, 14.0]),
            atol=1e-5,
        )


class TestApplySurfaceDisplacementField(chex.TestCase):
    """Tests for apply_surface_displacement_field."""

    def test_applies_displacements_only_near_surface(self):
        slab = _make_test_slab()
        modified = apply_surface_displacement_field(
            slab,
            0.8,
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, -0.2, 0.3],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.array(
                [
                    [0.2, 0.2, 0.5],
                    [0.6, 0.2, 5.1],
                    [1.5, 1.6, 4.8],
                ]
            ),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.asarray(slab.cart_positions[:, 3]),
            atol=1e-6,
        )

    def test_zero_displacement_field_is_identity(self):
        slab = _make_test_slab()
        modified = apply_surface_displacement_field(
            slab,
            0.8,
            jnp.zeros((3, 3)),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions),
            np.asarray(slab.cart_positions),
            atol=1e-6,
        )

    def test_grad_flows_through_surface_displacement(self):
        slab = _make_test_slab()

        def objective(delta_z):
            return apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, delta_z],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, 2]

        grad_value = jax.grad(objective)(0.3)
        chex.assert_trees_all_close(float(grad_value), 1.0, atol=1e-4)

    def test_jit_compiles(self):
        slab = _make_test_slab()
        compiled = jax.jit(
            lambda scale: apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1 * scale, -0.2 * scale, 0.3 * scale],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, :3]
        )

        result = compiled(1.0)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.6, 0.2, 5.1]),
            atol=1e-5,
        )

    def test_vmap_supports_batched_displacement_scales(self):
        slab = _make_test_slab()

        def top_atom_z(scale):
            return apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.3 * scale],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, 2]

        result = jax.vmap(top_atom_z)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([4.8, 4.95, 5.1]),
            atol=1e-5,
        )


class TestApplyStepEdgeField(chex.TestCase):
    """Tests for apply_step_edge_field."""

    def test_modulates_surface_heights_with_periodic_steps(self):
        slab = _make_test_slab()
        modified = apply_step_edge_field(
            slab,
            1.0,
            2.0,
            0.8,
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 2]),
            np.array([0.5, 5.3, 4.3]),
            atol=1e-3,
        )

    def test_zero_step_height_is_identity(self):
        slab = _make_test_slab()
        modified = apply_step_edge_field(
            slab,
            0.0,
            2.0,
            0.8,
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions),
            np.asarray(slab.cart_positions),
            atol=1e-6,
        )

    def test_grad_flows_through_step_height(self):
        slab = _make_test_slab()

        def objective(step_height):
            return apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[1, 2]

        grad_value = jax.grad(objective)(1.0)
        chex.assert_trees_all_close(float(grad_value), 0.5, atol=1e-3)

    def test_jit_compiles(self):
        slab = _make_test_slab()
        compiled = jax.jit(
            lambda step_height: apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[:, 2]
        )

        result = compiled(1.0)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.5, 5.3, 4.3]),
            atol=1e-3,
        )

    def test_vmap_supports_batched_step_heights(self):
        slab = _make_test_slab()

        def top_atom_z(step_height):
            return apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[1, 2]

        result = jax.vmap(top_atom_z)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([4.8, 5.05, 5.3]),
            atol=1e-3,
        )


class TestIncoherentDomainAverage(chex.TestCase, parameterized.TestCase):
    """Tests for incoherent_domain_average function."""

    def test_single_domain_unchanged(self):
        """Single domain with f=1 should return pattern unchanged."""
        pattern = jnp.ones((1, 8, 8)) * 5.0
        fractions = jnp.array([1.0])
        result = incoherent_domain_average(
            domain_patterns=pattern,
            domain_volume_fractions=fractions,
        )
        chex.assert_shape(result, (8, 8))
        chex.assert_trees_all_close(result, 5.0, atol=1e-6)

    def test_two_equal_domains_average(self):
        """50/50 mix should be the average of two patterns."""
        p1 = jnp.ones((8, 8)) * 2.0
        p2 = jnp.ones((8, 8)) * 6.0
        patterns = jnp.stack([p1, p2], axis=0)
        fractions = jnp.array([0.5, 0.5])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        expected = jnp.ones((8, 8)) * 4.0
        chex.assert_trees_all_close(result, expected, atol=1e-6)

    def test_output_shape(self):
        """Output should be (H, W) regardless of number of domains."""
        patterns = jnp.ones((3, 16, 32))
        fractions = jnp.array([0.5, 0.3, 0.2])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_shape(result, (16, 32))

    def test_output_nonnegative(self):
        """Result should be non-negative for non-negative inputs."""
        rng = np.random.default_rng(42)
        patterns = jnp.array(rng.uniform(0, 10, size=(4, 8, 8)))
        fractions = jnp.array([0.25, 0.25, 0.25, 0.25])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        assert float(jnp.min(result)) >= -1e-10

    def test_weighted_sum_correct(self):
        """Weighted sum should equal manual calculation."""
        p1 = jnp.ones((4, 4)) * 10.0
        p2 = jnp.ones((4, 4)) * 20.0
        patterns = jnp.stack([p1, p2], axis=0)
        fractions = jnp.array([0.3, 0.7])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        expected = 0.3 * 10.0 + 0.7 * 20.0
        chex.assert_trees_all_close(
            result,
            jnp.ones((4, 4)) * expected,
            atol=1e-6,
        )

    def test_fractions_auto_normalized(self):
        """Non-unit fractions should be auto-normalized."""
        p1 = jnp.ones((4, 4)) * 10.0
        patterns = jnp.expand_dims(p1, axis=0)
        fractions = jnp.array([2.0])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_trees_all_close(result, 10.0, atol=1e-6)

    def test_no_nan_or_inf(self):
        """Output should be finite."""
        rng = np.random.default_rng(0)
        patterns = jnp.array(rng.uniform(0, 100, size=(5, 8, 8)))
        fractions = jnp.array([0.1, 0.2, 0.3, 0.15, 0.25])
        result = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_tree_all_finite(result)
