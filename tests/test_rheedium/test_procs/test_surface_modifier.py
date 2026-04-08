"""Test suite for procs/surface_modifier.py.

Tests vicinal surface step splitting and incoherent domain averaging.
"""

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from rheedium.procs.surface_modifier import (
    incoherent_domain_average,
    vicinal_surface_step_splitting,
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
