"""Test suite for procs/grains.py."""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Complex, Float

from rheedium.procs.grains import (
    apply_misorientation_distribution,
    grain_distribution_average,
    grain_population_to_distribution,
)
from rheedium.simul.beam_averaging import apply_distribution
from rheedium.types import Distribution, ReductionMode
from rheedium.types.custom_types import scalar_float


def _public_function(path: Path, name: str) -> ast.FunctionDef:
    """Return a public function AST node by name."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


class TestR2InventoryGuards(chex.TestCase):
    """Guards for retired pattern-space averaging bodies."""

    repo_root = Path(__file__).parents[3]

    def test_retired_pattern_mixers_use_shared_reducer(self) -> None:
        """R2 retired pattern mixers should call the one reducer."""
        retired_mixers = {
            "src/rheedium/procs/grains.py": (
                "grain_distribution_average",
                "apply_misorientation_distribution",
            ),
            "src/rheedium/procs/surface_modifier.py": (
                "incoherent_domain_average",
            ),
        }
        reducer_calls: list[str] = []
        forbidden_weighted_broadcasts: list[str] = []
        for rel_path, function_names in retired_mixers.items():
            path = self.repo_root / rel_path
            source = path.read_text(encoding="utf-8")
            for function_name in function_names:
                function = _public_function(path, function_name)
                for node in ast.walk(function):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if (
                            isinstance(func, ast.Name)
                            and func.id == "apply_distributions"
                        ):
                            reducer_calls.append(f"{rel_path}:{function_name}")
                function_source = (
                    ast.get_source_segment(source, function) or ""
                )
                if "[:, None, None]" in function_source:
                    forbidden_weighted_broadcasts.append(
                        f"{rel_path}:{function_name}"
                    )

        self.assertEqual(
            reducer_calls,
            [
                "src/rheedium/procs/grains.py:grain_distribution_average",
                "src/rheedium/procs/surface_modifier.py:incoherent_domain_average",
            ],
        )
        self.assertEqual(forbidden_weighted_broadcasts, [])


class TestGrainDistributionAverage(chex.TestCase):
    """Tests for grain_distribution_average."""

    def test_computes_weighted_intensity_average(self) -> None:
        """Verify patterns are averaged weighted by grain fractions."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
                jnp.ones((2, 2)) * 5.0,
            ],
            axis=0,
        )
        result: Float[Array, "2 2"] = grain_distribution_average(
            patterns,
            jnp.array([0.2, 0.3, 0.5]),
        )

        chex.assert_trees_all_close(result, 3.6, atol=1e-6)

    def test_clips_negative_grain_weights(self) -> None:
        """Verify negative grain weights are clipped to zero."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
                jnp.ones((2, 2)) * 5.0,
            ],
            axis=0,
        )
        result: Float[Array, "2 2"] = grain_distribution_average(
            patterns,
            jnp.array([1.0, -2.0, 1.0]),
        )

        chex.assert_trees_all_close(result, 3.0, atol=1e-6)

    def test_grad_flows_through_grain_fraction(self) -> None:
        """Check gradients flow through the grain fraction weights."""
        patterns: Float[Array, "2 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )

        def objective(weight: scalar_float) -> scalar_float:
            return jnp.sum(
                grain_distribution_average(
                    patterns,
                    jnp.array([weight, 1.0]),
                )
            )

        grad_value: scalar_float = jax.grad(objective)(0.5)
        chex.assert_trees_all_close(
            float(grad_value),
            -8.0 / 2.25,
            atol=1e-6,
        )

    def test_jit_compiles(self) -> None:
        """Verify grain_distribution_average compiles under jit."""
        patterns: Float[Array, "2 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )
        compiled: Callable[..., Any] = jax.jit(
            lambda fractions: grain_distribution_average(patterns, fractions)
        )

        result: Float[Array, "2 2"] = compiled(jnp.array([1.0, 3.0]))
        chex.assert_trees_all_close(result, 2.5, atol=1e-6)

    def test_vmap_supports_batched_fraction_vectors(self) -> None:
        """Check the average maps over batched fraction vectors."""
        patterns: Float[Array, "2 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 3.0,
            ],
            axis=0,
        )

        def first_pixel(fractions: Float[Array, "2"]) -> scalar_float:
            return grain_distribution_average(patterns, fractions)[0, 0]

        batch: Float[Array, "3 2"] = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result: Float[Array, "3"] = jax.vmap(first_pixel)(batch)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 2.0, 3.0]),
            atol=1e-6,
        )


class TestGrainPopulationToDistribution(chex.TestCase):
    """Tests for grain_population_to_distribution."""

    def test_builds_incoherent_orientation_size_samples(self) -> None:
        """Verify grain metadata becomes an incoherent latent distribution."""
        distribution: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([-1.0, 2.0, 5.0]),
            grain_sizes_angstrom=jnp.array([80.0, 120.0, 160.0]),
            grain_volume_fractions=jnp.array([1.0, 2.0, 1.0]),
            axis_id="test_grains",
        )

        chex.assert_shape(distribution.samples, (3, 2))
        chex.assert_shape(distribution.weights, (3,))
        chex.assert_trees_all_close(
            distribution.samples,
            jnp.array(
                [
                    [-1.0, 80.0],
                    [2.0, 120.0],
                    [5.0, 160.0],
                ],
                dtype=jnp.float64,
            ),
        )
        chex.assert_trees_all_close(
            distribution.weights,
            jnp.array([0.25, 0.5, 0.25], dtype=jnp.float64),
        )
        assert distribution.reduction is ReductionMode.INCOHERENT
        assert distribution.axis_id == "test_grains"

    def test_matches_pattern_space_grain_average(self) -> None:
        """Verify generic Layer-1 reduction matches grain intensity mixing."""
        distribution: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([1.0, 3.0, 5.0]),
            grain_sizes_angstrom=jnp.array([50.0, 100.0, 150.0]),
            grain_volume_fractions=jnp.array([0.2, 0.3, 0.5]),
        )

        def _domain_pattern(
            sample: Float[Array, "2"],
        ) -> Float[Array, "2 2"]:
            angle_deg: Float[Array, ""] = sample[0]
            size_angstrom: Float[Array, ""] = sample[1]
            return jnp.array(
                [
                    [angle_deg + 0.01 * size_angstrom, angle_deg**2],
                    [0.001 * size_angstrom, 2.0 + 0.1 * angle_deg],
                ],
                dtype=jnp.float64,
            )

        domain_patterns: Float[Array, "3 2 2"] = jax.vmap(_domain_pattern)(
            distribution.samples
        )
        expected: Float[Array, "2 2"] = grain_distribution_average(
            domain_patterns,
            jnp.array([0.2, 0.3, 0.5]),
        )

        def _bound_amplitude(
            sample: Float[Array, "2"],
        ) -> Complex[Array, "2 2"]:
            return jnp.sqrt(_domain_pattern(sample)).astype(jnp.complex128)

        actual: Float[Array, "2 2"] = apply_distribution(
            distribution,
            _bound_amplitude,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_rejects_mismatched_grain_metadata_lengths(self) -> None:
        """Verify one-to-one grain metadata is required."""
        with pytest.raises(ValueError, match="share length"):
            grain_population_to_distribution(
                orientation_angles_deg=jnp.array([0.0, 1.0]),
                grain_sizes_angstrom=jnp.array([100.0]),
                grain_volume_fractions=jnp.array([0.4, 0.6]),
            )


class TestApplyMisorientationDistribution(chex.TestCase):
    """Tests for apply_misorientation_distribution."""

    def test_selects_patterns_near_distribution_center(self) -> None:
        """Verify a narrow width selects the centered pattern."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        result: Float[Array, "2 2"] = apply_misorientation_distribution(
            patterns,
            jnp.array([-1.0, 0.0, 1.0]),
            jnp.ones((3,)),
            0.0,
            0.05,
        )

        chex.assert_trees_all_close(result, 4.0, atol=1e-3)

    def test_broad_width_recovers_nearly_uniform_average(self) -> None:
        """Verify a broad width yields a near-uniform average."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        result: Float[Array, "2 2"] = apply_misorientation_distribution(
            patterns,
            jnp.array([-1.0, 0.0, 1.0]),
            jnp.ones((3,)),
            0.0,
            100.0,
        )

        chex.assert_trees_all_close(result, 14.0 / 3.0, atol=1e-4)

    def test_grad_flows_through_distribution_center(self) -> None:
        """Check gradients flow through the distribution center."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )

        def objective(mean_angle: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_misorientation_distribution(
                    patterns,
                    jnp.array([-1.0, 0.0, 1.0]),
                    jnp.ones((3,)),
                    mean_angle,
                    0.5,
                )
            )

        grad_value: scalar_float = jax.grad(objective)(0.0)
        assert np.isfinite(float(grad_value))
        assert float(grad_value) > 0.0

    def test_jit_compiles(self) -> None:
        """Verify apply_misorientation_distribution compiles under jit."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )
        compiled: Callable[..., Any] = jax.jit(
            lambda mean_angle: apply_misorientation_distribution(
                patterns,
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.ones((3,)),
                mean_angle,
                0.5,
            )
        )

        result: Float[Array, "2 2"] = compiled(0.0)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(np.asarray(result)))

    def test_vmap_supports_batched_distribution_centers(self) -> None:
        """Check the average maps over batched distribution centers."""
        patterns: Float[Array, "3 2 2"] = jnp.stack(
            [
                jnp.ones((2, 2)) * 1.0,
                jnp.ones((2, 2)) * 4.0,
                jnp.ones((2, 2)) * 9.0,
            ],
            axis=0,
        )

        def first_pixel(mean_angle: scalar_float) -> scalar_float:
            return apply_misorientation_distribution(
                patterns,
                jnp.array([-1.0, 0.0, 1.0]),
                jnp.ones((3,)),
                mean_angle,
                0.05,
            )[0, 0]

        result: Float[Array, "3"] = jax.vmap(first_pixel)(
            jnp.array([-1.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 4.0, 9.0]),
            atol=1e-3,
        )
