"""Tests for recon/deviation.py.

Verifies the recipe-deviation contract that turns fitted reconstruction
parameters into signed gaps, z-scores, and severity codes.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    RecipeDeviationReport,
    ReconProblem,
    recipe_deviation,
)

_DESIGN: Float[Array, "pixels params"] = jnp.array(
    [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
    dtype=jnp.float64,
)


def _forward(params: Float[Array, "params"]) -> Float[Array, "pixels"]:
    """Map recipe parameters to synthetic measurements."""
    pixels: Float[Array, "pixels"] = _DESIGN @ params
    return pixels


class TestRecipeDeviation(chex.TestCase):
    """Tests for fitted-minus-intended deviation reports.

    :see: :func:`~rheedium.recon.recipe_deviation`
    :see: :class:`~rheedium.recon.RecipeDeviationReport`
    """

    def test_recipe_deviation_flags_signed_mismatch(self) -> None:
        r"""Recipe deviation should report signed z-scored mismatches.

        Extended Summary
        ----------------
        Verifies that a planted mismatch between intended and true parameters
        is recovered with the right sign and critical severity.

        Notes
        -----
        It solves a small exact linear inverse problem and supplies explicit
        one-sigma parameter uncertainty to make the z-score deterministic.
        """
        true_params: Float[Array, "params"] = jnp.array(
            [1.5, -0.5],
            dtype=jnp.float64,
        )
        intended_params: Float[Array, "params"] = jnp.array(
            [1.0, -0.5],
            dtype=jnp.float64,
        )
        problem: ReconProblem = ReconProblem(
            forward=_forward,
            measured=_forward(true_params),
        )

        report: RecipeDeviationReport = recipe_deviation(
            problem=problem,
            intended_params=intended_params,
            initial_latent=jnp.zeros(2, dtype=jnp.float64),
            parameter_uncertainty=jnp.array([0.1, 0.1], dtype=jnp.float64),
            max_steps=32,
        )

        self.assertIsInstance(report, RecipeDeviationReport)
        chex.assert_trees_all_close(
            report.result.params,
            true_params,
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            report.deviation,
            jnp.array([0.5, 0.0], dtype=jnp.float64),
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            report.z_score,
            jnp.array([5.0, 0.0], dtype=jnp.float64),
            atol=1e-8,
        )
        self.assertEqual(int(report.severity), 2)


class TestRecipeDeviationNamespace(chex.TestCase):
    """Tests for public recipe-deviation exports."""

    def test_namespace_exports_recipe_deviation_entry_points(self) -> None:
        r"""Recipe-deviation APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the automaton-facing
        deviation report and entry point.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertIs(recon.RecipeDeviationReport, RecipeDeviationReport)
        self.assertIs(recon.recipe_deviation, recipe_deviation)
