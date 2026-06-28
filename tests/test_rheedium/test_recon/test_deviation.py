"""Tests for recon/deviation.py.

Verifies the recipe-deviation contract that turns fitted reconstruction
parameters into signed gaps, z-scores, and severity codes.
"""

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import rheedium.types as rh_types
from rheedium import recon
from rheedium.recon import (
    recipe_deviation,
    recipe_deviation_report_payload,
    recipe_deviation_report_schema,
    validate_recipe_deviation_report,
)
from rheedium.types import (
    RECIPE_DEVIATION_SCHEMA_VERSION,
    RecipeDeviationReport,
    ReconProblem,
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
    :see: :class:`~rheedium.types.RecipeDeviationReport`
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
        self.assertEqual(report.uncertainty_source, "supplied")
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
        chex.assert_trees_all_close(
            report.parameter_standard_deviation,
            jnp.array([0.1, 0.1], dtype=jnp.float64),
            atol=1e-12,
        )

    def test_recipe_deviation_uses_default_laplace_covariance(self) -> None:
        r"""Recipe deviation should default to K4 Laplace covariance.

        Extended Summary
        ----------------
        Verifies that K5 wires default K4 uncertainty into the report when no
        explicit parameter uncertainty is supplied, while still flagging the
        deliberately mismatched recipe with the correct sign and severity.

        Notes
        -----
        The linear synthetic inverse problem has an analytic Fisher covariance,
        so the default uncertainty source is deterministic and calibrated by
        the supplied noise variance.
        """
        true_params: Float[Array, "params"] = jnp.array(
            [1.4, -0.5],
            dtype=jnp.float64,
        )
        intended_params: Float[Array, "params"] = jnp.array(
            [1.0, -0.5],
            dtype=jnp.float64,
        )
        noise_sigma: Float[Array, ""] = jnp.asarray(0.05, dtype=jnp.float64)
        problem: ReconProblem = ReconProblem(
            forward=_forward,
            measured=_forward(true_params),
        )

        report: RecipeDeviationReport = recipe_deviation(
            problem=problem,
            intended_params=intended_params,
            initial_latent=jnp.zeros(2, dtype=jnp.float64),
            noise_variance=noise_sigma**2,
            uncertainty_regularization=1e-12,
            max_steps=32,
        )

        self.assertEqual(report.uncertainty_source, "laplace")
        chex.assert_trees_all_close(
            report.deviation,
            jnp.array([0.4, 0.0], dtype=jnp.float64),
            atol=1e-8,
        )
        self.assertGreater(float(report.z_score[0]), 3.0)
        self.assertEqual(int(report.severity), 2)
        chex.assert_tree_all_finite(report.parameter_covariance)

    def test_matched_recipe_stays_within_calibrated_noise(self) -> None:
        r"""Matched noisy recipes should not trigger a deviation warning.

        Extended Summary
        ----------------
        Verifies the K5 calibration half of the gate: when the measured pattern
        is generated from the intended recipe plus small noise, default K4
        covariance keeps z-scores below the warning threshold.

        Notes
        -----
        The fixed residual noise vector is intentionally well inside one sigma
        of the supplied detector noise model.
        """
        intended_params: Float[Array, "params"] = jnp.array(
            [1.0, -0.5],
            dtype=jnp.float64,
        )
        noise_sigma: Float[Array, ""] = jnp.asarray(0.05, dtype=jnp.float64)
        measured: Float[Array, "pixels"] = _forward(
            intended_params
        ) + jnp.array(
            [0.01, -0.015, 0.005],
            dtype=jnp.float64,
        )
        problem: ReconProblem = ReconProblem(
            forward=_forward,
            measured=measured,
        )

        report: RecipeDeviationReport = recipe_deviation(
            problem=problem,
            intended_params=intended_params,
            initial_latent=jnp.zeros(2, dtype=jnp.float64),
            noise_variance=noise_sigma**2,
            uncertainty_regularization=1e-12,
            max_steps=32,
        )

        self.assertLess(float(report.max_abs_z), 2.0)
        self.assertEqual(int(report.severity), 0)


class TestRecipeDeviationSchema(chex.TestCase):
    """Tests for the frozen recipe-deviation payload schema.

    :see: :func:`~rheedium.recon.recipe_deviation_report_payload`
    :see: :func:`~rheedium.recon.recipe_deviation_report_schema`
    :see: :func:`~rheedium.recon.validate_recipe_deviation_report`
    """

    def test_report_payload_validates_against_committed_schema(self) -> None:
        r"""Recipe-deviation payload should validate against its schema.

        Extended Summary
        ----------------
        Verifies the K5 schema-freeze gate: the report converts to a stable
        JSON-compatible payload whose version and required fields match the
        committed schema resource.

        Notes
        -----
        This is the shape the automatons should pin to when consuming
        recipe-deviation reports.
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

        payload: dict[str, object] = recipe_deviation_report_payload(report)
        schema: dict[str, object] = recipe_deviation_report_schema()
        validate_recipe_deviation_report(payload)

        self.assertEqual(
            payload["schema_version"], RECIPE_DEVIATION_SCHEMA_VERSION
        )
        self.assertEqual(schema["title"], "Rheedium Recipe Deviation Report")
        self.assertEqual(len(payload["parameters"]), 2)
        self.assertEqual(payload["uncertainty"]["source"], "supplied")

    def test_schema_validation_rejects_missing_required_keys(self) -> None:
        r"""Schema validation should reject incomplete payloads.

        Extended Summary
        ----------------
        Verifies that the lightweight validator fails loudly when an automaton
        payload omits a required top-level schema field.

        Notes
        -----
        The validator intentionally checks the committed schema's required key
        list without adding a runtime JSON-schema dependency.
        """
        invalid_payload: dict[str, object] = {
            "schema_version": RECIPE_DEVIATION_SCHEMA_VERSION,
        }

        with pytest.raises(ValueError, match="missing keys"):
            validate_recipe_deviation_report(invalid_payload)


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
        self.assertFalse(hasattr(recon, "RecipeDeviationReport"))
        self.assertFalse(hasattr(recon, "RECIPE_DEVIATION_SCHEMA_VERSION"))
        self.assertIs(rh_types.RecipeDeviationReport, RecipeDeviationReport)
        self.assertIs(
            rh_types.RECIPE_DEVIATION_SCHEMA_VERSION,
            RECIPE_DEVIATION_SCHEMA_VERSION,
        )
        self.assertIs(recon.recipe_deviation, recipe_deviation)
        self.assertIs(
            recon.recipe_deviation_report_payload,
            recipe_deviation_report_payload,
        )
        self.assertIs(
            recon.recipe_deviation_report_schema,
            recipe_deviation_report_schema,
        )
        self.assertIs(
            recon.validate_recipe_deviation_report,
            validate_recipe_deviation_report,
        )
