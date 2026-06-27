"""Tests for recon/transforms.py.

Verifies the optimizer-coordinate bijectors used by reconstruction solvers:
positive values, bounded intervals, probability simplexes, and ordered bounded
coordinates.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    bounded_from_unconstrained,
    ordered_bounded_from_unconstrained,
    positive_from_unconstrained,
    simplex_from_unconstrained,
    unconstrained_from_bounded,
    unconstrained_from_positive,
    unconstrained_from_simplex,
)


class TestReconTransforms(chex.TestCase):
    """Tests for reconstruction parameter transforms.

    :see: :func:`~rheedium.recon.positive_from_unconstrained`
    :see: :func:`~rheedium.recon.simplex_from_unconstrained`
    """

    def test_positive_and_bounded_transforms_round_trip(self) -> None:
        r"""Positive and bounded transforms should recover physical inputs.

        Extended Summary
        ----------------
        Verifies that the inverse maps for positive and bounded parameters
        round-trip representative physical values to numerical precision.

        Notes
        -----
        It constructs small vectors away from singular interval endpoints and
        checks tolerance-aware equality after applying inverse then forward
        transforms.
        """
        positive_values: Float[Array, "values"] = jnp.array(
            [0.1, 1.5, 4.0],
            dtype=jnp.float64,
        )
        recovered_positive: Float[Array, "values"] = (
            positive_from_unconstrained(
                unconstrained_from_positive(positive_values)
            )
        )
        chex.assert_trees_all_close(
            recovered_positive,
            positive_values,
            atol=1e-10,
        )

        bounded_values: Float[Array, "values"] = jnp.array(
            [-0.75, 0.0, 0.8],
            dtype=jnp.float64,
        )
        recovered_bounded: Float[Array, "values"] = bounded_from_unconstrained(
            unconstrained_from_bounded(bounded_values, -1.0, 1.0),
            -1.0,
            1.0,
        )
        chex.assert_trees_all_close(
            recovered_bounded,
            bounded_values,
            atol=1e-10,
        )

    def test_simplex_transform_round_trips_and_has_finite_grad(self) -> None:
        r"""Simplex logits should round-trip and remain differentiable.

        Extended Summary
        ----------------
        Verifies that centered logit inversion recovers normalized weights and
        that gradients flow through the softmax transform.

        Notes
        -----
        It compares the round-tripped weights directly and evaluates
        ``jax.grad`` of a smooth simplex objective for finite entries.
        """
        weights: Float[Array, "weights"] = jnp.array(
            [0.2, 0.3, 0.5],
            dtype=jnp.float64,
        )
        logits: Float[Array, "weights"] = unconstrained_from_simplex(weights)
        recovered_weights: Float[Array, "weights"] = (
            simplex_from_unconstrained(logits)
        )
        chex.assert_trees_all_close(
            recovered_weights,
            weights,
            atol=1e-10,
        )

        gradient: Float[Array, "weights"] = jax.grad(
            lambda trial_logits: jnp.sum(
                simplex_from_unconstrained(trial_logits) ** 2
            )
        )(logits)
        chex.assert_tree_all_finite(gradient)

    def test_ordered_bounded_transform_is_monotone(self) -> None:
        r"""Ordered bounded transform should produce sorted interval points.

        Extended Summary
        ----------------
        Verifies that unconstrained segment logits become monotone coordinates
        inside the requested finite interval.

        Notes
        -----
        It checks lower/upper bounds and non-negative first differences for a
        representative vector.
        """
        logits: Float[Array, "points"] = jnp.array(
            [1.0, -0.5, 0.25, 2.0],
            dtype=jnp.float64,
        )
        ordered: Float[Array, "points"] = ordered_bounded_from_unconstrained(
            logits, -2.0, 3.0
        )

        self.assertGreaterEqual(float(jnp.min(ordered)), -2.0)
        self.assertLessEqual(float(jnp.max(ordered)), 3.0)
        self.assertTrue(bool(jnp.all(jnp.diff(ordered) >= -1e-12)))


class TestReconTransformNamespace(chex.TestCase):
    """Tests for public transform exports."""

    def test_namespace_exports_transform_entry_points(self) -> None:
        r"""Transform APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the transform helpers
        documented as public reconstruction APIs.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertIs(
            recon.positive_from_unconstrained,
            positive_from_unconstrained,
        )
        self.assertIs(
            recon.simplex_from_unconstrained, simplex_from_unconstrained
        )
