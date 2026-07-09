"""Tests for recon/transforms.py.

Verifies the optimizer-coordinate bijectors used by reconstruction solvers:
positive values, bounded intervals, probability simplexes, and ordered bounded
coordinates, plus crystallographic lattice and Wyckoff constraints.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    bounded_from_unconstrained,
    fractional_from_unconstrained,
    lattice_from_unconstrained,
    ordered_bounded_from_unconstrained,
    positive_from_unconstrained,
    simplex_from_unconstrained,
    unconstrained_from_bounded,
    unconstrained_from_fractional,
    unconstrained_from_lattice,
    unconstrained_from_ordered_bounded,
    unconstrained_from_positive,
    unconstrained_from_simplex,
    wyckoff_fractional_from_unconstrained,
)


class TestReconTransforms(chex.TestCase):
    """Tests for reconstruction parameter transforms.

    :see: :func:`~rheedium.recon.positive_from_unconstrained`
    :see: :func:`~rheedium.recon.lattice_from_unconstrained`
    :see: :func:`~rheedium.recon.simplex_from_unconstrained`
    :see: :func:`~rheedium.recon.wyckoff_fractional_from_unconstrained`
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
        self.assertLess(float(ordered[-1]), 3.0)
        self.assertTrue(bool(jnp.all(jnp.diff(ordered) >= -1e-12)))

    def test_ordered_bounded_transform_round_trips_and_couples_outputs(
        self,
    ) -> None:
        r"""Ordered bounded transform should invert and stay fully coupled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Ordered
        bounded transform should invert and stay fully coupled.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        logits: Float[Array, "points"] = jnp.array(
            [-1.2, 0.4, 2.0, -0.3],
            dtype=jnp.float64,
        )
        ordered: Float[Array, "points"] = ordered_bounded_from_unconstrained(
            logits,
            -3.0,
            5.0,
        )
        recovered_logits: Float[Array, "points"] = (
            unconstrained_from_ordered_bounded(ordered, -3.0, 5.0)
        )
        chex.assert_trees_all_close(recovered_logits, logits, atol=1e-12)

        jacobian: Float[Array, "points points"] = jax.jacfwd(
            lambda z: ordered_bounded_from_unconstrained(z, -3.0, 5.0)
        )(logits)
        self.assertTrue(bool(jnp.all(jnp.abs(jacobian) > 1e-12)))
        self.assertTrue(bool(jnp.all(jnp.abs(jacobian[-1]) > 1e-12)))

    def test_fractional_lattice_and_wyckoff_transforms_are_smooth(
        self,
    ) -> None:
        r"""Crystallographic transforms should round-trip and differentiate.

        Extended Summary
        ----------------
        Verifies the lattice and Wyckoff-specific constraint maps added for
        structure inversion: fractional coordinates round-trip through logits,
        lattice lengths/angles round-trip through their physical bounds, and a
        constrained Wyckoff coordinate map has finite gradients.

        Notes
        -----
        The Wyckoff check evaluates away from unit-cell wrap discontinuities,
        which is the differentiable regime used by local optimizers.
        """
        fractional: Float[Array, "coords"] = jnp.array(
            [0.15, 0.4, 0.8],
            dtype=jnp.float64,
        )
        recovered_fractional: Float[Array, "coords"] = (
            fractional_from_unconstrained(
                unconstrained_from_fractional(fractional)
            )
        )
        chex.assert_trees_all_close(
            recovered_fractional,
            fractional,
            atol=1e-10,
        )

        lengths: Float[Array, "three"] = jnp.array(
            [3.1, 4.2, 5.3],
            dtype=jnp.float64,
        )
        angles: Float[Array, "three"] = jnp.array(
            [65.0, 90.0, 118.0],
            dtype=jnp.float64,
        )
        unconstrained_lengths: Float[Array, "three"]
        unconstrained_angles: Float[Array, "three"]
        unconstrained_lengths, unconstrained_angles = (
            unconstrained_from_lattice(
                lengths,
                angles,
                minimum_length=1.0,
                minimum_angle_deg=30.0,
                maximum_angle_deg=150.0,
            )
        )
        recovered_lengths: Float[Array, "three"]
        recovered_angles: Float[Array, "three"]
        recovered_lengths, recovered_angles = lattice_from_unconstrained(
            unconstrained_lengths,
            unconstrained_angles,
            minimum_length=1.0,
            minimum_angle_deg=30.0,
            maximum_angle_deg=150.0,
        )
        chex.assert_trees_all_close(recovered_lengths, lengths, atol=1e-10)
        chex.assert_trees_all_close(recovered_angles, angles, atol=1e-10)

        basis: Float[Array, "coords degrees"] = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float64,
        )
        offset: Float[Array, "coords"] = jnp.array(
            [0.05, 0.1, 0.2],
            dtype=jnp.float64,
        )
        degrees: Float[Array, "degrees"] = jnp.array(
            [-0.7, 0.3],
            dtype=jnp.float64,
        )
        wyckoff: Float[Array, "coords"] = (
            wyckoff_fractional_from_unconstrained(
                degrees,
                basis,
                offset,
            )
        )
        self.assertGreaterEqual(float(jnp.min(wyckoff)), 0.0)
        self.assertLess(float(jnp.max(wyckoff)), 1.0)

        gradient: Float[Array, "degrees"] = jax.grad(
            lambda trial_degrees: jnp.sum(
                wyckoff_fractional_from_unconstrained(
                    trial_degrees,
                    basis,
                    offset,
                )
                ** 2
            )
        )(degrees)
        chex.assert_tree_all_finite(gradient)


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
            recon.fractional_from_unconstrained,
            fractional_from_unconstrained,
        )
        self.assertIs(
            recon.lattice_from_unconstrained, lattice_from_unconstrained
        )
        self.assertIs(
            recon.simplex_from_unconstrained, simplex_from_unconstrained
        )
        self.assertIs(
            recon.wyckoff_fractional_from_unconstrained,
            wyckoff_fractional_from_unconstrained,
        )
        self.assertIs(
            recon.unconstrained_from_ordered_bounded,
            unconstrained_from_ordered_bounded,
        )
