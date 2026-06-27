"""Tests for recon/uncertainty.py.

Verifies generalized Fisher-information and Laplace covariance helpers on a
small linear residual model with an analytic reference matrix.
"""

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium import recon
from rheedium.recon import (
    LaplaceUncertainty,
    covariance_from_fisher,
    fisher_information_from_residual,
    laplace_uncertainty,
)

_SENSITIVITY: Float[Array, "residuals params"] = jnp.array(
    [[2.0, 0.0], [1.0, -1.0], [0.5, 2.0]],
    dtype=jnp.float64,
)


def _linear_residual(
    params: Float[Array, "params"],
) -> Float[Array, "residuals"]:
    """Return a linear residual vector for uncertainty tests."""
    residual: Float[Array, "residuals"] = _SENSITIVITY @ params
    return residual


class TestReconUncertainty(chex.TestCase):
    """Tests for Fisher and Laplace reconstruction uncertainty.

    :see: :func:`~rheedium.recon.fisher_information_from_residual`
    :see: :func:`~rheedium.recon.laplace_uncertainty`
    """

    def test_fisher_information_matches_linear_reference(self) -> None:
        r"""Fisher information should match the analytic linear result.

        Extended Summary
        ----------------
        Verifies that residual Jacobians are flattened correctly for arbitrary
        parameter pytrees by comparing against ``J.T @ J / variance``.

        Notes
        -----
        It evaluates the Fisher matrix at a nonzero parameter vector and uses
        chex closeness checks against the analytic expression.
        """
        params: Float[Array, "params"] = jnp.array(
            [1.0, -0.25],
            dtype=jnp.float64,
        )
        noise_variance: Float[Array, ""] = jnp.asarray(
            0.5,
            dtype=jnp.float64,
        )

        fisher: Float[Array, "params params"] = (
            fisher_information_from_residual(
                residual_fn=_linear_residual,
                params=params,
                noise_variance=noise_variance,
            )
        )

        expected: Float[Array, "params params"] = (
            _SENSITIVITY.T @ _SENSITIVITY
        ) / noise_variance
        chex.assert_trees_all_close(fisher, expected, atol=1e-12)

    def test_laplace_uncertainty_returns_psd_covariance(self) -> None:
        r"""Laplace uncertainty should produce finite covariance summaries.

        Extended Summary
        ----------------
        Verifies that the covariance inversion and correlation normalization
        return finite, symmetric uncertainty summaries.

        Notes
        -----
        It checks covariance symmetry, non-negative eigenvalues, finite
        standard deviations, and unit diagonal correlation entries.
        """
        params: Float[Array, "params"] = jnp.array(
            [0.5, 0.25],
            dtype=jnp.float64,
        )

        uncertainty: LaplaceUncertainty = laplace_uncertainty(
            residual_fn=_linear_residual,
            params=params,
            regularization=1e-8,
        )

        chex.assert_tree_all_finite(uncertainty.covariance)
        chex.assert_trees_all_close(
            uncertainty.covariance,
            uncertainty.covariance.T,
            atol=1e-12,
        )
        self.assertTrue(
            bool(
                jnp.all(jnp.linalg.eigvalsh(uncertainty.covariance) >= -1e-10)
            )
        )
        chex.assert_tree_all_finite(uncertainty.standard_deviation)
        chex.assert_trees_all_close(
            jnp.diag(uncertainty.correlation),
            jnp.ones(2, dtype=jnp.float64),
            atol=1e-8,
        )


class TestReconUncertaintyNamespace(chex.TestCase):
    """Tests for public uncertainty exports."""

    def test_namespace_exports_uncertainty_entry_points(self) -> None:
        r"""Uncertainty APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the Fisher and
        Laplace helpers documented as public reconstruction APIs.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertIs(recon.LaplaceUncertainty, LaplaceUncertainty)
        self.assertIs(
            recon.fisher_information_from_residual,
            fisher_information_from_residual,
        )
        self.assertIs(recon.covariance_from_fisher, covariance_from_fisher)
        self.assertIs(recon.laplace_uncertainty, laplace_uncertainty)
