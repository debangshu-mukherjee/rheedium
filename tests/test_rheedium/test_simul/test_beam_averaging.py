"""Test suite for beam_averaging.py.

Verifies angular divergence averaging, energy spread averaging,
coherence envelope damping, detector PSF convolution, and the full
instrument-broadened pipeline. Includes gradient tests to ensure
end-to-end differentiability through all beam averaging operations.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float

from rheedium.simul import (
    angular_divergence_average,
    coherence_envelope,
    detector_psf_convolve,
    energy_spread_average,
    instrument_broadened_pattern,
)
from rheedium.tools import gauss_hermite_nodes_weights
from rheedium.types import scalar_float

H: int = 32
W: int = 32


def _dummy_angle_sim(
    polar_rad: scalar_float,
    azimuth_rad: scalar_float,
) -> Float[Array, "H W"]:
    """Simulate pattern that broadens with polar angle."""
    y: Float[Array, " H"] = jnp.linspace(-1.0, 1.0, H)
    x: Float[Array, " W"] = jnp.linspace(-1.0, 1.0, W)
    yy: Float[Array, "H W"]
    xx: Float[Array, "H W"]
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    sigma: scalar_float = 0.1 + polar_rad * 10.0
    return jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))


def _dummy_energy_sim(
    energy_kev: scalar_float,
) -> Float[Array, "H W"]:
    """Simulate pattern that shifts peak with energy."""
    y: Float[Array, " H"] = jnp.linspace(-1.0, 1.0, H)
    x: Float[Array, " W"] = jnp.linspace(-1.0, 1.0, W)
    yy: Float[Array, "H W"]
    xx: Float[Array, "H W"]
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    shift: scalar_float = (energy_kev - 20.0) * 0.01
    return jnp.exp(-((xx - shift) ** 2 + yy**2) / 0.02)


class TestGaussHermiteNodesWeights(chex.TestCase):
    """Tests for Gauss-Hermite quadrature computation."""

    def test_correct_count(self) -> None:
        """Returned arrays have the requested number of points."""
        for n in [3, 5, 7, 9]:
            nodes, weights = gauss_hermite_nodes_weights(n)
            chex.assert_shape(nodes, (n,))
            chex.assert_shape(weights, (n,))

    def test_weights_positive(self) -> None:
        """All Gauss-Hermite weights are positive."""
        nodes, weights = gauss_hermite_nodes_weights(7)
        self.assertTrue(jnp.all(weights > 0.0))

    def test_nodes_symmetric(self) -> None:
        """Nodes are symmetric about zero."""
        nodes, weights = gauss_hermite_nodes_weights(7)
        sorted_nodes: Float[Array, " N"] = jnp.sort(nodes)
        chex.assert_trees_all_close(
            sorted_nodes,
            -jnp.flip(sorted_nodes),
            atol=1e-12,
        )

    def test_weights_sum(self) -> None:
        """Weights sum to sqrt(pi) (Gauss-Hermite normalization)."""
        _, weights = gauss_hermite_nodes_weights(7)
        chex.assert_trees_all_close(
            jnp.sum(weights),
            jnp.sqrt(jnp.pi),
            atol=1e-12,
        )


class TestAngularDivergenceAverage(chex.TestCase):
    """Tests for angular divergence averaging."""

    def test_shape_preserved(self) -> None:
        """Output shape matches single-pattern shape."""
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self) -> None:
        """All pixels in the averaged pattern are non-negative."""
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_broader_than_single(self) -> None:
        """Averaged pattern is broader than single-angle pattern."""
        single: Float[Array, "H W"] = _dummy_angle_sim(
            jnp.float64(0.035), jnp.float64(0.0)
        )
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(1.0),
            n_quadrature_points=7,
        )
        single_row: Float[Array, " W"] = single[H // 2, :]
        avg_row: Float[Array, " W"] = avg[H // 2, :]
        single_half_max: scalar_float = jnp.max(single_row) / 2.0
        avg_half_max: scalar_float = jnp.max(avg_row) / 2.0
        single_fwhm: int = int(jnp.sum(single_row > single_half_max))
        avg_fwhm: int = int(jnp.sum(avg_row > avg_half_max))
        self.assertGreaterEqual(avg_fwhm, single_fwhm)

    def test_zero_divergence_matches_single(self) -> None:
        """Zero divergence reproduces the single-angle pattern."""
        single: Float[Array, "H W"] = _dummy_angle_sim(
            jnp.float64(0.035), jnp.float64(0.0)
        )
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.0),
            n_quadrature_points=5,
        )
        chex.assert_trees_all_close(avg, single, atol=1e-10)

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestEnergySpreadAverage(chex.TestCase):
    """Tests for energy spread averaging."""

    def test_shape_preserved(self) -> None:
        """Output shape matches single-energy pattern shape."""
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self) -> None:
        """All pixels in the averaged pattern are non-negative."""
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_shifts_streaks(self) -> None:
        """Different energies produce slightly different patterns."""
        pattern_low: Float[Array, "H W"] = _dummy_energy_sim(jnp.float64(19.5))
        pattern_high: Float[Array, "H W"] = _dummy_energy_sim(
            jnp.float64(20.5)
        )
        diff: scalar_float = jnp.max(jnp.abs(pattern_low - pattern_high))
        self.assertTrue(diff > 1e-6)

    def test_zero_spread_matches_single(self) -> None:
        """Zero energy spread reproduces the single-energy pattern."""
        single: Float[Array, "H W"] = _dummy_energy_sim(jnp.float64(20.0))
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.0),
            n_quadrature_points=5,
        )
        chex.assert_trees_all_close(avg, single, atol=1e-10)

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestCoherenceEnvelope(chex.TestCase):
    """Tests for coherence envelope damping."""

    def test_damps_high_q(self) -> None:
        """High-q amplitude is reduced more than low-q."""
        amp: Complex[Array, "H W"] = jnp.ones((H, W), dtype=jnp.complex128)
        q_par_low: Float[Array, "H W"] = jnp.full((H, W), 0.01)
        q_par_high: Float[Array, "H W"] = jnp.full((H, W), 1.0)
        q_z: Float[Array, "H W"] = jnp.zeros((H, W))
        damped_low: Complex[Array, "H W"] = coherence_envelope(
            amp,
            jnp.float64(500.0),
            jnp.float64(1000.0),
            q_par_low,
            q_z,
        )
        damped_high: Complex[Array, "H W"] = coherence_envelope(
            amp,
            jnp.float64(500.0),
            jnp.float64(1000.0),
            q_par_high,
            q_z,
        )
        self.assertTrue(jnp.abs(damped_low[0, 0]) > jnp.abs(damped_high[0, 0]))

    def test_zero_q_unchanged(self) -> None:
        """At q=0 the envelope is unity (no damping)."""
        amp: Complex[Array, "H W"] = jnp.ones((H, W), dtype=jnp.complex128) * (
            2.0 + 1.0j
        )
        q_par: Float[Array, "H W"] = jnp.zeros((H, W))
        q_z: Float[Array, "H W"] = jnp.zeros((H, W))
        damped: Complex[Array, "H W"] = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z
        )
        chex.assert_trees_all_close(damped, amp, atol=1e-14)

    def test_shape_preserved(self) -> None:
        """Output shape matches input amplitude shape."""
        amp: Complex[Array, "H W"] = jnp.ones((H, W), dtype=jnp.complex128)
        q_par: Float[Array, "H W"] = jnp.ones((H, W)) * 0.1
        q_z: Float[Array, "H W"] = jnp.ones((H, W)) * 0.05
        damped: Complex[Array, "H W"] = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z
        )
        chex.assert_shape(damped, (H, W))

    def test_longitudinal_damping(self) -> None:
        """Longitudinal coherence damps along q_z."""
        amp: Complex[Array, "H W"] = jnp.ones((H, W), dtype=jnp.complex128)
        q_par: Float[Array, "H W"] = jnp.zeros((H, W))
        q_z_low: Float[Array, "H W"] = jnp.full((H, W), 0.001)
        q_z_high: Float[Array, "H W"] = jnp.full((H, W), 0.1)
        damped_low: Complex[Array, "H W"] = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z_low
        )
        damped_high: Complex[Array, "H W"] = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z_high
        )
        self.assertTrue(jnp.abs(damped_low[0, 0]) > jnp.abs(damped_high[0, 0]))


class TestDetectorPsfConvolve(chex.TestCase):
    """Tests for detector PSF convolution."""

    def test_shape_preserved(self) -> None:
        """Output shape equals input shape."""
        img: Float[Array, "H W"] = jnp.ones((H, W))
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(1.0)
        )
        chex.assert_shape(blurred, (H, W))

    def test_zero_sigma_unchanged(self) -> None:
        """Zero PSF sigma leaves image unchanged."""
        img: Float[Array, "H W"] = jnp.eye(H, W)
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(0.0)
        )
        chex.assert_trees_all_close(blurred, img, atol=1e-12)

    def test_energy_conserved(self) -> None:
        """Total intensity is preserved to within 1%."""
        img: Float[Array, "H W"] = jnp.eye(H, W) * 100.0
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(1.5)
        )
        original_sum: scalar_float = jnp.sum(img)
        blurred_sum: scalar_float = jnp.sum(blurred)
        relative_error: scalar_float = (
            jnp.abs(blurred_sum - original_sum) / original_sum
        )
        self.assertTrue(relative_error < 0.01)

    def test_nonnegative(self) -> None:
        """Output is non-negative."""
        img: Float[Array, "H W"] = jnp.ones((H, W))
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(2.0)
        )
        self.assertTrue(jnp.all(blurred >= 0.0))

    def test_blurs_delta(self) -> None:
        """PSF spreads a delta function into a wider peak."""
        img: Float[Array, "H W"] = jnp.zeros((H, W))
        img = img.at[H // 2, W // 2].set(1.0)
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(2.0)
        )
        self.assertTrue(blurred[H // 2, W // 2] < 1.0)
        self.assertTrue(blurred[H // 2 + 1, W // 2] > 0.0)

    def test_finite_values(self) -> None:
        """No NaN or Inf in output."""
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 50.0
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(1.0)
        )
        chex.assert_tree_all_finite(blurred)


class TestInstrumentBroadenedPattern(chex.TestCase):
    """Tests for the full instrument-broadened pipeline."""

    def test_shape_preserved(self) -> None:
        """Output shape matches pattern shape."""
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_angle_fn=_dummy_angle_sim,
            simulate_energy_fn=_dummy_energy_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_shape(pattern, (H, W))

    def test_finite_values(self) -> None:
        """No NaN or Inf in final output."""
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_angle_fn=_dummy_angle_sim,
            simulate_energy_fn=_dummy_energy_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_tree_all_finite(pattern)

    def test_nonnegative(self) -> None:
        """All pixels in the final pattern are non-negative."""
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_angle_fn=_dummy_angle_sim,
            simulate_energy_fn=_dummy_energy_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        self.assertTrue(jnp.all(pattern >= 0.0))

    def test_jit_agrees(self) -> None:
        """JIT and non-JIT results agree to 1e-4."""
        kwargs = dict(
            simulate_angle_fn=_dummy_angle_sim,
            simulate_energy_fn=_dummy_energy_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
            n_angular_samples=5,
            n_energy_samples=3,
        )
        nojit: Float[Array, "H W"] = instrument_broadened_pattern(**kwargs)
        jitted: Float[Array, "H W"] = jax.jit(
            instrument_broadened_pattern,
            static_argnames=(
                "simulate_angle_fn",
                "simulate_energy_fn",
                "n_angular_samples",
                "n_energy_samples",
            ),
        )(**kwargs)
        chex.assert_trees_all_close(nojit, jitted, atol=1e-4)


class TestGradients(chex.TestCase):
    """Gradient tests for beam averaging functions."""

    def test_grad_through_angular_average(self) -> None:
        """jax.grad of sum(averaged_pattern) w.r.t. divergence is finite."""

        def loss(divergence: scalar_float) -> scalar_float:
            pattern: Float[Array, "H W"] = angular_divergence_average(
                simulate_fn=_dummy_angle_sim,
                nominal_polar_angle_rad=jnp.float64(0.035),
                nominal_azimuth_angle_rad=jnp.float64(0.0),
                angular_divergence_mrad=divergence,
                n_quadrature_points=5,
            )
            return jnp.sum(pattern)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_energy_average(self) -> None:
        """jax.grad of sum(averaged_pattern) w.r.t. spread is finite."""

        def loss(spread: scalar_float) -> scalar_float:
            pattern: Float[Array, "H W"] = energy_spread_average(
                simulate_fn=_dummy_energy_sim,
                nominal_energy_kev=jnp.float64(20.0),
                energy_spread_ev=spread,
                n_quadrature_points=5,
            )
            return jnp.sum(pattern)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_psf_convolve(self) -> None:
        """jax.grad of sum(convolved) w.r.t. psf_sigma is finite."""

        def loss(sigma: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W))
            blurred: Float[Array, "H W"] = detector_psf_convolve(img, sigma)
            return jnp.sum(blurred)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_coherence_envelope(self) -> None:
        """jax.grad of sum(|damped|^2) w.r.t. coherence length is finite."""

        def loss(l_t: scalar_float) -> scalar_float:
            amp: Complex[Array, "H W"] = jnp.ones((H, W), dtype=jnp.complex128)
            q_par: Float[Array, "H W"] = jnp.ones((H, W)) * 0.005
            q_z: Float[Array, "H W"] = jnp.ones((H, W)) * 0.002
            damped: Complex[Array, "H W"] = coherence_envelope(
                amp, l_t, jnp.float64(100.0), q_par, q_z
            )
            return jnp.sum(jnp.abs(damped) ** 2)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(50.0))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(
            jnp.abs(grad_val) > 1e-20,
            "Gradient through coherence envelope should be non-zero",
        )

    def test_grad_through_full_pipeline(self) -> None:
        """jax.grad flows through the full instrument pipeline."""

        def loss(divergence: scalar_float) -> scalar_float:
            pattern: Float[Array, "H W"] = instrument_broadened_pattern(
                simulate_angle_fn=_dummy_angle_sim,
                simulate_energy_fn=_dummy_energy_sim,
                nominal_polar_angle_rad=jnp.float64(0.035),
                nominal_azimuth_angle_rad=jnp.float64(0.0),
                nominal_energy_kev=jnp.float64(20.0),
                angular_divergence_mrad=divergence,
                energy_spread_ev=jnp.float64(0.5),
                psf_sigma_pixels=jnp.float64(1.0),
                n_angular_samples=5,
                n_energy_samples=3,
            )
            return jnp.sum(pattern)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(
            jnp.abs(grad_val) > 1e-20,
            "Gradient through full pipeline should be non-zero",
        )
