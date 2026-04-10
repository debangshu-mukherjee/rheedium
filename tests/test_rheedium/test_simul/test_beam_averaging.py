"""Test suite for beam_averaging.py.

Verifies angular divergence averaging, energy spread averaging,
coherence envelope damping, detector PSF convolution, and the full
instrument-broadened pipeline. Includes gradient tests to ensure
end-to-end differentiability through all beam averaging operations.
"""

import chex
import jax
import jax.numpy as jnp

from rheedium.simul.beam_averaging import (
    angular_divergence_average,
    coherence_envelope,
    detector_psf_convolve,
    energy_spread_average,
    instrument_broadened_pattern,
)

H = 32
W = 32


def _dummy_angle_sim(
    polar_rad,
    azimuth_rad,
):
    """Simulate pattern that broadens with polar angle."""
    y = jnp.linspace(-1.0, 1.0, H)
    x = jnp.linspace(-1.0, 1.0, W)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    sigma = 0.1 + polar_rad * 10.0
    return jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))


def _dummy_energy_sim(
    energy_kev,
):
    """Simulate pattern that shifts peak with energy."""
    y = jnp.linspace(-1.0, 1.0, H)
    x = jnp.linspace(-1.0, 1.0, W)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    shift = (energy_kev - 20.0) * 0.01
    return jnp.exp(-((xx - shift) ** 2 + yy**2) / 0.02)


def _dummy_joint_sim(
    polar_rad,
    azimuth_rad,
    energy_kev,
):
    """Simulate a pattern with coupled angular and energy dependence."""
    return _dummy_angle_sim(polar_rad, azimuth_rad) * _dummy_energy_sim(
        energy_kev
    )


class TestAngularDivergenceAverage(chex.TestCase):
    """Tests for angular divergence averaging."""

    def test_shape_preserved(self):
        """Output shape matches single-pattern shape."""
        avg = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self):
        """All pixels in the averaged pattern are non-negative."""
        avg = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_broader_than_single(self):
        """Averaged pattern is broader than single-angle pattern."""
        single = _dummy_angle_sim(jnp.float64(0.035), jnp.float64(0.0))
        avg = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(1.0),
            n_quadrature_points=7,
        )
        single_row = single[H // 2, :]
        avg_row = avg[H // 2, :]
        single_half_max = jnp.max(single_row) / 2.0
        avg_half_max = jnp.max(avg_row) / 2.0
        single_fwhm = int(jnp.sum(single_row > single_half_max))
        avg_fwhm = int(jnp.sum(avg_row > avg_half_max))
        self.assertGreaterEqual(avg_fwhm, single_fwhm)

    def test_zero_divergence_matches_single(self):
        """Zero divergence reproduces the single-angle pattern."""
        single = _dummy_angle_sim(jnp.float64(0.035), jnp.float64(0.0))
        avg = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.0),
            n_quadrature_points=5,
        )
        chex.assert_trees_all_close(avg, single, atol=1e-10)

    def test_finite_values(self):
        """No NaN or Inf in output."""
        avg = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestEnergySpreadAverage(chex.TestCase):
    """Tests for energy spread averaging."""

    def test_shape_preserved(self):
        """Output shape matches single-energy pattern shape."""
        avg = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self):
        """All pixels in the averaged pattern are non-negative."""
        avg = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_shifts_streaks(self):
        """Different energies produce slightly different patterns."""
        pattern_low = _dummy_energy_sim(jnp.float64(19.5))
        pattern_high = _dummy_energy_sim(jnp.float64(20.5))
        diff = jnp.max(jnp.abs(pattern_low - pattern_high))
        self.assertTrue(diff > 1e-6)

    def test_zero_spread_matches_single(self):
        """Zero energy spread reproduces the single-energy pattern."""
        single = _dummy_energy_sim(jnp.float64(20.0))
        avg = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.0),
            n_quadrature_points=5,
        )
        chex.assert_trees_all_close(avg, single, atol=1e-10)

    def test_finite_values(self):
        """No NaN or Inf in output."""
        avg = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestCoherenceEnvelope(chex.TestCase):
    """Tests for coherence envelope damping."""

    def test_damps_high_q(self):
        """High-q amplitude is reduced more than low-q."""
        amp = jnp.ones((H, W), dtype=jnp.complex128)
        q_par_low = jnp.full((H, W), 0.01)
        q_par_high = jnp.full((H, W), 1.0)
        q_z = jnp.zeros((H, W))
        damped_low = coherence_envelope(
            amp,
            jnp.float64(500.0),
            jnp.float64(1000.0),
            q_par_low,
            q_z,
        )
        damped_high = coherence_envelope(
            amp,
            jnp.float64(500.0),
            jnp.float64(1000.0),
            q_par_high,
            q_z,
        )
        self.assertTrue(jnp.abs(damped_low[0, 0]) > jnp.abs(damped_high[0, 0]))

    def test_zero_q_unchanged(self):
        """At q=0 the envelope is unity (no damping)."""
        amp = jnp.ones((H, W), dtype=jnp.complex128) * (2.0 + 1.0j)
        q_par = jnp.zeros((H, W))
        q_z = jnp.zeros((H, W))
        damped = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z
        )
        chex.assert_trees_all_close(damped, amp, atol=1e-14)

    def test_shape_preserved(self):
        """Output shape matches input amplitude shape."""
        amp = jnp.ones((H, W), dtype=jnp.complex128)
        q_par = jnp.ones((H, W)) * 0.1
        q_z = jnp.ones((H, W)) * 0.05
        damped = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z
        )
        chex.assert_shape(damped, (H, W))

    def test_longitudinal_damping(self):
        """Longitudinal coherence damps along q_z."""
        amp = jnp.ones((H, W), dtype=jnp.complex128)
        q_par = jnp.zeros((H, W))
        q_z_low = jnp.full((H, W), 0.001)
        q_z_high = jnp.full((H, W), 0.1)
        damped_low = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z_low
        )
        damped_high = coherence_envelope(
            amp, jnp.float64(500.0), jnp.float64(1000.0), q_par, q_z_high
        )
        self.assertTrue(jnp.abs(damped_low[0, 0]) > jnp.abs(damped_high[0, 0]))

    def test_longer_coherence_damps_less(self):
        """Longer coherence lengths preserve more high-q amplitude."""
        amp = jnp.ones((H, W), dtype=jnp.complex128)
        q_par = jnp.full((H, W), 0.2)
        q_z = jnp.full((H, W), 0.1)
        damped_short = coherence_envelope(
            amp, jnp.float64(10.0), jnp.float64(20.0), q_par, q_z
        )
        damped_long = coherence_envelope(
            amp, jnp.float64(1000.0), jnp.float64(2000.0), q_par, q_z
        )
        self.assertTrue(
            jnp.abs(damped_long[0, 0]) > jnp.abs(damped_short[0, 0])
        )


class TestDetectorPsfConvolve(chex.TestCase):
    """Tests for detector PSF convolution."""

    def test_shape_preserved(self):
        """Output shape equals input shape."""
        img = jnp.ones((H, W))
        blurred = detector_psf_convolve(img, jnp.float64(1.0))
        chex.assert_shape(blurred, (H, W))

    def test_zero_sigma_unchanged(self):
        """Zero PSF sigma leaves image unchanged."""
        img = jnp.eye(H, W)
        blurred = detector_psf_convolve(img, jnp.float64(0.0))
        chex.assert_trees_all_close(blurred, img, atol=1e-12)

    def test_energy_conserved(self):
        """Total intensity is preserved to within 1%."""
        img = jnp.eye(H, W) * 100.0
        blurred = detector_psf_convolve(img, jnp.float64(1.5))
        original_sum = jnp.sum(img)
        blurred_sum = jnp.sum(blurred)
        relative_error = jnp.abs(blurred_sum - original_sum) / original_sum
        self.assertTrue(relative_error < 0.01)

    def test_nonnegative(self):
        """Output is non-negative."""
        img = jnp.ones((H, W))
        blurred = detector_psf_convolve(img, jnp.float64(2.0))
        self.assertTrue(jnp.all(blurred >= 0.0))

    def test_blurs_delta(self):
        """PSF spreads a delta function into a wider peak."""
        img = jnp.zeros((H, W))
        img = img.at[H // 2, W // 2].set(1.0)
        blurred = detector_psf_convolve(img, jnp.float64(2.0))
        self.assertTrue(blurred[H // 2, W // 2] < 1.0)
        self.assertTrue(blurred[H // 2 + 1, W // 2] > 0.0)

    def test_finite_values(self):
        """No NaN or Inf in output."""
        img = jnp.ones((H, W)) * 50.0
        blurred = detector_psf_convolve(img, jnp.float64(1.0))
        chex.assert_tree_all_finite(blurred)


class TestInstrumentBroadenedPattern(chex.TestCase):
    """Tests for the full instrument-broadened pipeline."""

    def test_shape_preserved(self):
        """Output shape matches pattern shape."""
        pattern = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_shape(pattern, (H, W))

    def test_finite_values(self):
        """No NaN or Inf in final output."""
        pattern = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_tree_all_finite(pattern)

    def test_nonnegative(self):
        """All pixels in the final pattern are non-negative."""
        pattern = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        self.assertTrue(jnp.all(pattern >= 0.0))

    def test_jit_agrees(self):
        """JIT and non-JIT results agree to 1e-4."""
        kwargs = dict(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
            n_angular_samples=5,
            n_energy_samples=3,
        )
        nojit = instrument_broadened_pattern(**kwargs)
        jitted = jax.jit(
            instrument_broadened_pattern,
            static_argnames=(
                "simulate_fn",
                "n_angular_samples",
                "n_energy_samples",
            ),
        )(**kwargs)
        chex.assert_trees_all_close(nojit, jitted, atol=1e-4)


class TestGradients(chex.TestCase):
    """Gradient tests for beam averaging functions."""

    def test_grad_through_angular_average(self):
        """jax.grad of sum(averaged_pattern) w.r.t. divergence is finite."""

        def loss(divergence):
            pattern = angular_divergence_average(
                simulate_fn=_dummy_angle_sim,
                nominal_polar_angle_rad=jnp.float64(0.035),
                nominal_azimuth_angle_rad=jnp.float64(0.0),
                angular_divergence_mrad=divergence,
                n_quadrature_points=5,
            )
            return jnp.sum(pattern)

        grad_val = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_energy_average(self):
        """jax.grad of sum(averaged_pattern) w.r.t. spread is finite."""

        def loss(spread):
            pattern = energy_spread_average(
                simulate_fn=_dummy_energy_sim,
                nominal_energy_kev=jnp.float64(20.0),
                energy_spread_ev=spread,
                n_quadrature_points=5,
            )
            return jnp.sum(pattern)

        grad_val = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_psf_convolve(self):
        """jax.grad of sum(convolved) w.r.t. psf_sigma is finite."""

        def loss(sigma):
            img = jnp.ones((H, W))
            blurred = detector_psf_convolve(img, sigma)
            return jnp.sum(blurred)

        grad_val = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_coherence_envelope(self):
        """jax.grad of sum(|damped|^2) w.r.t. coherence length is finite."""

        def loss(l_t):
            amp = jnp.ones((H, W), dtype=jnp.complex128)
            q_par = jnp.ones((H, W)) * 0.005
            q_z = jnp.ones((H, W)) * 0.002
            damped = coherence_envelope(
                amp, l_t, jnp.float64(100.0), q_par, q_z
            )
            return jnp.sum(jnp.abs(damped) ** 2)

        grad_val = jax.grad(loss)(jnp.float64(50.0))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(
            jnp.abs(grad_val) > 1e-20,
            "Gradient through coherence envelope should be non-zero",
        )

    def test_grad_through_full_pipeline(self):
        """jax.grad flows through the full instrument pipeline."""

        def loss(divergence):
            pattern = instrument_broadened_pattern(
                simulate_fn=_dummy_joint_sim,
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

        grad_val = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(
            jnp.abs(grad_val) > 1e-20,
            "Gradient through full pipeline should be non-zero",
        )
