"""Test suite for beam_averaging.py.

Verifies angular divergence averaging, energy spread averaging,
detector PSF convolution, and the full instrument-broadened pipeline.
Includes gradient tests to ensure end-to-end differentiability through
all beam averaging operations.
"""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float

from rheedium.simul.beam_averaging import (
    angular_divergence_average,
    apply_distribution,
    apply_distribution_intensity,
    apply_distributions,
    decompose_beam_modes,
    decompose_beam_modes_static,
    detector_psf_convolve,
    energy_spread_average,
    instrument_broadened_pattern,
)
from rheedium.types import (
    Distribution,
    ReductionMode,
    create_coherent_beam,
    create_distribution,
    create_gaussian_schell_beam,
)
from rheedium.types.custom_types import scalar_float

H: int = 32
W: int = 32


def _public_function(path: Path, name: str) -> ast.FunctionDef:
    """Return a public function AST node by name."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


class TestR2InventoryGuards(chex.TestCase):
    """Guards for retired instrument averaging bodies."""

    repo_root = Path(__file__).parents[3]

    def test_instrument_broadened_pattern_uses_shared_reducer(self) -> None:
        r"""R2 instrument quadrature should route through the one reducer.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: R2 instrument
        quadrature should route through the one reducer.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        path = self.repo_root / "src/rheedium/simul/beam_averaging.py"
        source = path.read_text(encoding="utf-8")
        function = _public_function(path, "instrument_broadened_pattern")
        function_source = ast.get_source_segment(source, function) or ""
        reducer_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_distributions"
        ]

        self.assertLen(reducer_calls, 1)
        self.assertNotIn("patterns:", function_source)
        self.assertNotIn("jnp.einsum", function_source)


def _linear_complex_field(sample: Float[Array, "D"]) -> Complex[Array, "H W"]:
    """Return a small complex field controlled by the sample value."""
    base: Complex[Array, "2 2"] = jnp.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [1.0 - 1.0j, -0.5 + 0.25j],
        ],
        dtype=jnp.complex128,
    )
    return sample[0] * base


class InstrumentBroadenedPatternKwargs(TypedDict):
    """Keyword arguments for instrument_broadened_pattern."""

    simulate_fn: Callable[
        [scalar_float, scalar_float, scalar_float], Float[Array, "H W"]
    ]
    nominal_polar_angle_rad: scalar_float
    nominal_azimuth_angle_rad: scalar_float
    nominal_energy_kev: scalar_float
    angular_divergence_mrad: scalar_float
    energy_spread_ev: scalar_float
    psf_sigma_pixels: scalar_float
    n_angular_samples: int
    n_energy_samples: int


def _weighted_variance(
    values: Float[Array, "N"],
    weights: Float[Array, "N"],
) -> Float[Array, ""]:
    """Return weighted variance for one distribution coordinate."""
    mean: Float[Array, ""] = jnp.sum(weights * values)
    return jnp.sum(weights * (values - mean) ** 2)


class TestBeamModeDecomposition(chex.TestCase):
    """Tests for Gaussian Schell-model beam-mode decomposition.

    :see: :func:`~rheedium.simul.decompose_beam_modes_static`
    :see: :func:`~rheedium.simul.decompose_beam_modes`
    """

    def test_decompose_beam_modes_normalizes_product_weights(self) -> None:
        r"""Beam modes flatten transverse and longitudinal products.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Beam modes flatten
        transverse and longitudinal products.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        beam = create_gaussian_schell_beam(
            beta_in_plane=0.25,
            beta_out_of_plane=0.5,
            divergence_in_plane_rad=2.0e-4,
            divergence_out_of_plane_rad=4.0e-4,
            energy_spread_ev=0.35,
            distribution_id="beam_test",
        )

        dist: Distribution = decompose_beam_modes(
            beam,
            n_modes_per_axis=4,
            n_modes_out_of_plane=3,
            n_energy_points=3,
        )

        chex.assert_shape(dist.samples, (36, 3))
        chex.assert_trees_all_close(jnp.sum(dist.weights), 1.0, atol=1e-12)
        assert dist.reduction is ReductionMode.INCOHERENT
        assert dist.axis_id == "beam_test"

    def test_decompose_beam_modes_matches_requested_variances(self) -> None:
        r"""Occupation-weighted spreads match the physical beam parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        Occupation-weighted spreads match the physical beam parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        theta_sigma: float = 3.0e-4
        phi_sigma: float = 6.0e-4
        energy_sigma: float = 0.4
        beam = create_gaussian_schell_beam(
            beta_in_plane=0.35,
            beta_out_of_plane=0.65,
            divergence_in_plane_rad=theta_sigma,
            divergence_out_of_plane_rad=phi_sigma,
            energy_spread_ev=energy_sigma,
        )

        dist: Distribution = decompose_beam_modes(
            beam,
            n_modes_per_axis=5,
            n_modes_out_of_plane=4,
            n_energy_points=3,
        )

        chex.assert_trees_all_close(
            _weighted_variance(dist.samples[:, 0], dist.weights),
            theta_sigma**2,
            rtol=1e-10,
            atol=1e-16,
        )
        chex.assert_trees_all_close(
            _weighted_variance(dist.samples[:, 1], dist.weights),
            phi_sigma**2,
            rtol=1e-10,
            atol=1e-16,
        )
        chex.assert_trees_all_close(
            _weighted_variance(dist.samples[:, 2], dist.weights),
            energy_sigma**2,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_decompose_beam_modes_static_collapses_coherent_limit(
        self,
    ) -> None:
        r"""Static decomposition prunes a sharp coherent beam to one mode.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Static
        decomposition prunes a sharp coherent beam to one mode.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        beam = create_coherent_beam()

        dist: Distribution = decompose_beam_modes_static(
            beam,
            n_modes_per_axis=8,
            n_energy_points=5,
        )

        chex.assert_shape(dist.samples, (1, 3))
        chex.assert_trees_all_close(dist.samples, jnp.zeros((1, 3)))
        chex.assert_trees_all_close(dist.weights, jnp.ones((1,)))

    def test_decompose_beam_modes_preserves_anisotropy(self) -> None:
        r"""Different axis divergences produce anisotropic sample variance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Different axis
        divergences produce anisotropic sample variance.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        beam = create_gaussian_schell_beam(
            beta_in_plane=0.4,
            beta_out_of_plane=0.4,
            divergence_in_plane_rad=8.0e-4,
            divergence_out_of_plane_rad=2.0e-4,
        )

        dist: Distribution = decompose_beam_modes(
            beam,
            n_modes_per_axis=4,
        )
        theta_variance: Float[Array, ""] = _weighted_variance(
            dist.samples[:, 0],
            dist.weights,
        )
        phi_variance: Float[Array, ""] = _weighted_variance(
            dist.samples[:, 1],
            dist.weights,
        )

        self.assertGreater(float(theta_variance), float(phi_variance))


class TestDistributionApply(chex.TestCase):
    """Tests for generic coherent/incoherent distribution reducers.

    :see: :func:`~rheedium.simul.apply_distribution`
    :see: :func:`~rheedium.simul.apply_distributions`
    """

    def test_apply_distribution_incoherent_matches_manual(self) -> None:
        r"""Incoherent reduction weights per-sample intensities.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Incoherent
        reduction weights per-sample intensities.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.25, 0.75]),
            reduction=ReductionMode.INCOHERENT,
        )

        actual: Float[Array, "2 2"] = apply_distribution(
            dist,
            _linear_complex_field,
        )
        fields: Complex[Array, "2 2 2"] = jax.vmap(_linear_complex_field)(
            dist.samples
        )
        expected: Float[Array, "2 2"] = jnp.einsum(
            "n,nhw->hw",
            dist.weights,
            jnp.abs(fields) ** 2,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_apply_distribution_intensity_incoherent_matches_manual(
        self,
    ) -> None:
        r"""Intensity reducer should weight intensities without amplitudes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Intensity
        reducer should weight intensities without amplitudes.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        dist = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.25, 0.75]),
            reduction=ReductionMode.INCOHERENT,
        )

        def intensity(sample: Float[Array, "1"]) -> Float[Array, "2 2"]:
            return jnp.ones((2, 2), dtype=jnp.float64) * sample[0] ** 2

        actual: Float[Array, "2 2"] = apply_distribution_intensity(
            dist,
            intensity,
        )

        chex.assert_trees_all_close(actual, 3.25, atol=1e-12)

    def test_apply_distribution_intensity_rejects_coherent_axes(self) -> None:
        r"""Intensity reducer is restricted to incoherent distributions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Intensity
        reducer is restricted to incoherent distributions.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        dist = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.25, 0.75]),
            reduction=ReductionMode.COHERENT,
        )

        with pytest.raises(ValueError, match="incoherent"):
            apply_distribution_intensity(
                dist,
                lambda sample: jnp.ones((2, 2), dtype=jnp.float64) * sample[0],
            )

    def test_apply_distribution_coherent_matches_manual(self) -> None:
        r"""Coherent reduction sums amplitudes before modulus squared.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Coherent reduction
        sums amplitudes before modulus squared.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.25, 0.75]),
            reduction=ReductionMode.COHERENT,
        )

        actual: Float[Array, "2 2"] = apply_distribution(
            dist,
            _linear_complex_field,
        )
        fields: Complex[Array, "2 2 2"] = jax.vmap(_linear_complex_field)(
            dist.samples
        )
        expected: Float[Array, "2 2"] = (
            jnp.abs(jnp.einsum("n,nhw->hw", dist.weights, fields)) ** 2
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_single_sample_reductions_coincide(self) -> None:
        r"""Coherent and incoherent reductions agree for one sample.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Coherent and
        incoherent reductions agree for one sample.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        coherent = create_distribution(
            samples=jnp.array([[2.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.COHERENT,
        )
        incoherent = create_distribution(
            samples=jnp.array([[2.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
        )

        coherent_image: Float[Array, "2 2"] = apply_distribution(
            coherent,
            _linear_complex_field,
        )
        incoherent_image: Float[Array, "2 2"] = apply_distribution(
            incoherent,
            _linear_complex_field,
        )

        chex.assert_trees_all_close(coherent_image, incoherent_image)

    def test_apply_distributions_matches_manual_nested_reduction(self) -> None:
        r"""Composed axes use coherent reduction inside incoherent averaging.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Composed axes use
        coherent reduction inside incoherent averaging.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        coherent = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.4, 0.6]),
            reduction=ReductionMode.COHERENT,
        )
        incoherent = create_distribution(
            samples=jnp.array([[10.0], [20.0]]),
            weights=jnp.array([0.25, 0.75]),
            reduction=ReductionMode.INCOHERENT,
        )

        def _bound(sample: Float[Array, "2"]) -> Complex[Array, "2 2"]:
            """Return a constant field for one composed beam sample."""
            field: Complex[Array, "2 2"] = (
                sample[0] + 0.1 * sample[1]
            ) * jnp.ones(
                (2, 2),
                dtype=jnp.complex128,
            )
            return field

        actual: Float[Array, "2 2"] = apply_distributions(
            [coherent, incoherent],
            _bound,
        )
        manual_terms: list[Float[Array, "2 2"]] = []
        for incoherent_idx in range(incoherent.samples.shape[0]):
            amplitudes: list[Complex[Array, "2 2"]] = []
            for coherent_idx in range(coherent.samples.shape[0]):
                sample = jnp.concatenate(
                    [
                        coherent.samples[coherent_idx],
                        incoherent.samples[incoherent_idx],
                    ]
                )
                amplitudes.append(
                    coherent.weights[coherent_idx] * _bound(sample)
                )
            coherent_sum: Complex[Array, "2 2"] = jnp.sum(
                jnp.stack(amplitudes),
                axis=0,
            )
            manual_terms.append(
                incoherent.weights[incoherent_idx] * jnp.abs(coherent_sum) ** 2
            )
        expected: Float[Array, "2 2"] = jnp.sum(
            jnp.stack(manual_terms),
            axis=0,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)


def _dummy_angle_sim(
    polar_rad: scalar_float,
    azimuth_rad: scalar_float,  # noqa: ARG001
) -> Float[Array, "H W"]:
    """Simulate pattern that broadens with polar angle."""
    y: Float[Array, "H"] = jnp.linspace(-1.0, 1.0, H)
    x: Float[Array, "W"] = jnp.linspace(-1.0, 1.0, W)
    yy: Float[Array, "H W"]
    xx: Float[Array, "H W"]
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    sigma: scalar_float = 0.1 + polar_rad * 10.0
    return jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))


def _dummy_energy_sim(
    energy_kev: scalar_float,
) -> Float[Array, "H W"]:
    """Simulate pattern that shifts peak with energy."""
    y: Float[Array, "H"] = jnp.linspace(-1.0, 1.0, H)
    x: Float[Array, "W"] = jnp.linspace(-1.0, 1.0, W)
    yy: Float[Array, "H W"]
    xx: Float[Array, "H W"]
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    shift: scalar_float = (energy_kev - 20.0) * 0.01
    return jnp.exp(-((xx - shift) ** 2 + yy**2) / 0.02)


def _dummy_joint_sim(
    polar_rad: scalar_float,
    azimuth_rad: scalar_float,
    energy_kev: scalar_float,
) -> Float[Array, "H W"]:
    """Simulate a pattern with coupled angular and energy dependence."""
    return _dummy_angle_sim(polar_rad, azimuth_rad) * _dummy_energy_sim(
        energy_kev
    )


class TestAngularDivergenceAverage(chex.TestCase):
    """Tests for angular divergence averaging.

    :see: :func:`~rheedium.simul.angular_divergence_average`
    """

    def test_shape_preserved(self) -> None:
        r"""Output shape matches single-pattern shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches single-pattern shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self) -> None:
        r"""All pixels in the averaged pattern are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All pixels in the
        averaged pattern are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_broader_than_single(self) -> None:
        r"""Averaged pattern is broader than single-angle pattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Averaged pattern
        is broader than single-angle pattern.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        single_row: Float[Array, "W"] = single[H // 2, :]
        avg_row: Float[Array, "W"] = avg[H // 2, :]
        single_half_max: scalar_float = jnp.max(single_row) / 2.0
        avg_half_max: scalar_float = jnp.max(avg_row) / 2.0
        single_fwhm: int = int(jnp.sum(single_row > single_half_max))
        avg_fwhm: int = int(jnp.sum(avg_row > avg_half_max))
        self.assertGreaterEqual(avg_fwhm, single_fwhm)

    def test_zero_divergence_matches_single(self) -> None:
        r"""Zero divergence reproduces the single-angle pattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero divergence
        reproduces the single-angle pattern.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""No NaN or Inf in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = angular_divergence_average(
            simulate_fn=_dummy_angle_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            angular_divergence_mrad=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestEnergySpreadAverage(chex.TestCase):
    """Tests for energy spread averaging.

    :see: :func:`~rheedium.simul.energy_spread_average`
    """

    def test_shape_preserved(self) -> None:
        r"""Output shape matches single-energy pattern shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches single-energy pattern shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
            n_quadrature_points=5,
        )
        chex.assert_shape(avg, (H, W))

    def test_nonnegative(self) -> None:
        r"""All pixels in the averaged pattern are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All pixels in the
        averaged pattern are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        self.assertTrue(jnp.all(avg >= 0.0))

    def test_shifts_streaks(self) -> None:
        r"""Different energies produce slightly different patterns.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Different energies
        produce slightly different patterns.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern_low: Float[Array, "H W"] = _dummy_energy_sim(jnp.float64(19.5))
        pattern_high: Float[Array, "H W"] = _dummy_energy_sim(
            jnp.float64(20.5)
        )
        diff: scalar_float = jnp.max(jnp.abs(pattern_low - pattern_high))
        self.assertTrue(diff > 1e-6)

    def test_zero_spread_matches_single(self) -> None:
        r"""Zero energy spread reproduces the single-energy pattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero energy spread
        reproduces the single-energy pattern.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        single: Float[Array, "H W"] = _dummy_energy_sim(jnp.float64(20.0))
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.0),
            n_quadrature_points=5,
        )
        chex.assert_trees_all_close(avg, single, atol=1e-10)

    def test_finite_values(self) -> None:
        r"""No NaN or Inf in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        avg: Float[Array, "H W"] = energy_spread_average(
            simulate_fn=_dummy_energy_sim,
            nominal_energy_kev=jnp.float64(20.0),
            energy_spread_ev=jnp.float64(0.5),
        )
        chex.assert_tree_all_finite(avg)


class TestDetectorPsfConvolve(chex.TestCase):
    """Tests for detector PSF convolution.

    :see: :func:`~rheedium.simul.detector_psf_convolve`
    """

    def test_shape_preserved(self) -> None:
        r"""Output shape equals input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        equals input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "H W"] = jnp.ones((H, W))
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(1.0)
        )
        chex.assert_shape(blurred, (H, W))

    def test_zero_sigma_unchanged(self) -> None:
        r"""Zero PSF sigma leaves image unchanged.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero PSF sigma
        leaves image unchanged.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "H W"] = jnp.eye(H, W)
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(0.0)
        )
        chex.assert_trees_all_close(blurred, img, atol=1e-12)

    def test_energy_conserved(self) -> None:
        r"""Total intensity is preserved to within 1%.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Total intensity is
        preserved to within 1%.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Output is non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output is
        non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "H W"] = jnp.ones((H, W))
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(2.0)
        )
        self.assertTrue(jnp.all(blurred >= 0.0))

    def test_blurs_delta(self) -> None:
        r"""PSF spreads a delta function into a wider peak.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PSF spreads a
        delta function into a wider peak.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "H W"] = jnp.zeros((H, W))
        img = img.at[H // 2, W // 2].set(1.0)
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(2.0)
        )
        self.assertTrue(blurred[H // 2, W // 2] < 1.0)
        self.assertTrue(blurred[H // 2 + 1, W // 2] > 0.0)

    def test_finite_values(self) -> None:
        r"""No NaN or Inf in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        img: Float[Array, "H W"] = jnp.ones((H, W)) * 50.0
        blurred: Float[Array, "H W"] = detector_psf_convolve(
            img, jnp.float64(1.0)
        )
        chex.assert_tree_all_finite(blurred)


class TestInstrumentBroadenedPattern(chex.TestCase):
    """Tests for the full instrument-broadened pipeline.

    :see: :func:`~rheedium.simul.instrument_broadened_pattern`
    """

    def test_shape_preserved(self) -> None:
        r"""Output shape matches pattern shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches pattern shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_shape(pattern, (H, W))

    def test_finite_values(self) -> None:
        r"""No NaN or Inf in final output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: No NaN or Inf in
        final output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        chex.assert_tree_all_finite(pattern)

    def test_nonnegative(self) -> None:
        r"""All pixels in the final pattern are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All pixels in the
        final pattern are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "H W"] = instrument_broadened_pattern(
            simulate_fn=_dummy_joint_sim,
            nominal_polar_angle_rad=jnp.float64(0.035),
            nominal_azimuth_angle_rad=jnp.float64(0.0),
            nominal_energy_kev=jnp.float64(20.0),
            angular_divergence_mrad=jnp.float64(0.5),
            energy_spread_ev=jnp.float64(0.5),
            psf_sigma_pixels=jnp.float64(1.0),
        )
        self.assertTrue(jnp.all(pattern >= 0.0))

    def test_jit_agrees(self) -> None:
        r"""JIT and non-JIT results agree to 1e-4.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT and non-JIT
        results agree to 1e-4.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: InstrumentBroadenedPatternKwargs = {
            "simulate_fn": _dummy_joint_sim,
            "nominal_polar_angle_rad": jnp.float64(0.035),
            "nominal_azimuth_angle_rad": jnp.float64(0.0),
            "nominal_energy_kev": jnp.float64(20.0),
            "angular_divergence_mrad": jnp.float64(0.5),
            "energy_spread_ev": jnp.float64(0.5),
            "psf_sigma_pixels": jnp.float64(1.0),
            "n_angular_samples": 5,
            "n_energy_samples": 3,
        }
        nojit: Float[Array, "H W"] = instrument_broadened_pattern(**kwargs)
        jitted: Float[Array, "H W"] = jax.jit(
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

    def test_grad_through_angular_average(self) -> None:
        r"""jax.grad of sum(averaged_pattern) w.r.t. divergence is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad of
        sum(averaged_pattern) w.r.t. divergence is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""jax.grad of sum(averaged_pattern) w.r.t. spread is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad of
        sum(averaged_pattern) w.r.t. spread is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""jax.grad of sum(convolved) w.r.t. psf_sigma is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad of
        sum(convolved) w.r.t. psf_sigma is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(sigma: scalar_float) -> scalar_float:
            img: Float[Array, "H W"] = jnp.ones((H, W))
            blurred: Float[Array, "H W"] = detector_psf_convolve(img, sigma)
            return jnp.sum(blurred)

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_val)

    def test_grad_through_full_pipeline(self) -> None:
        r"""jax.grad flows through the full instrument pipeline.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad flows
        through the full instrument pipeline.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_beam_averaging``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(divergence: scalar_float) -> scalar_float:
            pattern: Float[Array, "H W"] = instrument_broadened_pattern(
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

        grad_val: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(grad_val)
        self.assertTrue(
            jnp.abs(grad_val) > 1e-20,
            "Gradient through full pipeline should be non-zero",
        )
