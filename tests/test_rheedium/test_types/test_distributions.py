"""Tests for orientation-distribution probability types and integration."""

import chex
import jax
import jax.numpy as jnp
from jax import tree_util
from jaxtyping import Array, Complex, Float

from rheedium.simul.beam_averaging import apply_distributions
from rheedium.types import (
    TRIVIAL_DISTRIBUTION,
    BeamModeDistribution,
    Distribution,
    OrientationDistribution,
    ReductionMode,
    SizeDistribution,
    create_coherent_beam,
    create_discrete_orientation,
    create_distribution,
    create_gaussian_orientation,
    create_gaussian_schell_beam,
    create_lognormal_size,
    create_mixed_orientation,
    create_trivial_distribution,
    discretize_orientation,
    discretize_orientation_static,
    discretize_size_distribution,
    integrate_over_orientation,
    orientation_to_distribution,
    size_to_distribution,
)
from rheedium.types.custom_types import scalar_float

from ..._assertions import assert_rejects


class TestDistributionFactories(chex.TestCase):
    """Tests for generic Distribution factory helpers."""

    def test_create_distribution_normalizes_weights(self) -> None:
        """Generic distribution weights are normalized."""
        dist: Distribution = create_distribution(
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([1.0, 3.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="beam",
        )

        assert isinstance(dist, Distribution)
        chex.assert_trees_all_close(
            dist.weights,
            jnp.array([0.25, 0.75]),
            atol=1e-12,
        )
        assert dist.reduction is ReductionMode.INCOHERENT
        assert dist.axis_id == "beam"

    def test_create_distribution_accepts_string_reduction(self) -> None:
        """String reductions are canonicalized to ReductionMode."""
        dist: Distribution = create_distribution(
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([0.5, 0.5]),
            reduction="coherent",
        )

        assert dist.reduction is ReductionMode.COHERENT

    def test_create_distribution_rejects_negative_weights(self) -> None:
        """Generic distribution weights must be non-negative."""
        assert_rejects(
            create_distribution,
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([1.0, -1.0]),
            match="weights must be non-negative",
        )

    def test_create_distribution_rejects_shape_mismatch(self) -> None:
        """Samples and weights must share a leading dimension."""
        assert_rejects(
            create_distribution,
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([1.0, 2.0, 3.0]),
            match="samples and weights must share leading dimension",
        )

    def test_create_trivial_distribution_identity_sample(self) -> None:
        """Trivial distribution contains one zero sample with unit weight."""
        dist: Distribution = create_trivial_distribution(sample_dim=2)

        chex.assert_shape(dist.samples, (1, 2))
        chex.assert_trees_all_close(dist.samples, jnp.zeros((1, 2)))
        chex.assert_trees_all_close(dist.weights, jnp.ones((1,)))
        assert dist.reduction is ReductionMode.INCOHERENT

    def test_trivial_distribution_constant(self) -> None:
        """Module-level trivial distribution is the identity axis."""
        chex.assert_shape(TRIVIAL_DISTRIBUTION.samples, (1, 1))
        chex.assert_trees_all_close(
            TRIVIAL_DISTRIBUTION.weights,
            jnp.ones((1,)),
        )

    def test_distribution_is_pytree(self) -> None:
        """Generic Distribution should flatten and unflatten cleanly."""
        dist: Distribution = create_distribution(
            samples=jnp.array([[1.0], [2.0]]),
            weights=jnp.array([0.4, 0.6]),
            reduction=ReductionMode.COHERENT,
            axis_id="coherent_axis",
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(dist)
        reconstructed: Distribution = treedef.unflatten(flat)

        assert isinstance(reconstructed, Distribution)
        chex.assert_trees_all_close(reconstructed.samples, dist.samples)
        chex.assert_trees_all_close(reconstructed.weights, dist.weights)
        assert reconstructed.reduction is ReductionMode.COHERENT
        assert reconstructed.axis_id == "coherent_axis"


class TestBeamModeDistributionFactories(chex.TestCase):
    """Tests for Gaussian Schell-model beam producer factories."""

    def test_create_gaussian_schell_beam_validates_parameters(self) -> None:
        """GSM beam parameters are stored as scalar JAX arrays."""
        beam: BeamModeDistribution = create_gaussian_schell_beam(
            beta_in_plane=0.25,
            beta_out_of_plane=0.5,
            divergence_in_plane_rad=2.0e-4,
            divergence_out_of_plane_rad=4.0e-4,
            energy_spread_ev=0.35,
            distribution_id="schottky",
        )

        assert isinstance(beam, BeamModeDistribution)
        chex.assert_trees_all_close(beam.beta_in_plane, 0.25, atol=1e-12)
        chex.assert_trees_all_close(beam.beta_out_of_plane, 0.5, atol=1e-12)
        chex.assert_trees_all_close(
            beam.divergence_in_plane_rad,
            2.0e-4,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            beam.divergence_out_of_plane_rad,
            4.0e-4,
            atol=1e-12,
        )
        chex.assert_trees_all_close(beam.energy_spread_ev, 0.35, atol=1e-12)
        assert beam.distribution_id == "schottky"

    def test_create_coherent_beam_collapses_transverse_spread(self) -> None:
        """Coherent beam factory creates a sharp transverse source."""
        beam: BeamModeDistribution = create_coherent_beam()

        chex.assert_trees_all_close(beam.beta_in_plane, 0.0, atol=1e-12)
        chex.assert_trees_all_close(beam.beta_out_of_plane, 0.0, atol=1e-12)
        chex.assert_trees_all_close(
            beam.divergence_in_plane_rad,
            0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            beam.divergence_out_of_plane_rad,
            0.0,
            atol=1e-12,
        )
        assert beam.distribution_id == "coherent_beam"

    def test_create_gaussian_schell_beam_rejects_invalid_beta(self) -> None:
        """GSM beta values must be in the half-open unit interval."""
        assert_rejects(
            create_gaussian_schell_beam,
            beta_in_plane=1.0,
            match="beta_in_plane must be finite and in",
        )

    def test_create_gaussian_schell_beam_rejects_negative_spread(self) -> None:
        """Beam divergences and energy spread are non-negative."""
        assert_rejects(
            create_gaussian_schell_beam,
            divergence_out_of_plane_rad=-1.0e-4,
            match="divergence_out_of_plane_rad",
        )


class TestOrientationDistributionFactories(chex.TestCase):
    """Tests for OrientationDistribution factory helpers."""

    def test_create_discrete_orientation_defaults_equal_weights(self) -> None:
        """Discrete variants default to equal probability weights."""
        dist: OrientationDistribution = create_discrete_orientation(
            jnp.array([33.7, -33.7])
        )

        assert isinstance(dist, OrientationDistribution)
        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.0, atol=1e-12)

    def test_create_mixed_orientation_rejects_negative_weights(self) -> None:
        """Factory weights must be valid probabilities."""
        assert_rejects(
            create_mixed_orientation,
            match="weights must be non-negative",
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, -2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

    def test_create_mixed_orientation_normalizes_weights(self) -> None:
        """Factory weights are normalized when valid."""
        dist: OrientationDistribution = create_mixed_orientation(
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, 2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([0.25, 0.5, 0.25]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.3, atol=1e-12)

    def test_create_gaussian_orientation_builds_single_peak(self) -> None:
        """Gaussian orientation uses one center peak plus mosaic width."""
        dist: OrientationDistribution = create_gaussian_orientation(
            center_deg=12.5, fwhm_deg=0.8
        )

        chex.assert_trees_all_close(
            dist.discrete_angles_deg,
            jnp.array([12.5]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            dist.discrete_weights,
            jnp.array([1.0]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(dist.mosaic_fwhm_deg, 0.8, atol=1e-12)

    def test_orientation_distribution_is_pytree(self) -> None:
        """OrientationDistribution should flatten and unflatten cleanly."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([10.0, -10.0]),
            weights=jnp.array([0.25, 0.75]),
            distribution_id="twins",
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(dist)
        reconstructed: OrientationDistribution = treedef.unflatten(flat)

        assert isinstance(reconstructed, OrientationDistribution)
        chex.assert_trees_all_close(
            reconstructed.discrete_angles_deg,
            dist.discrete_angles_deg,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            reconstructed.discrete_weights,
            dist.discrete_weights,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            reconstructed.mosaic_fwhm_deg,
            dist.mosaic_fwhm_deg,
            atol=1e-12,
        )
        assert reconstructed.distribution_id == "twins"

    def test_create_discrete_orientation_rejects_nan_angle(self) -> None:
        """Factory angles must be finite."""
        assert_rejects(
            create_discrete_orientation,
            jnp.array([0.0, jnp.nan]),
            match="angles_deg must be finite",
        )

    def test_create_gaussian_orientation_rejects_negative_fwhm(self) -> None:
        """Mosaic FWHM must be non-negative."""
        assert_rejects(
            create_gaussian_orientation,
            fwhm_deg=-0.1,
            match="mosaic_fwhm_deg must be non-negative",
        )

    def test_create_mixed_orientation_rejects_zero_weight_sum(self) -> None:
        """Factory weights must have positive total probability."""
        assert_rejects(
            create_mixed_orientation,
            angles_deg=jnp.array([0.0, 90.0]),
            weights=jnp.array([0.0, 0.0]),
            match="weights must have positive total probability",
        )


class TestSizeDistributionFactories(chex.TestCase):
    """Tests for size-distribution factory helpers."""

    def test_create_lognormal_size_valid(self) -> None:
        """Valid lognormal size parameters should be preserved."""
        dist: SizeDistribution = create_lognormal_size(
            mean_ang=100.0,
            sigma_ang=30.0,
            min_size_ang=10.0,
            max_size_ang=500.0,
        )

        assert isinstance(dist, SizeDistribution)
        chex.assert_trees_all_close(dist.mean_ang, 100.0)
        chex.assert_trees_all_close(dist.sigma_ang, 30.0)
        chex.assert_trees_all_close(dist.min_size_ang, 10.0)
        chex.assert_trees_all_close(dist.max_size_ang, 500.0)

    def test_create_lognormal_size_rejects_negative_mean(self) -> None:
        """Mean domain size must be positive."""
        assert_rejects(
            create_lognormal_size,
            mean_ang=-1.0,
            match="mean_ang must be positive",
        )

    def test_create_lognormal_size_rejects_negative_sigma(self) -> None:
        """Size spread must be non-negative."""
        assert_rejects(
            create_lognormal_size,
            sigma_ang=-1.0,
            match="sigma_ang must be non-negative",
        )

    def test_create_lognormal_size_rejects_invalid_bounds(self) -> None:
        """Maximum size must exceed minimum size."""
        assert_rejects(
            create_lognormal_size,
            min_size_ang=100.0,
            max_size_ang=10.0,
            match="max_size_ang must be greater than min_size_ang",
        )


class TestOrientationDiscretization(chex.TestCase):
    """Tests for discretizing orientation probability distributions."""

    def test_discretize_orientation_returns_normalized_weights(self) -> None:
        """Quadrature weights remain a proper probability distribution."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        angles_deg: Float[Array, "10"]
        weights: Float[Array, "10"]
        angles_deg, weights = discretize_orientation(dist, n_mosaic_points=5)

        chex.assert_shape(angles_deg, (10,))
        chex.assert_shape(weights, (10,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-12)
        chex.assert_trees_all_equal(jnp.all(weights >= 0.0), True)

    def test_discretize_orientation_static_returns_discrete_support(
        self,
    ) -> None:
        """Avoid redundant quadrature for sharp discrete peaks."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([15.0, -15.0]),
            weights=jnp.array([0.6, 0.4]),
        )

        angles_deg: Float[Array, "2"]
        weights: Float[Array, "2"]
        angles_deg, weights = discretize_orientation_static(
            dist, n_mosaic_points=7
        )

        chex.assert_shape(angles_deg, (2,))
        chex.assert_trees_all_close(
            angles_deg,
            jnp.array([15.0, -15.0]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            weights,
            jnp.array([0.6, 0.4]),
            atol=1e-12,
        )

    def test_discretize_orientation_static_normalizes_manual_weights(
        self,
    ) -> None:
        """Normalize manual OrientationDistribution weights."""
        dist: OrientationDistribution = OrientationDistribution(
            discrete_angles_deg=jnp.array([0.0, 90.0]),
            discrete_weights=jnp.array([0.0, 0.0]),
            mosaic_fwhm_deg=jnp.array(0.0),
            distribution_id=None,
        )

        angles_deg: Float[Array, "2"]
        weights: Float[Array, "2"]
        angles_deg, weights = discretize_orientation_static(dist)
        chex.assert_trees_all_close(
            weights,
            jnp.array([0.5, 0.5]),
            atol=1e-12,
        )


class TestOrientationProducer(chex.TestCase):
    """Tests for converting orientation distributions to generic producers."""

    def test_orientation_to_distribution_matches_static_support(self) -> None:
        """Sharp orientations map directly to one-column phi samples."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([15.0, -15.0]),
            weights=jnp.array([0.6, 0.4]),
            distribution_id="twins",
        )

        produced: Distribution = orientation_to_distribution(
            dist,
            base_phi_deg=2.0,
            use_static_discretization=True,
        )

        chex.assert_shape(produced.samples, (2, 1))
        chex.assert_trees_all_close(
            produced.samples[:, 0],
            jnp.array([17.0, -13.0]),
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            produced.weights,
            jnp.array([0.6, 0.4]),
            atol=1e-12,
        )
        assert produced.reduction is ReductionMode.INCOHERENT
        assert produced.axis_id == "twins"

    def test_orientation_to_distribution_matches_quadrature(self) -> None:
        """Mosaic orientation producer matches discretize_orientation."""
        dist: OrientationDistribution = create_gaussian_orientation(
            center_deg=1.5,
            fwhm_deg=0.2,
        )
        angles: Float[Array, "5"]
        weights: Float[Array, "5"]
        angles, weights = discretize_orientation(dist, n_mosaic_points=5)

        produced: Distribution = orientation_to_distribution(
            dist,
            n_mosaic_points=5,
        )

        chex.assert_trees_all_close(produced.samples[:, 0], angles)
        chex.assert_trees_all_close(produced.weights, weights)


class TestSizeProducer(chex.TestCase):
    """Tests for converting size distributions to generic producers."""

    def test_discretize_size_distribution_returns_normalized_weights(
        self,
    ) -> None:
        """Size quadrature weights remain normalized and positive."""
        dist: SizeDistribution = create_lognormal_size(
            mean_ang=100.0,
            sigma_ang=30.0,
            min_size_ang=10.0,
            max_size_ang=500.0,
        )

        sizes: Float[Array, "5"]
        weights: Float[Array, "5"]
        sizes, weights = discretize_size_distribution(dist, n_points=5)

        chex.assert_shape(sizes, (5,))
        chex.assert_shape(weights, (5,))
        chex.assert_tree_all_finite(sizes)
        chex.assert_trees_all_equal(jnp.all(sizes >= dist.min_size_ang), True)
        chex.assert_trees_all_equal(jnp.all(sizes <= dist.max_size_ang), True)
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-12)

    def test_size_to_distribution_is_incoherent_one_column_samples(
        self,
    ) -> None:
        """Size producer emits one-column incoherent size samples."""
        dist: SizeDistribution = create_lognormal_size(
            mean_ang=100.0,
            sigma_ang=20.0,
            min_size_ang=20.0,
            max_size_ang=300.0,
        )

        produced: Distribution = size_to_distribution(dist, n_points=3)

        chex.assert_shape(produced.samples, (3, 1))
        chex.assert_shape(produced.weights, (3,))
        chex.assert_trees_all_close(jnp.sum(produced.weights), 1.0)
        assert produced.reduction is ReductionMode.INCOHERENT
        assert produced.axis_id == "size"

    def test_delta_size_distribution_collapses_to_one_sample(self) -> None:
        """Static delta size distributions keep one exact support point."""
        dist = SizeDistribution(
            distribution_type="delta",
            mean_ang=jnp.array(75.0),
            sigma_ang=jnp.array(0.0),
            min_size_ang=jnp.array(10.0),
            max_size_ang=jnp.array(200.0),
        )

        sizes: Float[Array, "1"]
        weights: Float[Array, "1"]
        sizes, weights = discretize_size_distribution(dist, n_points=7)

        chex.assert_trees_all_close(sizes, jnp.array([75.0]))
        chex.assert_trees_all_close(weights, jnp.array([1.0]))


class TestProducerComposition(chex.TestCase):
    """Tests for composing real producer distributions."""

    def test_orientation_size_composition_matches_manual_sum(self) -> None:
        """Orientation and size producers compose as nested incoherent axes."""
        orientation = orientation_to_distribution(
            create_discrete_orientation(
                angles_deg=jnp.array([0.0, 10.0]),
                weights=jnp.array([0.25, 0.75]),
            ),
            use_static_discretization=True,
        )
        size = size_to_distribution(
            create_lognormal_size(
                mean_ang=100.0,
                sigma_ang=10.0,
                min_size_ang=50.0,
                max_size_ang=150.0,
            ),
            n_points=3,
        )

        def _bound(sample: Float[Array, "2"]) -> Complex[Array, "2 2"]:
            phi_deg: Float[Array, ""] = sample[0]
            size_ang: Float[Array, ""] = sample[1]
            amplitude: Float[Array, ""] = phi_deg + 0.01 * size_ang
            return jnp.ones((2, 2), dtype=jnp.complex128) * amplitude

        actual: Float[Array, "2 2"] = apply_distributions(
            [orientation, size],
            _bound,
        )
        manual: Float[Array, "2 2"] = jnp.zeros((2, 2), dtype=jnp.float64)
        for orientation_idx in range(orientation.samples.shape[0]):
            for size_idx in range(size.samples.shape[0]):
                sample = jnp.concatenate(
                    [
                        orientation.samples[orientation_idx],
                        size.samples[size_idx],
                    ]
                )
                weight = (
                    orientation.weights[orientation_idx]
                    * size.weights[size_idx]
                )
                manual = manual + weight * jnp.abs(_bound(sample)) ** 2

        chex.assert_trees_all_close(actual, manual, atol=1e-12)


class TestOrientationIntegration(chex.TestCase):
    """Tests for orientation-driven incoherent pattern integration."""

    def test_integrate_over_orientation_computes_incoherent_sum(self) -> None:
        """The final pattern is the weighted intensity sum over variants."""
        dist: OrientationDistribution = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        def simulate_fn(phi_deg: scalar_float) -> Float[Array, "2 2"]:
            return jnp.ones((2, 2), dtype=jnp.float64) * phi_deg**2

        pattern: Float[Array, "2 2"] = integrate_over_orientation(
            simulate_fn, dist, 5
        )
        chex.assert_trees_all_close(pattern, 75.0, atol=1e-6)

    def test_grad_flows_through_orientation_angle(self) -> None:
        """Orientation integration remains differentiable in angle space."""

        def loss(angle_deg: scalar_float) -> scalar_float:
            dist: OrientationDistribution = create_discrete_orientation(
                jnp.atleast_1d(angle_deg)
            )
            pattern: Float[Array, "2 2"] = integrate_over_orientation(
                lambda phi_deg: jnp.ones((2, 2), dtype=jnp.float64)
                * phi_deg**2,
                dist,
                3,
            )
            return jnp.sum(pattern)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(grad_value)
        chex.assert_trees_all_close(grad_value, 16.0, atol=1e-6)

    def test_jit_compiles_orientation_integration(self) -> None:
        """Orientation integration should compile under jax.jit."""

        @jax.jit
        def run(center_deg: scalar_float) -> Float[Array, "3 3"]:
            dist: OrientationDistribution = create_gaussian_orientation(
                center_deg=center_deg, fwhm_deg=0.0
            )
            return integrate_over_orientation(
                lambda phi_deg: jnp.ones((3, 3), dtype=jnp.float64)
                * (phi_deg + 1.0),
                dist,
                3,
            )

        pattern: Float[Array, "3 3"] = run(jnp.float64(4.0))
        chex.assert_shape(pattern, (3, 3))
        chex.assert_trees_all_close(pattern, 5.0, atol=1e-6)
