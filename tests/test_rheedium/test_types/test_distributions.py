"""Tests for orientation-distribution probability types and integration."""

import inspect
from collections.abc import Callable

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
    beam_modes_from_electron_beam,
    create_coherent_beam,
    create_discrete_orientation,
    create_distribution,
    create_electron_beam,
    create_field_emission_beam,
    create_gaussian_orientation,
    create_gaussian_schell_beam,
    create_lognormal_size,
    create_mixed_orientation,
    create_thermionic_beam,
    create_trivial_distribution,
    discretize_orientation,
    discretize_orientation_static,
    discretize_size_distribution,
    integrate_over_orientation,
    orientation_to_distribution,
    reduction_mode_from_coherence_length,
    size_to_distribution,
)
from rheedium.types.custom_types import scalar_float

from ..._assertions import assert_rejects


class TestDistributionFactories(chex.TestCase):
    """Tests for generic Distribution factory helpers.

    :see: :class:`~rheedium.types.Distribution`
    :see: :class:`~rheedium.types.ReductionMode`
    :see: :func:`~rheedium.types.create_distribution`
    :see: :func:`~rheedium.types.create_trivial_distribution`
    """

    def test_create_distribution_normalizes_weights(self) -> None:
        r"""Generic distribution weights are normalized.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Generic
        distribution weights are normalized.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""String reductions are canonicalized to ReductionMode.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: String reductions
        are canonicalized to ReductionMode.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = create_distribution(
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([0.5, 0.5]),
            reduction="coherent",
        )

        assert dist.reduction is ReductionMode.COHERENT

    def test_create_distribution_rejects_negative_weights(self) -> None:
        r"""Generic distribution weights must be non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Generic
        distribution weights must be non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_distribution,
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([1.0, -1.0]),
            match="weights must be non-negative",
        )

    def test_create_distribution_rejects_shape_mismatch(self) -> None:
        r"""Samples and weights must share a leading dimension.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Samples and
        weights must share a leading dimension.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_distribution,
            samples=jnp.array([[0.0], [1.0]]),
            weights=jnp.array([1.0, 2.0, 3.0]),
            match="samples and weights must share leading dimension",
        )

    def test_create_trivial_distribution_identity_sample(self) -> None:
        r"""Trivial distribution contains one zero sample with unit weight.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Trivial
        distribution contains one zero sample with unit weight.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = create_trivial_distribution(sample_dim=2)

        chex.assert_shape(dist.samples, (1, 2))
        chex.assert_trees_all_close(dist.samples, jnp.zeros((1, 2)))
        chex.assert_trees_all_close(dist.weights, jnp.ones((1,)))
        assert dist.reduction is ReductionMode.INCOHERENT

    def test_trivial_distribution_constant(self) -> None:
        r"""Module-level trivial distribution is the identity axis.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Module-level
        trivial distribution is the identity axis.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        chex.assert_shape(TRIVIAL_DISTRIBUTION.samples, (1, 1))
        chex.assert_trees_all_close(
            TRIVIAL_DISTRIBUTION.weights,
            jnp.ones((1,)),
        )

    def test_distribution_is_pytree(self) -> None:
        r"""Generic Distribution should flatten and unflatten cleanly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Generic
        Distribution should flatten and unflatten cleanly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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

    def test_distribution_bind_delegates_to_kernel_binder(self) -> None:
        r"""Distribution exposes the public producer-bind contract.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Distribution
        exposes the public producer-bind contract.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = create_distribution(
            samples=jnp.array([[1.0, 2.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="test_axis",
        )

        def _binder(
            distribution: Distribution,
        ) -> Callable[[Float[Array, "D"]], Float[Array, "D"]]:
            """Bind a test distribution to a sample-shifting closure."""
            assert distribution.axis_id == "test_axis"

            def _bound(sample: Float[Array, "D"]) -> Float[Array, "D"]:
                """Shift a sample by the bound distribution sample."""
                shifted_sample: Float[Array, "D"] = (
                    sample + distribution.samples[0]
                )
                return shifted_sample

            bound: Callable[[Float[Array, "D"]], Float[Array, "D"]] = _bound
            return bound

        bound = dist.bind(_binder)

        chex.assert_trees_all_close(
            bound(jnp.array([3.0, 4.0])),
            jnp.array([4.0, 6.0]),
        )


class TestCoherenceReduction(chex.TestCase):
    """Tests for coherence-length reduction selection.

    :see: :func:`~rheedium.types.reduction_mode_from_coherence_length`
    """

    def test_sub_coherence_feature_is_coherent(self) -> None:
        r"""Features inside the coherent footprint reduce coherently.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Features inside
        the coherent footprint reduce coherently.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        mode: ReductionMode = reduction_mode_from_coherence_length(
            feature_length_angstrom=50.0,
            coherence_length_angstrom=100.0,
        )

        assert mode is ReductionMode.COHERENT

    def test_super_coherence_feature_is_incoherent(self) -> None:
        r"""Features larger than the coherent footprint reduce incoherently.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Features larger
        than the coherent footprint reduce incoherently.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        mode: ReductionMode = reduction_mode_from_coherence_length(
            feature_length_angstrom=150.0,
            coherence_length_angstrom=100.0,
        )

        assert mode is ReductionMode.INCOHERENT

    def test_rejects_non_positive_feature_length(self) -> None:
        r"""Feature length must be positive.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Feature length
        must be positive.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            reduction_mode_from_coherence_length,
            feature_length_angstrom=0.0,
            coherence_length_angstrom=100.0,
            match="feature_length_angstrom must be positive",
        )


class TestBeamModeDistributionFactories(chex.TestCase):
    """Tests for Gaussian Schell-model beam producer factories.

    :see: :class:`~rheedium.types.BeamModeDistribution`
    :see: :func:`~rheedium.types.beam_modes_from_electron_beam`
    :see: :func:`~rheedium.types.create_coherent_beam`
    :see: :func:`~rheedium.types.create_field_emission_beam`
    :see: :func:`~rheedium.types.create_gaussian_schell_beam`
    :see: :func:`~rheedium.types.create_thermionic_beam`
    """

    def test_create_gaussian_schell_beam_validates_parameters(self) -> None:
        r"""GSM beam parameters are stored as scalar JAX arrays.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: GSM beam
        parameters are stored as scalar JAX arrays.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Coherent beam factory creates a sharp transverse source.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Coherent beam
        factory creates a sharp transverse source.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""GSM beta values must be in the half-open unit interval.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: GSM beta values
        must be in the half-open unit interval.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_gaussian_schell_beam,
            beta_in_plane=1.0,
            match="beta_in_plane must be finite and in",
        )

    def test_create_gaussian_schell_beam_rejects_negative_spread(self) -> None:
        r"""Beam divergences and energy spread are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Beam divergences
        and energy spread are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_gaussian_schell_beam,
            divergence_out_of_plane_rad=-1.0e-4,
            match="divergence_out_of_plane_rad",
        )

    def test_beam_modes_from_electron_beam_projects_grazing_footprint(
        self,
    ) -> None:
        r"""ElectronBeam bridge maps grazing footprint to anisotropic beta.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ElectronBeam
        bridge maps grazing footprint to anisotropic beta.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        beam = create_electron_beam(
            energy_spread_ev=0.4,
            angular_divergence_mrad=0.3,
            coherence_length_transverse_angstrom=1000.0,
            spot_size_um=jnp.array([50.0, 25.0]),
        )

        modes: BeamModeDistribution = beam_modes_from_electron_beam(
            beam,
            incidence_angle_deg=2.0,
            distribution_id="bridge",
        )

        self.assertGreater(
            float(modes.beta_in_plane),
            float(modes.beta_out_of_plane),
        )
        chex.assert_trees_all_close(
            modes.divergence_in_plane_rad,
            3.0e-4,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            modes.divergence_out_of_plane_rad,
            3.0e-4,
            atol=1e-12,
        )
        chex.assert_trees_all_close(modes.energy_spread_ev, 0.4, atol=1e-12)
        assert modes.distribution_id == "bridge"

    def test_beam_mode_presets_rank_source_coherence(self) -> None:
        r"""Thermionic preset is broader and more mixed than field emission.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Thermionic preset
        is broader and more mixed than field emission.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        field_emission: BeamModeDistribution = create_field_emission_beam()
        thermionic: BeamModeDistribution = create_thermionic_beam()

        self.assertGreater(
            float(thermionic.beta_out_of_plane),
            float(field_emission.beta_out_of_plane),
        )
        self.assertGreater(
            float(thermionic.energy_spread_ev),
            float(field_emission.energy_spread_ev),
        )
        self.assertGreater(
            float(thermionic.divergence_in_plane_rad),
            float(field_emission.divergence_in_plane_rad),
        )
        assert field_emission.distribution_id == "field_emission_beam"
        assert thermionic.distribution_id == "thermionic_beam"


class TestOrientationDistributionFactories(chex.TestCase):
    """Tests for OrientationDistribution factory helpers.

    :see: :class:`~rheedium.types.OrientationDistribution`
    :see: :func:`~rheedium.types.create_discrete_orientation`
    :see: :func:`~rheedium.types.create_gaussian_orientation`
    :see: :func:`~rheedium.types.create_mixed_orientation`
    """

    def test_create_discrete_orientation_defaults_equal_weights(self) -> None:
        r"""Discrete variants default to equal probability weights.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Discrete variants
        default to equal probability weights.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Factory weights must be valid probabilities.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Factory weights
        must be valid probabilities.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_mixed_orientation,
            match="weights must be non-negative",
            angles_deg=jnp.array([0.0, 90.0, 180.0]),
            weights=jnp.array([1.0, -2.0, 1.0]),
            mosaic_fwhm_deg=0.3,
        )

    def test_create_mixed_orientation_normalizes_weights(self) -> None:
        r"""Factory weights are normalized when valid.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Factory weights
        are normalized when valid.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Gaussian orientation uses one center peak plus mosaic width.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gaussian
        orientation uses one center peak plus mosaic width.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""OrientationDistribution should flatten and unflatten cleanly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        OrientationDistribution should flatten and unflatten cleanly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Factory angles must be finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Factory angles
        must be finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_discrete_orientation,
            jnp.array([0.0, jnp.nan]),
            match="angles_deg must be finite",
        )

    def test_create_gaussian_orientation_rejects_negative_fwhm(self) -> None:
        r"""Mosaic FWHM must be non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mosaic FWHM must
        be non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_gaussian_orientation,
            fwhm_deg=-0.1,
            match="mosaic_fwhm_deg must be non-negative",
        )

    def test_create_mixed_orientation_rejects_zero_weight_sum(self) -> None:
        r"""Factory weights must have positive total probability.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Factory weights
        must have positive total probability.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_mixed_orientation,
            angles_deg=jnp.array([0.0, 90.0]),
            weights=jnp.array([0.0, 0.0]),
            match="weights must have positive total probability",
        )


class TestSizeDistributionFactories(chex.TestCase):
    """Tests for size-distribution factory helpers."""

    def test_create_lognormal_size_valid(self) -> None:
        r"""Valid lognormal size parameters should be preserved.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Valid lognormal
        size parameters should be preserved.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Mean domain size must be positive.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mean domain size
        must be positive.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_lognormal_size,
            mean_ang=-1.0,
            match="mean_ang must be positive",
        )

    def test_create_lognormal_size_rejects_negative_sigma(self) -> None:
        r"""Size spread must be non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Size spread must
        be non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_lognormal_size,
            sigma_ang=-1.0,
            match="sigma_ang must be non-negative",
        )

    def test_create_lognormal_size_rejects_invalid_bounds(self) -> None:
        r"""Maximum size must exceed minimum size.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Maximum size must
        exceed minimum size.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert_rejects(
            create_lognormal_size,
            min_size_ang=100.0,
            max_size_ang=10.0,
            match="max_size_ang must be greater than min_size_ang",
        )


class TestOrientationDiscretization(chex.TestCase):
    """Tests for discretizing orientation probability distributions.

    :see: :func:`~rheedium.types.discretize_orientation_static`
    :see: :func:`~rheedium.types.discretize_orientation`
    """

    def test_discretize_orientation_has_honest_signature(self) -> None:
        r"""The public discretizer exposes only parameters it uses.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The public
        discretizer exposes only parameters it uses.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        signature = inspect.signature(discretize_orientation)
        assert "n_sigma_range" not in signature.parameters

    def test_discretize_orientation_returns_normalized_weights(self) -> None:
        r"""Quadrature weights remain a proper probability distribution.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Quadrature weights
        remain a proper probability distribution.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Avoid redundant quadrature for sharp discrete peaks.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Avoid redundant
        quadrature for sharp discrete peaks.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Normalize manual OrientationDistribution weights.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Normalize manual
        OrientationDistribution weights.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
    """Tests for converting orientation distributions to generic producers.

    :see: :func:`~rheedium.types.orientation_to_distribution`
    """

    def test_orientation_to_distribution_matches_static_support(self) -> None:
        r"""Sharp orientations map directly to one-column phi samples.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Sharp orientations
        map directly to one-column phi samples.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Mosaic orientation producer matches discretize_orientation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mosaic orientation
        producer matches discretize_orientation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
    """Tests for converting size distributions to generic producers.

    :see: :func:`~rheedium.types.discretize_size_distribution`
    :see: :func:`~rheedium.types.size_to_distribution`
    """

    def test_discretize_size_distribution_returns_normalized_weights(
        self,
    ) -> None:
        r"""Size quadrature weights remain normalized and positive.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Size quadrature
        weights remain normalized and positive.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Size producer emits one-column incoherent size samples.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Size producer
        emits one-column incoherent size samples.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Static delta size distributions keep one exact support point.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Static delta size
        distributions keep one exact support point.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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

    def test_exponential_size_quantiles_match_untruncated_moments(
        self,
    ) -> None:
        r"""Truncated exponential bins recover untruncated moments.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Truncated
        exponential bins recover untruncated moments.

        Notes
        -----
        It constructs the representative inputs inside the test body,
        keeping the fixture and assertion path local to the documented
        case.
        """
        mean_ang: float = 40.0
        dist = SizeDistribution(
            distribution_type="exponential",
            mean_ang=jnp.asarray(mean_ang, dtype=jnp.float64),
            sigma_ang=jnp.asarray(0.0, dtype=jnp.float64),
            min_size_ang=jnp.asarray(0.0, dtype=jnp.float64),
            max_size_ang=jnp.asarray(50.0 * mean_ang, dtype=jnp.float64),
        )

        sizes, weights = discretize_size_distribution(dist, n_points=64)
        observed_mean: Float[Array, ""] = jnp.sum(weights * sizes)
        observed_second: Float[Array, ""] = jnp.sum(weights * sizes * sizes)
        centered: Float[Array, "N"] = sizes - observed_mean
        observed_variance: Float[Array, ""] = jnp.sum(
            weights * centered * centered
        )
        observed_skew: Float[Array, ""] = (
            jnp.sum(weights * centered * centered * centered)
            / observed_variance**1.5
        )

        chex.assert_trees_all_close(
            observed_mean,
            mean_ang,
            rtol=5e-3,
        )
        chex.assert_trees_all_close(
            observed_second,
            2.0 * mean_ang * mean_ang,
            rtol=2e-2,
        )
        chex.assert_trees_all_close(observed_skew, 2.0, rtol=1e-1)


class TestProducerComposition(chex.TestCase):
    """Tests for composing real producer distributions."""

    def test_orientation_size_composition_matches_manual_sum(self) -> None:
        r"""Orientation and size producers compose as nested incoherent axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation and
        size producers compose as nested incoherent axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            """Return a constant amplitude for one orientation-size sample."""
            phi_deg: Float[Array, ""] = sample[0]
            size_ang: Float[Array, ""] = sample[1]
            amplitude: Float[Array, ""] = phi_deg + 0.01 * size_ang
            field: Complex[Array, "2 2"] = (
                jnp.ones((2, 2), dtype=jnp.complex128) * amplitude
            )
            return field

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
    """Tests for orientation-driven incoherent pattern integration.

    :see: :func:`~rheedium.types.integrate_over_orientation`
    """

    def test_integrate_over_orientation_computes_incoherent_sum(self) -> None:
        r"""The final pattern is the weighted intensity sum over variants.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The final pattern
        is the weighted intensity sum over variants.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Orientation integration remains differentiable in angle space.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation
        integration remains differentiable in angle space.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""Orientation integration should compile under jax.jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation
        integration should compile under jax.jit.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_distributions``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
