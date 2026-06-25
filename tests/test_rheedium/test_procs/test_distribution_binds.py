"""Tests for producer-owned Distribution bind helpers."""

from typing import Any

import chex
import jax.numpy as jnp
import pytest

from rheedium.procs.distribution_binds import (
    bind_kinematic_axis_distribution,
    bind_multislice_axis_distribution,
)
from rheedium.types import ReductionMode, create_distribution


def _unused_structure_builder(_sample: Any) -> None:
    """Provide a placeholder builder for non-structure bind tests."""


class TestDistributionBindHelpers(chex.TestCase):
    """Producer-owned bind semantics for generic Distribution axes."""

    def test_kinematic_beam_axis_maps_sample_to_geometry_deltas(self) -> None:
        """Beam-like axes perturb theta, phi, and voltage."""
        distribution = create_distribution(
            samples=jnp.array([[1.0e-3, -2.0e-3, 4.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="legacy_instrument",
        )

        bound = distribution.bind(
            lambda dist: bind_kinematic_axis_distribution(
                dist,
                twin_builder=_unused_structure_builder,
                step_builder=_unused_structure_builder,
            )
        )
        update = bound(distribution.samples[0])

        assert update.crystal is None
        chex.assert_trees_all_close(update.voltage_delta_kv, 4.0e-3)
        chex.assert_trees_all_close(update.theta_delta_deg, jnp.rad2deg(1e-3))
        chex.assert_trees_all_close(update.phi_delta_deg, jnp.rad2deg(-2e-3))
        assert update.domain_size_angstrom is None

    def test_kinematic_grain_axis_carries_orientation_and_size(self) -> None:
        """Grain axes expose both orientation and finite-domain size."""
        distribution = create_distribution(
            samples=jnp.array([[5.0, 80.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="grains",
        )

        bound = distribution.bind(
            lambda dist: bind_kinematic_axis_distribution(
                dist,
                twin_builder=_unused_structure_builder,
                step_builder=_unused_structure_builder,
            )
        )
        update = bound(distribution.samples[0])

        chex.assert_trees_all_close(update.phi_delta_deg, 5.0)
        chex.assert_trees_all_close(update.domain_size_angstrom, 80.0)

    def test_multislice_size_axis_carries_domain_size(self) -> None:
        """Multislice size axes request a PotentialSlices domain envelope."""
        distribution = create_distribution(
            samples=jnp.array([[40.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="size",
        )

        bound = distribution.bind(
            lambda dist: bind_multislice_axis_distribution(
                dist,
                twin_builder=_unused_structure_builder,
                step_builder=_unused_structure_builder,
            )
        )
        update = bound(distribution.samples[0])

        assert update.crystal is None
        chex.assert_trees_all_close(update.domain_size_angstrom, 40.0)

    def test_multislice_unknown_axis_still_fails_loudly(self) -> None:
        """Unregistered multislice axes still fail before binding."""
        distribution = create_distribution(
            samples=jnp.array([[40.0]]),
            weights=jnp.array([1.0]),
            reduction=ReductionMode.INCOHERENT,
            axis_id="unknown_axis",
        )

        with pytest.raises(ValueError, match="registered multislice bind"):
            distribution.bind(
                lambda dist: bind_multislice_axis_distribution(
                    dist,
                    twin_builder=_unused_structure_builder,
                    step_builder=_unused_structure_builder,
                )
            )
