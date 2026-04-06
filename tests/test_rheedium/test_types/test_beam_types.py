"""Tests for ElectronBeam PyTree and create_electron_beam factory."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import tree_util
from jaxtyping import TypeCheckError

from rheedium.types import (
    ElectronBeam,
    create_electron_beam,
)


class TestElectronBeam(chex.TestCase):
    """Comprehensive test suite for ElectronBeam PyTree."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_electron_beam_defaults(self) -> None:
        """Default beam should have standard RHEED gun values."""
        var_create = self.variant(create_electron_beam)
        beam = var_create()

        chex.assert_trees_all_close(beam.energy_kev, 20.0)
        chex.assert_trees_all_close(beam.energy_spread_ev, 0.5)
        chex.assert_trees_all_close(beam.angular_divergence_mrad, 0.5)
        chex.assert_trees_all_close(
            beam.coherence_length_transverse_angstrom, 500.0
        )
        chex.assert_trees_all_close(
            beam.coherence_length_longitudinal_angstrom, 1000.0
        )
        chex.assert_shape(beam.spot_size_um, (2,))
        chex.assert_trees_all_close(
            beam.spot_size_um, jnp.array([100.0, 50.0])
        )

    def test_create_electron_beam_custom(self) -> None:
        """Custom values should be preserved."""
        beam = create_electron_beam(
            energy_kev=15.0,
            energy_spread_ev=0.2,
            angular_divergence_mrad=0.3,
            coherence_length_transverse_angstrom=800.0,
            coherence_length_longitudinal_angstrom=2000.0,
            spot_size_um=jnp.array([200.0, 100.0]),
        )
        chex.assert_trees_all_close(beam.energy_kev, 15.0)
        chex.assert_trees_all_close(beam.energy_spread_ev, 0.2)
        chex.assert_trees_all_close(beam.angular_divergence_mrad, 0.3)
        chex.assert_trees_all_close(
            beam.coherence_length_transverse_angstrom, 800.0
        )
        chex.assert_trees_all_close(
            beam.coherence_length_longitudinal_angstrom, 2000.0
        )
        chex.assert_trees_all_close(
            beam.spot_size_um, jnp.array([200.0, 100.0])
        )

    def test_electron_beam_pytree(self) -> None:
        """ElectronBeam should flatten and unflatten as a PyTree."""
        beam = create_electron_beam()
        flat, treedef = tree_util.tree_flatten(beam)

        assert len(flat) == 6
        reconstructed = treedef.unflatten(flat)
        assert isinstance(reconstructed, ElectronBeam)
        chex.assert_trees_all_close(beam, reconstructed)

    def test_electron_beam_jit(self) -> None:
        """ElectronBeam should be creatable inside jit."""

        @jax.jit
        def make_beam(energy):
            return create_electron_beam(energy_kev=energy)

        beam = make_beam(jnp.float64(25.0))
        chex.assert_trees_all_close(beam.energy_kev, 25.0)

    def test_electron_beam_vmap(self) -> None:
        """vmap over energy should produce batched beams."""

        def make_beam(energy):
            beam = create_electron_beam(energy_kev=energy)
            return beam.energy_kev

        energies = jnp.array([10.0, 20.0, 30.0])
        result = jax.vmap(make_beam)(energies)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(result, energies)

    def test_electron_beam_tree_map(self) -> None:
        """tree_map should apply to all leaves."""
        beam = create_electron_beam(energy_kev=20.0)
        doubled = jax.tree.map(lambda x: x * 2.0, beam)
        chex.assert_trees_all_close(doubled.energy_kev, 40.0)
        chex.assert_trees_all_close(doubled.energy_spread_ev, 1.0)

    def test_dtypes_are_float64(self) -> None:
        """All fields should be float64."""
        beam = create_electron_beam()
        assert beam.energy_kev.dtype == jnp.float64
        assert beam.energy_spread_ev.dtype == jnp.float64
        assert beam.angular_divergence_mrad.dtype == jnp.float64
        assert beam.coherence_length_transverse_angstrom.dtype == jnp.float64
        assert beam.coherence_length_longitudinal_angstrom.dtype == jnp.float64
        assert beam.spot_size_um.dtype == jnp.float64


class TestElectronBeamValidation(chex.TestCase):
    """Validation tests for create_electron_beam."""

    def test_energy_too_low(self) -> None:
        """Energy below 5 keV should produce NaN."""
        beam = create_electron_beam(energy_kev=1.0)
        assert jnp.isnan(beam.energy_kev)

    def test_energy_too_high(self) -> None:
        """Energy above 100 keV should produce NaN."""
        beam = create_electron_beam(energy_kev=200.0)
        assert jnp.isnan(beam.energy_kev)

    def test_energy_boundary_low(self) -> None:
        """Energy at exactly 5 keV should be valid."""
        beam = create_electron_beam(energy_kev=5.0)
        chex.assert_trees_all_close(beam.energy_kev, 5.0)

    def test_energy_boundary_high(self) -> None:
        """Energy at exactly 100 keV should be valid."""
        beam = create_electron_beam(energy_kev=100.0)
        chex.assert_trees_all_close(beam.energy_kev, 100.0)

    def test_negative_energy_spread(self) -> None:
        """Negative energy spread should produce NaN."""
        beam = create_electron_beam(energy_spread_ev=-0.1)
        assert jnp.isnan(beam.energy_spread_ev)

    def test_zero_energy_spread(self) -> None:
        """Zero energy spread should be valid (ideal source)."""
        beam = create_electron_beam(energy_spread_ev=0.0)
        chex.assert_trees_all_close(beam.energy_spread_ev, 0.0)

    def test_negative_divergence(self) -> None:
        """Negative angular divergence should produce NaN."""
        beam = create_electron_beam(angular_divergence_mrad=-0.1)
        assert jnp.isnan(beam.angular_divergence_mrad)

    def test_zero_divergence(self) -> None:
        """Zero divergence should be valid (perfect collimation)."""
        beam = create_electron_beam(angular_divergence_mrad=0.0)
        chex.assert_trees_all_close(beam.angular_divergence_mrad, 0.0)

    def test_negative_transverse_coherence(self) -> None:
        """Negative transverse coherence length should produce NaN."""
        beam = create_electron_beam(coherence_length_transverse_angstrom=-10.0)
        assert jnp.isnan(beam.coherence_length_transverse_angstrom)

    def test_zero_transverse_coherence(self) -> None:
        """Zero transverse coherence length should produce NaN."""
        beam = create_electron_beam(coherence_length_transverse_angstrom=0.0)
        assert jnp.isnan(beam.coherence_length_transverse_angstrom)

    def test_negative_longitudinal_coherence(self) -> None:
        """Negative longitudinal coherence length should produce NaN."""
        beam = create_electron_beam(
            coherence_length_longitudinal_angstrom=-10.0
        )
        assert jnp.isnan(beam.coherence_length_longitudinal_angstrom)

    def test_zero_longitudinal_coherence(self) -> None:
        """Zero longitudinal coherence length should produce NaN."""
        beam = create_electron_beam(coherence_length_longitudinal_angstrom=0.0)
        assert jnp.isnan(beam.coherence_length_longitudinal_angstrom)

    def test_negative_spot_size(self) -> None:
        """Negative spot size should produce NaN."""
        beam = create_electron_beam(spot_size_um=jnp.array([-100.0, 50.0]))
        assert jnp.any(jnp.isnan(beam.spot_size_um))

    def test_zero_spot_size(self) -> None:
        """Zero spot size should produce NaN."""
        beam = create_electron_beam(spot_size_um=jnp.array([0.0, 50.0]))
        assert jnp.any(jnp.isnan(beam.spot_size_um))

    def test_valid_fields_unaffected_by_invalid(self) -> None:
        """Invalid energy should not corrupt other fields."""
        beam = create_electron_beam(energy_kev=1.0, energy_spread_ev=0.3)
        assert jnp.isnan(beam.energy_kev)
        chex.assert_trees_all_close(beam.energy_spread_ev, 0.3)


class TestElectronBeamGradients(chex.TestCase):
    """Gradient tests for ElectronBeam through create_electron_beam."""

    def test_grad_energy(self) -> None:
        """Gradient should flow through energy_kev."""

        def loss(energy):
            beam = create_electron_beam(energy_kev=energy)
            return beam.energy_kev**2

        g = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)
        chex.assert_trees_all_close(g, 40.0)

    def test_grad_energy_spread(self) -> None:
        """Gradient should flow through energy_spread_ev."""

        def loss(spread):
            beam = create_electron_beam(energy_spread_ev=spread)
            return beam.energy_spread_ev**2

        g = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        chex.assert_trees_all_close(g, 1.0)

    def test_grad_divergence(self) -> None:
        """Gradient should flow through angular_divergence_mrad."""

        def loss(div):
            beam = create_electron_beam(angular_divergence_mrad=div)
            return beam.angular_divergence_mrad**2

        g = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        chex.assert_trees_all_close(g, 1.0)

    def test_grad_coherence_lengths(self) -> None:
        """Gradient should flow through both coherence lengths."""

        def loss(lt, ll):
            beam = create_electron_beam(
                coherence_length_transverse_angstrom=lt,
                coherence_length_longitudinal_angstrom=ll,
            )
            return (
                beam.coherence_length_transverse_angstrom
                + beam.coherence_length_longitudinal_angstrom
            )

        gt, gl = jax.grad(loss, argnums=(0, 1))(
            jnp.float64(500.0), jnp.float64(1000.0)
        )
        chex.assert_tree_all_finite(gt)
        chex.assert_tree_all_finite(gl)
        chex.assert_trees_all_close(gt, 1.0)
        chex.assert_trees_all_close(gl, 1.0)

    def test_grad_spot_size(self) -> None:
        """Gradient should flow through spot_size_um."""

        def loss(spot):
            beam = create_electron_beam(spot_size_um=spot)
            return jnp.sum(beam.spot_size_um**2)

        g = jax.grad(loss)(jnp.array([100.0, 50.0]))
        chex.assert_tree_all_finite(g)
        chex.assert_shape(g, (2,))
        chex.assert_trees_all_close(g, jnp.array([200.0, 100.0]))

    def test_jacrev_multi_param(self) -> None:
        """jacrev over (energy, spread) produces correct Jacobian."""

        def loss(params):
            beam = create_electron_beam(
                energy_kev=params[0],
                energy_spread_ev=params[1],
            )
            return beam.energy_kev + beam.energy_spread_ev

        jac = jax.jacrev(loss)(jnp.array([20.0, 0.5]))
        chex.assert_shape(jac, (2,))
        chex.assert_tree_all_finite(jac)
        chex.assert_trees_all_close(jac, jnp.array([1.0, 1.0]))
