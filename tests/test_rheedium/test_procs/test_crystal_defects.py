"""Test suite for procs/crystal_defects.py."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rheedium.procs.crystal_defects import (
    apply_antisite_field,
    apply_interstitial_field,
    apply_vacancy_field,
)
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.ucell.unitcell import build_cell_vectors


def _make_bulk_crystal():
    """Build a small orthorhombic crystal for defect tests."""
    frac_positions = jnp.array(
        [
            [0.0, 0.0, 0.0, 14.0],
            [0.5, 0.5, 0.5, 8.0],
        ]
    )
    cell_vectors = build_cell_vectors(4.0, 3.0, 5.0, 90.0, 90.0, 90.0)
    cart_positions = jnp.column_stack(
        [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([4.0, 3.0, 5.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestApplyVacancyField(chex.TestCase):
    """Tests for apply_vacancy_field."""

    def test_applies_continuous_site_occupancies(self):
        crystal = _make_bulk_crystal()
        modified = apply_vacancy_field(crystal, jnp.array([1.0, 0.25]))

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 2.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.asarray(crystal.cart_positions[:, :3]),
            atol=1e-6,
        )

    def test_clips_unphysical_occupancies(self):
        crystal = _make_bulk_crystal()
        modified = apply_vacancy_field(crystal, jnp.array([-1.0, 2.0]))

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([0.0, 8.0]),
            atol=1e-6,
        )

    def test_grad_flows_through_occupancy(self):
        crystal = _make_bulk_crystal()

        def objective(occupancy):
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).cart_positions[:, 3]
            )

        grad_value = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 8.0, atol=1e-6)

    def test_jit_compiles(self):
        crystal = _make_bulk_crystal()
        compiled = jax.jit(
            lambda occupancies: apply_vacancy_field(
                crystal,
                occupancies,
            ).cart_positions[:, 3]
        )

        result = compiled(jnp.array([1.0, 0.5]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 4.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_occupancies(self):
        crystal = _make_bulk_crystal()

        def summed_intensity(occupancy):
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).cart_positions[:, 3]
            )

        result = jax.vmap(summed_intensity)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 18.0, 22.0]),
            atol=1e-6,
        )


class TestApplyInterstitialField(chex.TestCase):
    """Tests for apply_interstitial_field."""

    def test_appends_weighted_interstitial_sites(self):
        crystal = _make_bulk_crystal()
        modified = apply_interstitial_field(
            crystal,
            jnp.array([[0.25, 0.5, 0.75]]),
            jnp.array([12.0]),
            jnp.array([0.5]),
        )

        assert modified.cart_positions.shape == (3, 4)
        np.testing.assert_allclose(
            np.asarray(modified.frac_positions[-1]),
            np.array([0.25, 0.5, 0.75, 6.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[-1]),
            np.array([1.0, 1.5, 3.75, 6.0]),
            atol=1e-6,
        )

    def test_empty_interstitial_bank_leaves_crystal_unchanged(self):
        crystal = _make_bulk_crystal()
        modified = apply_interstitial_field(
            crystal,
            jnp.zeros((0, 3), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions),
            np.asarray(crystal.cart_positions),
            atol=1e-6,
        )

    def test_grad_flows_through_interstitial_occupancy(self):
        crystal = _make_bulk_crystal()

        def objective(occupancy):
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).cart_positions[:, 3]
            )

        grad_value = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 12.0, atol=1e-6)

    def test_jit_compiles(self):
        crystal = _make_bulk_crystal()
        compiled = jax.jit(
            lambda occupancy: apply_interstitial_field(
                crystal,
                jnp.array([[0.25, 0.5, 0.75]]),
                jnp.array([12.0]),
                jnp.array([occupancy]),
            ).cart_positions[-1]
        )

        result = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 1.5, 3.75, 6.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_occupancies(self):
        crystal = _make_bulk_crystal()

        def summed_intensity(occupancy):
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).cart_positions[:, 3]
            )

        result = jax.vmap(summed_intensity)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([22.0, 28.0, 34.0]),
            atol=1e-6,
        )


class TestApplyAntisiteField(chex.TestCase):
    """Tests for apply_antisite_field."""

    def test_blends_host_and_substitute_species(self):
        crystal = _make_bulk_crystal()
        modified = apply_antisite_field(
            crystal,
            jnp.array([0.0, 0.25]),
            jnp.array([20.0, 20.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 11.0]),
            atol=1e-6,
        )

    def test_clips_mixing_fraction_to_physical_range(self):
        crystal = _make_bulk_crystal()
        modified = apply_antisite_field(
            crystal,
            jnp.array([-1.0, 2.0]),
            jnp.array([20.0, 20.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 20.0]),
            atol=1e-6,
        )

    def test_grad_flows_through_mixing_fraction(self):
        crystal = _make_bulk_crystal()

        def objective(mixing_fraction):
            return jnp.sum(
                apply_antisite_field(
                    crystal,
                    jnp.array([0.0, mixing_fraction]),
                    jnp.array([20.0, 20.0]),
                ).cart_positions[:, 3]
            )

        grad_value = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 12.0, atol=1e-6)

    def test_jit_compiles(self):
        crystal = _make_bulk_crystal()
        compiled = jax.jit(
            lambda mixing_fraction: apply_antisite_field(
                crystal,
                jnp.array([0.0, mixing_fraction]),
                jnp.array([20.0, 20.0]),
            ).cart_positions[:, 3]
        )

        result = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 14.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_mixing_fraction(self):
        crystal = _make_bulk_crystal()

        def summed_intensity(mixing_fraction):
            return jnp.sum(
                apply_antisite_field(
                    crystal,
                    jnp.array([0.0, mixing_fraction]),
                    jnp.array([20.0, 20.0]),
                ).cart_positions[:, 3]
            )

        result = jax.vmap(summed_intensity)(jnp.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([22.0, 28.0, 34.0]),
            atol=1e-6,
        )
