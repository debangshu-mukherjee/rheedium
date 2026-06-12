"""Test suite for procs/crystal_defects.py."""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rheedium.procs.crystal_defects import (
    apply_antisite_field,
    apply_interstitial_field,
    apply_vacancy_field,
)
from rheedium.types import CrystalStructure
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.types.custom_types import scalar_float
from rheedium.ucell.unitcell import build_cell_vectors


def _make_bulk_crystal() -> CrystalStructure:
    """Build a small orthorhombic crystal for defect tests."""
    frac_positions: Float[Array, "2 4"] = jnp.array(
        [
            [0.0, 0.0, 0.0, 14.0],
            [0.5, 0.5, 0.5, 8.0],
        ]
    )
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        4.0, 3.0, 5.0, 90.0, 90.0, 90.0
    )
    cart_positions: Float[Array, "2 4"] = jnp.column_stack(
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

    def test_applies_continuous_site_occupancies(self) -> None:
        """Verify occupancies scale atomic numbers continuously."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_vacancy_field(
            crystal, jnp.array([1.0, 0.25])
        )

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

    def test_clips_unphysical_occupancies(self) -> None:
        """Verify occupancies are clipped to the physical range."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_vacancy_field(
            crystal, jnp.array([-1.0, 2.0])
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([0.0, 8.0]),
            atol=1e-6,
        )

    def test_grad_flows_through_occupancy(self) -> None:
        """Check gradients flow through the occupancy parameter."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).cart_positions[:, 3]
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 8.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        """Verify apply_vacancy_field compiles under jit."""
        crystal: CrystalStructure = _make_bulk_crystal()
        compiled: Callable[..., Any] = jax.jit(
            lambda occupancies: apply_vacancy_field(
                crystal,
                occupancies,
            ).cart_positions[:, 3]
        )

        result: Float[Array, "2"] = compiled(jnp.array([1.0, 0.5]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 4.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_occupancies(self) -> None:
        """Check apply_vacancy_field maps over batched occupancies."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_intensity(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).cart_positions[:, 3]
            )

        result: Float[Array, "3"] = jax.vmap(summed_intensity)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 18.0, 22.0]),
            atol=1e-6,
        )


class TestApplyInterstitialField(chex.TestCase):
    """Tests for apply_interstitial_field."""

    def test_appends_weighted_interstitial_sites(self) -> None:
        """Verify occupancy-weighted interstitial sites are appended."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_interstitial_field(
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

    def test_empty_interstitial_bank_leaves_crystal_unchanged(self) -> None:
        """Verify an empty interstitial bank leaves the crystal as is."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_interstitial_field(
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

    def test_grad_flows_through_interstitial_occupancy(self) -> None:
        """Check gradients flow through interstitial occupancy."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).cart_positions[:, 3]
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 12.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        """Verify apply_interstitial_field compiles under jit."""
        crystal: CrystalStructure = _make_bulk_crystal()
        compiled: Callable[..., Any] = jax.jit(
            lambda occupancy: apply_interstitial_field(
                crystal,
                jnp.array([[0.25, 0.5, 0.75]]),
                jnp.array([12.0]),
                jnp.array([occupancy]),
            ).cart_positions[-1]
        )

        result: Float[Array, "4"] = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 1.5, 3.75, 6.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_occupancies(self) -> None:
        """Check apply_interstitial_field maps over batched occupancies."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_intensity(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).cart_positions[:, 3]
            )

        result: Float[Array, "3"] = jax.vmap(summed_intensity)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([22.0, 28.0, 34.0]),
            atol=1e-6,
        )


class TestApplyAntisiteField(chex.TestCase):
    """Tests for apply_antisite_field."""

    def test_blends_host_and_substitute_species(self) -> None:
        """Verify host and substitute species are blended by fraction."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_antisite_field(
            crystal,
            jnp.array([0.0, 0.25]),
            jnp.array([20.0, 20.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 11.0]),
            atol=1e-6,
        )

    def test_clips_mixing_fraction_to_physical_range(self) -> None:
        """Verify the mixing fraction is clipped to a physical range."""
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_antisite_field(
            crystal,
            jnp.array([-1.0, 2.0]),
            jnp.array([20.0, 20.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 20.0]),
            atol=1e-6,
        )

    def test_grad_flows_through_mixing_fraction(self) -> None:
        """Check gradients flow through the mixing fraction."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(mixing_fraction: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_antisite_field(
                    crystal,
                    jnp.array([0.0, mixing_fraction]),
                    jnp.array([20.0, 20.0]),
                ).cart_positions[:, 3]
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 12.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        """Verify apply_antisite_field compiles under jit."""
        crystal: CrystalStructure = _make_bulk_crystal()
        compiled: Callable[..., Any] = jax.jit(
            lambda mixing_fraction: apply_antisite_field(
                crystal,
                jnp.array([0.0, mixing_fraction]),
                jnp.array([20.0, 20.0]),
            ).cart_positions[:, 3]
        )

        result: Float[Array, "2"] = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([14.0, 14.0]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_mixing_fraction(self) -> None:
        """Check apply_antisite_field maps over batched fractions."""
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_intensity(mixing_fraction: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_antisite_field(
                    crystal,
                    jnp.array([0.0, mixing_fraction]),
                    jnp.array([20.0, 20.0]),
                ).cart_positions[:, 3]
            )

        result: Float[Array, "3"] = jax.vmap(summed_intensity)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([22.0, 28.0, 34.0]),
            atol=1e-6,
        )
