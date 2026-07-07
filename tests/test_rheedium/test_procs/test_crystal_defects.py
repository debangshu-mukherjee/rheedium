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
    """Tests for apply_vacancy_field.

    :see: :func:`~rheedium.procs.apply_vacancy_field`
    """

    def test_applies_continuous_site_occupancies(self) -> None:
        r"""Verify occupancies land in the first-class occupancy field.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: vacancy
        occupancies scale the crystal's ``occupancies`` field continuously
        while the atomic-number column stays the integral element identity
        (the C6 contract: a partially vacant silicon site is still silicon,
        never rescaled into a different element).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_vacancy_field(
            crystal, jnp.array([1.0, 0.25])
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 8.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 0.25]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.asarray(crystal.cart_positions[:, :3]),
            atol=1e-6,
        )

    def test_clips_unphysical_occupancies(self) -> None:
        r"""Verify occupancies are clipped to the physical range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: occupancies are
        clipped to the physical range [0, 1] in the ``occupancies`` field,
        with the atomic-number column untouched.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_vacancy_field(
            crystal, jnp.array([-1.0, 2.0])
        )

        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([0.0, 1.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 8.0]),
            atol=1e-6,
        )

    def test_keeps_atomic_numbers_integral_at_small_vacancy(self) -> None:
        r"""Verify a 1% vacancy never changes the element identity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a vacancy
        fraction of 0.01 leaves the atomic-number column exactly equal to
        the original element everywhere with occupancies of 0.99. The old
        effective-Z encoding turned a 1% silicon vacancy into aluminum
        (Z = 14 x 0.99 = 13.86 truncated to 13); this regression pins the
        C6 fix.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_vacancy_field(
            crystal, jnp.full((2,), 0.99)
        )

        np.testing.assert_array_equal(
            np.asarray(modified.cart_positions[:, 3]),
            np.asarray(crystal.cart_positions[:, 3]),
        )
        np.testing.assert_array_equal(
            np.asarray(modified.frac_positions[:, 3]),
            np.asarray(crystal.frac_positions[:, 3]),
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([0.99, 0.99]),
            atol=1e-12,
        )

    def test_grad_flows_through_occupancy(self) -> None:
        r"""Check gradients flow through the occupancy parameter.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through the occupancy parameter into the ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).occupancies
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 1.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_vacancy_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_vacancy_field compiles under jit and returns the vacancy
        occupancies in the ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        compiled: Callable[..., Any] = jax.jit(
            lambda occupancies: apply_vacancy_field(
                crystal,
                occupancies,
            ).occupancies
        )

        result: Float[Array, "2"] = compiled(jnp.array([1.0, 0.5]))
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 0.5]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_occupancies(self) -> None:
        r"""Check apply_vacancy_field maps over batched occupancies.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check
        apply_vacancy_field maps over batched occupancies through the
        ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_occupancy(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_vacancy_field(
                    crystal,
                    jnp.array([1.0, occupancy]),
                ).occupancies
            )

        result: Float[Array, "3"] = jax.vmap(summed_occupancy)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 1.5, 2.0]),
            atol=1e-6,
        )


class TestApplyInterstitialField(chex.TestCase):
    """Tests for apply_interstitial_field.

    :see: :func:`~rheedium.procs.apply_interstitial_field`
    """

    def test_appends_weighted_interstitial_sites(self) -> None:
        r"""Verify occupancy-weighted interstitial sites are appended.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: occupancy-weighted
        interstitial sites are appended with their integral atomic number in
        the species column and the occupancy in the first-class
        ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            np.array([0.25, 0.5, 0.75, 12.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[-1]),
            np.array([1.0, 1.5, 3.75, 12.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 1.0, 0.5]),
            atol=1e-6,
        )

    def test_empty_interstitial_bank_leaves_crystal_unchanged(self) -> None:
        r"""Verify an empty interstitial bank leaves the crystal as is.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: an empty
        interstitial bank leaves the crystal as is.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.ones(2),
            atol=1e-6,
        )

    def test_grad_flows_through_interstitial_occupancy(self) -> None:
        r"""Check gradients flow through interstitial occupancy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through interstitial occupancy into the ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).occupancies
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 1.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_interstitial_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_interstitial_field compiles under jit, keeping the appended
        site's integral atomic number and storing its occupancy in the
        ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def _decorated(occupancy: scalar_float) -> tuple:
            modified: CrystalStructure = apply_interstitial_field(
                crystal,
                jnp.array([[0.25, 0.5, 0.75]]),
                jnp.array([12.0]),
                jnp.array([occupancy]),
            )
            assert modified.occupancies is not None
            return modified.cart_positions[-1], modified.occupancies[-1]

        compiled: Callable[..., Any] = jax.jit(_decorated)
        result_row, result_occ = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result_row),
            np.array([1.0, 1.5, 3.75, 12.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(float(result_occ), 0.5, atol=1e-6)

    def test_vmap_supports_batched_occupancies(self) -> None:
        r"""Check apply_interstitial_field maps over batched occupancies.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check
        apply_interstitial_field maps over batched occupancies through the
        ``occupancies`` field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_occupancy(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_interstitial_field(
                    crystal,
                    jnp.array([[0.25, 0.5, 0.75]]),
                    jnp.array([12.0]),
                    jnp.array([occupancy]),
                ).occupancies
            )

        result: Float[Array, "3"] = jax.vmap(summed_occupancy)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([2.0, 2.5, 3.0]),
            atol=1e-6,
        )


class TestApplyAntisiteField(chex.TestCase):
    """Tests for apply_antisite_field.

    :see: :func:`~rheedium.procs.apply_antisite_field`
    """

    def test_blends_host_and_substitute_species(self) -> None:
        r"""Verify antisites become co-located complementary-occupancy pairs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: each site is
        split into two co-located sites — the host element at occupancy
        ``1 - f`` and the substitute element at occupancy ``f`` — with both
        atomic numbers kept integral, so the site's scattering amplitude is
        ``(1 - f) f_host + f f_substitute``.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_antisite_field(
            crystal,
            jnp.array([0.0, 0.25]),
            jnp.array([20.0, 20.0]),
        )

        assert modified.cart_positions.shape == (4, 4)
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 8.0, 20.0, 20.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 0.75, 0.0, 0.25]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[2:, :3]),
            np.asarray(crystal.cart_positions[:, :3]),
            atol=1e-6,
        )

    def test_clips_mixing_fraction_to_physical_range(self) -> None:
        r"""Verify the mixing fraction is clipped to a physical range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the mixing
        fraction is clipped to [0, 1] before building the complementary
        host/substitute occupancy pair.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        modified: CrystalStructure = apply_antisite_field(
            crystal,
            jnp.array([-1.0, 2.0]),
            jnp.array([20.0, 20.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 0.0, 0.0, 1.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 8.0, 20.0, 20.0]),
            atol=1e-6,
        )

    def test_grad_flows_through_mixing_fraction(self) -> None:
        r"""Check gradients flow through the mixing fraction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through the mixing fraction. The occupancy-weighted sum of
        atomic numbers ``sum(occ * Z)`` differentiates to
        ``Z_substitute - Z_host = 20 - 8 = 12`` for the mixed site.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def objective(mixing_fraction: scalar_float) -> scalar_float:
            modified: CrystalStructure = apply_antisite_field(
                crystal,
                jnp.array([0.0, mixing_fraction]),
                jnp.array([20.0, 20.0]),
            )
            assert modified.occupancies is not None
            return jnp.sum(
                modified.occupancies * modified.cart_positions[:, 3]
            )

        grad_value: scalar_float = jax.grad(objective)(0.25)
        chex.assert_trees_all_close(float(grad_value), 12.0, atol=1e-6)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_antisite_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_antisite_field compiles under jit and returns the
        complementary host/substitute occupancy pair.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()
        compiled: Callable[..., Any] = jax.jit(
            lambda mixing_fraction: apply_antisite_field(
                crystal,
                jnp.array([0.0, mixing_fraction]),
                jnp.array([20.0, 20.0]),
            ).occupancies
        )

        result: Float[Array, "4"] = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 0.5, 0.0, 0.5]),
            atol=1e-6,
        )

    def test_vmap_supports_batched_mixing_fraction(self) -> None:
        r"""Check apply_antisite_field maps over batched fractions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check
        apply_antisite_field maps over batched fractions. The
        occupancy-weighted atomic-number sum interpolates linearly from the
        pure-host value (22) to the fully substituted value (34).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_crystal_defects``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_bulk_crystal()

        def summed_effective(mixing_fraction: scalar_float) -> scalar_float:
            modified: CrystalStructure = apply_antisite_field(
                crystal,
                jnp.array([0.0, mixing_fraction]),
                jnp.array([20.0, 20.0]),
            )
            assert modified.occupancies is not None
            return jnp.sum(
                modified.occupancies * modified.cart_positions[:, 3]
            )

        result: Float[Array, "3"] = jax.vmap(summed_effective)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([22.0, 28.0, 34.0]),
            atol=1e-6,
        )
