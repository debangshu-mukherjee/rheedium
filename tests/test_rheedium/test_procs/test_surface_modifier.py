"""Test suite for procs/surface_modifier.py.

Tests vicinal surface step splitting and incoherent domain averaging.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float

from rheedium.procs.surface_modifier import (
    apply_step_edge_field,
    apply_surface_displacement_field,
    apply_surface_occupancy_field,
    apply_twin_wall_field,
    apply_vicinal_staircase,
    bind_step_edge_distribution,
    bind_twin_wall_distribution,
    incoherent_domain_average,
    step_edge_to_distribution,
    twin_wall_to_distribution,
    vicinal_surface_step_splitting,
)
from rheedium.simul.beam_averaging import apply_distribution
from rheedium.types import CrystalStructure, Distribution, ReductionMode
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.types.custom_types import scalar_float
from rheedium.ucell.unitcell import build_cell_vectors


def _make_test_slab() -> CrystalStructure:
    """Build a small orthorhombic slab with two surface atoms."""
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        2.0, 2.0, 6.0, 90.0, 90.0, 90.0
    )
    cart_positions: Float[Array, "3 4"] = jnp.array(
        [
            [0.2, 0.2, 0.5, 14.0],
            [0.5, 0.4, 4.8, 14.0],
            [1.5, 1.6, 4.8, 8.0],
        ]
    )
    frac_positions: Float[Array, "3 4"] = jnp.column_stack(
        [
            cart_positions[:, :3] @ jnp.linalg.inv(cell_vectors).T,
            cart_positions[:, 3],
        ]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([2.0, 2.0, 6.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestVicinalSurfaceStepSplitting(chex.TestCase, parameterized.TestCase):
    """Tests for vicinal_surface_step_splitting function.

    :see: :func:`~rheedium.procs.vicinal_surface_step_splitting`
    """

    def test_output_shape_matches_input(self) -> None:
        r"""Output should have same shape as q_z input.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should have
        same shape as q_z input.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        q_z: Float[Array, "100"] = jnp.linspace(0.0, 10.0, 100)
        result: Float[Array, "100"] = vicinal_surface_step_splitting(
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        chex.assert_shape(result, (100,))

    def test_output_nonnegative(self) -> None:
        r"""All intensity values should be >= 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All intensity
        values should be >= 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        q_z: Float[Array, "200"] = jnp.linspace(0.0, 10.0, 200)
        result: Float[Array, "200"] = vicinal_surface_step_splitting(
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(jnp.min(result)) >= -1e-10

    def test_output_bounded_by_one(self) -> None:
        r"""Normalized intensity should not exceed 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Normalized
        intensity should not exceed 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        q_z: Float[Array, "200"] = jnp.linspace(0.0, 10.0, 200)
        result: Float[Array, "200"] = vicinal_surface_step_splitting(
            step_height_angstrom=2.0,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(jnp.max(result)) <= 1.0 + 1e-10

    def test_antiphase_condition_dip(self) -> None:
        r"""At q_z * d = pi, intensity should show a minimum.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: At q_z * d = pi,
        intensity should show a minimum.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        step_height: float = 2.0
        q_at_pi: scalar_float = jnp.pi / step_height
        q_z: Float[Array, "2"] = jnp.array([0.0, q_at_pi])
        result: Float[Array, "2"] = vicinal_surface_step_splitting(
            step_height_angstrom=step_height,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        assert float(result[1]) < float(result[0])

    def test_in_phase_condition_peak(self) -> None:
        r"""At q_z * d = 2*pi, intensity should be at maximum.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: At q_z * d = 2*pi,
        intensity should be at maximum.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        step_height: float = 2.0
        q_at_2pi: scalar_float = 2.0 * jnp.pi / step_height
        q_z: Float[Array, "1"] = jnp.array([q_at_2pi])
        result: Float[Array, "1"] = vicinal_surface_step_splitting(
            step_height_angstrom=step_height,
            terrace_width_angstrom=50.0,
            q_z=q_z,
        )
        chex.assert_trees_all_close(float(result[0]), 1.0, atol=1e-6)

    def test_wider_terraces_sharper_peaks(self) -> None:
        r"""Wider terraces should produce sharper (narrower) peaks.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Wider terraces
        should produce sharper (narrower) peaks.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        q_z: Float[Array, "1000"] = jnp.linspace(0.0, 10.0, 1000)
        narrow: Float[Array, "1000"] = vicinal_surface_step_splitting(
            step_height_angstrom=2.0,
            terrace_width_angstrom=20.0,
            q_z=q_z,
        )
        wide: Float[Array, "1000"] = vicinal_surface_step_splitting(
            step_height_angstrom=2.0,
            terrace_width_angstrom=100.0,
            q_z=q_z,
        )
        narrow_mean: float = float(jnp.mean(narrow))
        wide_mean: float = float(jnp.mean(wide))
        assert wide_mean < narrow_mean

    def test_no_nan_or_inf(self) -> None:
        r"""Output should be finite everywhere.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should be
        finite everywhere.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        q_z: Float[Array, "500"] = jnp.linspace(0.0, 20.0, 500)
        result: Float[Array, "500"] = vicinal_surface_step_splitting(
            step_height_angstrom=3.0,
            terrace_width_angstrom=30.0,
            q_z=q_z,
        )
        chex.assert_tree_all_finite(result)

    def test_grid_independent_normalization(self) -> None:
        r"""Two q_z grids give the same value at shared points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: normalizing
        by the analytic zero-splitting (in-phase) peak of ``1`` makes the
        profile independent of the ``q_z`` sampling grid, so a coarse and
        a fine grid agree to 1 percent at the ``q_z`` values they share
        (the retired per-grid-max normalization did not).

        Notes
        -----
        It evaluates the splitting on a coarse and a dense ``q_z`` grid
        whose coarse points are a subset of the dense points, then
        compares the shared samples directly.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the
        Test Reference exposes both the guarantee and the implementation
        path.
        """
        q_coarse: Float[Array, "9"] = jnp.linspace(0.05, 4.0, 9)
        q_fine: Float[Array, "41"] = jnp.linspace(0.05, 4.0, 41)
        coarse: np.ndarray = np.asarray(
            vicinal_surface_step_splitting(
                step_height_angstrom=2.0,
                terrace_width_angstrom=40.0,
                q_z=q_coarse,
            )
        )
        fine: np.ndarray = np.asarray(
            vicinal_surface_step_splitting(
                step_height_angstrom=2.0,
                terrace_width_angstrom=40.0,
                q_z=q_fine,
            )
        )
        np.testing.assert_allclose(coarse, fine[::5], rtol=1e-2)

    def test_rejects_nonpositive_step_height_during_gradient_probe(
        self,
    ) -> None:
        r"""Nonpositive physical step heights are rejected under autodiff.

        Extended Summary
        ----------------
        Verifies that the former additive denominator offset cannot create a
        pole at ``step_height_angstrom = -1e-10`` because nonpositive step
        heights are now invalid inputs.

        Notes
        -----
        It runs the audited ``jax.grad`` boundary probe and checks the clear
        runtime validation message, then confirms a positive interior input
        still has a finite gradient.
        """
        q_z: Float[Array, "1"] = jnp.array([1.0])

        def objective(step_height: scalar_float) -> scalar_float:
            return jnp.sum(
                vicinal_surface_step_splitting(
                    step_height_angstrom=step_height,
                    terrace_width_angstrom=10.0,
                    q_z=q_z,
                )
            )

        with pytest.raises(RuntimeError, match="step_height_angstrom"):
            jax.grad(objective)(-1e-10)

        gradient: scalar_float = jax.grad(objective)(2.0)
        assert bool(jnp.isfinite(gradient))


class TestApplySurfaceOccupancyField(chex.TestCase):
    """Tests for apply_surface_occupancy_field.

    :see: :func:`~rheedium.procs.apply_surface_occupancy_field`
    """

    def test_scales_only_surface_region_occupancies(self) -> None:
        r"""Verify only surface-region site occupancies are scaled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: only
        surface-region site ``occupancies`` are scaled, while the
        atomic-number column stays the integral element identity (the
        C6 contract: occupancy is a first-class float field, never an
        "effective Z").

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_surface_occupancy_field(
            slab,
            0.8,
            jnp.array([0.1, 0.5, 0.25]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 14.0, 8.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 0.5, 0.25]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.asarray(slab.cart_positions[:, :3]),
            atol=1e-6,
        )

    def test_clips_surface_occupancies(self) -> None:
        r"""Verify surface occupancies are clipped to a physical range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: surface
        occupancies are clipped to a physical range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_surface_occupancy_field(
            slab,
            0.8,
            jnp.array([0.1, 2.0, -1.0]),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.array([14.0, 14.0, 8.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(modified.occupancies),
            np.array([1.0, 1.0, 0.0]),
            atol=1e-5,
        )

    def test_grad_flows_through_surface_occupancy(self) -> None:
        r"""Check gradients flow through the surface occupancy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through the surface occupancy.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(occupancy: scalar_float) -> scalar_float:
            return jnp.sum(
                apply_surface_occupancy_field(
                    slab,
                    0.8,
                    jnp.array([1.0, occupancy, 1.0]),
                ).occupancies
            )

        grad_value: scalar_float = jax.grad(objective)(0.5)
        chex.assert_trees_all_close(float(grad_value), 1.0, atol=1e-4)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_surface_occupancy_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_surface_occupancy_field compiles under jit.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        compiled: Callable[..., Any] = jax.jit(
            lambda occupancy: apply_surface_occupancy_field(
                slab,
                0.8,
                jnp.array([1.0, occupancy, 1.0]),
            ).occupancies
        )

        result: Float[Array, "3"] = compiled(0.5)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([1.0, 0.5, 1.0]),
            atol=1e-5,
        )

    def test_vmap_supports_batched_surface_occupancies(self) -> None:
        r"""Check the field maps over batched surface occupancies.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check the field
        maps over batched surface occupancies.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def top_layer_weight(occupancy: scalar_float) -> scalar_float:
            modified: CrystalStructure = apply_surface_occupancy_field(
                slab,
                0.8,
                jnp.array([1.0, occupancy, 1.0]),
            )
            assert modified.occupancies is not None
            return modified.occupancies[1]

        result: Float[Array, "3"] = jax.vmap(top_layer_weight)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.0, 0.5, 1.0]),
            atol=1e-5,
        )


class TestApplySurfaceDisplacementField(chex.TestCase):
    """Tests for apply_surface_displacement_field.

    :see: :func:`~rheedium.procs.apply_surface_displacement_field`
    """

    def test_applies_displacements_only_near_surface(self) -> None:
        r"""Verify displacements are applied only near the surface.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: displacements are
        applied only near the surface.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_surface_displacement_field(
            slab,
            0.8,
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, -0.2, 0.3],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, :3]),
            np.array(
                [
                    [0.2, 0.2, 0.5],
                    [0.6, 0.2, 5.1],
                    [1.5, 1.6, 4.8],
                ]
            ),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 3]),
            np.asarray(slab.cart_positions[:, 3]),
            atol=1e-6,
        )

    def test_zero_displacement_field_is_identity(self) -> None:
        r"""Verify a zero displacement field leaves the slab unchanged.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a zero
        displacement field leaves the slab unchanged.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_surface_displacement_field(
            slab,
            0.8,
            jnp.zeros((3, 3)),
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions),
            np.asarray(slab.cart_positions),
            atol=1e-6,
        )

    def test_grad_flows_through_surface_displacement(self) -> None:
        r"""Check gradients flow through the surface displacement.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through the surface displacement.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(delta_z: scalar_float) -> scalar_float:
            return apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, delta_z],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, 2]

        grad_value: scalar_float = jax.grad(objective)(0.3)
        chex.assert_trees_all_close(float(grad_value), 1.0, atol=1e-4)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_surface_displacement_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_surface_displacement_field compiles under jit.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        compiled: Callable[..., Any] = jax.jit(
            lambda scale: apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1 * scale, -0.2 * scale, 0.3 * scale],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, :3]
        )

        result: Float[Array, "3"] = compiled(1.0)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.6, 0.2, 5.1]),
            atol=1e-5,
        )

    def test_vmap_supports_batched_displacement_scales(self) -> None:
        r"""Check the field maps over batched displacement scales.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check the field
        maps over batched displacement scales.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def top_atom_z(scale: scalar_float) -> scalar_float:
            return apply_surface_displacement_field(
                slab,
                0.8,
                jnp.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.3 * scale],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ).cart_positions[1, 2]

        result: Float[Array, "3"] = jax.vmap(top_atom_z)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([4.8, 4.95, 5.1]),
            atol=1e-5,
        )


class TestApplyVicinalStaircase(chex.TestCase):
    """Tests for apply_vicinal_staircase.

    :see: :func:`~rheedium.procs.apply_vicinal_staircase`
    """

    def test_mean_slope_and_drop_and_gradient(self) -> None:
        r"""Staircase slope, five-terrace drop, and gradient are exact.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the smooth
        monotonic staircase ``h(x) = -s [x/w - sin(2 pi x/w)/(2 pi)]`` has
        mean slope exactly ``-s/w`` over whole terraces, drops exactly
        ``5 s`` over five terraces, and is differentiable in the terrace
        width.

        Notes
        -----
        It samples the height field on a dense line of atoms along the
        step direction, reads the z-displacement off the returned slab,
        and checks the closed-form slope and drop plus a finite
        ``jax.grad`` with respect to ``terrace_width_ang``.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the
        Test Reference exposes both the guarantee and the implementation
        path.
        """
        width: float = 12.0
        step: float = 1.5
        xs: Float[Array, "N"] = jnp.linspace(0.0, 5.0 * width, 400)
        z0: Float[Array, "N"] = jnp.full_like(xs, 3.0)
        cart: Float[Array, "N 4"] = jnp.column_stack(
            [xs, jnp.zeros_like(xs), z0, jnp.full_like(xs, 14.0)]
        )
        cell: Float[Array, "3 3"] = build_cell_vectors(
            5.0 * width + 10.0, 10.0, 40.0, 90.0, 90.0, 90.0
        )
        frac: Float[Array, "N 4"] = jnp.column_stack(
            [cart[:, :3] @ jnp.linalg.inv(cell), cart[:, 3]]
        )
        surface: CrystalStructure = create_crystal_structure(
            frac_positions=frac,
            cart_positions=cart,
            cell_lengths=jnp.array([5.0 * width + 10.0, 10.0, 40.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )
        result: CrystalStructure = apply_vicinal_staircase(
            surface, terrace_width_ang=width, step_height_ang=step
        )
        h: np.ndarray = np.asarray(result.cart_positions[:, 2]) - 3.0
        x_np: np.ndarray = np.asarray(xs)
        drop: float = float(h[0] - h[-1])
        np.testing.assert_allclose(drop, 5.0 * step, atol=1e-6)
        slope: float = float(np.polyfit(x_np, h, 1)[0])
        np.testing.assert_allclose(slope, -step / width, atol=2e-3)
        assert bool(np.all(np.diff(h) <= 1e-9))

        def total_drop(w: scalar_float) -> Float[Array, ""]:
            out: CrystalStructure = apply_vicinal_staircase(
                surface, terrace_width_ang=w, step_height_ang=step
            )
            return out.cart_positions[0, 2] - out.cart_positions[-1, 2]

        grad_w: Float[Array, ""] = jax.grad(total_drop)(width)
        assert bool(jnp.isfinite(grad_w))

    def test_rejects_zero_step_direction_during_gradient_probe(self) -> None:
        r"""A zero staircase direction is rejected under autodiff.

        Extended Summary
        ----------------
        Verifies that the meaningless zero direction cannot enter the former
        ``where`` normalization trap during a ``jax.grad`` evaluation.

        Notes
        -----
        It differentiates the summed output height with respect to the full
        direction vector at the audited zero boundary and checks the explicit
        validation error; a nonzero direction retains finite derivatives.
        """
        surface: CrystalStructure = _make_test_slab()

        def objective(direction: Float[Array, "2"]) -> scalar_float:
            modified: CrystalStructure = apply_vicinal_staircase(
                surface,
                terrace_width_ang=2.0,
                step_height_ang=1.0,
                step_direction_xy=direction,
            )
            return jnp.sum(modified.cart_positions[:, 2])

        with pytest.raises(RuntimeError, match="step_direction_xy"):
            jax.grad(objective)(jnp.zeros(2))

        gradient: Float[Array, "2"] = jax.grad(objective)(
            jnp.array([1.0, 0.0])
        )
        chex.assert_tree_all_finite(gradient)


class TestApplyStepEdgeField(chex.TestCase):
    """Tests for apply_step_edge_field.

    :see: :func:`~rheedium.procs.apply_step_edge_field`
    """

    def test_modulates_surface_heights_with_periodic_steps(self) -> None:
        r"""Verify surface heights are modulated by periodic steps.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: surface heights
        are modulated by periodic steps.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_step_edge_field(
            slab,
            1.0,
            2.0,
            0.8,
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions[:, 2]),
            np.array([0.5, 5.3, 4.3]),
            atol=1e-3,
        )

    def test_zero_step_height_is_identity(self) -> None:
        r"""Verify a zero step height leaves the slab unchanged.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a zero step height
        leaves the slab unchanged.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_step_edge_field(
            slab,
            0.0,
            2.0,
            0.8,
        )

        np.testing.assert_allclose(
            np.asarray(modified.cart_positions),
            np.asarray(slab.cart_positions),
            atol=1e-6,
        )

    def test_grad_flows_through_step_height(self) -> None:
        r"""Check gradients flow through the step height.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check gradients
        flow through the step height.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(step_height: scalar_float) -> scalar_float:
            return apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[1, 2]

        grad_value: scalar_float = jax.grad(objective)(1.0)
        chex.assert_trees_all_close(float(grad_value), 0.5, atol=1e-3)

    def test_jit_compiles(self) -> None:
        r"""Verify apply_step_edge_field compiles under jit.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        apply_step_edge_field compiles under jit.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        compiled: Callable[..., Any] = jax.jit(
            lambda step_height: apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[:, 2]
        )

        result: Float[Array, "3"] = compiled(1.0)
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([0.5, 5.3, 4.3]),
            atol=1e-3,
        )

    def test_vmap_supports_batched_step_heights(self) -> None:
        r"""Check the step edge field maps over batched step heights.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check the step
        edge field maps over batched step heights.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def top_atom_z(step_height: scalar_float) -> scalar_float:
            return apply_step_edge_field(
                slab,
                step_height,
                2.0,
                0.8,
            ).cart_positions[1, 2]

        result: Float[Array, "3"] = jax.vmap(top_atom_z)(
            jnp.array([0.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            np.asarray(result),
            np.array([4.8, 5.05, 5.3]),
            atol=1e-3,
        )

    def test_rejects_zero_step_direction_during_gradient_probe(self) -> None:
        r"""A zero periodic-step direction is rejected under autodiff.

        Extended Summary
        ----------------
        Verifies that a zero ``step_direction_xy`` cannot reach the former
        double-``where`` normalization trap during ``jax.grad``.

        Notes
        -----
        It differentiates a surface height with respect to the direction at
        the exact zero boundary and checks the validation message, then checks
        finite derivatives for a valid nonzero direction.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(direction: Float[Array, "2"]) -> scalar_float:
            modified: CrystalStructure = apply_step_edge_field(
                slab,
                step_height_angstrom=1.0,
                corrugation_period_ang=2.0,
                surface_layer_depth_angstrom=0.8,
                step_direction_xy=direction,
            )
            return modified.cart_positions[1, 2]

        with pytest.raises(RuntimeError, match="step_direction_xy"):
            jax.grad(objective)(jnp.zeros(2))

        gradient: Float[Array, "2"] = jax.grad(objective)(
            jnp.array([1.0, 0.0])
        )
        chex.assert_tree_all_finite(gradient)


class TestApplyTwinWallField(chex.TestCase):
    """Tests for apply_twin_wall_field.

    :see: :func:`~rheedium.procs.apply_twin_wall_field`
    """

    def test_rotates_surface_atoms_smoothly_across_wall(self) -> None:
        r"""Twin wall should move top-surface in-plane coordinates.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Twin wall should
        move top-surface in-plane coordinates.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_twin_wall_field(
            slab=slab,
            twin_angle_deg=10.0,
            wall_position_angstrom=0.8,
            surface_layer_depth_angstrom=0.8,
            wall_normal_xy=jnp.array([1.0, 0.0]),
            wall_width_angstrom=0.5,
        )

        chex.assert_shape(modified.cart_positions, slab.cart_positions.shape)
        chex.assert_trees_all_close(
            modified.cart_positions[:, 3],
            slab.cart_positions[:, 3],
            atol=1e-12,
        )
        assert (
            float(
                jnp.linalg.norm(
                    modified.cart_positions[1:, :2]
                    - slab.cart_positions[1:, :2]
                )
            )
            > 0.0
        )

    def test_zero_angle_is_identity(self) -> None:
        r"""Zero twin angle preserves coordinates.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero twin angle
        preserves coordinates.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        modified: CrystalStructure = apply_twin_wall_field(
            slab=slab,
            twin_angle_deg=0.0,
            wall_position_angstrom=0.8,
            surface_layer_depth_angstrom=0.8,
        )

        chex.assert_trees_all_close(
            modified.cart_positions,
            slab.cart_positions,
            atol=1e-12,
        )

    def test_grad_flows_through_twin_angle(self) -> None:
        r"""Twin-wall structure builder remains differentiable.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Twin-wall
        structure builder remains differentiable.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(angle: scalar_float) -> scalar_float:
            modified: CrystalStructure = apply_twin_wall_field(
                slab=slab,
                twin_angle_deg=angle,
                wall_position_angstrom=0.8,
                surface_layer_depth_angstrom=0.8,
            )
            return jnp.sum(modified.cart_positions[:, 0])

        grad_value: scalar_float = jax.grad(objective)(5.0)
        assert jnp.isfinite(grad_value)
        assert grad_value != 0.0

    def test_rejects_zero_wall_normal_during_gradient_probe(self) -> None:
        r"""A zero twin-wall normal is rejected under autodiff.

        Extended Summary
        ----------------
        Verifies that a meaningless zero wall normal cannot enter norm
        differentiation or the retired additive-epsilon normalization.

        Notes
        -----
        It runs the audited ``jax.grad`` probe at the exact zero vector and
        checks the validation error, then verifies finite derivatives for a
        valid wall normal.
        """
        slab: CrystalStructure = _make_test_slab()

        def objective(normal: Float[Array, "2"]) -> scalar_float:
            modified: CrystalStructure = apply_twin_wall_field(
                slab=slab,
                twin_angle_deg=5.0,
                wall_position_angstrom=0.8,
                surface_layer_depth_angstrom=0.8,
                wall_normal_xy=normal,
            )
            return jnp.sum(modified.cart_positions[:, :2])

        with pytest.raises(RuntimeError, match="wall_normal_xy"):
            jax.grad(objective)(jnp.zeros(2))

        gradient: Float[Array, "2"] = jax.grad(objective)(
            jnp.array([1.0, 0.0])
        )
        chex.assert_tree_all_finite(gradient)


class TestIncoherentDomainAverage(chex.TestCase, parameterized.TestCase):
    """Tests for incoherent_domain_average function.

    :see: :func:`~rheedium.procs.incoherent_domain_average`
    """

    def test_single_domain_unchanged(self) -> None:
        r"""Single domain with f=1 should return pattern unchanged.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Single domain with
        f=1 should return pattern unchanged.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = jnp.ones((1, 8, 8)) * 5.0
        fractions: Float[Array, "..."] = jnp.array([1.0])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=pattern,
            domain_volume_fractions=fractions,
        )
        chex.assert_shape(result, (8, 8))
        chex.assert_trees_all_close(result, 5.0, atol=1e-6)

    def test_two_equal_domains_average(self) -> None:
        r"""50/50 mix should be the average of two patterns.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 50/50 mix should
        be the average of two patterns.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        p1: Float[Array, "..."] = jnp.ones((8, 8)) * 2.0
        p2: Float[Array, "..."] = jnp.ones((8, 8)) * 6.0
        patterns: Float[Array, "..."] = jnp.stack([p1, p2], axis=0)
        fractions: Float[Array, "..."] = jnp.array([0.5, 0.5])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        expected: Float[Array, "..."] = jnp.ones((8, 8)) * 4.0
        chex.assert_trees_all_close(result, expected, atol=1e-6)

    def test_output_shape(self) -> None:
        r"""Output should be (H, W) regardless of number of domains.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should be
        (H, W) regardless of number of domains.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        patterns: Float[Array, "..."] = jnp.ones((3, 16, 32))
        fractions: Float[Array, "..."] = jnp.array([0.5, 0.3, 0.2])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_shape(result, (16, 32))

    def test_output_nonnegative(self) -> None:
        r"""Result should be non-negative for non-negative inputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Result should be
        non-negative for non-negative inputs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        rng: np.random.Generator = np.random.default_rng(42)
        patterns: Float[Array, "domains height width"] = jnp.array(
            rng.uniform(0, 10, size=(4, 8, 8))
        )
        fractions: Float[Array, "..."] = jnp.array([0.25, 0.25, 0.25, 0.25])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        assert float(jnp.min(result)) >= -1e-10

    def test_weighted_sum_correct(self) -> None:
        r"""Weighted sum should equal manual calculation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Weighted sum
        should equal manual calculation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        p1: Float[Array, "..."] = jnp.ones((4, 4)) * 10.0
        p2: Float[Array, "..."] = jnp.ones((4, 4)) * 20.0
        patterns: Float[Array, "..."] = jnp.stack([p1, p2], axis=0)
        fractions: Float[Array, "..."] = jnp.array([0.3, 0.7])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        expected: Float[Array, "..."] = 0.3 * 10.0 + 0.7 * 20.0
        chex.assert_trees_all_close(
            result,
            jnp.ones((4, 4)) * expected,
            atol=1e-6,
        )

    def test_fractions_auto_normalized(self) -> None:
        r"""Non-unit fractions should be auto-normalized.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-unit fractions
        should be auto-normalized.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        p1: Float[Array, "..."] = jnp.ones((4, 4)) * 10.0
        patterns: Float[Array, "..."] = jnp.expand_dims(p1, axis=0)
        fractions: Float[Array, "..."] = jnp.array([2.0])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_trees_all_close(result, 10.0, atol=1e-6)

    def test_no_nan_or_inf(self) -> None:
        r"""Output should be finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should be
        finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        rng: np.random.Generator = np.random.default_rng(0)
        patterns: Float[Array, "domains height width"] = jnp.array(
            rng.uniform(0, 100, size=(5, 8, 8))
        )
        fractions: Float[Array, "..."] = jnp.array([0.1, 0.2, 0.3, 0.15, 0.25])
        result: Float[Array, "..."] = incoherent_domain_average(
            domain_patterns=patterns,
            domain_volume_fractions=fractions,
        )
        chex.assert_tree_all_finite(result)

    def test_zero_intensity_pixel_has_finite_gradient(self) -> None:
        r"""Zero-intensity domain pixels retain their mixture derivative.

        Extended Summary
        ----------------
        Verifies that incoherent domain averaging sums intensities directly
        instead of passing them through a singular ``sqrt(I)`` round-trip.

        Notes
        -----
        It differentiates the mixed detector pixel with respect to a zero
        input pixel in a domain of normalized weight ``0.25`` and checks the
        exact closed-form derivative.
        """
        patterns: Float[Array, "2 1 1"] = jnp.array([[[0.0]], [[4.0]]])

        def objective(pixel: scalar_float) -> scalar_float:
            varied_patterns: Float[Array, "2 1 1"] = patterns.at[0, 0, 0].set(
                pixel
            )
            return incoherent_domain_average(
                varied_patterns,
                jnp.array([0.25, 0.75]),
            )[0, 0]

        gradient: scalar_float = jax.grad(objective)(0.0)

        assert bool(jnp.isfinite(gradient))
        chex.assert_trees_all_close(gradient, 0.25, atol=1e-12)


class TestTwinWallToDistribution(chex.TestCase):
    """Tests for twin-wall Distribution producer.

    :see: :func:`~rheedium.procs.twin_wall_to_distribution`
    """

    def test_sub_coherence_twins_reduce_coherently(self) -> None:
        r"""Fine twin spacing selects coherent amplitude reduction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fine twin spacing
        selects coherent amplitude reduction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([-1.0, 1.0]),
            wall_positions_angstrom=jnp.array([0.0, 20.0]),
            twin_fractions=jnp.array([1.0, 3.0]),
            twin_spacing_angstrom=25.0,
            coherence_length_angstrom=50.0,
        )

        chex.assert_shape(dist.samples, (2, 2))
        chex.assert_trees_all_close(dist.weights, jnp.array([0.25, 0.75]))
        assert dist.reduction is ReductionMode.COHERENT
        assert dist.axis_id == "twins"

    def test_coherent_reduction_matches_manual_amplitude_sum(self) -> None:
        r"""Twin producer composes with the Layer-1 coherent reducer.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Twin producer
        composes with the Layer-1 coherent reducer.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([0.0, 2.0]),
            wall_positions_angstrom=jnp.array([0.0, 1.0]),
            twin_fractions=jnp.array([0.25, 0.75]),
            twin_spacing_angstrom=10.0,
            coherence_length_angstrom=20.0,
        )

        def amp(sample: Float[Array, "2"]) -> Complex[Array, "2 2"]:
            return jnp.ones((2, 2), dtype=jnp.complex128) * (1.0 + sample[0])

        reduced: Float[Array, "2 2"] = apply_distribution(dist, amp)
        manual_amplitude: scalar_float = 0.25 * 1.0 + 0.75 * 3.0
        chex.assert_trees_all_close(
            reduced,
            jnp.ones((2, 2)) * manual_amplitude**2,
        )


class TestStepEdgeToDistribution(chex.TestCase):
    """Tests for step-edge Distribution producer.

    :see: :func:`~rheedium.procs.step_edge_to_distribution`
    """

    def test_regular_sub_coherence_steps_reduce_coherently(self) -> None:
        r"""Regular terrace arrays can interfere coherently.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Regular terrace
        arrays can interfere coherently.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([1.0, 2.0]),
            terrace_widths_angstrom=jnp.array([20.0, 30.0]),
            line_azimuths_deg=jnp.array([0.0, 90.0]),
            step_fractions=jnp.array([2.0, 2.0]),
            coherence_length_angstrom=50.0,
            regular=True,
        )

        chex.assert_shape(dist.samples, (2, 3))
        chex.assert_trees_all_close(dist.weights, jnp.array([0.5, 0.5]))
        assert dist.reduction is ReductionMode.COHERENT

    def test_random_steps_reduce_incoherently(self) -> None:
        r"""Random step populations remain intensity mixtures.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Random step
        populations remain intensity mixtures.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        dist: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([1.0, 3.0]),
            terrace_widths_angstrom=jnp.array([20.0, 30.0]),
            line_azimuths_deg=jnp.array([0.0, 90.0]),
            step_fractions=jnp.array([0.25, 0.75]),
            coherence_length_angstrom=50.0,
            regular=False,
        )

        assert dist.reduction is ReductionMode.INCOHERENT

        def amp(sample: Float[Array, "3"]) -> Complex[Array, "2 2"]:
            return jnp.ones((2, 2), dtype=jnp.complex128) * sample[0]

        reduced: Float[Array, "2 2"] = apply_distribution(dist, amp)
        manual_intensity: scalar_float = 0.25 * 1.0**2 + 0.75 * 3.0**2
        chex.assert_trees_all_close(
            reduced,
            jnp.ones((2, 2)) * manual_intensity,
        )


class TestBindTwinWallDistribution(chex.TestCase):
    """Tests for twin-wall sample-to-structure binding.

    :see: :func:`~rheedium.procs.bind_twin_wall_distribution`
    """

    def test_builder_matches_direct_twin_wall_modifier(self) -> None:
        r"""Bound twin samples should call the structure modifier.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Bound twin samples
        should call the structure modifier.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        builder: Callable[[Float[Array, "2"]], CrystalStructure] = (
            bind_twin_wall_distribution(
                slab=slab,
                surface_layer_depth_angstrom=0.8,
                wall_width_angstrom=0.5,
            )
        )
        sample: Float[Array, "2"] = jnp.array([8.0, 0.8])

        bound: CrystalStructure = builder(sample)
        direct: CrystalStructure = apply_twin_wall_field(
            slab=slab,
            twin_angle_deg=sample[0],
            wall_position_angstrom=sample[1],
            surface_layer_depth_angstrom=0.8,
            wall_width_angstrom=0.5,
        )

        chex.assert_trees_all_close(
            bound.cart_positions,
            direct.cart_positions,
        )

    def test_layer1_reduction_can_use_bound_twin_structures(self) -> None:
        r"""A twin Distribution can build structures inside Layer 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A twin
        Distribution can build structures inside Layer 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        dist: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([0.0, 6.0]),
            wall_positions_angstrom=jnp.array([0.8, 0.8]),
            twin_fractions=jnp.array([0.25, 0.75]),
            twin_spacing_angstrom=10.0,
            coherence_length_angstrom=20.0,
        )
        builder: Callable[[Float[Array, "2"]], CrystalStructure] = (
            bind_twin_wall_distribution(
                slab=slab,
                surface_layer_depth_angstrom=0.8,
                wall_width_angstrom=0.5,
            )
        )

        def amp(sample: Float[Array, "2"]) -> Complex[Array, "1 1"]:
            modified: CrystalStructure = builder(sample)
            value: scalar_float = jnp.sum(modified.cart_positions[:, 0])
            return jnp.asarray([[value]], dtype=jnp.complex128)

        reduced: Float[Array, "1 1"] = apply_distribution(dist, amp)
        manual_amplitude: scalar_float = (
            0.25 * amp(dist.samples[0])[0, 0]
            + 0.75 * amp(dist.samples[1])[0, 0]
        )

        chex.assert_trees_all_close(
            reduced,
            jnp.abs(manual_amplitude) ** 2,
        )


class TestBindStepEdgeDistribution(chex.TestCase):
    """Tests for step-edge sample-to-structure binding.

    :see: :func:`~rheedium.procs.bind_step_edge_distribution`
    """

    def test_builder_matches_direct_step_modifier(self) -> None:
        r"""Bound step samples should call the structure modifier.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Bound step samples
        should call the structure modifier.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        builder: Callable[[Float[Array, "3"]], CrystalStructure] = (
            bind_step_edge_distribution(
                slab=slab,
                surface_layer_depth_angstrom=0.8,
            )
        )
        sample: Float[Array, "3"] = jnp.array([1.0, 2.0, 0.0])

        bound: CrystalStructure = builder(sample)
        direct: CrystalStructure = apply_step_edge_field(
            slab=slab,
            step_height_angstrom=sample[0],
            corrugation_period_ang=sample[1],
            surface_layer_depth_angstrom=0.8,
            step_direction_xy=jnp.array([1.0, 0.0]),
        )

        chex.assert_trees_all_close(
            bound.cart_positions,
            direct.cart_positions,
        )

    def test_layer1_reduction_can_use_bound_step_structures(self) -> None:
        r"""A step Distribution can build structures inside Layer 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A step
        Distribution can build structures inside Layer 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_procs.test_surface_modifier``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slab: CrystalStructure = _make_test_slab()
        dist: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([1.0, 3.0]),
            terrace_widths_angstrom=jnp.array([2.0, 2.0]),
            line_azimuths_deg=jnp.array([0.0, 0.0]),
            step_fractions=jnp.array([0.25, 0.75]),
            coherence_length_angstrom=0.5,
            regular=False,
        )
        builder: Callable[[Float[Array, "3"]], CrystalStructure] = (
            bind_step_edge_distribution(
                slab=slab,
                surface_layer_depth_angstrom=0.8,
            )
        )

        def amp(sample: Float[Array, "3"]) -> Complex[Array, "1 1"]:
            modified: CrystalStructure = builder(sample)
            value: scalar_float = modified.cart_positions[1, 2]
            return jnp.asarray([[value]], dtype=jnp.complex128)

        reduced: Float[Array, "1 1"] = apply_distribution(dist, amp)
        manual_intensity: scalar_float = (
            0.25 * jnp.abs(amp(dist.samples[0])[0, 0]) ** 2
            + 0.75 * jnp.abs(amp(dist.samples[1])[0, 0]) ** 2
        )

        chex.assert_trees_all_close(reduced, manual_intensity)
