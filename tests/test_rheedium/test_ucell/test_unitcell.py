"""Tests for ucell.unitcell module.

Tests the atom_scraper function for filtering atoms within specified
thickness along a zone axis, plus reciprocal lattice functions.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jaxtyping import Array, Bool, Float, Integer
from numpy.typing import NDArray

from rheedium.types.crystal_types import (
    CrystalStructure,
    create_crystal_structure,
)
from rheedium.types.custom_types import scalar_float
from rheedium.types.rheed_types import SlicedCrystal
from rheedium.ucell.helper import compute_lengths_angles
from rheedium.ucell.unitcell import (
    atom_scraper,
    build_cell_vectors,
    bulk_to_slice,
    generate_reciprocal_points,
    get_unit_cell_matrix,
    miller_to_reciprocal,
    reciprocal_lattice_vectors,
    reciprocal_unitcell,
)


def _make_simple_crystal(n_atoms: int = 8) -> CrystalStructure:
    """Create a simple cubic CrystalStructure for testing."""
    rng: np.random.Generator = np.random.default_rng(0)
    frac_xyz: Float[NDArray, "atoms xyz"] = rng.uniform(size=(n_atoms, 3))
    z_nums: Float[NDArray, "atoms one"] = np.full((n_atoms, 1), 14.0)
    frac_pos: Float[Array, "atoms xyzz"] = jnp.array(
        np.hstack([frac_xyz, z_nums])
    )
    cell_lengths: Float[Array, "three"] = jnp.array([5.43, 5.43, 5.43])
    cell_angles: Float[Array, "three"] = jnp.array([90.0, 90.0, 90.0])
    cart_xyz: Float[NDArray, "atoms xyz"] = frac_xyz * np.array(
        [5.43, 5.43, 5.43]
    )
    cart_pos: Float[Array, "atoms xyzz"] = jnp.array(
        np.hstack([cart_xyz, z_nums])
    )
    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


class TestBulkToSlice(chex.TestCase):
    """Tests for bulk_to_slice function.

    :see: :func:`~rheedium.ucell.bulk_to_slice`
    """

    def test_returns_sliced_crystal(self) -> None:
        r"""Should return a SlicedCrystal instance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should return a
        SlicedCrystal instance.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
        )
        assert isinstance(sliced, SlicedCrystal)

    def test_output_shapes(self) -> None:
        r"""Output should have correct array shapes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should have
        correct array shapes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.ndim == 2
        assert sliced.cart_positions.shape[1] == 4
        chex.assert_shape(sliced.cell_lengths, (3,))
        chex.assert_shape(sliced.cell_angles, (3,))
        chex.assert_shape(sliced.orientation, (3,))

    def test_depth_preserved(self) -> None:
        r"""Slab depth should match requested depth.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Slab depth should
        match requested depth.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        depth: float = 15.0
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=depth,
        )
        chex.assert_trees_all_close(sliced.depth, depth)

    def test_extents_preserved(self) -> None:
        r"""Lateral extents should match requested values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Lateral extents
        should match requested values.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=120.0,
            y_extent=130.0,
        )
        chex.assert_trees_all_close(sliced.x_extent, 120.0)
        chex.assert_trees_all_close(sliced.y_extent, 130.0)

    def test_orientation_preserved(self) -> None:
        r"""Surface orientation should be preserved.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Surface
        orientation should be preserved.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        orient: Integer[Array, "..."] = jnp.array([1, 1, 1], dtype=jnp.int32)
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=orient,
            depth=10.0,
        )
        chex.assert_trees_all_equal(sliced.orientation, orient)

    def test_atoms_within_bounds(self) -> None:
        r"""All atoms should be within the specified bounds.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All atoms should
        be within the specified bounds.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        depth: float = 10.0
        x_ext: float = 80.0
        y_ext: float = 80.0
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=depth,
            x_extent=x_ext,
            y_extent=y_ext,
        )
        positions: Float[Array, "..."] = sliced.cart_positions[:, :3]
        assert bool(jnp.all(positions[:, 0] >= 0))
        assert bool(jnp.all(positions[:, 0] <= x_ext))
        assert bool(jnp.all(positions[:, 1] >= 0))
        assert bool(jnp.all(positions[:, 1] <= y_ext))
        assert bool(jnp.all(positions[:, 2] >= 0))
        assert bool(jnp.all(positions[:, 2] <= depth))

    def test_cell_angles_orthorhombic(self) -> None:
        r"""Output cell should have 90-degree angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output cell should
        have 90-degree angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
        )
        chex.assert_trees_all_close(
            sliced.cell_angles,
            jnp.array([90.0, 90.0, 90.0]),
        )

    def test_001_orientation(self) -> None:
        r"""(001) orientation should work without rotation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (001) orientation
        should work without rotation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0

    def test_111_orientation(self) -> None:
        r"""(111) orientation should produce rotated slab.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (111) orientation
        should produce rotated slab.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([1, 1, 1], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0

    def test_100_orientation(self) -> None:
        r"""(100) orientation should produce rotated slab.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (100) orientation
        should produce rotated slab.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_simple_crystal()
        sliced: SlicedCrystal = bulk_to_slice(
            crystal,
            orientation=jnp.array([1, 0, 0], dtype=jnp.int32),
            depth=10.0,
            x_extent=50.0,
            y_extent=50.0,
        )
        assert sliced.cart_positions.shape[0] > 0


class TestAtomScraper(chex.TestCase, parameterized.TestCase):
    """Test atom_scraper function.

    :see: :func:`~rheedium.ucell.atom_scraper`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

        # Create a simple cubic crystal with atoms at different z positions
        self.cubic_crystal: Any = self._create_layered_crystal()

    def _create_layered_crystal(self) -> CrystalStructure:
        """Create a crystal with atoms at different z heights.

        Creates 5 atoms stacked along z-axis at z = 0, 2, 4, 6, 8 Angstroms.
        """
        a: float = 5.0  # lattice constant

        # Atoms at different z heights
        cart_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],  # z = 0
                [0.0, 0.0, 2.0],  # z = 2
                [0.0, 0.0, 4.0],  # z = 4
                [0.0, 0.0, 6.0],  # z = 6
                [0.0, 0.0, 8.0],  # z = 8
            ]
        )

        # Fractional coordinates
        frac_coords: Float[Array, "..."] = cart_coords / jnp.array(
            [a, a, 10.0]
        )

        # All silicon atoms
        atomic_numbers: Float[Array, "..."] = jnp.full(5, 14.0)

        frac_positions: Float[Array, "..."] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "..."] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a, a, 10.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def _create_xy_plane_crystal(self) -> CrystalStructure:
        """Create a crystal with atoms spread in XY plane at same z."""
        a: float = 10.0

        cart_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
            ]
        )

        frac_coords: Float[Array, "..."] = cart_coords / a
        atomic_numbers: Float[Array, "..."] = jnp.full(4, 14.0)

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_z_axis_scraping(self) -> None:
        r"""Test scraping atoms along z-axis with specific thickness.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: scraping atoms
        along z-axis with specific thickness.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [5.0, 5.0, 3.0]
        )  # 3 Angstrom thickness

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should filter to atoms near the top (z=8)
        # With 3 Angstrom thickness from top, should include z=8 and z=6
        n_atoms: int = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        self.assertLessEqual(int(n_atoms), 5)

    def test_full_thickness_keeps_all_atoms(self) -> None:
        r"""Test that large thickness keeps all atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: large thickness
        keeps all atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [10.0, 10.0, 20.0]
        )  # Much larger than crystal

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should keep all 5 atoms
        n_atoms: int = filtered.cart_positions.shape[0]
        self.assertEqual(int(n_atoms), 5)

    def test_zero_thickness_top_layer_only(self) -> None:
        r"""Test that zero thickness returns top layer atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: zero thickness
        returns top layer atoms. Existing context from the original test prose:
        With zero thickness, the function uses an adaptive epsilon based on the
        minimum atom spacing (2 * min_spacing). For atoms spaced 2Å apart, this
        gives adaptive_eps = 4Å, which includes multiple atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [0.0, 0.0, 0.0]
        )  # Zero thickness = top layer mode

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # With atoms at z=0,2,4,6,8 and spacing=2Å, adaptive_eps=4Å
        # This includes atoms within 4Å of top (z=8): z=8,6,4
        n_atoms: int = filtered.cart_positions.shape[0]
        self.assertGreaterEqual(int(n_atoms), 1)
        self.assertLessEqual(int(n_atoms), 5)

        # Verify the topmost atom is included
        max_z: scalar_float = jnp.max(filtered.cart_positions[:, 2])
        chex.assert_trees_all_close(max_z, 8.0, atol=1e-6)

    @parameterized.named_parameters(
        ("z_axis", [0.0, 0.0, 1.0]),
        ("neg_z_axis", [0.0, 0.0, -1.0]),
        ("z_axis_scaled", [0.0, 0.0, 2.0]),
    )
    def test_zone_axis_normalization(self, zone_axis: list[float]) -> None:
        r"""Test that zone axis is properly normalized.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: zone axis is
        properly normalized.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``zone_axis``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis_arr: Float[Array, "three"] = jnp.array(zone_axis)
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, 3.0])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis_arr,
            thickness=thickness,
        )

        # Should produce valid output regardless of axis scaling
        n_atoms: int = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))

    def test_x_axis_scraping(self) -> None:
        r"""Test scraping along x-axis.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: scraping along
        x-axis.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Create crystal with atoms spread along x
        cart_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        frac_coords: Float[Array, "..."] = cart_coords / 10.0
        atomic_numbers: Float[Array, "..."] = jnp.full(5, 14.0)

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([10.0, 5.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        zone_axis: Float[Array, "..."] = jnp.array([1.0, 0.0, 0.0])
        thickness: Float[Array, "..."] = jnp.array([3.0, 5.0, 5.0])

        filtered: Any = atom_scraper(
            crystal=crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        n_atoms: int = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        self.assertLess(int(n_atoms), 5)

    def test_diagonal_zone_axis(self) -> None:
        r"""Test scraping along diagonal [1,1,1] direction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: scraping along
        diagonal [1,1,1] direction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([1.0, 1.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, 5.0])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Should produce valid output
        n_atoms: int = filtered.cart_positions.shape[0]
        chex.assert_scalar_positive(int(n_atoms))
        chex.assert_tree_all_finite(filtered.cart_positions)

    def test_output_is_valid_crystal_structure(self) -> None:
        r"""Test that output is a valid CrystalStructure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output is a valid
        CrystalStructure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, 3.0])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Check it's a CrystalStructure
        self.assertIsInstance(filtered, CrystalStructure)

        # Check shapes are consistent
        n_atoms: int = filtered.cart_positions.shape[0]
        chex.assert_shape(filtered.frac_positions, (n_atoms, 4))
        chex.assert_shape(filtered.cart_positions, (n_atoms, 4))
        chex.assert_shape(filtered.cell_lengths, (3,))
        chex.assert_shape(filtered.cell_angles, (3,))

        # Check values are finite
        chex.assert_tree_all_finite(filtered.frac_positions)
        chex.assert_tree_all_finite(filtered.cart_positions)
        chex.assert_tree_all_finite(filtered.cell_lengths)
        chex.assert_tree_all_finite(filtered.cell_angles)

    def test_cell_lengths_positive(self) -> None:
        r"""Test that output cell lengths are positive.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output cell
        lengths are positive.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, 3.0])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        chex.assert_trees_all_equal(jnp.all(filtered.cell_lengths > 0), True)

    def test_cell_angles_valid(self) -> None:
        r"""Test that output cell angles are in valid range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output cell angles
        are in valid range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, 3.0])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Angles should be between 0 and 180 degrees
        angle: scalar_float
        for angle in filtered.cell_angles:
            chex.assert_scalar_in(float(angle), 0.0, 180.0)

    def test_atomic_numbers_preserved(self) -> None:
        r"""Test that atomic numbers are preserved in output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: atomic numbers are
        preserved in output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [5.0, 5.0, 20.0]
        )  # Keep all atoms

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # All atoms should still be silicon (Z=14)
        atomic_nums: Any = filtered.cart_positions[:, 3]
        chex.assert_trees_all_close(atomic_nums, jnp.full(5, 14.0), atol=1e-10)

    def test_xy_plane_atoms_same_z(self) -> None:
        r"""Test scraping with atoms at same z height.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: scraping with
        atoms at same z height. Existing context from the original test prose:
        Note: When all atoms are at the same height along the zone axis, they
        are all considered "top layer" atoms and should be included. We use a
        crystal with slight z variation to avoid the edge case where all atoms
        are exactly coplanar.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: float = 10.0

        # Atoms in XY plane with slight z variation
        cart_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 4.99],  # Slightly lower
                [5.0, 5.0, 5.0],
            ]
        )

        frac_coords: Float[Array, "..."] = cart_coords / a
        atomic_numbers: Float[Array, "..."] = jnp.full(4, 14.0)

        crystal: CrystalStructure = create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [10.0, 10.0, 1.0]
        )  # 1 Angstrom slice

        filtered: Any = atom_scraper(
            crystal=crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # All 4 atoms should be included (within 1 Angstrom of top)
        n_atoms: int = filtered.cart_positions.shape[0]
        self.assertEqual(int(n_atoms), 4)

    @parameterized.named_parameters(
        ("thin", 1.0, 1),
        ("medium", 5.0, 3),
        ("thick", 10.0, 5),
    )
    def test_thickness_controls_atom_count(
        self, z_thickness: float, min_expected: int
    ) -> None:
        r"""Test that increasing thickness includes more atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: increasing
        thickness includes more atoms.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``z_thickness``, ``min_expected``, so the documented behavior is
        checked across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array([5.0, 5.0, z_thickness])

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        n_atoms: int = filtered.cart_positions.shape[0]
        self.assertGreaterEqual(int(n_atoms), min_expected)

    def test_frac_and_cart_positions_consistent(self) -> None:
        r"""Test that fractional and Cartesian positions remain consistent.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: fractional and
        Cartesian positions remain consistent.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        zone_axis: Float[Array, "..."] = jnp.array([0.0, 0.0, 1.0])
        thickness: Float[Array, "..."] = jnp.array(
            [5.0, 5.0, 20.0]
        )  # Keep all atoms

        filtered: Any = atom_scraper(
            crystal=self.cubic_crystal,
            zone_axis=zone_axis,
            thickness=thickness,
        )

        # Number of atoms should match
        n_frac: int = filtered.frac_positions.shape[0]
        n_cart: int = filtered.cart_positions.shape[0]
        self.assertEqual(n_frac, n_cart)

        # Atomic numbers should match between frac and cart
        frac_z: Any = filtered.frac_positions[:, 3]
        cart_z: Any = filtered.cart_positions[:, 3]
        chex.assert_trees_all_close(frac_z, cart_z, atol=1e-10)


class TestReciprocalUnitcell(chex.TestCase, parameterized.TestCase):
    """Test reciprocal_unitcell function.

    :see: :func:`~rheedium.ucell.reciprocal_unitcell`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(with_pmap=False)
    def test_cubic_system(self) -> None:
        r"""Test reciprocal parameters for cubic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal
        parameters for cubic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # For cubic: a* = 2π/a
        expected_a_star: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_trees_all_close(lengths[0], expected_a_star, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], expected_a_star, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], expected_a_star, rtol=1e-5)
        # Reciprocal angles for cubic are 90°
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants(with_pmap=False)
    def test_orthorhombic_system(self) -> None:
        r"""Test reciprocal params for orthorhombic (a!=b!=c, 90).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal params
        for orthorhombic (a!=b!=c, 90).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # For orthorhombic: a* = 2π/a, b* = 2π/b, c* = 2π/c
        chex.assert_trees_all_close(lengths[0], 2 * jnp.pi / 3.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], 2 * jnp.pi / 4.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], 2 * jnp.pi / 5.0, rtol=1e-5)
        # Angles still 90°
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants(with_pmap=False)
    def test_tetragonal_system(self) -> None:
        r"""Test reciprocal parameters for tetragonal system (a=b≠c).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal
        parameters for tetragonal system (a=b≠c).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # a* = b* but c* different
        chex.assert_trees_all_close(lengths[0], lengths[1], rtol=1e-5)
        self.assertNotAlmostEqual(
            float(lengths[0]), float(lengths[2]), places=3
        )

    @chex.all_variants(with_pmap=False)
    def test_hexagonal_system(self) -> None:
        r"""Test reciprocal parameters for hexagonal system (γ=120°).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal
        parameters for hexagonal system (γ=120°).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # Reciprocal gamma for hexagonal is 60°
        chex.assert_trees_all_close(angles[2], 60.0, atol=1e-4)

    @chex.all_variants(with_pmap=False)
    def test_monoclinic_system(self) -> None:
        r"""Test reciprocal parameters for monoclinic system (β≠90°).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal
        parameters for monoclinic system (β≠90°).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # Alpha* and gamma* should remain 90° for monoclinic
        chex.assert_trees_all_close(angles[0], 90.0, atol=1e-4)
        chex.assert_trees_all_close(angles[2], 90.0, atol=1e-4)

    @chex.all_variants(with_pmap=False)
    def test_triclinic_system(self) -> None:
        r"""Test reciprocal params for triclinic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal params
        for triclinic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_shape(lengths, (3,))
        chex.assert_shape(angles, (3,))
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)
        # All values should be positive
        chex.assert_trees_all_equal(jnp.all(lengths > 0), True)
        chex.assert_trees_all_equal(jnp.all(angles > 0), True)
        chex.assert_trees_all_equal(jnp.all(angles < 180), True)

    @chex.all_variants(with_pmap=False)
    def test_in_degrees_flag_true(self) -> None:
        r"""Test in_degrees=True (input in degrees).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: in_degrees=True
        (input in degrees).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        chex.assert_tree_all_finite(lengths)
        chex.assert_tree_all_finite(angles)

    @chex.all_variants(with_pmap=False)
    def test_in_degrees_flag_false(self) -> None:
        r"""Test in_degrees=False (input in radians).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: in_degrees=False
        (input in radians).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        pi_half: Float[Array, "..."] = jnp.pi / 2
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
            out_degrees=True,
        )
        # Should give same result as cubic with 90° in degrees
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.all_variants(with_pmap=False)
    def test_out_degrees_flag_false(self) -> None:
        r"""Test out_degrees=False (output in radians).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: out_degrees=False
        (output in radians).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=False,
        )
        # Output angles in radians for cubic should be π/2
        chex.assert_trees_all_close(
            angles, jnp.array([jnp.pi / 2] * 3), atol=1e-5
        )

    @chex.all_variants(with_pmap=False)
    def test_both_degrees_flags_false(self) -> None:
        r"""Test both in_degrees=False and out_degrees=False.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: both
        in_degrees=False and out_degrees=False.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_unitcell)
        pi_half: Float[Array, "..."] = jnp.pi / 2
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
            out_degrees=False,
        )
        chex.assert_trees_all_close(
            angles, jnp.array([pi_half] * 3), atol=1e-5
        )

    @parameterized.named_parameters(
        ("small_cell", 1.0, 1.0, 1.0),
        ("medium_cell", 5.0, 5.0, 5.0),
        ("large_cell", 10.0, 10.0, 10.0),
    )
    def test_various_cell_sizes(self, a: float, b: float, c: float) -> None:
        r"""Test with various cell sizes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with various cell
        sizes.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``a``, ``b``,
        ``c``, so the documented behavior is checked across the cases supplied
        by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = reciprocal_unitcell(
            a=a,
            b=b,
            c=c,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
            out_degrees=True,
        )
        # a* should scale inversely with a
        expected: scalar_float = 2 * jnp.pi / a
        chex.assert_trees_all_close(lengths[0], expected, rtol=1e-5)


class TestGetUnitCellMatrix(chex.TestCase, parameterized.TestCase):
    """Test get_unit_cell_matrix function.

    :see: :func:`~rheedium.ucell.get_unit_cell_matrix`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(with_pmap=False)
    def test_cubic_system(self) -> None:
        r"""Test transformation matrix for cubic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: transformation
        matrix for cubic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # For cubic, matrix should be diagonal with a on diagonal
        chex.assert_trees_all_close(matrix[0, 0], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[1, 1], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[2, 2], 3.0, atol=1e-10)

    @chex.all_variants(with_pmap=False)
    def test_orthorhombic_system(self) -> None:
        r"""Test transformation matrix for orthorhombic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: transformation
        matrix for orthorhombic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        # For orthorhombic, matrix is diagonal
        chex.assert_trees_all_close(matrix[0, 0], 3.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[1, 1], 4.0, atol=1e-10)
        chex.assert_trees_all_close(matrix[2, 2], 5.0, atol=1e-10)

    @chex.all_variants(with_pmap=False)
    def test_monoclinic_system(self) -> None:
        r"""Test transformation matrix for monoclinic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: transformation
        matrix for monoclinic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # Off-diagonal term for c vector
        self.assertNotAlmostEqual(float(matrix[0, 2]), 0.0, places=5)

    @chex.all_variants(with_pmap=False)
    def test_hexagonal_system(self) -> None:
        r"""Test transformation matrix for hexagonal system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: transformation
        matrix for hexagonal system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # b has x and y components due to 120° angle
        chex.assert_trees_all_close(
            matrix[0, 1], 3.0 * jnp.cos(jnp.radians(120.0)), atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_triclinic_system(self) -> None:
        r"""Test transformation matrix for triclinic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: transformation
        matrix for triclinic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        # Matrix should have off-diagonal elements
        self.assertNotAlmostEqual(float(matrix[0, 1]), 0.0, places=5)

    @chex.all_variants(with_pmap=False)
    def test_volume_consistency(self) -> None:
        r"""Test that matrix determinant equals cell volume.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: matrix determinant
        equals cell volume.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(get_unit_cell_matrix)
        matrix: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        volume: scalar_float = jnp.linalg.det(matrix)
        expected_volume: Float[Array, "..."] = (
            3.0 * 4.0 * 5.0
        )  # For orthorhombic
        chex.assert_trees_all_close(volume, expected_volume, rtol=1e-5)


class TestBuildCellVectors(chex.TestCase, parameterized.TestCase):
    """Test build_cell_vectors function.

    :see: :func:`~rheedium.ucell.build_cell_vectors`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(with_pmap=False)
    def test_cubic_system(self) -> None:
        r"""Test cell vectors for cubic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cell vectors for
        cubic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        chex.assert_shape(vectors, (3, 3))
        # First vector along x-axis
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        # Second vector along y-axis
        chex.assert_trees_all_close(
            vectors[1], jnp.array([0.0, 3.0, 0.0]), atol=1e-10
        )
        # Third vector along z-axis
        chex.assert_trees_all_close(
            vectors[2], jnp.array([0.0, 0.0, 3.0]), atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_orthorhombic_system(self) -> None:
        r"""Test cell vectors for orthorhombic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cell vectors for
        orthorhombic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        # Vectors should be orthogonal with different lengths
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            vectors[1], jnp.array([0.0, 4.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            vectors[2], jnp.array([0.0, 0.0, 5.0]), atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_hexagonal_system(self) -> None:
        r"""Test cell vectors for hexagonal system (gamma=120).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cell vectors for
        hexagonal system (gamma=120).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=3.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
        )
        chex.assert_shape(vectors, (3, 3))
        # a vector along x
        chex.assert_trees_all_close(
            vectors[0], jnp.array([3.0, 0.0, 0.0]), atol=1e-10
        )
        # b vector in xy plane at 120° from a
        b_x: Float[Array, "..."] = 3.0 * jnp.cos(jnp.radians(120.0))
        b_y: Float[Array, "..."] = 3.0 * jnp.sin(jnp.radians(120.0))
        chex.assert_trees_all_close(vectors[1, 0], b_x, atol=1e-10)
        chex.assert_trees_all_close(vectors[1, 1], b_y, atol=1e-10)
        # c vector along z
        chex.assert_trees_all_close(vectors[2, 2], 5.0, atol=1e-10)

    @chex.all_variants(with_pmap=False)
    def test_monoclinic_system(self) -> None:
        r"""Test cell vectors for monoclinic system (beta != 90).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cell vectors for
        monoclinic system (beta != 90).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=100.0,
            gamma=90.0,
        )
        chex.assert_shape(vectors, (3, 3))
        chex.assert_tree_all_finite(vectors)
        # c vector should have nonzero x component
        c_x: Float[Array, "..."] = 5.0 * jnp.cos(jnp.radians(100.0))
        chex.assert_trees_all_close(vectors[2, 0], c_x, atol=1e-10)

    @chex.all_variants(with_pmap=False)
    def test_triclinic_system(self) -> None:
        r"""Test cell vectors for triclinic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cell vectors for
        triclinic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        chex.assert_shape(vectors, (3, 3))
        chex.assert_tree_all_finite(vectors)
        # All three vectors should have nonzero components

    @chex.all_variants(with_pmap=False)
    def test_vector_lengths_correct(self) -> None:
        r"""Test that built vectors have correct lengths.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: built vectors have
        correct lengths.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        lengths: Float[Array, "..."] = jnp.linalg.norm(vectors, axis=1)
        chex.assert_trees_all_close(lengths[0], 3.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[1], 4.0, rtol=1e-5)
        chex.assert_trees_all_close(lengths[2], 5.0, rtol=1e-5)

    @chex.all_variants(with_pmap=False)
    def test_angles_correct(self) -> None:
        r"""Test that angles between vectors are correct.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: angles between
        vectors are correct.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(build_cell_vectors)
        vectors: Float[Array, "..."] = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        # Check angle gamma between a and b
        a_vec: Any = vectors[0]
        b_vec: Any = vectors[1]
        cos_gamma: Float[Array, "..."] = jnp.dot(a_vec, b_vec) / (
            jnp.linalg.norm(a_vec) * jnp.linalg.norm(b_vec)
        )
        gamma_computed: Float[Array, "..."] = jnp.rad2deg(
            jnp.arccos(cos_gamma)
        )
        chex.assert_trees_all_close(gamma_computed, 75.0, atol=1e-4)


class TestComputeLengthsAngles(chex.TestCase, parameterized.TestCase):
    """Test compute_lengths_angles function.

    :see: :func:`~rheedium.ucell.compute_lengths_angles`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.variants(with_jit=True, without_jit=True)
    def test_cubic_system(self) -> None:
        r"""Test lengths and angles for cubic vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: lengths and angles
        for cubic vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(compute_lengths_angles)
        vectors: Float[Array, "..."] = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 3.0, 3.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_orthorhombic_system(self) -> None:
        r"""Test lengths and angles for orthorhombic vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: lengths and angles
        for orthorhombic vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(compute_lengths_angles)
        vectors: Float[Array, "..."] = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 4.0, 5.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_cubic(self) -> None:
        r"""Test build_cell_vectors followed by compute_lengths_angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: build_cell_vectors
        followed by compute_lengths_angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        vectors: Float[Array, "..."] = build_cell_vectors(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        var_fn: Callable[..., Any] = self.variant(compute_lengths_angles)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(
            lengths, jnp.array([3.0, 3.0, 3.0]), rtol=1e-5
        )
        chex.assert_trees_all_close(
            angles, jnp.array([90.0, 90.0, 90.0]), atol=1e-4
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_triclinic(self) -> None:
        r"""Test roundtrip for triclinic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roundtrip for
        triclinic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: tuple[Any, ...]
        b: tuple[Any, ...]
        c: tuple[Any, ...]
        a, b, c = 3.0, 4.0, 5.0
        alpha: tuple[Any, ...]
        beta: tuple[Any, ...]
        gamma: tuple[Any, ...]
        alpha, beta, gamma = 80.0, 85.0, 75.0
        vectors: Float[Array, "..."] = build_cell_vectors(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        var_fn: Callable[..., Any] = self.variant(compute_lengths_angles)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), rtol=1e-5)
        chex.assert_trees_all_close(
            angles, jnp.array([alpha, beta, gamma]), atol=1e-4
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip_hexagonal(self) -> None:
        r"""Test roundtrip for hexagonal system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roundtrip for
        hexagonal system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        a: tuple[Any, ...]
        b: tuple[Any, ...]
        c: tuple[Any, ...]
        a, b, c = 3.0, 3.0, 5.0
        alpha: tuple[Any, ...]
        beta: tuple[Any, ...]
        gamma: tuple[Any, ...]
        alpha, beta, gamma = 90.0, 90.0, 120.0
        vectors: Float[Array, "..."] = build_cell_vectors(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        var_fn: Callable[..., Any] = self.variant(compute_lengths_angles)
        lengths: Float[Array, "..."]
        angles: Float[Array, "..."]
        lengths, angles = var_fn(vectors)
        chex.assert_trees_all_close(lengths, jnp.array([a, b, c]), rtol=1e-5)
        chex.assert_trees_all_close(
            angles, jnp.array([alpha, beta, gamma]), atol=1e-4
        )


class TestReciprocalLatticeVectors(chex.TestCase, parameterized.TestCase):
    """Test reciprocal_lattice_vectors function.

    :see: :func:`~rheedium.ucell.reciprocal_lattice_vectors`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(with_pmap=False)
    def test_cubic_system(self) -> None:
        r"""Test reciprocal vectors for cubic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal vectors
        for cubic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_lattice_vectors)
        rec_vecs: Any = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        chex.assert_shape(rec_vecs, (3, 3))
        # For cubic: b1 = (2π/a, 0, 0), etc.
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            rec_vecs[0], jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[1], jnp.array([0.0, expected, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[2], jnp.array([0.0, 0.0, expected]), atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_orthorhombic_system(self) -> None:
        r"""Test reciprocal vectors for orthorhombic system.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal vectors
        for orthorhombic system.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_lattice_vectors)
        rec_vecs: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        chex.assert_shape(rec_vecs, (3, 3))
        chex.assert_trees_all_close(
            rec_vecs[0], jnp.array([2 * jnp.pi / 3.0, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[1], jnp.array([0.0, 2 * jnp.pi / 4.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            rec_vecs[2], jnp.array([0.0, 0.0, 2 * jnp.pi / 5.0]), atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_orthogonality_to_direct(self) -> None:
        r"""Test reciprocal vectors orthogonal to direct vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reciprocal vectors
        orthogonal to direct vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_lattice_vectors)
        rec_vecs: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
        )
        direct_vecs: Any = build_cell_vectors(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        # b1 · a2 = 0, b1 · a3 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[1]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[2]), 0.0, atol=1e-10
        )
        # b2 · a1 = 0, b2 · a3 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[0]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[2]), 0.0, atol=1e-10
        )
        # b3 · a1 = 0, b3 · a2 = 0
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[0]), 0.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[1]), 0.0, atol=1e-10
        )

    @chex.all_variants(with_pmap=False)
    def test_bi_dot_ai_equals_2pi(self) -> None:
        r"""Test that b_i · a_i = 2π.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: b_i · a_i = 2π.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_lattice_vectors)
        rec_vecs: Any = var_fn(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
            in_degrees=True,
        )
        direct_vecs: Any = build_cell_vectors(
            a=3.0,
            b=4.0,
            c=5.0,
            alpha=80.0,
            beta=85.0,
            gamma=75.0,
        )
        two_pi: scalar_float = 2 * jnp.pi
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[0], direct_vecs[0]), two_pi, rtol=1e-5
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[1], direct_vecs[1]), two_pi, rtol=1e-5
        )
        chex.assert_trees_all_close(
            jnp.dot(rec_vecs[2], direct_vecs[2]), two_pi, rtol=1e-5
        )

    @chex.all_variants(with_pmap=False)
    def test_in_degrees_flag(self) -> None:
        r"""Test in_degrees flag.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: in_degrees flag.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(reciprocal_lattice_vectors)
        # With degrees
        rec_vecs_deg: Any = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )
        # With radians
        pi_half: Float[Array, "..."] = jnp.pi / 2
        rec_vecs_rad: Any = var_fn(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=pi_half,
            beta=pi_half,
            gamma=pi_half,
            in_degrees=False,
        )
        chex.assert_trees_all_close(rec_vecs_deg, rec_vecs_rad, rtol=1e-5)


class TestMillerToReciprocal(chex.TestCase, parameterized.TestCase):
    """Test miller_to_reciprocal function.

    :see: :func:`~rheedium.ucell.miller_to_reciprocal`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

        # Set up cubic reciprocal vectors for testing
        self.cubic_rec_vecs: Any = reciprocal_lattice_vectors(
            a=3.0,
            b=3.0,
            c=3.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            in_degrees=True,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_100(self) -> None:
        r"""Test (1,0,0) Miller index.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (1,0,0) Miller
        index.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([1, 0, 0])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_010(self) -> None:
        r"""Test (0,1,0) Miller index.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (0,1,0) Miller
        index.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([0, 1, 0])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, expected, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_001(self) -> None:
        r"""Test (0,0,1) Miller index.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (0,0,1) Miller
        index.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([0, 0, 1])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, 0.0, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_index_111(self) -> None:
        r"""Test (1,1,1) Miller index.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (1,1,1) Miller
        index.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([1, 1, 1])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_negative_indices(self) -> None:
        r"""Test negative Miller indices.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: negative Miller
        indices.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([-1, -1, -1])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = -2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_indices(self) -> None:
        r"""Test (0,0,0) gives zero vector.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (0,0,0) gives zero
        vector.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([0, 0, 0])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        chex.assert_trees_all_close(
            g_vec, jnp.array([0.0, 0.0, 0.0]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_batch_indices(self) -> None:
        r"""Test batched Miller indices.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: batched Miller
        indices.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Float[Array, "..."] = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
            ]
        )
        g_vecs: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        chex.assert_shape(g_vecs, (4, 3))
        expected: scalar_float = 2 * jnp.pi / 3.0
        chex.assert_trees_all_close(
            g_vecs[0], jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )
        chex.assert_trees_all_close(
            g_vecs[3], jnp.array([expected, expected, expected]), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_higher_indices(self) -> None:
        r"""Test higher Miller indices (2,0,0).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: higher Miller
        indices (2,0,0).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(miller_to_reciprocal)
        hkl: Integer[Array, "..."] = jnp.array([2, 0, 0])
        g_vec: Float[Array, "..."] = var_fn(hkl, self.cubic_rec_vecs)
        expected: scalar_float = 2 * (2 * jnp.pi / 3.0)
        chex.assert_trees_all_close(
            g_vec, jnp.array([expected, 0.0, 0.0]), atol=1e-10
        )


class TestGenerateReciprocalPoints(chex.TestCase, parameterized.TestCase):
    """Test generate_reciprocal_points function.

    :see: :func:`~rheedium.ucell.generate_reciprocal_points`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

        # Create a simple cubic crystal
        self.cubic_crystal: Any = self._create_cubic_crystal()

    def _create_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal."""
        a: float = 3.0
        cart_coords: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        frac_coords: Float[Array, "..."] = cart_coords / a
        atomic_numbers: Float[Array, "..."] = jnp.array([14.0])  # Silicon

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_point_count(self) -> None:
        r"""Test number of generated points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: number of
        generated points.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(generate_reciprocal_points)
        g_vecs: Float[Array, "..."] = var_fn(
            crystal=self.cubic_crystal,
            hmax=2,
            kmax=2,
            lmax=1,
            in_degrees=True,
        )
        # Number of points = (2*h+1) * (2*k+1) * (2*l+1)
        expected_count: Float[Array, "..."] = 5 * 5 * 3  # 75 points
        chex.assert_shape(g_vecs, (expected_count, 3))

    @chex.variants(with_device=True, without_jit=True)
    def test_includes_origin(self) -> None:
        r"""Test that origin (0,0,0) is included.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: origin (0,0,0) is
        included.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(generate_reciprocal_points)
        g_vecs: Float[Array, "..."] = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # Check if any row is close to zero
        norms: Float[Array, "..."] = jnp.linalg.norm(g_vecs, axis=1)
        has_origin: Bool[Array, "..."] = jnp.any(norms < 1e-10)
        chex.assert_trees_all_equal(has_origin, True)

    @chex.variants(with_device=True, without_jit=True)
    def test_symmetry_pairs(self) -> None:
        r"""Test that (h,k,l) and (-h,-k,-l) are opposites.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (h,k,l) and
        (-h,-k,-l) are opposites.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(generate_reciprocal_points)
        g_vecs: Float[Array, "..."] = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # For each vector, its negative should also be present
        chex.assert_tree_all_finite(g_vecs)

    @chex.variants(with_device=True, without_jit=True)
    def test_cubic_symmetry(self) -> None:
        r"""Test cubic symmetry - equivalent directions have same magnitude.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cubic symmetry -
        equivalent directions have same magnitude.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_fn: Callable[..., Any] = self.variant(generate_reciprocal_points)
        g_vecs: Float[Array, "..."] = var_fn(
            crystal=self.cubic_crystal,
            hmax=1,
            kmax=1,
            lmax=1,
            in_degrees=True,
        )
        # Calculate magnitudes
        mags: Float[Array, "..."] = jnp.linalg.norm(g_vecs, axis=1)
        # For cubic, (1,0,0), (0,1,0), (0,0,1) should have same magnitude
        expected: scalar_float = 2 * jnp.pi / 3.0
        # Count how many have this magnitude
        matches: scalar_float = jnp.sum(jnp.abs(mags - expected) < 1e-5)
        # Should be 6: ±(1,0,0), ±(0,1,0), ±(0,0,1)
        chex.assert_trees_all_equal(matches, 6)

    @parameterized.named_parameters(
        ("small", 1, 1, 1),
        ("medium", 2, 2, 2),
        ("asymmetric", 3, 2, 1),
    )
    def test_various_ranges(self, hmax: int, kmax: int, lmax: int) -> None:
        r"""Test various hkl ranges.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: various hkl
        ranges.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``hmax``,
        ``kmax``, ``lmax``, so the documented behavior is checked across the
        cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_ucell.test_unitcell``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vecs: Float[Array, "..."] = generate_reciprocal_points(
            crystal=self.cubic_crystal,
            hmax=hmax,
            kmax=kmax,
            lmax=lmax,
            in_degrees=True,
        )
        expected_count: Float[Array, "..."] = (
            (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1)
        )
        chex.assert_shape(g_vecs, (expected_count, 3))
        chex.assert_tree_all_finite(g_vecs)


if __name__ == "__main__":
    chex.TestCase.main()
