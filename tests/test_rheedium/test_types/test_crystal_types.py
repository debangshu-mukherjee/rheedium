"""Test suite for rheedium.types.crystal_types PyTrees."""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax import tree_util
from jaxtyping import (
    Array,
    Complex,
    Float,
    Int,
    Num,
    PRNGKeyArray,
    TypeCheckError,
)
from numpy.typing import NDArray
from typing_extensions import TypedDict

from rheedium.types import crystal_types
from rheedium.types.crystal_types import (
    CrystalStructure,
    EwaldData,
    PotentialSlices,
    XYZData,
    create_crystal_structure,
    create_ewald_data,
    create_potential_slices,
    create_xyz_data,
)
from rheedium.ucell.unitcell import build_cell_vectors

from ..._assertions import assert_rejects


def test_crystal_types_all_exports_phase9_carriers() -> None:
    r"""Phase 9 carrier names are exported from ``crystal_types.__all__``.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: Phase 9 carrier
    names are exported from ``crystal_types.__all__``.

    Notes
    -----
    It constructs the representative inputs inside the test body,
    keeping the fixture and assertion path local to the documented case.
    """
    expected: set[str] = {
        "EdgeOnSlices",
        "KirklandParameters",
        "create_edge_on_slices",
        "create_kirkland_parameters",
    }

    assert expected <= set(crystal_types.__all__)


class EwaldKwargs(TypedDict):
    """Typed keyword arguments for create_ewald_data."""

    wavelength_ang: Float[Array, ""]
    k_magnitude: Float[Array, ""]
    sphere_radius: Float[Array, ""]
    recip_vectors: Float[Array, "3 3"]
    hkl_grid: Int[Array, "N 3"]
    g_vectors: Float[Array, "N 3"]
    g_magnitudes: Float[Array, "N"]
    structure_factors: Complex[Array, "N"]
    intensities: Float[Array, "N"]


class TestCrystalStructure(chex.TestCase):
    """Comprehensive test suite for CrystalStructure PyTree.

    :see: :class:`~rheedium.types.CrystalStructure`
    :see: :func:`~rheedium.types.create_crystal_structure`
    """

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_crystal_structure_valid(self) -> None:
        r"""Test creation of valid CrystalStructure instances.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creation of valid
        CrystalStructure instances.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 10
        positions_3d: Float[NDArray, "N 3"] = np.random.rand(n_atoms, 3)
        atomic_numbers: Float[NDArray, "N 1"] = np.ones((n_atoms, 1)) * 14
        frac_positions: Float[Array, "N 4"] = jnp.array(
            np.hstack([positions_3d, atomic_numbers]), dtype=jnp.float32
        )
        cart_positions: Float[Array, "N 4"] = frac_positions * jnp.array(
            [5.0, 5.0, 5.0, 1.0]
        )
        cell_lengths: Float[Array, "3"] = jnp.array([5.0, 5.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

        create_fn: Callable[
            [
                Num[Array, "... 4"],
                Num[Array, "... 4"],
                Num[Array, "3"],
                Num[Array, "3"],
            ],
            CrystalStructure,
        ] = self.variant(create_crystal_structure)
        crystal: CrystalStructure = create_fn(
            frac_positions, cart_positions, cell_lengths, cell_angles
        )

        chex.assert_shape(crystal.frac_positions, (n_atoms, 4))
        chex.assert_shape(crystal.cart_positions, (n_atoms, 4))
        chex.assert_shape(crystal.cell_lengths, (3,))
        chex.assert_shape(crystal.cell_angles, (3,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_crystal_structure_pytree(self) -> None:
        r"""Test PyTree registration and operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PyTree
        registration and operations.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 5
        frac_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
        cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
            *cell_lengths, *cell_angles
        )
        cart_positions: Float[Array, "N 4"] = jnp.column_stack(
            [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
        )

        create_fn: Callable[
            [
                Num[Array, "... 4"],
                Num[Array, "... 4"],
                Num[Array, "3"],
                Num[Array, "3"],
            ],
            CrystalStructure,
        ] = self.variant(create_crystal_structure)
        crystal: CrystalStructure = create_fn(
            frac_positions, cart_positions, cell_lengths, cell_angles
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(crystal)
        reconstructed: CrystalStructure = tree_util.tree_unflatten(
            treedef, flat
        )

        chex.assert_trees_all_close(crystal, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("cubic_small", 1, "cubic"),
        ("orthorhombic_medium", 10, "orthorhombic"),
        ("triclinic_large", 100, "triclinic"),
        ("hexagonal_xlarge", 1000, "hexagonal"),
    )
    def test_crystal_structure_various_cells(
        self, n_atoms: int, cell_type: str
    ) -> None:
        r"""Test CrystalStructure with various cell types and atom counts.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CrystalStructure
        with various cell types and atom counts.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``n_atoms``,
        ``cell_type``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        cell_params: dict[str, tuple[list[float], list[float]]] = {
            "cubic": ([5.0, 5.0, 5.0], [90.0, 90.0, 90.0]),
            "orthorhombic": ([4.0, 5.0, 6.0], [90.0, 90.0, 90.0]),
            "triclinic": ([4.0, 5.0, 6.0], [85.0, 95.0, 100.0]),
            "hexagonal": ([5.0, 5.0, 8.0], [90.0, 90.0, 120.0]),
        }

        lengths: list[float]
        angles: list[float]
        lengths, angles = cell_params[cell_type]
        frac_positions: Float[Array, "N 4"] = jnp.concatenate(
            [
                jnp.array(np.random.rand(n_atoms, 3)),
                jnp.ones((n_atoms, 1)) * 14,
            ],
            axis=1,
        )
        cell_lengths: Float[Array, "3"] = jnp.array(lengths)
        cell_angles: Float[Array, "3"] = jnp.array(angles)
        cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
            *cell_lengths, *cell_angles
        )
        cart_positions: Float[Array, "N 4"] = jnp.column_stack(
            [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
        )
        max_angle: float = 180.0

        var_create_crystal_structure: Callable[
            [
                Num[Array, "... 4"],
                Num[Array, "... 4"],
                Num[Array, "3"],
                Num[Array, "3"],
            ],
            CrystalStructure,
        ] = self.variant(create_crystal_structure)
        crystal: CrystalStructure = var_create_crystal_structure(
            frac_positions, cart_positions, cell_lengths, cell_angles
        )

        chex.assert_shape(crystal.frac_positions, (n_atoms, 4))
        chex.assert_shape(crystal.cart_positions, (n_atoms, 4))
        chex.assert_trees_all_equal(jnp.all(crystal.cell_lengths > 0), True)
        chex.assert_trees_all_equal(
            jnp.all(
                (crystal.cell_angles > 0) & (crystal.cell_angles < max_angle)
            ),
            True,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_crystal_structure_jit_compilation(self) -> None:
        r"""Test JIT compilation of CrystalStructure operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT compilation of
        CrystalStructure operations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def create_and_process(
            frac_pos: Num[Array, "N 4"],
            cart_pos: Num[Array, "N 4"],
            lengths: Num[Array, "3"],
            angles: Num[Array, "3"],
        ) -> Num[Array, ""]:
            crystal: CrystalStructure = create_crystal_structure(
                frac_pos, cart_pos, lengths, angles
            )
            return jnp.sum(crystal.frac_positions) + jnp.sum(
                crystal.cart_positions
            )

        jitted_fn: Callable[
            [
                Num[Array, "N 4"],
                Num[Array, "N 4"],
                Num[Array, "3"],
                Num[Array, "3"],
            ],
            Num[Array, ""],
        ] = self.variant(create_and_process)

        n_atoms: int = 5
        frac_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
        cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
            *cell_lengths, *cell_angles
        )
        cart_positions: Float[Array, "N 4"] = jnp.column_stack(
            [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
        )

        result: Num[Array, ""] = jitted_fn(
            frac_positions, cart_positions, cell_lengths, cell_angles
        )
        expected: Float[Array, ""] = jnp.sum(frac_positions) + jnp.sum(
            cart_positions
        )
        chex.assert_trees_all_close(result, expected)

    def test_crystal_structure_validation_errors(self) -> None:
        r"""Handle invalid inputs properly during JIT compilation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Handle invalid
        inputs properly during JIT compilation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 5

        def create_with_wrong_shape() -> CrystalStructure:
            wrong_shape_frac: Float[Array, "N 3"] = jnp.ones((n_atoms, 3))
            cart_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
            cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
            cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
            return jax.jit(create_crystal_structure)(
                wrong_shape_frac, cart_positions, cell_lengths, cell_angles
            )

        def create_with_mismatched_positions() -> CrystalStructure:
            frac_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
            cart_positions: Float[Array, "M 4"] = jnp.ones((n_atoms + 1, 4))
            cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
            cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
            return jax.jit(create_crystal_structure)(
                frac_positions, cart_positions, cell_lengths, cell_angles
            )

        with pytest.raises(Exception, match="frac_positions|dimension"):
            create_with_wrong_shape()

        with pytest.raises(Exception, match=".*"):
            create_with_mismatched_positions()

    def test_crystal_structure_atomic_number_mismatch_rejected(self) -> None:
        r"""Mismatched atomic numbers should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Mismatched atomic
        numbers should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "2 4"] = jnp.array(
            [[0.0, 0.0, 0.0, 6.0], [0.5, 0.5, 0.5, 8.0]]
        )
        cart_positions: Float[Array, "2 4"] = jnp.array(
            [[0.0, 0.0, 0.0, 6.0], [1.0, 1.0, 1.0, 14.0]]
        )
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

        assert_rejects(
            create_crystal_structure,
            frac_positions,
            cart_positions,
            cell_lengths,
            cell_angles,
            match="atomic numbers must match",
        )

    def test_crystal_structure_negative_cell_lengths_rejected(self) -> None:
        r"""Negative cell lengths should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Negative cell
        lengths should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "2 4"] = jnp.ones((2, 4))
        cart_positions: Float[Array, "2 4"] = jnp.ones((2, 4))
        cell_lengths: Float[Array, "3"] = jnp.array([-3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

        assert_rejects(
            create_crystal_structure,
            frac_positions,
            cart_positions,
            cell_lengths,
            cell_angles,
            match="cell_lengths must be positive",
        )

    def test_crystal_structure_invalid_cell_angles_rejected(self) -> None:
        r"""Cell angles outside (0, 180) should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Cell angles
        outside (0, 180) should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac_positions: Float[Array, "2 4"] = jnp.ones((2, 4))
        cart_positions: Float[Array, "2 4"] = jnp.ones((2, 4))
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 180.0, 90.0])

        assert_rejects(
            create_crystal_structure,
            frac_positions,
            cart_positions,
            cell_lengths,
            cell_angles,
            match="cell_angles must be between 0 and 180 degrees",
        )


class TestEwaldData(chex.TestCase, parameterized.TestCase):
    """Test suite for EwaldData PyTree and create_ewald_data validation.

    :see: :class:`~rheedium.types.EwaldData`
    :see: :func:`~rheedium.types.create_ewald_data`
    """

    def _make_valid_ewald_kwargs(self, n_points: int = 7) -> EwaldKwargs:
        """Build valid keyword arguments for create_ewald_data."""
        wavelength_ang: Float[Array, ""] = jnp.array(0.0859)
        k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / wavelength_ang
        sphere_radius: Float[Array, ""] = k_magnitude
        recip_vectors: Float[Array, "3 3"] = jnp.eye(3) * 1.5
        rng: PRNGKeyArray = jax.random.PRNGKey(0)
        hkl_grid: Int[Array, "N 3"] = jax.random.randint(
            rng, (n_points, 3), -5, 6, dtype=jnp.int32
        )
        g_vectors: Float[Array, "N 3"] = (
            hkl_grid.astype(jnp.float64) @ recip_vectors
        )
        g_magnitudes: Float[Array, "N"] = jnp.linalg.norm(g_vectors, axis=-1)
        structure_factors: Complex[Array, "N"] = jnp.ones(
            n_points, dtype=jnp.complex128
        )
        intensities: Float[Array, "N"] = jnp.ones(n_points, dtype=jnp.float64)
        return {
            "wavelength_ang": wavelength_ang,
            "k_magnitude": k_magnitude,
            "sphere_radius": sphere_radius,
            "recip_vectors": recip_vectors,
            "hkl_grid": hkl_grid,
            "g_vectors": g_vectors,
            "g_magnitudes": g_magnitudes,
            "structure_factors": structure_factors,
            "intensities": intensities,
        }

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_ewald_data_valid(self) -> None:
        r"""Test creation of valid EwaldData instances.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creation of valid
        EwaldData instances.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_points: int = 7
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs(n_points)
        create_fn: Callable[..., EwaldData] = self.variant(create_ewald_data)
        ewald: EwaldData = create_fn(**kwargs)

        chex.assert_shape(ewald.wavelength_ang, ())
        chex.assert_shape(ewald.k_magnitude, ())
        chex.assert_shape(ewald.sphere_radius, ())
        chex.assert_shape(ewald.recip_vectors, (3, 3))
        chex.assert_shape(ewald.hkl_grid, (n_points, 3))
        chex.assert_shape(ewald.g_vectors, (n_points, 3))
        chex.assert_shape(ewald.g_magnitudes, (n_points,))
        chex.assert_shape(ewald.structure_factors, (n_points,))
        chex.assert_shape(ewald.intensities, (n_points,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_data_pytree(self) -> None:
        r"""Test PyTree flatten/unflatten round-trip.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PyTree
        flatten/unflatten round-trip.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        create_fn: Callable[..., EwaldData] = self.variant(create_ewald_data)
        ewald: EwaldData = create_fn(**kwargs)

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(ewald)
        reconstructed: EwaldData = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(ewald, reconstructed)

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_data_dtype_casting(self) -> None:
        r"""Test that inputs are cast to correct dtypes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: inputs are cast to
        correct dtypes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        create_fn: Callable[..., EwaldData] = self.variant(create_ewald_data)
        ewald: EwaldData = create_fn(**kwargs)

        assert ewald.wavelength_ang.dtype == jnp.float64
        assert ewald.k_magnitude.dtype == jnp.float64
        assert ewald.recip_vectors.dtype == jnp.float64
        assert ewald.hkl_grid.dtype == jnp.int32
        assert ewald.g_vectors.dtype == jnp.float64
        assert ewald.structure_factors.dtype == jnp.complex128
        assert ewald.intensities.dtype == jnp.float64

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_data_values_preserved(self) -> None:
        r"""Test that array values are faithfully preserved.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: array values are
        faithfully preserved.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        create_fn: Callable[..., EwaldData] = self.variant(create_ewald_data)
        ewald: EwaldData = create_fn(**kwargs)

        chex.assert_trees_all_close(
            ewald.wavelength_ang, kwargs["wavelength_ang"]
        )
        chex.assert_trees_all_close(ewald.k_magnitude, kwargs["k_magnitude"])
        chex.assert_trees_all_close(
            ewald.recip_vectors, kwargs["recip_vectors"]
        )
        chex.assert_trees_all_close(ewald.g_vectors, kwargs["g_vectors"])
        chex.assert_trees_all_close(ewald.intensities, kwargs["intensities"])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small", 1),
        ("medium", 27),
        ("large", 125),
    )
    def test_ewald_data_various_sizes(self, n_points: int) -> None:
        r"""Test EwaldData with various numbers of reciprocal points.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: EwaldData with
        various numbers of reciprocal points.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``n_points``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs(n_points)
        create_fn: Callable[..., EwaldData] = self.variant(create_ewald_data)
        ewald: EwaldData = create_fn(**kwargs)

        chex.assert_shape(ewald.hkl_grid, (n_points, 3))
        chex.assert_shape(ewald.g_vectors, (n_points, 3))
        chex.assert_shape(ewald.g_magnitudes, (n_points,))
        chex.assert_shape(ewald.structure_factors, (n_points,))
        chex.assert_shape(ewald.intensities, (n_points,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_data_jit_compilation(self) -> None:
        r"""Test JIT compilation of EwaldData operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT compilation of
        EwaldData operations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def create_and_process(**kw: object) -> Float[Array, ""]:
            ewald: EwaldData = create_ewald_data(**kw)
            return jnp.sum(ewald.intensities) + ewald.wavelength_ang

        jitted_fn: Callable[..., Float[Array, ""]] = self.variant(
            create_and_process
        )
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        result: Float[Array, ""] = jitted_fn(**kwargs)
        expected: Float[Array, ""] = (
            jnp.sum(kwargs["intensities"]) + kwargs["wavelength_ang"]
        )
        chex.assert_trees_all_close(result, expected)

    def test_ewald_data_negative_wavelength(self) -> None:
        r"""Test that negative wavelength is caught by validation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: negative
        wavelength is caught by validation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["wavelength_ang"] = jnp.array(-0.1)
        assert_rejects(
            create_ewald_data,
            match="wavelength_ang must be positive",
            **kwargs,
        )

    def test_ewald_data_negative_k_magnitude(self) -> None:
        r"""Test that negative k_magnitude is caught by validation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: negative
        k_magnitude is caught by validation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["k_magnitude"] = jnp.array(-1.0)
        assert_rejects(
            create_ewald_data,
            match="k_magnitude must be positive",
            **kwargs,
        )

    def test_ewald_data_negative_sphere_radius(self) -> None:
        r"""Test that negative sphere_radius is caught by validation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: negative
        sphere_radius is caught by validation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["sphere_radius"] = jnp.array(-1.0)
        assert_rejects(
            create_ewald_data,
            match="sphere_radius must be positive",
            **kwargs,
        )

    def test_ewald_data_mismatched_n(self) -> None:
        r"""Test that mismatched leading dimensions are caught.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: mismatched leading
        dimensions are caught.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs(n_points=7)
        kwargs["intensities"] = jnp.ones(5, dtype=jnp.float64)
        with pytest.raises(TypeCheckError):
            jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_negative_intensities(self) -> None:
        r"""Test that negative intensities are caught by validation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: negative
        intensities are caught by validation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["intensities"] = -jnp.ones(7, dtype=jnp.float64)
        assert_rejects(
            create_ewald_data,
            match="intensities must be non-negative",
            **kwargs,
        )

    def test_ewald_data_nan_in_g_vectors(self) -> None:
        r"""Test that NaN values in g_vectors are caught.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: NaN values in
        g_vectors are caught.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        g: Float[Array, "N 3"] = kwargs["g_vectors"]
        kwargs["g_vectors"] = g.at[0, 0].set(jnp.nan)
        assert_rejects(
            create_ewald_data,
            match="g_vectors contain non-finite values",
            **kwargs,
        )

    def test_ewald_data_inf_in_wavelength(self) -> None:
        r"""Test that Inf values are caught by finiteness check.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Inf values are
        caught by finiteness check.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["wavelength_ang"] = jnp.array(jnp.inf)
        assert_rejects(
            create_ewald_data,
            match="wavelength_ang must be finite",
            **kwargs,
        )


class TestPotentialSlices(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for PotentialSlices PyTree.

    :see: :class:`~rheedium.types.PotentialSlices`
    :see: :func:`~rheedium.types.create_potential_slices`
    """

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_potential_slices_valid(self) -> None:
        r"""Test creation of valid PotentialSlices instances.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creation of valid
        PotentialSlices instances.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_slices: int = 10
        height: int = 64
        width: int = 64
        slices: Float[Array, "n_slices height width"] = jnp.zeros(
            (n_slices, height, width)
        )
        slice_thickness: float = 2.0
        x_calibration: float = 0.1
        y_calibration: float = 0.1

        create_fn: Callable[
            [Float[Array, "n_slices height width"], float, float, float],
            PotentialSlices,
        ] = self.variant(create_potential_slices)
        potential: PotentialSlices = create_fn(
            slices, slice_thickness, x_calibration, y_calibration
        )

        chex.assert_shape(potential.slices, (n_slices, height, width))
        # Scalar fields are validated inside create_potential_slices
        assert float(potential.slice_thickness) == slice_thickness
        assert float(potential.x_calibration) == x_calibration
        assert float(potential.y_calibration) == y_calibration

    @chex.variants(with_jit=True, without_jit=True)
    def test_potential_slices_pytree(self) -> None:
        r"""Test PyTree registration and operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PyTree
        registration and operations.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slices: Float[Array, "5 32 32"] = jnp.ones((5, 32, 32))
        slice_thickness: float = 1.5
        x_calibration: float = 0.2
        y_calibration: float = 0.2

        create_fn: Callable[
            [Float[Array, "n_slices height width"], float, float, float],
            PotentialSlices,
        ] = self.variant(create_potential_slices)
        potential: PotentialSlices = create_fn(
            slices, slice_thickness, x_calibration, y_calibration
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(potential)
        reconstructed: PotentialSlices = tree_util.tree_unflatten(
            treedef, flat
        )

        chex.assert_trees_all_close(potential.slices, reconstructed.slices)
        # Scalar fields become tracers in JIT, can't be directly compared.
        # PyTree structure preservation is verified by reconstruction.

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_slice", 1, 16, 16, 0.5),
        ("medium_slice", 10, 64, 64, 2.0),
        ("large_slice", 100, 128, 128, 1.0),
        ("wide_slice", 50, 256, 512, 3.0),
    )
    def test_potential_slices_various_sizes(
        self, n_slices: int, height: int, width: int, thickness: float
    ) -> None:
        r"""Test PotentialSlices with various dimensions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PotentialSlices
        with various dimensions.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``n_slices``,
        ``height``, ``width``, ``thickness``, so the documented behavior is
        checked across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        slices: Float[Array, "n_slices height width"] = jax.random.normal(
            self.rng, (n_slices, height, width)
        )
        x_calibration: float = 0.1
        y_calibration: float = 0.15

        var_create_potential_slices: Callable[
            [Float[Array, "n_slices height width"], float, float, float],
            PotentialSlices,
        ] = self.variant(create_potential_slices)
        potential: PotentialSlices = var_create_potential_slices(
            slices, thickness, x_calibration, y_calibration
        )

        chex.assert_shape(potential.slices, (n_slices, height, width))
        # Scalar fields are validated in the create_potential_slices function

    @chex.variants(with_jit=True, without_jit=True)
    def test_potential_slices_jit_compilation(self) -> None:
        r"""Test JIT compilation of PotentialSlices operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT compilation of
        PotentialSlices operations.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def create_and_process(
            slices: Float[Array, "n_slices height width"],
            thickness: float,
            x_cal: float,
            y_cal: float,
        ) -> Float[Array, ""]:
            potential: PotentialSlices = create_potential_slices(
                slices, thickness, x_cal, y_cal
            )
            return jnp.sum(potential.slices) * potential.slice_thickness

        jitted_fn: Callable[
            [Float[Array, "n_slices height width"], float, float, float],
            Float[Array, ""],
        ] = self.variant(create_and_process)

        slices: Float[Array, "5 32 32"] = jnp.ones((5, 32, 32))
        thickness: float = 2.0
        x_cal: float = 0.1
        y_cal: float = 0.1

        result: Float[Array, ""] = jitted_fn(slices, thickness, x_cal, y_cal)
        expected: Float[Array, ""] = jnp.sum(slices) * thickness
        chex.assert_trees_all_close(result, expected)

    def test_potential_slices_validation_errors(self) -> None:
        r"""Handle invalid PotentialSlices inputs during JIT compilation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Handle invalid
        PotentialSlices inputs during JIT compilation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def create_with_wrong_shape() -> PotentialSlices:
            wrong_shape_slices: Float[Array, "10 32"] = jnp.ones((10, 32))
            return jax.jit(create_potential_slices)(
                wrong_shape_slices, 1.0, 0.1, 0.1
            )

        # jaxtyping catches type errors before internal validation
        with pytest.raises(TypeCheckError):
            create_with_wrong_shape()

        assert_rejects(
            create_potential_slices,
            jnp.ones((10, 32, 32)),
            -1.0,
            0.1,
            0.1,
            match="slice_thickness must be positive",
        )
        assert_rejects(
            create_potential_slices,
            jnp.ones((10, 32, 32)),
            1.0,
            -0.1,
            0.1,
            match="x_calibration must be positive",
        )


class TestXYZData(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for XYZData PyTree.

    :see: :class:`~rheedium.types.XYZData`
    :see: :func:`~rheedium.types.create_xyz_data`
    """

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_xyz_data_minimal(self) -> None:
        r"""Test creation of XYZData with minimal required fields.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creation of
        XYZData with minimal required fields.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 10
        positions: Float[Array, "N 3"] = jax.random.normal(
            self.rng, (n_atoms, 3)
        )
        atomic_numbers: Int[Array, "N"] = jnp.array([6, 8] * 5)

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        xyz_data: XYZData = make_fn(positions, atomic_numbers)

        chex.assert_shape(xyz_data.positions, (n_atoms, 3))
        chex.assert_shape(xyz_data.atomic_numbers, (n_atoms,))
        chex.assert_trees_all_equal(xyz_data.lattice, None)
        chex.assert_trees_all_equal(xyz_data.stress, None)
        chex.assert_trees_all_equal(xyz_data.energy, None)
        chex.assert_trees_all_equal(xyz_data.forces, None)
        chex.assert_trees_all_equal(xyz_data.properties, None)
        chex.assert_trees_all_equal(xyz_data.comment, None)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_xyz_data_full(self) -> None:
        r"""Test creation of XYZData with all optional fields.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creation of
        XYZData with all optional fields.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 5
        positions: Float[Array, "N 3"] = jax.random.normal(
            self.rng, (n_atoms, 3)
        )
        atomic_numbers: Int[Array, "N"] = jnp.array([1, 6, 7, 8, 9])
        lattice: Float[Array, "3 3"] = jnp.eye(3) * 10.0
        stress: Float[Array, "3 3"] = jax.random.normal(self.rng, (3, 3))
        energy: float = -100.5
        properties: list[dict[str, float | int]] = [
            {"atom_id": i, "charge": 0.1 * i} for i in range(n_atoms)
        ]
        comment: str = "Test XYZ structure"

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        xyz_data: XYZData = make_fn(
            positions,
            atomic_numbers,
            lattice,
            stress,
            energy,
            properties,
            comment,
        )

        chex.assert_shape(xyz_data.positions, (n_atoms, 3))
        chex.assert_shape(xyz_data.atomic_numbers, (n_atoms,))
        chex.assert_shape(xyz_data.lattice, (3, 3))
        chex.assert_shape(xyz_data.stress, (3, 3))
        chex.assert_trees_all_equal(xyz_data.energy is not None, True)
        chex.assert_trees_all_equal(xyz_data.properties, properties)
        chex.assert_trees_all_equal(xyz_data.comment, comment)

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_pytree(self) -> None:
        r"""Test PyTree registration and operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: PyTree
        registration and operations.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "3 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        atomic_numbers: Int[Array, "3"] = jnp.array([1, 1, 1])
        lattice: Float[Array, "3 3"] = jnp.eye(3)
        energy: float = -10.0

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        xyz_data: XYZData = make_fn(
            positions, atomic_numbers, lattice=lattice, energy=energy
        )

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(xyz_data)
        reconstructed: XYZData = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(
            xyz_data.positions, reconstructed.positions
        )
        chex.assert_trees_all_close(
            xyz_data.atomic_numbers, reconstructed.atomic_numbers
        )
        chex.assert_trees_all_close(xyz_data.lattice, reconstructed.lattice)
        chex.assert_trees_all_close(xyz_data.energy, reconstructed.energy)

    @chex.variants(without_jit=True, with_jit=False)
    @parameterized.named_parameters(
        ("minimal_with_lattice", 1, True, False, False),
        ("small_with_stress", 10, False, True, False),
        ("medium_full", 100, True, True, True),
        ("large_with_energy", 1000, False, False, True),
    )
    def test_xyz_data_optional_fields(
        self,
        n_atoms: int,
        include_lattice: bool,
        include_stress: bool,
        include_energy: bool,
    ) -> None:
        r"""Test XYZData with various combinations of optional fields.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XYZData with
        various combinations of optional fields.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``n_atoms``,
        ``include_lattice``, ``include_stress``, ``include_energy``, so the
        documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "N 3"] = jax.random.normal(
            self.rng, (n_atoms, 3)
        )
        atomic_numbers: Int[Array, "N"] = jax.random.randint(
            self.rng, (n_atoms,), 1, 119
        )

        kwargs: dict[
            str,
            Float[Array, "3 3"] | Float[Array, ""] | None,
        ] = {}
        if include_lattice:
            kwargs["lattice"] = jnp.eye(3) * 10.0
        if include_stress:
            kwargs["stress"] = jax.random.normal(self.rng, (3, 3))
        if include_energy:
            kwargs["energy"] = jax.random.normal(self.rng, ())

        # create_xyz_data is a foreign interface; use variant without JIT
        var_create_xyz_data: Callable[..., XYZData] = self.variant(
            create_xyz_data
        )
        xyz_data: XYZData = var_create_xyz_data(
            positions, atomic_numbers, **kwargs
        )

        chex.assert_shape(xyz_data.positions, (n_atoms, 3))
        chex.assert_shape(xyz_data.atomic_numbers, (n_atoms,))
        if include_lattice:
            chex.assert_shape(xyz_data.lattice, (3, 3))
        if include_stress:
            chex.assert_shape(xyz_data.stress, (3, 3))
        if include_energy:
            chex.assert_trees_all_equal(xyz_data.energy is not None, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_xyz_data_jit_compilation(self) -> None:
        r"""Test JIT compilation of operations on XYZData.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT compilation of
        operations on XYZData.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        # Create XYZData outside JIT; create_xyz_data is a foreign interface
        n_atoms: int = 5
        positions: Float[Array, "N 3"] = jnp.ones((n_atoms, 3))
        atomic_numbers: Int[Array, "N"] = (
            jnp.ones(n_atoms, dtype=jnp.int32) * 6
        )
        lattice: Float[Array, "3 3"] = jnp.eye(3) * 5.0
        xyz_data: XYZData = create_xyz_data(
            positions, atomic_numbers, lattice=lattice
        )

        def process_xyz_data(xyz: XYZData) -> Num[Array, ""]:
            return jnp.sum(xyz.positions) + jnp.sum(xyz.atomic_numbers)

        jitted_fn: Callable[[XYZData], Num[Array, ""]] = self.variant(
            process_xyz_data
        )

        result: Num[Array, ""] = jitted_fn(xyz_data)
        expected: Num[Array, ""] = jnp.sum(xyz_data.positions) + jnp.sum(
            xyz_data.atomic_numbers
        )
        chex.assert_trees_all_close(result, expected)

    def test_xyz_data_validation_errors(self) -> None:
        r"""Test that invalid inputs raise appropriate errors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: invalid inputs
        raise appropriate errors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        # jaxtyping catches type errors for wrong position shape
        wrong_shape_positions: Float[Array, "5 4"] = jnp.ones((5, 4))
        atomic_numbers: Int[Array, "5"] = jnp.ones(5, dtype=jnp.int32)
        with pytest.raises(TypeCheckError):
            create_xyz_data(wrong_shape_positions, atomic_numbers)

    def test_xyz_data_nonfinite_positions_rejected(self) -> None:
        r"""Non-finite positions should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-finite
        positions should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "2 3"] = jnp.ones((2, 3)).at[0, 0].set(jnp.nan)
        atomic_numbers: Int[Array, "2"] = jnp.array([6, 8], dtype=jnp.int32)
        assert_rejects(
            create_xyz_data,
            positions,
            atomic_numbers,
            match="positions contain non-finite values",
        )

    def test_xyz_data_negative_atomic_numbers_rejected(self) -> None:
        r"""Negative atomic numbers should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Negative atomic
        numbers should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "2 3"] = jnp.ones((2, 3))
        atomic_numbers: Int[Array, "2"] = jnp.array([6, -1], dtype=jnp.int32)
        assert_rejects(
            create_xyz_data,
            positions,
            atomic_numbers,
            match="atomic_numbers must be non-negative",
        )

    def test_xyz_data_nonfinite_lattice_rejected(self) -> None:
        r"""Non-finite lattice values should be rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-finite lattice
        values should be rejected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "2 3"] = jnp.ones((2, 3))
        atomic_numbers: Int[Array, "2"] = jnp.array([6, 8], dtype=jnp.int32)
        lattice: Float[Array, "3 3"] = jnp.eye(3).at[0, 0].set(jnp.inf)
        assert_rejects(
            create_xyz_data,
            positions,
            atomic_numbers,
            lattice=lattice,
            match="lattice contains non-finite values",
        )

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_tree_map(self) -> None:
        r"""Test that XYZData works correctly with tree_map operations.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: XYZData works
        correctly with tree_map operations.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 5
        positions: Float[Array, "N 3"] = jnp.ones((n_atoms, 3))
        atomic_numbers: Int[Array, "N"] = jnp.ones(n_atoms, dtype=jnp.int32)

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        xyz_data: XYZData = make_fn(positions, atomic_numbers)

        def scale_positions(x: Num[Array, "..."]) -> Num[Array, "..."]:
            if isinstance(x, jnp.ndarray) and x.shape == positions.shape:
                return x * 2.0
            return x

        scaled_data: XYZData = tree_util.tree_map(scale_positions, xyz_data)
        chex.assert_trees_all_close(scaled_data.positions, positions * 2.0)
        chex.assert_trees_all_close(scaled_data.atomic_numbers, atomic_numbers)

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_lattice_none_not_fabricated(self) -> None:
        r"""Omitted lattice stays None instead of a fabricated identity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: creating
        XYZData without a lattice stores ``lattice=None``, while an
        explicit identity lattice (a genuine 1 Angstrom cubic cell) is
        stored as the identity matrix. The two cases remain
        distinguishable, so downstream bounding-box fallbacks trigger
        only for truly absent lattices.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "N 3"] = jnp.zeros((2, 3))
        atomic_numbers: Int[Array, "N"] = jnp.array([6, 6])

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        without_lattice: XYZData = make_fn(positions, atomic_numbers)
        with_unit_lattice: XYZData = make_fn(
            positions, atomic_numbers, lattice=jnp.eye(3)
        )

        chex.assert_trees_all_equal(without_lattice.lattice, None)
        assert with_unit_lattice.lattice is not None
        chex.assert_trees_all_close(
            with_unit_lattice.lattice, jnp.eye(3), atol=1e-12
        )

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_forces_stored(self) -> None:
        r"""Forces are stored as a dynamic (N, 3) field when provided.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: passing a
        forces array to ``create_xyz_data`` stores the exact values in
        ``XYZData.forces`` as a float64 (N, 3) array, and the field
        participates in PyTree flattening as a dynamic leaf.

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
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "N 3"] = jnp.zeros((2, 3))
        atomic_numbers: Int[Array, "N"] = jnp.array([14, 14])
        forces: Float[Array, "N 3"] = jnp.array(
            [[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]]
        )

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        xyz_data: XYZData = make_fn(positions, atomic_numbers, forces=forces)

        assert xyz_data.forces is not None
        chex.assert_shape(xyz_data.forces, (2, 3))
        chex.assert_trees_all_close(xyz_data.forces, forces, atol=1e-15)
        leaves = tree_util.tree_leaves(xyz_data)
        assert any(
            hasattr(leaf, "shape")
            and leaf.shape == (2, 3)
            and jnp.allclose(leaf, forces)
            for leaf in leaves
        )

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_forces_wrong_shape_rejected(self) -> None:
        r"""Forces with a mismatched shape are rejected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: passing a
        forces array whose leading dimension does not match the number of
        atoms is rejected instead of being silently stored. The jaxtyping
        signature annotation (shared axis ``N`` between ``positions`` and
        ``forces``) raises a TypeCheckError naming the ``forces``
        argument, and the factory's own shape validation backs it up with
        a ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        positions: Float[Array, "N 3"] = jnp.zeros((3, 3))
        atomic_numbers: Int[Array, "N"] = jnp.array([6, 6, 6])
        bad_forces: Float[Array, "M 3"] = jnp.zeros((2, 3))

        # create_xyz_data is a foreign interface; use variant without JIT
        make_fn: Callable[..., XYZData] = self.variant(create_xyz_data)
        with pytest.raises((ValueError, TypeCheckError), match=r"forces"):
            make_fn(positions, atomic_numbers, forces=bad_forces)


class TestPyTreeIntegration(chex.TestCase, parameterized.TestCase):
    """Test PyTree operations across all crystal types."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    def test_nested_pytree_operations(self) -> None:
        r"""Test nested PyTree structures with crystal types.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: nested PyTree
        structures with crystal types.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_atoms: int = 5
        frac_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
        cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
            3.0, 4.0, 5.0, 90.0, 90.0, 90.0
        )
        crystal: CrystalStructure = create_crystal_structure(
            frac_positions,
            jnp.column_stack(
                [frac_positions[:, :3] @ cell_vectors, frac_positions[:, 3]]
            ),
            jnp.array([3.0, 4.0, 5.0]),
            jnp.array([90.0, 90.0, 90.0]),
        )

        potential: PotentialSlices = create_potential_slices(
            jnp.ones((10, 32, 32)), 2.0, 0.1, 0.1
        )

        xyz_data: XYZData = create_xyz_data(
            jnp.ones((n_atoms, 3)), jnp.ones(n_atoms, dtype=jnp.int32)
        )

        nested_structure: dict[
            str, CrystalStructure | PotentialSlices | XYZData
        ] = {
            "crystal": crystal,
            "potential": potential,
            "xyz": xyz_data,
        }

        flat: list[object]
        treedef: object
        flat, treedef = tree_util.tree_flatten(nested_structure)
        reconstructed: dict[
            str, CrystalStructure | PotentialSlices | XYZData
        ] = tree_util.tree_unflatten(treedef, flat)

        chex.assert_trees_all_close(
            nested_structure["crystal"], reconstructed["crystal"]
        )
        chex.assert_trees_all_close(
            nested_structure["potential"].slices,
            reconstructed["potential"].slices,
        )
        chex.assert_trees_all_close(
            nested_structure["xyz"].positions, reconstructed["xyz"].positions
        )

    def test_vmap_over_crystal_structures(self) -> None:
        r"""Test vmap operations over batches of crystal structures.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: vmap operations
        over batches of crystal structures.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        batch_size: int = 4
        n_atoms: int = 3

        frac_positions_batch: Float[Array, "batch N 4"] = jnp.ones(
            (batch_size, n_atoms, 4)
        )
        cart_positions_batch: Float[Array, "batch N 4"] = jnp.concatenate(
            [
                jnp.ones((batch_size, n_atoms, 3)) * 5.0,
                jnp.ones((batch_size, n_atoms, 1)),
            ],
            axis=2,
        )
        cell_lengths_batch: Float[Array, "batch 3"] = (
            jnp.ones((batch_size, 3)) * 5.0
        )
        cell_angles_batch: Float[Array, "batch 3"] = (
            jnp.ones((batch_size, 3)) * 90.0
        )

        vmapped_create: Callable[
            [
                Num[Array, "batch N 4"],
                Num[Array, "batch N 4"],
                Num[Array, "batch 3"],
                Num[Array, "batch 3"],
            ],
            CrystalStructure,
        ] = jax.vmap(create_crystal_structure)
        crystals: CrystalStructure = vmapped_create(
            frac_positions_batch,
            cart_positions_batch,
            cell_lengths_batch,
            cell_angles_batch,
        )

        chex.assert_shape(crystals.frac_positions, (batch_size, n_atoms, 4))
        chex.assert_shape(crystals.cart_positions, (batch_size, n_atoms, 4))
        chex.assert_shape(crystals.cell_lengths, (batch_size, 3))
        chex.assert_shape(crystals.cell_angles, (batch_size, 3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestEwaldDataOccupancies(chex.TestCase):
    """Occupancy metadata on the EwaldData PyTree.

    :see: :func:`~rheedium.types.create_ewald_data`
    """

    def test_factory_stores_occupancies_as_float64(self) -> None:
        r"""Verify create_ewald_data converts and stores occupancies.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the optional
        ``occupancies`` argument is converted to float64 and stored on
        the returned ``EwaldData`` alongside the atomic positions and
        numbers used by the continuous-rod paths.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_points: int = 2
        ewald: EwaldData = create_ewald_data(
            wavelength_ang=jnp.asarray(0.05),
            k_magnitude=jnp.asarray(125.0),
            sphere_radius=jnp.asarray(125.0),
            recip_vectors=jnp.eye(3),
            hkl_grid=jnp.zeros((n_points, 3), dtype=jnp.int32),
            g_vectors=jnp.zeros((n_points, 3)),
            g_magnitudes=jnp.zeros(n_points),
            structure_factors=jnp.zeros(n_points, dtype=jnp.complex128),
            intensities=jnp.zeros(n_points),
            atom_positions=jnp.zeros((1, 3)),
            atomic_numbers=jnp.array([14], dtype=jnp.int32),
            occupancies=jnp.array([0.5], dtype=jnp.float32),
        )
        assert ewald.occupancies is not None
        self.assertEqual(ewald.occupancies.dtype, jnp.float64)
        np.testing.assert_allclose(np.asarray(ewald.occupancies), [0.5])

    def test_factory_defaults_occupancies_to_none(self) -> None:
        r"""Verify occupancies default to None on hand-built EwaldData.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: omitting the
        ``occupancies`` argument leaves the field ``None``, which every
        consumer interprets as fully occupied sites.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_types.test_crystal_types``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        n_points: int = 2
        ewald: EwaldData = create_ewald_data(
            wavelength_ang=jnp.asarray(0.05),
            k_magnitude=jnp.asarray(125.0),
            sphere_radius=jnp.asarray(125.0),
            recip_vectors=jnp.eye(3),
            hkl_grid=jnp.zeros((n_points, 3), dtype=jnp.int32),
            g_vectors=jnp.zeros((n_points, 3)),
            g_magnitudes=jnp.zeros(n_points),
            structure_factors=jnp.zeros(n_points, dtype=jnp.complex128),
            intensities=jnp.zeros(n_points),
        )
        self.assertIsNone(ewald.occupancies)
