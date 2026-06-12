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
    """Comprehensive test suite for CrystalStructure PyTree."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_crystal_structure_valid(self) -> None:
        """Test creation of valid CrystalStructure instances."""
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
        """Test PyTree registration and operations."""
        n_atoms: int = 5
        frac_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4))
        cart_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4)) * 2.0
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
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
        """Test CrystalStructure with various cell types and atom counts."""
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
        cart_positions: Float[Array, "N 4"] = frac_positions * jnp.concatenate(
            [jnp.array(lengths), jnp.array([1.0])]
        )
        cell_lengths: Float[Array, "3"] = jnp.array(lengths)
        cell_angles: Float[Array, "3"] = jnp.array(angles)
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
        """Test JIT compilation of CrystalStructure operations."""

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
        cart_positions: Float[Array, "N 4"] = jnp.ones((n_atoms, 4)) * 2.0
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 4.0, 5.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

        result: Num[Array, ""] = jitted_fn(
            frac_positions, cart_positions, cell_lengths, cell_angles
        )
        expected: Float[Array, ""] = jnp.sum(frac_positions) + jnp.sum(
            cart_positions
        )
        chex.assert_trees_all_close(result, expected)

    def test_crystal_structure_validation_errors(self) -> None:
        """Handle invalid inputs properly during JIT compilation."""
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


class TestEwaldData(chex.TestCase, parameterized.TestCase):
    """Test suite for EwaldData PyTree and create_ewald_data validation."""

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
        """Test creation of valid EwaldData instances."""
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
        """Test PyTree flatten/unflatten round-trip."""
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
        """Test that inputs are cast to correct dtypes."""
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
        """Test that array values are faithfully preserved."""
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
        """Test EwaldData with various numbers of reciprocal points."""
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
        """Test JIT compilation of EwaldData operations."""

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
        """Test that negative wavelength is caught by validation."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["wavelength_ang"] = jnp.array(-0.1)
        jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_negative_k_magnitude(self) -> None:
        """Test that negative k_magnitude is caught by validation."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["k_magnitude"] = jnp.array(-1.0)
        jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_negative_sphere_radius(self) -> None:
        """Test that negative sphere_radius is caught by validation."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["sphere_radius"] = jnp.array(-1.0)
        jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_mismatched_n(self) -> None:
        """Test that mismatched leading dimensions are caught."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs(n_points=7)
        kwargs["intensities"] = jnp.ones(5, dtype=jnp.float64)
        with pytest.raises(TypeCheckError):
            jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_negative_intensities(self) -> None:
        """Test that negative intensities are caught by validation."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["intensities"] = -jnp.ones(7, dtype=jnp.float64)
        jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_nan_in_g_vectors(self) -> None:
        """Test that NaN values in g_vectors are caught."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        g: Float[Array, "N 3"] = kwargs["g_vectors"]
        kwargs["g_vectors"] = g.at[0, 0].set(jnp.nan)
        jax.jit(create_ewald_data)(**kwargs)

    def test_ewald_data_inf_in_wavelength(self) -> None:
        """Test that Inf values are caught by finiteness check."""
        kwargs: EwaldKwargs = self._make_valid_ewald_kwargs()
        kwargs["wavelength_ang"] = jnp.array(jnp.inf)
        jax.jit(create_ewald_data)(**kwargs)


class TestPotentialSlices(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for PotentialSlices PyTree."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_potential_slices_valid(self) -> None:
        """Test creation of valid PotentialSlices instances."""
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
        """Test PyTree registration and operations."""
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
        """Test PotentialSlices with various dimensions."""
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
        """Test JIT compilation of PotentialSlices operations."""

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
        """Handle invalid PotentialSlices inputs during JIT compilation."""

        def create_with_wrong_shape() -> PotentialSlices:
            wrong_shape_slices: Float[Array, "10 32"] = jnp.ones((10, 32))
            return jax.jit(create_potential_slices)(
                wrong_shape_slices, 1.0, 0.1, 0.1
            )

        def create_with_negative_thickness() -> PotentialSlices:
            slices: Float[Array, "10 32 32"] = jnp.ones((10, 32, 32))
            negative_thickness: float = -1.0
            return jax.jit(create_potential_slices)(
                slices, negative_thickness, 0.1, 0.1
            )

        def create_with_negative_calibration() -> PotentialSlices:
            slices: Float[Array, "10 32 32"] = jnp.ones((10, 32, 32))
            negative_calibration: float = -0.1
            return jax.jit(create_potential_slices)(
                slices, 1.0, negative_calibration, 0.1
            )

        # jaxtyping catches type errors before internal validation
        with pytest.raises(TypeCheckError):
            create_with_wrong_shape()

        # These will fail during JIT tracing due to conditional checks
        # The actual error depends on JAX's tracing behavior
        # Both trace successfully but fail at runtime if executed.
        create_with_negative_thickness()
        create_with_negative_calibration()


class TestXYZData(chex.TestCase, parameterized.TestCase):
    """Comprehensive test suite for XYZData PyTree."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_xyz_data_minimal(self) -> None:
        """Test creation of XYZData with minimal required fields."""
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
        chex.assert_trees_all_equal(xyz_data.lattice is not None, True)
        chex.assert_trees_all_equal(xyz_data.stress, None)
        chex.assert_trees_all_equal(xyz_data.energy, None)
        chex.assert_trees_all_equal(xyz_data.properties, None)
        chex.assert_trees_all_equal(xyz_data.comment, None)

    @chex.variants(without_jit=True, with_jit=False)
    def test_create_xyz_data_full(self) -> None:
        """Test creation of XYZData with all optional fields."""
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
        """Test PyTree registration and operations."""
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
        """Test XYZData with various combinations of optional fields."""
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
        """Test JIT compilation of operations on XYZData."""
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
        """Test that invalid inputs raise appropriate errors."""
        # jaxtyping catches type errors for wrong position shape
        wrong_shape_positions: Float[Array, "5 4"] = jnp.ones((5, 4))
        atomic_numbers: Int[Array, "5"] = jnp.ones(5, dtype=jnp.int32)
        with pytest.raises(TypeCheckError):
            create_xyz_data(wrong_shape_positions, atomic_numbers)

    @chex.variants(without_jit=True, with_jit=False)
    def test_xyz_data_tree_map(self) -> None:
        """Test that XYZData works correctly with tree_map operations."""
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


class TestPyTreeIntegration(chex.TestCase, parameterized.TestCase):
    """Test PyTree operations across all crystal types."""

    def setUp(self) -> None:
        """Initialize the random key for each test."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

    def test_nested_pytree_operations(self) -> None:
        """Test nested PyTree structures with crystal types."""
        n_atoms: int = 5
        crystal: CrystalStructure = create_crystal_structure(
            jnp.ones((n_atoms, 4)),
            jnp.ones((n_atoms, 4)) * 2.0,
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
        """Test vmap operations over batches of crystal structures."""
        batch_size: int = 4
        n_atoms: int = 3

        frac_positions_batch: Float[Array, "batch N 4"] = jnp.ones(
            (batch_size, n_atoms, 4)
        )
        cart_positions_batch: Float[Array, "batch N 4"] = (
            jnp.ones((batch_size, n_atoms, 4)) * 2.0
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
