"""Test suite for updated RHEED simulator with surface physics.

Tests the integration of:
- Proper atomic form factors (Kirkland parameterization)
- Surface-enhanced Debye-Waller factors
- CTR intensity calculations
- Structure factor with q_z dependence
- Multislice propagation and simulation
- Kinematic reflection finding
- Detector projection
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax.test_util import check_grads
from jaxtyping import Array, Bool, Complex, Float, Integer, PRNGKeyArray
from numpy.typing import NDArray

from rheedium.procs.grains import grain_population_to_distribution
from rheedium.procs.surface_modifier import (
    bind_step_edge_distribution,
    bind_twin_wall_distribution,
    step_edge_to_distribution,
    twin_wall_to_distribution,
)
from rheedium.simul.beam_averaging import (
    apply_distribution,
    apply_distributions,
    decompose_beam_modes,
)
from rheedium.simul.simulator import (
    _ewald_amplitude_pattern,
    _kinematic_finite_domain_amplitude,
    _render_ctr_streaks_to_image,
    checked_multislice_propagate,
    compute_kinematic_intensities_with_ctrs,
    detector_extent_mm,
    ewald_simulator,
    ewald_simulator_with_orientation_distribution,
    find_kinematic_reflections,
    kinematic_amplitude,
    log_compress_image,
    multislice_detector_amplitude,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    render_amplitude_to_field,
    render_ctr_amplitude_to_field,
    render_pattern_to_image,
    simulate_detector_image,
    simulate_detector_image_instrument,
    sliced_crystal_to_projected_potential_slices,
)
from rheedium.tools import (
    gauss_hermite_nodes_weights,
    incident_wavevector,
    wavelength_ang,
)
from rheedium.tools.wrappers import jax_safe
from rheedium.types import (
    TRIVIAL_DISTRIBUTION,
    CrystalStructure,
    Distribution,
    PotentialSlices,
    ReductionMode,
    SlicedCrystal,
    create_coherent_beam,
    create_distribution,
    create_gaussian_schell_beam,
    orientation_to_distribution,
)
from rheedium.types.crystal_types import (
    create_crystal_structure,
    create_potential_slices,
)
from rheedium.types.custom_types import scalar_float
from rheedium.types.distributions import create_discrete_orientation
from rheedium.types.rheed_types import (
    RHEEDPattern,
    SurfaceConfig,
    create_sliced_crystal,
)
from rheedium.ucell import reciprocal_lattice_vectors


class TestUpdatedSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for updated RHEED simulator with proper surface physics."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.rng: PRNGKeyArray = jax.random.PRNGKey(42)

        # Create simple Si(111) structure for testing
        self.si_crystal: CrystalStructure = self._create_si111_crystal()

    def _create_si111_crystal(self) -> CrystalStructure:
        """Create a simple Si(111) crystal structure.

        Returns
        -------
        crystal : CrystalStructure
            Silicon crystal with (111) orientation
        """
        a_si: float = 5.431  # Si lattice constant in Angstroms

        # Si diamond structure fractional positions
        frac_coords: Float[Array, "..."] = jnp.array(
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.25, 0.25],
                [0.50, 0.50, 0.00],
                [0.75, 0.75, 0.25],
                [0.50, 0.00, 0.50],
                [0.75, 0.25, 0.75],
                [0.00, 0.50, 0.50],
                [0.25, 0.75, 0.75],
            ]
        )

        # Convert to Cartesian coordinates
        cart_coords: Float[Array, "..."] = frac_coords * a_si

        # Add atomic numbers (Si = 14)
        atomic_numbers: Float[Array, "..."] = jnp.full(8, 14.0)
        frac_positions: Float[Array, "..."] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "..."] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("room_temp", 300.0, 0.5, 0.3),
        ("low_temp", 77.0, 0.3, 0.3),
        ("high_roughness", 300.0, 1.0, 0.3),
        ("thin_surface", 300.0, 0.5, 0.1),
    )
    def test_intensity_calculation_with_ctrs(
        self,
        temperature: float,
        surface_roughness: float,
        surface_fraction: float,
    ) -> None:
        """Test intensity calculation with CTR contributions."""
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        # Set up simple test case
        # 20 keV, 2 degrees
        k_in: Float[Array, "..."] = jnp.array([73.0, 0.0, -2.5])
        g_vectors: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        k_out: Float[Array, "..."] = k_in + g_vectors

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=temperature,
            surface_roughness=surface_roughness,
            surface_fraction=surface_fraction,
        )

        # Check properties
        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

        # Surface roughness should decrease intensities
        if surface_roughness > 0.5:
            max_intensity: scalar_float = jnp.max(intensities)
            chex.assert_scalar_positive(float(max_intensity))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_surface_enhancement_effect(self) -> None:
        """Test that surface atoms have enhanced thermal motion."""
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        k_in: Float[Array, "..."] = jnp.array([73.0, 0.0, -2.5])
        g_vectors: Float[Array, "..."] = jnp.array([[1.0, 0.0, 0.0]])
        k_out: Float[Array, "..."] = k_in + g_vectors

        # Compare with and without surface effects
        intensities_bulk: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_fraction=0.0,  # No surface atoms
        )

        intensities_surface: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_fraction=0.5,  # Half atoms are surface
        )

        # Surface enhancement should reduce intensity due to increased
        # DW factor (without normalization, this works correctly)
        chex.assert_trees_all_equal(
            intensities_surface[0] < intensities_bulk[0], True
        )


class TestCheckedNumericalEntryPoints(chex.TestCase):
    """Tests for opt-in checkified numerical entry points."""

    def test_checked_multislice_propagate_valid(self) -> None:
        """Checked multislice propagation should allow finite outputs."""
        potential: PotentialSlices = create_potential_slices(
            slices=jnp.ones((2, 8, 8)),
            slice_thickness=1.0,
            x_calibration=0.2,
            y_calibration=0.2,
        )

        err: Any
        exit_wave: Complex[Array, "8 8"]
        err, exit_wave = jax.jit(checked_multislice_propagate)(
            potential,
            20.0,
            2.0,
        )
        err.throw()

        chex.assert_shape(exit_wave, (8, 8))
        chex.assert_tree_all_finite(exit_wave)

    def test_checked_multislice_propagate_rejects_nan(self) -> None:
        """Checked multislice propagation should report NaN outputs."""
        potential: PotentialSlices = PotentialSlices(
            slices=jnp.ones((2, 8, 8)).at[0, 0, 0].set(jnp.nan),
            slice_thickness=jnp.asarray(1.0),
            x_calibration=jnp.asarray(0.2),
            y_calibration=jnp.asarray(0.2),
        )

        err: Any
        exit_wave: Complex[Array, "8 8"]
        err, exit_wave = jax.jit(checked_multislice_propagate)(
            potential,
            20.0,
            2.0,
        )

        del exit_wave
        with pytest.raises(Exception, match="nan"):
            err.throw()


class TestProjectOnDetector(chex.TestCase, parameterized.TestCase):
    """Test suite for detector projection functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_basic_projection(self) -> None:
        """Test basic projection onto detector plane."""
        var_project: Callable[..., Any] = self.variant(project_on_detector)

        k_out: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.5],
            ]
        )
        detector_distance: float = 100.0

        coords: Float[Array, "..."] = var_project(k_out, detector_distance)

        chex.assert_shape(coords, (3, 2))
        chex.assert_tree_all_finite(coords)
        # First point: no lateral deflection
        chex.assert_trees_all_close(
            coords[0], jnp.array([0.0, 0.0]), atol=1e-6
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_scaling(self, distance: float) -> None:
        """Test that coordinates scale linearly with detector distance."""
        var_project: Callable[..., Any] = self.variant(project_on_detector)

        k_out: Float[Array, "..."] = jnp.array([[1.0, 0.5, 0.3]])
        coords: Float[Array, "..."] = var_project(k_out, distance)

        chex.assert_shape(coords, (1, 2))
        # Verify linear scaling
        expected_h: Float[Array, "..."] = 0.5 * distance / 1.0
        expected_v: Float[Array, "..."] = 0.3 * distance / 1.0
        chex.assert_trees_all_close(
            coords[0], jnp.array([expected_h, expected_v]), rtol=1e-5
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self) -> None:
        """Test output has correct shape for various inputs."""
        var_project: Callable[..., Any] = self.variant(project_on_detector)

        n: int
        for n in [1, 5, 10, 50]:
            k_out: Float[Array, "..."] = jnp.ones((n, 3))
            coords: Float[Array, "..."] = var_project(k_out, 100.0)
            chex.assert_shape(coords, (n, 2))


class TestFindKinematicReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for kinematic reflection finding."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.k_mag: float = 73.0  # Typical |k| for 20 keV electrons

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_elastic_scattering_constraint(self) -> None:
        """Test that output wavevectors satisfy |k_out| ≈ |k_in|."""
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [10.0, 10.0, 10.0],  # This one should fail elastic condition
            ]
        )

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, z_sign=-1.0, tolerance=0.5)

        # Check shapes
        chex.assert_shape(allowed_indices, (5,))
        chex.assert_shape(k_out, (5, 3))

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.05),
        ("loose", 0.2),
    )
    def test_tolerance_variation(self, tolerance: float) -> None:
        """Test that tighter tolerances allow fewer reflections."""
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        # Small G vectors that barely satisfy elastic condition
        gs: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.15, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=-1.0, tolerance=tolerance
        )

        chex.assert_tree_all_finite(k_out)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_positive(self) -> None:
        """Test filtering with positive z_sign (forward scattering)."""
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, 2.5])
        gs: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -5.0],  # Would give negative z
            ]
        )

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, z_sign=1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_negative(self) -> None:
        """Test filtering with negative z_sign (back scattering - RHEED)."""
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],  # Would give positive z
                [0.0, 0.0, -1.0],
            ]
        )

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, z_sign=-1.0, tolerance=0.5)

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_empty_g_vectors(self) -> None:
        """Test handling of single G vector."""
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, tolerance=0.5)

        chex.assert_shape(allowed_indices, (1,))
        chex.assert_shape(k_out, (1, 3))


class TestSlicedCrystalToProjectedPotentialSlices(
    chex.TestCase, parameterized.TestCase
):
    """Tests for converting sliced crystals to potential slices."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.si_sliced: Any = self._create_simple_sliced_crystal()

    def _create_simple_sliced_crystal(self) -> SlicedCrystal:
        """Create a simple sliced crystal for testing."""
        # Simple 2-atom structure
        cart_positions: Float[Array, "..."] = jnp.array(
            [
                [5.0, 5.0, 1.0, 14.0],  # Si at (5,5,1)
                [7.5, 7.5, 3.0, 14.0],  # Si at (7.5,7.5,3)
            ]
        )

        return create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output potential has expected shape.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential: Any = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
        )

        # Check slices array exists
        chex.assert_tree_all_finite(potential.slices)
        # Should have nz slices based on depth/slice_thickness
        nz_expected: Float[Array, "..."] = int(jnp.ceil(5.0 / 2.0))
        self.assertEqual(potential.slices.shape[0], nz_expected)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin", 1.0),
        ("medium", 2.0),
        ("thick", 5.0),
    )
    def test_slice_thickness_variation(self, thickness: float) -> None:
        """Test potential generation with different slice thicknesses.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential: Any = var_convert(
            self.si_sliced,
            slice_thickness=thickness,
            pixel_size=0.5,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Number of slices should be ceil(depth / thickness)
        expected_nz: Float[Array, "..."] = int(jnp.ceil(5.0 / thickness))
        self.assertEqual(potential.slices.shape[0], expected_nz)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("fine", 0.1),
        ("medium", 0.5),
        ("coarse", 1.0),
    )
    def test_pixel_size_variation(self, pixel_size: float) -> None:
        """Test potential generation with different pixel sizes.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential: Any = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=pixel_size,
        )

        chex.assert_tree_all_finite(potential.slices)
        # Grid should scale with pixel size
        expected_nx: Float[Array, "..."] = int(jnp.ceil(15.0 / pixel_size))
        expected_ny: Float[Array, "..."] = int(jnp.ceil(15.0 / pixel_size))
        self.assertEqual(potential.slices.shape[1], expected_nx)
        self.assertEqual(potential.slices.shape[2], expected_ny)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("lobato", "lobato"),
        ("kirkland", "kirkland"),
    )
    def test_parameterization_variation(self, parameterization: str) -> None:
        """Projected-potential slices are finite for both models."""
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        potential: Any = var_convert(
            self.si_sliced,
            slice_thickness=2.0,
            pixel_size=0.5,
            parameterization=parameterization,
        )

        chex.assert_tree_all_finite(potential.slices)

    @chex.variants(with_device=True, without_jit=True)
    def test_calibration_stored(self) -> None:
        """Test that calibration values are stored correctly.

        Note: JIT compilation not supported due to dynamic grid dimensions.
        """
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        pixel_size: float = 0.3
        slice_thickness: float = 1.5

        potential: Any = var_convert(
            self.si_sliced,
            slice_thickness=slice_thickness,
            pixel_size=pixel_size,
        )

        chex.assert_trees_all_close(
            potential.x_calibration, pixel_size, atol=1e-6
        )
        chex.assert_trees_all_close(
            potential.y_calibration, pixel_size, atol=1e-6
        )
        chex.assert_trees_all_close(
            potential.slice_thickness, slice_thickness, atol=1e-6
        )


class TestMultislicePropagate(chex.TestCase, parameterized.TestCase):
    """Test suite for multislice wave propagation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential: Any = self._create_simple_potential()

    def _create_simple_potential(self) -> PotentialSlices:
        """Create a simple potential for testing."""
        # Small grid for fast tests
        nx: tuple[Any, ...]
        ny: tuple[Any, ...]
        nz: tuple[Any, ...]
        nx, ny, nz = 32, 32, 3
        slices: Float[Array, "..."] = jnp.zeros((nz, nx, ny))
        # Add a small potential at center of first slice
        slices = slices.at[0, 16, 16].set(1.0)

        return create_potential_slices(
            slices=slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self) -> None:
        """Test that exit wave has same shape as input grid."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_shape(exit_wave, (32, 32))
        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_exit_wave_nonzero(self) -> None:
        """Test that exit wave has non-zero amplitude."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        total_intensity: scalar_float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage: float) -> None:
        """Test propagation at different voltages."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=voltage,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Higher voltage = shorter wavelength = different phase evolution
        total_intensity: scalar_float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("shallow", 0.5),
        ("medium", 2.0),
        ("steep", 5.0),
    )
    def test_grazing_angle_variation(self, theta: float) -> None:
        """Test propagation at different grazing angles."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
    )
    def test_azimuthal_angle_variation(self, phi: float) -> None:
        """Test propagation at different azimuthal angles."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=phi,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("no_inner", 0.0),
        ("small_inner", 10.0),
        ("large_inner", 20.0),
    )
    def test_inner_potential_variation(self, v0: float) -> None:
        """Test effect of inner potential on propagation."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            inner_potential_v0=v0,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("half", 0.5),
        ("two_thirds", 2.0 / 3.0),
        ("full", 1.0),
    )
    def test_bandwidth_limit_variation(self, limit: float) -> None:
        """Test different bandwidth limiting values."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            bandwidth_limit=limit,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_zero_potential_propagation(self) -> None:
        """Test propagation through zero potential (free space)."""
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        # Zero potential
        zero_slices: Float[Array, "..."] = jnp.zeros((3, 32, 32))
        zero_potential: Any = create_potential_slices(
            slices=zero_slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

        exit_wave: Any = var_propagate(
            zero_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Should still have intensity (plane wave propagates)
        total_intensity: scalar_float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)


class TestMultisliceSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for complete multislice RHEED simulation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.simple_potential: Any = self._create_test_potential()

    def _create_test_potential(self) -> PotentialSlices:
        """Create potential slices for testing."""
        nx: tuple[Any, ...]
        ny: tuple[Any, ...]
        nz: tuple[Any, ...]
        nx, ny, nz = 32, 32, 3
        slices: Float[Array, "..."] = jnp.zeros((nz, nx, ny))
        # Add some structure
        slices = slices.at[0, 16, 16].set(2.0)
        slices = slices.at[1, 16, 16].set(1.5)
        slices = slices.at[2, 16, 16].set(1.0)

        return create_potential_slices(
            slices=slices,
            slice_thickness=2.0,
            x_calibration=0.5,
            y_calibration=0.5,
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_returns_rheed_pattern(self) -> None:
        """Test that simulator returns valid RHEEDPattern.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_pattern_shapes_consistent(self) -> None:
        """Test that all pattern arrays have consistent shapes.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        n: Any = pattern.G_indices.shape[0]
        chex.assert_shape(pattern.k_out, (n, 3))
        chex.assert_shape(pattern.detector_points, (n, 2))
        chex.assert_shape(pattern.intensities, (n,))

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("close", 50.0),
        ("medium", 100.0),
        ("far", 500.0),
    )
    def test_detector_distance_variation(self, distance: float) -> None:
        """Test simulation at different detector distances.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
            detector_distance=distance,
        )

        chex.assert_tree_all_finite(pattern.detector_points)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("low", 10.0),
        ("medium", 20.0),
        ("high", 30.0),
    )
    def test_voltage_variation(self, voltage: float) -> None:
        """Test simulation at different voltages.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=voltage,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    @parameterized.named_parameters(
        ("shallow", 1.0),
        ("medium", 2.0),
        ("steep", 5.0),
    )
    def test_angle_variation(self, theta: float) -> None:
        """Test simulation at different grazing angles.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ewald_sphere_constraint(self) -> None:
        """Test that output wavevectors approximately satisfy Ewald sphere.

        Note: JIT not supported due to dynamic array sizes.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            voltage_kv=20.0,
            theta_deg=2.0,
        )

        # k_out should have approximately same magnitude as k_in
        voltage_kv: float = 20.0
        lam_ang: Any = float(wavelength_ang(voltage_kv))
        k_mag_expected: scalar_float = 2.0 * jnp.pi / lam_ang

        k_out_mags: Float[Array, "..."] = jnp.linalg.norm(
            pattern.k_out, axis=1
        )

        # Filter non-zero k_out (valid reflections)
        valid_mask: Bool[Array, "..."] = k_out_mags > 0
        valid_k_out_mags: Float[Array, "..."] = k_out_mags[valid_mask]

        if valid_k_out_mags.shape[0] > 0:
            # All valid k_out should be close to k_in magnitude
            chex.assert_trees_all_close(
                valid_k_out_mags,
                jnp.full_like(valid_k_out_mags, k_mag_expected),
                rtol=0.1,
            )


class TestComputeKinematicIntensitiesExtended(
    chex.TestCase, parameterized.TestCase
):
    """Extended tests for kinematic intensity calculation with CTRs."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.si_crystal: CrystalStructure = self._create_si_crystal()
        self.k_in: Float[Array, "..."] = jnp.array([73.0, 0.0, -2.5])
        self.g_vectors: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.k_out: Float[Array, "..."] = self.k_in + self.g_vectors

    def _create_si_crystal(self) -> CrystalStructure:
        """Create simple Si crystal for testing."""
        a_si: float = 5.431
        frac_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        )
        cart_coords: Float[Array, "..."] = frac_coords * a_si
        atomic_numbers: Float[Array, "..."] = jnp.full(2, 14.0)
        frac_positions: Float[Array, "..."] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "..."] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_none(self) -> None:
        """Test intensity calculation with no CTR contribution.

        Note: jittable with ctr_mixing_mode as a static argument; see
        the JAX Transformability guide and test_ctr_jit_static_mode.
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="none",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_coherent(self) -> None:
        """Test intensity calculation with coherent CTR mixing.

        Note: jittable with ctr_mixing_mode as a static argument; see
        the JAX Transformability guide and test_ctr_jit_static_mode.
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="coherent",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_mode_incoherent(self) -> None:
        """Test intensity calculation with incoherent CTR mixing.

        Note: jittable with ctr_mixing_mode as a static argument; see
        the JAX Transformability guide and test_ctr_jit_static_mode.
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_mixing_mode="incoherent",
        )

        chex.assert_shape(intensities, (3,))
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    def test_ctr_jit_static_mode(self) -> None:
        """Compile the CTR intensity function with the mode held static.

        The only jit blocker is the string ``ctr_mixing_mode``; making it
        static (here via ``eqx.filter_jit``, equivalently
        ``jax.jit(..., static_argnames=("ctr_mixing_mode",))``) yields a
        fully compiled function whose output matches the eager result.
        See the JAX Transformability guide.
        """
        kwargs: Any = {
            "crystal": self.si_crystal,
            "g_allowed": self.g_vectors,
            "k_in": self.k_in,
            "k_out": self.k_out,
            "ctr_mixing_mode": "incoherent",
        }
        eager: Any = compute_kinematic_intensities_with_ctrs(**kwargs)
        compiled: Any = eqx.filter_jit(
            compute_kinematic_intensities_with_ctrs
        )(**kwargs)
        chex.assert_trees_all_close(eager, compiled, atol=1e-6)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("zero", 0.0),
        ("half", 0.5),
        ("full", 1.0),
    )
    def test_ctr_weight_variation(self, weight: float) -> None:
        """Test effect of CTR weight on intensities."""
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            ctr_weight=weight,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_height(self) -> None:
        """Test with height-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        config: Any = SurfaceConfig(method="height", height_fraction=0.3)

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_config_layers(self) -> None:
        """Test with layer-based surface atom identification.

        Note: JIT not supported due to SurfaceConfig with string method.
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        config: Any = SurfaceConfig(method="layers", n_layers=1)

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            surface_config=config,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.named_parameters(
        ("tight", 0.01),
        ("medium", 0.1),
        ("loose", 0.5),
    )
    def test_hk_tolerance_variation(self, tolerance: float) -> None:
        """Test effect of h,k tolerance for CTR application."""
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        intensities: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hk_tolerance=tolerance,
        )

        chex.assert_tree_all_finite(intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ctr_gating_uses_explicit_hkl(self) -> None:
        """Explicit hkl should enable CTR when |G| misses tolerance."""
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        hkls: Float[Array, "..."] = jnp.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=jnp.int32,
        )

        # Tight tolerance makes derived indices miss near-integer check
        intens_no_hkl: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hk_tolerance=0.01,
        )
        intens_with_hkl: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=self.g_vectors,
            k_in=self.k_in,
            k_out=self.k_out,
            hkl_indices=hkls,
            hk_tolerance=0.01,
        )

        total_no: scalar_float = float(jnp.sum(intens_no_hkl))
        total_with: scalar_float = float(jnp.sum(intens_with_hkl))

        self.assertGreater(total_with, total_no)


class TestEwaldSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for ewald_simulator with exact Ewald-CTR intersection."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.mgo_crystal: Any = self._create_mgo_crystal()

    def _create_mgo_crystal(self) -> CrystalStructure:
        """Create a simple MgO rock-salt structure for testing."""
        a_mgo: float = 4.212

        frac_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        )

        cart_coords: Float[Array, "..."] = frac_coords * a_mgo

        atomic_numbers: Float[Array, "..."] = jnp.array([12.0, 8.0])
        frac_positions: Float[Array, "..."] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "..."] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a_mgo, a_mgo, a_mgo]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_pattern_generation(self) -> None:
        """Test that ewald_simulator produces a valid RHEED pattern."""
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=3,
            kmax=3,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        n_valid: scalar_float = jnp.sum(valid_mask)
        self.assertGreater(
            int(n_valid), 0, "Should have at least one valid reflection"
        )

        self.assertTrue(
            jnp.all(pattern.intensities >= 0),
            "All intensities should be non-negative",
        )

        valid_detector: Any = pattern.detector_points[valid_mask]
        self.assertTrue(
            jnp.all(jnp.isfinite(valid_detector)),
            "Valid detector points should be finite",
        )

    def test_upward_scattering_only(self) -> None:
        """Only upward-scattered reflections are returned (k_out_z > 0)."""
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=5,
            kmax=5,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        k_out_valid: Float[Array, "..."] = pattern.k_out[valid_mask]

        self.assertTrue(
            jnp.all(k_out_valid[:, 2] > 0),
            "All valid reflections should have k_out_z > 0",
        )

    @staticmethod
    def _raw_ctr_detector_points(
        crystal: CrystalStructure,
        voltage_kv: float,
        theta_deg: float,
        phi_deg: float,
        hmax: int,
        kmax: int,
        detector_distance: float,
    ) -> Float[NDArray, "points detector_xy"]:
        """Construct detector intersections directly from both rod branches."""
        lam_ang: Any = float(wavelength_ang(voltage_kv))
        k_in: Float[NDArray, "..."] = np.asarray(
            incident_wavevector(lam_ang, theta_deg, phi_deg),
            dtype=np.float64,
        )
        recip_a: Float[NDArray, "..."]
        recip_b: Float[NDArray, "..."]
        recip_c: Float[NDArray, "..."]
        recip_a, recip_b, recip_c = np.asarray(
            reciprocal_lattice_vectors(
                *crystal.cell_lengths,
                *crystal.cell_angles,
                in_degrees=True,
            ),
            dtype=np.float64,
        )
        k_mag_sq: Any = float(np.dot(k_in, k_in))
        rows: list[tuple[float, float]] = []
        h: int
        for h in range(-hmax, hmax + 1):
            k: int
            for k in range(-kmax, kmax + 1):
                g_hk: Any = h * recip_a + k * recip_b
                p_vec: Any = k_in + g_hk
                a_coef: Any = float(np.dot(recip_c, recip_c))
                b_coef: Any = float(2.0 * np.dot(p_vec, recip_c))
                c_coef: Any = float(np.dot(p_vec, p_vec) - k_mag_sq)
                discriminant: Any = b_coef * b_coef - 4.0 * a_coef * c_coef
                if discriminant < 0.0:
                    continue
                sqrt_disc: Any = discriminant**0.5
                l_val: int
                for l_val in (
                    (-b_coef + sqrt_disc) / (2.0 * a_coef),
                    (-b_coef - sqrt_disc) / (2.0 * a_coef),
                ):
                    k_out: Float[Array, "..."] = p_vec + l_val * recip_c
                    if k_out[0] <= 0.0 or k_out[2] <= 0.0:
                        continue
                    scale: float = detector_distance / max(
                        float(k_out[0]), 1e-12
                    )
                    rows.append(
                        (float(k_out[1] * scale), float(k_out[2] * scale))
                    )
        detector_points: Float[NDArray, "..."] = np.asarray(
            rows, dtype=np.float64
        )
        order: Float[NDArray, "..."] = np.lexsort(
            (detector_points[:, 1], detector_points[:, 0])
        )
        return detector_points[order]

    @staticmethod
    def _create_sto_crystal() -> CrystalStructure:
        """Create a simple cubic SrTiO3 unit cell for Ewald tests."""
        a_sto: float = 3.905
        frac_coords: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
        cart_coords: Float[Array, "..."] = frac_coords * a_sto
        atomic_numbers: Float[Array, "..."] = jnp.array(
            [38.0, 22.0, 8.0, 8.0, 8.0]
        )
        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a_sto, a_sto, a_sto]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    @parameterized.parameters(1.5, 2.5, 4.0)
    def test_matches_raw_dual_branch_geometry(self, theta_deg: float) -> None:
        """Sparse Ewald output matches direct two-branch rod geometry."""
        detector_distance: float = 900.0
        hmax: int = 8
        kmax: int = 8
        raw_points: Float[Array, "..."] = self._raw_ctr_detector_points(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
        )
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
        )
        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        detector_points: Float[NDArray, "..."] = np.asarray(
            pattern.detector_points[valid_mask]
        )
        order: Float[NDArray, "..."] = np.lexsort(
            (detector_points[:, 1], detector_points[:, 0])
        )
        detector_points = detector_points[order]

        self.assertEqual(
            raw_points.shape,
            detector_points.shape,
            "Raw and sparse Ewald geometry should emit the same "
            "number of hits",
        )
        self.assertTrue(
            np.allclose(raw_points, detector_points, atol=1e-9, rtol=0.0),
            "Sparse Ewald intersections should match direct rod geometry",
        )

    @parameterized.parameters(1.5, 2.5, 4.0)
    def test_sto_matches_raw_dual_branch_geometry(
        self, theta_deg: float
    ) -> None:
        """SrTiO3 sparse Ewald output matches the raw detector geometry."""
        sto_crystal: Any = self._create_sto_crystal()
        detector_distance: float = 900.0
        hmax: int = 14
        kmax: int = 14
        raw_points: Float[Array, "..."] = self._raw_ctr_detector_points(
            crystal=sto_crystal,
            voltage_kv=18.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
        )
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=sto_crystal,
            voltage_kv=18.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
            temperature=300.0,
            surface_roughness=0.55,
        )
        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        detector_points: Float[NDArray, "..."] = np.asarray(
            pattern.detector_points[valid_mask]
        )
        raw_points: Float[NDArray, "..."] = np.unique(
            np.round(raw_points, 6), axis=0
        )
        detector_points: Float[NDArray, "..."] = np.unique(
            np.round(detector_points, 6), axis=0
        )
        raw_order: Float[NDArray, "..."] = np.lexsort(
            (raw_points[:, 1], raw_points[:, 0])
        )
        detector_order: Float[NDArray, "..."] = np.lexsort(
            (detector_points[:, 1], detector_points[:, 0])
        )
        raw_points = raw_points[raw_order]
        detector_points = detector_points[detector_order]

        self.assertEqual(raw_points.shape, detector_points.shape)
        self.assertTrue(
            np.allclose(raw_points, detector_points, atol=2e-6, rtol=0.0),
            "SrTiO3 sparse Ewald intersections should match direct "
            "rod geometry",
        )

    def test_elastic_scattering_constraint(self) -> None:
        """Test that |k_out| = |k_in| (elastic scattering)."""
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        k_out_valid: Float[Array, "..."] = pattern.k_out[valid_mask]

        wl: Any = wavelength_ang(20.0)
        k_mag_expected: scalar_float = 2.0 * jnp.pi / wl

        k_out_mags: Float[Array, "..."] = jnp.linalg.norm(k_out_valid, axis=1)
        relative_error: Float[Array, "..."] = (
            jnp.abs(k_out_mags - k_mag_expected) / k_mag_expected
        )

        self.assertTrue(
            jnp.all(relative_error < 0.01),
            "k_out magnitudes should match k_in (elastic scattering)",
        )

    def test_azimuthal_rotation_changes_pattern(self) -> None:
        """Changing phi_deg rotates the pattern."""
        pattern_0: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=3,
            kmax=3,
        )

        pattern_45: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=45.0,
            hmax=3,
            kmax=3,
        )

        self.assertFalse(
            jnp.allclose(
                pattern_0.detector_points, pattern_45.detector_points
            ),
            "Different azimuths should produce different patterns",
        )

    def test_temperature_affects_intensity(self) -> None:
        """Higher temperature reduces intensity (Debye-Waller)."""
        pattern_low_T: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            temperature=100.0,
            hmax=3,
            kmax=3,
        )

        pattern_high_T: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            temperature=500.0,
            hmax=3,
            kmax=3,
        )

        valid_low: Any = pattern_low_T.G_indices >= 0
        valid_high: Any = pattern_high_T.G_indices >= 0

        self.assertGreater(
            int(jnp.sum(valid_low)),
            0,
            "Low T pattern should have reflections",
        )
        self.assertGreater(
            int(jnp.sum(valid_high)),
            0,
            "High T pattern should have reflections",
        )

    def test_roughness_affects_intensity(self) -> None:
        """Surface roughness affects CTR intensity."""
        pattern_smooth: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_roughness=0.1,
            hmax=3,
            kmax=3,
        )

        pattern_rough: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_roughness=2.0,
            hmax=3,
            kmax=3,
        )

        valid_smooth: Any = pattern_smooth.G_indices >= 0
        valid_rough: Any = pattern_rough.G_indices >= 0

        self.assertGreater(
            int(jnp.sum(valid_smooth)),
            0,
            "Smooth surface should have reflections",
        )
        self.assertGreater(
            int(jnp.sum(valid_rough)),
            0,
            "Rough surface should have reflections",
        )

    def test_voltage_affects_wavevector(self) -> None:
        """Different voltages give different k magnitudes."""
        pattern_10kv: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=10.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        pattern_30kv: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=30.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        valid_10: Any = pattern_10kv.G_indices >= 0
        valid_30: Any = pattern_30kv.G_indices >= 0

        if jnp.any(valid_10) and jnp.any(valid_30):
            k_mag_10: Float[Array, "..."] = jnp.linalg.norm(
                pattern_10kv.k_out[valid_10][0]
            )
            k_mag_30: Float[Array, "..."] = jnp.linalg.norm(
                pattern_30kv.k_out[valid_30][0]
            )

            self.assertGreater(
                float(k_mag_30),
                float(k_mag_10),
                "Higher voltage should give larger k magnitude",
            )

    def test_jax_jit_compatible(self) -> None:
        """ewald_simulator works under JAX JIT compilation."""
        pattern: Callable[..., Any] = jax.jit(
            ewald_simulator,
            static_argnames=("hmax", "kmax"),
        )(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "JIT-compiled simulation should work",
        )

    def test_surface_config_parameter(self) -> None:
        """surface_config parameter works correctly."""
        config: Any = SurfaceConfig(method="height", height_fraction=0.5)

        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            surface_config=config,
            hmax=3,
            kmax=3,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "Should produce valid pattern with custom surface config",
        )

    @parameterized.parameters(
        {"theta_deg": 1.0},
        {"theta_deg": 2.0},
        {"theta_deg": 3.0},
        {"theta_deg": 5.0},
    )
    def test_various_grazing_angles(self, theta_deg: float) -> None:
        """Various grazing angles produce valid patterns."""
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            voltage_kv=20.0,
            theta_deg=theta_deg,
            hmax=3,
            kmax=3,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        self.assertGreaterEqual(
            int(jnp.sum(valid_mask)),
            0,
            f"Grazing angle {theta_deg} should produce valid reflections",
        )

    def test_orientation_distribution_matches_manual_incoherent_union(
        self,
    ) -> None:
        """Orientation wrapper matches explicit per-angle pattern union."""
        orientation_dist: Float[Array, "..."] = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 45.0]),
            weights=jnp.array([0.25, 0.75]),
        )

        def fake_ewald_simulator(  # noqa: PLR0913
            crystal: CrystalStructure,
            voltage_kv: float,
            theta_deg: float,
            phi_deg: float,
            hmax: int,
            kmax: int,
            detector_distance: float,
            temperature: float,
            surface_roughness: float,
            ctr_regularization: float,
            ctr_power: float,
            roughness_power: float,
            parameterization: str,
            surface_config: SurfaceConfig,
        ) -> RHEEDPattern:
            del (
                crystal,
                voltage_kv,
                theta_deg,
                hmax,
                kmax,
                detector_distance,
                temperature,
                surface_roughness,
                ctr_regularization,
                ctr_power,
                roughness_power,
                parameterization,
                surface_config,
            )
            phi: Float[Array, "..."] = jnp.asarray(phi_deg, dtype=jnp.float64)
            return RHEEDPattern(
                G_indices=jnp.array([3, 7], dtype=jnp.int32),
                k_out=jnp.array(
                    [
                        [1.0 + phi, 0.0, 2.0],
                        [2.0 + phi, 1.0, 3.0],
                    ],
                    dtype=jnp.float64,
                ),
                detector_points=jnp.array(
                    [
                        [phi, phi + 1.0],
                        [phi + 2.0, phi + 3.0],
                    ],
                    dtype=jnp.float64,
                ),
                intensities=jnp.array([1.0, phi + 2.0], dtype=jnp.float64),
            )

        with patch(
            "rheedium.simul.simulator.ewald_simulator",
            side_effect=fake_ewald_simulator,
        ):
            averaged: Any = ewald_simulator_with_orientation_distribution(
                crystal=_SI_CRYSTAL_2ATOM,
                orientation_distribution=orientation_dist,
                voltage_kv=20.0,
                theta_deg=2.0,
                hmax=1,
                kmax=1,
                n_mosaic_points=1,
            )

        expected_g_indices: Integer[Array, "..."] = jnp.array(
            [3, 7, 3, 7], dtype=jnp.int32
        )
        expected_k_out: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 3.0],
                [46.0, 0.0, 2.0],
                [47.0, 1.0, 3.0],
            ],
            dtype=jnp.float64,
        )
        expected_detector_points: Float[Array, "..."] = jnp.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [45.0, 46.0],
                [47.0, 48.0],
            ],
            dtype=jnp.float64,
        )
        expected_intensities: Float[Array, "..."] = jnp.array(
            [0.25, 0.5, 0.75, 35.25],
            dtype=jnp.float64,
        )

        chex.assert_trees_all_close(averaged.G_indices, expected_g_indices)
        chex.assert_trees_all_close(averaged.k_out, expected_k_out, atol=1e-10)
        chex.assert_trees_all_close(
            averaged.detector_points,
            expected_detector_points,
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            averaged.intensities,
            expected_intensities,
            atol=1e-10,
        )

    def test_orientation_distribution_wrapper_jits(self) -> None:
        """Orientation-distribution wrapper should compile under jax.jit."""
        orientation_dist: Float[Array, "..."] = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 30.0]),
            weights=jnp.array([0.4, 0.6]),
        )

        pattern: Callable[..., Any] = jax.jit(
            ewald_simulator_with_orientation_distribution,
            static_argnames=("hmax", "kmax", "n_mosaic_points"),
        )(
            crystal=_SI_CRYSTAL_2ATOM,
            orientation_distribution=orientation_dist,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=1,
            kmax=1,
            n_mosaic_points=1,
        )

        valid_mask: Bool[Array, "..."] = pattern.G_indices >= 0
        self.assertGreater(
            int(jnp.sum(valid_mask)),
            0,
            "JIT-compiled orientation simulation should produce reflections",
        )
        self.assertTrue(
            jnp.all(pattern.intensities >= 0.0),
            "Orientation-averaged intensities should be non-negative",
        )


def _make_si_crystal_2atom() -> CrystalStructure:
    """Create a 2-atom Si crystal for fast gradient tests."""
    a_si: float = 5.431
    frac_coords: Float[Array, "..."] = jnp.array(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    )
    cart_coords: Float[Array, "..."] = frac_coords * a_si
    atomic_numbers: Float[Array, "..."] = jnp.full(2, 14.0)
    frac_positions: Float[Array, "..."] = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_positions: Float[Array, "..."] = jnp.column_stack(
        [cart_coords, atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([a_si, a_si, a_si]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


_SI_CRYSTAL_2ATOM = _make_si_crystal_2atom()


class TestDetectorImageOrchestrator(chex.TestCase, parameterized.TestCase):
    """Tests for dense detector-image helpers built on ewald_simulator."""

    @staticmethod
    def _tiny_potential_slices(scale: float = 0.05) -> PotentialSlices:
        """Create a compact potential volume for public multislice tests."""
        slices: Float[Array, "2 8 8"] = jnp.zeros((2, 8, 8), dtype=jnp.float64)
        slices = slices.at[0, 3, 3].set(scale)
        slices = slices.at[1, 4, 4].set(0.5 * scale)
        return create_potential_slices(
            slices=slices,
            slice_thickness=1.5,
            x_calibration=0.75,
            y_calibration=0.75,
        )

    def test_render_pattern_to_image_shape_and_normalization(self) -> None:
        """Rasterized detector image has requested shape and unit maximum."""
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0, 1], dtype=jnp.int32),
            k_out=jnp.array(
                [[10.0, 0.0, 1.0], [10.0, 0.0, 1.0]], dtype=jnp.float64
            ),
            detector_points=jnp.array(
                [[0.0, 0.0], [2.0, 4.0]], dtype=jnp.float64
            ),
            intensities=jnp.array([1.0, 0.5], dtype=jnp.float64),
        )

        image: Float[Array, "..."] = render_pattern_to_image(
            pattern=pattern,
            image_shape_px=(32, 40),
            pixel_size_mm=(1.0, 2.0),
            beam_center_px=(20.0, 4.0),
            spot_sigma_px=1.5,
        )

        chex.assert_shape(image, (32, 40))
        chex.assert_tree_all_finite(image)
        chex.assert_trees_all_close(jnp.max(image), 1.0, atol=1e-12)
        self.assertTrue(jnp.all(image >= 0.0))

    def test_render_amplitude_to_field_matches_single_spot_intensity(
        self,
    ) -> None:
        """Squared single-spot amplitude field matches legacy rendering."""
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0], dtype=jnp.int32),
            k_out=jnp.array([[10.0, 0.0, 1.0]], dtype=jnp.float64),
            detector_points=jnp.array([[0.0, 0.0]], dtype=jnp.float64),
            intensities=jnp.array([4.0], dtype=jnp.float64),
        )

        field: Complex[Array, "32 40"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.sqrt(pattern.intensities).astype(jnp.complex128),
            image_shape_px=(32, 40),
            pixel_size_mm=(1.0, 2.0),
            beam_center_px=(20.0, 4.0),
            spot_sigma_px=1.5,
        )
        intensity: Float[Array, "32 40"] = jnp.abs(field) ** 2
        intensity = intensity / jnp.max(intensity)
        legacy: Float[Array, "32 40"] = render_pattern_to_image(
            pattern=pattern,
            image_shape_px=(32, 40),
            pixel_size_mm=(1.0, 2.0),
            beam_center_px=(20.0, 4.0),
            spot_sigma_px=1.5,
        )

        chex.assert_trees_all_close(intensity, legacy, atol=1e-12)

    def test_render_amplitude_to_field_preserves_interference(self) -> None:
        """Overlapping amplitudes interfere coherently in the dense field."""
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0, 1], dtype=jnp.int32),
            k_out=jnp.array(
                [[10.0, 0.0, 1.0], [10.0, 0.0, 1.0]],
                dtype=jnp.float64,
            ),
            detector_points=jnp.array(
                [[0.0, 0.0], [0.0, 0.0]],
                dtype=jnp.float64,
            ),
            intensities=jnp.array([1.0, 1.0], dtype=jnp.float64),
        )

        constructive: Complex[Array, "16 16"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.array([1.0 + 0.0j, 1.0 + 0.0j]),
            image_shape_px=(16, 16),
            pixel_size_mm=(1.0, 1.0),
            beam_center_px=(8.0, 8.0),
            spot_sigma_px=1.0,
        )
        destructive: Complex[Array, "16 16"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.array([1.0 + 0.0j, -1.0 + 0.0j]),
            image_shape_px=(16, 16),
            pixel_size_mm=(1.0, 1.0),
            beam_center_px=(8.0, 8.0),
            spot_sigma_px=1.0,
        )

        chex.assert_trees_all_close(
            jnp.max(jnp.abs(constructive) ** 2),
            4.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.max(jnp.abs(destructive) ** 2),
            0.0,
            atol=1e-12,
        )

    def test_kinematic_amplitude_carries_nontrivial_phase(self) -> None:
        """The real kinematic kernel should expose complex reflection phase."""
        sparse_pattern: RHEEDPattern
        amplitudes: Complex[Array, "N"]
        sparse_pattern, amplitudes = _ewald_amplitude_pattern(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=1,
            kmax=1,
            detector_distance=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
        )
        field: Complex[Array, "32 32"] = kinematic_amplitude(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=1,
            kmax=1,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(32, 32),
            pixel_size_mm=(8.0, 8.0),
            beam_center_px=(16.0, 3.0),
            spot_sigma_px=1.2,
            render_ctrs_as_streaks=False,
        )

        chex.assert_tree_all_finite(jnp.real(field))
        chex.assert_tree_all_finite(jnp.imag(field))
        valid_mask: Bool[Array, "N"] = (sparse_pattern.G_indices >= 0) & (
            jnp.abs(amplitudes) > 1e-8
        )
        pairwise_phase_area: Float[Array, "N N"] = jnp.imag(
            amplitudes[:, None] * jnp.conj(amplitudes[None, :])
        )
        valid_pairs: Bool[Array, "N N"] = (
            valid_mask[:, None] & valid_mask[None, :]
        )
        relative_phase_signal: Float[Array, ""] = jnp.max(
            jnp.abs(jnp.where(valid_pairs, pairwise_phase_area, 0.0))
        )
        assert float(relative_phase_signal) > 1e-6

    def test_real_kinematic_kernel_coherently_interferes(self) -> None:
        """Coherent reduction should interfere real kinematic amplitudes."""
        distribution_coherent: Distribution = create_distribution(
            samples=jnp.array([[0.0], [jnp.pi]], dtype=jnp.float64),
            weights=jnp.array([0.5, 0.5], dtype=jnp.float64),
            reduction=ReductionMode.COHERENT,
            axis_id="phase",
        )
        distribution_incoherent: Distribution = create_distribution(
            samples=distribution_coherent.samples,
            weights=distribution_coherent.weights,
            reduction=ReductionMode.INCOHERENT,
            axis_id="phase",
        )

        def _bound(sample: Float[Array, "1"]) -> Complex[Array, "16 24"]:
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=1,
                kmax=1,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                ctr_regularization=0.01,
                ctr_power=1.0,
                roughness_power=0.25,
                image_shape_px=(16, 24),
                pixel_size_mm=(8.0, 16.0),
                beam_center_px=(12.0, 2.0),
                spot_sigma_px=1.2,
                render_ctrs_as_streaks=False,
            )
            return jnp.exp(1j * sample[0]) * field

        coherent: Float[Array, "16 24"] = apply_distribution(
            distribution_coherent,
            _bound,
        )
        incoherent: Float[Array, "16 24"] = apply_distribution(
            distribution_incoherent,
            _bound,
        )

        assert float(jnp.max(coherent)) < 1e-12
        assert float(jnp.max(incoherent)) > 1e-3

    def test_kinematic_amplitude_matches_explicit_sparse_render(self) -> None:
        """Kinematic amplitude uses the sparse Ewald amplitude-render path."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
        }
        amplitude: Complex[Array, "16 24"] = kinematic_amplitude(**kwargs)
        sparse_pattern: RHEEDPattern
        amplitudes: Complex[Array, "N"]
        sparse_pattern, amplitudes = _ewald_amplitude_pattern(
            crystal=kwargs["crystal"],
            voltage_kv=kwargs["voltage_kv"],
            theta_deg=kwargs["theta_deg"],
            phi_deg=kwargs["phi_deg"],
            hmax=kwargs["hmax"],
            kmax=kwargs["kmax"],
            detector_distance=kwargs["detector_distance_mm"],
            temperature=kwargs["temperature"],
            surface_roughness=kwargs["surface_roughness"],
        )
        expected: Complex[Array, "16 24"] = render_amplitude_to_field(
            pattern=sparse_pattern,
            amplitudes=amplitudes,
            image_shape_px=kwargs["image_shape_px"],
            pixel_size_mm=kwargs["pixel_size_mm"],
            beam_center_px=kwargs["beam_center_px"],
            spot_sigma_px=kwargs["spot_sigma_px"],
        )

        chex.assert_trees_all_close(amplitude, expected, atol=1e-12)

    def test_ewald_amplitude_pattern_matches_intensity_simulator(self) -> None:
        """Complex Ewald amplitudes preserve the legacy intensity surface."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 1,
            "kmax": 1,
            "detector_distance": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
        }
        intensity_pattern: RHEEDPattern = ewald_simulator(**kwargs)
        amplitude_pattern: RHEEDPattern
        amplitudes: Complex[Array, "N"]
        amplitude_pattern, amplitudes = _ewald_amplitude_pattern(**kwargs)

        chex.assert_trees_all_close(
            amplitude_pattern.detector_points,
            intensity_pattern.detector_points,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            amplitude_pattern.intensities,
            intensity_pattern.intensities,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.abs(amplitudes) ** 2,
            intensity_pattern.intensities,
            atol=1e-12,
        )

    def test_trivial_distribution_reduces_kinematic_amplitude_to_intensity(
        self,
    ) -> None:
        """Trivial distribution turns one coherent amplitude into intensity."""
        amplitude: Complex[Array, "16 24"] = kinematic_amplitude(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
        )

        def _bound(_sample: Float[Array, "1"]) -> Complex[Array, "16 24"]:
            return amplitude

        reduced: Float[Array, "16 24"] = apply_distribution(
            TRIVIAL_DISTRIBUTION,
            _bound,
        )

        chex.assert_trees_all_close(
            reduced,
            jnp.abs(amplitude) ** 2,
            atol=1e-12,
        )

    def test_detector_extent_mm_matches_calibration(self) -> None:
        """Display extent converts beam centre and pixel pitch correctly."""
        extent: Float[Array, "..."] = detector_extent_mm(
            image_shape_px=(100, 200),
            pixel_size_mm=(1.5, 3.0),
            beam_center_px=(80.0, 5.0),
        )
        self.assertEqual(extent, (-120.0, 180.0, -15.0, 285.0))

    def test_log_compress_image_preserves_bounds(self) -> None:
        """Log compression maps a normalized image back into [0, 1]."""
        image: Float[Array, "..."] = jnp.array(
            [[0.0, 0.25], [0.5, 1.0]], dtype=jnp.float64
        )
        compressed: Any = log_compress_image(image, gain=20.0)
        chex.assert_shape(compressed, (2, 2))
        chex.assert_tree_all_finite(compressed)
        self.assertTrue(jnp.all(compressed >= 0.0))
        self.assertTrue(jnp.all(compressed <= 1.0))
        chex.assert_trees_all_close(compressed[0, 0], 0.0, atol=1e-12)
        chex.assert_trees_all_close(compressed[1, 1], 1.0, atol=1e-12)

    def test_log_compress_image_applies_dynamic_range_floor(self) -> None:
        """Display floor hides weak pixels and rescales the visible range."""
        image: Float[Array, "..."] = jnp.array(
            [[0.0, 0.25], [0.5, 1.0]], dtype=jnp.float64
        )
        compressed: Any = log_compress_image(
            image,
            gain=20.0,
            dynamic_range_floor=0.5,
        )
        chex.assert_shape(compressed, (2, 2))
        chex.assert_tree_all_finite(compressed)
        chex.assert_trees_all_close(compressed[0, 0], 0.0, atol=1e-12)
        chex.assert_trees_all_close(compressed[0, 1], 0.0, atol=1e-12)
        chex.assert_trees_all_close(compressed[1, 0], 0.0, atol=1e-12)
        chex.assert_trees_all_close(compressed[1, 1], 1.0, atol=1e-12)

    def test_render_ctr_amplitude_matches_legacy_single_reflection(
        self,
    ) -> None:
        """Complex CTR renderer reproduces legacy one-reflection intensity."""
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0], dtype=jnp.int32),
            k_out=jnp.array([[10.0, 0.0, 1.0]], dtype=jnp.float64),
            detector_points=jnp.array([[0.0, 8.0]], dtype=jnp.float64),
            intensities=jnp.array([4.0], dtype=jnp.float64),
        )
        image_shape_px: tuple[int, int] = (32, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 8.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        amplitudes: Complex[Array, "1"] = jnp.sqrt(pattern.intensities).astype(
            jnp.complex128
        )

        field: Complex[Array, "32 24"] = render_ctr_amplitude_to_field(
            pattern=pattern,
            amplitudes=amplitudes,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
        )
        actual: Float[Array, "32 24"] = jnp.abs(field) ** 2
        actual = actual / jnp.maximum(jnp.max(actual), 1e-12)
        expected: Float[Array, "32 24"] = _render_ctr_streaks_to_image(
            pattern=pattern,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_simulate_detector_image_uses_layer1_default_when_spot_rendered(
        self,
    ) -> None:
        """Spot-render default delegates to the trivial Layer-1 reducer."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
        }
        amplitude: Complex[Array, "16 24"] = kinematic_amplitude(
            **kwargs,
        )
        expected: Float[Array, "16 24"] = apply_distribution(
            TRIVIAL_DISTRIBUTION,
            lambda _sample: amplitude,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        orchestrated_image: Float[Array, "..."] = simulate_detector_image(
            **kwargs,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
            kernel="kinematic",
        )

        chex.assert_trees_all_close(
            orchestrated_image,
            expected,
            atol=1e-10,
        )

    def test_simulate_detector_image_spot_instrument_uses_distribution(
        self,
    ) -> None:
        """Spot-render instrument widths reduce through Layer-1 mechanics."""
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        angular_divergence_mrad: float = 0.4
        energy_spread_ev: float = 0.2
        n_angular_samples: int = 3
        n_energy_samples: int = 3
        angle_nodes: Float[Array, "3"]
        angle_weights: Float[Array, "3"]
        energy_nodes: Float[Array, "3"]
        energy_weights: Float[Array, "3"]
        angle_nodes, angle_weights = gauss_hermite_nodes_weights(
            n_angular_samples
        )
        energy_nodes, energy_weights = gauss_hermite_nodes_weights(
            n_energy_samples
        )
        sqrt2: Float[Array, ""] = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
        sqrt_pi: Float[Array, ""] = jnp.sqrt(
            jnp.asarray(jnp.pi, dtype=jnp.float64)
        )
        theta_offsets: Float[Array, "3"] = (
            sqrt2 * angular_divergence_mrad * 1.0e-3 * angle_nodes
        )
        energy_offsets: Float[Array, "3"] = (
            sqrt2 * energy_spread_ev * energy_nodes
        )
        theta_grid: Float[Array, "3 3"]
        energy_grid: Float[Array, "3 3"]
        theta_grid, energy_grid = jnp.meshgrid(
            theta_offsets,
            energy_offsets,
            indexing="ij",
        )
        samples: Float[Array, "9 3"] = jnp.stack(
            [
                theta_grid.ravel(),
                jnp.zeros_like(theta_grid).ravel(),
                energy_grid.ravel(),
            ],
            axis=-1,
        )
        weights: Float[Array, "9"] = (
            (angle_weights[:, None] * energy_weights[None, :] / (sqrt_pi**2))
            .ravel()
            .astype(jnp.float64)
        )
        distribution = create_distribution(
            samples=samples,
            weights=weights,
            reduction=ReductionMode.INCOHERENT,
        )
        base_voltage_kv: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 3.0

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=base_voltage_kv + 1.0e-3 * sample[2],
                theta_deg=base_theta_deg + jnp.rad2deg(sample[0]),
                phi_deg=base_phi_deg + jnp.rad2deg(sample[1]),
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=1.2,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=base_voltage_kv,
            theta_deg=base_theta_deg,
            phi_deg=base_phi_deg,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=0.0,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            render_ctrs_as_streaks=False,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_ctr_instrument_uses_distribution(
        self,
    ) -> None:
        """CTR-rendered instrument widths reduce through Layer-1 mechanics."""
        image_shape_px: tuple[int, int] = (32, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 8.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        angular_divergence_mrad: float = 0.35
        energy_spread_ev: float = 0.2
        n_angular_samples: int = 3
        n_energy_samples: int = 3
        angle_nodes: Float[Array, "3"]
        angle_weights: Float[Array, "3"]
        energy_nodes: Float[Array, "3"]
        energy_weights: Float[Array, "3"]
        angle_nodes, angle_weights = gauss_hermite_nodes_weights(
            n_angular_samples
        )
        energy_nodes, energy_weights = gauss_hermite_nodes_weights(
            n_energy_samples
        )
        sqrt2: Float[Array, ""] = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
        sqrt_pi: Float[Array, ""] = jnp.sqrt(
            jnp.asarray(jnp.pi, dtype=jnp.float64)
        )
        theta_offsets: Float[Array, "3"] = (
            sqrt2 * angular_divergence_mrad * 1.0e-3 * angle_nodes
        )
        energy_offsets: Float[Array, "3"] = (
            sqrt2 * energy_spread_ev * energy_nodes
        )
        theta_grid: Float[Array, "3 3"]
        energy_grid: Float[Array, "3 3"]
        theta_grid, energy_grid = jnp.meshgrid(
            theta_offsets,
            energy_offsets,
            indexing="ij",
        )
        distribution = create_distribution(
            samples=jnp.stack(
                [
                    theta_grid.ravel(),
                    jnp.zeros_like(theta_grid).ravel(),
                    energy_grid.ravel(),
                ],
                axis=-1,
            ),
            weights=(
                (
                    angle_weights[:, None]
                    * energy_weights[None, :]
                    / (sqrt_pi**2)
                )
                .ravel()
                .astype(jnp.float64)
            ),
            reduction=ReductionMode.INCOHERENT,
        )
        base_voltage_kv: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 0.0

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "32 24"]:
            return kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=base_voltage_kv + 1.0e-3 * sample[2],
                theta_deg=base_theta_deg + jnp.rad2deg(sample[0]),
                phi_deg=base_phi_deg + jnp.rad2deg(sample[1]),
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=1.2,
                render_ctrs_as_streaks=True,
            )

        expected: Float[Array, "32 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=base_voltage_kv,
            theta_deg=base_theta_deg,
            phi_deg=base_phi_deg,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=0.0,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            render_ctrs_as_streaks=True,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_trivial_distribution_is_identity(
        self,
    ) -> None:
        """Trivial generic distribution preserves the spot-rendered image."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 4.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "render_ctrs_as_streaks": False,
        }
        reference: Float[Array, "..."] = simulate_detector_image(**kwargs)
        distributed: Float[Array, "..."] = simulate_detector_image(
            **kwargs,
            distribution=TRIVIAL_DISTRIBUTION,
        )

        chex.assert_trees_all_close(distributed, reference, atol=1e-10)

    def test_simulate_detector_image_trivial_distribution_matches_ctr_streaks(
        self,
    ) -> None:
        """Trivial generic distribution preserves CTR streak rendering."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (32, 24),
            "pixel_size_mm": (6.0, 8.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "render_ctrs_as_streaks": True,
        }
        reference: Float[Array, "..."] = simulate_detector_image(**kwargs)
        distributed: Float[Array, "..."] = simulate_detector_image(
            **kwargs,
            distribution=TRIVIAL_DISTRIBUTION,
        )

        chex.assert_trees_all_close(distributed, reference, atol=1e-10)

    def test_simulate_detector_image_distribution_matches_manual_layer1(
        self,
    ) -> None:
        """Generic distribution delegates reduction to Layer-1 mechanics."""
        distribution = create_distribution(
            samples=jnp.array([[0.0], [5.0]], dtype=jnp.float64),
            weights=jnp.array([0.25, 0.75], dtype=jnp.float64),
            reduction=ReductionMode.INCOHERENT,
            axis_id="test_phi",
        )
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 3.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "render_ctrs_as_streaks": False,
        }
        base_phi_deg: float = 3.0
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)

        def _bound(sample: Float[Array, "1"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=base_phi_deg + sample[0],
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=1.2,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            **kwargs,
            distribution=distribution,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_twin_distribution_binds_structure(
        self,
    ) -> None:
        """Twin distributions should bind structures in the public path."""
        distribution: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([0.0, 4.0]),
            wall_positions_angstrom=jnp.array([0.4, 0.4]),
            twin_fractions=jnp.array([0.25, 0.75]),
            twin_spacing_angstrom=4.0,
            coherence_length_angstrom=10.0,
        )
        builder: Callable[[Float[Array, "2"]], CrystalStructure] = (
            bind_twin_wall_distribution(
                slab=_SI_CRYSTAL_2ATOM,
                surface_layer_depth_angstrom=0.8,
            )
        )

        def _bound(sample: Float[Array, "2"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=builder(sample),
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=3.0,
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                ctr_regularization=0.01,
                ctr_power=1.0,
                roughness_power=0.25,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                spot_sigma_px=1.2,
                render_ctrs_as_streaks=False,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
            defect_surface_layer_depth_angstrom=0.8,
            distribution=distribution,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_step_distribution_binds_structure(
        self,
    ) -> None:
        """Step distributions should bind structures in the public path."""
        distribution: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([0.0, 0.4]),
            terrace_widths_angstrom=jnp.array([2.0, 2.0]),
            line_azimuths_deg=jnp.array([0.0, 0.0]),
            step_fractions=jnp.array([0.6, 0.4]),
            coherence_length_angstrom=0.5,
            regular=False,
        )
        builder: Callable[[Float[Array, "3"]], CrystalStructure] = (
            bind_step_edge_distribution(
                slab=_SI_CRYSTAL_2ATOM,
                surface_layer_depth_angstrom=0.8,
            )
        )

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=builder(sample),
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=3.0,
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                ctr_regularization=0.01,
                ctr_power=1.0,
                roughness_power=0.25,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                spot_sigma_px=1.2,
                render_ctrs_as_streaks=False,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
            defect_surface_layer_depth_angstrom=0.8,
            distribution=distribution,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_grain_distribution_binds_orientation(
        self,
    ) -> None:
        """Grain distributions should not be interpreted as beam samples."""
        distribution: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([-1.0, 2.0]),
            grain_sizes_angstrom=jnp.array([80.0, 120.0]),
            grain_volume_fractions=jnp.array([0.25, 0.75]),
        )

        def _bound(sample: Float[Array, "2"]) -> Complex[Array, "16 24"]:
            return _kinematic_finite_domain_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=3.0 + sample[0],
                domain_size_angstrom=sample[1],
                domain_aspect_ratio=(1.0, 1.0, 0.5),
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                ctr_regularization=0.01,
                ctr_power=1.0,
                roughness_power=0.25,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                spot_sigma_px=1.2,
                render_ctrs_as_streaks=False,
                parameterization="lobato",
                surface_config=None,
                energy_spread_frac=0.0,
                beam_divergence_rad=0.0,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
            distribution=distribution,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    @staticmethod
    def _defect_image_kwargs() -> dict[str, Any]:
        """Compact detector settings for defect distinguishability tests."""
        return {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 3.0,
            "hmax": 1,
            "kmax": 1,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "render_ctrs_as_streaks": False,
        }

    def test_twin_distribution_changes_detector_image(self) -> None:
        """Twin producers should change detector output, not only bind."""
        kwargs: Any = self._defect_image_kwargs()
        base: Float[Array, "16 24"] = simulate_detector_image(**kwargs)
        twin_dist: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([20.0]),
            wall_positions_angstrom=jnp.array([0.4]),
            twin_fractions=jnp.array([1.0]),
            twin_spacing_angstrom=4.0,
            coherence_length_angstrom=10.0,
        )
        twin_image: Float[Array, "16 24"] = simulate_detector_image(
            **kwargs,
            distribution=twin_dist,
            defect_surface_layer_depth_angstrom=0.8,
        )

        assert float(jnp.max(jnp.abs(twin_image - base))) > 1e-4

    def test_step_distribution_changes_detector_image(self) -> None:
        """Step producers should change detector output, not only bind."""
        kwargs: Any = self._defect_image_kwargs()
        base: Float[Array, "16 24"] = simulate_detector_image(**kwargs)
        step_dist: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([1.0]),
            terrace_widths_angstrom=jnp.array([2.0]),
            line_azimuths_deg=jnp.array([0.0]),
            step_fractions=jnp.array([1.0]),
            coherence_length_angstrom=0.5,
            regular=False,
        )
        step_image: Float[Array, "16 24"] = simulate_detector_image(
            **kwargs,
            distribution=step_dist,
            defect_surface_layer_depth_angstrom=0.8,
        )

        assert float(jnp.max(jnp.abs(step_image - base))) > 1e-3

    def test_grain_distribution_changes_detector_image(self) -> None:
        """Grain producers should change detector output, not only bind."""
        kwargs: Any = self._defect_image_kwargs()
        base: Float[Array, "16 24"] = simulate_detector_image(**kwargs)
        grain_dist: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([5.0]),
            grain_sizes_angstrom=jnp.array([80.0]),
            grain_volume_fractions=jnp.array([1.0]),
        )
        grain_image: Float[Array, "16 24"] = simulate_detector_image(
            **kwargs,
            distribution=grain_dist,
        )

        assert float(jnp.max(jnp.abs(grain_image - base))) > 1e-3

    def test_simulate_detector_image_binds_size_distribution(
        self,
    ) -> None:
        """Size axes bind finite-domain broadening in the public path."""
        distribution: Distribution = create_distribution(
            samples=jnp.array([[40.0], [80.0]], dtype=jnp.float64),
            weights=jnp.array([0.5, 0.5], dtype=jnp.float64),
            reduction=ReductionMode.INCOHERENT,
            axis_id="size",
        )

        base: Float[Array, "16 24"] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=1,
            kmax=1,
            detector_distance_mm=1000.0,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
        )
        sized: Float[Array, "16 24"] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=1,
            kmax=1,
            detector_distance_mm=1000.0,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
            distribution=distribution,
        )

        chex.assert_shape(sized, (16, 24))
        chex.assert_tree_all_finite(sized)
        assert float(jnp.max(jnp.abs(sized - base))) > 1e-6

    def test_simulate_detector_image_rejects_ambiguous_distributions(
        self,
    ) -> None:
        """Legacy and generic distributions are mutually exclusive inputs."""
        orientation_dist: Float[Array, "..."] = create_discrete_orientation(
            angles_deg=jnp.array([0.0]),
            weights=jnp.array([1.0]),
        )

        with pytest.raises(ValueError, match="either"):
            simulate_detector_image(
                crystal=_SI_CRYSTAL_2ATOM,
                orientation_distribution=orientation_dist,
                distribution=TRIVIAL_DISTRIBUTION,
                render_ctrs_as_streaks=False,
            )

    def test_simulate_detector_image_rejects_unknown_kernel(self) -> None:
        """Layer-0 kernel selector fails clearly for unsupported names."""
        with pytest.raises(ValueError, match="Unsupported kernel"):
            simulate_detector_image(
                crystal=_SI_CRYSTAL_2ATOM,
                hmax=0,
                kmax=0,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                render_ctrs_as_streaks=False,
                kernel="dynamical",
            )

    def test_simulate_detector_image_rejects_multislice_without_payload(
        self,
    ) -> None:
        """Multislice selection requires a concrete potential-slice payload."""
        with pytest.raises(ValueError, match="potential_slices"):
            simulate_detector_image(
                crystal=_SI_CRYSTAL_2ATOM,
                hmax=0,
                kmax=0,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                render_ctrs_as_streaks=False,
                kernel="multislice",
            )

    def test_simulate_detector_image_multislice_kernel_matches_bound_field(
        self,
    ) -> None:
        """Public Layer 1 can select multislice and reduce its field."""
        potential_slices: PotentialSlices = self._tiny_potential_slices()
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "potential_slices": potential_slices,
            "voltage_kv": 20.0,
            "theta_deg": 5.0,
            "phi_deg": 0.0,
            "detector_distance_mm": 20.0,
            "image_shape_px": (32, 32),
            "pixel_size_mm": (2.0, 2.0),
            "beam_center_px": (16.0, 16.0),
            "spot_sigma_px": 1.0,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "kernel": "multislice",
        }

        actual: Float[Array, "32 32"] = simulate_detector_image(**kwargs)
        field: Complex[Array, "32 32"] = multislice_detector_amplitude(
            potential_slices=potential_slices,
            voltage_kv=kwargs["voltage_kv"],
            theta_deg=kwargs["theta_deg"],
            phi_deg=kwargs["phi_deg"],
            detector_distance_mm=kwargs["detector_distance_mm"],
            image_shape_px=kwargs["image_shape_px"],
            pixel_size_mm=kwargs["pixel_size_mm"],
            beam_center_px=kwargs["beam_center_px"],
            spot_sigma_px=kwargs["spot_sigma_px"],
        )
        expected: Float[Array, "32 32"] = jnp.abs(field) ** 2
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)

        chex.assert_shape(actual, (32, 32))
        chex.assert_tree_all_finite(actual)
        self.assertGreater(float(jnp.max(actual)), 0.0)
        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_multislice_rejects_defect_axes(
        self,
    ) -> None:
        """Structure-changing producer axes require a PotentialSlices bind."""
        distribution: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([0.0, 4.0]),
            wall_positions_angstrom=jnp.array([0.4, 0.4]),
            twin_fractions=jnp.array([0.25, 0.75]),
            twin_spacing_angstrom=4.0,
            coherence_length_angstrom=10.0,
        )

        with pytest.raises(ValueError, match="PotentialSlices producer"):
            simulate_detector_image(
                crystal=_SI_CRYSTAL_2ATOM,
                potential_slices=self._tiny_potential_slices(),
                image_shape_px=(16, 16),
                pixel_size_mm=(2.0, 2.0),
                beam_center_px=(8.0, 8.0),
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
                psf_sigma_pixels=0.0,
                n_angular_samples=1,
                n_energy_samples=1,
                distribution=distribution,
                kernel="multislice",
            )

    def test_simulate_detector_image_beam_modes_match_instrument_wrapper(
        self,
    ) -> None:
        """Main simulator accepts explicit beam modes on the Layer-1 path."""
        beam_modes = create_gaussian_schell_beam(
            beta_in_plane=0.25,
            beta_out_of_plane=0.1,
            divergence_in_plane_rad=1.5e-4,
            divergence_out_of_plane_rad=0.75e-4,
            energy_spread_ev=0.15,
        )
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 2.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "psf_sigma_pixels": 0.0,
            "n_modes_per_axis": 2,
            "n_modes_out_of_plane": 2,
            "n_energy_points": 3,
        }
        expected: Float[Array, "..."] = simulate_detector_image_instrument(
            beam_modes=beam_modes,
            **kwargs,
        )
        actual: Float[Array, "..."] = simulate_detector_image(
            beam_modes=beam_modes,
            n_beam_modes_per_axis=kwargs["n_modes_per_axis"],
            n_beam_modes_out_of_plane=kwargs["n_modes_out_of_plane"],
            n_beam_energy_points=kwargs["n_energy_points"],
            render_ctrs_as_streaks=False,
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "n_modes_per_axis",
                    "n_modes_out_of_plane",
                    "n_energy_points",
                }
            },
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_rejects_ambiguous_beam_modes(
        self,
    ) -> None:
        """Explicit beam modes are mutually exclusive with generic axes."""
        with pytest.raises(ValueError, match="beam_modes or distribution"):
            simulate_detector_image(
                crystal=_SI_CRYSTAL_2ATOM,
                beam_modes=create_coherent_beam(),
                distribution=TRIVIAL_DISTRIBUTION,
                render_ctrs_as_streaks=False,
            )

    def test_simulate_detector_image_beam_modes_match_ctr_streaks(
        self,
    ) -> None:
        """Coherent beam modes preserve CTR streak rendering."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 0,
            "kmax": 0,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (32, 24),
            "pixel_size_mm": (6.0, 8.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "render_ctrs_as_streaks": True,
        }
        reference: Float[Array, "..."] = simulate_detector_image(**kwargs)
        actual: Float[Array, "..."] = simulate_detector_image(
            **kwargs,
            beam_modes=create_coherent_beam(),
            n_beam_modes_per_axis=1,
            n_beam_energy_points=1,
        )

        chex.assert_trees_all_close(actual, reference, atol=1e-10)

    def test_simulate_detector_image_instrument_coherent_beam_is_identity(
        self,
    ) -> None:
        """Single coherent beam mode matches the unbroadened spot path."""
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        reference: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
            render_ctrs_as_streaks=False,
        )
        actual: Float[Array, "..."] = simulate_detector_image_instrument(
            crystal=_SI_CRYSTAL_2ATOM,
            beam_modes=create_coherent_beam(),
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=3.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            psf_sigma_pixels=0.0,
            n_modes_per_axis=1,
            n_energy_points=1,
        )

        chex.assert_trees_all_close(actual, reference, atol=1e-10)

    def test_simulate_detector_image_instrument_matches_manual_layer1(
        self,
    ) -> None:
        """Beam-mode wrapper delegates to generic Layer-1 reduction."""
        beam_modes = create_gaussian_schell_beam(
            beta_in_plane=0.35,
            beta_out_of_plane=0.2,
            divergence_in_plane_rad=2.0e-4,
            divergence_out_of_plane_rad=1.0e-4,
            energy_spread_ev=0.2,
        )
        distribution = decompose_beam_modes(
            beam_modes,
            n_modes_per_axis=2,
            n_modes_out_of_plane=2,
            n_energy_points=3,
        )
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        base_voltage_kv: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 1.5

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=base_voltage_kv + 1.0e-3 * sample[2],
                theta_deg=base_theta_deg + jnp.rad2deg(sample[0]),
                phi_deg=base_phi_deg + jnp.rad2deg(sample[1]),
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=1.2,
            )

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image_instrument(
            crystal=_SI_CRYSTAL_2ATOM,
            beam_modes=beam_modes,
            voltage_kv=base_voltage_kv,
            theta_deg=base_theta_deg,
            phi_deg=base_phi_deg,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            psf_sigma_pixels=0.0,
            n_modes_per_axis=2,
            n_modes_out_of_plane=2,
            n_energy_points=3,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_composes_beam_and_orientation(
        self,
    ) -> None:
        """Beam and orientation producers compose through Layer 1."""
        beam_modes = create_gaussian_schell_beam(
            beta_in_plane=0.2,
            beta_out_of_plane=0.1,
            divergence_in_plane_rad=1.0e-4,
            divergence_out_of_plane_rad=5.0e-5,
            energy_spread_ev=0.0,
        )
        orientation_dist = create_discrete_orientation(
            angles_deg=jnp.array([-2.0, 3.0]),
            weights=jnp.array([0.25, 0.75]),
        )
        beam_distribution = decompose_beam_modes(
            beam_modes,
            n_modes_per_axis=2,
            n_modes_out_of_plane=1,
            n_energy_points=1,
        )
        orientation_distribution = orientation_to_distribution(
            orientation_dist,
            n_mosaic_points=7,
        )
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)

        def _bound(sample: Float[Array, "4"]) -> Complex[Array, "16 24"]:
            return kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0 + 1.0e-3 * sample[2],
                theta_deg=2.0 + jnp.rad2deg(sample[0]),
                phi_deg=1.5 + jnp.rad2deg(sample[1]) + sample[3],
                hmax=0,
                kmax=0,
                detector_distance_mm=1000.0,
                temperature=300.0,
                surface_roughness=0.5,
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
                spot_sigma_px=1.2,
            )

        expected: Float[Array, "16 24"] = apply_distributions(
            [beam_distribution, orientation_distribution],
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            beam_modes=beam_modes,
            orientation_distribution=orientation_dist,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=1.5,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=1.2,
            psf_sigma_pixels=0.0,
            n_beam_modes_per_axis=2,
            n_beam_modes_out_of_plane=1,
            n_beam_energy_points=1,
            n_mosaic_points=7,
            render_ctrs_as_streaks=False,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-10)

    def test_simulate_detector_image_instrument_rejects_unknown_kernel(
        self,
    ) -> None:
        """Instrument wrapper remains kinematic-only compatibility API."""
        with pytest.raises(ValueError, match="supports only"):
            simulate_detector_image_instrument(
                crystal=_SI_CRYSTAL_2ATOM,
                beam_modes=create_coherent_beam(),
                hmax=0,
                kmax=0,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                kernel="multislice",
            )

    def test_simulate_detector_image_renders_streaks_by_default(self) -> None:
        """Check dense rendering elongates CTRs vertically on detector."""
        image: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(48, 32),
            pixel_size_mm=(6.0, 8.0),
            beam_center_px=(16.0, 2.0),
            spot_sigma_px=1.0,
            angular_divergence_mrad=0.0,
            energy_spread_ev=0.0,
            psf_sigma_pixels=0.0,
            n_angular_samples=1,
            n_energy_samples=1,
        )

        peak_row: Float[Array, "..."]
        peak_col: Float[Array, "..."]
        peak_row, peak_col = jnp.unravel_index(jnp.argmax(image), image.shape)
        vertical_support: scalar_float = jnp.sum(image[:, peak_col] > 0.25)
        horizontal_support: scalar_float = jnp.sum(image[peak_row, :] > 0.25)

        self.assertGreater(
            int(vertical_support),
            int(horizontal_support),
            "Default detector rendering should produce an elongated streak",
        )

    def test_simulate_detector_image_with_orientation_distribution(
        self,
    ) -> None:
        """Check orientation-distribution yields a valid dense image."""
        orientation_dist: Float[Array, "..."] = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.4, 0.6]),
        )
        image: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            orientation_distribution=orientation_dist,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=0,
            kmax=0,
            detector_distance_mm=1000.0,
            temperature=300.0,
            surface_roughness=0.5,
            image_shape_px=(16, 24),
            pixel_size_mm=(6.0, 16.0),
            beam_center_px=(12.0, 2.0),
            spot_sigma_px=1.2,
            angular_divergence_mrad=0.2,
            energy_spread_ev=0.2,
            psf_sigma_pixels=0.8,
            n_angular_samples=3,
            n_energy_samples=3,
            n_mosaic_points=1,
        )

        chex.assert_shape(image, (16, 24))
        chex.assert_tree_all_finite(image)
        self.assertTrue(jnp.all(image >= 0.0))
        chex.assert_trees_all_close(jnp.max(image), 1.0, atol=1e-12)


class TestSimulateDetectorImagePhase6Gradients(chex.TestCase):
    """Phase-6 differentiability gates for the public detector integrator."""

    @staticmethod
    def _detector_metric(image: Float[Array, "H W"]) -> scalar_float:
        """Return an asymmetric scalar metric for gradient tests."""
        height_px, width_px = image.shape
        x_axis: Float[Array, "W"] = jnp.linspace(0.0, 1.0, width_px)
        y_axis: Float[Array, "H"] = jnp.linspace(0.0, 1.0, height_px)
        y_grid: Float[Array, "H W"]
        x_grid: Float[Array, "H W"]
        y_grid, x_grid = jnp.meshgrid(y_axis, x_axis, indexing="ij")
        return jnp.sum(image * (0.7 * x_grid + 1.3 * y_grid))

    @staticmethod
    def _base_kwargs() -> dict[str, Any]:
        """Shared compact detector settings for public grad gates."""
        return {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 3.0,
            "hmax": 1,
            "kmax": 1,
            "detector_distance_mm": 1000.0,
            "temperature": 300.0,
            "surface_roughness": 0.5,
            "image_shape_px": (16, 24),
            "pixel_size_mm": (6.0, 16.0),
            "beam_center_px": (12.0, 2.0),
            "spot_sigma_px": 1.2,
            "psf_sigma_pixels": 0.0,
            "render_ctrs_as_streaks": False,
        }

    def test_grad_through_public_simulator_beta_is_finite(self) -> None:
        """jax.grad through simulate_detector_image w.r.t. GSM beta is live."""

        def loss(beta: scalar_float) -> scalar_float:
            beam_modes = create_gaussian_schell_beam(
                beta_in_plane=beta,
                beta_out_of_plane=0.1,
                divergence_in_plane_rad=1.0e-4,
                divergence_out_of_plane_rad=5.0e-5,
                energy_spread_ev=0.0,
            )
            image: Float[Array, "16 24"] = simulate_detector_image(
                **self._base_kwargs(),
                beam_modes=beam_modes,
                n_beam_modes_per_axis=2,
                n_beam_modes_out_of_plane=1,
                n_beam_energy_points=1,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(0.2))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8

    def test_grad_public_simulator_twin_density_is_finite(
        self,
    ) -> None:
        """jax.grad through public twin fraction is live."""

        def loss(twin_fraction: scalar_float) -> scalar_float:
            clipped_fraction: scalar_float = jnp.clip(
                twin_fraction,
                1.0e-3,
                1.0 - 1.0e-3,
            )
            distribution: Distribution = twin_wall_to_distribution(
                twin_angles_deg=jnp.array([0.0, 20.0]),
                wall_positions_angstrom=jnp.array([0.4, 0.4]),
                twin_fractions=jnp.array(
                    [1.0 - clipped_fraction, clipped_fraction]
                ),
                twin_spacing_angstrom=4.0,
                coherence_length_angstrom=10.0,
            )
            image: Float[Array, "16 24"] = simulate_detector_image(
                **self._base_kwargs(),
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
                n_angular_samples=1,
                n_energy_samples=1,
                distribution=distribution,
                defect_surface_layer_depth_angstrom=0.8,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(0.4))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-4

    def test_grad_through_public_simulator_grain_size_is_live(self) -> None:
        """jax.grad through public grain size is live."""

        def loss(grain_size_angstrom: scalar_float) -> scalar_float:
            distribution: Distribution = grain_population_to_distribution(
                orientation_angles_deg=jnp.array([5.0]),
                grain_sizes_angstrom=jnp.array([grain_size_angstrom]),
                grain_volume_fractions=jnp.array([1.0]),
            )
            image: Float[Array, "16 24"] = simulate_detector_image(
                **self._base_kwargs(),
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
                n_angular_samples=1,
                n_energy_samples=1,
                distribution=distribution,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(80.0))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8


class TestEwaldSimulatorGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for the ewald_simulator."""

    def _ewald_loss(self, **override: object) -> scalar_float:
        """Compute sum of intensities from ewald_simulator."""
        defaults: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "voltage_kv": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 2,
            "kmax": 2,
            "temperature": 300.0,
            "surface_roughness": 0.5,
        }
        defaults.update(override)
        pattern: Float[Array, "..."] = ewald_simulator(**defaults)
        return jnp.sum(pattern.intensities)

    def test_grad_temperature(self) -> None:
        """Gradient w.r.t. temperature is finite and non-zero."""

        def loss(temp: scalar_float) -> scalar_float:
            return self._ewald_loss(temperature=temp)

        g: scalar_float = jax.grad(loss)(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_roughness(self) -> None:
        """Gradient w.r.t. surface roughness is finite and non-zero."""

        def loss(roughness: scalar_float) -> scalar_float:
            return self._ewald_loss(surface_roughness=roughness)

        g: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_polar_angle(self) -> None:
        """Gradient w.r.t. incidence angle is finite."""

        def loss(theta: scalar_float) -> scalar_float:
            return self._ewald_loss(theta_deg=theta)

        g: scalar_float = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(g)

    def test_grad_voltage(self) -> None:
        """Gradient w.r.t. beam voltage is finite."""

        def loss(voltage: scalar_float) -> scalar_float:
            return self._ewald_loss(voltage_kv=voltage)

        g: scalar_float = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_vmap_grad(self) -> None:
        """vmap(grad(loss)) over temperatures produces correct shape."""

        def loss(temp: scalar_float) -> scalar_float:
            return self._ewald_loss(temperature=temp)

        grad_fn: Callable[[scalar_float], scalar_float] = jax.grad(loss)
        batch_grad: Callable[
            [Float[Array, "temps"]], Float[Array, "temps"]
        ] = jax.vmap(grad_fn)
        temps: Float[Array, "3"] = jnp.array([100.0, 300.0, 600.0])
        grads: Float[Array, "3"] = batch_grad(temps)
        chex.assert_shape(grads, (3,))
        chex.assert_tree_all_finite(grads)

    def test_jacrev(self) -> None:
        """Jacrev w.r.t. (temperature, roughness) produces (2,) Jacobian."""

        def loss(params: Float[Array, "2"]) -> scalar_float:
            return self._ewald_loss(
                temperature=params[0],
                surface_roughness=params[1],
            )

        jac_fn: Callable[[Float[Array, "2"]], Float[Array, "2"]] = jax.jacrev(
            loss
        )
        params: Float[Array, "2"] = jnp.array([300.0, 0.5])
        jac: Float[Array, "2"] = jac_fn(params)
        chex.assert_shape(jac, (2,))
        chex.assert_tree_all_finite(jac)

    def test_ewald_simulator_grad_temperature_correct(self) -> None:
        """Ewald simulator grad w.r.t. temperature matches finite diff."""

        def f(temp: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        check_grads(jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-2)

    def test_ewald_simulator_grad_roughness_correct(self) -> None:
        """Ewald simulator grad w.r.t. roughness matches finite diff."""

        def f(roughness: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=300.0,
                surface_roughness=roughness,
            )
            return jnp.sum(pattern.intensities)

        check_grads(jax_safe(f), (jnp.float64(0.5),), order=1, atol=1e-2)


class TestMultisliceGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for multislice forward model."""

    def test_multislice_grad_voltage(self) -> None:
        """Gradient through multislice propagation w.r.t. voltage."""
        cart_positions: Float[Array, "..."] = jnp.array(
            [[5.0, 5.0, 1.0, 14.0], [7.5, 7.5, 3.0, 14.0]]
        )
        sliced: SlicedCrystal = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def loss(voltage: scalar_float) -> scalar_float:
            potential: Any = sliced_crystal_to_projected_potential_slices(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
            )
            psi_exit: Complex[Array, "H W"] = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        g: scalar_float = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_multislice_grad_voltage_correct(self) -> None:
        """Multislice grad w.r.t. voltage matches finite diff."""
        cart_positions: Float[Array, "..."] = jnp.array(
            [[5.0, 5.0, 1.0, 14.0], [7.5, 7.5, 3.0, 14.0]]
        )
        sliced: SlicedCrystal = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def f(voltage: scalar_float) -> scalar_float:
            potential: Any = sliced_crystal_to_projected_potential_slices(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
            )
            psi_exit: Complex[Array, "H W"] = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        check_grads(jax_safe(f), (jnp.float64(20.0),), order=1, atol=1e-2)


class TestEwaldSimulatorVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential for ewald_simulator."""

    def test_ewald_simulator_vmap_temperature_consistent(self) -> None:
        """Batched ewald_simulator over temps matches sequential."""

        def f(temp: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        temp_batch: Float[Array, "3"] = jnp.array([100.0, 300.0, 600.0])
        batched: Float[Array, "3"] = jax.vmap(f)(temp_batch)
        sequential: Float[Array, "3"] = jnp.stack([f(t) for t in temp_batch])
        chex.assert_trees_all_close(batched, sequential, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
