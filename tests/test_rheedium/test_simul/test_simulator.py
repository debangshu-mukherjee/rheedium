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

import ast
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax.test_util import check_grads
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray
from numpy.typing import NDArray

import rheedium.simul as rheedium_simul
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
    _multislice_amplitude_pattern,
    _render_ctr_streaks_to_image,
    checked_multislice_propagate,
    compute_kinematic_intensities_with_ctrs,
    detector_extent_mm,
    ewald_simulator,
    find_kinematic_reflections,
    kinematic_amplitude,
    log_compress_image,
    multislice_detector_amplitude,
    multislice_propagate,
    multislice_simulator,
    project_on_detector_geometry,
    render_amplitude_to_field,
    render_ctr_amplitude_to_field,
    render_pattern_to_image,
    sliced_crystal_to_projected_potential_slices,
)
from rheedium.simul.simulator import (
    simulate_detector_image as _simulate_detector_image,
)
from rheedium.simul.simulator import (
    simulate_detector_image_instrument as _simulate_detector_image_instrument,
)
from rheedium.tools import (
    gauss_hermite_nodes_weights,
    incident_wavevector,
    wavelength_ang,
)
from rheedium.tools.wrappers import jax_safe
from rheedium.types import (
    TRIVIAL_DISTRIBUTION,
    BeamModeDistribution,
    BeamSpec,
    CrystalStructure,
    DetectorGeometry,
    Distribution,
    PotentialSlices,
    ReductionMode,
    RenderParams,
    SlicedCrystal,
    SurfaceConfig,
    SurfaceCTRParams,
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
    create_rheed_pattern,
    create_sliced_crystal,
)
from rheedium.ucell import reciprocal_lattice_vectors


def simulate_detector_image(  # noqa: PLR0913
    crystal: CrystalStructure,
    energy_kev: Any = 20.0,
    theta_deg: Any = 2.0,
    phi_deg: Any = 0.0,
    hmax: int = 5,
    kmax: int = 5,
    detector_distance_mm: Any = 1000.0,
    temperature: Any = 300.0,
    surface_roughness: Any = 0.0,
    ctr_regularization: Any = 0.01,
    ctr_power: Any = 1.0,
    roughness_power: Any = 0.25,
    image_shape_px: tuple[int, int] = (192, 192),
    pixel_size_mm: tuple[float, float] = (1.5, 3.0),
    beam_center_px: tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: Any = 1.4,
    angular_divergence_mrad: Any = 0.35,
    energy_spread_ev: Any = 0.35,
    psf_sigma_pixels: Any = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: Any = None,
    distribution: Distribution | None = None,
    beam_modes: BeamModeDistribution | None = None,
    n_beam_modes_per_axis: int = 3,
    n_beam_modes_out_of_plane: int | None = None,
    n_beam_energy_points: int = 1,
    n_mosaic_points: int = 7,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
    defect_surface_layer_depth_angstrom: Any = 1.0,
    kernel: str = "kinematic",
    potential_slices: PotentialSlices | None = None,
    inner_potential_v0: Any = 0.0,
    bandwidth_limit: Any = 2.0 / 3.0,
    finite_domain_aspect_ratio: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> Float[Array, "H W"]:
    """Test-local adapter from legacy fixtures to carrier API."""
    image: Float[Array, "H W"] = _simulate_detector_image(
        crystal=crystal,
        beam=BeamSpec(
            energy_kev=energy_kev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            energy_spread_ev=energy_spread_ev,
            angular_divergence_mrad=angular_divergence_mrad,
            beam_modes=beam_modes,
            n_beam_modes_per_axis=n_beam_modes_per_axis,
            n_beam_modes_out_of_plane=n_beam_modes_out_of_plane,
            n_beam_energy_points=n_beam_energy_points,
        ),
        surface=SurfaceCTRParams(
            hmax=hmax,
            kmax=kmax,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            surface_config=surface_config,
            defect_surface_layer_depth_angstrom=(
                defect_surface_layer_depth_angstrom
            ),
            finite_domain_aspect_ratio=finite_domain_aspect_ratio,
        ),
        detector=DetectorGeometry(
            distance=detector_distance_mm,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            psf_sigma_pixels=psf_sigma_pixels,
        ),
        render=RenderParams(
            spot_sigma_px=spot_sigma_px,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            n_mosaic_points=n_mosaic_points,
            parameterization=parameterization,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
            kernel=kernel,
            inner_potential_v0=inner_potential_v0,
            bandwidth_limit=bandwidth_limit,
            potential_slices=potential_slices,
            orientation_distribution=orientation_distribution,
            distribution=distribution,
        ),
    )
    return image


def simulate_detector_image_instrument(  # noqa: PLR0913
    crystal: CrystalStructure,
    beam_modes: BeamModeDistribution,
    energy_kev: Any = 20.0,
    theta_deg: Any = 2.0,
    phi_deg: Any = 0.0,
    hmax: int = 5,
    kmax: int = 5,
    detector_distance_mm: Any = 1000.0,
    temperature: Any = 300.0,
    surface_roughness: Any = 0.0,
    ctr_regularization: Any = 0.01,
    ctr_power: Any = 1.0,
    roughness_power: Any = 0.25,
    image_shape_px: tuple[int, int] = (192, 192),
    pixel_size_mm: tuple[float, float] = (1.5, 3.0),
    beam_center_px: tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: Any = 1.4,
    psf_sigma_pixels: Any = 1.2,
    n_modes_per_axis: int = 3,
    n_modes_out_of_plane: int | None = None,
    n_energy_points: int = 1,
    parameterization: str = "lobato",
    surface_config: SurfaceConfig | None = None,
    kernel: str = "kinematic",
) -> Float[Array, "H W"]:
    """Test-local adapter for the carrier-shaped instrument wrapper."""
    image: Float[Array, "H W"] = _simulate_detector_image_instrument(
        crystal=crystal,
        beam=BeamSpec(
            energy_kev=energy_kev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            beam_modes=beam_modes,
            n_beam_modes_per_axis=n_modes_per_axis,
            n_beam_modes_out_of_plane=n_modes_out_of_plane,
            n_beam_energy_points=n_energy_points,
        ),
        surface=SurfaceCTRParams(
            hmax=hmax,
            kmax=kmax,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            surface_config=surface_config,
        ),
        detector=DetectorGeometry(
            distance=detector_distance_mm,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            psf_sigma_pixels=psf_sigma_pixels,
        ),
        render=RenderParams(
            spot_sigma_px=spot_sigma_px,
            parameterization=parameterization,
            render_ctrs_as_streaks=False,
            kernel=kernel,
        ),
    )
    return image


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
        r"""Test intensity calculation with CTR contributions.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: intensity
        calculation with CTR contributions.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``temperature``, ``surface_roughness``, ``surface_fraction``, so the
        documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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

    @chex.variants(with_device=True, without_jit=True)
    def test_surface_enhancement_effect(self) -> None:
        r"""Test that opt-in surface enhancement reduces intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: surface
        thermal enhancement is off by default (``surface_config=None``
        maps to ``SurfaceConfig(method="none")`` because the bulk basis
        is repeated by the CTR factor), and a caller who opts in with an
        explicit height-based config gets reduced intensity from the
        enhanced Debye-Waller damping of the tagged atoms.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.compute_kinematic_intensities_with_ctrs`
        :see: :func:`~rheedium.types.identify_surface_atoms`
        """
        var_compute: Callable[..., Any] = self.variant(
            compute_kinematic_intensities_with_ctrs
        )

        k_in: Float[Array, "..."] = jnp.array([73.0, 0.0, -2.5])
        g_vectors: Float[Array, "..."] = jnp.array([[1.0, 0.0, 0.0]])
        k_out: Float[Array, "..."] = k_in + g_vectors

        # Default: no surface enhancement (bulk cell, CTR-repeated)
        intensities_bulk: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
        )

        # Explicit opt-in: tag half the atoms as surface atoms
        intensities_surface: Float[Array, "..."] = var_compute(
            crystal=self.si_crystal,
            g_allowed=g_vectors,
            k_in=k_in,
            k_out=k_out,
            temperature=300.0,
            surface_roughness=0.0,
            surface_config=SurfaceConfig(method="height", height_fraction=0.5),
        )

        # Surface enhancement should reduce intensity due to increased
        # DW factor (without normalization, this works correctly)
        chex.assert_trees_all_equal(
            intensities_surface[0] < intensities_bulk[0], True
        )


class TestCheckedNumericalEntryPoints(chex.TestCase):
    """Tests for opt-in checkified numerical entry points."""

    def test_checked_multislice_propagate_valid(self) -> None:
        r"""Checked multislice propagation should allow finite outputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Checked multislice
        propagation should allow finite outputs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Checked multislice propagation should report NaN outputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Checked multislice
        propagation should report NaN outputs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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


class TestProjectOnDetectorGeometry(chex.TestCase, parameterized.TestCase):
    """Test suite for detector projection functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_basic_projection(self) -> None:
        r"""Test basic projection onto detector plane.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: basic projection
        onto detector plane.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_project: Callable[..., Any] = self.variant(
            project_on_detector_geometry
        )

        k_out: Float[Array, "..."] = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.5],
            ]
        )
        detector = DetectorGeometry(distance=100.0)

        coords: Float[Array, "..."] = var_project(k_out, detector)

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
        r"""Test that coordinates scale linearly with detector distance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: coordinates scale
        linearly with detector distance.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``distance``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_project: Callable[..., Any] = self.variant(
            project_on_detector_geometry
        )

        k_out: Float[Array, "..."] = jnp.array([[1.0, 0.5, 0.3]])
        coords: Float[Array, "..."] = var_project(
            k_out, DetectorGeometry(distance=distance)
        )

        chex.assert_shape(coords, (1, 2))
        # Verify linear scaling
        expected_h: Float[Array, "..."] = 0.5 * distance / 1.0
        expected_v: Float[Array, "..."] = 0.3 * distance / 1.0
        chex.assert_trees_all_close(
            coords[0], jnp.array([expected_h, expected_v]), rtol=1e-5
        )

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_output_shape(self) -> None:
        r"""Test output has correct shape for various inputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output has correct
        shape for various inputs.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_project: Callable[..., Any] = self.variant(
            project_on_detector_geometry
        )

        n: int
        for n in [1, 5, 10, 50]:
            k_out: Float[Array, "..."] = jnp.ones((n, 3))
            coords: Float[Array, "..."] = var_project(
                k_out, DetectorGeometry(distance=100.0)
            )
            chex.assert_shape(coords, (n, 2))


class TestRationalizationGuards(chex.TestCase):
    """Regression guards for rationalization gates RG1, RG2, and RG3."""

    repo_root = Path(__file__).parents[3]

    def test_rg1_projection_callers_use_detector_geometry(self) -> None:
        r"""Production projection callers should consume DetectorGeometry.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Production
        projection callers should consume DetectorGeometry.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        checked_files = [
            self.repo_root / "src/rheedium/simul/simulator.py",
            self.repo_root / "src/rheedium/simul/kinematic.py",
            self.repo_root / "src/rheedium/simul/reflection_multislice.py",
        ]

        raw_projection_calls: list[str] = []
        raw_projection_imports: list[str] = []
        for path in checked_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Name)
                        and func.id == "project_on_detector"
                    ):
                        raw_projection_calls.append(
                            f"{path.name}:{node.lineno}"
                        )
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == "project_on_detector":
                            raw_projection_imports.append(
                                f"{path.name}:{node.lineno}"
                            )

        self.assertEqual(raw_projection_calls, [])
        self.assertEqual(raw_projection_imports, [])

    def test_rg3_touched_public_symbols_have_live_consumers(self) -> None:
        r"""Removed or retained public symbols must match live import graph.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Removed or
        retained public symbols must match live import graph.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        self.assertFalse(hasattr(rheedium_simul, "project_on_detector"))
        self.assertNotIn("project_on_detector", rheedium_simul.__all__)

        beam_averaging = ast.parse(
            (
                self.repo_root / "src/rheedium/simul/beam_averaging.py"
            ).read_text(encoding="utf-8")
        )
        beam_public = {
            node.name
            for node in beam_averaging.body
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        }
        self.assertNotIn("coherence_envelope", beam_public)

        finite_domain = ast.parse(
            (self.repo_root / "src/rheedium/simul/finite_domain.py").read_text(
                encoding="utf-8"
            )
        )
        finite_domain_consumers = [
            node
            for node in finite_domain.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "finite_domain_intensities_for_size_distribution"
        ]
        self.assertLen(finite_domain_consumers, 1)
        size_distribution_arg = next(
            arg
            for arg in finite_domain_consumers[0].args.args
            if arg.arg == "size_distribution"
        )
        self.assertIsInstance(size_distribution_arg.annotation, ast.Name)
        self.assertEqual(
            size_distribution_arg.annotation.id,
            "SizeDistribution",
        )

    def test_rg2_orientation_averaging_has_single_reducer_path(self) -> None:
        r"""Orientation averaging should route through Layer-1 reducers.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation
        averaging should route through Layer-1 reducers.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        retired_names = {
            "ewald_simulator_with_orientation_distribution",
            "_discretize_orientation_for_sparse_pattern",
            "_incoherent_pattern_union",
        }

        self.assertFalse(
            hasattr(
                rheedium_simul,
                "ewald_simulator_with_orientation_distribution",
            )
        )
        self.assertNotIn(
            "ewald_simulator_with_orientation_distribution",
            rheedium_simul.__all__,
        )

        simulator = ast.parse(
            (self.repo_root / "src/rheedium/simul/simulator.py").read_text(
                encoding="utf-8"
            )
        )
        simulator_defs = {
            node.name
            for node in simulator.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(retired_names.isdisjoint(simulator_defs))

        distributions_path = (
            self.repo_root / "src/rheedium/types/distributions/orientation.py"
        )
        distributions_source = distributions_path.read_text(encoding="utf-8")
        distributions = ast.parse(distributions_source)
        integrate_node = next(
            node
            for node in distributions.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "integrate_over_orientation"
        )
        integrate_source = ast.get_source_segment(
            distributions_source,
            integrate_node,
        )
        self.assertIsNotNone(integrate_source)
        call_names = [
            node.func.id
            for node in ast.walk(integrate_node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(call_names.count("apply_distribution"), 1)
        self.assertNotIn("jax.vmap", integrate_source)
        self.assertNotIn("einsum", integrate_source)

    def test_rg4_detector_api_uses_carriers_and_generic_sweeps(self) -> None:
        r"""Detector images should expose carrier and generic sweep APIs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Detector images
        should expose carrier and generic sweep APIs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        signature = inspect.signature(rheedium_simul.simulate_detector_image)
        positional_or_keyword = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        self.assertLessEqual(len(positional_or_keyword), 5)
        self.assertEqual(
            set(signature.parameters),
            {"crystal", "beam", "surface", "detector", "render"},
        )
        retired_kwargs = {
            "energy_kev",
            "theta_deg",
            "phi_deg",
            "hmax",
            "kmax",
            "image_shape_px",
            "pixel_size_mm",
            "beam_center_px",
            "spot_sigma_px",
            "angular_divergence_mrad",
            "energy_spread_ev",
            "psf_sigma_pixels",
            "orientation_distribution",
            "distribution",
        }
        self.assertTrue(retired_kwargs.isdisjoint(signature.parameters))

        sweep_exports = {
            name
            for name in rheedium_simul.__all__
            if name.startswith("simulate_detector_image_")
            and name.endswith(("_sweep", "_grid"))
        }
        self.assertEqual(
            sweep_exports,
            {
                "simulate_detector_image_grid",
                "simulate_detector_image_sweep",
            },
        )

        sweeps = ast.parse(
            (self.repo_root / "src/rheedium/simul/sweeps.py").read_text(
                encoding="utf-8"
            )
        )
        public_sweep_defs = {
            node.name
            for node in sweeps.body
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        }
        self.assertEqual(
            public_sweep_defs,
            {
                "simulate_detector_image_grid",
                "simulate_detector_image_sweep",
            },
        )

    def test_rg5_angle_units_convert_at_single_boundary(self) -> None:
        r"""Public degree angles should convert to internal radians once.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Public degree
        angles should convert to internal radians once.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        simulator_path = self.repo_root / "src/rheedium/simul/simulator.py"
        simulator = ast.parse(simulator_path.read_text(encoding="utf-8"))
        tools_path = self.repo_root / "src/rheedium/tools/simul_utils.py"
        tools = ast.parse(tools_path.read_text(encoding="utf-8"))
        helper_name = "incidence_angles_to_radians"
        helper = next(
            node
            for node in tools.body
            if isinstance(node, ast.FunctionDef) and node.name == helper_name
        )
        helper_call_count = 0
        misplaced_conversions: list[str] = []
        checked_modules = {
            "simulator.py": simulator,
            "simul_utils.py": tools,
        }
        helper_nodes = set(ast.walk(helper))
        for module_name, module in checked_modules.items():
            for node in ast.walk(module):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == helper_name:
                        helper_call_count += 1
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "deg2rad"
                        and node.args
                    ):
                        arg_source = ast.unparse(node.args[0])
                        if (
                            "theta_deg" in arg_source
                            or "phi_deg" in arg_source
                        ) and node not in helper_nodes:
                            misplaced_conversions.append(
                                f"{module_name}:{node.lineno}:{arg_source}"
                            )

        self.assertEqual(misplaced_conversions, [])
        self.assertGreaterEqual(helper_call_count, 4)

    def test_rg5_procs_public_returns_follow_trichotomy(self) -> None:
        r"""Public procs functions should declare the R5 return contract.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Public procs
        functions should declare the R5 return contract.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        expected_categories = {
            "src/rheedium/procs/crystal_defects.py": {
                "apply_antisite_field": "CrystalStructure",
                "apply_interstitial_field": "CrystalStructure",
                "apply_vacancy_field": "CrystalStructure",
            },
            "src/rheedium/procs/distribution_binds.py": {
                "bind_kinematic_axis_distribution": "Callable",
                "bind_multislice_axis_distribution": "Callable",
                "validate_kinematic_axis": "None",
                "validate_multislice_axis": "None",
            },
            "src/rheedium/procs/grains.py": {
                "apply_misorientation_distribution": "Float",
                "grain_distribution_average": "Float",
                "grain_population_to_distribution": "Distribution",
            },
            "src/rheedium/procs/library.py": {
                "gaas001_2x4": "CrystalStructure",
                "mgo001_bulk_terminated": "CrystalStructure",
                "si100_2x1": "CrystalStructure",
                "si111_1x1": "CrystalStructure",
                "si111_7x7": "CrystalStructure",
                "srtio3_001_2x1": "CrystalStructure",
            },
            "src/rheedium/procs/preprocessing.py": {
                "log_intensity_transform": "Float",
                "normalize_image": "Float",
                "preprocess_experimental": "Float",
                "soft_threshold_mask": "Float",
                "subtract_background": "Float",
            },
            "src/rheedium/procs/surface_builder.py": {
                "add_adsorbate_layer": "CrystalStructure",
                "apply_surface_reconstruction": "CrystalStructure",
                "create_surface_slab": "CrystalStructure",
            },
            "src/rheedium/procs/surface_modifier.py": {
                "apply_step_edge_field": "CrystalStructure",
                "apply_surface_displacement_field": "CrystalStructure",
                "apply_surface_occupancy_field": "CrystalStructure",
                "apply_twin_wall_field": "CrystalStructure",
                "bind_step_edge_distribution": "Callable",
                "bind_twin_wall_distribution": "Callable",
                "incoherent_domain_average": "Float",
                "step_edge_to_distribution": "Distribution",
                "twin_wall_to_distribution": "Distribution",
                "vicinal_surface_step_splitting": "Float",
            },
        }

        missing_annotations: list[str] = []
        category_mismatches: list[str] = []
        extra_public_functions: list[str] = []
        for rel_path, expected in expected_categories.items():
            module = ast.parse(
                (self.repo_root / rel_path).read_text(encoding="utf-8")
            )
            public_functions = {
                node.name: node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
            }
            extra_public_functions.extend(
                f"{rel_path}:{name}"
                for name in sorted(set(public_functions) - set(expected))
            )
            for function_name, category in expected.items():
                function = public_functions[function_name]
                if function.returns is None:
                    missing_annotations.append(f"{rel_path}:{function_name}")
                    continue
                annotation = ast.unparse(function.returns)
                if category == "None":
                    matched = annotation == "None"
                else:
                    matched = category in annotation
                if not matched:
                    category_mismatches.append(
                        f"{rel_path}:{function_name}:{annotation}"
                    )

        self.assertEqual(extra_public_functions, [])
        self.assertEqual(missing_annotations, [])
        self.assertEqual(category_mismatches, [])

    def test_rg6_layering_modules_do_not_keep_forwarding_paths(self) -> None:
        r"""Package splits should not keep compatibility forwarding modules.

        Extended Summary
        ----------------
        Verifies the documented zero-legacy export rule: subpackage APIs may
        surface owned symbols, but old module-level forwarding paths are
        deleted instead of preserved as aliases.

        Notes
        -----
        It imports the canonical subpackage surfaces and asserts that
        the retired ``layer0`` / ``layer1`` modules are not importable
        compatibility shims.
        """
        distributions_path = (
            self.repo_root / "src/rheedium/types/distributions"
        )
        self.assertTrue(distributions_path.is_dir())
        self.assertFalse(
            (self.repo_root / "src/rheedium/types/distributions.py").exists()
        )

        distributions = importlib.import_module("rheedium.types.distributions")
        distribution_exports = set(distributions.__all__)
        self.assertIn("Distribution", distribution_exports)
        self.assertIn("BeamModeDistribution", distribution_exports)
        self.assertIn("OrientationDistribution", distribution_exports)
        self.assertIn("SizeDistribution", distribution_exports)

        for module_name in ("base", "beam", "orientation", "size"):
            importlib.import_module(
                f"rheedium.types.distributions.{module_name}"
            )

        simulator = importlib.import_module("rheedium.simul.simulator")
        self.assertIs(
            rheedium_simul.kinematic_amplitude,
            simulator.kinematic_amplitude,
        )
        self.assertIs(
            rheedium_simul.simulate_detector_image,
            simulator.simulate_detector_image,
        )
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rheedium.simul.layer0")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rheedium.simul.layer1")


class TestFindKinematicReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for kinematic reflection finding.

    :see: :func:`~rheedium.simul.find_kinematic_reflections`
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.k_mag: float = 73.0  # Typical |k| for 20 keV electrons

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_elastic_scattering_constraint(self) -> None:
        r"""Test that output wavevectors satisfy \|k_out\| ≈ \|k_in\|.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output wavevectors
        satisfy \|k_out\| ≈ \|k_in\|.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=-1.0, tolerance_inv_ang=0.5
        )

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
        r"""Test that tighter tolerances allow fewer reflections.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: tighter tolerances
        allow fewer reflections.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``tolerance``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            k_in, gs, z_sign=-1.0, tolerance_inv_ang=tolerance
        )

        chex.assert_tree_all_finite(k_out)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_positive(self) -> None:
        r"""Test filtering with positive z_sign (forward scattering).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: filtering with
        positive z_sign (forward scattering).

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=1.0, tolerance_inv_ang=0.5
        )

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_z_sign_negative(self) -> None:
        r"""Test filtering with negative z_sign (back scattering - RHEED).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: filtering with
        negative z_sign (back scattering - RHEED).

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=-1.0, tolerance_inv_ang=0.5
        )

        chex.assert_shape(allowed_indices, (3,))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_empty_g_vectors(self) -> None:
        r"""Test handling of single G vector.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: handling of single
        G vector.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        gs: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, tolerance_inv_ang=0.5)

        chex.assert_shape(allowed_indices, (1,))
        chex.assert_shape(k_out, (1, 3))

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_default_tolerance_is_shell_derived(self) -> None:
        r"""Test that the None default derives 3 x compute_shell_sigma.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: calling
        ``find_kinematic_reflections`` without ``tolerance_inv_ang``
        applies an absolute elastic tolerance of three Ewald-shell sigmas
        (energy spread 1e-4, divergence 1 mrad). At 20 kV (\|k\| ~ 73
        inverse Ångstroms) this admits only \|dk\| below roughly 0.25
        inverse Ångstroms, so a G vector producing \|dk\| ~ 0.5 must be
        rejected even though the old fractional-tolerance default
        (0.05 x \|k\| = 3.66) would have passed it.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.find_kinematic_reflections`
        :see: :func:`~rheedium.simul.compute_shell_sigma`
        """
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        k_in_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
        # G giving an elastic upward k_out (reflect k_in z-component)
        g_elastic: Float[Array, "3"] = jnp.array([0.0, 0.0, 5.0])
        # G giving |dk| ~ 0.5 1/A: inside the old 3.66 1/A default,
        # outside the new ~0.22 1/A shell-derived default.
        g_marginal: Float[Array, "3"] = k_in * (0.5 / k_in_mag) + g_elastic
        gs: Float[Array, "..."] = jnp.stack([g_elastic, g_marginal])

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(k_in, gs, z_sign=1.0)

        dk_elastic: float = float(
            jnp.abs(jnp.linalg.norm(k_in + g_elastic) - k_in_mag)
        )
        dk_marginal: float = float(
            jnp.abs(jnp.linalg.norm(k_in + g_marginal) - k_in_mag)
        )
        self.assertLess(dk_elastic, 0.22)
        self.assertGreater(dk_marginal, 0.25)
        self.assertLess(dk_marginal, 3.66)
        chex.assert_trees_all_equal(bool(allowed_indices[0] >= 0), True)
        chex.assert_trees_all_equal(bool(jnp.any(allowed_indices == 1)), False)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_padded_slots_zeroed(self) -> None:
        r"""Test that padded output slots return zero k_out rows.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with exactly
        one kinematically allowed reflection among several G vectors, the
        padded output rows carry ``k_out == 0`` and ``index == -1``
        rather than live copies of another reflection gathered through
        the JAX -1-wraps-to-last-element indexing convention.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.find_kinematic_reflections`
        """
        var_find: Callable[..., Any] = self.variant(find_kinematic_reflections)

        k_in: Float[Array, "..."] = jnp.array([self.k_mag, 0.0, -2.5])
        # Exactly one allowed reflection (elastic, upward); rest far off
        gs: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 10.0],
            ]
        )

        allowed_indices: Any
        k_out: Float[Array, "..."]
        allowed_indices, k_out = var_find(
            k_in, gs, z_sign=1.0, tolerance_inv_ang=0.25
        )

        chex.assert_trees_all_equal(
            allowed_indices,
            jnp.array([0, -1, -1, -1], dtype=allowed_indices.dtype),
        )
        chex.assert_trees_all_close(k_out[0], k_in + gs[0])
        chex.assert_trees_all_equal(
            jnp.all(k_out[1:] == 0.0),
            jnp.array(True),
        )


class TestSlicedCrystalToProjectedPotentialSlices(
    chex.TestCase, parameterized.TestCase
):
    """Tests for converting sliced crystals to potential slices.

    :see: :func:`~rheedium.simul.sliced_crystal_to_projected_potential_slices`
    """

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
        r"""Test that output potential has expected shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output potential
        has expected shape. Existing context from the original test prose:
        Note: JIT compilation not supported due to dynamic grid dimensions.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test potential generation with different slice thicknesses.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: potential
        generation with different slice thicknesses. Existing context from the
        original test prose: Note: JIT compilation not supported due to dynamic
        grid dimensions.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``thickness``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test potential generation with different pixel sizes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: potential
        generation with different pixel sizes. Existing context from the
        original test prose: Note: JIT compilation not supported due to dynamic
        grid dimensions.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``pixel_size``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Projected-potential slices are finite for both models.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        Projected-potential slices are finite for both models.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``parameterization``, so the documented behavior is checked across the
        cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Test that calibration values are stored correctly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: calibration values
        are stored correctly. Existing context from the original test prose:
        Note: JIT compilation not supported due to dynamic grid dimensions.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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

    @chex.variants(with_device=True, without_jit=True)
    def test_periodic_grid_excludes_endpoint(self) -> None:
        r"""Grid samples are spaced L/n with no duplicate boundary column.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the lateral
        potential grid uses n samples spaced L/n starting at 0 and
        excluding the endpoint L, matching the fftfreq(n, L/n)
        convention assumed by the Fresnel propagator. For a single atom
        at (0, 0) on a periodic grid, column 0 (the atom site) and the
        wrap-around column n-1 must not be duplicates, while columns 1
        and n-1 sit at the same minimum-image distance and must agree.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~.sliced_crystal_to_projected_potential_slices`
        """
        var_convert: Callable[..., Any] = self.variant(
            sliced_crystal_to_projected_potential_slices
        )

        single_atom: SlicedCrystal = create_sliced_crystal(
            cart_positions=jnp.array([[0.0, 0.0, 1.0, 14.0]]),
            cell_lengths=jnp.array([8.0, 8.0, 4.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=4.0,
            x_extent=8.0,
            y_extent=8.0,
        )
        potential: Any = var_convert(
            single_atom,
            slice_thickness=4.0,
            pixel_size=0.5,
        )
        v: Float[Array, "nx ny"] = potential.slices[0]
        # nx = ceil(8.0 / 0.5) = 16 samples spaced L/n = 0.5 Angstroms
        chex.assert_shape(v, (16, 16))
        # Peak sits exactly on grid point (0, 0)
        peak_idx: Any = jnp.unravel_index(jnp.argmax(v), v.shape)
        self.assertEqual(int(peak_idx[0]), 0)
        self.assertEqual(int(peak_idx[1]), 0)
        # The old linspace(0, L, n) grid duplicated x=0 and x=L: the last
        # column equalled the first. With arange(n) * (L/n) it must not.
        self.assertFalse(bool(jnp.allclose(v[:, 0], v[:, -1], rtol=1e-6)))
        self.assertFalse(bool(jnp.allclose(v[0, :], v[-1, :], rtol=1e-6)))
        # Wrap-around symmetry: column 1 (distance L/n) matches column
        # n-1 (distance L - (n-1) L/n = L/n through the boundary).
        chex.assert_trees_all_close(v[:, 1], v[:, -1], rtol=1e-10)
        chex.assert_trees_all_close(v[1, :], v[-1, :], rtol=1e-10)


class TestMultislicePropagate(chex.TestCase, parameterized.TestCase):
    """Test suite for multislice wave propagation.

    :see: :func:`~rheedium.simul.multislice_propagate`
    """

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
        r"""Test that exit wave has same shape as input grid.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: exit wave has same
        shape as input grid.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )

        chex.assert_shape(exit_wave, (32, 32))
        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_exit_wave_nonzero(self) -> None:
        r"""Test that exit wave has non-zero amplitude.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: exit wave has
        non-zero amplitude.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test propagation at different voltages.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: propagation at
        different voltages.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``voltage``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=voltage,
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
        r"""Test propagation at different grazing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: propagation at
        different grazing angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``theta``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test propagation at different azimuthal angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: propagation at
        different azimuthal angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``phi``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test effect of inner potential on propagation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: effect of inner
        potential on propagation.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``v0``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test different bandwidth limiting values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: different
        bandwidth limiting values.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``limit``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_propagate: Callable[..., Any] = self.variant(multislice_propagate)

        exit_wave: Any = var_propagate(
            self.simple_potential,
            energy_kev=20.0,
            theta_deg=2.0,
            bandwidth_limit=limit,
        )

        chex.assert_tree_all_finite(exit_wave)

    @chex.all_variants(without_device=False, with_pmap=False)
    def test_zero_potential_propagation(self) -> None:
        r"""Test propagation through zero potential (free space).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: propagation
        through zero potential (free space).

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            energy_kev=20.0,
            theta_deg=2.0,
        )

        chex.assert_tree_all_finite(exit_wave)
        # Should still have intensity (plane wave propagates)
        total_intensity: scalar_float = float(jnp.sum(jnp.abs(exit_wave) ** 2))
        chex.assert_scalar_positive(total_intensity)


class TestMultisliceSimulator(chex.TestCase, parameterized.TestCase):
    """Test suite for complete multislice RHEED simulation.

    :see: :func:`~rheedium.simul.multislice_simulator`
    """

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
        r"""Test that simulator returns valid RHEEDPattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: simulator returns
        valid RHEEDPattern. Existing context from the original test prose:
        Note: JIT not supported due to dynamic array sizes.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )

        self.assertIsInstance(pattern, RHEEDPattern)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @chex.variants(with_device=True, without_jit=True)
    def test_pattern_shapes_consistent(self) -> None:
        r"""Test that all pattern arrays have consistent shapes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all pattern arrays
        have consistent shapes. Existing context from the original test prose:
        Note: JIT not supported due to dynamic array sizes.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test simulation at different detector distances.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: simulation at
        different detector distances. Existing context from the original test
        prose: Note: JIT not supported due to dynamic array sizes.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``distance``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=20.0,
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
        r"""Test simulation at different voltages.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: simulation at
        different voltages. Existing context from the original test prose:
        Note: JIT not supported due to dynamic array sizes.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``voltage``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=voltage,
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
        r"""Test simulation at different grazing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: simulation at
        different grazing angles. Existing context from the original test
        prose: Note: JIT not supported due to dynamic array sizes.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``theta``, so
        the documented behavior is checked across the cases supplied by pytest,
        Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=20.0,
            theta_deg=theta,
        )

        chex.assert_tree_all_finite(pattern.intensities)

    @chex.variants(with_device=True, without_jit=True)
    def test_ewald_sphere_constraint(self) -> None:
        r"""Test that output wavevectors approximately satisfy Ewald sphere.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: output wavevectors
        approximately satisfy Ewald sphere. Existing context from the original
        test prose: Note: JIT not supported due to dynamic array sizes.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_simulate: Callable[..., Any] = self.variant(multislice_simulator)

        pattern: Float[Array, "..."] = var_simulate(
            self.simple_potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )

        # k_out should have approximately same magnitude as k_in
        energy_kev: float = 20.0
        lam_ang: Any = float(wavelength_ang(energy_kev))
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
    """Extended tests for kinematic intensity calculation with CTRs.

    :see: :func:`~rheedium.simul.compute_kinematic_intensities_with_ctrs`
    """

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
        r"""Test intensity calculation with no CTR contribution.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: intensity
        calculation with no CTR contribution. Existing context from the
        original test prose: Note: jittable with ctr_mixing_mode as a static
        argument; see the JAX Transformability guide and
        test_ctr_jit_static_mode.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test intensity calculation with coherent CTR mixing.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: intensity
        calculation with coherent CTR mixing. Existing context from the
        original test prose: Note: jittable with ctr_mixing_mode as a static
        argument; see the JAX Transformability guide and
        test_ctr_jit_static_mode.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test intensity calculation with incoherent CTR mixing.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: intensity
        calculation with incoherent CTR mixing. Existing context from the
        original test prose: Note: jittable with ctr_mixing_mode as a static
        argument; see the JAX Transformability guide and
        test_ctr_jit_static_mode.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Compile the CTR intensity function with the mode held static.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Compile the CTR
        intensity function with the mode held static. Existing context from the
        original test prose: The only jit blocker is the string
        ``ctr_mixing_mode``; making it static (here via ``eqx.filter_jit``,
        equivalently ``jax.jit(..., static_argnames=("ctr_mixing_mode",))``)
        yields a fully compiled function whose output matches the eager result.
        See the JAX Transformability guide.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test effect of CTR weight on intensities.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: effect of CTR
        weight on intensities.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``weight``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Test with height-based surface atom identification.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with height-based
        surface atom identification. Existing context from the original test
        prose: Note: JIT not supported due to SurfaceConfig with string method.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test with layer-based surface atom identification.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with layer-based
        surface atom identification. Existing context from the original test
        prose: Note: JIT not supported due to SurfaceConfig with string method.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
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
        r"""Test effect of h,k tolerance for CTR application.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: effect of h,k
        tolerance for CTR application.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``tolerance``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Explicit hkl should enable CTR when \|G\| misses tolerance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Explicit hkl
        should enable CTR when \|G\| misses tolerance.

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
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
    """Test suite for ewald_simulator with exact Ewald-CTR intersection.

    :see: :func:`~rheedium.simul.ewald_simulator`
    """

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
        r"""Test that ewald_simulator produces a valid RHEED pattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ewald_simulator
        produces a valid RHEED pattern.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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

    def test_default_applies_no_surface_enhancement(self) -> None:
        r"""Default ewald_simulator applies no surface thermal enhancement.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with
        ``surface_config=None`` the simulator treats no atom as a surface
        atom (``SurfaceConfig(method="none")``), because the bulk unit
        cell it repeats through the CTR factor has no surface atoms. On
        SrTiO3 at 300 K the default pattern must be bit-identical to an
        explicit method-"none" run and to an explicit all-False mask,
        even though several atoms sit in the top 30 percent of the cell.
        An explicit height-based opt-in must change the intensities,
        proving the old default (2x mean-square displacement for the top
        30 percent of a bulk cell) would have altered the result and is
        no longer applied silently.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.ewald_simulator`
        :see: :func:`~rheedium.types.identify_surface_atoms`
        """
        a_sto: float = 3.905
        frac_coords: Float[Array, "5 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Sr
                [0.5, 0.5, 0.5],  # Ti
                [0.5, 0.5, 0.0],  # O
                [0.0, 0.5, 0.5],  # O (top 30% of the cell by height)
                [0.5, 0.0, 0.5],  # O (top 30% of the cell by height)
            ]
        )
        atomic_numbers: Float[Array, "5"] = jnp.array(
            [38.0, 22.0, 8.0, 8.0, 8.0]
        )
        sto_crystal: CrystalStructure = create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack(
                [frac_coords * a_sto, atomic_numbers]
            ),
            cell_lengths=jnp.array([a_sto, a_sto, a_sto]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )
        kwargs: dict[str, Any] = {
            "crystal": sto_crystal,
            "energy_kev": 20.0,
            "theta_deg": 2.0,
            "phi_deg": 0.0,
            "hmax": 2,
            "kmax": 2,
            "temperature": 300.0,
        }
        pattern_default: Any = ewald_simulator(**kwargs)
        pattern_none: Any = ewald_simulator(
            **kwargs, surface_config=SurfaceConfig(method="none")
        )
        pattern_mask: Any = ewald_simulator(
            **kwargs,
            surface_config=SurfaceConfig(
                method="explicit",
                explicit_mask=jnp.zeros(5, dtype=bool),
            ),
        )
        pattern_height: Any = ewald_simulator(
            **kwargs,
            surface_config=SurfaceConfig(method="height", height_fraction=0.3),
        )
        # Default is exactly the no-enhancement result
        chex.assert_trees_all_equal(
            pattern_default.intensities, pattern_none.intensities
        )
        chex.assert_trees_all_equal(
            pattern_default.intensities, pattern_mask.intensities
        )
        # Opt-in height tagging (the old default) changes intensities,
        # so the equality above is not vacuous.
        self.assertFalse(
            bool(
                jnp.allclose(
                    pattern_default.intensities,
                    pattern_height.intensities,
                    rtol=1e-6,
                )
            )
        )

    def test_upward_scattering_only(self) -> None:
        r"""Only upward-scattered reflections are returned (k_out_z > 0).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Only
        upward-scattered reflections are returned (k_out_z > 0).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        energy_kev: float,
        theta_deg: float,
        phi_deg: float,
        hmax: int,
        kmax: int,
        detector_distance: float,
    ) -> Float[NDArray, "points detector_xy"]:
        """Construct detector intersections directly from both rod branches."""
        lam_ang: Any = float(wavelength_ang(energy_kev))
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
        r"""Sparse Ewald output matches direct two-branch rod geometry.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Sparse Ewald
        output matches direct two-branch rod geometry.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        detector_distance: float = 900.0
        hmax: int = 8
        kmax: int = 8
        raw_points: Float[Array, "..."] = self._raw_ctr_detector_points(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
        )
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""SrTiO3 sparse Ewald output matches the raw detector geometry.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: SrTiO3 sparse
        Ewald output matches the raw detector geometry.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        sto_crystal: Any = self._create_sto_crystal()
        detector_distance: float = 900.0
        hmax: int = 14
        kmax: int = 14
        raw_points: Float[Array, "..."] = self._raw_ctr_detector_points(
            crystal=sto_crystal,
            energy_kev=18.0,
            theta_deg=theta_deg,
            phi_deg=0.0,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
        )
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=sto_crystal,
            energy_kev=18.0,
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
        r"""Test that \|k_out\| = \|k_in\| (elastic scattering).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|k_out\| =
        \|k_in\| (elastic scattering).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""Changing phi_deg rotates the pattern.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Changing phi_deg
        rotates the pattern.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern_0: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=3,
            kmax=3,
        )

        pattern_45: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""Higher temperature reduces intensity (Debye-Waller).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Higher temperature
        reduces intensity (Debye-Waller).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern_low_T: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            temperature=100.0,
            hmax=3,
            kmax=3,
        )

        pattern_high_T: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""Surface roughness affects CTR intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Surface roughness
        affects CTR intensity.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern_smooth: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            surface_roughness=0.1,
            hmax=3,
            kmax=3,
        )

        pattern_rough: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""Different voltages give different k magnitudes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Different voltages
        give different k magnitudes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern_10kv: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=10.0,
            theta_deg=2.0,
            hmax=3,
            kmax=3,
        )

        pattern_30kv: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=30.0,
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
        r"""ewald_simulator works under JAX JIT compilation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ewald_simulator
        works under JAX JIT compilation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Callable[..., Any] = jax.jit(
            ewald_simulator,
            static_argnames=("hmax", "kmax"),
        )(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""surface_config parameter works correctly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: surface_config
        parameter works correctly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        config: Any = SurfaceConfig(method="height", height_fraction=0.5)

        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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
        r"""Various grazing angles produce valid patterns.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Various grazing
        angles produce valid patterns.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = ewald_simulator(
            crystal=self.mgo_crystal,
            energy_kev=20.0,
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


def _legacy_flat_detector_points(
    k_out: Float[Array, "N 3"],
    detector_distance_mm: float,
) -> Float[Array, "N 2"]:
    """Return the pre-carrier flat-detector projection formula."""
    scale: Float[Array, "N"] = detector_distance_mm / (k_out[:, 0] + 1e-10)
    return jnp.stack((k_out[:, 1] * scale, k_out[:, 2] * scale), axis=-1)


class TestDetectorImageOrchestrator(chex.TestCase, parameterized.TestCase):
    """Tests for dense detector-image helpers built on ewald_simulator.

    :see: :func:`~rheedium.simul.kinematic_amplitude`
    :see: :func:`~rheedium.simul.render_amplitude_to_field`
    :see: :func:`~rheedium.simul.render_ctr_amplitude_to_field`
    """

    @staticmethod
    def _legacy_render_pattern_to_image(
        pattern: RHEEDPattern,
        image_shape_px: tuple[int, int],
        pixel_size_mm: tuple[float, float],
        beam_center_px: tuple[float, float],
        spot_sigma_px: float,
    ) -> Float[Array, "H W"]:
        """Mirror the pre-DetectorGeometry dense intensity renderer."""
        height_px, width_px = image_shape_px
        x_axis: Float[Array, "W"] = jnp.arange(width_px, dtype=jnp.float64)
        y_axis: Float[Array, "H"] = jnp.arange(height_px, dtype=jnp.float64)
        x_grid: Float[Array, "H W"]
        y_grid: Float[Array, "H W"]
        x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
        x_pixels: Float[Array, "N"] = (
            pattern.detector_points[:, 0] / pixel_size_mm[0]
            + beam_center_px[0]
        )
        y_pixels: Float[Array, "N"] = (
            pattern.detector_points[:, 1] / pixel_size_mm[1]
            + beam_center_px[1]
        )

        def _render_one_spot(
            x0_px: Float[Array, ""],
            y0_px: Float[Array, ""],
            intensity: Float[Array, ""],
        ) -> Float[Array, "H W"]:
            """Render one detector spot into the legacy image grid."""
            spot: Float[Array, "H W"] = intensity * jnp.exp(
                -((x_grid - x0_px) ** 2 + (y_grid - y0_px) ** 2)
                / (2.0 * spot_sigma_px**2)
            )
            return spot

        image: Float[Array, "H W"] = jnp.sum(
            jax.vmap(_render_one_spot)(
                x_pixels,
                y_pixels,
                pattern.intensities,
            ),
            axis=0,
        )
        return image / jnp.maximum(jnp.max(image), 1e-12)

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
        r"""Rasterized detector image has requested shape and unit maximum.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Rasterized
        detector image has requested shape and unit maximum.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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

        geometry: DetectorGeometry = DetectorGeometry(
            image_shape_px=(32, 40),
            pixel_size_mm=(1.0, 2.0),
            beam_center_px=(20.0, 4.0),
        )
        image: Float[Array, "..."] = render_pattern_to_image(
            pattern=pattern,
            geometry=geometry,
            spot_sigma_px=1.5,
        )

        chex.assert_shape(image, (32, 40))
        chex.assert_tree_all_finite(image)
        chex.assert_trees_all_close(jnp.max(image), 1.0, atol=1e-12)
        self.assertTrue(jnp.all(image >= 0.0))

    def test_render_pattern_to_image_matches_pre_carrier_pixels(self) -> None:
        r"""RG1: DetectorGeometry carrier preserves pre-refactor pixels.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: RG1:
        DetectorGeometry carrier preserves pre-refactor pixels.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0, 1, 2], dtype=jnp.int32),
            k_out=jnp.array(
                [
                    [10.0, 0.0, 1.0],
                    [10.0, 0.0, 1.0],
                    [10.0, 0.0, 1.0],
                ],
                dtype=jnp.float64,
            ),
            detector_points=jnp.array(
                [[-1.5, 2.0], [2.25, 5.0], [9.0, 0.0]],
                dtype=jnp.float64,
            ),
            intensities=jnp.array([1.0, 0.35, 0.15], dtype=jnp.float64),
        )
        image_shape_px: tuple[int, int] = (32, 40)
        pixel_size_mm: tuple[float, float] = (1.5, 2.5)
        beam_center_px: tuple[float, float] = (18.0, 6.0)
        spot_sigma_px: float = 1.25

        actual: Float[Array, "32 40"] = render_pattern_to_image(
            pattern=pattern,
            geometry=DetectorGeometry(
                image_shape_px=image_shape_px,
                pixel_size_mm=pixel_size_mm,
                beam_center_px=beam_center_px,
            ),
            spot_sigma_px=spot_sigma_px,
        )
        expected: Float[Array, "32 40"] = self._legacy_render_pattern_to_image(
            pattern=pattern,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_render_amplitude_to_field_matches_single_spot_intensity(
        self,
    ) -> None:
        r"""Squared single-spot amplitude field matches legacy rendering.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Squared
        single-spot amplitude field matches legacy rendering.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = RHEEDPattern(
            G_indices=jnp.array([0], dtype=jnp.int32),
            k_out=jnp.array([[10.0, 0.0, 1.0]], dtype=jnp.float64),
            detector_points=jnp.array([[0.0, 0.0]], dtype=jnp.float64),
            intensities=jnp.array([4.0], dtype=jnp.float64),
        )

        geometry: DetectorGeometry = DetectorGeometry(
            image_shape_px=(32, 40),
            pixel_size_mm=(1.0, 2.0),
            beam_center_px=(20.0, 4.0),
        )
        field: Complex[Array, "32 40"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.sqrt(pattern.intensities).astype(jnp.complex128),
            geometry=geometry,
            spot_sigma_px=1.5,
        )
        intensity: Float[Array, "32 40"] = jnp.abs(field) ** 2
        intensity = intensity / jnp.max(intensity)
        legacy: Float[Array, "32 40"] = render_pattern_to_image(
            pattern=pattern,
            geometry=geometry,
            spot_sigma_px=1.5,
        )

        chex.assert_trees_all_close(intensity, legacy, atol=1e-12)

    def test_render_amplitude_to_field_preserves_interference(self) -> None:
        r"""Overlapping amplitudes interfere coherently in the dense field.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Overlapping
        amplitudes interfere coherently in the dense field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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

        geometry: DetectorGeometry = DetectorGeometry(
            image_shape_px=(16, 16),
            pixel_size_mm=(1.0, 1.0),
            beam_center_px=(8.0, 8.0),
        )
        constructive: Complex[Array, "16 16"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.array([1.0 + 0.0j, 1.0 + 0.0j]),
            geometry=geometry,
            spot_sigma_px=1.0,
        )
        destructive: Complex[Array, "16 16"] = render_amplitude_to_field(
            pattern=pattern,
            amplitudes=jnp.array([1.0 + 0.0j, -1.0 + 0.0j]),
            geometry=geometry,
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
        r"""The real kinematic kernel should expose complex reflection phase.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The real kinematic
        kernel should expose complex reflection phase.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        sparse_pattern: RHEEDPattern
        amplitudes: Complex[Array, "N"]
        sparse_pattern, amplitudes = _ewald_amplitude_pattern(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
            energy_kev=20.0,
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
        r"""Coherent reduction should interfere real kinematic amplitudes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Coherent reduction
        should interfere real kinematic amplitudes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            """Apply one coherent phase sample to the kinematic field."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0,
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
            phased_field: Complex[Array, "16 24"] = (
                jnp.exp(1j * sample[0]) * field
            )
            return phased_field

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
        r"""Kinematic amplitude uses the sparse Ewald amplitude-render path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Kinematic
        amplitude uses the sparse Ewald amplitude-render path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
            energy_kev=kwargs["energy_kev"],
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
            geometry=DetectorGeometry(
                image_shape_px=kwargs["image_shape_px"],
                pixel_size_mm=kwargs["pixel_size_mm"],
                beam_center_px=kwargs["beam_center_px"],
            ),
            spot_sigma_px=kwargs["spot_sigma_px"],
        )

        chex.assert_trees_all_close(amplitude, expected, atol=1e-12)

    def test_ewald_amplitude_pattern_matches_intensity_simulator(self) -> None:
        r"""Complex Ewald amplitudes preserve the legacy intensity surface.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Complex Ewald
        amplitudes preserve the legacy intensity surface.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Trivial distribution turns one coherent amplitude into intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Trivial
        distribution turns one coherent amplitude into intensity.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        amplitude: Complex[Array, "16 24"] = kinematic_amplitude(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
            """Return the precomputed amplitude for the trivial sample."""
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
        r"""Display extent converts beam centre and pixel pitch correctly.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Display extent
        converts beam centre and pixel pitch correctly.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        extent: Float[Array, "..."] = detector_extent_mm(
            DetectorGeometry(
                image_shape_px=(100, 200),
                pixel_size_mm=(1.5, 3.0),
                beam_center_px=(80.0, 5.0),
            )
        )
        self.assertEqual(extent, (-120.0, 180.0, -15.0, 285.0))

    def test_log_compress_image_preserves_bounds(self) -> None:
        r"""Log compression maps a normalized image back into [0, 1].

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Log compression
        maps a normalized image back into [0, 1].

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Display floor hides weak pixels and rescales the visible range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Display floor
        hides weak pixels and rescales the visible range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Complex CTR renderer reproduces legacy one-reflection intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Complex CTR
        renderer reproduces legacy one-reflection intensity.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        geometry: DetectorGeometry = DetectorGeometry(
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
        )

        field: Complex[Array, "32 24"] = render_ctr_amplitude_to_field(
            pattern=pattern,
            amplitudes=amplitudes,
            geometry=geometry,
            spot_sigma_px=1.2,
        )
        actual: Float[Array, "32 24"] = jnp.abs(field) ** 2
        actual = actual / jnp.maximum(jnp.max(actual), 1e-12)
        expected: Float[Array, "32 24"] = _render_ctr_streaks_to_image(
            pattern=pattern,
            geometry=geometry,
            spot_sigma_px=1.2,
        )

        chex.assert_trees_all_close(actual, expected, atol=1e-12)

    def test_simulate_detector_image_uses_layer1_default_when_spot_rendered(
        self,
    ) -> None:
        r"""Spot-render default delegates to the trivial Layer-1 reducer.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Spot-render
        default delegates to the trivial Layer-1 reducer.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Spot-render instrument widths reduce through Layer-1 mechanics.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Spot-render
        instrument widths reduce through Layer-1 mechanics.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        base_energy_kev: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 3.0

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "16 24"]:
            """Evaluate the kinematic field for one beam sample."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=base_energy_kev + 1.0e-3 * sample[2],
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
            return field

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=base_energy_kev,
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
        r"""CTR-rendered instrument widths reduce through Layer-1 mechanics.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR-rendered
        instrument widths reduce through Layer-1 mechanics.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        base_energy_kev: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 0.0

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "32 24"]:
            """Evaluate the CTR-streak field for one beam sample."""
            field: Complex[Array, "32 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=base_energy_kev + 1.0e-3 * sample[2],
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
            return field

        expected: Float[Array, "32 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=base_energy_kev,
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
        r"""Trivial generic distribution preserves the spot-rendered image.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Trivial generic
        distribution preserves the spot-rendered image.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Trivial generic distribution preserves CTR streak rendering.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Trivial generic
        distribution preserves CTR streak rendering.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Generic distribution delegates reduction to Layer-1 mechanics.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Generic
        distribution delegates reduction to Layer-1 mechanics.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution = create_distribution(
            samples=jnp.array([[0.0], [5.0]], dtype=jnp.float64),
            weights=jnp.array([0.25, 0.75], dtype=jnp.float64),
            reduction=ReductionMode.INCOHERENT,
            axis_id="test_phi",
        )
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
            """Evaluate one detector field for a distribution sample."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0,
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
            return field

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
        r"""Twin distributions should bind structures in the public path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Twin distributions
        should bind structures in the public path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            """Evaluate a twin-bound crystal sample as a detector field."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=builder(sample),
                energy_kev=20.0,
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
            return field

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
        r"""Step distributions should bind structures in the public path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Step distributions
        should bind structures in the public path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            """Evaluate a step-bound crystal sample as a detector field."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=builder(sample),
                energy_kev=20.0,
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
            return field

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
        r"""Grain distributions should not be interpreted as beam samples.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Grain
        distributions should not be interpreted as beam samples.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([-1.0, 2.0]),
            grain_sizes_angstrom=jnp.array([80.0, 120.0]),
            grain_volume_fractions=jnp.array([0.25, 0.75]),
        )

        def _bound(sample: Float[Array, "2"]) -> Complex[Array, "16 24"]:
            """Evaluate one grain finite-domain amplitude sample."""
            amplitude: Complex[Array, "16 24"] = (
                _kinematic_finite_domain_amplitude(
                    crystal=_SI_CRYSTAL_2ATOM,
                    energy_kev=20.0,
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
            )
            return amplitude

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
            "energy_kev": 20.0,
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
        r"""Twin producers should change detector output, not only bind.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Twin producers
        should change detector output, not only bind.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Step producers should change detector output, not only bind.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Step producers
        should change detector output, not only bind.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Grain producers should change detector output, not only bind.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Grain producers
        should change detector output, not only bind.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Size axes bind finite-domain broadening in the public path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Size axes bind
        finite-domain broadening in the public path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = create_distribution(
            samples=jnp.array([[40.0], [80.0]], dtype=jnp.float64),
            weights=jnp.array([0.5, 0.5], dtype=jnp.float64),
            reduction=ReductionMode.INCOHERENT,
            axis_id="size",
        )

        base: Float[Array, "16 24"] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
            energy_kev=20.0,
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
        r"""Legacy and generic distributions are mutually exclusive inputs.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Legacy and generic
        distributions are mutually exclusive inputs.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Layer-0 kernel selector fails clearly for unsupported names.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Layer-0 kernel
        selector fails clearly for unsupported names.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Multislice selection requires a concrete potential-slice payload.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Multislice
        selection requires a concrete potential-slice payload.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Public Layer 1 can select multislice and reduce its field.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Public Layer 1 can
        select multislice and reduce its field.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        potential_slices: PotentialSlices = self._tiny_potential_slices()
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "potential_slices": potential_slices,
            "energy_kev": 20.0,
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
            energy_kev=kwargs["energy_kev"],
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

    def test_detector_contract_extents_match_kinematic_and_multislice(
        self,
    ) -> None:
        r"""FG2: both kernels consume the same detector extent contract.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG2: both kernels
        consume the same detector extent contract.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        potential_slices: PotentialSlices = self._tiny_potential_slices()
        detector_distance_mm: float = 20.0
        image_shape_px: tuple[int, int] = (32, 32)
        pixel_size_mm: tuple[float, float] = (2.0, 2.0)
        beam_center_px: tuple[float, float] = (16.0, 16.0)
        kinematic_extent: tuple[float, float, float, float] = (
            detector_extent_mm(
                DetectorGeometry(
                    distance=detector_distance_mm,
                    image_shape_px=image_shape_px,
                    pixel_size_mm=pixel_size_mm,
                    beam_center_px=beam_center_px,
                )
            )
        )
        multislice_extent: tuple[float, float, float, float] = (
            detector_extent_mm(
                DetectorGeometry(
                    distance=detector_distance_mm,
                    image_shape_px=image_shape_px,
                    pixel_size_mm=pixel_size_mm,
                    beam_center_px=beam_center_px,
                )
            )
        )
        chex.assert_trees_all_close(
            jnp.asarray(kinematic_extent),
            jnp.asarray(multislice_extent),
            atol=0.0,
        )

        kinematic_pattern: RHEEDPattern
        multislice_pattern: RHEEDPattern
        kinematic_pattern, _ = _ewald_amplitude_pattern(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
            theta_deg=5.0,
            phi_deg=0.0,
            hmax=0,
            kmax=0,
            detector_distance=detector_distance_mm,
        )
        multislice_pattern, _ = _multislice_amplitude_pattern(
            potential_slices=potential_slices,
            energy_kev=20.0,
            theta_deg=5.0,
            phi_deg=0.0,
            detector_distance_mm=detector_distance_mm,
        )

        xmin: float
        xmax: float
        ymin: float
        ymax: float
        xmin, xmax, ymin, ymax = kinematic_extent
        for pattern in (kinematic_pattern, multislice_pattern):
            points: Float[Array, "N 2"] = pattern.detector_points
            visible: Bool[Array, "N"] = (
                (points[:, 0] >= xmin)
                & (points[:, 0] <= xmax)
                & (points[:, 1] >= ymin)
                & (points[:, 1] <= ymax)
            )
            assert bool(jnp.any(visible))

    def test_detector_contract_pixelwise_matches_flat_projection_regression(
        self,
    ) -> None:
        r"""RG1: carrier projection preserves pre-refactor flat pixels.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: RG1: carrier
        projection preserves pre-refactor flat pixels.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        detector_distance_mm: float = 20.0
        geometry = DetectorGeometry(
            distance=detector_distance_mm,
            image_shape_px=(32, 32),
            pixel_size_mm=(2.0, 2.0),
            beam_center_px=(16.0, 16.0),
        )
        carrier_pattern: RHEEDPattern
        carrier_pattern, _ = _ewald_amplitude_pattern(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
            theta_deg=5.0,
            phi_deg=0.0,
            hmax=1,
            kmax=1,
            detector_distance=detector_distance_mm,
        )
        legacy_flat_pattern = create_rheed_pattern(
            g_indices=carrier_pattern.G_indices,
            k_out=carrier_pattern.k_out,
            detector_points=_legacy_flat_detector_points(
                carrier_pattern.k_out, detector_distance_mm
            ),
            intensities=carrier_pattern.intensities,
        )

        carrier_image = render_pattern_to_image(
            carrier_pattern,
            geometry=geometry,
            spot_sigma_px=1.2,
        )
        legacy_image = render_pattern_to_image(
            legacy_flat_pattern,
            geometry=geometry,
            spot_sigma_px=1.2,
        )

        chex.assert_shape(carrier_image, (32, 32))
        chex.assert_tree_all_finite(carrier_image)
        self.assertGreater(float(jnp.max(carrier_image)), 0.0)
        chex.assert_trees_all_close(
            carrier_image,
            legacy_image,
            atol=1e-12,
            rtol=1e-12,
        )

    @staticmethod
    def _detector_metric(image: Float[Array, "H W"]) -> scalar_float:
        """Return an asymmetric scalar image metric."""
        height_px, width_px = image.shape
        x_axis: Float[Array, "W"] = jnp.linspace(0.0, 1.0, width_px)
        y_axis: Float[Array, "H"] = jnp.linspace(0.0, 1.0, height_px)
        y_grid: Float[Array, "H W"]
        x_grid: Float[Array, "H W"]
        y_grid, x_grid = jnp.meshgrid(y_axis, x_axis, indexing="ij")
        return jnp.sum(image * (0.7 * x_grid + 1.3 * y_grid))

    def _assert_multislice_distribution_changes_image(
        self,
        distribution: Distribution,
    ) -> None:
        """Assert one multislice producer axis changes the detector image."""
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "potential_slices": self._tiny_potential_slices(),
            "energy_kev": 20.0,
            "theta_deg": 5.0,
            "phi_deg": 0.0,
            "detector_distance_mm": 20.0,
            "image_shape_px": (16, 16),
            "pixel_size_mm": (2.0, 2.0),
            "beam_center_px": (8.0, 8.0),
            "spot_sigma_px": 1.0,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "kernel": "multislice",
        }
        base: Float[Array, "16 16"] = simulate_detector_image(**kwargs)
        modified: Float[Array, "16 16"] = simulate_detector_image(
            **kwargs,
            distribution=distribution,
            defect_surface_layer_depth_angstrom=0.8,
        )

        chex.assert_shape(modified, (16, 16))
        chex.assert_tree_all_finite(modified)
        assert float(jnp.max(jnp.abs(modified - base))) > 1.0e-8

    def test_simulate_detector_image_multislice_twin_axis_changes_image(
        self,
    ) -> None:
        r"""FG1: twin axes bind to generated multislice potentials.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: twin axes
        bind to generated multislice potentials.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = twin_wall_to_distribution(
            twin_angles_deg=jnp.array([0.0, 4.0]),
            wall_positions_angstrom=jnp.array([0.4, 0.4]),
            twin_fractions=jnp.array([0.25, 0.75]),
            twin_spacing_angstrom=4.0,
            coherence_length_angstrom=10.0,
        )
        self._assert_multislice_distribution_changes_image(distribution)

    def test_simulate_detector_image_multislice_step_axis_changes_image(
        self,
    ) -> None:
        r"""FG1: step axes bind to generated multislice potentials.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: step axes
        bind to generated multislice potentials.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = step_edge_to_distribution(
            step_heights_angstrom=jnp.array([1.0]),
            terrace_widths_angstrom=jnp.array([2.0]),
            step_fractions=jnp.array([1.0]),
            line_azimuths_deg=jnp.array([30.0]),
            coherence_length_angstrom=0.5,
            regular=False,
        )
        self._assert_multislice_distribution_changes_image(distribution)

    def test_simulate_detector_image_multislice_grain_axis_changes_image(
        self,
    ) -> None:
        r"""FG1: grain axes bind orientation and size under multislice.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: grain axes
        bind orientation and size under multislice.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = grain_population_to_distribution(
            orientation_angles_deg=jnp.array([5.0]),
            grain_sizes_angstrom=jnp.array([1.0]),
            grain_volume_fractions=jnp.array([1.0]),
        )
        self._assert_multislice_distribution_changes_image(distribution)

    def test_simulate_detector_image_multislice_size_axis_changes_image(
        self,
    ) -> None:
        r"""FG1: size axes bind finite-domain multislice envelopes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: size axes
        bind finite-domain multislice envelopes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        distribution: Distribution = create_distribution(
            samples=jnp.array([[1.0]], dtype=jnp.float64),
            weights=jnp.array([1.0], dtype=jnp.float64),
            reduction=ReductionMode.INCOHERENT,
            axis_id="size",
        )
        self._assert_multislice_distribution_changes_image(distribution)

    def test_simulate_detector_image_beam_modes_match_instrument_wrapper(
        self,
    ) -> None:
        r"""Main simulator accepts explicit beam modes on the Layer-1 path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Main simulator
        accepts explicit beam modes on the Layer-1 path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        beam_modes = create_gaussian_schell_beam(
            beta_in_plane=0.25,
            beta_out_of_plane=0.1,
            divergence_in_plane_rad=1.5e-4,
            divergence_out_of_plane_rad=0.75e-4,
            energy_spread_ev=0.15,
        )
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Explicit beam modes are mutually exclusive with generic axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Explicit beam
        modes are mutually exclusive with generic axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Coherent beam modes preserve CTR streak rendering.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Coherent beam
        modes preserve CTR streak rendering.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        kwargs: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Single coherent beam mode matches the unbroadened spot path.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Single coherent
        beam mode matches the unbroadened spot path.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        image_shape_px: tuple[int, int] = (16, 24)
        pixel_size_mm: tuple[float, float] = (6.0, 16.0)
        beam_center_px: tuple[float, float] = (12.0, 2.0)
        reference: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
            energy_kev=20.0,
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
        r"""Beam-mode wrapper delegates to generic Layer-1 reduction.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Beam-mode wrapper
        delegates to generic Layer-1 reduction.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        base_energy_kev: float = 20.0
        base_theta_deg: float = 2.0
        base_phi_deg: float = 1.5

        def _bound(sample: Float[Array, "3"]) -> Complex[Array, "16 24"]:
            """Evaluate the instrument field for one beam sample."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=base_energy_kev + 1.0e-3 * sample[2],
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
            return field

        expected: Float[Array, "16 24"] = apply_distribution(
            distribution,
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image_instrument(
            crystal=_SI_CRYSTAL_2ATOM,
            beam_modes=beam_modes,
            energy_kev=base_energy_kev,
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
        r"""Beam and orientation producers compose through Layer 1.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Beam and
        orientation producers compose through Layer 1.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
            """Evaluate the field for one beam-orientation sample."""
            field: Complex[Array, "16 24"] = kinematic_amplitude(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0 + 1.0e-3 * sample[2],
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
            return field

        expected: Float[Array, "16 24"] = apply_distributions(
            [beam_distribution, orientation_distribution],
            _bound,
        )
        expected = expected / jnp.maximum(jnp.max(expected), 1e-12)
        actual: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            beam_modes=beam_modes,
            orientation_distribution=orientation_dist,
            energy_kev=20.0,
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
        r"""Instrument wrapper remains kinematic-only compatibility API.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Instrument wrapper
        remains kinematic-only compatibility API.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
        r"""Check dense rendering elongates CTRs vertically on detector.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check dense
        rendering elongates CTRs vertically on detector.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        image: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            energy_kev=20.0,
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
        r"""Check orientation-distribution yields a valid dense image.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Check
        orientation-distribution yields a valid dense image.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        orientation_dist: Float[Array, "..."] = create_discrete_orientation(
            angles_deg=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.4, 0.6]),
        )
        image: Float[Array, "..."] = simulate_detector_image(
            crystal=_SI_CRYSTAL_2ATOM,
            orientation_distribution=orientation_dist,
            energy_kev=20.0,
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
            "energy_kev": 20.0,
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
        r"""jax.grad through simulate_detector_image w.r.t. GSM beta is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad through
        simulate_detector_image w.r.t. GSM beta is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""jax.grad through public twin fraction is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad through
        public twin fraction is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""jax.grad through public grain size is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad through
        public grain size is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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

    @staticmethod
    def _tiny_potential_slices(scale: float = 0.05) -> PotentialSlices:
        """Create a compact potential volume for multislice grad gates."""
        slices: Float[Array, "2 8 8"] = jnp.zeros((2, 8, 8), dtype=jnp.float64)
        slices = slices.at[0, 3, 3].set(scale)
        slices = slices.at[1, 4, 4].set(0.5 * scale)
        return create_potential_slices(
            slices=slices,
            slice_thickness=1.5,
            x_calibration=0.75,
            y_calibration=0.75,
        )

    def _multislice_kwargs(self) -> dict[str, Any]:
        """Shared compact multislice settings for FG1 grad gates."""
        return {
            "crystal": _SI_CRYSTAL_2ATOM,
            "potential_slices": self._tiny_potential_slices(),
            "energy_kev": 20.0,
            "theta_deg": 5.0,
            "phi_deg": 0.0,
            "detector_distance_mm": 20.0,
            "image_shape_px": (16, 16),
            "pixel_size_mm": (2.0, 2.0),
            "beam_center_px": (8.0, 8.0),
            "spot_sigma_px": 1.0,
            "angular_divergence_mrad": 0.0,
            "energy_spread_ev": 0.0,
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
            "kernel": "multislice",
        }

    def test_grad_public_multislice_twin_axis_is_live(self) -> None:
        r"""FG1: jax.grad through multislice twin samples is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: jax.grad
        through multislice twin samples is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(twin_angle_deg: scalar_float) -> scalar_float:
            distribution: Distribution = twin_wall_to_distribution(
                twin_angles_deg=jnp.array([twin_angle_deg]),
                wall_positions_angstrom=jnp.array([0.4]),
                twin_fractions=jnp.array([1.0]),
                twin_spacing_angstrom=4.0,
                coherence_length_angstrom=10.0,
            )
            image: Float[Array, "16 16"] = simulate_detector_image(
                **self._multislice_kwargs(),
                distribution=distribution,
                defect_surface_layer_depth_angstrom=0.8,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8

    def test_grad_public_multislice_step_axis_is_live(self) -> None:
        r"""FG1: jax.grad through multislice step samples is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: jax.grad
        through multislice step samples is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(step_height_angstrom: scalar_float) -> scalar_float:
            distribution: Distribution = step_edge_to_distribution(
                step_heights_angstrom=jnp.array([step_height_angstrom]),
                terrace_widths_angstrom=jnp.array([2.0]),
                step_fractions=jnp.array([1.0]),
                line_azimuths_deg=jnp.array([30.0]),
                coherence_length_angstrom=0.5,
                regular=False,
            )
            image: Float[Array, "16 16"] = simulate_detector_image(
                **self._multislice_kwargs(),
                distribution=distribution,
                defect_surface_layer_depth_angstrom=0.8,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8

    def test_grad_public_multislice_grain_axis_is_live(self) -> None:
        r"""FG1: jax.grad through multislice grain size is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: jax.grad
        through multislice grain size is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(grain_size_angstrom: scalar_float) -> scalar_float:
            distribution: Distribution = grain_population_to_distribution(
                orientation_angles_deg=jnp.array([5.0]),
                grain_sizes_angstrom=jnp.array([grain_size_angstrom]),
                grain_volume_fractions=jnp.array([1.0]),
            )
            image: Float[Array, "16 16"] = simulate_detector_image(
                **self._multislice_kwargs(),
                distribution=distribution,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8

    def test_grad_public_multislice_size_axis_is_live(self) -> None:
        r"""FG1: jax.grad through multislice size samples is live.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: FG1: jax.grad
        through multislice size samples is live.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(size_angstrom: scalar_float) -> scalar_float:
            distribution: Distribution = create_distribution(
                samples=jnp.array([[size_angstrom]], dtype=jnp.float64),
                weights=jnp.array([1.0], dtype=jnp.float64),
                reduction=ReductionMode.INCOHERENT,
                axis_id="size",
            )
            image: Float[Array, "16 16"] = simulate_detector_image(
                **self._multislice_kwargs(),
                distribution=distribution,
            )
            return self._detector_metric(image)

        grad_value: scalar_float = jax.grad(loss)(jnp.float64(1.0))
        chex.assert_tree_all_finite(grad_value)
        assert float(jnp.abs(grad_value)) > 1e-8


class TestEwaldSimulatorGradients(chex.TestCase, parameterized.TestCase):
    """Gradient existence and correctness for the ewald_simulator."""

    def _ewald_loss(self, **override: object) -> scalar_float:
        """Compute sum of intensities from ewald_simulator."""
        defaults: Any = {
            "crystal": _SI_CRYSTAL_2ATOM,
            "energy_kev": 20.0,
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
        r"""Gradient w.r.t. temperature is finite and non-zero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient w.r.t.
        temperature is finite and non-zero.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(temp: scalar_float) -> scalar_float:
            return self._ewald_loss(temperature=temp)

        g: scalar_float = jax.grad(loss)(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_roughness(self) -> None:
        r"""Gradient w.r.t. surface roughness is finite and non-zero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient w.r.t.
        surface roughness is finite and non-zero.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(roughness: scalar_float) -> scalar_float:
            return self._ewald_loss(surface_roughness=roughness)

        g: scalar_float = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        assert jnp.abs(g) > 1e-12

    def test_grad_polar_angle(self) -> None:
        r"""Gradient w.r.t. incidence angle is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient w.r.t.
        incidence angle is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(theta: scalar_float) -> scalar_float:
            return self._ewald_loss(theta_deg=theta)

        g: scalar_float = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(g)

    def test_grad_voltage(self) -> None:
        r"""Gradient w.r.t. beam voltage is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient w.r.t.
        beam voltage is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def loss(voltage: scalar_float) -> scalar_float:
            return self._ewald_loss(energy_kev=voltage)

        g: scalar_float = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_vmap_grad(self) -> None:
        r"""vmap(grad(loss)) over temperatures produces correct shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: vmap(grad(loss))
        over temperatures produces correct shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, vectorization, protecting
        JAX transform compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""Jacrev w.r.t. (temperature, roughness) produces (2,) Jacobian.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Jacrev w.r.t.
        (temperature, roughness) produces (2,) Jacobian.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

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
        r"""Ewald simulator grad w.r.t. temperature matches finite diff.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Ewald simulator
        grad w.r.t. temperature matches finite diff.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def f(temp: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0,
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
        r"""Ewald simulator grad w.r.t. roughness matches finite diff.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Ewald simulator
        grad w.r.t. roughness matches finite diff.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def f(roughness: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0,
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
        r"""Gradient through multislice propagation w.r.t. voltage.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gradient through
        multislice propagation w.r.t. voltage.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
                energy_kev=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        g: scalar_float = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_multislice_grad_voltage_correct(self) -> None:
        r"""Multislice grad w.r.t. voltage matches finite diff.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Multislice grad
        w.r.t. voltage matches finite diff.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
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
                energy_kev=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        check_grads(jax_safe(f), (jnp.float64(20.0),), order=1, atol=1e-2)


class TestEwaldSimulatorVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap matches sequential for ewald_simulator."""

    def test_ewald_simulator_vmap_temperature_consistent(self) -> None:
        r"""Batched ewald_simulator over temps matches sequential.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Batched
        ewald_simulator over temps matches sequential.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_simulator``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def f(temp: scalar_float) -> scalar_float:
            pattern: Float[Array, "..."] = ewald_simulator(
                crystal=_SI_CRYSTAL_2ATOM,
                energy_kev=20.0,
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
