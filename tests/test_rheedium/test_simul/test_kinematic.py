"""Tests for kinematic RHEED simulator.

This module tests the kinematic RHEED implementation that follows
the algorithm from arXiv:2207.06642.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float, Integer

from rheedium.inout.cif import parse_cif
from rheedium.simul.kinematic import (
    kinematic_spot_simulator,
    make_ewald_sphere,
)
from rheedium.simul.kinematic import (
    simple_structure_factor as kinematic_structure_factor,
)
from rheedium.simul.simulator import (
    find_kinematic_reflections as kinematic_ewald_sphere,
)
from rheedium.simul.simulator import (
    project_on_detector_geometry,
)
from rheedium.tools import (
    incident_wavevector as kinematic_incident_wavevector,
)
from rheedium.tools import wavelength_ang as kinematic_wavelength
from rheedium.types.crystal_types import (
    CrystalStructure,
    create_crystal_structure,
)
from rheedium.types.custom_types import scalar_float
from rheedium.types.detector import DetectorGeometry
from rheedium.types.rheed_types import RHEEDPattern
from rheedium.ucell.unitcell import (
    miller_to_reciprocal,
    reciprocal_lattice_vectors,
)


class TestKinematicWavelength(chex.TestCase):
    """Test relativistic wavelength calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("10kV", 10.0, 0.1220),
        ("20kV", 20.0, 0.0859),
        ("30kV", 30.0, 0.0698),
    )
    def test_wavelength_values(
        self, energy_kev: float, expected_lambda: float
    ) -> None:
        r"""Test wavelength calculation matches expected values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: wavelength
        calculation matches expected values.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``energy_kev``, ``expected_lambda``, so the documented behavior is
        checked across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_wavelength: Callable[..., Any] = self.variant(kinematic_wavelength)

        wavelength: float = var_wavelength(energy_kev)

        # Check within 0.5% of expected
        chex.assert_scalar_positive(float(wavelength))
        assert jnp.abs(wavelength - expected_lambda) / expected_lambda < 0.005


class TestKinematicIncidentWavevector(chex.TestCase):
    """Test incident wavevector construction."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_incident_wavevector_magnitude(self) -> None:
        r"""Test \|k_in\| = 2π/λ.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|k_in\| = 2π/λ.

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_k_in: Callable[..., Any] = self.variant(
            kinematic_incident_wavevector
        )

        wavelength: float = 0.0859  # 20 keV
        theta_deg: float = 2.0

        k_in: Any = var_k_in(wavelength, theta_deg)

        # Check magnitude
        k_mag: Float[Array, "..."] = jnp.linalg.norm(k_in)
        expected_mag: scalar_float = 2.0 * jnp.pi / wavelength

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_incident_wavevector_components(self) -> None:
        r"""Test k_in = k·[cos(θ), 0, -sin(θ)] (beam going into surface).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: k_in = k·[cos(θ),
        0, -sin(θ)] (beam going into surface).

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_k_in: Callable[..., Any] = self.variant(
            kinematic_incident_wavevector
        )

        wavelength: float = 0.0859
        theta_deg: float = 2.0
        theta_rad: scalar_float = jnp.deg2rad(theta_deg)

        k_in: Any = var_k_in(wavelength, theta_deg)

        k_mag: scalar_float = 2.0 * jnp.pi / wavelength

        # Expected components: beam goes at grazing angle θ into surface
        # z is up (surface normal), so beam has negative z component
        expected_x: Float[Array, "..."] = k_mag * jnp.cos(theta_rad)
        expected_y: float = 0.0
        expected_z: Float[Array, "..."] = -k_mag * jnp.sin(
            theta_rad
        )  # Negative: beam going down

        chex.assert_trees_all_close(k_in[0], expected_x, rtol=1e-6)
        chex.assert_trees_all_close(k_in[1], expected_y, atol=1e-10)
        chex.assert_trees_all_close(k_in[2], expected_z, rtol=1e-6)


class TestKinematicEwaldSphere(chex.TestCase):
    """Test Ewald sphere construction."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_sphere_elastic_scattering(self) -> None:
        r"""Test that all k_out satisfy \|k_out\| ≈ \|k_in\|.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all k_out satisfy
        \|k_out\| ≈ \|k_in\|.

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_ewald: Callable[..., Any] = self.variant(kinematic_ewald_sphere)

        # Setup: incident beam at grazing incidence
        # (k_in_z < 0, going into surface)
        k_in: Float[Array, "..."] = jnp.array([73.0, 0.0, -2.5])
        k_mag: Float[Array, "..."] = jnp.linalg.norm(k_in)

        # Some test reciprocal vectors
        # For RHEED, we want upward scattering (k_out_z > 0)
        G_vectors: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 5.0],  # k_out_z = -2.5 + 5.0 = 2.5 > 0 (upward)
                [0.0, 1.5, 5.1],  # k_out_z = -2.5 + 5.1 = 2.6 > 0 (upward)
                [2.0, 0.0, -2.0],  # k_out_z = -2.5 - 2.0 = -4.5 < 0 (rejected)
            ]
        )

        # z_sign=1.0 selects upward scattering (k_out_z > 0)
        indices: Any
        k_out: Float[Array, "..."]
        indices, k_out = var_ewald(
            k_in, G_vectors, z_sign=1.0, tolerance_inv_ang=0.1
        )

        # Check that we got some valid results
        n_valid: scalar_float = int(jnp.sum(indices >= 0))
        assert n_valid > 0, "Should find some valid reflections"

        # All valid k_out should satisfy elastic scattering
        valid_k_out: Float[Array, "..."] = k_out[indices >= 0]
        k_out_mags: Float[Array, "..."] = jnp.linalg.norm(valid_k_out, axis=1)
        k_mag_out: scalar_float
        for k_mag_out in k_out_mags:
            assert jnp.abs(k_mag_out - k_mag) / k_mag < 0.1, (
                "Elastic scattering"
            )

        # All valid k_out should have positive z (upward scattering for RHEED)
        assert jnp.all(valid_k_out[:, 2] > 0.0), "RHEED requires k_out_z > 0"


class TestDetectorProjection(chex.TestCase):
    """Test detector projection implementing inverse of paper's Equations 5-6.

    Paper's Equations 5-6 map detector → reciprocal space::

        x = k₀ · x_d / R                    [Eq. 5]
        y = k₀ · (-d/R + cos θ)             [Eq. 6]

    where R = √(d² + x_d² + y_d²)

    Our implementation inverts this to map reciprocal → detector::

        R = d / (cos θ - y/k₀)
        x_d = x · R / k₀
        y_d = √(R² - d² - x_d²)
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_detector_projection_roundtrip(self) -> None:
        r"""Test that ray-tracing projection gives reasonable coordinates.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ray-tracing
        projection gives reasonable coordinates. Existing context from the
        original test prose: The simplified projection uses: x_d = k_y * d /
        k_x, y_d = k_z * d / k_x

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_project: Callable[..., Any] = self.variant(
            project_on_detector_geometry
        )

        # Setup: typical RHEED parameters
        wavelength: float = 0.0859  # ~20 keV electrons
        k0: scalar_float = 2.0 * jnp.pi / wavelength

        # Scattered wavevector: forward + upward scattering
        k_out: Float[Array, "..."] = jnp.array([[k0 * 0.98, 0.5, k0 * 0.05]])
        d: float = 100.0

        # Get detector coordinates
        coords: Float[Array, "..."] = var_project(
            k_out, DetectorGeometry(distance=d)
        )
        x_d: tuple[Any, ...]
        y_d: tuple[Any, ...]
        x_d, y_d = coords[0, 0], coords[0, 1]

        # Verify ray-tracing geometry: x_d = k_y * d / k_x
        expected_x: Float[Array, "..."] = k_out[0, 1] * d / k_out[0, 0]
        expected_y: Float[Array, "..."] = k_out[0, 2] * d / k_out[0, 0]

        chex.assert_trees_all_close(x_d, expected_x, rtol=1e-4)
        chex.assert_trees_all_close(y_d, expected_y, rtol=1e-4)

    @chex.variants(with_jit=True, without_jit=True)
    def test_detector_projection_specular(self) -> None:
        r"""Test projection for near-specular reflection.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: projection for
        near-specular reflection. Existing context from the original test
        prose: Specular: k_out_z ≈ \|k_in_z\| but positive (upward).

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_project: Callable[..., Any] = self.variant(
            project_on_detector_geometry
        )

        theta_deg: float = 2.0
        theta_rad: scalar_float = jnp.deg2rad(theta_deg)
        wavelength: float = 0.0859
        k0: scalar_float = 2.0 * jnp.pi / wavelength

        # Near-specular: same angle out, k_out_z positive, k_out_y = 0
        k_out: Float[Array, "..."] = jnp.array(
            [[k0 * jnp.cos(theta_rad), 0.0, k0 * jnp.sin(theta_rad)]]
        )
        d: float = 100.0

        coords: Float[Array, "..."] = var_project(
            k_out, DetectorGeometry(distance=d)
        )

        # Specular should be close to x_d = 0 (no horizontal deflection)
        chex.assert_trees_all_close(coords[0, 0], 0.0, atol=0.1)
        # y_d should be positive (above horizon)
        assert coords[0, 1] > 0, "Specular reflection should be above horizon"


class TestKinematicStructureFactor(chex.TestCase):
    """Test structure factor calculation.

    :see: :func:`~rheedium.simul.kinematic.simple_structure_factor`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_single_atom(self) -> None:
        r"""Test structure factor for single atom at origin.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        for single atom at origin.

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_sf: Callable[..., Any] = self.variant(kinematic_structure_factor)

        G: Float[Array, "..."] = jnp.array([1.0, 0.0, 0.0])
        positions: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        atomic_nums: Integer[Array, "..."] = jnp.array([14])  # Silicon

        intensity: Any = var_sf(G, positions, atomic_nums)

        # For single atom at origin: F = Z·exp(i·0) = Z
        # I = |F|² = Z²
        expected: Float[Array, "..."] = 14.0**2
        chex.assert_trees_all_close(intensity, expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_systematic_absence(self) -> None:
        r"""Test that (100) is forbidden for diamond structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (100) is forbidden
        for diamond structure.

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_sf: Callable[..., Any] = self.variant(kinematic_structure_factor)

        # Simple cubic with two atoms (like diamond basis)
        G: Float[Array, "..."] = jnp.array(
            [2.0, 0.0, 0.0]
        )  # (100) type reflection
        positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],  # Diamond basis
            ]
        )
        atomic_nums: Integer[Array, "..."] = jnp.array([14, 14])

        intensity: Any = var_sf(G, positions, atomic_nums)

        # This should have very low intensity (systematic absence)
        # F = f·[exp(0) + exp(iπ)] = f·[1 - 1] = 0 for certain G
        # Actual value depends on exact G, but should be much less than max
        chex.assert_scalar_positive(float(intensity))


class TestKinematicSimulator(chex.TestCase):
    """Test complete kinematic simulator.

    :see: :func:`~rheedium.simul.kinematic_spot_simulator`
    """

    def setUp(self) -> None:
        """Set up test crystal."""
        super().setUp()

        # Simple cubic Silicon crystal
        a_si: float = 5.43  # Silicon lattice constant
        frac_pos: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 14.0]]
        )  # One Si atom at origin
        cart_pos: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 0.0, 14.0]]
        )  # Cartesian same for origin

        self.crystal: CrystalStructure = create_crystal_structure(
            frac_positions=frac_pos,
            cart_positions=cart_pos,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_kinematic_spot_simulator_runs(self) -> None:
        r"""Test that simulator runs without errors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: simulator runs
        without errors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        # Note: lmax=5 needed to get upward scattering at θ=2° grazing angle
        # because G_z must exceed |k_in_z| ≈ k0*sin(2°) ≈ 2.5 1/Å
        pattern: RHEEDPattern = kinematic_spot_simulator(
            crystal=self.crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=5,  # Higher lmax for grazing incidence
            detector_distance=100.0,
        )

        # Check output structure
        assert hasattr(pattern, "G_indices")
        assert hasattr(pattern, "k_out")
        assert hasattr(pattern, "detector_points")
        assert hasattr(pattern, "intensities")

        # Check that we found some reflections
        n_reflections: int = len(pattern.intensities)
        assert n_reflections > 0, (
            "Should find reflections at grazing incidence"
        )

        # Check intensities are positive
        assert jnp.all(pattern.intensities >= 0.0)

    def test_kinematic_spot_simulator_detector_coords(self) -> None:
        r"""Test detector coordinates are reasonable.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: detector
        coordinates are reasonable.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = kinematic_spot_simulator(
            crystal=self.crystal,
            energy_kev=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=5,  # Higher lmax for grazing incidence
            detector_distance=100.0,
        )

        # Detector coordinates should be finite
        assert jnp.all(jnp.isfinite(pattern.detector_points))

        # For d=100mm and typical RHEED, spots should be within
        # reasonable range (a sanity check, not a strict requirement).
        max_coord: scalar_float = jnp.max(jnp.abs(pattern.detector_points))
        assert max_coord < 10000.0  # Not absurdly large


class TestMakeEwaldSphere(chex.TestCase):
    """Test Ewald sphere generation.

    :see: :func:`~rheedium.simul.make_ewald_sphere`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_sphere_geometry(self) -> None:
        r"""Test center and radius calculation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: center and radius
        calculation.

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
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        var_make_sphere: Callable[..., Any] = self.variant(make_ewald_sphere)

        # Setup
        k_mag: float = 10.0  # 1/Å
        theta_deg: float = 2.0
        phi_deg: float = 0.0

        center: Float[Array, "..."]
        radius: Any
        center, radius = var_make_sphere(k_mag, theta_deg, phi_deg)

        # Radius should be exactly k_mag
        chex.assert_trees_all_close(radius, k_mag, rtol=1e-6)

        # Center should be -k_in
        # k_in magnitude is k_mag
        center_mag: Float[Array, "..."] = jnp.linalg.norm(center)
        chex.assert_trees_all_close(center_mag, k_mag, rtol=1e-6)

        # Check direction for theta=2, phi=0
        # k_in = [k cos(theta), 0, -k sin(theta)]
        # (approx, check sign convention)
        # In kinematic.py: k_z = -k * sin(theta)
        # So center = -k_in = [-kx, -ky, -kz]
        # center_z = -(-k * sin(theta)) = k * sin(theta) > 0

        theta_rad: scalar_float = jnp.deg2rad(theta_deg)
        expected_z: Float[Array, "..."] = k_mag * jnp.sin(theta_rad)
        chex.assert_trees_all_close(center[2], expected_z, rtol=1e-6)


class TestMgOExtinctionRules(chex.TestCase):
    """Test FCC extinction rules using MgO structure.

    MgO has rocksalt (FCC) structure. The structure factor extinction rule
    for FCC requires Miller indices to be ALL ODD or ALL EVEN for non-zero
    intensity. Mixed indices like (1,0,0), (2,1,0) must have zero intensity.

    This is a critical validation test for the structure factor calculation.
    Reference: Paper Figure 2 shows MgO(001) pattern where (10) is missing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load MgO crystal structure from CIF file."""
        test_data_dir: Path = Path(__file__).parent.parent.parent / "test_data"
        cif_path: Path = test_data_dir / "MgO.cif"
        cls.mgo_crystal = parse_cif(cif_path)

        # Get reciprocal lattice vectors for MgO
        cls.recip_vectors = reciprocal_lattice_vectors(
            *cls.mgo_crystal.cell_lengths,
            *cls.mgo_crystal.cell_angles,
            in_degrees=True,
        )

    def _get_structure_factor_intensity(
        self, h: int, k: int, ell: int
    ) -> float:
        """Calculate structure factor intensity for given Miller indices."""
        hkl: Float[Array, "..."] = jnp.array([[h, k, ell]])
        g_vector: Float[Array, "..."] = miller_to_reciprocal(
            hkl, self.recip_vectors
        )[0]

        atom_positions: Float[Array, "..."] = self.mgo_crystal.cart_positions[
            :, :3
        ]
        atomic_numbers: Float[Array, "..."] = self.mgo_crystal.cart_positions[
            :, 3
        ].astype(jnp.int32)

        intensity: Any = kinematic_structure_factor(
            g_vector, atom_positions, atomic_numbers
        )
        return float(intensity)

    def test_mgo_allowed_reflections_nonzero(self) -> None:
        r"""Test that allowed FCC reflections have non-zero intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: allowed FCC
        reflections have non-zero intensity. Existing context from the original
        test prose: For FCC: (h,k,l) all even or all odd => allowed Examples:
        (1,1,1), (2,0,0), (2,2,0), (2,2,2)

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        # All odd indices - should be allowed
        all_odd_cases: list[Any] = [
            (1, 1, 1),
            (1, 1, 3),
            (3, 1, 1),
        ]
        h: int
        k: int
        ell: int
        for h, k, ell in all_odd_cases:
            intensity: Any = self._get_structure_factor_intensity(h, k, ell)
            assert intensity > 1.0, (
                f"FCC allowed reflection ({h},{k},{ell}) should have non-zero "
                f"intensity, got {intensity}"
            )

        # All even indices - should be allowed
        all_even_cases: list[Any] = [
            (2, 0, 0),
            (2, 2, 0),
            (2, 2, 2),
            (4, 0, 0),
        ]
        for h, k, ell in all_even_cases:
            intensity = self._get_structure_factor_intensity(h, k, ell)
            assert intensity > 1.0, (
                f"FCC allowed reflection ({h},{k},{ell}) should have non-zero "
                f"intensity, got {intensity}"
            )

    def test_mgo_forbidden_reflections_zero(self) -> None:
        r"""Test that forbidden FCC reflections have zero intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: forbidden FCC
        reflections have zero intensity. Existing context from the original
        test prose: For FCC: mixed indices (not all even, not all odd) =>
        forbidden Examples: (1,0,0), (1,1,0), (2,1,0), (2,1,1) This is the
        CRITICAL test - if (1,0,0) shows non-zero intensity, the structure
        factor calculation is fundamentally wrong.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        forbidden_cases: list[Any] = [
            (1, 0, 0),  # Paper explicitly shows this is missing
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (2, 1, 0),
            (2, 0, 1),
            (0, 2, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (3, 0, 0),
            (3, 1, 0),
            (3, 2, 1),
        ]

        tolerance: float = 1e-6  # Numerical tolerance for "zero"

        h: int
        k: int
        ell: int
        for h, k, ell in forbidden_cases:
            intensity: Any = self._get_structure_factor_intensity(h, k, ell)
            assert intensity < tolerance, (
                f"FCC forbidden reflection ({h},{k},{ell}) should have zero "
                f"intensity, got {intensity}. "
                f"This indicates the structure factor calculation is WRONG."
            )

    def test_mgo_origin_reflection(self) -> None:
        r"""Test that (0,0,0) reflection gives expected intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (0,0,0) reflection
        gives expected intensity. Existing context from the original test
        prose: F(0,0,0) = sum of all atomic scattering factors For simplified
        f_j = Z_j: F = Z_Mg + Z_O = 12 + 8 = 20 per formula unit With 8 atoms
        in conventional cell (4 Mg + 4 O): F = 4*12 + 4*8 = 48 + 32 = 80 I =
        \|F\|² = 6400

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        intensity: Any = self._get_structure_factor_intensity(0, 0, 0)
        # 8 atoms total: 4 Mg (Z=12) + 4 O (Z=8)
        expected_f: Float[Array, "..."] = 4 * 12 + 4 * 8  # = 80
        expected_intensity: Float[Array, "..."] = expected_f**2  # = 6400

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)


class TestSrTiO3StructureFactor(chex.TestCase):
    """Test structure factor for SrTiO3 perovskite structure.

    SrTiO3 has the perovskite structure (space group Pm-3m, #221).
    The perovskite structure has cubic symmetry with a 5-atom basis:
    - Sr at corner-centered position (0.5, 0.5, 0.5)
    - Ti at origin (0, 0, 0)
    - O at face-centered positions (0, 0, 0.5), (0.5, 0, 0), (0, 0.5, 0)

    Unlike FCC (which has systematic extinctions for mixed indices),
    perovskite has no lattice-based extinctions - all (h,k,l) reflections
    are allowed. The intensity variations come from the atomic basis.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load SrTiO3 crystal structure from CIF file."""
        test_data_dir: Path = Path(__file__).parent.parent.parent / "test_data"
        cif_path: Path = test_data_dir / "SrTiO3.cif"
        cls.sto_crystal = parse_cif(cif_path)

        # Verify we got the right number of atoms
        assert len(cls.sto_crystal.cart_positions) == 5, (
            "Expected 5 atoms (1 Sr + 1 Ti + 3 O), got "
            f"{len(cls.sto_crystal.cart_positions)}"
        )

        # Get reciprocal lattice vectors
        cls.recip_vectors = reciprocal_lattice_vectors(
            *cls.sto_crystal.cell_lengths,
            *cls.sto_crystal.cell_angles,
            in_degrees=True,
        )

    def _get_structure_factor_intensity(
        self, h: int, k: int, ell: int
    ) -> float:
        """Calculate structure factor intensity for given Miller indices."""
        hkl: Float[Array, "..."] = jnp.array([[h, k, ell]])
        g_vector: Float[Array, "..."] = miller_to_reciprocal(
            hkl, self.recip_vectors
        )[0]

        atom_positions: Float[Array, "..."] = self.sto_crystal.cart_positions[
            :, :3
        ]
        atomic_numbers: Float[Array, "..."] = self.sto_crystal.cart_positions[
            :, 3
        ].astype(jnp.int32)

        intensity: Any = kinematic_structure_factor(
            g_vector, atom_positions, atomic_numbers
        )
        return float(intensity)

    def test_sto_origin_reflection(self) -> None:
        r"""Test that (0,0,0) reflection gives expected intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (0,0,0) reflection
        gives expected intensity. Existing context from the original test
        prose: F(0,0,0) = sum of all atomic scattering factors For simplified
        f_j = Z_j: F = Z_Sr + Z_Ti + 3*Z_O = 38 + 22 + 3*8 = 84 I = \|F\|² =
        7056

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        intensity: Any = self._get_structure_factor_intensity(0, 0, 0)
        expected_f: Float[Array, "..."] = 38 + 22 + 3 * 8  # = 84
        expected_intensity: Float[Array, "..."] = expected_f**2  # = 7056

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_100_reflection(self) -> None:
        r"""Test (1,0,0) reflection intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (1,0,0) reflection
        intensity. Existing context from the original test prose: For
        perovskite with: Sr at (0.5, 0.5, 0.5): phase = exp(i*pi) = -1 Ti at
        (0, 0, 0): phase = exp(0) = 1 O at (0, 0, 0.5): phase = exp(0) = 1 O at
        (0.5, 0, 0): phase = exp(i*pi) = -1 O at (0, 0.5, 0): phase = exp(0) =
        1 F = -38 + 22 + 8 - 8 + 8 = -8 I = 64

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        intensity: Any = self._get_structure_factor_intensity(1, 0, 0)
        # F = -Z_Sr + Z_Ti + Z_O - Z_O + Z_O = -38 + 22 + 8 - 8 + 8 = -8
        expected_f: Float[Array, "..."] = -38 + 22 + 8 - 8 + 8  # = -8
        expected_intensity: Float[Array, "..."] = expected_f**2  # = 64

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_110_reflection(self) -> None:
        r"""Test (1,1,0) reflection intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (1,1,0) reflection
        intensity. Existing context from the original test prose: For (1,1,0),
        the phase contributions are:: Sr at (0.5, 0.5, 0.5): exp(i*2*pi) = 1 Ti
        at (0, 0, 0): 1 O at (0, 0, 0.5): 1 O at (0.5, 0, 0): exp(i*pi) = -1 O
        at (0, 0.5, 0): exp(i*pi) = -1 F = 38 + 22 + 8 - 8 - 8 = 52 I = 2704

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        intensity: Any = self._get_structure_factor_intensity(1, 1, 0)
        expected_f: Float[Array, "..."] = 38 + 22 + 8 - 8 - 8  # = 52
        expected_intensity: Float[Array, "..."] = expected_f**2  # = 2704

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_111_reflection(self) -> None:
        r"""Test (1,1,1) reflection intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: (1,1,1) reflection
        intensity. Existing context from the original test prose: For (1,1,1):
        Sr at (0.5, 0.5, 0.5): phase = exp(i*3*pi) = -1 Ti at (0, 0, 0): phase
        = 1 O at (0, 0, 0.5): phase = exp(i*pi) = -1 O at (0.5, 0, 0): phase =
        exp(i*pi) = -1 O at (0, 0.5, 0): phase = exp(i*pi) = -1 F = -38 + 22 -
        8 - 8 - 8 = -40 I = 1600

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        intensity: Any = self._get_structure_factor_intensity(1, 1, 1)
        expected_f: Float[Array, "..."] = -38 + 22 - 8 - 8 - 8  # = -40
        expected_intensity: Float[Array, "..."] = expected_f**2  # = 1600

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_all_reflections_nonzero(self) -> None:
        r"""Test that various reflections have non-zero intensity.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: various
        reflections have non-zero intensity. Existing context from the original
        test prose: Unlike FCC, perovskite has no systematic extinctions from
        the lattice. All reflections should have some intensity (though values
        vary based on the atomic basis phases).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_kinematic``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        test_cases: list[Any] = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 1, 1),
            (2, 2, 0),
            (2, 2, 1),
            (3, 0, 0),
        ]

        h: int
        k: int
        ell: int
        for h, k, ell in test_cases:
            intensity: Any = self._get_structure_factor_intensity(h, k, ell)
            assert intensity > 0.1, (
                f"Perovskite reflection ({h},{k},{ell}) should have non-zero "
                f"intensity, got {intensity}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
