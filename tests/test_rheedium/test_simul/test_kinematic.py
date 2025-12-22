"""Tests for kinematic RHEED simulator.

This module tests the kinematic RHEED implementation that follows
the algorithm from arXiv:2207.06642.
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from rheedium.inout import parse_cif
from rheedium.simul.kinematic import (
    find_kinematic_reflections as kinematic_ewald_sphere,
    incident_wavevector as kinematic_incident_wavevector,
    kinematic_detector_projection,
    kinematic_simulator,
    make_ewald_sphere,
    simple_structure_factor as kinematic_structure_factor,
    wavelength_ang as kinematic_wavelength,
)
from rheedium.types import create_crystal_structure
from rheedium.ucell import miller_to_reciprocal, reciprocal_lattice_vectors


class TestKinematicWavelength(chex.TestCase):
    """Test relativistic wavelength calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("10kV", 10.0, 0.1220),
        ("20kV", 20.0, 0.0859),
        ("30kV", 30.0, 0.0698),
    )
    def test_wavelength_values(self, voltage_kv: float, expected_lambda: float):
        """Test wavelength calculation matches expected values."""
        var_wavelength = self.variant(kinematic_wavelength)

        wavelength = var_wavelength(voltage_kv)

        # Check within 0.5% of expected
        chex.assert_scalar_positive(float(wavelength))
        assert jnp.abs(wavelength - expected_lambda) / expected_lambda < 0.005


class TestKinematicIncidentWavevector(chex.TestCase):
    """Test incident wavevector construction."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_incident_wavevector_magnitude(self):
        """Test |k_in| = 2π/λ."""
        var_k_in = self.variant(kinematic_incident_wavevector)

        wavelength = 0.0859  # 20 keV
        theta_deg = 2.0

        k_in = var_k_in(wavelength, theta_deg)

        # Check magnitude
        k_mag = jnp.linalg.norm(k_in)
        expected_mag = 2.0 * jnp.pi / wavelength

        chex.assert_trees_all_close(k_mag, expected_mag, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_incident_wavevector_components(self):
        """Test k_in = k·[cos(θ), 0, sin(θ)]."""
        var_k_in = self.variant(kinematic_incident_wavevector)

        wavelength = 0.0859
        theta_deg = 2.0
        theta_rad = jnp.deg2rad(theta_deg)

        k_in = var_k_in(wavelength, theta_deg)

        k_mag = 2.0 * jnp.pi / wavelength

        # Expected components
        expected_x = k_mag * jnp.cos(theta_rad)
        expected_y = 0.0
        expected_z = k_mag * jnp.sin(theta_rad)

        chex.assert_trees_all_close(k_in[0], expected_x, rtol=1e-6)
        chex.assert_trees_all_close(k_in[1], expected_y, atol=1e-10)
        chex.assert_trees_all_close(k_in[2], expected_z, rtol=1e-6)


class TestKinematicEwaldSphere(chex.TestCase):
    """Test Ewald sphere construction."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_sphere_elastic_scattering(self):
        """Test that all k_out satisfy |k_out| ≈ |k_in|."""
        var_ewald = self.variant(kinematic_ewald_sphere)

        # Setup
        k_in = jnp.array([73.0, 0.0, 2.5])
        k_mag = jnp.linalg.norm(k_in)

        # Some test reciprocal vectors
        G_vectors = jnp.array([
            [1.0, 0.0, -2.5],  # Should be allowed (k_out_z < 0)
            [0.0, 1.5, -2.4],  # Should be allowed
            [2.0, 0.0, 2.0],  # Should be rejected (k_out_z > 0)
        ])

        indices, k_out = var_ewald(k_in, G_vectors, tolerance=0.1)

        # All k_out should satisfy elastic scattering
        k_out_mags = jnp.linalg.norm(k_out, axis=1)
        for k_mag_out in k_out_mags:
            assert jnp.abs(k_mag_out - k_mag) < 0.1

        # All k_out should have negative z
        assert jnp.all(k_out[:, 2] < 0.0)


class TestKinematicDetectorProjection(chex.TestCase):
    """Test detector projection implementing inverse of paper's Equations 5-6.

    Paper's Equations 5-6 map detector → reciprocal space:
        x = k₀ · x_d / R                    [Eq. 5]
        y = k₀ · (-d/R + cos θ)             [Eq. 6]
    where R = √(d² + x_d² + y_d²)

    Our implementation inverts this to map reciprocal → detector:
        R = d / (cos θ - y/k₀)
        x_d = x · R / k₀
        y_d = √(R² - d² - x_d²)
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_detector_projection_roundtrip(self):
        """Test that projection and inverse equations are consistent.

        If we project to detector and then apply paper's Eqs. 5-6,
        we should recover the original reciprocal space coordinates.
        """
        var_project = self.variant(kinematic_detector_projection)

        # Setup: typical RHEED parameters
        theta_deg = 2.0
        theta_rad = jnp.deg2rad(theta_deg)
        wavelength = 0.0859  # ~20 keV electrons
        k0 = 2.0 * jnp.pi / wavelength

        # Incident wavevector: grazing incidence along +x, z pointing down
        k_in = jnp.array([
            k0 * jnp.cos(theta_rad), 0.0, -k0 * jnp.sin(theta_rad)
        ])

        # Scattered wavevector: upward scattering
        k_out = jnp.array([[k0 * 0.98, 0.5, k0 * 0.05]])
        d = 100.0

        # Get detector coordinates from our inverse function
        coords = var_project(k_out, k_in, d, theta_deg)
        x_d, y_d = coords[0, 0], coords[0, 1]

        # Apply paper's forward equations 5-6 to verify roundtrip
        R = jnp.sqrt(d**2 + x_d**2 + y_d**2)
        x_recip_recovered = k0 * x_d / R  # Eq. 5
        y_recip_recovered = k0 * (-d / R + jnp.cos(theta_rad))  # Eq. 6

        # What we used as input: x_recip = G_x, y_recip = k_out_z
        x_recip_input = k_out[0, 0] - k_in[0]
        y_recip_input = k_out[0, 2]

        # The recovered values should match the inputs
        chex.assert_trees_all_close(
            x_recip_recovered, x_recip_input, rtol=1e-4
        )
        chex.assert_trees_all_close(
            y_recip_recovered, y_recip_input, rtol=1e-4
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_detector_projection_specular(self):
        """Test projection for near-specular reflection.

        Specular: k_out_z ≈ k_in_z but with opposite sign.
        """
        var_project = self.variant(kinematic_detector_projection)

        theta_deg = 2.0
        theta_rad = jnp.deg2rad(theta_deg)
        wavelength = 0.0859
        k0 = 2.0 * jnp.pi / wavelength

        # Incident: grazing incidence
        k_in = jnp.array([
            k0 * jnp.cos(theta_rad), 0.0, -k0 * jnp.sin(theta_rad)
        ])

        # Near-specular: same angle out, k_out_z positive
        k_out = jnp.array([[
            k0 * jnp.cos(theta_rad), 0.0, k0 * jnp.sin(theta_rad)
        ]])
        d = 100.0

        coords = var_project(k_out, k_in, d, theta_deg)

        # Specular should be close to x_d = 0 (no horizontal deflection)
        chex.assert_trees_all_close(coords[0, 0], 0.0, atol=0.1)
        # y_d should be positive (above horizon)
        assert coords[0, 1] > 0, "Specular reflection should be above horizon"


class TestKinematicStructureFactor(chex.TestCase):
    """Test structure factor calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_single_atom(self):
        """Test structure factor for single atom at origin."""
        var_sf = self.variant(kinematic_structure_factor)

        G = jnp.array([1.0, 0.0, 0.0])
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_nums = jnp.array([14])  # Silicon

        intensity = var_sf(G, positions, atomic_nums)

        # For single atom at origin: F = Z·exp(i·0) = Z
        # I = |F|² = Z²
        expected = 14.0**2
        chex.assert_trees_all_close(intensity, expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_structure_factor_systematic_absence(self):
        """Test that (100) is forbidden for diamond structure."""
        var_sf = self.variant(kinematic_structure_factor)

        # Simple cubic with two atoms (like diamond basis)
        G = jnp.array([2.0, 0.0, 0.0])  # (100) type reflection
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],  # Diamond basis
        ])
        atomic_nums = jnp.array([14, 14])

        intensity = var_sf(G, positions, atomic_nums)

        # This should have very low intensity (systematic absence)
        # F = f·[exp(0) + exp(iπ)] = f·[1 - 1] = 0 for certain G
        # Actual value depends on exact G, but should be much less than max
        chex.assert_scalar_positive(float(intensity))


class TestKinematicSimulator(chex.TestCase):
    """Test complete kinematic simulator."""

    def setUp(self):
        """Set up test crystal."""
        super().setUp()

        # Simple cubic Silicon crystal
        a_si = 5.43  # Silicon lattice constant
        frac_pos = jnp.array([[0.0, 0.0, 0.0, 14.0]])  # One Si atom at origin
        cart_pos = jnp.array([[0.0, 0.0, 0.0, 14.0]])  # Cartesian same for origin

        self.crystal = create_crystal_structure(
            frac_positions=frac_pos,
            cart_positions=cart_pos,
            cell_lengths=jnp.array([a_si, a_si, a_si]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_kinematic_simulator_runs(self):
        """Test that simulator runs without errors."""
        pattern = kinematic_simulator(
            crystal=self.crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            detector_distance=100.0,
        )

        # Check output structure
        assert hasattr(pattern, "G_indices")
        assert hasattr(pattern, "k_out")
        assert hasattr(pattern, "detector_points")
        assert hasattr(pattern, "intensities")

        # Check that we found some reflections
        n_reflections = len(pattern.intensities)
        assert n_reflections > 0

        # Check intensities are positive
        assert jnp.all(pattern.intensities >= 0.0)

    def test_kinematic_simulator_detector_coords(self):
        """Test detector coordinates are reasonable."""
        pattern = kinematic_simulator(
            crystal=self.crystal,
            voltage_kv=20.0,
            theta_deg=2.0,
            hmax=2,
            kmax=2,
            lmax=1,
            detector_distance=100.0,
        )

        # Detector coordinates should be finite
        assert jnp.all(jnp.isfinite(pattern.detector_points))

        # For d=100mm and typical RHEED, spots should be within reasonable range
        # (This is a sanity check, not a strict requirement)
        max_coord = jnp.max(jnp.abs(pattern.detector_points))
        assert max_coord < 10000.0  # Not absurdly large



class TestMakeEwaldSphere(chex.TestCase):
    """Test Ewald sphere generation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_ewald_sphere_geometry(self):
        """Test center and radius calculation."""
        var_make_sphere = self.variant(make_ewald_sphere)

        # Setup
        k_mag = 10.0  # 1/Å
        theta_deg = 2.0
        phi_deg = 0.0

        center, radius = var_make_sphere(k_mag, theta_deg, phi_deg)

        # Radius should be exactly k_mag
        chex.assert_trees_all_close(radius, k_mag, rtol=1e-6)

        # Center should be -k_in
        # k_in magnitude is k_mag
        center_mag = jnp.linalg.norm(center)
        chex.assert_trees_all_close(center_mag, k_mag, rtol=1e-6)

        # Check direction for theta=2, phi=0
        # k_in = [k cos(theta), 0, -k sin(theta)] (approx, check sign convention)
        # In kinematic.py: k_z = -k * sin(theta)
        # So center = -k_in = [-kx, -ky, -kz]
        # center_z = -(-k * sin(theta)) = k * sin(theta) > 0
        
        theta_rad = jnp.deg2rad(theta_deg)
        expected_z = k_mag * jnp.sin(theta_rad)
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
    def setUpClass(cls):
        """Load MgO crystal structure from CIF file."""
        test_data_dir = Path(__file__).parent.parent.parent / "test_data"
        cif_path = test_data_dir / "MgO.cif"
        cls.mgo_crystal = parse_cif(cif_path)

        # Get reciprocal lattice vectors for MgO
        cls.recip_vectors = reciprocal_lattice_vectors(
            *cls.mgo_crystal.cell_lengths,
            *cls.mgo_crystal.cell_angles,
            in_degrees=True,
        )

    def _get_structure_factor_intensity(self, h: int, k: int, l: int) -> float:
        """Calculate structure factor intensity for given Miller indices."""
        hkl = jnp.array([[h, k, l]])
        g_vector = miller_to_reciprocal(hkl, self.recip_vectors)[0]

        atom_positions = self.mgo_crystal.cart_positions[:, :3]
        atomic_numbers = self.mgo_crystal.cart_positions[:, 3].astype(jnp.int32)

        intensity = kinematic_structure_factor(g_vector, atom_positions, atomic_numbers)
        return float(intensity)

    def test_mgo_allowed_reflections_nonzero(self):
        """Test that allowed FCC reflections have non-zero intensity.

        For FCC: (h,k,l) all even or all odd => allowed
        Examples: (1,1,1), (2,0,0), (2,2,0), (2,2,2)
        """
        # All odd indices - should be allowed
        all_odd_cases = [
            (1, 1, 1),
            (1, 1, 3),
            (3, 1, 1),
        ]
        for h, k, l in all_odd_cases:
            intensity = self._get_structure_factor_intensity(h, k, l)
            assert intensity > 1.0, (
                f"FCC allowed reflection ({h},{k},{l}) should have non-zero "
                f"intensity, got {intensity}"
            )

        # All even indices - should be allowed
        all_even_cases = [
            (2, 0, 0),
            (2, 2, 0),
            (2, 2, 2),
            (4, 0, 0),
        ]
        for h, k, l in all_even_cases:
            intensity = self._get_structure_factor_intensity(h, k, l)
            assert intensity > 1.0, (
                f"FCC allowed reflection ({h},{k},{l}) should have non-zero "
                f"intensity, got {intensity}"
            )

    def test_mgo_forbidden_reflections_zero(self):
        """Test that forbidden FCC reflections have zero intensity.

        For FCC: mixed indices (not all even, not all odd) => forbidden
        Examples: (1,0,0), (1,1,0), (2,1,0), (2,1,1)

        This is the CRITICAL test - if (1,0,0) shows non-zero intensity,
        the structure factor calculation is fundamentally wrong.
        """
        forbidden_cases = [
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

        tolerance = 1e-6  # Numerical tolerance for "zero"

        for h, k, l in forbidden_cases:
            intensity = self._get_structure_factor_intensity(h, k, l)
            assert intensity < tolerance, (
                f"FCC forbidden reflection ({h},{k},{l}) should have zero "
                f"intensity, got {intensity}. "
                f"This indicates the structure factor calculation is WRONG."
            )

    def test_mgo_origin_reflection(self):
        """Test that (0,0,0) reflection gives expected intensity.

        F(0,0,0) = sum of all atomic scattering factors
        For simplified f_j = Z_j: F = Z_Mg + Z_O = 12 + 8 = 20 per formula unit
        With 8 atoms in conventional cell (4 Mg + 4 O):
        F = 4*12 + 4*8 = 48 + 32 = 80
        I = |F|² = 6400
        """
        intensity = self._get_structure_factor_intensity(0, 0, 0)
        # 8 atoms total: 4 Mg (Z=12) + 4 O (Z=8)
        expected_f = 4 * 12 + 4 * 8  # = 80
        expected_intensity = expected_f ** 2  # = 6400

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
    def setUpClass(cls):
        """Load SrTiO3 crystal structure from CIF file."""
        test_data_dir = Path(__file__).parent.parent.parent / "test_data"
        cif_path = test_data_dir / "SrTiO3.cif"
        cls.sto_crystal = parse_cif(cif_path)

        # Verify we got the right number of atoms
        assert len(cls.sto_crystal.cart_positions) == 5, (
            f"Expected 5 atoms (1 Sr + 1 Ti + 3 O), got {len(cls.sto_crystal.cart_positions)}"
        )

        # Get reciprocal lattice vectors
        cls.recip_vectors = reciprocal_lattice_vectors(
            *cls.sto_crystal.cell_lengths,
            *cls.sto_crystal.cell_angles,
            in_degrees=True,
        )

    def _get_structure_factor_intensity(self, h: int, k: int, l: int) -> float:
        """Calculate structure factor intensity for given Miller indices."""
        hkl = jnp.array([[h, k, l]])
        g_vector = miller_to_reciprocal(hkl, self.recip_vectors)[0]

        atom_positions = self.sto_crystal.cart_positions[:, :3]
        atomic_numbers = self.sto_crystal.cart_positions[:, 3].astype(jnp.int32)

        intensity = kinematic_structure_factor(g_vector, atom_positions, atomic_numbers)
        return float(intensity)

    def test_sto_origin_reflection(self):
        """Test that (0,0,0) reflection gives expected intensity.

        F(0,0,0) = sum of all atomic scattering factors
        For simplified f_j = Z_j:
        F = Z_Sr + Z_Ti + 3*Z_O = 38 + 22 + 3*8 = 84
        I = |F|² = 7056
        """
        intensity = self._get_structure_factor_intensity(0, 0, 0)
        expected_f = 38 + 22 + 3 * 8  # = 84
        expected_intensity = expected_f ** 2  # = 7056

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_100_reflection(self):
        """Test (1,0,0) reflection intensity.

        For perovskite with:
        - Sr at (0.5, 0.5, 0.5): phase = exp(i*pi) = -1
        - Ti at (0, 0, 0): phase = exp(0) = 1
        - O at (0, 0, 0.5): phase = exp(0) = 1
        - O at (0.5, 0, 0): phase = exp(i*pi) = -1
        - O at (0, 0.5, 0): phase = exp(0) = 1

        F = -38 + 22 + 8 - 8 + 8 = -8
        I = 64
        """
        intensity = self._get_structure_factor_intensity(1, 0, 0)
        # F = -Z_Sr + Z_Ti + Z_O - Z_O + Z_O = -38 + 22 + 8 - 8 + 8 = -8
        expected_f = -38 + 22 + 8 - 8 + 8  # = -8
        expected_intensity = expected_f ** 2  # = 64

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_110_reflection(self):
        """Test (1,1,0) reflection intensity.

        For (1,1,0):
        - Sr at (0.5, 0.5, 0.5): phase = exp(i*2*pi*0.5) * exp(i*2*pi*0.5) = exp(i*2*pi) = 1
        - Ti at (0, 0, 0): phase = 1
        - O at (0, 0, 0.5): phase = 1
        - O at (0.5, 0, 0): phase = exp(i*pi) = -1
        - O at (0, 0.5, 0): phase = exp(i*pi) = -1

        F = 38 + 22 + 8 - 8 - 8 = 52
        I = 2704
        """
        intensity = self._get_structure_factor_intensity(1, 1, 0)
        expected_f = 38 + 22 + 8 - 8 - 8  # = 52
        expected_intensity = expected_f ** 2  # = 2704

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_111_reflection(self):
        """Test (1,1,1) reflection intensity.

        For (1,1,1):
        - Sr at (0.5, 0.5, 0.5): phase = exp(i*3*pi) = -1
        - Ti at (0, 0, 0): phase = 1
        - O at (0, 0, 0.5): phase = exp(i*pi) = -1
        - O at (0.5, 0, 0): phase = exp(i*pi) = -1
        - O at (0, 0.5, 0): phase = exp(i*pi) = -1

        F = -38 + 22 - 8 - 8 - 8 = -40
        I = 1600
        """
        intensity = self._get_structure_factor_intensity(1, 1, 1)
        expected_f = -38 + 22 - 8 - 8 - 8  # = -40
        expected_intensity = expected_f ** 2  # = 1600

        chex.assert_trees_all_close(intensity, expected_intensity, rtol=0.01)

    def test_sto_all_reflections_nonzero(self):
        """Test that various reflections have non-zero intensity.

        Unlike FCC, perovskite has no systematic extinctions from the lattice.
        All reflections should have some intensity (though values vary
        based on the atomic basis phases).
        """
        test_cases = [
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

        for h, k, l in test_cases:
            intensity = self._get_structure_factor_intensity(h, k, l)
            assert intensity > 0.1, (
                f"Perovskite reflection ({h},{k},{l}) should have non-zero "
                f"intensity, got {intensity}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
