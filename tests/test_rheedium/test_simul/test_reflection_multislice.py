"""Tests for reflection-geometry multislice RHEED simulation."""

from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout import parse_crystal
from rheedium.simul import ewald_simulator
from rheedium.simul.reflection_multislice import (
    _read_reflected_pattern,
    crystal_to_edge_on_slices,
    reflection_multislice_propagate,
    reflection_multislice_simulator,
)
from rheedium.tools import wavelength_ang
from rheedium.types import (
    CrystalStructure,
    EdgeOnSlices,
    RHEEDPattern,
    create_crystal_structure,
    create_edge_on_slices,
)

BI2SE3_DIR = Path("tests/test_data/bi2se3")


def _flat_step_slices(
    theta_deg: float = 2.0,
    *,
    vacuum_above: float = 16.0,
    cap_width: float = 8.0,
) -> EdgeOnSlices:
    """Build a compact uniform inner-potential step for reflection tests."""
    del theta_deg
    voltage_step = 15.0
    nx_slices = 16
    ny = 64
    dx_slice = 1.0
    dy = 0.25
    dz = 0.25
    z_lo = -cap_width
    z_surf = 0.0
    z_hi = z_surf + vacuum_above + cap_width
    nz = int(jnp.ceil((z_hi - z_lo) / dz))
    z_axis: Float[Array, "nz"] = z_lo + dz * jnp.arange(nz)
    potential_profile: Float[Array, "nz"] = jnp.where(
        z_axis < z_surf,
        voltage_step * dx_slice,
        0.0,
    )
    slices: Float[Array, "nx ny nz"] = jnp.tile(
        potential_profile[None, None, :],
        (nx_slices, ny, 1),
    )
    return create_edge_on_slices(
        slices=slices,
        dx_slice=dx_slice,
        dy=dy,
        dz=dz,
        y_extent=ny * dy,
        z_lo=z_lo,
        z_surf=z_surf,
        cap_width=cap_width,
    )


def _fresnel_step_reflectivity(
    theta_deg: float,
    *,
    voltage_kv: float = 30.0,
    inner_potential_v: float = 15.0,
) -> Float[Array, ""]:
    """Return the analytic electron inner-potential step reflectivity."""
    wavelength = wavelength_ang(voltage_kv)
    k_mag = 2.0 * jnp.pi / wavelength
    sin_theta = jnp.sin(jnp.deg2rad(theta_deg))
    k_perp_vac = k_mag * sin_theta
    k_perp_in = k_mag * jnp.sqrt(
        sin_theta**2 + inner_potential_v / (voltage_kv * 1000.0)
    )
    amplitude = (k_perp_vac - k_perp_in) / (k_perp_vac + k_perp_in)
    return jnp.abs(amplitude) ** 2


def _read_flat_step_pattern(
    theta_deg: float,
    *,
    vacuum_above: float = 16.0,
    cap_width: float = 8.0,
) -> RHEEDPattern:
    """Propagate and read a uniform flat-step test pattern."""
    slices: EdgeOnSlices = _flat_step_slices(
        theta_deg,
        vacuum_above=vacuum_above,
        cap_width=cap_width,
    )
    wavefield: Any = reflection_multislice_propagate(
        slices,
        voltage_kv=30.0,
        theta_deg=theta_deg,
        cap_strength=5.0,
    )
    return _read_reflected_pattern(
        wavefield=wavefield,
        slices=slices,
        voltage_kv=30.0,
        theta_deg=theta_deg,
        detector_distance=80.0,
    )


def _toy_crystal() -> CrystalStructure:
    """Create a small orthogonal slab with a top surface at positive z."""
    frac_positions: Float[Array, "N 4"] = jnp.array(
        [
            [0.25, 0.25, 0.25, 14.0],
            [0.75, 0.75, 0.50, 14.0],
        ]
    )
    cart_positions: Float[Array, "N 4"] = jnp.array(
        [
            [1.0, 1.0, 1.0, 14.0],
            [3.0, 3.0, 2.0, 14.0],
        ]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([4.0, 4.0, 4.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestReflectionMultislice(chex.TestCase, parameterized.TestCase):
    """Tests for edge-on reflection multislice routines."""

    def test_crystal_to_edge_on_slices_shapes(self) -> None:
        """Test crystal slabs convert to finite edge-on slices."""
        slices: EdgeOnSlices = crystal_to_edge_on_slices(
            _toy_crystal(),
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=4.0,
            cap_width=2.0,
            r_cutoff=2.0,
        )

        chex.assert_shape(slices.slices, (4, 8, 18))
        chex.assert_tree_all_finite(slices.slices)
        assert float(slices.z_surf) == 2.0
        assert float(slices.z_lo) == -1.0

    @parameterized.named_parameters(
        ("theta_1", 1.0),
        ("theta_2", 2.0),
        ("theta_3", 3.0),
    )
    def test_flat_slab_specular_geometry(self, theta_deg: float) -> None:
        """Test a flat step produces a specular up-going channel."""
        pattern: RHEEDPattern = _read_flat_step_pattern(theta_deg)

        brightest = int(jnp.argmax(pattern.intensities))
        wavelength = wavelength_ang(30.0)
        k_mag = 2.0 * jnp.pi / wavelength
        expected_kz = k_mag * jnp.sin(jnp.deg2rad(theta_deg))

        assert brightest == 0
        chex.assert_trees_all_close(pattern.k_out[brightest, 1], 0.0)
        chex.assert_trees_all_close(
            pattern.k_out[brightest, 2],
            expected_kz,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @parameterized.named_parameters(
        ("theta_0p5", 0.5),
        ("theta_1", 1.0),
        ("theta_2", 2.0),
        ("theta_3", 3.0),
    )
    def test_flat_slab_fresnel_reflectivity(self, theta_deg: float) -> None:
        """Test flat-step specular intensity matches Fresnel reflectivity."""
        pattern: RHEEDPattern = _read_flat_step_pattern(theta_deg)
        expected: Float[Array, ""] = _fresnel_step_reflectivity(theta_deg)

        chex.assert_trees_all_close(
            pattern.intensities[0],
            expected,
            rtol=0.15,
            atol=1e-8,
        )

    def test_flat_slab_reflectivity_increases_toward_grazing(self) -> None:
        """Test Fresnel reflectivity increases at smaller incidence angle."""
        reflectivities: Float[Array, "4"] = jnp.array(
            [
                _read_flat_step_pattern(theta).intensities[0]
                for theta in (0.5, 1.0, 2.0, 3.0)
            ]
        )

        chex.assert_trees_all_equal(
            jnp.all(jnp.diff(reflectivities) < 0), True
        )

    def test_cap_and_vacuum_convergence_for_flat_slab(self) -> None:
        """Test specular result is stable under larger vacuum and CAP."""
        base: RHEEDPattern = _read_flat_step_pattern(
            2.0,
            vacuum_above=16.0,
            cap_width=8.0,
        )
        expanded: RHEEDPattern = _read_flat_step_pattern(
            2.0,
            vacuum_above=24.0,
            cap_width=12.0,
        )

        chex.assert_trees_all_close(
            expanded.k_out[0],
            base.k_out[0],
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            expanded.intensities[0],
            base.intensities[0],
            rtol=0.05,
            atol=1e-8,
        )

    def test_reflection_multislice_simulator_returns_finite_pattern(
        self,
    ) -> None:
        """Test end-to-end atomic reflection simulation returns valid data."""
        pattern: RHEEDPattern = reflection_multislice_simulator(
            _toy_crystal(),
            voltage_kv=30.0,
            theta_deg=2.0,
            detector_distance=80.0,
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=4.0,
            cap_width=2.0,
        )

        chex.assert_shape(pattern.k_out, (8, 3))
        chex.assert_shape(pattern.detector_points, (8, 2))
        chex.assert_shape(pattern.intensities, (8,))
        chex.assert_tree_all_finite(pattern.k_out)
        chex.assert_tree_all_finite(pattern.detector_points)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    def test_bi2se3_reflection_ky_channels_match_ewald_rods(self) -> None:
        """Test strongest Bi2Se3 reflected beams lie on Ewald rod ky values."""
        crystal: CrystalStructure = parse_crystal(BI2SE3_DIR / "intial.xyz")
        reflection: RHEEDPattern = reflection_multislice_simulator(
            crystal,
            voltage_kv=30.0,
            theta_deg=2.5,
            detector_distance=80.0,
            dx_slice=4.0,
            dy=1.0,
            dz=1.0,
            vacuum_above=6.0,
            cap_width=3.0,
        )
        ewald: RHEEDPattern = ewald_simulator(
            crystal,
            voltage_kv=30.0,
            theta_deg=2.5,
            phi_deg=0.0,
            hmax=4,
            kmax=12,
            detector_distance=80.0,
        )
        strongest = jnp.argsort(reflection.intensities)[-5:]
        reflected_ky = reflection.k_out[strongest, 1]
        ewald_ky = ewald.k_out[:, 1]
        nearest = jnp.min(
            jnp.abs(reflected_ky[:, None] - ewald_ky[None, :]),
            axis=1,
        )

        assert jnp.all(nearest <= jnp.pi / crystal.cell_lengths[1])

    @parameterized.named_parameters(
        ("initial", "intial.xyz"),
        ("500K", "500K.final.xyz"),
        ("750K", "750K.final.xyz"),
        ("1000K", "1000K.final.xyz"),
        ("1250K", "1250K.final.xyz"),
    )
    def test_bi2se3_xyz_fixtures_run_with_finite_intensities(
        self,
        filename: str,
    ) -> None:
        """Test all Bi2Se3 XYZ fixtures run with finite normalized output."""
        crystal: CrystalStructure = parse_crystal(BI2SE3_DIR / filename)
        pattern: RHEEDPattern = reflection_multislice_simulator(
            crystal,
            voltage_kv=30.0,
            theta_deg=2.5,
            detector_distance=80.0,
            dx_slice=4.0,
            dy=1.5,
            dz=1.5,
            vacuum_above=4.5,
            cap_width=3.0,
        )

        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)
        chex.assert_trees_all_close(jnp.max(pattern.intensities), 1.0)
