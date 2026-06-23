"""Tests for reflection-geometry multislice RHEED simulation."""

from typing import Any

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

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


def _flat_step_slices(theta_deg: float = 2.0) -> EdgeOnSlices:
    """Build a compact uniform inner-potential step for reflection tests."""
    del theta_deg
    voltage_step = 15.0
    nx_slices = 16
    ny = 64
    nz = 160
    dx_slice = 1.0
    dy = 0.25
    dz = 0.25
    z_lo = -15.0
    z_surf = 0.0
    cap_width = 8.0
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
        slices: EdgeOnSlices = _flat_step_slices(theta_deg)
        wavefield: Any = reflection_multislice_propagate(
            slices,
            voltage_kv=30.0,
            theta_deg=theta_deg,
            cap_strength=5.0,
        )
        pattern: RHEEDPattern = _read_reflected_pattern(
            wavefield=wavefield,
            slices=slices,
            voltage_kv=30.0,
            theta_deg=theta_deg,
            detector_distance=80.0,
        )

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
