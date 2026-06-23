"""Tests for vmapped detector-image sweep helpers."""

from typing import Any
from unittest.mock import patch

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.simul.simulator import simulate_detector_image
from rheedium.simul.sweeps import (
    simulate_detector_image_all_sweep,
    simulate_detector_image_energy_sweep,
    simulate_detector_image_orientation_sweep,
    simulate_detector_image_parameter_grid,
    simulate_detector_image_phi_sweep,
    simulate_detector_image_roughness_sweep,
    simulate_detector_image_theta_sweep,
)
from rheedium.types import CrystalStructure
from rheedium.types.custom_types import scalar_float

from ..._assertions import assert_crystal_structure_arrays
from ..._factories import make_si_crystal_2atom

_SI_CRYSTAL_2ATOM: CrystalStructure = make_si_crystal_2atom()


class TestDetectorImageSweeps(chex.TestCase):
    """Tests for dense detector-image sweep utilities."""

    def test_shared_si_crystal_fixture_valid(self) -> None:
        """Shared typed crystal factory returns valid sweep test data."""
        assert_crystal_structure_arrays(_SI_CRYSTAL_2ATOM, n_atoms=2)

    def _small_detector_kwargs(self) -> dict[str, Any]:
        """Return fast detector settings shared by dense-image tests."""
        return {
            "crystal": _SI_CRYSTAL_2ATOM,
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

    @staticmethod
    def _encoded_detector_image(**kwargs: Any) -> Float[Array, "2 3"]:
        """Encode sweep parameters into a tiny synthetic detector image."""
        value: scalar_float = (
            kwargs["phi_deg"]
            + 10.0 * kwargs["theta_deg"]
            + 100.0 * kwargs["voltage_kv"]
            + 1000.0 * kwargs["surface_roughness"]
        )
        return jnp.full((2, 3), value, dtype=jnp.float64)

    def test_phi_sweep_maps_phi_axis(self) -> None:
        """Phi sweep evaluates one detector image per azimuth."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = (
                simulate_detector_image_phi_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    phi_deg_values=jnp.array([0.0, 15.0, 30.0]),
                    voltage_kv=1.0,
                    theta_deg=2.0,
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([120.0, 135.0, 150.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_orientation_sweep_maps_phi_axis(self) -> None:
        """Orientation sweep evaluates one detector image per orientation."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = (
                simulate_detector_image_orientation_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    orientation_deg_values=jnp.array([0.0, 15.0, 30.0]),
                    voltage_kv=1.0,
                    theta_deg=2.0,
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([120.0, 135.0, 150.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_theta_sweep_maps_angle_axis(self) -> None:
        """Theta sweep evaluates one detector image per incidence angle."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = (
                simulate_detector_image_theta_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    theta_deg_values=jnp.array([1.0, 2.0, 3.0]),
                    voltage_kv=1.0,
                    phi_deg=5.0,
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([115.0, 125.0, 135.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_energy_sweep_maps_voltage_axis(self) -> None:
        """Energy sweep evaluates one detector image per beam voltage."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = (
                simulate_detector_image_energy_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    voltage_kv_values=jnp.array([1.0, 2.0, 3.0]),
                    theta_deg=2.0,
                    phi_deg=5.0,
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([125.0, 225.0, 325.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_roughness_sweep_maps_roughness_axis(self) -> None:
        """Roughness sweep evaluates one detector image per roughness value."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = (
                simulate_detector_image_roughness_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    surface_roughness_values=jnp.array([0.0, 0.5, 1.0]),
                    voltage_kv=1.0,
                    theta_deg=2.0,
                    phi_deg=5.0,
                )
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([125.0, 625.0, 1125.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_parameter_grid_orders_axes(self) -> None:
        """Combined grid axes are phi, theta, voltage, height, width."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_grid: Float[Array, "..."] = (
                simulate_detector_image_parameter_grid(
                    crystal=_SI_CRYSTAL_2ATOM,
                    phi_deg_values=jnp.array([0.0, 5.0]),
                    theta_deg_values=jnp.array([1.0, 2.0, 3.0]),
                    voltage_kv_values=jnp.array([1.0, 2.0]),
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_grid, (2, 3, 2, 2, 3))
        expected: Float[Array, "2 3 2"] = (
            jnp.array([0.0, 5.0])[:, None, None]
            + 10.0 * jnp.array([1.0, 2.0, 3.0])[None, :, None]
            + 100.0 * jnp.array([1.0, 2.0])[None, None, :]
        )
        chex.assert_trees_all_close(image_grid[..., 0, 0], expected)

    def test_all_sweep_orders_axes(self) -> None:
        """All-sweep axes are orientation, theta, voltage, height, width."""
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_grid: Float[Array, "..."] = (
                simulate_detector_image_all_sweep(
                    crystal=_SI_CRYSTAL_2ATOM,
                    orientation_deg_values=jnp.array([0.0, 5.0]),
                    theta_deg_values=jnp.array([1.0, 2.0, 3.0]),
                    voltage_kv_values=jnp.array([1.0, 2.0]),
                    surface_roughness=0.0,
                )
            )

        chex.assert_shape(image_grid, (2, 3, 2, 2, 3))
        expected: Float[Array, "2 3 2"] = (
            jnp.array([0.0, 5.0])[:, None, None]
            + 10.0 * jnp.array([1.0, 2.0, 3.0])[None, :, None]
            + 100.0 * jnp.array([1.0, 2.0])[None, None, :]
        )
        chex.assert_trees_all_close(image_grid[..., 0, 0], expected)

    def test_orientation_sweep_matches_single_images(self) -> None:
        """Orientation sweep batches detector images over phi angles."""
        kwargs: dict[str, Any] = self._small_detector_kwargs()
        orientations: Float[Array, "2"] = jnp.array([0.0, 15.0])

        image_bank: Float[Array, "..."] = (
            simulate_detector_image_orientation_sweep(
                orientation_deg_values=orientations,
                voltage_kv=20.0,
                theta_deg=2.0,
                **kwargs,
            )
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=orientations[0],
            **kwargs,
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)

    def test_theta_sweep_matches_single_images(self) -> None:
        """Theta sweep batches detector images over grazing angles."""
        kwargs: dict[str, Any] = self._small_detector_kwargs()
        theta_values: Float[Array, "2"] = jnp.array([1.5, 2.5])

        image_bank: Float[Array, "..."] = simulate_detector_image_theta_sweep(
            theta_deg_values=theta_values,
            voltage_kv=20.0,
            phi_deg=0.0,
            **kwargs,
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            voltage_kv=20.0,
            theta_deg=theta_values[0],
            phi_deg=0.0,
            **kwargs,
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)

    def test_energy_sweep_matches_single_images(self) -> None:
        """Energy sweep batches detector images over beam voltages."""
        kwargs: dict[str, Any] = self._small_detector_kwargs()
        voltages: Float[Array, "2"] = jnp.array([15.0, 25.0])

        image_bank: Float[Array, "..."] = simulate_detector_image_energy_sweep(
            voltage_kv_values=voltages,
            theta_deg=2.0,
            phi_deg=0.0,
            **kwargs,
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            voltage_kv=voltages[0],
            theta_deg=2.0,
            phi_deg=0.0,
            **kwargs,
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)
