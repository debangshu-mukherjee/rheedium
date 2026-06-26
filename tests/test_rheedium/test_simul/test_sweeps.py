"""Tests for vmapped detector-image sweep helpers."""

from typing import Any
from unittest.mock import patch

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from rheedium.simul.simulator import simulate_detector_image
from rheedium.simul.sweeps import (
    simulate_detector_image_grid,
    simulate_detector_image_sweep,
)
from rheedium.types import (
    BeamSpec,
    CrystalStructure,
    DetectorGeometry,
    RenderParams,
    SurfaceCTRParams,
)
from rheedium.types.custom_types import scalar_float

from ..._assertions import assert_crystal_structure_arrays
from ..._factories import make_si_crystal_2atom

_SI_CRYSTAL_2ATOM: CrystalStructure = make_si_crystal_2atom()


class TestDetectorImageSweeps(chex.TestCase):
    """Tests for dense detector-image sweep utilities."""

    def test_shared_si_crystal_fixture_valid(self) -> None:
        r"""Shared typed crystal factory returns valid sweep test data.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Shared typed
        crystal factory returns valid sweep test data.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        assert_crystal_structure_arrays(_SI_CRYSTAL_2ATOM, n_atoms=2)

    def _small_detector_config(self) -> dict[str, Any]:
        """Return fast carrier settings shared by dense-image tests."""
        return {
            "crystal": _SI_CRYSTAL_2ATOM,
            "beam": BeamSpec(
                energy_kev=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            ),
            "surface": SurfaceCTRParams(
                hmax=0,
                kmax=0,
                temperature=300.0,
                surface_roughness=0.5,
            ),
            "detector": DetectorGeometry(
                distance=1000.0,
                image_shape_px=(16, 24),
                pixel_size_mm=(6.0, 16.0),
                beam_center_px=(12.0, 2.0),
                psf_sigma_pixels=0.0,
            ),
            "render": RenderParams(
                spot_sigma_px=1.2,
                n_angular_samples=1,
                n_energy_samples=1,
                render_ctrs_as_streaks=False,
            ),
        }

    @staticmethod
    def _encoded_detector_image(**kwargs: Any) -> Float[Array, "2 3"]:
        """Encode sweep parameters into a tiny synthetic detector image."""
        beam: BeamSpec = kwargs["beam"]
        surface: SurfaceCTRParams = kwargs["surface"]
        value: scalar_float = (
            beam.phi_deg
            + 10.0 * beam.theta_deg
            + 100.0 * beam.energy_kev
            + 1000.0 * surface.surface_roughness
        )
        return jnp.full((2, 3), value, dtype=jnp.float64)

    def test_phi_sweep_maps_phi_axis(self) -> None:
        r"""Phi sweep evaluates one detector image per azimuth.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Phi sweep
        evaluates one detector image per azimuth.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
                crystal=_SI_CRYSTAL_2ATOM,
                axis=("phi_deg", jnp.array([0.0, 15.0, 30.0])),
                beam=BeamSpec(energy_kev=1.0, theta_deg=2.0),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([120.0, 135.0, 150.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_orientation_sweep_maps_phi_axis(self) -> None:
        r"""Orientation sweep evaluates one detector image per orientation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation sweep
        evaluates one detector image per orientation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
                crystal=_SI_CRYSTAL_2ATOM,
                axis=("phi_deg", jnp.array([0.0, 15.0, 30.0])),
                beam=BeamSpec(energy_kev=1.0, theta_deg=2.0),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([120.0, 135.0, 150.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_theta_sweep_maps_angle_axis(self) -> None:
        r"""Theta sweep evaluates one detector image per incidence angle.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Theta sweep
        evaluates one detector image per incidence angle.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
                crystal=_SI_CRYSTAL_2ATOM,
                axis=("theta_deg", jnp.array([1.0, 2.0, 3.0])),
                beam=BeamSpec(energy_kev=1.0, phi_deg=5.0),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([115.0, 125.0, 135.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_energy_sweep_maps_voltage_axis(self) -> None:
        r"""Energy sweep evaluates one detector image per beam voltage.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Energy sweep
        evaluates one detector image per beam voltage.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
                crystal=_SI_CRYSTAL_2ATOM,
                axis=("energy_kev", jnp.array([1.0, 2.0, 3.0])),
                beam=BeamSpec(theta_deg=2.0, phi_deg=5.0),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([125.0, 225.0, 325.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_roughness_sweep_maps_roughness_axis(self) -> None:
        r"""Roughness sweep evaluates one detector image per roughness value.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Roughness sweep
        evaluates one detector image per roughness value.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
                crystal=_SI_CRYSTAL_2ATOM,
                axis=(
                    "surface_roughness",
                    jnp.array([0.0, 0.5, 1.0]),
                ),
                beam=BeamSpec(energy_kev=1.0, theta_deg=2.0, phi_deg=5.0),
            )

        chex.assert_shape(image_bank, (3, 2, 3))
        expected: Float[Array, "3"] = jnp.array([125.0, 625.0, 1125.0])
        chex.assert_trees_all_close(image_bank[:, 0, 0], expected)

    def test_parameter_grid_orders_axes(self) -> None:
        r"""Combined grid axes are phi, theta, voltage, height, width.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Combined grid axes
        are phi, theta, voltage, height, width.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_grid: Float[Array, "..."] = simulate_detector_image_grid(
                crystal=_SI_CRYSTAL_2ATOM,
                axes=(
                    ("phi_deg", jnp.array([0.0, 5.0])),
                    ("theta_deg", jnp.array([1.0, 2.0, 3.0])),
                    ("energy_kev", jnp.array([1.0, 2.0])),
                ),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_grid, (2, 3, 2, 2, 3))
        expected: Float[Array, "2 3 2"] = (
            jnp.array([0.0, 5.0])[:, None, None]
            + 10.0 * jnp.array([1.0, 2.0, 3.0])[None, :, None]
            + 100.0 * jnp.array([1.0, 2.0])[None, None, :]
        )
        chex.assert_trees_all_close(image_grid[..., 0, 0], expected)

    def test_all_sweep_orders_axes(self) -> None:
        r"""All-sweep axes are orientation, theta, voltage, height, width.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: All-sweep axes are
        orientation, theta, voltage, height, width.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with patch(
            "rheedium.simul.sweeps.simulate_detector_image",
            side_effect=self._encoded_detector_image,
        ):
            image_grid: Float[Array, "..."] = simulate_detector_image_grid(
                crystal=_SI_CRYSTAL_2ATOM,
                axes=(
                    ("phi_deg", jnp.array([0.0, 5.0])),
                    ("theta_deg", jnp.array([1.0, 2.0, 3.0])),
                    ("energy_kev", jnp.array([1.0, 2.0])),
                ),
                surface=SurfaceCTRParams(surface_roughness=0.0),
            )

        chex.assert_shape(image_grid, (2, 3, 2, 2, 3))
        expected: Float[Array, "2 3 2"] = (
            jnp.array([0.0, 5.0])[:, None, None]
            + 10.0 * jnp.array([1.0, 2.0, 3.0])[None, :, None]
            + 100.0 * jnp.array([1.0, 2.0])[None, None, :]
        )
        chex.assert_trees_all_close(image_grid[..., 0, 0], expected)

    def test_orientation_sweep_matches_single_images(self) -> None:
        r"""Orientation sweep batches detector images over phi angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Orientation sweep
        batches detector images over phi angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        kwargs: dict[str, Any] = self._small_detector_config()
        orientations: Float[Array, "2"] = jnp.array([0.0, 15.0])

        image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
            axis=("phi_deg", orientations),
            **kwargs,
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            beam=BeamSpec(
                energy_kev=20.0,
                theta_deg=2.0,
                phi_deg=orientations[0],
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            ),
            surface=kwargs["surface"],
            detector=kwargs["detector"],
            render=kwargs["render"],
            crystal=kwargs["crystal"],
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)

    def test_theta_sweep_matches_single_images(self) -> None:
        r"""Theta sweep batches detector images over grazing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Theta sweep
        batches detector images over grazing angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        kwargs: dict[str, Any] = self._small_detector_config()
        theta_values: Float[Array, "2"] = jnp.array([1.5, 2.5])

        image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
            axis=("theta_deg", theta_values),
            **kwargs,
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            beam=BeamSpec(
                energy_kev=20.0,
                theta_deg=theta_values[0],
                phi_deg=0.0,
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            ),
            surface=kwargs["surface"],
            detector=kwargs["detector"],
            render=kwargs["render"],
            crystal=kwargs["crystal"],
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)

    def test_energy_sweep_matches_single_images(self) -> None:
        r"""Energy sweep batches detector images over beam voltages.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Energy sweep
        batches detector images over beam voltages.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_sweeps``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        kwargs: dict[str, Any] = self._small_detector_config()
        voltages: Float[Array, "2"] = jnp.array([15.0, 25.0])

        image_bank: Float[Array, "..."] = simulate_detector_image_sweep(
            axis=("energy_kev", voltages),
            **kwargs,
        )
        reference: Float[Array, "..."] = simulate_detector_image(
            beam=BeamSpec(
                energy_kev=voltages[0],
                theta_deg=2.0,
                phi_deg=0.0,
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            ),
            surface=kwargs["surface"],
            detector=kwargs["detector"],
            render=kwargs["render"],
            crystal=kwargs["crystal"],
        )

        chex.assert_shape(image_bank, (2, 16, 24))
        chex.assert_tree_all_finite(image_bank)
        self.assertTrue(jnp.all(image_bank >= 0.0))
        chex.assert_trees_all_close(image_bank[0], reference, atol=1e-10)
