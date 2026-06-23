"""Vectorized detector-image sweep utilities.

Extended Summary
----------------
This module contains batched RHEED detector-image helpers built as
``jax.vmap`` wrappers around
:func:`rheedium.simul.simulator.simulate_detector_image`.
The functions keep parameter scans separate from the core simulator while
preserving the public ``rheedium.simul`` API.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    CrystalStructure,
    OrientationDistribution,
    SurfaceConfig,
    scalar_float,
    scalar_int,
    scalar_num,
)

from .simulator import simulate_detector_image


@jaxtyped(typechecker=beartype)
def simulate_detector_image_phi_sweep(  # noqa: PLR0913
    crystal: CrystalStructure,
    phi_deg_values: Float[Array, "N"],
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N H W"]:
    """Simulate detector images over azimuthal angle using ``jax.vmap``."""
    phi_bank: Float[Array, "N"] = jnp.asarray(phi_deg_values)

    def _simulate_one(phi_deg: scalar_float) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    return jax.vmap(_simulate_one)(phi_bank)


@jaxtyped(typechecker=beartype)
def simulate_detector_image_orientation_sweep(  # noqa: PLR0913
    crystal: CrystalStructure,
    orientation_deg_values: Float[Array, "N"],
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N H W"]:
    """Simulate detector images over crystal orientation using ``jax.vmap``."""
    orientation_bank: Float[Array, "N"] = jnp.asarray(orientation_deg_values)

    def _simulate_one(orientation_deg: scalar_float) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=orientation_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    return jax.vmap(_simulate_one)(orientation_bank)


@jaxtyped(typechecker=beartype)
def simulate_detector_image_theta_sweep(  # noqa: PLR0913
    crystal: CrystalStructure,
    theta_deg_values: Float[Array, "N"],
    voltage_kv: scalar_num = 20.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N H W"]:
    """Simulate detector images over grazing incidence using ``jax.vmap``."""
    theta_bank: Float[Array, "N"] = jnp.asarray(theta_deg_values)

    def _simulate_one(theta_deg: scalar_float) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    return jax.vmap(_simulate_one)(theta_bank)


@jaxtyped(typechecker=beartype)
def simulate_detector_image_energy_sweep(  # noqa: PLR0913
    crystal: CrystalStructure,
    voltage_kv_values: Float[Array, "N"],
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N H W"]:
    """Simulate detector images over beam energy using ``jax.vmap``."""
    voltage_bank: Float[Array, "N"] = jnp.asarray(voltage_kv_values)

    def _simulate_one(voltage_kv: scalar_float) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    return jax.vmap(_simulate_one)(voltage_bank)


@jaxtyped(typechecker=beartype)
def simulate_detector_image_parameter_grid(  # noqa: PLR0913
    crystal: CrystalStructure,
    phi_deg_values: Float[Array, "N_phi"],
    theta_deg_values: Float[Array, "N_theta"],
    voltage_kv_values: Float[Array, "N_voltage"],
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N_phi N_theta N_voltage H W"]:
    """Simulate images over orientation, angle, and energy using vmaps."""
    phi_bank: Float[Array, "N_phi"] = jnp.asarray(phi_deg_values)
    theta_bank: Float[Array, "N_theta"] = jnp.asarray(theta_deg_values)
    voltage_bank: Float[Array, "N_voltage"] = jnp.asarray(voltage_kv_values)

    def _simulate_one(
        phi_deg: scalar_float,
        theta_deg: scalar_float,
        voltage_kv: scalar_float,
    ) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    def _simulate_voltage_axis(
        phi_deg: scalar_float,
        theta_deg: scalar_float,
    ) -> Float[Array, "N_voltage H W"]:
        return jax.vmap(
            lambda voltage: _simulate_one(phi_deg, theta_deg, voltage)
        )(voltage_bank)

    def _simulate_theta_axis(
        phi_deg: scalar_float,
    ) -> Float[Array, "N_theta N_voltage H W"]:
        return jax.vmap(lambda theta: _simulate_voltage_axis(phi_deg, theta))(
            theta_bank
        )

    return jax.vmap(_simulate_theta_axis)(phi_bank)


@jaxtyped(typechecker=beartype)
def simulate_detector_image_roughness_sweep(  # noqa: PLR0913
    crystal: CrystalStructure,
    surface_roughness_values: Float[Array, "N"],
    voltage_kv: scalar_num = 20.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
    hmax: scalar_int = 5,
    kmax: scalar_int = 5,
    detector_distance_mm: scalar_float = 1000.0,
    temperature: scalar_float = 300.0,
    ctr_regularization: scalar_float = 0.01,
    ctr_power: scalar_float = 1.0,
    roughness_power: scalar_float = 0.25,
    image_shape_px: Tuple[int, int] = (192, 192),
    pixel_size_mm: Tuple[float, float] = (1.5, 3.0),
    beam_center_px: Tuple[float, float] = (96.0, 8.0),
    spot_sigma_px: scalar_float = 1.4,
    angular_divergence_mrad: scalar_float = 0.35,
    energy_spread_ev: scalar_float = 0.35,
    psf_sigma_pixels: scalar_float = 1.2,
    n_angular_samples: int = 5,
    n_energy_samples: int = 5,
    orientation_distribution: OrientationDistribution | None = None,
    n_mosaic_points: scalar_int = 7,
    surface_config: SurfaceConfig | None = None,
    render_ctrs_as_streaks: bool = True,
) -> Float[Array, "N H W"]:
    """Simulate detector images over surface roughness using ``jax.vmap``."""
    roughness_bank: Float[Array, "N"] = jnp.asarray(surface_roughness_values)

    def _simulate_one(surface_roughness: scalar_float) -> Float[Array, "H W"]:
        return simulate_detector_image(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance_mm=detector_distance_mm,
            temperature=temperature,
            surface_roughness=surface_roughness,
            ctr_regularization=ctr_regularization,
            ctr_power=ctr_power,
            roughness_power=roughness_power,
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
            angular_divergence_mrad=angular_divergence_mrad,
            energy_spread_ev=energy_spread_ev,
            psf_sigma_pixels=psf_sigma_pixels,
            n_angular_samples=n_angular_samples,
            n_energy_samples=n_energy_samples,
            orientation_distribution=orientation_distribution,
            n_mosaic_points=n_mosaic_points,
            surface_config=surface_config,
            render_ctrs_as_streaks=render_ctrs_as_streaks,
        )

    return jax.vmap(_simulate_one)(roughness_bank)


__all__: list[str] = [
    "simulate_detector_image_energy_sweep",
    "simulate_detector_image_orientation_sweep",
    "simulate_detector_image_parameter_grid",
    "simulate_detector_image_phi_sweep",
    "simulate_detector_image_roughness_sweep",
    "simulate_detector_image_theta_sweep",
]
