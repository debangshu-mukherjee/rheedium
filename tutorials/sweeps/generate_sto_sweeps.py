from pathlib import Path

import jax.numpy as jnp
import numpy as np

import rheedium as rh


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sweeps_dir = repo_root / "tutorials" / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    crystal = rh.inout.parse_cif(
        repo_root / "tests" / "test_data" / "SrTiO3.cif"
    )

    settings = {
        "voltage_kv": 18.0,
        "theta_deg": 4.0,
        "phi_deg": 0.0,
        "hmax": 14,
        "kmax": 14,
        "detector_distance_mm": 900.0,
        "temperature": 300.0,
        "surface_roughness": 0.0,
        "ctr_regularization": 0.01,
        "ctr_power": 1.0,
        "roughness_power": 0.25,
        "image_shape_px": (300, 300),
        "pixel_size_mm": (2.16, 2.16),
        "beam_center_px": (150.0, 0.0),
        "spot_sigma_px": 1.1,
        "angular_divergence_mrad": 0.35,
        "energy_spread_ev": 0.35,
        "psf_sigma_pixels": 1.0,
        "n_angular_samples": 5,
        "n_energy_samples": 3,
        "log_gain": 22.0,
        "dynamic_range_floor": 1.3001876993458826e-05,
    }

    extent_mm = np.asarray(
        rh.simul.detector_extent_mm(
            image_shape_px=settings["image_shape_px"],
            pixel_size_mm=settings["pixel_size_mm"],
            beam_center_px=settings["beam_center_px"],
        )
    )

    phi_values = np.linspace(0.0, 45.0, 10)
    phi_bank = rh.simul.simulate_detector_image_phi_sweep(
        crystal=crystal,
        phi_deg_values=jnp.asarray(phi_values),
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance_mm=settings["detector_distance_mm"],
        temperature=settings["temperature"],
        surface_roughness=settings["surface_roughness"],
        ctr_regularization=settings["ctr_regularization"],
        ctr_power=settings["ctr_power"],
        roughness_power=settings["roughness_power"],
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
        angular_divergence_mrad=settings["angular_divergence_mrad"],
        energy_spread_ev=settings["energy_spread_ev"],
        psf_sigma_pixels=settings["psf_sigma_pixels"],
        n_angular_samples=settings["n_angular_samples"],
        n_energy_samples=settings["n_energy_samples"],
        render_ctrs_as_streaks=True,
    )
    phi_display_bank = np.asarray(
        [
            rh.simul.log_compress_image(
                image,
                gain=settings["log_gain"],
                dynamic_range_floor=settings["dynamic_range_floor"],
            )
            for image in np.asarray(phi_bank)
        ]
    )
    np.savez_compressed(
        sweeps_dir / "sto_theta4_phi_sweep.npz",
        image_bank=phi_display_bank,
        parameter_values=phi_values,
        parameter_name="phi_deg",
        title_prefix="phi",
        extent_mm=extent_mm,
        xlim=np.asarray([-300.0, 300.0]),
        ylim=np.asarray([0.0, 300.0]),
        theta_deg=settings["theta_deg"],
        voltage_kv=settings["voltage_kv"],
        dynamic_range_floor=settings["dynamic_range_floor"],
        log_gain=settings["log_gain"],
    )

    roughness_values = np.asarray([0.0, 0.25, 0.5, 1.0])
    roughness_bank = rh.simul.simulate_detector_image_roughness_sweep(
        crystal=crystal,
        surface_roughness_values=jnp.asarray(roughness_values),
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance_mm=settings["detector_distance_mm"],
        temperature=settings["temperature"],
        ctr_regularization=settings["ctr_regularization"],
        ctr_power=settings["ctr_power"],
        roughness_power=settings["roughness_power"],
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
        angular_divergence_mrad=settings["angular_divergence_mrad"],
        energy_spread_ev=settings["energy_spread_ev"],
        psf_sigma_pixels=settings["psf_sigma_pixels"],
        n_angular_samples=settings["n_angular_samples"],
        n_energy_samples=settings["n_energy_samples"],
        render_ctrs_as_streaks=True,
    )
    roughness_display_bank = np.asarray(
        [
            rh.simul.log_compress_image(
                image,
                gain=settings["log_gain"],
                dynamic_range_floor=settings["dynamic_range_floor"],
            )
            for image in np.asarray(roughness_bank)
        ]
    )
    np.savez_compressed(
        sweeps_dir / "sto_theta4_roughness_sweep.npz",
        image_bank=roughness_display_bank,
        parameter_values=roughness_values,
        parameter_name="surface_roughness",
        title_prefix="surface roughness",
        extent_mm=extent_mm,
        xlim=np.asarray([-300.0, 300.0]),
        ylim=np.asarray([0.0, 300.0]),
        phi_deg=settings["phi_deg"],
        theta_deg=settings["theta_deg"],
        voltage_kv=settings["voltage_kv"],
        dynamic_range_floor=settings["dynamic_range_floor"],
        log_gain=settings["log_gain"],
    )

    print(sweeps_dir / "sto_theta4_phi_sweep.npz")
    print(sweeps_dir / "sto_theta4_roughness_sweep.npz")


if __name__ == "__main__":
    main()
