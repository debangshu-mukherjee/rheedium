from pathlib import Path

import jax.numpy as jnp
import numpy as np

import rheedium as rh


def _chunked_phi_bank(crystal, phi_values, settings, batch_size):
    bank_parts = []
    for start in range(0, len(phi_values), batch_size):
        stop = min(start + batch_size, len(phi_values))
        chunk = jnp.asarray(phi_values[start:stop])
        print(
            f"building Bi2Se3 phi chunk {start}:{stop} of {len(phi_values)}",
            flush=True,
        )
        chunk_bank = rh.simul.simulate_detector_image_phi_sweep(
            crystal=crystal,
            phi_deg_values=chunk,
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
        bank_parts.append(np.asarray(chunk_bank))
    return np.concatenate(bank_parts, axis=0)


def _chunked_roughness_bank(crystal, roughness_values, settings, batch_size):
    bank_parts = []
    for start in range(0, len(roughness_values), batch_size):
        stop = min(start + batch_size, len(roughness_values))
        chunk = jnp.asarray(roughness_values[start:stop])
        print(
            "building Bi2Se3 roughness chunk "
            f"{start}:{stop} of {len(roughness_values)}",
            flush=True,
        )
        chunk_bank = rh.simul.simulate_detector_image_roughness_sweep(
            crystal=crystal,
            surface_roughness_values=chunk,
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
        bank_parts.append(np.asarray(chunk_bank))
    return np.concatenate(bank_parts, axis=0)


def _compute_dynamic_range_floor(crystal, settings):
    sparse_pattern = rh.simul.ewald_simulator(
        crystal=crystal,
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance=settings["detector_distance_mm"],
        temperature=settings["temperature"],
        surface_roughness=settings["surface_roughness"],
        ctr_regularization=settings["ctr_regularization"],
        ctr_power=settings["ctr_power"],
        roughness_power=settings["roughness_power"],
    )
    detector_image = rh.simul.simulate_detector_image(
        crystal=crystal,
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
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
    valid = np.asarray(sparse_pattern.G_indices) >= 0
    bragg_points_mm = np.asarray(sparse_pattern.detector_points)[valid]
    x_px = np.rint(
        settings["beam_center_px"][0]
        + bragg_points_mm[:, 0] / settings["pixel_size_mm"][0]
    ).astype(int)
    y_px = np.rint(
        settings["beam_center_px"][1]
        + bragg_points_mm[:, 1] / settings["pixel_size_mm"][1]
    ).astype(int)
    x_px = np.clip(x_px, 0, settings["image_shape_px"][1] - 1)
    y_px = np.clip(y_px, 0, settings["image_shape_px"][0] - 1)
    sampled = np.asarray(detector_image)[y_px, x_px]
    positive = sampled[sampled > 0.0]
    faintest = float(positive.min()) if len(positive) else 0.0
    return settings["dynamic_range_scale"] * faintest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sweeps_dir = repo_root / "tutorials" / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    crystal = rh.inout.parse_cif(
        repo_root / "tests" / "test_data" / "bi2se3" / "Bi2Se3.cif"
    )
    settings = {
        "material": "Bi2Se3",
        "voltage_kv": 30.0,
        "theta_deg": 2.5,
        "phi_deg": 0.0,
        "hmax": 3,
        "kmax": 3,
        "detector_distance_mm": 80.0,
        "temperature": 300.0,
        "surface_roughness": 0.0,
        "ctr_regularization": 0.01,
        "ctr_power": 1.0,
        "roughness_power": 0.25,
        "image_shape_px": (300, 300),
        "pixel_size_mm": (0.8, 0.8),
        "beam_center_px": (150.0, 0.0),
        "spot_sigma_px": 1.1,
        "angular_divergence_mrad": 0.35,
        "energy_spread_ev": 0.35,
        "psf_sigma_pixels": 1.0,
        "n_angular_samples": 1,
        "n_energy_samples": 1,
        "log_gain": 22.0,
        "dynamic_range_scale": 0.8,
    }
    dynamic_range_floor = _compute_dynamic_range_floor(crystal, settings)
    extent_mm = np.asarray(
        rh.simul.detector_extent_mm(
            image_shape_px=settings["image_shape_px"],
            pixel_size_mm=settings["pixel_size_mm"],
            beam_center_px=settings["beam_center_px"],
        )
    )

    phi_values = np.linspace(0.0, 45.0, 10)
    phi_bank = _chunked_phi_bank(
        crystal=crystal,
        phi_values=phi_values,
        settings=settings,
        batch_size=2,
    )
    phi_display_bank = np.asarray(
        [
            rh.simul.log_compress_image(
                jnp.asarray(image),
                gain=settings["log_gain"],
                dynamic_range_floor=dynamic_range_floor,
            )
            for image in phi_bank
        ]
    )
    np.savez_compressed(
        sweeps_dir / "bi2se3_theta2p5_phi_sweep.npz",
        image_bank=phi_display_bank,
        parameter_values=phi_values,
        parameter_name="phi_deg",
        title_prefix="phi",
        material=settings["material"],
        extent_mm=extent_mm,
        xlim=np.asarray([-120.0, 120.0]),
        ylim=np.asarray([0.0, 120.0]),
        theta_deg=settings["theta_deg"],
        voltage_kv=settings["voltage_kv"],
        dynamic_range_floor=dynamic_range_floor,
        log_gain=settings["log_gain"],
    )

    roughness_values = np.asarray([0.0, 0.25, 0.5, 1.0])
    roughness_bank = _chunked_roughness_bank(
        crystal=crystal,
        roughness_values=roughness_values,
        settings=settings,
        batch_size=2,
    )
    roughness_display_bank = np.asarray(
        [
            rh.simul.log_compress_image(
                jnp.asarray(image),
                gain=settings["log_gain"],
                dynamic_range_floor=dynamic_range_floor,
            )
            for image in roughness_bank
        ]
    )
    np.savez_compressed(
        sweeps_dir / "bi2se3_theta2p5_roughness_sweep.npz",
        image_bank=roughness_display_bank,
        parameter_values=roughness_values,
        parameter_name="surface_roughness",
        title_prefix="surface roughness",
        material=settings["material"],
        extent_mm=extent_mm,
        xlim=np.asarray([-120.0, 120.0]),
        ylim=np.asarray([0.0, 120.0]),
        phi_deg=settings["phi_deg"],
        theta_deg=settings["theta_deg"],
        voltage_kv=settings["voltage_kv"],
        dynamic_range_floor=dynamic_range_floor,
        log_gain=settings["log_gain"],
    )

    print(sweeps_dir / "bi2se3_theta2p5_phi_sweep.npz")
    print(sweeps_dir / "bi2se3_theta2p5_roughness_sweep.npz")


if __name__ == "__main__":
    main()
