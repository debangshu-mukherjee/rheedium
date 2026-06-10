"""Generate synthetic reference images for the audit benchmark bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rheedium.audit import (
    ReferenceMetadata,
    dominant_peak_positions,
    extract_streak_profile,
    peak_centroid,
    simulate_detector_image_from_metadata,
    streak_fwhm_px,
)

_REFERENCE_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _REFERENCE_DIR.parents[3]

_BASE_CASES: tuple[ReferenceMetadata, ...] = (
    ReferenceMetadata(
        reference_id="mgo001_synthetic",
        material="MgO",
        surface="(001)",
        surface_normal_hkl=(0, 0, 1),
        source_kind="synthetic_detector_reference",
        simulation_mode="ewald",
        cif_path="tests/test_data/MgO.cif",
        image_path="mgo001_reference.npz",
        image_shape_px=(192, 192),
        pixel_size_mm=(1.5, 3.0),
        detector_pixel_pitch_mm=(1.5, 3.0),
        beam_center_px=(96.0, 8.0),
        specular_position_px=(0.0, 0.0),
        voltage_kv=20.0,
        theta_deg=2.2,
        phi_deg=0.0,
        hmax=5,
        kmax=5,
        detector_distance_mm=1000.0,
        camera_length_mm=1000.0,
        temperature_kelvin=300.0,
        surface_roughness_angstrom=0.45,
        angular_divergence_mrad=0.35,
        energy_spread_ev=0.35,
        psf_sigma_pixels=1.2,
        spot_sigma_px=1.4,
        n_angular_samples=5,
        n_energy_samples=5,
        n_metric_peaks=3,
        peak_min_separation_px=8,
        profile_band_half_width_px=2,
        notes="Synthetic MgO detector image for audit-pipeline validation.",
    ),
    ReferenceMetadata(
        reference_id="srtio3_001_synthetic",
        material="SrTiO3",
        surface="(001)",
        surface_normal_hkl=(0, 0, 1),
        source_kind="synthetic_detector_reference",
        simulation_mode="ewald",
        cif_path="tests/test_data/SrTiO3.cif",
        image_path="srtio3_001_reference.npz",
        image_shape_px=(208, 224),
        pixel_size_mm=(6.0, 5.0),
        detector_pixel_pitch_mm=(6.0, 5.0),
        beam_center_px=(12.0, 8.0),
        specular_position_px=(0.0, 0.0),
        voltage_kv=18.0,
        theta_deg=1.6,
        phi_deg=45.0,
        hmax=5,
        kmax=5,
        detector_distance_mm=900.0,
        camera_length_mm=900.0,
        temperature_kelvin=300.0,
        surface_roughness_angstrom=0.55,
        angular_divergence_mrad=0.45,
        energy_spread_ev=0.40,
        psf_sigma_pixels=1.4,
        spot_sigma_px=1.6,
        n_angular_samples=5,
        n_energy_samples=5,
        n_metric_peaks=3,
        peak_min_separation_px=8,
        profile_band_half_width_px=2,
        notes="Synthetic SrTiO3 detector image for audit-pipeline validation.",
    ),
)


def _write_case(case: ReferenceMetadata) -> None:
    """Generate, annotate, and write one synthetic reference case."""
    image = simulate_detector_image_from_metadata(case, repo_root=_REPO_ROOT)
    image_array = np.asarray(image, dtype=np.float64)
    specular_position = np.asarray(
        peak_centroid(jnp.asarray(image_array)), dtype=np.float64
    )
    peak_positions = np.asarray(
        dominant_peak_positions(
            jnp.asarray(image_array),
            axis="horizontal",
            n_peaks=case.n_metric_peaks,
            min_separation_px=case.peak_min_separation_px,
        ),
        dtype=np.float64,
    )
    streak_profile = extract_streak_profile(
        jnp.asarray(image_array),
        jnp.asarray(specular_position),
        axis="vertical",
        band_half_width_px=case.profile_band_half_width_px,
    )
    metadata = case.with_updates(
        specular_position_px=tuple(float(v) for v in specular_position),
        dominant_peak_positions_px=tuple(float(v) for v in peak_positions),
        reference_streak_fwhm_px=float(streak_fwhm_px(streak_profile)),
    )

    image_path = _REFERENCE_DIR / case.image_path
    np.savez_compressed(image_path, image=image_array)

    metadata_path = _REFERENCE_DIR / f"{case.reference_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")


def main() -> None:
    """Generate all synthetic audit reference fixtures."""
    _REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for case in _BASE_CASES:
        _write_case(case)


if __name__ == "__main__":
    main()
