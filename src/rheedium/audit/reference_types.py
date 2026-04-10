"""Typed containers for audit reference cases and benchmark results.

Extended Summary
----------------
This module defines the structured objects used by the audit benchmark
pipeline. They replace loose metadata dictionaries with explicit,
serializable dataclasses so the public API is easier to reason about and
extend.

Routine Listings
----------------
:class:`BenchmarkCaseResult`
    Quantitative comparison for one reference case.
:class:`BenchmarkSuiteResult`
    Aggregate benchmark summary across all reference cases.
:class:`ReferenceCase`
    One stored reference image and its metadata.
:class:`ReferenceMetadata`
    Simulation and detector metadata for one reference image.
:obj:`REQUIRED_REFERENCE_METADATA_KEYS`
    Required JSON metadata keys for persisted reference cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from jaxtyping import Float
from numpy import ndarray as NDArray  # noqa: N812

REQUIRED_REFERENCE_METADATA_KEYS: tuple[str, ...] = (
    "reference_id",
    "material",
    "surface",
    "surface_normal_hkl",
    "source_kind",
    "simulation_mode",
    "cif_path",
    "image_path",
    "image_shape_px",
    "pixel_size_mm",
    "detector_pixel_pitch_mm",
    "beam_center_px",
    "specular_position_px",
    "voltage_kv",
    "theta_deg",
    "phi_deg",
    "hmax",
    "kmax",
    "detector_distance_mm",
    "camera_length_mm",
    "temperature_kelvin",
    "surface_roughness_angstrom",
    "angular_divergence_mrad",
    "energy_spread_ev",
    "psf_sigma_pixels",
    "spot_sigma_px",
)

_SUMMARY_CANONICAL_ABS_TOL: float = 1e-10


def _canonicalize_summary_float(value: float) -> float:
    """Snap machine-noise values to exact metric boundaries for JSON output."""
    float_value = float(value)
    if abs(float_value) <= _SUMMARY_CANONICAL_ABS_TOL:
        return 0.0
    if abs(float_value - 1.0) <= _SUMMARY_CANONICAL_ABS_TOL:
        return 1.0
    if abs(float_value + 1.0) <= _SUMMARY_CANONICAL_ABS_TOL:
        return -1.0
    return float_value


@dataclass(frozen=True)
class ReferenceMetadata:
    """Structured metadata for one audit reference image."""

    reference_id: str
    material: str
    surface: str
    surface_normal_hkl: tuple[int, int, int]
    source_kind: str
    simulation_mode: str
    cif_path: str
    image_path: str
    image_shape_px: tuple[int, int]
    pixel_size_mm: tuple[float, float]
    detector_pixel_pitch_mm: tuple[float, float]
    beam_center_px: tuple[float, float]
    specular_position_px: tuple[float, float]
    voltage_kv: float
    theta_deg: float
    phi_deg: float
    hmax: int
    kmax: int
    detector_distance_mm: float
    camera_length_mm: float
    temperature_kelvin: float
    surface_roughness_angstrom: float
    angular_divergence_mrad: float
    energy_spread_ev: float
    psf_sigma_pixels: float
    spot_sigma_px: float
    n_angular_samples: int = 5
    n_energy_samples: int = 5
    n_metric_peaks: int = 3
    peak_min_separation_px: int = 5
    profile_band_half_width_px: int = 2
    dominant_peak_positions_px: tuple[float, ...] = ()
    reference_streak_fwhm_px: float | None = None
    notes: str = ""
    metadata_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceMetadata":
        """Create metadata from a JSON-compatible mapping."""
        missing_keys = sorted(
            key for key in REQUIRED_REFERENCE_METADATA_KEYS if key not in data
        )
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"Missing required reference metadata keys: {missing}"
            )

        return cls(
            reference_id=str(data["reference_id"]),
            material=str(data["material"]),
            surface=str(data["surface"]),
            surface_normal_hkl=tuple(
                int(v) for v in data["surface_normal_hkl"]
            ),
            source_kind=str(data["source_kind"]),
            simulation_mode=str(data["simulation_mode"]),
            cif_path=str(data["cif_path"]),
            image_path=str(data["image_path"]),
            image_shape_px=tuple(int(v) for v in data["image_shape_px"]),
            pixel_size_mm=tuple(float(v) for v in data["pixel_size_mm"]),
            detector_pixel_pitch_mm=tuple(
                float(v) for v in data["detector_pixel_pitch_mm"]
            ),
            beam_center_px=tuple(float(v) for v in data["beam_center_px"]),
            specular_position_px=tuple(
                float(v) for v in data["specular_position_px"]
            ),
            voltage_kv=float(data["voltage_kv"]),
            theta_deg=float(data["theta_deg"]),
            phi_deg=float(data["phi_deg"]),
            hmax=int(data["hmax"]),
            kmax=int(data["kmax"]),
            detector_distance_mm=float(data["detector_distance_mm"]),
            camera_length_mm=float(data["camera_length_mm"]),
            temperature_kelvin=float(data["temperature_kelvin"]),
            surface_roughness_angstrom=float(
                data["surface_roughness_angstrom"]
            ),
            angular_divergence_mrad=float(data["angular_divergence_mrad"]),
            energy_spread_ev=float(data["energy_spread_ev"]),
            psf_sigma_pixels=float(data["psf_sigma_pixels"]),
            spot_sigma_px=float(data["spot_sigma_px"]),
            n_angular_samples=int(data.get("n_angular_samples", 5)),
            n_energy_samples=int(data.get("n_energy_samples", 5)),
            n_metric_peaks=int(data.get("n_metric_peaks", 3)),
            peak_min_separation_px=int(data.get("peak_min_separation_px", 5)),
            profile_band_half_width_px=int(
                data.get("profile_band_half_width_px", 2)
            ),
            dominant_peak_positions_px=tuple(
                float(v) for v in data.get("dominant_peak_positions_px", [])
            ),
            reference_streak_fwhm_px=(
                None
                if data.get("reference_streak_fwhm_px") is None
                else float(data["reference_streak_fwhm_px"])
            ),
            notes=str(data.get("notes", "")),
            metadata_path=(
                None
                if data.get("metadata_path") is None
                else str(data["metadata_path"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "reference_id": self.reference_id,
            "material": self.material,
            "surface": self.surface,
            "surface_normal_hkl": list(self.surface_normal_hkl),
            "source_kind": self.source_kind,
            "simulation_mode": self.simulation_mode,
            "cif_path": self.cif_path,
            "image_path": self.image_path,
            "image_shape_px": list(self.image_shape_px),
            "pixel_size_mm": list(self.pixel_size_mm),
            "detector_pixel_pitch_mm": list(self.detector_pixel_pitch_mm),
            "beam_center_px": list(self.beam_center_px),
            "specular_position_px": list(self.specular_position_px),
            "voltage_kv": self.voltage_kv,
            "theta_deg": self.theta_deg,
            "phi_deg": self.phi_deg,
            "hmax": self.hmax,
            "kmax": self.kmax,
            "detector_distance_mm": self.detector_distance_mm,
            "camera_length_mm": self.camera_length_mm,
            "temperature_kelvin": self.temperature_kelvin,
            "surface_roughness_angstrom": self.surface_roughness_angstrom,
            "angular_divergence_mrad": self.angular_divergence_mrad,
            "energy_spread_ev": self.energy_spread_ev,
            "psf_sigma_pixels": self.psf_sigma_pixels,
            "spot_sigma_px": self.spot_sigma_px,
            "n_angular_samples": self.n_angular_samples,
            "n_energy_samples": self.n_energy_samples,
            "n_metric_peaks": self.n_metric_peaks,
            "peak_min_separation_px": self.peak_min_separation_px,
            "profile_band_half_width_px": self.profile_band_half_width_px,
            "notes": self.notes,
        }
        if self.dominant_peak_positions_px:
            payload["dominant_peak_positions_px"] = list(
                self.dominant_peak_positions_px
            )
        if self.reference_streak_fwhm_px is not None:
            payload["reference_streak_fwhm_px"] = self.reference_streak_fwhm_px
        if self.metadata_path is not None:
            payload["metadata_path"] = self.metadata_path
        return payload

    def with_updates(self, **changes: Any) -> "ReferenceMetadata":
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class ReferenceCase:
    """One loaded reference image and its metadata."""

    metadata: ReferenceMetadata
    image: Float[NDArray, "H W"]


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Quantitative benchmark result for a single reference case."""

    reference_id: str
    material: str
    surface: str
    source_kind: str
    simulation_mode: str
    normalized_cross_correlation: float
    specular_offset_px: float
    peak_centroid_error_px: float
    rod_spacing_error_px: float
    reference_streak_fwhm_px: float
    simulated_streak_fwhm_px: float
    streak_fwhm_abs_error_px: float

    def to_dict(self) -> dict[str, Any]:
        """Convert one benchmark result to a JSON-compatible mapping."""
        return {
            "reference_id": self.reference_id,
            "material": self.material,
            "surface": self.surface,
            "source_kind": self.source_kind,
            "simulation_mode": self.simulation_mode,
            "normalized_cross_correlation": _canonicalize_summary_float(
                self.normalized_cross_correlation
            ),
            "specular_offset_px": _canonicalize_summary_float(
                self.specular_offset_px
            ),
            "peak_centroid_error_px": _canonicalize_summary_float(
                self.peak_centroid_error_px
            ),
            "rod_spacing_error_px": _canonicalize_summary_float(
                self.rod_spacing_error_px
            ),
            "reference_streak_fwhm_px": _canonicalize_summary_float(
                self.reference_streak_fwhm_px
            ),
            "simulated_streak_fwhm_px": _canonicalize_summary_float(
                self.simulated_streak_fwhm_px
            ),
            "streak_fwhm_abs_error_px": _canonicalize_summary_float(
                self.streak_fwhm_abs_error_px
            ),
        }


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    """Aggregate benchmark result across all reference cases."""

    reference_count: int
    mean_normalized_cross_correlation: float
    mean_specular_offset_px: float
    max_streak_fwhm_abs_error_px: float
    cases: tuple[BenchmarkCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the suite result to a JSON-compatible mapping."""
        return {
            "reference_count": self.reference_count,
            "mean_normalized_cross_correlation": (
                _canonicalize_summary_float(
                    self.mean_normalized_cross_correlation
                )
            ),
            "mean_specular_offset_px": _canonicalize_summary_float(
                self.mean_specular_offset_px
            ),
            "max_streak_fwhm_abs_error_px": _canonicalize_summary_float(
                self.max_streak_fwhm_abs_error_px
            ),
            "cases": [case.to_dict() for case in self.cases],
        }


__all__: list[str] = [
    "BenchmarkCaseResult",
    "BenchmarkSuiteResult",
    "ReferenceCase",
    "ReferenceMetadata",
    "REQUIRED_REFERENCE_METADATA_KEYS",
]
