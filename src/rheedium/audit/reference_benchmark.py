"""Reference-bundle benchmarking for detector-image realism audits.

Extended Summary
----------------
This module loads stored reference detector images, regenerates the same
simulation conditions, and reports a compact JSON-friendly summary of
agreement metrics. The shipped reference bundle is synthetic for now,
which validates the benchmark mechanics before calibrated experimental
images are introduced.

Routine Listings
----------------
:func:`benchmark_reference_case`
    Compare one reference case against a regenerated simulation.
:func:`benchmark_reference_suite`
    Run the full reference bundle and summarize the results.
:func:`load_reference_cases`
    Load stored reference images and metadata from disk.
:func:`render_pattern_to_image`
    Rasterize sparse detector points onto a pixel grid.
:func:`simulate_detector_image_from_metadata`
    Regenerate a reference detector image from stored metadata.

Notes
-----
The benchmark runner intentionally uses the current public simulation
path. That makes it useful as a regression tool for future realism work:
if detector rendering, broadening, or geometry calibration changes, the
summary will move in a measurable way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Float
from numpy import ndarray as NDArray  # noqa: N812

from rheedium.inout import parse_cif
from rheedium.simul import (
    detector_psf_convolve,
    ewald_simulator,
)
from rheedium.tools import gauss_hermite_nodes_weights

from .metrics import (
    dominant_peak_positions,
    extract_streak_profile,
    normalized_cross_correlation,
    peak_centroid,
    peak_centroid_error_px,
    rod_spacing_error_px,
    specular_offset_px,
    streak_fwhm_px,
)
from .reference_types import (
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ReferenceCase,
    ReferenceMetadata,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_DEFAULT_REFERENCE_DIR: Path = (
    _REPO_ROOT / "tests" / "test_data" / "reference_data" / "synthetic"
)


@beartype
def _resolve_path(root: Path, maybe_relative_path: str) -> Path:
    """Resolve a metadata path relative to the repository root."""
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return path
    return root / path


@beartype
def _load_image_array(
    reference_dir: Path,
    image_path: str,
) -> Float[NDArray, "H W"]:
    """Load a compressed reference image array from disk."""
    npz_path: Path = _resolve_path(reference_dir, image_path)
    with np.load(npz_path) as data:
        return np.asarray(data["image"], dtype=np.float64)


@beartype
def load_reference_cases(
    reference_dir: Path = _DEFAULT_REFERENCE_DIR,
) -> list[ReferenceCase]:
    """Load reference metadata and images from a benchmark directory.

    Parameters
    ----------
    reference_dir : Path, optional
        Directory containing ``*_metadata.json`` files and associated
        ``.npz`` images. Default:
        ``tests/test_data/reference_data/synthetic``.

    Returns
    -------
    cases : list[ReferenceCase]
        Loaded reference cases with typed metadata and image arrays.
    """
    cases: list[ReferenceCase] = []
    for metadata_path in sorted(reference_dir.glob("*_metadata.json")):
        metadata_payload: dict[str, Any] = json.loads(
            metadata_path.read_text()
        )
        metadata = ReferenceMetadata.from_dict(
            {
                **metadata_payload,
                "metadata_path": str(metadata_path),
            }
        )
        cases.append(
            ReferenceCase(
                metadata=metadata,
                image=_load_image_array(reference_dir, metadata.image_path),
            )
        )
    return cases


@beartype
def _coerce_metadata(
    reference: ReferenceMetadata | ReferenceCase,
) -> ReferenceMetadata:
    """Normalize a reference input down to its metadata object."""
    if isinstance(reference, ReferenceCase):
        return reference.metadata
    return reference


@beartype
def render_pattern_to_image(
    detector_points_mm: Float[NDArray, "N 2"],
    intensities: Float[NDArray, "N"],
    image_shape_px: tuple[int, int],
    pixel_size_mm: tuple[float, float],
    beam_center_px: tuple[float, float],
    spot_sigma_px: float,
) -> Float[NDArray, "H W"]:
    """Rasterize sparse detector hits into a dense detector image.

    Parameters
    ----------
    detector_points_mm : Float[NDArray, "N 2"]
        Detector coordinates in millimeters with shape ``(N, 2)``.
    intensities : Float[NDArray, "N"]
        Non-negative reflection intensities with shape ``(N,)``.
    image_shape_px : tuple[int, int]
        Output image shape ``(height, width)`` in pixels.
    pixel_size_mm : tuple[float, float]
        Detector calibration ``(x_mm_per_px, y_mm_per_px)``.
    beam_center_px : tuple[float, float]
        Pixel coordinate corresponding to detector ``(0 mm, 0 mm)``.
    spot_sigma_px : float
        Gaussian spot width in pixels.

    Returns
    -------
    image : Float[NDArray, "H W"]
        Rasterized detector image, normalized to unit maximum.
    """
    image_height, image_width = image_shape_px
    x_mm_per_px, y_mm_per_px = pixel_size_mm
    center_x_px, center_y_px = beam_center_px

    x_pixels = detector_points_mm[:, 0] / x_mm_per_px + center_x_px
    y_pixels = detector_points_mm[:, 1] / y_mm_per_px + center_y_px

    x_axis = np.arange(image_width, dtype=np.float64)
    y_axis = np.arange(image_height, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    image = np.zeros((image_height, image_width), dtype=np.float64)

    for x0_px, y0_px, intensity in zip(
        x_pixels, y_pixels, intensities, strict=True
    ):
        image += intensity * np.exp(
            -((x_grid - x0_px) ** 2 + (y_grid - y0_px) ** 2)
            / (2.0 * spot_sigma_px**2)
        )

    max_intensity = float(np.max(image))
    if max_intensity > 0.0:
        image /= max_intensity

    return image


@beartype
def simulate_detector_image_from_metadata(
    reference: ReferenceMetadata | ReferenceCase,
    repo_root: Path = _REPO_ROOT,
) -> Float[NDArray, "H W"]:
    """Regenerate a detector image from stored benchmark metadata.

    Parameters
    ----------
    reference : ReferenceMetadata | ReferenceCase
        Reference metadata object, or a loaded reference case.
    repo_root : Path, optional
        Repository root used to resolve relative paths.

    Returns
    -------
    detector_image : Float[NDArray, "H W"]
        Simulated detector image.
    """
    metadata = _coerce_metadata(reference)
    if metadata.simulation_mode != "ewald":
        raise ValueError(
            f"Unsupported simulation_mode: {metadata.simulation_mode}"
        )

    crystal = parse_cif(_resolve_path(repo_root, metadata.cif_path))
    image_shape_px = metadata.image_shape_px
    pixel_size_mm = metadata.pixel_size_mm
    beam_center_px = metadata.beam_center_px
    spot_sigma_px = metadata.spot_sigma_px

    def _simulate_raw_image(
        energy_kev: float,
        theta_deg: float,
        phi_deg: float,
    ) -> Float[NDArray, "H W"]:
        pattern = ewald_simulator(
            crystal=crystal,
            voltage_kv=energy_kev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            hmax=metadata.hmax,
            kmax=metadata.kmax,
            detector_distance=metadata.detector_distance_mm,
            temperature=metadata.temperature_kelvin,
            surface_roughness=metadata.surface_roughness_angstrom,
        )
        return render_pattern_to_image(
            detector_points_mm=np.asarray(pattern.detector_points),
            intensities=np.asarray(pattern.intensities),
            image_shape_px=image_shape_px,
            pixel_size_mm=pixel_size_mm,
            beam_center_px=beam_center_px,
            spot_sigma_px=spot_sigma_px,
        )

    theta_deg = metadata.theta_deg
    phi_deg = metadata.phi_deg
    energy_kev = metadata.voltage_kv
    divergence_rad = metadata.angular_divergence_mrad * 1e-3
    spread_kev = metadata.energy_spread_ev * 1e-3

    angle_nodes, angle_weights = gauss_hermite_nodes_weights(
        metadata.n_angular_samples
    )
    angle_average = np.zeros(image_shape_px, dtype=np.float64)
    for node, weight in zip(
        np.asarray(angle_nodes),
        np.asarray(angle_weights),
        strict=True,
    ):
        theta_sample_deg = theta_deg + np.rad2deg(
            np.sqrt(2.0) * divergence_rad * float(node)
        )
        angle_average += float(weight) * _simulate_raw_image(
            energy_kev=energy_kev,
            theta_deg=theta_sample_deg,
            phi_deg=phi_deg,
        )
    angle_average /= np.sqrt(np.pi)

    energy_nodes, energy_weights = gauss_hermite_nodes_weights(
        metadata.n_energy_samples
    )
    energy_average = np.zeros(image_shape_px, dtype=np.float64)
    for node, weight in zip(
        np.asarray(energy_nodes),
        np.asarray(energy_weights),
        strict=True,
    ):
        energy_sample_kev = energy_kev + np.sqrt(2.0) * spread_kev * float(
            node
        )
        energy_average += float(weight) * _simulate_raw_image(
            energy_kev=energy_sample_kev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )
    energy_average /= np.sqrt(np.pi)

    combined_image = 0.5 * (angle_average + energy_average)
    broadened_image = detector_psf_convolve(
        detector_image=jnp.asarray(combined_image, dtype=jnp.float64),
        psf_sigma_pixels=jnp.asarray(
            metadata.psf_sigma_pixels, dtype=jnp.float64
        ),
    )
    final_image = np.array(broadened_image, dtype=np.float64, copy=True)
    max_intensity = float(np.max(final_image))
    if max_intensity > 0.0:
        final_image /= max_intensity
    return final_image


@beartype
def benchmark_reference_case(
    reference_case: ReferenceCase,
    repo_root: Path = _REPO_ROOT,
) -> BenchmarkCaseResult:
    """Compare one stored reference case to a regenerated simulation.

    Parameters
    ----------
    reference_case : ReferenceCase
        Loaded reference case with metadata and image.
    repo_root : Path, optional
        Repository root used to resolve relative paths.

    Returns
    -------
    summary : BenchmarkCaseResult
        Structured benchmark summary for the case.
    """
    metadata = reference_case.metadata
    reference_image = jnp.asarray(reference_case.image, dtype=jnp.float64)
    simulated_image = jnp.asarray(
        simulate_detector_image_from_metadata(metadata, repo_root=repo_root),
        dtype=jnp.float64,
    )

    reference_specular = jnp.asarray(
        metadata.specular_position_px, dtype=jnp.float64
    )
    simulated_specular = peak_centroid(simulated_image)
    if metadata.dominant_peak_positions_px:
        reference_peak_positions = jnp.asarray(
            metadata.dominant_peak_positions_px,
            dtype=jnp.float64,
        )
    else:
        reference_peak_positions = dominant_peak_positions(
            reference_image,
            axis="horizontal",
            n_peaks=metadata.n_metric_peaks,
            min_separation_px=metadata.peak_min_separation_px,
        )
    simulated_peak_positions = dominant_peak_positions(
        simulated_image,
        axis="horizontal",
        n_peaks=metadata.n_metric_peaks,
        min_separation_px=metadata.peak_min_separation_px,
    )
    reference_profile = extract_streak_profile(
        reference_image,
        reference_specular,
        axis="vertical",
        band_half_width_px=metadata.profile_band_half_width_px,
    )
    reference_fwhm = (
        jnp.asarray(metadata.reference_streak_fwhm_px, dtype=jnp.float64)
        if metadata.reference_streak_fwhm_px is not None
        else streak_fwhm_px(reference_profile)
    )
    simulated_profile = extract_streak_profile(
        simulated_image,
        simulated_specular,
        axis="vertical",
        band_half_width_px=metadata.profile_band_half_width_px,
    )
    simulated_fwhm = streak_fwhm_px(simulated_profile)

    return BenchmarkCaseResult(
        reference_id=metadata.reference_id,
        material=metadata.material,
        surface=metadata.surface,
        source_kind=metadata.source_kind,
        simulation_mode=metadata.simulation_mode,
        normalized_cross_correlation=float(
            normalized_cross_correlation(reference_image, simulated_image)
        ),
        specular_offset_px=float(
            specular_offset_px(reference_specular, simulated_specular)
        ),
        peak_centroid_error_px=float(
            peak_centroid_error_px(reference_image, simulated_image)
        ),
        rod_spacing_error_px=float(
            rod_spacing_error_px(
                reference_peak_positions, simulated_peak_positions
            )
        ),
        reference_streak_fwhm_px=float(reference_fwhm),
        simulated_streak_fwhm_px=float(simulated_fwhm),
        streak_fwhm_abs_error_px=float(
            jnp.abs(reference_fwhm - simulated_fwhm)
        ),
    )


@beartype
def benchmark_reference_suite(
    reference_dir: Path = _DEFAULT_REFERENCE_DIR,
    output_path: Path | None = None,
    repo_root: Path = _REPO_ROOT,
) -> BenchmarkSuiteResult:
    """Run the full reference benchmark suite and aggregate the results.

    Parameters
    ----------
    reference_dir : Path, optional
        Directory containing benchmark metadata and images.
    output_path : Path | None, optional
        Optional JSON file to write with the benchmark summary.
    repo_root : Path, optional
        Repository root used to resolve relative paths.

    Returns
    -------
    suite_summary : BenchmarkSuiteResult
        Aggregate benchmark summary with one entry per case.
    """
    cases = load_reference_cases(reference_dir)
    if not cases:
        raise ValueError(f"No reference cases found in {reference_dir}")

    case_summaries: tuple[BenchmarkCaseResult, ...] = tuple(
        benchmark_reference_case(reference_case=case, repo_root=repo_root)
        for case in cases
    )
    mean_ncc = float(
        np.mean([case.normalized_cross_correlation for case in case_summaries])
    )
    mean_specular_offset = float(
        np.mean([case.specular_offset_px for case in case_summaries])
    )
    max_streak_fwhm_error = float(
        np.max([case.streak_fwhm_abs_error_px for case in case_summaries])
    )
    summary = BenchmarkSuiteResult(
        reference_count=len(case_summaries),
        mean_normalized_cross_correlation=mean_ncc,
        mean_specular_offset_px=mean_specular_offset,
        max_streak_fwhm_abs_error_px=max_streak_fwhm_error,
        cases=case_summaries,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")

    return summary


@beartype
def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run the rheedium reference-image audit benchmark."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=_DEFAULT_REFERENCE_DIR,
        help="Directory containing *_metadata.json and .npz image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the benchmark summary.",
    )
    return parser


@beartype
def main() -> None:
    """Run the reference benchmark suite from the command line."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    summary = benchmark_reference_suite(
        reference_dir=args.reference_dir,
        output_path=args.output,
    )
    print(json.dumps(summary.to_dict(), indent=2))


__all__: list[str] = [
    "benchmark_reference_case",
    "benchmark_reference_suite",
    "load_reference_cases",
    "render_pattern_to_image",
    "simulate_detector_image_from_metadata",
]


if __name__ == "__main__":
    main()
