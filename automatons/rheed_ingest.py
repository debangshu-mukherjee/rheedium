# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.10"]
# ///
"""Ingest one RHEED frame and emit per-frame growth observables.

The automaton is the Loop A per-frame-stateless entry point: it loads one TIFF,
NumPy, or HDF5 detector frame, computes compact growth observables, writes a
state JSON artifact, and exits. Smoke mode uses the committed
``tests/test_data/rheed_loop_a`` fixture so repeatability and the run→emit→exit
contract are tested without a live detector.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from beartype.typing import Any

import rheedium as rh
from rheedium.harness import Param, experiment

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_LOOP_A_FIXTURE_DIR: Path = _REPO_ROOT / "tests" / "test_data" / "rheed_loop_a"
_EPS: float = 1e-12


def _fixture_metadata() -> dict[str, Any]:
    """Load the committed Loop A fixture metadata."""
    return json.loads(
        (_LOOP_A_FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8")
    )


def _first_array_from_mapping(data: Mapping[str, Any]) -> Any:
    """Return an image-like value from a deterministic mapping."""
    for key in ("image", "frame", "series", "data"):
        if key in data:
            return data[key]
    if not data:
        raise ValueError("HDF5/NPZ mapping contains no arrays")
    return data[sorted(data)[0]]


def _array_from_loaded(value: Any) -> np.ndarray[Any, Any]:
    """Convert a loaded object or PyTree image carrier to an ndarray."""
    if isinstance(value, Mapping):
        return _array_from_loaded(_first_array_from_mapping(value))
    if hasattr(value, "img_array"):
        return np.asarray(value.img_array, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _load_frame(path: Path, *, hdf5_dataset: str) -> np.ndarray[Any, Any]:
    """Load a single detector frame from TIFF, NumPy, or HDF5."""
    suffix: str = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = rh.inout.load_tiff_as_rheed_image(
            path,
            incoming_angle_deg=2.0,
            energy_kev=20.0,
            detector_distance_mm=350.0,
        )
        frame = np.asarray(image.img_array, dtype=np.float64)
    elif suffix == ".npy":
        frame = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            frame = _array_from_loaded(dict(data))
    elif suffix in {".h5", ".hdf5"}:
        loaded = rh.inout.load_from_h5(
            path,
            name=hdf5_dataset or None,
        )
        frame = _array_from_loaded(loaded)
    else:
        raise ValueError(f"unsupported frame extension: {suffix}")
    if frame.ndim > 2:
        frame = frame[0]
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D; got shape {frame.shape}")
    return np.nan_to_num(frame, copy=True)


def _center_from_args(
    args: Any,
    image: np.ndarray[Any, Any],
    metadata: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """Return the requested, fixture, or brightest-pixel specular center."""
    if args.specular_center_y >= 0 and args.specular_center_x >= 0:
        return int(args.specular_center_y), int(args.specular_center_x)
    if args.smoke and metadata is not None:
        center_y, center_x = metadata["specular_center_yx"]
        return int(center_y), int(center_x)
    center_y, center_x = np.unravel_index(int(np.argmax(image)), image.shape)
    return int(center_y), int(center_x)


def _window_bounds(
    center: int,
    radius: int,
    upper: int,
) -> tuple[int, int]:
    """Return inclusive-exclusive clipped bounds around a pixel center."""
    start: int = max(0, int(center) - int(radius))
    stop: int = min(int(upper), int(center) + int(radius) + 1)
    return start, stop


def _roi_sum(
    image: np.ndarray[Any, Any],
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> float:
    """Sum one square region of interest."""
    y0, y1 = _window_bounds(center_y, radius, image.shape[0])
    x0, x1 = _window_bounds(center_x, radius, image.shape[1])
    return float(np.sum(image[y0:y1, x0:x1]))


def _row_profile(
    image: np.ndarray[Any, Any],
    *,
    center_x: int,
    half_width: int,
) -> np.ndarray[Any, Any]:
    """Return a vertical profile from a detector-column window."""
    x0, x1 = _window_bounds(center_x, half_width, image.shape[1])
    return np.asarray(image[:, x0:x1], dtype=np.float64).sum(axis=1)


def _peak_rows(
    profile: np.ndarray[Any, Any],
    *,
    min_separation: int,
    n_peaks: int,
) -> list[int]:
    """Return the strongest separated row peaks."""
    order = np.argsort(profile)[::-1]
    peaks: list[int] = []
    for raw_index in order:
        index: int = int(raw_index)
        if all(abs(index - existing) >= min_separation for existing in peaks):
            peaks.append(index)
        if len(peaks) >= n_peaks:
            break
    return sorted(peaks)


def _streak_observables(
    image: np.ndarray[Any, Any],
    *,
    center_x: int,
    half_width: int,
    min_separation: int,
) -> dict[str, Any]:
    """Estimate streak spacing and sharpness from a vertical profile."""
    profile = _row_profile(image, center_x=center_x, half_width=half_width)
    peaks = _peak_rows(profile, min_separation=min_separation, n_peaks=5)
    spacings: list[int] = [
        right - left for left, right in zip(peaks, peaks[1:], strict=False)
    ]
    spacing_px: float = float(np.median(spacings)) if spacings else 0.0
    profile_median: float = float(np.median(profile))
    profile_peak: float = float(np.max(profile))
    profile_mean: float = float(np.mean(profile))
    sharpness: float = (profile_peak - profile_median) / (profile_mean + _EPS)
    return {
        "profile": profile,
        "peak_rows": peaks,
        "streak_spacing_px": spacing_px,
        "streak_sharpness": sharpness,
    }


def _growth_state(
    *,
    is_2d: bool,
    specular_fraction: float,
    streak_spacing_px: float,
) -> str:
    """Return a compact per-frame growth-state label."""
    if is_2d and streak_spacing_px > 0.0 and specular_fraction >= 0.02:
        return "2d_streaky"
    if is_2d:
        return "2d_weak"
    return "3d_spotty"


@experiment(
    name="rheed-ingest",
    params=[
        Param(
            "frame",
            str,
            default="",
            help="Detector frame path (.tif/.tiff/.npy/.npz/.h5/.hdf5).",
            example="tests/test_data/rheed_loop_a/rheed_frame_000.tif",
        ),
        Param(
            "hdf5_dataset",
            str,
            default="",
            help="Optional HDF5 top-level dataset/group name.",
            example="image",
        ),
        Param(
            "specular_center_y",
            int,
            default=-1,
            help="Specular ROI center row; negative auto-detects.",
            unit="px",
        ),
        Param(
            "specular_center_x",
            int,
            default=-1,
            help="Specular ROI center column; negative auto-detects.",
            unit="px",
        ),
        Param(
            "specular_radius_px",
            int,
            default=3,
            help="Square half-width for specular ROI integration.",
            unit="px",
            bounds=(1.0, 32.0),
            example=3,
        ),
        Param(
            "profile_half_width_px",
            int,
            default=18,
            help="Detector-column half-width for streak row profile.",
            unit="px",
            bounds=(1.0, 256.0),
            example=18,
        ),
        Param(
            "min_peak_separation_px",
            int,
            default=8,
            help="Minimum row separation for streak peak picking.",
            unit="px",
            bounds=(1.0, 128.0),
            example=8,
        ),
        Param(
            "streak_sharpness_threshold",
            float,
            default=1.0,
            help="Sharpness threshold for 2D/streaky classification.",
            example=1.0,
        ),
    ],
    returns={
        "metrics": {
            "specular_intensity": {"type": "number"},
            "streak_spacing_px": {"type": "number"},
            "streak_sharpness": {"type": "number"},
            "is_2d": {"type": "boolean"},
        },
        "artifacts": {"roles": ["growth_state", "observables", "rheed_frame"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run one stateless RHEED-frame ingest step."""
    metadata: dict[str, Any] | None = (
        _fixture_metadata() if args.smoke else None
    )
    frame_path = Path(args.frame)
    if args.smoke and not args.frame:
        assert metadata is not None
        frame_path = _LOOP_A_FIXTURE_DIR / str(metadata["single_frame"])
    if not str(frame_path):
        raise ValueError("frame is required unless --smoke is set")

    image = _load_frame(frame_path, hdf5_dataset=args.hdf5_dataset)
    center_y, center_x = _center_from_args(args, image, metadata)
    specular_intensity = _roi_sum(
        image,
        center_y=center_y,
        center_x=center_x,
        radius=args.specular_radius_px,
    )
    integrated_intensity: float = float(np.sum(image))
    specular_fraction: float = specular_intensity / (
        integrated_intensity + _EPS
    )
    streak = _streak_observables(
        image,
        center_x=center_x,
        half_width=args.profile_half_width_px,
        min_separation=args.min_peak_separation_px,
    )
    is_2d: bool = bool(
        streak["streak_sharpness"] >= args.streak_sharpness_threshold
        and streak["streak_spacing_px"] > 0.0
    )
    state_label: str = _growth_state(
        is_2d=is_2d,
        specular_fraction=specular_fraction,
        streak_spacing_px=float(streak["streak_spacing_px"]),
    )
    state: dict[str, Any] = {
        "frame": frame_path.as_posix(),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "specular_center_yx": [center_y, center_x],
        "specular_intensity": specular_intensity,
        "specular_fraction": specular_fraction,
        "streak_peak_rows": streak["peak_rows"],
        "streak_spacing_px": float(streak["streak_spacing_px"]),
        "streak_sharpness": float(streak["streak_sharpness"]),
        "surface_state": state_label,
        "is_2d": is_2d,
    }
    state_artifact = ctx.save_json(
        "growth_state.json",
        state,
        role="growth_state",
    )
    observables_artifact = ctx.save_array(
        "frame_observables.npz",
        {
            "image": image,
            "row_profile": streak["profile"],
            "specular_center_yx": np.asarray([center_y, center_x]),
        },
        role="observables",
    )
    frame_artifact = ctx.save_image(
        "rheed_frame.png",
        image,
        cmap="phosphor",
        role="rheed_frame",
    )
    metrics: dict[str, Any] = {
        "specular_intensity": specular_intensity,
        "specular_fraction": specular_fraction,
        "specular_center_y": center_y,
        "specular_center_x": center_x,
        "streak_spacing_px": float(streak["streak_spacing_px"]),
        "streak_sharpness": float(streak["streak_sharpness"]),
        "integrated_intensity": integrated_intensity,
        "is_2d": is_2d,
        "surface_state": state_label,
    }
    return {
        "metrics": metrics,
        "artifacts": [state_artifact, observables_artifact, frame_artifact],
        "growth_state": state,
    }


if __name__ == "__main__":
    main()
