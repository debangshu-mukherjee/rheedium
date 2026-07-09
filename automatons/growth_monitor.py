# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Monitor a rolling RHEED frame series for growth dynamics.

The automaton is the Loop A temporal companion to ``rheed_ingest``. It reads an
explicit rolling frame-series file, computes a specular intensity trace,
recovers the dominant oscillation period, summarizes a roughness trend, writes
machine-readable artifacts, and exits. Smoke mode uses the committed synthetic
Loop A TIFF series with a planted six-frame oscillation period.
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
    """Return an image-series-like value from a deterministic mapping."""
    for key in ("series", "frames", "image", "data"):
        if key in data:
            return data[key]
    if not data:
        raise ValueError("HDF5/NPZ mapping contains no arrays")
    return data[sorted(data)[0]]


def _array_from_loaded(value: Any) -> np.ndarray[Any, Any]:
    """Convert a loaded object or mapping to an ndarray."""
    if isinstance(value, Mapping):
        return _array_from_loaded(_first_array_from_mapping(value))
    if hasattr(value, "img_array"):
        return np.asarray(value.img_array, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _load_series(path: Path, *, hdf5_dataset: str) -> np.ndarray[Any, Any]:
    """Load a detector frame series from TIFF, NumPy, or HDF5."""
    suffix: str = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        series, _metadata = rh.inout.load_tiff_sequence(path)
        frames = np.asarray(series, dtype=np.float64)
    elif suffix == ".npy":
        frames = np.asarray(
            np.load(path, allow_pickle=False), dtype=np.float64
        )
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            frames = _array_from_loaded(dict(data))
    elif suffix in {".h5", ".hdf5"}:
        loaded = rh.inout.load_from_h5(
            path,
            name=hdf5_dataset or None,
        )
        frames = _array_from_loaded(loaded)
    else:
        raise ValueError(f"unsupported series extension: {suffix}")
    if frames.ndim == 2:
        frames = frames[None, :, :]
    if frames.ndim != 3:
        raise ValueError(f"series must have shape (T,H,W); got {frames.shape}")
    return np.nan_to_num(frames, copy=True)


def _center_from_args(
    args: Any,
    frames: np.ndarray[Any, Any],
    metadata: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """Return the requested, fixture, or brightest-pixel specular center."""
    if args.specular_center_y >= 0 and args.specular_center_x >= 0:
        return int(args.specular_center_y), int(args.specular_center_x)
    if args.smoke and metadata is not None:
        center_y, center_x = metadata["specular_center_yx"]
        return int(center_y), int(center_x)
    mean_frame = np.mean(frames, axis=0)
    center_y, center_x = np.unravel_index(
        int(np.argmax(mean_frame)),
        mean_frame.shape,
    )
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


def _specular_trace(
    frames: np.ndarray[Any, Any],
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray[Any, Any]:
    """Return the specular ROI intensity trace across a frame series."""
    y0, y1 = _window_bounds(center_y, radius, frames.shape[1])
    x0, x1 = _window_bounds(center_x, radius, frames.shape[2])
    return np.asarray(frames[:, y0:y1, x0:x1]).sum(axis=(1, 2))


def _dominant_period_frames(
    trace: np.ndarray[Any, Any],
    *,
    min_period: float,
    max_period: float,
) -> tuple[float, float, int]:
    """Return dominant nonzero Fourier period, amplitude, and index."""
    centered = np.asarray(trace, dtype=np.float64) - float(np.mean(trace))
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.shape[0] <= 1:
        return 0.0, 0.0, 0
    spectrum[0] = 0.0
    indices = np.arange(spectrum.shape[0], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = centered.shape[0] / indices
    mask = np.isfinite(periods)
    mask &= periods >= min_period
    mask &= periods <= max_period
    mask[0] = False
    if not np.any(mask):
        frequency_index = int(np.argmax(spectrum))
    else:
        masked = np.where(mask, spectrum, -np.inf)
        frequency_index = int(np.argmax(masked))
    if frequency_index <= 0:
        return 0.0, 0.0, 0
    return (
        float(centered.shape[0] / frequency_index),
        float(spectrum[frequency_index]),
        frequency_index,
    )


def _linear_slope(values: np.ndarray[Any, Any]) -> float:
    """Return least-squares slope over frame index."""
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.shape[0], dtype=np.float64)
    centered_x = x - float(np.mean(x))
    centered_y = y - float(np.mean(y))
    denom: float = float(np.sum(centered_x**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(centered_x * centered_y) / denom)


def _roughness_trace(frames: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a simple diffuse-background fraction per frame."""
    totals = np.sum(frames, axis=(1, 2))
    medians = np.median(frames, axis=(1, 2))
    diffuse = medians * frames.shape[1] * frames.shape[2]
    return np.asarray(diffuse / (totals + _EPS), dtype=np.float64)


def _trend_label(slope: float, threshold: float) -> str:
    """Convert a roughness slope to a compact trend label."""
    if slope > threshold:
        return "roughening"
    if slope < -threshold:
        return "smoothing"
    return "stable"


@experiment(
    name="growth-monitor",
    params=[
        Param(
            "series",
            str,
            default="",
            help="Rolling detector series path (.tif/.npy/.npz/.h5/.hdf5).",
            example="tests/test_data/rheed_loop_a/rheed_series.tif",
        ),
        Param(
            "hdf5_dataset",
            str,
            default="",
            help="Optional HDF5 top-level dataset/group name.",
            example="series",
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
            "min_period_frames",
            float,
            default=2.0,
            help="Shortest accepted oscillation period.",
            unit="frames",
            bounds=(1.0, 10000.0),
        ),
        Param(
            "max_period_frames",
            float,
            default=24.0,
            help="Longest accepted oscillation period.",
            unit="frames",
            bounds=(1.0, 100000.0),
        ),
        Param(
            "roughness_slope_threshold",
            float,
            default=5e-4,
            help="Absolute slope threshold for roughness trend labels.",
        ),
    ],
    returns={
        "metrics": {
            "dominant_period_frames": {"type": "number"},
            "oscillation_count": {"type": "number"},
            "roughness_trend": {"type": "string"},
            "n_frames": {"type": "integer"},
        },
        "artifacts": {
            "roles": [
                "growth_monitor",
                "time_series",
                "mean_rheed_frame",
                "mean_rheed_frame_linear",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run one stateless RHEED growth-monitoring step."""
    metadata: dict[str, Any] | None = (
        _fixture_metadata() if args.smoke else None
    )
    series_path = Path(args.series)
    if args.smoke and not args.series:
        assert metadata is not None
        series_path = _LOOP_A_FIXTURE_DIR / str(metadata["series"])
    if not str(series_path):
        raise ValueError("series is required unless --smoke is set")

    frames = _load_series(series_path, hdf5_dataset=args.hdf5_dataset)
    center_y, center_x = _center_from_args(args, frames, metadata)
    trace = _specular_trace(
        frames,
        center_y=center_y,
        center_x=center_x,
        radius=args.specular_radius_px,
    )
    period, amplitude, frequency_index = _dominant_period_frames(
        trace,
        min_period=args.min_period_frames,
        max_period=args.max_period_frames,
    )
    n_frames: int = int(frames.shape[0])
    oscillation_count: float = (
        float(n_frames / period) if period > 0.0 else 0.0
    )
    roughness = _roughness_trace(frames)
    roughness_slope: float = _linear_slope(roughness)
    roughness_trend: str = _trend_label(
        roughness_slope,
        args.roughness_slope_threshold,
    )
    oscillation_detected: bool = bool(period > 0.0 and amplitude > 0.0)
    monitor: dict[str, Any] = {
        "series": series_path.as_posix(),
        "image_shape": [int(frames.shape[1]), int(frames.shape[2])],
        "n_frames": n_frames,
        "specular_center_yx": [center_y, center_x],
        "dominant_period_frames": period,
        "dominant_frequency_index": frequency_index,
        "dominant_amplitude": amplitude,
        "oscillation_count": oscillation_count,
        "roughness_index": float(np.mean(roughness)),
        "roughness_slope": roughness_slope,
        "roughness_trend": roughness_trend,
        "transition_flags": {
            "oscillation_detected": oscillation_detected,
            "roughening": roughness_trend == "roughening",
            "smoothing": roughness_trend == "smoothing",
        },
    }
    monitor_artifact = ctx.save_json(
        "growth_monitor.json",
        monitor,
        role="growth_monitor",
    )
    series_artifact = ctx.save_array(
        "growth_trace.npz",
        {
            "specular_trace": trace,
            "roughness_trace": roughness,
            "frame_index": np.arange(n_frames, dtype=np.int32),
        },
        role="time_series",
    )
    mean_artifacts = ctx.save_image_scales(
        "mean_rheed_frame.png",
        np.mean(frames, axis=0),
        cmap="phosphor",
        role="mean_rheed_frame",
    )
    metrics: dict[str, Any] = {
        "n_frames": n_frames,
        "dominant_period_frames": period,
        "dominant_amplitude": amplitude,
        "oscillation_count": oscillation_count,
        "roughness_index": float(np.mean(roughness)),
        "roughness_slope": roughness_slope,
        "roughness_trend": roughness_trend,
        "oscillation_detected": oscillation_detected,
        "specular_trace_mean": float(np.mean(trace)),
    }
    return {
        "metrics": metrics,
        "artifacts": [monitor_artifact, series_artifact, *mean_artifacts],
        "growth_monitor": monitor,
    }


if __name__ == "__main__":
    main()
