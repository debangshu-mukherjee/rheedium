"""Generate deterministic synthetic RHEED fixtures for Loop A automatons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray

HEIGHT: int = 48
WIDTH: int = 64
N_FRAMES: int = 24
PERIOD_FRAMES: float = 6.0
SPECULAR_CENTER_YX: tuple[int, int] = (12, 32)
SPECULAR_ROI_RADIUS: int = 3
STREAK_SPACING_PX: int = 10


def _gaussian(
    yy: NDArray[np.float64],
    xx: NDArray[np.float64],
    *,
    center_y: float,
    center_x: float,
    sigma_y: float,
    sigma_x: float,
) -> NDArray[np.float64]:
    """Return a two-dimensional Gaussian field."""
    return np.exp(
        -0.5
        * (((yy - center_y) / sigma_y) ** 2 + ((xx - center_x) / sigma_x) ** 2)
    )


def _frame(index: int) -> NDArray[np.uint16]:
    """Return one deterministic synthetic RHEED frame."""
    y_axis = np.arange(HEIGHT, dtype=np.float64)
    x_axis = np.arange(WIDTH, dtype=np.float64)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    center_y, center_x = SPECULAR_CENTER_YX
    phase = 2.0 * np.pi * index / PERIOD_FRAMES

    frame = 110.0 + 0.8 * yy + 0.35 * xx
    specular_scale = 2200.0 + 750.0 * np.cos(phase)
    frame += specular_scale * _gaussian(
        yy,
        xx,
        center_y=float(center_y),
        center_x=float(center_x),
        sigma_y=2.0,
        sigma_x=3.0,
    )

    for offset, scale in (
        (-STREAK_SPACING_PX, 900.0),
        (0, 1200.0),
        (STREAK_SPACING_PX, 800.0),
    ):
        frame += scale * _gaussian(
            yy,
            xx,
            center_y=float(center_y + offset),
            center_x=float(center_x),
            sigma_y=0.9,
            sigma_x=18.0,
        )

    frame += 220.0 * _gaussian(
        yy,
        xx,
        center_y=float(center_y + 18),
        center_x=float(center_x - 12),
        sigma_y=2.4,
        sigma_x=5.0,
    )
    return np.clip(np.rint(frame), 0, np.iinfo(np.uint16).max).astype(
        np.uint16
    )


def main() -> None:
    """Write the single-frame and multi-frame TIFF fixtures."""
    outdir = Path(__file__).resolve().parent
    series = np.stack([_frame(index) for index in range(N_FRAMES)], axis=0)
    tifffile.imwrite(
        outdir / "rheed_frame_000.tif",
        series[0],
        photometric="minisblack",
    )
    tifffile.imwrite(
        outdir / "rheed_series.tif",
        series,
        photometric="minisblack",
    )
    metadata = {
        "description": (
            "Deterministic synthetic RHEED frame/series fixture for Loop A "
            "online-ingest automatons."
        ),
        "frame_shape": [HEIGHT, WIDTH],
        "n_frames": N_FRAMES,
        "period_frames": PERIOD_FRAMES,
        "specular_center_yx": list(SPECULAR_CENTER_YX),
        "specular_roi_radius": SPECULAR_ROI_RADIUS,
        "streak_spacing_px": STREAK_SPACING_PX,
        "single_frame": "rheed_frame_000.tif",
        "series": "rheed_series.tif",
        "units": {
            "intensity": "uint16_counts",
            "period_frames": "frames",
            "specular_center_yx": "pixels",
            "streak_spacing_px": "pixels",
        },
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
