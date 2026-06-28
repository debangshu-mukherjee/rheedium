"""Fixture contract tests for future Loop A RHEED automatons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rheedium.inout import load_tiff_as_rheed_image, load_tiff_sequence

_REPO_ROOT: Path = Path(__file__).parents[2]
_FIXTURE_DIR: Path = _REPO_ROOT / "tests" / "test_data" / "rheed_loop_a"


def _metadata() -> dict[str, Any]:
    """Load the committed Loop A fixture metadata."""
    return json.loads(
        (_FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8")
    )


def _specular_trace(
    sequence: np.ndarray[Any, Any],
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray[Any, Any]:
    """Return the specular ROI intensity trace across a frame series."""
    return np.asarray(
        sequence[
            :,
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
    ).sum(axis=(1, 2))


def _dominant_period_frames(trace: np.ndarray[Any, Any]) -> float:
    """Estimate the dominant nonzero Fourier period in frames."""
    centered = np.asarray(trace, dtype=np.float64) - float(np.mean(trace))
    spectrum = np.abs(np.fft.rfft(centered))
    spectrum[0] = 0.0
    frequency_index = int(np.argmax(spectrum))
    if frequency_index <= 0:
        raise ValueError("trace has no nonzero dominant frequency")
    return float(centered.shape[0] / frequency_index)


def test_loop_a_rheed_frame_fixture_loads_as_image() -> None:
    r"""The committed single-frame fixture loads as a RHEED image.

    Extended Summary
    ----------------
    Verifies the pre-G4 fixture contract for online ingest: a committed TIFF
    frame can be loaded through rheedium's normal RHEED-image reader and has
    the expected detector shape.

    Notes
    -----
    This test intentionally uses the same public I/O function that
    ``rheed_ingest.py`` should call, so future Loop A automata start from a
    stable file contract rather than a temporary generated frame.
    """
    meta = _metadata()
    image = load_tiff_as_rheed_image(
        _FIXTURE_DIR / str(meta["single_frame"]),
        incoming_angle_deg=2.0,
        energy_kev=20.0,
        detector_distance_mm=350.0,
    )

    assert tuple(image.img_array.shape) == tuple(meta["frame_shape"])
    assert float(np.asarray(image.img_array).max()) > 0.0


def test_loop_a_rheed_series_fixture_has_known_period() -> None:
    r"""The committed frame-series fixture carries a known oscillation period.

    Extended Summary
    ----------------
    Verifies the pre-G4 series contract for growth monitoring: the committed
    multipage TIFF has the declared shape and a specular ROI trace whose
    dominant period matches the planted oscillation period.

    Notes
    -----
    ``growth_monitor.py`` can use this fixture as its smoke input. The test
    computes the period directly from the committed series, so regressions in
    fixture generation or accidental binary churn are caught by ordinary
    ``pytest``.
    """
    meta = _metadata()
    sequence, frame_metadata = load_tiff_sequence(
        _FIXTURE_DIR / str(meta["series"])
    )
    center_y, center_x = meta["specular_center_yx"]
    trace = _specular_trace(
        np.asarray(sequence),
        center_y=int(center_y),
        center_x=int(center_x),
        radius=int(meta["specular_roi_radius"]),
    )

    assert tuple(sequence.shape) == (
        int(meta["n_frames"]),
        *tuple(meta["frame_shape"]),
    )
    assert len(frame_metadata) == int(meta["n_frames"])
    assert _dominant_period_frames(trace) == pytest.approx(
        float(meta["period_frames"]),
        abs=0.1,
    )
