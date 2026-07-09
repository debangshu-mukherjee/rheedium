"""Export stable README figures from committed tutorial artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rheedium.plots import create_phosphor_colormap  # noqa: E402


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SWEEP_DIR: Final[Path] = PROJECT_ROOT / "tutorials" / "sweeps"
OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "docs" / "source" / "_static" / "readme"
)
README_GALLERY: Final[Path] = OUTPUT_DIR / "rheed-sweep-gallery.png"


@dataclass(frozen=True)
class SweepPanel:
    """One source sweep and frame to render in the README gallery."""

    label: str
    filename: str
    frame_index: int


@dataclass(frozen=True)
class RenderedPanel:
    """Loaded image data and display metadata for one README panel."""

    label: str
    image: NDArray[np.float64]
    extent_mm: tuple[float, float, float, float]
    xlim_mm: tuple[float, float]
    ylim_mm: tuple[float, float]
    parameter_label: str
    parameter_unit: str
    parameter_value: float
    theta_deg: float
    energy_kev: float


README_PANELS: Final[tuple[SweepPanel, ...]] = (
    # Frame 0 is phi = 0 deg (a symmetry azimuth), so the streaks are
    # mirror-symmetric about the specular rod. Off-axis frames (e.g. phi = 10
    # deg) are physically asymmetric and read poorly as a hero image.
    SweepPanel("SrTiO3", "sto_theta4_phi_sweep.npz", 0),
    SweepPanel("MgO", "mgo_theta2p2_phi_sweep.npz", 0),
    SweepPanel("Bi2Se3", "bi2se3_theta2p5_phi_sweep.npz", 0),
)


def _load_float_array(data: NpzFile, key: str) -> NDArray[np.float64]:
    """Load an array from an ``.npz`` file as ``float64``."""
    array: NDArray[np.float64] = np.asarray(data[key], dtype=np.float64)
    return array


def _load_float_scalar(data: NpzFile, key: str) -> float:
    """Load a scalar from an ``.npz`` file as ``float``."""
    value: float = float(np.asarray(data[key], dtype=np.float64).item())
    return value


def _load_text_scalar(data: NpzFile, key: str) -> str:
    """Load a scalar from an ``.npz`` file as ``str``."""
    value: str = str(np.asarray(data[key]).item())
    return value


def _extent_tuple(extent: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert a four-element NumPy extent into a typed tuple."""
    if extent.shape != (4,):
        raise ValueError(f"Expected extent shape (4,), got {extent.shape}.")
    typed_extent: tuple[float, float, float, float] = (
        float(extent[0]),
        float(extent[1]),
        float(extent[2]),
        float(extent[3]),
    )
    return typed_extent


def _limit_tuple(limits: NDArray[np.float64]) -> tuple[float, float]:
    """Convert a two-element NumPy axis limit into a typed tuple."""
    if limits.shape != (2,):
        raise ValueError(f"Expected limit shape (2,), got {limits.shape}.")
    typed_limits: tuple[float, float] = (
        float(limits[0]),
        float(limits[1]),
    )
    return typed_limits


def _parameter_display_name(name: str) -> tuple[str, str]:
    """Return a compact README label and unit for a sweep parameter."""
    if name == "phi_deg":
        return "phi", "deg"
    if name.endswith("_deg"):
        return name.removesuffix("_deg").replace("_", " "), "deg"
    return name.replace("_", " "), ""


def _load_panel(panel: SweepPanel) -> RenderedPanel:
    """Load one selected sweep frame for plotting."""
    source_path: Path = SWEEP_DIR / panel.filename
    data: NpzFile = cast(
        "NpzFile",
        np.load(source_path, allow_pickle=False),
    )
    try:
        image_bank: NDArray[np.float64] = _load_float_array(
            data,
            "image_bank",
        )
        parameter_values: NDArray[np.float64] = _load_float_array(
            data,
            "parameter_values",
        )
        frame_count: int = int(image_bank.shape[0])
        if not 0 <= panel.frame_index < frame_count:
            raise IndexError(
                f"{panel.filename} has {frame_count} frames; "
                f"cannot use frame {panel.frame_index}."
            )

        parameter_name: str = _load_text_scalar(data, "parameter_name")
        parameter_label: str
        parameter_unit: str
        parameter_label, parameter_unit = _parameter_display_name(
            parameter_name,
        )
        rendered: RenderedPanel = RenderedPanel(
            label=panel.label,
            image=image_bank[panel.frame_index],
            extent_mm=_extent_tuple(_load_float_array(data, "extent_mm")),
            xlim_mm=_limit_tuple(_load_float_array(data, "xlim")),
            ylim_mm=_limit_tuple(_load_float_array(data, "ylim")),
            parameter_label=parameter_label,
            parameter_unit=parameter_unit,
            parameter_value=float(parameter_values[panel.frame_index]),
            theta_deg=_load_float_scalar(data, "theta_deg"),
            energy_kev=_load_float_scalar(data, "energy_kev"),
        )
        return rendered
    finally:
        data.close()


def _display_window(
    image: NDArray[np.float64],
    extent_mm: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Zoom to the intensity-carrying region, keeping the specular centered.

    Uses intensity-weighted quantiles so faint rod tails do not blow the
    window up to the full detector, and mirrors the x-window about x = 0 so
    the specular rod stays visually centered.
    """
    height_px, width_px = image.shape
    x_mm = np.linspace(extent_mm[0], extent_mm[1], width_px)
    y_mm = np.linspace(extent_mm[2], extent_mm[3], height_px)
    rows, cols = np.where(image > 1e-3)
    if rows.size == 0:
        return (extent_mm[0], extent_mm[1]), (extent_mm[2], extent_mm[3])
    x_half = float(np.max(np.abs(x_mm[cols])))
    # Rods extend upward with faint tails; cap the window at the height
    # containing 98% of the above-threshold pixels instead of the extreme.
    y_top = float(np.percentile(y_mm[rows], 98.0))
    x_half = 1.3 * max(x_half, 5.0)
    y_top = 1.2 * max(y_top, 5.0)
    # Every panel gets the same window aspect (width : height) so the three
    # mm-true (aspect="equal") axes render as identical rectangles instead
    # of collapsing into thin columns when the content is taller than wide.
    target_ratio = 1.4
    if 2.0 * x_half < target_ratio * y_top:
        x_half = 0.5 * target_ratio * y_top
    else:
        y_top = 2.0 * x_half / target_ratio
    return (-x_half, x_half), (0.0, y_top)


def export_readme_gallery(output_path: Path = README_GALLERY) -> Path:
    """Render a compact README gallery from tutorial sweep banks."""
    panels: tuple[RenderedPanel, ...] = tuple(
        _load_panel(panel) for panel in README_PANELS
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure: Figure
    axes_raw: object
    figure, axes_raw = plt.subplots(
        1,
        len(panels),
        figsize=(11.5, 3.6),
        constrained_layout=True,
    )
    axes: list[Axes] = cast("list[Axes]", list(np.ravel(axes_raw)))
    figure.patch.set_facecolor("#050505")
    figure.suptitle(
        "Differentiable RHEED simulations",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    for axis, panel in zip(axes, panels, strict=True):
        parameter_value: str = f"{panel.parameter_value:.0f}"
        if panel.parameter_unit:
            parameter_value = f"{parameter_value} {panel.parameter_unit}"
        # Gamma-brighten the stored log-compressed frame so the fainter
        # diffracted rods are visible next to the specular spot.
        image_artist: AxesImage = axis.imshow(
            np.power(np.clip(panel.image, 0.0, 1.0), 0.4),
            cmap=create_phosphor_colormap(),
            extent=panel.extent_mm,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic",
        )
        image_artist.set_rasterized(True)
        axis.set_facecolor("#050505")
        axis.set_title(
            (
                f"{panel.label}\n"
                f"{panel.parameter_label}={parameter_value}"
            ),
            color="white",
            fontsize=11,
            pad=8,
        )
        window_x, window_y = _display_window(panel.image, panel.extent_mm)
        axis.set_xlim(window_x)
        axis.set_ylim(window_y)
        axis.set_xlabel("detector x (mm)", color="#d6d6d6", fontsize=8)
        axis.set_ylabel("detector y (mm)", color="#d6d6d6", fontsize=8)
        axis.tick_params(colors="#b8b8b8", labelsize=7, length=2)
        for spine in axis.spines.values():
            spine.set_color("#545454")
            spine.set_linewidth(0.6)
        axis.text(
            0.03,
            0.04,
            f"{panel.theta_deg:g} deg, {panel.energy_kev:g} keV",
            color="#ededed",
            fontsize=8,
            transform=axis.transAxes,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#050505",
                "edgecolor": "#3f3f3f",
                "alpha": 0.78,
            },
        )

    figure.savefig(
        output_path,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)
    return output_path


def main() -> int:
    """Generate all README figures."""
    output_path: Path = export_readme_gallery()
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
