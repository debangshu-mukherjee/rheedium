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
    voltage_kv: float


README_PANELS: Final[tuple[SweepPanel, ...]] = (
    SweepPanel("SrTiO3", "sto_theta4_phi_sweep.npz", 2),
    SweepPanel("MgO", "mgo_theta2p2_phi_sweep.npz", 2),
    SweepPanel("Bi2Se3", "bi2se3_theta2p5_phi_sweep.npz", 2),
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
            voltage_kv=_load_float_scalar(data, "voltage_kv"),
        )
        return rendered
    finally:
        data.close()


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
        image_artist: AxesImage = axis.imshow(
            panel.image,
            cmap="inferno",
            extent=panel.extent_mm,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
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
        axis.set_xlim(panel.xlim_mm)
        axis.set_ylim(panel.ylim_mm)
        axis.set_xlabel("detector x (mm)", color="#d6d6d6", fontsize=8)
        axis.set_ylabel("detector y (mm)", color="#d6d6d6", fontsize=8)
        axis.tick_params(colors="#b8b8b8", labelsize=7, length=2)
        for spine in axis.spines.values():
            spine.set_color("#545454")
            spine.set_linewidth(0.6)
        axis.text(
            0.03,
            0.04,
            f"{panel.theta_deg:g} deg, {panel.voltage_kv:g} kV",
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
