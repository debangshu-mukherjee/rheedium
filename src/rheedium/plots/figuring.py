import jax
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

import rheedium as rh
from rheedium.types import *

jax.config.update("jax_enable_x64", True)


@beartype
def create_phosphor_colormap(
    name: Optional[str] = "phosphor",
) -> LinearSegmentedColormap:
    """
    Description
    -----------
    Create a custom colormap that simulates a phosphor screen appearance.
    The colormap transitions from black through a bright phosphorescent green,
    with a slight white bloom at maximum intensity.

    Parameters
    ----------
    - `name` (str, optional):
        Name for the colormap.
        Default is 'phosphor'

    Returns
    -------
    - `matplotlib.colors.LinearSegmentedColormap`
        Custom phosphor screen colormap
    """
    colors: List[
        Tuple[scalar_float, Tuple[scalar_float, scalar_float, scalar_float]]
    ] = [
        (0.0, (0.0, 0.0, 0.0)),
        (0.4, (0.0, 0.05, 0.0)),
        (0.7, (0.15, 0.85, 0.15)),
        (0.9, (0.45, 0.95, 0.45)),
        (1.0, (0.8, 1.0, 0.8)),
    ]
    positions: List[scalar_float] = [x[0] for x in colors]
    rgb_values: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        x[1] for x in colors
    ]
    red: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values)
    ]
    green: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values)
    ]
    blue: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values)
    ]
    cmap = LinearSegmentedColormap(name, {"red": red, "green": green, "blue": blue})
    return cmap


def plot_rheed(
    rheed_pattern: RHEEDPattern,
    grid_size: Optional[int] = 200,
    interp_type: Optional[str] = "cubic",
    cmap_name: Optional[str] = "phosphor",
) -> None:
    """
    Description
    -----------
    Interpolate the RHEED spots onto a uniform grid using various methods,
    then display using the phosphor colormap.

    The parameter `interp_type` controls which interpolation method is used:
      - "cubic"    => calls griddata(..., method="cubic")
      - "linear"   => calls griddata(..., method="linear")
      - "nearest"  => calls griddata(..., method="nearest")

    Parameters
    ----------
    - `rheed_pattern` (RHEEDPattern)
        Must have `detector_points` of shape (M, 2) and
        `intensities` of shape (M,).
    - `grid_size` (int)
        Controls how many grid points in Y and Z directions.
        Default is 200.
    - `interp_type` (str)
        Which interpolation approach to use. Default is "cubic".
        The parameter `interp_type` controls which interpolation method is used:
        - "cubic"    => calls griddata(..., method="cubic")
        - "linear"   => calls griddata(..., method="linear")
        - "nearest"  => calls griddata(..., method="nearest")
    - `cmap_name` (str):
        Name for your custom phosphor colormap.
        Default is 'phosphor'.
    """
    coords = rheed_pattern.detector_points
    Y = coords[:, 0]
    Z = coords[:, 1]
    intensities = rheed_pattern.intensities
    Y_np = np.asarray(Y)
    Z_np = np.asarray(Z)
    I_np = np.asarray(intensities)
    if interp_type in ("cubic", "linear", "nearest"):
        method = interp_type
    else:
        raise ValueError(
            "interp_type must be one of: 'cubic', 'linear', or 'nearest'. "
            f"Got: {interp_type}"
        )
    y_min, y_max = float(Y_np.min()), float(Y_np.max())
    z_min, z_max = float(Z_np.min()), float(Z_np.max())
    y_lin = np.linspace(y_min, y_max, grid_size)
    z_lin = np.linspace(z_min, z_max, grid_size)
    Yg, Zg = np.meshgrid(y_lin, z_lin, indexing="xy")
    grid_points = np.column_stack([Yg.ravel(), Zg.ravel()])
    interpolated = griddata(
        points=(Y_np, Z_np), values=I_np, xi=grid_points, method=method, fill_value=0.0
    )
    intensity_grid = interpolated.reshape((grid_size, grid_size))
    phosphor_cmap = rh.inout.create_phosphor_colormap(cmap_name)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(
        intensity_grid.T,
        origin="lower",
        cmap=phosphor_cmap,
        extent=[y_min, y_max, z_min, z_max],
        aspect="equal",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Interpolated Intensity (arb. units)")
    ax.set_title(f"RHEED Pattern ({method} interpolation)")
    ax.set_xlabel("Y (Å)")
    ax.set_ylabel("Z (Å)")
    plt.tight_layout()
    plt.show()
