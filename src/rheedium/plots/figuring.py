"""Functions for creating and customizing RHEED pattern visualizations.

Extended Summary
----------------
This module provides specialized visualization functions for RHEED patterns,
including custom colormaps that simulate the phosphor screen appearance
commonly seen in experimental RHEED systems.

Routine Listings
----------------
create_phosphor_colormap : function
    Create custom colormap simulating phosphor screen appearance
plot_rheed : function
    Plot RHEED pattern with interpolation and phosphor colormap

Notes
-----
Visualization functions use matplotlib for rendering and scipy for
interpolation.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jaxtyping import Float
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

from rheedium.types import RHEEDPattern, scalar_float

jax.config.update("jax_enable_x64", True)


@beartype
def create_phosphor_colormap(
    name: Optional[str] = "phosphor",
) -> LinearSegmentedColormap:
    """Create a custom colormap that simulates a phosphor screen appearance.

    The colormap transitions from black through a bright phosphorescent green,
    with a slight white bloom at maximum intensity.

    Parameters
    ----------
    name : str, optional
        Name for the colormap. Default is 'phosphor'.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Custom phosphor screen colormap.

    Notes
    -----
    - Define color transition points and RGB values from black through dark
      green, bright green, lighter green, to white bloom.
    - Extract positions and RGB values from color definitions
    - Create color channel definitions for red, green, and blue
    - Create and return LinearSegmentedColormap with custom colors

    Examples
    --------
    >>> from rheedium.plots.figuring import create_phosphor_colormap
    >>> import matplotlib.pyplot as plt
    >>> # Create and display the colormap
    >>> cmap = create_phosphor_colormap()
    >>> plt.figure(figsize=(8, 1))
    >>> plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    >>> plt.title("Phosphor Screen Colormap")
    >>> plt.show()
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
        (pos, rgb[0], rgb[0])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    green: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[1], rgb[1])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    blue: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[2], rgb[2])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    cmap: LinearSegmentedColormap = LinearSegmentedColormap(
        name, {"red": red, "green": green, "blue": blue}
    )
    return cmap


@beartype
def plot_rheed(
    rheed_pattern: RHEEDPattern,
    grid_size: int = 200,
    interp_type: str = "gaussian",
    cmap_name: Optional[str] = "phosphor",
    spot_width: float = 0.08,
    figsize: Tuple[float, float] = (8, 10),
    x_extent: Optional[Tuple[float, float]] = None,
    y_extent: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot RHEED pattern with multiple rendering options.

    Description
    -----------
    Renders RHEED pattern to 2D image using interpolation or Gaussian
    broadening, then displays with phosphor-screen colormap.

    Parameters
    ----------
    rheed_pattern : RHEEDPattern
        Pattern with detector_points (N, 2) and intensities (N,).
    grid_size : int, optional
        Number of pixels along each axis. Default: 200
    interp_type : str, optional
        Rendering method. Default: "gaussian"
        - "gaussian": Sum of Gaussian spots (realistic, recommended)
        - "cubic": Cubic interpolation
        - "linear": Linear interpolation
        - "nearest": Nearest neighbor
    cmap_name : str, optional
        Colormap name. Default: "phosphor"
    spot_width : float, optional
        Gaussian spot width in mm (only for interp_type="gaussian").
        Default: 0.08
    figsize : Tuple[float, float], optional
        Figure size. Default: (8, 10)
    x_extent : Tuple[float, float], optional
        X-axis range (min, max) in mm. Default: auto from data with padding
    y_extent : Tuple[float, float], optional
        Y-axis range (min, max) in mm. Default: auto from data with padding
    """
    coords: Float[np.ndarray, "N 2"] = np.asarray(rheed_pattern.detector_points)
    x_np: Float[np.ndarray, "N"] = coords[:, 0]
    y_np: Float[np.ndarray, "N"] = coords[:, 1]
    i_np: Float[np.ndarray, "N"] = np.asarray(rheed_pattern.intensities)

    if x_extent is None:
        x_min: float = float(x_np.min()) - 0.5
        x_max: float = float(x_np.max()) + 0.5
    else:
        x_min, x_max = x_extent

    if y_extent is None:
        y_min: float = float(y_np.min()) - 0.5
        y_max: float = float(y_np.max()) + 0.5
    else:
        y_min, y_max = y_extent

    x_axis: Float[np.ndarray, "W"] = np.linspace(x_min, x_max, grid_size)
    y_axis: Float[np.ndarray, "H"] = np.linspace(y_min, y_max, grid_size)

    if interp_type == "gaussian":
        xx: Float[np.ndarray, "H W"]
        yy: Float[np.ndarray, "H W"]
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
        image: Float[np.ndarray, "H W"] = np.zeros((grid_size, grid_size))

        for idx in range(len(i_np)):
            x0: float = x_np[idx]
            y0: float = y_np[idx]
            i0: float = i_np[idx]
            image += i0 * np.exp(
                -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * spot_width ** 2)
            )

    elif interp_type in ("cubic", "linear", "nearest"):
        points: Float[np.ndarray, "N 2"] = np.column_stack([x_np, y_np])
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
        grid_points: Float[np.ndarray, "M 2"] = np.column_stack(
            [xx.ravel(), yy.ravel()]
        )
        image_flat: Float[np.ndarray, "M"] = griddata(
            points, i_np, grid_points, method=interp_type, fill_value=0.0
        )
        image = image_flat.reshape((grid_size, grid_size))

    else:
        raise ValueError(
            f"interp_type must be 'gaussian', 'cubic', 'linear', or 'nearest'."
            f"Got: {interp_type}"
        )

    if cmap_name == "phosphor":
        cmap = create_phosphor_colormap()
    else:
        cmap = plt.get_cmap(cmap_name)

    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        image,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    ax.set_xlabel("x_d (mm)")
    ax.set_ylabel("y_d (mm)")
    ax.set_title(f"RHEED Pattern ({interp_type})")
    plt.colorbar(im, ax=ax, label="Intensity (arb. units)")
    plt.show()