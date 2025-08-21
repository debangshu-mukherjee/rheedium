"""
Module: plots.figuring
----------------------
Functions for creating and customizing RHEED pattern visualizations.

Functions
---------
- `create_phosphor_colormap`:
    Create custom colormap simulating phosphor screen appearance
- `plot_rheed`:
    Plot RHEED pattern with interpolation and phosphor colormap
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import Any, List, Optional, Tuple
from jaxtyping import Float
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

import rheedium as rh
from rheedium._decorators import beartype
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
    matplotlib.colors.LinearSegmentedColormap
        Custom phosphor screen colormap.

    Algorithm
    ---------
    - Define color transition points and RGB values from black through dark green,
      bright green, lighter green, to white bloom
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
        (pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    green: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    blue: List[Tuple[scalar_float, scalar_float, scalar_float]] = [
        (pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    cmap: LinearSegmentedColormap = LinearSegmentedColormap(
        name, {"red": red, "green": green, "blue": blue}
    )
    return cmap


@beartype
def plot_rheed(
    rheed_pattern: RHEEDPattern,
    grid_size: Optional[int] = 200,
    interp_type: Optional[str] = "cubic",
    cmap_name: Optional[str] = "phosphor",
) -> None:
    """Interpolate the RHEED spots onto a uniform grid using various methods.

    Then display using the phosphor colormap.

    Parameters
    ----------
    rheed_pattern : RHEEDPattern
        Must have `detector_points` of shape (M, 2) and `intensities` of shape (M,).
    grid_size : int, optional
        Controls how many grid points in Y and Z directions. Default is 200.
    interp_type : str, optional
        Which interpolation approach to use. Default is "cubic". Options are:
        - "cubic" => calls griddata(..., method="cubic")
        - "linear" => calls griddata(..., method="linear")
        - "nearest" => calls griddata(..., method="nearest")
    cmap_name : str, optional
        Name for your custom phosphor colormap. Default is 'phosphor'.

    Algorithm
    ---------
    - Extract coordinates and intensities from RHEED pattern
    - Convert JAX arrays to NumPy arrays
    - Validate interpolation method
    - Calculate coordinate ranges for grid and create uniform grid points
    - Interpolate intensities onto grid using griddata
    - Reshape result to 2D grid
    - Create phosphor colormap
    - Create figure and plot with colorbar, labels, and title
    - Show plot

    Examples
    --------
    >>> from rheedium.plots.figuring import plot_rheed
    >>> from rheedium.types.rheed_types import RHEEDPattern
    >>> import jax.numpy as jnp
    >>> # Create a simple RHEED pattern
    >>> points = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    >>> intensities = jnp.array([1.0, 0.5, 0.5, 0.5, 0.5])
    >>> pattern = RHEEDPattern(points=points, intensities=intensities)
    >>> # Plot the pattern
    >>> plot_rheed(pattern, figsize=(6, 6))
    >>> plt.show()
    """
    coords: Float[np.ndarray, "M 2"] = rheed_pattern.detector_points
    yy: Float[np.ndarray, M] = coords[:, 0]
    zz: Float[np.ndarray, M] = coords[:, 1]
    intensities: Float[np.ndarray, M] = rheed_pattern.intensities
    y_np: np.ndarray = np.asarray(yy)
    z_np: np.ndarray = np.asarray(zz)
    i_np: np.ndarray = np.asarray(intensities)
    if interp_type in ("cubic", "linear", "nearest"):
        method: str = interp_type
    else:
        raise ValueError(
            f"interp_type must be one of: 'cubic', 'linear', or 'nearest'. Got: {interp_type}"
        )
    y_min: float = float(y_np.min())
    y_max: float = float(y_np.max())
    z_min: float = float(z_np.min())
    z_max: float = float(z_np.max())
    y_lin: np.ndarray = np.linspace(y_min, y_max, grid_size)
    z_lin: np.ndarray = np.linspace(z_min, z_max, grid_size)
    yg: np.ndarray
    zg: np.ndarray
    yg, zg = np.meshgrid(y_lin, z_lin, indexing="xy")
    grid_points: np.ndarray = np.column_stack([yg.ravel(), zg.ravel()])
    interpolated: np.ndarray = griddata(
        points=(y_np, z_np), values=i_np, xi=grid_points, method=method, fill_value=0.0
    )
    intensity_grid: np.ndarray = interpolated.reshape((grid_size, grid_size))
    phosphor_cmap: LinearSegmentedColormap = rh.inout.create_phosphor_colormap(
        cmap_name
    )
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(6, 6))
    cax: Any = ax.imshow(
        intensity_grid.T,
        origin="lower",
        cmap=phosphor_cmap,
        extent=[y_min, y_max, z_min, z_max],
        aspect="equal",
        interpolation="bilinear",
    )
    cbar: Any = fig.colorbar(cax, ax=ax)
    cbar.set_label("Interpolated Intensity (arb. units)")
    ax.set_title(f"RHEED Pattern ({method} interpolation)")
    ax.set_xlabel("Y (Å)")
    ax.set_ylabel("Z (Å)")
    plt.tight_layout()
    plt.show()
