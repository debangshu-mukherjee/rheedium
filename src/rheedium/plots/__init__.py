"""
Module: plots
-------------
Plotting and visualization utilities for RHEED data.

Functions
---------
- `create_phosphor_colormap`:
    Create a custom colormap that simulates a phosphor screen appearance
- `plot_rheed`:
    Interpolate RHEED spots onto a uniform grid and display using phosphor colormap
"""

from .figuring import create_phosphor_colormap, plot_rheed

__all__ = [
    "create_phosphor_colormap",
    "plot_rheed",
]
