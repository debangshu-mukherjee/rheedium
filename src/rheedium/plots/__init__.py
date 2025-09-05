"""Plotting and visualization utilities for RHEED data.

Extended Summary
----------------
This module provides functions for visualizing RHEED patterns and crystal
structures. It includes specialized colormaps that simulate the appearance
of phosphor screens used in experimental RHEED setups.

Routine Listings
----------------
create_phosphor_colormap : function
    Create a custom colormap that simulates a phosphor screen appearance
plot_rheed : function
    Interpolate RHEED spots onto a uniform grid and display using phosphor colormap

Notes
-----
Visualization functions are designed to closely match the appearance of
experimental RHEED patterns for direct comparison.
"""

from .figuring import create_phosphor_colormap, plot_rheed

__all__ = [
    "create_phosphor_colormap",
    "plot_rheed",
]
