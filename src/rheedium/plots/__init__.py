"""Plotting and visualization utilities for RHEED data.

Extended Summary
----------------
This module provides functions for visualizing RHEED patterns, crystal
structures, and physics concepts. It includes specialized colormaps that
simulate the appearance of phosphor screens used in experimental RHEED
setups, as well as pedagogical diagrams for documentation.

Routine Listings
----------------
create_phosphor_colormap : function
    Create a custom colormap that simulates a phosphor screen appearance.
plot_crystal_structure_3d : function
    Plot 3D crystal structure with atomic positions.
plot_ctr_profile : function
    Plot crystal truncation rod intensity profile.
plot_debye_waller : function
    Plot Debye-Waller damping factor at different temperatures.
plot_ewald_sphere_2d : function
    Plot 2D cross-section of Ewald sphere construction.
plot_ewald_sphere_3d : function
    Plot 3D visualization of Ewald sphere with reciprocal rods.
plot_form_factors : function
    Plot atomic form factors f(q) for multiple elements.
plot_grazing_incidence_geometry : function
    Plot grazing incidence geometry diagram for RHEED.
plot_rheed : function
    Interpolate RHEED spots onto a uniform grid and display.
plot_rod_broadening : function
    Plot lateral rod broadening for different correlation lengths.
plot_roughness_damping : function
    Plot surface roughness damping for different roughness values.
plot_structure_factor_phases : function
    Plot Argand diagram showing structure factor phase contributions.
plot_unit_cell_3d : function
    Plot 3D unit cell with lattice vectors.
plot_wavelength_curve : function
    Plot electron wavelength vs accelerating voltage.
view_atoms : function
    View atoms in a CrystalStructure with 3D visualization.

Notes
-----
Visualization functions are designed to closely match the appearance of
experimental RHEED patterns for direct comparison. Diagram functions
provide publication-quality figures for documentation and teaching.
"""

from .diagrams import (
    plot_crystal_structure_3d,
    plot_ctr_profile,
    plot_debye_waller,
    plot_ewald_sphere_2d,
    plot_ewald_sphere_3d,
    plot_form_factors,
    plot_grazing_incidence_geometry,
    plot_rod_broadening,
    plot_roughness_damping,
    plot_structure_factor_phases,
    plot_unit_cell_3d,
    plot_wavelength_curve,
    view_atoms,
)
from .figuring import create_phosphor_colormap, plot_rheed

__all__ = [
    "create_phosphor_colormap",
    "plot_crystal_structure_3d",
    "plot_ctr_profile",
    "plot_debye_waller",
    "plot_ewald_sphere_2d",
    "plot_ewald_sphere_3d",
    "plot_form_factors",
    "plot_grazing_incidence_geometry",
    "plot_rheed",
    "plot_rod_broadening",
    "plot_roughness_damping",
    "plot_structure_factor_phases",
    "plot_unit_cell_3d",
    "plot_wavelength_curve",
    "view_atoms",
]
