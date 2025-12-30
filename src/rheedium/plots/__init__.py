"""Plotting and visualization utilities for RHEED data.

Extended Summary
----------------
This module provides functions for visualizing RHEED patterns, crystal
structures, and physics concepts. It includes specialized colormaps that
simulate the appearance of phosphor screens used in experimental RHEED
setups, as well as pedagogical diagrams for documentation.

Routine Listings
----------------
:func:`create_phosphor_colormap`
    Create a custom colormap that simulates a phosphor screen appearance.
:func:`plot_crystal_structure_3d`
    Plot 3D crystal structure with atomic positions.
:func:`plot_ctr_profile`
    Plot crystal truncation rod intensity profile.
:func:`plot_debye_waller`
    Plot Debye-Waller damping factor at different temperatures.
:func:`plot_ewald_sphere_2d`
    Plot 2D cross-section of Ewald sphere construction.
:func:`plot_ewald_sphere_3d`
    Plot 3D visualization of Ewald sphere with reciprocal rods.
:func:`plot_form_factors`
    Plot atomic form factors f(q) for multiple elements.
:func:`plot_grazing_incidence_geometry`
    Plot grazing incidence geometry diagram for RHEED.
:func:`plot_rheed`
    Interpolate RHEED spots onto a uniform grid and display.
:func:`plot_rod_broadening`
    Plot lateral rod broadening for different correlation lengths.
:func:`plot_roughness_damping`
    Plot surface roughness damping for different roughness values.
:func:`plot_structure_factor_phases`
    Plot Argand diagram showing structure factor phase contributions.
:func:`plot_unit_cell_3d`
    Plot 3D unit cell with lattice vectors.
:func:`plot_wavelength_curve`
    Plot electron wavelength vs accelerating voltage.
:func:`view_atoms`
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
