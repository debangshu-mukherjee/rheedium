"""RHEED pattern simulation utilities.

Extended Summary
----------------
This module provides functions for simulating RHEED patterns using kinematic
approximations. It includes utilities for calculating electron wavelengths,
scattering intensities, and complete diffraction patterns from crystal
structures.

Routine Listings
----------------
atomic_potential : function
    Calculate atomic scattering potential for given atomic number
atomic_scattering_factor : function
    Combined form factor with Debye-Waller damping
compute_kinematic_intensities : function
    Calculate kinematic diffraction intensities for reciprocal lattice points
crystal_potential : function
    Calculate multislice potential for a crystal structure
debye_waller_factor : function
    Calculate Debye-Waller damping factor for thermal vibrations
find_kinematic_reflections : function
    Find kinematically allowed reflections for given experimental conditions
get_mean_square_displacement : function
    Calculate mean square displacement for given temperature
incident_wavevector : function
    Calculate incident electron wavevector from beam parameters
kirkland_form_factor : function
    Calculate atomic form factor f(q) using Kirkland parameterization
load_kirkland_parameters : function
    Load Kirkland scattering parameters from data file
project_on_detector : function
    Project reciprocal lattice points onto detector screen
simulate_rheed_pattern : function
    Complete RHEED pattern simulation from crystal structure to detector
    pattern.
wavelength_ang : function
    Calculate electron wavelength in angstroms
"""

from .form_factors import (
    atomic_scattering_factor,
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
    load_kirkland_parameters,
)
from .simulator import (
    atomic_potential,
    compute_kinematic_intensities,
    crystal_potential,
    find_kinematic_reflections,
    incident_wavevector,
    project_on_detector,
    simulate_rheed_pattern,
    wavelength_ang,
)

__all__ = [
    "atomic_potential",
    "atomic_scattering_factor",
    "compute_kinematic_intensities",
    "crystal_potential",
    "debye_waller_factor",
    "find_kinematic_reflections",
    "get_mean_square_displacement",
    "incident_wavevector",
    "kirkland_form_factor",
    "load_kirkland_parameters",
    "project_on_detector",
    "simulate_rheed_pattern",
    "wavelength_ang",
]
