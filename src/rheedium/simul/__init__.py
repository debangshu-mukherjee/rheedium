"""RHEED pattern simulation utilities.

Extended Summary
----------------
This module provides functions for simulating RHEED patterns using kinematic
approximations. It includes utilities for calculating electron wavelengths,
scattering intensities, and complete diffraction patterns from crystal structures.

Routine Listings
----------------
wavelength_ang : function
    Calculate electron wavelength in angstroms
incident_wavevector : function
    Calculate incident electron wavevector from beam parameters
project_on_detector : function
    Project reciprocal lattice points onto detector screen
find_kinematic_reflections : function
    Find kinematically allowed reflections for given experimental conditions
compute_kinematic_intensities : function
    Calculate kinematic diffraction intensities for reciprocal lattice points
simulate_rheed_pattern : function
    Complete RHEED pattern simulation from crystal structure to detector pattern
atomic_potential : function
    Calculate atomic scattering potential for given atomic number
crystal_potential : function
    Calculate multislice potential for a crystal structure

Notes
-----
All functions support JAX transformations and automatic differentiation.
"""

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
    "wavelength_ang",
    "incident_wavevector",
    "project_on_detector",
    "find_kinematic_reflections",
    "compute_kinematic_intensities",
    "simulate_rheed_pattern",
    "atomic_potential",
    "crystal_potential",
]
