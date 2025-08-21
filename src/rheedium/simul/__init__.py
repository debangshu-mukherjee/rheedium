"""
Module: simul
-------------
RHEED pattern simulation utilities.

Functions
---------
- `wavelength_ang`:
    Calculate electron wavelength in angstroms
- `incident_wavevector`:
    Calculate incident electron wavevector from beam parameters
- `project_on_detector`:
    Project reciprocal lattice points onto detector screen
- `find_kinematic_reflections`:
    Find kinematically allowed reflections for given experimental conditions
- `compute_kinematic_intensities`:
    Calculate kinematic diffraction intensities for reciprocal lattice points
- `simulate_rheed_pattern`:
    Complete RHEED pattern simulation from crystal structure to detector pattern
- `atomic_potential`:
    Calculate atomic scattering potential for given atomic number
- `crystal_potential`:
    Calculate multislice potential for a crystal structure
"""

from .simulator import (atomic_potential, compute_kinematic_intensities,
                        crystal_potential, find_kinematic_reflections,
                        incident_wavevector, project_on_detector,
                        simulate_rheed_pattern, wavelength_ang)

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
