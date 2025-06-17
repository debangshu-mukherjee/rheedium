"""
Module: simul
-------------
RHEED pattern simulation utilities.

Functions
---------
- From `simulator.py` submodule:
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
"""

from .simulator import (atomic_potential, compute_kinematic_intensities,
                        find_kinematic_reflections, incident_wavevector,
                        project_on_detector, simulate_rheed_pattern)

__all__ = [
    "incident_wavevector",
    "project_on_detector",
    "find_kinematic_reflections",
    "compute_kinematic_intensities",
    "simulate_rheed_pattern",
    "atomic_potential",
]
