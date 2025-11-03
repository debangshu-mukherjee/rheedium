"""RHEED pattern simulation utilities.

Extended Summary
----------------
This module provides functions for simulating RHEED patterns using both kinematic
and dynamical (multislice) approximations with surface physics. It includes utilities
for calculating electron wavelengths, scattering intensities, crystal truncation rods
(CTRs), and complete diffraction patterns from crystal structures.

Routine Listings
----------------
atomic_scattering_factor : function
    Combined form factor with Debye-Waller damping
calculate_ctr_intensity : function
    Calculate continuous intensity along crystal truncation rods
compute_kinematic_intensities_with_ctrs : function
    Calculate kinematic diffraction intensities with CTR contributions
debye_waller_factor : function
    Calculate Debye-Waller damping factor for thermal vibrations
find_kinematic_reflections : function
    Find kinematically allowed reflections for given experimental conditions
gaussian_rod_profile : function
    Gaussian lateral width profile of rods due to finite correlation length
get_mean_square_displacement : function
    Calculate mean square displacement for given temperature
incident_wavevector : function
    Calculate incident electron wavevector from beam parameters
integrated_rod_intensity : function
    Integrate CTR intensity over finite detector acceptance
kirkland_form_factor : function
    Calculate atomic form factor f(q) using Kirkland parameterization
load_kirkland_parameters : function
    Load Kirkland scattering parameters from data file
lorentzian_rod_profile : function
    Lorentzian lateral width profile of rods due to finite correlation length
kinematic_simulator : function
    Complete RHEED pattern simulation from crystal structure (kinematic)
multislice_propagate : function
    Propagate electron wave through potential slices using multislice algorithm
multislice_simulator : function
    Simulate RHEED pattern from potential slices using multislice (dynamical)
project_on_detector : function
    Project reciprocal lattice points onto detector screen
rod_profile_function : function
    Lateral width profile of rods due to finite correlation length
roughness_damping : function
    Gaussian roughness damping factor for CTR intensities
sliced_crystal_to_potential : function
    Convert SlicedCrystal to PotentialSlices for multislice simulation
surface_structure_factor : function
    Calculate structure factor for surface with q_z dependence
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
from .kinematic import (
    kinematic_detector_projection,
    kinematic_simulator,
    kinematic_structure_factor,
    paper_detector_projection,
    paper_kinematic_simulator,
    simple_structure_factor,
)
from .simulator import (
    compute_kinematic_intensities_with_ctrs,
    find_kinematic_reflections,
    incident_wavevector,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    sliced_crystal_to_potential,
    wavelength_ang,
)
from .surface_rods import (
    calculate_ctr_intensity,
    gaussian_rod_profile,
    integrated_rod_intensity,
    lorentzian_rod_profile,
    rod_profile_function,
    roughness_damping,
    surface_structure_factor,
)

__all__ = [
    "atomic_scattering_factor",
    "calculate_ctr_intensity",
    "compute_kinematic_intensities_with_ctrs",
    "debye_waller_factor",
    "find_kinematic_reflections",
    "gaussian_rod_profile",
    "get_mean_square_displacement",
    "incident_wavevector",
    "integrated_rod_intensity",
    "kinematic_detector_projection",
    "kinematic_simulator",
    "kinematic_structure_factor",
    "kirkland_form_factor",
    "load_kirkland_parameters",
    "lorentzian_rod_profile",
    "multislice_propagate",
    "multislice_simulator",
    "paper_detector_projection",
    "paper_kinematic_simulator",
    "project_on_detector",
    "rod_profile_function",
    "roughness_damping",
    "simple_structure_factor",
    "sliced_crystal_to_potential",
    "surface_structure_factor",
    "wavelength_ang",
]
