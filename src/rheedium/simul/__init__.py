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
build_ewald_data : function
    Build angle-independent EwaldData from crystal and beam parameters
calculate_ctr_intensity : function
    Calculate continuous intensity along crystal truncation rods
compute_domain_extent : function
    Compute domain extent from atomic positions bounding box
compute_kinematic_intensities_with_ctrs : function
    Calculate kinematic diffraction intensities with CTR contributions
compute_shell_sigma : function
    Compute Ewald shell Gaussian thickness from beam parameters
debye_waller_factor : function
    Calculate Debye-Waller damping factor for thermal vibrations
ewald_allowed_reflections : function
    Find reflections satisfying Ewald sphere condition for given beam angles
extent_to_rod_sigma : function
    Convert domain extent to reciprocal-space rod widths
find_ctr_ewald_intersection : function
    Find intersection of CTR with Ewald sphere for given (h, k) rod
find_kinematic_reflections : function
    Find kinematically allowed reflections for given experimental conditions
finite_domain_intensities : function
    Compute intensities with finite domain broadening
gaussian_rod_profile : function
    Gaussian lateral width profile of rods due to finite correlation length
get_mean_square_displacement : function
    Calculate mean square displacement for given temperature
incident_wavevector : function
    Calculate incident electron wavevector from beam parameters
integrated_rod_intensity : function
    Integrate CTR intensity over finite detector acceptance
kinematic_ctr_simulator : function
    RHEED simulation using continuous crystal truncation rods (streaks)
kinematic_simulator : function
    Simulate RHEED pattern with proper atomic form factors and CTRs
kinematic_spot_simulator : function
    RHEED simulation using discrete 3D reciprocal lattice (spots)
kirkland_form_factor : function
    Calculate atomic form factor f(q) using Kirkland parameterization
load_kirkland_parameters : function
    Load Kirkland scattering parameters from data file
lorentzian_rod_profile : function
    Lorentzian lateral width profile of rods due to finite correlation length
make_ewald_sphere : function
    Create incident wavevector k_in from beam parameters
multislice_propagate : function
    Propagate electron wave through potential slices using multislice algorithm
multislice_simulator : function
    Simulate RHEED pattern from potential slices using multislice (dynamical)
project_on_detector : function
    Project reciprocal lattice points onto detector screen
rod_ewald_overlap : function
    Compute overlap between broadened rods and Ewald shell
rod_profile_function : function
    Lateral width profile of rods due to finite correlation length
roughness_damping : function
    Gaussian roughness damping factor for CTR intensities
simple_structure_factor : function
    Calculate structure factor F(G) for given G vector and atomic positions
sliced_crystal_to_potential : function
    Convert SlicedCrystal to PotentialSlices for multislice simulation
surface_structure_factor : function
    Calculate structure factor for surface with q_z dependence
wavelength_ang : function
    Calculate electron wavelength in angstroms
"""

from .ewald import build_ewald_data, ewald_allowed_reflections
from .finite_domain import (
    compute_domain_extent,
    compute_shell_sigma,
    extent_to_rod_sigma,
    finite_domain_intensities,
    rod_ewald_overlap,
)
from .form_factors import (
    atomic_scattering_factor,
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
    load_kirkland_parameters,
)
from .kinematic import (
    find_ctr_ewald_intersection,
    kinematic_ctr_simulator,
    kinematic_spot_simulator,
    make_ewald_sphere,
    simple_structure_factor,
)
from .simul_utils import incident_wavevector, wavelength_ang
from .simulator import (
    compute_kinematic_intensities_with_ctrs,
    find_kinematic_reflections,
    kinematic_simulator,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    sliced_crystal_to_potential,
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
    "build_ewald_data",
    "calculate_ctr_intensity",
    "compute_domain_extent",
    "compute_kinematic_intensities_with_ctrs",
    "compute_shell_sigma",
    "debye_waller_factor",
    "ewald_allowed_reflections",
    "extent_to_rod_sigma",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "finite_domain_intensities",
    "gaussian_rod_profile",
    "get_mean_square_displacement",
    "incident_wavevector",
    "integrated_rod_intensity",
    "kinematic_ctr_simulator",
    "kinematic_simulator",
    "kinematic_spot_simulator",
    "kirkland_form_factor",
    "load_kirkland_parameters",
    "lorentzian_rod_profile",
    "make_ewald_sphere",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector",
    "rod_ewald_overlap",
    "rod_profile_function",
    "roughness_damping",
    "simple_structure_factor",
    "sliced_crystal_to_potential",
    "surface_structure_factor",
    "wavelength_ang",
]
