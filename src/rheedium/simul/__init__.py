"""RHEED pattern simulation utilities.

Extended Summary
----------------
This module provides functions for simulating RHEED patterns using both
kinematic and dynamical (multislice) approximations with surface physics. It
includes utilities for calculating electron wavelengths, scattering
intensities, crystal truncation rods (CTRs), and complete diffraction patterns
from crystal structures.

Routine Listings
----------------
:func:`atomic_scattering_factor`
    Combined form factor with Debye-Waller damping.
:func:`build_ewald_data`
    Build angle-independent EwaldData from crystal and beam parameters.
:func:`calculate_ctr_intensity`
    Calculate continuous intensity along crystal truncation rods.
:func:`compute_domain_extent`
    Compute domain extent from atomic positions bounding box.
:func:`compute_kinematic_intensities_with_ctrs`
    Calculate kinematic diffraction intensities with CTR contributions.
:func:`compute_shell_sigma`
    Compute Ewald shell Gaussian thickness from beam parameters.
:func:`debye_waller_factor`
    Calculate Debye-Waller damping factor for thermal vibrations.
:func:`ewald_allowed_reflections`
    Find reflections satisfying Ewald sphere condition for given beam angles.
:func:`extent_to_rod_sigma`
    Convert domain extent to reciprocal-space rod widths.
:func:`find_ctr_ewald_intersection`
    Find intersection of CTR with Ewald sphere for given (h, k) rod.
:func:`find_kinematic_reflections`
    Find kinematically allowed reflections for given experimental conditions.
:func:`finite_domain_intensities`
    Compute intensities with finite domain broadening.
:func:`gaussian_rod_profile`
    Gaussian lateral width profile of rods due to finite correlation length.
:func:`get_mean_square_displacement`
    Calculate mean square displacement for given temperature.
:func:`incident_wavevector`
    Calculate incident electron wavevector from beam parameters.
:func:`integrated_rod_intensity`
    Integrate CTR intensity over finite detector acceptance.
:func:`kinematic_ctr_simulator`
    RHEED simulation using continuous crystal truncation rods (streaks).
:func:`kinematic_simulator`
    Simulate RHEED pattern with proper atomic form factors and CTRs.
:func:`kinematic_spot_simulator`
    RHEED simulation using discrete 3D reciprocal lattice (spots).
:func:`kirkland_form_factor`
    Calculate atomic form factor f(q) using Kirkland parameterization.
:func:`load_kirkland_parameters`
    Load Kirkland scattering parameters from data file.
:func:`lorentzian_rod_profile`
    Lorentzian lateral width profile of rods due to finite correlation length.
:func:`make_ewald_sphere`
    Create incident wavevector k_in from beam parameters.
:func:`multislice_propagate`
    Propagate electron wave through potential slices via multislice.
:func:`multislice_simulator`
    Simulate RHEED pattern from potential slices using multislice (dynamical).
:func:`project_on_detector`
    Project reciprocal lattice points onto detector screen.
:func:`project_on_detector_geometry`
    Project reciprocal lattice points with full detector geometry support.
:func:`rod_ewald_overlap`
    Compute overlap between broadened rods and Ewald shell.
:func:`rod_profile_function`
    Lateral width profile of rods due to finite correlation length.
:func:`roughness_damping`
    Gaussian roughness damping factor for CTR intensities.
:func:`simple_structure_factor`
    Calculate structure factor F(G) for given G vector and atomic positions.
:func:`sliced_crystal_to_potential`
    Convert SlicedCrystal to PotentialSlices for multislice simulation.
:func:`surface_structure_factor`
    Calculate structure factor for surface with q_z dependence.
:func:`wavelength_ang`
    Calculate electron wavelength in angstroms.
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
from .simul_utils import (
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from .simulator import (
    compute_kinematic_intensities_with_ctrs,
    find_kinematic_reflections,
    kinematic_simulator,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    project_on_detector_geometry,
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
    "interaction_constant",
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
    "project_on_detector_geometry",
    "rod_ewald_overlap",
    "rod_profile_function",
    "roughness_damping",
    "simple_structure_factor",
    "sliced_crystal_to_potential",
    "surface_structure_factor",
    "wavelength_ang",
]
