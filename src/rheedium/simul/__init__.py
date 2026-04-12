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
:func:`angular_divergence_average`
    Average pattern over Gaussian angular divergence distribution.
:func:`atomic_scattering_factor`
    Combined form factor with Debye-Waller damping.
:func:`build_ewald_data`
    Build angle-independent EwaldData from crystal and beam parameters.
:func:`build_transmission_function`
    Construct ``T(x,y) = exp(i sigma V_proj)`` from a complex projected
    potential.
:func:`crystal_projected_potential`
    Build complex projected potential V_real + i*V_abs for one
    multislice slice.
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
:func:`detector_psf_convolve`
    Convolve detector image with Gaussian point spread function.
:func:`energy_spread_average`
    Average pattern over Gaussian energy spread distribution.
:func:`ewald_allowed_reflections`
    Find reflections satisfying Ewald sphere condition for given beam angles.
:func:`extent_to_rod_sigma`
    Convert domain extent to reciprocal-space rod widths.
:func:`ewald_simulator`
    Simulate RHEED using exact Ewald sphere-CTR intersection (recommended).
:func:`ewald_simulator_with_orientation_distribution`
    Simulate and incoherently combine an orientation distribution of
    Ewald patterns.
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
:func:`instrument_broadened_pattern`
    Full instrument-averaged RHEED pattern combining all effects.
:func:`integrated_rod_intensity`
    Integrate CTR intensity over finite detector acceptance.
:func:`kinematic_spot_simulator`
    RHEED simulation using discrete 3D reciprocal lattice (spots).
:func:`kirkland_form_factor`
    Calculate atomic form factor f(q) using Kirkland parameterization.
:func:`kirkland_projected_potential`
    Calculate projected atomic potential using Kirkland parameterization.
:func:`load_lobato_parameters`
    Load Lobato-van Dyck scattering parameters from data file.
:func:`lobato_form_factor`
    Calculate atomic form factor f_e(q) using Lobato-van Dyck
    parameterization.
:func:`lobato_projected_potential`
    Calculate projected atomic potential using Lobato-van Dyck
    parameterization.
:func:`projected_potential`
    Projected potential with selectable parameterization.
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
:func:`rod_profile_function`
    Lateral width profile of rods due to finite correlation length.
:func:`roughness_damping`
    Gaussian roughness damping factor for CTR intensities.
:func:`sliced_crystal_to_projected_potential_slices`
    Convert SlicedCrystal to projected-potential slices for multislice
    simulation.
:func:`surface_structure_factor`
    Calculate structure factor for surface with q_z dependence.
"""

from .beam_averaging import (
    angular_divergence_average,
    coherence_envelope,
    detector_psf_convolve,
    energy_spread_average,
    instrument_broadened_pattern,
)
from .ewald import build_ewald_data, ewald_allowed_reflections
from .finite_domain import (
    compute_domain_extent,
    compute_shell_sigma,
    extent_to_rod_sigma,
    finite_domain_intensities,
)
from .form_factors import (
    atomic_scattering_factor,
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
    kirkland_projected_potential,
    load_lobato_parameters,
    lobato_form_factor,
    lobato_projected_potential,
    projected_potential,
)
from .kinematic import kinematic_spot_simulator, make_ewald_sphere
from .multislice import (
    build_transmission_function,
)
from .potential import crystal_projected_potential
from .simulator import (
    compute_kinematic_intensities_with_ctrs,
    ewald_simulator,
    ewald_simulator_with_orientation_distribution,
    find_ctr_ewald_intersection,
    find_kinematic_reflections,
    multislice_propagate,
    multislice_simulator,
    project_on_detector,
    project_on_detector_geometry,
    sliced_crystal_to_projected_potential_slices,
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

__all__: list[str] = [
    "angular_divergence_average",
    "atomic_scattering_factor",
    "build_ewald_data",
    "build_transmission_function",
    "coherence_envelope",
    "crystal_projected_potential",
    "calculate_ctr_intensity",
    "compute_domain_extent",
    "compute_kinematic_intensities_with_ctrs",
    "compute_shell_sigma",
    "debye_waller_factor",
    "detector_psf_convolve",
    "energy_spread_average",
    "ewald_allowed_reflections",
    "ewald_simulator",
    "ewald_simulator_with_orientation_distribution",
    "extent_to_rod_sigma",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "finite_domain_intensities",
    "gaussian_rod_profile",
    "get_mean_square_displacement",
    "instrument_broadened_pattern",
    "integrated_rod_intensity",
    "kinematic_spot_simulator",
    "kirkland_form_factor",
    "kirkland_projected_potential",
    "load_lobato_parameters",
    "lobato_form_factor",
    "lobato_projected_potential",
    "lorentzian_rod_profile",
    "projected_potential",
    "make_ewald_sphere",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector",
    "project_on_detector_geometry",
    "rod_profile_function",
    "roughness_damping",
    "sliced_crystal_to_projected_potential_slices",
    "surface_structure_factor",
]
