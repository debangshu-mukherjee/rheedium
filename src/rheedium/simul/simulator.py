"""Functions for simulating RHEED patterns and diffraction patterns.

Extended Summary
----------------
This module provides functions for simulating Reflection High-Energy Electron
Diffraction (RHEED) patterns using kinematic approximations with proper atomic
form factors and surface physics. It includes utilities for calculating 
electron wavelengths, incident wavevectors, diffraction intensities with CTRs, 
and complete RHEED patterns from crystal structures.

Routine Listings
----------------
wavelength_ang : function
    Calculate electron wavelength in angstroms
incident_wavevector : function
    Calculate incident electron wavevector
project_on_detector : function
    Project wavevectors onto detector plane
find_kinematic_reflections : function
    Find reflections satisfying kinematic conditions
compute_kinematic_intensities_with_ctrs : function
    Calculate kinematic intensities with CTR contributions
simulate_rheed_pattern : function
    Simulate complete RHEED pattern with surface physics

Notes
-----
All functions support JAX transformations and automatic differentiation for
gradient-based optimization and inverse problems.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Bool, Float, Int, Num, jaxtyped

from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_rheed_pattern,
    scalar_float,
    scalar_int,
    scalar_num,
)
from rheedium.ucell import generate_reciprocal_points

from .form_factors import atomic_scattering_factor
from .surface_rods import integrated_rod_intensity

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def wavelength_ang(
    voltage_kv: Union[scalar_num, Num[Array, "..."]],
) -> Float[Array, "..."]:
    """Calculate the relativistic electron wavelength in angstroms.

    Parameters
    ----------
    voltage_kv : Union[scalar_num, Num[Array, "..."]]
        Electron energy in kiloelectron volts.
        Could be either a scalar or an array.

    Returns
    -------
    wavelength : Float[Array, "..."]
        Electron wavelength in angstroms.

    Notes
    -----
    Uses relativistic corrections for accurate wavelength at high energies.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> lam = rh.simul.wavelength_ang(jnp.asarray(20.0))  # 20 keV
    >>> print(f"λ = {lam:.4f} Å")
    λ = 0.0859 Å
    """
    rest_mass_energy_kev: Float[Array, "..."] = 511.0
    # Convert kV to V for the formula
    voltage_v: Float[Array, "..."] = voltage_kv * 1000.0
    corrected_voltage: Float[Array, "..."] = voltage_v * (
        1.0 + voltage_v / (2.0 * rest_mass_energy_kev * 1000.0)
    )
    h_over_2me: Float[Array, "..."] = 12.26
    wavelength: Float[Array, "..."] = h_over_2me / jnp.sqrt(corrected_voltage)
    return wavelength


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_float,
    theta_deg: scalar_float,
) -> Float[Array, "3"]:
    """Calculate the incident electron wavevector for RHEED geometry.

    Parameters
    ----------
    lam_ang : scalar_float
        Electron wavelength in angstroms.
    theta_deg : scalar_float
        Grazing angle of incidence in degrees.

    Returns
    -------
    k_in : Float[Array, "3"]
        Incident wavevector [k_x, k_y, k_z] in reciprocal angstroms.
        Direction along x, perpendicular to surface along z.
    """
    k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta_rad: Float[Array, ""] = jnp.deg2rad(theta_deg)
    k_x: Float[Array, ""] = k_magnitude * jnp.cos(theta_rad)
    k_y: Float[Array, ""] = 0.0
    k_z: Float[Array, ""] = -k_magnitude * jnp.sin(theta_rad)
    k_in: Float[Array, "3"] = jnp.array([k_x, k_y, k_z])
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out: Float[Array, "N 3"],
    detector_distance: scalar_float,
) -> Float[Array, "N 2"]:
    """Project output wavevectors onto detector plane.

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Array of output wavevectors.
    detector_distance : scalar_float
        Distance from sample to detector in angstroms.

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [x, y] coordinates on detector plane in angstroms.
    """
    scale_factor: Float[Array, "N"] = detector_distance / (
        k_out[:, 0] + 1e-10
    )
    detector_y: Float[Array, "N"] = k_out[:, 1] * scale_factor
    detector_z: Float[Array, "N"] = k_out[:, 2] * scale_factor
    detector_coords: Float[Array, "N 2"] = jnp.stack(
        [detector_y, detector_z], axis=-1
    )
    return detector_coords


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, "3"],
    gs: Float[Array, "M 3"],
    z_sign: scalar_float = 1.0,
    tolerance: scalar_float = 0.05,
) -> Tuple[Int[Array, "N"], Float[Array, "N 3"]]:
    """Find kinematically allowed reflections.

    Parameters
    ----------
    k_in : Float[Array, "3"]
        Incident wavevector.
    gs : Float[Array, "M 3"]
        Array of reciprocal lattice vectors.
    z_sign : scalar_float, optional
        If +1, keep reflections with positive z in k_out.
        If -1, keep reflections with negative z.
        Default: 1.0
    tolerance : scalar_float, optional
        Tolerance for reflection condition |k_out| = |k_in|.
        Default: 0.05

    Returns
    -------
    allowed_indices : Int[Array, "N"]
        Indices of allowed reflections in gs array.
    k_out : Float[Array, "N 3"]
        Output wavevectors for allowed reflections.
    """
    k_out_all: Float[Array, "M 3"] = k_in + gs
    k_in_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    k_out_mags: Float[Array, "M"] = jnp.linalg.norm(k_out_all, axis=1)
    elastic_condition: Bool[Array, "M"] = (
        jnp.abs(k_out_mags - k_in_mag) < tolerance
    )
    z_condition: Bool[Array, "M"] = k_out_all[:, 2] * z_sign > 0
    allowed: Bool[Array, "M"] = elastic_condition & z_condition
    allowed_indices: Int[Array, "N"] = jnp.where(allowed, size=gs.shape[0])[0]
    n_allowed: Int[Array, ""] = jnp.sum(allowed)
    allowed_indices = allowed_indices[:n_allowed]
    k_out: Float[Array, "N 3"] = k_out_all[allowed_indices]
    return allowed_indices, k_out


@jaxtyped(typechecker=beartype)
def compute_kinematic_intensities_with_ctrs(
    crystal: CrystalStructure,
    g_allowed: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    k_out: Float[Array, "N 3"],
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.5,
    detector_acceptance: scalar_float = 0.01,
    surface_fraction: scalar_float = 0.3,
) -> Float[Array, "N"]:
    """Calculate kinematic diffraction intensities with CTR contributions.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure containing atomic positions and types.
    g_allowed : Float[Array, "N 3"]
        Allowed reciprocal lattice vectors.
    k_in : Float[Array, "3"]
        Incident wavevector.
    k_out : Float[Array, "N 3"]
        Output wavevectors.
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors.
        Default: 300.0
    surface_roughness : scalar_float, optional
        RMS surface roughness in angstroms.
        Default: 0.5
    detector_acceptance : scalar_float, optional
        Detector angular acceptance in reciprocal angstroms.
        Default: 0.01
    surface_fraction : scalar_float, optional
        Fraction of atoms considered as surface atoms.
        Default: 0.3

    Returns
    -------
    intensities : Float[Array, "N"]
        Diffraction intensities for each allowed reflection.

    Algorithm
    ---------
    - Extract atomic positions and numbers from crystal
    - Determine surface atoms based on z-coordinate
    - For each allowed reflection:
        - Calculate momentum transfer q = k_out - k_in
        - Compute structure factor with proper form factors
        - Apply Debye-Waller factors (enhanced for surface atoms)
        - Add CTR contributions for surface reflections
    - Return normalized intensities
    """
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    z_coords: Float[Array, "M"] = atom_positions[:, 2]
    z_max: scalar_float = jnp.max(z_coords)
    z_min: scalar_float = jnp.min(z_coords)
    z_threshold: scalar_float = z_max - surface_fraction * (z_max - z_min)
    is_surface_atom: Bool[Array, "M"] = z_coords >= z_threshold

    def _calculate_reflection_intensity(
        idx: Int[Array, ""],
    ) -> Float[Array, ""]:
        g_vec: Float[Array, "3"] = g_allowed[idx]
        k_out_vec: Float[Array, "3"] = k_out[idx]
        q_vector: Float[Array, "3"] = k_out_vec - k_in

        def _atomic_contribution(
            atom_idx: Int[Array, ""],
        ) -> Float[Array, ""]:
            atomic_num: scalar_int = atomic_numbers[atom_idx]
            atom_pos: Float[Array, "3"] = atom_positions[atom_idx]
            is_surface: bool = is_surface_atom[atom_idx]
            
            form_factor: scalar_float = atomic_scattering_factor(
                atomic_number=atomic_num,
                q_vector=q_vector,
                temperature=temperature,
                is_surface=is_surface,
            )
            phase: scalar_float = jnp.dot(g_vec, atom_pos)
            contribution: complex = form_factor * jnp.exp(
                2.0j * jnp.pi * phase
            )
            return contribution

        n_atoms: Int[Array, ""] = atom_positions.shape[0]
        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contributions: Float[Array, "M"] = jax.vmap(_atomic_contribution)(
            atom_indices
        )
        structure_factor: complex = jnp.sum(contributions)
        
        # Calculate CTR contribution
        hk_index: Int[Array, "2"] = jnp.array(
            [jnp.round(g_vec[0]).astype(jnp.int32),
             jnp.round(g_vec[1]).astype(jnp.int32)]
        )
        q_z_value: Float[Array, ""] = q_vector[2]

        # Define integration range around q_z with detector acceptance
        q_z_range: Float[Array, "2"] = jnp.array([
            q_z_value - detector_acceptance,
            q_z_value + detector_acceptance
        ])

        ctr_intensity: scalar_float = integrated_rod_intensity(
            hk_index=hk_index,
            q_z_range=q_z_range,
            crystal=crystal,
            surface_roughness=surface_roughness,
            detector_acceptance=detector_acceptance,
            temperature=temperature,
        )
        
        kinematic_intensity: scalar_float = jnp.abs(structure_factor) ** 2
        total_intensity: scalar_float = kinematic_intensity + ctr_intensity
        
        return total_intensity

    n_reflections: Int[Array, ""] = g_allowed.shape[0]
    reflection_indices: Int[Array, "N"] = jnp.arange(n_reflections)
    intensities: Float[Array, "N"] = jax.vmap(
        _calculate_reflection_intensity
    )(reflection_indices)

    return intensities


@jaxtyped(typechecker=beartype)
def simulate_rheed_pattern(
    crystal: CrystalStructure,
    voltage_kv: scalar_num = 10.0,
    theta_deg: scalar_num = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    lmax: scalar_int = 1,
    tolerance: scalar_float = 0.05,
    detector_distance: scalar_float = 1000.0,
    z_sign: scalar_float = -1.0,
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.5,
    detector_acceptance: scalar_float = 0.01,
    surface_fraction: scalar_float = 0.3,
) -> RHEEDPattern:
    """Simulate RHEED pattern with proper atomic form factors and CTRs.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    voltage_kv : scalar_num, optional
        Electron beam energy in kiloelectron volts.
        Default: 20.0
    theta_deg : scalar_num, optional
        Grazing angle of incidence in degrees.
        Default: 2.0
    hmax : scalar_int, optional
        Maximum h Miller index for reciprocal point generation.
        Default: 3
    kmax : scalar_int, optional
        Maximum k Miller index for reciprocal point generation.
        Default: 3
    lmax : scalar_int, optional
        Maximum l Miller index for reciprocal point generation.
        Default: 1
    tolerance : scalar_float, optional
        Tolerance for reflection condition |k_out| = |k_in|.
        Default: 0.05
    detector_distance : scalar_float, optional
        Distance from sample to detector plane in angstroms.
        Default: 1000.0
    z_sign : scalar_float, optional
        If -1, keep reflections with negative z in k_out (standard RHEED).
        If +1, keep reflections with positive z.
        Default: -1.0
    temperature : scalar_float, optional
        Temperature in Kelvin for thermal factors.
        Default: 300.0
    surface_roughness : scalar_float, optional
        RMS surface roughness in angstroms.
        Default: 0.5
    detector_acceptance : scalar_float, optional
        Detector angular acceptance in reciprocal angstroms.
        Default: 0.01
    surface_fraction : scalar_float, optional
        Fraction of atoms considered as surface atoms.
        Default: 0.3

    Returns
    -------
    pattern : RHEEDPattern
        A NamedTuple capturing reflection indices, k_out, detector coords,
        and intensities with proper surface physics.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Load crystal structure from CIF file
    >>> crystal = rh.inout.parse_cif("path/to/crystal.cif")
    >>>
    >>> # Simulate RHEED pattern with surface physics
    >>> pattern = rh.simul.simulate_rheed_pattern(
    ...     crystal=crystal,
    ...     voltage_kv=20.0,
    ...     theta_deg=2.0,
    ...     temperature=300.0,
    ...     surface_roughness=0.8,
    ... )
    >>>
    >>> # Plot the pattern
    >>> rh.plots.plot_rheed(pattern, grid_size=400)

    Algorithm
    ---------
    - Generate reciprocal lattice points up to specified bounds
    - Calculate electron wavelength from voltage
    - Build incident wavevector at specified angle
    - Find G vectors satisfying reflection condition
    - Project resulting k_out onto detector plane
    - Calculate intensities with proper atomic form factors
    - Include CTR contributions for surface reflections
    - Apply surface-enhanced Debye-Waller factors
    - Create and return RHEEDPattern with computed data
    """
    # Convert scalar inputs to JAX arrays
    voltage_kv = jnp.asarray(voltage_kv)
    theta_deg = jnp.asarray(theta_deg)
    hmax = jnp.asarray(hmax, dtype=jnp.int32)
    kmax = jnp.asarray(kmax, dtype=jnp.int32)
    lmax = jnp.asarray(lmax, dtype=jnp.int32)

    gs: Float[Array, "M 3"] = generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    lam_ang: Float[Array, ""] = wavelength_ang(voltage_kv)
    k_in: Float[Array, "3"] = incident_wavevector(lam_ang, theta_deg)
    allowed_indices: Int[Array, "K"]
    k_out: Float[Array, "K 3"]
    allowed_indices, k_out = find_kinematic_reflections(
        k_in=k_in, gs=gs, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, "K 2"] = project_on_detector(
        k_out, detector_distance
    )
    g_allowed: Float[Array, "K 3"] = gs[allowed_indices]

    intensities: Float[Array, "K"] = compute_kinematic_intensities_with_ctrs(
        crystal=crystal,
        g_allowed=g_allowed,
        k_in=k_in,
        k_out=k_out,
        temperature=temperature,
        surface_roughness=surface_roughness,
        detector_acceptance=detector_acceptance,
        surface_fraction=surface_fraction,
    )
    
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern

