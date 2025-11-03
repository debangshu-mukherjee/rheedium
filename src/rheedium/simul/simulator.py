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
    PotentialSlices,
    RHEEDPattern,
    create_potential_slices,
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
    phi_deg: scalar_float = 0.0,
) -> Float[Array, "3"]:
    """Calculate the incident electron wavevector for RHEED geometry.

    Parameters
    ----------
    lam_ang : scalar_float
        Electron wavelength in angstroms.
    theta_deg : scalar_float
        Grazing angle of incidence in degrees (angle from surface).
    phi_deg : scalar_float, optional
        Azimuthal angle in degrees (in-plane rotation).
        phi=0: beam along +x axis (default, gives horizontal streaks)
        phi=90: beam along +y axis (gives vertical streaks)
        Default: 0.0

    Returns
    -------
    k_in : Float[Array, "3"]
        Incident wavevector [k_x, k_y, k_z] in reciprocal angstroms.
        The beam propagates in the surface plane at azimuthal angle phi,
        with a downward z-component determined by the grazing angle theta.
    """
    k_magnitude: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta_rad: Float[Array, ""] = jnp.deg2rad(theta_deg)
    phi_rad: Float[Array, ""] = jnp.deg2rad(phi_deg)

    # In-plane component magnitude
    k_parallel: Float[Array, ""] = k_magnitude * jnp.cos(theta_rad)

    # Split in-plane component into x and y based on azimuthal angle
    k_x: Float[Array, ""] = k_parallel * jnp.cos(phi_rad)
    k_y: Float[Array, ""] = k_parallel * jnp.sin(phi_rad)
    k_z: Float[Array, ""] = -k_magnitude * jnp.sin(theta_rad)

    k_in: Float[Array, "3"] = jnp.array([k_x, k_y, k_z])
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out: Float[Array, "N 3"],
    detector_distance: scalar_float,
    k_in: Float[Array, "3"],
) -> Float[Array, "N 2"]:
    """Project output wavevectors onto detector plane.

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Array of output wavevectors.
    detector_distance : scalar_float
        Distance from sample to detector in angstroms.
    k_in : Float[Array, "3"]
        Incident wavevector used to determine detector orientation.
        The detector is perpendicular to the in-plane component of k_in.

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [horizontal, vertical] coordinates on detector plane in angstroms.
        Horizontal direction is perpendicular to beam (in surface plane).
        Vertical direction is along surface normal (z-direction).
    """
    # Get the in-plane (x-y) component of the incident beam direction
    beam_direction_xy: Float[Array, "2"] = k_in[:2]
    beam_mag_xy: Float[Array, ""] = jnp.linalg.norm(beam_direction_xy) + 1e-10

    # Normalized in-plane beam direction
    beam_unit_xy: Float[Array, "2"] = beam_direction_xy / beam_mag_xy

    # For detector projection: use the dominant in-plane direction
    # If beam is along x (phi=0), project onto y-z plane (detector sees y, z)
    # If beam is along y (phi=90), project onto x-z plane (detector sees x, z)

    # Calculate scale factor based on the in-plane beam component
    k_parallel: Float[Array, "N"] = (
        k_out[:, 0] * beam_unit_xy[0] + k_out[:, 1] * beam_unit_xy[1]
    )
    scale_factor: Float[Array, "N"] = detector_distance / (k_parallel + 1e-10)

    # Horizontal detector coordinate: perpendicular to beam in x-y plane
    # For phi=0 (beam along x): perpendicular is y-direction
    # For phi=90 (beam along y): perpendicular is x-direction
    detector_h: Float[Array, "N"] = (
        k_out[:, 0] * (-beam_unit_xy[1]) + k_out[:, 1] * beam_unit_xy[0]
    ) * scale_factor

    # Vertical detector coordinate: always z-direction
    detector_v: Float[Array, "N"] = k_out[:, 2] * scale_factor

    detector_coords: Float[Array, "N 2"] = jnp.stack(
        [detector_h, detector_v], axis=-1
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
def kinematic_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_num = 10.0,
    theta_deg: scalar_num = 2.0,
    phi_deg: scalar_num = 0.0,
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
        Grazing angle of incidence in degrees (angle from surface).
        Default: 2.0
    phi_deg : scalar_num, optional
        Azimuthal angle in degrees (in-plane rotation).
        phi=0: beam along +x axis (gives horizontal streaks)
        phi=90: beam along +y axis (gives vertical streaks)
        Default: 0.0
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
    phi_deg = jnp.asarray(phi_deg)
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
    k_in: Float[Array, "3"] = incident_wavevector(lam_ang, theta_deg, phi_deg)
    allowed_indices: Int[Array, "K"]
    k_out: Float[Array, "K 3"]
    allowed_indices, k_out = find_kinematic_reflections(
        k_in=k_in, gs=gs, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, "K 2"] = project_on_detector(
        k_out, detector_distance, k_in
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


@jaxtyped(typechecker=beartype)
def sliced_crystal_to_potential(
    sliced_crystal: "SlicedCrystal",
    slice_thickness: scalar_float = 2.0,
    pixel_size: scalar_float = 0.1,
    voltage_kv: scalar_float = 20.0,
) -> "PotentialSlices":
    """Convert a SlicedCrystal into PotentialSlices for multislice calculation.

    This function takes a surface-oriented crystal slab and generates 3D
    potential slices suitable for multislice electron diffraction simulations.
    The potential is calculated from atomic positions using proper scattering
    factors and projected onto a discrete grid.

    Parameters
    ----------
    sliced_crystal : SlicedCrystal
        Surface-oriented crystal structure with atoms and extents.
    slice_thickness : scalar_float, optional
        Thickness of each potential slice in Ångstroms. Default: 2.0 Å
        Determines the z-spacing between consecutive slices.
    pixel_size : scalar_float, optional
        Real-space pixel size in Ångstroms. Default: 0.1 Å
        Sets the lateral resolution of the potential grid.
    voltage_kv : scalar_float, optional
        Electron beam voltage in kV. Default: 20.0 kV
        Used for interaction constant calculation.

    Returns
    -------
    potential_slices : PotentialSlices
        3D potential array with calibration information.

    Algorithm
    ---------
    1. Determine grid dimensions from x_extent, y_extent, and pixel_size
    2. Calculate number of slices from depth and slice_thickness
    3. For each slice z-range:
       a. Select atoms within [z, z+slice_thickness]
       b. Project atomic potentials onto x-y grid
       c. Use proper scattering factors for each element
       d. Sum contributions from all atoms in slice
    4. Apply appropriate units (Volts or interaction potential)
    5. Return PotentialSlices with grid and calibration data

    Notes
    -----
    - The potential includes proper atomic scattering factors
    - Assumes independent atom approximation
    - Periodic boundary conditions in x-y plane
    - Non-periodic in z-direction (surface slab)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>>
    >>> # Create surface slab
    >>> bulk = rh.inout.parse_cif("SrTiO3.cif")
    >>> slab = rh.types.bulk_to_slice(
    ...     bulk_crystal=bulk,
    ...     orientation=jnp.array([1, 1, 1]),
    ...     depth=20.0
    ... )
    >>>
    >>> # Convert to potential slices
    >>> potential = rh.simul.sliced_crystal_to_potential(
    ...     sliced_crystal=slab,
    ...     slice_thickness=2.0,
    ...     pixel_size=0.1
    ... )
    """
    # Convert inputs to JAX arrays
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    pixel_size = jnp.asarray(pixel_size, dtype=jnp.float64)
    voltage_kv = jnp.asarray(voltage_kv, dtype=jnp.float64)

    # Extract crystal information
    positions: Float[Array, "N 3"] = sliced_crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = sliced_crystal.cart_positions[:, 3]
    x_extent: Float[Array, ""] = sliced_crystal.x_extent
    y_extent: Float[Array, ""] = sliced_crystal.y_extent
    depth: Float[Array, ""] = sliced_crystal.depth

    # Calculate grid dimensions
    nx: int = int(jnp.ceil(x_extent / pixel_size))
    ny: int = int(jnp.ceil(y_extent / pixel_size))
    nz: int = int(jnp.ceil(depth / slice_thickness))

    # Create coordinate grids
    x_coords: Float[Array, "nx"] = jnp.linspace(0, x_extent, nx)
    y_coords: Float[Array, "ny"] = jnp.linspace(0, y_extent, ny)

    # Create meshgrid for potential calculation
    X: Float[Array, "nx ny"]
    Y: Float[Array, "nx ny"]
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij")

    # Calculate wavelength and interaction constant
    wavelength: Float[Array, ""] = wavelength_ang(voltage_kv)

    # Interaction constant sigma = (2 * pi * m * e * lambda) / h^2
    # For practical calculations, use simplified form
    # Units: 1/Volt-Angstrom^2
    sigma: Float[Array, ""] = 2.0 * jnp.pi / (wavelength * voltage_kv * 1000.0)

    # Number of atoms
    n_atoms: int = positions.shape[0]

    def _calculate_slice_potential(slice_idx: int) -> Float[Array, "nx ny"]:
        """Calculate potential for a single slice."""
        z_start: Float[Array, ""] = slice_idx * slice_thickness
        z_end: Float[Array, ""] = (slice_idx + 1) * slice_thickness

        # Select atoms in this slice
        z_positions: Float[Array, "N"] = positions[:, 2]
        in_slice: Bool[Array, "N"] = jnp.logical_and(
            z_positions >= z_start, z_positions < z_end
        )

        def _atom_contribution(atom_idx: int) -> Float[Array, "nx ny"]:
            """Calculate contribution from single atom to potential.

            Uses masked computation to handle atoms outside slice.
            """
            # Get position and atomic number
            pos: Float[Array, "3"] = positions[atom_idx]
            Z: Float[Array, ""] = atomic_numbers[atom_idx]
            is_in_slice: Bool[Array, ""] = in_slice[atom_idx]

            # Distance from atom to each grid point
            dx: Float[Array, "nx ny"] = X - pos[0]
            dy: Float[Array, "nx ny"] = Y - pos[1]
            r: Float[Array, "nx ny"] = jnp.sqrt(dx**2 + dy**2 + 1e-10)

            # Projected potential (simplified Doyle-Turner approximation)
            # V(r) ~ Z / r * exp(-a*r^2) for each Gaussian in scattering factor
            # For simplicity, use screened Coulomb potential
            a: Float[Array, ""] = 0.5  # Screening parameter (Å^-2)
            V: Float[Array, "nx ny"] = (
                Z * sigma * jnp.exp(-a * r**2) / (r + 1e-10)
            )

            # Zero out contribution if atom not in slice
            return jnp.where(is_in_slice, V, 0.0)

        # Sum contributions from ALL atoms (masked by in_slice)
        atom_indices: Int[Array, "N"] = jnp.arange(n_atoms)
        contributions: Float[Array, "N nx ny"] = jax.vmap(_atom_contribution)(
            atom_indices
        )
        slice_potential: Float[Array, "nx ny"] = jnp.sum(contributions, axis=0)

        return slice_potential

    # Calculate all slices
    slice_indices: Int[Array, "nz"] = jnp.arange(nz)
    all_slices: Float[Array, "nz nx ny"] = jax.vmap(_calculate_slice_potential)(
        slice_indices
    )

    # Create PotentialSlices
    potential_slices: PotentialSlices = create_potential_slices(
        slices=all_slices,
        slice_thickness=slice_thickness,
        x_calibration=pixel_size,
        y_calibration=pixel_size,
    )

    return potential_slices


@beartype
def multislice_propagate(
    potential_slices: PotentialSlices,
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
) -> Array:
    """Propagate electron wave through potential slices using multislice algorithm.

    This implements the multislice algorithm for dynamical electron diffraction,
    which accounts for multiple scattering events. The algorithm alternates between:
    1. Transmission through a slice: ψ' = ψ × exp(iσV)
    2. Fresnel propagation to next slice: ψ → FFT⁻¹[FFT[ψ] × P(kx,ky)]

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials with shape (nz, nx, ny)
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts
    theta_deg : scalar_float
        Grazing incidence angle in degrees
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
        phi=0: beam along +x axis, phi=90: beam along +y axis

    Returns
    -------
    exit_wave : Float[Array, "nx ny"]
        Complex exit wave after propagation through all slices

    Notes
    -----
    The transmission function is:
        T(x,y) = exp(iσV(x,y))
    where σ = 2πme/(h²k) is the interaction constant.

    The Fresnel propagator in reciprocal space is:
        P(kx,ky,Δz) = exp(-iπλΔz(kx² + ky²))

    For RHEED geometry with grazing incidence, we:
    1. Start with a tilted plane wave
    2. Propagate through slices perpendicular to surface normal
    3. Account for the projection of k_in onto the surface

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy, 2nd ed.
    .. [2] Cowley & Moodie (1957). Acta Cryst. 10, 609-619.
    """
    # Extract potential array and parameters
    V_slices: Float[Array, "nz nx ny"] = potential_slices.slices
    dz: scalar_float = potential_slices.slice_thickness
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    nz: int = V_slices.shape[0]
    nx: int = V_slices.shape[1]
    ny: int = V_slices.shape[2]

    # Calculate electron wavelength
    lam_ang: scalar_float = wavelength_ang(voltage_kv)

    # Wave magnitude k = 2π/λ
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang

    # Calculate interaction constant σ = 2π/(λV)
    # This is a simplified form suitable for high-energy electrons
    # Units: 1/(Volt·Angstrom²)
    # Full form: σ = 2πme/(h²k), but for practical calculations we use:
    sigma: scalar_float = 2.0 * jnp.pi / (lam_ang * voltage_kv * 1000.0)

    # Set up coordinate grids in real space
    x: Float[Array, "nx"] = jnp.arange(nx) * dx
    y: Float[Array, "ny"] = jnp.arange(ny) * dy

    # Set up reciprocal space coordinates
    # kx, ky are in units of 1/Angstrom
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    KX: Float[Array, "nx ny"]
    KY: Float[Array, "nx ny"]
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

    # Initialize incident wave as tilted plane wave
    # For RHEED, we have grazing incidence at angle theta_deg
    # The wave propagates at angle theta from the surface
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)

    # Incident wavevector components
    k_in_x: scalar_float = k_mag * jnp.cos(theta_rad) * jnp.cos(phi_rad)
    k_in_y: scalar_float = k_mag * jnp.cos(theta_rad) * jnp.sin(phi_rad)
    k_in_z: scalar_float = k_mag * jnp.sin(theta_rad)

    # Create initial tilted plane wave: exp(i*k_in·r)
    X: Float[Array, "nx ny"]
    Y: Float[Array, "nx ny"]
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Initial wave has phase corresponding to propagation direction
    # At z=0, phase is k_in_x*x + k_in_y*y
    psi: Float[Array, "nx ny"] = jnp.exp(1j * (k_in_x * X + k_in_y * Y))

    # Fresnel propagator in reciprocal space
    # P(kx,ky) = exp(-iπλΔz(kx² + ky²))
    # This accounts for free-space propagation between slices
    propagator: Float[Array, "nx ny"] = jnp.exp(
        -1j * jnp.pi * lam_ang * dz * (KX**2 + KY**2)
    )

    # Multislice propagation loop
    def _propagate_one_slice(
        psi_in: Float[Array, "nx ny"],
        V_slice: Float[Array, "nx ny"]
    ) -> tuple[Float[Array, "nx ny"], None]:
        """Propagate through one slice: transmit then propagate.

        Parameters
        ----------
        psi_in : Float[Array, "nx ny"]
            Input wavefunction
        V_slice : Float[Array, "nx ny"]
            Potential for this slice

        Returns
        -------
        psi_out : Float[Array, "nx ny"]
            Output wavefunction after transmission and propagation
        None
            Dummy return for scan compatibility
        """
        # Step 1: Transmission through slice
        # T(x,y) = exp(iσV(x,y))
        transmission: Float[Array, "nx ny"] = jnp.exp(1j * sigma * V_slice)
        psi_transmitted: Float[Array, "nx ny"] = psi_in * transmission

        # Step 2: Fresnel propagation to next slice
        # ψ(z+Δz) = FFT⁻¹[FFT[ψ(z)] × P(kx,ky)]
        psi_k: Float[Array, "nx ny"] = jnp.fft.fft2(psi_transmitted)
        psi_k_propagated: Float[Array, "nx ny"] = psi_k * propagator
        psi_out: Float[Array, "nx ny"] = jnp.fft.ifft2(psi_k_propagated)

        return psi_out, None

    # Propagate through all slices using scan for efficiency
    psi_exit: Float[Array, "nx ny"]
    psi_exit, _ = jax.lax.scan(_propagate_one_slice, psi, V_slices)

    return psi_exit


@beartype
def multislice_simulator(
    potential_slices: PotentialSlices,
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance: scalar_float = 100.0,
    detector_width: scalar_float = 100.0,
    detector_height: scalar_float = 100.0,
    detector_pixels_x: int = 512,
    detector_pixels_y: int = 512,
) -> RHEEDPattern:
    """Simulate RHEED pattern from potential slices using multislice algorithm.

    This function implements the complete multislice RHEED simulation pipeline:
    1. Propagate electron wave through crystal (multislice_propagate)
    2. Extract exit wave at surface
    3. Apply Ewald sphere constraint for elastic scattering
    4. Project diffracted beams onto detector screen
    5. Calculate intensity distribution

    Parameters
    ----------
    potential_slices : PotentialSlices
        3D array of projected potentials from sliced_crystal_to_potential()
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts (typically 10-30 keV for RHEED)
    theta_deg : scalar_float
        Grazing incidence angle in degrees (typically 1-5°)
    phi_deg : scalar_float, optional
        Azimuthal angle of incident beam in degrees (default: 0.0)
    detector_distance : scalar_float, optional
        Distance from sample to detector screen in mm (default: 100.0)
    detector_width : scalar_float, optional
        Physical width of detector in mm (default: 100.0)
    detector_height : scalar_float, optional
        Physical height of detector in mm (default: 100.0)
    detector_pixels_x : int, optional
        Number of detector pixels in x direction (default: 512)
    detector_pixels_y : int, optional
        Number of detector pixels in y direction (default: 512)

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with detector coordinates and intensities

    Notes
    -----
    The multislice algorithm captures dynamical diffraction effects including:
    - Multiple scattering events
    - Absorption and inelastic processes (if imaginary potential included)
    - Thickness-dependent intensity oscillations
    - Kikuchi lines from diffuse scattering

    Unlike the kinematic approximation, multislice is quantitatively accurate
    for thick samples and strong scattering conditions.

    For RHEED geometry, the exit wave is projected onto the Ewald sphere
    to satisfy elastic scattering constraint |k_out| = |k_in|.

    See Also
    --------
    multislice_propagate : Core propagation algorithm
    simulate_rheed_pattern : Kinematic approximation simulator
    sliced_crystal_to_potential : Convert SlicedCrystal to potential slices

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy, 2nd ed.
    .. [2] Ichimiya & Cohen (2004). Reflection High-Energy Electron Diffraction
    """
    # Step 1: Propagate through crystal to get exit wave
    exit_wave: Float[Array, "nx ny"] = multislice_propagate(
        potential_slices=potential_slices,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )

    # Step 2: Fourier transform to get reciprocal space amplitude
    # The exit wave in real space becomes diffraction amplitude in k-space
    exit_wave_k: Float[Array, "nx ny"] = jnp.fft.fft2(exit_wave)

    # Get grid parameters
    nx: int = potential_slices.slices.shape[1]
    ny: int = potential_slices.slices.shape[2]
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration

    # Reciprocal space sampling
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    KX: Float[Array, "nx ny"]
    KY: Float[Array, "nx ny"]
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

    # Step 3: Apply Ewald sphere constraint
    # For elastic scattering: |k_out| = |k_in| = 2π/λ
    lam_ang: scalar_float = wavelength_ang(voltage_kv)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang

    # Incident wavevector
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)
    k_in: Float[Array, "3"] = k_mag * jnp.array([
        jnp.cos(theta_rad) * jnp.cos(phi_rad),
        jnp.cos(theta_rad) * jnp.sin(phi_rad),
        jnp.sin(theta_rad),
    ])

    # For each point in k-space, calculate k_out on Ewald sphere
    # k_out = k_in + G, where G = (kx, ky, kz)
    # Constraint: |k_out|² = k_mag²
    # This gives: kz = f(kx, ky)

    # Parallel components are direct: k_out_x = k_in_x + kx, k_out_y = k_in_y + ky
    k_out_x: Float[Array, "nx ny"] = k_in[0] + KX
    k_out_y: Float[Array, "nx ny"] = k_in[1] + KY

    # Perpendicular component from Ewald sphere:
    # k_out_z² = k_mag² - k_out_x² - k_out_y²
    k_out_z_squared: Float[Array, "nx ny"] = (
        k_mag**2 - k_out_x**2 - k_out_y**2
    )

    # Only real solutions (positive k_out_z²) correspond to propagating waves
    # Evanescent waves (k_out_z² < 0) don't reach detector
    valid_mask: Float[Array, "nx ny"] = k_out_z_squared > 0
    k_out_z: Float[Array, "nx ny"] = jnp.where(
        valid_mask, jnp.sqrt(k_out_z_squared), 0.0
    )

    # Step 4: Project onto detector
    # For RHEED, detector is typically perpendicular to incident beam
    # We use similar triangle geometry to project k-space to detector

    # Convert k-space coordinates to angles
    # For small angles: θ_x ≈ k_x / k_z, θ_y ≈ k_y / k_z
    theta_x: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_x / k_out_z, 0.0
    )
    theta_y: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_y / k_out_z, 0.0
    )

    # Detector position in mm
    # Origin at center of detector
    det_x: Float[Array, "nx ny"] = detector_distance * theta_x
    det_y: Float[Array, "nx ny"] = detector_distance * theta_y

    # Step 5: Calculate intensity on detector
    # Intensity = |amplitude|²
    intensity_k: Float[Array, "nx ny"] = jnp.abs(exit_wave_k) ** 2

    # Apply mask for evanescent waves
    intensity_k = jnp.where(valid_mask, intensity_k, 0.0)

    # Flatten arrays for RHEEDPattern output
    det_x_flat: Float[Array, "n"] = det_x.ravel()
    det_y_flat: Float[Array, "n"] = det_y.ravel()
    intensity_flat: Float[Array, "n"] = intensity_k.ravel()
    k_out_x_flat: Float[Array, "n"] = k_out_x.ravel()
    k_out_y_flat: Float[Array, "n"] = k_out_y.ravel()
    k_out_z_flat: Float[Array, "n"] = k_out_z.ravel()

    # Filter out zero intensities for efficiency
    nonzero_mask: Float[Array, "n"] = intensity_flat > 0
    det_x_filtered: Float[Array, "m"] = det_x_flat[nonzero_mask]
    det_y_filtered: Float[Array, "m"] = det_y_flat[nonzero_mask]
    intensity_filtered: Float[Array, "m"] = intensity_flat[nonzero_mask]

    # Reconstruct k_out vectors
    k_out_filtered: Float[Array, "m 3"] = jnp.column_stack([
        k_out_x_flat[nonzero_mask],
        k_out_y_flat[nonzero_mask],
        k_out_z_flat[nonzero_mask],
    ])

    # Create detector points array
    detector_points: Float[Array, "m 2"] = jnp.column_stack([
        det_x_filtered, det_y_filtered
    ])

    # Create dummy g_indices (not well-defined for multislice)
    # Use flattened grid indices instead
    grid_indices: Int[Array, "m"] = jnp.where(nonzero_mask)[0]

    # Create RHEEDPattern
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=grid_indices,
        k_out=k_out_filtered,
        detector_points=detector_points,
        intensities=intensity_filtered,
    )

    return pattern

