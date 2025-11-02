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
atomic_potential : function
    Calculate atomic potential for multislice calculations
crystal_potential : function
    Calculate multislice potential for a crystal structure

Notes
-----
All functions support JAX transformations and automatic differentiation for
gradient-based optimization and inverse problems.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
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
from rheedium.ucell import bessel_kv, generate_reciprocal_points

from .form_factors import atomic_scattering_factor
from .surface_rods import integrated_rod_intensity

jax.config.update("jax_enable_x64", True)
DEFAULT_KIRKLAND_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "Kirkland_Potentials.csv"
)


@jaxtyped(typechecker=beartype)
def wavelength_ang(
    voltage_kv: Union[scalar_num, Num[Array, " ..."]],
) -> Float[Array, " ..."]:
    """Calculate the relativistic electron wavelength in angstroms.

    Parameters
    ----------
    voltage_kv : Union[scalar_num, Num[Array, " ..."]]
        Electron energy in kiloelectron volts.
        Could be either a scalar or an array.

    Returns
    -------
    wavelength : Float[Array, " ..."]
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
    rest_mass_energy_kev: Float[Array, " ..."] = 511.0
    # Convert kV to V for the formula
    voltage_v: Float[Array, " ..."] = voltage_kv * 1000.0
    corrected_voltage: Float[Array, " ..."] = voltage_v * (
        1.0 + voltage_v / (2.0 * rest_mass_energy_kev * 1000.0)
    )
    h_over_2me: Float[Array, " ..."] = 12.26
    wavelength: Float[Array, " ..."] = h_over_2me / jnp.sqrt(corrected_voltage)
    return wavelength


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_float,
    theta_deg: scalar_float,
) -> Float[Array, " 3"]:
    """Calculate the incident electron wavevector for RHEED geometry.

    Parameters
    ----------
    lam_ang : scalar_float
        Electron wavelength in angstroms.
    theta_deg : scalar_float
        Grazing angle of incidence in degrees.

    Returns
    -------
    k_in : Float[Array, " 3"]
        Incident wavevector [k_x, k_y, k_z] in reciprocal angstroms.
        Direction along x, perpendicular to surface along z.
    """
    k_magnitude: Float[Array, " "] = 2.0 * jnp.pi / lam_ang
    theta_rad: Float[Array, " "] = jnp.deg2rad(theta_deg)
    k_x: Float[Array, " "] = k_magnitude * jnp.cos(theta_rad)
    k_y: Float[Array, " "] = 0.0
    k_z: Float[Array, " "] = -k_magnitude * jnp.sin(theta_rad)
    k_in: Float[Array, " 3"] = jnp.array([k_x, k_y, k_z])
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out: Float[Array, " N 3"],
    detector_distance: scalar_float,
) -> Float[Array, " N 2"]:
    """Project output wavevectors onto detector plane.

    Parameters
    ----------
    k_out : Float[Array, " N 3"]
        Array of output wavevectors.
    detector_distance : scalar_float
        Distance from sample to detector in angstroms.

    Returns
    -------
    detector_coords : Float[Array, " N 2"]
        [x, y] coordinates on detector plane in angstroms.
    """
    scale_factor: Float[Array, " N"] = detector_distance / (
        k_out[:, 0] + 1e-10
    )
    detector_y: Float[Array, " N"] = k_out[:, 1] * scale_factor
    detector_z: Float[Array, " N"] = k_out[:, 2] * scale_factor
    detector_coords: Float[Array, " N 2"] = jnp.stack(
        [detector_y, detector_z], axis=-1
    )
    return detector_coords


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, " 3"],
    gs: Float[Array, " M 3"],
    z_sign: scalar_float = 1.0,
    tolerance: scalar_float = 0.05,
) -> Tuple[Int[Array, " N"], Float[Array, " N 3"]]:
    """Find kinematically allowed reflections.

    Parameters
    ----------
    k_in : Float[Array, " 3"]
        Incident wavevector.
    gs : Float[Array, " M 3"]
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
    allowed_indices : Int[Array, " N"]
        Indices of allowed reflections in gs array.
    k_out : Float[Array, " N 3"]
        Output wavevectors for allowed reflections.
    """
    k_out_all: Float[Array, " M 3"] = k_in + gs
    k_in_mag: Float[Array, " "] = jnp.linalg.norm(k_in)
    k_out_mags: Float[Array, " M"] = jnp.linalg.norm(k_out_all, axis=1)
    elastic_condition: Bool[Array, " M"] = (
        jnp.abs(k_out_mags - k_in_mag) < tolerance
    )
    z_condition: Bool[Array, " M"] = k_out_all[:, 2] * z_sign > 0
    allowed: Bool[Array, " M"] = elastic_condition & z_condition
    allowed_indices: Int[Array, " N"] = jnp.where(allowed, size=gs.shape[0])[0]
    n_allowed: Int[Array, " "] = jnp.sum(allowed)
    allowed_indices = allowed_indices[:n_allowed]
    k_out: Float[Array, " N 3"] = k_out_all[allowed_indices]
    return allowed_indices, k_out


@jaxtyped(typechecker=beartype)
def compute_kinematic_intensities_with_ctrs(
    crystal: CrystalStructure,
    g_allowed: Float[Array, " N 3"],
    k_in: Float[Array, " 3"],
    k_out: Float[Array, " N 3"],
    temperature: scalar_float = 300.0,
    surface_roughness: scalar_float = 0.5,
    detector_acceptance: scalar_float = 0.01,
    surface_fraction: scalar_float = 0.3,
) -> Float[Array, " N"]:
    """Calculate kinematic diffraction intensities with CTR contributions.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure containing atomic positions and types.
    g_allowed : Float[Array, " N 3"]
        Allowed reciprocal lattice vectors.
    k_in : Float[Array, " 3"]
        Incident wavevector.
    k_out : Float[Array, " N 3"]
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
    intensities : Float[Array, " N"]
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
    atom_positions: Float[Array, " M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, " M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    z_coords: Float[Array, " M"] = atom_positions[:, 2]
    z_max: scalar_float = jnp.max(z_coords)
    z_min: scalar_float = jnp.min(z_coords)
    z_threshold: scalar_float = z_max - surface_fraction * (z_max - z_min)
    is_surface_atom: Bool[Array, " M"] = z_coords >= z_threshold

    def _calculate_reflection_intensity(
        idx: Int[Array, " "],
    ) -> Float[Array, " "]:
        g_vec: Float[Array, " 3"] = g_allowed[idx]
        k_out_vec: Float[Array, " 3"] = k_out[idx]
        q_vector: Float[Array, " 3"] = k_out_vec - k_in

        def _atomic_contribution(
            atom_idx: Int[Array, " "],
        ) -> Float[Array, " "]:
            atomic_num: scalar_int = atomic_numbers[atom_idx]
            atom_pos: Float[Array, " 3"] = atom_positions[atom_idx]
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

        n_atoms: Int[Array, " "] = atom_positions.shape[0]
        atom_indices: Int[Array, " M"] = jnp.arange(n_atoms)
        contributions: Float[Array, " M"] = jax.vmap(_atomic_contribution)(
            atom_indices
        )
        structure_factor: complex = jnp.sum(contributions)
        
        # Calculate CTR contribution
        hk_index: Int[Array, " 2"] = jnp.array(
            [jnp.round(g_vec[0]).astype(jnp.int32),
             jnp.round(g_vec[1]).astype(jnp.int32)]
        )
        q_z_value: Float[Array, " "] = q_vector[2]

        # Define integration range around q_z with detector acceptance
        q_z_range: Float[Array, " 2"] = jnp.array([
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

    n_reflections: Int[Array, " "] = g_allowed.shape[0]
    reflection_indices: Int[Array, " N"] = jnp.arange(n_reflections)
    intensities: Float[Array, " N"] = jax.vmap(
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
    z_sign: scalar_float = 1.0,
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
        If +1, keep reflections with positive z in k_out.
        Default: 1.0
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

    gs: Float[Array, " M 3"] = generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    lam_ang: Float[Array, " "] = wavelength_ang(voltage_kv)
    k_in: Float[Array, " 3"] = incident_wavevector(lam_ang, theta_deg)
    allowed_indices: Int[Array, " K"]
    k_out: Float[Array, " K 3"]
    allowed_indices, k_out = find_kinematic_reflections(
        k_in=k_in, gs=gs, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, " K 2"] = project_on_detector(
        k_out, detector_distance
    )
    g_allowed: Float[Array, " K 3"] = gs[allowed_indices]

    intensities: Float[Array, " K"] = compute_kinematic_intensities_with_ctrs(
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
def atomic_potential(
    atom_no: scalar_int,
    pixel_size: scalar_float,
    grid_shape: Optional[Tuple[scalar_int, scalar_int]] = None,
    center_coords: Optional[Float[Array, " 2"]] = None,
    sampling: Optional[scalar_int] = 16,
    potential_extent: Optional[scalar_float] = 4.0,
    datafile: Optional[str] = str(DEFAULT_KIRKLAND_PATH),
) -> Float[Array, " h w"]:
    """Calculate the projected Kirklans potential of a single atom.

    The potential can be centered at arbitrary coordinates within a
    custom grid.

    Parameters
    ----------
    atom_no : scalar_int
        Atomic number of the atom whose potential is being calculated
    pixel_size : scalar_float
        Real space pixel size in Ångstroms
    grid_shape : Tuple[scalar_int, scalar_int], optional
        Shape of the output grid (height, width). If None, calculated from
        potential_extent.
    center_coords : Float[Array, " 2"], optional
        (x, y) coordinates in Ångstroms where atom should be centered.
        If None, centers at grid center
    sampling : scalar_int, optional
        Supersampling factor for increased accuracy. Default is 16
    potential_extent : scalar_float, optional
        Distance in Ångstroms from atom center to calculate potential.
        Default is 4.0 Å.
    datafile : str, optional
        Path to CSV file containing Kirkland scattering factors

    Returns
    -------
    potential : Float[Array, " h w"]
        Projected potential matrix with atom centered at specified coordinates

    Notes
    -----
    The algorithm proceeds as follows:

    1. Define physical constants and load Kirkland parameters
    2. Determine grid size and center coordinates
    3. Calculate step size for supersampling
    4. Create coordinate grid with atom centered at specified position
    5. Calculate radial distances from atom center
    6. Compute Bessel and Gaussian terms using Kirkland parameters
    7. Combine terms to get total potential
    8. Downsample to target resolution using average pooling
    9. Return final potential matrix
    """
    a0: Float[Array, " "] = jnp.asarray(0.5292)
    ek: Float[Array, " "] = jnp.asarray(14.4)
    term1: Float[Array, " "] = 4.0 * (jnp.pi**2) * a0 * ek
    term2: Float[Array, " "] = 2.0 * (jnp.pi**2) * a0 * ek
    kirkland_df: pd.DataFrame = pd.read_csv(datafile, header=None)
    kirkland_array: Float[Array, " 103 12"] = jnp.array(kirkland_df.values)
    kirk_params: Float[Array, " 12"] = kirkland_array[atom_no - 1, :]
    step_size: Float[Array, " "] = pixel_size / sampling
    if grid_shape is None:
        grid_extent: Float[Array, " "] = potential_extent
        n_points: Int[Array, " "] = jnp.ceil(
            2.0 * grid_extent / step_size
        ).astype(jnp.int32)
        grid_height: Int[Array, " "] = n_points
        grid_width: Int[Array, " "] = n_points
    else:
        grid_height: Int[Array, " "] = jnp.asarray(
            grid_shape[0] * sampling, dtype=jnp.int32
        )
        grid_width: Int[Array, " "] = jnp.asarray(
            grid_shape[1] * sampling, dtype=jnp.int32
        )
    if center_coords is None:
        center_x: Float[Array, " "] = 0.0
        center_y: Float[Array, " "] = 0.0
    else:
        center_x: Float[Array, " "] = center_coords[0]
        center_y: Float[Array, " "] = center_coords[1]
    y_coords: Float[Array, " h"] = (
        jnp.arange(grid_height) - grid_height // 2
    ) * step_size + center_y
    x_coords: Float[Array, " w"] = (
        jnp.arange(grid_width) - grid_width // 2
    ) * step_size + center_x
    ya: Float[Array, " h w"]
    xa: Float[Array, " h w"]
    ya, xa = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    r: Float[Array, " h w"] = jnp.sqrt(
        (xa - center_x) ** 2 + (ya - center_y) ** 2
    )
    bessel_term1: Float[Array, " h w"] = kirk_params[0] * bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[1]) * r
    )
    bessel_term2: Float[Array, " h w"] = kirk_params[2] * bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[3]) * r
    )
    bessel_term3: Float[Array, " h w"] = kirk_params[4] * bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[5]) * r
    )
    part1: Float[Array, " h w"] = term1 * (
        bessel_term1 + bessel_term2 + bessel_term3
    )
    gauss_term1: Float[Array, " h w"] = (
        kirk_params[6] / kirk_params[7]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[7]) * r**2)
    gauss_term2: Float[Array, " h w"] = (
        kirk_params[8] / kirk_params[9]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[9]) * r**2)
    gauss_term3: Float[Array, " h w"] = (
        kirk_params[10] / kirk_params[11]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[11]) * r**2)
    part2: Float[Array, " h w"] = term2 * (
        gauss_term1 + gauss_term2 + gauss_term3
    )
    supersampled_potential: Float[Array, " h w"] = part1 + part2
    if grid_shape is None:
        target_height: Int[Array, " "] = grid_height // sampling
        target_width: Int[Array, " "] = grid_width // sampling
    else:
        target_height: Int[Array, " "] = jnp.asarray(
            grid_shape[0], dtype=jnp.int32
        )
        target_width: Int[Array, " "] = jnp.asarray(
            grid_shape[1], dtype=jnp.int32
        )
    height: Int[Array, " "] = supersampled_potential.shape[0]
    width: Int[Array, " "] = supersampled_potential.shape[1]
    new_height: Int[Array, " "] = (height // sampling) * sampling
    new_width: Int[Array, " "] = (width // sampling) * sampling
    cropped: Float[Array, " h_crop w_crop"] = supersampled_potential[
        :new_height, :new_width
    ]
    reshaped: Float[Array, " h_new sampling w_new sampling"] = cropped.reshape(
        new_height // sampling, sampling, new_width // sampling, sampling
    )
    potential: Float[Array, " h_new w_new"] = jnp.mean(reshaped, axis=(1, 3))
    potential_resized: Float[Array, " h w"] = potential[
        :target_height, :target_width
    ]
    return potential_resized


@jaxtyped(typechecker=beartype)
def crystal_potential(
    crystal: CrystalStructure,
    slice_thickness: scalar_float,
    grid_shape: Tuple[scalar_int, scalar_int],
    physical_extent: Tuple[scalar_float, scalar_float],
    pixel_size: Optional[scalar_float] = 0.1,
    sampling: Optional[scalar_int] = 16,
) -> PotentialSlices:
    """Calculate the multislice potential for a crystal structure.

    Uses an optimized approach: compute atomic potentials once per unique atom
    type, then use Fourier shifts to position them at their actual coordinates.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to compute potential for
    slice_thickness : scalar_float
        Thickness of each slice in angstroms
    grid_shape : Tuple[scalar_int, scalar_int]
        Shape of the output grid (height, width) for each slice
    physical_extent : Tuple[scalar_float, scalar_float]
        Physical size of the grid (y_extent, x_extent) in angstroms
    pixel_size : scalar_float, optional
        Real space pixel size in angstroms.
        Default: 0.1
    sampling : scalar_int, optional
        Supersampling factor for potential calculation.
        Default: 16

    Returns
    -------
    potential_slices : PotentialSlices
        Structured potential data containing slice arrays and calibration
        information.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Extract atomic positions and numbers from crystal structure
    2. Find unique atomic numbers and compute their centered potentials once
    3. Calculate z-range and determine number of slices needed
    4. Calculate pixel calibrations and coordinate grids
    5. For each slice:
    6. Find atoms within the slice boundaries
    7. Group atoms by atomic number
    8. For each unique atom type in slice:
    9. Use Fourier shifts to position atoms at their x,y coordinates
    10. Sum shifted potentials for all atoms of this type
    11. Sum contributions from all atom types to get total slice potential
    12. Create PotentialSlices object with slice data and metadata
    13. Return structured potential slices
    """
    atom_positions: Float[Array, " N 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, " N"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )
    unique_atomic_numbers: Int[Array, " U"] = jnp.unique(atomic_numbers)
    y_calibration: Float[Array, " "] = physical_extent[0] / grid_shape[0]
    x_calibration: Float[Array, " "] = physical_extent[1] / grid_shape[1]

    def _compute_centered_potential(
        atomic_num: Int[Array, " "],
    ) -> Float[Array, " h w"]:
        return atomic_potential(
            atom_no=atomic_num,
            pixel_size=pixel_size,
            grid_shape=grid_shape,
            center_coords=None,  # Centered potential
            sampling=sampling,
        )

    centered_potentials: Float[Array, " U h w"] = jax.vmap(
        _compute_centered_potential
    )(unique_atomic_numbers)
    z_coords: Float[Array, " N"] = atom_positions[:, 2]
    z_min: Float[Array, " "] = jnp.min(z_coords)
    z_max: Float[Array, " "] = jnp.max(z_coords)
    z_range: Float[Array, " "] = z_max - z_min
    n_slices: Int[Array, " "] = jnp.ceil(z_range / slice_thickness).astype(
        jnp.int32
    )
    n_slices = jnp.maximum(n_slices, 1)

    def _fourier_shift_potential(
        potential: Float[Array, " h w"],
        shift_x: Float[Array, " "],
        shift_y: Float[Array, " "],
    ) -> Float[Array, " h w"]:
        """Apply Fourier shift theorem to translate potential in real space."""
        shift_pixels_x: Float[Array, " "] = shift_x / x_calibration
        shift_pixels_y: Float[Array, " "] = shift_y / y_calibration
        ky: Float[Array, " h"] = jnp.fft.fftfreq(grid_shape[0], d=1.0)
        kx: Float[Array, " w"] = jnp.fft.fftfreq(grid_shape[1], d=1.0)
        ky_grid: Float[Array, " h w"]
        kx_grid: Float[Array, " h w"]
        ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
        phase_shift: Float[Array, " h w"] = jnp.exp(
            -2j
            * jnp.pi
            * (kx_grid * shift_pixels_x + ky_grid * shift_pixels_y)
        )
        potential_fft: Float[Array, " h w"] = jnp.fft.fft2(potential)
        shifted_fft: Float[Array, " h w"] = potential_fft * phase_shift
        shifted_potential: Float[Array, " h w"] = jnp.real(
            jnp.fft.ifft2(shifted_fft)
        )
        return shifted_potential

    def _calculate_slice_potential(
        slice_idx: Int[Array, " "],
    ) -> Float[Array, " h w"]:
        """Calculate the potential for a single slice of the crystal."""
        slice_z_start: Float[Array, " "] = z_min + slice_idx * slice_thickness
        slice_z_end: Float[Array, " "] = slice_z_start + slice_thickness
        atoms_in_slice: Bool[Array, " N"] = jnp.logical_and(
            z_coords >= slice_z_start, z_coords < slice_z_end
        )
        slice_atom_positions: Float[Array, " M 3"] = atom_positions[
            atoms_in_slice
        ]
        slice_atomic_numbers: Int[Array, " M"] = atomic_numbers[atoms_in_slice]

        def _process_atom_type(
            unique_atomic_num: Int[Array, " "],
        ) -> Float[Array, " h w"]:
            """Process the potential for a single atom type."""
            potential_idx: Int[Array, " "] = jnp.where(
                unique_atomic_numbers == unique_atomic_num, size=1
            )[0][0]
            base_potential: Float[Array, " h w"] = centered_potentials[
                potential_idx
            ]
            atoms_of_type: Bool[Array, " M"] = (
                slice_atomic_numbers == unique_atomic_num
            )
            positions_of_type: Float[Array, " K 3"] = slice_atom_positions[
                atoms_of_type
            ]

            def _shift_single_atom(
                atom_pos: Float[Array, " 3"],
            ) -> Float[Array, " h w"]:
                """Shift the potential for a single atom."""
                shift_x: Float[Array, " "] = atom_pos[0]
                shift_y: Float[Array, " "] = atom_pos[1]
                return _fourier_shift_potential(
                    base_potential, shift_x, shift_y
                )

            n_atoms_of_type: Int[Array, " "] = positions_of_type.shape[0]

            def _compute_type_contribution() -> Float[Array, " h w"]:
                """Compute the sum of the potential for a single atom type."""
                shifted_potentials: Float[Array, " K h w"] = jax.vmap(
                    _shift_single_atom
                )(positions_of_type)
                return jnp.sum(shifted_potentials, axis=0)

            def _return_zero_contribution() -> Float[Array, " h w"]:
                """Return an empty contribution."""
                return jnp.zeros(grid_shape, dtype=jnp.float64)

            type_contribution: Float[Array, " h w"] = jax.lax.cond(
                n_atoms_of_type > 0,
                _compute_type_contribution,
                _return_zero_contribution,
            )
            return type_contribution

        n_atoms_in_slice: Int[Array, " "] = slice_atom_positions.shape[0]

        def _compute_slice_sum() -> Float[Array, " h w"]:
            """Compute the sum of the potential for a single slice."""
            unique_slice_numbers: Int[Array, " V"] = jnp.unique(
                slice_atomic_numbers
            )
            type_contributions: Float[Array, " V h w"] = jax.vmap(
                _process_atom_type
            )(unique_slice_numbers)
            return jnp.sum(type_contributions, axis=0)

        def _return_empty_slice() -> Float[Array, " h w"]:
            """Return an empty slice."""
            return jnp.zeros(grid_shape, dtype=jnp.float64)

        slice_potential: Float[Array, " h w"] = jax.lax.cond(
            n_atoms_in_slice > 0, _compute_slice_sum, _return_empty_slice
        )
        return slice_potential

    slice_indices: Int[Array, " n_slices"] = jnp.arange(n_slices)
    slice_arrays: Float[Array, " n_slices h w"] = jax.vmap(
        _calculate_slice_potential
    )(slice_indices)
    potential_slices: PotentialSlices = create_potential_slices(
        slices=slice_arrays,
        slice_thickness=slice_thickness,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    return potential_slices
