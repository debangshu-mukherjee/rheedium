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
compute_kinematic_intensities_with_ctrs : function
    Calculate kinematic intensities with CTR contributions
find_kinematic_reflections : function
    Find reflections satisfying kinematic conditions
incident_wavevector : function
    Calculate incident electron wavevector
kinematic_simulator : function
    Simulate complete RHEED pattern using kinematic approximation
multislice_propagate : function
    Propagate electron wave through potential slices using multislice algorithm
multislice_simulator : function
    Simulate RHEED pattern from potential slices using multislice (dynamical)
project_on_detector : function
    Project wavevectors onto detector plane
sliced_crystal_to_potential : function
    Convert SlicedCrystal to PotentialSlices for multislice simulation
wavelength_ang : function
    Calculate electron wavelength in angstroms

Notes
-----
All functions support JAX transformations and automatic differentiation for
gradient-based optimization and inverse problems.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    PotentialSlices,
    RHEEDPattern,
    SlicedCrystal,
    create_potential_slices,
    create_rheed_pattern,
    scalar_float,
    scalar_int,
    scalar_num,
)
from rheedium.ucell import generate_reciprocal_points

from .finite_domain import (
    compute_shell_sigma,
    extent_to_rod_sigma,
    rod_ewald_overlap,
)
from .form_factors import atomic_scattering_factor
from .simul_utils import incident_wavevector, wavelength_ang
from .surface_rods import integrated_rod_intensity

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out: Float[Array, "N 3"],
    detector_distance: scalar_float,
) -> Float[Array, "N 2"]:
    """Project output wavevectors onto detector plane.

    Uses ray-tracing projection to a vertical detector screen at distance d.
    The scale factor is computed as d/k_x (with small epsilon to avoid
    division by zero), then multiplied by k_y and k_z to get horizontal
    and vertical detector coordinates.

    Parameters
    ----------
    k_out : Float[Array, "N 3"]
        Output wavevectors.
    detector_distance : scalar_float
        Distance from sample to detector in mm.

    Returns
    -------
    detector_coords : Float[Array, "N 2"]
        [horizontal, vertical] coordinates on detector in mm.
    """
    scale: Float[Array, "N"] = detector_distance / (k_out[:, 0] + 1e-10)
    detector_h: Float[Array, "N"] = k_out[:, 1] * scale
    detector_v: Float[Array, "N"] = k_out[:, 2] * scale
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

    Computes k_out = k_in + G for all reciprocal lattice vectors G, then
    filters based on elastic scattering condition |k_out| ≈ |k_in| and
    z-direction constraint. Returns fixed-size arrays for JIT compatibility,
    with -1 marking invalid entries.

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
        Tolerance for reflection condition :math:`|k_{out}| = |k_{in}|`.
        Default: 0.05

    Returns
    -------
    allowed_indices : Int[Array, "M"]
        Indices of allowed reflections in gs array. Invalid entries are -1.
        Use `allowed_indices >= 0` to filter valid results.
    k_out : Float[Array, "M 3"]
        Output wavevectors for allowed reflections. Invalid entries
        correspond to `allowed_indices == -1`.

    Notes
    -----
    Returns fixed-size arrays for JIT compatibility. Filter results using:
        valid_mask = allowed_indices >= 0
        valid_indices = allowed_indices[valid_mask]
        valid_k_out = k_out[valid_mask]
    """
    k_out_all: Float[Array, "M 3"] = k_in + gs
    k_in_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    k_out_mags: Float[Array, "M"] = jnp.linalg.norm(k_out_all, axis=1)
    elastic_condition: Bool[Array, "M"] = (
        jnp.abs(k_out_mags - k_in_mag) < tolerance
    )
    z_condition: Bool[Array, "M"] = k_out_all[:, 2] * z_sign > 0
    allowed: Bool[Array, "M"] = elastic_condition & z_condition
    allowed_indices: Int[Array, "M"] = jnp.where(
        allowed, size=gs.shape[0], fill_value=-1
    )[0]
    safe_indices: Int[Array, "M"] = jnp.maximum(allowed_indices, 0)
    k_out: Float[Array, "M 3"] = k_out_all[safe_indices]
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

    For each reflection, computes the structure factor by summing atomic
    contributions (form factor × phase factor). The phase is computed as
    G·r where G vectors already include the 2π factor from reciprocal
    lattice generation. CTR intensity is added to the kinematic intensity.

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
            contribution: complex = form_factor * jnp.exp(1j * phase)
            return contribution

        n_atoms: Int[Array, ""] = atom_positions.shape[0]
        atom_indices: Int[Array, "M"] = jnp.arange(n_atoms)
        contributions: Float[Array, "M"] = jax.vmap(_atomic_contribution)(
            atom_indices
        )
        structure_factor: complex = jnp.sum(contributions)
        hk_index: Int[Array, "2"] = jnp.array(
            [
                jnp.round(g_vec[0]).astype(jnp.int32),
                jnp.round(g_vec[1]).astype(jnp.int32),
            ]
        )
        q_z_value: Float[Array, ""] = q_vector[2]
        q_z_range: Float[Array, "2"] = jnp.array(
            [q_z_value - detector_acceptance, q_z_value + detector_acceptance]
        )
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
    intensities: Float[Array, "N"] = jax.vmap(_calculate_reflection_intensity)(
        reflection_indices
    )
    return intensities


@jaxtyped(typechecker=beartype)
def kinematic_simulator(  # noqa: PLR0913
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
    domain_extent_ang: Float[Array, "3"] | None = None,
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
) -> RHEEDPattern:
    """Simulate RHEED pattern with proper atomic form factors and CTRs.

    Supports two modes for Ewald sphere intersection:

    1. **Binary mode** (default): Uses tolerance-based filtering for
       reflections satisfying |k_out| ≈ |k_in|.
    2. **Finite domain mode**: Uses continuous overlap weighting based on
       domain size and beam parameters.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    voltage_kv : scalar_num, optional
        Electron beam energy in kiloelectron volts.
        Default: 10.0
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
        Tolerance for reflection condition :math:`|k_{out}| = |k_{in}|`.
        Only used in binary mode (when domain_extent_ang is None).
        Default: 0.05
    detector_distance : scalar_float, optional
        Distance from sample to detector plane in mm.
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
    domain_extent_ang : Float[Array, "3"], optional
        Physical domain size [Lx, Ly, Lz] in Ångstroms. If provided, enables
        finite domain mode with continuous overlap weighting instead of
        binary tolerance-based filtering. Default: None
    energy_spread_frac : scalar_float, optional
        Fractional energy spread ΔE/E for Ewald shell thickness.
        Only used in finite domain mode. Default: 1e-4
    beam_divergence_rad : scalar_float, optional
        Beam angular divergence in radians for Ewald shell thickness.
        Only used in finite domain mode. Default: 1e-3

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
    k_mag: Float[Array, ""] = jnp.linalg.norm(k_in)
    k_out_all: Float[Array, "M 3"] = k_in + gs
    z_condition: Bool[Array, "M"] = k_out_all[:, 2] * z_sign > 0

    if domain_extent_ang is not None:
        domain_arr: Float[Array, "3"] = jnp.asarray(
            domain_extent_ang, dtype=jnp.float64
        )
        rod_sigma: Float[Array, "2"] = extent_to_rod_sigma(domain_arr)
        shell_sigma: Float[Array, ""] = compute_shell_sigma(
            k_magnitude=k_mag,
            energy_spread_frac=energy_spread_frac,
            beam_divergence_rad=beam_divergence_rad,
        )
        overlap: Float[Array, "M"] = rod_ewald_overlap(
            g_vectors=gs,
            k_in=k_in,
            k_magnitude=k_mag,
            rod_sigma=rod_sigma,
            shell_sigma=shell_sigma,
        )
        overlap = jnp.where(z_condition, overlap, 0.0)
        overlap_threshold: float = 1e-6
        is_allowed: Bool[Array, "M"] = overlap > overlap_threshold
        allowed_indices: Int[Array, "K"] = jnp.where(
            is_allowed, size=gs.shape[0], fill_value=-1
        )[0]
        k_out: Float[Array, "K 3"] = k_out_all[allowed_indices]
        g_allowed: Float[Array, "K 3"] = gs[allowed_indices]
        overlap_weights: Float[Array, "K"] = overlap[allowed_indices]
    else:
        kr: Tuple[Int[Array, "K"], Float[Array, "K 3"]] = (
            find_kinematic_reflections(
                k_in=k_in, gs=gs, z_sign=z_sign, tolerance=tolerance
            )
        )
        allowed_indices: Int[Array, "K"] = kr[0]
        k_out: Float[Array, "K 3"] = kr[1]
        g_allowed: Float[Array, "K 3"] = gs[allowed_indices]
        overlap_weights = None

    detector_points: Float[Array, "K 2"] = project_on_detector(
        k_out,
        detector_distance,
    )
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
    if overlap_weights is not None:
        intensities = intensities * overlap_weights
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern


@jaxtyped(typechecker=beartype)
def sliced_crystal_to_potential(
    sliced_crystal: SlicedCrystal,
    slice_thickness: scalar_float = 2.0,
    pixel_size: scalar_float = 0.1,
    voltage_kv: scalar_float = 20.0,
) -> PotentialSlices:
    """Convert a SlicedCrystal into PotentialSlices for multislice calculation.

    This function takes a surface-oriented crystal slab and generates 3D
    potential slices suitable for multislice electron diffraction simulations.
    The potential is calculated from atomic positions using proper scattering
    factors and projected onto a discrete grid.

    The interaction constant sigma = 2*pi*m*e*lambda/h^2 is computed in
    simplified form as 2*pi/(lambda*V) with units 1/(Volt*Angstrom^2).
    Each atom contributes a screened Coulomb potential V(r) ~ Z/r*exp(-a*r^2)
    where a=0.5 Å^-2 is the screening parameter.

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
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    pixel_size = jnp.asarray(pixel_size, dtype=jnp.float64)
    voltage_kv = jnp.asarray(voltage_kv, dtype=jnp.float64)
    positions: Float[Array, "N 3"] = sliced_crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = sliced_crystal.cart_positions[:, 3]
    x_extent: Float[Array, ""] = sliced_crystal.x_extent
    y_extent: Float[Array, ""] = sliced_crystal.y_extent
    depth: Float[Array, ""] = sliced_crystal.depth
    nx: int = int(jnp.ceil(x_extent / pixel_size))
    ny: int = int(jnp.ceil(y_extent / pixel_size))
    nz: int = int(jnp.ceil(depth / slice_thickness))
    x_coords: Float[Array, "nx"] = jnp.linspace(0, x_extent, nx)
    y_coords: Float[Array, "ny"] = jnp.linspace(0, y_extent, ny)
    xx: Float[Array, "nx ny"]
    yy: Float[Array, "nx ny"]
    xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
    wavelength: Float[Array, ""] = wavelength_ang(voltage_kv)
    sigma: Float[Array, ""] = 2.0 * jnp.pi / (wavelength * voltage_kv * 1000.0)
    n_atoms: int = positions.shape[0]

    def _calculate_slice_potential(slice_idx: int) -> Float[Array, "nx ny"]:
        """Calculate potential for a single slice."""
        z_start: Float[Array, ""] = slice_idx * slice_thickness
        z_end: Float[Array, ""] = (slice_idx + 1) * slice_thickness
        z_positions: Float[Array, "N"] = positions[:, 2]
        in_slice: Bool[Array, "N"] = jnp.logical_and(
            z_positions >= z_start, z_positions < z_end
        )

        def _atom_contribution(atom_idx: int) -> Float[Array, "nx ny"]:
            """Calculate contribution from single atom to potential."""
            pos: Float[Array, "3"] = positions[atom_idx]
            z_number: Float[Array, ""] = atomic_numbers[atom_idx]
            is_in_slice: Bool[Array, ""] = in_slice[atom_idx]
            dx: Float[Array, "nx ny"] = xx - pos[0]
            dy: Float[Array, "nx ny"] = yy - pos[1]
            r: Float[Array, "nx ny"] = jnp.sqrt(dx**2 + dy**2 + 1e-10)
            a: Float[Array, ""] = 0.5
            atom_potential: Float[Array, "nx ny"] = (
                z_number * sigma * jnp.exp(-a * r**2) / (r + 1e-10)
            )
            return jnp.where(is_in_slice, atom_potential, 0.0)

        atom_indices: Int[Array, "N"] = jnp.arange(n_atoms)
        contributions: Float[Array, "N nx ny"] = jax.vmap(_atom_contribution)(
            atom_indices
        )
        slice_potential: Float[Array, "nx ny"] = jnp.sum(contributions, axis=0)
        return slice_potential

    slice_indices: Int[Array, "nz"] = jnp.arange(nz)
    all_slices: Float[Array, "nz nx ny"] = jax.vmap(
        _calculate_slice_potential
    )(slice_indices)
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
    """Propagate electron wave through potential slices.

    This implements the multislice algorithm for dynamical electron
    diffraction, which accounts for multiple scattering events. The
    algorithm alternates between:
    1. Transmission through a slice: ψ' = ψ × exp(iσV)
    2. Fresnel propagation: ψ → FFT⁻¹[FFT[ψ] × P(kx,ky)]

    The interaction constant σ = 2π/(λV) is computed in simplified form
    suitable for high-energy electrons. The Fresnel propagator in reciprocal
    space is P(kx,ky) = exp(-iπλΔz(kx² + ky²)) which accounts for free-space
    propagation between slices. The initial wave is a tilted plane wave with
    phase k_in_x*x + k_in_y*y at z=0.

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
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron
       Microscopy, 2nd ed.
    .. [2] Cowley & Moodie (1957). Acta Cryst. 10, 609-619.
    """
    v_slices: Float[Array, "nz nx ny"] = potential_slices.slices
    dz: scalar_float = potential_slices.slice_thickness
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    nx: int = v_slices.shape[1]
    ny: int = v_slices.shape[2]
    lam_ang: scalar_float = wavelength_ang(voltage_kv)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    sigma: scalar_float = 2.0 * jnp.pi / (lam_ang * voltage_kv * 1000.0)
    x: Float[Array, "nx"] = jnp.arange(nx) * dx
    y: Float[Array, "ny"] = jnp.arange(ny) * dy
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, "nx ny"]
    ky_grid: Float[Array, "nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)
    k_in_x: scalar_float = k_mag * jnp.cos(theta_rad) * jnp.cos(phi_rad)
    k_in_y: scalar_float = k_mag * jnp.cos(theta_rad) * jnp.sin(phi_rad)
    x_grid: Float[Array, "nx ny"]
    y_grid: Float[Array, "nx ny"]
    x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
    phase_init: Float[Array, "nx ny"] = k_in_x * x_grid + k_in_y * y_grid
    psi: Float[Array, "nx ny"] = jnp.exp(1j * phase_init)
    propagator: Float[Array, "nx ny"] = jnp.exp(
        -1j * jnp.pi * lam_ang * dz * (kx_grid**2 + ky_grid**2)
    )

    def _propagate_one_slice(
        psi_in: Float[Array, "nx ny"], v_slice: Float[Array, "nx ny"]
    ) -> tuple[Float[Array, "nx ny"], None]:
        """Propagate through one slice: transmit then propagate."""
        transmission: Float[Array, "nx ny"] = jnp.exp(1j * sigma * v_slice)
        psi_transmitted: Float[Array, "nx ny"] = psi_in * transmission
        psi_k: Float[Array, "nx ny"] = jnp.fft.fft2(psi_transmitted)
        psi_k_propagated: Float[Array, "nx ny"] = psi_k * propagator
        psi_out: Float[Array, "nx ny"] = jnp.fft.ifft2(psi_k_propagated)
        return psi_out, None

    psi_exit: Float[Array, "nx ny"]
    psi_exit, _ = jax.lax.scan(_propagate_one_slice, psi, v_slices)
    return psi_exit


@beartype
def multislice_simulator(
    potential_slices: PotentialSlices,
    voltage_kv: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
    detector_distance: scalar_float = 100.0,
) -> RHEEDPattern:
    """Simulate RHEED pattern from potential slices using multislice algorithm.

    This function implements the complete multislice RHEED simulation pipeline:
    1. Propagate electron wave through crystal (multislice_propagate)
    2. Fourier transform exit wave to get reciprocal space amplitude
    3. Apply Ewald sphere constraint for elastic scattering where
       |k_out| = |k_in| = 2π/λ
    4. Project diffracted beams onto detector using angle approximation
       θ_x ≈ k_x/k_z, θ_y ≈ k_y/k_z
    5. Calculate intensity as |amplitude|²

    The Ewald sphere constraint gives k_out_z² = k_mag² - k_out_x² - k_out_y².
    Only real solutions (positive k_out_z²) correspond to propagating waves;
    evanescent waves don't reach the detector.

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

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with detector coordinates and intensities.
        The g_indices field contains flattened grid indices since Miller
        indices are not well-defined for multislice simulation.

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
    to satisfy elastic scattering constraint :math:`|k_{out}| = |k_{in}|`.

    See Also
    --------
    multislice_propagate : Core propagation algorithm
    simulate_rheed_pattern : Kinematic approximation simulator
    sliced_crystal_to_potential : Convert SlicedCrystal to potential slices

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron
       Microscopy, 2nd ed.
    .. [2] Ichimiya & Cohen (2004). Reflection High-Energy Electron
       Diffraction
    """
    exit_wave: Float[Array, "nx ny"] = multislice_propagate(
        potential_slices=potential_slices,
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )
    exit_wave_k: Float[Array, "nx ny"] = jnp.fft.fft2(exit_wave)
    nx: int = potential_slices.slices.shape[1]
    ny: int = potential_slices.slices.shape[2]
    dx: scalar_float = potential_slices.x_calibration
    dy: scalar_float = potential_slices.y_calibration
    kx: Float[Array, "nx"] = jnp.fft.fftfreq(nx, dx)
    ky: Float[Array, "ny"] = jnp.fft.fftfreq(ny, dy)
    kx_grid: Float[Array, "nx ny"]
    ky_grid: Float[Array, "nx ny"]
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    lam_ang: scalar_float = wavelength_ang(voltage_kv)
    k_mag: scalar_float = 2.0 * jnp.pi / lam_ang
    theta_rad: scalar_float = jnp.deg2rad(theta_deg)
    phi_rad: scalar_float = jnp.deg2rad(phi_deg)
    k_in: Float[Array, "3"] = k_mag * jnp.array(
        [
            jnp.cos(theta_rad) * jnp.cos(phi_rad),
            jnp.cos(theta_rad) * jnp.sin(phi_rad),
            jnp.sin(theta_rad),
        ]
    )
    k_out_x: Float[Array, "nx ny"] = k_in[0] + kx_grid
    k_out_y: Float[Array, "nx ny"] = k_in[1] + ky_grid
    k_out_z_squared: Float[Array, "nx ny"] = k_mag**2 - k_out_x**2 - k_out_y**2
    valid_mask: Float[Array, "nx ny"] = k_out_z_squared > 0
    k_out_z: Float[Array, "nx ny"] = jnp.where(
        valid_mask, jnp.sqrt(k_out_z_squared), 0.0
    )
    theta_x: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_x / k_out_z, 0.0
    )
    theta_y: Float[Array, "nx ny"] = jnp.where(
        valid_mask, k_out_y / k_out_z, 0.0
    )
    det_x: Float[Array, "nx ny"] = detector_distance * theta_x
    det_y: Float[Array, "nx ny"] = detector_distance * theta_y
    intensity_k: Float[Array, "nx ny"] = jnp.abs(exit_wave_k) ** 2
    intensity_k = jnp.where(valid_mask, intensity_k, 0.0)
    det_x_flat: Float[Array, "n"] = det_x.ravel()
    det_y_flat: Float[Array, "n"] = det_y.ravel()
    intensity_flat: Float[Array, "n"] = intensity_k.ravel()
    k_out_x_flat: Float[Array, "n"] = k_out_x.ravel()
    k_out_y_flat: Float[Array, "n"] = k_out_y.ravel()
    k_out_z_flat: Float[Array, "n"] = k_out_z.ravel()
    nonzero_mask: Float[Array, "n"] = intensity_flat > 0
    det_x_filtered: Float[Array, "m"] = det_x_flat[nonzero_mask]
    det_y_filtered: Float[Array, "m"] = det_y_flat[nonzero_mask]
    intensity_filtered: Float[Array, "m"] = intensity_flat[nonzero_mask]
    k_out_filtered: Float[Array, "m 3"] = jnp.column_stack(
        [
            k_out_x_flat[nonzero_mask],
            k_out_y_flat[nonzero_mask],
            k_out_z_flat[nonzero_mask],
        ]
    )
    detector_points: Float[Array, "m 2"] = jnp.column_stack(
        [det_x_filtered, det_y_filtered]
    )
    grid_indices: Int[Array, "m"] = jnp.where(nonzero_mask)[0]
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=grid_indices,
        k_out=k_out_filtered,
        detector_points=detector_points,
        intensities=intensity_filtered,
    )
    return pattern
