"""
Module: simul.simulator
-----------------------
Functions for simulating RHEED patterns and calculating diffraction intensities.

Functions
---------
- `incident_wavevector`:
    Calculate incident electron wavevector
- `project_on_detector`:
    Project wavevectors onto detector plane
- `find_kinematic_reflections`:
    Find reflections satisfying kinematic conditions
- `compute_kinematic_intensities`:
    Calculate kinematic diffraction intensities
- `simulate_rheed_pattern`:
    Simulate complete RHEED pattern
- `atomic_potential`:
    Calculate atomic potential for intensity computation
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

import rheedium as rh
from rheedium.types import (CrystalStructure, RHEEDPattern, scalar_float,
                            scalar_int)

jax.config.update("jax_enable_x64", True)
DEFAULT_KIRKLAND_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "Kirkland_Potentials.csv"
)


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_float, theta_deg: scalar_float
) -> Float[Array, "3"]:
    """
    Description
    -----------
    Build an incident wavevector k_in with magnitude (2π / λ),
    traveling mostly along +x, with a small angle theta from the x-y plane.

    Parameters
    ----------
    - `lam_ang` (scalar_float):
        Electron wavelength in angstroms
    - `theta_deg` (scalar_float):
        Grazing angle in degrees

    Returns
    -------
    - `k_in` (Float[Array, "3"]):
        The 3D incident wavevector (1/angstrom)

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Calculate wavelength for 20 kV electrons
    >>> lam = rh.ucell.wavelength_ang(20.0)
    >>> 
    >>> # Calculate incident wavevector at 2 degree grazing angle
    >>> k_in = rh.simul.incident_wavevector(lam, 2.0)
    >>> print(f"Incident wavevector: {k_in}")

    Flow
    ----
    - Calculate wavevector magnitude as 2π/λ
    - Convert theta from degrees to radians
    - Calculate x-component using cosine of theta
    - Calculate z-component using negative sine of theta
    - Return 3D wavevector array with y-component as 0
    """
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta: Float[Array, ""] = jnp.deg2rad(theta_deg)
    kx: Float[Array, ""] = k_mag * jnp.cos(theta)
    kz: Float[Array, ""] = -k_mag * jnp.sin(theta)
    k_in: Float[Array, "3"] = jnp.array([kx, 0.0, kz], dtype=jnp.float64)
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out_set: Float[Array, "M 3"],
    detector_distance: scalar_float,
) -> Float[Array, "M 2"]:
    """
    Description
    -----------
    Project wavevectors k_out onto a plane at x = detector_distance.
    Returns (M, 2) array of [Y, Z] coordinates on the detector.

    Parameters
    ----------
    - `k_out_set` (Float[Array, "M 3"]):
        (M, 3) array of outgoing wavevectors
    - `detector_distance` (scalar_float):
        distance (in angstroms, or same unit) where screen is placed at x = L

    Returns
    -------
    - `coords` (Float[Array, "M 2"]):
        (M, 2) array of projected [Y, Z]

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Create some outgoing wavevectors
    >>> k_out = jnp.array([
    ...     [1.0, 0.1, 0.1],  # First reflection
    ...     [1.0, -0.1, 0.2], # Second reflection
    ...     [1.0, 0.2, -0.1]  # Third reflection
    ... ])
    >>> 
    >>> # Project onto detector at 1000 Å distance
    >>> detector_points = rh.simul.project_on_detector(k_out, 1000.0)
    >>> print(f"Detector points: {detector_points}")

    Flow
    ----
    - Calculate norms of each wavevector
    - Normalize wavevectors to get unit directions
    - Calculate time parameter t for each ray to reach detector
    - Calculate Y coordinates using y-component of direction
    - Calculate Z coordinates using z-component of direction
    - Stack Y and Z coordinates into final array
    """
    norms: Float[Array, "M 1"] = jnp.linalg.norm(k_out_set, axis=1, keepdims=True)
    directions: Float[Array, "M 3"] = k_out_set / (norms + 1e-12)
    t_vals: Float[Array, "M"] = detector_distance / (directions[:, 0] + 1e-12)
    Y: Float[Array, "M"] = directions[:, 1] * t_vals
    Z: Float[Array, "M"] = directions[:, 2] * t_vals
    coords: Float[Array, "M 2"] = jnp.stack([Y, Z], axis=-1)
    return coords


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, "3"],
    Gs: Float[Array, "M 3"],
    lam_ang: Float[Array, ""],
    z_sign: Optional[Float[Array, ""]] = jnp.asarray(1.0),
    tolerance: Optional[scalar_float] = 0.05,
) -> Tuple[Int[Array, "K"], Float[Array, "K 3"]]:
    """
    Description
    -----------
    Returns indices of G for which ||k_in + G|| ~ 2π/lam
    and the z-component of (k_in + G) has the specified sign.

    Parameters
    ----------
    - `k_in` (Float[Array, "3"]):
        shape (3,)
    - `Gs` (Float[Array, "M 3]"):
        G vector
    - `lam_ang` (Float[Array, ""):
        electron wavelength in Å
    - `z_sign` (Float[Array, ""]):
        sign for z-component of k_out
    - `tolerance` (scalar_float, optional),
        how close to the Ewald sphere in 1/Å
        Optional. Default: 0.05

    Returns
    -------
    - `allowed_indices` (Int[Array, "K"]):
        Allowed indices that will kinematically reflect.
    - `k_out` (Float[Array, "K 3"]):
        Outgoing wavevectors (in 1/Å) for those reflections.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Calculate incident wavevector
    >>> lam = rh.ucell.wavelength_ang(20.0)
    >>> k_in = rh.simul.incident_wavevector(lam, 2.0)
    >>> 
    >>> # Generate some reciprocal lattice points
    >>> Gs = jnp.array([
    ...     [0, 0, 0],    # (000)
    ...     [1, 0, 0],    # (100)
    ...     [0, 1, 0],    # (010)
    ...     [1, 1, 0]     # (110)
    ... ])
    >>> 
    >>> # Find allowed reflections
    >>> indices, k_out = rh.simul.find_kinematic_reflections(
    ...     k_in=k_in,
    ...     Gs=Gs,
    ...     lam_ang=lam,
    ...     tolerance=0.1  # More lenient tolerance
    ... )
    >>> print(f"Allowed indices: {indices}")
    >>> print(f"Outgoing wavevectors: {k_out}")

    Flow
    ----
    - Calculate wavevector magnitude as 2π/λ
    - Calculate candidate outgoing wavevectors by adding k_in to each G
    - Calculate norms of candidate wavevectors
    - Create mask for wavevectors close to Ewald sphere
    - Create mask for wavevectors with correct z-sign
    - Combine masks to get final allowed indices
    - Return allowed indices and corresponding outgoing wavevectors
    """
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    k_out_candidates: Float[Array, "M 3"] = k_in[None, :] + Gs
    norms: Float[Array, "M"] = jnp.linalg.norm(k_out_candidates, axis=1)
    cond_mag: Bool[Array, "M"] = jnp.abs(norms - k_mag) < tolerance
    cond_z: Bool[Array, "M"] = jnp.sign(k_out_candidates[:, 2]) == jnp.sign(z_sign)
    mask: Bool[Array, "M"] = jnp.logical_and(cond_mag, cond_z)
    allowed_indices: Int[Array, "K"] = jnp.where(mask)[0]
    k_out: Float[Array, "K 3"] = k_out_candidates[allowed_indices]
    return (allowed_indices, k_out)


def compute_kinematic_intensities(
    positions: Float[Array, "N 3"], G_allowed: Float[Array, "M 3"]
) -> Float[Array, "M"]:
    """
    Description
    -----------
    Given the atomic Cartesian positions (N,3) and the
    reciprocal vectors G_allowed (M,3),
    compute the kinematic intensity for each reflection:
        I(G) = | sum_j exp(i G·r_j) |^2
    ignoring atomic form factors, etc.

    Parameters
    ----------
    - `positions` (Float[Array, "N 3]):
        Atomic positions in Cartesian coordinates.
    - `G_allowed` (Float[Array, "M 3]):
        Reciprocal lattice vectors that satisfy reflection condition.

    Returns
    -------
    - `intensities` (Float[Array, "M"]):
        Intensities for each reflection.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Create a simple unit cell with two atoms
    >>> positions = jnp.array([
    ...     [0.0, 0.0, 0.0],  # First atom at origin
    ...     [0.5, 0.5, 0.5]   # Second atom at cell center
    ... ])
    >>> 
    >>> # Define some allowed G vectors
    >>> G_allowed = jnp.array([
    ...     [1, 0, 0],    # (100)
    ...     [0, 1, 0],    # (010)
    ...     [1, 1, 0]     # (110)
    ... ])
    >>> 
    >>> # Calculate intensities
    >>> intensities = rh.simul.compute_kinematic_intensities(
    ...     positions=positions,
    ...     G_allowed=G_allowed
    ... )
    >>> print(f"Reflection intensities: {intensities}")

    Flow
    ----
    - Define inner function to compute intensity for single G vector
    - Calculate phase factors for each atom position
    - Sum real and imaginary parts of phase factors
    - Compute intensity as sum of squared real and imaginary parts
    - Vectorize computation over all allowed G vectors
    """

    def intensity_for_G(G_):
        phases = jnp.einsum("j,ij->i", G_, positions)
        re = jnp.sum(jnp.cos(phases))
        im = jnp.sum(jnp.sin(phases))
        return re * re + im * im

    intensities = jax.vmap(intensity_for_G)(G_allowed)
    return intensities


@jaxtyped(typechecker=beartype)
def simulate_rheed_pattern(
    crystal: CrystalStructure,
    voltage_kV: Optional[Float[Array, ""]] = jnp.asarray(10.0),
    theta_deg: Optional[Float[Array, ""]] = jnp.asarray(1.0),
    hmax: Optional[Int[Array, ""]] = jnp.asarray(3),
    kmax: Optional[Int[Array, ""]] = jnp.asarray(3),
    lmax: Optional[Int[Array, ""]] = jnp.asarray(1),
    tolerance: Optional[Float[Array, ""]] = jnp.asarray(0.05),
    detector_distance: Optional[Float[Array, ""]] = jnp.asarray(1000.0),
    z_sign: Optional[Float[Array, ""]] = jnp.asarray(1.0),
    pixel_size: Optional[Float[Array, ""]] = jnp.asarray(0.1),
) -> RHEEDPattern:
    """
    Description
    -----------
    Compute a kinematic RHEED pattern for the given crystal using
    atomic form factors from Kirkland potentials for realistic intensities.

    This function combines several steps:
    1. Generates reciprocal lattice points using :func:`rheedium.ucell.generate_reciprocal_points`
    2. Calculates incident wavevector using :func:`incident_wavevector`
    3. Finds allowed reflections using :func:`find_kinematic_reflections`
    4. Projects points onto detector using :func:`project_on_detector`
    5. Computes intensities using atomic form factors from :func:`atomic_potential`

    Parameters
    ----------
    - `crystal` (CrystalStructure):
        Crystal structure to simulate. Can be created using :func:`rheedium.types.create_crystal_structure`
        or loaded from a CIF file using :func:`rheedium.inout.parse_cif`
    - `voltage_kV` (Float[Array, ""]):
        Accelerating voltage in kilovolts.
        Optional. Default: 10.0
    - `theta_deg` (Float[Array, ""]):
        Grazing angle in degrees
        Optional. Default: 1.0
    - `hmax, kmax, lmax` (Int[Array, ""]):
        Bounds on reciprocal lattice indices
        Optional. Default: 3, 3, 1
    - `tolerance` (Float[Array, ""]):
        How close to the Ewald sphere in 1/Å
        Optional. Default: 0.05
    - `detector_distance` (Float[Array, ""]):
        Distance from sample to detector plane in angstroms
        Optional. Default: 1000.0
    - `z_sign` (Float[Array, ""]):
        If +1, keep reflections with positive z in k_out
        Optional. Default: 1.0
    - `pixel_size` (Float[Array, ""]):
        Pixel size for atomic potential calculation in angstroms
        Optional. Default: 0.1

    Returns
    -------
    - `pattern` (RHEEDPattern):
        A NamedTuple capturing reflection indices, k_out, and detector coords.
        Can be visualized using :func:`rheedium.plots.plot_rheed`

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Load crystal structure from CIF file
    >>> crystal = rh.inout.parse_cif("path/to/crystal.cif")
    >>> 
    >>> # Simulate RHEED pattern
    >>> pattern = rh.simul.simulate_rheed_pattern(
    ...     crystal=crystal,
    ...     voltage_kV=jnp.asarray(20.0),  # 20 kV beam
    ...     theta_deg=jnp.asarray(2.0),    # 2 degree grazing angle
    ...     hmax=jnp.asarray(4),           # Generate more reflections
    ...     kmax=jnp.asarray(4),
    ...     lmax=jnp.asarray(2)
    ... )
    >>> 
    >>> # Plot the pattern
    >>> rh.plots.plot_rheed(pattern, grid_size=400)

    Flow
    ----
    - Build real-space cell vectors from cell parameters
    - Generate reciprocal lattice points up to specified bounds
    - Calculate electron wavelength from voltage
    - Build incident wavevector at specified angle
    - Find G vectors satisfying reflection condition
    - Project resulting k_out onto detector plane
    - Extract unique atomic numbers from crystal
    - Calculate atomic potentials for each element type
    - Compute structure factors with atomic form factors
    - Create and return RHEEDPattern with computed data
    """
    Gs: Float[Array, "M 3"] = rh.ucell.generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    lam_ang: Float[Array, ""] = rh.ucell.wavelength_ang(voltage_kV)
    k_in: Float[Array, "3"] = rh.simul.incident_wavevector(lam_ang, theta_deg)
    allowed_indices: Int[Array, "K"]
    k_out: Float[Array, "K 3"]
    allowed_indices, k_out = rh.simul.find_kinematic_reflections(
        k_in=k_in, Gs=Gs, lam_ang=lam_ang, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, "K 2"] = project_on_detector(k_out, detector_distance)
    G_allowed: Float[Array, "K 3"] = Gs[allowed_indices]
    atom_positions: Float[Array, "N 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "N"] = crystal.cart_positions[:, 3]
    unique_atomic_numbers: Float[Array, "U"] = jnp.unique(atomic_numbers)

    def calculate_form_factor_for_atom(
        atomic_num: Float[Array, ""],
    ) -> Float[Array, "n n"]:
        atomic_num_int: scalar_int = int(atomic_num)
        return rh.simul.atomic_potential(
            atom_no=atomic_num_int,
            pixel_size=pixel_size,
            sampling=16,
            potential_extent=4.0,
        )

    form_factors: Float[Array, "U n n"] = jax.vmap(calculate_form_factor_for_atom)(
        unique_atomic_numbers
    )

    def compute_structure_factor_with_form_factors(
        G_vec: Float[Array, "3"],
    ) -> Float[Array, ""]:
        phases: Float[Array, "N"] = jnp.einsum("j,ij->i", G_vec, atom_positions)

        def get_form_factor_for_atom(atom_idx: Int[Array, ""]) -> Float[Array, ""]:
            atomic_num: Float[Array, ""] = atomic_numbers[atom_idx]
            form_factor_idx: Int[Array, ""] = jnp.where(
                unique_atomic_numbers == atomic_num, size=1
            )[0][0]
            form_factor_matrix: Float[Array, "n n"] = form_factors[form_factor_idx]
            center_idx: Int[Array, ""] = form_factor_matrix.shape[0] // 2
            return form_factor_matrix[center_idx, center_idx]

        atom_indices: Int[Array, "N"] = jnp.arange(len(atomic_numbers))
        form_factor_values: Float[Array, "N"] = jax.vmap(get_form_factor_for_atom)(
            atom_indices
        )
        complex_amplitudes: Float[Array, "N"] = form_factor_values * jnp.exp(
            1j * phases
        )
        total_amplitude: Float[Array, ""] = jnp.sum(complex_amplitudes)
        intensity: Float[Array, ""] = jnp.real(
            total_amplitude * jnp.conj(total_amplitude)
        )
        return intensity

    intensities: Float[Array, "K"] = jax.vmap(
        compute_structure_factor_with_form_factors
    )(G_allowed)
    pattern: RHEEDPattern = rh.types.create_rheed_pattern(
        G_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern


@jaxtyped(typechecker=beartype)
def atomic_potential(
    atom_no: scalar_int,
    pixel_size: scalar_float,
    sampling: Optional[scalar_int] = 16,
    potential_extent: Optional[scalar_float] = 4.0,
    datafile: Optional[str] = str(DEFAULT_KIRKLAND_PATH),
) -> Float[Array, "n n"]:
    """
    Description
    -----------
    Calculate the atomic potential for a given element using Kirkland's parameterization.
    This function is used internally by :func:`simulate_rheed_pattern` to compute
    realistic diffraction intensities.

    The potential is computed on a grid with size determined by `sampling` and `potential_extent`.
    The grid is centered on the atom position, with the potential value at the center
    used as the atomic form factor in structure factor calculations.

    Parameters
    ----------
    - `atom_no` (scalar_int):
        Atomic number of the element
    - `pixel_size` (scalar_float):
        Size of each pixel in angstroms
    - `sampling` (scalar_int, optional):
        Number of pixels in each dimension
        Default: 16
    - `potential_extent` (scalar_float, optional):
        Extent of the potential in angstroms
        Default: 4.0
    - `datafile` (str, optional):
        Path to Kirkland potentials data file
        Default: '<project_root>/data/Kirkland_Potentials.csv'

    Returns
    -------
    - `potential` (Float[Array, "n n"]):
        Atomic potential on a 2D grid, where n = sampling

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>> 
    >>> # Calculate potential for gold (Z=79)
    >>> potential = rh.simul.atomic_potential(
    ...     atom_no=79,
    ...     pixel_size=0.1,  # 0.1 Å per pixel
    ...     sampling=32,     # 32x32 grid
    ...     potential_extent=5.0  # 5 Å extent
    ... )
    >>> 
    >>> # The center value is used as the atomic form factor
    >>> form_factor = potential[16, 16]  # Center of 32x32 grid

    Flow
    ----
    - Load Kirkland potential parameters from CSV file
    - Create coordinate grid for potential calculation
    - Calculate radial distances from center
    - Compute Bessel functions for each parameter
    - Sum contributions from all parameters
    - Return 2D potential array
    """
    a0: Float[Array, ""] = jnp.asarray(0.5292)
    ek: Float[Array, ""] = jnp.asarray(14.4)
    term1: Float[Array, ""] = 4.0 * (jnp.pi**2) * a0 * ek
    term2: Float[Array, ""] = 2.0 * (jnp.pi**2) * a0 * ek
    kirkland_df = pd.read_csv(datafile, header=None)
    kirkland_array: Float[Array, "103 12"] = jnp.array(kirkland_df.values)
    kirk_params: Float[Array, "12"] = kirkland_array[atom_no - 1, :]
    step_size: Float[Array, ""] = pixel_size / sampling
    grid_extent: Float[Array, ""] = potential_extent
    n_points: Int[Array, ""] = jnp.ceil(2.0 * grid_extent / step_size).astype(jnp.int32)
    coords: Float[Array, "n"] = jnp.linspace(-grid_extent, grid_extent, n_points)
    ya: Float[Array, "n n"]
    xa: Float[Array, "n n"]
    ya, xa = jnp.meshgrid(coords, coords, indexing="ij")
    r: Float[Array, "n n"] = jnp.sqrt(xa**2 + ya**2)
    bessel_term1: Float[Array, "n n"] = kirk_params[0] * rh.ucell.bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[1]) * r
    )
    bessel_term2: Float[Array, "n n"] = kirk_params[2] * rh.ucell.bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[3]) * r
    )
    bessel_term3: Float[Array, "n n"] = kirk_params[4] * rh.ucell.bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[5]) * r
    )
    part1: Float[Array, "n n"] = term1 * (bessel_term1 + bessel_term2 + bessel_term3)
    gauss_term1: Float[Array, "n n"] = (kirk_params[6] / kirk_params[7]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[7]) * r**2
    )
    gauss_term2: Float[Array, "n n"] = (kirk_params[8] / kirk_params[9]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[9]) * r**2
    )
    gauss_term3: Float[Array, "n n"] = (kirk_params[10] / kirk_params[11]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[11]) * r**2
    )
    part2: Float[Array, "n n"] = term2 * (gauss_term1 + gauss_term2 + gauss_term3)
    supersampled_potential: Float[Array, "n n"] = part1 + part2
    height: Int[Array, ""] = supersampled_potential.shape[0]
    width: Int[Array, ""] = supersampled_potential.shape[1]
    new_height: Int[Array, ""] = (height // sampling) * sampling
    new_width: Int[Array, ""] = (width // sampling) * sampling
    cropped: Float[Array, "h w"] = supersampled_potential[:new_height, :new_width]
    reshaped: Float[Array, "h_new sampling w_new sampling"] = cropped.reshape(
        new_height // sampling, sampling, new_width // sampling, sampling
    )
    potential: Float[Array, "h_new w_new"] = jnp.mean(reshaped, axis=(1, 3))

    return potential
