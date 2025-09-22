"""Functions for simulating RHEED patterns and diffraction patterns.

Extended Summary
----------------
This module provides functions for simulating Reflection High-Energy Electron
Diffraction (RHEED) patterns using kinematic approximations. It includes
utilities for calculating electron wavelengths, incident wavevectors,
diffraction intensities, and complete RHEED patterns from crystal structures.

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
compute_kinematic_intensities : function
    Calculate kinematic diffraction intensities
simulate_rheed_pattern : function
    Simulate complete RHEED pattern
atomic_potential : function
    Calculate atomic potential for intensity computation
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
    Float[Array, " ..."]
        Electron wavelength in angstroms

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> energy = jnp.array([10.0, 20.0, 30.0])
    >>> wavelengths = rh.simul.wavelength_ang(energy)
    >>> print(wavelengths)
    [0.12204694 0.08588511 0.06979081]
    """
    m: scalar_float = jnp.asarray(9.109383e-31)
    e: scalar_float = jnp.asarray(1.602177e-19)
    c: scalar_float = jnp.asarray(299792458.0)
    h: scalar_float = jnp.asarray(6.62607e-34)

    ev: Float[Array, " ..."] = (
        jnp.float64(voltage_kv) * jnp.float64(1000.0) * jnp.float64(e)
    )
    numerator: scalar_float = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, " ..."] = jnp.multiply(
        ev, ((2 * m * jnp.square(c)) + ev)
    )
    wavelength_meters: Float[Array, " ..."] = jnp.sqrt(numerator / denominator)
    lambda_angstroms: Float[Array, " ..."] = (
        jnp.asarray(1e10) * wavelength_meters
    )
    return lambda_angstroms


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: scalar_float, theta_deg: scalar_float
) -> Float[Array, " 3"]:
    """Build an incident wavevector k_in with magnitude (2π / λ).

    Traveling mostly along +x, with a small angle theta from the x-y plane.

    Parameters
    ----------
    lam_ang : scalar_float
        Electron wavelength in angstroms
    theta_deg : scalar_float
        Grazing angle in degrees

    Returns
    -------
    k_in : Float[Array, " 3"]
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

    Notes
    -----
    Algorithm:

    - Calculate wavevector magnitude as 2π/λ
    - Convert theta from degrees to radians
    - Calculate x-component using cosine of theta
    - Calculate z-component using negative sine of theta
    - Return 3D wavevector array with y-component as 0
    """
    k_mag: Float[Array, " "] = 2.0 * jnp.pi / lam_ang
    theta: Float[Array, " "] = jnp.deg2rad(theta_deg)
    kx: Float[Array, " "] = k_mag * jnp.cos(theta)
    kz: Float[Array, " "] = -k_mag * jnp.sin(theta)
    k_in: Float[Array, " 3"] = jnp.array([kx, 0.0, kz], dtype=jnp.float64)
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out_set: Float[Array, " M 3"],
    detector_distance: scalar_float,
) -> Float[Array, " M 2"]:
    """Project wavevectors k_out onto a plane at x = detector_distance.

    Parameters
    ----------
    k_out_set : Float[Array, " M 3"]
        (M, 3) array of outgoing wavevectors
    detector_distance : scalar_float
        Distance (in angstroms, or same unit) where screen is placed at x = L

    Returns
    -------
    Float[Array, " M 2"]
        (M, 2) array of projected [Y, Z] coordinates on the detector

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Create some outgoing wavevectors
    >>> k_out = jnp.array(
    ...     [
    ...         [1.0, 0.1, 0.1],  # First reflection
    ...         [1.0, -0.1, 0.2],  # Second reflection
    ...         [1.0, 0.2, -0.1],  # Third reflection
    ...     ]
    ... )
    >>>
    >>> # Project onto detector at 1000 Å distance
    >>> detector_points = rh.simul.project_on_detector(k_out, 1000.0)
    >>> print(f"Detector points: {detector_points}")

    Notes
    -----
    Algorithm:

    - Calculate norms of each wavevector
    - Normalize wavevectors to get unit directions
    - Calculate time parameter t for each ray to reach detector
    - Calculate Y coordinates using y-component of direction
    - Calculate Z coordinates using z-component of direction
    - Stack Y and Z coordinates into final array
    """
    norms: Float[Array, " M 1"] = jnp.linalg.norm(
        k_out_set, axis=1, keepdims=True
    )
    directions: Float[Array, " M 3"] = k_out_set / (norms + 1e-12)
    t_vals: Float[Array, " M"] = detector_distance / (directions[:, 0] + 1e-12)
    yy: Float[Array, " M"] = directions[:, 1] * t_vals
    zz: Float[Array, " M"] = directions[:, 2] * t_vals
    coords: Float[Array, " M 2"] = jnp.stack([yy, zz], axis=-1)
    return coords


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, " 3"],
    gs: Float[Array, " M 3"],
    lam_ang: Float[Array, " "],
    z_sign: Optional[scalar_float] = 1.0,
    tolerance: Optional[scalar_float] = 0.05,
) -> Tuple[Int[Array, " K"], Float[Array, " K 3"]]:
    """Find reflections satisfying kinematic scattering conditions.

    Returns indices of G for which ||k_in + G|| ~ 2π/lam.
    The z-component of (k_in + G) must have the specified sign.

    Parameters
    ----------
    k_in : Float[Array, " 3"]
        Incident wavevector (shape 3)
    gs : Float[Array, " M 3"]
        Reciprocal lattice vectors
    lam_ang : Float[Array, " "]
        Electron wavelength in angstroms
    z_sign : scalar_float, optional
        Sign for z-component of k_out. Default is 1.0
    tolerance : scalar_float, optional
        How close to the Ewald sphere in 1/Å. Default is 0.05

    Returns
    -------
    Tuple[Int[Array, " K"], Float[Array, " K 3"]]
        Allowed indices that will kinematically reflect and
        outgoing wavevectors (in 1/Å) for those reflections.

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
    >>> Gs = jnp.array(
    ...     [
    ...         [0, 0, 0],  # (000)
    ...         [1, 0, 0],  # (100)
    ...         [0, 1, 0],  # (010)
    ...         [1, 1, 0],  # (110)
    ...     ]
    ... )
    >>>
    >>> # Find allowed reflections
    >>> indices, k_out = rh.simul.find_kinematic_reflections(
    ...     k_in=k_in,
    ...     Gs=Gs,
    ...     lam_ang=lam,
    ...     tolerance=0.1,  # More lenient tolerance
    ... )
    >>> print(f"Allowed indices: {indices}")
    >>> print(f"Outgoing wavevectors: {k_out}")

    Algorithm
    ---------
    - Calculate wavevector magnitude as 2π/λ
    - Calculate candidate outgoing wavevectors by adding k_in to each G
    - Calculate norms of candidate wavevectors
    - Create mask for wavevectors close to Ewald sphere
    - Create mask for wavevectors with correct z-sign
    - Combine masks to get final allowed indices
    - Return allowed indices and corresponding outgoing wavevectors
    """
    k_mag: Float[Array, " "] = 2.0 * jnp.pi / lam_ang
    k_out_candidates: Float[Array, " M 3"] = k_in[None, :] + gs
    norms: Float[Array, " M"] = jnp.linalg.norm(k_out_candidates, axis=1)
    cond_mag: Bool[Array, " M"] = jnp.abs(norms - k_mag) < tolerance
    cond_z: Bool[Array, " M"] = jnp.sign(k_out_candidates[:, 2]) == jnp.sign(
        jnp.asarray(z_sign)
    )
    mask: Bool[Array, " M"] = jnp.logical_and(cond_mag, cond_z)
    allowed_indices: Int[Array, " K"] = jnp.where(mask)[0]
    k_out: Float[Array, " K 3"] = k_out_candidates[allowed_indices]
    return (allowed_indices, k_out)


@jaxtyped(typechecker=beartype)
def compute_kinematic_intensities(
    positions: Float[Array, " N 3"], g_allowed: Float[Array, " M 3"]
) -> Float[Array, " M"]:
    """Compute the kinematic intensity for each reflection.

    Given the atomic Cartesian positions (N,3) and the
    reciprocal vectors G_allowed (M,3), compute::

        I(G) = | sum_j exp(i G·r_j) |^2

    ignoring atomic form factors, etc.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Atomic positions in Cartesian coordinates.
    G_allowed : Float[Array, " M 3"]
        Reciprocal lattice vectors that satisfy reflection condition.

    Returns
    -------
    intensities : Float[Array, " M"]
        Intensities for each reflection.

    Examples
    --------
    >>> import rheedium as rh
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a simple unit cell with two atoms
    >>> positions = jnp.array(
    ...     [
    ...         [0.0, 0.0, 0.0],  # First atom at origin
    ...         [0.5, 0.5, 0.5],  # Second atom at cell center
    ...     ]
    ... )
    >>>
    >>> # Define some allowed G vectors
    >>> G_allowed = jnp.array(
    ...     [
    ...         [1, 0, 0],  # (100)
    ...         [0, 1, 0],  # (010)
    ...         [1, 1, 0],  # (110)
    ...     ]
    ... )
    >>>
    >>> # Calculate intensities
    >>> intensities = rh.simul.compute_kinematic_intensities(
    ...     positions=positions, G_allowed=G_allowed
    ... )
    >>> print(f"Reflection intensities: {intensities}")

    Algorithm
    ---------
    - Define inner function to compute intensity for single G vector
    - Calculate phase factors for each atom position
    - Sum real and imaginary parts of phase factors
    - Compute intensity as sum of squared real and imaginary parts
    - Vectorize computation over all allowed G vectors
    """

    def _intensity_for_g(g_: Float[Array, " 3"]) -> Float[Array, " "]:
        """Calculate intensity for a single G vector.

        Parameters
        ----------
        g_ : Float[Array, " 3"]
            Single reciprocal lattice vector

        Returns
        -------
        Float[Array, " "]
            Intensity value for this G vector
        """
        phases: Float[Array, " N"] = jnp.einsum("j,ij->i", g_, positions)
        re: Float[Array, " "] = jnp.sum(jnp.cos(phases))
        im: Float[Array, " "] = jnp.sum(jnp.sin(phases))
        return re * re + im * im

    intensities: Float[Array, " M"] = jax.vmap(_intensity_for_g)(g_allowed)
    return intensities


@jaxtyped(typechecker=beartype)
def simulate_rheed_pattern(
    crystal: CrystalStructure,
    voltage_kv: Optional[scalar_num] = 10,
    theta_deg: Optional[scalar_float] = 1.0,
    hmax: Optional[scalar_int] = 3,
    kmax: Optional[scalar_int] = 3,
    lmax: Optional[scalar_int] = 1,
    tolerance: Optional[scalar_float] = 0.05,
    detector_distance: Optional[scalar_num] = 1000.0,
    z_sign: Optional[scalar_float] = 1.0,
    pixel_size: Optional[scalar_float] = 0.1,
) -> RHEEDPattern:
    """Compute a kinematic RHEED pattern for the given crystal.

    Uses atomic form factors from Kirkland potentials for realistic intensities.

    This function combines several steps:
    1. Generates reciprocal lattice points
        using :func:`generate_reciprocal_points`
    2. Calculates incident wavevector using :func:`incident_wavevector`
    3. Finds allowed reflections using :func:`find_kinematic_reflections`
    4. Projects points onto detector using :func:`project_on_detector`
    5. Computes intensities using atomic form factors
        from :func:`atomic_potential`

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure to simulate.
        Can be created using :func:`rheedium.types.create_crystal_structure`
        or loaded from a CIF file using :func:`rheedium.inout.parse_cif`
    voltage_kv : scalar_num, optional
        Accelerating voltage in kilovolts.
        Default: 10.0
    theta_deg : scalar_float, optional
        Grazing angle in degrees.
        Default: 1.0
    hmax : scalar_int, optional
        h Bound on reciprocal lattice indices.
        Default is 3.
    kmax : scalar_int, optional
        k Bound on reciprocal lattice indices.
        Default is 3.
    lmax : scalar_int, optional
        l Bound on reciprocal lattice indices.
        Default is 1.
    tolerance : scalar_float, optional
        How close to the Ewald sphere in 1/Å.
        Default: 0.05
    detector_distance : scalar_float, optional
        Distance from sample to detector plane in angstroms.
        Default: 1000.0
    z_sign : scalar_float, optional
        If +1, keep reflections with positive z in k_out.
        Default: 1.0
    pixel_size : scalar_float, optional
        Pixel size for atomic potential calculation in angstroms.
        Default: 0.1

    Returns
    -------
    pattern : RHEEDPattern
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
    ...     voltage_kv=jnp.asarray(20.0),  # 20 kV beam
    ...     theta_deg=jnp.asarray(2.0),  # 2 degree grazing angle
    ...     hmax=jnp.asarray(4),  # Generate more reflections
    ...     kmax=jnp.asarray(4),
    ...     lmax=jnp.asarray(2),
    ... )
    >>>
    >>> # Plot the pattern
    >>> rh.plots.plot_rheed(pattern, grid_size=400)

    Algorithm
    ---------
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
        k_in=k_in, gs=gs, lam_ang=lam_ang, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, " K 2"] = project_on_detector(
        k_out, detector_distance
    )
    g_allowed: Float[Array, " K 3"] = gs[allowed_indices]
    atom_positions: Float[Array, " N 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, " N"] = crystal.cart_positions[:, 3]
    unique_atomic_numbers: Float[Array, " U"] = jnp.unique(atomic_numbers)

    def _calculate_form_factor_for_atom(
        atomic_num: Float[Array, " "],
    ) -> Float[Array, " n n"]:
        atomic_num_int: scalar_int = int(atomic_num)
        return atomic_potential(
            atom_no=atomic_num_int,
            pixel_size=pixel_size,
            sampling=16,
            potential_extent=4.0,
        )

    form_factors: Float[Array, " U n n"] = jax.vmap(
        _calculate_form_factor_for_atom
    )(unique_atomic_numbers)

    def _compute_structure_factor_with_form_factors(
        g_vec: Float[Array, " 3"],
    ) -> Float[Array, " "]:
        phases: Float[Array, " N"] = jnp.einsum(
            "j,ij->i", g_vec, atom_positions
        )

        def _get_form_factor_for_atom(
            atom_idx: Int[Array, " "],
        ) -> Float[Array, " "]:
            atomic_num: Float[Array, " "] = atomic_numbers[atom_idx]
            form_factor_idx: Int[Array, " "] = jnp.where(
                unique_atomic_numbers == atomic_num, size=1
            )[0][0]
            form_factor_matrix: Float[Array, " n n"] = form_factors[
                form_factor_idx
            ]
            center_idx: Int[Array, " "] = form_factor_matrix.shape[0] // 2
            return form_factor_matrix[center_idx, center_idx]

        atom_indices: Int[Array, " N"] = jnp.arange(len(atomic_numbers))
        form_factor_values: Float[Array, " N"] = jax.vmap(
            _get_form_factor_for_atom
        )(atom_indices)
        complex_amplitudes: Float[Array, " N"] = form_factor_values * jnp.exp(
            1j * phases
        )
        total_amplitude: Float[Array, " "] = jnp.sum(complex_amplitudes)
        intensity: Float[Array, " "] = jnp.real(
            total_amplitude * jnp.conj(total_amplitude)
        )
        return intensity

    intensities: Float[Array, " K"] = jax.vmap(
        _compute_structure_factor_with_form_factors
    )(g_allowed)
    pattern: RHEEDPattern = create_rheed_pattern(
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

    Algorithm
    ---------
    - Define physical constants and load Kirkland parameters
    - Determine grid size and center coordinates
    - Calculate step size for supersampling
    - Create coordinate grid with atom centered at specified position
    - Calculate radial distances from atom center
    - Compute Bessel and Gaussian terms using Kirkland parameters
    - Combine terms to get total potential
    - Downsample to target resolution using average pooling
    - Return final potential matrix
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

    Algorithm
    ---------
    - Extract atomic positions and numbers from crystal structure
    - Find unique atomic numbers and compute their centered potentials once
    - Calculate z-range and determine number of slices needed
    - Calculate pixel calibrations and coordinate grids
    - For each slice:
        - Find atoms within the slice boundaries
        - Group atoms by atomic number
        - For each unique atom type in slice:
            - Use Fourier shifts to position atoms at their x,y coordinates
            - Sum shifted potentials for all atoms of this type
        - Sum contributions from all atom types to get total slice potential
    - Create PotentialSlices object with slice data and metadata
    - Return structured potential slices
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
        """Apply Fourier shift theorem to translate potential in real space.

        Parameters
        ----------
        potential : Float[Array, " h w"]
            Input potential to be shifted
        shift_x : Float[Array, " "]
            Shift in x-direction in angstroms
        shift_y : Float[Array, " "]
            Shift in y-direction in angstroms

        Returns
        -------
        shifted_potential : Float[Array, " h w"]
            Potential shifted to new position

        Algorithm
        ---------
        - Convert shifts from physical units to pixel units
        - Create frequency grids for FFT
        - Apply phase shift in Fourier domain
        - Transform back to real space
        """
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
                shift_x: Float[Array, " "] = atom_pos[0]
                shift_y: Float[Array, " "] = atom_pos[1]
                return _fourier_shift_potential(
                    base_potential, shift_x, shift_y
                )

            n_atoms_of_type: Int[Array, " "] = positions_of_type.shape[0]

            def _compute_type_contribution() -> Float[Array, " h w"]:
                shifted_potentials: Float[Array, " K h w"] = jax.vmap(
                    _shift_single_atom
                )(positions_of_type)
                return jnp.sum(shifted_potentials, axis=0)

            def _return_zero_contribution() -> Float[Array, " h w"]:
                return jnp.zeros(grid_shape, dtype=jnp.float64)

            type_contribution: Float[Array, " h w"] = jax.lax.cond(
                n_atoms_of_type > 0,
                _compute_type_contribution,
                _return_zero_contribution,
            )
            return type_contribution

        n_atoms_in_slice: Int[Array, " "] = slice_atom_positions.shape[0]

        def _compute_slice_sum() -> Float[Array, " h w"]:
            unique_slice_numbers: Int[Array, " V"] = jnp.unique(
                slice_atomic_numbers
            )
            type_contributions: Float[Array, " V h w"] = jax.vmap(
                _process_atom_type
            )(unique_slice_numbers)
            return jnp.sum(type_contributions, axis=0)

        def _return_empty_slice() -> Float[Array, " h w"]:
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
