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
from beartype import beartype
from beartype.typing import Optional, Tuple
import pandas as pd
from jaxtyping import Array, Bool, Float, Int, jaxtyped

import rheedium as rh
from rheedium.types import CrystalStructure, RHEEDPattern, scalar_int, scalar_float

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
) -> RHEEDPattern:
    """
    Description
    -----------
    Compute a simple kinematic RHEED pattern for the given crystal.

    Parameters
    ----------
    - `crystal` (io.CrystalStructure):
        Crystal structure to simulate
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
        Distance from the sample to the detector plane in angstroms
        Optional. Default: 1000.0
    - `z_sign` (Float[Array, ""]):
        If +1, keep reflections with positive z in k_out
        Optional. Default: 1.0

    Returns
    -------
    - `pattern` (RHEEDPattern):
        A NamedTuple capturing reflection indices, k_out, and detector coords.

    Flow
    ----
    - Build real-space cell vectors from cell parameters
    - Generate reciprocal lattice points up to specified bounds
    - Calculate electron wavelength from voltage
    - Build incident wavevector at specified angle
    - Find G vectors satisfying reflection condition
    - Project resulting k_out onto detector plane
    - Calculate kinematic intensities for allowed reflections
    - Create and return RHEEDPattern with all computed data
    """
    cell_vecs = rh.ucell.build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    Gs: Float[Array, "M 3"] = rh.ucell.generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    lam_ang: Float[Array, ""] = rh.ucell.wavelength_ang(voltage_kV)
    k_in: Float[Array, "3"] = rh.simul.incident_wavevector(lam_ang, theta_deg)
    allowed_indices, k_out = rh.simul.find_kinematic_reflections(
        k_in=k_in, Gs=Gs, lam_ang=lam_ang, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, "M 2"] = project_on_detector(
        k_out, jnp.asarray(detector_distance)
    )
    G_allowed = Gs[allowed_indices]
    atom_positions = crystal.cart_positions[:, :3]
    intensities: Float[Array, "M"] = rh.simul.compute_kinematic_intensities(
        positions=atom_positions, G_allowed=G_allowed
    )
    pattern = RHEEDPattern(
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
    Calculate the projected potential of a single atom using Kirkland scattering factors.

    This function computes the projected screened potential of an independent atom
    using the Kirkland parameterization with modified Bessel functions. The potential
    is calculated on a 2D grid and downsampled to the target pixel size.

    Parameters
    ----------
    - `atom_no` (scalar_int):
        Atomic number of the atom whose potential is being calculated
    - `pixel_size` (scalar_float):
        Real space pixel size in Ångstroms
    - `sampling` (scalar_int, optional):
        Supersampling factor for increased accuracy. Higher values improve precision
        with larger pixel sizes. Default is 16
    - `potential_extent` (scalar_float, optional):
        Distance in Ångstroms from atom center to which the projected potential
        is calculated. Default is 4.0 Ångstroms
    - `datafile` (str, optional):
        Path to the CSV file containing Kirkland scattering factors.
        Default points to data/Kirkland_Potentials.csv

    Returns
    -------
    - `potential` (Float[Array, "n n"]):
        Projected potential matrix in the appropriate units

    Flow
    ----
    - Define physical constants (Bohr radius and electron charge)
    - Calculate constant terms for potential calculation
    - Load Kirkland scattering factors from CSV file
    - Extract parameters for specified atomic number
    - Calculate step size and grid extent
    - Create coordinate grid around atom center
    - Calculate radial distances from atom center
    - Compute Bessel function terms using Kirkland parameters
    - Compute Gaussian terms using Kirkland parameters
    - Combine terms to get total potential
    - Downsample potential to target resolution using average pooling
    - Return final potential matrix
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

    target_size: Int[Array, ""] = jnp.ceil(n_points / sampling).astype(jnp.int32)

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
