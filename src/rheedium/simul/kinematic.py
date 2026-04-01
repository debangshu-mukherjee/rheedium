"""Kinematic RHEED simulator.

Extended Summary
----------------
This module provides kinematic RHEED simulation functions including
structure factor calculation, and complete pattern simulation.
Implements the algorithm from arXiv:2207.06642.

Routine Listings
----------------
:func:`kinematic_spot_simulator`
    RHEED simulation using discrete 3D reciprocal lattice (spots).
:func:`make_ewald_sphere`
    Generate Ewald sphere geometry from scattering parameters.
:func:`simple_structure_factor`
    Calculate structure factor for a single reflection.

Notes
-----
Key difference from simulator.py:
- Simplified structure factors (f_j ≈ Z_j instead of Kirkland)
- For detector projection, use :func:`project_on_detector` from simulator

References
----------
.. [1] arXiv:2207.06642 - "A Python program for simulating RHEED patterns"
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_rheed_pattern,
    scalar_float,
    scalar_int,
)
from rheedium.ucell import generate_reciprocal_points

from .simul_utils import incident_wavevector, wavelength_ang
from .simulator import find_kinematic_reflections, project_on_detector


@jaxtyped(typechecker=beartype)
def make_ewald_sphere(
    wavevector_magnitude: scalar_float,
    theta_deg: scalar_float,
    phi_deg: scalar_float = 0.0,
) -> Tuple[Float[Array, "3"], scalar_float]:
    r"""Generate Ewald sphere geometry from scattering parameters.

    Parameters
    ----------
    wavevector_magnitude : scalar_float
        Magnitude of the wavevector (k = 2π/λ) in 1/Å.
    theta_deg : scalar_float
        Grazing incidence angle in degrees.
    phi_deg : scalar_float, optional
        Azimuthal angle in degrees. Default: 0.0

    Returns
    -------
    center : Float[Array, "3"]
        Center of the Ewald sphere (-k_in).
    radius : scalar_float
        Radius of the Ewald sphere (k).

    Implementation
    --------------
    1. **Derive wavelength** --
       :math:`\\lambda = 2\\pi / k`.
    2. **Compute incident wavevector** --
       Call :func:`incident_wavevector` with wavelength and
       angles to obtain :math:`k_{in}`.
    3. **Set sphere geometry** --
       Center at :math:`-k_{in}`, radius equals :math:`k`.

    See Also
    --------
    incident_wavevector : Calculate incident wavevector from angles
    build_ewald_data : Pre-compute full Ewald geometry for efficiency
    """
    wavelength: scalar_float = 2.0 * jnp.pi / wavevector_magnitude
    k_in: Float[Array, "3"] = incident_wavevector(
        wavelength, theta_deg, phi_deg
    )
    center: Float[Array, "3"] = -k_in
    radius: scalar_float = wavevector_magnitude
    return center, radius


@jaxtyped(typechecker=beartype)
def simple_structure_factor(
    reciprocal_vector: Float[Array, "3"],
    atom_positions: Float[Array, "M 3"],
    atomic_numbers: Int[Array, "M"],
) -> Float[Array, ""]:
    r"""Calculate structure factor for a single reflection.

    Following paper's Equation 7:

    .. math::

        F(G) = \sum_j f_j \cdot \exp(i \cdot G \cdot r_j)

    Parameters
    ----------
    reciprocal_vector : Float[Array, "3"]
        Reciprocal lattice vector G for this reflection
    atom_positions : Float[Array, "M 3"]
        Cartesian positions of atoms in unit cell
    atomic_numbers : Int[Array, "M"]
        Atomic numbers (Z) for each atom

    Returns
    -------
    intensity : Float[Array, ""]
        Diffraction intensity :math:`I = |F(G)|^2`

    Implementation
    --------------
    1. **Approximate scattering factors** --
       Use :math:`f_j \\approx Z_j` (atomic number) as a
       simplified form factor.
    2. **Compute phase factors** --
       Calculate :math:`\\exp(i \\cdot G \\cdot r_j)` for all
       atoms via vectorized dot products.
    3. **Sum contributions** --
       :math:`F = \\sum f_j \\cdot \\exp(i \\cdot G \\cdot r_j)`.
    4. **Return intensity** --
       :math:`I(G) = |F(G)|^2`.

    Notes
    -----
    Structure factor (Paper's Eq. 7):

    .. math::

        F(G) = \\sum_j f_j(G) \\cdot \\exp(i \\cdot G \\cdot r_j)

    For more accurate scattering, use Kirkland parameterization
    (see :func:`kirkland_form_factor` in form_factors).

    Examples
    --------
    >>> G = jnp.array([2.0, 0.0, 1.0])  # (100) reflection
    >>> positions = jnp.array([[0, 0, 0], [0.5, 0.5, 0.5]])  # Two atoms
    >>> atomic_nums = jnp.array([14, 14])  # Silicon
    >>> I = simple_structure_factor(G, positions, atomic_nums)
    >>> print(f"I(100) = {I:.2f}")

    See Also
    --------
    atomic_scattering_factor : Accurate form factor with thermal damping
    surface_structure_factor : Structure factor for CTR calculations
    """
    f_j: Float[Array, "M"] = atomic_numbers.astype(jnp.float64)
    dot_products: Float[Array, "M"] = jnp.dot(
        atom_positions, reciprocal_vector
    )
    phases: Complex[Array, "M"] = jnp.exp(1j * dot_products)
    structure_factor: Complex[Array, ""] = jnp.sum(f_j * phases)
    intensity: Float[Array, ""] = jnp.abs(structure_factor) ** 2
    return intensity


@jaxtyped(typechecker=beartype)
def kinematic_spot_simulator(
    crystal: CrystalStructure,
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    lmax: scalar_int = 1,
    detector_distance: scalar_float = 100.0,
    tolerance: scalar_float = 0.05,
) -> RHEEDPattern:
    r"""Kinematic RHEED spot simulator using discrete 3D reciprocal lattice.

    Simulates RHEED pattern as discrete spots where integer (h,k,l) reciprocal
    lattice points intersect the Ewald sphere. Useful for bulk-like diffraction
    or when only spot positions matter.

    Parameters
    ----------
    crystal : CrystalStructure
        Crystal structure with atomic positions and cell parameters.
    voltage_kv : scalar_float, optional
        Electron beam voltage in kilovolts. Default: 20.0
    theta_deg : scalar_float, optional
        Grazing incidence angle in degrees. Default: 2.0
    hmax : scalar_int, optional
        Maximum h Miller index. Default: 3
    kmax : scalar_int, optional
        Maximum k Miller index. Default: 3
    lmax : scalar_int, optional
        Maximum l Miller index. Default: 1
    detector_distance : scalar_float, optional
        Sample-to-screen distance in mm. Default: 100.0
    tolerance : scalar_float, optional
        Tolerance for Ewald sphere constraint. Default: 0.05

    Returns
    -------
    pattern : RHEEDPattern
        RHEED diffraction pattern with spot positions and intensities.

    Notes
    -----
    This simulator treats the reciprocal lattice as discrete 3D points.
    For surface-sensitive RHEED with continuous crystal truncation rods
    (CTRs) and streak patterns, use `kinematic_ctr_simulator` instead.

    Implementation
    --------------
    1. **Generate reciprocal lattice** --
       Create G(h,k,l) up to (hmax, kmax, lmax).
    2. **Compute beam parameters** --
       Calculate electron wavelength :math:`\\lambda` from
       voltage and build incident wavevector :math:`k_{in}`.
    3. **Find allowed reflections** --
       Apply Ewald sphere construction to identify
       kinematically allowed spots.
    4. **Project onto detector** --
       Map outgoing wavevectors :math:`k_{out}` to
       detector screen coordinates.
    5. **Calculate intensities** --
       Compute :math:`I = |F(G)|^2` using structure factors.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> pattern = rh.simul.kinematic_spot_simulator(
    ...     crystal=crystal,
    ...     voltage_kv=20.0,
    ...     theta_deg=2.0,
    ...     hmax=3, kmax=3, lmax=5,
    ... )
    >>> print(f"Found {len(pattern.intensities)} spots")

    See Also
    --------
    ewald_simulator : Exact Ewald sphere-CTR intersection (recommended)
    """
    reciprocal_points: Float[Array, "M 3"] = generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    wavelength: scalar_float = wavelength_ang(voltage_kv)
    k_in: Float[Array, "3"] = incident_wavevector(
        wavelength, theta_deg, phi_deg=0.0
    )
    all_indices, all_k_out = find_kinematic_reflections(
        k_in=k_in, gs=reciprocal_points, z_sign=1.0, tolerance=tolerance
    )

    # Filter to valid reflections only (indices >= 0)
    valid_mask: Bool[Array, "M"] = all_indices >= 0
    allowed_indices: Int[Array, "K"] = all_indices[valid_mask]
    k_out: Float[Array, "K 3"] = all_k_out[valid_mask]
    reciprocal_allowed: Float[Array, "K 3"] = reciprocal_points[
        allowed_indices
    ]

    detector_coords: Float[Array, "K 2"] = project_on_detector(
        k_out=k_out,
        detector_distance=detector_distance,
    )
    atom_positions: Float[Array, "M 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "M"] = crystal.cart_positions[:, 3].astype(
        jnp.int32
    )

    def _calculate_intensity(gg: Float[Array, "3"]) -> Float[Array, ""]:
        return simple_structure_factor(gg, atom_positions, atomic_numbers)

    intensities: Float[Array, "N"] = jax.vmap(_calculate_intensity)(
        reciprocal_allowed
    )
    pattern: RHEEDPattern = create_rheed_pattern(
        g_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_coords,
        intensities=intensities,
    )
    return pattern
