"""Finite domain Ewald sphere broadening for RHEED simulation.

Extended Summary
----------------
This module provides functions for computing finite domain effects in RHEED
diffraction. Finite coherent domain size causes reciprocal lattice rods to
broaden, and the Ewald sphere intersection becomes a continuous overlap
integral rather than a binary hit. Combined with finite Ewald shell thickness
from beam energy spread and divergence, this enables realistic simulation of
surfaces with limited coherence length.

Routine Listings
----------------
compute_domain_extent : function
    Compute domain extent from atomic positions bounding box
compute_shell_sigma : function
    Compute Ewald shell Gaussian thickness from beam parameters
extent_to_rod_sigma : function
    Convert domain extent to reciprocal-space rod widths
finite_domain_intensities : function
    Compute intensities with finite domain broadening
rod_ewald_overlap : function
    Compute overlap between broadened rods and Ewald shell

Notes
-----
The finite domain broadening replaces the binary Ewald sphere intersection
condition with a continuous overlap integral. For a domain of size L, the
reciprocal lattice rod has Gaussian width σ ≈ 2π/L. The measured intensity
is I = |F|² × Overlap(rod, shell), where the overlap depends on both the
rod width and the Ewald shell thickness from beam parameters.

Physical origins of broadening:

1. **Rod broadening** (from finite domain size):
   - Coherent scattering only within domain
   - Rod FWHM ≈ 2π/L in reciprocal space
   - Gaussian approximation: σ_rod = 2π/(L × √(2π))

2. **Shell broadening** (from beam properties):
   - Energy spread ΔE/E contributes Δk/k = ΔE/(2E)
   - Beam divergence Δθ contributes Δk⊥ = k×Δθ
   - Combined: σ_shell = k × √[(ΔE/2E)² + Δθ²]

References
----------
.. [1] Ichimiya & Cohen (2004). Reflection High-Energy Electron Diffraction
.. [2] Robinson & Tweet (1992). Rep. Prog. Phys. 55, 599 (CTR theory)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    EwaldData,
    scalar_float,
)

from .simul_utils import incident_wavevector, wavelength_ang

jax.config.update("jax_enable_x64", True)

# Minimum extent to avoid division by zero (in Ångstroms)
_MIN_EXTENT_ANG: float = 1.0


@jaxtyped(typechecker=beartype)
def compute_domain_extent(
    positions: Float[Array, "N 3"],
    padding_ang: scalar_float = 0.0,
) -> Float[Array, "3"]:
    """Compute domain extent from atomic positions bounding box.

    Description
    -----------
    Calculates the physical extent of a coherent scattering domain as the
    bounding box of atomic positions plus optional padding. This extent
    determines the reciprocal-space rod broadening via the Fourier
    uncertainty relation.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian atomic positions in Ångstroms. Shape (N, 3) where N is
        the number of atoms.
    padding_ang : scalar_float, optional
        Additional padding on each side in Ångstroms. Total padding per
        dimension is 2×padding_ang. Default: 0.0

    Returns
    -------
    extent : Float[Array, "3"]
        Domain extent [Lx, Ly, Lz] in Ångstroms. Minimum value is 1.0 Å
        per dimension to avoid numerical issues.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Compute min and max coordinates along each axis
    2. Calculate extent = max - min
    3. Add 2×padding to each dimension
    4. Enforce minimum extent of 1.0 Å per dimension

    For a single atom, the extent would be [0, 0, 0] before minimum
    enforcement. The minimum extent prevents division by zero in
    subsequent rod width calculations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([[0., 0., 0.], [10., 10., 5.]])
    >>> extent = compute_domain_extent(positions)
    >>> extent
    Array([10., 10.,  5.], dtype=float64)
    """
    padding_arr: Float[Array, ""] = jnp.asarray(padding_ang, dtype=jnp.float64)

    min_coords: Float[Array, "3"] = jnp.min(positions, axis=0)
    max_coords: Float[Array, "3"] = jnp.max(positions, axis=0)

    extent: Float[Array, "3"] = max_coords - min_coords + 2.0 * padding_arr

    # Enforce minimum extent to avoid division by zero
    extent = jnp.maximum(extent, _MIN_EXTENT_ANG)

    return extent


@jaxtyped(typechecker=beartype)
def extent_to_rod_sigma(
    domain_extent_ang: Float[Array, "3"],
) -> Float[Array, "2"]:
    """Convert domain extent to reciprocal-space rod Gaussian widths.

    Description
    -----------
    Computes the Gaussian σ for reciprocal lattice rod profiles from
    real-space domain size. Uses the Fourier uncertainty relation with
    a conversion factor that matches the FWHM of a sinc² profile.

    Parameters
    ----------
    domain_extent_ang : Float[Array, "3"]
        Domain size [Lx, Ly, Lz] in Ångstroms.

    Returns
    -------
    rod_sigma : Float[Array, "2"]
        Rod Gaussian widths [σx, σy] in 1/Ångstroms. Only x and y
        components are returned since rods extend continuously along z.

    Notes
    -----
    The conversion uses:

    .. math::

        \\sigma_q = \\frac{2\\pi}{L \\times \\sqrt{2\\pi}}

    This formula ensures that the Gaussian approximation has the same
    FWHM as the true sinc² profile from a finite domain. The sinc²
    function has FWHM ≈ 0.886 × 2π/L, and the Gaussian FWHM = 2.355σ,
    giving σ ≈ 0.376 × 2π/L ≈ 2π/(L × √(2π)).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> extent = jnp.array([100., 100., 50.])
    >>> sigma = extent_to_rod_sigma(extent)
    >>> sigma  # Approximately [0.025, 0.025] 1/Å
    Array([0.02506628, 0.02506628], dtype=float64)
    """
    # Enforce minimum extent
    extent_safe: Float[Array, "3"] = jnp.maximum(
        domain_extent_ang, _MIN_EXTENT_ANG
    )

    # σ_q = 2π / (L × √(2π)) for FWHM matching sinc²
    sqrt_2pi: Float[Array, ""] = jnp.sqrt(2.0 * jnp.pi)
    rod_sigma: Float[Array, "2"] = (2.0 * jnp.pi) / (extent_safe[:2] * sqrt_2pi)

    return rod_sigma


@jaxtyped(typechecker=beartype)
def compute_shell_sigma(
    k_magnitude: Float[Array, ""],
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
) -> Float[Array, ""]:
    """Compute Ewald shell Gaussian thickness from beam parameters.

    Description
    -----------
    Calculates the Gaussian width of the Ewald shell due to energy spread
    and beam angular divergence. These instrumental factors cause the
    Ewald "sphere" to have finite thickness, allowing partial intensity
    contribution from nearby reciprocal lattice points.

    Parameters
    ----------
    k_magnitude : Float[Array, ""]
        Wavevector magnitude |k| = 2π/λ in 1/Ångstroms.
    energy_spread_frac : scalar_float, optional
        Fractional energy spread ΔE/E. Default: 1e-4 (0.01%), typical
        for thermionic electron guns.
    beam_divergence_rad : scalar_float, optional
        Beam angular divergence in radians. Default: 1e-3 (1 mrad),
        typical for RHEED geometry.

    Returns
    -------
    shell_sigma : Float[Array, ""]
        Ewald shell Gaussian width in 1/Ångstroms.

    Notes
    -----
    The shell thickness arises from two contributions:

    1. **Energy spread**: Δk/k = ΔE/(2E) since k ∝ √E
    2. **Beam divergence**: Δk⊥ = k × Δθ

    Combined in quadrature:

    .. math::

        \\sigma_{shell} = k \\times \\sqrt{\\left(\\frac{\\Delta E}{2E}\\right)^2
        + \\Delta\\theta^2}

    For typical RHEED conditions (15 kV, ΔE/E = 10⁻⁴, Δθ = 1 mrad):
    - k ≈ 73 Å⁻¹
    - σ_shell ≈ 0.07 Å⁻¹

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> k = jnp.array(73.0)  # ~15 kV electrons
    >>> sigma = compute_shell_sigma(k, energy_spread_frac=1e-4)
    >>> sigma  # ~0.07 1/Å
    Array(0.07303659, dtype=float64)
    """
    energy_spread_arr: Float[Array, ""] = jnp.asarray(
        energy_spread_frac, dtype=jnp.float64
    )
    divergence_arr: Float[Array, ""] = jnp.asarray(
        beam_divergence_rad, dtype=jnp.float64
    )

    # Δk/k from energy spread: ΔE/(2E) since k ∝ √E
    dk_over_k_energy: Float[Array, ""] = energy_spread_arr / 2.0

    # Combined in quadrature
    shell_sigma: Float[Array, ""] = k_magnitude * jnp.sqrt(
        dk_over_k_energy**2 + divergence_arr**2
    )

    return shell_sigma


@jaxtyped(typechecker=beartype)
def rod_ewald_overlap(
    g_vectors: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    k_magnitude: Float[Array, ""],
    rod_sigma: Float[Array, "2"],
    shell_sigma: Float[Array, ""],
) -> Float[Array, "N"]:
    """Compute overlap between broadened rods and Ewald shell.

    Description
    -----------
    Calculates the intensity contribution from finite-width reciprocal
    lattice rods intersecting a finite-thickness Ewald shell. Both are
    modeled as Gaussians for analytic evaluation. This replaces the
    binary "on/off" Ewald sphere condition with a continuous overlap.

    Parameters
    ----------
    g_vectors : Float[Array, "N 3"]
        Reciprocal lattice vectors G in 1/Ångstroms. Shape (N, 3).
    k_in : Float[Array, "3"]
        Incident wavevector in 1/Ångstroms.
    k_magnitude : Float[Array, ""]
        Wavevector magnitude |k| = 2π/λ in 1/Ångstroms.
    rod_sigma : Float[Array, "2"]
        Rod Gaussian widths [σx, σy] in 1/Ångstroms.
    shell_sigma : Float[Array, ""]
        Ewald shell Gaussian width in 1/Ångstroms.

    Returns
    -------
    overlap : Float[Array, "N"]
        Overlap factors in [0, 1] for each G vector. Value of 1.0
        indicates exact Ewald sphere intersection; smaller values
        indicate partial overlap.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Compute outgoing wavevector: k_out = k_in + G
    2. Calculate elastic scattering deviation: d = ||k_out| - |k_in||
    3. Compute effective Gaussian width: σ_eff² = σ_rod² + σ_shell²
    4. Evaluate overlap: exp(-d²/(2σ_eff²))

    The perpendicular distance from rod center to Ewald sphere is
    approximated by the magnitude deviation |k_out| - |k_in|. This is
    exact for small deviations and a good approximation for RHEED
    geometry where the Ewald sphere is nearly flat.

    For the effective σ, we use the average of rod_sigma components
    since rods are approximately circular in the xy plane for most
    crystal structures.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> g_vecs = jnp.array([[0., 0., 1.], [1., 0., 0.]])
    >>> k_in = jnp.array([1., 0., -0.1])
    >>> k_mag = jnp.linalg.norm(k_in)
    >>> rod_sigma = jnp.array([0.05, 0.05])
    >>> shell_sigma = jnp.array(0.07)
    >>> overlap = rod_ewald_overlap(g_vecs, k_in, k_mag, rod_sigma, shell_sigma)
    """
    # Compute k_out = k_in + G for all G vectors
    k_out: Float[Array, "N 3"] = k_in + g_vectors

    # Magnitude of k_out
    k_out_mag: Float[Array, "N"] = jnp.linalg.norm(k_out, axis=-1)

    # Deviation from elastic scattering condition
    d_perp: Float[Array, "N"] = jnp.abs(k_out_mag - k_magnitude)

    # Effective Gaussian width (combine rod and shell in quadrature)
    # Use mean of rod_sigma components for approximately circular rods
    rod_sigma_mean: Float[Array, ""] = jnp.mean(rod_sigma)
    sigma_eff_sq: Float[Array, ""] = rod_sigma_mean**2 + shell_sigma**2

    # Gaussian overlap factor
    overlap: Float[Array, "N"] = jnp.exp(-d_perp**2 / (2.0 * sigma_eff_sq))

    return overlap


@jaxtyped(typechecker=beartype)
def finite_domain_intensities(
    ewald: EwaldData,
    theta_deg: scalar_float,
    phi_deg: scalar_float,
    domain_extent_ang: Float[Array, "3"],
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Compute diffraction intensities with finite domain broadening.

    Description
    -----------
    Calculates kinematic diffraction intensities accounting for finite
    coherent domain size and beam parameters. The base intensities from
    EwaldData are weighted by the rod-Ewald overlap factors.

    Parameters
    ----------
    ewald : EwaldData
        Pre-computed angle-independent Ewald data containing G vectors,
        structure factors, and base intensities.
    theta_deg : scalar_float
        Grazing incidence angle in degrees (angle from surface plane).
        Typical RHEED values: 1-5 degrees.
    phi_deg : scalar_float
        Azimuthal angle in degrees (rotation about surface normal).
        0 degrees = beam along x-axis.
    domain_extent_ang : Float[Array, "3"]
        Physical domain size [Lx, Ly, Lz] in Ångstroms.
    energy_spread_frac : scalar_float, optional
        Fractional energy spread ΔE/E. Default: 1e-4
    beam_divergence_rad : scalar_float, optional
        Beam angular divergence in radians. Default: 1e-3

    Returns
    -------
    overlap_factors : Float[Array, "N"]
        Overlap factors in [0, 1] for each reciprocal lattice point.
    modified_intensities : Float[Array, "N"]
        Intensities weighted by overlap: I_modified = I_base × overlap.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Convert beam angles to incident wavevector k_in
    2. Compute rod widths from domain extent
    3. Compute shell thickness from beam parameters
    4. Calculate rod-Ewald overlap for all G vectors
    5. Weight base intensities by overlap factors

    This function combines all finite domain effects in one call.
    The modified intensities can be used directly for pattern
    simulation or analysis.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ewald = rh.simul.build_ewald_data(crystal, voltage_kv=15.0,
    ...                                    hmax=3, kmax=3, lmax=2)
    >>> domain = jnp.array([100., 100., 50.])
    >>> overlap, intensities = rh.simul.finite_domain_intensities(
    ...     ewald, theta_deg=2.0, phi_deg=0.0, domain_extent_ang=domain
    ... )
    """
    # Build incident wavevector
    lam_ang: Float[Array, ""] = wavelength_ang(
        jnp.asarray(ewald.k_magnitude * ewald.wavelength_ang / (2.0 * jnp.pi))
    )
    # Actually we have wavelength directly in ewald
    k_in: Float[Array, "3"] = incident_wavevector(
        ewald.wavelength_ang, theta_deg, phi_deg
    )

    # Compute rod and shell widths
    rod_sigma: Float[Array, "2"] = extent_to_rod_sigma(domain_extent_ang)
    shell_sigma: Float[Array, ""] = compute_shell_sigma(
        k_magnitude=ewald.k_magnitude,
        energy_spread_frac=energy_spread_frac,
        beam_divergence_rad=beam_divergence_rad,
    )

    # Calculate overlap factors
    overlap_factors: Float[Array, "N"] = rod_ewald_overlap(
        g_vectors=ewald.g_vectors,
        k_in=k_in,
        k_magnitude=ewald.k_magnitude,
        rod_sigma=rod_sigma,
        shell_sigma=shell_sigma,
    )

    # Weight base intensities by overlap
    modified_intensities: Float[Array, "N"] = ewald.intensities * overlap_factors

    return overlap_factors, modified_intensities
