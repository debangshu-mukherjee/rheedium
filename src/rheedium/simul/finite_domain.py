r"""Finite domain Ewald sphere broadening for RHEED simulation.

Extended Summary
----------------
This module provides functions for computing finite domain effects in RHEED
diffraction. Finite coherent domain size causes reciprocal lattice rods to
broaden, and the Ewald sphere intersection becomes a continuous overlap
integral rather than a binary hit. Combined with finite Ewald shell thickness
from beam energy spread and divergence, this enables realistic simulation of
surfaces with limited coherence length.

The overlap model operates on (h, k) **rods**, not on discrete (h, k, l)
points: for each rod the rod-sphere quadratic is solved, real intersections
with :math:`k_{out,z} > 0` contribute at the continuous intersection
:math:`l^*` (composing with the semi-infinite CTR truncation shape), lateral
misses are weighted by a Gaussian in the closed-form miss distance
:math:`d_{miss} = \\sqrt{-\\Delta} / (2\\sqrt{a})`, and the finite domain
thickness enters as a Gauss-Hermite window over :math:`\\delta l` with
:math:`\\sigma_l = \\sigma_z / |b_3|`.

Routine Listings
----------------
:func:`compute_domain_extent`
    Compute domain extent from atomic positions bounding box.
:func:`compute_shell_sigma`
    Compute Ewald shell Gaussian thickness from beam parameters.
:func:`extent_to_rod_sigma`
    Convert domain extent to reciprocal-space rod widths.
:func:`find_ctr_ewald_intersection`
    Find intersection of a (h, k) CTR with the Ewald sphere.
:func:`finite_domain_intensities`
    Compute intensities with rod-based finite domain broadening.
:func:`finite_domain_intensities_for_size_distribution`
    Incoherently average finite-domain intensities over a SizeDistribution.
:func:`rod_domain_overlap`
    Rod-based overlap between broadened (h, k) rods and the Ewald shell.

Notes
-----
For a domain of size L, the reciprocal lattice rod has a Gaussian width
matched to the sinc² shape function: the sinc² FWHM is 0.886·(2π/L) and the
Gaussian FWHM is 2.355σ, giving σ = (0.886/2.355)·(2π/L) ≈ 0.376·(2π/L).

Physical origins of broadening:

1. **Rod broadening** (from finite domain size):
   - Coherent scattering only within domain
   - Rod FWHM ≈ 0.886 × 2π/L in reciprocal space
   - Gaussian approximation: σ = 0.376 × 2π/L per dimension

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

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from rheedium.tools import (
    gauss_hermite_nodes_weights,
    incident_wavevector,
    safe_sqrt,
)
from rheedium.types import (
    Distribution,
    EwaldData,
    SizeDistribution,
    scalar_float,
    scalar_int,
    size_to_distribution,
)

from .surface_rods import ctr_truncation_intensity

_MIN_EXTENT_ANG: float = 1.0
# Gaussian sigma matching the sinc^2 shape-function FWHM:
# sinc^2 FWHM = 0.886 * (2 pi / L); Gaussian FWHM = 2.355 sigma;
# sigma = (0.886 / 2.355) * (2 pi / L) = 0.376 * (2 pi / L).
_SINC2_SIGMA_FACTOR: float = 0.886 / 2.355
_N_GAUSS_HERMITE_L: int = 7


@jaxtyped(typechecker=beartype)
def compute_domain_extent(
    positions: Float[Array, "N 3"],
    padding_ang: scalar_float = 0.0,
) -> Float[Array, "3"]:
    r"""Compute domain extent from atomic positions bounding box.

    Calculates the physical extent of a coherent scattering domain as the
    bounding box of atomic positions plus optional padding. This extent
    determines the reciprocal-space rod broadening via the Fourier
    uncertainty relation.

    :see: :class:`~.test_finite_domain.TestComputeDomainExtent`

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
    1. **Bounding box** --
       Compute min and max coordinates along each axis.
    2. **Raw extent** --
       :math:`L = \\text{max} - \\text{min}`.
    3. **Add padding** --
       :math:`L \\leftarrow L + 2 \\times \\text{padding}`.
    4. **Enforce minimum** --
       Clip to 1.0 Å per dimension to prevent division
       by zero in rod width calculations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([[0.0, 0.0, 0.0], [10.0, 10.0, 5.0]])
    >>> extent = compute_domain_extent(positions)
    >>> extent
    Array([10., 10.,  5.], dtype=float64)

    See Also
    --------
    extent_to_rod_sigma : Convert extent to reciprocal-space widths
    finite_domain_intensities : Full intensity calculation
    """
    padding_arr: Float[Array, ""] = jnp.asarray(padding_ang, dtype=jnp.float64)
    min_coords: Float[Array, "3"] = jnp.min(positions, axis=0)
    max_coords: Float[Array, "3"] = jnp.max(positions, axis=0)
    extent: Float[Array, "3"] = max_coords - min_coords + 2.0 * padding_arr
    extent: Float[Array, "3"] = jnp.maximum(extent, _MIN_EXTENT_ANG)
    return extent


@jaxtyped(typechecker=beartype)
def extent_to_rod_sigma(
    domain_extent_ang: Float[Array, "3"],
) -> Float[Array, "3"]:
    r"""Convert domain extent to reciprocal-space rod Gaussian widths.

    Computes the Gaussian σ for reciprocal lattice rod profiles from
    real-space domain size for **all three** dimensions. Uses the Fourier
    uncertainty relation with a conversion factor that matches the FWHM
    of a sinc² profile.

    :see: :class:`~.test_finite_domain.TestExtentToRodSigma`
    :see: :func:`~.test_physics_anchors.test_rod_sigma_matches_sinc_fwhm`

    Parameters
    ----------
    domain_extent_ang : Float[Array, "3"]
        Domain size [Lx, Ly, Lz] in Ångstroms.

    Returns
    -------
    rod_sigma : Float[Array, "3"]
        Rod Gaussian widths [σx, σy, σz] in 1/Ångstroms. σx and σy are
        the lateral rod widths; σz sets the finite-thickness l-window
        used by the rod-based overlap.

    Notes
    -----
    The conversion uses:

    .. math::

        \\sigma_q = \\frac{0.886}{2.355} \\cdot \\frac{2\\pi}{L}
        \\approx 0.376 \\cdot \\frac{2\\pi}{L}

    The finite-domain shape function is sinc²(q L / 2) with FWHM
    0.886 × 2π/L; a Gaussian has FWHM 2.355σ, so matching the two
    gives σ = (0.886/2.355) × 2π/L ≈ 2.363/L. (The previous constant
    2π/(L√(2π)) ≈ 2.507/L was ≈6% too wide.)

    1. **Enforce minimum extent** --
       Clip domain sizes to 1.0 Å minimum.
    2. **Fourier relation** --
       :math:`\\sigma_q = 0.376 \\times 2\\pi / L` for each dimension.
    3. **Return all three widths** --
       :math:`[\\sigma_x, \\sigma_y, \\sigma_z]`; the z-width enters the
       finite-thickness l-window (the domain thickness no longer drops
       out of the model).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> extent = jnp.array([100.0, 100.0, 50.0])
    >>> sigma = extent_to_rod_sigma(extent)
    >>> sigma  # Approximately [0.0236, 0.0236, 0.0473] 1/Å
    Array([0.02363736, 0.02363736, 0.04727471], dtype=float64)

    See Also
    --------
    compute_domain_extent : Calculate extent from atomic positions
    rod_domain_overlap : Use rod widths in the rod-based overlap
    """
    extent_safe: Float[Array, "3"] = jnp.maximum(
        domain_extent_ang, _MIN_EXTENT_ANG
    )
    rod_sigma: Float[Array, "3"] = (
        _SINC2_SIGMA_FACTOR * 2.0 * jnp.pi / extent_safe
    )
    return rod_sigma


@jaxtyped(typechecker=beartype)
def compute_shell_sigma(
    k_magnitude: Float[Array, ""],
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
) -> Float[Array, ""]:
    r"""Compute Ewald shell Gaussian thickness from beam parameters.

    Calculates the Gaussian width of the Ewald shell due to energy spread
    and beam angular divergence. These instrumental factors cause the
    Ewald "sphere" to have finite thickness, allowing partial intensity
    contribution from nearby reciprocal lattice points.

    :see: :class:`~.test_finite_domain.TestComputeShellSigma`

    Parameters
    ----------
    k_magnitude : Float[Array, ""]
        Wavevector magnitude :math:`|k| = 2π/λ` in 1/Ångstroms.
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

        \\sigma_{shell} = k \\times
        \\sqrt{\\left(\\frac{\\Delta E}{2E}\\right)^2
        + \\Delta\\theta^2}


    For typical RHEED conditions (15 kV, ΔE/E = 10⁻⁴, Δθ = 1 mrad):
    - k ≈ 73 Å⁻¹
    - σ_shell ≈ 0.07 Å⁻¹

    1. **Energy contribution** --
       :math:`\\Delta k / k = \\Delta E / (2E)`.
    2. **Divergence contribution** --
       :math:`\\Delta k_{\\perp} = k \\times \\Delta\\theta`.
    3. **Combine in quadrature** --
       :math:`\\sigma_{shell} = k \\sqrt{(\\Delta E/2E)^2
       + \\Delta\\theta^2}`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> k = jnp.array(73.0)  # ~15 kV electrons
    >>> sigma = compute_shell_sigma(k, energy_spread_frac=1e-4)
    >>> sigma  # ~0.07 1/Å
    Array(0.07303659, dtype=float64)

    See Also
    --------
    rod_domain_overlap : Use shell width in overlap calculation
    finite_domain_intensities : Full intensity calculation
    """
    energy_spread_arr: Float[Array, ""] = jnp.asarray(
        energy_spread_frac, dtype=jnp.float64
    )
    divergence_arr: Float[Array, ""] = jnp.asarray(
        beam_divergence_rad, dtype=jnp.float64
    )
    dk_over_k_energy: Float[Array, ""] = energy_spread_arr / 2.0
    shell_sigma: Float[Array, ""] = k_magnitude * safe_sqrt(
        dk_over_k_energy**2 + divergence_arr**2
    )
    return shell_sigma


@jaxtyped(typechecker=beartype)
def find_ctr_ewald_intersection(
    h: scalar_int,
    k: scalar_int,
    k_in: Float[Array, "3"],
    recip_a: Float[Array, "3"],
    recip_b: Float[Array, "3"],
    recip_c: Float[Array, "3"],
) -> Tuple[Float[Array, "2"], Float[Array, "2 3"], Float[Array, "2"]]:
    """Find where a crystal truncation rod intersects the Ewald sphere.

    Solves the quadratic rod-sphere equation and returns both valid
    upward-scattering branches, if present.

    :see: :class:`~.test_simulator.TestEwaldSimulator`

    Parameters
    ----------
    h : scalar_int
        Miller index h for the rod.
    k : scalar_int
        Miller index k for the rod.
    k_in : Float[Array, "3"]
        Incident wavevector.
    recip_a : Float[Array, "3"]
        First reciprocal lattice vector (a*).
    recip_b : Float[Array, "3"]
        Second reciprocal lattice vector (b*).
    recip_c : Float[Array, "3"]
        Third reciprocal lattice vector (c*), defines rod direction.

    Returns
    -------
    l_intersect : Float[Array, "2"]
        The plus and minus branch l values, zeroed when invalid.
    k_out : Float[Array, "2 3"]
        Outgoing wavevectors for the plus and minus branches.
    valid : Float[Array, "2"]
        Branch validity flags as floats in ``{0.0, 1.0}``.

    Notes
    -----
    1. **Rod line** --
       Parameterize the rod as :math:`k_{in} + h a^* + k b^* + l c^*`.
    2. **Quadratic** --
       Insert into :math:`|k_{out}|^2 = |k_{in}|^2` to obtain
       :math:`a l^2 + b l + c = 0`.
    3. **Branches** --
       Solve for both roots; keep those with positive discriminant and
       :math:`k_{out,z} > 0`.
    """
    g_hk: Float[Array, "3"] = h * recip_a + k * recip_b
    c_star: Float[Array, "3"] = recip_c
    k_mag_sq: Float[Array, ""] = jnp.dot(k_in, k_in)
    p_vec: Float[Array, "3"] = k_in + g_hk
    a_coef: Float[Array, ""] = jnp.dot(c_star, c_star)
    b_coef: Float[Array, ""] = 2.0 * jnp.dot(p_vec, c_star)
    c_coef: Float[Array, ""] = jnp.dot(p_vec, p_vec) - k_mag_sq
    discriminant: Float[Array, ""] = b_coef**2 - 4.0 * a_coef * c_coef
    safe_disc: Float[Array, ""] = jnp.maximum(discriminant, 1e-30)
    sqrt_disc: Float[Array, ""] = jnp.sqrt(safe_disc)
    l_plus: Float[Array, ""] = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    l_minus: Float[Array, ""] = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    k_out_plus: Float[Array, "3"] = p_vec + l_plus * c_star
    k_out_minus: Float[Array, "3"] = p_vec + l_minus * c_star
    valid_plus: Float[Array, ""] = (discriminant >= 0) & (k_out_plus[2] > 0)
    valid_minus: Float[Array, ""] = (discriminant >= 0) & (k_out_minus[2] > 0)
    l_intersect: Float[Array, "2"] = jnp.array([l_plus, l_minus])
    k_out: Float[Array, "2 3"] = jnp.stack([k_out_plus, k_out_minus], axis=0)
    valid_mask: Bool[Array, "2"] = jnp.array([valid_plus, valid_minus])
    l_intersect = jnp.where(valid_mask, l_intersect, 0.0)
    k_out = jnp.where(valid_mask[:, None], k_out, 0.0)
    valid: Float[Array, "2"] = valid_mask.astype(jnp.float64)
    return l_intersect, k_out, valid


@jaxtyped(typechecker=beartype)
def _point_domain_overlap(
    g_vectors: Float[Array, "N 3"],
    k_in: Float[Array, "3"],
    k_magnitude: Float[Array, ""],
    rod_sigma: Float[Array, "2"],
    shell_sigma: Float[Array, ""],
) -> Float[Array, "N"]:
    r"""Legacy point-based Gaussian overlap (reference only, unused).

    This is the retired point-based model: it broadens each discrete
    (h, k, l) point radially instead of treating the (h, k) rod as
    continuous along l, so inter-order intensity exists only where an
    integer-l point happens to sit near the sphere and the domain
    thickness never enters. Kept privately for reference and tests; the
    production path is :func:`rod_domain_overlap`.

    :see: :class:`~.test_finite_domain.TestPointDomainOverlap`

    Parameters
    ----------
    g_vectors : Float[Array, "N 3"]
        Reciprocal lattice vectors G in 1/Ångstroms. Shape (N, 3).
    k_in : Float[Array, "3"]
        Incident wavevector in 1/Ångstroms.
    k_magnitude : Float[Array, ""]
        Wavevector magnitude :math:`|k| = 2π/λ` in 1/Ångstroms.
    rod_sigma : Float[Array, "2"]
        Rod Gaussian widths [σx, σy] in 1/Ångstroms.
    shell_sigma : Float[Array, ""]
        Ewald shell Gaussian width in 1/Ångstroms.

    Returns
    -------
    overlap : Float[Array, "N"]
        Overlap factors in [0, 1] for each G vector.

    Notes
    -----
    1. **Outgoing wavevector** --
       :math:`k_{out} = k_{in} + G`.
    2. **Radial distance** --
       :math:`d = ||k_{out}| - |k_{in}||`.
    3. **Anisotropic rod width** --
       Effective width from the azimuthal angle of :math:`k_{out}`.
    4. **Evaluate overlap** --
       :math:`\exp(-d^2 / (2 \sigma_{eff}^2))`.
    """
    k_out: Float[Array, "N 3"] = k_in + g_vectors
    k_out_mag: Float[Array, "N"] = jnp.linalg.norm(k_out, axis=-1)
    d_perp: Float[Array, "N"] = jnp.abs(k_out_mag - k_magnitude)
    k_out_xy: Float[Array, "N 2"] = k_out[:, :2]
    k_out_xy_mag: Float[Array, "N"] = jnp.linalg.norm(k_out_xy, axis=-1)
    k_out_xy_mag_safe: Float[Array, "N"] = jnp.maximum(k_out_xy_mag, 1e-10)
    cos_phi: Float[Array, "N"] = k_out[:, 0] / k_out_xy_mag_safe
    sin_phi: Float[Array, "N"] = k_out[:, 1] / k_out_xy_mag_safe
    rod_sigma_x: Float[Array, ""] = rod_sigma[0]
    rod_sigma_y: Float[Array, ""] = rod_sigma[1]
    rod_sigma_eff_sq: Float[Array, "N"] = (rod_sigma_x * cos_phi) ** 2 + (
        rod_sigma_y * sin_phi
    ) ** 2
    vertical_threshold: float = 1e-8
    is_vertical: Bool[Array, "N"] = k_out_xy_mag < vertical_threshold
    rod_sigma_mean_sq: Float[Array, ""] = (rod_sigma_x**2 + rod_sigma_y**2) / 2
    rod_sigma_eff_sq: Float[Array, "N"] = jnp.where(
        is_vertical, rod_sigma_mean_sq, rod_sigma_eff_sq
    )
    sigma_eff_sq: Float[Array, "N"] = rod_sigma_eff_sq + shell_sigma**2
    overlap: Float[Array, "N"] = jnp.exp(-(d_perp**2) / (2.0 * sigma_eff_sq))
    return overlap


@jaxtyped(typechecker=beartype)
def rod_domain_overlap(  # noqa: PLR0913
    hkl_points: Int[Array, "N 3"] | Float[Array, "N 3"],
    recip_vectors: Float[Array, "3 3"],
    k_in: Float[Array, "3"],
    k_magnitude: Float[Array, ""],
    rod_sigma: Float[Array, "3"],
    shell_sigma: Float[Array, ""],
    layer_attenuation: scalar_float = 0.01,
) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N 3"]]:
    r"""Rod-based overlap between broadened (h, k) rods and the Ewald shell.

    Treats each (h, k) rod as **continuous along l**: the rod-sphere
    quadratic is solved per rod, real intersections with
    :math:`k_{out,z} > 0` receive unit envelope weight at the continuous
    intersection :math:`l^*`, lateral misses receive
    :math:`\exp(-d_{miss}^2 / (2\sigma_{lat}^2 + 2\sigma_{shell}^2))`
    with the closed-form miss distance
    :math:`d_{miss} = \sqrt{-\Delta} / (2\sqrt{a})`, and the finite
    domain thickness enters as a Gauss-Hermite window (7 points) of the
    CTR truncation intensity over :math:`\delta l` with
    :math:`\sigma_l = \sigma_z / |b_3|`.

    The input is keyed per (h, k, l) grid point only for fixed-shape JIT
    compatibility: each rod's contribution is assigned to exactly one
    **representative** grid point on that rod (the one whose integer l is
    nearest to :math:`l^*`, clipped to the grid's l range), so summing
    over grid points counts every rod exactly once per intersection
    branch.

    :see: :class:`~.test_finite_domain.TestRodDomainOverlap`
    :see: :func:`~.test_physics_anchors.test_rod_miss_distance_closed_form`

    Parameters
    ----------
    hkl_points : Int[Array, "N 3"] | Float[Array, "N 3"]
        Miller indices (h, k, l) of the reciprocal grid points. The
        (h, k) columns key the rods; the l column selects representative
        slots.
    recip_vectors : Float[Array, "3 3"]
        Reciprocal lattice basis [b1, b2, b3] as rows in 1/Ångstroms.
    k_in : Float[Array, "3"]
        Incident wavevector in 1/Ångstroms.
    k_magnitude : Float[Array, ""]
        Wavevector magnitude :math:`|k| = 2π/λ` in 1/Ångstroms.
    rod_sigma : Float[Array, "3"]
        Rod Gaussian widths [σx, σy, σz] in 1/Ångstroms from
        :func:`extent_to_rod_sigma`.
    shell_sigma : Float[Array, ""]
        Ewald shell Gaussian width in 1/Ångstroms.
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation ε of the CTR truncation factor
        evaluated along the rod. Default: 0.01

    Returns
    -------
    overlap : Float[Array, "N"]
        Envelope weights in [0, 1]: 1 where the rod truly intersects the
        sphere, the lateral-miss Gaussian otherwise, and 0 for
        non-representative grid points.
    rod_factor : Float[Array, "N"]
        Envelope weight × Gauss-Hermite-windowed CTR truncation
        intensity at :math:`l^*`; multiply by the point's :math:`|F|^2`
        to obtain the rod-based intensity.
    k_out : Float[Array, "N 3"]
        Outgoing wavevectors evaluated at the continuous intersection
        (or closest approach) of the dominant branch.

    Notes
    -----
    The closed-form miss distance follows from the rod-sphere quadratic
    :math:`f(l) = |k_{in} + G_{hk} + l b_3|^2 - k^2
    = a l^2 + b l + c`: at the closest approach
    :math:`l^* = -b/(2a)` the minimum is
    :math:`f(l^*) = -\Delta/(4a)` with :math:`\Delta = b^2 - 4ac`, so
    :math:`d_{miss} = \sqrt{f(l^*)} = \sqrt{-\Delta}/(2\sqrt{a})`
    satisfies :math:`\min_l |k_{out}(l)|^2 = k^2 + d_{miss}^2` — the
    lateral distance between the rod line and the sphere in quadrature.

    1. **Rod quadratic** --
       For each point's (h, k), build
       :math:`p = k_{in} + h b_1 + k b_2` and solve
       :math:`a l^2 + b l + c = 0` along :math:`b_3`.
    2. **Intersection branches** --
       Real roots with :math:`k_{out,z} > 0` get weight 1 at
       :math:`l^* = l_\pm`.
    3. **Lateral miss** --
       Negative discriminant: evaluate at :math:`l^* = -b/(2a)` with
       weight :math:`\exp(-d_{miss}^2/(2\sigma_{lat}^2 +
       2\sigma_{shell}^2))`, σ_lat from the azimuth-weighted lateral
       widths.
    4. **Finite thickness** --
       Convolve the truncation intensity with a Gaussian l-window of
       width :math:`\sigma_l = \sigma_z/|b_3|` by 7-point Gauss-Hermite
       quadrature.
    5. **Representative gating** --
       Assign each branch to the grid point with the nearest integer l
       (clipped to the grid range) so each rod is counted once.

    See Also
    --------
    find_ctr_ewald_intersection : Single-rod intersection solver.
    extent_to_rod_sigma : Rod widths from the domain extent.
    finite_domain_intensities : Consumer applying these factors.
    """
    hkl: Float[Array, "N 3"] = jnp.asarray(hkl_points, dtype=jnp.float64)
    b1: Float[Array, "3"] = recip_vectors[0]
    b2: Float[Array, "3"] = recip_vectors[1]
    b3: Float[Array, "3"] = recip_vectors[2]
    b3_norm: Float[Array, ""] = jnp.linalg.norm(b3)
    p_vec: Float[Array, "N 3"] = (
        k_in[None, :] + hkl[:, 0:1] * b1[None, :] + hkl[:, 1:2] * b2[None, :]
    )
    a_coef: Float[Array, ""] = jnp.dot(b3, b3)
    b_coef: Float[Array, "N"] = 2.0 * jnp.einsum("nj,j->n", p_vec, b3)
    c_coef: Float[Array, "N"] = (
        jnp.einsum("nj,nj->n", p_vec, p_vec) - k_magnitude**2
    )
    disc: Float[Array, "N"] = b_coef**2 - 4.0 * a_coef * c_coef
    has_intersection: Bool[Array, "N"] = disc >= 0.0
    sqrt_disc: Float[Array, "N"] = safe_sqrt(disc)
    d_miss: Float[Array, "N"] = safe_sqrt(-disc) / (2.0 * jnp.sqrt(a_coef))
    l_closest: Float[Array, "N"] = -b_coef / (2.0 * a_coef)
    # Branch l values: the two intersections when they exist, otherwise
    # the closest approach carried on branch 0 only.
    l_plus: Float[Array, "N"] = l_closest + sqrt_disc / (2.0 * a_coef)
    l_minus: Float[Array, "N"] = l_closest - sqrt_disc / (2.0 * a_coef)
    l_branches: Float[Array, "N 2"] = jnp.stack(
        [
            jnp.where(has_intersection, l_plus, l_closest),
            jnp.where(has_intersection, l_minus, l_closest),
        ],
        axis=-1,
    )
    k_out_branches: Float[Array, "N 2 3"] = (
        p_vec[:, None, :] + l_branches[:, :, None] * b3[None, None, :]
    )
    upward: Bool[Array, "N 2"] = k_out_branches[:, :, 2] > 0.0
    # Lateral-miss Gaussian weight (miss case): sigma_lat from the
    # azimuth of k_out at the closest approach, combined with the shell.
    k_out_xy_mag: Float[Array, "N 2"] = jnp.linalg.norm(
        k_out_branches[:, :, :2], axis=-1
    )
    k_out_xy_safe: Float[Array, "N 2"] = jnp.maximum(k_out_xy_mag, 1e-10)
    cos_phi: Float[Array, "N 2"] = k_out_branches[:, :, 0] / k_out_xy_safe
    sin_phi: Float[Array, "N 2"] = k_out_branches[:, :, 1] / k_out_xy_safe
    sigma_lat_sq: Float[Array, "N 2"] = (rod_sigma[0] * cos_phi) ** 2 + (
        rod_sigma[1] * sin_phi
    ) ** 2
    is_vertical: Bool[Array, "N 2"] = k_out_xy_mag < 1e-8
    sigma_lat_mean_sq: Float[Array, ""] = (
        rod_sigma[0] ** 2 + rod_sigma[1] ** 2
    ) / 2.0
    sigma_lat_sq = jnp.where(is_vertical, sigma_lat_mean_sq, sigma_lat_sq)
    miss_weight: Float[Array, "N 2"] = jnp.exp(
        -(d_miss[:, None] ** 2) / (2.0 * sigma_lat_sq + 2.0 * shell_sigma**2)
    )
    branch_weight: Float[Array, "N 2"] = jnp.where(
        has_intersection[:, None], 1.0, miss_weight
    )
    # Branch validity: upward scattering; the miss case only occupies
    # branch 0 (branch 1 would double-count the same closest approach).
    branch_valid: Bool[Array, "N 2"] = upward & (
        has_intersection[:, None]
        | jnp.stack(
            [
                ~has_intersection,
                jnp.zeros_like(has_intersection),
            ],
            axis=-1,
        )
    )
    # Representative gating: this grid point carries the branch iff its
    # own integer l is the grid point nearest to l* on this rod.
    l_grid: Float[Array, "N"] = hkl[:, 2]
    l_lo: Float[Array, ""] = jnp.min(l_grid)
    l_hi: Float[Array, ""] = jnp.max(l_grid)
    l_nearest: Float[Array, "N 2"] = jnp.clip(
        jnp.round(l_branches), l_lo, l_hi
    )
    is_representative: Bool[Array, "N 2"] = l_nearest == l_grid[:, None]
    active: Float[Array, "N 2"] = (is_representative & branch_valid).astype(
        jnp.float64
    ) * branch_weight
    # Finite-thickness l-window: Gauss-Hermite average of the CTR
    # truncation intensity around l* with sigma_l = sigma_z / |b3|.
    gh_nodes: Float[Array, "Q"]
    gh_weights: Float[Array, "Q"]
    gh_nodes, gh_weights = gauss_hermite_nodes_weights(_N_GAUSS_HERMITE_L)
    sigma_l: Float[Array, ""] = rod_sigma[2] / b3_norm
    l_samples: Float[Array, "N 2 Q"] = (
        l_branches[:, :, None]
        + jnp.sqrt(2.0) * sigma_l * gh_nodes[None, None, :]
    )
    truncation_samples: Float[Array, "N 2 Q"] = ctr_truncation_intensity(
        l_values=l_samples,
        layer_attenuation=layer_attenuation,
    )
    truncation_window: Float[Array, "N 2"] = jnp.einsum(
        "q,nbq->nb", gh_weights, truncation_samples
    ) / jnp.sqrt(jnp.pi)
    overlap: Float[Array, "N"] = jnp.minimum(jnp.sum(active, axis=-1), 1.0)
    rod_factor: Float[Array, "N"] = jnp.sum(
        active * truncation_window, axis=-1
    )
    dominant: Int[Array, "N"] = jnp.argmax(active, axis=-1)
    k_out: Float[Array, "N 3"] = jnp.take_along_axis(
        k_out_branches, dominant[:, None, None], axis=1
    )[:, 0, :]
    return overlap, rod_factor, k_out


@jaxtyped(typechecker=beartype)
def rod_base_intensities(
    ewald: EwaldData,
    k_in: Float[Array, "3"],
    k_out_rod: Float[Array, "N 3"],
    parameterization: str = "lobato",
) -> Float[Array, "N"]:
    r"""Per-rod :math:`|F(q^*)|^2` at the continuous rod intersection.

    Evaluates the cell structure factor at the rod point actually selected
    by :func:`rod_domain_overlap` (:math:`q^* = k_{out} - k_{in}`), rather
    than at the nearest integer-``l`` grid point. Falls back to the stored
    integer-grid intensities with a warning when the ``EwaldData`` carries
    no atomic data (hand-built instances).

    Parameters
    ----------
    ewald : EwaldData
        Angle-independent Ewald data. Exact evaluation requires
        ``atom_positions``/``atomic_numbers`` (filled by
        :func:`build_ewald_data`).
    k_in : Float[Array, "3"]
        Incident wavevector in 1/Angstroms.
    k_out_rod : Float[Array, "N 3"]
        Rod-intersection outgoing wavevectors from
        :func:`rod_domain_overlap`.
    parameterization : str, optional
        Form-factor model; must match the one used to build ``ewald``.
        Default: ``"lobato"``.

    Returns
    -------
    base_intensities : Float[Array, "N"]
        :math:`|F(q^*)|^2` per rod slot (or grid intensities on fallback).

    Notes
    -----
    1. Fallback path returns ``ewald.intensities`` when atomic data are
       absent, warning that rod intensities use the integer-``l`` grid.
    2. Otherwise compute :math:`q^* = k_{out} - k_{in}` per slot and
       evaluate the structure factor with the same form-factor,
       per-site occupancy, and Debye-Waller machinery used on the grid
       (``ewald.occupancies``; ones when absent).
    """
    if ewald.atom_positions is None or ewald.atomic_numbers is None:
        warnings.warn(
            "EwaldData lacks atomic data; finite-domain rod intensities "
            "fall back to integer-l grid |F|^2. Build via build_ewald_data "
            "for exact evaluation at the rod intersection.",
            UserWarning,
            stacklevel=3,
        )
        return ewald.intensities
    from rheedium.simul.ewald import _compute_structure_factor_single

    temperature = (
        ewald.temperature
        if ewald.temperature is not None
        else jnp.asarray(300.0, dtype=jnp.float64)
    )
    q_star: Float[Array, "N 3"] = k_out_rod - k_in[None, :]

    def _one(q_vec: Float[Array, "3"]) -> Float[Array, ""]:
        structure_factor = _compute_structure_factor_single(
            g_vector=q_vec,
            atom_positions=ewald.atom_positions,
            atomic_numbers=ewald.atomic_numbers,
            temperature=temperature,
            parameterization=parameterization,
            occupancies=ewald.occupancies,
        )
        return jnp.abs(structure_factor) ** 2

    return jax.vmap(_one)(q_star)


@jaxtyped(typechecker=beartype)
def finite_domain_intensities(
    ewald: EwaldData,
    theta_deg: scalar_float,
    phi_deg: scalar_float,
    domain_extent_ang: Float[Array, "3"],
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
    layer_attenuation: scalar_float = 0.01,
    parameterization: str = "lobato",
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    r"""Compute diffraction intensities with rod-based domain broadening.

    Calculates kinematic diffraction intensities accounting for finite
    coherent domain size and beam parameters using the **rod-based**
    overlap: each (h, k) rod is continuous along l, its intensity is
    evaluated at the continuous rod-sphere intersection :math:`l^*`
    (CTR truncation shape, Gauss-Hermite windowed over the finite
    thickness), and the result is assigned to the grid point on that rod
    whose integer l is nearest to :math:`l^*`.

    :see: :class:`~.test_finite_domain.TestFiniteDomainIntensities`

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
        Physical domain size [Lx, Ly, Lz] in Ångstroms. The z-extent now
        enters through the finite-thickness l-window.
    energy_spread_frac : scalar_float, optional
        Fractional energy spread ΔE/E. Default: 1e-4
    beam_divergence_rad : scalar_float, optional
        Beam angular divergence in radians. Default: 1e-3
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation ε of the CTR truncation factor.
        Default: 0.01

    Returns
    -------
    overlap_factors : Float[Array, "N"]
        Rod envelope weights in [0, 1] for each reciprocal grid point;
        nonzero only on each rod's representative point.
    modified_intensities : Float[Array, "N"]
        Rod-based intensities
        :math:`I = |F|^2 \times w \times \langle T(l^*)\rangle_{GH}`
        using the representative point's structure factor as the
        single-cell :math:`|F|^2` along the rod.

    Notes
    -----
    1. **Incident wavevector** --
       Build :math:`k_{in}` from beam angles and wavelength.
    2. **Rod widths** --
       Convert domain extent to [σx, σy, σz] via
       :func:`extent_to_rod_sigma`.
    3. **Shell thickness** --
       Compute Ewald shell σ from energy spread and beam divergence.
    4. **Rod overlap** --
       Evaluate :func:`rod_domain_overlap` for all grid points.
    5. **Weight intensities** --
       :math:`I_{mod} = I_{base} \times \text{rod factor}`.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ewald = rh.simul.build_ewald_data(
    ...     crystal, energy_kev=15.0, hmax=3, kmax=3, lmax=2
    ... )
    >>> domain = jnp.array([100.0, 100.0, 50.0])
    >>> overlap, intensities = rh.simul.finite_domain_intensities(
    ...     ewald, theta_deg=2.0, phi_deg=0.0, domain_extent_ang=domain
    ... )

    See Also
    --------
    build_ewald_data : Create EwaldData for input
    ewald_allowed_reflections : Alternative with binary Ewald condition
    compute_domain_extent : Calculate domain extent from positions
    extent_to_rod_sigma : Rod width calculation
    compute_shell_sigma : Shell thickness calculation
    rod_domain_overlap : Core rod-based overlap calculation
    """
    k_in: Float[Array, "3"] = incident_wavevector(
        ewald.wavelength_ang, theta_deg, phi_deg
    )
    rod_sigma: Float[Array, "3"] = extent_to_rod_sigma(domain_extent_ang)
    shell_sigma: Float[Array, ""] = compute_shell_sigma(
        k_magnitude=ewald.k_magnitude,
        energy_spread_frac=energy_spread_frac,
        beam_divergence_rad=beam_divergence_rad,
    )
    overlap_factors: Float[Array, "N"]
    rod_factor: Float[Array, "N"]
    overlap_factors, rod_factor, k_out_rod = rod_domain_overlap(
        hkl_points=ewald.hkl_grid,
        recip_vectors=ewald.recip_vectors,
        k_in=k_in,
        k_magnitude=ewald.k_magnitude,
        rod_sigma=rod_sigma,
        shell_sigma=shell_sigma,
        layer_attenuation=layer_attenuation,
    )
    base_intensities: Float[Array, "N"] = rod_base_intensities(
        ewald=ewald,
        k_in=k_in,
        k_out_rod=k_out_rod,
        parameterization=parameterization,
    )
    modified_intensities: Float[Array, "N"] = base_intensities * rod_factor
    return overlap_factors, modified_intensities


@jaxtyped(typechecker=beartype)
def finite_domain_intensities_for_size_distribution(
    ewald: EwaldData,
    theta_deg: scalar_float,
    phi_deg: scalar_float,
    size_distribution: SizeDistribution,
    domain_aspect_ratio: Tuple[float, float, float] = (1.0, 1.0, 0.5),
    n_size_points: scalar_int = 7,
    energy_spread_frac: scalar_float = 1e-4,
    beam_divergence_rad: scalar_float = 1e-3,
    layer_attenuation: scalar_float = 0.01,
    parameterization: str = "lobato",
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    r"""Average finite-domain intensities over a domain-size distribution.

    :see: :class:`~.test_finite_domain.TestFiniteDomainIntensities`

    Parameters
    ----------
    ewald : EwaldData
        Pre-computed angle-independent Ewald data.
    theta_deg, phi_deg : scalar_float
        Beam incidence and azimuth angles in degrees.
    size_distribution : SizeDistribution
        Statistical distribution over coherent lateral domain sizes.
    domain_aspect_ratio : Tuple[float, float, float], optional
        Relative mapping from each scalar size sample to
        ``[Lx, Ly, Lz]`` domain extent. Default: ``(1.0, 1.0, 0.5)``.
    n_size_points : scalar_int, optional
        Quadrature point count for non-delta size distributions. Default: 7.
    energy_spread_frac : scalar_float, optional
        Fractional energy spread used in shell broadening. Default: 1e-4.
    beam_divergence_rad : scalar_float, optional
        Beam divergence in radians used in shell broadening. Default: 1e-3.
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation ε of the CTR truncation factor.
        Default: 0.01

    Returns
    -------
    averaged_overlap : Float[Array, "N"]
        Probability-weighted overlap factors for each reciprocal point.
    averaged_intensities : Float[Array, "N"]
        Probability-weighted finite-domain intensities.

    Notes
    -----
    Domain size is an incoherent ensemble axis: each finite coherent domain
    scatters with its own rod width, and the measured signal is the weighted
    sum of per-size intensities.
    """
    aspect_ratio: Float[Array, "3"] = jnp.asarray(
        domain_aspect_ratio, dtype=jnp.float64
    )
    if aspect_ratio.shape != (3,):
        raise ValueError("domain_aspect_ratio must have shape (3,)")
    checked_aspect_ratio: Float[Array, "3"] = eqx.error_if(
        aspect_ratio,
        jnp.any(~jnp.isfinite(aspect_ratio)),
        "domain_aspect_ratio must be finite",
    )
    checked_aspect_ratio = eqx.error_if(
        checked_aspect_ratio,
        jnp.any(checked_aspect_ratio <= 0.0),
        "domain_aspect_ratio must be positive",
    )
    size_axis: Distribution = size_to_distribution(
        size_distribution,
        n_points=n_size_points,
    )

    def _evaluate_size_sample(
        sample: Float[Array, "1"],
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
        domain_extent_ang: Float[Array, "3"] = sample[0] * checked_aspect_ratio
        return finite_domain_intensities(
            ewald=ewald,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            domain_extent_ang=domain_extent_ang,
            energy_spread_frac=energy_spread_frac,
            beam_divergence_rad=beam_divergence_rad,
            layer_attenuation=layer_attenuation,
            parameterization=parameterization,
        )

    overlap_bank: Float[Array, "S N"]
    intensity_bank: Float[Array, "S N"]
    overlap_bank, intensity_bank = jax.vmap(_evaluate_size_sample)(
        size_axis.samples
    )
    averaged_overlap: Float[Array, "N"] = jnp.einsum(
        "s,sn->n",
        size_axis.weights,
        overlap_bank,
    )
    averaged_intensities: Float[Array, "N"] = jnp.einsum(
        "s,sn->n",
        size_axis.weights,
        intensity_bank,
    )
    return averaged_overlap, averaged_intensities


__all__: list[str] = [
    "compute_domain_extent",
    "compute_shell_sigma",
    "extent_to_rod_sigma",
    "find_ctr_ewald_intersection",
    "finite_domain_intensities",
    "finite_domain_intensities_for_size_distribution",
    "rod_domain_overlap",
]
