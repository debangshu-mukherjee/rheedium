r"""Surface reciprocal lattice rod calculations for RHEED simulations.

Extended Summary
----------------
This module provides functions for calculating Crystal Truncation Rod (CTR)
intensities, which are continuous scattering features along surface-normal
reciprocal space directions. CTRs are essential for accurate RHEED pattern
simulation as they produce the characteristic streaks observed experimentally.

The CTR model implemented here is the semi-infinite truncation rod: the
complex amplitude along the (h, k) rod is

.. math::

    A_{hk}(l) = F_{cell}(q) \\cdot
    \\frac{1}{1 - e^{-2\\pi i l} e^{-\\epsilon}}
    \\cdot e^{-q_z^2 \\sigma^2 / 2}

with :math:`q(h, k, l) = h b_1 + k b_2 + l b_3` built from the full
reciprocal basis, :math:`\\epsilon \\geq 0` a per-layer attenuation
absorbing the finite penetration depth, and the last factor the Gaussian
roughness amplitude damping. Intensities are the squared modulus, so the
truncation factor becomes
:math:`1 / (1 - 2 e^{-\\epsilon} \\cos 2\\pi l + e^{-2\\epsilon})` and
the roughness factor becomes :math:`e^{-q_z^2 \\sigma^2}`.

Routine Listings
----------------
:func:`calculate_ctr_intensity`
    Calculate continuous intensity along crystal truncation rods with
    the semi-infinite truncation factor.
:func:`calculate_ctr_amplitude`
    Calculate complex amplitude along crystal truncation rods (for
    coherent mixing with kinematic scattering).
:func:`ctr_truncation_amplitude`
    Semi-infinite truncation-rod complex amplitude factor.
:func:`ctr_truncation_intensity`
    Semi-infinite truncation-rod intensity factor.
:func:`gaussian_rod_profile`
    Gaussian lateral width profile of rods due to finite correlation
    length.
:func:`lorentzian_rod_profile`
    Lorentzian lateral width profile of rods due to finite correlation
    length.
:func:`roughness_damping`
    Gaussian roughness amplitude damping factor for CTR amplitudes.
:func:`rod_profile_function`
    Lateral width profile of rods due to finite correlation length.
:func:`surface_structure_factor`
    Calculate structure factor for surface with q_z dependence.
:func:`integrated_rod_intensity`
    Acceptance-window-weighted mean CTR intensity.
:func:`integrated_ctr_amplitude`
    Acceptance-window-weighted mean CTR amplitude (for coherent mixing).

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
CTR calculations follow the kinematic approximation with proper surface
physics.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Union
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from rheedium.types import CrystalStructure, scalar_float
from rheedium.ucell import reciprocal_lattice_vectors

from .form_factors import atomic_scattering_factor


@jaxtyped(typechecker=beartype)
def ctr_truncation_amplitude(
    l_values: Union[scalar_float, Float[Array, "..."]],
    layer_attenuation: scalar_float = 0.01,
) -> Complex[Array, "..."]:
    r"""Semi-infinite truncation-rod complex amplitude factor.

    Computes the geometric-series sum over the semi-infinite stack of
    unit cells below the surface,
    :math:`1 / (1 - e^{-2\pi i l} e^{-\epsilon})`, which multiplies the
    single-cell structure factor to produce the crystal truncation rod.

    :see: :class:`~.test_surface_rods.TestCtrTruncationFactors`

    Parameters
    ----------
    l_values : Float[Array, "..."]
        Continuous Miller index l along the rod (in units of the third
        reciprocal basis vector b3). Can be scalar or array.
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon \geq 0` absorbing
        the finite penetration depth of the electron beam. Default: 0.01

    Returns
    -------
    amplitude_factor : Complex[Array, "..."]
        Complex truncation amplitude factor. Its squared modulus equals
        :func:`ctr_truncation_intensity` at the same arguments.

    Notes
    -----
    1. **Clamp attenuation** --
       Ensure :math:`\epsilon \geq 0` so the geometric series converges.
    2. **Layer phase** --
       Compute :math:`e^{-2\pi i l}` for the cell-to-cell phase below the
       surface.
    3. **Geometric sum** --
       Return :math:`1 / (1 - e^{-2\pi i l} e^{-\epsilon})`.

    See Also
    --------
    ctr_truncation_intensity : Squared modulus of this factor.
    calculate_ctr_amplitude : Full rod amplitude including this factor.
    """
    epsilon: Float[Array, ""] = jnp.maximum(
        jnp.asarray(layer_attenuation, dtype=jnp.float64), 0.0
    )
    layer_phase: Complex[Array, "..."] = jnp.exp(
        -2.0j * jnp.pi * jnp.asarray(l_values, dtype=jnp.float64)
    )
    amplitude_factor: Complex[Array, "..."] = 1.0 / (
        1.0 - layer_phase * jnp.exp(-epsilon)
    )
    return amplitude_factor


@jaxtyped(typechecker=beartype)
def ctr_truncation_intensity(
    l_values: Union[scalar_float, Float[Array, "..."]],
    layer_attenuation: scalar_float = 0.01,
) -> Float[Array, "..."]:
    r"""Semi-infinite truncation-rod intensity factor.

    Computes
    :math:`1 / (1 - 2 e^{-\epsilon} \cos 2\pi l + e^{-2\epsilon})`, the
    squared modulus of :func:`ctr_truncation_amplitude`. For small
    :math:`\epsilon` this equals :math:`1 / (4 \sin^2 \pi l)` away from
    Bragg points and caps at
    :math:`1 / (1 - e^{-\epsilon})^2 = 1/(4 e^{-\epsilon}
    \sinh^2(\epsilon/2)) \approx 1/\epsilon^2` at integer l.

    :see: :class:`~.test_surface_rods.TestCtrTruncationFactors`

    Parameters
    ----------
    l_values : Float[Array, "..."]
        Continuous Miller index l along the rod. Can be scalar or array.
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon \geq 0`.
        Default: 0.01

    Returns
    -------
    intensity_factor : Float[Array, "..."]
        Truncation-rod intensity factor, equal to
        ``abs(ctr_truncation_amplitude(...)) ** 2``.

    Notes
    -----
    1. **Clamp attenuation** --
       Ensure :math:`\epsilon \geq 0`.
    2. **Denominator** --
       Evaluate :math:`1 - 2 e^{-\epsilon} \cos 2\pi l + e^{-2\epsilon}`,
       the squared modulus of :math:`1 - e^{-2\pi i l} e^{-\epsilon}`.
    3. **Invert** --
       Return the reciprocal.

    See Also
    --------
    ctr_truncation_amplitude : Complex amplitude version of this factor.
    calculate_ctr_intensity : Full rod intensity including this factor.
    """
    epsilon: Float[Array, ""] = jnp.maximum(
        jnp.asarray(layer_attenuation, dtype=jnp.float64), 0.0
    )
    l_arr: Float[Array, "..."] = jnp.asarray(l_values, dtype=jnp.float64)
    attenuation: Float[Array, ""] = jnp.exp(-epsilon)
    denominator: Float[Array, "..."] = (
        1.0
        - 2.0 * attenuation * jnp.cos(2.0 * jnp.pi * l_arr)
        + attenuation**2
    )
    intensity_factor: Float[Array, "..."] = 1.0 / denominator
    return intensity_factor


def _rod_l_from_qz(
    hk: Int[Array, "2"],
    q_z: Float[Array, "..."],
    reciprocal_vectors: Float[Array, "3 3"],
) -> Float[Array, "..."]:
    r"""Convert Cartesian q_z on the (h, k) rod to continuous l.

    The rod point at height q_z satisfies
    :math:`(h b_1 + k b_2 + l b_3) \cdot \hat{z} = q_z`, so
    :math:`l = (q_z - (h b_1 + k b_2)_z) / (b_3)_z`.
    """
    g_parallel: Float[Array, "3"] = (
        jnp.float64(hk[0]) * reciprocal_vectors[0]
        + jnp.float64(hk[1]) * reciprocal_vectors[1]
    )
    b3_z: Float[Array, ""] = reciprocal_vectors[2, 2]
    l_values: Float[Array, "..."] = (q_z - g_parallel[2]) / b3_z
    return l_values


@jaxtyped(typechecker=beartype)
def calculate_ctr_intensity(
    hk_indices: Int[Array, "N 2"],
    l_values: Float[Array, "M"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    layer_attenuation: scalar_float = 0.01,
    temperature: scalar_float = 300.0,
    is_surface_atom: Bool[Array, "n_atoms"] | None = None,
    parameterization: str = "lobato",
) -> Float[Array, "N M"]:
    r"""Calculate continuous intensity along crystal truncation rods (CTRs).

    Computes the semi-infinite truncation-rod intensity along each (h, k)
    rod at continuous l values:
    :math:`I_{hk}(l) = |F_{cell}(q)|^2 /
    (1 - 2 e^{-\epsilon} \cos 2\pi l + e^{-2\epsilon})
    \times e^{-q_z^2 \sigma^2}` with
    :math:`q = h b_1 + k b_2 + l b_3` built from the full reciprocal
    basis.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`
    :see: :func:`~.test_physics_anchors.test_ctr_shape`

    Parameters
    ----------
    hk_indices : Int[Array, "N 2"]
        In-plane Miller indices (h,k) for each rod. Shape (N, 2) where
        N is the number of rods to calculate.
    l_values : Float[Array, "M"]
        Continuous Miller index l along the rod (units of b3) where
        intensity is calculated. Shape (M,) for M points along each rod.
    crystal : CrystalStructure
        Crystal structure containing atomic positions and cell parameters
    surface_roughness : scalar_float
        RMS surface roughness σ_h in Angstroms
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon \geq 0` in the
        truncation factor, absorbing the finite penetration depth.
        Default: 0.01
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors. Default: 300.0
    is_surface_atom : Bool[Array, "n_atoms"] | None, optional
        Per-atom boolean mask indicating which atoms are surface atoms.
        If None, no surface enhancement is applied (all atoms treated as
        bulk). This prevents double-application when used with kinematic
        calculations that already apply surface enhancement.
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    intensities : Float[Array, "N M"]
        CTR intensities for each (h,k) rod at each l value.
        Shape (N, M) where N is number of rods, M is number of l points.

    Notes
    -----
    1. **Extract atomic data** --
       Get positions and atomic numbers from crystal
       structure.
    2. **Build reciprocal lattice** --
       Compute the full reciprocal basis :math:`b_1, b_2, b_3` from cell
       parameters.
    3. **Rod q vectors** --
       For each (h, k, l), build :math:`q = h b_1 + k b_2 + l b_3`; the
       rod direction is :math:`b_3` (never a hard-coded Cartesian z).
    4. **Structure factor** --
       Evaluate the single-cell structure factor with atomic form
       factors and Debye-Waller damping.
    5. **Truncation factor** --
       Multiply :math:`|F|^2` by :func:`ctr_truncation_intensity`.
    6. **Roughness damping** --
       Multiply by the squared amplitude damping
       :math:`e^{-q_z^2 \sigma^2}` with :math:`q_z = q \cdot \hat{z}`.
    7. **Assemble output** --
       Return the intensity array for all rods and l values.

    See Also
    --------
    calculate_ctr_amplitude : Complex amplitude version for coherent mixing
    ctr_truncation_intensity : Truncation-rod intensity factor
    surface_structure_factor : Per-q structure factor calculation
    roughness_damping : Surface roughness amplitude attenuation
    """
    amplitudes: Complex[Array, "N M"] = calculate_ctr_amplitude(
        hk_indices=hk_indices,
        l_values=l_values,
        crystal=crystal,
        surface_roughness=surface_roughness,
        layer_attenuation=layer_attenuation,
        temperature=temperature,
        is_surface_atom=is_surface_atom,
        parameterization=parameterization,
    )
    intensities: Float[Array, "N M"] = jnp.abs(amplitudes) ** 2
    return intensities


@jaxtyped(typechecker=beartype)
def calculate_ctr_amplitude(
    hk_indices: Int[Array, "N 2"],
    l_values: Float[Array, "M"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    layer_attenuation: scalar_float = 0.01,
    temperature: scalar_float = 300.0,
    is_surface_atom: Bool[Array, "n_atoms"] | None = None,
    parameterization: str = "lobato",
) -> Complex[Array, "N M"]:
    r"""Calculate complex amplitude along crystal truncation rods (CTRs).

    Computes the semi-infinite truncation-rod complex amplitude
    :math:`A_{hk}(l) = F_{cell}(q) / (1 - e^{-2\pi i l} e^{-\epsilon})
    \times e^{-q_z^2 \sigma^2 / 2}` with
    :math:`q = h b_1 + k b_2 + l b_3`. This is used for coherent mixing
    with kinematic scattering amplitudes; its squared modulus is
    :func:`calculate_ctr_intensity`.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`

    Parameters
    ----------
    hk_indices : Int[Array, "N 2"]
        In-plane Miller indices (h,k) for each rod. Shape (N, 2) where
        N is the number of rods to calculate.
    l_values : Float[Array, "M"]
        Continuous Miller index l along the rod (units of b3) where
        amplitude is calculated. Shape (M,) for M points along each rod.
    crystal : CrystalStructure
        Crystal structure containing atomic positions and cell parameters
    surface_roughness : scalar_float
        RMS surface roughness σ_h in Angstroms
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon \geq 0` in the
        truncation factor. Default: 0.01
    temperature : scalar_float, optional
        Temperature in Kelvin for Debye-Waller factors. Default: 300.0
    is_surface_atom : Bool[Array, "n_atoms"] | None, optional
        Per-atom boolean mask indicating which atoms are surface atoms.
        If None, no surface enhancement is applied. Default: None
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    amplitudes : Complex[Array, "N M"]
        CTR complex amplitudes for each (h,k) rod at each l value.
        Shape (N, M) where N is number of rods, M is number of l points.

    Notes
    -----
    The amplitude is
    :math:`F_{cell}(q) \times T(l) \times R_a(q_z)` where
    :math:`T(l)` is :func:`ctr_truncation_amplitude` and
    :math:`R_a(q_z) = e^{-q_z^2 \sigma^2 / 2}` is the roughness
    amplitude factor, so
    :math:`|A|^2 = |F|^2 \times |T|^2 \times e^{-q_z^2 \sigma^2}`.

    1. **Extract atomic data** --
       Get positions and atomic numbers from crystal
       structure.
    2. **Build reciprocal lattice** --
       Compute the full reciprocal basis from cell parameters.
    3. **Per-rod amplitude** --
       For each (h, k, l), build :math:`q = h b_1 + k b_2 + l b_3` and
       compute the single-cell structure factor.
    4. **Apply truncation factor** --
       Multiply by :func:`ctr_truncation_amplitude` at the same l.
    5. **Apply roughness** --
       Multiply by :func:`roughness_damping` (the amplitude factor,
       applied exactly once).

    See Also
    --------
    calculate_ctr_intensity : Intensity version (amplitude squared)
    ctr_truncation_amplitude : Truncation-rod amplitude factor
    integrated_ctr_amplitude : Detector-integrated amplitude
    surface_structure_factor : Per-q structure factor
    """
    atomic_positions: Float[Array, "n_atoms 3"] = crystal.cart_positions[:, :3]
    atomic_numbers: Int[Array, "n_atoms"] = crystal.cart_positions[
        :, 3
    ].astype(jnp.int32)
    n_atoms: int = atomic_positions.shape[0]
    occupancies: Float[Array, "n_atoms"] = (
        jnp.ones(n_atoms, dtype=jnp.float64)
        if crystal.occupancies is None
        else jnp.asarray(crystal.occupancies, dtype=jnp.float64)
    )
    cell_lengths: Float[Array, "3"] = crystal.cell_lengths
    cell_angles: Float[Array, "3"] = crystal.cell_angles
    reciprocal_vectors: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        cell_lengths[0],
        cell_lengths[1],
        cell_lengths[2],
        cell_angles[0],
        cell_angles[1],
        cell_angles[2],
    )
    reciprocal_a: Float[Array, "3"] = reciprocal_vectors[0]
    reciprocal_b: Float[Array, "3"] = reciprocal_vectors[1]
    reciprocal_c: Float[Array, "3"] = reciprocal_vectors[2]

    # Default: no surface enhancement (prevents double-application)
    surface_mask: Bool[Array, "n_atoms"] = (
        jnp.zeros(n_atoms, dtype=jnp.bool_)
        if is_surface_atom is None
        else is_surface_atom
    )

    def calculate_single_rod_amplitude(
        hk: Int[Array, "2"],
    ) -> Complex[Array, "M"]:
        """Calculate amplitude for a single (h,k) rod at all l values."""
        h_val: Float[Array, ""] = jnp.float64(hk[0])
        k_val: Float[Array, ""] = jnp.float64(hk[1])
        q_parallel: Float[Array, "3"] = (
            h_val * reciprocal_a + k_val * reciprocal_b
        )

        def calculate_at_l(l_val: Float[Array, ""]) -> Complex[Array, ""]:
            """Calculate amplitude at a single l value."""
            q_vector: Float[Array, "3"] = q_parallel + l_val * reciprocal_c
            structure_factor: Complex[Array, ""] = surface_structure_factor(
                q_vector=q_vector,
                atomic_positions=atomic_positions,
                atomic_numbers=atomic_numbers,
                temperature=temperature,
                is_surface_atom=surface_mask,
                parameterization=parameterization,
                occupancies=occupancies,
            )
            truncation: Complex[Array, ""] = ctr_truncation_amplitude(
                l_values=l_val, layer_attenuation=layer_attenuation
            )
            damping: Float[Array, ""] = roughness_damping(
                q_z=q_vector[2], sigma_height=surface_roughness
            )
            amplitude: Complex[Array, ""] = (
                structure_factor * truncation * damping
            )
            return amplitude

        rod_amplitudes: Complex[Array, "M"] = jax.vmap(calculate_at_l)(
            l_values
        )
        return rod_amplitudes

    all_amplitudes: Complex[Array, "N M"] = jax.vmap(
        calculate_single_rod_amplitude
    )(hk_indices)
    return all_amplitudes


@jaxtyped(typechecker=beartype)
def integrated_ctr_amplitude(
    hk_index: Int[Array, "2"],
    q_z_range: Float[Array, "2"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    detector_acceptance_inv_ang: scalar_float,
    n_integration_points: int = 50,
    temperature: scalar_float = 300.0,
    layer_attenuation: scalar_float = 0.01,
    is_surface_atom: Bool[Array, "n_atoms"] | None = None,
    parameterization: str = "lobato",
) -> Complex[Array, ""]:
    r"""Acceptance-window-weighted mean CTR amplitude.

    Calculates an effective complex amplitude for a detector with finite
    q_z acceptance by coherently averaging CTR amplitudes with a Gaussian
    acceptance window: :math:`\sum_i w_i A_i / \sum_i w_i` with
    :math:`w_i = \exp(-\tfrac{1}{2}((q_i - q_0)/\sigma)^2)`. The same
    normalization is used by :func:`integrated_rod_intensity`, so
    :math:`|A_{int}|^2 \approx I_{int}` for a slowly varying rod.

    :see: :class:`~.test_surface_rods.TestIntegratedWindowConsistency`

    Parameters
    ----------
    hk_index : Int[Array, "2"]
        In-plane Miller indices (h, k) for the rod
    q_z_range : Float[Array, "2"]
        Range of q_z values (min, max) in 1/Å to sample over
    crystal : CrystalStructure
        Crystal structure for calculation
    surface_roughness : scalar_float
        RMS surface roughness in Angstroms
    detector_acceptance_inv_ang : scalar_float
        Gaussian σ of the q_z acceptance window in 1/Å (this parameter
        was never an angle).
    n_integration_points : int, optional
        Number of integration points. Default: 50
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon` in the
        truncation factor. Default: 0.01
    is_surface_atom : Bool[Array, "n_atoms"] | None, optional
        Per-atom boolean mask indicating which atoms are surface atoms.
        If None, no surface enhancement is applied. Default: None
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    integrated_amplitude : Complex[Array, ""]
        Window-weighted mean complex amplitude over detector acceptance

    Notes
    -----
    1. **Sample q_z** --
       Create linearly spaced :math:`q_z` values over
       the integration range and convert them to continuous l on the
       (h, k) rod.
    2. **Compute amplitudes** --
       Evaluate the CTR amplitude at all sampled points.
    3. **Gaussian window** --
       Weight with the acceptance window centred on the :math:`q_z`
       midpoint with σ = ``detector_acceptance_inv_ang``.
    4. **Weighted mean** --
       Return :math:`\sum w_i A_i / \sum w_i` (coherent average).
    """
    q_z_values: Float[Array, "n_points"] = jnp.linspace(
        q_z_range[0], q_z_range[1], n_integration_points
    )
    cell_lengths: Float[Array, "3"] = crystal.cell_lengths
    cell_angles: Float[Array, "3"] = crystal.cell_angles
    reciprocal_vectors: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        cell_lengths[0],
        cell_lengths[1],
        cell_lengths[2],
        cell_angles[0],
        cell_angles[1],
        cell_angles[2],
    )
    l_values: Float[Array, "n_points"] = _rod_l_from_qz(
        hk=hk_index, q_z=q_z_values, reciprocal_vectors=reciprocal_vectors
    )
    amplitudes: Complex[Array, "1 n_points"] = calculate_ctr_amplitude(
        hk_indices=hk_index[None, :],
        l_values=l_values,
        crystal=crystal,
        surface_roughness=surface_roughness,
        layer_attenuation=layer_attenuation,
        temperature=temperature,
        is_surface_atom=is_surface_atom,
        parameterization=parameterization,
    )
    rod_amplitudes: Complex[Array, "n_points"] = amplitudes[0]
    q_z_center: Float[Array, ""] = jnp.mean(q_z_values)
    q_z_width: Float[Array, ""] = jnp.asarray(
        detector_acceptance_inv_ang, dtype=jnp.float64
    )
    acceptance_window: Float[Array, "n_points"] = jnp.exp(
        -0.5 * jnp.square((q_z_values - q_z_center) / q_z_width)
    )
    window_sum: Float[Array, ""] = jnp.sum(acceptance_window)
    integrated_amplitude: Complex[Array, ""] = (
        jnp.sum(rod_amplitudes * acceptance_window) / window_sum
    )
    return integrated_amplitude


@jaxtyped(typechecker=beartype)
def roughness_damping(
    q_z: Float[Array, "..."],
    sigma_height: scalar_float,
) -> Float[Array, "..."]:
    r"""Gaussian roughness amplitude damping factor for CTR amplitudes.

    Calculates the **amplitude** damping factor
    :math:`R_a(q_z) = \exp(-q_z^2 \sigma_h^2 / 2)` due to surface
    roughness, which reduces the CTR amplitude especially at large q_z
    values. Assumes a Gaussian height distribution with RMS roughness
    σ_h. This is the single roughness-exponent definition in the
    package: amplitude paths multiply by this factor exactly once, and
    intensity paths multiply by its square
    :math:`e^{-q_z^2 \sigma_h^2}`.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`
    :see: :func:`~.test_physics_anchors.test_roughness_intensity_ratio`

    Parameters
    ----------
    q_z : Float[Array, "..."]
        Perpendicular momentum transfer in 1/Å. Can be scalar or array.
    sigma_height : scalar_float
        RMS surface roughness in Angstroms

    Returns
    -------
    damping : Float[Array, "..."]
        Amplitude damping factor exp(-q_z²σ_h²/2) between 0 and 1

    Notes
    -----
    1. **Clamp roughness** --
       Ensure :math:`\sigma_h \geq 0`.
    2. **Compute exponent** --
       :math:`W = \tfrac{1}{2} q_z^2 \sigma_h^2`.
    3. **Evaluate damping** --
       Return :math:`\exp(-W)` (amplitude convention; square it for
       intensities).
    4. **Handle zero roughness** --
       If :math:`\sigma_h < \epsilon`, return 1.0
       (no damping).

    See Also
    --------
    calculate_ctr_amplitude : Multiplies by this factor once
    calculate_ctr_intensity : Multiplies by the square of this factor
    debye_waller_factor : Similar damping for thermal vibrations
    """
    sigma: Float[Array, ""] = jnp.maximum(
        jnp.asarray(sigma_height, dtype=jnp.float64), 0.0
    )
    epsilon: Float[Array, ""] = jnp.asarray(1e-10, dtype=jnp.float64)
    half: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
    q_z_squared: Float[Array, "..."] = jnp.square(q_z)
    sigma_squared: Float[Array, ""] = jnp.square(sigma)
    exponent: Float[Array, "..."] = half * q_z_squared * sigma_squared
    damping: Float[Array, ". .."] = jnp.exp(-exponent)
    damping_final: Float[Array, "..."] = jnp.where(
        sigma < epsilon, jnp.ones_like(q_z), damping
    )
    return damping_final


@jaxtyped(typechecker=beartype)
def gaussian_rod_profile(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
) -> Float[Array, "..."]:
    r"""Gaussian lateral width profile of rods with finite correlation length.

    Calculates the Gaussian lateral intensity profile of CTRs perpendicular
    to the rod direction. The width in reciprocal space is inversely
    proportional to the real-space correlation length.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized Gaussian intensity profile perpendicular to rod

    Notes
    -----
    1. **Clamp correlation length** --
       Ensure :math:`\\xi > 0`.
    2. **Reciprocal width** --
       :math:`\\sigma_q = 1/\\xi`.
    3. **Normalize** --
       Compute :math:`q_{\\perp} / \\sigma_q`.
    4. **Gaussian profile** --
       :math:`\\exp(-\\tfrac{1}{2}(q_{\\perp}/\\sigma_q)^2)`.
    """
    xi: Float[Array, ""] = jnp.maximum(
        jnp.asarray(correlation_length, dtype=jnp.float64), 1e-10
    )
    sigma_q: Float[Array, ""] = 1.0 / xi
    half: Float[Array, ""] = jnp.asarray(0.5, dtype=jnp.float64)
    q_perp_normalized: Float[Array, "..."] = q_perpendicular / sigma_q
    profile: Float[Array, "..."] = jnp.exp(
        -half * jnp.square(q_perp_normalized)
    )
    return profile


@jaxtyped(typechecker=beartype)
def lorentzian_rod_profile(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
) -> Float[Array, "..."]:
    r"""Lorentzian lateral width profile for finite correlation length rods.

    Calculates the Lorentzian lateral intensity profile of CTRs perpendicular
    to the rod direction. This profile corresponds to exponentially decaying
    surface correlations.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized Lorentzian intensity profile perpendicular to rod

    Notes
    -----
    1. **Clamp correlation length** --
       Ensure :math:`\\xi > 0`.
    2. **Compute product** --
       :math:`q_{\\perp} \\times \\xi`.
    3. **Lorentzian profile** --
       :math:`1 / (1 + (q_{\\perp} \\xi)^2)`.
    """
    xi: Float[Array, ""] = jnp.maximum(
        jnp.asarray(correlation_length, dtype=jnp.float64), 1e-10
    )
    q_xi_product: Float[Array, "..."] = q_perpendicular * xi
    profile: Float[Array, "..."] = 1.0 / (1.0 + jnp.square(q_xi_product))
    return profile


@jaxtyped(typechecker=beartype)
def rod_profile_function(
    q_perpendicular: Float[Array, "..."],
    correlation_length: scalar_float,
    profile_type: str = "gaussian",
) -> Float[Array, "..."]:
    """Lateral width profile of rods due to finite correlation length.

    Calculates the lateral intensity profile of CTRs perpendicular to
    the rod direction using JAX-safe conditional logic. Finite correlation
    length of surface features causes rods to have finite width in reciprocal
    space.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`

    Parameters
    ----------
    q_perpendicular : Float[Array, "..."]
        Perpendicular distance from rod center in 1/Å
    correlation_length : scalar_float
        Surface correlation length in Angstroms
    profile_type : str, optional
        Type of profile: "gaussian" or "lorentzian".
        Default is "gaussian".

    Returns
    -------
    profile : Float[Array, "..."]
        Normalized intensity profile perpendicular to rod

    Notes
    -----
    1. **Select profile type** --
       Use JAX-safe conditional to choose between
       Gaussian and Lorentzian.
    2. **Evaluate profile** --
       Dispatch to :func:`gaussian_rod_profile` or
       :func:`lorentzian_rod_profile`.
    """
    is_lorentzian: Bool[Array, ""] = jnp.asarray(
        profile_type == "lorentzian", dtype=jnp.bool_
    )
    profile: Float[Array, "..."] = jax.lax.cond(
        is_lorentzian,
        lambda: lorentzian_rod_profile(q_perpendicular, correlation_length),
        lambda: gaussian_rod_profile(q_perpendicular, correlation_length),
    )
    return profile


@jaxtyped(typechecker=beartype)
def surface_structure_factor(
    q_vector: Float[Array, "3"],
    atomic_positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    temperature: scalar_float = 300.0,
    is_surface_atom: Bool[Array, "N"] | None = None,
    parameterization: str = "lobato",
    occupancies: Float[Array, "N"] | None = None,
) -> Complex[Array, ""]:
    r"""Calculate structure factor for surface with q_z dependence.

    Computes the complex structure factor F(q) for a surface, including
    atomic form factors, per-site occupancy weights, and Debye-Waller
    factors. Surface atoms can be treated with enhanced thermal
    vibrations via per-atom masking.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`

    Parameters
    ----------
    q_vector : Float[Array, "3"]
        3D scattering vector in 1/Å
    atomic_positions : Float[Array, "N 3"]
        Cartesian atomic positions in Angstroms
    atomic_numbers : Int[Array, "N"]
        Atomic numbers for each atom
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0
    is_surface_atom : Bool[Array, "N"] | None, optional
        Per-atom boolean mask indicating which atoms are surface atoms.
        Surface atoms receive enhanced Debye-Waller factors.
        If None, all atoms are treated as bulk (no enhancement).
        Default: None (prevents double-application with kinematic path)
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.
    occupancies : Float[Array, "N"] | None, optional
        Per-site occupancies in [0, 1] multiplying each atom's form
        factor. Default: None (all sites fully occupied).

    Returns
    -------
    structure_factor : Complex[Array, ""]
        Complex structure factor F(q)

    Notes
    -----
    1. **Phase factors** --
       Compute :math:`\\exp(i q \\cdot r_j)` for each atom.
    2. **Scattering factors** --
       Evaluate per-atom form factor with Debye-Waller
       damping and surface enhancement flag, weighted by the site
       occupancy.
    3. **Sum contributions** --
       Accumulate weighted complex contributions.
    4. **Return result** --
       Complex structure factor :math:`F(q)`.

    See Also
    --------
    atomic_scattering_factor : Per-atom form factor with thermal damping
    calculate_ctr_intensity : Uses structure factor for CTR calculation
    """
    n_atoms: int = atomic_positions.shape[0]

    # Default: no surface enhancement
    surface_mask: Bool[Array, "N"] = (
        jnp.zeros(n_atoms, dtype=jnp.bool_)
        if is_surface_atom is None
        else is_surface_atom
    )
    occupancy_weights: Float[Array, "N"] = (
        jnp.ones(n_atoms, dtype=jnp.float64)
        if occupancies is None
        else jnp.asarray(occupancies, dtype=jnp.float64)
    )

    phases: Float[Array, "N"] = jnp.einsum(
        "i,ji->j", q_vector, atomic_positions
    )
    phase_factors: Complex[Array, "N"] = jnp.exp(1j * phases)

    def get_atom_scattering(atom_idx: Int[Array, ""]) -> Float[Array, ""]:
        """Get scattering factor for single atom."""
        atomic_num: Int[Array, ""] = atomic_numbers[atom_idx]
        is_surf: Bool[Array, ""] = surface_mask[atom_idx]
        q_vec_expanded: Float[Array, "1 3"] = q_vector[jnp.newaxis, :]
        scattering: Float[Array, "1"] = atomic_scattering_factor(
            atomic_number=atomic_num,
            q_vector=q_vec_expanded,
            temperature=temperature,
            is_surface=is_surf,
            parameterization=parameterization,
        )
        return jnp.squeeze(scattering)

    atom_indices: Int[Array, "N"] = jnp.arange(n_atoms)
    scattering_factors: Float[Array, "N"] = occupancy_weights * jax.vmap(
        get_atom_scattering
    )(atom_indices)
    weighted_contributions: Complex[Array, "N"] = (
        scattering_factors * phase_factors
    )
    structure_factor: Complex[Array, ""] = jnp.sum(weighted_contributions)
    return structure_factor


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def integrated_rod_intensity(
    hk_index: Int[Array, "2"],
    q_z_range: Float[Array, "2"],
    crystal: CrystalStructure,
    surface_roughness: scalar_float,
    detector_acceptance_inv_ang: scalar_float,
    n_integration_points: int = 50,
    temperature: scalar_float = 300.0,
    layer_attenuation: scalar_float = 0.01,
    is_surface_atom: Bool[Array, "n_atoms"] | None = None,
    parameterization: str = "lobato",
) -> scalar_float:
    r"""Acceptance-window-weighted mean CTR intensity.

    Calculates the effective intensity seen by a detector with finite
    q_z acceptance as the Gaussian-window-weighted mean of the CTR
    intensity: :math:`\sum_i w_i I_i / \sum_i w_i` with
    :math:`w_i = \exp(-\tfrac{1}{2}((q_i - q_0)/\sigma)^2)`. The same
    normalization is used by :func:`integrated_ctr_amplitude`, so
    :math:`|A_{int}|^2 \approx I_{int}` for a slowly varying rod.

    :see: :class:`~.test_surface_rods.TestSurfaceRods`
    :see: :class:`~.test_surface_rods.TestIntegratedWindowConsistency`

    Parameters
    ----------
    hk_index : Int[Array, "2"]
        In-plane Miller indices (h, k) for the rod
    q_z_range : Float[Array, "2"]
        Range of q_z values (min, max) in 1/Å to sample over
    crystal : CrystalStructure
        Crystal structure for calculation
    surface_roughness : scalar_float
        RMS surface roughness in Angstroms
    detector_acceptance_inv_ang : scalar_float
        Gaussian σ of the q_z acceptance window in 1/Å (this parameter
        was never an angle).
    n_integration_points : int, optional
        Number of integration points. Default: 50
    temperature : scalar_float, optional
        Temperature in Kelvin. Default: 300.0
    layer_attenuation : scalar_float, optional
        Per-layer amplitude attenuation :math:`\epsilon` in the
        truncation factor. Default: 0.01
    is_surface_atom : Bool[Array, "n_atoms"] | None, optional
        Per-atom boolean mask indicating which atoms are surface atoms.
        If None, no surface enhancement is applied. Default: None
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    integrated_intensity : scalar_float
        Window-weighted mean intensity over the detector acceptance

    Notes
    -----
    1. **Sample q_z** --
       Create linearly spaced :math:`q_z` values over the range and
       convert them to continuous l on the (h, k) rod.
    2. **CTR intensity** --
       Evaluate the truncation-rod intensity at all sampled points.
    3. **Acceptance window** --
       Weight with the Gaussian acceptance window with
       σ = ``detector_acceptance_inv_ang``.
    4. **Weighted mean** --
       Return :math:`\sum w_i I_i / \sum w_i` (consistent with the
       amplitude version).
    """
    q_z_values: Float[Array, "n_points"] = jnp.linspace(
        q_z_range[0], q_z_range[1], n_integration_points
    )
    cell_lengths: Float[Array, "3"] = crystal.cell_lengths
    cell_angles: Float[Array, "3"] = crystal.cell_angles
    reciprocal_vectors: Float[Array, "3 3"] = reciprocal_lattice_vectors(
        cell_lengths[0],
        cell_lengths[1],
        cell_lengths[2],
        cell_angles[0],
        cell_angles[1],
        cell_angles[2],
    )
    l_values: Float[Array, "n_points"] = _rod_l_from_qz(
        hk=hk_index, q_z=q_z_values, reciprocal_vectors=reciprocal_vectors
    )
    intensities: Float[Array, "1 n_points"] = calculate_ctr_intensity(
        hk_indices=hk_index[None, :],
        l_values=l_values,
        crystal=crystal,
        surface_roughness=surface_roughness,
        layer_attenuation=layer_attenuation,
        temperature=temperature,
        is_surface_atom=is_surface_atom,
        parameterization=parameterization,
    )
    rod_intensities: Float[Array, "n_points"] = intensities[0]
    q_z_center: Float[Array, ""] = jnp.mean(q_z_values)
    q_z_width: Float[Array, ""] = jnp.asarray(
        detector_acceptance_inv_ang, dtype=jnp.float64
    )
    acceptance_window: Float[Array, "n_points"] = jnp.exp(
        -0.5 * jnp.square((q_z_values - q_z_center) / q_z_width)
    )
    window_sum: Float[Array, ""] = jnp.sum(acceptance_window)
    integrated_intensity: Float[Array, ""] = (
        jnp.sum(rod_intensities * acceptance_window) / window_sum
    )
    return integrated_intensity


__all__: list[str] = [
    "calculate_ctr_amplitude",
    "calculate_ctr_intensity",
    "ctr_truncation_amplitude",
    "ctr_truncation_intensity",
    "gaussian_rod_profile",
    "integrated_ctr_amplitude",
    "integrated_rod_intensity",
    "lorentzian_rod_profile",
    "rod_profile_function",
    "roughness_damping",
    "surface_structure_factor",
]
