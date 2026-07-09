r"""Beam averaging and instrument broadening for RHEED simulations.

Extended Summary
----------------
This module provides functions for modeling finite beam properties and
detector response in RHEED simulations. A real electron beam has finite
angular divergence (:math:`\sigma_\theta \approx 0.1\text{--}1` mrad),
energy spread (:math:`\sigma_E \approx 0.1\text{--}1` eV), finite
transverse and longitudinal coherence lengths, and the detector
introduces a point spread function (PSF). Without these effects,
simulated patterns are 5--20x sharper than experiment.

All functions are end-to-end differentiable via ``jax.grad``.

Routine Listings
----------------
:func:`angular_divergence_average`
    Average pattern over Gaussian angular divergence distribution.
:func:`apply_distribution`
    Apply a weighted distribution to a coherent amplitude closure.
:func:`apply_distribution_intensity`
    Apply an incoherent weighted distribution to an intensity closure.
:func:`apply_distributions`
    Apply multiple distribution axes with nested coherent/incoherent reduction.
:func:`decompose_beam_modes`
    Convert GSM beam parameters to a generic incoherent Distribution.
:func:`decompose_beam_modes_static`
    Eager tolerance-pruned beam-mode decomposition.
:func:`energy_spread_average`
    Average pattern over Gaussian energy spread distribution.
:func:`detector_psf_convolve`
    Convolve detector image with Gaussian point spread function.
:func:`instrument_broadened_pattern`
    Full instrument-averaged RHEED pattern combining all effects.

Notes
-----
Angular and energy averaging use Gauss-Hermite quadrature, which
integrates functions of the form :math:`f(x) \exp(-x^2)` exactly for
polynomials up to degree :math:`2n - 1`. For smooth RHEED intensity
profiles, 5--9 quadrature points give integration errors below
:math:`10^{-10}`.

The detector PSF convolution uses FFT for :math:`O(HW \log HW)` complexity and
is fully differentiable through ``jnp.fft``. Partial coherence is represented
through explicit beam-mode distributions rather than a separate multiplicative
coherence envelope.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from rheedium.tools import (
    gauss_hermite_nodes_weights as _gauss_hermite_nodes_weights,
)
from rheedium.types import (
    BeamModeDistribution,
    Distribution,
    ReductionMode,
    create_distribution,
    scalar_float,
)


@jaxtyped(typechecker=beartype)
def apply_distribution(
    distribution: Distribution,
    bound_amplitude_fn: Callable[[Float[Array, "D"]], Complex[Array, "H W"]],
) -> Float[Array, "H W"]:
    r"""Apply one distribution axis to a coherent amplitude function.

    :see: :class:`~.test_beam_averaging.TestDistributionApply`

    Parameters
    ----------
    distribution : Distribution
        Weighted latent samples and a static coherent/incoherent reduction
        mode.
    bound_amplitude_fn : Callable[[Float[Array, "D"]], Complex[Array, "H W"]]
        Closure mapping one sample vector to a dense coherent detector
        amplitude field.

    Returns
    -------
    intensity : Float[Array, "H W"]
        Reduced detector intensity.

    Notes
    -----
    1. Evaluate the bound coherent kernel for every sample with ``vmap``.
    2. For coherent axes, sum weighted amplitudes then take ``|.|^2``.
    3. For incoherent axes, take ``|.|^2`` per sample then weight and sum.

    See Also
    --------
    apply_distributions : Compose multiple distribution axes.
    """
    amplitudes: Complex[Array, "N H W"] = jax.vmap(bound_amplitude_fn)(
        distribution.samples
    )
    if distribution.reduction is ReductionMode.COHERENT:
        coherent_amplitude: Complex[Array, "H W"] = jnp.einsum(
            "n,nhw->hw",
            distribution.weights,
            amplitudes,
        )
        return jnp.abs(coherent_amplitude) ** 2
    return jnp.einsum(
        "n,nhw->hw",
        distribution.weights,
        jnp.abs(amplitudes) ** 2,
    )


@jaxtyped(typechecker=beartype)
def apply_distribution_intensity(
    distribution: Distribution,
    bound_intensity_fn: Callable[[Float[Array, "D"]], Float[Array, "H W"]],
) -> Float[Array, "H W"]:
    r"""Apply one incoherent distribution axis to an intensity function.

    Parameters
    ----------
    distribution : Distribution
        Weighted latent samples with ``ReductionMode.INCOHERENT``.
    bound_intensity_fn : Callable[[Float[Array, "D"]], Float[Array, "H W"]]
        Closure mapping one sample vector directly to detector intensity.

    Returns
    -------
    intensity : Float[Array, "H W"]
        Weighted incoherent detector intensity.

    Raises
    ------
    ValueError
        If ``distribution`` is configured for coherent amplitude reduction.

    Notes
    -----
    This is the intensity-space companion to :func:`apply_distribution` for
    callers that already produce intensities. It avoids the numerically
    fragile ``sqrt(I)`` round-trip at exactly zero-intensity pixels.
    """
    if distribution.reduction is not ReductionMode.INCOHERENT:
        raise ValueError(
            "apply_distribution_intensity only supports incoherent axes"
        )
    intensities: Float[Array, "N H W"] = jax.vmap(bound_intensity_fn)(
        distribution.samples
    )
    return jnp.einsum("n,nhw->hw", distribution.weights, intensities)


def _index_product(axis_sizes: Sequence[int]) -> Int[Array, "P K"]:
    """Return row-wise Cartesian products of axis indices."""
    if len(axis_sizes) == 0:
        return jnp.zeros((1, 0), dtype=jnp.int32)
    index_axes: list[Int[Array, "M"]] = [
        jnp.arange(axis_size, dtype=jnp.int32) for axis_size in axis_sizes
    ]
    grids = jnp.meshgrid(*index_axes, indexing="ij")
    return jnp.stack([grid.ravel() for grid in grids], axis=-1)


@jaxtyped(typechecker=beartype)
def apply_distributions(
    distributions: Sequence[Distribution],
    bound_amplitude_fn: Callable[[Float[Array, "D"]], Complex[Array, "H W"]],
) -> Float[Array, "H W"]:
    r"""Apply composed distribution axes to a coherent amplitude closure.

    :see: :class:`~.test_beam_averaging.TestDistributionApply`

    Parameters
    ----------
    distributions : Sequence[Distribution]
        Ordered distribution axes. Each sample passed to ``bound_amplitude_fn``
        is the concatenation of one sample from each axis in this order.
    bound_amplitude_fn : Callable[[Float[Array, "D"]], Complex[Array, "H W"]]
        Closure mapping one concatenated latent sample to a dense coherent
        detector amplitude field.

    Returns
    -------
    intensity : Float[Array, "H W"]
        Nested coherent/incoherent reduced detector intensity.

    Notes
    -----
    1. Split axes by static reduction mode.
    2. Iterate over the incoherent product outside the modulus.
    3. For each incoherent sample, sum the coherent product in amplitude.
    4. Weight and sum the resulting intensities.

    See Also
    --------
    apply_distribution : Single-axis distribution reduction.
    """
    if len(distributions) == 0:
        raise ValueError("distributions must contain at least one axis")

    coherent_positions: tuple[int, ...] = tuple(
        idx
        for idx, distribution in enumerate(distributions)
        if distribution.reduction is ReductionMode.COHERENT
    )
    incoherent_positions: tuple[int, ...] = tuple(
        idx
        for idx, distribution in enumerate(distributions)
        if distribution.reduction is ReductionMode.INCOHERENT
    )
    coherent_indices: Int[Array, "P C"] = _index_product(
        [distributions[idx].samples.shape[0] for idx in coherent_positions]
    )
    incoherent_indices: Int[Array, "Q I"] = _index_product(
        [distributions[idx].samples.shape[0] for idx in incoherent_positions]
    )

    def _assemble_sample(
        coherent_row: Int[Array, "C"],
        incoherent_row: Int[Array, "I"],
    ) -> Float[Array, "D"]:
        sample_parts: list[Float[Array, "D_i"]] = []
        coherent_cursor: int = 0
        incoherent_cursor: int = 0
        for idx, distribution in enumerate(distributions):
            if idx in coherent_positions:
                sample_idx = coherent_row[coherent_cursor]
                coherent_cursor += 1
            else:
                sample_idx = incoherent_row[incoherent_cursor]
                incoherent_cursor += 1
            sample_parts.append(distribution.samples[sample_idx])
        return jnp.concatenate(sample_parts)

    def _coherent_weight(coherent_row: Int[Array, "C"]) -> Float[Array, ""]:
        weight: Float[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
        for axis_pos, distribution_idx in enumerate(coherent_positions):
            axis_weight: Float[Array, ""] = distributions[
                distribution_idx
            ].weights[coherent_row[axis_pos]]
            weight = weight * axis_weight
        return weight

    def _incoherent_weight(
        incoherent_row: Int[Array, "I"],
    ) -> Float[Array, ""]:
        weight: Float[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
        for axis_pos, distribution_idx in enumerate(incoherent_positions):
            weight = (
                weight
                * distributions[distribution_idx].weights[
                    incoherent_row[axis_pos]
                ]
            )
        return weight

    coherent_weights: Float[Array, "P"] = jax.vmap(_coherent_weight)(
        coherent_indices
    )
    incoherent_weights: Float[Array, "Q"] = jax.vmap(_incoherent_weight)(
        incoherent_indices
    )

    def _intensity_for_incoherent_row(
        incoherent_row: Int[Array, "I"],
    ) -> Float[Array, "H W"]:
        def _amplitude_for_coherent_row(
            coherent_row: Int[Array, "C"],
        ) -> Complex[Array, "H W"]:
            sample: Float[Array, "D"] = _assemble_sample(
                coherent_row,
                incoherent_row,
            )
            return bound_amplitude_fn(sample)

        amplitudes: Complex[Array, "P H W"] = jax.vmap(
            _amplitude_for_coherent_row
        )(coherent_indices)
        coherent_amplitude: Complex[Array, "H W"] = jnp.einsum(
            "p,phw->hw",
            coherent_weights,
            amplitudes,
        )
        return jnp.abs(coherent_amplitude) ** 2

    intensities: Float[Array, "Q H W"] = jax.vmap(
        _intensity_for_incoherent_row
    )(incoherent_indices)
    return jnp.einsum("q,qhw->hw", incoherent_weights, intensities)


@jaxtyped(typechecker=beartype)
def _gsm_axis_modes(
    beta: Float[Array, ""],
    divergence_rad: Float[Array, ""],
    n_modes: int,
) -> tuple[Float[Array, "M"], Float[Array, "M"]]:
    """Return one transverse GSM axis with variance-matched offsets."""
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    indices: Float[Array, "M"] = jnp.arange(n_modes, dtype=jnp.float64)
    raw_weights: Float[Array, "M"] = (1.0 - beta) * beta**indices
    weights: Float[Array, "M"] = raw_weights / jnp.maximum(
        jnp.sum(raw_weights),
        1e-12,
    )
    mean_index: Float[Array, ""] = jnp.sum(weights * indices)
    centered_indices: Float[Array, "M"] = indices - mean_index
    index_variance: Float[Array, ""] = jnp.sum(weights * centered_indices**2)
    scale: Float[Array, ""] = jnp.where(
        index_variance > 0.0,
        divergence_rad
        / jnp.sqrt(jnp.where(index_variance > 0.0, index_variance, 1.0)),
        0.0,
    )
    offsets: Float[Array, "M"] = centered_indices * scale
    return offsets, weights


@jaxtyped(typechecker=beartype)
def _energy_mode_axis(
    energy_spread_ev: Float[Array, ""],
    n_energy_points: int,
) -> tuple[Float[Array, "E"], Float[Array, "E"]]:
    """Return longitudinal incoherent energy offsets and weights."""
    if n_energy_points <= 0:
        raise ValueError("n_energy_points must be positive")
    if n_energy_points == 1:
        offsets: Float[Array, "1"] = jnp.zeros((1,), dtype=jnp.float64)
        weights: Float[Array, "1"] = jnp.ones((1,), dtype=jnp.float64)
        return offsets, weights
    nodes: Float[Array, "E"]
    quad_weights: Float[Array, "E"]
    nodes, quad_weights = _gauss_hermite_nodes_weights(n_energy_points)
    sqrt2: Float[Array, ""] = jnp.sqrt(jnp.array(2.0, dtype=jnp.float64))
    sqrt_pi: Float[Array, ""] = jnp.sqrt(jnp.array(jnp.pi, dtype=jnp.float64))
    offsets = sqrt2 * energy_spread_ev * nodes
    weights = quad_weights / sqrt_pi
    return offsets, weights / jnp.sum(weights)


@jaxtyped(typechecker=beartype)
def decompose_beam_modes(
    beam_modes: BeamModeDistribution,
    n_modes_per_axis: int = 3,
    n_modes_out_of_plane: int | None = None,
    n_energy_points: int = 1,
) -> Distribution:
    """Convert GSM beam modes to a generic incoherent Distribution.

    :see: :class:`~.test_beam_averaging.TestBeamModeDecomposition`

    Parameters
    ----------
    beam_modes : BeamModeDistribution
        Physical Gaussian Schell-model source parameters.
    n_modes_per_axis : int, optional
        Fixed in-plane transverse mode count. Also used out-of-plane when
        ``n_modes_out_of_plane`` is not supplied. Default: 3.
    n_modes_out_of_plane : int | None, optional
        Optional separate out-of-plane mode count. Default: None.
    n_energy_points : int, optional
        Longitudinal Gauss-Hermite energy samples. Default: 1.

    Returns
    -------
    distribution : Distribution
        Incoherent distribution with samples
        ``[delta_theta_rad, delta_phi_rad, delta_energy_ev]``.
    """
    theta_offsets: Float[Array, "T"]
    theta_weights: Float[Array, "T"]
    theta_offsets, theta_weights = _gsm_axis_modes(
        beam_modes.beta_in_plane,
        beam_modes.divergence_in_plane_rad,
        n_modes_per_axis,
    )
    phi_offsets: Float[Array, "P"]
    phi_weights: Float[Array, "P"]
    phi_mode_count: int = (
        n_modes_per_axis
        if n_modes_out_of_plane is None
        else n_modes_out_of_plane
    )
    phi_offsets, phi_weights = _gsm_axis_modes(
        beam_modes.beta_out_of_plane,
        beam_modes.divergence_out_of_plane_rad,
        phi_mode_count,
    )
    energy_offsets: Float[Array, "E"]
    energy_weights: Float[Array, "E"]
    energy_offsets, energy_weights = _energy_mode_axis(
        beam_modes.energy_spread_ev,
        n_energy_points,
    )
    theta_grid: Float[Array, "T P E"]
    phi_grid: Float[Array, "T P E"]
    energy_grid: Float[Array, "T P E"]
    theta_grid, phi_grid, energy_grid = jnp.meshgrid(
        theta_offsets,
        phi_offsets,
        energy_offsets,
        indexing="ij",
    )
    weight_grid: Float[Array, "T P E"] = (
        theta_weights[:, None, None]
        * phi_weights[None, :, None]
        * energy_weights[None, None, :]
    )
    samples: Float[Array, "N 3"] = jnp.stack(
        [
            theta_grid.ravel(),
            phi_grid.ravel(),
            energy_grid.ravel(),
        ],
        axis=-1,
    )
    axis_id: str = (
        beam_modes.distribution_id
        if beam_modes.distribution_id is not None
        else "beam_modes"
    )
    return create_distribution(
        samples=samples,
        weights=weight_grid.ravel(),
        reduction=ReductionMode.INCOHERENT,
        axis_id=axis_id,
    )


def _static_mode_count(
    beta: Float[Array, ""], cap: int, weight_tol: float
) -> int:
    """Choose an eager GSM mode count from cumulative geometric mass."""
    if cap <= 0:
        raise ValueError("n_modes_per_axis must be positive")
    beta_value: float = float(beta)
    if beta_value <= 0.0:
        return 1
    cumulative_weight: float = 0.0
    for mode_idx in range(cap):
        cumulative_weight += (1.0 - beta_value) * (beta_value**mode_idx)
        if cumulative_weight >= 1.0 - weight_tol:
            return mode_idx + 1
    return cap


@jaxtyped(typechecker=beartype)
def decompose_beam_modes_static(
    beam_modes: BeamModeDistribution,
    n_modes_per_axis: int = 16,
    n_energy_points: int = 1,
    weight_tol: float = 1e-6,
) -> Distribution:
    """Eager tolerance-pruned GSM beam-mode decomposition.

    :see: :class:`~.test_beam_averaging.TestBeamModeDecomposition`

    Parameters
    ----------
    beam_modes : BeamModeDistribution
        Physical Gaussian Schell-model source parameters.
    n_modes_per_axis : int, optional
        Maximum transverse mode count per axis. Default: 16.
    n_energy_points : int, optional
        Longitudinal energy quadrature point count. Default: 1.
    weight_tol : float, optional
        Tail probability tolerance for transverse mode truncation.

    Returns
    -------
    distribution : Distribution
        Incoherent distribution with negligible transverse tails pruned.
    """
    theta_modes: int = _static_mode_count(
        beam_modes.beta_in_plane,
        n_modes_per_axis,
        weight_tol,
    )
    phi_modes: int = _static_mode_count(
        beam_modes.beta_out_of_plane,
        n_modes_per_axis,
        weight_tol,
    )
    energy_points: int = (
        1 if float(beam_modes.energy_spread_ev) <= 0.0 else n_energy_points
    )
    return decompose_beam_modes(
        beam_modes,
        n_modes_per_axis=theta_modes,
        n_modes_out_of_plane=phi_modes,
        n_energy_points=energy_points,
    )


@jaxtyped(typechecker=beartype)
def angular_divergence_average(
    simulate_fn: Callable[[scalar_float, scalar_float], Float[Array, "H W"]],
    nominal_polar_angle_rad: scalar_float,
    nominal_azimuth_angle_rad: scalar_float,
    angular_divergence_mrad: scalar_float,
    n_quadrature_points: int = 7,
) -> Float[Array, "H W"]:
    r"""Average RHEED intensity over Gaussian angular divergence.

    Extended Summary
    ----------------
    The incident polar angle :math:`\alpha` is distributed as
    :math:`p(\alpha) = \mathcal{N}(\alpha_0, \sigma_\theta^2)`. The
    pattern is averaged incoherently (intensity sum, not amplitude sum)
    because different incident angles correspond to uncorrelated
    wavepackets. Uses Gauss-Hermite quadrature for accurate integration
    with minimal function evaluations.

    :see: :class:`~.test_beam_averaging.TestAngularDivergenceAverage`

    Parameters
    ----------
    simulate_fn : Callable[..., Float[Array, "H W"]]
        Function mapping ``(polar_angle_rad, azimuth_angle_rad)`` to a
        2-D intensity pattern. Must be vmappable.
    nominal_polar_angle_rad : scalar_float
        Mean incidence angle :math:`\alpha_0` in radians.
    nominal_azimuth_angle_rad : scalar_float
        Mean azimuth :math:`\varphi_0` in radians.
    angular_divergence_mrad : scalar_float
        1-sigma angular divergence :math:`\sigma_\theta` in
        milliradians.
    n_quadrature_points : int, optional
        Number of Gauss-Hermite quadrature nodes. Default: 7

    Returns
    -------
    averaged_pattern : Float[Array, "H W"]
        Intensity-weighted average pattern.

    Notes
    -----
    1. **Compute quadrature** --
       Gauss-Hermite nodes :math:`x_i` and weights :math:`w_i`.
    2. **Map to angle samples** --
       :math:`\alpha_i = \alpha_0 + \sqrt{2}\,\sigma_\theta\,x_i`.
    3. **Evaluate patterns** --
       ``vmap`` ``simulate_fn`` over :math:`(\alpha_i, \varphi_0)` to
       obtain patterns of shape ``(N, H, W)``.
    4. **Weighted sum** --
       :math:`\bar{I} = \sum_i w_i I(\alpha_i) / \sqrt{\pi}`.
    5. **Clip** --
       Ensure non-negative intensities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> def dummy_sim(polar, azimuth):
    ...     return jnp.ones((64, 64)) * polar
    >>> avg = rh.simul.angular_divergence_average(
    ...     dummy_sim,
    ...     jnp.float64(0.035),
    ...     jnp.float64(0.0),
    ...     jnp.float64(0.5),
    ...     n_quadrature_points=5,
    ... )
    >>> avg.shape
    (64, 64)
    """
    divergence_rad: scalar_float = angular_divergence_mrad * 1e-3
    nodes: Float[Array, " N"]
    weights: Float[Array, " N"]
    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature_points)
    angle_samples: Float[Array, " N"] = (
        nominal_polar_angle_rad + jnp.sqrt(2.0) * divergence_rad * nodes
    )
    azimuth_samples: Float[Array, " N"] = jnp.full_like(
        angle_samples, nominal_azimuth_angle_rad
    )
    patterns: Float[Array, "N H W"] = jax.vmap(simulate_fn)(
        angle_samples, azimuth_samples
    )
    weighted_sum: Float[Array, "H W"] = jnp.einsum(
        "i,ijk->jk", weights, patterns
    )
    averaged_pattern: Float[Array, "H W"] = weighted_sum / jnp.sqrt(jnp.pi)
    return jnp.maximum(averaged_pattern, 0.0)


@jaxtyped(typechecker=beartype)
def energy_spread_average(
    simulate_fn: Callable[[scalar_float], Float[Array, "H W"]],
    nominal_energy_kev: scalar_float,
    energy_spread_ev: scalar_float,
    n_quadrature_points: int = 5,
) -> Float[Array, "H W"]:
    r"""Average RHEED intensity over Gaussian energy spread.

    Extended Summary
    ----------------
    The beam energy :math:`E` is distributed as
    :math:`p(E) = \mathcal{N}(E_0, \sigma_E^2)`. Averaging is
    incoherent because electrons with different energies do not
    interfere. Uses Gauss-Hermite quadrature with the same structure
    as angular averaging.

    :see: :class:`~.test_beam_averaging.TestEnergySpreadAverage`

    Parameters
    ----------
    simulate_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Function mapping ``energy_kev`` to a 2-D intensity pattern.
        Must be vmappable.
    nominal_energy_kev : scalar_float
        Mean beam energy :math:`E_0` in keV.
    energy_spread_ev : scalar_float
        1-sigma energy spread :math:`\sigma_E` in eV.
    n_quadrature_points : int, optional
        Number of Gauss-Hermite quadrature nodes. Default: 5

    Returns
    -------
    averaged_pattern : Float[Array, "H W"]
        Intensity-weighted average pattern.

    Notes
    -----
    1. **Compute quadrature** --
       Gauss-Hermite nodes :math:`x_i` and weights :math:`w_i`.
    2. **Map to energy samples** --
       Convert spread to keV:
       :math:`\sigma_{keV} = \sigma_E / 1000`. Then
       :math:`E_i = E_0 + \sqrt{2}\,\sigma_{keV}\,x_i`.
    3. **Evaluate patterns** --
       ``vmap`` ``simulate_fn`` over :math:`E_i`.
    4. **Weighted sum** --
       :math:`\bar{I} = \sum_i w_i I(E_i) / \sqrt{\pi}`.
    5. **Clip** --
       Ensure non-negative intensities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> def dummy_sim(energy_kev):
    ...     return jnp.ones((64, 64)) * energy_kev
    >>> avg = rh.simul.energy_spread_average(
    ...     dummy_sim,
    ...     jnp.float64(20.0),
    ...     jnp.float64(0.5),
    ... )
    >>> avg.shape
    (64, 64)
    """
    spread_kev: scalar_float = energy_spread_ev * 1e-3
    nodes: Float[Array, " N"]
    weights: Float[Array, " N"]
    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature_points)
    energy_samples: Float[Array, " N"] = (
        nominal_energy_kev + jnp.sqrt(2.0) * spread_kev * nodes
    )
    patterns: Float[Array, "N H W"] = jax.vmap(simulate_fn)(energy_samples)
    weighted_sum: Float[Array, "H W"] = jnp.einsum(
        "i,ijk->jk", weights, patterns
    )
    averaged_pattern: Float[Array, "H W"] = weighted_sum / jnp.sqrt(jnp.pi)
    return jnp.maximum(averaged_pattern, 0.0)


@jaxtyped(typechecker=beartype)
def detector_psf_convolve(
    detector_image: Float[Array, "H W"],
    psf_sigma_pixels: scalar_float,
) -> Float[Array, "H W"]:
    r"""Convolve detector image with Gaussian point spread function.

    Extended Summary
    ----------------
    Models phosphor screen grain size, camera lens aberrations, and CCD
    pixel diffusion as a 2-D Gaussian PSF. Convolution is computed in
    Fourier space for :math:`O(HW \log HW)` complexity:

    .. math::

        I_{\text{blurred}} = \mathcal{F}^{-1}\!\bigl[
            \mathcal{F}[I] \cdot \mathcal{F}[\text{PSF}]
        \bigr]

    The PSF is applied as a multiplicative Gaussian in Fourier space
    (the Fourier transform of a Gaussian is a Gaussian), avoiding the
    need to explicitly build and FFT the kernel.

    :see: :class:`~.test_beam_averaging.TestDetectorPsfConvolve`

    Parameters
    ----------
    detector_image : Float[Array, "H W"]
        Raw simulated intensity map before instrument broadening.
    psf_sigma_pixels : scalar_float
        1-sigma width of the Gaussian PSF in pixel units. Typical:
        0.5--2.0 pixels. Use 0.0 to skip convolution.

    Returns
    -------
    broadened_image : Float[Array, "H W"]
        Convolved intensity map with same shape as input.

    Notes
    -----
    1. **Build frequency grids** --
       Compute normalized frequency coordinates
       :math:`f_y, f_x` using ``jnp.fft.fftfreq``.
    2. **Gaussian in Fourier space** --
       :math:`\hat{G}(f) = \exp(-2\pi^2 \sigma^2 (f_x^2 + f_y^2))`.
    3. **Apply filter** --
       Multiply FFT of image by Gaussian filter.
    4. **Inverse FFT** --
       Take real part and clip to non-negative.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> img = jnp.ones((64, 64))
    >>> blurred = rh.simul.detector_psf_convolve(
    ...     img,
    ...     jnp.float64(1.5),
    ... )
    >>> blurred.shape
    (64, 64)
    """
    height: int = detector_image.shape[0]
    width: int = detector_image.shape[1]
    freq_y: Float[Array, " H"] = jnp.fft.fftfreq(height)
    freq_x: Float[Array, " W"] = jnp.fft.fftfreq(width)
    freq_grid_y: Float[Array, "H W"]
    freq_grid_x: Float[Array, "H W"]
    freq_grid_y, freq_grid_x = jnp.meshgrid(freq_y, freq_x, indexing="ij")
    freq_sq: Float[Array, "H W"] = freq_grid_y**2 + freq_grid_x**2
    gaussian_filter: Float[Array, "H W"] = jnp.exp(
        -2.0 * jnp.pi**2 * psf_sigma_pixels**2 * freq_sq
    )
    image_fft: Complex[Array, "H W"] = jnp.fft.fft2(detector_image)
    filtered_fft: Complex[Array, "H W"] = image_fft * gaussian_filter
    broadened_image: Float[Array, "H W"] = jnp.real(
        jnp.fft.ifft2(filtered_fft)
    )
    return jnp.maximum(broadened_image, 0.0)


@jaxtyped(typechecker=beartype)
def instrument_broadened_pattern(
    simulate_fn: Callable[
        [scalar_float, scalar_float, scalar_float], Float[Array, "H W"]
    ],
    nominal_polar_angle_rad: scalar_float,
    nominal_azimuth_angle_rad: scalar_float,
    nominal_energy_kev: scalar_float,
    angular_divergence_mrad: scalar_float,
    energy_spread_ev: scalar_float,
    psf_sigma_pixels: scalar_float,
    n_angular_samples: int = 7,
    n_energy_samples: int = 5,
) -> Float[Array, "H W"]:  # noqa: PLR0913
    r"""Compute fully instrument-broadened RHEED pattern.

    Extended Summary
    ----------------
    Combines all instrument response effects into a single pipeline:

    1. Angular divergence averaging (incoherent sum over incident
       angles).
    2. Energy spread averaging (incoherent sum over beam energies).
    3. Detector PSF convolution (spatial blurring).

    This is the function to call for any comparison with experimental
    data. All operations are differentiable via ``jax.grad``.

    :see: :class:`~.test_beam_averaging.TestInstrumentBroadenedPattern`

    Parameters
    ----------
    simulate_fn : Callable[..., Float[Array, "H W"]]
        Function mapping
        ``(polar_angle_rad, azimuth_angle_rad, energy_kev)`` to an
        intensity pattern. Must be vmappable over both angle and energy
        samples.
    nominal_polar_angle_rad : scalar_float
        Mean incidence angle :math:`\alpha_0` in radians.
    nominal_azimuth_angle_rad : scalar_float
        Mean azimuth :math:`\varphi_0` in radians.
    nominal_energy_kev : scalar_float
        Mean beam energy :math:`E_0` in keV.
    angular_divergence_mrad : scalar_float
        1-sigma angular divergence :math:`\sigma_\theta` in
        milliradians.
    energy_spread_ev : scalar_float
        1-sigma energy spread :math:`\sigma_E` in eV.
    psf_sigma_pixels : scalar_float
        Detector PSF 1-sigma width in pixels. Use 0.0 to skip.
    n_angular_samples : int, optional
        Gauss-Hermite quadrature points for angular average.
        Default: 7
    n_energy_samples : int, optional
        Gauss-Hermite quadrature points for energy average. Default: 5

    Returns
    -------
    final_pattern : Float[Array, "H W"]
        Fully instrument-broadened intensity pattern.

    Notes
    -----
    1. **Build a Distribution** --
       Convert angular and energy quadrature grids to one incoherent
       distribution over ``[polar_angle_rad, azimuth_angle_rad, energy_kev]``.
    2. **Incoherent reduction** --
       Average the simulated intensities directly through
       :func:`apply_distribution_intensity`.
    3. **PSF convolution** --
       Apply :func:`detector_psf_convolve` to the combined pattern.
    4. **Clip** --
       Ensure non-negative output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> def joint_sim(polar, azimuth, energy_kev):
    ...     del azimuth
    ...     return jnp.ones((64, 64)) * (polar + energy_kev)
    >>> pattern = rh.simul.instrument_broadened_pattern(
    ...     joint_sim,
    ...     jnp.float64(0.035),
    ...     jnp.float64(0.0),
    ...     jnp.float64(20.0),
    ...     jnp.float64(0.5),
    ...     jnp.float64(0.5),
    ...     jnp.float64(1.0),
    ... )
    >>> pattern.shape
    (64, 64)
    """
    divergence_rad: scalar_float = angular_divergence_mrad * 1e-3
    spread_kev: scalar_float = energy_spread_ev * 1e-3
    angle_nodes: Float[Array, " N_a"]
    angle_weights: Float[Array, " N_a"]
    energy_nodes: Float[Array, " N_e"]
    energy_weights: Float[Array, " N_e"]
    angle_nodes, angle_weights = _gauss_hermite_nodes_weights(
        n_angular_samples
    )
    energy_nodes, energy_weights = _gauss_hermite_nodes_weights(
        n_energy_samples
    )
    angle_samples: Float[Array, "N_a"] = (
        nominal_polar_angle_rad + jnp.sqrt(2.0) * divergence_rad * angle_nodes
    )
    energy_samples: Float[Array, "N_e"] = (
        nominal_energy_kev + jnp.sqrt(2.0) * spread_kev * energy_nodes
    )
    polar_grid: Float[Array, "N_a N_e"]
    energy_grid: Float[Array, "N_a N_e"]
    polar_grid, energy_grid = jnp.meshgrid(
        angle_samples,
        energy_samples,
        indexing="ij",
    )
    azimuth_grid: Float[Array, "N_a N_e"] = jnp.full_like(
        polar_grid,
        nominal_azimuth_angle_rad,
    )
    weight_grid: Float[Array, "N_a N_e"] = (
        angle_weights[:, None] * energy_weights[None, :]
    )
    distribution: Distribution = create_distribution(
        samples=jnp.stack(
            [
                polar_grid.ravel(),
                azimuth_grid.ravel(),
                energy_grid.ravel(),
            ],
            axis=-1,
        ),
        weights=weight_grid.ravel(),
        reduction=ReductionMode.INCOHERENT,
        axis_id="instrument_quadrature",
    )

    def _instrument_intensity(
        sample: Float[Array, "3"],
    ) -> Float[Array, "H W"]:
        return simulate_fn(
            sample[0],
            sample[1],
            sample[2],
        )

    combined: Float[Array, "H W"] = apply_distribution_intensity(
        distribution,
        _instrument_intensity,
    )
    final_pattern: Float[Array, "H W"] = detector_psf_convolve(
        detector_image=combined,
        psf_sigma_pixels=psf_sigma_pixels,
    )
    return jnp.maximum(final_pattern, 0.0)


__all__: list[str] = [
    "angular_divergence_average",
    "apply_distribution",
    "apply_distribution_intensity",
    "apply_distributions",
    "decompose_beam_modes",
    "decompose_beam_modes_static",
    "detector_psf_convolve",
    "energy_spread_average",
    "instrument_broadened_pattern",
]
