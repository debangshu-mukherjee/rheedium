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
:func:`gauss_hermite_nodes_weights`
    Gauss-Hermite quadrature nodes and weights for Gaussian averaging.
:func:`angular_divergence_average`
    Average pattern over Gaussian angular divergence distribution.
:func:`energy_spread_average`
    Average pattern over Gaussian energy spread distribution.
:func:`coherence_envelope`
    Apply partial coherence damping envelope in reciprocal space.
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

Coherence envelopes act as multiplicative Gaussian damping in
reciprocal space. The PSF convolution uses FFT for
:math:`O(HW \log HW)` complexity and is fully differentiable
through ``jnp.fft``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812

from rheedium.types import scalar_float


@jaxtyped(typechecker=beartype)
def gauss_hermite_nodes_weights(
    n_points: int,
) -> Tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Compute Gauss-Hermite quadrature nodes and weights.

    Extended Summary
    ----------------
    Returns the nodes :math:`x_i` and weights :math:`w_i` for
    Gauss-Hermite quadrature, which integrates functions against the
    weight :math:`\exp(-x^2)`:

    .. math::

        \int_{-\infty}^{\infty} f(x) e^{-x^2} dx
        \approx \sum_{i=1}^{n} w_i f(x_i)

    To average over a Gaussian distribution
    :math:`\mathcal{N}(\mu, \sigma^2)`, map nodes as
    :math:`\alpha_i = \mu + \sqrt{2} \sigma x_i` and normalize by
    :math:`1/\sqrt{\pi}`.

    Parameters
    ----------
    n_points : int
        Number of quadrature nodes. Must be positive. For RHEED
        averaging, 5--9 is sufficient.

    Returns
    -------
    nodes : Float[Array, " N"]
        Quadrature nodes :math:`x_i`.
    weights : Float[Array, " N"]
        Quadrature weights :math:`w_i`.

    Notes
    -----
    1. **Compute nodes and weights** --
       Use NumPy polynomial Gauss-Hermite routine (exact for
       polynomials up to degree :math:`2n - 1`).
    2. **Convert to JAX arrays** --
       Cast to float64 JAX arrays for downstream computation.

    Examples
    --------
    >>> import rheedium as rh
    >>> nodes, weights = rh.simul.gauss_hermite_nodes_weights(7)
    >>> nodes.shape
    (7,)
    """
    np_nodes: Float[NDArray, "N"]
    np_weights: Float[NDArray, "N"]
    np_nodes, np_weights = np.polynomial.hermite.hermgauss(n_points)
    nodes: Float[Array, " N"] = jnp.asarray(np_nodes, dtype=jnp.float64)
    weights: Float[Array, " N"] = jnp.asarray(np_weights, dtype=jnp.float64)
    return nodes, weights


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

    Parameters
    ----------
    simulate_fn : Callable[[scalar_float, scalar_float], \
            Float[Array, "H W"]]
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
    ...     dummy_sim, jnp.float64(0.035), jnp.float64(0.0),
    ...     jnp.float64(0.5), n_quadrature_points=5,
    ... )
    >>> avg.shape
    (64, 64)
    """
    divergence_rad: scalar_float = angular_divergence_mrad * 1e-3
    nodes: Float[Array, " N"]
    weights: Float[Array, " N"]
    nodes, weights = gauss_hermite_nodes_weights(n_quadrature_points)
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
    ...     dummy_sim, jnp.float64(20.0), jnp.float64(0.5),
    ... )
    >>> avg.shape
    (64, 64)
    """
    spread_kev: scalar_float = energy_spread_ev * 1e-3
    nodes: Float[Array, " N"]
    weights: Float[Array, " N"]
    nodes, weights = gauss_hermite_nodes_weights(n_quadrature_points)
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
def coherence_envelope(
    reciprocal_space_amplitude: Complex[Array, "H W"],
    transverse_coherence_length_angstrom: scalar_float,
    longitudinal_coherence_length_angstrom: scalar_float,
    q_parallel: Float[Array, "H W"],
    q_z: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    r"""Apply partial coherence damping to diffraction amplitude.

    Extended Summary
    ----------------
    Coherence effects act as Gaussian envelopes that damp high-q
    components of the diffraction amplitude:

    .. math::

        E_t(q_\parallel) = \exp\!\bigl(
            -q_\parallel^2 L_t^2 / 2
        \bigr)

    .. math::

        E_l(q_z) = \exp\!\bigl(-q_z^2 L_l^2 / 2\bigr)

    These envelopes model the finite spatial extent of electron
    wavepackets and set the maximum spatial frequency that can
    produce visible interference fringes.

    Parameters
    ----------
    reciprocal_space_amplitude : Complex[Array, "H W"]
        Diffraction amplitude :math:`F(\mathbf{q})` at each detector
        pixel.
    transverse_coherence_length_angstrom : scalar_float
        Transverse coherence length :math:`L_t` in Angstroms. Controls
        damping of in-plane high-q components.
    longitudinal_coherence_length_angstrom : scalar_float
        Longitudinal coherence length :math:`L_l` in Angstroms.
        Controls damping along :math:`q_z` (streak modulation).
    q_parallel : Float[Array, "H W"]
        In-plane momentum transfer magnitude
        :math:`q_\parallel = \sqrt{q_x^2 + q_y^2}` at each pixel,
        in inverse Angstroms.
    q_z : Float[Array, "H W"]
        Out-of-plane momentum transfer :math:`q_z` at each pixel,
        in inverse Angstroms.

    Returns
    -------
    damped_amplitude : Complex[Array, "H W"]
        Amplitude multiplied by coherence envelope.

    Notes
    -----
    1. **Transverse envelope** --
       :math:`E_t = \exp(-q_\parallel^2 L_t^2 / 2)`.
    2. **Longitudinal envelope** --
       :math:`E_l = \exp(-q_z^2 L_l^2 / 2)`.
    3. **Apply damping** --
       Return :math:`F(\mathbf{q}) \cdot E_t \cdot E_l`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> amp = jnp.ones((64, 64), dtype=jnp.complex128)
    >>> q_par = jnp.ones((64, 64)) * 0.1
    >>> q_z = jnp.ones((64, 64)) * 0.05
    >>> damped = rh.simul.coherence_envelope(
    ...     amp, jnp.float64(500.0), jnp.float64(1000.0),
    ...     q_par, q_z,
    ... )
    >>> damped.shape
    (64, 64)
    """
    l_t: scalar_float = transverse_coherence_length_angstrom
    l_l: scalar_float = longitudinal_coherence_length_angstrom
    envelope_transverse: Float[Array, "H W"] = jnp.exp(
        -0.5 * q_parallel**2 * l_t**2
    )
    envelope_longitudinal: Float[Array, "H W"] = jnp.exp(
        -0.5 * q_z**2 * l_l**2
    )
    damped_amplitude: Complex[Array, "H W"] = (
        reciprocal_space_amplitude
        * envelope_transverse
        * envelope_longitudinal
    )
    return damped_amplitude


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
    ...     img, jnp.float64(1.5),
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
    simulate_angle_fn: Callable[
        [scalar_float, scalar_float], Float[Array, "H W"]
    ],
    simulate_energy_fn: Callable[[scalar_float], Float[Array, "H W"]],
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

    Parameters
    ----------
    simulate_angle_fn : Callable[[scalar_float, scalar_float], \
            Float[Array, "H W"]]
        Function mapping ``(polar_angle_rad, azimuth_angle_rad)`` to
        intensity pattern. Must be vmappable.
    simulate_energy_fn : Callable[[scalar_float], Float[Array, "H W"]]
        Function mapping ``energy_kev`` to intensity pattern at the
        nominal angles. Must be vmappable.
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
    1. **Angular averaging** --
       Call :func:`angular_divergence_average` with ``simulate_angle_fn``
       and beam divergence.
    2. **Energy averaging** --
       Call :func:`energy_spread_average` with ``simulate_energy_fn``
       and energy spread.
    3. **Combine** --
       Average the angular-averaged and energy-averaged patterns with
       equal weight.
    4. **PSF convolution** --
       Apply :func:`detector_psf_convolve` to the combined pattern.
    5. **Clip** --
       Ensure non-negative output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> def angle_sim(polar, azimuth):
    ...     return jnp.ones((64, 64)) * polar
    >>> def energy_sim(energy_kev):
    ...     return jnp.ones((64, 64)) * energy_kev
    >>> pattern = rh.simul.instrument_broadened_pattern(
    ...     angle_sim, energy_sim,
    ...     jnp.float64(0.035), jnp.float64(0.0),
    ...     jnp.float64(20.0), jnp.float64(0.5),
    ...     jnp.float64(0.5), jnp.float64(1.0),
    ... )
    >>> pattern.shape
    (64, 64)
    """
    angle_averaged: Float[Array, "H W"] = angular_divergence_average(
        simulate_fn=simulate_angle_fn,
        nominal_polar_angle_rad=nominal_polar_angle_rad,
        nominal_azimuth_angle_rad=nominal_azimuth_angle_rad,
        angular_divergence_mrad=angular_divergence_mrad,
        n_quadrature_points=n_angular_samples,
    )
    energy_averaged: Float[Array, "H W"] = energy_spread_average(
        simulate_fn=simulate_energy_fn,
        nominal_energy_kev=nominal_energy_kev,
        energy_spread_ev=energy_spread_ev,
        n_quadrature_points=n_energy_samples,
    )
    combined: Float[Array, "H W"] = 0.5 * (angle_averaged + energy_averaged)
    final_pattern: Float[Array, "H W"] = detector_psf_convolve(
        detector_image=combined,
        psf_sigma_pixels=psf_sigma_pixels,
    )
    return jnp.maximum(final_pattern, 0.0)


__all__: list[str] = [
    "angular_divergence_average",
    "coherence_envelope",
    "detector_psf_convolve",
    "energy_spread_average",
    "gauss_hermite_nodes_weights",
    "instrument_broadened_pattern",
]
