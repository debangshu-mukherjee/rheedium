"""Physics-anchor tests for red-team remediation.

These tests compare against analytic identities or external numerical
references rather than against rheedium's own implementation choices.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from rheedium.inout import kirkland_potentials, lobato_potentials
from rheedium.inout.cif import parse_cif
from rheedium.inout.interop import from_ase, to_ase
from rheedium.simul.ewald import _compute_structure_factor_single
from rheedium.simul.form_factors import (
    get_atomic_mass,
    get_debye_temperature,
    get_mean_square_displacement,
    kirkland_form_factor,
    kirkland_projected_potential,
    lobato_form_factor,
    lobato_projected_potential,
)
from rheedium.tools import bessel_k0, bessel_k1
from rheedium.types import (
    AMU_TO_KG,
    BOLTZMANN_CONSTANT_JK,
    HBAR_JS,
    M2_TO_ANG2,
)
from rheedium.ucell import reciprocal_lattice_vectors
from rheedium.ucell.unitcell import bulk_to_slice


def test_lobato_table_bethe_sum_rule() -> None:
    r"""Lobato coefficients obey the exact Bethe asymptote sum rule.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the shipped Lobato
    parameter table satisfies the analytic Mott-Bethe sum rule
    ``sum(a_i / b_i) = 0.0957336 Z`` for every element.

    Notes
    -----
    It loads the full Lobato table via ``lobato_potentials`` and splits the
    interleaved amplitude/scale columns inside the test body.

    Numerical expectations are checked with tolerance-aware closeness
    assertions against the analytic sum rule for Z from 1 to 103, which is
    appropriate for floating-point table data.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    params = np.asarray(lobato_potentials())
    a = params[:, 0::2]
    b = params[:, 1::2]
    z = np.arange(1, 104, dtype=np.float64)
    expected = 0.0957336 * z
    np.testing.assert_array_less(0.0, b)
    np.testing.assert_allclose(np.sum(a / b, axis=1), expected, rtol=1e-3)


def test_lobato_vs_kirkland_f0() -> None:
    r"""Lobato and Kirkland f(0) agree at table-load time.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the zero-angle
    scattering factors implied by the two independent parameter tables agree
    element by element, guarding against a corrupted or misordered table.

    Notes
    -----
    It computes ``f(0)`` analytically from both raw tables inside the test
    body, without calling any rheedium form-factor kernels.

    The result is checked with direct ratio bounds: every element within 15
    percent and the median within 5 percent, tolerances that reflect genuine
    parameterization differences.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    lobato_params = np.asarray(lobato_potentials())
    kirkland_params = np.asarray(kirkland_potentials())
    lobato_f0 = 2.0 * np.sum(lobato_params[:, 0::2], axis=1)
    kirkland_f0 = np.sum(
        kirkland_params[:, 0:6:2] / kirkland_params[:, 1:6:2], axis=1
    ) + np.sum(kirkland_params[:, 6:12:2], axis=1)
    ratio = lobato_f0 / kirkland_f0
    assert np.all(np.abs(ratio - 1.0) < 0.15)
    assert np.median(np.abs(ratio - 1.0)) < 0.05


def test_kirkland_lobato_cross_agreement() -> None:
    r"""Independent Kirkland/Lobato parameterizations of f_e agree.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the packaged
    ``kirkland_form_factor`` and ``lobato_form_factor`` kernels produce
    mutually consistent electron scattering factors at finite momentum
    transfer, so neither kernel misinterprets its table units.

    Notes
    -----
    It evaluates both kernels for seven representative elements at four
    momentum transfers inside the test body, collecting the pairwise ratios.

    The result is checked with direct ratio bounds: maximum deviation below 20
    percent and median below 8 percent, tolerances that reflect genuine
    parameterization differences.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    ratios = []
    for z in (6, 8, 13, 14, 26, 38, 79):
        for g_inv_ang in (0.5, 1.0, 2.0, 4.0):
            q_pkg = jnp.asarray([2.0 * jnp.pi * g_inv_ang], dtype=jnp.float64)
            f_lobato = float(lobato_form_factor(z, q_pkg)[0])
            f_kirkland = float(kirkland_form_factor(z, q_pkg)[0])
            ratios.append(f_kirkland / f_lobato)
    deviations = np.abs(np.asarray(ratios) - 1.0)
    assert float(np.max(deviations)) < 0.20
    assert float(np.median(deviations)) < 0.08


def test_kirkland_bethe_asymptote() -> None:
    r"""Kirkland's corrected argument follows the Mott-Bethe asymptote.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: at large momentum
    transfer the Kirkland form factor approaches the analytic Mott-Bethe
    limit ``0.023934 Z / g^2``, anchoring the kernel's argument convention.

    Notes
    -----
    It evaluates ``kirkland_form_factor`` for silicon at ``g = 4`` inverse
    Angstroms inside the test body, comparing against the closed-form
    asymptote.

    The result is checked with a direct ratio bound of 15 percent, which is
    appropriate because the asymptote is only approached at finite ``g``.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    q_pkg = jnp.asarray([2.0 * jnp.pi * 4.0], dtype=jnp.float64)
    f_si = float(kirkland_form_factor(14, q_pkg)[0])
    expected = 0.023934 * 14.0 / (2.0**2)
    assert abs(f_si / expected - 1.0) < 0.15


@pytest.mark.parametrize(
    ("form_factor_fn", "potential_fn"),
    [
        (kirkland_form_factor, kirkland_projected_potential),
        (lobato_form_factor, lobato_projected_potential),
    ],
)
@pytest.mark.parametrize("z", [14, 79])
@pytest.mark.parametrize("r_ang", [0.2, 0.5, 1.0])
def test_projected_potential_is_hankel_pair_of_form_factor(
    form_factor_fn: Callable[[Any, Any], Any],
    potential_fn: Callable[[Any, Any], Any],
    z: int,
    r_ang: float,
) -> None:
    r"""Each projected potential matches the Hankel transform of its f(g).

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the projected
    potential kernels are the zeroth-order Hankel transforms of their own
    form factors, so the real-space and reciprocal-space parameterizations
    stay mutually consistent.

    Notes
    -----
    It receives parametrized inputs named ``form_factor_fn``,
    ``potential_fn``, ``z``, ``r_ang``, so the documented behavior is checked
    across both Kirkland and Lobato kernels, two elements, and three radii.

    Numerical expectations are checked with a tolerance-aware ratio bound of
    2 percent against a dense scipy trapezoid quadrature of the analytic
    Hankel integral.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    pref = 47.87801 * 2.0 * np.pi
    g = np.linspace(0.0, 80.0, 200_000)
    q_pkg = jnp.asarray(2.0 * np.pi * g, dtype=jnp.float64)
    f = np.asarray(form_factor_fn(z, q_pkg))
    integrand = f * scipy.special.j0(2.0 * np.pi * g * r_ang) * g
    ref = pref * np.trapezoid(integrand, g)
    code_vz = float(potential_fn(z, jnp.asarray([r_ang]))[0])
    assert abs(code_vz / ref - 1.0) < 0.02


@pytest.mark.parametrize(
    ("kernel", "scipy_ref"),
    [(bessel_k0, scipy.special.k0), (bessel_k1, scipy.special.k1)],
)
def test_bessel_k_gradients_are_finite_at_large_x(
    kernel: Callable[[Any], Any],
    scipy_ref: Callable[[Any], Any],
) -> None:
    r"""Untaken small-x branches do not NaN-poison large-x gradients.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the modified Bessel
    kernels remain differentiable at large arguments where the untaken
    small-argument branch would overflow, and their values track scipy.

    Notes
    -----
    It receives parametrized inputs named ``kernel``, ``scipy_ref``, so the
    documented behavior is checked for both ``bessel_k0`` and ``bessel_k1``.

    The gradient at ``x = 800`` is checked for finiteness through
    ``jax.grad``, and values across both branch regimes are checked with
    tolerance-aware closeness assertions against the scipy references.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    grad_val = jax.grad(lambda x: kernel(jnp.asarray(x, dtype=jnp.float64)))(
        800.0
    )
    assert math.isfinite(float(grad_val))
    x = np.asarray([1e-3, 1.9, 2.1, 100.0])
    got = np.asarray(kernel(jnp.asarray(x, dtype=jnp.float64)))
    np.testing.assert_allclose(got, scipy_ref(x), rtol=1e-6, atol=1e-10)


def test_debye_msd_full_model_limits() -> None:
    r"""The full Debye MSD has the correct high-T and zero-point limits.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: the mean-square
    displacement model reproduces the classical high-temperature Debye limit,
    the quantum zero-point limit, and monotonic growth with temperature.

    Notes
    -----
    It constructs the analytic limit expressions for silicon inside the test
    body from the same physical constants the source uses.

    The high-temperature ratio is checked within 3 percent, the zero-point
    ratio within 1 percent, and monotonicity is checked with an exact
    positivity assertion on finite differences.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    z = 14
    theta = float(get_debye_temperature(z))
    mass_kg = float(get_atomic_mass(z)) * AMU_TO_KG
    high_t = 2000.0
    old_high_t = (
        3.0
        * HBAR_JS**2
        * high_t
        / (mass_kg * BOLTZMANN_CONSTANT_JK * theta**2)
        * M2_TO_ANG2
    )
    new_high_t = float(get_mean_square_displacement(z, high_t))
    assert abs(new_high_t / old_high_t - 1.0) < 0.03

    low_t = float(get_mean_square_displacement(z, 1.0))
    zero_point = (
        3.0
        * HBAR_JS**2
        / (4.0 * mass_kg * BOLTZMANN_CONSTANT_JK * theta)
        * M2_TO_ANG2
    )
    assert abs(low_t / zero_point - 1.0) < 0.01

    temps = jnp.asarray([1.0, 100.0, 300.0, 1000.0], dtype=jnp.float64)
    msds = jnp.asarray([get_mean_square_displacement(z, t) for t in temps])
    assert bool(jnp.all(jnp.diff(msds) > 0.0))


def test_frame_contract_roundtrip_from_ase() -> None:
    r"""Primitive ASE cells canonicalize positions without losing phases.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: importing a
    primitive ASE cell preserves the frame contract, so structure-factor
    phases computed from Cartesian positions honor diamond extinction rules
    and the roundtrip back to ASE keeps scaled positions.

    Notes
    -----
    It constructs a diamond silicon primitive cell through ``ase.build.bulk``
    inside the test body, skipping cleanly when ASE is unavailable.

    The forbidden (222)/(111) structure-factor ratio is checked against a
    strict extinction bound, and the ASE roundtrip is checked with
    tolerance-aware closeness assertions on scaled positions.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    bulk = pytest.importorskip("ase.build").bulk
    atoms = bulk("Si", "diamond", a=5.43)

    crystal = from_ase(atoms)
    recip = reciprocal_lattice_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    atomic_numbers = crystal.cart_positions[:, 3].astype(jnp.int32)

    f111 = _compute_structure_factor_single(
        jnp.asarray([1.0, 1.0, 1.0]) @ recip,
        crystal.cart_positions[:, :3],
        atomic_numbers,
        300.0,
        "kirkland",
    )
    f222 = _compute_structure_factor_single(
        jnp.asarray([2.0, 2.0, 2.0]) @ recip,
        crystal.cart_positions[:, :3],
        atomic_numbers,
        300.0,
        "kirkland",
    )
    assert float(jnp.abs(f222) / jnp.abs(f111)) < 1e-6

    np.testing.assert_allclose(
        to_ase(crystal).get_scaled_positions(),
        atoms.get_scaled_positions(),
        atol=1e-8,
    )


@pytest.mark.parametrize("orientation", [(0, 0, 1), (1, 1, 1)])
def test_slab_density(orientation: tuple[int, int, int]) -> None:
    r"""SrTiO3 slabs tile all three lattice directions at bulk density.

    Extended Summary
    ----------------
    Verifies the documented behavior for this test case: ``bulk_to_slice``
    fills the requested slab volume at bulk atomic density with no missing
    layers or truncated lateral tiling.

    Notes
    -----
    It receives a parametrized input named ``orientation``, so the documented
    behavior is checked for both (001) and (111) surface cuts of the SrTiO3
    fixture crystal.

    The atom count is checked within 5 percent of the bulk-density
    expectation, the slab depth within 10 percent of the request, and every
    5 Angstrom depth band is checked for occupancy and lateral extent.

    The documented check is rendered from
    ``tests.test_rheedium.test_physics_anchors``, so the Test Reference
    exposes both the guarantee and the implementation path.
    """
    crystal = parse_cif(Path("tests/test_data/SrTiO3.cif"))
    depth = 20.0
    x_extent = 40.0
    y_extent = 40.0
    slab = bulk_to_slice(
        crystal,
        jnp.asarray(orientation, dtype=jnp.int32),
        depth,
        x_extent,
        y_extent,
    )
    xyz = np.asarray(slab.cart_positions[:, :3])
    bulk_density = 5.0 / float(crystal.cell_lengths[0]) ** 3
    expected_atoms = bulk_density * x_extent * y_extent * depth

    assert np.ptp(xyz[:, 2]) >= 0.9 * depth
    assert abs(xyz.shape[0] / expected_atoms - 1.0) < 0.05

    for z_min in np.arange(0.0, depth, 5.0):
        band = xyz[(xyz[:, 2] >= z_min) & (xyz[:, 2] < z_min + 5.0)]
        assert band.shape[0] > 0
        assert np.ptp(band[:, 0]) >= 0.8 * x_extent


def test_ewald_default_tolerance_matches_beam_physics() -> None:
    r"""Default Ewald tolerance is 3 sigma of the physical shell at 20 kV.

    Extended Summary
    ----------------
    Anchors the WP4.1 tolerance convention against closed-form beam
    physics computed with scipy constants only. The relativistic electron
    wavelength at 20 kV gives k = 2 pi / lambda ~ 73.2 inverse Angstroms,
    and the Ewald-shell Gaussian width from an energy spread of 1e-4 and
    a divergence of 1 mrad is sigma = k sqrt((dE/2E)^2 + dtheta^2), so
    the default acceptance window 3 sigma must be about 0.22 inverse
    Angstroms - an order of magnitude below the 3.66 the old fractional
    default (0.05 |k|) allowed, and below reciprocal-lattice spacings.

    Notes
    -----
    It computes the reference wavelength from CODATA constants via
    scipy.constants and the shell width from the quadrature formula,
    entirely outside rheedium, then compares the package's derived
    default tolerance (3 x compute_shell_sigma at the package k) against
    that external value.

    :see: :func:`~rheedium.simul.ewald_allowed_reflections`
    :see: :func:`~rheedium.simul.find_kinematic_reflections`
    :see: :func:`~rheedium.simul.compute_shell_sigma`
    """
    from scipy import constants

    from rheedium.simul.finite_domain import compute_shell_sigma
    from rheedium.tools import wavelength_ang

    voltage_v = 20.0e3
    e_joule = constants.e * voltage_v
    p = math.sqrt(
        2.0
        * constants.m_e
        * e_joule
        * (1.0 + e_joule / (2.0 * constants.m_e * constants.c**2))
    )
    lambda_ref_ang = constants.h / p * 1e10
    k_ref = 2.0 * math.pi / lambda_ref_ang
    sigma_ref = k_ref * math.sqrt((1e-4 / 2.0) ** 2 + (1e-3) ** 2)

    k_pkg = float(2.0 * jnp.pi / wavelength_ang(jnp.asarray(20.0)))
    tol_pkg = 3.0 * float(
        compute_shell_sigma(
            k_magnitude=jnp.asarray(k_pkg),
            energy_spread_frac=1e-4,
            beam_divergence_rad=1e-3,
        )
    )
    assert abs(k_pkg / k_ref - 1.0) < 1e-4
    assert abs(tol_pkg / (3.0 * sigma_ref) - 1.0) < 1e-4
    # The physical default admits |dk| < ~0.25 1/A, not 3.66 1/A
    assert tol_pkg < 0.25
    assert tol_pkg > 0.1


def test_periodic_potential_grid_fft_is_real_for_centrosymmetric_atom() -> (
    None
):
    r"""A single atom at the origin yields a real FFT on the periodic grid.

    Extended Summary
    ----------------
    Anchors the WP4.4 grid convention against a Fourier identity: a real
    potential that is even under x -> L - x on a periodic grid sampled at
    x_i = i L / n has a purely real DFT. A single atom at (0, 0) with
    minimum-image wrapping produces exactly such an even function only
    when the grid excludes the duplicated endpoint; the old
    linspace(0, L, n) sampling (spacing L/(n-1)) broke the symmetry and
    double-weighted the boundary column, leaving a spurious imaginary
    part inconsistent with the fftfreq(n, L/n) propagator lattice.

    Notes
    -----
    It builds the elastic projected potential of one silicon atom at the
    origin on a 16 x 16 periodic grid with numpy's FFT and checks that
    the maximum imaginary magnitude is negligible against the zero-
    frequency component, an identity that holds independently of the
    potential parameterization.

    :see: :func:`~rheedium.simul.crystal_projected_potential`
    """
    from rheedium.simul.potential import crystal_projected_potential

    v = crystal_projected_potential(
        atomic_positions_angstrom=jnp.asarray([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.asarray([14], dtype=jnp.int32),
        grid_shape=(16, 16),
        cell_dimensions_angstrom=jnp.asarray([8.0, 8.0]),
        absorption_fraction=0.0,
        parameterization="lobato",
    )
    v_real = np.asarray(jnp.real(v))
    spectrum = np.fft.fft2(v_real)
    assert np.abs(spectrum.imag).max() < 1e-10 * np.abs(spectrum[0, 0])
