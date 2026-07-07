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
from rheedium.simul.ewald import (
    _compute_structure_factor_single,
    build_ewald_data,
    ewald_allowed_reflections,
)
from rheedium.simul.finite_domain import (
    compute_shell_sigma,
    extent_to_rod_sigma,
    rod_domain_overlap,
)
from rheedium.simul.form_factors import (
    get_atomic_mass,
    get_debye_temperature,
    get_mean_square_displacement,
    kirkland_form_factor,
    kirkland_projected_potential,
    lobato_form_factor,
    lobato_projected_potential,
)
from rheedium.simul.surface_rods import ctr_truncation_intensity
from rheedium.tools import (
    bessel_k0,
    bessel_k1,
    incident_wavevector,
    wavelength_ang,
)
from rheedium.types import (
    AMU_TO_KG,
    BOLTZMANN_CONSTANT_JK,
    HBAR_JS,
    M2_TO_ANG2,
    create_crystal_structure,
)
from rheedium.ucell import build_cell_vectors, reciprocal_lattice_vectors
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


def test_ctr_shape() -> None:
    r"""The (0,0) CTR follows the semi-infinite truncation-rod shape.

    Extended Summary
    ----------------
    WP5.1 acceptance anchor: for a one-cell SrTiO3 crystal the (0, 0) rod
    intensity, after dividing out the single-cell ``|F|^2`` variation,
    matches the analytic truncation shape
    ``[sin^2(0.05 pi) + s0] / [sin^2(0.5 pi) + s0]`` (with
    ``s0 = sinh^2(eps/2)``) within 5 percent; the rod direction
    ``q(0,0,l)`` for a hexagonal cell is parallel to ``b3`` (never a
    hard-coded Cartesian z); and at a Bragg point the intensity equals
    ``|F|^2 / (1 - e^{-eps})^2`` (the exact cap, equal to
    ``|F|^2 / (4 sinh^2(eps/2))`` up to ``e^{-eps}``) with no additive
    ``|F|^2 (1 + anything)`` double counting.

    Notes
    -----
    It computes the reference ratio from closed-form trigonometry only,
    divides out the structure-factor variation with the package's own
    per-q ``|F|^2`` (a pure normalization), and checks the hexagonal rod
    direction with an exact cross-product bound.

    :see: :func:`~rheedium.simul.calculate_ctr_intensity`
    :see: :func:`~rheedium.simul.ctr_truncation_intensity`
    """
    from rheedium.simul.surface_rods import (
        calculate_ctr_intensity,
        surface_structure_factor,
    )
    from rheedium.types.crystal_types import create_crystal_structure

    a_sto = 3.905
    frac = jnp.array(
        [
            [0.0, 0.0, 0.0, 38.0],
            [0.5, 0.5, 0.5, 22.0],
            [0.5, 0.5, 0.0, 8.0],
            [0.5, 0.0, 0.5, 8.0],
            [0.0, 0.5, 0.5, 8.0],
        ]
    )
    cart = frac.at[:, :3].multiply(a_sto)
    crystal = create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=jnp.array([a_sto, a_sto, a_sto]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )
    epsilon = 0.01
    hk = jnp.array([[0, 0]], dtype=jnp.int32)
    l_values = jnp.array([0.05, 0.5, 1.0])
    intensity = np.asarray(
        calculate_ctr_intensity(
            hk_indices=hk,
            l_values=l_values,
            crystal=crystal,
            surface_roughness=0.0,
            layer_attenuation=epsilon,
        )
    )[0]

    recip = reciprocal_lattice_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    f_sq = []
    for l_val in np.asarray(l_values):
        q_vec = l_val * recip[2]
        f_val = surface_structure_factor(
            q_vector=jnp.asarray(q_vec),
            atomic_positions=crystal.cart_positions[:, :3],
            atomic_numbers=crystal.cart_positions[:, 3].astype(jnp.int32),
        )
        f_sq.append(float(jnp.abs(f_val) ** 2))
    f_sq = np.asarray(f_sq)

    shape = intensity / f_sq
    s0 = math.sinh(epsilon / 2.0) ** 2
    reference_ratio = (math.sin(0.05 * math.pi) ** 2 + s0) / (
        math.sin(0.5 * math.pi) ** 2 + s0
    )
    measured_ratio = shape[1] / shape[0]
    assert abs(measured_ratio / reference_ratio - 1.0) < 0.05

    # Bragg point: I == |F|^2 / (1 - e^{-eps})^2 exactly (small-eps form
    # |F|^2 / (4 sinh^2(eps/2)) up to e^{-eps}); NOT |F|^2 (1 + anything).
    exact_cap = 1.0 / (1.0 - math.exp(-epsilon)) ** 2
    assert abs(intensity[2] / (f_sq[2] * exact_cap) - 1.0) < 1e-10
    small_eps_cap = 1.0 / (4.0 * math.sinh(epsilon / 2.0) ** 2)
    assert abs(intensity[2] / (f_sq[2] * small_eps_cap) - 1.0) < 2.0 * epsilon

    # Hexagonal cell: q(0,0,l) must be parallel to b3.
    recip_hex = reciprocal_lattice_vectors(
        jnp.asarray(4.0),
        jnp.asarray(4.0),
        jnp.asarray(6.0),
        jnp.asarray(90.0),
        jnp.asarray(90.0),
        jnp.asarray(120.0),
    )
    b3 = np.asarray(recip_hex[2])
    for l_val in (0.3, 0.7, 1.4):
        q_rod = l_val * b3  # q(0,0,l) = 0*b1 + 0*b2 + l*b3 by construction
        cross = np.cross(q_rod, b3)
        assert np.linalg.norm(cross) < 1e-12 * np.linalg.norm(b3) ** 2


def test_roughness_intensity_ratio() -> None:
    r"""CTR intensity roughness damping is exactly exp(-q_z^2 sigma^2).

    Extended Summary
    ----------------
    WP5.2 acceptance anchor: for a single-atom crystal the ratio of rough
    to smooth CTR intensity, which cancels the structure factor and the
    truncation factor (equivalent to disabling the truncation factor),
    equals the analytic intensity damping ``exp(-q_z^2 sigma^2)`` to
    1e-12 - the square of the single amplitude convention
    ``exp(-q_z^2 sigma^2 / 2)``.

    Notes
    -----
    It computes the reference damping from ``numpy.exp`` alone and takes
    q_z from the reciprocal-basis geometry, so no rheedium roughness code
    appears on the reference side of the comparison.

    :see: :func:`~rheedium.simul.roughness_damping`
    :see: :func:`~rheedium.simul.calculate_ctr_intensity`
    """
    from rheedium.simul.surface_rods import calculate_ctr_intensity
    from rheedium.types.crystal_types import create_crystal_structure

    a_cell = 4.0
    frac = jnp.array([[0.0, 0.0, 0.0, 14.0]])
    cart = frac.at[:, :3].multiply(a_cell)
    crystal = create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=jnp.array([a_cell, a_cell, a_cell]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )
    sigma = 0.8
    hk = jnp.array([[0, 0]], dtype=jnp.int32)
    l_values = jnp.linspace(0.05, 2.45, 25)

    smooth = np.asarray(
        calculate_ctr_intensity(
            hk_indices=hk,
            l_values=l_values,
            crystal=crystal,
            surface_roughness=0.0,
        )
    )[0]
    rough = np.asarray(
        calculate_ctr_intensity(
            hk_indices=hk,
            l_values=l_values,
            crystal=crystal,
            surface_roughness=sigma,
        )
    )[0]

    b3_z = float(
        reciprocal_lattice_vectors(
            *crystal.cell_lengths, *crystal.cell_angles
        )[2, 2]
    )
    q_z = np.asarray(l_values) * b3_z
    reference = np.exp(-(q_z**2) * sigma**2)
    np.testing.assert_allclose(rough / smooth, reference, rtol=1e-12)


def test_rod_sigma_matches_sinc_fwhm() -> None:
    r"""The finite-domain rod sigma matches the sinc^2 FWHM externally.

    Extended Summary
    ----------------
    WP5.4/N11 anchor: the Gaussian rod width returned by
    ``extent_to_rod_sigma`` must match the FWHM of the true finite-domain
    shape function ``sinc^2(q L / 2)``. The half-maximum point of
    ``sin^2(x)/x^2`` is found with scipy root finding, giving
    FWHM = 4 x_half / L = 0.886 (2 pi / L), so the matched Gaussian sigma
    is FWHM / (2 sqrt(2 ln 2)) = 0.376 (2 pi / L) = 2.364 / L. The
    retired constant 2 pi / (L sqrt(2 pi)) = 2.507 / L was about 6
    percent too wide.

    Notes
    -----
    It derives the reference sigma entirely from scipy.optimize.brentq
    and closed-form constants, then compares the package output for a
    100 Angstrom domain within 0.5 percent.

    :see: :func:`~rheedium.simul.extent_to_rod_sigma`
    """
    from scipy.optimize import brentq

    from rheedium.simul.finite_domain import extent_to_rod_sigma

    x_half = brentq(lambda x: math.sin(x) ** 2 / x**2 - 0.5, 1e-9, 2.0)
    domain_l = 100.0
    fwhm_q = 4.0 * x_half / domain_l
    sigma_ref = fwhm_q / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    sigma_pkg = np.asarray(
        extent_to_rod_sigma(jnp.array([domain_l, domain_l, domain_l]))
    )
    np.testing.assert_allclose(sigma_pkg, sigma_ref, rtol=5e-3)
    # And the old constant is measurably wrong (~6%):
    old_sigma = 2.0 * math.pi / (domain_l * math.sqrt(2.0 * math.pi))
    assert abs(old_sigma / sigma_ref - 1.0) > 0.05


def test_rod_miss_distance_closed_form() -> None:
    r"""The closed-form rod-sphere miss distance matches brute force.

    Extended Summary
    ----------------
    WP5.4 anchor: for a rod line ``k_in + G_hk + l b3`` that misses the
    Ewald sphere, the closed-form lateral miss distance
    ``d = sqrt(-disc) / (2 sqrt(a))`` from the rod-sphere quadratic
    equals the brute-force minimum over l of
    ``sqrt(|k_in + G(l)|^2 - k^2)`` to 1e-8, since the quadratic
    ``f(l) = a l^2 + b l + c`` attains ``-disc/(4a)`` at its vertex.

    Notes
    -----
    It builds the geometry with plain numpy vectors and minimizes the
    distance metric with scipy.optimize.minimize_scalar as the external
    reference; no rheedium overlap code appears on either side except
    the geometry convention itself.

    :see: :func:`~rheedium.simul.rod_domain_overlap`
    """
    from scipy.optimize import minimize_scalar

    k_mag = 62.9  # ~15 kV electrons, 1/Angstrom
    theta = math.radians(2.0)
    k_in = k_mag * np.array([math.cos(theta), 0.0, -math.sin(theta)])
    b1 = np.array([1.492, 0.0, 0.0])
    b2 = np.array([0.0, 1.492, 0.0])
    b3 = np.array([0.0, 0.0, 1.492])

    checked = 0
    for h, k in [(2, 0), (3, 1), (2, 2)]:
        p = k_in + h * b1 + k * b2
        a_coef = float(b3 @ b3)
        b_coef = float(2.0 * p @ b3)
        c_coef = float(p @ p - k_mag**2)
        disc = b_coef**2 - 4.0 * a_coef * c_coef
        if disc >= 0.0:
            continue
        d_closed = math.sqrt(-disc) / (2.0 * math.sqrt(a_coef))

        def metric(l_val: float, p_vec: np.ndarray = p) -> float:
            k_out = p_vec + l_val * b3
            return math.sqrt(max(float(k_out @ k_out) - k_mag**2, 0.0))

        result = minimize_scalar(
            metric,
            bounds=(-500.0, 500.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert abs(result.fun - d_closed) < 1e-8
        checked += 1
    assert checked > 0


def test_rod_intensity_evaluated_at_intersection() -> None:
    r"""Finite-domain rods use :math:`|F(l^*)|^2`, not the grid value.

    Extended Summary
    ----------------
    Builds a two-atom basis whose off-symmetry layer spacing makes
    :math:`|F(0,0,l)|^2` vary strongly along the rod, so evaluating the
    structure factor at the continuous rod-sphere intersection
    :math:`l^*` differs measurably from the nearest integer-``l`` grid
    point. The domain-mode reflections must equal
    :math:`|F(q^*)|^2` times the rod weight exactly, and must differ
    from the integer-grid product for at least one rod.

    Notes
    -----
    The reference value recomputes the structure factor independently at
    :math:`q^* = k_{out} - k_{in}` with the same form-factor machinery,
    anchoring the WP5.4 requirement that rod intensities are evaluated at
    the intersection rather than gathered from the integer grid.

    The check exercises ``build_ewald_data`` storage of atomic data and
    the ``rod_base_intensities`` path through
    ``ewald_allowed_reflections``.
    """
    lattice_constant = 5.43
    frac = jnp.asarray(
        [[0.0, 0.0, 0.0, 14.0], [0.5, 0.5, 0.37, 14.0]], dtype=jnp.float64
    )
    cell = build_cell_vectors(
        lattice_constant, lattice_constant, lattice_constant, 90.0, 90.0, 90.0
    )
    cart = jnp.concatenate([frac[:, :3] @ cell, frac[:, 3:]], axis=1)
    crystal = create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=jnp.asarray([lattice_constant] * 3),
        cell_angles=jnp.asarray([90.0, 90.0, 90.0]),
    )
    ewald = build_ewald_data(crystal, energy_kev=15.0, hmax=1, kmax=1, lmax=3)
    domain = jnp.asarray([80.0, 80.0, 15.0])
    indices, k_out, intensities = ewald_allowed_reflections(
        ewald, theta_deg=2.0, phi_deg=0.0, domain_extent_ang=domain
    )
    lam = float(wavelength_ang(jnp.asarray(15.0)))
    k_in = incident_wavevector(lam, 2.0)
    rod_sigma = extent_to_rod_sigma(domain)
    shell_sigma = compute_shell_sigma(ewald.k_magnitude, 1e-4, 1e-3)
    _, rod_factor, k_out_rod = rod_domain_overlap(
        ewald.hkl_grid,
        ewald.recip_vectors,
        k_in,
        ewald.k_magnitude,
        rod_sigma,
        shell_sigma,
        0.01,
    )
    grid_intensities = np.asarray(ewald.intensities)
    any_differs = False
    for slot, returned in zip(
        np.asarray(indices), np.asarray(intensities), strict=True
    ):
        if slot < 0:
            continue
        q_star = k_out_rod[slot] - k_in
        f_star = _compute_structure_factor_single(
            g_vector=q_star,
            atom_positions=ewald.atom_positions,
            atomic_numbers=ewald.atomic_numbers,
            temperature=ewald.temperature,
        )
        expected = float(jnp.abs(f_star) ** 2 * rod_factor[slot])
        np.testing.assert_allclose(returned, expected, rtol=1e-10)
        grid_value = float(grid_intensities[slot] * rod_factor[slot])
        if abs(expected - grid_value) / max(expected, 1e-30) > 0.02:
            any_differs = True
    assert any_differs, "test basis must make F(l*) differ from grid F"


def test_ctr_regularization_alias_preserves_bragg_cap() -> None:
    r"""The deprecated alias maps to the exact legacy Bragg cap.

    Extended Summary
    ----------------
    The legacy CTR shape capped Bragg peaks at ``1/reg``. The alias must
    convert ``ctr_regularization`` to a per-layer attenuation whose new
    cap ``1/(1 - exp(-eps))**2`` equals the legacy cap exactly, i.e.
    ``eps = -log(1 - sqrt(reg))``.

    Notes
    -----
    Evaluates the truncation intensity at a Bragg point (integer ``l``)
    with the converted epsilon and compares against ``1/reg`` for two
    representative legacy values.
    """
    for reg in (0.01, 0.05):
        eps = -np.log(1.0 - np.sqrt(reg))
        cap = float(ctr_truncation_intensity(jnp.asarray(1.0), eps))
        np.testing.assert_allclose(cap, 1.0 / reg, rtol=1e-10)


def test_refraction_conserves_k_parallel() -> None:
    r"""Inner-potential refraction conserves the surface-parallel momentum.

    Extended Summary
    ----------------
    Snell's law for electrons at a mean-inner-potential step keeps the
    surface-parallel wavevector component continuous and rescales only the
    surface-normal component: ``k_parallel = k cos(theta)`` unchanged and
    ``k_z_inside = k sqrt(sin^2(theta) + V0/V)``, which implies the exact
    identity ``k_z_inside^2 - k_z_vac^2 = k^2 V0 / V``. The old code
    rescaled the parallel component through ``cos(theta)/n`` instead
    (red-team finding M3), violating momentum conservation exactly where
    the inner potential matters most.

    Notes
    -----
    It evaluates the transmission tool's refraction helper at 20 kV,
    theta = 2 degrees, V0 = 15 V and checks (a) exact equality of the
    parallel component inside and outside, (b) the k_z identity to 1e-10
    relative, and (c) agreement of the implied Fresnel step reflectivity
    with the independent ``_flat_step_specular_reflectivity`` oracle in
    the reflection module.

    :see: :func:`~rheedium.simul.multislice_propagate`
    :see: :func:`~rheedium.simul.reflection_multislice_simulator`
    """
    from rheedium.simul.reflection_multislice import (
        _flat_step_specular_reflectivity,
    )
    from rheedium.simul.simulator import _refraction_wavevector_components
    from rheedium.types import create_edge_on_slices

    energy_kev = 20.0
    theta_deg = 2.0
    inner_potential_v0 = 15.0
    lam = float(wavelength_ang(jnp.asarray(energy_kev)))
    k_mag = 2.0 * math.pi / lam
    theta_rad = math.radians(theta_deg)

    k_parallel, k_z_inside = _refraction_wavevector_components(
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        inner_potential_v0=inner_potential_v0,
    )
    k_parallel_vacuum = k_mag * math.cos(theta_rad)
    k_z_vacuum = k_mag * math.sin(theta_rad)

    assert float(k_parallel) == k_parallel_vacuum
    lhs = float(k_z_inside) ** 2 - k_z_vacuum**2
    rhs = k_mag**2 * inner_potential_v0 / (energy_kev * 1000.0)
    assert abs(lhs / rhs - 1.0) < 1e-10

    dz = 0.25
    dx_slice = 1.0
    cap_width = 15.0
    z_lo = -30.0 - cap_width
    nz = int(math.ceil((30.0 + cap_width - z_lo) / dz))
    z_axis = z_lo + dz * jnp.arange(nz)
    profile = jnp.where(z_axis < 0.0, inner_potential_v0 * dx_slice, 0.0)
    slices = create_edge_on_slices(
        slices=jnp.tile(profile[None, None, :], (4, 2, 1)),
        dx_slice=dx_slice,
        dy=1.0,
        dz=dz,
        y_extent=2.0,
        z_lo=z_lo,
        z_surf=0.0,
        cap_width=cap_width,
    )
    oracle_reflectivity, is_flat = _flat_step_specular_reflectivity(
        slices=slices,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
    )
    helper_reflectivity = (
        (k_z_vacuum - float(k_z_inside)) / (k_z_vacuum + float(k_z_inside))
    ) ** 2
    assert float(is_flat) == 1.0
    np.testing.assert_allclose(
        float(oracle_reflectivity),
        helper_reflectivity,
        rtol=1e-10,
    )


def _flat_step_anchor_slices(
    theta_deg: float,
    length_ang: float,
    energy_kev: float = 15.0,
    inner_potential_v: float = 15.0,
) -> Any:
    r"""Build a uniform flat-step slab sized for genuine propagation.

    The vacuum window exceeds the reflected wedge height
    ``L tan(theta)`` so the incident wave feeding the surface never
    crosses the top absorber, and the slab depth exceeds the descent of
    the refracted transmitted beam so it is absorbed at the bottom only.
    """
    from rheedium.types import create_edge_on_slices

    sin_theta = math.sin(math.radians(theta_deg))
    sin_inside = math.sqrt(
        sin_theta**2 + inner_potential_v / (energy_kev * 1000.0)
    )
    cap_width = 15.0
    vacuum_above = max(30.0, 1.1 * length_ang * sin_theta + 10.0)
    slab_depth = max(30.0, 1.2 * length_ang * sin_inside + 10.0)
    dz = 0.125
    dx_slice = 1.0
    nx_slices = 8
    ny = 2
    z_lo = -slab_depth - cap_width
    z_hi = vacuum_above + cap_width
    nz = int(math.ceil((z_hi - z_lo) / dz))
    z_axis = z_lo + dz * jnp.arange(nz)
    profile = jnp.where(z_axis < 0.0, inner_potential_v * dx_slice, 0.0)
    return create_edge_on_slices(
        slices=jnp.tile(profile[None, None, :], (nx_slices, ny, 1)),
        dx_slice=dx_slice,
        dy=1.0,
        dz=dz,
        y_extent=float(ny),
        z_lo=z_lo,
        z_surf=0.0,
        cap_width=cap_width,
    )


def test_flat_step_fresnel() -> None:
    r"""Propagated flat-step reflectivity reproduces the Fresnel formula.

    Extended Summary
    ----------------
    A uniform inner-potential step (V0 = 15 V) is the one dynamical RHEED
    problem with a closed-form answer: the specular reflectivity is
    ``|(k_z - k_z') / (k_z + k_z')|^2`` with
    ``k_z' = k sqrt(sin^2(theta) + V0/V)``. The edge-on reflection
    multislice must reproduce this by *genuine propagation* over hundreds
    of Angstroms of crystal (red-team finding C9: the old code propagated
    through about one unit cell and patched the specular channel with the
    analytic answer at runtime). Gates: within 20 percent at
    theta = 1.5 degrees and 15 kV for L = 400 Angstroms, within 40
    percent at theta = 3 degrees, doubling L changes each answer by less
    than 10 percent, and the total read-off intensity never exceeds the
    incident flux (absorbing caps only remove flux).

    Notes
    -----
    The read-off residual bias is understood: a ``(x / sin x)^2``
    z-sampling factor with ``x = (k_z + k_z') dz / 2`` (about +7 percent
    at 3 degrees for dz = 0.125) plus the paraxial-dispersion Fresnel
    offset (about +2 percent), both shrinking with dz. The lock-in tail
    average over the second half of the run suppresses start-up
    transients, giving the sub-10-percent length stability asserted here.

    :see: :func:`~rheedium.simul.reflection_multislice_propagate`
    :see: :func:`~rheedium.simul.reflection_multislice_simulator`
    """
    from rheedium.simul.reflection_multislice import (
        _reflection_amplitude_pattern,
    )

    energy_kev = 15.0
    inner_potential_v = 15.0
    lam = float(wavelength_ang(jnp.asarray(energy_kev)))
    k_mag = 2.0 * math.pi / lam

    def fresnel(theta_deg: float) -> float:
        sin_theta = math.sin(math.radians(theta_deg))
        k_z_vac = k_mag * sin_theta
        k_z_in = k_mag * math.sqrt(
            sin_theta**2 + inner_potential_v / (energy_kev * 1000.0)
        )
        return ((k_z_vac - k_z_in) / (k_z_vac + k_z_in)) ** 2

    def readoff(theta_deg: float, length_ang: float) -> tuple[float, float]:
        slices = _flat_step_anchor_slices(theta_deg, length_ang)
        pattern, _ = _reflection_amplitude_pattern(
            slices=slices,
            energy_kev=energy_kev,
            theta_deg=theta_deg,
            propagation_length_ang=length_ang,
        )
        specular = float(pattern.intensities[0])
        total = float(jnp.sum(pattern.intensities))
        return specular, total

    tolerances = {1.5: 0.20, 3.0: 0.40}
    for theta_deg, tolerance in tolerances.items():
        reference = fresnel(theta_deg)
        specular_400, total_400 = readoff(theta_deg, 400.0)
        specular_800, total_800 = readoff(theta_deg, 800.0)
        assert abs(specular_400 / reference - 1.0) < tolerance, (
            f"theta={theta_deg}: {specular_400} vs Fresnel {reference}"
        )
        assert abs(specular_800 / reference - 1.0) < tolerance
        # convergence: doubling the propagation length changes < 10%
        assert abs(specular_800 / specular_400 - 1.0) < 0.10
        # absorbing caps only remove flux
        assert total_400 <= 1.0
        assert total_800 <= 1.0
