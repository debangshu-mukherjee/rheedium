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
    """Lobato coefficients obey the exact Bethe asymptote sum rule."""
    params = np.asarray(lobato_potentials())
    a = params[:, 0::2]
    b = params[:, 1::2]
    z = np.arange(1, 104, dtype=np.float64)
    expected = 0.0957336 * z
    np.testing.assert_array_less(0.0, b)
    np.testing.assert_allclose(np.sum(a / b, axis=1), expected, rtol=1e-3)


def test_lobato_vs_kirkland_f0() -> None:
    """Lobato and Kirkland f(0) agree at table-load time."""
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
    """Independent Kirkland/Lobato parameterizations of f_e agree."""
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
    """Kirkland's corrected argument follows the Mott-Bethe asymptote."""
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
    """Each projected potential matches the Hankel transform of its f(g)."""
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
    """Untaken small-x branches do not NaN-poison large-x gradients."""
    grad_val = jax.grad(lambda x: kernel(jnp.asarray(x, dtype=jnp.float64)))(
        800.0
    )
    assert math.isfinite(float(grad_val))
    x = np.asarray([1e-3, 1.9, 2.1, 100.0])
    got = np.asarray(kernel(jnp.asarray(x, dtype=jnp.float64)))
    np.testing.assert_allclose(got, scipy_ref(x), rtol=1e-6, atol=1e-10)


def test_debye_msd_full_model_limits() -> None:
    """The full Debye MSD has the correct high-T and zero-point limits."""
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
    """Primitive ASE cells canonicalize positions without losing phases."""
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
    """SrTiO3 slabs tile all three lattice directions at bulk density."""
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
