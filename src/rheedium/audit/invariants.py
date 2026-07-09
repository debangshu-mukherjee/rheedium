r"""Physics-based sanity checks for the rheedium simulation pipeline.

Extended Summary
----------------
This module verifies physical invariants that any correct RHEED
simulator must satisfy, independent of stored reference images. Each
invariant returns a structured :class:`InvariantResult` reporting the
numerical residual against an explicit tolerance and a one-line
explanation.

The intent is to catch the kind of bug that regression-style image
comparison cannot: sign errors, missing factors of :math:`2\pi`,
normalization mistakes, frame-of-reference confusion, or transcribed
parameterization tables. These bugs typically produce a result that
looks plausible but violates a physical law that has nothing to do with
any particular crystal or detector geometry.

Routine Listings
----------------
:class:`InvariantResult`
    Structured pass/fail container for one invariant check.
:func:`check_form_factor_positivity`
    Verify :math:`f(q) > 0` over the usable q range for both
    parameterizations.
:func:`check_form_factor_monotonic_decrease`
    Verify the electron form factor decreases monotonically with q.
:func:`check_form_factor_kirkland_lobato_close`
    Coarse sanity check that Kirkland and Lobato parameterizations are
    in the same ballpark over their common validity range.
:func:`check_wavelength_relativistic_consistency`
    Verify ``wavelength_ang`` matches an independent relativistic
    de Broglie evaluation.
:func:`check_friedel_law_structure_factor`
    Verify :math:`I(\mathbf{G}) = I(-\mathbf{G})` for the kinematic
    structure factor.
:func:`check_elastic_closure_ewald`
    Verify :math:`\|\mathbf{k}_{out}\| = \|\mathbf{k}_{in}\|` for
    every reflection emitted by ``ewald_simulator``.
:func:`run_default_invariants`
    Run the full default invariant suite and return all results.

Notes
-----
Every check in this module is **stateless** with respect to disk: no
fixtures, no stored references. Inputs are either pure physical
constants (atomic number, voltage) or minimal crystal stubs constructed
in-memory. This is what makes the suite suitable as a CI guardrail
against silent physics regressions.

The default tolerances are chosen to be loose enough that legitimate
floating-point and numerical-quadrature noise pass, but tight enough
that a real bug will produce a residual one or more orders of magnitude
above the threshold. They are exposed as keyword arguments so callers
can tighten them when running on a higher-precision build.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.special
from beartype import beartype
from jaxtyping import Array, Float

import rheedium.simul.ewald as ewald_module
from rheedium.inout import lobato_potentials
from rheedium.simul import (
    ewald_simulator,
    kirkland_form_factor,
    kirkland_projected_potential,
    lobato_form_factor,
    lobato_projected_potential,
)
from rheedium.tools import incident_wavevector, wavelength_ang
from rheedium.types import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    PLANCK_CONSTANT_JS,
    SPEED_OF_LIGHT_MS,
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
    scalar_int,
)
from rheedium.ucell.unitcell import bulk_to_slice


@dataclass(frozen=True)
class InvariantResult:
    """Structured outcome of a single physics invariant check.

    :see: :func:`~.test_invariant_result_is_immutable`

    Attributes
    ----------
    name : str
        Short identifier for the invariant, suitable for logs and
        aggregate reports.
    passed : bool
        ``True`` when ``residual <= tolerance``.
    residual : float
        Worst-case numerical deviation observed during the check, in
        the units given by ``units``.
    tolerance : float
        Threshold against which ``residual`` was compared.
    units : str
        Human-readable units for ``residual`` and ``tolerance``.
    detail : str
        One-line explanation of what the invariant tests and how the
        residual was measured.
    """

    name: str
    passed: bool
    residual: float
    tolerance: float
    units: str
    detail: str


def _result(
    name: str,
    residual: scalar_float,
    tolerance: scalar_float,
    units: str,
    detail: str,
) -> InvariantResult:
    """Build an :class:`InvariantResult` and decide pass/fail."""
    return InvariantResult(
        name=name,
        passed=bool(residual <= tolerance),
        residual=float(residual),
        tolerance=float(tolerance),
        units=units,
        detail=detail,
    )


def _direct_cell_vectors(
    lengths: Float[Array, "3"],
    angles_deg: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """Build direct cell vectors from lengths and angles."""
    a_len, b_len, c_len = [float(item) for item in lengths]
    alpha, beta, gamma = [math.radians(float(item)) for item in angles_deg]
    sin_gamma = math.sin(gamma)
    a_vec = np.array([a_len, 0.0, 0.0], dtype=np.float64)
    b_vec = np.array(
        [b_len * math.cos(gamma), b_len * sin_gamma, 0.0],
        dtype=np.float64,
    )
    c_x = c_len * math.cos(beta)
    c_y = (
        c_len
        * (math.cos(alpha) - math.cos(beta) * math.cos(gamma))
        / (sin_gamma)
    )
    c_z = math.sqrt(max(c_len**2 - c_x**2 - c_y**2, 0.0))
    c_vec = np.array([c_x, c_y, c_z], dtype=np.float64)
    return jnp.asarray(np.stack([a_vec, b_vec, c_vec]), dtype=jnp.float64)


def _reciprocal_from_direct(
    direct_vectors: Float[Array, "3 3"],
) -> Float[Array, "3 3"]:
    """Build reciprocal vectors as rows from a direct-cell matrix."""
    return 2.0 * jnp.pi * jnp.linalg.inv(direct_vectors).T


def _relativistic_wavelength_ang_from_constants(
    energy_kev: scalar_float,
) -> float:
    """Return relativistic electron wavelength from constants only."""
    voltage = float(energy_kev) * 1.0e3
    kinetic_energy_j = ELEMENTARY_CHARGE_C * voltage
    rest_energy_j = ELECTRON_MASS_KG * SPEED_OF_LIGHT_MS**2
    relativistic_factor = 1.0 + kinetic_energy_j / (2.0 * rest_energy_j)
    momentum = math.sqrt(
        2.0 * ELECTRON_MASS_KG * kinetic_energy_j * relativistic_factor
    )
    return PLANCK_CONSTANT_JS / momentum * 1.0e10


def _crystal_from_fractional(
    frac_xyz: Float[Array, "M 3"],
    atomic_numbers: Float[Array, "M"],
    lengths: Float[Array, "3"],
    angles_deg: Float[Array, "3"],
) -> CrystalStructure:
    """Create a crystal from fractional coordinates and atomic numbers."""
    direct_vectors = _direct_cell_vectors(lengths, angles_deg)
    cart_xyz = frac_xyz @ direct_vectors
    frac_positions = jnp.concatenate(
        [frac_xyz, atomic_numbers[:, None]],
        axis=1,
    )
    cart_positions = jnp.concatenate(
        [cart_xyz, atomic_numbers[:, None]],
        axis=1,
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=lengths,
        cell_angles=angles_deg,
    )


def _gaas_zincblende_crystal() -> CrystalStructure:
    """Return a fixed non-centrosymmetric conventional GaAs cell."""
    frac_xyz = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.asarray([31.0] * 4 + [33.0] * 4, dtype=jnp.float64)
    lengths = jnp.asarray([5.6533, 5.6533, 5.6533], dtype=jnp.float64)
    angles = jnp.asarray([90.0, 90.0, 90.0], dtype=jnp.float64)
    return _crystal_from_fractional(frac_xyz, atomic_numbers, lengths, angles)


@beartype
def check_form_factor_positivity(
    atomic_numbers: Sequence[scalar_int] = (1, 6, 14, 29, 47, 79),
    q_min: scalar_float = 0.1,
    q_max: scalar_float = 6.0,
    n_samples: scalar_int = 64,
) -> tuple[InvariantResult, InvariantResult]:
    r"""Verify :math:`f(q) > 0` for both parameterizations.

    The electron scattering factor in the first Born approximation is
    a positive real number for any neutral atom over its physically
    meaningful q range. A negative value indicates a sign error in the
    parameterization, an unintended subtraction, or evaluation outside
    the validity range. This check is the cheapest possible smoke test
    against gross corruption of either form factor table.

    :see: :func:`~.test_form_factor_positivity_both_parameterizations`

    Parameters
    ----------
    atomic_numbers : sequence of int, optional
        Atomic numbers to probe. Defaults to a span from H to Au.
    q_min, q_max : float, optional
        Inclusive endpoints of the q range in 1/Å.
    n_samples : int, optional
        Number of linearly spaced q points sampled.

    Returns
    -------
    kirkland_result, lobato_result : InvariantResult
        Pass/fail per parameterization. Residual is the most negative
        :math:`f(q)` observed (or 0 if all values are positive).
    """
    q_grid: Float[Array, " n"] = jnp.linspace(q_min, q_max, n_samples)

    def _most_negative(
        form_factor_fn: Callable[
            [int, Float[Array, " n"]], Float[Array, " n"]
        ],
    ) -> float:
        worst = 0.0
        for z in atomic_numbers:
            f_values = form_factor_fn(int(z), q_grid)
            min_value = float(jnp.min(f_values))
            worst = min(worst, min_value)
        return -worst

    kirkland_residual = _most_negative(kirkland_form_factor)
    lobato_residual = _most_negative(lobato_form_factor)

    detail = (
        f"most-negative f(q) over q in [{q_min}, {q_max}] 1/Å for "
        f"Z in {tuple(atomic_numbers)}; expects f(q) > 0"
    )
    kirkland_result = _result(
        name="form_factor_positivity_kirkland",
        residual=kirkland_residual,
        tolerance=0.0,
        units="form factor units",
        detail=detail,
    )
    lobato_result = _result(
        name="form_factor_positivity_lobato",
        residual=lobato_residual,
        tolerance=0.0,
        units="form factor units",
        detail=detail,
    )
    return kirkland_result, lobato_result


@beartype
def check_form_factor_monotonic_decrease(
    atomic_numbers: Sequence[scalar_int] = (1, 6, 14, 29, 47, 79),
    q_min: scalar_float = 0.2,
    q_max: scalar_float = 5.0,
    n_samples: scalar_int = 64,
) -> tuple[InvariantResult, InvariantResult]:
    r"""Verify the electron form factor decreases monotonically with q.

    The electron scattering factor falls off with momentum transfer
    because higher-q components of the atomic charge density are
    progressively out of phase. Any local maximum away from the
    high-q tail is unphysical and signals a parameterization error
    (negative coefficient, swapped table row, or arithmetic bug).

    :see: :func:`~.test_form_factor_monotonic_decrease_both_parameterizations`

    Parameters
    ----------
    atomic_numbers : sequence of int, optional
        Atomic numbers to probe.
    q_min, q_max : float, optional
        Endpoints of the q range in 1/Å where monotonicity should
        hold. Avoids the very-small-q region where parameterizations
        diverge by design.
    n_samples : int, optional
        Number of linearly spaced q points sampled.

    Returns
    -------
    kirkland_result, lobato_result : InvariantResult
        Pass/fail per parameterization. Residual is the largest
        positive :math:`f(q_{i+1}) - f(q_i)` observed (0 if strictly
        monotonic).
    """
    q_grid: Float[Array, " n"] = jnp.linspace(q_min, q_max, n_samples)

    def _worst_increase(
        form_factor_fn: Callable[
            [int, Float[Array, " n"]], Float[Array, " n"]
        ],
    ) -> float:
        worst = 0.0
        for z in atomic_numbers:
            f_values = form_factor_fn(int(z), q_grid)
            diffs = jnp.diff(f_values)
            largest_increase = float(jnp.max(diffs))
            worst = max(worst, largest_increase)
        return worst

    kirkland_residual = _worst_increase(kirkland_form_factor)
    lobato_residual = _worst_increase(lobato_form_factor)

    detail = (
        f"largest positive Δf between adjacent q samples on q in "
        f"[{q_min}, {q_max}] 1/Å; expects strict monotonic decrease"
    )
    kirkland_result = _result(
        name="form_factor_monotonic_kirkland",
        residual=kirkland_residual,
        tolerance=0.0,
        units="form factor units",
        detail=detail,
    )
    lobato_result = _result(
        name="form_factor_monotonic_lobato",
        residual=lobato_residual,
        tolerance=0.0,
        units="form factor units",
        detail=detail,
    )
    return kirkland_result, lobato_result


@beartype
def check_form_factor_kirkland_lobato_close(
    atomic_numbers: Sequence[scalar_int] = (6, 14, 29, 47),
    q_min: scalar_float = 0.5,
    q_max: scalar_float = 3.0,
    n_samples: scalar_int = 32,
    relative_tolerance: scalar_float = 0.30,
) -> InvariantResult:
    r"""Coarse sanity check that Kirkland and Lobato are in the same ballpark.

    Kirkland and Lobato are independent fits with different design
    constraints (Kirkland uses sums of Gaussians and Lorentzians;
    Lobato is constructed to satisfy the Bethe high-q asymptotic
    exactly). They are *not* expected to agree exactly, even in their
    common range. They should, however, be close enough that a
    transcription error, off-by-one in atomic number, or factor-of-10
    bug in either table would show up clearly.

    The default tolerance of 30% is intentionally loose. The point is
    not to assert equality but to catch gross corruption.

    Parameters
    ----------
    atomic_numbers : sequence of int, optional
        Elements to compare. Defaults to a span from C to Ag.
    q_min, q_max : float, optional
        Endpoints of the q range in 1/Å. Defaults to the central
        region where both parameterizations are most reliable.
    n_samples : int, optional
        Number of log-spaced q points sampled in the range.
    relative_tolerance : float, optional
        Maximum allowed
        :math:`|f_{K} - f_{L}| / \max(|f_{K}|, |f_{L}|)` over the
        sampled grid for any element. Default 30% reflects "close,
        not equal".

    Returns
    -------
    InvariantResult
        Worst-case relative disagreement across all probed atoms and q.
    """
    q_grid: Float[Array, " n"] = jnp.logspace(
        math.log10(q_min), math.log10(q_max), n_samples
    )

    worst = 0.0
    worst_z = atomic_numbers[0]
    for z in atomic_numbers:
        f_kirkland = kirkland_form_factor(int(z), q_grid)
        f_lobato = lobato_form_factor(int(z), q_grid)
        denom = jnp.maximum(jnp.abs(f_kirkland), jnp.abs(f_lobato))
        rel_error = jnp.max(jnp.abs(f_kirkland - f_lobato) / denom)
        rel_error_value = float(rel_error)
        if rel_error_value > worst:
            worst = rel_error_value
            worst_z = z

    return _result(
        name="form_factor_kirkland_lobato_close",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            f"worst-case relative disagreement on q in "
            f"[{q_min}, {q_max}] 1/Å, dominated by Z={worst_z}; "
            "loose ballpark check, not strict equality"
        ),
    )


@beartype
def check_wavelength_relativistic_consistency(
    voltages_kv: Sequence[scalar_float] = (
        10.0,
        20.0,
        50.0,
        100.0,
        300.0,
    ),
    relative_tolerance: scalar_float = 1.0e-4,
) -> InvariantResult:
    r"""Verify ``wavelength_ang`` matches an independent de Broglie eval.

    Computes the relativistic electron wavelength

    .. math::

        \lambda = \frac{h}
                  {\sqrt{2 m_e e V \left(1 + \frac{eV}{2 m_e c^2}\right)}}

    using independent CODATA constants and compares against
    :func:`rheedium.tools.wavelength_ang`. A mismatch points to a wrong
    constant, a unit confusion, or a missing relativistic correction.

    :see: :func:`~.test_wavelength_relativistic_consistency`

    Parameters
    ----------
    voltages_kv : sequence of float, optional
        Beam voltages in kV at which to evaluate.
    relative_tolerance : float, optional
        Maximum allowed
        :math:`|\lambda_{ref} - \lambda_{rheedium}| / \lambda_{ref}`.

    Returns
    -------
    InvariantResult
        Worst-case relative wavelength error across the voltage grid.
    """
    planck_constant_js = 6.62607015e-34
    electron_mass_kg = 9.1093837015e-31
    elementary_charge_c = 1.602176634e-19
    speed_of_light_m_s = 299792458.0

    worst = 0.0
    worst_voltage = voltages_kv[0]
    for voltage in voltages_kv:
        v_volts = float(voltage) * 1.0e3
        kinetic_energy_j = elementary_charge_c * v_volts
        rest_energy_j = electron_mass_kg * speed_of_light_m_s**2
        relativistic_factor = 1.0 + kinetic_energy_j / (2.0 * rest_energy_j)
        denominator = math.sqrt(
            2.0 * electron_mass_kg * kinetic_energy_j * relativistic_factor
        )
        lambda_meters = planck_constant_js / denominator
        lambda_ang_reference = lambda_meters * 1.0e10
        lambda_ang_rheedium = float(wavelength_ang(voltage))
        rel_error = (
            abs(lambda_ang_reference - lambda_ang_rheedium)
            / lambda_ang_reference
        )
        if rel_error > worst:
            worst = rel_error
            worst_voltage = voltage

    return _result(
        name="wavelength_relativistic_consistency",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst relative error vs CODATA-derived de Broglie "
            f"formula at V={worst_voltage} kV"
        ),
    )


@beartype
def check_friedel_law_structure_factor(
    g_vectors: Sequence[Sequence[scalar_float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 2.0, 1.0),
        (2.0, 0.0, 1.0),
    ),
    relative_tolerance: scalar_float = 1.0e-10,
) -> InvariantResult:
    r"""Verify :math:`I(\mathbf{G}) = I(-\mathbf{G})` for the structure factor.

    In the absence of anomalous scattering the kinematic intensity is
    invariant under :math:`\mathbf{G} \to -\mathbf{G}` because the form
    factors are real and the structure factor satisfies
    :math:`F(-\mathbf{G}) = F(\mathbf{G})^{*}`, so
    :math:`|F(-\mathbf{G})|^{2} = |F(\mathbf{G})|^{2}`. This must hold
    for any (non-centrosymmetric) crystal.

    Uses a fixed non-centrosymmetric GaAs zincblende cell and the production
    structure-factor path, including real form factors and Debye-Waller
    damping, so the check exercises the same code used by simulation.

    :see: :func:`~.test_friedel_law_structure_factor`

    Parameters
    ----------
    g_vectors : sequence of 3-sequence of float, optional
        Reciprocal lattice vectors to test. Each is paired with its
        negation.
    relative_tolerance : float, optional
        Maximum allowed
        :math:`|I(\mathbf{G}) - I(-\mathbf{G})| / I(\mathbf{G})` over
        all tested vectors.

    Returns
    -------
    InvariantResult
        Worst-case relative asymmetry between paired reflections.
    """
    crystal = _gaas_zincblende_crystal()
    reciprocal_basis = _reciprocal_from_direct(
        _direct_cell_vectors(crystal.cell_lengths, crystal.cell_angles)
    )
    atom_positions = crystal.cart_positions[:, :3]
    atomic_numbers = crystal.cart_positions[:, 3].astype(jnp.int32)

    worst = 0.0
    worst_g = g_vectors[0]
    for g in g_vectors:
        hkl = jnp.array(g, dtype=jnp.float64)
        g_pos = hkl @ reciprocal_basis
        g_neg = -g_pos
        structure_factor_pos = ewald_module._compute_structure_factor_single(
            g_pos,
            atom_positions,
            atomic_numbers,
            300.0,
            "kirkland",
        )
        structure_factor_neg = ewald_module._compute_structure_factor_single(
            g_neg,
            atom_positions,
            atomic_numbers,
            300.0,
            "kirkland",
        )
        intensity_pos = float(jnp.abs(structure_factor_pos) ** 2)
        intensity_neg = float(jnp.abs(structure_factor_neg) ** 2)
        denom = max(abs(intensity_pos), 1.0e-30)
        rel_error = abs(intensity_pos - intensity_neg) / denom
        if rel_error > worst:
            worst = rel_error
            worst_g = g

    return _result(
        name="friedel_law_structure_factor",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst |I(G) - I(-G)| / I(G) through production structure "
            "factors for non-centrosymmetric GaAs; worst at "
            f"hkl={tuple(worst_g)}"
        ),
    )


@beartype
def check_elastic_closure_ewald(
    energy_kev: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    relative_tolerance: scalar_float = 1.0e-4,
) -> InvariantResult:
    r"""Verify the rod-Ewald quadratic closure used by ``ewald_simulator``.

    This check is scoped to the algebra of the exact rod-sphere quadratic:
    each returned outgoing vector must remain on the elastic sphere, and its
    momentum transfer must match the returned rod index when recomputed from
    an independent reciprocal basis. It is not a full detector-calibration or
    intensity invariant.

    Uses :func:`rheedium.simul.ewald_simulator`, which solves the
    rod-sphere intersection exactly rather than searching a discrete
    lattice with a tolerance window. The expected residual is
    therefore at floating-point precision, not at simulator-filter
    precision. Constructs an in-memory cubic crystal so the test does
    not depend on any disk fixture.

    :see: :func:`~.test_elastic_closure_ewald_simulator`

    Parameters
    ----------
    energy_kev : scalar_float, optional
        Beam voltage in kV.
    theta_deg : scalar_float, optional
        Grazing angle of incidence in degrees.
    hmax, kmax : scalar_int, optional
        Reciprocal-lattice search half-extents passed to the
        simulator.
    relative_tolerance : scalar_float, optional
        Maximum allowed
        :math:`(\|\mathbf{k}_{out}\| - \|\mathbf{k}_{in}\|) /
        \|\mathbf{k}_{in}\|` for any returned reflection.

    Returns
    -------
    InvariantResult
        Worst-case relative violation of elastic closure across all
        reflections in the returned pattern.
    """
    a = 4.0
    cell_lengths = jnp.array([a, a, a], dtype=jnp.float64)
    cell_angles = jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64)
    frac_positions = jnp.array(
        [
            [0.0, 0.0, 0.0, 14.0],
            [0.5, 0.5, 0.0, 14.0],
            [0.5, 0.0, 0.5, 14.0],
            [0.0, 0.5, 0.5, 14.0],
        ],
        dtype=jnp.float64,
    )
    cart_positions = frac_positions.at[:, :3].set(frac_positions[:, :3] * a)
    crystal = create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )

    pattern = ewald_simulator(
        crystal=crystal,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        hmax=hmax,
        kmax=kmax,
    )

    wavelength_reference = _relativistic_wavelength_ang_from_constants(
        energy_kev
    )
    k_magnitude_reference = 2.0 * math.pi / wavelength_reference
    k_in = incident_wavevector(
        wavelength_ang(energy_kev),
        theta_deg,
        0.0,
    )
    k_in_magnitude = float(jnp.linalg.norm(k_in))
    k_in_reference_error = (
        abs(k_in_magnitude - k_magnitude_reference) / k_magnitude_reference
    )

    intensities = jnp.asarray(pattern.intensities)
    g_indices = jnp.asarray(pattern.G_indices)
    k_out = jnp.asarray(pattern.k_out)
    valid_mask = (intensities > 0.0) & (g_indices >= 0)
    if not bool(jnp.any(valid_mask)):
        return _result(
            name="elastic_closure_ewald",
            residual=float("inf"),
            tolerance=relative_tolerance,
            units="dimensionless (relative)",
            detail=(
                "ewald_simulator returned no valid reflections; "
                "cannot evaluate elastic closure"
            ),
        )
    k_out_valid = k_out[valid_mask]
    g_indices_valid = g_indices[valid_mask]
    k_out_magnitudes = jnp.linalg.norm(k_out_valid, axis=1)
    rel_errors = jnp.abs(k_out_magnitudes - k_in_magnitude) / k_in_magnitude
    elastic_error = float(jnp.max(rel_errors))

    direct_vectors = _direct_cell_vectors(cell_lengths, cell_angles)
    reciprocal_basis = _reciprocal_from_direct(direct_vectors)
    inverse_reciprocal = jnp.linalg.inv(reciprocal_basis)
    q_vectors: Float[Array, "N 3"] = k_out_valid - k_in
    hkl_coefficients: Float[Array, "N 3"] = q_vectors @ inverse_reciprocal
    n_k: int = 2 * int(kmax) + 1
    h_values = (g_indices_valid // n_k - int(hmax)).astype(jnp.float64)
    k_values = (g_indices_valid % n_k - int(kmax)).astype(jnp.float64)
    index_error = float(
        jnp.max(
            jnp.maximum(
                jnp.abs(hkl_coefficients[:, 0] - h_values),
                jnp.abs(hkl_coefficients[:, 1] - k_values),
            )
        )
    )
    rebuilt_q: Float[Array, "N 3"] = (
        h_values[:, None] * reciprocal_basis[0]
        + k_values[:, None] * reciprocal_basis[1]
        + hkl_coefficients[:, 2:3] * reciprocal_basis[2]
    )
    q_error = float(
        jnp.max(
            jnp.linalg.norm(rebuilt_q - q_vectors, axis=1)
            / jnp.maximum(jnp.linalg.norm(q_vectors, axis=1), 1.0)
        )
    )
    worst = max(elastic_error, k_in_reference_error, index_error, q_error)

    return _result(
        name="elastic_closure_ewald",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst of elastic-sphere error, constants-derived |k_in| error, "
            "and independent reciprocal-basis reconstruction error over "
            f"valid ewald_simulator reflections at V={energy_kev} kV"
        ),
    )


@beartype
def check_lobato_bethe_sum_rule(
    relative_tolerance: scalar_float = 1.0e-3,
) -> InvariantResult:
    r"""Verify the loaded Lobato table satisfies the Bethe sum rule."""
    params = np.asarray(lobato_potentials(), dtype=np.float64)
    amplitudes = params[:, 0::2]
    scales = params[:, 1::2]
    atomic_numbers = np.arange(1, params.shape[0] + 1, dtype=np.float64)
    expected = 0.0957336 * atomic_numbers
    actual = np.sum(amplitudes / scales, axis=1)
    relative_errors = np.abs(actual - expected) / expected
    worst = float(np.max(relative_errors))
    worst_z = int(np.argmax(relative_errors) + 1)
    return _result(
        name="lobato_bethe_sum_rule",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst relative error in sum(a_i / b_i) = 0.0957336 Z "
            f"over the loaded Lobato table, dominated by Z={worst_z}"
        ),
    )


@beartype
def check_projected_potential_hankel_pair(
    relative_tolerance: scalar_float = 5.0e-2,
) -> InvariantResult:
    r"""Verify one Hankel-pair point for each form-factor parameterization."""
    g_grid = np.linspace(0.0, 80.0, 50_000)
    q_grid = jnp.asarray(2.0 * np.pi * g_grid, dtype=jnp.float64)
    prefactor = 47.87801 * 2.0 * np.pi
    checks: tuple[
        tuple[
            str,
            Callable[[int, Float[Array, " n"]], Float[Array, " n"]],
            Callable[[int, Float[Array, " r"]], Float[Array, " r"]],
        ],
        ...,
    ] = (
        ("kirkland", kirkland_form_factor, kirkland_projected_potential),
        ("lobato", lobato_form_factor, lobato_projected_potential),
    )
    worst = 0.0
    worst_name = checks[0][0]
    z_value = 14
    r_ang = 0.5
    for name, form_factor_fn, potential_fn in checks:
        form_factor = np.asarray(form_factor_fn(z_value, q_grid))
        integrand = (
            form_factor
            * scipy.special.j0(2.0 * np.pi * g_grid * r_ang)
            * g_grid
        )
        reference = prefactor * np.trapezoid(integrand, g_grid)
        observed = float(potential_fn(z_value, jnp.asarray([r_ang]))[0])
        relative_error = abs(observed / reference - 1.0)
        if relative_error > worst:
            worst = relative_error
            worst_name = name
    return _result(
        name="projected_potential_hankel_pair",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst Hankel-pair relative error at Z=14, r=0.5 Å, "
            f"dominated by {worst_name}"
        ),
    )


@beartype
def check_srtio3_001_slab_density(
    relative_tolerance: scalar_float = 5.0e-2,
) -> InvariantResult:
    r"""Verify ``bulk_to_slice`` preserves SrTiO3(001) bulk density."""
    lattice_constant = 3.905
    frac_xyz = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.asarray([38.0, 22.0, 8.0, 8.0, 8.0])
    lengths = jnp.asarray([lattice_constant] * 3, dtype=jnp.float64)
    angles = jnp.asarray([90.0, 90.0, 90.0], dtype=jnp.float64)
    crystal = _crystal_from_fractional(
        frac_xyz,
        atomic_numbers,
        lengths,
        angles,
    )
    depth = 20.0
    x_extent = 40.0
    y_extent = 40.0
    slab = bulk_to_slice(
        crystal,
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        depth,
        x_extent,
        y_extent,
    )
    bulk_density = 5.0 / lattice_constant**3
    expected_atoms = bulk_density * depth * x_extent * y_extent
    actual_atoms = float(slab.cart_positions.shape[0])
    residual = abs(actual_atoms / expected_atoms - 1.0)
    return _result(
        name="srtio3_001_slab_density",
        residual=residual,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "relative atom-count density error for an in-memory SrTiO3(001) "
            "bulk_to_slice slab"
        ),
    )


@beartype
def check_triclinic_frame_contract(
    tolerance: scalar_float = 1.0e-10,
) -> InvariantResult:
    r"""Validate fractional/cartesian and reciprocal contracts in triclinic."""
    lengths = jnp.asarray([3.2, 4.1, 5.3], dtype=jnp.float64)
    angles = jnp.asarray([73.0, 82.0, 111.0], dtype=jnp.float64)
    frac_xyz = jnp.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.6, 0.4, 0.8],
            [0.25, 0.75, 0.5],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.asarray([14.0, 8.0, 22.0], dtype=jnp.float64)
    crystal = _crystal_from_fractional(
        frac_xyz,
        atomic_numbers,
        lengths,
        angles,
    )
    direct_vectors = _direct_cell_vectors(lengths, angles)
    recovered_frac = crystal.cart_positions[:, :3] @ jnp.linalg.inv(
        direct_vectors
    )
    reciprocal_basis = _reciprocal_from_direct(direct_vectors)
    reciprocal_contract = direct_vectors @ reciprocal_basis.T / (2.0 * jnp.pi)
    residual = float(
        jnp.maximum(
            jnp.max(jnp.abs(recovered_frac - frac_xyz)),
            jnp.max(
                jnp.abs(
                    reciprocal_contract
                    - jnp.eye(3, dtype=reciprocal_contract.dtype)
                )
            ),
        )
    )
    return _result(
        name="triclinic_frame_contract",
        residual=residual,
        tolerance=tolerance,
        units="absolute",
        detail=(
            "max absolute fractional-coordinate and A·Bᵀ/(2π) contract "
            "error for an in-memory triclinic crystal"
        ),
    )


@beartype
def check_refraction_k_parallel_conservation(
    relative_tolerance: scalar_float = 1.0e-10,
) -> InvariantResult:
    r"""Verify inner-potential refraction conserves surface-parallel k."""
    from rheedium.simul.simulator import _refraction_wavevector_components

    energy_kev = 20.0
    theta_deg = 2.0
    inner_potential_v0 = 15.0
    wavelength = float(wavelength_ang(energy_kev))
    k_magnitude = 2.0 * math.pi / wavelength
    theta_rad = math.radians(theta_deg)
    k_parallel_expected = k_magnitude * math.cos(theta_rad)
    k_z_vacuum = k_magnitude * math.sin(theta_rad)
    k_parallel, k_z_inside = _refraction_wavevector_components(
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        inner_potential_v0=inner_potential_v0,
    )
    parallel_error = abs(float(k_parallel) / k_parallel_expected - 1.0)
    lhs = float(k_z_inside) ** 2 - k_z_vacuum**2
    rhs = k_magnitude**2 * inner_potential_v0 / (energy_kev * 1000.0)
    normal_error = abs(lhs / rhs - 1.0)
    residual = max(parallel_error, normal_error)
    return _result(
        name="refraction_k_parallel_conservation",
        residual=residual,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "max relative error in k_parallel continuity and the k_z "
            "inner-potential identity at 20 kV"
        ),
    )


@beartype
def run_default_invariants() -> list[InvariantResult]:
    """Run the full default invariant suite and return all results.

    The suite is intentionally fast (no large simulations) so it can
    be invoked from CI on every change. Tolerances use the per-check
    defaults; tighten by calling individual checks with explicit
    keyword arguments.

    :see: :func:`~.test_run_default_invariants_returns_full_suite`

    Returns
    -------
    list of InvariantResult
        One entry per invariant, in execution order.
    """
    results: list[InvariantResult] = []
    pos_kirkland: InvariantResult
    pos_lobato: InvariantResult
    pos_kirkland, pos_lobato = check_form_factor_positivity()
    results.append(pos_kirkland)
    results.append(pos_lobato)
    mono_kirkland: InvariantResult
    mono_lobato: InvariantResult
    mono_kirkland, mono_lobato = check_form_factor_monotonic_decrease()
    results.append(mono_kirkland)
    results.append(mono_lobato)
    results.append(check_wavelength_relativistic_consistency())
    results.append(check_friedel_law_structure_factor())
    results.append(check_elastic_closure_ewald())
    results.append(check_lobato_bethe_sum_rule())
    results.append(check_projected_potential_hankel_pair())
    results.append(check_srtio3_001_slab_density())
    results.append(check_triclinic_frame_contract())
    results.append(check_refraction_k_parallel_conservation())
    return results
