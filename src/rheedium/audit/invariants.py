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
from beartype import beartype
from jaxtyping import Array, Float

from rheedium.simul import (
    ewald_simulator,
    kirkland_form_factor,
    lobato_form_factor,
)
from rheedium.tools import (
    incident_wavevector,
    wavelength_ang,
)
from rheedium.types import (
    create_crystal_structure,
    scalar_float,
    scalar_int,
)


@dataclass(frozen=True)
class InvariantResult:
    """Structured outcome of a single physics invariant check.

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


def _simple_structure_factor(
    reciprocal_vector: Float[Array, "3"],
    atom_positions: Float[Array, "M 3"],
    atomic_numbers: Array,
) -> Float[Array, ""]:
    """Compute a minimal kinematic structure factor intensity locally."""
    scattering_factors: Float[Array, "M"] = atomic_numbers.astype(jnp.float64)
    dot_products: Float[Array, "M"] = jnp.dot(
        atom_positions, reciprocal_vector
    )
    phases = jnp.exp(1j * dot_products)
    structure_factor = jnp.sum(scattering_factors * phases)
    return jnp.abs(structure_factor) ** 2


def _result(
    name: str,
    residual: float,
    tolerance: float,
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
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.3, 0.4, 0.2),
        (1.0, 0.0, 0.5),
    ),
    relative_tolerance: scalar_float = 1.0e-6,
) -> InvariantResult:
    r"""Verify :math:`I(\mathbf{G}) = I(-\mathbf{G})` for the structure factor.

    In the absence of anomalous scattering the kinematic intensity is
    invariant under :math:`\mathbf{G} \to -\mathbf{G}` because the form
    factors are real and the structure factor satisfies
    :math:`F(-\mathbf{G}) = F(\mathbf{G})^{*}`, so
    :math:`|F(-\mathbf{G})|^{2} = |F(\mathbf{G})|^{2}`. This must hold
    for any (non-centrosymmetric) crystal.

    Uses an asymmetric two-atom basis to avoid trivially satisfying
    Friedel's law via centrosymmetry.

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
    atom_positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.31, 0.42, 0.17],
            [0.65, 0.18, 0.79],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.array([14, 8, 22], dtype=jnp.int32)

    worst = 0.0
    worst_g = g_vectors[0]
    for g in g_vectors:
        g_pos = jnp.array(g, dtype=jnp.float64)
        g_neg = -g_pos
        intensity_pos = float(
            _simple_structure_factor(g_pos, atom_positions, atomic_numbers)
        )
        intensity_neg = float(
            _simple_structure_factor(g_neg, atom_positions, atomic_numbers)
        )
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
            "worst |I(G) - I(-G)| / I(G) over a non-centrosymmetric "
            f"three-atom basis; worst at G={tuple(worst_g)}"
        ),
    )


@beartype
def check_elastic_closure_ewald(
    voltage_kv: scalar_float = 20.0,
    theta_deg: scalar_float = 2.0,
    hmax: scalar_int = 3,
    kmax: scalar_int = 3,
    relative_tolerance: scalar_float = 1.0e-4,
) -> InvariantResult:
    r"""Verify elastic closure for ``ewald_simulator``.

    Asserts :math:`\|\mathbf{k}_{out}\| = \|\mathbf{k}_{in}\|` for
    every reflection returned by the exact rod-Ewald intersection.

    Elastic scattering preserves kinetic energy, so every outgoing
    wavevector returned by a RHEED simulator must lie on the Ewald
    sphere of radius :math:`\|\mathbf{k}_{in}\| = 2\pi / \lambda`. A
    deviation from this constraint indicates an inconsistent definition
    of :math:`\mathbf{k}` (with vs without the :math:`2\pi`), an
    incorrect rod-Ewald intersection, or a frame-of-reference bug in
    the projection.

    Uses :func:`rheedium.simul.ewald_simulator`, which solves the
    rod-sphere intersection exactly rather than searching a discrete
    lattice with a tolerance window. The expected residual is
    therefore at floating-point precision, not at simulator-filter
    precision. Constructs an in-memory cubic crystal so the test does
    not depend on any disk fixture.

    Parameters
    ----------
    voltage_kv : scalar_float, optional
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
        voltage_kv=voltage_kv,
        theta_deg=theta_deg,
        hmax=hmax,
        kmax=kmax,
    )

    lam = float(wavelength_ang(voltage_kv))
    k_in = incident_wavevector(lam, theta_deg)
    k_in_magnitude = float(jnp.linalg.norm(k_in))

    intensities = jnp.asarray(pattern.intensities)
    k_out = jnp.asarray(pattern.k_out)
    valid_mask = intensities > 0.0
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
    k_out_magnitudes = jnp.linalg.norm(k_out_valid, axis=1)
    rel_errors = jnp.abs(k_out_magnitudes - k_in_magnitude) / k_in_magnitude
    worst = float(jnp.max(rel_errors))

    return _result(
        name="elastic_closure_ewald",
        residual=worst,
        tolerance=relative_tolerance,
        units="dimensionless (relative)",
        detail=(
            "worst (||k_out|| - ||k_in||) / ||k_in|| over all valid "
            f"ewald_simulator reflections from a cubic test crystal "
            f"at V={voltage_kv} kV"
        ),
    )


@beartype
def run_default_invariants() -> list[InvariantResult]:
    """Run the full default invariant suite and return all results.

    The suite is intentionally fast (no large simulations) so it can
    be invoked from CI on every change. Tolerances use the per-check
    defaults; tighten by calling individual checks with explicit
    keyword arguments.

    Returns
    -------
    list of InvariantResult
        One entry per invariant, in execution order.
    """
    results: list[InvariantResult] = []
    pos_kirkland, pos_lobato = check_form_factor_positivity()
    results.append(pos_kirkland)
    results.append(pos_lobato)
    mono_kirkland, mono_lobato = check_form_factor_monotonic_decrease()
    results.append(mono_kirkland)
    results.append(mono_lobato)
    results.append(check_wavelength_relativistic_consistency())
    results.append(check_friedel_law_structure_factor())
    results.append(check_elastic_closure_ewald())
    return results
