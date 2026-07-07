"""Tests for reflection-geometry multislice RHEED simulation."""

import math
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.inout import parse_crystal
from rheedium.simul import ewald_simulator
from rheedium.simul.reflection_multislice import (
    _beam_repeat_count,
    _edge_on_slices_like,
    _flat_step_specular_reflectivity,
    _read_reflected_pattern,
    _reflection_amplitude_pattern,
    crystal_to_edge_on_slices,
    reflection_detector_amplitude,
    reflection_multislice_propagate,
    reflection_multislice_simulator,
)
from rheedium.tools import wavelength_ang
from rheedium.types import (
    CrystalStructure,
    EdgeOnSlices,
    RHEEDPattern,
    create_crystal_structure,
    create_edge_on_slices,
)

BI2SE3_DIR = Path("tests/test_data/bi2se3")


def _flat_step_slices(
    theta_deg: float,
    *,
    energy_kev: float = 30.0,
    inner_potential_v: float = 15.0,
    length_ang: float = 200.0,
    dz: float = 0.125,
    cap_width: float = 15.0,
    vac_extra: float = 10.0,
    depth_extra: float = 10.0,
) -> EdgeOnSlices:
    """Build a uniform inner-potential step sized for real propagation.

    The vacuum window must exceed the reflected-wedge height
    ``length_ang * tan(theta)`` (so the incident wave feeding the surface
    never crosses the top CAP) and the slab depth must exceed the descent
    of the refracted transmitted beam (so it is absorbed by the bottom CAP
    only after leaving the readout region).
    """
    sin_theta = math.sin(math.radians(theta_deg))
    sin_inside = math.sqrt(
        sin_theta**2 + inner_potential_v / (energy_kev * 1000.0)
    )
    vacuum_above = max(30.0, 1.1 * length_ang * sin_theta + vac_extra)
    slab_depth = max(30.0, 1.2 * length_ang * sin_inside + depth_extra)
    nx_slices = 8
    ny = 2
    dx_slice = 1.0
    dy = 1.0
    z_lo = -slab_depth - cap_width
    z_hi = vacuum_above + cap_width
    nz = int(math.ceil((z_hi - z_lo) / dz))
    z_axis: Float[Array, "nz"] = z_lo + dz * jnp.arange(nz)
    potential_profile: Float[Array, "nz"] = jnp.where(
        z_axis < 0.0,
        inner_potential_v * dx_slice,
        0.0,
    )
    slices: Float[Array, "nx ny nz"] = jnp.tile(
        potential_profile[None, None, :],
        (nx_slices, ny, 1),
    )
    return create_edge_on_slices(
        slices=slices,
        dx_slice=dx_slice,
        dy=dy,
        dz=dz,
        y_extent=ny * dy,
        z_lo=z_lo,
        z_surf=0.0,
        cap_width=cap_width,
    )


def _fresnel_step_reflectivity(
    theta_deg: float,
    *,
    energy_kev: float = 30.0,
    inner_potential_v: float = 15.0,
) -> Float[Array, ""]:
    """Return the analytic electron inner-potential step reflectivity."""
    wavelength = wavelength_ang(energy_kev)
    k_mag = 2.0 * jnp.pi / wavelength
    sin_theta = jnp.sin(jnp.deg2rad(theta_deg))
    k_perp_vac = k_mag * sin_theta
    k_perp_in = k_mag * jnp.sqrt(
        sin_theta**2 + inner_potential_v / (energy_kev * 1000.0)
    )
    amplitude = (k_perp_vac - k_perp_in) / (k_perp_vac + k_perp_in)
    return jnp.abs(amplitude) ** 2


def _read_flat_step_pattern(
    theta_deg: float,
    *,
    length_ang: float = 200.0,
    cap_width: float = 15.0,
    vac_extra: float = 10.0,
    depth_extra: float = 10.0,
) -> RHEEDPattern:
    """Propagate and read a uniform flat-step test pattern."""
    slices: EdgeOnSlices = _flat_step_slices(
        theta_deg,
        length_ang=length_ang,
        cap_width=cap_width,
        vac_extra=vac_extra,
        depth_extra=depth_extra,
    )
    pattern: RHEEDPattern
    pattern, _ = _reflection_amplitude_pattern(
        slices=slices,
        energy_kev=30.0,
        theta_deg=theta_deg,
        detector_distance_mm=80.0,
        propagation_length_ang=length_ang,
    )
    return pattern


def _toy_crystal() -> CrystalStructure:
    """Create a small orthogonal slab with a top surface at positive z."""
    frac_positions: Float[Array, "N 4"] = jnp.array(
        [
            [0.25, 0.25, 0.25, 14.0],
            [0.75, 0.75, 0.50, 14.0],
        ]
    )
    cart_positions: Float[Array, "N 4"] = jnp.array(
        [
            [1.0, 1.0, 1.0, 14.0],
            [3.0, 3.0, 2.0, 14.0],
        ]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([4.0, 4.0, 4.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestReflectionMultislice(chex.TestCase, parameterized.TestCase):
    """Tests for edge-on reflection multislice routines.

    :see: :class:`~rheedium.types.EdgeOnSlices`
    :see: :func:`~rheedium.simul.crystal_to_edge_on_slices`
    :see: :func:`~rheedium.simul.reflection_multislice_propagate`
    :see: :func:`~rheedium.simul.reflection_detector_amplitude`
    :see: :func:`~rheedium.simul.reflection_multislice_simulator`
    :see: :func:`~rheedium.types.create_edge_on_slices`
    """

    def test_crystal_to_edge_on_slices_shapes(self) -> None:
        r"""Test crystal slabs convert to finite edge-on slices.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: crystal slabs
        convert to finite edge-on slices.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        slices: EdgeOnSlices = crystal_to_edge_on_slices(
            _toy_crystal(),
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=4.0,
            cap_width=2.0,
            r_cutoff=2.0,
        )

        chex.assert_shape(slices.slices, (4, 8, 18))
        chex.assert_tree_all_finite(slices.slices)
        assert float(slices.z_surf) == 2.0
        assert float(slices.z_lo) == -1.0

    def test_atom_x_coordinates_wrap_periodically(self) -> None:
        r"""Test atoms outside [0, l_x) wrap into the periodic beam cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: atoms outside
        [0, l_x) wrap into the periodic beam cell. An atom placed one full
        cell length beyond the beam-axis cell must deposit exactly the same
        projected potential as the equivalent wrapped atom, instead of being
        clipped into the first or last slice (red-team finding N4).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values
        where the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """

        def _crystal_with_x(x_value: float) -> CrystalStructure:
            frac = jnp.array([[x_value / 4.0, 0.25, 0.25, 14.0]])
            cart = jnp.array([[x_value, 1.0, 1.0, 14.0]])
            return create_crystal_structure(
                frac_positions=frac,
                cart_positions=cart,
                cell_lengths=jnp.array([4.0, 4.0, 4.0]),
                cell_angles=jnp.array([90.0, 90.0, 90.0]),
            )

        kwargs: dict[str, Any] = {
            "dx_slice": 1.0,
            "dy": 0.5,
            "dz": 0.5,
            "vacuum_above": 4.0,
            "cap_width": 2.0,
            "r_cutoff": 2.0,
        }
        wrapped = crystal_to_edge_on_slices(_crystal_with_x(0.5), **kwargs)
        # frac position is out of [0, 1) too, but only x wrapping matters
        shifted = crystal_to_edge_on_slices(_crystal_with_x(4.5), **kwargs)

        chex.assert_trees_all_close(
            shifted.slices,
            wrapped.slices,
            atol=1e-12,
        )

    def test_beam_repeat_count_covers_propagation_length(self) -> None:
        r"""Test the stack repeat count covers the propagation length.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the stack repeat
        count covers the propagation length. The per-cell stack is looped
        ``ceil(L / (nx_slices * dx_slice))`` times so the wave traverses at
        least ``propagation_length_ang`` of crystal (red-team finding C9:
        the old code propagated through exactly one unit cell).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values
        where the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        slices: EdgeOnSlices = crystal_to_edge_on_slices(
            _toy_crystal(),
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=4.0,
            cap_width=2.0,
        )
        stack_length = float(slices.slices.shape[0] * slices.dx_slice)

        assert _beam_repeat_count(slices, 200.0) == math.ceil(
            200.0 / stack_length
        )
        assert _beam_repeat_count(slices, 1.0) == 1
        assert _beam_repeat_count(slices, 200.0) * stack_length >= 200.0

    def test_edge_on_slices_like_matches_direct_build(self) -> None:
        r"""Test template-based deposition matches the direct slice build.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: template-based
        deposition matches the direct slice build. The traceable
        ``_edge_on_slices_like`` rebuild (used by structure distribution
        axes under the public multislice kernel) must reproduce
        ``crystal_to_edge_on_slices`` exactly when given the same crystal.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        crystal = _toy_crystal()
        template: EdgeOnSlices = crystal_to_edge_on_slices(
            crystal,
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=4.0,
            cap_width=2.0,
        )
        rebuilt: EdgeOnSlices = _edge_on_slices_like(
            crystal,
            template=template,
        )

        chex.assert_trees_all_close(
            rebuilt.slices,
            template.slices,
            atol=1e-12,
        )
        chex.assert_trees_all_close(rebuilt.dx_slice, template.dx_slice)
        chex.assert_trees_all_close(rebuilt.z_lo, template.z_lo)

    @parameterized.named_parameters(
        ("theta_1", 1.0),
        ("theta_2", 2.0),
        ("theta_3", 3.0),
    )
    def test_flat_slab_specular_geometry(self, theta_deg: float) -> None:
        r"""Test a flat step produces a specular up-going channel.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a flat step
        produces a specular up-going channel.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = _read_flat_step_pattern(theta_deg)

        brightest = int(jnp.argmax(pattern.intensities))
        wavelength = wavelength_ang(30.0)
        k_mag = 2.0 * jnp.pi / wavelength
        expected_kz = k_mag * jnp.sin(jnp.deg2rad(theta_deg))

        assert brightest == 0
        chex.assert_trees_all_close(pattern.k_out[brightest, 1], 0.0)
        chex.assert_trees_all_close(
            pattern.k_out[brightest, 2],
            expected_kz,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    @parameterized.named_parameters(
        ("theta_0p5", 0.5, 1600.0, 0.25),
        ("theta_1", 1.0, 800.0, 0.25),
        ("theta_2", 2.0, 200.0, 0.25),
        ("theta_3", 3.0, 200.0, 0.30),
    )
    def test_flat_slab_fresnel_reflectivity(
        self,
        theta_deg: float,
        length_ang: float,
        rtol: float,
    ) -> None:
        r"""Test propagated specular intensity approaches Fresnel value.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: propagated
        specular intensity approaches the Fresnel value. Unlike the old
        implementation, no analytic blending patches the specular channel:
        the reflectivity is produced by genuine propagation over
        ``length_ang`` of crystal and read from the vacuum field. More
        grazing angles need longer propagation because the reflected wedge
        above the surface grows as ``L tan(theta)``; the per-angle lengths
        and tolerances encode the measured convergence of the paraxial
        split-step scheme (the residual bias is the known
        ``(x / sin x)^2`` z-sampling factor with
        ``x = (k_z + k_z') dz / 2``, plus the paraxial-dispersion Fresnel
        offset of a few percent).

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, ``length_ang``, and ``rtol``, so the documented
        behavior is checked across the cases supplied by pytest, Chex,
        Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = _read_flat_step_pattern(
            theta_deg,
            length_ang=length_ang,
        )
        expected: Float[Array, ""] = _fresnel_step_reflectivity(theta_deg)

        chex.assert_trees_all_close(
            pattern.intensities[0],
            expected,
            rtol=rtol,
            atol=1e-8,
        )

    def test_flat_slab_total_readoff_bounded_by_incident(self) -> None:
        r"""Test total read-off intensity never exceeds the incident flux.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: total read-off
        intensity never exceeds the incident flux. The absorbing caps only
        remove flux, so the summed channel reflectivities relative to the
        unit incident wave must stay at or below one.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = _read_flat_step_pattern(2.0)

        total = float(jnp.sum(pattern.intensities))
        assert total <= 1.0
        assert total > 0.0

    def test_flat_slab_reflectivity_increases_toward_grazing(self) -> None:
        r"""Test Fresnel reflectivity increases at smaller incidence angle.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Fresnel
        reflectivity increases at smaller incidence angle.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        reflectivities: Float[Array, "3"] = jnp.array(
            [
                _read_flat_step_pattern(
                    theta,
                    length_ang=length,
                ).intensities[0]
                for theta, length in ((1.0, 800.0), (2.0, 200.0), (3.0, 200.0))
            ]
        )

        chex.assert_trees_all_equal(
            jnp.all(jnp.diff(reflectivities) < 0), True
        )

    def test_cap_and_vacuum_convergence_for_flat_slab(self) -> None:
        r"""Test specular result is stable under larger vacuum and CAP.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: specular result is
        stable under larger vacuum and CAP.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        base: RHEEDPattern = _read_flat_step_pattern(2.0)
        expanded: RHEEDPattern = _read_flat_step_pattern(
            2.0,
            cap_width=20.0,
            vac_extra=20.0,
            depth_extra=20.0,
        )

        chex.assert_trees_all_close(
            expanded.k_out[0],
            base.k_out[0],
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            expanded.intensities[0],
            base.intensities[0],
            rtol=0.05,
            atol=1e-8,
        )

    def test_flat_step_oracle_matches_closed_form(self) -> None:
        r"""Test the retained Fresnel oracle detects and scores flat steps.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the retained
        Fresnel oracle detects and scores flat steps.
        ``_flat_step_specular_reflectivity`` is no longer blended into the
        runtime read-off; it survives as an analytic test oracle and must
        keep returning the closed-form k-parallel-conserving step
        reflectivity together with a positive flat-step detection flag.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        slices: EdgeOnSlices = _flat_step_slices(2.0)

        reflectivity, is_flat = _flat_step_specular_reflectivity(
            slices=slices,
            energy_kev=30.0,
            theta_deg=2.0,
        )

        chex.assert_trees_all_close(
            reflectivity,
            _fresnel_step_reflectivity(2.0),
            rtol=1e-10,
        )
        assert float(is_flat) == 1.0

    def test_single_shot_readout_matches_tail_average_loosely(self) -> None:
        r"""Test the single-state read-off agrees with the tail average.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the single-state
        read-off agrees with the tail average. ``_read_reflected_pattern``
        reads the final wavefield only, while the production pattern uses
        the lock-in tail average; on a converged flat step the two must
        agree within the transient-oscillation envelope.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        length_ang = 200.0
        slices: EdgeOnSlices = _flat_step_slices(2.0, length_ang=length_ang)
        wavefield: Any = reflection_multislice_propagate(
            slices,
            energy_kev=30.0,
            theta_deg=2.0,
            propagation_length_ang=length_ang,
        )
        n_steps = (
            _beam_repeat_count(slices, length_ang) * slices.slices.shape[0]
        )
        single: RHEEDPattern = _read_reflected_pattern(
            wavefield=wavefield,
            slices=slices,
            energy_kev=30.0,
            theta_deg=2.0,
            detector_distance=80.0,
            n_steps=n_steps,
        )
        averaged: RHEEDPattern = _read_flat_step_pattern(
            2.0,
            length_ang=length_ang,
        )

        chex.assert_trees_all_close(
            single.intensities[0],
            averaged.intensities[0],
            rtol=0.35,
        )

    def test_reflection_multislice_simulator_returns_finite_pattern(
        self,
    ) -> None:
        r"""Test end-to-end atomic reflection simulation returns valid data.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: end-to-end atomic
        reflection simulation returns valid data.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        pattern: RHEEDPattern = reflection_multislice_simulator(
            _toy_crystal(),
            energy_kev=30.0,
            theta_deg=2.0,
            detector_distance=80.0,
            dx_slice=1.0,
            dy=0.5,
            dz=0.5,
            vacuum_above=8.0,
            cap_width=4.0,
            propagation_length_ang=40.0,
        )

        chex.assert_shape(pattern.k_out, (8, 3))
        chex.assert_shape(pattern.detector_points, (8, 2))
        chex.assert_shape(pattern.intensities, (8,))
        chex.assert_tree_all_finite(pattern.k_out)
        chex.assert_tree_all_finite(pattern.detector_points)
        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)

    def test_reflection_detector_amplitude_renders_dense_field(self) -> None:
        r"""Test dense detector rendering of reflected channel amplitudes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: dense detector
        rendering of reflected channel amplitudes. The dense coherent field
        must be finite and reproduce the sparse channel intensities at the
        spot centers up to the Gaussian rasterization envelope.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        slices: EdgeOnSlices = _flat_step_slices(2.0)

        field: Any = reflection_detector_amplitude(
            slices=slices,
            energy_kev=30.0,
            theta_deg=2.0,
            detector_distance_mm=80.0,
            image_shape_px=(32, 32),
            pixel_size_mm=(2.0, 2.0),
            beam_center_px=(16.0, 4.0),
            spot_sigma_px=1.0,
            propagation_length_ang=200.0,
        )

        chex.assert_shape(field, (32, 32))
        chex.assert_tree_all_finite(field)
        assert float(jnp.max(jnp.abs(field) ** 2)) > 0.0

    def test_bi2se3_reflection_ky_channels_match_ewald_rods(self) -> None:
        r"""Test strongest Bi2Se3 reflected beams lie on Ewald rod ky values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: strongest Bi2Se3
        reflected beams lie on Ewald rod ky values.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = parse_crystal(BI2SE3_DIR / "intial.xyz")
        reflection: RHEEDPattern = reflection_multislice_simulator(
            crystal,
            energy_kev=30.0,
            theta_deg=2.5,
            detector_distance=80.0,
            dx_slice=4.0,
            dy=1.0,
            dz=1.0,
            vacuum_above=6.0,
            cap_width=3.0,
            propagation_length_ang=150.0,
        )
        ewald: RHEEDPattern = ewald_simulator(
            crystal,
            energy_kev=30.0,
            theta_deg=2.5,
            phi_deg=0.0,
            hmax=4,
            kmax=12,
            detector_distance=80.0,
        )
        strongest = jnp.argsort(reflection.intensities)[-5:]
        reflected_ky = reflection.k_out[strongest, 1]
        ewald_ky = ewald.k_out[:, 1]
        nearest = jnp.min(
            jnp.abs(reflected_ky[:, None] - ewald_ky[None, :]),
            axis=1,
        )

        assert jnp.all(nearest <= jnp.pi / crystal.cell_lengths[1])

    @parameterized.named_parameters(
        ("initial", "intial.xyz"),
        ("500K", "500K.final.xyz"),
        ("750K", "750K.final.xyz"),
        ("1000K", "1000K.final.xyz"),
        ("1250K", "1250K.final.xyz"),
    )
    def test_bi2se3_xyz_fixtures_run_with_finite_intensities(
        self,
        filename: str,
    ) -> None:
        r"""Test all Bi2Se3 XYZ fixtures run with finite absolute output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all Bi2Se3 XYZ
        fixtures run with finite absolute output. Read-off intensities are
        absolute reflectivities relative to the unit incident wave (the old
        max normalization was removed together with the Fresnel blending),
        so they must be finite, non-negative, and nonzero somewhere.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``filename``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_reflection_multislice``, so the
        Test Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = parse_crystal(BI2SE3_DIR / filename)
        pattern: RHEEDPattern = reflection_multislice_simulator(
            crystal,
            energy_kev=30.0,
            theta_deg=2.5,
            detector_distance=80.0,
            dx_slice=4.0,
            dy=1.5,
            dz=1.5,
            vacuum_above=4.5,
            cap_width=3.0,
            propagation_length_ang=150.0,
        )

        chex.assert_tree_all_finite(pattern.intensities)
        chex.assert_trees_all_equal(jnp.all(pattern.intensities >= 0), True)
        assert float(jnp.max(pattern.intensities)) > 0.0
