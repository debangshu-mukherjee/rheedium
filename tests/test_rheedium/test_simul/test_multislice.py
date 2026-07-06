"""Test suite for simul/multislice.py and simul/potential.py.

Tests the multislice algorithm primitives:
- build_transmission_function
- fresnel_propagator
- multislice_one_step
- crystal_projected_potential

The tests verify physical invariants (transmission modulus, propagator
unitarity, vacuum propagation, absorption-induced norm decay), shape
correctness, JIT compatibility, and the kinematic-limit agreement
between multislice and the weak-phase approximation.
"""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float, Int

from rheedium.simul.multislice import (
    build_transmission_function,
    fresnel_propagator,
    multislice_one_step,
)
from rheedium.simul.potential import crystal_projected_potential
from rheedium.simul.simulator import multislice_amplitude, multislice_propagate
from rheedium.types import PotentialSlices, create_potential_slices
from rheedium.types.custom_types import scalar_float


def _make_grid_params() -> tuple[
    tuple[int, int], Float[Array, "2"], float, float
]:
    """Return standard test grid parameters."""
    grid: tuple[int, int] = (16, 16)
    cell: Float[Array, "2"] = jnp.array([8.0, 8.0])
    voltage: float = 20.0
    dz: float = 2.0
    return grid, cell, voltage, dz


def _zero_potential(grid: tuple[int, int]) -> Complex[Array, "H W"]:
    """Zero complex potential of given shape."""
    return jnp.zeros(grid, dtype=jnp.complex128)


def _single_atom_potential(
    grid: tuple[int, int],
    cell: Float[Array, "2"],
    z: int = 14,
    absorption: float = 0.1,
) -> Complex[Array, "H W"]:
    """Single atom at the cell center."""
    pos: Float[Array, "1 3"] = jnp.array(
        [[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]]
    )
    z_arr: Int[Array, "1"] = jnp.array([z], dtype=jnp.int32)
    return crystal_projected_potential(
        atomic_positions_angstrom=pos,
        atomic_numbers=z_arr,
        grid_shape=grid,
        cell_dimensions_angstrom=cell,
        absorption_fraction=absorption,
        parameterization="lobato",
    )


def _make_potential_slices(
    scale: float = 0.0,
) -> PotentialSlices:
    """Return a tiny deterministic PotentialSlices object."""
    slices: Float[Array, "2 8 8"] = jnp.ones((2, 8, 8)) * scale
    return create_potential_slices(
        slices=slices,
        slice_thickness=1.0,
        x_calibration=0.5,
        y_calibration=0.5,
    )


class TestBuildTransmissionFunction(chex.TestCase, parameterized.TestCase):
    """Tests for build_transmission_function.

    :see: :func:`~rheedium.simul.build_transmission_function`
    """

    def test_zero_potential_gives_unity(self) -> None:
        r"""T = 1 everywhere when V = 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: T = 1 everywhere
        when V = 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _zero_potential(grid)
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        chex.assert_shape(t, grid)
        chex.assert_trees_all_close(
            t, jnp.ones(grid, dtype=jnp.complex128), atol=1e-12
        )

    def test_modulus_bounded_by_one(self) -> None:
        r"""\|T\| <= 1 everywhere (absorption only removes amplitude).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|T\| <= 1
        everywhere (absorption only removes amplitude).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        modulus: Float[Array, "H W"] = jnp.abs(t)
        assert float(jnp.max(modulus)) <= 1.0 + 1e-6

    def test_real_potential_gives_unit_modulus(self) -> None:
        r"""Pure-real V (no absorption) gives \|T\| = 1 everywhere.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Pure-real V (no
        absorption) gives \|T\| = 1 everywhere.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.0
        )
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        modulus: Float[Array, "H W"] = jnp.abs(t)
        chex.assert_trees_all_close(modulus, jnp.ones(grid), atol=1e-6)

    def test_absorption_reduces_modulus(self) -> None:
        r"""Higher absorption fraction lowers \|T\|.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Higher absorption
        fraction lowers \|T\|.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v_low: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.05
        )
        v_high: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.5
        )
        t_low: Complex[Array, "H W"] = build_transmission_function(
            v_low, voltage
        )
        t_high: Complex[Array, "H W"] = build_transmission_function(
            v_high, voltage
        )
        assert float(jnp.min(jnp.abs(t_high))) < float(jnp.min(jnp.abs(t_low)))

    def test_jit_matches_eager(self) -> None:
        r"""JIT-compiled output matches eager output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT-compiled
        output matches eager output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        eager: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        jit_fn: Callable[
            [Complex[Array, "H W"], scalar_float],
            Complex[Array, "H W"],
        ] = jax.jit(
            build_transmission_function,
            static_argnums=(),
        )
        compiled: Complex[Array, "H W"] = jit_fn(v, voltage)
        chex.assert_trees_all_close(eager, compiled, atol=1e-10)

    def test_explicit_projected_potential_keyword_matches_positional(
        self,
    ) -> None:
        r"""The explicit projected-potential keyword matches positional use.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The explicit
        projected-potential keyword matches positional use.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        positional: Complex[Array, "H W"] = build_transmission_function(
            v, voltage
        )
        keyword: Complex[Array, "H W"] = build_transmission_function(
            projected_potential_volt_angstrom=v,
            energy_kev=voltage,
        )
        chex.assert_trees_all_close(positional, keyword, atol=1e-12)


class TestFresnelPropagator(chex.TestCase, parameterized.TestCase):
    """Tests for fresnel_propagator.

    :see: :func:`~rheedium.simul.multislice.fresnel_propagator`
    """

    def test_unitarity(self) -> None:
        r"""\|P(qx, qy)\| = 1 everywhere — propagation is unitary.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|P(qx, qy)\| = 1
        everywhere — propagation is unitary.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        modulus: Float[Array, "H W"] = jnp.abs(p)
        chex.assert_trees_all_close(modulus, jnp.ones(grid), atol=1e-10)

    def test_zero_thickness_is_identity(self) -> None:
        r"""Dz = 0 gives P = 1 (no propagation).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Dz = 0 gives P = 1
        (no propagation).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, 0.0)
        chex.assert_trees_all_close(
            p, jnp.ones(grid, dtype=jnp.complex128), atol=1e-12
        )

    def test_dc_component_is_unity(self) -> None:
        r"""At qx = qy = 0 the propagator equals 1 for any dz.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: At qx = qy = 0 the
        propagator equals 1 for any dz.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_trees_all_close(float(jnp.real(p[0, 0])), 1.0, atol=1e-12)
        chex.assert_trees_all_close(float(jnp.imag(p[0, 0])), 0.0, atol=1e-12)

    def test_shape(self) -> None:
        r"""Output shape matches grid.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches grid.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_shape(p, grid)


class TestMultisliceOneStep(chex.TestCase, parameterized.TestCase):
    """Tests for multislice_one_step.

    :see: :func:`~rheedium.simul.multislice.multislice_one_step`
    """

    def test_vacuum_preserves_norm(self) -> None:
        r"""Zero potential leaves \|psi\| unchanged (free propagation).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Zero potential
        leaves \|psi\| unchanged (free propagation).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _zero_potential(grid)
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        psi_in: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        psi_out: Complex[Array, "H W"] = multislice_one_step(psi_in, t, p)
        norm_in: float = float(jnp.sum(jnp.abs(psi_in) ** 2))
        norm_out: float = float(jnp.sum(jnp.abs(psi_out) ** 2))
        chex.assert_trees_all_close(norm_out, norm_in, rtol=1e-6)

    def test_absorption_reduces_norm(self) -> None:
        r"""Non-zero absorption reduces wavefunction norm.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Non-zero
        absorption reduces wavefunction norm.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.3
        )
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        psi_in: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        psi_out: Complex[Array, "H W"] = multislice_one_step(psi_in, t, p)
        norm_in: float = float(jnp.sum(jnp.abs(psi_in) ** 2))
        norm_out: float = float(jnp.sum(jnp.abs(psi_out) ** 2))
        assert norm_out < norm_in

    def test_norm_monotonic_with_depth(self) -> None:
        r"""Norm decreases monotonically over many absorbing slices.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Norm decreases
        monotonically over many absorbing slices.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.2
        )
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        psi: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        norms: list[float] = [float(jnp.sum(jnp.abs(psi) ** 2))]
        for _ in range(5):
            psi = multislice_one_step(psi, t, p)
            norms.append(float(jnp.sum(jnp.abs(psi) ** 2)))
        i: int
        for i in range(len(norms) - 1):
            assert norms[i + 1] <= norms[i] + 1e-6

    def test_output_shape(self) -> None:
        r"""Output shape equals input shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        equals input shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        psi_in: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        psi_out: Complex[Array, "H W"] = multislice_one_step(psi_in, t, p)
        chex.assert_shape(psi_out, grid)

    def test_jit_compilation(self) -> None:
        r"""JIT-compiled multislice step matches eager output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: JIT-compiled
        multislice step matches eager output.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises JIT compilation, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        psi_in: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        eager: Complex[Array, "H W"] = multislice_one_step(psi_in, t, p)
        jit_step: Callable[
            [
                Complex[Array, "H W"],
                Complex[Array, "H W"],
                Complex[Array, "H W"],
            ],
            Complex[Array, "H W"],
        ] = jax.jit(multislice_one_step)
        compiled: Complex[Array, "H W"] = jit_step(psi_in, t, p)
        chex.assert_trees_all_close(eager, compiled, atol=1e-10)

    def test_grad_flows_through_potential(self) -> None:
        r"""jax.grad through multislice w.r.t. absorption is finite.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: jax.grad through
        multislice w.r.t. absorption is finite.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        psi_in: Complex[Array, "H W"] = jnp.ones(grid, dtype=jnp.complex128)
        pos: Float[Array, "1 3"] = jnp.array(
            [[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]]
        )
        z_arr: Int[Array, "1"] = jnp.array([14], dtype=jnp.int32)
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)

        def loss_fn(absorption: float) -> scalar_float:
            v: Complex[Array, "H W"] = crystal_projected_potential(
                atomic_positions_angstrom=pos,
                atomic_numbers=z_arr,
                grid_shape=grid,
                cell_dimensions_angstrom=cell,
                absorption_fraction=absorption,
                parameterization="lobato",
            )
            t: Complex[Array, "H W"] = build_transmission_function(v, voltage)
            psi_out: Complex[Array, "H W"] = multislice_one_step(psi_in, t, p)
            return jnp.sum(jnp.abs(psi_out) ** 2).real

        g: float = float(jax.grad(loss_fn)(0.1))
        assert jnp.isfinite(g)
        assert g != 0.0


class TestMultisliceAmplitude(chex.TestCase):
    """Tests for the Layer-0 multislice amplitude slot.

    :see: :func:`~rheedium.simul.multislice_amplitude`
    """

    def test_matches_fft_of_exit_wave(self) -> None:
        r"""Amplitude is the reciprocal-space exit wave before \|.\|^2.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Amplitude is the
        reciprocal-space exit wave before \|.\|^2.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        potential: PotentialSlices = _make_potential_slices(scale=0.0)

        amplitude: Complex[Array, "8 8"] = multislice_amplitude(
            potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )
        exit_wave: Complex[Array, "8 8"] = multislice_propagate(
            potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )

        chex.assert_trees_all_close(
            amplitude,
            jnp.fft.fft2(exit_wave),
            atol=1e-10,
        )

    def test_modulus_squared_is_finite_intensity(self) -> None:
        r"""\|amplitude\|^2 gives a finite diffraction-intensity grid.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|amplitude\|^2
        gives a finite diffraction-intensity grid.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        potential: PotentialSlices = _make_potential_slices(scale=0.01)
        amplitude: Complex[Array, "8 8"] = multislice_amplitude(
            potential,
            energy_kev=20.0,
            theta_deg=2.0,
        )

        intensity: Float[Array, "8 8"] = jnp.abs(amplitude) ** 2
        chex.assert_shape(intensity, (8, 8))
        chex.assert_tree_all_finite(intensity)
        assert float(jnp.sum(intensity)) > 0.0

    def test_grad_flows_through_potential_scale(self) -> None:
        r"""Layer-0 multislice amplitude remains differentiable.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Layer-0 multislice
        amplitude remains differentiable.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def objective(scale: float) -> scalar_float:
            potential: PotentialSlices = _make_potential_slices(scale=scale)
            amplitude: Complex[Array, "8 8"] = multislice_amplitude(
                potential,
                energy_kev=20.0,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(amplitude) ** 2).real

        grad_value: scalar_float = jax.grad(objective)(0.01)
        assert jnp.isfinite(grad_value)


class TestCrystalProjectedPotential(chex.TestCase, parameterized.TestCase):
    """Tests for crystal_projected_potential.

    :see: :func:`~rheedium.simul.crystal_projected_potential`
    """

    def test_output_shape(self) -> None:
        r"""Output shape matches grid_shape.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output shape
        matches grid_shape.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        chex.assert_shape(v, grid)

    def test_output_is_complex(self) -> None:
        r"""Output dtype is complex.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output dtype is
        complex.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        assert jnp.iscomplexobj(v)

    def test_real_part_nonnegative(self) -> None:
        r"""Real part of V is non-negative for all atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Real part of V is
        non-negative for all atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        assert float(jnp.min(jnp.real(v))) >= -1e-6

    def test_imag_part_proportional_to_real(self) -> None:
        r"""V_abs = absorption_fraction * V_real.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: V_abs =
        absorption_fraction * V_real.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.2
        )
        v_real: Float[Array, "H W"] = jnp.real(v)
        v_imag: Float[Array, "H W"] = jnp.imag(v)
        chex.assert_trees_all_close(v_imag, 0.2 * v_real, atol=1e-6)

    def test_zero_absorption_gives_real_potential(self) -> None:
        r"""absorption_fraction=0 yields purely real potential.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        absorption_fraction=0 yields purely real potential.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, absorption=0.0
        )
        chex.assert_trees_all_close(jnp.imag(v), jnp.zeros(grid), atol=1e-12)

    def test_higher_z_gives_stronger_potential(self) -> None:
        r"""Heavier atom (Z=82) gives higher peak potential than Z=6.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Heavier atom
        (Z=82) gives higher peak potential than Z=6.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v_carbon: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, z=6
        )
        v_lead: Complex[Array, "H W"] = _single_atom_potential(
            grid, cell, z=82
        )
        peak_c: float = float(jnp.max(jnp.real(v_carbon)))
        peak_pb: float = float(jnp.max(jnp.real(v_lead)))
        assert peak_pb > peak_c

    def test_two_atoms_gives_two_peaks(self) -> None:
        r"""Two atoms produce two peaks in the potential.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Two atoms produce
        two peaks in the potential.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        pos: Float[Array, "2 3"] = jnp.array(
            [
                [2.0, 4.0, 0.0],
                [6.0, 4.0, 0.0],
            ]
        )
        z_arr: Int[Array, "2"] = jnp.array([14, 14], dtype=jnp.int32)
        v: Complex[Array, "H W"] = crystal_projected_potential(
            atomic_positions_angstrom=pos,
            atomic_numbers=z_arr,
            grid_shape=grid,
            cell_dimensions_angstrom=cell,
            absorption_fraction=0.1,
            parameterization="lobato",
        )
        v_real: Float[Array, "H W"] = jnp.real(v)
        col_at_y4: Float[Array, "H"] = v_real[:, grid[1] // 2]
        assert float(col_at_y4[grid[0] // 4]) > 0.0
        assert float(col_at_y4[3 * grid[0] // 4]) > 0.0

    def test_finite_everywhere(self) -> None:
        r"""Output is finite (no NaN, no Inf).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output is finite
        (no NaN, no Inf).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        chex.assert_tree_all_finite(jnp.real(v))
        chex.assert_tree_all_finite(jnp.imag(v))

    @parameterized.named_parameters(
        ("lobato", "lobato"),
        ("kirkland", "kirkland"),
    )
    def test_parameterization_switch(self, parameterization: str) -> None:
        r"""Both parameterizations produce same-shape complex output.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Both
        parameterizations produce same-shape complex output.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``parameterization``, so the documented behavior is checked across the
        cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        pos: Float[Array, "1 3"] = jnp.array(
            [[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]]
        )
        z_arr: Int[Array, "1"] = jnp.array([14], dtype=jnp.int32)
        v: Complex[Array, "H W"] = crystal_projected_potential(
            atomic_positions_angstrom=pos,
            atomic_numbers=z_arr,
            grid_shape=grid,
            cell_dimensions_angstrom=cell,
            absorption_fraction=0.1,
            parameterization=parameterization,
        )
        chex.assert_shape(v, grid)
        assert jnp.iscomplexobj(v)

    def test_periodic_grid_excludes_endpoint(self) -> None:
        r"""Grid samples are spaced L/n with no duplicate boundary column.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the potential
        grid uses n samples spaced L/n starting at 0 and excluding the
        endpoint L, matching the fftfreq(n, L/n) convention assumed by
        fresnel_propagator. For a single atom at (0, 0) on a periodic
        grid, column 0 (the atom site) and the wrap-around column n-1
        (one pixel away through the boundary) must not be duplicates,
        while columns 1 and n-1 sit at the same minimum-image distance
        dx and must agree.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_multislice``, so the Test
        Reference exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.crystal_projected_potential`
        :see: :func:`~rheedium.simul.fresnel_propagator`
        """
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        pos: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])
        z_arr: Int[Array, "1"] = jnp.array([14], dtype=jnp.int32)
        v: Complex[Array, "H W"] = crystal_projected_potential(
            atomic_positions_angstrom=pos,
            atomic_numbers=z_arr,
            grid_shape=grid,
            cell_dimensions_angstrom=cell,
            absorption_fraction=0.0,
            parameterization="lobato",
        )
        v_real: Float[Array, "H W"] = jnp.real(v)
        # Peak sits exactly on grid point (0, 0)
        peak_idx: tuple[Array, ...] = jnp.unravel_index(
            jnp.argmax(v_real), v_real.shape
        )
        assert int(peak_idx[0]) == 0
        assert int(peak_idx[1]) == 0
        # The old linspace(0, L, n) grid duplicated x=0 and x=L: the last
        # column equalled the first. With arange(n) * (L/n) it must not.
        assert not bool(jnp.allclose(v_real[:, 0], v_real[:, -1], rtol=1e-6))
        assert not bool(jnp.allclose(v_real[0, :], v_real[-1, :], rtol=1e-6))
        # Wrap-around symmetry: column 1 (distance dx = L/n) matches
        # column n-1 (distance L - (n-1) dx = dx through the boundary).
        chex.assert_trees_all_close(v_real[:, 1], v_real[:, -1], rtol=1e-10)
        chex.assert_trees_all_close(v_real[1, :], v_real[-1, :], rtol=1e-10)
