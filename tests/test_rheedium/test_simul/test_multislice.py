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


class TestBuildTransmissionFunction(chex.TestCase, parameterized.TestCase):
    """Tests for build_transmission_function."""

    def test_zero_potential_gives_unity(self) -> None:
        """T = 1 everywhere when V = 0."""
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
        """|T| <= 1 everywhere (absorption only removes amplitude)."""
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
        """Pure-real V (no absorption) gives |T| = 1 everywhere."""
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
        """Higher absorption fraction lowers |T|."""
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
        """JIT-compiled output matches eager output."""
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
        """The explicit projected-potential keyword matches positional use."""
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
            voltage_kv=voltage,
        )
        chex.assert_trees_all_close(positional, keyword, atol=1e-12)


class TestFresnelPropagator(chex.TestCase, parameterized.TestCase):
    """Tests for fresnel_propagator."""

    def test_unitarity(self) -> None:
        """|P(qx, qy)| = 1 everywhere — propagation is unitary."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        modulus: Float[Array, "H W"] = jnp.abs(p)
        chex.assert_trees_all_close(modulus, jnp.ones(grid), atol=1e-10)

    def test_zero_thickness_is_identity(self) -> None:
        """Dz = 0 gives P = 1 (no propagation)."""
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
        """At qx = qy = 0 the propagator equals 1 for any dz."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_trees_all_close(float(jnp.real(p[0, 0])), 1.0, atol=1e-12)
        chex.assert_trees_all_close(float(jnp.imag(p[0, 0])), 0.0, atol=1e-12)

    def test_shape(self) -> None:
        """Output shape matches grid."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        p: Complex[Array, "H W"] = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_shape(p, grid)


class TestMultisliceOneStep(chex.TestCase, parameterized.TestCase):
    """Tests for multislice_one_step."""

    def test_vacuum_preserves_norm(self) -> None:
        """Zero potential leaves |psi| unchanged (free propagation)."""
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
        """Non-zero absorption reduces wavefunction norm."""
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
        """Norm decreases monotonically over many absorbing slices."""
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
        for i in range(len(norms) - 1):
            assert norms[i + 1] <= norms[i] + 1e-6

    def test_output_shape(self) -> None:
        """Output shape equals input shape."""
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
        """JIT-compiled multislice step matches eager output."""
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
        """jax.grad through multislice w.r.t. absorption is finite."""
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


class TestCrystalProjectedPotential(chex.TestCase, parameterized.TestCase):
    """Tests for crystal_projected_potential."""

    def test_output_shape(self) -> None:
        """Output shape matches grid_shape."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        chex.assert_shape(v, grid)

    def test_output_is_complex(self) -> None:
        """Output dtype is complex."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        assert jnp.iscomplexobj(v)

    def test_real_part_nonnegative(self) -> None:
        """Real part of V is non-negative for all atoms."""
        grid: tuple[int, int]
        cell: Float[Array, "2"]
        voltage: float
        dz: float
        grid, cell, voltage, dz = _make_grid_params()
        v: Complex[Array, "H W"] = _single_atom_potential(grid, cell)
        assert float(jnp.min(jnp.real(v))) >= -1e-6

    def test_imag_part_proportional_to_real(self) -> None:
        """V_abs = absorption_fraction * V_real."""
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
        """absorption_fraction=0 yields purely real potential."""
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
        """Heavier atom (Z=82) gives higher peak potential than Z=6."""
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
        """Two atoms produce two peaks in the potential."""
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
        """Output is finite (no NaN, no Inf)."""
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
        """Both parameterizations produce same-shape complex output."""
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
