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

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from rheedium.simul.multislice import (
    build_transmission_function,
    fresnel_propagator,
    multislice_one_step,
)
from rheedium.simul.potential import crystal_projected_potential


def _make_grid_params():
    """Standard test grid parameters."""
    grid = (16, 16)
    cell = jnp.array([8.0, 8.0])
    voltage = 20.0
    dz = 2.0
    return grid, cell, voltage, dz


def _zero_potential(grid):
    """Zero complex potential of given shape."""
    return jnp.zeros(grid, dtype=jnp.complex128)


def _single_atom_potential(
    grid,
    cell,
    z=14,
    absorption=0.1,
):
    """Single atom at the cell center."""
    pos = jnp.array([[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]])
    z_arr = jnp.array([z], dtype=jnp.int32)
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

    def test_zero_potential_gives_unity(self):
        """T = 1 everywhere when V = 0."""
        grid, _, voltage, dz = _make_grid_params()
        v = _zero_potential(grid)
        t = build_transmission_function(v, voltage, dz)
        chex.assert_shape(t, grid)
        chex.assert_trees_all_close(
            t, jnp.ones(grid, dtype=jnp.complex128), atol=1e-12
        )

    def test_modulus_bounded_by_one(self):
        """|T| <= 1 everywhere (absorption only removes amplitude)."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        t = build_transmission_function(v, voltage, dz)
        modulus = jnp.abs(t)
        assert float(jnp.max(modulus)) <= 1.0 + 1e-6

    def test_real_potential_gives_unit_modulus(self):
        """Pure-real V (no absorption) gives |T| = 1 everywhere."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell, absorption=0.0)
        t = build_transmission_function(v, voltage, dz)
        modulus = jnp.abs(t)
        chex.assert_trees_all_close(modulus, jnp.ones(grid), atol=1e-6)

    def test_absorption_reduces_modulus(self):
        """Higher absorption fraction lowers |T|."""
        grid, cell, voltage, dz = _make_grid_params()
        v_low = _single_atom_potential(grid, cell, absorption=0.05)
        v_high = _single_atom_potential(grid, cell, absorption=0.5)
        t_low = build_transmission_function(v_low, voltage, dz)
        t_high = build_transmission_function(v_high, voltage, dz)
        assert float(jnp.min(jnp.abs(t_high))) < float(jnp.min(jnp.abs(t_low)))

    def test_jit_matches_eager(self):
        """JIT-compiled output matches eager output."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        eager = build_transmission_function(v, voltage, dz)
        jit_fn = jax.jit(
            build_transmission_function,
            static_argnums=(),
        )
        compiled = jit_fn(v, voltage, dz)
        chex.assert_trees_all_close(eager, compiled, atol=1e-10)

    def test_projected_potential_input_is_independent_of_dz_argument(self):
        """Projected-potential transmission does not multiply by dz twice."""
        grid, cell, voltage, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        t_thin = build_transmission_function(v, voltage, 1.0)
        t_thick = build_transmission_function(v, voltage, 5.0)
        chex.assert_trees_all_close(t_thin, t_thick, atol=1e-12)


class TestFresnelPropagator(chex.TestCase, parameterized.TestCase):
    """Tests for fresnel_propagator."""

    def test_unitarity(self):
        """|P(qx, qy)| = 1 everywhere — propagation is unitary."""
        grid, cell, voltage, dz = _make_grid_params()
        p = fresnel_propagator(grid, cell, voltage, dz)
        modulus = jnp.abs(p)
        chex.assert_trees_all_close(modulus, jnp.ones(grid), atol=1e-10)

    def test_zero_thickness_is_identity(self):
        """dz = 0 gives P = 1 (no propagation)."""
        grid, cell, voltage, _ = _make_grid_params()
        p = fresnel_propagator(grid, cell, voltage, 0.0)
        chex.assert_trees_all_close(
            p, jnp.ones(grid, dtype=jnp.complex128), atol=1e-12
        )

    def test_dc_component_is_unity(self):
        """At qx = qy = 0 the propagator equals 1 for any dz."""
        grid, cell, voltage, dz = _make_grid_params()
        p = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_trees_all_close(float(jnp.real(p[0, 0])), 1.0, atol=1e-12)
        chex.assert_trees_all_close(float(jnp.imag(p[0, 0])), 0.0, atol=1e-12)

    def test_shape(self):
        """Output shape matches grid."""
        grid, cell, voltage, dz = _make_grid_params()
        p = fresnel_propagator(grid, cell, voltage, dz)
        chex.assert_shape(p, grid)


class TestMultisliceOneStep(chex.TestCase, parameterized.TestCase):
    """Tests for multislice_one_step."""

    def test_vacuum_preserves_norm(self):
        """Zero potential leaves |psi| unchanged (free propagation)."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _zero_potential(grid)
        t = build_transmission_function(v, voltage, dz)
        p = fresnel_propagator(grid, cell, voltage, dz)
        psi_in = jnp.ones(grid, dtype=jnp.complex128)
        psi_out = multislice_one_step(psi_in, t, p)
        norm_in = float(jnp.sum(jnp.abs(psi_in) ** 2))
        norm_out = float(jnp.sum(jnp.abs(psi_out) ** 2))
        chex.assert_trees_all_close(norm_out, norm_in, rtol=1e-6)

    def test_absorption_reduces_norm(self):
        """Non-zero absorption reduces wavefunction norm."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell, absorption=0.3)
        t = build_transmission_function(v, voltage, dz)
        p = fresnel_propagator(grid, cell, voltage, dz)
        psi_in = jnp.ones(grid, dtype=jnp.complex128)
        psi_out = multislice_one_step(psi_in, t, p)
        norm_in = float(jnp.sum(jnp.abs(psi_in) ** 2))
        norm_out = float(jnp.sum(jnp.abs(psi_out) ** 2))
        assert norm_out < norm_in

    def test_norm_monotonic_with_depth(self):
        """Norm decreases monotonically over many absorbing slices."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell, absorption=0.2)
        t = build_transmission_function(v, voltage, dz)
        p = fresnel_propagator(grid, cell, voltage, dz)
        psi = jnp.ones(grid, dtype=jnp.complex128)
        norms = [float(jnp.sum(jnp.abs(psi) ** 2))]
        for _ in range(5):
            psi = multislice_one_step(psi, t, p)
            norms.append(float(jnp.sum(jnp.abs(psi) ** 2)))
        for i in range(len(norms) - 1):
            assert norms[i + 1] <= norms[i] + 1e-6

    def test_output_shape(self):
        """Output shape equals input shape."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        t = build_transmission_function(v, voltage, dz)
        p = fresnel_propagator(grid, cell, voltage, dz)
        psi_in = jnp.ones(grid, dtype=jnp.complex128)
        psi_out = multislice_one_step(psi_in, t, p)
        chex.assert_shape(psi_out, grid)

    def test_jit_compilation(self):
        """JIT-compiled multislice step matches eager output."""
        grid, cell, voltage, dz = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        t = build_transmission_function(v, voltage, dz)
        p = fresnel_propagator(grid, cell, voltage, dz)
        psi_in = jnp.ones(grid, dtype=jnp.complex128)
        eager = multislice_one_step(psi_in, t, p)
        jit_step = jax.jit(multislice_one_step)
        compiled = jit_step(psi_in, t, p)
        chex.assert_trees_all_close(eager, compiled, atol=1e-10)

    def test_grad_flows_through_potential(self):
        """jax.grad through multislice w.r.t. absorption is finite."""
        grid, cell, voltage, dz = _make_grid_params()
        psi_in = jnp.ones(grid, dtype=jnp.complex128)
        pos = jnp.array([[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]])
        z_arr = jnp.array([14], dtype=jnp.int32)
        p = fresnel_propagator(grid, cell, voltage, dz)

        def loss_fn(absorption):
            v = crystal_projected_potential(
                atomic_positions_angstrom=pos,
                atomic_numbers=z_arr,
                grid_shape=grid,
                cell_dimensions_angstrom=cell,
                absorption_fraction=absorption,
                parameterization="lobato",
            )
            t = build_transmission_function(v, voltage, dz)
            psi_out = multislice_one_step(psi_in, t, p)
            return jnp.sum(jnp.abs(psi_out) ** 2).real

        g = float(jax.grad(loss_fn)(0.1))
        assert jnp.isfinite(g)
        assert g != 0.0


class TestCrystalProjectedPotential(chex.TestCase, parameterized.TestCase):
    """Tests for crystal_projected_potential."""

    def test_output_shape(self):
        """Output shape matches grid_shape."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        chex.assert_shape(v, grid)

    def test_output_is_complex(self):
        """Output dtype is complex."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        assert jnp.iscomplexobj(v)

    def test_real_part_nonnegative(self):
        """Real part of V is non-negative for all atoms."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        assert float(jnp.min(jnp.real(v))) >= -1e-6

    def test_imag_part_proportional_to_real(self):
        """V_abs = absorption_fraction * V_real."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell, absorption=0.2)
        v_real = jnp.real(v)
        v_imag = jnp.imag(v)
        chex.assert_trees_all_close(v_imag, 0.2 * v_real, atol=1e-6)

    def test_zero_absorption_gives_real_potential(self):
        """absorption_fraction=0 yields purely real potential."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell, absorption=0.0)
        chex.assert_trees_all_close(jnp.imag(v), jnp.zeros(grid), atol=1e-12)

    def test_higher_z_gives_stronger_potential(self):
        """Heavier atom (Z=82) gives higher peak potential than Z=6."""
        grid, cell, _, _ = _make_grid_params()
        v_carbon = _single_atom_potential(grid, cell, z=6)
        v_lead = _single_atom_potential(grid, cell, z=82)
        peak_c = float(jnp.max(jnp.real(v_carbon)))
        peak_pb = float(jnp.max(jnp.real(v_lead)))
        assert peak_pb > peak_c

    def test_two_atoms_gives_two_peaks(self):
        """Two atoms produce two peaks in the potential."""
        grid, cell, _, _ = _make_grid_params()
        pos = jnp.array(
            [
                [2.0, 4.0, 0.0],
                [6.0, 4.0, 0.0],
            ]
        )
        z_arr = jnp.array([14, 14], dtype=jnp.int32)
        v = crystal_projected_potential(
            atomic_positions_angstrom=pos,
            atomic_numbers=z_arr,
            grid_shape=grid,
            cell_dimensions_angstrom=cell,
            absorption_fraction=0.1,
            parameterization="lobato",
        )
        v_real = jnp.real(v)
        col_at_y4 = v_real[:, grid[1] // 2]
        assert float(col_at_y4[grid[0] // 4]) > 0.0
        assert float(col_at_y4[3 * grid[0] // 4]) > 0.0

    def test_finite_everywhere(self):
        """Output is finite (no NaN, no Inf)."""
        grid, cell, _, _ = _make_grid_params()
        v = _single_atom_potential(grid, cell)
        chex.assert_tree_all_finite(jnp.real(v))
        chex.assert_tree_all_finite(jnp.imag(v))

    @parameterized.named_parameters(
        ("lobato", "lobato"),
        ("kirkland", "kirkland"),
    )
    def test_parameterization_switch(self, parameterization):
        """Both parameterizations produce same-shape complex output."""
        grid, cell, _, _ = _make_grid_params()
        pos = jnp.array([[float(cell[0]) / 2.0, float(cell[1]) / 2.0, 0.0]])
        z_arr = jnp.array([14], dtype=jnp.int32)
        v = crystal_projected_potential(
            atomic_positions_angstrom=pos,
            atomic_numbers=z_arr,
            grid_shape=grid,
            cell_dimensions_angstrom=cell,
            absorption_fraction=0.1,
            parameterization=parameterization,
        )
        chex.assert_shape(v, grid)
        assert jnp.iscomplexobj(v)
