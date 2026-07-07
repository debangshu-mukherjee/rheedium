"""Test suite for ewald.py internal functions and public API.

This module provides comprehensive testing for Ewald sphere construction,
structure factor calculations, and allowed reflection finding used in
RHEED simulations.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Bool, Complex, Float, Int

from rheedium.simul.ewald import (
    _compute_structure_factor_single,
    build_ewald_data,
    ewald_allowed_reflections,
)
from rheedium.simul.finite_domain import compute_shell_sigma
from rheedium.types import CrystalStructure, EwaldData, create_ewald_data
from rheedium.types.crystal_types import create_crystal_structure
from rheedium.types.custom_types import scalar_float


class TestComputeStructureFactorSingle(chex.TestCase, parameterized.TestCase):
    """Test suite for _compute_structure_factor_single internal function.

    :see: :func:`~rheedium.simul.ewald._compute_structure_factor_single`
    """

    def setUp(self) -> None:
        """Set up test fixtures with simple crystal structures."""
        super().setUp()

        # Simple cubic structure with one atom at origin
        self.single_atom_positions: Float[Array, "1 3"] = jnp.array(
            [[0.0, 0.0, 0.0]]
        )
        self.single_atom_numbers: Int[Array, "1"] = jnp.array([14])

        # Two-atom structure for phase testing
        self.two_atom_positions: Float[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]  # 2 Angstrom separation
        )
        self.two_atom_numbers: Int[Array, "2"] = jnp.array([14, 14])

        # Multi-element structure
        self.multi_element_positions: Float[Array, "2 3"] = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        self.multi_element_numbers: Int[Array, "2"] = jnp.array([14, 8])

        self.temperature: float = 300.0

    def test_single_atom_at_origin_g_zero(self) -> None:
        r"""Test structure factor for single atom at origin with G=0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        for single atom at origin with G=0. Existing context from the original
        test prose: For a single atom at the origin, the phase factor
        exp(i*G*r) = 1 when G=0, so F(0) equals the atomic form factor times DW
        factor.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vector: Float[Array, "3"] = jnp.array([0.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        # Should be real and positive for G=0
        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.real(sf)))
        # Imaginary part should be negligible
        chex.assert_trees_all_close(jnp.imag(sf), 0.0, atol=1e-10)

    @parameterized.named_parameters(
        ("g_x", jnp.array([1.0, 0.0, 0.0])),
        ("g_y", jnp.array([0.0, 1.0, 0.0])),
        ("g_z", jnp.array([0.0, 0.0, 1.0])),
        ("g_diagonal", jnp.array([1.0, 1.0, 1.0])),
    )
    def test_single_atom_at_origin_nonzero_g(
        self, g_vector: Float[Array, "3"]
    ) -> None:
        r"""Test structure factor for single atom at origin with nonzero G.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        for single atom at origin with nonzero G. Existing context from the
        original test prose: For atom at origin, phase factor is always 1
        regardless of G, so the structure factor is just f(\|G\|) * DW(\|G\|).

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``g_vector``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        # For atom at origin, imaginary part should be zero
        chex.assert_trees_all_close(jnp.imag(sf), 0.0, atol=1e-10)
        # Real part should be positive (form factor is positive)
        chex.assert_scalar_positive(float(jnp.real(sf)))

    def test_structure_factor_decreases_with_g_magnitude(self) -> None:
        r"""Test that \|F(G)\| decreases with increasing \|G\|.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: \|F(G)\| decreases
        with increasing \|G\|. Existing context from the original test prose:
        Due to the form factor and Debye-Waller factor both decreasing with
        \|G\|, the structure factor magnitude should decrease.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vectors: Float[Array, "4 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )

        sf_magnitudes: list[Float[Array, ""]] = []
        g: Any
        for g in g_vectors:
            sf: Complex[Array, ""] = _compute_structure_factor_single(
                g_vector=g,
                atom_positions=self.single_atom_positions,
                atomic_numbers=self.single_atom_numbers,
                temperature=self.temperature,
            )
            sf_magnitudes.append(jnp.abs(sf))

        sf_magnitudes_array: Float[Array, "4"] = jnp.array(sf_magnitudes)
        differences: Float[Array, "3"] = jnp.diff(sf_magnitudes_array)
        chex.assert_trees_all_equal(jnp.all(differences <= 0), True)

    def test_two_atom_interference(self) -> None:
        r"""Test interference effects between two atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: interference
        effects between two atoms. Existing context from the original test
        prose: For two atoms separated by distance d along x, G = (2*pi/d, 0,
        0) should give constructive interference (phase = 2*pi), while G =
        (pi/d, 0, 0) gives destructive interference (phase = pi).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Constructive interference when G times d equals two pi.
        g_constructive: Float[Array, "3"] = jnp.array([jnp.pi, 0.0, 0.0])
        sf_constr: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_constructive,
            atom_positions=self.two_atom_positions,
            atomic_numbers=self.two_atom_numbers,
            temperature=self.temperature,
        )

        # Destructive interference when G times d equals pi.
        g_destructive: Float[Array, "3"] = jnp.array([jnp.pi / 2, 0.0, 0.0])
        sf_destr: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_destructive,
            atom_positions=self.two_atom_positions,
            atomic_numbers=self.two_atom_numbers,
            temperature=self.temperature,
        )

        # Constructive should have larger magnitude
        chex.assert_scalar_positive(
            float(jnp.abs(sf_constr) - jnp.abs(sf_destr))
        )

    def test_multi_element_structure(self) -> None:
        r"""Test structure factor with multiple element types.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        with multiple element types. Existing context from the original test
        prose: Different atomic numbers should contribute different form
        factors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vector: Float[Array, "3"] = jnp.array([1.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.multi_element_positions,
            atomic_numbers=self.multi_element_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        # Intensity should be non-zero
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    @parameterized.named_parameters(
        ("low_temp", 100.0),
        ("room_temp", 300.0),
        ("high_temp", 600.0),
    )
    def test_temperature_dependence(self, temperature: float) -> None:
        r"""Test that structure factor depends on temperature.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        depends on temperature. Existing context from the original test prose:
        Higher temperature increases Debye-Waller damping, reducing \|F(G)\|.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``temperature``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vector: Float[Array, "3"] = jnp.array([2.0, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=temperature,
        )

        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    def test_temperature_ordering(self) -> None:
        r"""Test that higher temperature gives lower \|F(G)\| for G != 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: higher temperature
        gives lower \|F(G)\| for G != 0.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vector: Float[Array, "3"] = jnp.array([2.0, 0.0, 0.0])

        sf_low: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=100.0,
        )
        sf_high: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=600.0,
        )

        # Lower temperature should give larger |F(G)|
        chex.assert_scalar_positive(float(jnp.abs(sf_low) - jnp.abs(sf_high)))

    def test_complex_phase_correctness(self) -> None:
        r"""Test that phase is computed correctly for off-origin atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: phase is computed
        correctly for off-origin atoms. Existing context from the original test
        prose: For an atom at position r, the phase contribution is exp(i*G*r).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Single atom at (1, 0, 0)
        atom_pos: Float[Array, "1 3"] = jnp.array([[1.0, 0.0, 0.0]])
        atom_nums: Int[Array, "1"] = jnp.array([14])

        # G = (pi, 0, 0) should give phase = exp(i*pi) = -1
        g_vector: Float[Array, "3"] = jnp.array([jnp.pi, 0.0, 0.0])

        sf: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=atom_pos,
            atomic_numbers=atom_nums,
            temperature=self.temperature,
        )

        # Real part should be negative (due to phase = -1)
        self.assertLess(float(jnp.real(sf)), 0.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_compatibility(self) -> None:
        r"""Test that _compute_structure_factor_single works with JIT.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        _compute_structure_factor_single works with JIT.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        It runs through the Chex variant wrapper where present, so the same
        assertion covers both transformed and untransformed JAX execution
        paths.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        var_compute_sf: Callable[..., Complex[Array, ""]] = self.variant(
            _compute_structure_factor_single
        )

        g_vector: Float[Array, "3"] = jnp.array([1.0, 0.0, 0.0])

        sf: Complex[Array, ""] = var_compute_sf(
            g_vector=g_vector,
            atom_positions=self.single_atom_positions,
            atomic_numbers=self.single_atom_numbers,
            temperature=self.temperature,
        )

        chex.assert_tree_all_finite(sf)
        chex.assert_scalar_positive(float(jnp.abs(sf)))

    def test_vmap_over_g_vectors(self) -> None:
        r"""Test vmapping _compute_structure_factor_single over G vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: vmapping
        _compute_structure_factor_single over G vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises vectorization, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        g_vectors: Float[Array, "5 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        vmapped_sf: Callable[[Float[Array, "5 3"]], Complex[Array, "5"]] = (
            jax.vmap(
                lambda g: _compute_structure_factor_single(
                    g_vector=g,
                    atom_positions=self.single_atom_positions,
                    atomic_numbers=self.single_atom_numbers,
                    temperature=self.temperature,
                )
            )
        )

        sfs: Complex[Array, "5"] = vmapped_sf(g_vectors)

        chex.assert_shape(sfs, (5,))
        chex.assert_tree_all_finite(sfs)

    def test_gradient_flow(self) -> None:
        r"""Test that gradients flow through structure factor calculation.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: gradients flow
        through structure factor calculation.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """

        def loss_fn(temperature: scalar_float) -> scalar_float:
            g_vector: Float[Array, "3"] = jnp.array([1.0, 0.0, 0.0])
            sf: Complex[Array, ""] = _compute_structure_factor_single(
                g_vector=g_vector,
                atom_positions=self.single_atom_positions,
                atomic_numbers=self.single_atom_numbers,
                temperature=temperature,
            )
            return jnp.abs(sf)

        grad_fn: Callable[[scalar_float], scalar_float] = jax.grad(loss_fn)
        gradient: scalar_float = grad_fn(300.0)

        chex.assert_tree_all_finite(gradient)
        # Gradient should be negative (higher temp -> lower |F|)
        self.assertLess(float(gradient), 0.0)


class TestBuildEwaldData(chex.TestCase, parameterized.TestCase):
    """Test suite for build_ewald_data public function.

    :see: :func:`~rheedium.simul.build_ewald_data`
    """

    def setUp(self) -> None:
        """Set up test fixtures with a simple crystal structure."""
        super().setUp()
        self.crystal: CrystalStructure = self._create_simple_cubic_crystal()

    def _create_simple_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal for testing."""
        a: float = 4.0

        frac_coords: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])
        cart_coords: Float[Array, "1 3"] = frac_coords * a
        atomic_numbers: Float[Array, "1"] = jnp.array([14.0])

        frac_positions: Float[Array, "1 4"] = jnp.column_stack(
            [frac_coords, atomic_numbers]
        )
        cart_positions: Float[Array, "1 4"] = jnp.column_stack(
            [cart_coords, atomic_numbers]
        )

        return create_crystal_structure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_ewald_data_creation(self) -> None:
        r"""Test basic EwaldData creation with minimal parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: basic EwaldData
        creation with minimal parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=300.0,
        )

        # Check that all fields are populated
        chex.assert_tree_all_finite(ewald.wavelength_ang)
        chex.assert_tree_all_finite(ewald.k_magnitude)
        chex.assert_tree_all_finite(ewald.sphere_radius)
        chex.assert_tree_all_finite(ewald.recip_vectors)
        chex.assert_tree_all_finite(ewald.g_vectors)
        chex.assert_tree_all_finite(ewald.g_magnitudes)
        chex.assert_tree_all_finite(ewald.structure_factors)
        chex.assert_tree_all_finite(ewald.intensities)

    def test_wavelength_calculation(self) -> None:
        r"""Test that wavelength is correctly computed from voltage.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: wavelength is
        correctly computed from voltage.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald_10kv: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=10.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )
        ewald_20kv: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        # Higher voltage should give shorter wavelength
        chex.assert_scalar_positive(
            float(ewald_10kv.wavelength_ang - ewald_20kv.wavelength_ang)
        )

        # Wavelength should be reasonable for RHEED (0.05 - 0.15 Angstroms)
        self.assertGreater(float(ewald_10kv.wavelength_ang), 0.05)
        self.assertLess(float(ewald_10kv.wavelength_ang), 0.15)

    def test_k_magnitude_relation(self) -> None:
        r"""Test that k_magnitude = 2*pi / wavelength.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: k_magnitude = 2*pi
        / wavelength.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        expected_k: scalar_float = 2.0 * jnp.pi / ewald.wavelength_ang
        chex.assert_trees_all_close(ewald.k_magnitude, expected_k, rtol=1e-10)

    def test_sphere_radius_equals_k_magnitude(self) -> None:
        r"""Test that Ewald sphere radius equals k magnitude.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Ewald sphere
        radius equals k magnitude.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=1,
            kmax=1,
            lmax=1,
        )

        chex.assert_trees_all_close(
            ewald.sphere_radius, ewald.k_magnitude, rtol=1e-10
        )

    @parameterized.named_parameters(
        ("small_grid", 1, 1, 1, 27),  # (2*1+1)^3 = 27
        ("medium_grid", 2, 2, 1, 75),  # 5*5*3 = 75
        ("asymmetric", 3, 2, 1, 105),  # 7*5*3 = 105
    )
    def test_grid_size(
        self, hmax: int, kmax: int, lmax: int, expected_n: int
    ) -> None:
        r"""Test that the correct number of G vectors are generated.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the correct number
        of G vectors are generated.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``hmax``,
        ``kmax``, ``lmax``, ``expected_n``, so the documented behavior is
        checked across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=hmax,
            kmax=kmax,
            lmax=lmax,
        )

        chex.assert_shape(ewald.hkl_grid, (expected_n, 3))
        chex.assert_shape(ewald.g_vectors, (expected_n, 3))
        chex.assert_shape(ewald.g_magnitudes, (expected_n,))
        chex.assert_shape(ewald.structure_factors, (expected_n,))
        chex.assert_shape(ewald.intensities, (expected_n,))

    def test_intensities_are_squared_magnitudes(self) -> None:
        r"""Test that intensities = \|structure_factors\|^2.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: intensities =
        \|structure_factors\|^2.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        expected_intensities: Float[Array, "N"] = (
            jnp.abs(ewald.structure_factors) ** 2
        )
        chex.assert_trees_all_close(
            ewald.intensities, expected_intensities, rtol=1e-10
        )

    def test_intensities_nonnegative(self) -> None:
        r"""Test that all intensities are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all intensities
        are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        chex.assert_trees_all_equal(jnp.all(ewald.intensities >= 0), True)

    def test_g_magnitudes_nonnegative(self) -> None:
        r"""Test that all G magnitudes are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all G magnitudes
        are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        chex.assert_trees_all_equal(jnp.all(ewald.g_magnitudes >= 0), True)

    def test_g_magnitude_consistency(self) -> None:
        r"""Test that g_magnitudes matches norm of g_vectors.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: g_magnitudes
        matches norm of g_vectors.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
        )

        computed_mags: Float[Array, "N"] = jnp.linalg.norm(
            ewald.g_vectors, axis=-1
        )
        chex.assert_trees_all_close(
            ewald.g_magnitudes, computed_mags, rtol=1e-10
        )

    def test_temperature_affects_intensities(self) -> None:
        r"""Test that temperature affects structure factors/intensities.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: temperature
        affects structure factors/intensities.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ewald_low_t: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=100.0,
        )
        ewald_high_t: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=600.0,
        )

        # Higher temperature should reduce total intensity (more DW damping)
        total_low: scalar_float = jnp.sum(ewald_low_t.intensities)
        total_high: scalar_float = jnp.sum(ewald_high_t.intensities)

        chex.assert_scalar_positive(float(total_low - total_high))


class TestEwaldAllowedReflections(chex.TestCase, parameterized.TestCase):
    """Test suite for ewald_allowed_reflections function.

    :see: :func:`~rheedium.simul.ewald_allowed_reflections`
    """

    def setUp(self) -> None:
        """Set up test fixtures with pre-computed EwaldData."""
        super().setUp()
        self.crystal: CrystalStructure = self._create_simple_cubic_crystal()
        self.ewald: EwaldData = build_ewald_data(
            crystal=self.crystal,
            energy_kev=20.0,
            hmax=3,
            kmax=3,
            lmax=2,
            temperature=300.0,
        )

    def _create_simple_cubic_crystal(self) -> CrystalStructure:
        """Create a simple cubic crystal for testing."""
        a: float = 4.0
        frac_coords: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])
        cart_coords: Float[Array, "1 3"] = frac_coords * a
        atomic_numbers: Float[Array, "1"] = jnp.array([14.0])

        return create_crystal_structure(
            frac_positions=jnp.column_stack([frac_coords, atomic_numbers]),
            cart_positions=jnp.column_stack([cart_coords, atomic_numbers]),
            cell_lengths=jnp.array([a, a, a]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
        )

    def test_basic_reflection_finding(self) -> None:
        r"""Test basic reflection finding with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: basic reflection
        finding with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        # Check output shapes are consistent
        n: int = indices.shape[0]
        chex.assert_shape(k_out, (n, 3))
        chex.assert_shape(intensities, (n,))

    def test_upward_scattering_only(self) -> None:
        r"""Test that only upward scattering reflections are returned.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: only upward
        scattering reflections are returned.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        # Valid reflections should have k_out_z > 0
        valid_mask: Bool[Array, "N"] = indices >= 0
        valid_k_out: Float[Array, "M 3"] = k_out[valid_mask]

        if valid_k_out.shape[0] > 0:
            chex.assert_trees_all_equal(jnp.all(valid_k_out[:, 2] > 0), True)

    def test_intensities_nonnegative(self) -> None:
        r"""Test that all returned intensities are non-negative.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: all returned
        intensities are non-negative.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )

        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    @parameterized.named_parameters(
        ("theta_1", 1.0),
        ("theta_2", 2.0),
        ("theta_5", 5.0),
    )
    def test_different_theta_angles(self, theta_deg: float) -> None:
        r"""Test reflection finding at different incidence angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reflection finding
        at different incidence angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``theta_deg``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=theta_deg,
            phi_deg=0.0,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)

    @parameterized.named_parameters(
        ("phi_0", 0.0),
        ("phi_45", 45.0),
        ("phi_90", 90.0),
    )
    def test_different_phi_angles(self, phi_deg: float) -> None:
        r"""Test reflection finding at different azimuthal angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: reflection finding
        at different azimuthal angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``phi_deg``,
        so the documented behavior is checked across the cases supplied by
        pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        indices: Any
        k_out: Float[Array, "..."]
        intensities: Float[Array, "..."]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=phi_deg,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)

    def test_tolerance_effect(self) -> None:
        r"""Test that larger absolute tolerance allows more reflections.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: a larger
        ``tolerance_inv_ang`` (absolute Ewald-shell half-thickness in
        inverse Ångstroms) admits at least as many reflections as a
        tighter one.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.ewald_allowed_reflections`
        """
        indices_tight: Int[Array, "N"]
        k_out_tight: Float[Array, "N 3"]
        intensities_tight: Float[Array, "N"]
        indices_tight, k_out_tight, intensities_tight = (
            ewald_allowed_reflections(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                tolerance_inv_ang=0.05,
            )
        )
        indices_loose: Int[Array, "N"]
        k_out_loose: Float[Array, "N 3"]
        intensities_loose: Float[Array, "N"]
        indices_loose, k_out_loose, intensities_loose = (
            ewald_allowed_reflections(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                tolerance_inv_ang=1.0,
            )
        )

        # More allowed reflections with larger tolerance
        n_tight: scalar_float = jnp.sum(indices_tight >= 0)
        n_loose: scalar_float = jnp.sum(indices_loose >= 0)

        self.assertGreaterEqual(int(n_loose), int(n_tight))

    def test_finite_domain_mode(self) -> None:
        r"""Test finite domain mode with domain_extent_ang parameter.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: finite domain mode
        with domain_extent_ang parameter.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        domain: Float[Array, "3"] = jnp.array([100.0, 100.0, 50.0])

        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )

        chex.assert_tree_all_finite(k_out)
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(intensities >= 0), True)

    def test_finite_domain_overlap_weighting(self) -> None:
        r"""Test that finite domain mode applies overlap weighting.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: finite domain mode
        applies overlap weighting.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        domain: Float[Array, "3"] = jnp.array([50.0, 50.0, 25.0])

        indices_binary: Int[Array, "N"]
        k_out_binary: Float[Array, "N 3"]
        intensities_binary: Float[Array, "N"]
        indices_binary, k_out_binary, intensities_binary = (
            ewald_allowed_reflections(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                tolerance_inv_ang=0.25,
            )
        )
        indices_finite: Int[Array, "N"]
        k_out_finite: Float[Array, "N 3"]
        intensities_finite: Float[Array, "N"]
        indices_finite, k_out_finite, intensities_finite = (
            ewald_allowed_reflections(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                domain_extent_ang=domain,
            )
        )

        # Finite domain should produce different intensity distribution
        # (not necessarily larger or smaller total)
        chex.assert_tree_all_finite(intensities_binary)
        chex.assert_tree_all_finite(intensities_finite)

    def test_k_out_is_k_in_plus_g(self) -> None:
        r"""Test that k_out = k_in + G for allowed reflections.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: k_out = k_in + G
        for allowed reflections.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        theta_deg: float = 2.0
        phi_deg: float = 0.0

        theta_rad: scalar_float = jnp.deg2rad(theta_deg)
        phi_rad: scalar_float = jnp.deg2rad(phi_deg)
        k_mag: scalar_float = self.ewald.k_magnitude

        k_in: Float[Array, "3"] = k_mag * jnp.array(
            [
                jnp.cos(theta_rad) * jnp.cos(phi_rad),
                jnp.cos(theta_rad) * jnp.sin(phi_rad),
                -jnp.sin(theta_rad),
            ]
        )

        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )

        # For valid indices, check k_out = k_in + G
        i: int
        idx: int
        for i, idx in enumerate(indices):
            if idx >= 0:
                g_vec: Float[Array, "3"] = self.ewald.g_vectors[idx]
                expected_k_out: Float[Array, "3"] = k_in + g_vec
                chex.assert_trees_all_close(
                    k_out[i], expected_k_out, rtol=1e-10
                )

    def test_default_tolerance_derived_from_shell_sigma(self) -> None:
        r"""Test that the default tolerance is 3 x compute_shell_sigma.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: when
        ``tolerance_inv_ang`` is None, the binary Ewald condition uses an
        absolute half-thickness of three shell sigmas derived from the
        default beam parameters (energy spread 1e-4, divergence 1 mrad).
        At 20 kV this admits reflections with :math:`|\Delta k|` below
        roughly 0.25 inverse Ångstroms, not the 3.66 inverse Ångstroms
        the old fractional default allowed.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.ewald_allowed_reflections`
        :see: :func:`~rheedium.simul.compute_shell_sigma`
        """
        tol_default: Float[Array, ""] = 3.0 * compute_shell_sigma(
            k_magnitude=self.ewald.k_magnitude,
            energy_spread_frac=1e-4,
            beam_divergence_rad=1e-3,
        )
        # Physical shell half-thickness at 20 kV, well under 0.25 1/A
        self.assertLess(float(tol_default), 0.25)
        self.assertGreater(float(tol_default), 0.0)

        indices_none: Int[Array, "N"]
        k_out_none: Float[Array, "N 3"]
        intensities_none: Float[Array, "N"]
        indices_none, k_out_none, intensities_none = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
        )
        indices_explicit: Int[Array, "N"]
        k_out_explicit: Float[Array, "N 3"]
        intensities_explicit: Float[Array, "N"]
        indices_explicit, k_out_explicit, intensities_explicit = (
            ewald_allowed_reflections(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                tolerance_inv_ang=tol_default,
            )
        )
        chex.assert_trees_all_equal(indices_none, indices_explicit)
        chex.assert_trees_all_close(k_out_none, k_out_explicit)
        chex.assert_trees_all_close(intensities_none, intensities_explicit)

        # All admitted reflections satisfy |dk| < tol_default (absolute)
        theta_rad: Float[Array, ""] = jnp.deg2rad(2.0)
        k_mag: Float[Array, ""] = self.ewald.k_magnitude
        k_in: Float[Array, "3"] = k_mag * jnp.array(
            [jnp.cos(theta_rad), 0.0, -jnp.sin(theta_rad)]
        )
        valid_mask: Bool[Array, "N"] = indices_none >= 0
        for i, idx in enumerate(indices_none):
            if valid_mask[i]:
                dk: Float[Array, ""] = jnp.abs(
                    jnp.linalg.norm(k_in + self.ewald.g_vectors[idx]) - k_mag
                )
                self.assertLess(float(dk), float(tol_default))

    def test_single_reflection_padding_zeroed(self) -> None:
        r"""Test padded slots are zeroed with exactly one allowed reflection.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: with a
        hand-built EwaldData containing exactly one G vector on the Ewald
        sphere and several far off it, the padded output slots carry
        ``k_out == 0``, ``intensity == 0``, and ``index == -1``, and the
        intensity sum equals the single reflection's intensity. This
        guards against the JAX gather semantics where index -1 silently
        reads the last array element, producing phantom copies.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.ewald_allowed_reflections`
        """
        k_mag: Float[Array, ""] = self.ewald.k_magnitude
        theta_rad: Float[Array, ""] = jnp.deg2rad(2.0)
        k_in: Float[Array, "3"] = k_mag * jnp.array(
            [jnp.cos(theta_rad), 0.0, -jnp.sin(theta_rad)]
        )
        # One G exactly on the sphere with upward k_out; three far off
        k_out_target: Float[Array, "3"] = k_mag * jnp.array(
            [jnp.cos(theta_rad), 0.0, jnp.sin(theta_rad)]
        )
        g_on_sphere: Float[Array, "3"] = k_out_target - k_in
        g_vectors: Float[Array, "4 3"] = jnp.stack(
            [
                g_on_sphere,
                jnp.array([10.0, 0.0, 0.0]),
                jnp.array([0.0, 10.0, 0.0]),
                jnp.array([10.0, 10.0, 10.0]),
            ]
        )
        structure_factors: Complex[Array, "4"] = jnp.array(
            [2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j]
        )
        intensities_all: Float[Array, "4"] = jnp.abs(structure_factors) ** 2
        ewald_single: EwaldData = create_ewald_data(
            wavelength_ang=self.ewald.wavelength_ang,
            k_magnitude=k_mag,
            sphere_radius=k_mag,
            recip_vectors=self.ewald.recip_vectors,
            hkl_grid=jnp.zeros((4, 3), dtype=jnp.int32),
            g_vectors=g_vectors,
            g_magnitudes=jnp.linalg.norm(g_vectors, axis=-1),
            structure_factors=structure_factors,
            intensities=intensities_all,
        )
        indices: Int[Array, "4"]
        k_out: Float[Array, "4 3"]
        intensities: Float[Array, "4"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=ewald_single,
            theta_deg=2.0,
            phi_deg=0.0,
        )
        chex.assert_trees_all_equal(
            indices, jnp.array([0, -1, -1, -1], dtype=indices.dtype)
        )
        chex.assert_trees_all_equal(k_out[1:], jnp.zeros((3, 3)))
        chex.assert_trees_all_equal(intensities[1:], jnp.zeros(3))
        chex.assert_trees_all_close(intensities[0], intensities_all[0])
        # Sum counts the single reflection exactly once - no phantom copies
        chex.assert_trees_all_close(jnp.sum(intensities), intensities_all[0])
        chex.assert_trees_all_close(k_out[0], k_out_target)

    def test_finite_domain_padding_zeroed(self) -> None:
        r"""Test padded slots are zeroed in finite-domain mode.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: in finite
        domain mode every output slot with ``index == -1`` carries
        exactly zero ``k_out`` and zero intensity, so summing the
        returned intensities never double-counts a live reflection.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Exact tree equality assertions check structure, dtype, and values where
        the expected result is discrete or deterministic.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test Reference
        exposes both the guarantee and the implementation path.

        :see: :func:`~rheedium.simul.ewald_allowed_reflections`
        """
        domain: Float[Array, "3"] = jnp.array([100.0, 100.0, 50.0])
        indices: Int[Array, "N"]
        k_out: Float[Array, "N 3"]
        intensities: Float[Array, "N"]
        indices, k_out, intensities = ewald_allowed_reflections(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )
        invalid_mask: Bool[Array, "N"] = indices < 0
        chex.assert_trees_all_equal(
            jnp.all(jnp.where(invalid_mask[:, None], k_out, 0.0) == 0.0),
            True,
        )
        chex.assert_trees_all_equal(
            jnp.all(jnp.where(invalid_mask, intensities, 0.0) == 0.0),
            True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOccupancyWeightedEwald(chex.TestCase):
    """Occupancy weighting through the Ewald structure-factor stack.

    :see: :func:`~rheedium.simul.ewald._compute_structure_factor_single`
    :see: :func:`~rheedium.simul.build_ewald_data`
    :see: :func:`~rheedium.simul.ewald_allowed_reflections`
    """

    def test_structure_factor_scales_linearly_with_occupancy(self) -> None:
        r"""Verify F(q) scales as occ * f_Z(q) with exact zero at occ 0.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the per-site
        occupancy multiplies the atomic form-factor amplitude, so a
        half-occupied site produces exactly half the full-occupancy
        structure factor and a zero-occupancy site contributes exactly
        zero (the C6 contract).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        atom_positions: Float[Array, "1 3"] = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers: Int[Array, "1"] = jnp.array([14], dtype=jnp.int32)
        g_vector: Float[Array, "3"] = jnp.array([1.157, 0.4, 0.2])
        f_full: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=atom_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
        )
        f_half: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=atom_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            occupancies=jnp.array([0.5]),
        )
        f_zero: Complex[Array, ""] = _compute_structure_factor_single(
            g_vector=g_vector,
            atom_positions=atom_positions,
            atomic_numbers=atomic_numbers,
            temperature=300.0,
            occupancies=jnp.array([0.0]),
        )
        chex.assert_trees_all_close(
            complex(f_half), 0.5 * complex(f_full), rtol=1e-12
        )
        self.assertEqual(complex(f_zero), 0.0 + 0.0j)

    def test_build_ewald_data_carries_occupancies(self) -> None:
        r"""Verify build_ewald_data stores and applies crystal occupancies.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the crystal's
        per-site occupancies are stored on the returned ``EwaldData`` and
        already weight the pre-computed grid intensities, so a uniform
        occupancy of 0.5 quarters every stored intensity.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac: Float[Array, "1 4"] = jnp.array([[0.0, 0.0, 0.0, 14.0]])
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 3.0, 3.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
        crystal_full: CrystalStructure = create_crystal_structure(
            frac, frac, cell_lengths, cell_angles
        )
        crystal_half: CrystalStructure = create_crystal_structure(
            frac,
            frac,
            cell_lengths,
            cell_angles,
            occupancies=jnp.array([0.5]),
        )
        ewald_full: EwaldData = build_ewald_data(
            crystal_full, energy_kev=20.0, hmax=1, kmax=1, lmax=1
        )
        ewald_half: EwaldData = build_ewald_data(
            crystal_half, energy_kev=20.0, hmax=1, kmax=1, lmax=1
        )
        chex.assert_trees_all_close(ewald_half.occupancies, jnp.array([0.5]))
        chex.assert_trees_all_close(
            ewald_half.intensities,
            0.25 * ewald_full.intensities,
            rtol=1e-12,
        )

    def test_allowed_reflections_reflect_occupancy_in_both_modes(
        self,
    ) -> None:
        r"""Verify both reflection modes weight intensities by occupancy.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: binary-mode
        reflections (stored grid intensities) and finite-domain-mode
        reflections (continuous rod-intersection intensities re-evaluated
        through ``rod_base_intensities`` from ``EwaldData.occupancies``)
        both scale by the squared occupancy.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac: Float[Array, "1 4"] = jnp.array([[0.0, 0.0, 0.0, 14.0]])
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 3.0, 3.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])
        ewald_full: EwaldData = build_ewald_data(
            create_crystal_structure(frac, frac, cell_lengths, cell_angles),
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=2,
        )
        ewald_half: EwaldData = build_ewald_data(
            create_crystal_structure(
                frac,
                frac,
                cell_lengths,
                cell_angles,
                occupancies=jnp.array([0.5]),
            ),
            energy_kev=20.0,
            hmax=2,
            kmax=2,
            lmax=2,
        )
        indices_full: Int[Array, "N"]
        intensities_full: Float[Array, "N"]
        indices_full, _, intensities_full = ewald_allowed_reflections(
            ewald_full, theta_deg=2.0, phi_deg=0.0
        )
        _, _, intensities_half = ewald_allowed_reflections(
            ewald_half, theta_deg=2.0, phi_deg=0.0
        )
        valid: Bool[Array, "N"] = indices_full >= 0
        self.assertGreater(int(jnp.sum(valid)), 0)
        chex.assert_trees_all_close(
            intensities_half[valid],
            0.25 * intensities_full[valid],
            rtol=1e-12,
        )
        domain: Float[Array, "3"] = jnp.array([100.0, 100.0, 50.0])
        rod_indices: Int[Array, "N"]
        rod_full: Float[Array, "N"]
        rod_indices, _, rod_full = ewald_allowed_reflections(
            ewald_full,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )
        _, _, rod_half = ewald_allowed_reflections(
            ewald_half,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )
        rod_valid: Bool[Array, "N"] = (rod_indices >= 0) & (rod_full > 1e-12)
        self.assertGreater(int(jnp.sum(rod_valid)), 0)
        chex.assert_trees_all_close(
            rod_half[rod_valid],
            0.25 * rod_full[rod_valid],
            rtol=1e-12,
        )

    def test_grad_of_intensity_wrt_occupancy_is_finite_nonzero(self) -> None:
        r"""Check d(intensity)/d(occupancy) is finite and nonzero.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the summed
        allowed-reflection intensity is differentiable with respect to a
        site occupancy with a finite, nonzero gradient (the
        differentiability contract that the pre-fix integer-Z truncation
        broke).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The body also exercises differentiability, protecting JAX transform
        compatibility for this path.

        The documented check is rendered from
        ``tests.test_rheedium.test_simul.test_ewald``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        frac: Float[Array, "1 4"] = jnp.array([[0.0, 0.0, 0.0, 14.0]])
        cell_lengths: Float[Array, "3"] = jnp.array([3.0, 3.0, 3.0])
        cell_angles: Float[Array, "3"] = jnp.array([90.0, 90.0, 90.0])

        def summed_intensity(occupancy: scalar_float) -> scalar_float:
            crystal: CrystalStructure = create_crystal_structure(
                frac,
                frac,
                cell_lengths,
                cell_angles,
                occupancies=jnp.array([occupancy]),
            )
            ewald: EwaldData = build_ewald_data(
                crystal, energy_kev=20.0, hmax=2, kmax=2, lmax=2
            )
            _, _, intensities = ewald_allowed_reflections(
                ewald, theta_deg=2.0, phi_deg=0.0
            )
            return jnp.sum(intensities)

        value: scalar_float
        gradient: scalar_float
        value, gradient = jax.value_and_grad(summed_intensity)(0.5)
        self.assertTrue(bool(jnp.isfinite(gradient)))
        self.assertGreater(abs(float(gradient)), 0.0)
        chex.assert_trees_all_close(
            float(gradient), 4.0 * float(value) / 1.0, rtol=1e-10
        )
