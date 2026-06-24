"""Test suite for finite_domain.py broadening calculations.

This module provides comprehensive testing for finite domain Ewald sphere
broadening functions used in RHEED simulations. Tests verify rod width
calculations, shell thickness, overlap integrals, and integration with
existing Ewald infrastructure.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.simul.ewald import build_ewald_data
from rheedium.simul.finite_domain import (
    compute_domain_extent,
    compute_shell_sigma,
    extent_to_rod_sigma,
    finite_domain_intensities,
    finite_domain_intensities_for_size_distribution,
    rod_ewald_overlap,
)
from rheedium.types import (
    SizeDistribution,
    create_lognormal_size,
    size_to_distribution,
)
from rheedium.types.crystal_types import (
    CrystalStructure,
    create_crystal_structure,
)
from rheedium.types.custom_types import scalar_float


class TestComputeDomainExtent(chex.TestCase, parameterized.TestCase):
    """Test suite for compute_domain_extent function."""

    def setUp(self) -> None:
        """Set up test fixtures for domain extent calculations.

        Creates various atomic position configurations including single atom,
        symmetric cube, rectangular slab, and configurations requiring padding.
        """
        super().setUp()
        # Single atom at origin
        self.single_atom: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])

        # Cube of atoms 10×10×10 Å
        self.cube_positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 10.0, 10.0],
            ]
        )

        # Rectangular slab 20×15×5 Å
        self.slab_positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 15.0, 0.0],
                [0.0, 0.0, 5.0],
                [20.0, 15.0, 5.0],
            ]
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_atom_minimum_extent(self) -> None:
        """Test that single atom returns minimum extent (1.0 Å).

        A single atom has zero extent, but minimum enforcement should
        return [1.0, 1.0, 1.0] to avoid numerical issues.
        """
        var_compute: Callable[..., Any] = self.variant(compute_domain_extent)

        extent: Float[Array, "..."] = var_compute(
            self.single_atom, padding_ang=0.0
        )

        chex.assert_shape(extent, (3,))
        # Single atom has zero extent, but minimum is enforced
        chex.assert_trees_all_close(
            extent, jnp.array([1.0, 1.0, 1.0]), rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_cube_extent(self) -> None:
        """Test extent calculation for cubic arrangement.

        Atoms at corners of 10×10×10 Å cube should give extent [10, 10, 10].
        """
        var_compute: Callable[..., Any] = self.variant(compute_domain_extent)

        extent: Float[Array, "..."] = var_compute(
            self.cube_positions, padding_ang=0.0
        )

        chex.assert_shape(extent, (3,))
        chex.assert_trees_all_close(
            extent, jnp.array([10.0, 10.0, 10.0]), rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_slab_extent(self) -> None:
        """Test extent calculation for rectangular slab.

        Atoms in 20×15×5 Å slab should give corresponding extent.
        """
        var_compute: Callable[..., Any] = self.variant(compute_domain_extent)

        extent: Float[Array, "..."] = var_compute(
            self.slab_positions, padding_ang=0.0
        )

        chex.assert_shape(extent, (3,))
        chex.assert_trees_all_close(
            extent, jnp.array([20.0, 15.0, 5.0]), rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_padding", 1.0),
        ("medium_padding", 5.0),
        ("large_padding", 10.0),
    )
    def test_padding_applied_correctly(self, padding: float) -> None:
        """Test that padding is added correctly (2×padding per dimension).

        Padding should be applied symmetrically on both sides.
        """
        var_compute: Callable[..., Any] = self.variant(compute_domain_extent)
        extent_no_pad: Float[Array, "..."] = var_compute(
            self.cube_positions, padding_ang=0.0
        )
        extent_with_pad: Float[Array, "..."] = var_compute(
            self.cube_positions, padding_ang=padding
        )
        expected: Float[Array, "..."] = extent_no_pad + 2.0 * padding
        chex.assert_trees_all_close(extent_with_pad, expected, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_is_positive(self) -> None:
        """Test that extent is always positive."""
        var_compute: Callable[..., Any] = self.variant(compute_domain_extent)
        extent: Float[Array, "..."] = var_compute(
            self.single_atom, padding_ang=0.0
        )
        chex.assert_trees_all_equal(jnp.all(extent > 0), True)


class TestExtentToRodSigma(chex.TestCase, parameterized.TestCase):
    """Test suite for extent_to_rod_sigma function."""

    def setUp(self) -> None:
        """Set up test fixtures for rod sigma calculations.

        Creates domain extents ranging from small (10 Å) to large (1000 Å)
        to test the inverse scaling relationship.
        """
        super().setUp()
        self.small_extent: Float[Array, "..."] = jnp.array([10.0, 10.0, 10.0])
        self.medium_extent: Float[Array, "..."] = jnp.array(
            [100.0, 100.0, 100.0]
        )
        self.large_extent: Float[Array, "..."] = jnp.array(
            [1000.0, 1000.0, 1000.0]
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has shape (2,) for x,y rod widths."""
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        sigma: Float[Array, "..."] = var_sigma(self.medium_extent)

        chex.assert_shape(sigma, (2,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_inverse_scaling(self) -> None:
        """Test that rod sigma scales inversely with domain size.

        σ_rod ∝ 1/L, so doubling L should halve σ.
        """
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        sigma_10: Float[Array, "..."] = var_sigma(
            jnp.array([10.0, 10.0, 10.0])
        )
        sigma_100: Float[Array, "..."] = var_sigma(
            jnp.array([100.0, 100.0, 100.0])
        )

        # sigma_10 should be 10× larger than sigma_100
        ratio: Any = sigma_10 / sigma_100
        chex.assert_trees_all_close(ratio, jnp.array([10.0, 10.0]), rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_numerical_value_100A(self) -> None:  # noqa: N802
        """Test numerical value for 100 Å domain.

        σ = 2π/(L×√(2π)) = 2π/(100×2.507) ≈ 0.0251 Å⁻¹
        """
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        sigma: Float[Array, "..."] = var_sigma(self.medium_extent)

        expected: scalar_float = (
            2.0 * jnp.pi / (100.0 * jnp.sqrt(2.0 * jnp.pi))
        )
        chex.assert_trees_all_close(
            sigma, jnp.array([expected, expected]), rtol=1e-6
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_positive_output(self) -> None:
        """Test that rod sigma is always positive."""
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        sigma: Float[Array, "..."] = var_sigma(self.small_extent)

        chex.assert_trees_all_equal(jnp.all(sigma > 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_tiny_extent_no_nan(self) -> None:
        """Test that tiny extent gives no NaN via minimum enforcement."""
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        tiny_extent: Float[Array, "..."] = jnp.array([0.1, 0.1, 0.1])
        sigma: Float[Array, "..."] = var_sigma(tiny_extent)

        chex.assert_tree_all_finite(sigma)


class TestComputeShellSigma(chex.TestCase, parameterized.TestCase):
    """Test suite for compute_shell_sigma function."""

    def setUp(self) -> None:
        """Set up test fixtures for shell sigma calculations.

        Creates wavevector magnitudes for common RHEED voltages.
        """
        super().setUp()
        # k = 2π/λ for various voltages
        # λ ≈ 0.086 Å at 20 kV → k ≈ 73 Å⁻¹
        self.k_15kv: Float[Array, "..."] = jnp.array(63.0)  # approximate
        self.k_20kv: Float[Array, "..."] = jnp.array(73.0)  # approximate
        self.k_30kv: Float[Array, "..."] = jnp.array(89.0)  # approximate

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_is_scalar(self) -> None:
        """Test that output is a scalar."""
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        sigma: Float[Array, "..."] = var_shell(self.k_20kv)

        chex.assert_shape(sigma, ())

    @chex.variants(with_jit=True, without_jit=True)
    def test_default_parameters(self) -> None:
        """Test shell sigma with default beam parameters.

        At 20 kV (k≈73), with ΔE/E=1e-4 and Δθ=1e-3:
        σ_shell = 73 × √[(5e-5)² + (1e-3)²] ≈ 73 × 1e-3 ≈ 0.073 Å⁻¹
        """
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        sigma: Float[Array, "..."] = var_shell(self.k_20kv)

        # Divergence dominates: σ ≈ k × Δθ = 73 × 0.001 = 0.073
        chex.assert_trees_all_close(sigma, 0.073, atol=0.01)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_divergence_energy_only(self) -> None:
        """Test shell sigma with only energy spread (zero divergence)."""
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        sigma: Float[Array, "..."] = var_shell(
            self.k_20kv, energy_spread_frac=1e-4, beam_divergence_rad=0.0
        )

        # σ = k × (ΔE/2E) = 73 × 5e-5 = 0.00365
        expected: Float[Array, "..."] = 73.0 * (1e-4 / 2.0)
        chex.assert_trees_all_close(sigma, expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_scaling_with_k(self) -> None:
        """Test that shell sigma scales linearly with k."""
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        sigma_15: Float[Array, "..."] = var_shell(self.k_15kv)
        sigma_30: Float[Array, "..."] = var_shell(self.k_30kv)

        # σ ∝ k, so ratio should equal k ratio
        expected_ratio: Float[Array, "..."] = float(self.k_30kv / self.k_15kv)
        actual_ratio: Any = float(sigma_30 / sigma_15)
        chex.assert_trees_all_close(actual_ratio, expected_ratio, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_positive_output(self) -> None:
        """Test that shell sigma is always positive."""
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        sigma: Float[Array, "..."] = var_shell(self.k_20kv)

        chex.assert_scalar_positive(float(sigma))


class TestRodEwaldOverlap(chex.TestCase, parameterized.TestCase):
    """Test suite for rod_ewald_overlap function."""

    def setUp(self) -> None:
        """Set up test fixtures for overlap calculations.

        Creates test G vectors, incident wavevector, and broadening parameters.
        """
        super().setUp()
        # Set up a simple scattering geometry
        # k_in pointing at grazing angle
        self.k_magnitude: Float[Array, "..."] = jnp.array(73.0)  # ~20 kV
        theta_rad: scalar_float = jnp.deg2rad(2.0)  # 2° grazing
        self.k_in: Float[Array, "..."] = self.k_magnitude * jnp.array(
            [jnp.cos(theta_rad), 0.0, -jnp.sin(theta_rad)]
        )

        # G vector that satisfies Ewald condition (roughly)
        # k_out = k_in + G should have |k_out| ≈ |k_in|
        self.g_on_sphere: Float[Array, "..."] = jnp.array(
            [[0.0, 0.0, 2.0 * self.k_magnitude * jnp.sin(theta_rad)]]
        )

        # G vector far from Ewald sphere
        self.g_off_sphere: Float[Array, "..."] = jnp.array(
            [[10.0, 10.0, 10.0]]
        )

        # Multiple G vectors for batch testing
        self.g_batch: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Specular
                [1.0, 0.0, 0.0],  # Off sphere
                [0.0, 1.0, 0.0],  # Off sphere
            ]
        )

        # Broadening parameters
        self.rod_sigma_large: Float[Array, "..."] = jnp.array(
            [0.5, 0.5]
        )  # Large broadening
        self.rod_sigma_small: Float[Array, "..."] = jnp.array(
            [0.01, 0.01]
        )  # Small broadening
        self.shell_sigma: Float[Array, "..."] = jnp.array(0.07)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shape_single(self) -> None:
        """Test output shape for single G vector."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap: Any = var_overlap(
            self.g_on_sphere,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        chex.assert_shape(overlap, (1,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shape_batch(self) -> None:
        """Test output shape for batch of G vectors."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap: Any = var_overlap(
            self.g_batch,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        chex.assert_shape(overlap, (3,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_specular_reflection_high_overlap(self) -> None:
        """Test that specular reflection (G=0) has high overlap.

        For G=0, k_out = k_in, so |k_out| = |k_in| exactly.
        Overlap should be 1.0.
        """
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        g_specular: Float[Array, "..."] = jnp.array([[0.0, 0.0, 0.0]])
        overlap: Any = var_overlap(
            g_specular,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        chex.assert_trees_all_close(overlap[0], 1.0, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_far_from_sphere_low_overlap(self) -> None:
        """Test that G vectors far from Ewald sphere have low overlap."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap: Any = var_overlap(
            self.g_off_sphere,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_small,  # Small σ → sharp cutoff
            self.shell_sigma,
        )

        # Should be very small
        chex.assert_trees_all_equal(overlap[0] < 0.01, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_overlap_bounded_zero_one(self) -> None:
        """Test that overlap values are bounded between 0 and 1."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap: Any = var_overlap(
            self.g_batch,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        chex.assert_trees_all_equal(jnp.all(overlap >= 0), True)
        chex.assert_trees_all_equal(jnp.all(overlap <= 1), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_larger_sigma_broader_overlap(self) -> None:
        """Test that larger σ gives broader (more uniform) overlap."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap_small: Any = var_overlap(
            self.g_off_sphere,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_small,
            self.shell_sigma,
        )
        overlap_large: Any = var_overlap(
            self.g_off_sphere,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        # Larger σ should give larger overlap for off-sphere points
        chex.assert_trees_all_equal(overlap_large[0] > overlap_small[0], True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_finite(self) -> None:
        """Test that output contains no NaN or Inf."""
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        overlap: Any = var_overlap(
            self.g_batch,
            self.k_in,
            self.k_magnitude,
            self.rod_sigma_large,
            self.shell_sigma,
        )

        chex.assert_tree_all_finite(overlap)


class TestFiniteDomainIntensities(chex.TestCase, parameterized.TestCase):
    """Test suite for finite_domain_intensities function."""

    def setUp(self) -> None:
        """Set up test fixtures for intensity calculations.

        Creates a simple crystal structure and pre-computes EwaldData.
        """
        super().setUp()
        # Simple cubic crystal (MgO-like)
        self.cell_lengths: Float[Array, "..."] = jnp.array([4.21, 4.21, 4.21])
        self.cell_angles: Float[Array, "..."] = jnp.array([90.0, 90.0, 90.0])

        # Atoms: Mg at (0,0,0), O at (0.5,0.5,0.5)
        self.frac_positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 12.0],  # Mg
                [0.5, 0.5, 0.5, 8.0],  # O
            ]
        )

        # Cartesian positions
        self.cart_positions: Float[Array, "..."] = jnp.array(
            [
                [0.0, 0.0, 0.0, 12.0],
                [2.105, 2.105, 2.105, 8.0],
            ]
        )

        self.crystal: CrystalStructure = create_crystal_structure(
            frac_positions=self.frac_positions,
            cart_positions=self.cart_positions,
            cell_lengths=self.cell_lengths,
            cell_angles=self.cell_angles,
        )

        # Build EwaldData
        self.ewald: Any = build_ewald_data(
            crystal=self.crystal,
            voltage_kv=15.0,
            hmax=2,
            kmax=2,
            lmax=1,
            temperature=300.0,
        )

        # Domain extents
        self.small_domain: Float[Array, "..."] = jnp.array([20.0, 20.0, 10.0])
        self.large_domain: Float[Array, "..."] = jnp.array(
            [1000.0, 1000.0, 500.0]
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shapes(self) -> None:
        """Test that output shapes match EwaldData.intensities."""
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap: Any
        intensities: Float[Array, "..."]
        overlap, intensities = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )

        chex.assert_shape(overlap, self.ewald.intensities.shape)
        chex.assert_shape(intensities, self.ewald.intensities.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_overlap_bounded(self) -> None:
        """Test that overlap factors are in [0, 1]."""
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap: Any
        overlap, _ = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )

        chex.assert_trees_all_equal(jnp.all(overlap >= 0), True)
        chex.assert_trees_all_equal(jnp.all(overlap <= 1), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_modified_intensities_bounded(self) -> None:
        """Test that modified intensities ≤ original intensities."""
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        modified: Any
        _, modified = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )

        # I_modified = I_base × overlap ≤ I_base (since overlap ≤ 1)
        chex.assert_trees_all_equal(
            jnp.all(modified <= self.ewald.intensities + 1e-10), True
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_large_domain_preserves_intensities(self) -> None:
        """Test that large domain gives overlap ≈ 1 for allowed reflections.

        For a very large domain (1000 Å), the rod width is very small,
        and only reflections exactly on the Ewald sphere should have
        significant overlap. The specular (0,0,0) should always have
        overlap = 1.0.
        """
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap: Any
        overlap, _ = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.large_domain,
        )

        # Find the specular reflection (0,0,0) index
        # It should have the highest overlap (1.0)
        max_overlap: scalar_float = jnp.max(overlap)
        chex.assert_trees_all_close(max_overlap, 1.0, rtol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_small_domain_broader_distribution(self) -> None:
        """Test that small domain gives more uniform overlap distribution.

        Smaller domains have broader rods, so more reflections contribute.
        The overlap distribution should be "flatter" than for large domains.
        """
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap_small: Any
        overlap_small, _ = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )
        overlap_large: Any
        overlap_large, _ = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.large_domain,
        )

        # Small domain should have more "active" reflections
        # Count reflections with overlap > 0.1
        active_small: scalar_float = jnp.sum(overlap_small > 0.1)
        active_large: scalar_float = jnp.sum(overlap_large > 0.1)

        chex.assert_trees_all_equal(active_small >= active_large, True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_finite(self) -> None:
        """Test that output contains no NaN or Inf."""
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap: Any
        intensities: Float[Array, "..."]
        overlap, intensities = var_intensities(
            self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )

        chex.assert_tree_all_finite(overlap)
        chex.assert_tree_all_finite(intensities)

    def test_size_distribution_delta_matches_single_domain_extent(
        self,
    ) -> None:
        """Delta size distribution reproduces one finite-domain evaluation."""
        size_dist: SizeDistribution = SizeDistribution(
            distribution_type="delta",
            mean_ang=jnp.asarray(40.0, dtype=jnp.float64),
            sigma_ang=jnp.asarray(0.0, dtype=jnp.float64),
            min_size_ang=jnp.asarray(10.0, dtype=jnp.float64),
            max_size_ang=jnp.asarray(80.0, dtype=jnp.float64),
        )
        aspect_ratio: tuple[float, float, float] = (1.0, 1.0, 0.25)
        expected_overlap: Float[Array, "N"]
        expected_intensities: Float[Array, "N"]
        expected_overlap, expected_intensities = finite_domain_intensities(
            ewald=self.ewald,
            theta_deg=2.0,
            phi_deg=0.0,
            domain_extent_ang=jnp.array([40.0, 40.0, 10.0]),
        )

        actual_overlap: Float[Array, "N"]
        actual_intensities: Float[Array, "N"]
        actual_overlap, actual_intensities = (
            finite_domain_intensities_for_size_distribution(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                size_distribution=size_dist,
                domain_aspect_ratio=aspect_ratio,
                n_size_points=5,
            )
        )

        chex.assert_trees_all_close(
            actual_overlap,
            expected_overlap,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            actual_intensities,
            expected_intensities,
            atol=1e-12,
        )

    def test_size_distribution_matches_manual_weighted_sum(self) -> None:
        """SizeDistribution bridge equals explicit incoherent averaging."""
        size_dist: SizeDistribution = create_lognormal_size(
            mean_ang=80.0,
            sigma_ang=12.0,
            min_size_ang=40.0,
            max_size_ang=140.0,
        )
        size_axis = size_to_distribution(size_dist, n_points=3)
        aspect_ratio: tuple[float, float, float] = (1.0, 1.0, 0.5)
        aspect_array: Float[Array, "3"] = jnp.array(aspect_ratio)

        def _manual_size_sample(
            sample: Float[Array, "1"],
        ) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
            return finite_domain_intensities(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                domain_extent_ang=sample[0] * aspect_array,
            )

        overlap_bank: Float[Array, "S N"]
        intensity_bank: Float[Array, "S N"]
        overlap_bank, intensity_bank = jax.vmap(_manual_size_sample)(
            size_axis.samples
        )
        expected_overlap: Float[Array, "N"] = jnp.einsum(
            "s,sn->n",
            size_axis.weights,
            overlap_bank,
        )
        expected_intensities: Float[Array, "N"] = jnp.einsum(
            "s,sn->n",
            size_axis.weights,
            intensity_bank,
        )
        actual_overlap: Float[Array, "N"]
        actual_intensities: Float[Array, "N"]
        actual_overlap, actual_intensities = (
            finite_domain_intensities_for_size_distribution(
                ewald=self.ewald,
                theta_deg=2.0,
                phi_deg=0.0,
                size_distribution=size_dist,
                domain_aspect_ratio=aspect_ratio,
                n_size_points=3,
            )
        )

        chex.assert_trees_all_close(
            actual_overlap,
            expected_overlap,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            actual_intensities,
            expected_intensities,
            atol=1e-12,
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("theta_1deg", 1.0),
        ("theta_2deg", 2.0),
        ("theta_5deg", 5.0),
    )
    def test_different_angles(self, theta: float) -> None:
        """Test that function works for various incidence angles."""
        var_intensities: Callable[..., Any] = self.variant(
            finite_domain_intensities
        )

        overlap: Any
        intensities: Float[Array, "..."]
        overlap, intensities = var_intensities(
            self.ewald,
            theta_deg=theta,
            phi_deg=0.0,
            domain_extent_ang=self.small_domain,
        )

        chex.assert_tree_all_finite(overlap)
        chex.assert_tree_all_finite(intensities)
        chex.assert_trees_all_equal(jnp.all(overlap >= 0), True)


class TestPhysicsValidation(chex.TestCase, parameterized.TestCase):
    """Physics validation tests for finite domain broadening."""

    def setUp(self) -> None:
        """Set up fixtures for physics validation."""
        super().setUp()
        # Reference values
        self.sqrt_2pi: scalar_float = jnp.sqrt(2.0 * jnp.pi)

    @chex.variants(with_jit=True, without_jit=True)
    def test_rod_sigma_formula(self) -> None:
        """Test that rod sigma formula is correct.

        σ_q = 2π / (L × √(2π))

        For L = 100 Å:
        σ_q = 2π / (100 × 2.5066) = 6.283 / 250.66 ≈ 0.0251 Å⁻¹
        """
        var_sigma: Callable[..., Any] = self.variant(extent_to_rod_sigma)

        L: float = 100.0
        sigma: Float[Array, "..."] = var_sigma(jnp.array([L, L, L]))

        expected: scalar_float = 2.0 * jnp.pi / (L * self.sqrt_2pi)
        chex.assert_trees_all_close(sigma[0], expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_shell_sigma_formula(self) -> None:
        """Test that shell sigma formula is correct.

        σ_shell = k × √[(ΔE/2E)² + Δθ²]

        For k = 73 Å⁻¹, ΔE/E = 1e-4, Δθ = 1e-3:
        σ_shell = 73 × √[(5e-5)² + (1e-3)²]
               = 73 × √[2.5e-9 + 1e-6]
               = 73 × √[1.0025e-6]
               ≈ 73 × 1.001e-3 ≈ 0.073 Å⁻¹
        """
        var_shell: Callable[..., Any] = self.variant(compute_shell_sigma)

        k: scalar_float = jnp.array(73.0)
        dE_E: float = 1e-4
        dtheta: float = 1e-3

        sigma: Float[Array, "..."] = var_shell(
            k, energy_spread_frac=dE_E, beam_divergence_rad=dtheta
        )

        expected: scalar_float = k * jnp.sqrt((dE_E / 2.0) ** 2 + dtheta**2)
        chex.assert_trees_all_close(sigma, expected, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gaussian_overlap_formula(self) -> None:
        """Test that overlap follows Gaussian formula.

        overlap = exp(-d²/(2σ_eff²))
        where σ_eff² = σ_rod² + σ_shell²
        """
        var_overlap: Callable[..., Any] = self.variant(rod_ewald_overlap)

        # Set up geometry where we can calculate d analytically
        k: scalar_float = jnp.array(73.0)
        k_in: Float[Array, "..."] = jnp.array([k, 0.0, 0.0])  # Along x

        # G that gives k_out with different magnitude
        delta_k: float = 0.1  # Deviation from elastic condition
        g: Float[Array, "..."] = jnp.array(
            [[delta_k, 0.0, 0.0]]
        )  # k_out = [k + delta, 0, 0]

        rod_sigma: Float[Array, "..."] = jnp.array([0.05, 0.05])
        shell_sigma: scalar_float = jnp.array(0.07)

        overlap: Any = var_overlap(g, k_in, k, rod_sigma, shell_sigma)

        # Calculate expected value
        # k_out = k_in + g = [k + delta, 0, 0]
        # |k_out| = k + delta
        # d = ||k_out| - k| = delta
        d: Any = delta_k
        sigma_eff_sq: Float[Array, "..."] = (
            0.05**2 + 0.07**2
        )  # Simplified: use mean rod sigma
        expected: Float[Array, "..."] = jnp.exp(-(d**2) / (2.0 * sigma_eff_sq))

        chex.assert_trees_all_close(overlap[0], expected, rtol=1e-3)
