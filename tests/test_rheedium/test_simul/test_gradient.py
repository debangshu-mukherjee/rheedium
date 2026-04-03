"""Gradient verification tests for the RHEED forward model.

Verifies that jax.grad produces finite, non-zero gradients through
the entire forward model pipeline: form factors, Debye-Waller,
CTR intensity, ewald_simulator, and multislice_simulator.

Gradient correctness is verified by comparing analytical gradients
(via jax.grad / reverse-mode AD) against numerical finite-difference
approximations using jax.test_util.check_grads. This catches bugs
where a function is differentiable and returns finite gradients that
are nonetheless wrong (e.g. missing negative sign, incorrect chain
rule in a custom operation).

These tests establish the differentiability baseline required for
inverse problems on experimental RHEED data.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.test_util import check_grads
from jaxtyping import Array, Float, Int

from rheedium.simul import (
    calculate_ctr_intensity,
    ewald_simulator,
    multislice_propagate,
    sliced_crystal_to_potential,
    wavelength_ang,
)
from rheedium.simul.form_factors import (
    debye_waller_factor,
    get_mean_square_displacement,
    kirkland_form_factor,
)
from rheedium.types import (
    CrystalStructure,
    RHEEDPattern,
    create_crystal_structure,
    create_sliced_crystal,
    scalar_float,
)


def _jax_safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap fn to convert numpy scalar args to JAX arrays.

    jax.test_util.check_grads perturbs inputs via numpy arithmetic,
    producing numpy scalars that fail beartype's Float[Array, '']
    checks. This wrapper ensures all scalar arguments are converted
    to JAX arrays before reaching the decorated function.
    """

    def wrapper(*args: Any) -> Any:
        return fn(*(jnp.asarray(a) for a in args))

    return wrapper


def _make_si_crystal() -> CrystalStructure:
    """Create a 2-atom Si crystal for fast gradient tests."""
    a_si: float = 5.431
    frac_coords: Float[Array, "2 3"] = jnp.array(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    )
    cart_coords: Float[Array, "2 3"] = frac_coords * a_si
    atomic_numbers: Float[Array, "2"] = jnp.full(2, 14.0)
    frac_positions: Float[Array, "2 4"] = jnp.column_stack(
        [frac_coords, atomic_numbers]
    )
    cart_positions: Float[Array, "2 4"] = jnp.column_stack(
        [cart_coords, atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=jnp.array([a_si, a_si, a_si]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


SI_CRYSTAL: CrystalStructure = _make_si_crystal()


class TestFormFactorGradients(chex.TestCase, parameterized.TestCase):
    """Gradient tests for atomic form factor functions."""

    def test_form_factor_grad_q(self) -> None:
        """Kirkland form factor gradient w.r.t. q is finite and smooth."""

        def f(q: scalar_float) -> scalar_float:
            return jnp.squeeze(kirkland_form_factor(14, q))

        grad_fn = jax.grad(f)
        q_values = [0.5, 1.0, 2.0, 4.0]
        for q in q_values:
            g = grad_fn(jnp.float64(q))
            chex.assert_tree_all_finite(g)
            self.assertTrue(
                jnp.abs(g) > 1e-12,
                f"Form factor gradient should be non-zero at q={q}",
            )

    def test_debye_waller_grad_temperature(self) -> None:
        """DW factor gradient w.r.t. temperature is negative.

        Increasing temperature increases MSD, which reduces the
        Debye-Waller factor (more thermal damping).
        """

        def dw_at_temp(temp: scalar_float) -> scalar_float:
            msd = get_mean_square_displacement(
                atomic_number=14,
                temperature=temp,
            )
            return debye_waller_factor(
                q_magnitude=jnp.float64(2.0),
                mean_square_displacement=msd,
            )

        grad_fn = jax.grad(dw_at_temp)
        g = grad_fn(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        self.assertTrue(
            g < 0,
            "DW gradient w.r.t. temperature should be negative",
        )

    def test_msd_grad_temperature(self) -> None:
        """Mean square displacement gradient w.r.t. temperature is positive.

        Higher temperature means more atomic vibration.
        """

        def msd_fn(temp: scalar_float) -> scalar_float:
            return get_mean_square_displacement(
                atomic_number=14, temperature=temp
            )

        grad_fn = jax.grad(msd_fn)
        g = grad_fn(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        self.assertTrue(
            g > 0,
            "MSD gradient w.r.t. temperature should be positive",
        )


class TestCTRIntensityGradients(chex.TestCase, parameterized.TestCase):
    """Gradient tests for CTR intensity calculations."""

    def test_ctr_intensity_grad_roughness(self) -> None:
        """CTR intensity gradient w.r.t. roughness is finite."""
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        q_z: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def loss(roughness: scalar_float) -> scalar_float:
            intensities = calculate_ctr_intensity(
                hk_indices=hk_indices,
                q_z=q_z,
                crystal=SI_CRYSTAL,
                surface_roughness=roughness,
                temperature=300.0,
            )
            return jnp.sum(intensities)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        self.assertTrue(
            jnp.abs(g) > 1e-12,
            "CTR intensity gradient w.r.t. roughness should be non-zero",
        )

    def test_ctr_intensity_grad_temperature(self) -> None:
        """CTR intensity gradient w.r.t. temperature is finite."""
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        q_z: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def loss(temp: scalar_float) -> scalar_float:
            intensities = calculate_ctr_intensity(
                hk_indices=hk_indices,
                q_z=q_z,
                crystal=SI_CRYSTAL,
                surface_roughness=0.5,
                temperature=temp,
            )
            return jnp.sum(intensities)

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)


class TestEwaldSimulatorGradients(chex.TestCase, parameterized.TestCase):
    """Gradient tests for the ewald_simulator forward model."""

    def _ewald_loss(self, **override) -> scalar_float:
        """Compute sum of intensities from ewald_simulator."""
        defaults = dict(
            crystal=SI_CRYSTAL,
            voltage_kv=20.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=2,
            kmax=2,
            temperature=300.0,
            surface_roughness=0.5,
        )
        defaults.update(override)
        pattern: RHEEDPattern = ewald_simulator(**defaults)
        return jnp.sum(pattern.intensities)

    def test_grad_temperature(self) -> None:
        """Gradient w.r.t. temperature is finite and non-zero."""

        def loss(temp: scalar_float) -> scalar_float:
            return self._ewald_loss(temperature=temp)

        g = jax.grad(loss)(jnp.float64(300.0))
        chex.assert_tree_all_finite(g)
        self.assertTrue(
            jnp.abs(g) > 1e-12,
            "Gradient w.r.t. temperature should be non-zero",
        )

    def test_grad_roughness(self) -> None:
        """Gradient w.r.t. surface roughness is finite and non-zero."""

        def loss(roughness: scalar_float) -> scalar_float:
            return self._ewald_loss(surface_roughness=roughness)

        g = jax.grad(loss)(jnp.float64(0.5))
        chex.assert_tree_all_finite(g)
        self.assertTrue(
            jnp.abs(g) > 1e-12,
            "Gradient w.r.t. roughness should be non-zero",
        )

    def test_grad_polar_angle(self) -> None:
        """Gradient w.r.t. incidence angle is finite."""

        def loss(theta: scalar_float) -> scalar_float:
            return self._ewald_loss(theta_deg=theta)

        g = jax.grad(loss)(jnp.float64(2.0))
        chex.assert_tree_all_finite(g)

    def test_grad_voltage(self) -> None:
        """Gradient w.r.t. beam voltage is finite."""

        def loss(voltage: scalar_float) -> scalar_float:
            return self._ewald_loss(voltage_kv=voltage)

        g = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)

    def test_vmap_grad(self) -> None:
        """vmap(grad(loss)) over temperatures produces correct shape."""

        def loss(temp: scalar_float) -> scalar_float:
            return self._ewald_loss(temperature=temp)

        grad_fn = jax.grad(loss)
        batch_grad = jax.vmap(grad_fn)
        temps: Float[Array, "3"] = jnp.array([100.0, 300.0, 600.0])
        grads: Float[Array, "3"] = batch_grad(temps)
        chex.assert_shape(grads, (3,))
        chex.assert_tree_all_finite(grads)

    def test_jacrev(self) -> None:
        """jacrev w.r.t. (temperature, roughness) produces (2,) Jacobian."""

        def loss(params: Float[Array, "2"]) -> scalar_float:
            return self._ewald_loss(
                temperature=params[0],
                surface_roughness=params[1],
            )

        jac_fn = jax.jacrev(loss)
        params: Float[Array, "2"] = jnp.array([300.0, 0.5])
        jac: Float[Array, "2"] = jac_fn(params)
        chex.assert_shape(jac, (2,))
        chex.assert_tree_all_finite(jac)


class TestMultisliceGradients(chex.TestCase, parameterized.TestCase):
    """Gradient tests for multislice forward model."""

    def test_multislice_grad_voltage(self) -> None:
        """Gradient through multislice propagation w.r.t. voltage."""
        cart_positions: Float[Array, "2 4"] = jnp.array(
            [
                [5.0, 5.0, 1.0, 14.0],
                [7.5, 7.5, 3.0, 14.0],
            ]
        )
        sliced = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def loss(voltage: scalar_float) -> scalar_float:
            potential = sliced_crystal_to_potential(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
                voltage_kv=voltage,
            )
            psi_exit = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        g = jax.grad(loss)(jnp.float64(20.0))
        chex.assert_tree_all_finite(g)


class TestGradientCorrectness(chex.TestCase, parameterized.TestCase):
    """Verify analytical gradients match finite differences.

    These tests catch silent gradient bugs where jax.grad returns
    finite values that are numerically wrong (e.g. sign error,
    missing chain rule term). Uses jax.test_util.check_grads
    which compares reverse-mode AD against central finite
    differences at the specified order.
    """

    def test_form_factor_grad_correct(self) -> None:
        """Kirkland form factor analytical grad matches finite diff."""

        def f(q: scalar_float) -> scalar_float:
            return jnp.squeeze(kirkland_form_factor(14, q))

        check_grads(_jax_safe(f), (jnp.float64(2.0),), order=1, atol=1e-3)

    def test_debye_waller_grad_correct(self) -> None:
        """DW factor analytical grad matches finite diff."""

        def f(temp: scalar_float) -> scalar_float:
            msd = get_mean_square_displacement(
                atomic_number=14, temperature=temp
            )
            return debye_waller_factor(
                q_magnitude=jnp.float64(2.0),
                mean_square_displacement=msd,
            )

        check_grads(_jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-3)

    def test_msd_grad_correct(self) -> None:
        """Mean square displacement analytical grad matches finite diff."""

        def f(temp: scalar_float) -> scalar_float:
            return get_mean_square_displacement(
                atomic_number=14, temperature=temp
            )

        check_grads(_jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-4)

    def test_ctr_intensity_grad_roughness_correct(self) -> None:
        """CTR intensity grad w.r.t. roughness matches finite diff."""
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        q_z: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(roughness: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    q_z=q_z,
                    crystal=SI_CRYSTAL,
                    surface_roughness=roughness,
                    temperature=300.0,
                )
            )

        check_grads(_jax_safe(f), (jnp.float64(0.5),), order=1, atol=1e-3)

    def test_ctr_intensity_grad_temperature_correct(self) -> None:
        """CTR intensity grad w.r.t. temperature matches finite diff."""
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        q_z: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(temp: scalar_float) -> scalar_float:
            return jnp.sum(
                calculate_ctr_intensity(
                    hk_indices=hk_indices,
                    q_z=q_z,
                    crystal=SI_CRYSTAL,
                    surface_roughness=0.5,
                    temperature=temp,
                )
            )

        check_grads(_jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-3)

    def test_ewald_simulator_grad_temperature_correct(self) -> None:
        """Ewald simulator grad w.r.t. temperature matches finite diff."""

        def f(temp: scalar_float) -> scalar_float:
            pattern = ewald_simulator(
                crystal=SI_CRYSTAL,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        check_grads(_jax_safe(f), (jnp.float64(300.0),), order=1, atol=1e-2)

    def test_ewald_simulator_grad_roughness_correct(self) -> None:
        """Ewald simulator grad w.r.t. roughness matches finite diff."""

        def f(roughness: scalar_float) -> scalar_float:
            pattern = ewald_simulator(
                crystal=SI_CRYSTAL,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=300.0,
                surface_roughness=roughness,
            )
            return jnp.sum(pattern.intensities)

        check_grads(_jax_safe(f), (jnp.float64(0.5),), order=1, atol=1e-2)

    def test_wavelength_grad_correct(self) -> None:
        """Relativistic wavelength grad matches finite diff to 2nd order."""

        def f(voltage: scalar_float) -> scalar_float:
            return wavelength_ang(voltage)

        check_grads(_jax_safe(f), (jnp.float64(20.0),), order=2, atol=1e-4)

    def test_multislice_grad_voltage_correct(self) -> None:
        """Multislice propagation grad w.r.t. voltage matches finite diff."""
        cart_positions: Float[Array, "2 4"] = jnp.array(
            [
                [5.0, 5.0, 1.0, 14.0],
                [7.5, 7.5, 3.0, 14.0],
            ]
        )
        sliced = create_sliced_crystal(
            cart_positions=cart_positions,
            cell_lengths=jnp.array([15.0, 15.0, 5.0]),
            cell_angles=jnp.array([90.0, 90.0, 90.0]),
            orientation=jnp.array([0, 0, 1]),
            depth=5.0,
            x_extent=15.0,
            y_extent=15.0,
        )

        def f(voltage: scalar_float) -> scalar_float:
            potential = sliced_crystal_to_potential(
                sliced,
                slice_thickness=2.0,
                pixel_size=0.5,
                voltage_kv=voltage,
            )
            psi_exit = multislice_propagate(
                potential,
                voltage_kv=voltage,
                theta_deg=2.0,
            )
            return jnp.sum(jnp.abs(psi_exit) ** 2)

        check_grads(_jax_safe(f), (jnp.float64(20.0),), order=1, atol=1e-2)


class TestVmapConsistency(chex.TestCase, parameterized.TestCase):
    """Verify vmap produces results consistent with sequential eval.

    This is critical for beam_averaging where angular_divergence_average
    relies on vmap producing identical results to sequential evaluation.
    Divergence between the two means the weighted sum is silently wrong.
    """

    def test_form_factor_vmap_consistent(self) -> None:
        """Batched form factor matches sequential per-element result."""

        def f(q: scalar_float) -> scalar_float:
            return jnp.squeeze(kirkland_form_factor(14, q))

        q_batch: Float[Array, "4"] = jnp.array([0.5, 1.0, 2.0, 4.0])
        batched: Float[Array, "4"] = jax.vmap(f)(q_batch)
        sequential: Float[Array, "4"] = jnp.stack([f(q) for q in q_batch])
        chex.assert_trees_all_close(batched, sequential, atol=1e-6)

    def test_ctr_intensity_vmap_consistent(self) -> None:
        """Batched CTR intensity matches sequential evaluation."""
        hk_indices: Int[Array, "1 2"] = jnp.array([[1, 0]])
        q_z: Float[Array, "5"] = jnp.linspace(0.5, 3.0, 5)

        def f(roughness: scalar_float) -> Float[Array, "1 5"]:
            return calculate_ctr_intensity(
                hk_indices=hk_indices,
                q_z=q_z,
                crystal=SI_CRYSTAL,
                surface_roughness=roughness,
                temperature=300.0,
            )

        roughness_batch: Float[Array, "3"] = jnp.array([0.1, 0.5, 1.0])
        batched: Float[Array, "3 1 5"] = jax.vmap(f)(roughness_batch)
        sequential: Float[Array, "3 1 5"] = jnp.stack(
            [f(r) for r in roughness_batch]
        )
        chex.assert_trees_all_close(batched, sequential, atol=1e-6)

    def test_debye_waller_vmap_consistent(self) -> None:
        """Batched DW factor matches sequential evaluation."""

        def f(temp: scalar_float) -> scalar_float:
            msd = get_mean_square_displacement(
                atomic_number=14, temperature=temp
            )
            return debye_waller_factor(
                q_magnitude=jnp.float64(2.0),
                mean_square_displacement=msd,
            )

        temp_batch: Float[Array, "4"] = jnp.array(
            [100.0, 300.0, 600.0, 1000.0]
        )
        batched: Float[Array, "4"] = jax.vmap(f)(temp_batch)
        sequential: Float[Array, "4"] = jnp.stack([f(t) for t in temp_batch])
        chex.assert_trees_all_close(batched, sequential, atol=1e-6)

    def test_ewald_simulator_vmap_temperature_consistent(self) -> None:
        """Batched ewald_simulator over temps matches sequential."""

        def f(temp: scalar_float) -> scalar_float:
            pattern = ewald_simulator(
                crystal=SI_CRYSTAL,
                voltage_kv=20.0,
                theta_deg=2.0,
                phi_deg=0.0,
                hmax=2,
                kmax=2,
                temperature=temp,
                surface_roughness=0.5,
            )
            return jnp.sum(pattern.intensities)

        temp_batch: Float[Array, "3"] = jnp.array([100.0, 300.0, 600.0])
        batched: Float[Array, "3"] = jax.vmap(f)(temp_batch)
        sequential: Float[Array, "3"] = jnp.stack([f(t) for t in temp_batch])
        chex.assert_trees_all_close(batched, sequential, atol=1e-4)

    def test_wavelength_vmap_consistent(self) -> None:
        """Batched wavelength matches sequential evaluation."""
        voltages: Float[Array, "4"] = jnp.array([10.0, 20.0, 30.0, 50.0])
        batched: Float[Array, "4"] = jax.vmap(wavelength_ang)(voltages)
        sequential: Float[Array, "4"] = jnp.stack(
            [wavelength_ang(v) for v in voltages]
        )
        chex.assert_trees_all_close(batched, sequential, atol=1e-8)
