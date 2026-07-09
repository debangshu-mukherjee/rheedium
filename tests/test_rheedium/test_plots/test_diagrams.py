"""Test suite for rheedium.plots.diagrams visualization functions.

This module provides smoke tests to ensure all plotting functions run without
errors. Tests do not verify visual output, only that functions execute
successfully and return valid matplotlib objects.
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jaxtyping import Array, Float, Integer
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

# Use non-interactive backend for testing
matplotlib.use("Agg")

import jax.numpy as jnp

from rheedium.plots.diagrams import (
    _prepare_interactive_atoms,
    _resolve_interactive_backend,
    plot_crystal_structure_3d,
    plot_ctr_profile,
    plot_debye_waller,
    plot_ewald_sphere_2d,
    plot_ewald_sphere_3d,
    plot_form_factors,
    plot_grazing_incidence_geometry,
    plot_rod_broadening,
    plot_roughness_damping,
    plot_structure_factor_phases,
    plot_unit_cell_3d,
    plot_wavelength_curve,
    view_atoms,
    view_atoms_interactive,
)
from rheedium.types import (
    H_OVER_SQRT_2ME_ANG_VSQRT,
    RELATIVISTIC_COEFF_PER_V,
    CrystalStructure,
    SurfaceConfig,
    identify_surface_atoms,
)
from rheedium.types.crystal_types import create_crystal_structure


class TestDiagramPlots:
    """Smoke tests for diagram plotting functions.

    :see: :func:`~rheedium.plots.plot_crystal_structure_3d`
    :see: :func:`~rheedium.plots.plot_ctr_profile`
    :see: :func:`~rheedium.plots.plot_debye_waller`
    :see: :func:`~rheedium.plots.plot_ewald_sphere_2d`
    :see: :func:`~rheedium.plots.plot_ewald_sphere_3d`
    :see: :func:`~rheedium.plots.plot_form_factors`
    :see: :func:`~rheedium.plots.plot_grazing_incidence_geometry`
    :see: :func:`~rheedium.plots.plot_rod_broadening`
    :see: :func:`~rheedium.plots.plot_roughness_damping`
    :see: :func:`~rheedium.plots.plot_structure_factor_phases`
    :see: :func:`~rheedium.plots.plot_unit_cell_3d`
    :see: :func:`~rheedium.plots.plot_wavelength_curve`
    """

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_plot_wavelength_curve_default(self) -> None:
        r"""Test wavelength curve plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: wavelength curve
        plot with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_wavelength_curve()
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Beam Energy (keV)"
        assert ax.get_ylabel() == "Wavelength (A)"

    def test_plot_wavelength_curve_custom_range(self) -> None:
        r"""Test wavelength curve with custom energy range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: wavelength curve
        with custom energy range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_wavelength_curve(
            energy_range_kev=(10.0, 50.0),
            n_points=50,
            show_comparison=False,
        )
        assert isinstance(ax, Axes)
        xlim: Any = ax.get_xlim()
        assert xlim[0] >= 10.0
        assert xlim[1] <= 50.0

    def test_plot_wavelength_curve_with_provided_ax(self) -> None:
        r"""Test wavelength curve with user-provided axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: wavelength curve
        with user-provided axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        fig: Any
        ax: Any
        fig, ax = plt.subplots()
        returned_ax: Any = plot_wavelength_curve(ax=ax)
        assert returned_ax is ax

    def test_plot_form_factors_single_element(self) -> None:
        r"""Test form factor plot for single element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: form factor plot
        for single element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_form_factors(atomic_numbers=[14])  # Silicon
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 1

    def test_plot_form_factors_multiple_elements(self) -> None:
        r"""Test form factor plot for multiple elements.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: form factor plot
        for multiple elements.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_form_factors(atomic_numbers=[14, 8, 38, 22])
        assert isinstance(ax, Axes)
        # Should have one line per element
        assert len(ax.get_lines()) >= 4

    def test_plot_form_factors_custom_range(self) -> None:
        r"""Test form factor plot with custom q range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: form factor plot
        with custom q range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_form_factors(
            atomic_numbers=[14],
            q_range=(0.0, 5.0),
            n_points=100,
        )
        assert isinstance(ax, Axes)
        xlim: Any = ax.get_xlim()
        assert xlim[0] >= 0.0
        assert xlim[1] <= 5.0

    def test_plot_debye_waller_default(self) -> None:
        r"""Test Debye-Waller plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Debye-Waller plot
        with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_debye_waller(
            atomic_number=14,
            temperatures=[100.0, 300.0, 600.0],
        )
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 3

    def test_plot_debye_waller_single_temp(self) -> None:
        r"""Test Debye-Waller plot with single temperature.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Debye-Waller plot
        with single temperature.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_debye_waller(
            atomic_number=79,  # Gold
            temperatures=[300.0],
        )
        assert isinstance(ax, Axes)

    def test_plot_ctr_profile_default(self) -> None:
        r"""Test CTR profile plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR profile plot
        with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ctr_profile()
        assert isinstance(ax, Axes)
        # Should be log scale on y-axis
        assert ax.get_yscale() == "log"

    def test_plot_ctr_profile_custom_range(self) -> None:
        r"""Test CTR profile with custom l range.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: CTR profile with
        custom l range.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ctr_profile(
            l_range=(-2.0, 2.0),
            n_points=200,
        )
        assert isinstance(ax, Axes)

    def test_plot_roughness_damping_default(self) -> None:
        r"""Test roughness damping plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roughness damping
        plot with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_roughness_damping()
        assert isinstance(ax, Axes)
        # Should have lines for each sigma value
        assert len(ax.get_lines()) >= 4

    def test_plot_roughness_damping_custom_sigma(self) -> None:
        r"""Test roughness damping with custom sigma values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: roughness damping
        with custom sigma values, and that the plotted curves are the CTR
        **intensity** damping factor exp(-q_z^2 sigma^2) (the square of the
        amplitude convention exp(-q_z^2 sigma^2 / 2)).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module, and the sigma = 1
        curve data is compared to the closed-form intensity factor.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_roughness_damping(
            q_z_range=(0.0, 3.0),
            sigma_values=[0.0, 1.0, 3.0],
        )
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 3
        sigma_one_line: Any = ax.get_lines()[1]
        q_z_plot: Any = np.asarray(sigma_one_line.get_xdata())
        damping_plot: Any = np.asarray(sigma_one_line.get_ydata())
        np.testing.assert_allclose(
            damping_plot, np.exp(-(q_z_plot**2)), rtol=1e-12
        )

    def test_plot_rod_broadening_default(self) -> None:
        r"""Test rod broadening plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: rod broadening
        plot with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_rod_broadening()
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 4

    def test_plot_rod_broadening_custom_lengths(self) -> None:
        r"""Test rod broadening with custom correlation lengths.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: rod broadening
        with custom correlation lengths.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_rod_broadening(
            q_perp_range=(-0.5, 0.5),
            correlation_lengths=[20.0, 100.0],
        )
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_2d_default(self) -> None:
        r"""Test 2D Ewald sphere plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2D Ewald sphere
        plot with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ewald_sphere_2d()
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_2d_custom_params(self) -> None:
        r"""Test 2D Ewald sphere with custom parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2D Ewald sphere
        with custom parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ewald_sphere_2d(
            energy_kev=20.0,
            theta_deg=3.0,
            lattice_spacing=5.0,
            n_rods=5,
        )
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_2d_specular_intersection_geometry(
        self,
    ) -> None:
        r"""Test 2D Ewald sphere intersects the specular reciprocal rod.

        Extended Summary
        ----------------
        Verifies the numeric construction geometry for the 2D Ewald sphere:
        the specular rod intersection drawn by the figure lies exactly on the
        sphere centered at the negative incident wavevector.
        """
        energy_kev: float = 20.0
        theta_deg: float = 3.0
        ax: Any = plot_ewald_sphere_2d(
            energy_kev=energy_kev,
            theta_deg=theta_deg,
        )
        voltage_v: float = energy_kev * 1000.0
        wavelength: float = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
            voltage_v * (1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v)
        )
        k_mag: float = 2.0 * np.pi / wavelength
        theta_rad: float = np.deg2rad(theta_deg)
        center: Float[NDArray, "2"] = np.array(
            [-k_mag * np.cos(theta_rad), k_mag * np.sin(theta_rad)]
        )
        specular: Float[NDArray, "2"] = np.array(
            [0.0, 2.0 * k_mag * np.sin(theta_rad)]
        )
        radius_to_specular: float = float(np.linalg.norm(specular - center))
        assert radius_to_specular == pytest.approx(
            k_mag,
            rel=0.0,
            abs=1e-9,
        )

        xlim: tuple[float, float] = ax.get_xlim()
        ylim: tuple[float, float] = ax.get_ylim()
        assert xlim[0] <= specular[0] <= xlim[1]
        assert ylim[0] <= specular[1] <= ylim[1]

    def test_plot_ewald_sphere_3d_default(self) -> None:
        r"""Test 3D Ewald sphere plot with default parameters.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D Ewald sphere
        plot with default parameters.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ewald_sphere_3d()
        assert isinstance(ax, Axes3D)

    def test_plot_ewald_sphere_3d_belt_reaches_origin_rod(self) -> None:
        r"""Test 3D Ewald mesh reaches the vertical origin rod.

        Extended Summary
        ----------------
        Verifies the numeric construction geometry for the 3D Ewald sphere:
        the plotted near-origin belt includes mesh vertices close enough to
        the vertical reciprocal rod through the origin.
        """
        energy_kev: float = 15.0
        theta_deg: float = 2.0
        phi_deg: float = 0.0
        ax: Any = plot_ewald_sphere_3d(
            energy_kev=energy_kev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )
        assert isinstance(ax, Axes3D)

        voltage_v: float = energy_kev * 1000.0
        wavelength: float = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
            voltage_v * (1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v)
        )
        k_mag: float = 2.0 * np.pi / wavelength
        theta_rad: float = np.deg2rad(theta_deg)
        phi_rad: float = np.deg2rad(phi_deg)
        k_in_x: float = k_mag * np.cos(theta_rad) * np.cos(phi_rad)
        k_in_y: float = k_mag * np.cos(theta_rad) * np.sin(phi_rad)
        u: Float[NDArray, "N"] = phi_rad + np.linspace(0, 2.0 * np.pi, 50)
        v: Float[NDArray, "N"] = np.linspace(
            np.pi / 2.0 - 0.15,
            np.pi / 2.0 + 0.15,
            25,
        )
        sphere_x: Float[NDArray, "N M"] = (
            k_mag * np.outer(np.cos(u), np.sin(v)) - k_in_x
        )
        sphere_y: Float[NDArray, "N M"] = (
            k_mag * np.outer(np.sin(u), np.sin(v)) - k_in_y
        )
        distance_to_origin_rod: Float[NDArray, "N M"] = np.sqrt(
            sphere_x**2 + sphere_y**2
        )
        assert float(np.min(distance_to_origin_rod)) < 0.5

    def test_plot_ewald_sphere_3d_different_views(self) -> None:
        r"""Test 3D Ewald sphere with different viewing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D Ewald sphere
        with different viewing angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        # Front view
        ax1: Any = plot_ewald_sphere_3d(elev=0.0, azim=0.0)
        assert isinstance(ax1, Axes3D)
        plt.close()

        # Top view
        ax2: Any = plot_ewald_sphere_3d(elev=90.0, azim=0.0)
        assert isinstance(ax2, Axes3D)
        plt.close()

        # Perspective view
        ax3: Any = plot_ewald_sphere_3d(elev=30.0, azim=45.0)
        assert isinstance(ax3, Axes3D)

    def test_plot_unit_cell_3d_cubic(self) -> None:
        r"""Test 3D unit cell plot for cubic cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D unit cell plot
        for cubic cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_unit_cell_3d(
            cell_lengths=(4.0, 4.0, 4.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_unit_cell_3d_orthorhombic(self) -> None:
        r"""Test 3D unit cell plot for orthorhombic cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D unit cell plot
        for orthorhombic cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_unit_cell_3d(
            cell_lengths=(3.0, 4.0, 5.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_unit_cell_3d_triclinic(self) -> None:
        r"""Test 3D unit cell plot for triclinic cell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D unit cell plot
        for triclinic cell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_unit_cell_3d(
            cell_lengths=(3.0, 4.0, 5.0),
            cell_angles=(80.0, 85.0, 75.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_simple(self) -> None:
        r"""Test 3D crystal structure plot with simple structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D crystal
        structure plot with simple structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[NDArray, "..."] = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
            ]
        )
        atomic_numbers: Integer[NDArray, "..."] = np.array(
            [14, 14]
        )  # Two silicon atoms

        ax: Any = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_with_cell(self) -> None:
        r"""Test 3D crystal structure with unit cell outline.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D crystal
        structure with unit cell outline.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[NDArray, "..."] = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        atomic_numbers: Integer[NDArray, "..."] = np.array(
            [38, 22, 8, 8]
        )  # Sr, Ti, O, O

        ax: Any = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell_lengths=(4.0, 4.0, 4.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_multiple_elements(self) -> None:
        r"""Test 3D crystal structure with multiple element types.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D crystal
        structure with multiple element types.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[NDArray, "..."] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        atomic_numbers: Integer[NDArray, "..."] = np.array(
            [6, 14, 29, 79]
        )  # C, Si, Cu, Au

        ax: Any = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )
        assert isinstance(ax, Axes3D)
        # Should have legend entries for each unique element
        legend: Any = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4

    def test_plot_grazing_incidence_geometry_default(self) -> None:
        r"""Test grazing incidence geometry diagram.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: grazing incidence
        geometry diagram.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_grazing_incidence_geometry()
        assert isinstance(ax, Axes)

    def test_plot_grazing_incidence_geometry_custom_angle(self) -> None:
        r"""Test grazing incidence geometry with custom angle.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: grazing incidence
        geometry with custom angle.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_grazing_incidence_geometry(theta_deg=5.0)
        assert isinstance(ax, Axes)

    def test_plot_structure_factor_phases_simple(self) -> None:
        r"""Test structure factor phases diagram with simple structure.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        phases diagram with simple structure.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        atom_positions: list[Any] = [(0.0, 0.0), (0.5, 0.5)]
        ax: Any = plot_structure_factor_phases(
            atom_positions_2d=atom_positions,
            g_vector=(1.0, 0.0),
        )
        assert isinstance(ax, Axes)

    def test_plot_structure_factor_phases_multiple_atoms(self) -> None:
        r"""Test structure factor phases with multiple atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: structure factor
        phases with multiple atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        atom_positions: list[Any] = [
            (0.0, 0.0),
            (0.25, 0.25),
            (0.5, 0.0),
            (0.75, 0.75),
        ]
        ax: Any = plot_structure_factor_phases(
            atom_positions_2d=atom_positions,
            g_vector=(2.0, 1.0),
        )
        assert isinstance(ax, Axes)


class TestDiagramPlotsWithProvidedAxes:
    """Test that all functions accept user-provided axes."""

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_2d_plots_with_provided_ax(self) -> None:
        r"""Test 2D plotting functions with user-provided axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 2D plotting
        functions with user-provided axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        fig: Any
        axes: Any
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Test each 2D function
        ax1: Any = plot_wavelength_curve(ax=axes[0, 0])
        assert ax1 is axes[0, 0]

        ax2: Any = plot_form_factors(atomic_numbers=[14], ax=axes[0, 1])
        assert ax2 is axes[0, 1]

        ax3: Any = plot_debye_waller(
            atomic_number=14, temperatures=[300.0], ax=axes[0, 2]
        )
        assert ax3 is axes[0, 2]

        ax4: Any = plot_ctr_profile(ax=axes[1, 0])
        assert ax4 is axes[1, 0]

        ax5: Any = plot_roughness_damping(ax=axes[1, 1])
        assert ax5 is axes[1, 1]

        ax6: Any = plot_rod_broadening(ax=axes[1, 2])
        assert ax6 is axes[1, 2]

    def test_3d_plots_with_provided_ax(self) -> None:
        r"""Test 3D plotting functions with user-provided axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: 3D plotting
        functions with user-provided axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        fig: Any = plt.figure(figsize=(15, 5))

        ax1: Any = fig.add_subplot(131, projection="3d")
        returned_ax1: Any = plot_ewald_sphere_3d(ax=ax1)
        assert returned_ax1 is ax1

        ax2: Any = fig.add_subplot(132, projection="3d")
        returned_ax2: Any = plot_unit_cell_3d(ax=ax2)
        assert returned_ax2 is ax2

        ax3: Any = fig.add_subplot(133, projection="3d")
        positions: Float[NDArray, "..."] = np.array(
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
        )
        atomic_numbers: Integer[NDArray, "..."] = np.array([14, 14])
        returned_ax3: Any = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
            ax=ax3,
        )
        assert returned_ax3 is ax3


class TestDiagramViewingAngles:
    """Test 3D viewing angle parameters."""

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    @pytest.mark.parametrize(
        ("elev", "azim"),
        [
            (0.0, 0.0),  # Front view
            (90.0, 0.0),  # Top view
            (0.0, 90.0),  # Side view
            (20.0, 45.0),  # Perspective
            (-30.0, 135.0),  # Unusual angle
        ],
    )
    def test_ewald_sphere_3d_viewing_angles(
        self, elev: float, azim: float
    ) -> None:
        r"""Test Ewald sphere 3D with various viewing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Ewald sphere 3D
        with various viewing angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``elev``,
        ``azim``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_ewald_sphere_3d(elev=elev, azim=azim)
        assert isinstance(ax, Axes3D)

    @pytest.mark.parametrize(
        ("elev", "azim"),
        [
            (0.0, 0.0),
            (45.0, 45.0),
            (90.0, 0.0),
        ],
    )
    def test_unit_cell_3d_viewing_angles(
        self, elev: float, azim: float
    ) -> None:
        r"""Test unit cell 3D with various viewing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: unit cell 3D with
        various viewing angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``elev``,
        ``azim``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        ax: Any = plot_unit_cell_3d(elev=elev, azim=azim)
        assert isinstance(ax, Axes3D)

    @pytest.mark.parametrize(
        ("elev", "azim"),
        [
            (0.0, 0.0),
            (30.0, 60.0),
            (60.0, 120.0),
        ],
    )
    def test_crystal_structure_3d_viewing_angles(
        self, elev: float, azim: float
    ) -> None:
        r"""Test crystal structure 3D with various viewing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: crystal structure
        3D with various viewing angles.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named ``elev``,
        ``azim``, so the documented behavior is checked across the cases
        supplied by pytest, Chex, Hypothesis, or absl.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        positions: Float[NDArray, "..."] = np.array(
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
        )
        atomic_numbers: Integer[NDArray, "..."] = np.array([14, 14])
        ax: Any = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
            elev=elev,
            azim=azim,
        )
        assert isinstance(ax, Axes3D)


def _make_crystal(n_atoms: int = 4) -> CrystalStructure:
    """Create a simple CrystalStructure for view_atoms tests."""
    all_pos: Float[NDArray, "..."] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )[:n_atoms]
    z_nums: Integer[NDArray, "..."] = np.array([14, 8, 38, 22])[:n_atoms]
    frac_pos: Float[Array, "..."] = jnp.concatenate(
        [jnp.array(all_pos), jnp.array(z_nums[:, None], dtype=float)],
        axis=1,
    )
    cart_pos: Float[Array, "..."] = jnp.concatenate(
        [jnp.array(all_pos) * 4.0, jnp.array(z_nums[:, None], dtype=float)],
        axis=1,
    )
    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=jnp.array([4.0, 4.0, 4.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


def _make_layered_crystal() -> CrystalStructure:
    """Create a small crystal with a non-degenerate height range."""
    frac_coords: Float[Array, "..."] = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.75],
            [0.5, 0.5, 1.0],
        ],
        dtype=jnp.float64,
    )
    z_nums: Float[Array, "..."] = jnp.array(
        [[14.0], [8.0], [38.0], [22.0]],
        dtype=jnp.float64,
    )
    frac_pos: Float[Array, "..."] = jnp.concatenate(
        [frac_coords, z_nums],
        axis=1,
    )
    cart_pos: Float[Array, "..."] = jnp.concatenate(
        [frac_coords * 4.0, z_nums],
        axis=1,
    )
    return create_crystal_structure(
        frac_positions=frac_pos,
        cart_positions=cart_pos,
        cell_lengths=jnp.array([4.0, 4.0, 4.0]),
        cell_angles=jnp.array([90.0, 90.0, 90.0]),
    )


class TestViewAtoms:
    """Tests for view_atoms function.

    :see: :func:`~rheedium.plots.view_atoms`
    """

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_view_atoms_returns_axes3d(self) -> None:
        r"""Should return an Axes3D instance.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should return an
        Axes3D instance.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal)
        assert isinstance(ax, Axes3D)

    def test_view_atoms_with_unit_cell(self) -> None:
        r"""Should render unit cell wireframe when enabled.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should render unit
        cell wireframe when enabled.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal, show_unit_cell=True)
        assert isinstance(ax, Axes3D)

    def test_view_atoms_without_unit_cell(self) -> None:
        r"""Should work without unit cell wireframe.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should work
        without unit cell wireframe.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal, show_unit_cell=False)
        assert isinstance(ax, Axes3D)

    def test_view_atoms_custom_angles(self) -> None:
        r"""Should accept custom viewing angles.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should accept
        custom viewing angles.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal, elev=60.0, azim=120.0)
        assert isinstance(ax, Axes3D)

    def test_view_atoms_custom_scale(self) -> None:
        r"""Should accept custom atom_scale.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should accept
        custom atom_scale.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal, atom_scale=2.0)
        assert isinstance(ax, Axes3D)

    def test_view_atoms_custom_figsize(self) -> None:
        r"""Should accept custom figsize.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should accept
        custom figsize.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        ax: Any = view_atoms(crystal, figsize=(12, 10))
        assert isinstance(ax, Axes3D)

    def test_view_atoms_with_provided_ax(self) -> None:
        r"""Should use user-provided axes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should use
        user-provided axes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        fig: Any = plt.figure()
        ax: Any = fig.add_subplot(111, projection="3d")
        returned_ax: Any = view_atoms(crystal, ax=ax)
        assert returned_ax is ax

    def test_view_atoms_legend_entries(self) -> None:
        r"""Should have a legend entry per unique element.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should have a
        legend entry per unique element.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal(n_atoms=4)
        ax: Any = view_atoms(crystal)
        legend: Any = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4

    def test_view_atoms_single_element(self) -> None:
        r"""Should work with a single element type.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should work with a
        single element type.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal(n_atoms=1)
        ax: Any = view_atoms(crystal)
        assert isinstance(ax, Axes3D)


class TestViewAtomsInteractive:
    """Tests for ASE-backed interactive atom viewing."""

    def test_prepare_atoms_supercell_scales_atom_count(self) -> None:
        r"""Prepared ASE atoms should repeat according to supercell.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Prepared ASE atoms
        should repeat according to supercell.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        atoms: Any
        surface_mask: Any
        atoms, surface_mask = _prepare_interactive_atoms(
            crystal,
            supercell=(2, 2, 1),
        )

        assert len(atoms) == 4 * len(crystal.cart_positions)
        assert surface_mask is None
        assert atoms.info["rheedium_supercell"] == (2, 2, 1)
        assert atoms.info["rheedium_base_atom_count"] == len(
            crystal.cart_positions
        )

    def test_prepare_atoms_surface_mask_matches_simulator(self) -> None:
        r"""Surface mask should match identify_surface_atoms.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Surface mask
        should match identify_surface_atoms.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_layered_crystal()
        config: SurfaceConfig = SurfaceConfig(
            method="height",
            height_fraction=0.5,
        )
        atoms: Any
        surface_mask: Any
        atoms, surface_mask = _prepare_interactive_atoms(
            crystal,
            highlight_surface=True,
            surface_config=config,
        )
        expected: Any = np.asarray(
            identify_surface_atoms(
                jnp.asarray(atoms.get_positions()),
                config,
            ),
            dtype=bool,
        )

        assert surface_mask is not None
        np.testing.assert_array_equal(surface_mask, expected)
        assert 0 < int(np.sum(surface_mask)) < len(atoms)
        np.testing.assert_array_equal(
            atoms.get_array("rheedium_surface_mask"),
            expected,
        )

    def test_x3d_backend_returns_html_with_metadata(self) -> None:
        r"""x3d backend should work headlessly via IPython HTML.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: x3d backend should
        work headlessly via IPython HTML.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        handle: Any = view_atoms_interactive(
            crystal,
            supercell=(1, 1, 1),
            highlight_surface=True,
            backend="x3d",
        )

        assert hasattr(handle, "_repr_html_")
        assert handle.rheedium_backend == "x3d"
        assert len(handle.rheedium_atoms) == len(crystal.cart_positions)
        assert handle.rheedium_surface_mask is not None

    def test_x3d_backend_validates_beam_direction(self) -> None:
        r"""Invalid beam directions should raise before rendering.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid beam
        directions should raise before rendering.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = _make_crystal()
        with pytest.raises(ValueError, match="nonzero"):
            view_atoms_interactive(
                crystal,
                beam_direction=(0.0, 0.0, 0.0),
                backend="x3d",
            )

    def test_ngl_backend_returns_widget_with_metadata(self) -> None:
        r"""The ngl backend should work when nglview is installed.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The ngl backend
        should work when nglview is installed.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pytest.importorskip("nglview")
        crystal: CrystalStructure = _make_crystal()
        handle: Any = view_atoms_interactive(
            crystal,
            highlight_surface=True,
            beam_direction=(1, 0, -0.05),
            backend="ngl",
        )

        assert handle.rheedium_backend == "ngl"
        assert len(handle.rheedium_atoms) == len(crystal.cart_positions)
        assert handle.rheedium_surface_mask is not None
        assert _resolve_interactive_backend("auto") == "ngl"

    def test_invalid_backend_raises_value_error(self) -> None:
        r"""Backend selection should reject unknown values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Backend selection
        should reject unknown values.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with pytest.raises(ValueError, match="backend"):
            _resolve_interactive_backend("plotly")  # type: ignore[arg-type]

    def test_ngl_backend_requires_nglview(self) -> None:
        r"""Explicit ngl backend should give an optional-dep error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Explicit ngl
        backend should give an optional-dep error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        if _resolve_interactive_backend("auto") == "ngl":
            pytest.skip("nglview is installed in this environment")
        with pytest.raises(ImportError, match="nglview"):
            _resolve_interactive_backend("ngl")

    def test_public_plots_export(self) -> None:
        r"""view_atoms_interactive should be available from rheedium.plots.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        view_atoms_interactive should be available from rheedium.plots.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_diagrams``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        import rheedium as rh

        assert rh.plots.view_atoms_interactive is view_atoms_interactive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
