"""Test suite for rheedium.plots.diagrams visualization functions.

This module provides smoke tests to ensure all plotting functions run without
errors. Tests do not verify visual output, only that functions execute
successfully and return valid matplotlib objects.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# Use non-interactive backend for testing
matplotlib.use("Agg")

from rheedium.plots.diagrams import (
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
)


class TestDiagramPlots:
    """Smoke tests for diagram plotting functions."""

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_plot_wavelength_curve_default(self) -> None:
        """Test wavelength curve plot with default parameters."""
        ax = plot_wavelength_curve()
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Accelerating Voltage (kV)"
        assert ax.get_ylabel() == "Wavelength (A)"

    def test_plot_wavelength_curve_custom_range(self) -> None:
        """Test wavelength curve with custom voltage range."""
        ax = plot_wavelength_curve(
            voltage_range_kv=(10.0, 50.0),
            n_points=50,
            show_comparison=False,
        )
        assert isinstance(ax, Axes)
        xlim = ax.get_xlim()
        assert xlim[0] >= 10.0
        assert xlim[1] <= 50.0

    def test_plot_wavelength_curve_with_provided_ax(self) -> None:
        """Test wavelength curve with user-provided axes."""
        fig, ax = plt.subplots()
        returned_ax = plot_wavelength_curve(ax=ax)
        assert returned_ax is ax

    def test_plot_form_factors_single_element(self) -> None:
        """Test form factor plot for single element."""
        ax = plot_form_factors(atomic_numbers=[14])  # Silicon
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 1

    def test_plot_form_factors_multiple_elements(self) -> None:
        """Test form factor plot for multiple elements."""
        ax = plot_form_factors(atomic_numbers=[14, 8, 38, 22])
        assert isinstance(ax, Axes)
        # Should have one line per element
        assert len(ax.get_lines()) >= 4

    def test_plot_form_factors_custom_range(self) -> None:
        """Test form factor plot with custom q range."""
        ax = plot_form_factors(
            atomic_numbers=[14],
            q_range=(0.0, 5.0),
            n_points=100,
        )
        assert isinstance(ax, Axes)
        xlim = ax.get_xlim()
        assert xlim[0] >= 0.0
        assert xlim[1] <= 5.0

    def test_plot_debye_waller_default(self) -> None:
        """Test Debye-Waller plot with default parameters."""
        ax = plot_debye_waller(
            atomic_number=14,
            temperatures=[100.0, 300.0, 600.0],
        )
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 3

    def test_plot_debye_waller_single_temp(self) -> None:
        """Test Debye-Waller plot with single temperature."""
        ax = plot_debye_waller(
            atomic_number=79,  # Gold
            temperatures=[300.0],
        )
        assert isinstance(ax, Axes)

    def test_plot_ctr_profile_default(self) -> None:
        """Test CTR profile plot with default parameters."""
        ax = plot_ctr_profile()
        assert isinstance(ax, Axes)
        # Should be log scale on y-axis
        assert ax.get_yscale() == "log"

    def test_plot_ctr_profile_custom_range(self) -> None:
        """Test CTR profile with custom l range."""
        ax = plot_ctr_profile(
            l_range=(-2.0, 2.0),
            n_points=200,
        )
        assert isinstance(ax, Axes)

    def test_plot_roughness_damping_default(self) -> None:
        """Test roughness damping plot with default parameters."""
        ax = plot_roughness_damping()
        assert isinstance(ax, Axes)
        # Should have lines for each sigma value
        assert len(ax.get_lines()) >= 4

    def test_plot_roughness_damping_custom_sigma(self) -> None:
        """Test roughness damping with custom sigma values."""
        ax = plot_roughness_damping(
            q_z_range=(0.0, 3.0),
            sigma_values=[0.0, 1.0, 3.0],
        )
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 3

    def test_plot_rod_broadening_default(self) -> None:
        """Test rod broadening plot with default parameters."""
        ax = plot_rod_broadening()
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 4

    def test_plot_rod_broadening_custom_lengths(self) -> None:
        """Test rod broadening with custom correlation lengths."""
        ax = plot_rod_broadening(
            q_perp_range=(-0.5, 0.5),
            correlation_lengths=[20.0, 100.0],
        )
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_2d_default(self) -> None:
        """Test 2D Ewald sphere plot with default parameters."""
        ax = plot_ewald_sphere_2d()
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_2d_custom_params(self) -> None:
        """Test 2D Ewald sphere with custom parameters."""
        ax = plot_ewald_sphere_2d(
            voltage_kv=20.0,
            theta_deg=3.0,
            lattice_spacing=5.0,
            n_rods=5,
        )
        assert isinstance(ax, Axes)

    def test_plot_ewald_sphere_3d_default(self) -> None:
        """Test 3D Ewald sphere plot with default parameters."""
        ax = plot_ewald_sphere_3d()
        assert isinstance(ax, Axes3D)

    def test_plot_ewald_sphere_3d_different_views(self) -> None:
        """Test 3D Ewald sphere with different viewing angles."""
        # Front view
        ax1 = plot_ewald_sphere_3d(elev=0.0, azim=0.0)
        assert isinstance(ax1, Axes3D)
        plt.close()

        # Top view
        ax2 = plot_ewald_sphere_3d(elev=90.0, azim=0.0)
        assert isinstance(ax2, Axes3D)
        plt.close()

        # Perspective view
        ax3 = plot_ewald_sphere_3d(elev=30.0, azim=45.0)
        assert isinstance(ax3, Axes3D)

    def test_plot_unit_cell_3d_cubic(self) -> None:
        """Test 3D unit cell plot for cubic cell."""
        ax = plot_unit_cell_3d(
            cell_lengths=(4.0, 4.0, 4.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_unit_cell_3d_orthorhombic(self) -> None:
        """Test 3D unit cell plot for orthorhombic cell."""
        ax = plot_unit_cell_3d(
            cell_lengths=(3.0, 4.0, 5.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_unit_cell_3d_triclinic(self) -> None:
        """Test 3D unit cell plot for triclinic cell."""
        ax = plot_unit_cell_3d(
            cell_lengths=(3.0, 4.0, 5.0),
            cell_angles=(80.0, 85.0, 75.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_simple(self) -> None:
        """Test 3D crystal structure plot with simple structure."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
            ]
        )
        atomic_numbers = np.array([14, 14])  # Two silicon atoms

        ax = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_with_cell(self) -> None:
        """Test 3D crystal structure with unit cell outline."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        atomic_numbers = np.array([38, 22, 8, 8])  # Sr, Ti, O, O

        ax = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell_lengths=(4.0, 4.0, 4.0),
            cell_angles=(90.0, 90.0, 90.0),
        )
        assert isinstance(ax, Axes3D)

    def test_plot_crystal_structure_3d_multiple_elements(self) -> None:
        """Test 3D crystal structure with multiple element types."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        atomic_numbers = np.array([6, 14, 29, 79])  # C, Si, Cu, Au

        ax = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
        )
        assert isinstance(ax, Axes3D)
        # Should have legend entries for each unique element
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4

    def test_plot_grazing_incidence_geometry_default(self) -> None:
        """Test grazing incidence geometry diagram."""
        ax = plot_grazing_incidence_geometry()
        assert isinstance(ax, Axes)

    def test_plot_grazing_incidence_geometry_custom_angle(self) -> None:
        """Test grazing incidence geometry with custom angle."""
        ax = plot_grazing_incidence_geometry(theta_deg=5.0)
        assert isinstance(ax, Axes)

    def test_plot_structure_factor_phases_simple(self) -> None:
        """Test structure factor phases diagram with simple structure."""
        atom_positions = [(0.0, 0.0), (0.5, 0.5)]
        ax = plot_structure_factor_phases(
            atom_positions_2d=atom_positions,
            g_vector=(1.0, 0.0),
        )
        assert isinstance(ax, Axes)

    def test_plot_structure_factor_phases_multiple_atoms(self) -> None:
        """Test structure factor phases with multiple atoms."""
        atom_positions = [
            (0.0, 0.0),
            (0.25, 0.25),
            (0.5, 0.0),
            (0.75, 0.75),
        ]
        ax = plot_structure_factor_phases(
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
        """Test 2D plotting functions with user-provided axes."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Test each 2D function
        ax1 = plot_wavelength_curve(ax=axes[0, 0])
        assert ax1 is axes[0, 0]

        ax2 = plot_form_factors(atomic_numbers=[14], ax=axes[0, 1])
        assert ax2 is axes[0, 1]

        ax3 = plot_debye_waller(
            atomic_number=14, temperatures=[300.0], ax=axes[0, 2]
        )
        assert ax3 is axes[0, 2]

        ax4 = plot_ctr_profile(ax=axes[1, 0])
        assert ax4 is axes[1, 0]

        ax5 = plot_roughness_damping(ax=axes[1, 1])
        assert ax5 is axes[1, 1]

        ax6 = plot_rod_broadening(ax=axes[1, 2])
        assert ax6 is axes[1, 2]

    def test_3d_plots_with_provided_ax(self) -> None:
        """Test 3D plotting functions with user-provided axes."""
        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(131, projection="3d")
        returned_ax1 = plot_ewald_sphere_3d(ax=ax1)
        assert returned_ax1 is ax1

        ax2 = fig.add_subplot(132, projection="3d")
        returned_ax2 = plot_unit_cell_3d(ax=ax2)
        assert returned_ax2 is ax2

        ax3 = fig.add_subplot(133, projection="3d")
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        atomic_numbers = np.array([14, 14])
        returned_ax3 = plot_crystal_structure_3d(
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
        "elev,azim",
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
        """Test Ewald sphere 3D with various viewing angles."""
        ax = plot_ewald_sphere_3d(elev=elev, azim=azim)
        assert isinstance(ax, Axes3D)

    @pytest.mark.parametrize(
        "elev,azim",
        [
            (0.0, 0.0),
            (45.0, 45.0),
            (90.0, 0.0),
        ],
    )
    def test_unit_cell_3d_viewing_angles(
        self, elev: float, azim: float
    ) -> None:
        """Test unit cell 3D with various viewing angles."""
        ax = plot_unit_cell_3d(elev=elev, azim=azim)
        assert isinstance(ax, Axes3D)

    @pytest.mark.parametrize(
        "elev,azim",
        [
            (0.0, 0.0),
            (30.0, 60.0),
            (60.0, 120.0),
        ],
    )
    def test_crystal_structure_3d_viewing_angles(
        self, elev: float, azim: float
    ) -> None:
        """Test crystal structure 3D with various viewing angles."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        atomic_numbers = np.array([14, 14])
        ax = plot_crystal_structure_3d(
            positions=positions,
            atomic_numbers=atomic_numbers,
            elev=elev,
            azim=azim,
        )
        assert isinstance(ax, Axes3D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
