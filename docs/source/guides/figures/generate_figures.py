#!/usr/bin/env python
"""Generate all documentation figures for rheedium guides.

This script generates publication-quality SVG figures for the documentation
guides. All figures are created using rheedium's plotting functions and
matplotlib. SVG format is used for vector graphics that:
- Scale without pixelation
- Are natively supported by web browsers (for HTML docs)
- Work well with Sphinx/ReadTheDocs

Usage
-----
    python generate_figures.py

All figures are saved to the current directory (docs/source/guides/figures/).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for headless generation
matplotlib.use("Agg")

# Common style settings
plt.style.use("default")
FIGSIZE = (8, 6)
FIGSIZE_WIDE = (10, 6)
FIGSIZE_TALL = (8, 8)
SAVE_DIR = Path(__file__).parent


def save_fig(name: str) -> None:
    """Save current figure as SVG and close it."""
    # Replace .png or .pdf extension with .svg if present
    if name.endswith(".png"):
        name = name[:-4] + ".svg"
    elif name.endswith(".pdf"):
        name = name[:-4] + ".svg"
    plt.savefig(
        SAVE_DIR / name,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"  Saved: {name}")


# =============================================================================
# Kinematic Scattering Figures
# =============================================================================


def generate_kinematic_figures() -> None:
    """Generate figures for kinematic-scattering.md guide."""
    print("\nGenerating kinematic scattering figures...")

    from rheedium.plots import plot_form_factors, plot_wavelength_curve

    # 1. Wavelength vs voltage
    plt.figure(figsize=FIGSIZE)
    plot_wavelength_curve(voltage_range_kv=(5.0, 30.0), show_comparison=True)
    save_fig("wavelength_vs_voltage.png")

    # 2. Form factor curves for common elements
    plt.figure(figsize=FIGSIZE)
    plot_form_factors(atomic_numbers=[14, 8, 38, 22], q_range=(0.0, 8.0))
    save_fig("form_factor_curves.png")

    # 3. CTR intensity profile
    from rheedium.plots import plot_ctr_profile

    plt.figure(figsize=FIGSIZE_WIDE)
    plot_ctr_profile(l_range=(-3.0, 3.0), n_points=500)
    save_fig("ctr_intensity_profile.png")

    # 4. Structure factor phases (Argand diagram)
    from rheedium.plots import plot_structure_factor_phases

    plt.figure(figsize=FIGSIZE_TALL)
    # Simple perovskite-like positions
    positions = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)]
    plot_structure_factor_phases(
        atom_positions_2d=positions, g_vector=(1.0, 1.0)
    )
    save_fig("structure_factor_phases.png")


# =============================================================================
# Ewald Sphere Figures
# =============================================================================


def generate_ewald_figures() -> None:
    """Generate figures for ewald-sphere.md guide."""
    print("\nGenerating Ewald sphere figures...")

    from rheedium.plots import plot_ewald_sphere_2d, plot_ewald_sphere_3d

    # 1. 2D cross-section
    plt.figure(figsize=FIGSIZE_WIDE)
    plot_ewald_sphere_2d(voltage_kv=15.0, theta_deg=2.0, n_rods=7)
    save_fig("ewald_sphere_2d.png")

    # 2. 3D front view (elev=0, azim=0)
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")
    plot_ewald_sphere_3d(
        voltage_kv=15.0, theta_deg=2.0, elev=0.0, azim=0.0, ax=ax
    )
    save_fig("ewald_sphere_3d_front.png")

    # 3. 3D perspective view (elev=20, azim=45)
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")
    plot_ewald_sphere_3d(
        voltage_kv=15.0, theta_deg=2.0, elev=20.0, azim=45.0, ax=ax
    )
    save_fig("ewald_sphere_3d_perspective.png")

    # 4. 3D top view (elev=90, azim=0)
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")
    plot_ewald_sphere_3d(
        voltage_kv=15.0, theta_deg=2.0, elev=90.0, azim=0.0, ax=ax
    )
    save_fig("ewald_sphere_3d_top.png")

    # 5. Grazing incidence geometry
    from rheedium.plots import plot_grazing_incidence_geometry

    plt.figure(figsize=FIGSIZE_WIDE)
    plot_grazing_incidence_geometry(theta_deg=2.0)
    save_fig("grazing_incidence_geometry.png")


# =============================================================================
# Form Factors Figures
# =============================================================================


def generate_form_factor_figures() -> None:
    """Generate figures for form-factors.md guide."""
    print("\nGenerating form factor figures...")

    from rheedium.plots import plot_debye_waller, plot_form_factors

    # 1. Form factor curves - light vs heavy elements
    plt.figure(figsize=FIGSIZE)
    plot_form_factors(atomic_numbers=[6, 14, 29, 79], q_range=(0.0, 10.0))
    save_fig("element_comparison.png")

    # 2. Debye-Waller damping at different temperatures (Silicon)
    plt.figure(figsize=FIGSIZE)
    plot_debye_waller(
        atomic_number=14,
        temperatures=[100.0, 300.0, 500.0, 800.0],
        q_range=(0.0, 8.0),
    )
    save_fig("debye_waller_damping.png")

    # 3. Combined scattering: f(q) with Debye-Waller
    import jax.numpy as jnp

    from rheedium.simul.form_factors import (
        debye_waller_factor,
        get_mean_square_displacement,
        kirkland_form_factor,
    )

    plt.figure(figsize=FIGSIZE)
    q_values = np.linspace(0.0, 8.0, 200)
    q_jax = jnp.array(q_values)

    # Get form factor
    ff = np.array(kirkland_form_factor(14, q_jax))

    # Get DW factor at room temperature
    msd = float(get_mean_square_displacement(14, 300.0))
    dw = np.array(debye_waller_factor(q_jax, msd))

    combined = ff * dw

    plt.plot(q_values, ff, "b-", linewidth=2, label="f(q) only")
    plt.plot(q_values, combined, "r-", linewidth=2, label="f(q) × DW (300K)")
    plt.fill_between(q_values, combined, alpha=0.3, color="red")
    plt.xlabel("q (1/A)", fontsize=12)
    plt.ylabel("Scattering Factor", fontsize=12)
    plt.title("Combined Scattering: Form Factor × Debye-Waller", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8)
    plt.ylim(bottom=0)
    save_fig("combined_scattering.png")


# =============================================================================
# Surface Rods Figures
# =============================================================================


def generate_surface_rod_figures() -> None:
    """Generate figures for surface-rods.md guide."""
    print("\nGenerating surface rod figures...")

    from rheedium.plots import plot_rod_broadening, plot_roughness_damping

    # 1. Roughness damping
    plt.figure(figsize=FIGSIZE)
    plot_roughness_damping(
        q_z_range=(0.0, 5.0),
        sigma_values=[0.0, 0.5, 1.0, 2.0, 5.0],
    )
    save_fig("roughness_damping.png")

    # 2. Rod broadening from finite domain size
    plt.figure(figsize=FIGSIZE)
    plot_rod_broadening(
        q_perp_range=(-0.5, 0.5),
        correlation_lengths=[20.0, 50.0, 100.0, 500.0],
    )
    save_fig("rod_broadening.png")

    # 3. CTR origin diagram (schematic)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Bulk crystal (periodic)
    ax1 = axes[0]
    for i in range(-2, 3):
        for j in range(-2, 3):
            ax1.scatter(i, j, c="blue", s=100)
    ax1.set_title("Bulk Crystal\n(Infinite)", fontsize=12)
    ax1.set_aspect("equal")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axis("off")

    # Panel 2: Truncated crystal
    ax2 = axes[1]
    for i in range(-2, 3):
        for j in range(0, 3):  # Only above surface
            ax2.scatter(i, j, c="blue", s=100)
    ax2.axhline(-0.3, color="brown", linewidth=3)
    ax2.fill_between([-3, 3], [-3, -3], [-0.3, -0.3], color="tan", alpha=0.3)
    ax2.set_title("Truncated Crystal\n(Surface)", fontsize=12)
    ax2.set_aspect("equal")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.axis("off")

    # Panel 3: Reciprocal space rods
    ax3 = axes[2]
    for h in range(-2, 3):
        ax3.axvline(h, color="green", linewidth=2, alpha=0.7)
    ax3.scatter([0], [0], c="red", s=150, zorder=5, label="Origin")
    ax3.set_title("Reciprocal Space\n(CTRs)", fontsize=12)
    ax3.set_xlabel("h", fontsize=10)
    ax3.set_ylabel("l", fontsize=10)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("ctr_origin_diagram.png")


# =============================================================================
# Unit Cell Figures
# =============================================================================


def generate_unit_cell_figures() -> None:
    """Generate figures for unit-cell.md guide."""
    print("\nGenerating unit cell figures...")

    from rheedium.plots import plot_crystal_structure_3d, plot_unit_cell_3d

    # 1. Lattice vector construction
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")
    plot_unit_cell_3d(
        cell_lengths=(4.0, 4.0, 4.0),
        cell_angles=(90.0, 90.0, 90.0),
        elev=20.0,
        azim=30.0,
        ax=ax,
    )
    save_fig("lattice_vector_construction.png")

    # 2. Different viewing angles - perspective
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")
    plot_unit_cell_3d(
        cell_lengths=(4.0, 5.0, 3.0),  # Orthorhombic
        cell_angles=(90.0, 90.0, 90.0),
        elev=25.0,
        azim=45.0,
        ax=ax,
    )
    ax.set_title("Orthorhombic Unit Cell", fontsize=12)
    save_fig("unit_cell_orthorhombic.png")

    # 3. Crystal structure visualization (simple perovskite-like)
    fig = plt.figure(figsize=FIGSIZE_TALL)
    ax = fig.add_subplot(111, projection="3d")

    # SrTiO3-like positions
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Sr at corner
            [1.95, 1.95, 1.95],  # Ti at body center
            [1.95, 1.95, 0.0],  # O on face
            [1.95, 0.0, 1.95],  # O on face
            [0.0, 1.95, 1.95],  # O on face
        ]
    )
    atomic_numbers = np.array([38, 22, 8, 8, 8])  # Sr, Ti, O, O, O

    plot_crystal_structure_3d(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell_lengths=(3.9, 3.9, 3.9),
        cell_angles=(90.0, 90.0, 90.0),
        elev=25.0,
        azim=35.0,
        ax=ax,
    )
    ax.set_title("SrTiO3-like Perovskite Structure", fontsize=12)
    save_fig("crystal_structure_example.png")

    # 4. Miller indices diagram (conceptual)
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"}
    )

    for idx, (hkl, title) in enumerate(
        [((1, 0, 0), "(100)"), ((1, 1, 0), "(110)"), ((1, 1, 1), "(111)")]
    ):
        ax = axes[idx]

        # Draw unit cell
        cell = 2.0
        for i in [0, cell]:
            for j in [0, cell]:
                ax.plot3D([i, i], [j, j], [0, cell], "k-", alpha=0.3)
                ax.plot3D([i, i], [0, cell], [j, j], "k-", alpha=0.3)
                ax.plot3D([0, cell], [i, i], [j, j], "k-", alpha=0.3)

        # Draw plane
        h, k, l = hkl
        if hkl == (1, 0, 0):
            xx, zz = np.meshgrid([cell], np.linspace(0, cell, 2))
            yy = np.linspace(0, cell, 2)[:, np.newaxis] * np.ones_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.5, color="blue")
        elif hkl == (1, 1, 0):
            xx = np.linspace(0, cell, 2)
            yy = cell - xx
            zz = np.array([[0, 0], [cell, cell]])
            ax.plot_surface(
                xx[:, np.newaxis] * np.ones((2, 2)),
                yy[:, np.newaxis] * np.ones((2, 2)),
                zz,
                alpha=0.5,
                color="green",
            )
        elif hkl == (1, 1, 1):
            # (111) plane - triangle
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            verts = [[(cell, 0, 0), (0, cell, 0), (0, 0, cell)]]
            ax.add_collection3d(
                Poly3DCollection(verts, alpha=0.5, color="red")
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{title} Plane", fontsize=12)
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)
        ax.set_zlim(0, cell)
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_fig("miller_indices.png")


# =============================================================================
# Data Wrangling Figures
# =============================================================================


def generate_data_wrangling_figures() -> None:
    """Generate figures for data-wrangling.md guide."""
    print("\nGenerating data wrangling figures...")

    # 1. Coordinate systems diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fractional coordinates
    ax1 = axes[0]
    ax1.set_xlim(-0.2, 1.4)
    ax1.set_ylim(-0.2, 1.4)

    # Draw unit cell
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", linewidth=2)

    # Draw axes with arrows
    ax1.annotate(
        "",
        xy=(1.2, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax1.annotate(
        "",
        xy=(0, 1.2),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax1.text(1.25, 0, "a (1.0)", fontsize=11, color="red")
    ax1.text(0.05, 1.25, "b (1.0)", fontsize=11, color="green")

    # Plot atoms at fractional positions
    atoms = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)]
    for x, y in atoms:
        ax1.scatter(x, y, s=150, c="blue", zorder=5)
        ax1.text(x + 0.05, y + 0.05, f"({x:.1f}, {y:.1f})", fontsize=9)

    ax1.set_title("Fractional Coordinates", fontsize=14)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Fractional x")
    ax1.set_ylabel("Fractional y")

    # Cartesian coordinates
    ax2 = axes[1]
    a, b = 4.0, 5.0  # Different lengths
    ax2.set_xlim(-0.5, a + 1)
    ax2.set_ylim(-0.5, b + 1)

    # Draw unit cell
    ax2.plot([0, a, a, 0, 0], [0, 0, b, b, 0], "k-", linewidth=2)

    # Draw axes
    ax2.annotate(
        "",
        xy=(a + 0.5, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax2.annotate(
        "",
        xy=(0, b + 0.5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax2.text(a + 0.6, 0, f"a = {a:.1f} A", fontsize=11, color="red")
    ax2.text(0.1, b + 0.6, f"b = {b:.1f} A", fontsize=11, color="green")

    # Plot atoms at Cartesian positions
    for fx, fy in atoms:
        x, y = fx * a, fy * b
        ax2.scatter(x, y, s=150, c="blue", zorder=5)
        ax2.text(x + 0.1, y + 0.1, f"({x:.1f}, {y:.1f})", fontsize=9)

    ax2.set_title("Cartesian Coordinates", fontsize=14)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("x (A)")
    ax2.set_ylabel("y (A)")

    plt.tight_layout()
    save_fig("coordinate_systems.png")

    # 2. Data flow diagram (text-based -> visual)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Draw boxes
    boxes = [
        (0.05, 0.6, 0.15, 0.25, "CIF File"),
        (0.25, 0.6, 0.15, 0.25, "parse_cif()"),
        (0.45, 0.6, 0.18, 0.25, "CrystalStructure"),
        (0.68, 0.6, 0.15, 0.25, "build_ewald_data()"),
        (0.88, 0.6, 0.1, 0.25, "EwaldData"),
        (0.05, 0.15, 0.15, 0.25, "XYZ File"),
        (0.25, 0.15, 0.15, 0.25, "parse_xyz()"),
        (0.45, 0.15, 0.15, 0.25, "XYZData"),
        (0.65, 0.15, 0.18, 0.25, "xyz_to_crystal()"),
    ]

    colors = {
        "CIF File": "#E8F4FD",
        "XYZ File": "#E8F4FD",
        "parse_cif()": "#FFF3CD",
        "parse_xyz()": "#FFF3CD",
        "CrystalStructure": "#D4EDDA",
        "XYZData": "#D4EDDA",
        "build_ewald_data()": "#FFF3CD",
        "xyz_to_crystal()": "#FFF3CD",
        "EwaldData": "#D4EDDA",
    }

    for x, y, w, h, label in boxes:
        color = colors.get(label, "#FFFFFF")
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=True,
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Draw arrows
    arrows = [
        (0.20, 0.725, 0.05, 0),
        (0.40, 0.725, 0.05, 0),
        (0.63, 0.725, 0.05, 0),
        (0.83, 0.725, 0.05, 0),
        (0.20, 0.275, 0.05, 0),
        (0.40, 0.275, 0.05, 0),
        (0.60, 0.275, 0.05, 0),
        (0.54, 0.40, 0, 0.15),  # XYZData to CrystalStructure
    ]

    for x, y, dx, dy in arrows:
        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Data Processing Pipeline", fontsize=14, fontweight="bold")

    save_fig("data_flow_diagram.png")


# =============================================================================
# PyTree Architecture Figures
# =============================================================================


def generate_pytree_figures() -> None:
    """Generate figures for pytree-architecture.md guide."""
    print("\nGenerating PyTree architecture figures...")

    # 1. PyTree hierarchy diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Level 0: Root
    ax.text(
        0.5,
        0.95,
        "rheedium Data Structures",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Level 1: Main categories
    categories = [
        (0.2, 0.78, "Input Data"),
        (0.5, 0.78, "Crystal Data"),
        (0.8, 0.78, "Output Data"),
    ]
    for x, y, label in categories:
        rect = plt.Rectangle(
            (x - 0.08, y - 0.04),
            0.16,
            0.08,
            fill=True,
            facecolor="#E8F4FD",
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Level 2: PyTree types
    pytrees = [
        # Input
        (0.12, 0.58, "XYZData", "#FFF3CD"),
        (0.28, 0.58, "RHEEDImage", "#FFF3CD"),
        # Crystal
        (0.42, 0.58, "CrystalStructure", "#D4EDDA"),
        (0.58, 0.58, "EwaldData", "#D4EDDA"),
        # Output
        (0.72, 0.58, "RHEEDPattern", "#F8D7DA"),
        (0.88, 0.58, "SlicedCrystal", "#F8D7DA"),
    ]

    for x, y, label, color in pytrees:
        rect = plt.Rectangle(
            (x - 0.07, y - 0.04),
            0.14,
            0.08,
            fill=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    # Level 3: Key attributes (simplified)
    attrs = [
        # XYZData
        (0.12, 0.42, "positions\natomic_numbers\nlattice"),
        # CrystalStructure
        (
            0.42,
            0.42,
            "frac_positions\ncart_positions\ncell_lengths\ncell_angles",
        ),
        # EwaldData
        (0.58, 0.42, "wavelength\ng_vectors\nstructure_factors"),
        # RHEEDPattern
        (0.72, 0.42, "G_indices\nk_out\ndetector_points\nintensities"),
    ]

    for x, y, text in attrs:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
            linespacing=1.5,
        )

    # Draw connecting lines
    connections = [
        (0.2, 0.74, 0.12, 0.62),
        (0.2, 0.74, 0.28, 0.62),
        (0.5, 0.74, 0.42, 0.62),
        (0.5, 0.74, 0.58, 0.62),
        (0.8, 0.74, 0.72, 0.62),
        (0.8, 0.74, 0.88, 0.62),
    ]
    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)

    # Workflow arrows
    ax.annotate(
        "",
        xy=(0.42, 0.58),
        xytext=(0.28, 0.58),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.annotate(
        "",
        xy=(0.58, 0.58),
        xytext=(0.50, 0.58),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.annotate(
        "",
        xy=(0.72, 0.58),
        xytext=(0.66, 0.58),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )

    # Legend
    ax.text(
        0.5,
        0.18,
        "Workflow: Parse → Build Crystal → Compute Ewald → Simulate",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        color="blue",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("PyTree Data Structure Hierarchy", fontsize=14, pad=20)

    save_fig("pytree_hierarchy.png")


# =============================================================================
# RHEED Pattern Figures
# =============================================================================


def _render_rheed_to_image(
    rheed_pattern, grid_size: int = 300, spot_width: float = 0.04
):
    """Render RHEED pattern to a 2D image array for subplotting."""
    coords = np.asarray(rheed_pattern.detector_points)
    x_np = coords[:, 0]
    y_np = coords[:, 1]
    i_np = np.asarray(rheed_pattern.intensities)

    x_min = float(x_np.min()) - 0.5
    x_max = float(x_np.max()) + 0.5
    y_min = float(y_np.min()) - 0.5
    y_max = float(y_np.max()) + 0.5

    x_axis = np.linspace(x_min, x_max, grid_size)
    y_axis = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    image = np.zeros((grid_size, grid_size))

    for idx in range(len(i_np)):
        x0 = x_np[idx]
        y0 = y_np[idx]
        i0 = i_np[idx]
        image += i0 * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * spot_width**2)
        )

    extent = [x_min, x_max, y_min, y_max]
    return image, extent


def generate_rheed_pattern_figures() -> None:
    """Generate RHEED pattern figures for guides."""
    print("\nGenerating RHEED pattern figures...")

    from rheedium.inout.cif import parse_cif
    from rheedium.plots.figuring import create_phosphor_colormap
    from rheedium.simul.simulator import ewald_simulator, kinematic_simulator
    from rheedium.types.rheed_types import SurfaceConfig

    # Use test CIF files
    test_data_dir = (
        Path(__file__).parent.parent.parent.parent.parent
        / "tests"
        / "test_data"
    )
    mgo_cif = test_data_dir / "MgO.cif"
    srtio3_cif = test_data_dir / "SrTiO3.cif"

    cmap = create_phosphor_colormap()

    # 1. MgO kinematic RHEED pattern
    print("  Generating MgO RHEED pattern...")
    mgo_crystal = parse_cif(str(mgo_cif))
    mgo_pattern = kinematic_simulator(
        mgo_crystal,
        voltage_kv=15.0,
        theta_deg=2.0,
        phi_deg=0.0,
        hmax=5,
        kmax=5,
        lmax=3,
        surface_roughness=0.3,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    image, extent = _render_rheed_to_image(
        mgo_pattern, grid_size=400, spot_width=0.04
    )
    ax.imshow(image, extent=extent, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel("x_d (mm)")
    ax.set_ylabel("y_d (mm)")
    ax.set_title("MgO RHEED Pattern (θ=2°, φ=0°)", fontsize=14)
    save_fig("mgo_kinematic_rheed.svg")

    # 2. Structure factor comparison: MgO vs SrTiO3
    print("  Generating structure factor comparison...")
    srtio3_crystal = parse_cif(str(srtio3_cif))
    srtio3_pattern = kinematic_simulator(
        srtio3_crystal,
        voltage_kv=15.0,
        theta_deg=2.0,
        phi_deg=0.0,
        hmax=5,
        kmax=5,
        lmax=3,
        surface_roughness=0.3,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    image_mgo, extent_mgo = _render_rheed_to_image(
        mgo_pattern, grid_size=350, spot_width=0.04
    )
    axes[0].imshow(
        image_mgo, extent=extent_mgo, origin="lower", cmap=cmap, aspect="auto"
    )
    axes[0].set_xlabel("x_d (mm)")
    axes[0].set_ylabel("y_d (mm)")
    axes[0].set_title("Rock Salt (MgO)", fontsize=12)

    image_sto, extent_sto = _render_rheed_to_image(
        srtio3_pattern, grid_size=350, spot_width=0.04
    )
    axes[1].imshow(
        image_sto, extent=extent_sto, origin="lower", cmap=cmap, aspect="auto"
    )
    axes[1].set_xlabel("x_d (mm)")
    axes[1].set_ylabel("y_d (mm)")
    axes[1].set_title("Perovskite (SrTiO₃)", fontsize=12)

    plt.suptitle("Structure Factor Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("structure_factor_comparison.svg")

    # 3. Layer depth comparison (surface_fraction effect)
    print("  Generating layer depth comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fractions = [0.1, 0.3, 0.5]

    for ax, frac in zip(axes, fractions):
        pattern = kinematic_simulator(
            srtio3_crystal,
            voltage_kv=15.0,
            theta_deg=2.0,
            phi_deg=0.0,
            hmax=5,
            kmax=5,
            lmax=3,
            surface_fraction=frac,
            surface_roughness=0.3,
        )
        image, extent = _render_rheed_to_image(
            pattern, grid_size=300, spot_width=0.04
        )
        ax.imshow(
            image, extent=extent, origin="lower", cmap=cmap, aspect="auto"
        )
        ax.set_xlabel("x_d (mm)")
        ax.set_ylabel("y_d (mm)")
        ax.set_title(f"surface_fraction = {frac}", fontsize=12)

    plt.suptitle(
        "Effect of Surface Depth on RHEED Pattern", fontsize=14, y=1.02
    )
    plt.tight_layout()
    save_fig("layer_depth_comparison.svg")

    # 4. Selection method comparison using ewald_simulator (supports surface_config)
    print("  Generating selection method comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Default selection with ewald_simulator
    pattern_height = ewald_simulator(
        srtio3_crystal,
        voltage_kv=15.0,
        theta_deg=2.0,
        phi_deg=0.0,
        hmax=5,
        kmax=5,
        surface_roughness=0.3,
    )
    image_h, extent_h = _render_rheed_to_image(
        pattern_height, grid_size=350, spot_width=0.04
    )
    axes[0].imshow(
        image_h, extent=extent_h, origin="lower", cmap=cmap, aspect="auto"
    )
    axes[0].set_xlabel("x_d (mm)")
    axes[0].set_ylabel("y_d (mm)")
    axes[0].set_title("Default selection", fontsize=12)

    # Layer-based using SurfaceConfig
    config = SurfaceConfig(method="layers", n_layers=2)
    pattern_layers = ewald_simulator(
        srtio3_crystal,
        voltage_kv=15.0,
        theta_deg=2.0,
        phi_deg=0.0,
        hmax=5,
        kmax=5,
        surface_config=config,
        surface_roughness=0.3,
    )
    image_l, extent_l = _render_rheed_to_image(
        pattern_layers, grid_size=350, spot_width=0.04
    )
    axes[1].imshow(
        image_l, extent=extent_l, origin="lower", cmap=cmap, aspect="auto"
    )
    axes[1].set_xlabel("x_d (mm)")
    axes[1].set_ylabel("y_d (mm)")
    axes[1].set_title("Layer-based selection (2 layers)", fontsize=12)

    plt.suptitle("Surface Atom Selection Methods", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("selection_method_comparison.svg")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Generate all documentation figures."""
    print("=" * 60)
    print("Generating documentation figures for rheedium")
    print("=" * 60)
    print(f"\nOutput directory: {SAVE_DIR}")

    generate_kinematic_figures()
    generate_ewald_figures()
    generate_form_factor_figures()
    generate_surface_rod_figures()
    generate_unit_cell_figures()
    generate_data_wrangling_figures()
    generate_pytree_figures()
    generate_rheed_pattern_figures()

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
