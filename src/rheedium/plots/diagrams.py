"""Visualization functions for RHEED physics and crystallography diagrams.

Extended Summary
----------------
This module provides reusable visualization functions for creating publication-
quality figures explaining RHEED physics concepts. All functions return
matplotlib axes objects and accept optional axis parameters for compositing
multiple plots.

Routine Listings
----------------
plot_wavelength_curve : function
    Plot electron wavelength vs accelerating voltage with relativistic correction
plot_form_factors : function
    Plot atomic form factors f(q) for multiple elements
plot_debye_waller : function
    Plot Debye-Waller damping factor at different temperatures
plot_ctr_profile : function
    Plot crystal truncation rod intensity profile
plot_roughness_damping : function
    Plot surface roughness damping for different roughness values
plot_ewald_sphere_2d : function
    Plot 2D cross-section of Ewald sphere construction
plot_ewald_sphere_3d : function
    Plot 3D visualization of Ewald sphere with reciprocal rods
plot_unit_cell_3d : function
    Plot 3D unit cell with lattice vectors
plot_reciprocal_lattice_3d : function
    Plot 3D reciprocal lattice with real-space overlay
plot_crystal_structure_3d : function
    Plot 3D crystal structure with atomic positions

Notes
-----
All plotting functions follow matplotlib conventions and support:
- Optional axis parameter for embedding in multi-panel figures
- Consistent styling with publication-quality defaults
- 3D functions accept elev and azim parameters for viewing angle control
"""

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Element symbols for labeling
ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    12: "Mg",
    13: "Al",
    14: "Si",
    20: "Ca",
    22: "Ti",
    26: "Fe",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    38: "Sr",
    47: "Ag",
    56: "Ba",
    79: "Au",
    82: "Pb",
}

# CPK-style colors for common elements
ELEMENT_COLORS = {
    1: "#FFFFFF",
    6: "#909090",
    7: "#3050F8",
    8: "#FF0D0D",
    12: "#8AFF00",
    13: "#BFA6A6",
    14: "#F0C8A0",
    20: "#3DFF00",
    22: "#BFC2C7",
    26: "#E06633",
    28: "#50D050",
    29: "#C88033",
    30: "#7D80B0",
    38: "#00FF00",
    47: "#C0C0C0",
    56: "#00C900",
    79: "#FFD123",
    82: "#575961",
}


@beartype
def plot_wavelength_curve(
    voltage_range_kv: Tuple[float, float] = (5.0, 30.0),
    n_points: int = 100,
    show_comparison: bool = True,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot electron wavelength vs accelerating voltage.

    Shows the relativistic wavelength formula and optionally compares
    with the non-relativistic approximation.

    Parameters
    ----------
    voltage_range_kv : Tuple[float, float], optional
        Range of voltages to plot in kV. Default: (5.0, 30.0)
    n_points : int, optional
        Number of points to plot. Default: 100
    show_comparison : bool, optional
        If True, also plot non-relativistic approximation. Default: True
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    voltages = np.linspace(voltage_range_kv[0], voltage_range_kv[1], n_points)

    # Relativistic wavelength: lambda = 12.2643 / sqrt(V * (1 + 0.978476e-6 * V))
    voltage_v = voltages * 1000.0
    relativistic_coeff = 0.978476e-6
    h_over_sqrt_2me = 12.2643

    wavelength_rel = h_over_sqrt_2me / np.sqrt(
        voltage_v * (1.0 + relativistic_coeff * voltage_v)
    )

    ax.plot(voltages, wavelength_rel, "b-", linewidth=2, label="Relativistic")

    if show_comparison:
        # Non-relativistic: lambda = 12.2643 / sqrt(V)
        wavelength_nonrel = h_over_sqrt_2me / np.sqrt(voltage_v)
        ax.plot(
            voltages,
            wavelength_nonrel,
            "r--",
            linewidth=1.5,
            label="Non-relativistic",
        )

        # Show percentage difference at max voltage
        rel_diff = (
            (wavelength_nonrel[-1] - wavelength_rel[-1])
            / wavelength_rel[-1]
            * 100
        )
        ax.annotate(
            f"{rel_diff:.1f}% difference\nat {voltages[-1]:.0f} kV",
            xy=(voltages[-1], wavelength_nonrel[-1]),
            xytext=(voltages[-1] - 5, wavelength_nonrel[-1] + 0.005),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    ax.set_xlabel("Accelerating Voltage (kV)", fontsize=12)
    ax.set_ylabel("Wavelength (A)", fontsize=12)
    ax.set_title("Electron Wavelength vs Voltage", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(voltage_range_kv)

    return ax


@beartype
def plot_form_factors(
    atomic_numbers: List[int],
    q_range: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot atomic form factors f(q) for multiple elements.

    Uses Kirkland parameterization for electron scattering.

    Parameters
    ----------
    atomic_numbers : List[int]
        List of atomic numbers to plot (e.g., [14, 8, 38, 22] for Si, O, Sr, Ti)
    q_range : Tuple[float, float], optional
        Range of scattering vector magnitudes in 1/A. Default: (0.0, 10.0)
    n_points : int, optional
        Number of points to plot. Default: 200
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    # Import rheedium functions for form factor calculation
    import jax.numpy as jnp
    from rheedium.simul.form_factors import kirkland_form_factor

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    q_values = np.linspace(q_range[0], q_range[1], n_points)
    q_jax = jnp.array(q_values)  # Convert to JAX array for rheedium functions

    colors = plt.cm.tab10(np.linspace(0, 1, len(atomic_numbers)))

    for i, z in enumerate(atomic_numbers):
        ff_jax = kirkland_form_factor(z, q_jax)
        ff = np.array(ff_jax)  # Convert back to numpy for plotting
        symbol = ELEMENT_SYMBOLS.get(z, f"Z={z}")
        ax.plot(
            q_values,
            ff,
            color=colors[i],
            linewidth=2,
            label=f"{symbol} (Z={z})",
        )

    ax.set_xlabel("q (1/A)", fontsize=12)
    ax.set_ylabel("f(q) (electron units)", fontsize=12)
    ax.set_title("Atomic Form Factors (Kirkland)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_range)
    ax.set_ylim(bottom=0)

    return ax


@beartype
def plot_debye_waller(
    atomic_number: int,
    temperatures: List[float],
    q_range: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot Debye-Waller damping factor at different temperatures.

    Shows how thermal vibrations reduce scattering intensity at high q.

    Parameters
    ----------
    atomic_number : int
        Atomic number of element to plot
    temperatures : List[float]
        List of temperatures in Kelvin to plot
    q_range : Tuple[float, float], optional
        Range of q values in 1/A. Default: (0.0, 10.0)
    n_points : int, optional
        Number of points to plot. Default: 200
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    import jax.numpy as jnp
    from rheedium.simul.form_factors import (
        debye_waller_factor,
        get_mean_square_displacement,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    q_values = np.linspace(q_range[0], q_range[1], n_points)
    q_jax = jnp.array(q_values)  # Convert to JAX array

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
    symbol = ELEMENT_SYMBOLS.get(atomic_number, f"Z={atomic_number}")

    for i, temp in enumerate(temperatures):
        msd = float(get_mean_square_displacement(atomic_number, temp))
        dw_jax = debye_waller_factor(q_jax, msd)
        dw = np.array(dw_jax)  # Convert back to numpy for plotting
        ax.plot(
            q_values,
            dw,
            color=colors[i],
            linewidth=2,
            label=f"T = {temp:.0f} K",
        )

    ax.set_xlabel("q (1/A)", fontsize=12)
    ax.set_ylabel("Debye-Waller Factor exp(-W)", fontsize=12)
    ax.set_title(f"Debye-Waller Damping for {symbol}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_range)
    ax.set_ylim(0, 1.05)

    return ax


@beartype
def plot_ctr_profile(
    l_range: Tuple[float, float] = (-3.0, 3.0),
    n_points: int = 500,
    c_spacing: float = 4.0,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot crystal truncation rod intensity profile I(l).

    Shows the characteristic 1/sin^2(pi*l) modulation with Bragg peaks.

    Parameters
    ----------
    l_range : Tuple[float, float], optional
        Range of l values (Miller index along rod). Default: (-3.0, 3.0)
    n_points : int, optional
        Number of points to plot. Default: 500
    c_spacing : float, optional
        Lattice spacing c in Angstroms (for q_z conversion). Default: 4.0
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    l_values = np.linspace(l_range[0], l_range[1], n_points)

    # CTR intensity: I(l) ~ 1/sin^2(pi*l) for l not integer
    # Add small epsilon to avoid division by zero at Bragg peaks
    epsilon = 0.01
    sin_term = np.sin(np.pi * l_values)
    intensity = 1.0 / (sin_term**2 + epsilon)

    # Normalize for display
    intensity = intensity / intensity.max()

    ax.semilogy(l_values, intensity, "b-", linewidth=2)

    # Mark Bragg peak positions
    bragg_l = np.arange(int(l_range[0]), int(l_range[1]) + 1)
    for l_bragg in bragg_l:
        ax.axvline(
            l_bragg, color="red", linestyle="--", alpha=0.5, linewidth=1
        )

    ax.set_xlabel("l (reciprocal lattice units)", fontsize=12)
    ax.set_ylabel("Intensity (normalized, log scale)", fontsize=12)
    ax.set_title("Crystal Truncation Rod Intensity Profile", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(l_range)

    # Add annotation
    ax.annotate(
        "Bragg peaks",
        xy=(1.0, 1.0),
        xytext=(1.5, 0.5),
        fontsize=10,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    return ax


@beartype
def plot_roughness_damping(
    q_z_range: Tuple[float, float] = (0.0, 5.0),
    sigma_values: List[float] = [0.0, 0.5, 1.0, 2.0],
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot surface roughness damping for different roughness values.

    Shows how surface roughness attenuates CTR intensity at high q_z.

    Parameters
    ----------
    q_z_range : Tuple[float, float], optional
        Range of q_z values in 1/A. Default: (0.0, 5.0)
    sigma_values : List[float], optional
        List of RMS roughness values in Angstroms. Default: [0, 0.5, 1.0, 2.0]
    n_points : int, optional
        Number of points to plot. Default: 200
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    q_z = np.linspace(q_z_range[0], q_z_range[1], n_points)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_values)))

    for i, sigma in enumerate(sigma_values):
        # Roughness damping: exp(-0.5 * q_z^2 * sigma^2)
        damping = np.exp(-0.5 * q_z**2 * sigma**2)
        ax.plot(
            q_z,
            damping,
            color=colors[i],
            linewidth=2,
            label=f"$\\sigma_h$ = {sigma:.1f} A",
        )

    ax.set_xlabel("$q_z$ (1/A)", fontsize=12)
    ax.set_ylabel("Roughness Damping Factor", fontsize=12)
    ax.set_title("Surface Roughness Damping of CTR Intensity", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_z_range)
    ax.set_ylim(0, 1.05)

    return ax


@beartype
def plot_rod_broadening(
    q_perp_range: Tuple[float, float] = (-1.0, 1.0),
    correlation_lengths: List[float] = [10.0, 50.0, 100.0, 500.0],
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot lateral rod broadening for different correlation lengths.

    Shows how finite domain size broadens reciprocal rods.

    Parameters
    ----------
    q_perp_range : Tuple[float, float], optional
        Range of perpendicular q in 1/A. Default: (-1.0, 1.0)
    correlation_lengths : List[float], optional
        Correlation lengths in Angstroms. Default: [10, 50, 100, 500]
    n_points : int, optional
        Number of points to plot. Default: 200
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    q_perp = np.linspace(q_perp_range[0], q_perp_range[1], n_points)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(correlation_lengths)))

    for i, xi in enumerate(correlation_lengths):
        # Gaussian broadening: width ~ 1/xi
        sigma_q = 1.0 / xi
        profile = np.exp(-0.5 * (q_perp / sigma_q) ** 2)
        ax.plot(
            q_perp,
            profile,
            color=colors[i],
            linewidth=2,
            label=f"$\\xi$ = {xi:.0f} A",
        )

    ax.set_xlabel("$q_\\perp$ (1/A)", fontsize=12)
    ax.set_ylabel("Normalized Intensity", fontsize=12)
    ax.set_title("Rod Broadening from Finite Domain Size", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_perp_range)
    ax.set_ylim(0, 1.05)

    return ax


@beartype
def plot_ewald_sphere_2d(
    voltage_kv: float = 15.0,
    theta_deg: float = 2.0,
    lattice_spacing: float = 4.0,
    n_rods: int = 7,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot 2D cross-section of Ewald sphere construction.

    Shows the Ewald sphere, incident/diffracted beams, and reciprocal rods.

    Parameters
    ----------
    voltage_kv : float, optional
        Electron beam voltage in kV. Default: 15.0
    theta_deg : float, optional
        Grazing angle in degrees. Default: 2.0
    lattice_spacing : float, optional
        Real-space lattice parameter in Angstroms. Default: 4.0
    n_rods : int, optional
        Number of reciprocal rods to show. Default: 7
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    # Calculate wavelength and k magnitude
    voltage_v = voltage_kv * 1000.0
    rel_coeff = 0.978476e-6
    wavelength = 12.2643 / np.sqrt(voltage_v * (1.0 + rel_coeff * voltage_v))
    k_mag = 2 * np.pi / wavelength

    # Convert angle to radians
    theta_rad = np.deg2rad(theta_deg)

    # Incident wavevector
    k_in_x = k_mag * np.cos(theta_rad)
    k_in_z = -k_mag * np.sin(theta_rad)

    # Ewald sphere center (at -k_in from origin)
    center_x = -k_in_x
    center_z = -k_in_z

    # Reciprocal lattice spacing
    g_spacing = 2 * np.pi / lattice_spacing

    # Plot the Ewald sphere (only the relevant part)
    theta_sphere = np.linspace(-np.pi / 4, np.pi / 4, 200)
    sphere_x = center_x + k_mag * np.cos(theta_sphere)
    sphere_z = center_z + k_mag * np.sin(theta_sphere)
    ax.plot(sphere_x, sphere_z, "b-", linewidth=2, label="Ewald sphere")

    # Plot reciprocal rods (vertical lines at G_x positions)
    rod_indices = np.arange(-(n_rods // 2), n_rods // 2 + 1)
    for h in rod_indices:
        g_x = h * g_spacing
        ax.axvline(g_x, color="green", linestyle="-", linewidth=1.5, alpha=0.7)
        if h == 0:
            ax.annotate(f"(0,0)", (g_x, 2), fontsize=9, ha="center")
        else:
            ax.annotate(f"({h},0)", (g_x, 2), fontsize=9, ha="center")

    # Plot incident beam
    ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-k_in_x, -k_in_z),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.text(
        -k_in_x / 2 - 0.5,
        -k_in_z / 2 - 0.5,
        "$\\mathbf{k}_{in}$",
        fontsize=12,
        color="red",
    )

    # Plot example diffracted beam
    g_example = g_spacing  # (1,0) rod
    # Find intersection with Ewald sphere
    # Simplified: assume intersection near grazing exit
    k_out_x = k_mag * np.cos(theta_rad)
    k_out_z = k_mag * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(k_out_x, k_out_z),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="purple", lw=2),
    )
    ax.text(
        k_out_x / 2 + 0.5,
        k_out_z / 2 + 0.5,
        "$\\mathbf{k}_{out}$",
        fontsize=12,
        color="purple",
    )

    # Mark origin
    ax.plot(0, 0, "ko", markersize=8)
    ax.text(0.5, -0.5, "O", fontsize=12)

    # Mark sphere center
    ax.plot(center_x, center_z, "b+", markersize=10)

    # Surface line
    ax.axhline(0, color="gray", linestyle="-", linewidth=2)
    ax.text(-8, 0.3, "Surface", fontsize=10, color="gray")

    ax.set_xlabel("$q_x$ (1/A)", fontsize=12)
    ax.set_ylabel("$q_z$ (1/A)", fontsize=12)
    ax.set_title(
        f"Ewald Sphere Construction ({voltage_kv:.0f} kV, "
        f"$\\theta$ = {theta_deg}$^\\circ$)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # Set reasonable limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-3, 5)

    return ax


@beartype
def plot_ewald_sphere_3d(
    voltage_kv: float = 15.0,
    theta_deg: float = 2.0,
    phi_deg: float = 0.0,
    lattice_spacing: float = 4.0,
    n_rods_h: int = 5,
    n_rods_k: int = 5,
    elev: float = 20.0,
    azim: float = 45.0,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D visualization of Ewald sphere with reciprocal rods.

    Parameters
    ----------
    voltage_kv : float, optional
        Electron beam voltage in kV. Default: 15.0
    theta_deg : float, optional
        Grazing angle in degrees. Default: 2.0
    phi_deg : float, optional
        Azimuthal angle in degrees. Default: 0.0
    lattice_spacing : float, optional
        Real-space lattice parameter in Angstroms. Default: 4.0
    n_rods_h : int, optional
        Number of rods in h direction. Default: 5
    n_rods_k : int, optional
        Number of rods in k direction. Default: 5
    elev : float, optional
        Elevation viewing angle in degrees. Default: 20.0
    azim : float, optional
        Azimuthal viewing angle in degrees. Default: 45.0
    ax : Axes3D, optional
        3D matplotlib axes. If None, creates new figure.

    Returns
    -------
    ax : Axes3D
        The 3D matplotlib axes with the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Calculate wavelength and k magnitude
    voltage_v = voltage_kv * 1000.0
    rel_coeff = 0.978476e-6
    wavelength = 12.2643 / np.sqrt(voltage_v * (1.0 + rel_coeff * voltage_v))
    k_mag = 2 * np.pi / wavelength

    # Reciprocal lattice spacing
    g_spacing = 2 * np.pi / lattice_spacing

    # Draw partial Ewald sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 4, 25)  # Only upper hemisphere portion
    sphere_x = k_mag * np.outer(np.cos(u), np.sin(v))
    sphere_y = k_mag * np.outer(np.sin(u), np.sin(v))
    sphere_z = k_mag * np.outer(np.ones(np.size(u)), np.cos(v))

    # Shift sphere so origin is on surface
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    k_in_x = k_mag * np.cos(theta_rad) * np.cos(phi_rad)
    k_in_y = k_mag * np.cos(theta_rad) * np.sin(phi_rad)
    k_in_z = -k_mag * np.sin(theta_rad)

    sphere_x = sphere_x - k_in_x
    sphere_y = sphere_y - k_in_y
    sphere_z = sphere_z - k_in_z

    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.2, color="blue")

    # Draw reciprocal rods
    h_indices = np.arange(-(n_rods_h // 2), n_rods_h // 2 + 1)
    k_indices = np.arange(-(n_rods_k // 2), n_rods_k // 2 + 1)

    for h in h_indices:
        for k in k_indices:
            g_x = h * g_spacing
            g_y = k * g_spacing
            z_range = np.linspace(-2, 5, 2)
            ax.plot(
                [g_x, g_x], [g_y, g_y], z_range, "g-", linewidth=1.5, alpha=0.7
            )

    # Draw incident beam
    ax.quiver(
        -k_in_x,
        -k_in_y,
        -k_in_z,
        k_in_x,
        k_in_y,
        k_in_z,
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2,
    )

    # Mark origin
    ax.scatter([0], [0], [0], color="black", s=50)

    # Surface plane
    xx, yy = np.meshgrid(np.linspace(-8, 8, 2), np.linspace(-8, 8, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color="gray")

    ax.set_xlabel("$q_x$ (1/A)", fontsize=10)
    ax.set_ylabel("$q_y$ (1/A)", fontsize=10)
    ax.set_zlabel("$q_z$ (1/A)", fontsize=10)
    ax.set_title(f"3D Ewald Sphere ({voltage_kv:.0f} kV)", fontsize=12)

    ax.view_init(elev=elev, azim=azim)

    return ax


@beartype
def plot_unit_cell_3d(
    cell_lengths: Tuple[float, float, float] = (4.0, 4.0, 4.0),
    cell_angles: Tuple[float, float, float] = (90.0, 90.0, 90.0),
    elev: float = 20.0,
    azim: float = 45.0,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D unit cell with lattice vectors.

    Parameters
    ----------
    cell_lengths : Tuple[float, float, float], optional
        Lattice parameters (a, b, c) in Angstroms. Default: (4.0, 4.0, 4.0)
    cell_angles : Tuple[float, float, float], optional
        Lattice angles (alpha, beta, gamma) in degrees. Default: (90, 90, 90)
    elev : float, optional
        Elevation viewing angle. Default: 20.0
    azim : float, optional
        Azimuthal viewing angle. Default: 45.0
    ax : Axes3D, optional
        3D matplotlib axes. If None, creates new figure.

    Returns
    -------
    ax : Axes3D
        The 3D matplotlib axes with the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    a, b, c = cell_lengths
    alpha, beta, gamma = [np.deg2rad(ang) for ang in cell_angles]

    # Build lattice vectors
    # a along x-axis
    vec_a = np.array([a, 0, 0])

    # b in xy-plane
    vec_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

    # c general direction
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    vec_c = np.array([cx, cy, cz])

    # Draw cell edges
    origin = np.array([0, 0, 0])
    corners = [
        origin,
        vec_a,
        vec_b,
        vec_c,
        vec_a + vec_b,
        vec_a + vec_c,
        vec_b + vec_c,
        vec_a + vec_b + vec_c,
    ]

    # Edges of the unit cell
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),  # from origin
        (1, 4),
        (1, 5),  # from a
        (2, 4),
        (2, 6),  # from b
        (3, 5),
        (3, 6),  # from c
        (4, 7),
        (5, 7),
        (6, 7),  # to far corner
    ]

    for i, j in edges:
        ax.plot3D(
            [corners[i][0], corners[j][0]],
            [corners[i][1], corners[j][1]],
            [corners[i][2], corners[j][2]],
            "k-",
            linewidth=1,
            alpha=0.5,
        )

    # Draw lattice vectors with arrows
    ax.quiver(
        0,
        0,
        0,
        vec_a[0],
        vec_a[1],
        vec_a[2],
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2,
    )
    ax.quiver(
        0,
        0,
        0,
        vec_b[0],
        vec_b[1],
        vec_b[2],
        color="green",
        arrow_length_ratio=0.1,
        linewidth=2,
    )
    ax.quiver(
        0,
        0,
        0,
        vec_c[0],
        vec_c[1],
        vec_c[2],
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
    )

    # Labels
    ax.text(
        vec_a[0] / 2,
        vec_a[1] / 2 - 0.5,
        vec_a[2] / 2,
        "a",
        color="red",
        fontsize=12,
    )
    ax.text(
        vec_b[0] / 2 - 0.5,
        vec_b[1] / 2,
        vec_b[2] / 2,
        "b",
        color="green",
        fontsize=12,
    )
    ax.text(
        vec_c[0] / 2 - 0.5,
        vec_c[1] / 2,
        vec_c[2] / 2,
        "c",
        color="blue",
        fontsize=12,
    )

    ax.set_xlabel("x (A)", fontsize=10)
    ax.set_ylabel("y (A)", fontsize=10)
    ax.set_zlabel("z (A)", fontsize=10)
    ax.set_title("Unit Cell Lattice Vectors", fontsize=12)

    # Equal aspect ratio
    max_range = max(a, b, c) * 1.2
    ax.set_xlim([-0.5, max_range])
    ax.set_ylim([-0.5, max_range])
    ax.set_zlim([-0.5, max_range])

    ax.view_init(elev=elev, azim=azim)

    return ax


@beartype
def plot_crystal_structure_3d(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell_lengths: Optional[Tuple[float, float, float]] = None,
    cell_angles: Optional[Tuple[float, float, float]] = None,
    elev: float = 20.0,
    azim: float = 45.0,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D crystal structure with atomic positions.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions with shape (N, 3) in Angstroms
    atomic_numbers : np.ndarray
        Atomic numbers with shape (N,)
    cell_lengths : Tuple[float, float, float], optional
        Lattice parameters (a, b, c) to draw unit cell outline
    cell_angles : Tuple[float, float, float], optional
        Lattice angles (alpha, beta, gamma) in degrees
    elev : float, optional
        Elevation viewing angle. Default: 20.0
    azim : float, optional
        Azimuthal viewing angle. Default: 45.0
    ax : Axes3D, optional
        3D matplotlib axes. If None, creates new figure.

    Returns
    -------
    ax : Axes3D
        The 3D matplotlib axes with the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot atoms colored by element
    unique_z = np.unique(atomic_numbers)
    for z in unique_z:
        mask = atomic_numbers == z
        pos_subset = positions[mask]
        color = ELEMENT_COLORS.get(int(z), "#808080")
        symbol = ELEMENT_SYMBOLS.get(int(z), f"Z={z}")
        # Size proportional to atomic number
        size = 50 + z * 2
        ax.scatter(
            pos_subset[:, 0],
            pos_subset[:, 1],
            pos_subset[:, 2],
            c=color,
            s=size,
            label=symbol,
            edgecolors="black",
            linewidth=0.5,
        )

    # Draw unit cell if provided
    if cell_lengths is not None:
        cell_angles = cell_angles or (90.0, 90.0, 90.0)
        a, b, c = cell_lengths
        alpha, beta, gamma = [np.deg2rad(ang) for ang in cell_angles]

        vec_a = np.array([a, 0, 0])
        vec_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0))
        vec_c = np.array([cx, cy, cz])

        origin = np.array([0, 0, 0])
        corners = [
            origin,
            vec_a,
            vec_b,
            vec_c,
            vec_a + vec_b,
            vec_a + vec_c,
            vec_b + vec_c,
            vec_a + vec_b + vec_c,
        ]
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        for i, j in edges:
            ax.plot3D(
                [corners[i][0], corners[j][0]],
                [corners[i][1], corners[j][1]],
                [corners[i][2], corners[j][2]],
                "k--",
                linewidth=1,
                alpha=0.3,
            )

    ax.set_xlabel("x (A)", fontsize=10)
    ax.set_ylabel("y (A)", fontsize=10)
    ax.set_zlabel("z (A)", fontsize=10)
    ax.set_title("Crystal Structure", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)

    ax.view_init(elev=elev, azim=azim)

    return ax


@beartype
def plot_grazing_incidence_geometry(
    theta_deg: float = 2.0,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot grazing incidence geometry diagram.

    Shows beam path, surface, and angle definitions for RHEED.

    Parameters
    ----------
    theta_deg : float, optional
        Grazing angle in degrees. Default: 2.0
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Draw surface
    ax.axhline(0, color="brown", linewidth=3)
    ax.fill_between([-2, 12], [-1, -1], [0, 0], color="tan", alpha=0.3)
    ax.text(5, -0.5, "Sample Surface", fontsize=11, ha="center")

    # Incident beam
    theta_rad = np.deg2rad(theta_deg)
    beam_length = 8
    start_x = 0
    start_y = beam_length * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(beam_length, 0),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.text(2, start_y / 2 + 0.3, "Incident\nBeam", fontsize=10, color="red")

    # Diffracted beam (specular)
    end_x = beam_length + beam_length * np.cos(theta_rad)
    end_y = beam_length * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(beam_length, 0),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.text(
        end_x - 2,
        end_y / 2 + 0.3,
        "Diffracted\nBeam",
        fontsize=10,
        color="blue",
    )

    # Angle arc
    arc_radius = 2
    arc_angles = np.linspace(0, theta_rad, 20)
    arc_x = beam_length + arc_radius * np.cos(arc_angles)
    arc_y = arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, "k-", linewidth=1)
    ax.text(
        beam_length + arc_radius + 0.5,
        0.15,
        f"$\\theta$ = {theta_deg}$^\\circ$",
        fontsize=11,
    )

    # Surface normal
    ax.annotate(
        "",
        xy=(beam_length, 1.5),
        xytext=(beam_length, 0),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax.text(beam_length + 0.2, 1.0, "n", fontsize=11, color="gray")

    ax.set_xlim(-1, 14)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RHEED Grazing Incidence Geometry", fontsize=14)

    return ax


@beartype
def plot_structure_factor_phases(
    atom_positions_2d: List[Tuple[float, float]],
    g_vector: Tuple[float, float] = (1.0, 0.0),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot Argand diagram showing structure factor phase contributions.

    Parameters
    ----------
    atom_positions_2d : List[Tuple[float, float]]
        List of 2D fractional coordinates (x, y) for atoms
    g_vector : Tuple[float, float], optional
        Reciprocal lattice vector (h, k). Default: (1.0, 0.0)
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    h, k = g_vector

    # Calculate phase factors
    colors = plt.cm.tab10(np.linspace(0, 1, len(atom_positions_2d)))
    total_real = 0
    total_imag = 0

    for i, (x, y) in enumerate(atom_positions_2d):
        phase = 2 * np.pi * (h * x + k * y)
        real_part = np.cos(phase)
        imag_part = np.sin(phase)

        # Draw arrow from previous sum to new sum
        ax.annotate(
            "",
            xy=(total_real + real_part, total_imag + imag_part),
            xytext=(total_real, total_imag),
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=2),
        )
        ax.text(
            total_real + real_part / 2 + 0.1,
            total_imag + imag_part / 2 + 0.1,
            f"Atom {i+1}",
            fontsize=9,
            color=colors[i],
        )

        total_real += real_part
        total_imag += imag_part

    # Draw resultant
    ax.annotate(
        "",
        xy=(total_real, total_imag),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=3),
    )
    ax.text(
        total_real / 2 - 0.2,
        total_imag / 2 - 0.2,
        "F(G)",
        fontsize=11,
        fontweight="bold",
    )

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Real", fontsize=12)
    ax.set_ylabel("Imaginary", fontsize=12)
    ax.set_title(
        f"Structure Factor Phase Diagram (G = ({h:.0f}, {k:.0f}))", fontsize=14
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Set limits based on content
    max_val = max(abs(total_real), abs(total_imag), 1.5)
    ax.set_xlim(-max_val - 0.5, max_val + 0.5)
    ax.set_ylim(-max_val - 0.5, max_val + 0.5)

    return ax
