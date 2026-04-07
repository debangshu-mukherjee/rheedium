import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Ewald Sphere Geometry

        This notebook is intentionally WASM-safe: it uses only NumPy,
        Matplotlib, and Marimo widgets. The geometry mirrors the same
        relativistic wavelength convention used in ``rheedium.simul``.
        """
    )
    return


@app.cell
def _(mo):
    beam_energy = mo.ui.slider(
        5.0,
        40.0,
        value=20.0,
        step=0.5,
        label="Beam energy (keV)",
    )
    grazing_angle = mo.ui.slider(
        0.5,
        6.0,
        value=2.0,
        step=0.1,
        label="Grazing angle (deg)",
    )
    lattice_spacing = mo.ui.slider(
        2.5,
        6.5,
        value=4.2,
        step=0.1,
        label="In-plane lattice spacing (Angstrom)",
    )
    rod_count = mo.ui.slider(
        2,
        8,
        value=4,
        step=1,
        label="Half-width in rods",
    )

    mo.vstack(
        [
            beam_energy,
            grazing_angle,
            lattice_spacing,
            rod_count,
        ]
    )
    return beam_energy, grazing_angle, lattice_spacing, rod_count


@app.cell
def _(np):
    h_over_sqrt_2me = 12.2643
    relativistic_coeff = 0.978476e-6

    def wavelength_ang(energy_kev: float) -> float:
        voltage_v = float(energy_kev) * 1000.0
        return h_over_sqrt_2me / np.sqrt(
            voltage_v * (1.0 + relativistic_coeff * voltage_v)
        )

    def incident_wavevector_2d(
        lam_ang: float, theta_deg: float
    ) -> tuple[float, float, float]:
        k_mag = 2.0 * np.pi / lam_ang
        theta_rad = np.deg2rad(theta_deg)
        k_x = k_mag * np.cos(theta_rad)
        k_z = -k_mag * np.sin(theta_rad)
        return k_mag, k_x, k_z

    return incident_wavevector_2d, wavelength_ang


@app.cell
def _(
    beam_energy,
    grazing_angle,
    incident_wavevector_2d,
    lattice_spacing,
    mo,
    np,
    plt,
    rod_count,
    wavelength_ang,
):
    lam_ang = wavelength_ang(beam_energy.value)
    k_mag, k_x, k_z = incident_wavevector_2d(lam_ang, grazing_angle.value)
    center_x = -k_x
    center_z = -k_z
    g_spacing = 2.0 * np.pi / lattice_spacing.value

    h_values = np.arange(-rod_count.value, rod_count.value + 1)
    rod_positions = h_values * g_spacing

    x_margin = 1.5 * g_spacing
    x_min = rod_positions.min() - x_margin
    x_max = rod_positions.max() + x_margin
    x_grid = np.linspace(x_min, x_max, 1500)

    radicand = k_mag**2 - (x_grid - center_x) ** 2
    ewald_arc = np.full_like(x_grid, np.nan)
    valid_arc = radicand >= 0.0
    ewald_arc[valid_arc] = center_z + np.sqrt(radicand[valid_arc])

    intersections_x: list[float] = []
    intersections_z: list[float] = []
    visible_indices: list[int] = []

    for h_index, rod_x in zip(h_values, rod_positions, strict=True):
        inner = k_mag**2 - (rod_x - center_x) ** 2
        if inner < 0.0:
            continue
        root = np.sqrt(inner)
        candidate_z = [center_z + root, center_z - root]
        for z_value in candidate_z:
            if z_value >= 0.0:
                intersections_x.append(float(rod_x))
                intersections_z.append(float(z_value))
                visible_indices.append(int(h_index))

    max_intersection = max(intersections_z, default=0.0)
    y_max = max(max_intersection * 1.2, center_z * 2.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        x_grid,
        ewald_arc,
        color="#005f73",
        linewidth=2.5,
        label="Ewald circle",
    )

    for h_index, rod_x in zip(h_values, rod_positions, strict=True):
        color = "#bb3e03" if h_index == 0 else "#94a3b8"
        ax.plot(
            [rod_x, rod_x],
            [0.0, y_max],
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.9,
        )

    if intersections_x:
        ax.scatter(
            intersections_x,
            intersections_z,
            s=50,
            color="#ca6702",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label="Allowed intersections",
        )

    ax.scatter(
        [0.0],
        [0.0],
        s=55,
        color="#0f172a",
        zorder=4,
        label="Reciprocal origin",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel(r"$q_x$ (1/Angstrom)")
    ax.set_ylabel(r"$q_z$ (1/Angstrom)")
    ax.set_title("Reciprocal rods intersecting the Ewald circle")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    summary = mo.md(
        f"""
        **Relativistic wavelength:** `{lam_ang:.4f}` Angstrom

        **Wavevector magnitude:** `{k_mag:.2f}` 1/Angstrom

        **Rod spacing:** `{g_spacing:.3f}` 1/Angstrom

        **Visible rod intersections:** `{len(intersections_x)}`
        """
    )

    mo.vstack([summary, fig])
    return


if __name__ == "__main__":
    app.run()
