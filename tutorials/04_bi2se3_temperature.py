import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bi2Se3 RHEED Patterns: Temperature-Dependent Recrystallization

    This tutorial demonstrates kinematic RHEED simulation for Bi2Se3 structures
    at different recrystallization temperatures. We observe how the diffraction
    patterns evolve across the full Ewald sphere rotation (1-4 degrees).

    ## Structures
    - **Initial**: Starting Bi2Se3 structure
    - **500K**: Recrystallized at 500 K
    - **750K**: Recrystallized at 750 K
    - **1000K**: Recrystallized at 1000 K
    - **1250K**: Recrystallized at 1250 K

    ## Setup
    - **Electron energy**: 30 keV
    - **Grazing angles**: 1° to 4°
    - **Zone axis**: [001] (viewing from z-axis)
    """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import rheedium as rh
    import numpy as np
    from pathlib import Path

    return Path, jnp, np, plt, rh


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load Crystal Structures

    We use `parse_crystal` to directly read XYZ files. The function auto-detects
    the file format and handles lattice information from extended XYZ metadata.
    """
    )
    return


@app.cell
def _(Path, rh):
    data_dir = Path("../tests/test_data/bi2se3")
    pristine = rh.inout.parse_cif(data_dir / "Bi2Se3.cif")
    # Load pristine Bi2Se3 from CIF file
    print(
        f"Pristine: {pristine.cart_positions.shape[0]} atoms, cell = [{pristine.cell_lengths[0]:.2f}, {pristine.cell_lengths[1]:.2f}, {pristine.cell_lengths[2]:.2f}] Å"
    )
    structure_files = {
        "Initial": data_dir / "intial.xyz",
        "500K": data_dir / "500K.final.xyz",
        "750K": data_dir / "750K.final.xyz",
        "1000K": data_dir / "1000K.final.xyz",
        "1250K": data_dir / "1250K.final.xyz",
    }
    crystals = {}
    for _label, filepath in structure_files.items():
        crystals[_label] = rh.inout.parse_crystal(filepath)
        # Define structures with labels
        # Load all structures
        print(
            f"{_label}: {crystals[_label].cart_positions.shape[0]} atoms, cell = [{crystals[_label].cell_lengths[0]:.2f}, {crystals[_label].cell_lengths[1]:.2f}, {crystals[_label].cell_lengths[2]:.2f}] Å"
        )
    return crystals, pristine


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualize Crystal Structures

    First, let's view the atomic arrangements in the pristine unit cell and the
    recrystallized structures. The pristine Bi2Se3 has a rhombohedral structure
    with quintuple layers (Se-Bi-Se-Bi-Se).
    """
    )
    return


@app.cell
def _(plt, pristine, rh):
    # View the pristine Bi2Se3 unit cell
    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(111, projection="3d")
    rh.plots.view_atoms(pristine, elev=10, azim=30, atom_scale=1.5, ax=_ax)
    _ax.set_title("Pristine Bi2Se3 Unit Cell", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(crystals, plt, rh):
    # Compare atomic structures at different recrystallization temperatures
    _fig, _axes = plt.subplots(
        1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"}
    )
    compare_labels = ["Initial", "750K", "1250K"]
    # Select representative structures to compare
    for _ax, _label in zip(_axes, compare_labels):
        rh.plots.view_atoms(
            crystals[_label], elev=5, azim=30, atom_scale=0.3, ax=_ax
        )
        _ax.set_title(f"{_label}", fontsize=14)
    plt.suptitle(
        "Bi2Se3 Structures at Different Recrystallization Temperatures",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulation Parameters
    """
    )
    return


@app.cell
def _(jnp, rh):
    # RHEED parameters
    voltage_kV = 30.0
    hmax, kmax = 3, 3
    detector_distance = 80.0  # mm

    # Grazing angles for Ewald sphere rotation
    theta_range = [1.0, 2.0, 3.0, 4.0]  # degrees

    # Electron wavelength and wavevector
    wavelength = rh.simul.wavelength_ang(voltage_kV)
    k_magnitude = 2 * jnp.pi / wavelength
    print(f"Electron wavelength: {float(wavelength):.4f} Å")
    print(f"Electron energy: {voltage_kV} keV")
    print(f"Wavevector magnitude: {float(k_magnitude):.2f} Å⁻¹")
    return detector_distance, hmax, kmax, theta_range, voltage_kV


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RHEED Patterns at Different Grazing Angles

    For each structure, we simulate the RHEED pattern at grazing angles from 1° to 4°.
    This shows the full Ewald sphere rotation and how different reflections become active.

    **Note on l-range**: These structures have a large unit cell (~100 Å along c-axis),
    resulting in a small reciprocal lattice spacing (c* ≈ 0.063 Å⁻¹). To satisfy the
    Ewald sphere condition (k_out_z > 0), we need l values around 50-150. We calculate
    the appropriate range based on the crystal geometry.
    """
    )
    return


@app.cell
def _(crystals, jnp, rh, voltage_kV):
    def compute_l_range(crystal, voltage_kv, theta_deg):
        """Compute appropriate l-range for Ewald sphere intersection.

        For large unit cells, the reciprocal lattice is dense and we need
        large l values to satisfy k_out_z > 0.
        """
        c_length = crystal.cell_lengths[2]  # Get c* magnitude
        c_star = 2 * jnp.pi / c_length
        wavelength = rh.simul.wavelength_ang(voltage_kv)
        theta_rad = jnp.radians(theta_deg)
        k_in_z = (
            -2 * jnp.pi / wavelength * jnp.sin(theta_rad)
        )  # Incident wavevector z-component (negative for grazing incidence)
        l_min = float(-k_in_z / c_star) + 1.0
        l_max = l_min + 100.0
        return (l_min, l_max)

    def simulate_rheed_pattern(
        crystal,
        theta_deg,
        voltage_kv=30.0,
        hmax=3,
        kmax=3,
        detector_distance=80.0,
    ):  # Minimum l for k_out_z > 0: l * c* > -k_in_z
        """Simulate kinematic CTR RHEED pattern with auto-computed l-range."""  # Add margin
        l_min, l_max = compute_l_range(
            crystal, voltage_kv, theta_deg
        )  # Sample 100 l-units
        _pattern = rh.simul.kinematic_ctr_simulator(
            crystal=crystal,
            voltage_kv=voltage_kv,
            theta_deg=theta_deg,
            hmax=hmax,
            kmax=kmax,
            detector_distance=detector_distance,
            l_min=l_min,
            l_max=l_max,
            n_points_per_rod=200,
        )
        return _pattern

    example_crystal = crystals["Initial"]
    l_min, l_max = compute_l_range(example_crystal, voltage_kV, 2.0)
    print(f"For c = {float(example_crystal.cell_lengths[2]):.1f} Å at θ = 2°:")
    print(f"  c* = {2 * jnp.pi / example_crystal.cell_lengths[2]:.4f} Å⁻¹")
    # Show l-range for initial structure at 2 degrees
    print(f"  l-range: [{l_min:.1f}, {l_max:.1f}]")
    return (simulate_rheed_pattern,)


@app.cell
def _(
    crystals,
    detector_distance,
    hmax,
    kmax,
    simulate_rheed_pattern,
    theta_range,
    voltage_kV,
):
    # Simulate patterns for all structures and angles
    patterns = {}
    for _label, crystal in crystals.items():
        patterns[_label] = {}
        for _theta in theta_range:
            patterns[_label][_theta] = simulate_rheed_pattern(
                crystal, _theta, voltage_kV, hmax, kmax, detector_distance
            )
        print(f"Simulated {_label}: {len(theta_range)} angles")
    return (patterns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualization: Full Ewald Sphere Rotation

    Each row shows a different structure (temperature), each column shows a different grazing angle.
    """
    )
    return


@app.cell
def _(np):
    def render_pattern(
        pattern,
        grid_size=300,
        spot_width=0.03,
        x_range=(-1.0, 1.0),
        y_range=(0, 6),
    ):
        """Render RHEED pattern to image array."""
        valid = _pattern.intensities > 0  # Filter to valid points only
        x_np = np.asarray(_pattern.detector_points[valid, 0])
        y_np = np.asarray(_pattern.detector_points[valid, 1])
        i_np = np.asarray(_pattern.intensities[valid])
        _x_axis = np.linspace(x_range[0], x_range[1], grid_size)
        _y_axis = np.linspace(y_range[0], y_range[1], grid_size)
        xx, yy = np.meshgrid(_x_axis, _y_axis, indexing="xy")
        _image = np.zeros_like(xx)
        for idx in range(len(i_np)):
            _image += i_np[idx] * np.exp(
                -((xx - x_np[idx]) ** 2 + (yy - y_np[idx]) ** 2)
                / (2 * spot_width**2)
            )
        return (_image, _x_axis, _y_axis)

    return (render_pattern,)


@app.cell
def _(crystals, patterns, plt, render_pattern, theta_range):
    # Create comprehensive figure
    structure_labels = list(crystals.keys())
    n_structures = len(structure_labels)
    n_angles = len(theta_range)
    _fig, _axes = plt.subplots(
        n_structures, n_angles, figsize=(4 * n_angles, 4 * n_structures)
    )
    for _i, _label in enumerate(structure_labels):
        for _j, _theta in enumerate(theta_range):
            _ax = _axes[_i, _j]
            _pattern = patterns[_label][_theta]
            _image, _x_axis, _y_axis = render_pattern(_pattern)
            _ax.imshow(
                _image,
                extent=[_x_axis[0], _x_axis[-1], _y_axis[0], _y_axis[-1]],
                origin="lower",
                cmap="Greens",
                aspect="auto",
            )
            if _i == 0:
                _ax.set_title(f"θ = {_theta}°", fontsize=14)
            if _j == 0:
                _ax.set_ylabel(f"{_label}\ny_d (mm)", fontsize=12)
            else:
                _ax.set_ylabel("")
            if _i == n_structures - 1:
                _ax.set_xlabel("x_d (mm)", fontsize=12)
            else:
                _ax.set_xlabel("")
    plt.suptitle(
        "Bi2Se3 RHEED Patterns: Temperature-Dependent Recrystallization",
        fontsize=16,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()
    return n_structures, structure_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Single Structure Comparison: Ewald Sphere Rotation

    Let's examine the initial structure in detail across all grazing angles.
    """
    )
    return


@app.cell
def _(patterns, plt, render_pattern, theta_range):
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))
    for _j, _theta in enumerate(theta_range):
        _ax = _axes[_j]
        _pattern = patterns["Initial"][_theta]
        _image, _x_axis, _y_axis = render_pattern(_pattern, grid_size=400)
        _ax.imshow(
            _image,
            extent=[_x_axis[0], _x_axis[-1], _y_axis[0], _y_axis[-1]],
            origin="lower",
            cmap="Greens",
            aspect="auto",
        )
        _ax.set_title(f"θ = {_theta}°", fontsize=14)
        _ax.set_xlabel("x_d (mm)", fontsize=12)
        if _j == 0:
            _ax.set_ylabel("y_d (mm)", fontsize=12)
    plt.suptitle(
        "Initial Bi2Se3 Structure: Ewald Sphere Rotation", fontsize=16, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Temperature Comparison at Fixed Angle

    Compare all temperature conditions at a fixed grazing angle (2°).
    """
    )
    return


@app.cell
def _(n_structures, patterns, plt, render_pattern, structure_labels):
    theta_fixed = 2.0
    _fig, _axes = plt.subplots(1, n_structures, figsize=(4 * n_structures, 4))
    for _i, _label in enumerate(structure_labels):
        _ax = _axes[_i]
        _pattern = patterns[_label][theta_fixed]
        _image, _x_axis, _y_axis = render_pattern(_pattern, grid_size=400)
        _ax.imshow(
            _image,
            extent=[_x_axis[0], _x_axis[-1], _y_axis[0], _y_axis[-1]],
            origin="lower",
            cmap="Greens",
            aspect="auto",
        )
        _ax.set_title(f"{_label}", fontsize=14)
        _ax.set_xlabel("x_d (mm)", fontsize=12)
        if _i == 0:
            _ax.set_ylabel("y_d (mm)", fontsize=12)
    plt.suptitle(
        f"Bi2Se3 Recrystallization at θ = {theta_fixed}°", fontsize=16, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary Statistics
    """
    )
    return


@app.cell
def _(jnp, patterns, structure_labels):
    print("Pattern Statistics (θ = 2.0°):")
    print("=" * 60)
    print(
        f"{'Structure':<12} {'Streak Points':<15} {'Unique Rods':<15} {'Max Intensity':<15}"
    )
    print("-" * 60)
    for _label in structure_labels:
        _pattern = patterns[_label][2.0]
        n_points = len(_pattern.intensities)
        n_rods = len(jnp.unique(_pattern.G_indices))
        max_i = float(jnp.max(_pattern.intensities))
        print(f"{_label:<12} {n_points:<15} {n_rods:<15} {max_i:<15.4f}")
    return


if __name__ == "__main__":
    app.run()
