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
    # Calibrated Kinematic RHEED: SrTiO3 (001)

    This notebook uses the exact `SrTiO3` setup we validated during the
    simulator calibration work:

    - `theta = 4 deg`
    - `phi = 0 deg`
    - square detector pixels
    - perfect-crystal baseline (`surface_roughness = 0`)

    The goal is to show three things on one consistent detector calibration:

    1. the sparse Ewald/Bragg intersections,
    2. the actual dense RHEED image,
    3. a detector-style dynamic-range cutoff that makes faint Bragg spots
       visible without changing the underlying simulation.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    import rheedium as rh

    return Path, np, plt, rh


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root,)


@app.cell
def _(repo_root, rh):
    crystal = rh.inout.parse_cif(
        repo_root / "tests" / "test_data" / "SrTiO3.cif"
    )
    print(f"Loaded SrTiO3 with a = {float(crystal.cell_lengths[0]):.3f} A")
    return (crystal,)


@app.cell
def _():
    settings = {
        "voltage_kv": 18.0,
        "theta_deg": 4.0,
        "phi_deg": 0.0,
        "hmax": 14,
        "kmax": 14,
        "detector_distance_mm": 900.0,
        "temperature": 300.0,
        "surface_roughness": 0.0,
        "ctr_regularization": 0.01,
        "ctr_power": 1.0,
        "roughness_power": 0.25,
        "image_shape_px": (300, 300),
        "pixel_size_mm": (2.16, 2.16),
        "beam_center_px": (150.0, 0.0),
        "spot_sigma_px": 1.1,
        "angular_divergence_mrad": 0.35,
        "energy_spread_ev": 0.35,
        "psf_sigma_pixels": 1.0,
        "n_angular_samples": 5,
        "n_energy_samples": 3,
        "log_gain": 22.0,
        "dynamic_range_floor": 1.3001876993458826e-05,
    }
    return (settings,)


@app.cell(hide_code=True)
def _(mo, settings):
    mo.md(
        f"""
    ## Fixed Setup

    - Energy: `{settings["voltage_kv"]:.1f}` keV
    - Grazing angle: `{settings["theta_deg"]:.1f}` deg
    - Azimuth: `{settings["phi_deg"]:.1f}` deg
    - CTR grid: `h,k = +/-{settings["hmax"]}`
    - Detector distance: `{settings["detector_distance_mm"]:.0f}` mm
    - Surface roughness: `{settings["surface_roughness"]:.1f}` A
    - Dynamic-range floor: `{settings["dynamic_range_floor"]:.3e}`

    The dynamic-range floor is a display cutoff only. It does not modify the
    underlying sparse or dense simulation values.
    """
    )
    return


@app.cell
def _(crystal, rh, settings):
    sparse_pattern = rh.simul.ewald_simulator(
        crystal=crystal,
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance=settings["detector_distance_mm"],
        temperature=settings["temperature"],
        surface_roughness=settings["surface_roughness"],
        ctr_regularization=settings["ctr_regularization"],
        ctr_power=settings["ctr_power"],
        roughness_power=settings["roughness_power"],
    )
    detector_image = rh.simul.simulate_detector_image(
        crystal=crystal,
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance_mm=settings["detector_distance_mm"],
        temperature=settings["temperature"],
        surface_roughness=settings["surface_roughness"],
        ctr_regularization=settings["ctr_regularization"],
        ctr_power=settings["ctr_power"],
        roughness_power=settings["roughness_power"],
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
        angular_divergence_mrad=settings["angular_divergence_mrad"],
        energy_spread_ev=settings["energy_spread_ev"],
        psf_sigma_pixels=settings["psf_sigma_pixels"],
        n_angular_samples=settings["n_angular_samples"],
        n_energy_samples=settings["n_energy_samples"],
        render_ctrs_as_streaks=True,
    )
    print(f"Sparse intersections: {len(sparse_pattern.intensities)}")
    return detector_image, sparse_pattern


@app.cell
def _(detector_image, np, rh, settings, sparse_pattern):
    sparse_image = rh.simul.render_pattern_to_image(
        pattern=sparse_pattern,
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
    )
    sparse_display = rh.simul.log_compress_image(
        sparse_image,
        gain=settings["log_gain"],
    )
    detector_display = rh.simul.log_compress_image(
        detector_image,
        gain=settings["log_gain"],
    )
    detector_cutoff_display = rh.simul.log_compress_image(
        detector_image,
        gain=settings["log_gain"],
        dynamic_range_floor=settings["dynamic_range_floor"],
    )
    extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )

    valid = np.asarray(sparse_pattern.G_indices) >= 0
    bragg_points_mm = np.asarray(sparse_pattern.detector_points)[valid]
    return (
        bragg_points_mm,
        detector_cutoff_display,
        detector_display,
        extent_mm,
        sparse_display,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bragg Spots

    Start with the sparse Ewald intersections. These are the Bragg/CTR hits on
    the detector before any streak rendering or detector-style display cutoff.
    """
    )
    return


@app.cell
def _(extent_mm, np, plt, rh, sparse_display):
    cmap = rh.plots.create_phosphor_colormap()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(
        np.asarray(sparse_display),
        extent=extent_mm,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlim(-300.0, 300.0)
    ax.set_ylim(0.0, 300.0)
    ax.set_title("Sparse Bragg Spots")
    ax.set_xlabel("detector x (mm)")
    ax.set_ylabel("detector y (mm)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, settings):
    mo.md(
        f"""
    ## Perfect-Crystal RHEED

    The left panel applies the fixed detector-style display cutoff,
    `dynamic_range_floor = {settings["dynamic_range_floor"]:.3e}`, and overlays
    the Bragg spots. The right panel is the plain log-compressed detector image
    from the same roughness-free simulation.
    """
    )
    return


@app.cell
def _(
    bragg_points_mm,
    detector_cutoff_display,
    detector_display,
    extent_mm,
    np,
    plt,
    rh,
):
    cmap = rh.plots.create_phosphor_colormap()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    panels = [
        (
            np.asarray(detector_cutoff_display),
            "Cutoff Display With Bragg Overlay",
        ),
        (np.asarray(detector_display), "Plain Log-Compressed Image"),
    ]
    for ax, (image, title) in zip(axes, panels, strict=True):
        ax.imshow(
            image,
            extent=extent_mm,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlim(-300.0, 300.0)
        ax.set_ylim(0.0, 300.0)
        ax.set_title(title)
        ax.set_xlabel("detector x (mm)")
    axes[0].set_ylabel("detector y (mm)")
    axes[0].scatter(
        bragg_points_mm[:, 0],
        bragg_points_mm[:, 1],
        s=18,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.9,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, settings):
    mo.md(
        f"""
    ## Roughness Sweep

    The panels below keep the same detector calibration and the same display
    cutoff, `dynamic_range_floor = {settings["dynamic_range_floor"]:.3e}`, while
    varying only the physical surface roughness parameter. This makes it easier
    to see how roughness changes the visible spot family without conflating that
    with autoscaled display contrast.
    """
    )
    return


@app.cell
def _(crystal, np, plt, rh, settings):
    roughness_values = [0.0, 0.25, 0.5, 1.0]
    extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )
    cmap = rh.plots.create_phosphor_colormap()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, roughness in zip(axes.flat, roughness_values, strict=True):
        detector_image = rh.simul.simulate_detector_image(
            crystal=crystal,
            voltage_kv=settings["voltage_kv"],
            theta_deg=settings["theta_deg"],
            phi_deg=settings["phi_deg"],
            hmax=settings["hmax"],
            kmax=settings["kmax"],
            detector_distance_mm=settings["detector_distance_mm"],
            temperature=settings["temperature"],
            surface_roughness=roughness,
            ctr_regularization=settings["ctr_regularization"],
            ctr_power=settings["ctr_power"],
            roughness_power=settings["roughness_power"],
            image_shape_px=settings["image_shape_px"],
            pixel_size_mm=settings["pixel_size_mm"],
            beam_center_px=settings["beam_center_px"],
            spot_sigma_px=settings["spot_sigma_px"],
            angular_divergence_mrad=settings["angular_divergence_mrad"],
            energy_spread_ev=settings["energy_spread_ev"],
            psf_sigma_pixels=settings["psf_sigma_pixels"],
            n_angular_samples=settings["n_angular_samples"],
            n_energy_samples=settings["n_energy_samples"],
            render_ctrs_as_streaks=True,
        )
        detector_display = rh.simul.log_compress_image(
            detector_image,
            gain=settings["log_gain"],
            dynamic_range_floor=settings["dynamic_range_floor"],
        )
        ax.imshow(
            np.asarray(detector_display),
            extent=extent_mm,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlim(-300.0, 300.0)
        ax.set_ylim(0.0, 300.0)
        ax.set_title(f"surface_roughness = {roughness:.2f} A")
        ax.set_xlabel("detector x (mm)")
        ax.set_ylabel("detector y (mm)")
    plt.suptitle(
        "SrTiO3(001) RHEED With Fixed Display Dynamic Range",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
