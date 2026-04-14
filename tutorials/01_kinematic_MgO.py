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
    # Full Kinematic RHEED: MgO (001)

    This notebook is intentionally thin. The heavy lifting now lives in
    `rh.simul.simulate_detector_image`, which orchestrates:

    - exact CTR-Ewald intersections
    - detector rasterization
    - angular and energy broadening
    - detector PSF blur

    The notebook just picks parameters and visualizes the result.
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
    crystal = rh.inout.parse_cif(repo_root / "tests" / "test_data" / "MgO.cif")
    print(f"Loaded MgO with a = {float(crystal.cell_lengths[0]):.3f} A")
    return (crystal,)


@app.cell
def _():
    settings = {
        "voltage_kv": 20.0,
        "theta_deg": 2.2,
        "phi_deg": 0.0,
        "hmax": 5,
        "kmax": 5,
        "detector_distance_mm": 1000.0,
        "temperature": 300.0,
        "surface_roughness": 0.45,
        "image_shape_px": (132, 160),
        "pixel_size_mm": (1.8, 4.0),
        "beam_center_px": (80.0, 4.0),
        "spot_sigma_px": 1.4,
        "angular_divergence_mrad": 0.35,
        "energy_spread_ev": 0.35,
        "psf_sigma_pixels": 1.2,
        "n_angular_samples": 5,
        "n_energy_samples": 3,
        "log_gain": 24.0,
    }
    return (settings,)


@app.cell(hide_code=True)
def _(mo, settings):
    mo.md(
        f"""
    ## Parameter Set

    - Energy: `{settings["voltage_kv"]:.1f}` keV
    - Grazing angle: `{settings["theta_deg"]:.1f}` deg
    - Azimuth: `{settings["phi_deg"]:.1f}` deg
    - CTR grid: `h,k = +/-{settings["hmax"]}`
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
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
        angular_divergence_mrad=settings["angular_divergence_mrad"],
        energy_spread_ev=settings["energy_spread_ev"],
        psf_sigma_pixels=settings["psf_sigma_pixels"],
        n_angular_samples=settings["n_angular_samples"],
        n_energy_samples=settings["n_energy_samples"],
    )
    print(f"Sparse intersections: {len(sparse_pattern.intensities)}")
    return detector_image, sparse_pattern


@app.cell
def _(detector_image, np, plt, rh, settings, sparse_pattern):
    sparse_image = rh.simul.render_pattern_to_image(
        pattern=sparse_pattern,
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
        spot_sigma_px=settings["spot_sigma_px"],
    )
    sparse_display = rh.simul.log_compress_image(
        sparse_image, gain=settings["log_gain"]
    )
    detector_display = rh.simul.log_compress_image(
        detector_image, gain=settings["log_gain"]
    )
    extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )
    cmap = rh.plots.create_phosphor_colormap()

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    panels = [
        (np.asarray(sparse_display), "Sparse CTR Intersections"),
        (np.asarray(detector_display), "Broadened Detector Image"),
    ]
    for ax, (image, title) in zip(axes, panels, strict=True):
        ax.imshow(
            image,
            extent=extent_mm,
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("detector x (mm)")
    axes[0].set_ylabel("detector y (mm)")
    plt.suptitle("MgO(001) Kinematic RHEED", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
