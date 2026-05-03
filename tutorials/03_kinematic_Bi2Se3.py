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
    # Calibrated Kinematic RHEED: Bi2Se3 (001)

    This notebook mirrors the `SrTiO3` tutorial structure, but for pristine
    `Bi2Se3`. The goal is the same: separate the geometric question of where
    the allowed Bragg intersections land from the display question of which of
    those intersections are actually visible in a detector-style image.

    The structure is layered and strongly anisotropic, so `Bi2Se3` makes a good
    contrast case to the more compact cubic `SrTiO3` example.

    We hold one geometry fixed:

    - `theta = 2.5 deg`
    - `phi = 0 deg`
    - square detector pixels
    - perfect-crystal baseline (`surface_roughness = 0`)

    Then we look at:

    1. the sparse Bragg intersections,
    2. the roughness-free dense detector image,
    3. a fixed display cutoff derived from the faintest Bragg-associated pixels,
    4. a roughness sweep at the same detector calibration and display floor.
    """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    import rheedium as rh

    return Path, jnp, np, plt, rh


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root,)


@app.cell
def _(repo_root, rh):
    crystal = rh.inout.parse_cif(
        repo_root / "tests" / "test_data" / "bi2se3" / "Bi2Se3.cif"
    )
    print(
        "Loaded Bi2Se3 with cell = "
        f"[{float(crystal.cell_lengths[0]):.3f}, "
        f"{float(crystal.cell_lengths[1]):.3f}, "
        f"{float(crystal.cell_lengths[2]):.3f}] A"
    )
    return (crystal,)


@app.cell
def _():
    settings = {
        "voltage_kv": 30.0,
        "theta_deg": 2.5,
        "phi_deg": 0.0,
        "hmax": 3,
        "kmax": 3,
        "detector_distance_mm": 80.0,
        "temperature": 300.0,
        "surface_roughness": 0.0,
        "ctr_regularization": 0.01,
        "ctr_power": 1.0,
        "roughness_power": 0.25,
        "image_shape_px": (300, 300),
        "pixel_size_mm": (0.8, 0.8),
        "beam_center_px": (150.0, 0.0),
        "spot_sigma_px": 1.1,
        "angular_divergence_mrad": 0.35,
        "energy_spread_ev": 0.35,
        "psf_sigma_pixels": 1.0,
        "n_angular_samples": 5,
        "n_energy_samples": 3,
        "log_gain": 22.0,
        "dynamic_range_scale": 0.8,
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
    - Detector distance: `{settings["detector_distance_mm"]:.1f}` mm
    - Surface roughness: `{settings["surface_roughness"]:.1f}` A

    As in the `SrTiO3` notebook, `theta` is the grazing incidence angle and
    `phi` is the in-plane azimuthal rotation around the surface normal. Here we
    keep `phi = 0 deg` fixed so the only later changes come from display cutoff
    or roughness, not from changing the in-plane crystal direction.

    Instead of hard-coding a display floor ahead of time, this notebook derives
    one from the simulated perfect-crystal detector image. The workflow is:

    1. simulate the roughness-free dense image,
    2. sample the dense image at the sparse Bragg-hit pixel positions,
    3. find the faintest nonzero Bragg-associated intensity,
    4. set the display floor to `{settings["dynamic_range_scale"]:.1f} x` that value.

    That produces a detector-style cutoff that is tied to the actual Bragg
    family for this geometry rather than to an arbitrary fixed number.
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
    valid = np.asarray(sparse_pattern.G_indices) >= 0
    bragg_points_mm = np.asarray(sparse_pattern.detector_points)[valid]

    x_px = np.rint(
        settings["beam_center_px"][0]
        + bragg_points_mm[:, 0] / settings["pixel_size_mm"][0]
    ).astype(int)
    y_px = np.rint(
        settings["beam_center_px"][1]
        + bragg_points_mm[:, 1] / settings["pixel_size_mm"][1]
    ).astype(int)
    x_px = np.clip(x_px, 0, settings["image_shape_px"][1] - 1)
    y_px = np.clip(y_px, 0, settings["image_shape_px"][0] - 1)
    bragg_pixel_intensities = np.asarray(detector_image)[y_px, x_px]
    positive = bragg_pixel_intensities[bragg_pixel_intensities > 0.0]
    faintest_bragg_pixel = float(positive.min()) if len(positive) else 0.0
    dynamic_range_floor = (
        settings["dynamic_range_scale"] * faintest_bragg_pixel
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
        dynamic_range_floor=dynamic_range_floor,
    )
    extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )
    return (
        bragg_points_mm,
        detector_cutoff_display,
        detector_display,
        dynamic_range_floor,
        extent_mm,
        faintest_bragg_pixel,
        sparse_display,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bragg Spots

    Start with the sparse Ewald intersections. These are the allowed detector
    hits from the scattering geometry before detector-space broadening and
    before any display cutoff is applied.
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
    ax.set_xlim(-120.0, 120.0)
    ax.set_ylim(0.0, 120.0)
    ax.set_title("Sparse Bragg Spots")
    ax.set_xlabel("detector x (mm)")
    ax.set_ylabel("detector y (mm)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(dynamic_range_floor, faintest_bragg_pixel, mo):
    mo.md(
        f"""
    ## Perfect-Crystal RHEED

    The cutoff display uses a floor derived from the simulated image itself:

    - faintest Bragg-associated pixel intensity:
      `{faintest_bragg_pixel:.3e}`
    - display floor:
      `{dynamic_range_floor:.3e}`

    This is still only a display threshold. It does not change the underlying
    detector image or the scattering calculation. It only suppresses the dimmest
    part of the already-simulated intensity scale so the weaker allowed spots
    become easier to inspect.
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
        ax.set_xlim(-120.0, 120.0)
        ax.set_ylim(0.0, 120.0)
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
def _(dynamic_range_floor, mo):
    mo.md(
        f"""
    ## Roughness Sweep

    The panels below keep the detector geometry and display floor fixed while
    varying only the physical roughness parameter.

    The display floor remains `{dynamic_range_floor:.3e}` in every panel so the
    visual comparison is on one consistent scale rather than being autoscaled
    independently.
    """
    )
    return


@app.cell
def _(crystal, dynamic_range_floor, jnp, np, plt, rh, settings):
    roughness_values = [0.0, 0.25, 0.5, 1.0]
    extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )
    cmap = rh.plots.create_phosphor_colormap()
    image_bank = rh.simul.simulate_detector_image_roughness_sweep(
        crystal=crystal,
        surface_roughness_values=jnp.asarray(roughness_values),
        voltage_kv=settings["voltage_kv"],
        theta_deg=settings["theta_deg"],
        phi_deg=settings["phi_deg"],
        hmax=settings["hmax"],
        kmax=settings["kmax"],
        detector_distance_mm=settings["detector_distance_mm"],
        temperature=settings["temperature"],
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, roughness, detector_image in zip(
        axes.flat,
        roughness_values,
        image_bank,
        strict=True,
    ):
        detector_display = rh.simul.log_compress_image(
            detector_image,
            gain=settings["log_gain"],
            dynamic_range_floor=dynamic_range_floor,
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
        ax.set_xlim(-120.0, 120.0)
        ax.set_ylim(0.0, 120.0)
        ax.set_title(f"surface_roughness = {roughness:.2f} A")
        ax.set_xlabel("detector x (mm)")
        ax.set_ylabel("detector y (mm)")
    plt.suptitle(
        "Bi2Se3(001) RHEED With Fixed Display Dynamic Range",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
