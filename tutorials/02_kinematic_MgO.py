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
    # Calibrated Kinematic RHEED: MgO (001)

    This notebook uses the same tutorial structure as the `SrTiO3` and
    `Bi2Se3` kinematic examples, but for `MgO (001)`.

    The goal is to separate three things:

    1. the sparse Bragg intersections set by the Ewald construction,
    2. the dense detector image built from those intersections,
    3. the display cutoff that controls which weak reflections are visible.

    We hold one geometry fixed:

    - `theta = 2.2 deg`
    - `phi = 0 deg`
    - square detector pixels
    - perfect-crystal baseline (`surface_roughness = 0`)
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

    `theta` is the grazing incidence angle and `phi` is the in-plane azimuthal
    rotation about the surface normal. Here `phi = 0 deg` is held fixed so the
    only later changes come from the intensity/display model and roughness.

    As in the `Bi2Se3` notebook, the display floor is derived from the
    roughness-free detector image by sampling the Bragg-hit pixels and using
    the faintest nonzero Bragg-associated intensity as the reference scale.
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
    hits before detector-space streak rendering and before any display cutoff is
    applied.
    """
    )
    return


@app.cell
def _(extent_mm, np, plt, rh, sparse_display):
    _cmap = rh.plots.create_phosphor_colormap()
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.imshow(
        np.asarray(sparse_display),
        extent=extent_mm,
        origin="lower",
        cmap=_cmap,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    _ax.set_xlim(-220.0, 220.0)
    _ax.set_ylim(0.0, 220.0)
    _ax.set_title("Sparse Bragg Spots")
    _ax.set_xlabel("detector x (mm)")
    _ax.set_ylabel("detector y (mm)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(dynamic_range_floor, faintest_bragg_pixel, mo):
    mo.md(
        f"""
    ## Perfect-Crystal RHEED

    - faintest Bragg-associated pixel intensity:
      `{faintest_bragg_pixel:.3e}`
    - display floor:
      `{dynamic_range_floor:.3e}`

    The cutoff is a display threshold only. It does not change the scattering
    calculation or move any spots.
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
    _cmap = rh.plots.create_phosphor_colormap()
    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    _panels = [
        (
            np.asarray(detector_cutoff_display),
            "Cutoff Display With Bragg Overlay",
        ),
        (np.asarray(detector_display), "Plain Log-Compressed Image"),
    ]
    for _ax, (image, title) in zip(_axes, _panels, strict=True):
        _ax.imshow(
            image,
            extent=extent_mm,
            origin="lower",
            cmap=_cmap,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        _ax.set_xlim(-220.0, 220.0)
        _ax.set_ylim(0.0, 220.0)
        _ax.set_title(title)
        _ax.set_xlabel("detector x (mm)")
    _axes[0].set_ylabel("detector y (mm)")
    _axes[0].scatter(
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

    The detector geometry and display floor remain fixed at
    `{dynamic_range_floor:.3e}` while varying only the physical roughness
    parameter.
    """
    )
    return


@app.cell
def _(crystal, dynamic_range_floor, jnp, np, plt, rh, settings):
    roughness_values = [0.0, 0.25, 0.5, 1.0]
    _extent_mm = rh.simul.detector_extent_mm(
        image_shape_px=settings["image_shape_px"],
        pixel_size_mm=settings["pixel_size_mm"],
        beam_center_px=settings["beam_center_px"],
    )
    _cmap = rh.plots.create_phosphor_colormap()
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
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for _ax, roughness, _detector_image in zip(
        _axes.flat,
        roughness_values,
        image_bank,
        strict=True,
    ):
        _detector_display = rh.simul.log_compress_image(
            _detector_image,
            gain=settings["log_gain"],
            dynamic_range_floor=dynamic_range_floor,
        )
        _ax.imshow(
            np.asarray(_detector_display),
            extent=_extent_mm,
            origin="lower",
            cmap=_cmap,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        _ax.set_xlim(-220.0, 220.0)
        _ax.set_ylim(0.0, 220.0)
        _ax.set_title(f"surface_roughness = {roughness:.2f} A")
        _ax.set_xlabel("detector x (mm)")
        _ax.set_ylabel("detector y (mm)")
    plt.suptitle(
        "MgO(001) RHEED With Fixed Display Dynamic Range",
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
    ## Precomputed Sweep Viewer

    This final section loads the precomputed `MgO` sweep banks from
    `tutorials/sweeps` and lets you scrub through them instantly with a slider.
    """
    )
    return


@app.cell
def _(np, repo_root):
    mgo_phi_sweep_data = np.load(
        repo_root / "tutorials" / "sweeps" / "mgo_theta2p2_phi_sweep.npz",
        allow_pickle=False,
    )
    mgo_roughness_sweep_data = np.load(
        repo_root
        / "tutorials"
        / "sweeps"
        / "mgo_theta2p2_roughness_sweep.npz",
        allow_pickle=False,
    )
    return mgo_phi_sweep_data, mgo_roughness_sweep_data


@app.cell
def _(mo):
    mgo_sweep_kind = mo.ui.dropdown(
        options={
            "Phi sweep": "phi",
            "Roughness sweep": "roughness",
        },
        value="phi",
        label="MgO sweep",
    )
    mo.vstack([mgo_sweep_kind])
    return (mgo_sweep_kind,)


@app.cell
def _(mgo_phi_sweep_data, mgo_roughness_sweep_data, mgo_sweep_kind, np):
    _selected_data = (
        mgo_phi_sweep_data
        if mgo_sweep_kind.value == "phi"
        else mgo_roughness_sweep_data
    )
    mgo_sweep_extent_mm = _selected_data["extent_mm"]
    mgo_sweep_image_bank = _selected_data["image_bank"]
    mgo_sweep_parameter_name = str(_selected_data["parameter_name"])
    mgo_sweep_parameter_values = _selected_data["parameter_values"]
    mgo_sweep_title_prefix = str(_selected_data["title_prefix"])
    mgo_sweep_xlim = _selected_data["xlim"]
    mgo_sweep_ylim = _selected_data["ylim"]
    mgo_sweep_metadata = {
        key: _selected_data[key].item()
        if np.asarray(_selected_data[key]).shape == ()
        else _selected_data[key]
        for key in _selected_data.files
        if key
        not in {
            "image_bank",
            "parameter_values",
            "parameter_name",
            "title_prefix",
            "extent_mm",
            "xlim",
            "ylim",
        }
    }
    return (
        mgo_sweep_extent_mm,
        mgo_sweep_image_bank,
        mgo_sweep_metadata,
        mgo_sweep_parameter_name,
        mgo_sweep_parameter_values,
        mgo_sweep_title_prefix,
        mgo_sweep_xlim,
        mgo_sweep_ylim,
    )


@app.cell
def _(mgo_sweep_parameter_name, mgo_sweep_parameter_values, mo):
    mgo_sweep_index = mo.ui.slider(
        start=0,
        stop=len(mgo_sweep_parameter_values) - 1,
        step=1,
        value=0,
        label=f"{mgo_sweep_parameter_name} index",
    )
    mgo_sweep_value = mo.md(
        f"**Current {mgo_sweep_parameter_name}:** "
        f"`{mgo_sweep_parameter_values[mgo_sweep_index.value]:.3f}`"
    )
    mo.vstack([mgo_sweep_index, mgo_sweep_value])
    return mgo_sweep_index, mgo_sweep_value


@app.cell
def _(  # noqa: PLR0913
    mgo_sweep_extent_mm,
    mgo_sweep_image_bank,
    mgo_sweep_index,
    mgo_sweep_metadata,
    mgo_sweep_parameter_values,
    mgo_sweep_title_prefix,
    mgo_sweep_xlim,
    mgo_sweep_ylim,
    np,
    plt,
):
    mgo_sweep_fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.imshow(
        np.asarray(mgo_sweep_image_bank[mgo_sweep_index.value]),
        extent=mgo_sweep_extent_mm,
        origin="lower",
        cmap="inferno",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    _ax.set_xlim(float(mgo_sweep_xlim[0]), float(mgo_sweep_xlim[1]))
    _ax.set_ylim(float(mgo_sweep_ylim[0]), float(mgo_sweep_ylim[1]))
    _ax.set_xlabel("detector x (mm)")
    _ax.set_ylabel("detector y (mm)")
    _ax.set_title(
        f"MgO sweep: {mgo_sweep_title_prefix} = "
        f"{mgo_sweep_parameter_values[mgo_sweep_index.value]:.3f}"
    )
    mgo_sweep_summary = "\n".join(
        [
            f"**theta:** `{float(mgo_sweep_metadata['theta_deg']):.1f} deg`",
            f"**voltage:** `{float(mgo_sweep_metadata['voltage_kv']):.1f} keV`",
            "**dynamic range floor:** "
            f"`{float(mgo_sweep_metadata['dynamic_range_floor']):.3e}`",
            f"**phi:** `{float(mgo_sweep_metadata.get('phi_deg', 0.0)):.1f} deg`",
        ]
    )
    return mgo_sweep_fig, mgo_sweep_summary


@app.cell(hide_code=True)
def _(mgo_sweep_fig, mgo_sweep_summary, mgo_sweep_value, mo):
    mo.vstack(
        [
            mgo_sweep_value,
            mo.md(mgo_sweep_summary),
            mgo_sweep_fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
