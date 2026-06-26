# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -trusted
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calibrated Kinematic RHEED: SrTiO3 (001)
#
# This notebook is a worked `SrTiO3 (001)` RHEED example built around one
# fixed and calibrated geometry. The goal is not just to produce a plausible
# image, but to make each stage of the simulation interpretable.
#
# We use the exact `SrTiO3` setup that was validated during the simulator
# calibration work:
#
# - `theta = 4 deg`
# - `phi = 0 deg`
# - square detector pixels
# - perfect-crystal baseline (`surface_roughness = 0`)
#
# The notebook is organized around three questions:
#
# 1. Where are the Bragg intersections on the detector?
# 2. How do those discrete intersections turn into a dense RHEED pattern?
# 3. How much of what we "see" depends on the detector display range rather
#    than on the underlying scattering physics?
#
# A recurring theme in this tutorial is that **spot position** and **spot
# visibility** are different problems. The Ewald construction determines where
# the allowed intersections land. The intensity model and the display model
# determine which of those allowed intersections are obvious to the eye.

# %%
from IPython.display import Markdown, display
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import rheedium as rh


# %%
def find_repo_root() -> Path:
    """Find the repository root from Jupyter, VS Code, or a shell."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    for candidate in candidates:
        if (candidate / "tests" / "test_data").exists():
            return candidate
    raise RuntimeError("Could not find tests/test_data.")


repo_root = find_repo_root()

# %%
crystal = rh.inout.parse_cif(repo_root / "tests" / "test_data" / "SrTiO3.cif")
print(f"Loaded SrTiO3 with a = {float(crystal.cell_lengths[0]):.3f} A")

# %%
settings = {
    "energy_kev": 9.0,
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

if os.environ.get("RHEEDIUM_TUTORIAL_FAST_DOCS") == "1":
    settings.update(
        {
            "hmax": 4,
            "kmax": 4,
            "image_shape_px": (160, 160),
            "beam_center_px": (80.0, 0.0),
            "psf_sigma_pixels": 0.0,
            "n_angular_samples": 1,
            "n_energy_samples": 1,
        }
    )

# %%
Markdown(f"""
## Fixed Setup

This tutorial intentionally holds almost everything fixed so that the role
of each parameter is clear.

- Energy: `{settings["energy_kev"]:.1f}` keV
- Grazing angle: `{settings["theta_deg"]:.1f}` deg
- Azimuth: `{settings["phi_deg"]:.1f}` deg
- CTR grid: `h,k = +/-{settings["hmax"]}`
- Detector distance: `{settings["detector_distance_mm"]:.0f}` mm
- Surface roughness: `{settings["surface_roughness"]:.1f}` A
- Dynamic-range floor: `{settings["dynamic_range_floor"]:.3e}`

Two angular parameters matter most for interpreting the geometry:

- `theta` is the grazing incidence angle of the incoming electron beam with
  respect to the sample surface. Increasing `theta` tilts the beam more
  strongly into the crystal, which changes where the Ewald sphere cuts the
  surface rods. In practical detector space, changing `theta` moves the
  visible Bragg family up and down and can change how many low-order spots
  appear in the first visible arc.
- `phi` is the in-plane azimuthal rotation of the sample around the surface
  normal. Changing `phi` rotates which reciprocal-lattice directions are
  aligned with the beam. In practice, this changes the lateral arrangement
  of the diffraction family on the detector and determines which surface
  symmetry direction you are probing.

In this tutorial we set `phi = 0 deg` and keep it there on purpose. That
means every change you see later is coming from intensity or roughness, not
from changing the in-plane orientation.

The `dynamic_range_floor` is a **display cutoff only**. It does not change
the underlying sparse intersections, and it does not alter the detector
image before compression. Instead, it tells the plotting pipeline to hide
everything below a chosen normalized intensity threshold. This is useful
because real phosphor screens and cameras also have a limited visible dynamic
range: a physically present but extremely weak spot can be mathematically in
the image while still being effectively invisible to the detector or to the
person inspecting the frame.
""")

# %%
sparse_pattern = rh.simul.ewald_simulator(
    crystal=crystal,
    energy_kev=settings["energy_kev"],
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
beam = rh.types.BeamSpec(
    energy_kev=settings["energy_kev"],
    theta_deg=settings["theta_deg"],
    phi_deg=settings["phi_deg"],
    angular_divergence_mrad=settings["angular_divergence_mrad"],
    energy_spread_ev=settings["energy_spread_ev"],
)
surface = rh.types.SurfaceCTRParams(
    hmax=settings["hmax"],
    kmax=settings["kmax"],
    temperature=settings["temperature"],
    surface_roughness=settings["surface_roughness"],
    ctr_regularization=settings["ctr_regularization"],
    ctr_power=settings["ctr_power"],
    roughness_power=settings["roughness_power"],
)
detector_geometry = rh.types.DetectorGeometry(
    distance=settings["detector_distance_mm"],
    image_shape_px=settings["image_shape_px"],
    pixel_size_mm=settings["pixel_size_mm"],
    beam_center_px=settings["beam_center_px"],
    psf_sigma_pixels=settings["psf_sigma_pixels"],
)
render = rh.types.RenderParams(
    spot_sigma_px=settings["spot_sigma_px"],
    n_angular_samples=settings["n_angular_samples"],
    n_energy_samples=settings["n_energy_samples"],
    render_ctrs_as_streaks=True,
)
detector_image = rh.simul.simulate_detector_image(
    crystal=crystal,
    beam=beam,
    surface=surface,
    detector=detector_geometry,
    render=render,
)
print(f"Sparse intersections: {len(sparse_pattern.intensities)}")

# %%
sparse_image = rh.simul.render_pattern_to_image(
    pattern=sparse_pattern,
    geometry=detector_geometry,
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
extent_mm = rh.simul.detector_extent_mm(detector_geometry)

valid = np.asarray(sparse_pattern.G_indices) >= 0
bragg_points_mm = np.asarray(sparse_pattern.detector_points)[valid]

# %% [markdown]
# ## Bragg Spots
#
# Start with the sparse Ewald intersections. These are the detector positions
# obtained directly from the scattering geometry before any detector-space
# streak rendering is added.
#
# This panel is the cleanest way to answer the question: **where is the
# simulator saying diffraction is allowed?** Each bright point corresponds to
# an allowed reciprocal-space intersection projected onto the detector. At
# this stage there is no attempt to model extended streak shape, phosphor
# response, or detector cutoff. The image is therefore discrete and idealized.
#
# This separation is important. If the Bragg spots are in the wrong places,
# the problem is geometric. If the Bragg spots are in believable places but
# are hard to see later in the dense pattern, the problem is in intensity
# weighting, broadening, or display range.

# %%
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
_ax.set_xlim(-300.0, 300.0)
_ax.set_ylim(0.0, 300.0)
_ax.set_title("Sparse Bragg Spots")
_ax.set_xlabel("detector x (mm)")
_ax.set_ylabel("detector y (mm)")
plt.tight_layout()
plt.show()

# %%
Markdown(f"""
## Perfect-Crystal RHEED

These two panels use the same roughness-free simulation and the same spot
positions, but they are shown in two different display modes.

The left panel applies the fixed detector-style display cutoff,
`dynamic_range_floor = {settings["dynamic_range_floor"]:.3e}`, and overlays
the Bragg spot locations. The cutoff is chosen so that even very faint Bragg
spots start to become visible without rescaling the image panel-by-panel.
In other words, it is a controlled visibility threshold.

The right panel is the plain log-compressed detector image from the same
simulation, without that extra floor. Comparing the two panels shows why a
display model matters in RHEED: many allowed reflections are physically
present, but only a subset are immediately visible unless the dynamic range
is restricted.

This is also a good place to stress what the cutoff is **not** doing. It is
not adding intensity to missing reflections. It is not moving spots. It is
not changing the scattering calculation. It only suppresses the dimmest part
of the rendered intensity scale so that the eye can separate weak features
from background more easily.
""")

# %%
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
    _ax.set_xlim(-300.0, 300.0)
    _ax.set_ylim(0.0, 300.0)
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

# %%
Markdown(f"""
## Roughness Sweep

The final comparison keeps the detector geometry, the beam settings, and the
display cutoff fixed while varying only the physical surface roughness
parameter.

The display floor remains
`dynamic_range_floor = {settings["dynamic_range_floor"]:.3e}` in every
panel. That matters because it lets you compare roughness cases on a common
visible scale. If each panel were autoscaled independently, a rough sample
could be made to look deceptively similar to a perfect one simply because
the colormap stretched to fill the available range.

In this model, increasing `surface_roughness` damps higher-`q_z` features
more strongly. Qualitatively, that should reduce the visibility of weaker
and higher-order spots first, while leaving the strongest low-order family
relatively easier to see. This sweep is therefore a controlled way to study
how much of the visible RHEED pattern survives as the surface departs from
the perfect-crystal limit.
""")

# %%
roughness_values = [0.0, 0.25, 0.5, 1.0]
_extent_mm = rh.simul.detector_extent_mm(detector_geometry)
_cmap = rh.plots.create_phosphor_colormap()
image_bank = rh.simul.simulate_detector_image_sweep(
    crystal=crystal,
    axis=("surface_roughness", jnp.asarray(roughness_values)),
    beam=beam,
    surface=surface,
    detector=detector_geometry,
    render=render,
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
        dynamic_range_floor=settings["dynamic_range_floor"],
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
    _ax.set_xlim(-300.0, 300.0)
    _ax.set_ylim(0.0, 300.0)
    _ax.set_title(f"surface_roughness = {roughness:.2f} A")
    _ax.set_xlabel("detector x (mm)")
    _ax.set_ylabel("detector y (mm)")
plt.suptitle(
    "SrTiO3(001) RHEED With Fixed Display Dynamic Range",
    fontsize=16,
    y=1.02,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Precomputed Sweep Viewer
#
# The tutorial above works through one fixed geometry in detail. The section
# below switches to precomputed sweep banks stored under
# `tutorials/sweeps`. This is the intended interactive workflow for heavier
# parameter exploration:
#
# - generate the expensive detector-image bank once,
# - save it as a compressed `.npz`,
# - scrub through the bank instantly with a slider.
#
# Here we keep the scope to `SrTiO3` only, so the same notebook now covers
# both the calibrated single-geometry explanation and the fast sweep
# exploration workflow.

# %%
sweep_data = np.load(
    repo_root / "tutorials" / "sweeps" / "sto_theta4_phi_sweep.npz",
    allow_pickle=False,
)
roughness_sweep_data = np.load(
    repo_root / "tutorials" / "sweeps" / "sto_theta4_roughness_sweep.npz",
    allow_pickle=False,
)

# %%
sweep_kind = "phi"
# Change to "roughness" to view the roughness sweep.

# %%
selected_data = sweep_data if sweep_kind == "phi" else roughness_sweep_data
sweep_image_bank = selected_data["image_bank"]
sweep_parameter_values = selected_data["parameter_values"]
sweep_parameter_name = str(selected_data["parameter_name"])
sweep_title_prefix = str(selected_data["title_prefix"])
sweep_extent_mm = selected_data["extent_mm"]
sweep_xlim = selected_data["xlim"]
sweep_ylim = selected_data["ylim"]
sweep_metadata = {
    key: selected_data[key].item()
    if np.asarray(selected_data[key]).shape == ()
    else selected_data[key]
    for key in selected_data.files
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

# %%
sweep_index = 0
# Increase this index to inspect another precomputed image.

# %%
sweep_value = Markdown(
    f"**Current {sweep_parameter_name}:** "
    f"`{sweep_parameter_values[sweep_index]:.3f}`"
)

# %%
sweep_fig, _ax = plt.subplots(figsize=(7, 6))
_ax.imshow(
    np.asarray(sweep_image_bank[sweep_index]),
    extent=sweep_extent_mm,
    origin="lower",
    cmap="inferno",
    aspect="equal",
    vmin=0.0,
    vmax=1.0,
)
_ax.set_xlim(float(sweep_xlim[0]), float(sweep_xlim[1]))
_ax.set_ylim(float(sweep_ylim[0]), float(sweep_ylim[1]))
_ax.set_xlabel("detector x (mm)")
_ax.set_ylabel("detector y (mm)")
_ax.set_title(
    f"SrTiO3 sweep: {sweep_title_prefix} = "
    f"{sweep_parameter_values[sweep_index]:.3f}"
)
sweep_energy_kev = float(
    sweep_metadata.get("energy_kev", settings["energy_kev"])
)
sweep_summary_lines = [
    f"**theta:** `{float(sweep_metadata['theta_deg']):.1f} deg`",
    f"**voltage:** `{sweep_energy_kev:.1f} keV`",
    "**dynamic range floor:** "
    f"`{float(sweep_metadata['dynamic_range_floor']):.3e}`",
]
if "phi_deg" in sweep_metadata:
    sweep_summary_lines.append(
        f"**phi:** `{float(sweep_metadata['phi_deg']):.1f} deg`"
    )
sweep_summary = "\n".join(sweep_summary_lines)

# %%
display(sweep_value)
display(Markdown(sweep_summary))
display(sweep_fig)
