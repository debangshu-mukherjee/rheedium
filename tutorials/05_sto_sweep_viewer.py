import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    return Path, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SrTiO3 Sweep Viewer

    This notebook does not recompute the RHEED simulation live. Instead, it
    loads precomputed `SrTiO3` sweep banks from `tutorials/sweeps` and lets you
    scrub through them with sliders.

    That is the intended workflow for interactive exploration:

    - generate expensive sweep banks once with JAX,
    - store them as compressed `.npz` files,
    - inspect them instantly with lightweight widgets.
    """
    )
    return


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    sweeps_dir = repo_root / "tutorials" / "sweeps"
    return repo_root, sweeps_dir


@app.cell
def _(mo, sweeps_dir):
    sweep_kind = mo.ui.dropdown(
        options={
            "Phi sweep": "sto_theta4_phi_sweep.npz",
            "Roughness sweep": "sto_theta4_roughness_sweep.npz",
        },
        value="sto_theta4_phi_sweep.npz",
        label="Sweep bank",
    )
    mo.vstack(
        [
            mo.md(f"Loading sweep banks from `{sweeps_dir}`"),
            sweep_kind,
        ]
    )
    return (sweep_kind,)


@app.cell
def _(np, sweep_kind, sweeps_dir):
    data = np.load(sweeps_dir / sweep_kind.value, allow_pickle=False)
    image_bank = data["image_bank"]
    parameter_values = data["parameter_values"]
    parameter_name = str(data["parameter_name"])
    title_prefix = str(data["title_prefix"])
    extent_mm = data["extent_mm"]
    xlim = data["xlim"]
    ylim = data["ylim"]
    metadata = {
        key: data[key].item()
        if np.asarray(data[key]).shape == ()
        else data[key]
        for key in data.files
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
        extent_mm,
        image_bank,
        metadata,
        parameter_name,
        parameter_values,
        title_prefix,
        xlim,
        ylim,
    )


@app.cell
def _(mo, parameter_name, parameter_values):
    sweep_index = mo.ui.slider(
        start=0,
        stop=len(parameter_values) - 1,
        step=1,
        value=0,
        label=f"{parameter_name} index",
    )
    sweep_value = mo.md(
        f"**Current {parameter_name}:** `{parameter_values[sweep_index.value]:.3f}`"
    )
    mo.vstack([sweep_index, sweep_value])
    return sweep_index, sweep_value


@app.cell
def _(  # noqa: PLR0913
    extent_mm,
    image_bank,
    metadata,
    np,
    parameter_name,
    parameter_values,
    plt,
    sweep_index,
    sweep_kind,
    title_prefix,
    xlim,
    ylim,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        np.asarray(image_bank[sweep_index.value]),
        extent=extent_mm,
        origin="lower",
        cmap="inferno",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel("detector x (mm)")
    ax.set_ylabel("detector y (mm)")
    ax.set_title(
        f"SrTiO3 sweep: {title_prefix} = "
        f"{parameter_values[sweep_index.value]:.3f}"
    )
    summary_lines = [
        f"**Sweep file:** `{sweep_kind.value}`",
        f"**theta:** `{float(metadata['theta_deg']):.1f} deg`",
        f"**voltage:** `{float(metadata['voltage_kv']):.1f} keV`",
    ]
    if "phi_deg" in metadata:
        summary_lines.append(
            f"**phi:** `{float(metadata['phi_deg']):.1f} deg`"
        )
    summary_lines.append(
        "**dynamic range floor:** "
        f"`{float(metadata['dynamic_range_floor']):.3e}`"
    )
    summary = "\n".join(summary_lines)
    return fig, summary


@app.cell(hide_code=True)
def _(fig, mo, sweep_value, summary):
    mo.vstack([sweep_value, mo.md(summary), fig])
    return


if __name__ == "__main__":
    app.run()
