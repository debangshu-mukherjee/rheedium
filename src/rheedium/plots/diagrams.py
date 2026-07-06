"""Visualization functions for RHEED physics and crystallography diagrams.

Extended Summary
----------------
This module provides reusable visualization functions for creating publication-
quality figures explaining RHEED physics concepts. All functions return
matplotlib axes objects and accept optional axis parameters for compositing
multiple plots.

Routine Listings
----------------
:func:`_load_element_symbols`
    Load element symbols from JSON file.
:func:`_load_element_colors`
    Load CPK colors from JSON file.
:func:`plot_wavelength_curve`
    Plot electron wavelength vs accelerating voltage.
:func:`plot_form_factors`
    Plot atomic form factors f(q) for multiple elements.
:func:`plot_debye_waller`
    Plot Debye-Waller damping factor at different temperatures.
:func:`plot_ctr_profile`
    Plot crystal truncation rod intensity profile.
:func:`plot_roughness_damping`
    Plot surface roughness damping for different roughness values.
:func:`plot_rod_broadening`
    Plot lateral rod broadening for different correlation lengths.
:func:`plot_ewald_sphere_2d`
    Plot 2D cross-section of Ewald sphere construction.
:func:`plot_ewald_sphere_3d`
    Plot 3D visualization of Ewald sphere with reciprocal rods.
:func:`plot_unit_cell_3d`
    Plot 3D unit cell with lattice vectors.
:func:`plot_crystal_structure_3d`
    Plot 3D crystal structure with atomic positions.
:func:`plot_grazing_incidence_geometry`
    Plot grazing incidence geometry diagram for RHEED.
:func:`plot_structure_factor_phases`
    Plot Argand diagram showing structure factor phase contributions.
:func:`view_atoms`
    View atoms in a CrystalStructure with 3D visualization.
:func:`view_atoms_interactive`
    View atoms with an ASE-backed interactive notebook widget.

Notes
-----
All plotting functions follow matplotlib conventions and support:
- Optional axis parameter for embedding in multi-panel figures
- Consistent styling with publication-quality defaults
- 3D functions accept elev and azim parameters for viewing angle control
"""

import importlib.util
import json
from pathlib import Path
from typing import Any as TypingAny

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.visualize import view as ase_view
from beartype import beartype
from beartype.typing import Any, Dict, List, Literal, Optional, Tuple, Union
from jaxtyping import Bool, Float, Int
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

from rheedium.inout import to_ase
from rheedium.simul import (
    debye_waller_factor,
    get_mean_square_displacement,
    lobato_form_factor,
)
from rheedium.types import (
    H_OVER_SQRT_2ME_ANG_VSQRT,
    RELATIVISTIC_COEFF_PER_V,
    CrystalStructure,
    SurfaceConfig,
    identify_surface_atoms,
)

_LUGGAGE_DIR: Path = Path(__file__).resolve().parent.parent / "_luggage"
_ATOMS_PATH: Path = _LUGGAGE_DIR / "atom_numbers.json"
_COLORS_PATH: Path = _LUGGAGE_DIR / "atom_colors.json"


def _load_element_symbols() -> Dict[int, str]:
    """Load atomic number to element symbol mapping from JSON file."""
    with open(_ATOMS_PATH, encoding="utf-8") as f:
        symbol_to_z: Dict[str, int] = json.load(f)
    return {z: symbol for symbol, z in symbol_to_z.items()}


def _load_element_colors() -> Dict[int, str]:
    """Load atomic number to CPK color mapping from JSON file."""
    with open(_ATOMS_PATH, encoding="utf-8") as f:
        symbol_to_z: Dict[str, int] = json.load(f)
    with open(_COLORS_PATH, encoding="utf-8") as f:
        symbol_to_color: Dict[str, str] = json.load(f)
    return {
        symbol_to_z[symbol]: color for symbol, color in symbol_to_color.items()
    }


_ELEMENT_SYMBOLS: Dict[int, str] = _load_element_symbols()
_ELEMENT_COLORS: Dict[int, str] = _load_element_colors()
_BeamDirection = Tuple[Union[int, float], Union[int, float], Union[int, float]]


def _validate_supercell(
    supercell: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Validate and normalize an ASE repeat tuple."""
    if len(supercell) != 3:
        raise ValueError("supercell must contain exactly three integers.")
    normalized: Tuple[int, int, int] = (
        int(supercell[0]),
        int(supercell[1]),
        int(supercell[2]),
    )
    if any(v < 1 for v in normalized):
        raise ValueError("supercell repeat counts must be positive.")
    return normalized


def _prepare_interactive_atoms(
    crystal: CrystalStructure,
    *,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    highlight_surface: bool = False,
    surface_config: SurfaceConfig | None = None,
) -> tuple[Atoms, Bool[NDArray, "N"] | None]:
    """Convert a crystal to ASE atoms and compute visualizer metadata."""
    repeat: Tuple[int, int, int] = _validate_supercell(supercell)
    atoms: Atoms = to_ase(crystal)
    base_atom_count: int = len(atoms)
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)

    surface_mask: Bool[NDArray, "N"] | None = None
    if highlight_surface:
        config: SurfaceConfig = surface_config or SurfaceConfig(
            method="height",
            height_fraction=0.3,
        )
        positions: Float[NDArray, "N 3"] = np.asarray(
            atoms.get_positions(),
            dtype=np.float64,
        )
        surface_mask = np.asarray(
            identify_surface_atoms(jnp.asarray(positions), config),
            dtype=bool,
        )
        atoms.set_array("rheedium_surface_mask", surface_mask)

    atoms.info["rheedium_supercell"] = repeat
    atoms.info["rheedium_base_atom_count"] = base_atom_count
    return atoms, surface_mask


def _resolve_interactive_backend(
    backend: str,
) -> Literal["ngl", "x3d"]:
    """Resolve the requested notebook viewer backend."""
    if backend not in {"auto", "ngl", "x3d"}:
        raise ValueError("backend must be one of 'auto', 'ngl', or 'x3d'.")
    has_nglview: bool = importlib.util.find_spec("nglview") is not None
    if backend == "auto":
        return "ngl" if has_nglview else "x3d"
    if backend == "ngl" and not has_nglview:
        raise ImportError(
            "backend='ngl' requires nglview. Install it with "
            "`pip install nglview` or `uv add --optional notebooks nglview`."
        )
    if backend == "ngl":
        return "ngl"
    return "x3d"


def _set_widget_metadata(
    handle: object,
    name: str,
    value: object,
) -> None:
    """Set metadata on dynamic notebook widget objects."""
    setattr(handle, name, value)


def _attach_interactive_metadata(
    handle: object,
    *,
    atoms: Atoms,
    surface_mask: Bool[NDArray, "N"] | None,
    backend: str,
    beam_direction: _BeamDirection | None,
) -> object:
    """Attach rheedium metadata to an interactive viewer handle."""
    _set_widget_metadata(handle, "rheedium_atoms", atoms)
    _set_widget_metadata(handle, "rheedium_surface_mask", surface_mask)
    _set_widget_metadata(handle, "rheedium_backend", backend)
    _set_widget_metadata(handle, "rheedium_beam_direction", beam_direction)
    return handle


def _normalize_beam_direction(
    beam_direction: _BeamDirection | None,
) -> Float[NDArray, "3"] | None:
    """Return a unit beam direction or None."""
    if beam_direction is None:
        return None
    direction: Float[NDArray, "3"] = np.asarray(
        beam_direction,
        dtype=np.float64,
    )
    if direction.shape != (3,):
        raise ValueError("beam_direction must contain exactly three values.")
    norm: float = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("beam_direction must have nonzero length.")
    return direction / norm


def _add_ngl_beam_arrow(
    widget: object,
    atoms: Atoms,
    direction: Float[NDArray, "3"] | None,
) -> None:
    """Best-effort RHEED beam marker for nglview widgets."""
    if direction is None:
        return
    positions: Float[NDArray, "N 3"] = np.asarray(
        atoms.get_positions(),
        dtype=np.float64,
    )
    center: Float[NDArray, "3"] = np.mean(positions, axis=0)
    extent: Float[NDArray, "3"] = np.ptp(positions, axis=0)
    length: float = max(float(np.max(extent)), 1.0)
    start: Float[NDArray, "3"] = center - 0.6 * length * direction
    end: Float[NDArray, "3"] = center + 0.6 * length * direction
    try:
        shape_attr: str = "shape"
        shape: TypingAny = getattr(widget, shape_attr)
        shape.add_arrow(
            start.tolist(),
            end.tolist(),
            [1.0, 0.25, 0.05],
            0.25,
            "RHEED beam",
        )
    except (AttributeError, TypeError, ValueError):
        return


def _find_ngl_widget(widget: object) -> object:
    """Return the nested NGLWidget when ASE wraps it in controls."""
    if hasattr(widget, "add_representation"):
        return widget
    for child in getattr(widget, "children", ()):
        if hasattr(child, "add_representation"):
            return child
    return widget


def _view_atoms_ngl(
    atoms: Atoms,
    *,
    surface_mask: Bool[NDArray, "N"] | None,
    show_cell: bool,
    beam_direction: _BeamDirection | None,
) -> object:
    """Create an nglview widget through ASE."""
    handle: object = ase_view(atoms, viewer="ngl")
    if handle is None:
        raise RuntimeError("ASE ngl viewer did not create a widget.")
    widget: object = _find_ngl_widget(handle)

    add_unitcell: object = getattr(widget, "add_unitcell", None)
    if show_cell and callable(add_unitcell):
        add_unitcell()

    if surface_mask is not None and bool(np.any(surface_mask)):
        indices: str = ",".join(str(i) for i in np.flatnonzero(surface_mask))
        add_representation: object = getattr(
            widget,
            "add_representation",
            None,
        )
        if callable(add_representation):
            add_representation(
                "spacefill",
                selection=f"@{indices}",
                color="#ff6b35",
                radius=0.6,
            )

    direction: Float[NDArray, "3"] | None = _normalize_beam_direction(
        beam_direction
    )
    _add_ngl_beam_arrow(widget, atoms, direction)
    return handle


def _x3d_atom_element(
    atom: Atom,
    *,
    color: tuple[float, float, float],
    radius_scale: float = 1.0,
) -> object:
    """Build one ASE x3d atom element with an optional custom color."""
    from ase.data import covalent_radii  # noqa: PLC0415
    from ase.io.x3d import element, translate  # noqa: PLC0415

    x: float
    y: float
    z: float
    x, y, z = atom.position
    r, g, b = color
    material: object = element("material", diffuseColor=f"{r} {g} {b}")
    appearance: object = element("appearance", child=material)
    radius: float = float(covalent_radii[atom.number]) * radius_scale
    sphere: object = element("sphere", radius=f"{radius}")
    shape: object = element("shape", children=(appearance, sphere))
    return translate(shape, x, y, z)


def _x3d_beam_element(
    atoms: Atoms,
    direction: Float[NDArray, "3"] | None,
) -> object | None:
    """Build a simple x3d line marker for the RHEED beam direction."""
    if direction is None:
        return None
    from ase.io.x3d import element  # noqa: PLC0415

    positions: Float[NDArray, "N 3"] = np.asarray(
        atoms.get_positions(),
        dtype=np.float64,
    )
    center: Float[NDArray, "3"] = np.mean(positions, axis=0)
    extent: Float[NDArray, "3"] = np.ptp(positions, axis=0)
    length: float = max(float(np.max(extent)), 1.0)
    start: Float[NDArray, "3"] = center - 0.7 * length * direction
    end: Float[NDArray, "3"] = center + 0.7 * length * direction
    point: str = (
        f"{start[0]} {start[1]} {start[2]}, {end[0]} {end[1]} {end[2]}"
    )
    material: object = element(
        "material",
        diffuseColor="1.0 0.25 0.05",
        emissiveColor="1.0 0.25 0.05",
    )
    appearance: object = element("appearance", child=material)
    coordinates: object = element("coordinate", point=point)
    line_set: object = element("lineset", vertexCount="2", child=coordinates)
    return element("shape", children=(appearance, line_set))


def _view_atoms_x3d(
    atoms: Atoms,
    *,
    show_cell: bool,
    surface_mask: Bool[NDArray, "N"] | None,
    beam_direction: _BeamDirection | None,
) -> object:
    """Create an ASE x3d widget, raising if IPython HTML is unavailable."""
    if show_cell and surface_mask is None and beam_direction is None:
        handle: object = ase_view(atoms, viewer="x3d")
        if handle is not None:
            return handle

    try:
        from IPython.display import HTML  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "ASE x3d viewer requires IPython.display.HTML. Install IPython "
            "or use backend='ngl' in a notebook with nglview installed."
        ) from exc

    from ase.data.colors import jmol_colors  # noqa: PLC0415
    from ase.io.x3d import (  # noqa: PLC0415
        X3DOM_template,
        element,
        get_maximum_extent,
        group,
        pretty_print,
        translate,
        x3d_wireframe_box,
    )

    atom_elements: list[object] = []
    for index, atom in enumerate(atoms):
        is_surface: bool = surface_mask is not None and bool(
            surface_mask[index]
        )
        color: tuple[float, float, float] = (
            (1.0, 0.42, 0.21)
            if is_surface
            else tuple(float(v) for v in jmol_colors[atom.number])
        )
        atom_elements.append(
            _x3d_atom_element(
                atom,
                color=color,
                radius_scale=1.2 if is_surface else 1.0,
            )
        )

    children: list[object] = []
    if show_cell:
        children.append(x3d_wireframe_box(atoms.cell))
    children.append(group(atom_elements))

    direction: Float[NDArray, "3"] | None = _normalize_beam_direction(
        beam_direction
    )
    beam_element: object | None = _x3d_beam_element(atoms, direction)
    if beam_element is not None:
        children.append(beam_element)

    positions: Float[NDArray, "N 3"] = np.asarray(
        atoms.get_positions(),
        dtype=np.float64,
    )
    if show_cell:
        center: Float[NDArray, "3"] = np.asarray(atoms.cell.diagonal()) / 2.0
        points: Float[NDArray, "M 3"] = np.vstack(
            (positions, atoms.cell[:])  # pyright: ignore[reportIndexIssue]
        )
    else:
        center = np.mean(positions, axis=0)
        points = positions
    max_xyz_extent: Float[NDArray, "3"] = get_maximum_extent(points - center)
    max_dim: float = max(float(np.max(max_xyz_extent)), 1.0)
    viewpoint: object = element(
        "viewpoint",
        position=f"0 0 {max_dim * 2}",
        child=element("group"),
    )
    centered_atoms: object = translate(group(children), *(-center))
    scene: object = element("scene", children=(viewpoint, centered_atoms))
    document: str = X3DOM_template.format(
        scene=pretty_print(scene),
        style='width="400px"; height="300px";',
    )
    return HTML(document)


@beartype
def plot_wavelength_curve(
    energy_range_kev: Tuple[float, float] = (5.0, 30.0),
    n_points: int = 100,
    show_comparison: bool = True,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot electron wavelength vs beam energy.

    Shows the relativistic wavelength formula lambda = 12.2643 / sqrt(V * (1 +
    0.978476e-6 * V)) and optionally compares with the non-relativistic
    approximation lambda = 12.2643 / sqrt(V).

    :see: :class:`~.test_diagrams.TestDiagramPlots`

    Parameters
    ----------
    energy_range_kev : Tuple[float, float], optional
        Range of beam energies to plot in keV. Default: (5.0, 30.0)
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

    Notes
    -----
    1. **Compute Relativistic Wavelength** --
       Evaluate lambda = 12.2643 / sqrt(V * (1 + 0.978476e-6
       * V)) over the requested energy range.
    2. **Optional Non-relativistic Curve** --
       When show_comparison is True, overlay the classical
       approximation and annotate the percentage difference.
    3. **Render Plot** --
       Draw curves with labels, grid, and axis formatting.
    """
    if ax is None:
        fig: Figure
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 6))
    energies_kev: Float[NDArray, "N"] = np.linspace(
        energy_range_kev[0], energy_range_kev[1], n_points
    )
    voltage_v: Float[NDArray, "N"] = energies_kev * 1000.0
    wavelength_rel: Float[NDArray, "N"] = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
        voltage_v * (1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v)
    )
    ax.plot(
        energies_kev,
        wavelength_rel,
        "b-",
        linewidth=2,
        label="Relativistic",
    )
    if show_comparison:
        wavelength_nonrel: Float[NDArray, "N"] = (
            H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(voltage_v)
        )
        ax.plot(
            energies_kev,
            wavelength_nonrel,
            "r--",
            linewidth=1.5,
            label="Non-relativistic",
        )
        rel_diff: float = (
            (wavelength_nonrel[-1] - wavelength_rel[-1])
            / wavelength_rel[-1]
            * 100
        )
        ax.annotate(
            f"{rel_diff:.1f}% difference\nat {energies_kev[-1]:.0f} keV",
            xy=(energies_kev[-1], wavelength_nonrel[-1]),
            xytext=(energies_kev[-1] - 5, wavelength_nonrel[-1] + 0.005),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "gray"},
        )
    ax.set_xlabel("Beam Energy (keV)", fontsize=12)
    ax.set_ylabel("Wavelength (A)", fontsize=12)
    ax.set_title("Electron Wavelength vs Energy", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(energy_range_kev)
    return ax


@beartype
def plot_form_factors(
    atomic_numbers: List[int],
    q_range: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot atomic form factors f(q) for multiple elements.

    Uses Lobato-van Dyck parameterization for electron scattering.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

    Parameters
    ----------
    atomic_numbers : List[int]
        List of atomic numbers to plot (e.g., [14, 8, 38, 22])
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

    Notes
    -----
    1. **Evaluate Form Factors** --
       For each atomic number, call lobato_form_factor
       over the q grid and convert to NumPy.
    2. **Plot Per-element Curves** --
       Assign distinct colors from tab10 and draw each
       element curve with its symbol label.

    See Also
    --------
    lobato_form_factor : Compute atomic form factor for element.
    plot_debye_waller : Plot thermal damping factors.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 6))
    q_values: Float[NDArray, "N"] = np.linspace(
        q_range[0], q_range[1], n_points
    )
    q_jax: Any = jnp.array(q_values)
    colors: Float[NDArray, "N 4"] = plt.get_cmap("tab10")(
        np.linspace(0, 1, len(atomic_numbers))
    )
    for i, z in enumerate(atomic_numbers):
        ff_jax: Any = lobato_form_factor(z, q_jax)
        ff: Float[NDArray, "N"] = np.array(ff_jax)
        symbol: str = _ELEMENT_SYMBOLS.get(z, f"Z={z}")
        ax.plot(
            q_values,
            ff,
            color=colors[i],
            linewidth=2,
            label=f"{symbol} (Z={z})",
        )
    ax.set_xlabel("q (1/A)", fontsize=12)
    ax.set_ylabel("f(q) (electron units)", fontsize=12)
    ax.set_title("Atomic Form Factors (Lobato-van Dyck)", fontsize=14)
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

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Compute MSD Per Temperature** --
       Call get_mean_square_displacement for the given
       element and each temperature.
    2. **Evaluate Damping Curves** --
       Apply debye_waller_factor over the q grid for
       each MSD value and plot with coolwarm colors.

    See Also
    --------
    debye_waller_factor : Compute thermal damping factor.
    get_mean_square_displacement : Get MSD for element and temperature.
    plot_form_factors : Plot atomic form factors.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 6))
    q_values: Float[NDArray, "N"] = np.linspace(
        q_range[0], q_range[1], n_points
    )
    q_jax: Any = jnp.array(q_values)
    colors: Float[NDArray, "N 4"] = plt.get_cmap("coolwarm")(
        np.linspace(0, 1, len(temperatures))
    )
    symbol: str = _ELEMENT_SYMBOLS.get(atomic_number, f"Z={atomic_number}")
    for i, temp in enumerate(temperatures):
        msd: Any = float(get_mean_square_displacement(atomic_number, temp))
        dw_jax: Any = debye_waller_factor(q_jax, msd)
        dw: Float[NDArray, "N"] = np.array(dw_jax)
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
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot crystal truncation rod intensity profile I(l).

    Shows the semi-infinite truncation-rod factor
    1/(1 - 2 e^(-eps) cos(2 pi l) + e^(-2 eps)) with Bragg peaks capped by
    the per-layer attenuation eps (the same model as
    :func:`rheedium.simul.ctr_truncation_intensity`).

    :see: :class:`~.test_diagrams.TestDiagramPlots`

    Parameters
    ----------
    l_range : Tuple[float, float], optional
        Range of l values (Miller index along rod). Default: (-3.0, 3.0)
    n_points : int, optional
        Number of points to plot. Default: 500
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Notes
    -----
    1. **Compute CTR Intensity** --
       Evaluate the semi-infinite truncation factor with attenuation to
       avoid divergence at integer l, then normalize.
    2. **Mark Bragg Peaks** --
       Draw vertical dashed lines at integer l values
       and annotate the Bragg peak positions.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(10, 6))
    l_values: Float[NDArray, "N"] = np.linspace(
        l_range[0], l_range[1], n_points
    )
    layer_attenuation: float = 0.01
    attenuation: float = float(np.exp(-layer_attenuation))
    intensity: Float[NDArray, "N"] = 1.0 / (
        1.0
        - 2.0 * attenuation * np.cos(2.0 * np.pi * l_values)
        + attenuation**2
    )
    intensity = intensity / intensity.max()
    ax.semilogy(l_values, intensity, "b-", linewidth=2)
    bragg_l: Int[NDArray, "N"] = np.arange(
        int(l_range[0]), int(l_range[1]) + 1
    )
    for l_bragg in bragg_l:
        ax.axvline(l_bragg, color="red", linestyle="--", alpha=0.5, lw=1)
    ax.set_xlabel("l (reciprocal lattice units)", fontsize=12)
    ax.set_ylabel("Intensity (normalized, log scale)", fontsize=12)
    ax.set_title("Crystal Truncation Rod Intensity Profile", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(l_range)
    ax.annotate(
        "Bragg peaks",
        xy=(1.0, 1.0),
        xytext=(1.5, 0.5),
        fontsize=10,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red"},
    )
    return ax


@beartype
def plot_roughness_damping(
    q_z_range: Tuple[float, float] = (0.0, 5.0),
    sigma_values: Optional[List[float]] = None,
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot surface roughness damping for different roughness values.

    Shows how surface roughness attenuates CTR **intensity** at high q_z
    using the intensity damping factor exp(-q_z^2 * sigma^2), the square
    of the amplitude factor exp(-q_z^2 * sigma^2 / 2) implemented by
    :func:`~rheedium.simul.roughness_damping`.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Compute Damping Curves** --
       For each sigma value, evaluate the intensity factor
       exp(-q_z^2 * sigma^2) over the q_z grid (amplitude factor
       squared).
    2. **Plot with Color Gradient** --
       Use viridis colormap to distinguish roughness
       levels and label each curve.
    """
    if sigma_values is None:
        sigma_values = [0.0, 0.5, 1.0, 2.0]
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 6))
    q_z: Float[NDArray, "N"] = np.linspace(
        q_z_range[0], q_z_range[1], n_points
    )
    colors: Float[NDArray, "N 4"] = plt.get_cmap("viridis")(
        np.linspace(0, 0.9, len(sigma_values))
    )
    for i, sigma in enumerate(sigma_values):
        # Intensity damping: square of the amplitude factor
        # exp(-q_z^2 sigma^2 / 2) from rheedium.simul.roughness_damping.
        damping: Float[NDArray, "N"] = np.exp(-(q_z**2) * sigma**2)
        ax.plot(
            q_z,
            damping,
            color=colors[i],
            linewidth=2,
            label=f"$\\sigma_h$ = {sigma:.1f} A",
        )
    ax.set_xlabel("$q_z$ (1/A)", fontsize=12)
    ax.set_ylabel(
        "Intensity Damping Factor $e^{-q_z^2\\sigma^2}$", fontsize=12
    )
    ax.set_title("Surface Roughness Damping of CTR Intensity", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_z_range)
    ax.set_ylim(0, 1.05)
    return ax


@beartype
def plot_rod_broadening(
    q_perp_range: Tuple[float, float] = (-1.0, 1.0),
    correlation_lengths: Optional[List[float]] = None,
    n_points: int = 200,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot lateral rod broadening for different correlation lengths.

    Shows how finite domain size broadens reciprocal rods using Gaussian
    broadening with width proportional to 1/correlation_length.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Compute Broadening Profiles** --
       For each correlation length xi, set sigma_q =
       1/xi and evaluate a Gaussian profile in q_perp.
    2. **Plot Per-length Curves** --
       Use plasma colormap to distinguish correlation
       lengths and label each curve.
    """
    if correlation_lengths is None:
        correlation_lengths = [10.0, 50.0, 100.0, 500.0]
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 6))
    q_perp: Float[NDArray, "N"] = np.linspace(
        q_perp_range[0], q_perp_range[1], n_points
    )
    colors: Float[NDArray, "N 4"] = plt.get_cmap("plasma")(
        np.linspace(0.1, 0.9, len(correlation_lengths))
    )
    for i, xi in enumerate(correlation_lengths):
        sigma_q: float = 1.0 / xi
        profile: Float[NDArray, "N"] = np.exp(-0.5 * (q_perp / sigma_q) ** 2)
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
    energy_kev: float = 15.0,
    theta_deg: float = 2.0,
    lattice_spacing: float = 4.0,
    n_rods: int = 7,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot 2D cross-section of Ewald sphere construction.

    Shows the Ewald sphere, incident/diffracted beams, and reciprocal rods.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

    Parameters
    ----------
    energy_kev : float, optional
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

    Notes
    -----
    1. **Compute Wavevector** --
       Convert voltage to relativistic wavelength, then
       derive the wavevector magnitude, k = 2*pi/lambda.
    2. **Draw Ewald Sphere** --
       Trace an arc centered at the back of the incident
       wavevector with radius k.
    3. **Draw Reciprocal Rods** --
       Place vertical lines at integer multiples of the
       reciprocal lattice spacing g = 2*pi/a.
    4. **Annotate Beam Vectors** --
       Draw incident and diffracted k-vectors as arrows
       with labels and mark the surface plane.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(12, 8))
    voltage_v: float = energy_kev * 1000.0
    wavelength: float = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
        voltage_v * (1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v)
    )
    k_mag: float = 2 * np.pi / wavelength
    theta_rad: float = np.deg2rad(theta_deg)
    k_in_x: float = k_mag * np.cos(theta_rad)
    k_in_z: float = -k_mag * np.sin(theta_rad)
    center_x: float = -k_in_x
    center_z: float = -k_in_z
    g_spacing: float = 2 * np.pi / lattice_spacing
    theta_sphere: Float[NDArray, "N"] = np.linspace(-np.pi / 4, np.pi / 4, 200)
    sphere_x: Float[NDArray, "N"] = center_x + k_mag * np.cos(theta_sphere)
    sphere_z: Float[NDArray, "N"] = center_z + k_mag * np.sin(theta_sphere)
    ax.plot(sphere_x, sphere_z, "b-", linewidth=2, label="Ewald sphere")
    rod_indices: Int[NDArray, "N"] = np.arange(-(n_rods // 2), n_rods // 2 + 1)
    for h in rod_indices:
        g_x: float = h * g_spacing
        ax.axvline(g_x, color="green", linestyle="-", linewidth=1.5, alpha=0.7)
        label: str = "(0,0)" if h == 0 else f"({h},0)"
        ax.annotate(label, (g_x, 2), fontsize=9, ha="center")
    ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-k_in_x, -k_in_z),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
    )
    ax.text(
        -k_in_x / 2 - 0.5,
        -k_in_z / 2 - 0.5,
        "$\\mathbf{k}_{in}$",
        fontsize=12,
        color="red",
    )
    k_out_x: float = k_mag * np.cos(theta_rad)
    k_out_z: float = k_mag * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(k_out_x, k_out_z),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": "purple", "lw": 2},
    )
    ax.text(
        k_out_x / 2 + 0.5,
        k_out_z / 2 + 0.5,
        "$\\mathbf{k}_{out}$",
        fontsize=12,
        color="purple",
    )
    ax.plot(0, 0, "ko", markersize=8)
    ax.text(0.5, -0.5, "O", fontsize=12)
    ax.plot(center_x, center_z, "b+", markersize=10)
    ax.axhline(0, color="gray", linestyle="-", linewidth=2)
    ax.text(-8, 0.3, "Surface", fontsize=10, color="gray")
    ax.set_xlabel("$q_x$ (1/A)", fontsize=12)
    ax.set_ylabel("$q_z$ (1/A)", fontsize=12)
    ax.set_title(
        f"Ewald Sphere Construction ({energy_kev:.0f} kV, "
        f"$\\theta$ = {theta_deg}$^\\circ$)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-3, 5)
    return ax


@beartype
def plot_ewald_sphere_3d(
    energy_kev: float = 15.0,
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

    :see: :class:`~.test_diagrams.TestDiagramPlots`

    Parameters
    ----------
    energy_kev : float, optional
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

    Notes
    -----
    1. **Build Sphere Surface** --
        Compute a partial spherical mesh of radius k,
        offset by the incident wavevector direction.
    2. **Place Reciprocal Rod Grid** --
       Draw vertical line segments at each (h, k)
       reciprocal lattice point on the surface plane.
    3. **Draw Incident Beam** --
       Render a 3D quiver arrow for the incident
       wavevector and mark the origin.
    """
    if ax is None:
        fig: Any = plt.figure(figsize=(10, 8))
        ax: Any = fig.add_subplot(111, projection="3d")
    voltage_v: float = energy_kev * 1000.0
    wavelength: float = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
        voltage_v * (1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v)
    )
    k_mag: float = 2 * np.pi / wavelength
    g_spacing: float = 2 * np.pi / lattice_spacing
    u: Float[NDArray, "N"] = np.linspace(0, 2 * np.pi, 50)
    v: Float[NDArray, "N"] = np.linspace(0, np.pi / 4, 25)
    sphere_x: Float[NDArray, "N M"] = k_mag * np.outer(np.cos(u), np.sin(v))
    sphere_y: Float[NDArray, "N M"] = k_mag * np.outer(np.sin(u), np.sin(v))
    sphere_z: Float[NDArray, "N M"] = k_mag * np.outer(
        np.ones(np.size(u)), np.cos(v)
    )
    theta_rad: float = np.deg2rad(theta_deg)
    phi_rad: float = np.deg2rad(phi_deg)
    k_in_x: float = k_mag * np.cos(theta_rad) * np.cos(phi_rad)
    k_in_y: float = k_mag * np.cos(theta_rad) * np.sin(phi_rad)
    k_in_z: float = -k_mag * np.sin(theta_rad)
    sphere_x = sphere_x - k_in_x
    sphere_y = sphere_y - k_in_y
    sphere_z = sphere_z - k_in_z
    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.2, color="blue")
    h_indices: Int[NDArray, "N"] = np.arange(
        -(n_rods_h // 2), n_rods_h // 2 + 1
    )
    k_indices: Int[NDArray, "N"] = np.arange(
        -(n_rods_k // 2), n_rods_k // 2 + 1
    )
    for h in h_indices:
        for k in k_indices:
            g_x: float = h * g_spacing
            g_y: float = k * g_spacing
            z_range: Float[NDArray, "N"] = np.linspace(-2, 5, 2)
            ax.plot(
                [g_x, g_x], [g_y, g_y], z_range, "g-", linewidth=1.5, alpha=0.7
            )
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
    ax.scatter([0], [0], [0], color="black", s=50)
    xx, yy = np.meshgrid(np.linspace(-8, 8, 2), np.linspace(-8, 8, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color="gray")
    ax.set_xlabel("$q_x$ (1/A)", fontsize=10)
    ax.set_ylabel("$q_y$ (1/A)", fontsize=10)
    ax.set_zlabel("$q_z$ (1/A)", fontsize=10)
    ax.set_title(f"3D Ewald Sphere ({energy_kev:.0f} kV)", fontsize=12)
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

    Builds lattice vectors with a along x-axis, b in xy-plane, and c in
    general direction based on the provided cell angles.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Build Lattice Vectors** --
       Place a along x, b in the xy-plane using gamma,
       and c in general direction from alpha and beta.
    2. **Draw Unit Cell Wireframe** --
       Enumerate all 12 edges of the parallelepiped
       and draw each as a thin black line.
    3. **Draw Basis Vectors** --
       Render colored quiver arrows for a (red),
       b (green), and c (blue) from the origin.
    """
    if ax is None:
        fig: Any = plt.figure(figsize=(8, 8))
        ax: Any = fig.add_subplot(111, projection="3d")
    a, b, c = cell_lengths
    alpha, beta, gamma = [np.deg2rad(ang) for ang in cell_angles]
    vec_a: Float[NDArray, "3"] = np.array([a, 0, 0])
    vec_b: Float[NDArray, "3"] = np.array(
        [b * np.cos(gamma), b * np.sin(gamma), 0]
    )
    cx: float = c * np.cos(beta)
    cy: float = (
        c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    )
    cz: float = np.sqrt(c**2 - cx**2 - cy**2)
    vec_c: Float[NDArray, "3"] = np.array([cx, cy, cz])
    origin: Float[NDArray, "3"] = np.array([0, 0, 0])
    corners: List[Float[NDArray, "3"]] = [
        origin,
        vec_a,
        vec_b,
        vec_c,
        vec_a + vec_b,
        vec_a + vec_c,
        vec_b + vec_c,
        vec_a + vec_b + vec_c,
    ]
    edges: List[Tuple[int, int]] = [
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
            "k-",
            linewidth=1,
            alpha=0.5,
        )
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
    max_range: float = max(a, b, c) * 1.2
    ax.set_xlim([-0.5, max_range])
    ax.set_ylim([-0.5, max_range])
    ax.set_zlim([-0.5, max_range])
    ax.view_init(elev=elev, azim=azim)
    return ax


@beartype
def plot_crystal_structure_3d(
    positions: Float[NDArray, "N 3"],
    atomic_numbers: Int[NDArray, "N"],
    cell_lengths: Optional[Tuple[float, float, float]] = None,
    cell_angles: Optional[Tuple[float, float, float]] = None,
    elev: float = 20.0,
    azim: float = 45.0,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D crystal structure with atomic positions.

    Atoms are colored by element using CPK colors and sized proportionally
    to atomic number.

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Group by Element** --
       Identify unique atomic numbers and scatter each
       group with CPK color and scaled marker size.
    2. **Optional Cell Outline** --
       When cell_lengths is provided, build lattice
       vectors and draw the 12-edge wireframe.

    See Also
    --------
    view_atoms : Plot atoms from CrystalStructure object.
    plot_unit_cell_3d : Plot unit cell vectors.
    """
    if ax is None:
        fig: Any = plt.figure(figsize=(10, 8))
        ax: Any = fig.add_subplot(111, projection="3d")
    unique_z: Int[NDArray, "N"] = np.unique(atomic_numbers)
    for z in unique_z:
        mask: Bool[NDArray, "N"] = atomic_numbers == z
        pos_subset: Float[NDArray, "M 3"] = positions[mask]
        color: Any = _ELEMENT_COLORS.get(int(z), "#808080")
        symbol: str = _ELEMENT_SYMBOLS.get(int(z), f"Z={z}")
        size: int = 50 + z * 2
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
    if cell_lengths is not None:
        cell_angles_tuple: Tuple[float, float, float] = cell_angles or (
            90.0,
            90.0,
            90.0,
        )
        a, b, c = cell_lengths
        alpha, beta, gamma = [np.deg2rad(ang) for ang in cell_angles_tuple]
        vec_a: Float[NDArray, "3"] = np.array([a, 0, 0])
        vec_b: Float[NDArray, "3"] = np.array(
            [b * np.cos(gamma), b * np.sin(gamma), 0]
        )
        cx: float = c * np.cos(beta)
        cy: float = (
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        )
        cz: float = np.sqrt(max(c**2 - cx**2 - cy**2, 0))
        vec_c: Float[NDArray, "3"] = np.array([cx, cy, cz])
        origin: Float[NDArray, "3"] = np.array([0, 0, 0])
        corners: List[Float[NDArray, "3"]] = [
            origin,
            vec_a,
            vec_b,
            vec_c,
            vec_a + vec_b,
            vec_a + vec_c,
            vec_b + vec_c,
            vec_a + vec_b + vec_c,
        ]
        edges: List[Tuple[int, int]] = [
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

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Draw Surface** --
       Render a horizontal line and filled region to
       represent the sample surface.
    2. **Draw Beam Geometry** --
       Place incident and diffracted beam arrows at the
       grazing angle theta with labeled annotations.
    3. **Annotate Angles** --
       Draw an arc for theta and a surface normal arrow
       to complete the geometry diagram.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="brown", linewidth=3)
    ax.fill_between([-2, 12], [-1, -1], [0, 0], color="tan", alpha=0.3)
    ax.text(5, -0.5, "Sample Surface", fontsize=11, ha="center")
    theta_rad: float = np.deg2rad(theta_deg)
    beam_length: float = 8
    start_x: float = 0
    start_y: float = beam_length * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(beam_length, 0),
        xytext=(start_x, start_y),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
    )
    ax.text(2, start_y / 2 + 0.3, "Incident\nBeam", fontsize=10, color="red")
    end_x: float = beam_length + beam_length * np.cos(theta_rad)
    end_y: float = beam_length * np.sin(theta_rad)
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(beam_length, 0),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 2},
    )
    ax.text(
        end_x - 2,
        end_y / 2 + 0.3,
        "Diffracted\nBeam",
        fontsize=10,
        color="blue",
    )
    arc_radius: float = 2
    arc_angles: Float[NDArray, "N"] = np.linspace(0, theta_rad, 20)
    arc_x: Float[NDArray, "N"] = beam_length + arc_radius * np.cos(arc_angles)
    arc_y: Float[NDArray, "N"] = arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, "k-", linewidth=1)
    ax.text(
        beam_length + arc_radius + 0.5,
        0.15,
        f"$\\theta$ = {theta_deg}$^\\circ$",
        fontsize=11,
    )
    ax.annotate(
        "",
        xy=(beam_length, 1.5),
        xytext=(beam_length, 0),
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5},
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

    :see: :class:`~.test_diagrams.TestDiagramPlots`

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

    Notes
    -----
    1. **Compute Phase Vectors** --
       For each atom at (x, y), compute phase = 2*pi *
       (h*x + k*y) and the corresponding unit phasor.
    2. **Chain Arrows** --
       Draw each phasor as a colored arrow appended
       tip-to-tail to build the resultant F(G).
    3. **Draw Resultant** --
       Add a bold arrow from the origin to the total
       sum and overlay a unit circle for reference.
    """
    if ax is None:
        fig: Figure
        fig, ax = plt.subplots(figsize=(8, 8))
    h, k = g_vector
    colors: Float[NDArray, "N 4"] = plt.get_cmap("tab10")(
        np.linspace(0, 1, len(atom_positions_2d))
    )
    total_real: float = 0
    total_imag: float = 0
    for i, (x, y) in enumerate(atom_positions_2d):
        phase: float = 2 * np.pi * (h * x + k * y)
        real_part: float = np.cos(phase)
        imag_part: float = np.sin(phase)
        ax.annotate(
            "",
            xy=(total_real + real_part, total_imag + imag_part),
            xytext=(total_real, total_imag),
            arrowprops={"arrowstyle": "->", "color": colors[i], "lw": 2},
        )
        ax.text(
            total_real + real_part / 2 + 0.1,
            total_imag + imag_part / 2 + 0.1,
            f"Atom {i + 1}",
            fontsize=9,
            color=colors[i],
        )
        total_real += real_part
        total_imag += imag_part
    ax.annotate(
        "",
        xy=(total_real, total_imag),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 3},
    )
    ax.text(
        total_real / 2 - 0.2,
        total_imag / 2 - 0.2,
        "F(G)",
        fontsize=11,
        fontweight="bold",
    )
    theta: Float[NDArray, "N"] = np.linspace(0, 2 * np.pi, 100)
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
    max_val: float = max(abs(total_real), abs(total_imag), 1.5)
    ax.set_xlim(-max_val - 0.5, max_val + 0.5)
    ax.set_ylim(-max_val - 0.5, max_val + 0.5)
    return ax


@beartype
def view_atoms_interactive(
    crystal: CrystalStructure,
    *,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    show_cell: bool = True,
    highlight_surface: bool = False,
    surface_config: SurfaceConfig | None = None,
    beam_direction: _BeamDirection | None = None,
    backend: Literal["auto", "ngl", "x3d"] = "auto",
) -> object:
    """View atoms with an ASE-backed interactive notebook widget.

    Parameters
    ----------
    crystal : CrystalStructure
        Structure to visualize.
    supercell : Tuple[int, int, int], optional
        Repeat counts passed to ``ase.Atoms.repeat`` before rendering.
        Default: (1, 1, 1)
    show_cell : bool, optional
        If True, show the unit-cell box when the backend supports it.
        Default: True
    highlight_surface : bool, optional
        If True, identify surface atoms using
        :func:`rheedium.types.identify_surface_atoms` and expose the mask on
        the returned widget as ``rheedium_surface_mask``. NGL backends also add
        a highlighted surface representation. Default: False
    surface_config : SurfaceConfig | None, optional
        Surface identification configuration. If None, uses
        ``SurfaceConfig(method="height", height_fraction=0.3)`` to match the
        simulator default.
    beam_direction : tuple[int | float, int | float, int | float] | None
        Optional RHEED beam direction. NGL backends add a best-effort arrow;
        all backends store the value as ``rheedium_beam_direction``.
    backend : {"auto", "ngl", "x3d"}, optional
        Viewer backend. ``"auto"`` prefers nglview when importable, otherwise
        falls back to ASE x3d. Default: "auto"

    Returns
    -------
    widget : object
        ``nglview.NGLWidget`` for the NGL backend, or an IPython HTML object
        for the ASE x3d backend. The returned object has ``rheedium_atoms``,
        ``rheedium_surface_mask``, ``rheedium_backend``, and
        ``rheedium_beam_direction`` attributes for inspection.

    Notes
    -----
    This function is a display wrapper over :func:`rheedium.inout.to_ase`.
    It does not alter the simulation data model.
    """
    atoms: Atoms
    surface_mask: Bool[NDArray, "N"] | None
    atoms, surface_mask = _prepare_interactive_atoms(
        crystal,
        supercell=supercell,
        highlight_surface=highlight_surface,
        surface_config=surface_config,
    )
    resolved_backend: Literal["ngl", "x3d"] = _resolve_interactive_backend(
        backend
    )
    if resolved_backend == "ngl":
        handle: object = _view_atoms_ngl(
            atoms,
            surface_mask=surface_mask,
            show_cell=show_cell,
            beam_direction=beam_direction,
        )
    else:
        _normalize_beam_direction(beam_direction)
        handle = _view_atoms_x3d(
            atoms,
            show_cell=show_cell,
            surface_mask=surface_mask,
            beam_direction=beam_direction,
        )

    return _attach_interactive_metadata(
        handle,
        atoms=atoms,
        surface_mask=surface_mask,
        backend=resolved_backend,
        beam_direction=beam_direction,
    )


@beartype
def view_atoms(
    crystal: CrystalStructure,
    elev: Union[int, float] = 20.0,
    azim: Union[int, float] = 45.0,
    show_unit_cell: bool = True,
    atom_scale: Union[int, float] = 1.0,
    figsize: Tuple[Union[int, float], Union[int, float]] = (10, 8),
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """View atoms in a CrystalStructure with 3D visualization.

    Creates an interactive 3D matplotlib plot of the crystal structure where
    each atom type is displayed with a different color using CPK coloring.
    Atom sizes are proportional to atomic number with optional scaling.

    :see: :class:`~.test_diagrams.TestViewAtoms`

    Parameters
    ----------
    crystal : CrystalStructure
        The crystal structure containing atomic positions and cell parameters.
        Uses cart_positions for display (columns: x, y, z, atomic_number).
    elev : float, optional
        Elevation viewing angle in degrees. Default: 20.0
    azim : float, optional
        Azimuthal viewing angle in degrees. Default: 45.0
    show_unit_cell : bool, optional
        If True, draw the unit cell outline. Default: True
    atom_scale : float, optional
        Scale factor for atom sizes. Default: 1.0
    figsize : Tuple[float, float], optional
        Figure size in inches. Default: (10, 8)
    ax : Axes3D, optional
        Existing 3D axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : Axes3D
        The 3D matplotlib axes with the plot.

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("MgO.cif")
    >>> ax = rh.plots.view_atoms(crystal, elev=30, azim=60)
    >>> import matplotlib.pyplot as plt
    >>> plt.savefig("crystal_view.png", dpi=150)

    Notes
    -----
    1. **Extract Positions** --
       Read cart_positions from CrystalStructure and
       split into xyz coordinates and atomic numbers.
    2. **Scatter by Element** --
       Group atoms by atomic number, assign CPK colors
       and scaled marker sizes, then scatter in 3D.
    3. **Optional Cell Wireframe** --
       When show_unit_cell is True, build lattice
       vectors from cell parameters and draw edges.

    See Also
    --------
    plot_crystal_structure_3d : Plot structure from raw arrays.
    plot_unit_cell_3d : Plot unit cell vectors.
    parse_cif : Load crystal structure from CIF file.
    """
    if ax is None:
        fig: Any = plt.figure(figsize=figsize)
        ax: Any = fig.add_subplot(111, projection="3d")
    cart_positions: Float[NDArray, "N 4"] = np.asarray(crystal.cart_positions)
    positions: Float[NDArray, "N 3"] = cart_positions[:, :3]
    atomic_numbers: Int[NDArray, "N"] = cart_positions[:, 3].astype(int)
    unique_z: Int[NDArray, "N"] = np.unique(atomic_numbers)
    for z in unique_z:
        mask: Bool[NDArray, "N"] = atomic_numbers == z
        pos_subset: Float[NDArray, "M 3"] = positions[mask]
        color: Any = _ELEMENT_COLORS.get(int(z), "#808080")
        symbol: str = _ELEMENT_SYMBOLS.get(int(z), f"Z={z}")
        size: int = (50 + z * 2) * atom_scale
        ax.scatter(
            pos_subset[:, 0],
            pos_subset[:, 1],
            pos_subset[:, 2],
            c=color,
            s=size,
            label=symbol,
            edgecolors="black",
            linewidth=0.5,
            depthshade=True,
        )
    if show_unit_cell:
        cell_lengths: Float[NDArray, "3"] = np.asarray(crystal.cell_lengths)
        cell_angles: Float[NDArray, "3"] = np.asarray(crystal.cell_angles)
        a, b, c = cell_lengths
        alpha, beta, gamma = np.deg2rad(cell_angles)
        vec_a: Float[NDArray, "3"] = np.array([a, 0, 0])
        vec_b: Float[NDArray, "3"] = np.array(
            [b * np.cos(gamma), b * np.sin(gamma), 0]
        )
        cx: float = c * np.cos(beta)
        cy: float = (
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        )
        cz: float = np.sqrt(max(c**2 - cx**2 - cy**2, 0))
        vec_c: Float[NDArray, "3"] = np.array([cx, cy, cz])
        origin: Float[NDArray, "3"] = np.array([0, 0, 0])
        corners: List[Float[NDArray, "3"]] = [
            origin,
            vec_a,
            vec_b,
            vec_c,
            vec_a + vec_b,
            vec_a + vec_c,
            vec_b + vec_c,
            vec_a + vec_b + vec_c,
        ]
        edges: List[Tuple[int, int]] = [
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


__all__: list[str] = [
    "_load_element_colors",
    "_load_element_symbols",
    "plot_crystal_structure_3d",
    "plot_ctr_profile",
    "plot_debye_waller",
    "plot_ewald_sphere_2d",
    "plot_ewald_sphere_3d",
    "plot_form_factors",
    "plot_grazing_incidence_geometry",
    "plot_rod_broadening",
    "plot_roughness_damping",
    "plot_structure_factor_phases",
    "plot_unit_cell_3d",
    "plot_wavelength_curve",
    "view_atoms",
    "view_atoms_interactive",
]
