# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.9"]
# ///
"""Run a distributed energy-by-incidence detector grid.

The automaton evaluates a Cartesian grid over electron energy and grazing
incidence angle, using the dense detector-image simulator behind
``rheedium.simul`` and distributing the flattened grid with
``rheedium.tools.distribute_batched``. It writes a numeric grid artifact and a
phosphor heatmap of integrated intensity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import (
    BeamSpec,
    CrystalStructure,
    DetectorGeometry,
    RenderParams,
    SurfaceCTRParams,
)


@jaxtyped(typechecker=beartype)
def _smoke_crystal() -> CrystalStructure:
    """Return a tiny MgO-like cubic crystal for smoke testing."""
    lattice_a: float = 4.21
    frac: Float[Array, "2 4"] = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 12.0],
            [0.5, 0.5, 0.5, 8.0],
        ],
        dtype=jnp.float64,
    )
    cart_xyz: Float[Array, "2 3"] = frac[:, :3] * lattice_a
    cart: Float[Array, "2 4"] = jnp.concatenate(
        [cart_xyz, frac[:, 3:4]],
        axis=1,
    )
    return rh.types.create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=jnp.asarray([lattice_a, lattice_a, lattice_a]),
        cell_angles=jnp.asarray([90.0, 90.0, 90.0]),
    )


def _load_crystal(path: str, *, smoke: bool) -> CrystalStructure:
    """Load a user crystal or generate the smoke fixture."""
    if smoke and not path:
        return _smoke_crystal()
    if not path:
        raise ValueError("crystal is required unless --smoke is set")
    return rh.inout.parse_crystal(path)


def _grid_carriers(
    *,
    hmax: int,
    kmax: int,
    image_size: int,
    spot_sigma_px: float,
) -> tuple[SurfaceCTRParams, DetectorGeometry, RenderParams]:
    """Build carrier objects shared by every grid point."""
    surface = SurfaceCTRParams(hmax=hmax, kmax=kmax, surface_roughness=0.0)
    detector = DetectorGeometry(
        distance=1000.0,
        image_shape_px=(image_size, image_size),
        pixel_size_mm=(6.0, 12.0),
        beam_center_px=(image_size / 2.0, max(1.0, image_size * 0.08)),
        psf_sigma_pixels=0.0,
    )
    render = RenderParams(
        spot_sigma_px=spot_sigma_px,
        n_angular_samples=1,
        n_energy_samples=1,
        render_ctrs_as_streaks=False,
    )
    return surface, detector, render


def _grid_rows(
    energies: Float[Array, "E"],
    thetas: Float[Array, "T"],
) -> Float[Array, "N two"]:
    """Return flattened ``[energy_kev, theta_deg]`` grid rows."""
    energy_grid, theta_grid = jnp.meshgrid(energies, thetas, indexing="ij")
    rows: Float[Array, "N two"] = jnp.stack(
        [energy_grid.ravel(), theta_grid.ravel()],
        axis=-1,
    )
    return rows


@experiment(
    name="parameter-grid",
    params=[
        Param(
            "crystal",
            str,
            default="",
            help="Path to a CIF, XYZ, or POSCAR structure file.",
            example="tests/test_data/SrTiO3.cif",
        ),
        Param(
            "energy_min_kev",
            float,
            default=15.0,
            help="Minimum beam energy.",
            unit="keV",
        ),
        Param(
            "energy_max_kev",
            float,
            default=25.0,
            help="Maximum beam energy.",
            unit="keV",
        ),
        Param("n_energy", int, default=3, help="Number of energy samples."),
        Param(
            "theta_min_deg",
            float,
            default=1.0,
            help="Minimum grazing angle.",
            unit="deg",
        ),
        Param(
            "theta_max_deg",
            float,
            default=3.0,
            help="Maximum grazing angle.",
            unit="deg",
        ),
        Param("n_theta", int, default=3, help="Number of theta samples."),
        Param(
            "phi_deg", float, default=0.0, help="Fixed azimuth.", unit="deg"
        ),
        Param("hmax", int, default=1, help="Maximum absolute h index."),
        Param("kmax", int, default=1, help="Maximum absolute k index."),
        Param("image_size", int, default=48, help="Square detector size."),
        Param("spot_sigma_px", float, default=1.2, help="Spot width."),
    ],
    returns={
        "metrics": {
            "n_grid_points": {"type": "integer"},
            "best_energy_kev": {"type": "number"},
            "best_theta_deg": {"type": "number"},
        },
        "artifacts": {
            "roles": ["grid_summary", "grid_arrays", "grid_heatmap"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the energy-by-theta grid and write diagnostics."""
    hmax: int = min(args.hmax, 1) if args.smoke else args.hmax
    kmax: int = min(args.kmax, 1) if args.smoke else args.kmax
    image_size: int = (
        min(args.image_size, 32) if args.smoke else args.image_size
    )
    n_energy: int = max(
        2, min(args.n_energy, 2) if args.smoke else args.n_energy
    )
    n_theta: int = max(2, min(args.n_theta, 3) if args.smoke else args.n_theta)

    crystal: CrystalStructure = _load_crystal(args.crystal, smoke=args.smoke)
    energies: Float[Array, "E"] = jnp.linspace(
        args.energy_min_kev,
        args.energy_max_kev,
        n_energy,
    )
    thetas: Float[Array, "T"] = jnp.linspace(
        args.theta_min_deg,
        args.theta_max_deg,
        n_theta,
    )
    rows: Float[Array, "N two"] = _grid_rows(energies, thetas)
    surface, detector, render = _grid_carriers(
        hmax=hmax,
        kmax=kmax,
        image_size=image_size,
        spot_sigma_px=args.spot_sigma_px,
    )

    def _run_grid(bank: Float[Array, "N two"]) -> Float[Array, "N H W"]:
        """Run one flattened detector grid for a bank of energy/theta rows."""

        def _simulate_one(row: Float[Array, "two"]) -> Float[Array, "H W"]:
            beam = BeamSpec(
                energy_kev=row[0],
                theta_deg=row[1],
                phi_deg=args.phi_deg,
                angular_divergence_mrad=0.0,
                energy_spread_ev=0.0,
            )
            image: Float[Array, "H W"] = rh.simul.simulate_detector_image_grid(
                crystal=crystal,
                axes=(),
                beam=beam,
                surface=surface,
                detector=detector,
                render=render,
            )
            return image

        images: Float[Array, "N H W"] = jax.vmap(_simulate_one)(bank)
        return images

    image_bank: Float[Array, "N H W"] = rh.tools.distribute_batched(
        _run_grid,
        rows,
    )
    integrated: Float[Array, "N"] = jnp.sum(image_bank, axis=(1, 2))
    heatmap: Float[Array, "E T"] = integrated.reshape((n_energy, n_theta))
    best_index: int = int(jnp.argmax(integrated))
    best_row: Float[Array, "two"] = rows[best_index]
    summary: dict[str, Any] = {
        "energy_kev": np.asarray(energies).tolist(),
        "theta_deg": np.asarray(thetas).tolist(),
        "integrated_intensity": np.asarray(heatmap).tolist(),
        "best": {
            "energy_kev": float(best_row[0]),
            "theta_deg": float(best_row[1]),
            "integrated_intensity": float(integrated[best_index]),
        },
    }

    summary_artifact = ctx.save_json(
        "parameter_grid.json",
        summary,
        role="grid_summary",
    )
    array_artifact = ctx.save_array(
        "parameter_grid.npz",
        {
            "energy_kev": np.asarray(energies),
            "theta_deg": np.asarray(thetas),
            "images": np.asarray(image_bank).reshape(
                (n_energy, n_theta, image_size, image_size)
            ),
            "integrated_intensity": np.asarray(heatmap),
        },
        role="grid_arrays",
    )
    heatmap_artifact = ctx.save_image(
        "parameter_grid_heatmap.png",
        heatmap,
        cmap="phosphor",
        role="grid_heatmap",
    )
    metrics: dict[str, Any] = {
        "n_grid_points": int(rows.shape[0]),
        "grid_shape": [n_energy, n_theta],
        "best_energy_kev": float(best_row[0]),
        "best_theta_deg": float(best_row[1]),
        "best_integrated_intensity": float(integrated[best_index]),
    }
    return {
        "metrics": metrics,
        "artifacts": [summary_artifact, array_artifact, heatmap_artifact],
        "grid": summary,
    }


if __name__ == "__main__":
    main()
