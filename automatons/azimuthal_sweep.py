# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.9"]
# ///
"""Run a distributed azimuthal detector-image sweep.

The automaton loads a crystal, sweeps the in-plane azimuth angle through
``rheedium.simul.simulate_detector_image_sweep``, distributes the batched
sweep with ``rheedium.tools.distribute_batched``, and emits per-angle
intensity diagnostics plus phosphor previews. Smoke mode uses a tiny MgO-like
crystal and a small detector grid.
"""

from __future__ import annotations

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


def _carriers(
    args: Any,
    *,
    hmax: int,
    kmax: int,
    image_size: int,
) -> tuple[BeamSpec, SurfaceCTRParams, DetectorGeometry, RenderParams]:
    """Build small carrier objects for dense detector-image sweeps."""
    beam = BeamSpec(
        energy_kev=args.energy_kev,
        theta_deg=args.theta_deg,
        phi_deg=0.0,
        angular_divergence_mrad=0.0,
        energy_spread_ev=0.0,
    )
    surface = SurfaceCTRParams(
        hmax=hmax,
        kmax=kmax,
        surface_roughness=args.surface_roughness,
    )
    detector = DetectorGeometry(
        distance=1000.0,
        image_shape_px=(image_size, image_size),
        pixel_size_mm=(6.0, 12.0),
        beam_center_px=(image_size / 2.0, max(1.0, image_size * 0.08)),
        psf_sigma_pixels=0.0,
    )
    render = RenderParams(
        spot_sigma_px=args.spot_sigma_px,
        n_angular_samples=1,
        n_energy_samples=1,
        render_ctrs_as_streaks=False,
    )
    return beam, surface, detector, render


def _angle_rows(
    phi_values: Float[Array, "N"],
    intensities: Float[Array, "N"],
    maxima: Float[Array, "N"],
) -> list[dict[str, float]]:
    """Build JSON-safe per-angle diagnostic rows."""
    return [
        {
            "phi_deg": float(phi),
            "integrated_intensity": float(integrated),
            "max_intensity": float(maximum),
        }
        for phi, integrated, maximum in zip(
            np.asarray(phi_values),
            np.asarray(intensities),
            np.asarray(maxima),
            strict=True,
        )
    ]


@experiment(
    name="azimuthal-sweep",
    params=[
        Param(
            "crystal",
            str,
            default="",
            help="Path to a CIF, XYZ, or POSCAR structure file.",
            example="tests/test_data/SrTiO3.cif",
        ),
        Param(
            "phi_start_deg",
            float,
            default=-20.0,
            help="First azimuth angle.",
            unit="deg",
        ),
        Param(
            "phi_stop_deg",
            float,
            default=20.0,
            help="Last azimuth angle.",
            unit="deg",
        ),
        Param(
            "n_angles",
            int,
            default=7,
            help="Number of azimuth samples.",
            bounds=(3.0, 181.0),
        ),
        Param(
            "energy_kev", float, default=20.0, help="Beam energy.", unit="keV"
        ),
        Param(
            "theta_deg",
            float,
            default=2.0,
            help="Grazing incidence angle.",
            unit="deg",
        ),
        Param("hmax", int, default=1, help="Maximum absolute h index."),
        Param("kmax", int, default=1, help="Maximum absolute k index."),
        Param(
            "surface_roughness",
            float,
            default=0.0,
            help="CTR roughness parameter for the dense-image simulator.",
        ),
        Param("image_size", int, default=48, help="Square detector size."),
        Param("spot_sigma_px", float, default=1.2, help="Spot width."),
    ],
    returns={
        "metrics": {
            "n_angles": {"type": "integer"},
            "mean_integrated_intensity": {"type": "number"},
            "max_integrated_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": ["sweep_summary", "sweep_arrays", "sweep_preview"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the azimuthal sweep and write diagnostics."""
    hmax: int = min(args.hmax, 1) if args.smoke else args.hmax
    kmax: int = min(args.kmax, 1) if args.smoke else args.kmax
    image_size: int = (
        min(args.image_size, 32) if args.smoke else args.image_size
    )
    n_angles: int = max(
        3, min(args.n_angles, 5) if args.smoke else args.n_angles
    )

    crystal: CrystalStructure = _load_crystal(args.crystal, smoke=args.smoke)
    beam, surface, detector, render = _carriers(
        args,
        hmax=hmax,
        kmax=kmax,
        image_size=image_size,
    )
    phi_values: Float[Array, "N"] = jnp.linspace(
        args.phi_start_deg,
        args.phi_stop_deg,
        n_angles,
    )

    def _run_phi_sweep(bank: Float[Array, "N"]) -> Float[Array, "N H W"]:
        """Run one distributed phi sweep for a bank of azimuths."""
        images: Float[Array, "N H W"] = rh.simul.simulate_detector_image_sweep(
            crystal=crystal,
            axis=("phi_deg", bank),
            beam=beam,
            surface=surface,
            detector=detector,
            render=render,
        )
        return images

    image_bank: Float[Array, "N H W"] = rh.tools.distribute_batched(
        _run_phi_sweep,
        phi_values,
    )
    intensities: Float[Array, "N"] = jnp.sum(image_bank, axis=(1, 2))
    maxima: Float[Array, "N"] = jnp.max(image_bank, axis=(1, 2))
    mean_image: Float[Array, "H W"] = jnp.mean(image_bank, axis=0)
    montage = np.concatenate(
        [np.asarray(image) for image in image_bank], axis=1
    )
    rows: list[dict[str, float]] = _angle_rows(
        phi_values,
        intensities,
        maxima,
    )

    summary_artifact = ctx.save_json(
        "azimuthal_sweep.json",
        {"rows": rows},
        role="sweep_summary",
    )
    array_artifact = ctx.save_array(
        "azimuthal_sweep.npz",
        {
            "phi_deg": np.asarray(phi_values),
            "images": np.asarray(image_bank),
            "integrated_intensity": np.asarray(intensities),
            "max_intensity": np.asarray(maxima),
        },
        role="sweep_arrays",
    )
    mean_artifact = ctx.save_image(
        "azimuthal_mean.png",
        mean_image,
        cmap="phosphor",
        role="sweep_preview",
    )
    montage_artifact = ctx.save_image(
        "azimuthal_montage.png",
        montage,
        cmap="phosphor",
        role="sweep_montage",
    )
    metrics: dict[str, Any] = {
        "n_angles": n_angles,
        "image_shape": [image_size, image_size],
        "mean_integrated_intensity": float(jnp.mean(intensities)),
        "max_integrated_intensity": float(jnp.max(intensities)),
        "peak_phi_deg": float(phi_values[jnp.argmax(intensities)]),
    }
    return {
        "metrics": metrics,
        "artifacts": [
            summary_artifact,
            array_artifact,
            mean_artifact,
            montage_artifact,
        ],
        "sweep": rows,
    }


if __name__ == "__main__":
    main()
