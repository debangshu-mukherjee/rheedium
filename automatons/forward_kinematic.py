# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Simulate one kinematic RHEED detector image from a structure file.

The automaton loads a CIF, XYZ, or POSCAR structure, runs rheedium's
kinematic Ewald forward model, rasterizes the sparse diffraction pattern to a
dense detector image, and writes both a PNG preview and an ``.npz`` numeric
artifact. In ``--smoke`` mode it generates a tiny MgO-like cubic cell in code
so the contract can be tested without external fixtures.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import CrystalStructure, DetectorGeometry


@jaxtyped(typechecker=beartype)
def _smoke_crystal() -> CrystalStructure:
    """Return a tiny MgO-like cubic crystal for smoke testing."""
    lattice_a: float = 4.21
    frac: Float[Array, "2 4"] = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 12.0],
            [0.5, 0.5, 0.5, 8.0],
        ]
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


@jaxtyped(typechecker=beartype)
def _image_metrics(image: Float[Array, "height width"]) -> dict[str, float]:
    """Summarize a detector image without crossing arrays into JSON."""
    image_np: np.ndarray[Any, Any] = np.asarray(image)
    return {
        "max_intensity": float(image_np.max()),
        "mean_intensity": float(image_np.mean()),
        "integrated_intensity": float(image_np.sum()),
    }


def _load_crystal(path: str, *, smoke: bool) -> CrystalStructure:
    """Load a user crystal or generate the smoke fixture."""
    if smoke and not path:
        return _smoke_crystal()
    if not path:
        raise ValueError("crystal is required unless --smoke is set")
    return rh.inout.parse_crystal(path)


@experiment(
    name="forward-kinematic",
    params=[
        Param(
            "crystal",
            str,
            default="",
            help="Path to a CIF, XYZ, or POSCAR structure file.",
            example="tests/test_data/SrTiO3.cif",
        ),
        Param(
            "energy_kev",
            float,
            default=20.0,
            help="Incident electron beam energy.",
            unit="keV",
            bounds=(5.0, 50.0),
            example=20.0,
        ),
        Param(
            "theta_deg",
            float,
            default=2.0,
            help="Grazing incidence angle from the surface.",
            unit="deg",
            bounds=(0.1, 10.0),
            example=2.0,
        ),
        Param(
            "phi_deg",
            float,
            default=0.0,
            help="In-plane azimuth angle.",
            unit="deg",
            bounds=(-180.0, 180.0),
            example=0.0,
        ),
        Param(
            "hmax",
            int,
            default=3,
            help="Maximum absolute h Miller index.",
            bounds=(1.0, 10.0),
            example=3,
        ),
        Param(
            "kmax",
            int,
            default=3,
            help="Maximum absolute k Miller index.",
            bounds=(1.0, 10.0),
            example=3,
        ),
        Param(
            "image_size",
            int,
            default=96,
            help="Square detector image size.",
            unit="px",
            bounds=(16.0, 512.0),
            example=96,
        ),
        Param(
            "spot_sigma_px",
            float,
            default=1.4,
            help="Gaussian detector spot width.",
            unit="px",
            bounds=(0.2, 10.0),
            example=1.4,
        ),
    ],
    returns={
        "metrics": {
            "n_reflections": {"type": "integer"},
            "max_intensity": {"type": "number"},
            "integrated_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": ["detector_image", "detector_array"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the kinematic forward simulation and write detector artifacts."""
    hmax: int = min(args.hmax, 1) if args.smoke else args.hmax
    kmax: int = min(args.kmax, 1) if args.smoke else args.kmax
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )

    crystal: CrystalStructure = _load_crystal(args.crystal, smoke=args.smoke)
    pattern = rh.simul.ewald_simulator(
        crystal,
        energy_kev=args.energy_kev,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        hmax=hmax,
        kmax=kmax,
    )
    geometry = DetectorGeometry(
        distance=1000.0,
        image_shape_px=(image_size, image_size),
        pixel_size_mm=(1.5, 3.0),
        beam_center_px=(image_size / 2.0, max(1.0, image_size * 0.08)),
        psf_sigma_pixels=0.0,
    )
    image = rh.simul.render_pattern_to_image(
        pattern,
        geometry,
        spot_sigma_px=args.spot_sigma_px,
    )

    png_artifact = ctx.save_image(
        "pattern.png",
        image,
        cmap="phosphor",
        role="detector_image",
    )
    npz_artifact = ctx.save_array(
        "pattern.npz",
        {
            "image": np.asarray(image),
            "detector_points": np.asarray(pattern.detector_points),
            "intensities": np.asarray(pattern.intensities),
        },
        role="detector_array",
    )

    metrics: dict[str, Any] = {
        "n_reflections": int(pattern.intensities.shape[0]),
        "image_shape": [image_size, image_size],
        **_image_metrics(image),
    }
    ctx.save_json("metrics.json", metrics, role="metrics")
    return {
        "metrics": metrics,
        "artifacts": [png_artifact, npz_artifact],
    }


if __name__ == "__main__":
    main()
