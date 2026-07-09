# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Simulate one reflection multislice RHEED detector image.

The automaton loads a CIF, XYZ, or POSCAR surface structure, runs rheedium's
edge-on reflection multislice forward model, rasterizes the sparse reflected
beam pattern to a dense detector image, and writes log and linear PNGs plus an
``.npz`` artifact containing raw intensity. In ``--smoke`` mode it uses a tiny
two-atom slab generated in code so the backend contract is testable without
external fixtures.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import CrystalStructure, DetectorGeometry, RHEEDPattern


@jaxtyped(typechecker=beartype)
def _smoke_crystal() -> CrystalStructure:
    """Return a tiny orthogonal surface slab for smoke testing."""
    frac: Float[Array, "2 4"] = jnp.asarray(
        [
            [0.25, 0.25, 0.25, 14.0],
            [0.75, 0.75, 0.50, 14.0],
        ]
    )
    cart: Float[Array, "2 4"] = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 14.0],
            [3.0, 3.0, 2.0, 14.0],
        ]
    )
    return rh.types.create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=jnp.asarray([4.0, 4.0, 4.0]),
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


def _reflection_metrics(pattern: RHEEDPattern) -> dict[str, Any]:
    """Summarize reflected beams with a specular proxy."""
    intensities: np.ndarray[Any, Any] = np.asarray(pattern.intensities)
    detector_points: np.ndarray[Any, Any] = np.asarray(pattern.detector_points)
    if not intensities.size:
        return {
            "n_reflected_beams": 0,
            "reflectivity": 0.0,
            "strongest_detector_point": [0.0, 0.0],
        }
    strongest_idx: int = int(np.argmax(intensities))
    total: float = float(intensities.sum())
    strongest: float = float(intensities[strongest_idx])
    return {
        "n_reflected_beams": int(intensities.shape[0]),
        "reflectivity": strongest / total if total > 0.0 else 0.0,
        "strongest_detector_point": [
            float(detector_points[strongest_idx, 0]),
            float(detector_points[strongest_idx, 1]),
        ],
        "max_beam_intensity": strongest,
        "total_beam_intensity": total,
    }


def _zone_axis(args: Any) -> tuple[int, int, int]:
    """Return the requested surface zone axis as Miller indices."""
    zone: tuple[int, int, int] = (
        int(args.zone_h),
        int(args.zone_k),
        int(args.zone_l),
    )
    if zone == (0, 0, 0):
        raise ValueError("zone axis cannot be [0, 0, 0]")
    return zone


def _load_crystal(args: Any, *, smoke: bool) -> CrystalStructure:
    """Load a user crystal (or smoke fixture) reoriented to the zone axis."""
    if smoke and not args.crystal:
        crystal: CrystalStructure = _smoke_crystal()
    elif not args.crystal:
        raise ValueError("crystal is required unless --smoke is set")
    else:
        crystal = rh.inout.parse_crystal(args.crystal)
    return rh.ucell.reorient_to_zone_axis(
        crystal, jnp.asarray(_zone_axis(args), dtype=jnp.int32)
    )


@experiment(
    name="forward-reflection",
    params=[
        Param(
            "crystal",
            str,
            default="",
            help="Path to a CIF, XYZ, or POSCAR surface structure file.",
            example="tests/test_data/bi2se3/intial.xyz",
        ),
        Param(
            "energy_kev",
            float,
            default=30.0,
            help="Incident electron beam energy.",
            unit="keV",
            bounds=(5.0, 50.0),
            example=30.0,
        ),
        Param(
            "theta_deg",
            float,
            default=2.5,
            help="Grazing incidence angle from the surface.",
            unit="deg",
            bounds=(0.1, 10.0),
            example=2.5,
        ),
        Param(
            "phi_deg",
            float,
            default=0.0,
            help="In-plane azimuth angle; reflection backend supports 0.",
            unit="deg",
            bounds=(0.0, 0.0),
            example=0.0,
        ),
        Param(
            "zone_h",
            int,
            default=0,
            help="Surface zone-axis Miller h index.",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "zone_k",
            int,
            default=0,
            help="Surface zone-axis Miller k index.",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "zone_l",
            int,
            default=1,
            help="Surface zone-axis Miller l index.",
            bounds=(-8.0, 8.0),
            example=1,
        ),
        Param(
            "detector_distance_mm",
            float,
            default=80.0,
            help="Sample-to-detector distance.",
            unit="mm",
            bounds=(1.0, 5000.0),
            example=80.0,
        ),
        Param(
            "dx_slice_ang",
            float,
            default=1.0,
            help="Edge-on slice spacing along the beam direction.",
            unit="Angstrom",
            bounds=(0.1, 20.0),
            example=1.0,
        ),
        Param(
            "dy_ang",
            float,
            default=0.25,
            help="Lateral edge-on grid spacing.",
            unit="Angstrom",
            bounds=(0.1, 10.0),
            example=0.25,
        ),
        Param(
            "dz_ang",
            float,
            default=0.25,
            help="Vertical edge-on grid spacing.",
            unit="Angstrom",
            bounds=(0.1, 10.0),
            example=0.25,
        ),
        Param(
            "vacuum_above_ang",
            float,
            default=30.0,
            help="Vacuum thickness above the surface read-off band.",
            unit="Angstrom",
            bounds=(1.0, 200.0),
            example=30.0,
        ),
        Param(
            "cap_width_ang",
            float,
            default=15.0,
            help="Absorbing cap width.",
            unit="Angstrom",
            bounds=(0.1, 100.0),
            example=15.0,
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
        Param(
            "parameterization",
            str,
            default="lobato",
            help="Atomic potential parameterization.",
            choices=("lobato", "kirkland"),
            example="lobato",
        ),
    ],
    returns={
        "metrics": {
            "n_reflected_beams": {"type": "integer"},
            "reflectivity": {"type": "number"},
            "max_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "detector_image",
                "detector_image_linear",
                "detector_array",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the edge-on reflection multislice forward simulation."""
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )
    dy_ang: float = max(args.dy_ang, 0.5) if args.smoke else args.dy_ang
    dz_ang: float = max(args.dz_ang, 0.5) if args.smoke else args.dz_ang
    dx_slice_ang: float = (
        max(args.dx_slice_ang, 1.0) if args.smoke else args.dx_slice_ang
    )
    vacuum_above_ang: float = (
        min(args.vacuum_above_ang, 4.0)
        if args.smoke
        else args.vacuum_above_ang
    )
    cap_width_ang: float = (
        min(args.cap_width_ang, 2.0) if args.smoke else args.cap_width_ang
    )

    crystal: CrystalStructure = _load_crystal(args, smoke=args.smoke)
    pattern: RHEEDPattern = rh.simul.reflection_multislice_simulator(
        crystal,
        energy_kev=args.energy_kev,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        detector_distance=args.detector_distance_mm,
        dx_slice=dx_slice_ang,
        dy=dy_ang,
        dz=dz_ang,
        vacuum_above=vacuum_above_ang,
        cap_width=cap_width_ang,
        parameterization=args.parameterization,
    )
    geometry = DetectorGeometry(
        distance=args.detector_distance_mm,
        image_shape_px=(image_size, image_size),
        pixel_size_mm=(1.5, 3.0),
        beam_center_px=(image_size / 2.0, max(1.0, image_size * 0.08)),
        psf_sigma_pixels=0.0,
    )
    image = rh.simul.render_ctr_streaks_to_image(
        pattern,
        geometry,
        spot_sigma_px=args.spot_sigma_px,
    )

    png_artifacts = ctx.save_image_scales(
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
            "k_out": np.asarray(pattern.k_out),
        },
        role="detector_array",
    )

    metrics: dict[str, Any] = {
        "image_shape": [image_size, image_size],
        "zone_axis": list(_zone_axis(args)),
        **_reflection_metrics(pattern),
        **_image_metrics(image),
    }
    ctx.save_json("metrics.json", metrics, role="metrics")
    return {
        "metrics": metrics,
        "artifacts": [*png_artifacts, npz_artifact],
    }


if __name__ == "__main__":
    main()
