# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Simulate one reflection multislice RHEED detector image.

The automaton loads a CIF, XYZ, or POSCAR bulk structure, re-expresses it in
the requested surface cell, runs rheedium's edge-on reflection multislice
forward model (beam along the surface, absorbing caps, genuine propagation
over ``propagation_length_ang``), rasterizes the sparse reflected pattern to
a dense detector image, and writes log and linear PNGs plus an ``.npz``
artifact containing raw intensity. In ``--smoke`` mode it uses a tiny crystal
generated in code so the backend contract is testable without external
fixtures.
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
    CrystalStructure,
    DetectorGeometry,
    EdgeOnSlices,
    RHEEDPattern,
)


@jaxtyped(typechecker=beartype)
def _smoke_crystal() -> CrystalStructure:
    """Return a tiny frame-consistent crystal for smoke testing."""
    frac = jnp.asarray(
        [
            [0.25, 0.25, 0.25, 14.0],
            [0.75, 0.75, 0.5, 14.0],
        ]
    )
    cell_lengths = jnp.asarray([4.0, 4.0, 4.0])
    cell_angles = jnp.asarray([90.0, 90.0, 90.0])
    cell = rh.ucell.build_cell_vectors(*cell_lengths, *cell_angles)
    cart = jnp.concatenate([frac[:, :3] @ cell, frac[:, 3:]], axis=1)
    return rh.types.create_crystal_structure(
        frac_positions=frac,
        cart_positions=cart,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
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


def _pattern_metrics(pattern: RHEEDPattern) -> dict[str, float | int]:
    """Summarize the sparse multislice pattern."""
    intensities: np.ndarray[Any, Any] = np.asarray(pattern.intensities)
    return {
        "n_beams": int(intensities.shape[0]),
        "max_beam_intensity": (
            float(intensities.max()) if intensities.size else 0.0
        ),
        "total_beam_intensity": float(intensities.sum()),
    }


def _zone_axis(args: Any) -> tuple[int, int, int]:
    """Return the requested surface orientation as Miller indices."""
    orientation = (
        int(args.zone_h),
        int(args.zone_k),
        int(args.zone_l),
    )
    if orientation == (0, 0, 0):
        raise ValueError("zone axis cannot be [0, 0, 0]")
    return orientation


def _surface_crystal(args: Any, *, smoke: bool) -> CrystalStructure:
    """Load a surface-oriented crystal or return the smoke fixture."""
    if smoke and not args.crystal:
        return _smoke_crystal()
    if not args.crystal:
        raise ValueError("crystal is required unless --smoke is set")
    crystal = rh.inout.parse_crystal(args.crystal)
    return rh.ucell.reorient_to_zone_axis(
        crystal, jnp.asarray(_zone_axis(args), dtype=jnp.int32)
    )


@experiment(
    name="forward-multislice",
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
            "zone_h",
            int,
            default=0,
            help="Surface zone-axis Miller h index (always applied).",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "zone_k",
            int,
            default=0,
            help="Surface zone-axis Miller k index (always applied).",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "zone_l",
            int,
            default=1,
            help="Surface zone-axis Miller l index (always applied).",
            bounds=(-8.0, 8.0),
            example=1,
        ),
        Param(
            "depth_ang",
            float,
            default=20.0,
            help="Penetration depth below the surface in the edge-on slab.",
            unit="Angstrom",
            bounds=(1.0, 200.0),
            example=20.0,
        ),
        Param(
            "x_extent_ang",
            float,
            default=40.0,
            help=(
                "Unused by the reflection kernel; kept for recipe "
                "compatibility."
            ),
            unit="Angstrom",
            bounds=(1.0, 500.0),
            example=40.0,
        ),
        Param(
            "y_extent_ang",
            float,
            default=40.0,
            help=(
                "Unused by the reflection kernel; kept for recipe "
                "compatibility."
            ),
            unit="Angstrom",
            bounds=(1.0, 500.0),
            example=40.0,
        ),
        Param(
            "slice_thickness_ang",
            float,
            default=2.0,
            help="Beam-axis slice step dx of the edge-on propagation.",
            unit="Angstrom",
            bounds=(0.1, 20.0),
            example=2.0,
        ),
        Param(
            "pixel_size_ang",
            float,
            default=0.5,
            help="Transverse grid step (dy = dz) of the edge-on wavefield.",
            unit="Angstrom",
            bounds=(0.1, 10.0),
            example=0.5,
        ),
        Param(
            "detector_distance_mm",
            float,
            default=100.0,
            help="Sample-to-detector distance.",
            unit="mm",
            bounds=(1.0, 5000.0),
            example=100.0,
        ),
        Param(
            "inner_potential_v0",
            float,
            default=0.0,
            help=(
                "Ignored: the reflection kernel derives refraction from "
                "the potential itself; kept for recipe compatibility."
            ),
            unit="V",
            bounds=(-100.0, 100.0),
            example=0.0,
        ),
        Param(
            "bandwidth_limit",
            float,
            default=2.0 / 3.0,
            help=(
                "Ignored by the reflection simulator (internal 2/3 "
                "limit); kept for recipe compatibility."
            ),
            bounds=(0.05, 1.0),
            example=2.0 / 3.0,
        ),
        Param(
            "propagation_length_ang",
            float,
            default=200.0,
            help="Beam-axis propagation length through the crystal.",
            unit="Angstrom",
            bounds=(10.0, 2000.0),
            example=200.0,
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
            "n_beams": {"type": "integer"},
            "max_intensity": {"type": "number"},
            "integrated_intensity": {"type": "number"},
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
    """Run the reflection multislice forward simulation."""
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )
    slice_thickness_ang: float = (
        max(args.slice_thickness_ang, 2.0)
        if args.smoke
        else args.slice_thickness_ang
    )
    pixel_size_ang: float = (
        max(args.pixel_size_ang, 1.0) if args.smoke else args.pixel_size_ang
    )

    vacuum_above_ang: float = 8.0 if args.smoke else 30.0
    cap_width_ang: float = 4.0 if args.smoke else 15.0
    propagation_length_ang: float = (
        min(args.propagation_length_ang, 16.0)
        if args.smoke
        else args.propagation_length_ang
    )
    crystal: CrystalStructure = _surface_crystal(args, smoke=args.smoke)
    edge_on: EdgeOnSlices = rh.simul.crystal_to_edge_on_slices(
        crystal,
        phi_deg=args.phi_deg,
        dx_slice=slice_thickness_ang,
        dy=pixel_size_ang,
        dz=pixel_size_ang,
        vacuum_above=vacuum_above_ang,
        cap_width=cap_width_ang,
        penetration_depth=args.depth_ang,
        parameterization=args.parameterization,
    )
    pattern: RHEEDPattern = rh.simul.reflection_multislice_simulator(
        crystal,
        energy_kev=args.energy_kev,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        detector_distance=args.detector_distance_mm,
        dx_slice=slice_thickness_ang,
        dy=pixel_size_ang,
        dz=pixel_size_ang,
        vacuum_above=vacuum_above_ang,
        cap_width=cap_width_ang,
        propagation_length_ang=propagation_length_ang,
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
            "edge_on_slices": np.asarray(edge_on.slices),
        },
        role="detector_array",
    )

    metrics: dict[str, Any] = {
        "image_shape": [image_size, image_size],
        "edge_on_slices_shape": list(edge_on.slices.shape),
        "zone_axis": list(_zone_axis(args)),
        **_pattern_metrics(pattern),
        **_image_metrics(image),
    }
    ctx.save_json("metrics.json", metrics, role="metrics")
    return {
        "metrics": metrics,
        "artifacts": [*png_artifacts, npz_artifact],
    }


if __name__ == "__main__":
    main()
