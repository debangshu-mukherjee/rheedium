# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.8"]
# ///
"""Simulate one transmission multislice RHEED detector image.

The automaton loads a CIF, XYZ, or POSCAR bulk structure, converts it to a
surface slab and projected potential slices, runs rheedium's transmission
multislice forward model, rasterizes the sparse diffraction pattern to a dense
detector image, and writes both PNG and ``.npz`` artifacts. In ``--smoke`` mode
it uses a tiny slab generated in code so the backend contract is testable
without external fixtures.
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
    DetectorGeometry,
    PotentialSlices,
    RHEEDPattern,
    SlicedCrystal,
)


@jaxtyped(typechecker=beartype)
def _smoke_sliced_crystal() -> SlicedCrystal:
    """Return a tiny orthogonal slab for smoke testing."""
    return rh.types.create_sliced_crystal(
        cart_positions=jnp.asarray(
            [
                [1.0, 1.0, 1.0, 14.0],
                [3.0, 3.0, 2.0, 14.0],
            ]
        ),
        cell_lengths=jnp.asarray([4.0, 4.0, 4.0]),
        cell_angles=jnp.asarray([90.0, 90.0, 90.0]),
        orientation=jnp.asarray([0, 0, 1]),
        depth=4.0,
        x_extent=4.0,
        y_extent=4.0,
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


def _orientation(args: Any) -> tuple[int, int, int]:
    """Return the requested surface orientation as Miller indices."""
    orientation = (
        int(args.orientation_h),
        int(args.orientation_k),
        int(args.orientation_l),
    )
    if orientation == (0, 0, 0):
        raise ValueError("surface orientation cannot be [0, 0, 0]")
    return orientation


def _sliced_crystal(args: Any, *, smoke: bool) -> SlicedCrystal:
    """Load a user crystal slab or return the smoke fixture."""
    if smoke and not args.crystal:
        return _smoke_sliced_crystal()
    if not args.crystal:
        raise ValueError("crystal is required unless --smoke is set")
    crystal = rh.inout.parse_crystal(args.crystal)
    return rh.ucell.bulk_to_slice(
        bulk_crystal=crystal,
        orientation=jnp.asarray(_orientation(args), dtype=jnp.int32),
        depth=args.depth_ang,
        x_extent=args.x_extent_ang,
        y_extent=args.y_extent_ang,
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
            "orientation_h",
            int,
            default=0,
            help="Surface-orientation Miller h index.",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "orientation_k",
            int,
            default=0,
            help="Surface-orientation Miller k index.",
            bounds=(-8.0, 8.0),
            example=0,
        ),
        Param(
            "orientation_l",
            int,
            default=1,
            help="Surface-orientation Miller l index.",
            bounds=(-8.0, 8.0),
            example=1,
        ),
        Param(
            "depth_ang",
            float,
            default=20.0,
            help="Surface slab depth used for bulk-to-slab lowering.",
            unit="Angstrom",
            bounds=(1.0, 200.0),
            example=20.0,
        ),
        Param(
            "x_extent_ang",
            float,
            default=40.0,
            help="Surface slab x extent used for bulk-to-slab lowering.",
            unit="Angstrom",
            bounds=(1.0, 500.0),
            example=40.0,
        ),
        Param(
            "y_extent_ang",
            float,
            default=40.0,
            help="Surface slab y extent used for bulk-to-slab lowering.",
            unit="Angstrom",
            bounds=(1.0, 500.0),
            example=40.0,
        ),
        Param(
            "slice_thickness_ang",
            float,
            default=2.0,
            help="Projected-potential slice thickness.",
            unit="Angstrom",
            bounds=(0.1, 20.0),
            example=2.0,
        ),
        Param(
            "pixel_size_ang",
            float,
            default=0.5,
            help="Projected-potential lateral pixel size.",
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
            help="Mean inner potential used in transmission propagation.",
            unit="V",
            bounds=(-100.0, 100.0),
            example=0.0,
        ),
        Param(
            "bandwidth_limit",
            float,
            default=2.0 / 3.0,
            help="Fraction of Nyquist frequency retained during propagation.",
            bounds=(0.05, 1.0),
            example=2.0 / 3.0,
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
        Param(
            "cmap",
            str,
            default="phosphor",
            help="Matplotlib colormap for the PNG artifact.",
            choices=("phosphor", "viridis", "magma", "gray"),
            example="phosphor",
        ),
    ],
    returns={
        "metrics": {
            "n_beams": {"type": "integer"},
            "max_intensity": {"type": "number"},
            "integrated_intensity": {"type": "number"},
        },
        "artifacts": {
            "roles": ["detector_image", "detector_array"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Run the transmission multislice forward simulation."""
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

    sliced: SlicedCrystal = _sliced_crystal(args, smoke=args.smoke)
    potential: PotentialSlices = (
        rh.simul.sliced_crystal_to_projected_potential_slices(
            sliced,
            slice_thickness=slice_thickness_ang,
            pixel_size=pixel_size_ang,
            parameterization=args.parameterization,
        )
    )
    pattern: RHEEDPattern = rh.simul.multislice_simulator(
        potential,
        energy_kev=args.energy_kev,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        detector_distance=args.detector_distance_mm,
        inner_potential_v0=args.inner_potential_v0,
        bandwidth_limit=args.bandwidth_limit,
    )
    geometry = DetectorGeometry(
        distance=args.detector_distance_mm,
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
        cmap=args.cmap,
        role="detector_image",
    )
    npz_artifact = ctx.save_array(
        "pattern.npz",
        {
            "image": np.asarray(image),
            "detector_points": np.asarray(pattern.detector_points),
            "intensities": np.asarray(pattern.intensities),
            "potential_slices": np.asarray(potential.slices),
        },
        role="detector_array",
    )

    metrics: dict[str, Any] = {
        "image_shape": [image_size, image_size],
        "potential_shape": list(potential.slices.shape),
        **_pattern_metrics(pattern),
        **_image_metrics(image),
    }
    ctx.save_json("metrics.json", metrics, role="metrics")
    return {
        "metrics": metrics,
        "artifacts": [png_artifact, npz_artifact],
    }


if __name__ == "__main__":
    main()
