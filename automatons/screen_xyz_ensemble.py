# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Rank a structure-file ensemble by simulated RHEED similarity.

The automaton accepts a directory, glob, or single candidate structure file,
simulates each candidate with the kinematic forward model, compares the images
against a measured array or simulated target crystal, and emits a ranked JSON
table plus best-match image artifacts. In ``--smoke`` mode it writes two tiny
extended-XYZ candidates under ``--outdir`` and ranks them against one of them.
"""

from __future__ import annotations

import glob
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Float, jaxtyped
from numpy.typing import NDArray

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import DetectorGeometry

_STRUCTURE_SUFFIXES: tuple[str, ...] = (".xyz", ".cif", ".vasp")


@jaxtyped(typechecker=beartype)
def _normalise_image(
    image: Float[NDArray, "height width"],
) -> Float[NDArray, "height width"]:
    """Return a finite float image scaled to unit maximum."""
    image_np: Float[NDArray, "height width"] = np.asarray(
        image,
        dtype=np.float64,
    )
    image_np = np.nan_to_num(image_np)
    max_value: float = float(np.max(np.abs(image_np)))
    if max_value <= 1e-12:
        return image_np
    return image_np / max_value


@jaxtyped(typechecker=beartype)
def _score_images(
    target: Float[NDArray, "height width"],
    candidate: Float[NDArray, "height width"],
) -> dict[str, float]:
    """Compute simple image similarity metrics for two detector images."""
    target_norm: Float[NDArray, "height width"] = _normalise_image(target)
    candidate_norm: Float[NDArray, "height width"] = _normalise_image(
        candidate
    )
    if target_norm.shape != candidate_norm.shape:
        raise ValueError(
            "target and candidate images must have the same shape; "
            f"got {target_norm.shape} and {candidate_norm.shape}"
        )
    residual: Float[NDArray, "height width"] = candidate_norm - target_norm
    target_centered: Float[NDArray, "height width"] = (
        target_norm - target_norm.mean()
    )
    candidate_centered: Float[NDArray, "height width"] = (
        candidate_norm - candidate_norm.mean()
    )
    denom: float = float(
        np.linalg.norm(target_centered) * np.linalg.norm(candidate_centered)
    )
    ncc: float = (
        0.0
        if denom <= 1e-12
        else float(np.vdot(target_centered, candidate_centered) / denom)
    )
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "ncc": ncc,
    }


def _metric_score(metrics: dict[str, float], metric: str) -> float:
    """Return a sortable score where larger means better."""
    if metric == "ncc":
        return metrics["ncc"]
    if metric == "mse":
        return -metrics["mse"]
    if metric == "mae":
        return -metrics["mae"]
    raise ValueError(f"unsupported metric: {metric}")


def _load_image(path: str) -> Float[NDArray, "height width"]:
    """Load a 2D detector image from ``.npy`` or ``.npz``."""
    image_path = Path(path)
    if image_path.suffix == ".npy":
        image = np.load(image_path)
    elif image_path.suffix == ".npz":
        with np.load(image_path) as data:
            key: str = "image" if "image" in data.files else data.files[0]
            image = data[key]
    else:
        raise ValueError("target_array must be a .npy or .npz file")
    image_np: Float[NDArray, "height width"] = np.asarray(
        image,
        dtype=np.float64,
    )
    if image_np.ndim != 2:
        raise ValueError(f"image must be 2D; got shape {image_np.shape}")
    return image_np


def _resolve_structure_paths(spec: str) -> list[Path]:
    """Resolve a file, directory, or glob into sorted candidate paths."""
    if not spec:
        raise ValueError("structures is required unless --smoke is set")
    root = Path(spec)
    if root.is_dir():
        paths: list[Path] = []
        for suffix in _STRUCTURE_SUFFIXES:
            paths.extend(root.glob(f"*{suffix}"))
        for special_name in ("POSCAR", "CONTCAR"):
            paths.extend(root.glob(special_name))
        return sorted({path.resolve() for path in paths})
    if any(char in spec for char in "*?[]"):
        return sorted(Path(item).resolve() for item in glob.glob(spec))
    return [root.resolve()]


def _write_smoke_xyz(path: Path, *, lattice_a: float, shift: float) -> None:
    """Write a two-atom extended XYZ file for smoke screening."""
    half: float = lattice_a / 2.0 + shift
    path.write_text(
        "\n".join(
            [
                "2",
                (
                    'Lattice="'
                    f"{lattice_a} 0 0 0 {lattice_a} 0 0 0 {lattice_a}"
                    '" Properties=species:S:1:pos:R:3'
                ),
                "Mg 0.0 0.0 0.0",
                f"O {half:.6f} {half:.6f} {half:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _smoke_files(ctx: Any) -> tuple[str, str]:
    """Create smoke candidate files and return structures spec plus target."""
    smoke_dir = ctx.outdir / "smoke_candidates"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    target_path = smoke_dir / "mgo_target.xyz"
    shifted_path = smoke_dir / "mgo_shifted.xyz"
    _write_smoke_xyz(target_path, lattice_a=4.21, shift=0.0)
    _write_smoke_xyz(shifted_path, lattice_a=4.21, shift=0.25)
    return str(smoke_dir), str(target_path)


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


def _simulate_image(
    structure_path: Path,
    args: Any,
    *,
    hmax: int,
    kmax: int,
    image_size: int,
) -> tuple[Float[NDArray, "height width"], int]:
    """Simulate one structure and return its detector image and hit count."""
    crystal = rh.inout.parse_crystal(structure_path)
    crystal = rh.ucell.reorient_to_zone_axis(
        crystal, jnp.asarray(_zone_axis(args), dtype=jnp.int32)
    )
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
    image = rh.simul.render_ctr_streaks_to_image(
        pattern,
        geometry,
        spot_sigma_px=args.spot_sigma_px,
    )
    return np.asarray(image, dtype=np.float64), int(
        pattern.intensities.shape[0]
    )


@experiment(
    name="screen-xyz-ensemble",
    params=[
        Param(
            "structures",
            str,
            default="",
            help="Candidate structure file, directory, or glob.",
            example="matensemble/*.xyz",
        ),
        Param(
            "target_array",
            str,
            default="",
            help="Measured/target detector image as .npy or .npz.",
            example="measured_frame.npz",
        ),
        Param(
            "target_crystal",
            str,
            default="",
            help=(
                "Structure file to simulate as the target if no array is set."
            ),
            example="target.xyz",
        ),
        Param(
            "metric",
            str,
            default="ncc",
            help="Ranking metric; ncc is higher-better, mse/mae lower-better.",
            choices=("ncc", "mse", "mae"),
            example="ncc",
        ),
        Param("top_k", int, default=5, help="Number of ranked rows to emit."),
        Param(
            "energy_kev",
            float,
            default=20.0,
            help="Incident electron beam energy.",
            unit="keV",
            bounds=(5.0, 50.0),
        ),
        Param(
            "theta_deg",
            float,
            default=2.0,
            help="Grazing incidence angle from the surface.",
            unit="deg",
            bounds=(0.1, 10.0),
        ),
        Param(
            "phi_deg",
            float,
            default=0.0,
            help="In-plane azimuth angle.",
            unit="deg",
            bounds=(-180.0, 180.0),
        ),
        Param(
            "zone_h",
            int,
            default=0,
            help="Surface zone-axis Miller h index.",
            bounds=(-8.0, 8.0),
        ),
        Param(
            "zone_k",
            int,
            default=0,
            help="Surface zone-axis Miller k index.",
            bounds=(-8.0, 8.0),
        ),
        Param(
            "zone_l",
            int,
            default=1,
            help="Surface zone-axis Miller l index.",
            bounds=(-8.0, 8.0),
        ),
        Param("hmax", int, default=3, help="Maximum absolute h index."),
        Param("kmax", int, default=3, help="Maximum absolute k index."),
        Param(
            "image_size",
            int,
            default=96,
            help="Square detector image size.",
            unit="px",
            bounds=(16.0, 512.0),
        ),
        Param(
            "spot_sigma_px",
            float,
            default=1.4,
            help="Gaussian detector spot width.",
            unit="px",
            bounds=(0.2, 10.0),
        ),
    ],
    returns={
        "metrics": {
            "n_candidates": {"type": "integer"},
            "best_score": {"type": "number"},
            "best_candidate": {"type": "string"},
        },
        "artifacts": {
            "roles": [
                "ranking",
                "score_table",
                "best_match_image",
                "best_match_image_linear",
                "residual_image",
                "residual_image_linear",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Simulate and rank a structure-file ensemble against a target image."""
    structures: str = args.structures
    target_crystal: str = args.target_crystal
    if args.smoke and not structures:
        structures, target_crystal = _smoke_files(ctx)

    hmax: int = min(args.hmax, 1) if args.smoke else args.hmax
    kmax: int = min(args.kmax, 1) if args.smoke else args.kmax
    image_size: int = (
        min(args.image_size, 48) if args.smoke else args.image_size
    )
    candidate_paths: list[Path] = _resolve_structure_paths(structures)
    if not candidate_paths:
        raise ValueError("no candidate structures matched")

    if args.target_array:
        target_image = _load_image(args.target_array)
    elif target_crystal:
        target_image, _target_hits = _simulate_image(
            Path(target_crystal),
            args,
            hmax=hmax,
            kmax=kmax,
            image_size=image_size,
        )
    else:
        raise ValueError("target_array or target_crystal is required")

    rows: list[dict[str, Any]] = []
    images: dict[str, Float[NDArray, "height width"]] = {}
    for path in candidate_paths:
        image, n_reflections = _simulate_image(
            path,
            args,
            hmax=hmax,
            kmax=kmax,
            image_size=image_size,
        )
        metrics = _score_images(target_image, image)
        score: float = _metric_score(metrics, args.metric)
        images[path.as_posix()] = image
        rows.append(
            {
                "candidate": path.as_posix(),
                "score": score,
                "rank_metric": args.metric,
                "n_reflections": n_reflections,
                **metrics,
            }
        )

    ranked: list[dict[str, Any]] = sorted(
        rows,
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    best = ranked[0]
    best_image = images[str(best["candidate"])]
    residual = np.abs(
        _normalise_image(best_image) - _normalise_image(target_image)
    )

    ranking_artifact = ctx.save_json(
        "ranked_candidates.json",
        {"ranked": ranked},
        role="ranking",
    )
    score_artifact = ctx.save_array(
        "scores.npz",
        {
            "score": np.asarray([row["score"] for row in ranked]),
            "mse": np.asarray([row["mse"] for row in ranked]),
            "mae": np.asarray([row["mae"] for row in ranked]),
            "ncc": np.asarray([row["ncc"] for row in ranked]),
            "candidate": np.asarray([row["candidate"] for row in ranked]),
        },
        role="score_table",
    )
    best_artifacts = ctx.save_image_scales(
        "best_match.png",
        best_image,
        cmap="phosphor",
        role="best_match_image",
    )
    residual_artifacts = ctx.save_image_scales(
        "best_residual.png",
        residual,
        cmap="phosphor",
        role="residual_image",
    )

    top_k: int = max(1, min(args.top_k, len(ranked)))
    return {
        "metrics": {
            "n_candidates": len(ranked),
            "zone_axis": list(_zone_axis(args)),
            "best_score": float(best["score"]),
            "best_candidate": str(best["candidate"]),
            "best_mse": float(best["mse"]),
            "best_ncc": float(best["ncc"]),
        },
        "artifacts": [
            ranking_artifact,
            score_artifact,
            *best_artifacts,
            *residual_artifacts,
        ],
        "ranked": ranked[:top_k],
    }


if __name__ == "__main__":
    main()
