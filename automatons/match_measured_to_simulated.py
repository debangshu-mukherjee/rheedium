# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.10"]
# ///
"""Match a measured RHEED image against simulated detector images.

The automaton is the image-scoring bridge for Loop B: it loads one measured
detector image and a set of simulated ``.npy``/``.npz`` images, computes simple
similarity metrics, and emits ranked scores plus best-match residual artifacts.
In ``--smoke`` mode it writes small synthetic image arrays under ``--outdir``
and runs the same file-based matching path.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Float, jaxtyped
from numpy.typing import NDArray

from rheedium.harness import Param, experiment


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
    measured: Float[NDArray, "height width"],
    simulated: Float[NDArray, "height width"],
) -> dict[str, float]:
    """Compute image similarity metrics for measured/simulated images."""
    measured_norm: Float[NDArray, "height width"] = _normalise_image(measured)
    simulated_norm: Float[NDArray, "height width"] = _normalise_image(
        simulated
    )
    if measured_norm.shape != simulated_norm.shape:
        raise ValueError(
            "measured and simulated images must have the same shape; "
            f"got {measured_norm.shape} and {simulated_norm.shape}"
        )
    residual: Float[NDArray, "height width"] = simulated_norm - measured_norm
    measured_centered: Float[NDArray, "height width"] = (
        measured_norm - measured_norm.mean()
    )
    simulated_centered: Float[NDArray, "height width"] = (
        simulated_norm - simulated_norm.mean()
    )
    denom: float = float(
        np.linalg.norm(measured_centered) * np.linalg.norm(simulated_centered)
    )
    ncc: float = (
        0.0
        if denom <= 1e-12
        else float(np.vdot(measured_centered, simulated_centered) / denom)
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


def _load_image(path: Path) -> Float[NDArray, "height width"]:
    """Load a 2D image from a ``.npy`` or ``.npz`` artifact."""
    if path.suffix == ".npy":
        image = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            key: str = "image" if "image" in data.files else data.files[0]
            image = data[key]
    else:
        raise ValueError(
            f"unsupported image artifact extension: {path.suffix}"
        )
    image_np: Float[NDArray, "height width"] = np.asarray(
        image,
        dtype=np.float64,
    )
    if image_np.ndim != 2:
        raise ValueError(f"image must be 2D; got shape {image_np.shape}")
    return image_np


def _resolve_image_paths(spec: str) -> list[Path]:
    """Resolve a file, directory, or glob into sorted image artifact paths."""
    if not spec:
        raise ValueError("simulated_arrays is required unless --smoke is set")
    root = Path(spec)
    if root.is_dir():
        return sorted(
            path.resolve()
            for pattern in ("*.npz", "*.npy")
            for path in root.glob(pattern)
        )
    if any(char in spec for char in "*?[]"):
        return sorted(Path(item).resolve() for item in glob.glob(spec))
    return [root.resolve()]


@jaxtyped(typechecker=beartype)
def _gaussian_image(
    size: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> Float[NDArray, "height width"]:
    """Create a normalized Gaussian spot image for smoke matching."""
    axis: Float[NDArray, "axis"] = np.arange(size, dtype=np.float64)
    xx: Float[NDArray, "height width"]
    yy: Float[NDArray, "height width"]
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    image: Float[NDArray, "height width"] = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
    )
    return _normalise_image(image)


def _smoke_files(ctx: Any) -> tuple[str, str]:
    """Write smoke measured/simulated arrays and return their path specs."""
    smoke_dir = ctx.outdir / "smoke_images"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    measured = _gaussian_image(48, 22.0, 27.0, 4.0)
    best = _gaussian_image(48, 22.0, 27.0, 4.0)
    shifted = _gaussian_image(48, 28.0, 20.0, 4.0)
    measured_path = smoke_dir / "measured.npz"
    np.savez_compressed(measured_path, image=measured)
    np.savez_compressed(smoke_dir / "simulated_best.npz", image=best)
    np.savez_compressed(smoke_dir / "simulated_shifted.npz", image=shifted)
    return str(measured_path), str(smoke_dir)


@experiment(
    name="match-measured-to-simulated",
    params=[
        Param(
            "measured_array",
            str,
            default="",
            help="Measured detector image as .npy or .npz.",
            example="measured_frame.npz",
        ),
        Param(
            "simulated_arrays",
            str,
            default="",
            help=(
                "Simulated image file, directory, or glob of .npy/.npz files."
            ),
            example="simulated/*.npz",
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
    ],
    returns={
        "metrics": {
            "n_simulated": {"type": "integer"},
            "best_score": {"type": "number"},
            "best_artifact": {"type": "string"},
        },
        "artifacts": {
            "roles": ["scores", "best_match_image", "residual_image"],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Rank simulated detector images against a measured detector image."""
    measured_array: str = args.measured_array
    simulated_arrays: str = args.simulated_arrays
    if args.smoke and (not measured_array or not simulated_arrays):
        measured_array, simulated_arrays = _smoke_files(ctx)

    measured = _load_image(Path(measured_array))
    simulated_paths: list[Path] = _resolve_image_paths(simulated_arrays)
    if not simulated_paths:
        raise ValueError("no simulated image artifacts matched")

    rows: list[dict[str, Any]] = []
    images: dict[str, Float[NDArray, "height width"]] = {}
    for path in simulated_paths:
        image = _load_image(path)
        metrics = _score_images(measured, image)
        score: float = _metric_score(metrics, args.metric)
        images[path.as_posix()] = image
        rows.append(
            {
                "artifact": path.as_posix(),
                "score": score,
                "rank_metric": args.metric,
                **metrics,
            }
        )

    ranked: list[dict[str, Any]] = sorted(
        rows,
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    best = ranked[0]
    best_image = images[str(best["artifact"])]
    residual = np.abs(
        _normalise_image(best_image) - _normalise_image(measured)
    )

    scores_artifact = ctx.save_json(
        "match_scores.json",
        {"ranked": ranked},
        role="scores",
    )
    best_artifact = ctx.save_image(
        "best_match.png",
        best_image,
        cmap="phosphor",
        role="best_match_image",
    )
    residual_artifact = ctx.save_image(
        "best_residual.png",
        residual,
        cmap="phosphor",
        role="residual_image",
    )

    top_k: int = max(1, min(args.top_k, len(ranked)))
    return {
        "metrics": {
            "n_simulated": len(ranked),
            "best_score": float(best["score"]),
            "best_artifact": str(best["artifact"]),
            "best_mse": float(best["mse"]),
            "best_ncc": float(best["ncc"]),
        },
        "artifacts": [scores_artifact, best_artifact, residual_artifact],
        "ranked": ranked[:top_k],
    }


if __name__ == "__main__":
    main()
