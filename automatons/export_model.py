# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.15"]
# ///
"""Export a portable StableHLO kinematic forward artifact.

The automaton lowers the kinematic Ewald forward path with a symbolic atom
count, writes the serialized StableHLO artifact and manifest, then reloads the
artifact in a separate Python process and checks it against the in-process
forward result. The exported callable accepts crystal arrays rather than a
Python structure object so it can be reused by deployment code without the
development-time wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("EQX_ON_ERROR", "nan")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from beartype.typing import Any  # noqa: E402
from jaxtyping import Array, Float  # noqa: E402

import rheedium as rh  # noqa: E402
from rheedium.harness import Param, experiment  # noqa: E402
from rheedium.types import CrystalStructure  # noqa: E402

_RELOAD_PROBE: str = r"""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import rheedium as rh

artifact = Path(sys.argv[1])
blob = artifact.read_bytes()
exported = rh.tools.deserialize_exported(blob)
frac2 = jnp.asarray(
    [[0.0, 0.0, 0.0, 12.0], [0.5, 0.5, 0.5, 8.0]],
    dtype=jnp.float64,
)
cart2 = jnp.asarray(
    [[0.0, 0.0, 0.0, 12.0], [2.105, 2.105, 2.105, 8.0]],
    dtype=jnp.float64,
)
frac3 = jnp.asarray(
    [
        [0.0, 0.0, 0.0, 12.0],
        [0.5, 0.5, 0.5, 8.0],
        [0.25, 0.25, 0.25, 12.0],
    ],
    dtype=jnp.float64,
)
cart3 = jnp.asarray(
    [
        [0.0, 0.0, 0.0, 12.0],
        [2.105, 2.105, 2.105, 8.0],
        [1.0525, 1.0525, 1.0525, 12.0],
    ],
    dtype=jnp.float64,
)
lengths = jnp.asarray([4.21, 4.21, 4.21], dtype=jnp.float64)
angles = jnp.asarray([90.0, 90.0, 90.0], dtype=jnp.float64)
points2, intensities2 = exported.call(frac2, cart2, lengths, angles)
points3, intensities3 = exported.call(frac3, cart3, lengths, angles)
payload = {
    "atom_counts": [int(frac2.shape[0]), int(frac3.shape[0])],
    "n_reflections": [int(intensities2.shape[0]), int(intensities3.shape[0])],
    "sum_intensity": [
        float(jnp.sum(intensities2)),
        float(jnp.sum(intensities3)),
    ],
    "sum_detector_points": [float(jnp.sum(points2)), float(jnp.sum(points3))],
}
print(json.dumps(payload, sort_keys=True))
"""


def _kinematic_forward(
    frac_positions: Float[Array, "N 4"],
    cart_positions: Float[Array, "N 4"],
    cell_lengths: Float[Array, "3"],
    cell_angles: Float[Array, "3"],
    *,
    energy_kev: float,
    theta_deg: float,
    hmax: int,
    kmax: int,
) -> tuple[Float[Array, "R two"], Float[Array, "R"]]:
    """Return detector points and intensities for one crystal array set."""
    crystal = CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    pattern = rh.simul.ewald_simulator(
        crystal,
        energy_kev=energy_kev,
        theta_deg=theta_deg,
        hmax=hmax,
        kmax=kmax,
    )
    return pattern.detector_points, pattern.intensities


def _sample_crystals() -> dict[str, Float[Array, "..."]]:
    """Return two tiny crystals used for export validation."""
    frac2: Float[Array, "2 4"] = jnp.asarray(
        [[0.0, 0.0, 0.0, 12.0], [0.5, 0.5, 0.5, 8.0]],
        dtype=jnp.float64,
    )
    cart2: Float[Array, "2 4"] = jnp.asarray(
        [[0.0, 0.0, 0.0, 12.0], [2.105, 2.105, 2.105, 8.0]],
        dtype=jnp.float64,
    )
    frac3: Float[Array, "3 4"] = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 12.0],
            [0.5, 0.5, 0.5, 8.0],
            [0.25, 0.25, 0.25, 12.0],
        ],
        dtype=jnp.float64,
    )
    cart3: Float[Array, "3 4"] = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 12.0],
            [2.105, 2.105, 2.105, 8.0],
            [1.0525, 1.0525, 1.0525, 12.0],
        ],
        dtype=jnp.float64,
    )
    return {
        "frac2": frac2,
        "cart2": cart2,
        "frac3": frac3,
        "cart3": cart3,
        "lengths": jnp.asarray([4.21, 4.21, 4.21], dtype=jnp.float64),
        "angles": jnp.asarray([90.0, 90.0, 90.0], dtype=jnp.float64),
    }


def _artifact_entry(
    ctx: Any, path: Path, *, role: str, mime: str
) -> dict[str, str]:
    """Build a schema-compatible artifact entry for a manually written file."""
    relative: Path = path.resolve().relative_to(ctx.outdir.resolve())
    return {"role": role, "mime": mime, "path": relative.as_posix()}


def _run_reloaded_process(artifact_path: Path) -> dict[str, Any]:
    """Reload and call a serialized export in a separate Python process."""
    env: dict[str, str] = dict(os.environ)
    env.setdefault("EQX_ON_ERROR", "nan")
    env.setdefault("JAX_PLATFORMS", "cpu")
    result = subprocess.run(
        [sys.executable, "-c", _RELOAD_PROBE, str(artifact_path)],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


@experiment(
    name="export-model",
    params=[
        Param(
            "model",
            str,
            default="kinematic",
            choices=("kinematic",),
            help="Forward model family to export.",
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
    ],
    returns={
        "metrics": {
            "artifact_bytes": {"type": "integer"},
            "separate_process_ok": {"type": "boolean"},
            "same_result_max_abs_error": {"type": "number"},
        },
        "artifacts": {
            "roles": [
                "stablehlo_artifact",
                "export_manifest",
                "export_arrays",
            ],
        },
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Export, serialize, reload, and validate the kinematic forward kernel."""
    hmax: int = min(args.hmax, 1) if args.smoke else args.hmax
    kmax: int = min(args.kmax, 1) if args.smoke else args.kmax
    samples: dict[str, Float[Array, "..."]] = _sample_crystals()

    def _forward(
        frac_positions: Float[Array, "N 4"],
        cart_positions: Float[Array, "N 4"],
        cell_lengths: Float[Array, "3"],
        cell_angles: Float[Array, "3"],
    ) -> tuple[Float[Array, "R two"], Float[Array, "R"]]:
        return _kinematic_forward(
            frac_positions,
            cart_positions,
            cell_lengths,
            cell_angles,
            energy_kev=args.energy_kev,
            theta_deg=args.theta_deg,
            hmax=hmax,
            kmax=kmax,
        )

    (n_atoms,) = jax.export.symbolic_shape("n_atoms")
    exported = rh.tools.export_forward(
        _forward,
        jax.ShapeDtypeStruct((n_atoms, 4), jnp.float64),
        jax.ShapeDtypeStruct((n_atoms, 4), jnp.float64),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jax.ShapeDtypeStruct((3,), jnp.float64),
    )
    blob: bytes = rh.tools.serialize_exported(exported)
    artifact_path: Path = ctx.path_for_artifact("forward_kinematic.stablehlo")
    artifact_path.write_bytes(blob)
    stablehlo_artifact = _artifact_entry(
        ctx,
        artifact_path,
        role="stablehlo_artifact",
        mime="application/vnd.stablehlo",
    )

    in_points2, in_intensities2 = _forward(
        samples["frac2"],
        samples["cart2"],
        samples["lengths"],
        samples["angles"],
    )
    ex_points2, ex_intensities2 = exported.call(
        samples["frac2"],
        samples["cart2"],
        samples["lengths"],
        samples["angles"],
    )
    in_points3, in_intensities3 = _forward(
        samples["frac3"],
        samples["cart3"],
        samples["lengths"],
        samples["angles"],
    )
    ex_points3, ex_intensities3 = exported.call(
        samples["frac3"],
        samples["cart3"],
        samples["lengths"],
        samples["angles"],
    )
    inprocess_errors: list[float] = [
        float(jnp.max(jnp.abs(in_points2 - ex_points2))),
        float(jnp.max(jnp.abs(in_intensities2 - ex_intensities2))),
        float(jnp.max(jnp.abs(in_points3 - ex_points3))),
        float(jnp.max(jnp.abs(in_intensities3 - ex_intensities3))),
    ]
    process_payload: dict[str, Any] = _run_reloaded_process(artifact_path)
    process_expected: list[float] = [
        float(jnp.sum(ex_intensities2)),
        float(jnp.sum(ex_intensities3)),
    ]
    process_errors: list[float] = [
        abs(observed - expected)
        for observed, expected in zip(
            process_payload["sum_intensity"],
            process_expected,
            strict=True,
        )
    ]
    same_result_error: float = max([*inprocess_errors, *process_errors])
    sha256: str = hashlib.sha256(blob).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": "rheedium.export_model.v1",
        "model": args.model,
        "rheedium_version": rh.__version__,
        "artifact": stablehlo_artifact["path"],
        "sha256": sha256,
        "artifact_bytes": len(blob),
        "export_tool": "rheedium.tools.export_forward",
        "input_schema": {
            "symbolic_axes": {"n_atoms": "leading atom axis"},
            "args": [
                {"name": "frac_positions", "shape": ["n_atoms", 4]},
                {"name": "cart_positions", "shape": ["n_atoms", 4]},
                {"name": "cell_lengths", "shape": [3]},
                {"name": "cell_angles", "shape": [3]},
            ],
            "returns": [
                {"name": "detector_points", "shape": ["reflections", 2]},
                {"name": "intensities", "shape": ["reflections"]},
            ],
            "dtype": "float64",
        },
        "validation": {
            "atom_counts": process_payload["atom_counts"],
            "separate_process": process_payload,
            "same_result_max_abs_error": same_result_error,
        },
    }
    manifest_artifact = ctx.save_json(
        "forward_kinematic_export.json",
        manifest,
        role="export_manifest",
    )
    arrays_artifact = ctx.save_array(
        "forward_kinematic_export_samples.npz",
        {
            "frac2": np.asarray(samples["frac2"]),
            "cart2": np.asarray(samples["cart2"]),
            "frac3": np.asarray(samples["frac3"]),
            "cart3": np.asarray(samples["cart3"]),
            "lengths": np.asarray(samples["lengths"]),
            "angles": np.asarray(samples["angles"]),
            "intensities2": np.asarray(ex_intensities2),
            "intensities3": np.asarray(ex_intensities3),
        },
        role="export_arrays",
    )
    metrics: dict[str, Any] = {
        "artifact_bytes": len(blob),
        "separate_process_ok": process_payload["atom_counts"] == [2, 3],
        "same_result_max_abs_error": same_result_error,
        "atom_counts_verified": process_payload["atom_counts"],
        "n_reflections": process_payload["n_reflections"],
    }
    return {
        "metrics": metrics,
        "artifacts": [
            stablehlo_artifact,
            manifest_artifact,
            arrays_artifact,
        ],
        "export": manifest,
    }


if __name__ == "__main__":
    main()
