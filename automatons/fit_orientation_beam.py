# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rheedium==2026.6.16"]
# ///
"""Fit orientation and beam parameters with rheedium.recon.

The automaton wraps :func:`rheedium.recon.fit_geometry_beam` around a compact
linear detector fixture. It is intentionally small enough for smoke-mode CI,
but still exercises the published recon API: a fixed crystal carrier, a
measured detector vector, a latent-to-physical transform, fitted orientation
and beam-mode parameters, and a Laplace covariance artifact.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, jaxtyped

import rheedium as rh
from rheedium.harness import Param, experiment
from rheedium.types import BeamModeDistribution, CrystalStructure

_GEOMETRY_BEAM_MATRIX: Float[Array, "pixels params"] = jnp.asarray(
    [
        [1.0, 0.0, 2.0, -0.5],
        [0.0, 1.0, -1.0, 1.5],
        [1.5, -0.5, 0.25, 0.75],
        [-0.25, 1.25, 1.0, 0.5],
        [0.75, 0.5, -0.5, 1.0],
    ],
    dtype=jnp.float64,
)


@jaxtyped(typechecker=beartype)
def _small_crystal() -> CrystalStructure:
    """Return a two-atom cubic crystal carrier for recon fixtures."""
    return rh.types.create_crystal_structure(
        frac_positions=jnp.asarray(
            [
                [0.0, 0.0, 0.0, 14.0],
                [0.5, 0.5, 0.5, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cart_positions=jnp.asarray(
            [
                [0.0, 0.0, 0.0, 14.0],
                [2.0, 2.0, 2.0, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cell_lengths=jnp.asarray([4.0, 4.0, 4.0], dtype=jnp.float64),
        cell_angles=jnp.asarray([90.0, 90.0, 90.0], dtype=jnp.float64),
    )


def _load_crystal(path: str) -> CrystalStructure:
    """Load a crystal path or return the synthetic fixture."""
    if not path:
        return _small_crystal()
    return rh.inout.parse_crystal(path)


def _array_param(
    values: list[Any], *, size: int, name: str
) -> Float[Array, "n"]:
    """Convert a JSON list parameter to a fixed-length float array."""
    array: Float[Array, "n"] = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have length {size}")
    return array


def _load_array(path: str) -> Float[Array, "pixels"]:
    """Load a one-dimensional detector vector from ``.npy`` or ``.npz``."""
    artifact = Path(path)
    if artifact.suffix == ".npy":
        raw = np.load(artifact)
    elif artifact.suffix == ".npz":
        with np.load(artifact) as data:
            key: str = (
                "measured" if "measured" in data.files else data.files[0]
            )
            raw = data[key]
    else:
        raise ValueError("measured_array must be a .npy or .npz file")
    array: Float[Array, "pixels"] = jnp.asarray(raw, dtype=jnp.float64)
    if array.ndim != 1:
        raise ValueError(f"measured array must be 1D; got {array.shape}")
    return array


def _geometry_beam_transform(
    latent: Float[Array, "params"],
) -> tuple[Float[Array, "orientation"], BeamModeDistribution]:
    """Map synthetic latent coordinates to orientation and beam modes."""
    orientation: Float[Array, "orientation"] = latent[:2]
    beam_modes = BeamModeDistribution(
        beta_in_plane=jnp.asarray(0.0, dtype=jnp.float64),
        beta_out_of_plane=jnp.asarray(0.0, dtype=jnp.float64),
        divergence_in_plane_rad=latent[2],
        divergence_out_of_plane_rad=latent[3],
        energy_spread_ev=jnp.asarray(0.0, dtype=jnp.float64),
        distribution_id="synthetic_geometry_beam",
    )
    return orientation, beam_modes


def _geometry_beam_forward(
    crystal: CrystalStructure,
    orientation: Float[Array, "orientation"],
    beam_modes: BeamModeDistribution,
) -> Float[Array, "pixels"]:
    """Return a linear detector fixture from orientation and beam modes."""
    del crystal
    parameter_vector: Float[Array, "params"] = jnp.asarray(
        [
            orientation[0],
            orientation[1],
            beam_modes.divergence_in_plane_rad,
            beam_modes.divergence_out_of_plane_rad,
        ],
        dtype=jnp.float64,
    )
    pixels: Float[Array, "pixels"] = _GEOMETRY_BEAM_MATRIX @ parameter_vector
    return pixels


@experiment(
    name="fit-orientation-beam",
    params=[
        Param(
            "crystal",
            str,
            default="",
            help="Optional crystal file used as the fixed carrier.",
            example="tests/test_data/SrTiO3.cif",
        ),
        Param(
            "measured_array",
            str,
            default="",
            help="Optional measured detector vector as .npy or .npz.",
            example="measured_geometry_beam.npz",
        ),
        Param(
            "true_latent",
            list,
            default=[1.25, -0.4, 0.2, 0.35],
            help="Synthetic target [theta, phi, div_in, div_out].",
            example=[1.25, -0.4, 0.2, 0.35],
        ),
        Param(
            "initial_latent",
            list,
            default=[0.0, 0.0, 0.0, 0.0],
            help="Initial latent [theta, phi, div_in, div_out].",
            example=[0.0, 0.0, 0.0, 0.0],
        ),
        Param(
            "mode",
            str,
            default="least_squares",
            help="Recon solver mode.",
            choices=("least_squares", "bfgs", "adamw"),
            example="least_squares",
        ),
        Param("max_steps", int, default=32, help="Maximum solver steps."),
        Param("rtol", float, default=1e-8, help="Relative solver tolerance."),
        Param("atol", float, default=1e-10, help="Absolute solver tolerance."),
        Param(
            "noise_variance",
            float,
            default=1.0,
            help="Noise variance for Laplace covariance.",
        ),
        Param(
            "uncertainty_regularization",
            float,
            default=1e-5,
            help="Fisher diagonal regularization.",
        ),
    ],
    returns={
        "metrics": {
            "orientation_l2_error": {"type": "number"},
            "beam_l2_error": {"type": "number"},
            "residual_mse": {"type": "number"},
        },
        "artifacts": {"roles": ["fit", "covariance"]},
    },
)
def main(args: Any, ctx: Any) -> dict[str, Any]:
    """Fit the synthetic orientation/beam inverse problem."""
    crystal: CrystalStructure = _load_crystal(args.crystal)
    true_latent: Float[Array, "params"] = _array_param(
        args.true_latent,
        size=4,
        name="true_latent",
    )
    initial_latent: Float[Array, "params"] = _array_param(
        args.initial_latent,
        size=4,
        name="initial_latent",
    )
    if args.measured_array:
        measured: Float[Array, "pixels"] = _load_array(args.measured_array)
    else:
        true_orientation, true_beam = _geometry_beam_transform(true_latent)
        measured = _geometry_beam_forward(crystal, true_orientation, true_beam)

    orientation: Float[Array, "orientation"]
    beam_modes: BeamModeDistribution
    covariance: Float[Array, "cov cov"]
    orientation, beam_modes, covariance = rh.recon.fit_geometry_beam(
        crystal=crystal,
        measured=measured,
        forward=_geometry_beam_forward,
        initial_latent=initial_latent,
        transform=_geometry_beam_transform,
        mode=args.mode,
        max_steps=args.max_steps,
        rtol=args.rtol,
        atol=args.atol,
        noise_variance=args.noise_variance,
        uncertainty_regularization=args.uncertainty_regularization,
    )

    fitted_vector: Float[Array, "params"] = jnp.asarray(
        [
            orientation[0],
            orientation[1],
            beam_modes.divergence_in_plane_rad,
            beam_modes.divergence_out_of_plane_rad,
        ],
        dtype=jnp.float64,
    )
    residual: Float[Array, "pixels"] = (
        _geometry_beam_forward(crystal, orientation, beam_modes) - measured
    )
    orientation_error: float = float(
        jnp.linalg.norm(fitted_vector[:2] - true_latent[:2])
    )
    beam_error: float = float(
        jnp.linalg.norm(fitted_vector[2:] - true_latent[2:])
    )
    residual_mse: float = float(jnp.mean(residual**2))
    fit_payload: dict[str, Any] = {
        "orientation": np.asarray(orientation).tolist(),
        "beam": {
            "divergence_in_plane_rad": float(
                beam_modes.divergence_in_plane_rad
            ),
            "divergence_out_of_plane_rad": float(
                beam_modes.divergence_out_of_plane_rad
            ),
            "energy_spread_ev": float(beam_modes.energy_spread_ev),
        },
        "fitted_latent": np.asarray(fitted_vector).tolist(),
        "target_latent": np.asarray(true_latent).tolist(),
        "covariance_diag": np.asarray(jnp.diag(covariance)).tolist(),
    }
    fit_artifact = ctx.save_json(
        "fit_geometry_beam.json",
        fit_payload,
        role="fit",
    )
    covariance_artifact = ctx.save_array(
        "fit_covariance.npz",
        {"covariance": np.asarray(covariance)},
        role="covariance",
    )
    metrics: dict[str, Any] = {
        "orientation_l2_error": orientation_error,
        "beam_l2_error": beam_error,
        "residual_mse": residual_mse,
        "covariance_trace": float(jnp.trace(covariance)),
    }
    return {
        "metrics": metrics,
        "artifacts": [fit_artifact, covariance_artifact],
        "fit": fit_payload,
    }


if __name__ == "__main__":
    main()
