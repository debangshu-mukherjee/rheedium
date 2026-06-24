"""Ahead-of-time export helpers for distributing rheedium forward models.

Extended Summary
----------------
:func:`jax.export.export` lowers a function to a portable StableHLO artifact
that can be serialized, shipped, reloaded, and called -- optionally across a
*range* of input shapes via symbolic dimensions. This module wraps that flow
with rheedium-specific guidance for the two failure modes the forward models
hit in practice.

The kinematic / Ewald path is shape-polymorphic in the atom count: a single
artifact handles any crystal size because the reflection grid is sized
statically from ``hmax``/``kmax``. The multislice path contains FFTs, which
cannot lower with a symbolic transform size, so it must be exported once per
detector grid and dispatched with :func:`bucketize_grid`.

Routine Listings
----------------
:func:`export_forward`
    Export a forward function to a :class:`jax.export.Exported` artifact.
:func:`serialize_exported`
    Serialize an exported artifact to portable bytes.
:func:`deserialize_exported`
    Reload an exported artifact from bytes.
:func:`bucketize_grid`
    Snap a requested detector grid up to the nearest pre-exported bucket.

Notes
-----
Functions carrying equinox runtime checks (``equinox.error_if``, used by the
``checked_*`` simulators) lower to host callbacks that cannot be serialized.
Because ``equinox`` reads ``EQX_ON_ERROR`` at import, the only way to strip
those callbacks is to set ``EQX_ON_ERROR=nan`` (or ``off``) *before* importing
rheedium, or to export the unchecked simulator. :func:`export_forward`
detects the resulting error and re-raises it with that guidance.
"""

import jax
from beartype import beartype
from beartype.typing import Any, Callable, Sequence, Tuple
from jaxtyping import jaxtyped


class ExportError(RuntimeError):
    """Raised when a forward model cannot be exported as-is.

    :see: :class:`~.test_exporting.TestExportForward`
    """


@jaxtyped(typechecker=beartype)
def export_forward(
    fn: Callable[..., Any],
    *example_args: Any,
) -> jax.export.Exported:
    """Export a forward function to a portable StableHLO artifact.

    Lowers ``fn`` over ``example_args`` -- which may carry symbolic shapes via
    :func:`jax.export.symbolic_shape` and :class:`jax.ShapeDtypeStruct` -- to a
    :class:`jax.export.Exported` artifact.

    :see: :class:`~.test_exporting.TestExportForward`

    Parameters
    ----------
    fn : Callable[..., Any]
        Pure forward function of array arguments. Static options (for example
        ``hmax``, ``kmax``, ``parameterization``) must be closed over rather
        than passed as traced arguments.
    *example_args : Any
        Example arguments defining the input signature. Use
        :class:`jax.ShapeDtypeStruct` with symbolic dimensions for
        shape-polymorphic axes (for example the atom count) and concrete
        dimensions elsewhere (for example FFT grids).

    Returns
    -------
    exported : jax.export.Exported
        The exported artifact, callable via ``exported.call`` and serializable
        via :func:`serialize_exported`.

    Raises
    ------
    ExportError
        If ``fn`` contains equinox runtime checks (host callbacks) or an FFT
        with a symbolic transform size, with guidance on how to proceed.

    Notes
    -----
    1. Wrap ``fn`` in :func:`jax.jit` and lower it over ``example_args``.
    2. Translate the two rheedium-specific ``NotImplementedError`` failures
       (host callbacks, symbolic FFT) into an :class:`ExportError`.
    """
    try:
        return jax.export.export(jax.jit(fn))(*example_args)
    except NotImplementedError as exc:
        message: str = str(exc)
        if "host_callbacks" in message:
            raise ExportError(
                "Function contains host callbacks from equinox runtime "
                "checks (equinox.error_if). Set EQX_ON_ERROR=nan (or off) "
                "before importing rheedium, or export the unchecked "
                "simulator, then retry."
            ) from exc
        if "FFT" in message or "fft" in message:
            raise ExportError(
                "Function contains an FFT with a symbolic transform size, "
                "which XLA cannot lower. Export one artifact per concrete "
                "detector grid and dispatch with bucketize_grid."
            ) from exc
        raise


@jaxtyped(typechecker=beartype)
def serialize_exported(exported: jax.export.Exported) -> bytes:
    """Serialize an exported artifact to portable bytes.

    :see: :class:`~.test_exporting.TestSerializeRoundTrip`

    Parameters
    ----------
    exported : jax.export.Exported
        Artifact produced by :func:`export_forward`.

    Returns
    -------
    blob : bytes
        Serialized StableHLO artifact suitable for writing to disk.
    """
    return bytes(exported.serialize())


@jaxtyped(typechecker=beartype)
def deserialize_exported(blob: bytes) -> jax.export.Exported:
    """Reload an exported artifact from bytes.

    :see: :class:`~.test_exporting.TestSerializeRoundTrip`

    Parameters
    ----------
    blob : bytes
        Bytes produced by :func:`serialize_exported`.

    Returns
    -------
    exported : jax.export.Exported
        The reloaded artifact, callable via ``exported.call``.
    """
    return jax.export.deserialize(blob)


@jaxtyped(typechecker=beartype)
def bucketize_grid(
    height: int,
    width: int,
    buckets: Sequence[int],
) -> Tuple[int, int]:
    """Snap a requested detector grid up to the nearest pre-exported bucket.

    Multislice artifacts are exported per concrete grid size; this picks the
    smallest bucket greater than or equal to the request in each dimension so
    the input can be padded to a size an existing artifact supports.

    :see: :class:`~.test_exporting.TestBucketizeGrid`

    Parameters
    ----------
    height : int
        Requested detector grid height in pixels.
    width : int
        Requested detector grid width in pixels.
    buckets : Sequence[int]
        Available pre-exported grid sizes (per dimension), e.g.
        ``(256, 512, 1024)``. Order does not matter.

    Returns
    -------
    grid : Tuple[int, int]
        ``(bucket_height, bucket_width)`` -- the smallest bucket that fits each
        requested dimension.

    Raises
    ------
    ValueError
        If ``buckets`` is empty, or a requested dimension exceeds every bucket.

    Notes
    -----
    1. Validate that buckets exist and the request fits the largest one.
    2. Select, per dimension, the smallest bucket that is at least the request.
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    ordered: list[int] = sorted(buckets)
    largest: int = ordered[-1]
    if height > largest or width > largest:
        raise ValueError(
            f"requested grid ({height}, {width}) exceeds the largest bucket "
            f"{largest}"
        )
    bucket_height: int = next(b for b in ordered if b >= height)
    bucket_width: int = next(b for b in ordered if b >= width)
    return (bucket_height, bucket_width)


__all__: list[str] = [
    "ExportError",
    "bucketize_grid",
    "deserialize_exported",
    "export_forward",
    "serialize_exported",
]
