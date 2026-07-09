"""Automaton process-boundary implementation for the rheedium harness.

Extended Summary
----------------
This module holds the concrete implementation of the automaton harness whose
public surface is re-exported by :mod:`rheedium.harness`. It defines the
declarative :class:`Param` metadata, the per-run :class:`ExperimentContext`,
the structured :class:`AutomatonError` taxonomy, the :func:`experiment`
decorator that turns a ``main(args, ctx)`` function into a runnable CLI, and
the :func:`emit` result writer.

Notes
-----
Import public names from :mod:`rheedium.harness` rather than from this module
directly; the package ``__init__`` is the stable interface and this module is
an implementation detail that may be reorganized.

:see: rheedium.harness
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import jaxtyped

from rheedium.tools import enable_compilation_cache

DESCRIBE_SCHEMA_VERSION: str = "rheedium.automaton.describe.v1"
RESULT_SCHEMA_VERSION: str = "rheedium.automaton.result.v1"
_MISSING: object = object()
_PREVIEW_MAX_BYTES: int = 65536


class ErrorKind:
    """Agent-readable error categories emitted by the harness."""

    INVALID_STRUCTURE: str = "InvalidStructure"
    PARAM_OUT_OF_RANGE: str = "ParamOutOfRange"
    UNSUPPORTED: str = "Unsupported"
    RESOURCE_EXHAUSTED: str = "ResourceExhausted"
    NUMERICAL_FAILURE: str = "NumericalFailure"
    TIMEOUT: str = "Timeout"
    UNKNOWN: str = "Unknown"


class AutomatonError(Exception):
    """Structured error raised at an automaton process boundary.

    Parameters
    ----------
    message : str
        Human-readable error message.
    error_kind : str, optional
        Machine-readable category from :class:`ErrorKind`.
    field : str, optional
        Parameter or input field that caused the error.
    """

    def __init__(
        self,
        message: str,
        *,
        error_kind: str = ErrorKind.UNKNOWN,
        field: str | None = None,
    ) -> None:
        """Initialize the structured automaton error."""
        super().__init__(message)
        self.error_kind: str = error_kind
        self.field: str | None = field


class DeadlineExceededError(AutomatonError):
    """Raised when ``--deadline`` interrupts an automaton run."""

    def __init__(self, seconds: float) -> None:
        """Initialize the deadline error."""
        super().__init__(
            f"deadline exceeded after {seconds:.3f} seconds",
            error_kind=ErrorKind.TIMEOUT,
        )


@dataclass(frozen=True)
class Param:
    """Typed parameter declaration for an automaton.

    Parameters
    ----------
    name : str
        Python-safe parameter name, for example ``"energy_kev"``.
    python_type : type
        Expected Python type. Supported CLI scalar types are ``str``, ``int``,
        ``float``, and ``bool``. ``dict`` and ``list`` accept JSON strings.
    default : Any, optional
        Default value. If omitted, the parameter is required.
    help : str, optional
        Human-facing help text.
    unit : str, optional
        Physical unit surfaced in ``--describe``.
    bounds : tuple[float | None, float | None], optional
        Inclusive numeric lower and upper bounds.
    choices : Sequence[Any], optional
        Allowed values.
    example : Any, optional
        Example value surfaced in ``--describe``.
    """

    name: str
    python_type: type[Any]
    default: Any = _MISSING
    help: str = ""
    unit: str | None = None
    bounds: tuple[float | None, float | None] | None = None
    choices: Sequence[Any] | None = None
    example: Any = _MISSING

    @property
    def required(self) -> bool:
        """Return whether this parameter has no default."""
        return self.default is _MISSING

    @property
    def json_type(self) -> str:
        """Return this parameter's JSON type name."""
        return _json_type_for_python_type(self.python_type)

    def describe(self) -> dict[str, Any]:
        """Return agent-facing parameter metadata."""
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.json_type,
            "required": self.required,
            "help": self.help,
        }
        if not self.required:
            payload["default"] = _json_ready(self.default)
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.bounds is not None:
            payload["bounds"] = list(self.bounds)
        if self.choices is not None:
            payload["choices"] = _json_ready(list(self.choices))
        if self.example is not _MISSING:
            payload["example"] = _json_ready(self.example)
        return payload

    def to_json_schema(self) -> dict[str, Any]:
        """Return this parameter as a JSON-Schema property."""
        schema: dict[str, Any] = {
            "type": self.json_type,
            "description": self.help,
        }
        if self.choices is not None:
            schema["enum"] = _json_ready(list(self.choices))
        if self.bounds is not None:
            lower, upper = self.bounds
            if lower is not None:
                schema["minimum"] = lower
            if upper is not None:
                schema["maximum"] = upper
        if not self.required:
            schema["default"] = _json_ready(self.default)
        if self.unit is not None:
            schema["unit"] = self.unit
        if self.example is not _MISSING:
            schema["examples"] = [_json_ready(self.example)]
        return schema

    def validate(self, value: Any) -> Any:
        """Validate and coerce a value for this parameter."""
        coerced: Any = _coerce_value(self, value)
        if self.choices is not None and coerced not in self.choices:
            raise AutomatonError(
                f"{self.name} must be one of {list(self.choices)!r}",
                error_kind=ErrorKind.UNSUPPORTED,
                field=self.name,
            )
        if self.bounds is not None:
            _validate_bounds(self, coerced)
        return coerced


@dataclass(frozen=True)
class ExperimentSpec:
    """Static declaration attached to a decorated automaton."""

    name: str
    params: tuple[Param, ...]
    returns: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    estimate: Callable[[SimpleNamespace], Mapping[str, Any]] | None = None

    def describe(self) -> dict[str, Any]:
        """Return the full ``--describe`` payload for this experiment."""
        return {
            "schema_version": DESCRIBE_SCHEMA_VERSION,
            "experiment": self.name,
            "summary": _summary_from_description(self.description),
            "description": self.description,
            "params": [param.describe() for param in self.params],
            "params_schema": _params_json_schema(self.params),
            "returns": _json_ready(dict(self.returns)),
            "inherited_flags": _inherited_flags(),
        }


@dataclass
class ExperimentContext:
    """Per-run context passed to an automaton body.

    Parameters
    ----------
    outdir : Path
        Directory where artifacts are written.
    seed : int
        Deterministic run seed.
    experiment : str
        Experiment name.
    json_mode : bool, optional
        Whether strict JSON stdout mode was requested.
    """

    outdir: Path
    seed: int
    experiment: str
    json_mode: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create the output directory and deterministic JAX key."""
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rng_key = jax.random.PRNGKey(self.seed)

    def path_for_artifact(self, name: str) -> Path:
        """Return a safe output path for an artifact name."""
        base: Path = self.outdir.resolve()
        target: Path = (self.outdir / name).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise AutomatonError(
                f"artifact path escapes outdir: {name}",
                error_kind=ErrorKind.UNSUPPORTED,
                field="artifact",
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(
        self,
        name: str,
        data: Mapping[str, Any],
        *,
        role: str = "metrics",
    ) -> dict[str, Any]:
        """Serialize a JSON artifact and return its manifest entry."""
        path: Path = self.path_for_artifact(name)
        path.write_text(
            json.dumps(_json_ready(dict(data)), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return self._record_artifact(path, role=role, mime="application/json")

    def save_array(
        self,
        name: str,
        data: Any,
        *,
        role: str = "array",
    ) -> dict[str, Any]:
        """Serialize an array or array mapping as a compressed ``.npz``."""
        path: Path = self.path_for_artifact(name)
        if isinstance(data, Mapping):
            arrays: dict[str, Any] = {
                str(key): np.asarray(value) for key, value in data.items()
            }
            np.savez_compressed(path, **arrays)
        else:
            np.savez_compressed(path, array=np.asarray(data))
        return self._record_artifact(path, role=role, mime="application/npz")

    def save_image(
        self,
        name: str,
        image: Any,
        *,
        role: str = "detector_image",
        cmap: str = "phosphor",
        preview: bool = True,
    ) -> dict[str, Any]:
        """Render a 2D array as a PNG artifact."""
        from matplotlib import pyplot as plt  # noqa: PLC0415

        path: Path = self.path_for_artifact(name)
        image_np: np.ndarray[Any, Any] = np.asarray(image)
        if cmap == "phosphor":
            from rheedium.plots import (  # noqa: I001, PLC0415
                create_phosphor_colormap,
            )

            cmap_obj: Any = create_phosphor_colormap()
        else:
            cmap_obj = cmap
        plt.imsave(path, image_np, cmap=cmap_obj)
        return self._record_artifact(
            path,
            role=role,
            mime="image/png",
            preview=preview,
        )

    def save_figure(
        self,
        name: str,
        figure: Any | None = None,
        *,
        role: str = "figure",
        preview: bool = True,
    ) -> dict[str, Any]:
        """Save a matplotlib figure as a PNG artifact."""
        from matplotlib import pyplot as plt  # noqa: PLC0415

        path: Path = self.path_for_artifact(name)
        fig: Any = figure if figure is not None else plt.gcf()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return self._record_artifact(
            path,
            role=role,
            mime="image/png",
            preview=preview,
        )

    def log(self, message: str) -> None:
        """Write a human-facing message to stderr."""
        print(f"[{self.experiment}] {message}", file=sys.stderr)

    def _record_artifact(
        self,
        path: Path,
        *,
        role: str,
        mime: str,
        preview: bool = False,
    ) -> dict[str, Any]:
        """Record and return a typed artifact manifest entry."""
        relative: Path = path.resolve().relative_to(self.outdir.resolve())
        artifact: dict[str, Any] = {
            "role": role,
            "mime": mime,
            "path": relative.as_posix(),
        }
        if preview and path.stat().st_size <= _PREVIEW_MAX_BYTES:
            artifact["preview_b64"] = base64.b64encode(
                path.read_bytes()
            ).decode("ascii")
        self.artifacts.append(artifact)
        return artifact


def experiment(
    *,
    name: str,
    params: Sequence[Param],
    returns: Mapping[str, Any] | None = None,
    estimate: Callable[[SimpleNamespace], Mapping[str, Any]] | None = None,
) -> Callable[
    [Callable[[SimpleNamespace, ExperimentContext], Mapping[str, Any] | None]],
    Callable[[Sequence[str] | None], int],
]:
    """Decorate an automaton ``main(args, ctx)`` function.

    The wrapped function builds a CLI around the declared parameters and owns
    result emission. When called with ``argv=None`` it exits the process with
    the correct status code, matching ordinary script execution. Tests may pass
    an explicit argument list to receive the integer status code directly.
    """

    def _decorate(
        func: Callable[
            [SimpleNamespace, ExperimentContext],
            Mapping[str, Any] | None,
        ],
    ) -> Callable[[Sequence[str] | None], int]:
        spec: ExperimentSpec = ExperimentSpec(
            name=name,
            params=tuple(params),
            returns=returns or {},
            description=func.__doc__ or "",
            estimate=estimate,
        )

        @beartype
        def _wrapped(argv: Sequence[str] | None = None) -> int:
            code: int = _run_experiment(spec, func, argv)
            if argv is None:
                raise SystemExit(code)
            return code

        _wrapped.__experiment_spec__ = spec  # type: ignore[attr-defined]
        return _wrapped

    return _decorate


@jaxtyped(typechecker=beartype)
def emit(result: Mapping[str, Any]) -> None:
    """Write a machine-readable JSON result as one stdout line."""
    print(json.dumps(_json_ready(dict(result)), sort_keys=True), flush=True)


def _run_experiment(  # noqa: PLR0911, PLR0912, PLR0915
    spec: ExperimentSpec,
    func: Callable[
        [SimpleNamespace, ExperimentContext], Mapping[str, Any] | None
    ],
    argv: Sequence[str] | None,
) -> int:
    """Run one automaton invocation from parsed command-line inputs."""
    parser: argparse.ArgumentParser = _build_parser(spec)
    namespace: argparse.Namespace = parser.parse_args(argv)

    if namespace.describe:
        emit(spec.describe())
        return 0

    start: float = time.perf_counter()
    params: dict[str, Any] = {}
    seed: int = int(namespace.seed)
    context: ExperimentContext | None = None

    try:
        params = _merge_and_validate_params(spec.params, namespace)
        context = ExperimentContext(
            outdir=Path(namespace.outdir),
            seed=seed,
            experiment=spec.name,
            json_mode=bool(namespace.json),
        )
        if namespace.cache:
            cache_dir: str = enable_compilation_cache()
            context.log(f"persistent compilation cache enabled at {cache_dir}")
        if namespace.unchecked:
            os.environ["RHEEDIUM_DISABLE_RUNTIME_CHECKS"] = "1"
        run_args: SimpleNamespace = SimpleNamespace(
            **params,
            smoke=bool(namespace.smoke),
            seed=seed,
        )

        if namespace.validate:
            result: dict[str, Any] = _build_result(
                spec=spec,
                status="ok",
                seed=seed,
                params=params,
                metrics={"valid": True},
                artifacts=[],
                start=start,
            )
            emit(result)
            return 0

        if namespace.estimate:
            metrics: Mapping[str, Any]
            if spec.estimate is None:
                metrics = {
                    "est_wall_s": None,
                    "needs_gpu": False,
                    "est_mem_gb": None,
                    "cache_warm": None,
                }
            else:
                metrics = spec.estimate(run_args)
            result = _build_result(
                spec=spec,
                status="ok",
                seed=seed,
                params=params,
                metrics=metrics,
                artifacts=[],
                start=start,
            )
            emit(result)
            return 0

        if namespace.warmup:
            context.log("warmup requested; no shared warmup hook is declared")

        with _deadline_context(float(namespace.deadline)):
            returned: Mapping[str, Any] | None = func(run_args, context)
        metrics, artifacts, extras = _normalise_run_payload(
            returned,
            context.artifacts,
        )
        result = _build_result(
            spec=spec,
            status="ok",
            seed=seed,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            start=start,
            extras=extras,
        )
        emit(result)
        return 0
    except DeadlineExceededError as exc:
        return _emit_error(spec, exc, seed, params, start, status="timeout")
    except Exception as exc:
        if not isinstance(exc, AutomatonError):
            traceback.print_exc(file=sys.stderr)
        return _emit_error(spec, exc, seed, params, start)


def _build_parser(spec: ExperimentSpec) -> argparse.ArgumentParser:
    """Build the automaton argument parser from a specification."""
    parser = argparse.ArgumentParser(
        prog=spec.name,
        description=_summary_from_description(spec.description),
    )
    parser.add_argument("--outdir", default="automaton_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--unchecked", action="store_true")
    parser.add_argument("--params", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--deadline", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    for param in spec.params:
        flags: tuple[str, ...] = (
            f"--{param.name.replace('_', '-')}",
            f"--{param.name}",
        )
        kwargs: dict[str, Any] = {
            "dest": param.name,
            "default": argparse.SUPPRESS,
            "help": param.help,
        }
        if param.python_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = _argparse_type(param.python_type)
        if param.choices is not None and param.python_type not in {dict, list}:
            kwargs["choices"] = list(param.choices)
        parser.add_argument(*flags, **kwargs)
    return parser


def _argparse_type(python_type: type[Any]) -> Callable[[str], Any]:
    """Return an argparse converter for a supported Python type."""
    if python_type in {str, int, float}:
        return python_type
    if python_type in {dict, list}:
        return json.loads
    raise TypeError(f"unsupported Param type: {python_type!r}")


def _merge_and_validate_params(
    params: Sequence[Param],
    namespace: argparse.Namespace,
) -> dict[str, Any]:
    """Merge defaults, ``--params`` JSON, and explicit CLI flags."""
    declared: dict[str, Param] = {param.name: param for param in params}
    raw: dict[str, Any] = {}
    for param in params:
        if not param.required:
            raw[param.name] = param.default

    if namespace.params is not None:
        incoming: Mapping[str, Any] = _read_params_document(namespace.params)
        unknown: set[str] = set(incoming) - set(declared)
        if unknown:
            raise AutomatonError(
                f"unknown parameter(s): {sorted(unknown)!r}",
                error_kind=ErrorKind.UNSUPPORTED,
                field="params",
            )
        raw.update(dict(incoming))

    for param in params:
        if hasattr(namespace, param.name):
            raw[param.name] = getattr(namespace, param.name)

    validated: dict[str, Any] = {}
    for param in params:
        if param.name not in raw:
            raise AutomatonError(
                f"missing required parameter: {param.name}",
                error_kind=ErrorKind.UNSUPPORTED,
                field=param.name,
            )
        validated[param.name] = param.validate(raw[param.name])
    return validated


def _read_params_document(source: str) -> Mapping[str, Any]:
    """Read a JSON params object from a file, stdin, or inline string."""
    if source == "-":
        text: str = sys.stdin.read()
    else:
        try:
            path = Path(source)
            text = (
                path.read_text(encoding="utf-8") if path.exists() else source
            )
        except OSError:
            text = source
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AutomatonError(
            f"--params is not valid JSON: {exc}",
            error_kind=ErrorKind.UNSUPPORTED,
            field="params",
        ) from exc
    if not isinstance(data, Mapping):
        raise AutomatonError(
            "--params must be a JSON object",
            error_kind=ErrorKind.UNSUPPORTED,
            field="params",
        )
    return data


def _coerce_value(  # noqa: PLR0911, PLR0912
    param: Param,
    value: Any,
) -> Any:
    """Coerce one raw value according to a parameter declaration."""
    expected: type[Any] = param.python_type
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered: str = value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise _type_error(param, value)
    if expected is int:
        if isinstance(value, bool):
            raise _type_error(param, value)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise _type_error(param, value) from exc
    if expected is float:
        if isinstance(value, bool):
            raise _type_error(param, value)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise _type_error(param, value) from exc
    if expected is str:
        if isinstance(value, str):
            return value
        raise _type_error(param, value)
    if expected in {dict, list}:
        if isinstance(value, expected):
            return value
        if isinstance(value, str):
            parsed: Any = json.loads(value)
            if isinstance(parsed, expected):
                return parsed
        raise _type_error(param, value)
    if isinstance(value, expected):
        return value
    raise _type_error(param, value)


def _type_error(param: Param, value: Any) -> AutomatonError:
    """Build a structured parameter type error."""
    expected: str = param.python_type.__name__
    return AutomatonError(
        f"{param.name} must be {expected}; got {type(value).__name__}",
        error_kind=ErrorKind.UNSUPPORTED,
        field=param.name,
    )


def _validate_bounds(param: Param, value: Any) -> None:
    """Validate inclusive numeric bounds for one parameter."""
    if param.bounds is None:
        return
    lower, upper = param.bounds
    try:
        numeric: float = float(value)
    except (TypeError, ValueError) as exc:
        raise AutomatonError(
            f"{param.name} has bounds but is not numeric",
            error_kind=ErrorKind.UNSUPPORTED,
            field=param.name,
        ) from exc
    if lower is not None and numeric < lower:
        raise AutomatonError(
            f"{param.name}={numeric} is below minimum {lower}",
            error_kind=ErrorKind.PARAM_OUT_OF_RANGE,
            field=param.name,
        )
    if upper is not None and numeric > upper:
        raise AutomatonError(
            f"{param.name}={numeric} exceeds maximum {upper}",
            error_kind=ErrorKind.PARAM_OUT_OF_RANGE,
            field=param.name,
        )


def _params_json_schema(params: Sequence[Param]) -> dict[str, Any]:
    """Build the per-experiment JSON input schema."""
    properties: dict[str, Any] = {
        param.name: param.to_json_schema() for param in params
    }
    required: list[str] = [param.name for param in params if param.required]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _normalise_run_payload(
    payload: Mapping[str, Any] | None,
    context_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Extract metrics, artifacts, and extras from a run return value."""
    if payload is None:
        return {}, [dict(item) for item in context_artifacts], {}
    data: dict[str, Any] = dict(payload)
    metrics: Mapping[str, Any] = data.pop("metrics", {})
    raw_artifacts: Any = data.pop("artifacts", None)
    if raw_artifacts is None:
        artifacts = [dict(item) for item in context_artifacts]
    else:
        artifacts = [_normalise_artifact(item) for item in raw_artifacts]
    return metrics, artifacts, data


def _normalise_artifact(item: Any) -> dict[str, Any]:
    """Convert a path-like artifact into a typed manifest entry."""
    if isinstance(item, Mapping):
        return dict(item)
    return {
        "role": "artifact",
        "mime": "application/octet-stream",
        "path": str(item),
    }


def _build_result(
    *,
    spec: ExperimentSpec,
    status: str,
    seed: int,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    start: float,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical automaton result object."""
    wall_seconds: float = max(0.0, time.perf_counter() - start)
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "experiment": spec.name,
        "rheedium_version": _rheedium_version(),
        "jax_backend": _jax_backend(),
        "seed": seed,
        "params": _json_ready(dict(params)),
        "metrics": _json_ready(dict(metrics)),
        "artifacts": _json_ready([dict(item) for item in artifacts]),
        "wall_seconds": wall_seconds,
        "returns": _json_ready(dict(spec.returns)),
    }
    payload["result_key"] = _result_key(
        experiment=spec.name,
        params=payload["params"],
        seed=seed,
        rheedium_version=payload["rheedium_version"],
    )
    if extras:
        payload.update(_json_ready(dict(extras)))
    return payload


def _emit_error(
    spec: ExperimentSpec,
    exc: Exception,
    seed: int,
    params: Mapping[str, Any],
    start: float,
    *,
    status: str = "error",
) -> int:
    """Emit a structured error result and return an exit code."""
    automaton_error: AutomatonError = _classify_exception(exc)
    result: dict[str, Any] = _build_result(
        spec=spec,
        status=status,
        seed=seed,
        params=params,
        metrics={},
        artifacts=[],
        start=start,
    )
    result["error"] = str(automaton_error)
    result["error_kind"] = automaton_error.error_kind
    if automaton_error.field is not None:
        result["field"] = automaton_error.field
    emit(result)
    if automaton_error.error_kind == ErrorKind.TIMEOUT:
        return 124
    if automaton_error.error_kind in {
        ErrorKind.PARAM_OUT_OF_RANGE,
        ErrorKind.UNSUPPORTED,
    }:
        return 2
    return 1


def _classify_exception(exc: Exception) -> AutomatonError:
    """Map arbitrary exceptions onto the automaton error taxonomy."""
    if isinstance(exc, AutomatonError):
        return exc
    if isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        return AutomatonError(
            str(exc),
            error_kind=ErrorKind.INVALID_STRUCTURE,
            field="input",
        )
    message: str = str(exc)
    lowered: str = message.lower()
    if "memory" in lowered or "resource exhausted" in lowered:
        return AutomatonError(message, error_kind=ErrorKind.RESOURCE_EXHAUSTED)
    if isinstance(exc, (FloatingPointError, OverflowError, ZeroDivisionError)):
        return AutomatonError(message, error_kind=ErrorKind.NUMERICAL_FAILURE)
    if isinstance(exc, ValueError):
        return AutomatonError(message, error_kind=ErrorKind.INVALID_STRUCTURE)
    return AutomatonError(message, error_kind=ErrorKind.UNKNOWN)


def _deadline_context(seconds: float) -> AbstractContextManager[None]:
    """Return a wall-clock deadline context manager."""
    if seconds <= 0.0 or not hasattr(signal, "SIGALRM"):
        return _NullDeadline()
    return _SignalDeadline(seconds)


class _NullDeadline(AbstractContextManager[None]):
    """No-op deadline context for disabled or unsupported deadlines."""

    def __enter__(self) -> None:
        """Enter the no-op context."""

    def __exit__(self, *exc_info: Any) -> None:
        """Exit the no-op context."""


class _SignalDeadline(AbstractContextManager[None]):
    """SIGALRM-backed wall-clock deadline context."""

    def __init__(self, seconds: float) -> None:
        """Initialize the signal deadline."""
        self.seconds: float = seconds
        self.previous_handler: Any = None
        self.previous_timer: tuple[float, float] = (0.0, 0.0)

    def __enter__(self) -> None:
        """Install the deadline alarm."""
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        self.previous_timer = signal.setitimer(
            signal.ITIMER_REAL,
            self.seconds,
        )
        signal.signal(signal.SIGALRM, self._handle_timeout)

    def __exit__(self, *exc_info: Any) -> None:
        """Restore the previous alarm state."""
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        if self.previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)

    def _handle_timeout(self, signum: int, frame: Any) -> None:
        """Raise a deadline error from the signal handler."""
        del signum, frame
        raise DeadlineExceededError(self.seconds)


def _json_type_for_python_type(python_type: type[Any]) -> str:
    """Map a Python type to a JSON type name."""
    type_names: dict[type[Any], str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }
    return type_names.get(python_type, "object")


def _json_ready(value: Any) -> Any:  # noqa: PLR0911
    """Convert common scientific Python values to JSON-ready values."""
    if value is _MISSING:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except (TypeError, ValueError, AttributeError):
            return str(value)
    return value


def _result_key(
    *,
    experiment: str,
    params: Mapping[str, Any],
    seed: int,
    rheedium_version: str,
) -> str:
    """Derive a deterministic content key for a run request."""
    content: dict[str, Any] = {
        "experiment": experiment,
        "params": params,
        "seed": seed,
        "rheedium_version": rheedium_version,
    }
    encoded: bytes = json.dumps(
        _json_ready(content),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rheedium_version() -> str:
    """Return the installed rheedium package version."""
    try:
        return version("rheedium")
    except PackageNotFoundError:
        return "unknown"


def _jax_backend() -> str:
    """Return the active JAX default backend name."""
    try:
        return jax.default_backend()
    except RuntimeError:
        return "unknown"


def _summary_from_description(description: str) -> str:
    """Extract the first non-empty line from a docstring."""
    for line in description.strip().splitlines():
        stripped: str = line.strip()
        if stripped:
            return stripped
    return ""


def _inherited_flags() -> list[dict[str, str]]:
    """Return metadata for CLI flags provided by every automaton."""
    return [
        {
            "name": "--outdir",
            "type": "string",
            "help": "Directory where artifacts are written.",
        },
        {
            "name": "--seed",
            "type": "integer",
            "help": "Deterministic seed recorded in the result.",
        },
        {
            "name": "--smoke",
            "type": "boolean",
            "help": "Run a tiny end-to-end workload for CI and pre-flight.",
        },
        {
            "name": "--describe",
            "type": "boolean",
            "help": "Emit the parameter and return schema as JSON.",
        },
        {
            "name": "--params",
            "type": "object",
            "help": "Read parameter values from JSON file, stdin, or string.",
        },
        {
            "name": "--validate",
            "type": "boolean",
            "help": "Validate parameters and emit a result without running.",
        },
        {
            "name": "--estimate",
            "type": "boolean",
            "help": "Emit a cost estimate without running the experiment.",
        },
        {
            "name": "--deadline",
            "type": "number",
            "help": "Wall-clock deadline in seconds; 0 disables it.",
        },
        {
            "name": "--json",
            "type": "boolean",
            "help": "Reserve stdout for machine-readable JSON output.",
        },
    ]


__all__: list[str] = [
    "AutomatonError",
    "DESCRIBE_SCHEMA_VERSION",
    "DeadlineExceededError",
    "ErrorKind",
    "ExperimentContext",
    "ExperimentSpec",
    "Param",
    "RESULT_SCHEMA_VERSION",
    "emit",
    "experiment",
]
