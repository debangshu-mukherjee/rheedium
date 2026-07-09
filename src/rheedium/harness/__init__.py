"""Automaton harness for agent-runnable rheedium experiments.

Extended Summary
----------------
This module provides the small process-boundary contract used by scripts in
``automatons/``. An automaton declares its parameters with :class:`Param`,
decorates one ``main(args, ctx)`` function with :func:`experiment`, and
receives consistent CLI parsing, JSON input, validation, artifact helpers, and
final-line JSON result emission.

Routine Listings
----------------
:class:`Param`
    Declarative parameter metadata used for CLI and ``--describe`` schemas.
:class:`ExperimentContext`
    Per-run output directory, random seed, and artifact writer helpers.
:class:`AutomatonError`
    Structured boundary error with an agent-readable ``error_kind``.
:func:`experiment`
    Decorator that turns ``main(args, ctx)`` into a runnable automaton.
:func:`emit`
    Write one JSON result object to stdout.

Notes
-----
The harness deliberately uses only the standard library plus dependencies that
already ship with :mod:`rheedium`. Automaton scripts can therefore declare a
single PEP 723 dependency on ``rheedium`` while still exposing a machine-facing
interface. The concrete implementation lives in
:mod:`rheedium.harness.automaton`; this package re-exports its public surface.
"""

from .automaton import (
    DESCRIBE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    AutomatonError,
    DeadlineExceededError,
    ErrorKind,
    ExperimentContext,
    ExperimentSpec,
    Param,
    emit,
    experiment,
)

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
