"""JAX-based RHEED simulation and analysis package.

Extended Summary
----------------
Rheedium provides a comprehensive suite of tools for simulating and analyzing
Reflection High-Energy Electron Diffraction (RHEED) patterns. Built on JAX,
it offers differentiable simulations suitable for optimization and machine
learning applications in materials science and surface physics. Distributed
computing is supported through device mesh parallelism.

Routine Listings
----------------
:func:`init_distributed`
    Initialize JAX multi-host execution (idempotent and guarded).
:mod:`audit`
    Benchmarking and realism-audit utilities for detector images.
:mod:`inout`
    Data input/output operations for crystal structures and RHEED images.
:mod:`harness`
    Process-boundary helpers for agent-runnable automaton scripts.
:mod:`plots`
    Visualization tools for RHEED patterns and crystal structures.
:mod:`procs`
    Differentiable procedural models and preprocessing utilities.
:mod:`recon`
    Inverse-problem utilities for inferring structure from RHEED data.
:mod:`simul`
    RHEED pattern simulation using kinematic diffraction theory.
:mod:`tools`
    Utility tools for parallel processing and distributed computing.
:mod:`types`
    Custom type definitions and data structures for JAX compatibility.
:mod:`ucell`
    Unit cell and crystallographic computation utilities.

Examples
--------
>>> import rheedium as rh
>>> crystal = rh.inout.parse_cif("structure.cif")
>>> pattern = rh.simul.ewald_simulator(crystal)
>>> rh.plots.plot_rheed(pattern)

Notes
-----
All computations are JAX-compatible and support automatic differentiation for
gradient-based optimization of crystal structures and simulation parameters.
64-bit precision is enabled at import.

Multi-node distributed execution is opt-in. For batch (e.g. SLURM) jobs, set
``RHEEDIUM_DISTRIBUTED=1`` before launching with ``srun`` or equivalent and the
package initializes the cluster on import. An optional
``RHEEDIUM_COORDINATOR_ADDRESS`` overrides automatic SLURM coordinator
detection. For interactive or non-batch use, call :func:`init_distributed`
directly rather than relying on the environment trigger.

Initialization is guarded: ``jax.distributed.initialize`` runs at most once and
degrades to a warning rather than an error if the runtime is already
initialized. It is nonetheless a *collective* operation that blocks until every
process in the job connects, so it should be reached by all ranks, not fired
from a transitive single-rank import.

The XLA persistent compilation cache is opt-in. Set ``RHEEDIUM_CACHE_DIR`` (or
``RHEEDIUM_COMPILATION_CACHE=1`` to use the default location) before import and
the package enables it here -- before any compilation -- via
:func:`~rheedium.tools.enable_compilation_cache`. Interactive users can call
that function directly instead.

Runtime input validation (``equinox.error_if`` in the factory functions and
``checked_*`` simulators) is **on by default**. Equinox resolves the
``EQX_ON_ERROR`` default once, at import, so it cannot be toggled afterwards.
For trusted, pre-validated data -- or to export a forward model, whose host
callbacks are otherwise unserializable -- set
``RHEEDIUM_DISABLE_RUNTIME_CHECKS=1`` before import and the package sets
``EQX_ON_ERROR=off`` here, ahead of the first equinox import. An explicit
``EQX_ON_ERROR`` is always respected. The default leaves every check active.
"""

import os
import warnings
from importlib.metadata import version

_RHEEDIUM_XLA_FLAGS: tuple[str, ...] = ("--xla_cpu_multi_thread_eigen=true",)
_existing_xla: str = os.environ.get("XLA_FLAGS", "")
_xla_parts: list[str] = _existing_xla.split()
_xla_keys: set[str] = {_part.split("=", 1)[0] for _part in _xla_parts}
for _flag in _RHEEDIUM_XLA_FLAGS:
    _flag_key: str = _flag.split("=", 1)[0]
    if _flag_key not in _xla_keys:
        _xla_parts.append(_flag)
        _xla_keys.add(_flag_key)
os.environ["XLA_FLAGS"] = " ".join(_xla_parts).strip()

if os.environ.get("RHEEDIUM_DISABLE_RUNTIME_CHECKS", "0") == "1":
    os.environ.setdefault("EQX_ON_ERROR", "off")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

if os.environ.get("RHEEDIUM_COMPILATION_CACHE", "0") == "1" or os.environ.get(
    "RHEEDIUM_CACHE_DIR"
):
    from .tools.caching import enable_compilation_cache  # noqa: E402

    enable_compilation_cache()


def init_distributed(
    coordinator_address: str | None = None,
    *,
    force: bool = False,
) -> bool:
    """Initialize JAX multi-host execution, idempotently and safely.

    Extended Summary
    ----------------
    Wraps ``jax.distributed.initialize`` with two guards bare initialization
    lacks: it is a no-op when the runtime is already initialized (so a
    re-import, or a user who already called ``jax.distributed.initialize``,
    does not raise), and it degrades to a :class:`RuntimeWarning` instead of
    crashing if initialization fails.

    ``jax.distributed.initialize`` is a *collective* operation -- it blocks
    until every process in the job has connected -- so call it from a context
    all ranks reach, not from a transitive import on a single rank.

    Parameters
    ----------
    coordinator_address : str | None, optional
        Coordinator ``host:port``. If ``None``, falls back to
        ``RHEEDIUM_COORDINATOR_ADDRESS`` and then to automatic SLURM detection.
    force : bool, optional
        If ``True``, attempt initialization even when the environment opt-in
        (``RHEEDIUM_DISTRIBUTED`` / ``SLURM_NTASKS``) is not satisfied. Use for
        interactive multi-host sessions launched without the batch env vars.

    Returns
    -------
    bool
        ``True`` if the runtime is initialized on return, ``False`` otherwise.
    """
    if not force:
        if os.environ.get("RHEEDIUM_DISTRIBUTED", "0") != "1":
            return False
        if int(os.environ.get("SLURM_NTASKS", "1")) <= 1:
            return False

    is_initialized = getattr(jax.distributed, "is_initialized", None)
    if callable(is_initialized) and is_initialized():
        return True

    address: str | None = coordinator_address or os.environ.get(
        "RHEEDIUM_COORDINATOR_ADDRESS"
    )
    try:
        if address is not None:
            jax.distributed.initialize(coordinator_address=address)
        else:
            jax.distributed.initialize()
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"rheedium: jax.distributed.initialize() skipped ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


init_distributed()

from . import (  # noqa: E402, I001
    audit,
    harness,
    inout,
    plots,
    procs,
    recon,
    simul,
    tools,
    types,
    ucell,
)

__version__: str = version("rheedium")

__all__: list[str] = [
    "init_distributed",
    "audit",
    "harness",
    "inout",
    "plots",
    "procs",
    "recon",
    "simul",
    "tools",
    "types",
    "ucell",
]
