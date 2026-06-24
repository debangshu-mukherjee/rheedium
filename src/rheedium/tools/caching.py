"""Persistent XLA compilation cache configuration for rheedium.

Extended Summary
----------------
JAX specializes every compiled executable on the input shapes and dtypes it
is traced with, recompiling whenever a new shape signature appears. The XLA
persistent cache writes those compiled executables to disk so that a later
process -- a re-run, a fresh worker, a CI job -- loads them instead of paying
the compilation cost again.

The cache must be configured *before* the first compilation, so the enabling
call lives at import time in the top-level :mod:`rheedium` package, guarded by
an environment opt-in. This module holds the reusable logic; interactive users
can call :func:`enable_compilation_cache` directly before their first
simulation.

Routine Listings
----------------
:func:`enable_compilation_cache`
    Point JAX's persistent compilation cache at a directory and return it.

Notes
-----
XLA:CPU executables are codegen-specialized on the host CPU feature set
(``avx512`` and similar). A cache directory shared between nodes with
different CPUs can hand a worker an executable its host cannot run, risking
``SIGILL``. By default the cache directory is namespaced per architecture
(:func:`enable_compilation_cache` with ``per_arch=True``) so heterogeneous
clusters never cross executables. The compiled artifacts are also not portable
across jaxlib versions; treat the cache as a local accelerator, not a
distributable binary (use :mod:`rheedium.tools.exporting` for that).
"""

import hashlib
import os
import pathlib
import platform

import jax
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import jaxtyped

_DEFAULT_CACHE_ROOT: str = "~/.cache/rheedium/xla"


def _architecture_tag() -> str:
    """Build a directory tag that discriminates XLA codegen targets."""
    system: str = platform.system()
    machine: str = platform.machine()
    flags: str = ""
    cpuinfo: pathlib.Path = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("flags"):
                flags = line.split(":", 1)[1].strip()
                break
    digest: str = (
        hashlib.sha1(flags.encode()).hexdigest()[:8] if flags else "noflags"
    )
    return f"{system}-{machine}-{digest}"


@jaxtyped(typechecker=beartype)
def enable_compilation_cache(
    cache_dir: Optional[str] = None,
    *,
    per_arch: bool = True,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = 0,
) -> str:
    """Enable JAX's persistent compilation cache and return its directory.

    Configures the XLA persistent cache so compiled executables are written to
    and reloaded from disk across processes. Call before the first compilation
    for the executables of interest to be cached.

    :see: :class:`~.test_caching.TestEnableCompilationCache`

    Parameters
    ----------
    cache_dir : str, optional
        Cache root directory. If ``None``, falls back to the
        ``RHEEDIUM_CACHE_DIR`` environment variable and then to
        ``~/.cache/rheedium/xla``. A leading ``~`` is expanded.
    per_arch : bool, optional
        If ``True`` (default), append an architecture tag (operating system,
        machine, and a hash of the CPU feature flags) to ``cache_dir`` so that
        nodes with different CPUs never load each other's executables.
        Default: True
    min_compile_time_secs : float, optional
        Only cache executables whose compilation took at least this many
        seconds. ``0.0`` caches everything. Default: 0.0
    min_entry_size_bytes : int, optional
        Only cache executables at least this many bytes in size. ``0`` caches
        everything. Default: 0

    Returns
    -------
    resolved_dir : str
        Absolute path of the directory the cache was pointed at (including the
        architecture tag when ``per_arch`` is set).

    Notes
    -----
    1. Resolve the cache root from the argument, environment, or default.
    2. Append the architecture tag from :func:`_architecture_tag` when
       ``per_arch`` is requested.
    3. Create the directory and set the three ``jax_*`` configuration keys.

    The cache is keyed on the computation and its input shapes, so each new
    shape signature still compiles once before being reused. It does not make
    executables portable across machines or jaxlib versions.
    """
    root: str = (
        cache_dir
        or os.environ.get("RHEEDIUM_CACHE_DIR")
        or _DEFAULT_CACHE_ROOT
    )
    resolved: pathlib.Path = pathlib.Path(root).expanduser()
    if per_arch:
        resolved = resolved / _architecture_tag()
    resolved.mkdir(parents=True, exist_ok=True)
    resolved_dir: str = str(resolved)

    jax.config.update("jax_compilation_cache_dir", resolved_dir)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", min_compile_time_secs
    )
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes
    )
    return resolved_dir


__all__: list[str] = [
    "enable_compilation_cache",
]
