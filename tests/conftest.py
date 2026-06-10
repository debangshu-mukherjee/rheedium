"""Pytest configuration for rheedium test suite.

Extended Summary
----------------
Manages JAX memory during test runs by:

1. Dynamically computing the ``pytest-xdist`` worker count from
   available system memory. Running ``pytest -n auto`` triggers the
   ``pytest_xdist_auto_num_workers`` hook, which computes
   ``int(available_ram_gb * AVAILABLE_MEM_FRACTION // MEM_PER_WORKER_GB)``.
   An explicit ``-n 16`` overrides this calculation.

2. Exposing 8 virtual CPU devices via
   ``XLA_FLAGS=--xla_force_host_platform_device_count=8``.
   This enables meaningful ``pmap`` and multi-device sharding
   tests on the CPU backend.

3. Disabling JAX GPU memory pre-allocation via
   ``XLA_PYTHON_CLIENT_PREALLOCATE=false``. This must be set before
   ``import jax``; the ``os.environ`` call at module level ensures
   this regardless of import order.

4. Clearing the JIT compilation cache after each test via
   ``jax.clear_caches()``. Without this, each unique array shape
   compiled through a ``@jax.jit`` function accumulates a cached XLA
   binary that is never evicted, causing monotonic memory growth.

5. Detecting per-test memory leaks by comparing RSS before and after
   each test. If a single test increases RSS by more than
   ``MEM_LEAK_THRESHOLD_GB``, the test is failed with a diagnostic
   message.

Routine Listings
----------------
:func:`pytest_xdist_auto_num_workers`
    Compute xdist worker count from available system memory.
:func:`manage_jax_memory`
    Autouse fixture that clears JIT caches and detects leaks.

Notes
-----
Memory detection is cross-platform (Linux, macOS, Windows). On
Linux, available memory is read from ``/proc/meminfo`` (includes
reclaimable buff/cache). On macOS, ``vm_stat`` is parsed. On
Windows, ``GlobalMemoryStatusEx`` is called via ctypes. Unrecognised
platforms fall back to a single-worker conservative default.

Available memory is queried once at session start. The figure is a
snapshot — as workers launch and caches grow it becomes stale.
``MEM_PER_WORKER_GB`` and ``MEM_LEAK_THRESHOLD_GB`` should be treated
as living configuration and tuned as the test suite grows.
"""

import os
from collections.abc import Generator
from typing import Final

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import platform  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402

MEM_PER_WORKER_GB: Final[int] = 5
MEM_LEAK_THRESHOLD_GB: Final[float] = 0.5
AVAILABLE_MEM_FRACTION: Final[float] = 0.80


def _available_ram_gb() -> float:
    """Return available system memory in GB, cross-platform.

    Returns
    -------
    available_gb : float
        Memory that can be allocated without swapping.
        On Linux this includes reclaimable buff/cache.
        Falls back to ``float(MEM_PER_WORKER_GB)`` on unrecognised
        platforms or when the target field is not found, so that
        ``pytest_xdist_auto_num_workers`` yields exactly one worker.

    Notes
    -----
    1. On Linux, parse ``/proc/meminfo`` for the ``MemAvailable``
       field, which accounts for free memory plus reclaimable
       page cache and slab.
    2. On macOS, run ``vm_stat`` and sum free, inactive, and
       purgeable pages, then multiply by the Mach page size.
    3. On Windows, call ``GlobalMemoryStatusEx`` via ctypes and
       read ``ullAvailPhys``.
    4. On unrecognised platforms, or if the target field is absent,
       return ``float(MEM_PER_WORKER_GB)`` as a conservative fallback.
    """
    system: str = platform.system()
    if system == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
        return float(MEM_PER_WORKER_GB)
    elif system == "Darwin":
        import subprocess  # noqa: PLC0415

        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        page_size: int = 16384
        pages_free: int = 0
        pages_inactive: int = 0
        pages_purgeable: int = 0
        for line in result.stdout.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free:" in line:
                pages_free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split()[-1].rstrip("."))
            elif "Pages purgeable:" in line:
                pages_purgeable = int(line.split()[-1].rstrip("."))
        available_bytes: int = (
            pages_free + pages_inactive + pages_purgeable
        ) * page_size
        return available_bytes / 1024**3
    elif system == "Windows":
        import ctypes  # noqa: PLC0415

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem: MEMORYSTATUSEX = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullAvailPhys / 1024**3
    return float(MEM_PER_WORKER_GB)


def _worker_rss_gb() -> float:
    """Return current process RSS in GB, cross-platform.

    Returns
    -------
    rss_gb : float
        Resident set size of the current process in GB.
        Returns 0.0 on unrecognised platforms or when the target
        field is absent, so the leak check in ``manage_jax_memory``
        never triggers a false positive.

    Notes
    -----
    1. On Linux, parse ``/proc/<pid>/status`` for the ``VmRSS``
       field (reported in kB by the kernel).
    2. On macOS, run ``ps -o rss= -p <pid>`` which reports RSS
       in kB.
    3. On Windows, call ``GetProcessMemoryInfo`` via ctypes and
       read ``WorkingSetSize`` (reported in bytes).
    4. On unrecognised platforms, or if the target field is absent,
       return 0.0.
    """
    system: str = platform.system()
    pid: int = os.getpid()
    if system == "Linux":
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024**2
        return 0.0
    elif system == "Darwin":
        import subprocess  # noqa: PLC0415

        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) / 1024**2
    elif system == "Windows":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        process: int = ctypes.windll.kernel32.GetCurrentProcess()

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc: PROCESS_MEMORY_COUNTERS = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / 1024**3
    return 0.0


def pytest_xdist_auto_num_workers() -> int:
    """Compute xdist worker count from available system memory.

    Returns
    -------
    n_workers : int
        ``available_ram_gb * AVAILABLE_MEM_FRACTION
        // MEM_PER_WORKER_GB``, minimum 1.

    Notes
    -----
    1. Query available system memory via ``_available_ram_gb``.
    2. Multiply by ``AVAILABLE_MEM_FRACTION`` to reserve headroom
       for the OS and other processes.
    3. Integer-divide by ``MEM_PER_WORKER_GB`` to get the maximum
       number of workers that fit in memory.
    4. Clamp to a minimum of 1.

    This hook is called by ``pytest-xdist`` when ``-n auto`` is
    passed. An explicit ``-n <N>`` bypasses it entirely. The available
    memory figure is a snapshot taken at session start; it does not
    account for memory consumed as workers launch and caches grow.
    ``MEM_PER_WORKER_GB`` should be tuned as the test suite grows.
    """
    available_gb: float = _available_ram_gb() * AVAILABLE_MEM_FRACTION
    return max(1, int(available_gb // MEM_PER_WORKER_GB))


@pytest.fixture(autouse=True)
def manage_jax_memory() -> Generator[None, None, None]:
    """Clear JAX JIT caches and detect memory leaks per test.

    Yields
    ------
    None
        Control is yielded to the test body.

    Raises
    ------
    pytest.fail
        If the test increases the worker process RSS by more than
        ``MEM_LEAK_THRESHOLD_GB``.

    Notes
    -----
    1. Record baseline RSS before the test runs.
    2. Yield control to the test.
    3. Call ``jax.clear_caches()`` to evict all JIT-compiled XLA
       binaries from the in-process cache.
    4. Measure RSS again and compute the delta from baseline.
    5. If the delta exceeds ``MEM_LEAK_THRESHOLD_GB``, fail the
       test with a diagnostic message identifying the leak size.

    The delta-based comparison avoids false positives from the
    worker process baseline (Python interpreter, JAX runtime,
    imported libraries), which can consume 2-4 GB before any
    test runs.
    """
    baseline_gb: float = _worker_rss_gb()
    yield
    jax.clear_caches()
    delta_gb: float = _worker_rss_gb() - baseline_gb
    if delta_gb > MEM_LEAK_THRESHOLD_GB:
        pytest.fail(
            f"Test leaked {delta_gb:.1f} GB "
            f"(limit per test: {MEM_LEAK_THRESHOLD_GB} GB)"
        )
