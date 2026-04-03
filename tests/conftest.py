"""Pytest configuration for rheedium test suite.

Extended Summary
----------------
Manages JAX memory during test runs by:

1. Dynamically computing the xdist worker count from available
   system memory (``-n auto`` → ``available_ram // MEM_PER_WORKER_GB``).
2. Disabling JAX GPU memory pre-allocation.
3. Clearing the JIT compilation cache after each test.
4. Failing any test whose worker process exceeds the per-worker
   memory budget.

Notes
-----
``XLA_PYTHON_CLIENT_PREALLOCATE`` must be set before JAX is imported.
The ``os.environ`` call at module level ensures this regardless of
import order. Use ``pytest -n auto`` to activate dynamic worker
calculation; an explicit ``-n 16`` overrides it.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import platform  # noqa: E402
import shutil  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402

MEM_PER_WORKER_GB = 10
AVAILABLE_MEM_FRACTION = 0.80


def _available_ram_gb() -> float:
    """Return available system memory in GB, cross-platform.

    Returns the memory that can be allocated without swapping.
    On Linux this includes reclaimable buff/cache.
    """
    system = platform.system()
    if system == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    elif system == "Darwin":
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        page_size = 16384
        pages_free = 0
        pages_inactive = 0
        pages_purgeable = 0
        for line in result.stdout.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free:" in line:
                pages_free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                pages_inactive = int(line.split()[-1].rstrip("."))
            elif "Pages purgeable:" in line:
                pages_purgeable = int(line.split()[-1].rstrip("."))
        available_bytes = (
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

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullAvailPhys / 1024**3
    return shutil.disk_usage("/").total / 1024**3


def _worker_rss_gb() -> float:
    """Return current process RSS in GB, cross-platform."""
    system = platform.system()
    pid = os.getpid()
    if system == "Linux":
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024**2
    elif system == "Darwin":
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) / 1024**2
    elif system == "Windows":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        process = ctypes.windll.kernel32.GetCurrentProcess()

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

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / 1024**3
    return 0.0


def pytest_xdist_auto_num_workers():
    """Compute worker count from available system memory.

    Returns
    -------
    n_workers : int
        ``available_ram_gb // MEM_PER_WORKER_GB``, minimum 1.
    """
    available_gb = _available_ram_gb() * AVAILABLE_MEM_FRACTION
    return max(1, int(available_gb // MEM_PER_WORKER_GB))


@pytest.fixture(autouse=True)
def manage_jax_memory():
    """Clear JAX JIT caches and check RSS after each test."""
    yield
    jax.clear_caches()
    rss_gb = _worker_rss_gb()
    if rss_gb > MEM_PER_WORKER_GB:
        pytest.fail(
            f"Worker RSS reached {rss_gb:.1f} GB "
            f"(limit: {MEM_PER_WORKER_GB} GB)"
        )
