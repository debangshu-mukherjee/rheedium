"""Tests for the persistent compilation cache helper in rheedium.tools."""

import os
import tempfile

import chex
import jax

from rheedium.tools.caching import enable_compilation_cache

_CACHE_KEYS: tuple[str, ...] = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
)


class TestEnableCompilationCache(chex.TestCase):
    """Tests for :func:`enable_compilation_cache`.

    :see: :func:`~rheedium.tools.enable_compilation_cache`
    """

    def setUp(self) -> None:
        """Snapshot the cache-related JAX config before each test."""
        super().setUp()
        self._saved: dict[str, object] = {
            key: getattr(jax.config, key, None) for key in _CACHE_KEYS
        }
        self._saved_env: str | None = os.environ.pop(
            "RHEEDIUM_CACHE_DIR", None
        )

    def tearDown(self) -> None:
        """Restore the JAX config and environment after each test."""
        for key, value in self._saved.items():
            if value is not None:
                jax.config.update(key, value)
        if self._saved_env is not None:
            os.environ["RHEEDIUM_CACHE_DIR"] = self._saved_env
        super().tearDown()

    def test_sets_config_and_creates_dir(self) -> None:
        r"""The cache dir is created and registered with JAX config.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: The cache dir is
        created and registered with JAX config.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_caching``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with tempfile.TemporaryDirectory() as tmp:
            resolved: str = enable_compilation_cache(tmp, per_arch=False)
            assert resolved == os.path.abspath(tmp)
            assert os.path.isdir(resolved)
            assert jax.config.jax_compilation_cache_dir == resolved

    def test_per_arch_namespaces_subdirectory(self) -> None:
        r"""``per_arch`` appends an architecture tag below the root.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: ``per_arch``
        appends an architecture tag below the root.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_caching``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with tempfile.TemporaryDirectory() as tmp:
            resolved: str = enable_compilation_cache(tmp, per_arch=True)
            assert resolved != os.path.abspath(tmp)
            assert os.path.dirname(resolved) == os.path.abspath(tmp)
            assert os.path.isdir(resolved)

    def test_threshold_arguments_registered(self) -> None:
        r"""Compile-time and entry-size thresholds reach the JAX config.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Compile-time and
        entry-size thresholds reach the JAX config.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_caching``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with tempfile.TemporaryDirectory() as tmp:
            enable_compilation_cache(
                tmp,
                per_arch=False,
                min_compile_time_secs=2.5,
                min_entry_size_bytes=128,
            )
            assert jax.config.jax_persistent_cache_min_compile_time_secs == 2.5
            assert jax.config.jax_persistent_cache_min_entry_size_bytes == 128

    def test_environment_fallback(self) -> None:
        r"""``RHEEDIUM_CACHE_DIR`` is used when no argument is given.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case:
        ``RHEEDIUM_CACHE_DIR`` is used when no argument is given.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_caching``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["RHEEDIUM_CACHE_DIR"] = tmp
            resolved: str = enable_compilation_cache(per_arch=False)
            assert resolved == os.path.abspath(tmp)
