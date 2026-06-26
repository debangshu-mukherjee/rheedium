"""Tests for the ahead-of-time export helpers in rheedium.tools."""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

from rheedium.tools.exporting import (
    ExportError,
    bucketize_grid,
    deserialize_exported,
    export_forward,
    serialize_exported,
)


def _sum_of_squares(x: Float[Array, " n"]) -> Float[Array, ""]:
    """Shape-polymorphic reduction used as an exportable forward stand-in."""
    return jnp.sum(x**2)


class TestExportForward(chex.TestCase):
    """Tests for :func:`export_forward`.

    :see: :class:`~rheedium.tools.ExportError`
    :see: :func:`~rheedium.tools.export_forward`
    """

    def test_symbolic_axis_round_trips(self) -> None:
        r"""One artifact serves several sizes of a symbolic leading axis.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: One artifact
        serves several sizes of a symbolic leading axis.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        (n,) = jax.export.symbolic_shape("n")
        exported: jax.export.Exported = export_forward(
            _sum_of_squares, jax.ShapeDtypeStruct((n,), jnp.float64)
        )
        size: int
        for size in (3, 8, 21):
            values: Float[Array, " n"] = jnp.arange(size, dtype=jnp.float64)
            result: Float[Array, ""] = exported.call(values)
            chex.assert_trees_all_close(result, jnp.sum(values**2))

    def test_host_callback_raises_export_error(self) -> None:
        r"""Equinox runtime checks (host callbacks) raise ``ExportError``.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Equinox runtime
        checks (host callbacks) raise ``ExportError``. Existing context from
        the original test prose: Mirrors the ``checked_*`` simulators: the
        check operates on a concrete-shaped quantity, so export reaches the
        unserializable host-callback rather than a symbolic-shape lowering
        error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def checked(x: Float[Array, "4"]) -> Float[Array, "4"]:
            x = eqx.error_if(x, jnp.any(x < -1e30), "unreachable")
            return x * 2.0

        with pytest.raises(ExportError):
            export_forward(checked, jax.ShapeDtypeStruct((4,), jnp.float64))

    def test_symbolic_fft_raises_export_error(self) -> None:
        r"""A symbolic FFT transform size raises ``ExportError``.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A symbolic FFT
        transform size raises ``ExportError``.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def fft_forward(x: Float[Array, "h w"]) -> Float[Array, "h w"]:
            return jnp.abs(jnp.fft.fft2(x))

        h, w = jax.export.symbolic_shape("h, w")
        with pytest.raises(ExportError):
            export_forward(
                fft_forward, jax.ShapeDtypeStruct((h, w), jnp.float64)
            )

    def test_concrete_fft_exports(self) -> None:
        r"""A concrete FFT grid exports and runs (the bucketing path).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A concrete FFT
        grid exports and runs (the bucketing path).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """

        def fft_forward(x: Float[Array, "h w"]) -> Float[Array, "h w"]:
            return jnp.abs(jnp.fft.fft2(x))

        exported: jax.export.Exported = export_forward(
            fft_forward, jax.ShapeDtypeStruct((16, 16), jnp.float64)
        )
        result: Float[Array, "h w"] = exported.call(
            jnp.ones((16, 16), jnp.float64)
        )
        chex.assert_shape(result, (16, 16))
        chex.assert_tree_all_finite(result)


class TestSerializeRoundTrip(chex.TestCase):
    """Tests for :func:`serialize_exported` / :func:`deserialize_exported`.

    :see: :func:`~rheedium.tools.deserialize_exported`
    :see: :func:`~rheedium.tools.serialize_exported`
    """

    def test_serialize_then_deserialize(self) -> None:
        r"""A serialized artifact reloads and computes identical results.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A serialized
        artifact reloads and computes identical results.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with tolerance-aware closeness
        assertions, which is appropriate for floating-point JAX arrays.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        (n,) = jax.export.symbolic_shape("n")
        exported: jax.export.Exported = export_forward(
            _sum_of_squares, jax.ShapeDtypeStruct((n,), jnp.float64)
        )
        blob: bytes = serialize_exported(exported)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        reloaded: jax.export.Exported = deserialize_exported(blob)
        values: Float[Array, " n"] = jnp.arange(6, dtype=jnp.float64)
        chex.assert_trees_all_close(reloaded.call(values), jnp.sum(values**2))


class TestBucketizeGrid(chex.TestCase):
    """Tests for :func:`bucketize_grid`.

    :see: :func:`~rheedium.tools.bucketize_grid`
    """

    @parameterized.parameters(
        ((200, 200), (256, 256)),
        ((300, 600), (512, 1024)),
        ((512, 512), (512, 512)),
        ((1, 1024), (256, 1024)),
    )
    def test_snaps_up_to_nearest_bucket(
        self,
        request_hw: tuple[int, int],
        expected: tuple[int, int],
    ) -> None:
        r"""Each dimension snaps to the smallest bucket that fits it.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Each dimension
        snaps to the smallest bucket that fits it.

        Notes
        -----
        It receives parametrized or fixture-provided inputs named
        ``request_hw``, ``expected``, so the documented behavior is checked
        across the cases supplied by pytest, Chex, Hypothesis, or absl.

        It uses the declared parameter table to exercise multiple named
        examples with the same assertion logic.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        assert bucketize_grid(*request_hw, (1024, 256, 512)) == expected

    def test_exceeds_largest_bucket_raises(self) -> None:
        r"""A request larger than every bucket raises ``ValueError``.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A request larger
        than every bucket raises ``ValueError``.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        with pytest.raises(ValueError, match="exceeds the largest bucket"):
            bucketize_grid(2048, 256, (256, 512, 1024))

    def test_empty_buckets_raises(self) -> None:
        r"""An empty bucket list raises ``ValueError``.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: An empty bucket
        list raises ``ValueError``.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_tools.test_exporting``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        with pytest.raises(ValueError, match="must be non-empty"):
            bucketize_grid(256, 256, ())
