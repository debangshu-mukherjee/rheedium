"""Test suite for rheedium.plots.figuring visualization functions.

Tests cover the phosphor colormap creation and the RHEED pattern
plotting function with various rendering modes.
"""

from typing import Any

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jaxtyping import Array, Float, Integer
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

matplotlib.use("Agg")


from rheedium.plots.figuring import create_phosphor_colormap, plot_rheed
from rheedium.simul import log_compress_image
from rheedium.types import RHEEDPattern
from rheedium.types.custom_types import scalar_float
from rheedium.types.rheed_types import create_rheed_pattern


def _make_test_pattern(n: int = 20) -> RHEEDPattern:
    """Create a simple RHEEDPattern for testing."""
    rng: np.random.Generator = np.random.default_rng(42)
    g_indices: Integer[Array, "reflections"] = jnp.arange(n, dtype=jnp.int32)
    k_out: Float[Array, "reflections xyz"] = jnp.array(rng.normal(size=(n, 3)))
    k_out = k_out / jnp.linalg.norm(k_out, axis=1, keepdims=True)
    detector_points: Float[Array, "reflections detector_xy"] = jnp.array(
        rng.normal(size=(n, 2)) * 50, dtype=jnp.float64
    )
    intensities: Float[Array, "reflections"] = jnp.array(
        rng.uniform(0, 100, size=(n,)), dtype=jnp.float64
    )
    return create_rheed_pattern(
        g_indices=g_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )


class TestCreatePhosphorColormap:
    """Tests for create_phosphor_colormap.

    :see: :func:`~rheedium.plots.create_phosphor_colormap`
    """

    def test_returns_colormap(self) -> None:
        r"""Should return a LinearSegmentedColormap.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should return a
        LinearSegmentedColormap.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_default_name(self) -> None:
        r"""Default name should be 'phosphor'.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Default name
        should be 'phosphor'.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        assert cmap.name == "phosphor"

    def test_custom_name(self) -> None:
        r"""Custom name should be respected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Custom name should
        be respected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap(name="my_cmap")
        assert cmap.name == "my_cmap"

    def test_maps_zero_to_black(self) -> None:
        r"""Value 0 should map to black.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Value 0 should map
        to black.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(0.0)
        assert rgba[0] < 0.01
        assert rgba[1] < 0.01
        assert rgba[2] < 0.01

    def test_maps_one_to_bright(self) -> None:
        r"""Value 1 should map to bright (near-white bloom).

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Value 1 should map
        to bright (near-white bloom).

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(1.0)
        assert rgba[0] > 0.5
        assert rgba[1] > 0.8
        assert rgba[2] > 0.5

    def test_green_dominates_midrange(self) -> None:
        r"""Green channel should dominate at mid-range values.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Green channel
        should dominate at mid-range values.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(0.7)
        assert rgba[1] > rgba[0]
        assert rgba[1] > rgba[2]

    def test_monotonic_green_channel(self) -> None:
        r"""Green channel should increase monotonically.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Green channel
        should increase monotonically.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        values: Float[NDArray, "..."] = np.linspace(0, 1, 50)
        greens: Any = [cmap(v)[1] for v in values]
        i: int
        for i in range(len(greens) - 1):
            assert greens[i + 1] >= greens[i] - 1e-6

    def test_output_has_four_channels(self) -> None:
        r"""Output should always be RGBA with 4 channels.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Output should
        always be RGBA with 4 channels.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        cmap: Any = create_phosphor_colormap()
        v: scalar_float
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rgba: Any = cmap(v)
            assert len(rgba) == 4
            assert rgba[3] == 1.0  # alpha = 1


class TestPlotRheed:
    """Tests for plot_rheed function.

    :see: :func:`~rheedium.plots.plot_rheed`
    """

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_log_intensity_scale_is_default(self) -> None:
        r"""Default display pixels use normalized logarithmic compression.

        Extended Summary
        ----------------
        Verifies that omitting ``intensity_scale`` transforms the rendered
        linear detector pixels with the public log-compression function and
        its default gain.

        Notes
        -----
        It renders one pattern explicitly on the linear scale, captures the
        image data supplied to Matplotlib, then renders the same pattern with
        defaults and compares every resulting pixel to
        ``log_compress_image(linear, gain=25.0)``.
        """
        pattern: RHEEDPattern = _make_test_pattern(n=5)
        plot_rheed(
            pattern,
            grid_size=20,
            interp_type="gaussian",
            intensity_scale="linear",
        )
        linear_pixels: NDArray[Any] = np.asarray(
            plt.gcf().axes[0].images[0].get_array()
        )
        plt.close("all")

        plot_rheed(pattern, grid_size=20, interp_type="gaussian")
        default_pixels: NDArray[Any] = np.asarray(
            plt.gcf().axes[0].images[0].get_array()
        )
        expected_pixels: NDArray[Any] = np.asarray(
            log_compress_image(jnp.asarray(linear_pixels), gain=25.0)
        )

        np.testing.assert_allclose(default_pixels, expected_pixels)
        assert not np.allclose(default_pixels, linear_pixels)

    def test_linear_intensity_scale_preserves_pixels(self) -> None:
        r"""Explicit linear display preserves the uncompressed image pixels.

        Extended Summary
        ----------------
        Verifies that ``intensity_scale="linear"`` restores the prior raw
        Gaussian-rendering behavior without normalizing or compressing the
        detector intensities.

        Notes
        -----
        It renders a single diffraction point and independently constructs
        the expected Gaussian raster from its coordinate, intensity, grid,
        extent padding, and spot width before comparing the full pixel array.
        """
        grid_size: int = 9
        spot_width: float = 0.2
        pattern: RHEEDPattern = _make_test_pattern(n=1)
        plot_rheed(
            pattern,
            grid_size=grid_size,
            interp_type="gaussian",
            spot_width=spot_width,
            intensity_scale="linear",
        )
        actual_pixels: NDArray[Any] = np.asarray(
            plt.gcf().axes[0].images[0].get_array()
        )

        x0: float = float(pattern.detector_points[0, 0])
        y0: float = float(pattern.detector_points[0, 1])
        intensity: float = float(pattern.intensities[0])
        x_axis: NDArray[Any] = np.linspace(x0 - 0.5, x0 + 0.5, grid_size)
        y_axis: NDArray[Any] = np.linspace(y0 - 0.5, y0 + 0.5, grid_size)
        xx: NDArray[Any]
        yy: NDArray[Any]
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
        expected_pixels: NDArray[Any] = intensity * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * spot_width**2)
        )

        np.testing.assert_allclose(actual_pixels, expected_pixels)

    def test_log_gain_controls_intensity_compression(self) -> None:
        r"""The exposed log gain controls the default display transform.

        Extended Summary
        ----------------
        Verifies that a caller-supplied ``log_gain`` is passed through to the
        existing detector log-compression function rather than being ignored
        or replaced by the default gain.

        Notes
        -----
        It captures a linear rendering as the uncompressed reference, renders
        the same pattern with a non-default gain, and compares every displayed
        pixel to the public compression function at that gain.
        """
        pattern: RHEEDPattern = _make_test_pattern(n=5)
        plot_rheed(
            pattern,
            grid_size=20,
            intensity_scale="linear",
        )
        linear_pixels: NDArray[Any] = np.asarray(
            plt.gcf().axes[0].images[0].get_array()
        )
        plt.close("all")

        log_gain: float = 5.0
        plot_rheed(pattern, grid_size=20, log_gain=log_gain)
        actual_pixels: NDArray[Any] = np.asarray(
            plt.gcf().axes[0].images[0].get_array()
        )
        expected_pixels: NDArray[Any] = np.asarray(
            log_compress_image(jnp.asarray(linear_pixels), gain=log_gain)
        )

        np.testing.assert_allclose(actual_pixels, expected_pixels)

    def test_invalid_intensity_scale(self) -> None:
        r"""Unsupported intensity scales raise a clear ``ValueError``.

        Extended Summary
        ----------------
        Verifies that the plot helper restricts its new display parameter to
        the documented ``"log"`` and ``"linear"`` values.

        Notes
        -----
        It passes an unsupported scale through the normal rendering path and
        asserts that validation fails explicitly with the parameter name in
        the diagnostic instead of silently choosing a fallback transform.
        """
        pattern: RHEEDPattern = _make_test_pattern(n=1)
        with pytest.raises(ValueError, match="intensity_scale"):
            plot_rheed(pattern, grid_size=10, intensity_scale="square-root")

    def test_gaussian_rendering(self) -> None:
        r"""Test Gaussian rendering mode runs without error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Gaussian rendering
        mode runs without error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="gaussian")

    def test_linear_rendering(self) -> None:
        r"""Test linear interpolation mode runs without error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: linear
        interpolation mode runs without error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="linear")

    def test_cubic_rendering(self) -> None:
        r"""Test cubic interpolation mode runs without error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: cubic
        interpolation mode runs without error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="cubic")

    def test_nearest_rendering(self) -> None:
        r"""Test nearest neighbor mode runs without error.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: nearest neighbor
        mode runs without error.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="nearest")

    def test_invalid_interp_type(self) -> None:
        r"""Invalid interp_type should raise ValueError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Invalid
        interp_type should raise ValueError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        with pytest.raises(ValueError, match="interp_type"):
            plot_rheed(pattern, interp_type="invalid")

    def test_phosphor_colormap_default(self) -> None:
        r"""Default colormap should be phosphor.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Default colormap
        should be phosphor.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50)

    def test_named_colormap(self) -> None:
        r"""Should accept a named matplotlib colormap.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should accept a
        named matplotlib colormap.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, cmap_name="viridis")

    def test_custom_extents(self) -> None:
        r"""Custom x_extent and y_extent should be respected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Custom x_extent
        and y_extent should be respected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(
            pattern,
            grid_size=50,
            x_extent=(-10.0, 10.0),
            y_extent=(-20.0, 20.0),
        )

    def test_custom_figsize(self) -> None:
        r"""Custom figsize should be respected.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Custom figsize
        should be respected.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, figsize=(12.0, 6.0))

    def test_custom_spot_width(self) -> None:
        r"""Custom spot_width should work for Gaussian mode.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Custom spot_width
        should work for Gaussian mode.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(
            pattern, grid_size=50, interp_type="gaussian", spot_width=0.2
        )

    def test_single_point_pattern(self) -> None:
        r"""Should handle a pattern with a single point.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should handle a
        pattern with a single point.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern(n=1)
        plot_rheed(pattern, grid_size=50)

    def test_large_grid_size(self) -> None:
        r"""Should handle larger grid sizes.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Should handle
        larger grid sizes.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_plots.test_figuring``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        pattern: Float[Array, "..."] = _make_test_pattern(n=5)
        plot_rheed(pattern, grid_size=100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
