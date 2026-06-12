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
    """Tests for create_phosphor_colormap."""

    def test_returns_colormap(self) -> None:
        """Should return a LinearSegmentedColormap."""
        cmap: Any = create_phosphor_colormap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_default_name(self) -> None:
        """Default name should be 'phosphor'."""
        cmap: Any = create_phosphor_colormap()
        assert cmap.name == "phosphor"

    def test_custom_name(self) -> None:
        """Custom name should be respected."""
        cmap: Any = create_phosphor_colormap(name="my_cmap")
        assert cmap.name == "my_cmap"

    def test_maps_zero_to_black(self) -> None:
        """Value 0 should map to black."""
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(0.0)
        assert rgba[0] < 0.01
        assert rgba[1] < 0.01
        assert rgba[2] < 0.01

    def test_maps_one_to_bright(self) -> None:
        """Value 1 should map to bright (near-white bloom)."""
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(1.0)
        assert rgba[0] > 0.5
        assert rgba[1] > 0.8
        assert rgba[2] > 0.5

    def test_green_dominates_midrange(self) -> None:
        """Green channel should dominate at mid-range values."""
        cmap: Any = create_phosphor_colormap()
        rgba: Any = cmap(0.7)
        assert rgba[1] > rgba[0]
        assert rgba[1] > rgba[2]

    def test_monotonic_green_channel(self) -> None:
        """Green channel should increase monotonically."""
        cmap: Any = create_phosphor_colormap()
        values: Float[NDArray, "..."] = np.linspace(0, 1, 50)
        greens: Any = [cmap(v)[1] for v in values]
        i: int
        for i in range(len(greens) - 1):
            assert greens[i + 1] >= greens[i] - 1e-6

    def test_output_has_four_channels(self) -> None:
        """Output should always be RGBA with 4 channels."""
        cmap: Any = create_phosphor_colormap()
        v: scalar_float
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rgba: Any = cmap(v)
            assert len(rgba) == 4
            assert rgba[3] == 1.0  # alpha = 1


class TestPlotRheed:
    """Tests for plot_rheed function."""

    def teardown_method(self) -> None:
        """Clean up matplotlib figures after each test."""
        plt.close("all")

    def test_gaussian_rendering(self) -> None:
        """Test Gaussian rendering mode runs without error."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="gaussian")

    def test_linear_rendering(self) -> None:
        """Test linear interpolation mode runs without error."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="linear")

    def test_cubic_rendering(self) -> None:
        """Test cubic interpolation mode runs without error."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="cubic")

    def test_nearest_rendering(self) -> None:
        """Test nearest neighbor mode runs without error."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, interp_type="nearest")

    def test_invalid_interp_type(self) -> None:
        """Invalid interp_type should raise ValueError."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        with pytest.raises(ValueError, match="interp_type"):
            plot_rheed(pattern, interp_type="invalid")

    def test_phosphor_colormap_default(self) -> None:
        """Default colormap should be phosphor."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50)

    def test_named_colormap(self) -> None:
        """Should accept a named matplotlib colormap."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, cmap_name="viridis")

    def test_custom_extents(self) -> None:
        """Custom x_extent and y_extent should be respected."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(
            pattern,
            grid_size=50,
            x_extent=(-10.0, 10.0),
            y_extent=(-20.0, 20.0),
        )

    def test_custom_figsize(self) -> None:
        """Custom figsize should be respected."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(pattern, grid_size=50, figsize=(12.0, 6.0))

    def test_custom_spot_width(self) -> None:
        """Custom spot_width should work for Gaussian mode."""
        pattern: Float[Array, "..."] = _make_test_pattern()
        plot_rheed(
            pattern, grid_size=50, interp_type="gaussian", spot_width=0.2
        )

    def test_single_point_pattern(self) -> None:
        """Should handle a pattern with a single point."""
        pattern: Float[Array, "..."] = _make_test_pattern(n=1)
        plot_rheed(pattern, grid_size=50)

    def test_large_grid_size(self) -> None:
        """Should handle larger grid sizes."""
        pattern: Float[Array, "..."] = _make_test_pattern(n=5)
        plot_rheed(pattern, grid_size=100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
