import os
import tempfile
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import colormaps
from pymatgen.core import Lattice, Structure

from rheedium.io import parse_cif_to_jax


class PhosphorColormap:
    """
    A class that creates and manages a custom colormap simulating a phosphor screen appearance.
    The colormap transitions from black through a bright phosphorescent green,
    with a slight white bloom at maximum intensity.
    """

    def __init__(self, name: Optional[str] = "phosphor") -> None:
        """
        Initialize the PhosphorColormap with a given name.

        Parameters
        ----------
        name : str, optional
            Name for the colormap (default is "phosphor")
        """
        self.name = name or "phosphor"
        self.colors: List[Tuple[float, Tuple[float, float, float]]] = [
            (0.0, (0.0, 0.0, 0.0)),  # Black at minimum
            (0.4, (0.0, 0.05, 0.0)),  # Very dark green
            (0.7, (0.15, 0.85, 0.15)),  # Bright phosphorescent green
            (0.9, (0.45, 0.95, 0.45)),  # Lighter green
            (1.0, (0.8, 1.0, 0.8)),  # Slight white bloom at maximum
        ]
        self._cmap: Optional[LinearSegmentedColormap] = None

    def _create_segments(self) -> dict:
        """
        Create color segments for the colormap.

        Returns
        -------
        dict
            Dictionary containing the RGB segments
        """
        positions = [x[0] for x in self.colors]
        rgb_values = [x[1] for x in self.colors]

        red = [(pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_values, strict=False)]
        green = [(pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_values, strict=False)]
        blue = [(pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_values, strict=False)]

        return {"red": red, "green": green, "blue": blue}

    def create(self) -> LinearSegmentedColormap:
        """
        Create and return the phosphor screen colormap.

        Returns
        -------
        LinearSegmentedColormap
            The created colormap
        """
        segments = self._create_segments()
        self._cmap = LinearSegmentedColormap(self.name, segments)
        return self._cmap

    @property
    def colormap(self) -> LinearSegmentedColormap:
        """
        Get the colormap, creating it if it doesn't exist.

        Returns
        -------
        LinearSegmentedColormap
            The phosphor screen colormap
        """
        if self._cmap is None:
            self._cmap = self.create()
        return self._cmap


class TestPhosphorColormap:
    """Test suite for PhosphorColormap class."""

    @pytest.fixture
    def phosphor_cmap(self):
        """Fixture providing a PhosphorColormap instance."""
        return PhosphorColormap()

    def test_initialization(self) -> None:
        """Test if the class initializes correctly with default and custom names."""
        cmap = PhosphorColormap()
        assert cmap.name == "phosphor"

        custom_name = "test_phosphor"
        cmap = PhosphorColormap(name=custom_name)
        assert cmap.name == custom_name

    def test_colormap_creation(self, phosphor_cmap) -> None:
        """Test if the colormap is created correctly."""
        cmap = phosphor_cmap.create()
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.name == "phosphor"

    def test_colormap_property(self, phosphor_cmap) -> None:
        """Test if the colormap property works correctly."""
        # First access should create the colormap
        cmap1 = phosphor_cmap.colormap
        assert isinstance(cmap1, LinearSegmentedColormap)

        # Second access should return the same colormap
        cmap2 = phosphor_cmap.colormap
        assert cmap1 is cmap2

    def test_color_range(self, phosphor_cmap) -> None:
        """Test if the colormap properly handles the full range of values [0,1]."""
        cmap = phosphor_cmap.colormap

        # Test minimum value (black)
        color_min = cmap(0.0)
        np.testing.assert_array_almost_equal(color_min[:3], [0.0, 0.0, 0.0])

        # Test maximum value (slight white bloom)
        color_max = cmap(1.0)
        np.testing.assert_array_almost_equal(color_max[:3], [0.8, 1.0, 0.8])

    def test_green_dominance(self, phosphor_cmap) -> None:
        """Test if the green channel is dominant in mid-range values."""
        cmap = phosphor_cmap.colormap
        color_mid = cmap(0.7)
        assert color_mid[1] > color_mid[0]  # Green should be greater than red
        assert color_mid[1] > color_mid[2]  # Green should be greater than blue

    def test_monotonicity(self, phosphor_cmap) -> None:
        """Test if the green channel increases monotonically."""
        cmap = phosphor_cmap.colormap
        values = np.linspace(0, 1, 100)
        green_values = [cmap(v)[1] for v in values]

        assert all(green_values[i] <= green_values[i + 1] for i in range(len(green_values) - 1))

    def test_alpha_channel(self, phosphor_cmap) -> None:
        """Test if the colormap properly sets alpha channel."""
        cmap = phosphor_cmap.colormap
        color = cmap(0.5)
        assert len(color) == 4  # Should have RGBA
        assert color[3] == 1.0  # Alpha should be 1.0 (fully opaque)

    def test_matplotlib_registration(self) -> None:
        """Test if the colormap gets properly registered in matplotlib."""
        name = "test_phosphor_registration"
        PhosphorColormap(name=name).create()
        assert name in colormaps()

    def test_invalid_input(self) -> None:
        """Test if the class handles invalid input appropriately."""
        with pytest.raises(TypeError):
            PhosphorColormap(name=123)  # type: ignore

    def test_array_input(self, phosphor_cmap) -> None:
        """Test if the colormap can handle array inputs."""
        cmap = phosphor_cmap.colormap
        values = np.array([0.0, 0.5, 1.0])
        colors = cmap(values)
        assert colors.shape == (3, 4)  # Should return 3 RGBA colors

    def test_segments_creation(self, phosphor_cmap) -> None:
        """Test if the color segments are created correctly."""
        segments = phosphor_cmap._create_segments()
        assert all(key in segments for key in ["red", "green", "blue"])
        assert all(len(segments[key]) == len(phosphor_cmap.colors) for key in segments)


class TestCIFParser:
    """Test suite for parse_cif_to_jax function."""

    @pytest.fixture
    def mock_structure(self):
        """Create a mock pymatgen Structure for testing."""
        # Create a simple cubic lattice
        lattice = Lattice.cubic(5.0)
        # Create a structure with two atoms
        coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        species = ["Si", "O"]
        return Structure(lattice, species, coords)

    @pytest.fixture
    def sample_cif_file(self):
        """Create a temporary CIF file for testing."""
        cif_content = """
        data_test
        _cell_length_a 5.0
        _cell_length_b 5.0
        _cell_length_c 5.0
        _cell_angle_alpha 90.0
        _cell_angle_beta  90.0
        _cell_angle_gamma 90.0
        _symmetry_space_group_name_H-M 'P 1'
        _atom_site_label Si1
        _atom_site_fract_x 0.0
        _atom_site_fract_y 0.0
        _atom_site_fract_z 0.0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
            f.write(cif_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)  # Cleanup after tests

    def test_file_not_found(self) -> None:
        """Test if function raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            parse_cif_to_jax("nonexistent.cif")

    def test_invalid_extension(self) -> None:
        """Test if function raises ValueError for invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f, pytest.raises(ValueError):
            parse_cif_to_jax(f.name)

    @patch("pymatgen.io.cif.CifParser")
    def test_invalid_cell_lengths(self, MockCifParser) -> None:
        """Test if function raises ValueError for invalid cell lengths."""
        mock_structure = Mock()
        mock_structure.lattice.abc = (-1.0, 5.0, 5.0)
        mock_structure.lattice.angles = (90.0, 90.0, 90.0)
        MockCifParser.return_value.parse_structures.return_value = [mock_structure]

        with pytest.raises(ValueError, match="Cell lengths must be positive"):
            parse_cif_to_jax("test.cif")

    @patch("pymatgen.io.cif.CifParser")
    def test_invalid_cell_angles(self, MockCifParser) -> None:
        """Test if function raises ValueError for invalid cell angles."""
        mock_structure = Mock()
        mock_structure.lattice.abc = (5.0, 5.0, 5.0)
        mock_structure.lattice.angles = (90.0, 181.0, 90.0)
        MockCifParser.return_value.parse_structures.return_value = [mock_structure]

        with pytest.raises(ValueError, match="Cell angles must be between 0 and 180 degrees"):
            parse_cif_to_jax("test.cif")

    def test_successful_parsing(self, sample_cif_file) -> None:
        """Test successful parsing of a valid CIF file."""
        result = parse_cif_to_jax(sample_cif_file)

        # Check if all required attributes are present
        assert hasattr(result, "frac_positions")
        assert hasattr(result, "cart_positions")
        assert hasattr(result, "cell_lengths")
        assert hasattr(result, "cell_angles")

        # Check array shapes
        assert result.frac_positions.shape[1] == 4  # x, y, z, atomic_number
        assert result.cart_positions.shape[1] == 4
        assert result.cell_lengths.shape == (3,)
        assert result.cell_angles.shape == (3,)

    @patch("pymatgen.io.cif.CifParser")
    def test_primitive_cell_option(self, MockCifParser, mock_structure) -> None:
        """Test if primitive cell option is correctly passed."""
        MockCifParser.return_value.parse_structures.return_value = [mock_structure]

        # Test with primitive=True
        parse_cif_to_jax("test.cif", primitive=True)
        MockCifParser.return_value.parse_structures.assert_called_with(primitive=True)

        # Test with primitive=False
        parse_cif_to_jax("test.cif", primitive=False)
        MockCifParser.return_value.parse_structures.assert_called_with(primitive=False)

    def test_array_types(self, sample_cif_file) -> None:
        """Test if the function returns JAX arrays with correct types."""
        result = parse_cif_to_jax(sample_cif_file)

        assert isinstance(result.frac_positions, jnp.ndarray)
        assert isinstance(result.cart_positions, jnp.ndarray)
        assert isinstance(result.cell_lengths, jnp.ndarray)
        assert isinstance(result.cell_angles, jnp.ndarray)

    def test_coordinate_conversion(self, sample_cif_file) -> None:
        """Test if fractional and Cartesian coordinates are correctly related."""
        result = parse_cif_to_jax(sample_cif_file)

        # For a cubic cell, conversion should be straightforward
        # Scale fractional coordinates by cell length
        scaled_coords = result.frac_positions[:, :3] * result.cell_lengths[0]
        np.testing.assert_array_almost_equal(scaled_coords, result.cart_positions[:, :3], decimal=5)

    def test_atomic_numbers(self, sample_cif_file) -> None:
        """Test if atomic numbers are correctly assigned."""
        result = parse_cif_to_jax(sample_cif_file)

        # Si atomic number is 14
        assert result.frac_positions[0, 3] == 14
        assert result.cart_positions[0, 3] == 14

    @pytest.mark.parametrize(
        "dimension", ["frac_positions", "cart_positions", "cell_lengths", "cell_angles"]
    )
    def test_array_dimensions(self, sample_cif_file, dimension) -> None:
        """Test if all arrays have correct dimensions."""
        result = parse_cif_to_jax(sample_cif_file)
        array = getattr(result, dimension)

        if dimension in ["frac_positions", "cart_positions"]:
            assert array.ndim == 2
            assert array.shape[1] == 4  # x, y, z, atomic_number
        else:
            assert array.ndim == 1
            assert array.shape[0] == 3  # a, b, c or α, β, γ


if __name__ == "__main__":
    pytest.main([__file__])
