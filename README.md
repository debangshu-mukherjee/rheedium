# Rheedium

[![PyPI Downloads](https://static.pepy.tech/badge/rheedium)](https://pepy.tech/projects/rheedium)
[![License](https://img.shields.io/pypi/l/rheedium.svg)](https://github.com/debangshu-mukherjee/rheedium/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/rheedium.svg)](https://pypi.python.org/pypi/rheedium)
[![Python Versions](https://img.shields.io/pypi/pyversions/rheedium.svg)](https://pypi.python.org/pypi/rheedium)
[![Tests](https://github.com/debangshu-mukherjee/rheedium/actions/workflows/test.yml/badge.svg)](https://github.com/debangshu-mukherjee/rheedium/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/rheedium/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/rheedium)
[![Documentation Status](https://readthedocs.org/projects/rheedium/badge/?version=latest)](https://rheedium.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14757400.svg)](https://doi.org/10.5281/zenodo.14757400)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![jax_badge](https://tinyurl.com/mucknrvu)](https://docs.jax.dev/)

**High-Performance RHEED Pattern Simulation for Crystal Surface Analysis**

*A JAX-accelerated Python package for realistic Reflection High-Energy Electron Diffraction (RHEED) pattern simulation using kinematic theory and atomic form factors.*

[Documentation](https://rheedium.readthedocs.io/) • [Installation](#installation) • [Quick Start](#quick-start) • [Examples](#examples) • [Contributing](#contributing)

</div>

## Overview

Rheedium is a modern computational framework for simulating RHEED patterns with scientific rigor and computational efficiency. Built on JAX for automatic differentiation and GPU acceleration, it provides researchers with tools to:

- **Simulate realistic RHEED patterns** using Ewald sphere construction and kinematic diffraction theory
- **Analyze crystal surface structures** with atomic-resolution precision
- **Handle complex reconstructions** including domains, supercells, and surface modifications
- **Leverage high-performance computing** with JAX's JIT compilation and GPU support

### Key Features

- **JAX-Accelerated**: GPU-ready computations with automatic differentiation
- **Physically Accurate**: Kirkland atomic potentials and kinematic scattering theory
- **Comprehensive Analysis**: Support for CIF files, surface reconstructions, and domains
- **Visualization Tools**: Phosphor screen colormap and interpolation for realistic display
- **Research-Ready**: Designed for thin-film growth, MBE, and surface science studies

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Install from PyPI

```bash
pip install rheedium
```

### Install for Development

```bash
git clone https://github.com/your-username/rheedium.git
cd rheedium
pip install -e ".[dev]"
```

### Dependencies

- JAX (with GPU support if available)
- NumPy
- Matplotlib
- SciPy
- Pandas
- Beartype (for runtime type checking)

## Quick Start

### Basic RHEED Simulation

```python
import rheedium as rh
import jax.numpy as jnp

# Load crystal structure from CIF file
crystal = rh.inout.parse_cif("data/SrTiO3.cif")

# Simulate RHEED pattern
pattern = rh.simul.simulate_rheed_pattern(
    crystal=crystal,
    voltage_kV=10.0,        # Beam energy
    theta_deg=2.0,          # Grazing angle
    detector_distance=1000.0 # Screen distance (mm)
)

# Visualize results
rh.plots.plot_rheed(pattern, interp_type="cubic")
```

### Working with Surface Reconstructions

```python
# Filter atoms within penetration depth
filtered_crystal = rh.ucell.atom_scraper(
    crystal=crystal,
    zone_axis=jnp.array([0, 0, 1]),  # Surface normal
    penetration_depth=5.0            # Angstroms
)

# Simulate pattern for surface layer
surface_pattern = rh.simul.simulate_rheed_pattern(
    crystal=filtered_crystal,
    voltage_kV=15.0,
    theta_deg=1.5
)
```

### Advanced Analysis

```python
# Generate reciprocal lattice points
reciprocal_points = rh.ucell.generate_reciprocal_points(
    crystal=crystal,
    hmax=5, kmax=5, lmax=2
)

# Calculate kinematic intensities
intensities = rh.simul.compute_kinematic_intensities(
    positions=crystal.cart_positions[:, :3],
    G_allowed=reciprocal_points
)
```

## Examples

### 1. Single Crystal Analysis

```python
import rheedium as rh

# Load SrTiO3 structure
crystal = rh.inout.parse_cif("examples/SrTiO3.cif")

# High-resolution simulation
pattern = rh.simul.simulate_rheed_pattern(
    crystal=crystal,
    voltage_kV=30.0,
    theta_deg=1.0,
    hmax=6, kmax=6, lmax=2,
    tolerance=0.01
)

# Create publication-quality plot
rh.plots.plot_rheed(
    pattern, 
    grid_size=400,
    interp_type="cubic",
    cmap_name="phosphor"
)
```

### 2. Surface Reconstruction Study

```python
# Analyze (√13×√13)-R33.7° reconstruction
reconstructed_crystal = rh.ucell.parse_cif_and_scrape(
    cif_path="data/SrTiO3.cif",
    zone_axis=jnp.array([0, 0, 1]),
    thickness_xyz=jnp.array([0, 0, 3.9])  # Single unit cell
)

# Compare patterns at different azimuths
azimuths = [0, 15, 30, 45]
patterns = []

for azimuth in azimuths:
    # Rotate crystal
    rotation_matrix = rh.ucell.build_rotation_matrix(azimuth)
    rotated_crystal = rh.ucell.rotate_crystal(reconstructed_crystal, rotation_matrix)
    
    # Simulate pattern
    pattern = rh.simul.simulate_rheed_pattern(rotated_crystal, theta_deg=2.6)
    patterns.append(pattern)
```

### 3. Domain Analysis

```python
# Multi-domain simulation
domains = []
for rotation_angle in [33.7, -33.7]:  # Twin domains
    rotated_crystal = rh.ucell.rotate_crystal(crystal, rotation_angle)
    domain_pattern = rh.simul.simulate_rheed_pattern(rotated_crystal)
    domains.append(domain_pattern)

# Combine domain contributions
combined_pattern = rh.types.combine_rheed_patterns(domains)
```

## Supported File Formats

- **CIF files**: Crystallographic Information Format with symmetry operations
- **CSV data**: Kirkland atomic potential parameters
- **Image formats**: PNG, TIFF, SVG for visualization output

## Configuration

### Performance Optimization

```python
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Use GPU if available
jax.config.update("jax_platform_name", "gpu")

# JIT compilation for speed
@jax.jit
def fast_simulation(crystal, voltage):
    return rh.simul.simulate_rheed_pattern(crystal, voltage_kV=voltage)
```

### Custom Atomic Potentials

```python
# Use custom Kirkland parameters
custom_potential = rh.simul.atomic_potential(
    atom_no=38,  # Strontium
    pixel_size=0.05,
    sampling=32,
    potential_extent=6.0,
    datafile="custom_potentials.csv"
)
```

## Applications

Rheedium is designed for researchers working in:

- **Molecular Beam Epitaxy (MBE)**: Real-time growth monitoring and optimization
- **Pulsed Laser Deposition (PLD)**: Surface quality assessment and phase identification
- **Surface Science**: Reconstruction analysis and domain characterization
- **Materials Engineering**: Thin film quality control and defect analysis
- **Method Development**: New RHEED analysis technique validation

## Documentation

Full documentation is available at [rheedium.readthedocs.io](https://rheedium.readthedocs.io/), including:

- **API Reference**: Complete function and class documentation
- **Tutorials**: Step-by-step guides for common workflows
- **Theory Guide**: Mathematical background and implementation details
- **Examples Gallery**: Real-world usage scenarios with code

## Contributing

We welcome contributions from the community! Please see our [Contributing Guide](https://github.com/debangshu-mukherjee/rheedium/blob/main/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

### Development Setup

```bash
git clone https://github.com/your-username/rheedium.git
cd rheedium
pip install -e ".[dev,test,docs]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
pytest --cov=rheedium tests/  # With coverage
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/debangshu-mukherjee/rheedium/blob/main/LICENSE) file for details.

## Citation

If you use Rheedium in your research, please cite:

```bibtex
@software{rheedium2024,
  title={Rheedium: High-Performance RHEED Pattern Simulation},
  author={Mukherjee, Debangshu},
  year={2025},
  url={https://github.com/debangshu-mukherjee/rheedium},
  version={2025.6.16},
  doi={10.5281/zenodo.14757400},
}
```