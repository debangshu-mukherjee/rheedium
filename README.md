# Rheedium

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/rheedium.svg)](https://badge.fury.io/py/rheedium)
[![PyPI Downloads](https://static.pepy.tech/badge/rheedium)](https://pepy.tech/projects/rheedium)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/rheedium/badge/?version=latest)](https://rheedium.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14757400.svg)](https://doi.org/10.5281/zenodo.14757400)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![jax_badge][https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC]

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