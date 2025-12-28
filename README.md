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
[![Lines of Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/debangshu-mukherjee/rheedium/main/.github/badges/loc.json)](https://github.com/debangshu-mukherjee/rheedium)

## Overview

Rheedium is a JAX based computational framework for simulating RHEED patterns with automatic differentiation capabilities and GPU acceleration.

## Documentation

### Theory and Architecture Guides

- [Kinematic Scattering](https://rheedium.readthedocs.io/en/latest/guides/kinematic-scattering.html) - Diffraction theory, structure factors, and intensity calculations
- [Ewald Sphere](https://rheedium.readthedocs.io/en/latest/guides/ewald-sphere.html) - Geometric diffraction conditions in reciprocal space
- [Form Factors](https://rheedium.readthedocs.io/en/latest/guides/form-factors.html) - Atomic scattering amplitudes and thermal (Debye-Waller) effects
- [Surface Rods](https://rheedium.readthedocs.io/en/latest/guides/surface-rods.html) - Crystal truncation rods, roughness, and finite domain effects
- [Data Wrangling](https://rheedium.readthedocs.io/en/latest/guides/data-wrangling.html) - Parsing XYZ, CIF, and POSCAR files
- [Unit Cell](https://rheedium.readthedocs.io/en/latest/guides/unit-cell.html) - Lattice vectors, reciprocal space, and surface slabs
- [PyTree Architecture](https://rheedium.readthedocs.io/en/latest/guides/pytree-architecture.html) - JAX data structures for GPU acceleration

### API Reference

See the [full API documentation](https://rheedium.readthedocs.io/en/latest/api/index.html) on Read the Docs.

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
  version={2025.10.05},
  doi={10.5281/zenodo.14757400},
}
```