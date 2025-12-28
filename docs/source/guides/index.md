# Rheedium Theory and Architecture Guides

This documentation provides comprehensive coverage of the physics and software architecture underlying rheedium, a JAX-based framework for simulating Reflection High-Energy Electron Diffraction (RHEED) patterns.

## Target Audience

These guides are written for **physics researchers** working with RHEED who want to understand:

- The mathematical foundations of kinematic diffraction theory
- How crystallographic data flows through the simulation pipeline
- The physical meaning of simulation parameters and outputs

## Guide Overview

### Physics Foundations

| Guide | Description |
|-------|-------------|
| [Kinematic Scattering](kinematic-scattering.md) | Single-scattering approximation, structure factors, and intensity calculations |
| [Ewald Sphere](ewald-sphere.md) | Geometric diffraction conditions in reciprocal space |
| [Form Factors](form-factors.md) | Atomic scattering amplitudes and thermal (Debye-Waller) effects |
| [Surface Rods](surface-rods.md) | Crystal truncation rods, roughness damping, and finite domain effects |

### Data and Architecture

| Guide | Description |
|-------|-------------|
| [Data Wrangling](data-wrangling.md) | Parsing XYZ, CIF, and POSCAR files; coordinate transformations |
| [Unit Cell](unit-cell.md) | Lattice vector construction, reciprocal space, and surface slabs |
| [PyTree Architecture](pytree-architecture.md) | JAX data structures enabling GPU acceleration and autodiff |

## Quick Start

For hands-on examples, see the [tutorials](../source/tutorials/index.rst) which demonstrate:

1. **MgO kinematic simulation** - Basic RHEED pattern generation
2. **SrTiO3 simulation** - Perovskite surface diffraction
3. **Finite domain effects** - Beam broadening and coherence

## Mathematical Notation

Throughout these guides, we use:

- $\mathbf{k}$ for wavevectors (in $\text{Å}^{-1}$)
- $\mathbf{G}$ for reciprocal lattice vectors
- $\mathbf{r}$ for atomic positions (in Å)
- $(h, k, l)$ for Miller indices
- $\theta$ for grazing incidence angle
- $\phi$ for azimuthal angle

Equations are rendered using LaTeX notation compatible with GitHub and MathJax.
