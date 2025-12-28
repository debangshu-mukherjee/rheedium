# Unit Cell Operations

This guide covers crystallographic operations on unit cells: lattice vector construction, reciprocal space, supercells, and surface slab generation for RHEED simulation.

## Lattice Vector Construction

### From Cell Parameters

Given cell parameters $(a, b, c, \alpha, \beta, \gamma)$, rheedium constructs lattice vectors in the standard crystallographic convention:

- $\mathbf{a}$ along the $x$-axis
- $\mathbf{b}$ in the $xy$-plane
- $\mathbf{c}$ determined by angles

### Mathematical Construction

$$
\mathbf{a} = \begin{pmatrix} a \\ 0 \\ 0 \end{pmatrix}
$$

$$
\mathbf{b} = \begin{pmatrix} b \cos\gamma \\ b \sin\gamma \\ 0 \end{pmatrix}
$$

$$
\mathbf{c} = \begin{pmatrix}
c \cos\beta \\
c \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma} \\
c \sqrt{1 - \cos^2\beta - \left(\frac{\cos\alpha - \cos\beta\cos\gamma}{\sin\gamma}\right)^2}
\end{pmatrix}
$$

### Implementation

```python
from rheedium.ucell import build_cell_vectors

# Cubic cell (e.g., MgO)
lattice_cubic = build_cell_vectors(
    a=4.21, b=4.21, c=4.21,
    alpha=90.0, beta=90.0, gamma=90.0,
)
# Returns:
# [[4.21, 0.00, 0.00],
#  [0.00, 4.21, 0.00],
#  [0.00, 0.00, 4.21]]

# Hexagonal cell
lattice_hex = build_cell_vectors(
    a=3.0, b=3.0, c=5.0,
    alpha=90.0, beta=90.0, gamma=120.0,
)
# Returns:
# [[ 3.00,  0.00, 0.00],
#  [-1.50,  2.60, 0.00],
#  [ 0.00,  0.00, 5.00]]
```

## Extracting Cell Parameters

The inverse operation extracts parameters from vectors:

```python
from rheedium.ucell import compute_lengths_angles

a, b, c, alpha, beta, gamma = compute_lengths_angles(lattice)
```

### Formulas

$$
a = |\mathbf{a}|, \quad b = |\mathbf{b}|, \quad c = |\mathbf{c}|
$$

$$
\cos\alpha = \frac{\mathbf{b} \cdot \mathbf{c}}{bc}, \quad
\cos\beta = \frac{\mathbf{a} \cdot \mathbf{c}}{ac}, \quad
\cos\gamma = \frac{\mathbf{a} \cdot \mathbf{b}}{ab}
$$

## Reciprocal Lattice

### Definition

The reciprocal lattice vectors satisfy:

$$
\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}
$$

where $\mathbf{a}_i$ are real-space vectors and $\mathbf{b}_j$ are reciprocal vectors.

### Explicit Formulas

$$
\mathbf{a}^* = \frac{2\pi (\mathbf{b} \times \mathbf{c})}{V}
$$

$$
\mathbf{b}^* = \frac{2\pi (\mathbf{c} \times \mathbf{a})}{V}
$$

$$
\mathbf{c}^* = \frac{2\pi (\mathbf{a} \times \mathbf{b})}{V}
$$

where the unit cell volume is:

$$
V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})
$$

### Implementation

```python
from rheedium.ucell import reciprocal_lattice_vectors

recip = reciprocal_lattice_vectors(lattice)
# recip[i] is the i-th reciprocal lattice vector (Å⁻¹)

# Verify orthogonality
import jax.numpy as jnp
dot_product = lattice @ recip.T  # Should be 2π × identity
```

### Generating Reciprocal Lattice Points

```python
from rheedium.ucell import generate_reciprocal_points

# Generate all (h,k,l) within bounds
hkl_grid, g_vectors = generate_reciprocal_points(
    recip_vectors=recip,
    hmax=3, kmax=3, lmax=2,
)
# hkl_grid: [N, 3] integer Miller indices
# g_vectors: [N, 3] reciprocal space vectors (Å⁻¹)
```

## Miller Indices

### Notation

Miller indices $(hkl)$ describe planes perpendicular to $\mathbf{G}_{hkl}$:

$$
\mathbf{G}_{hkl} = h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*
$$

### Physical Interpretation

| Indices | Meaning |
|---------|---------|
| $(100)$ | Planes perpendicular to $\mathbf{a}^*$ (parallel to $bc$ face) |
| $(110)$ | Diagonal planes |
| $(111)$ | Body-diagonal planes (important for FCC, BCC) |
| $(00l)$ | Planes parallel to the surface (for $l$ non-zero) |

### Direction Notation

Square brackets $[hkl]$ denote directions (parallel to $h\mathbf{a} + k\mathbf{b} + l\mathbf{c}$).

For RHEED, the **surface normal** is typically along $[001]$ (the $\mathbf{c}$-axis).

## Supercell Generation

Supercells are created by replicating the unit cell:

### 2D Supercell (In-Plane)

For RHEED, we typically expand in the surface plane only:

```python
from rheedium.ucell import make_supercell

# Create a 3×3×1 supercell
supercell = make_supercell(
    crystal=unit_cell,
    nx=3, ny=3, nz=1,
)
# supercell has 9× the atoms, with scaled lattice vectors
```

### Lattice Vector Scaling

$$
\mathbf{a}' = n_x \mathbf{a}, \quad
\mathbf{b}' = n_y \mathbf{b}, \quad
\mathbf{c}' = n_z \mathbf{c}
$$

### Position Replication

For each atom at $\mathbf{r}$ in the original cell, the supercell contains atoms at:

$$
\mathbf{r}' = \mathbf{r} + i\mathbf{a} + j\mathbf{b} + k\mathbf{c}
$$

for $i = 0, \ldots, n_x-1$, etc.

## Surface Slab Construction

RHEED simulates diffraction from a surface-oriented slab. This requires rotating the crystal so the desired surface normal aligns with the $z$-axis.

### Surface Orientation

Given a surface defined by Miller indices $(hkl)$, the surface normal is:

$$
\mathbf{n} = h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*
$$

### Rotation to Align with z-Axis

Rheedium constructs a rotation matrix $\mathbf{R}$ such that:

$$
\mathbf{R} \cdot \mathbf{n} = |\mathbf{n}| \hat{\mathbf{z}}
$$

All atomic positions are then transformed:

$$
\mathbf{r}' = \mathbf{R} \cdot \mathbf{r}
$$

### Implementation

```python
from rheedium.ucell import build_surface_slab

# Create a (001) surface slab
slab_001 = build_surface_slab(
    crystal=bulk_crystal,
    orientation=[0, 0, 1],  # (001) surface
    depth=20.0,             # Slab thickness in Å
    vacuum=10.0,            # Vacuum spacing in Å
)

# Create a (110) surface slab
slab_110 = build_surface_slab(
    crystal=bulk_crystal,
    orientation=[1, 1, 0],  # (110) surface
    depth=20.0,
)
```

### SlicedCrystal Structure

The result is a `SlicedCrystal` PyTree:

```python
@register_pytree_node_class
class SlicedCrystal(NamedTuple):
    cart_positions: Float[Array, "N 4"]  # Rotated Cartesian coords + Z
    cell_lengths: Float[Array, "3"]       # Supercell dimensions
    cell_angles: Float[Array, "3"]        # Typically [90, 90, 90]
    orientation: Int[Array, "3"]          # [h, k, l] of surface
    depth: float                          # Slab thickness (Å)
    x_extent: float                       # Lateral size in x (Å)
    y_extent: float                       # Lateral size in y (Å)
```

## Atom Scraping by Depth

For RHEED, only atoms near the surface contribute to scattering:

```python
from rheedium.ucell import atom_scraper

# Keep only atoms within 30 Å of the surface
surface_atoms = atom_scraper(
    crystal=slab,
    thickness=30.0,  # Å from top surface
)
```

This reduces computation by excluding deeply buried atoms.

## Lattice Systems

### Seven Crystal Systems

| System | Constraints | Example |
|--------|-------------|---------|
| Cubic | $a = b = c$, $\alpha = \beta = \gamma = 90°$ | NaCl, MgO |
| Tetragonal | $a = b \neq c$, all angles 90° | SrTiO₃, rutile |
| Orthorhombic | $a \neq b \neq c$, all angles 90° | Olivine |
| Hexagonal | $a = b \neq c$, $\gamma = 120°$, others 90° | Graphite, ZnO |
| Trigonal | $a = b = c$, $\alpha = \beta = \gamma \neq 90°$ | Quartz |
| Monoclinic | $\alpha = \gamma = 90°$, $\beta \neq 90°$ | Gypsum |
| Triclinic | No constraints | Feldspar |

### Convenience Functions

```python
from rheedium.ucell import (
    cubic_cell,
    tetragonal_cell,
    hexagonal_cell,
)

# Quick cubic cell
lattice = cubic_cell(a=4.21)

# Tetragonal cell
lattice = tetragonal_cell(a=3.905, c=3.905)

# Hexagonal cell
lattice = hexagonal_cell(a=3.0, c=5.0)
```

## Coordinate Wrapping

Fractional coordinates should be in [0, 1) for proper periodicity:

```python
from rheedium.ucell import wrap_to_unit_cell

# Wrap fractional coordinates to [0, 1)
wrapped = wrap_to_unit_cell(frac_positions)
# Uses modulo operation: wrapped = frac_positions % 1.0
```

This is automatically applied during symmetry expansion and supercell generation.

## Volume and Density

### Unit Cell Volume

$$
V = |\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})|
$$

```python
from rheedium.ucell import cell_volume

vol = cell_volume(lattice)  # Å³
```

### Density Calculation

$$
\rho = \frac{\sum_j M_j}{V \cdot N_A}
$$

where $M_j$ is atomic mass and $N_A$ is Avogadro's number.

## Workflow: Bulk to Surface Simulation

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. LOAD BULK CRYSTAL                         │
├─────────────────────────────────────────────────────────────────┤
│  crystal = parse_cif("bulk.cif")                                │
│  • Cell parameters: a, b, c, α, β, γ                            │
│  • Atomic positions: frac_positions, cart_positions             │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  2. BUILD SURFACE SLAB                           │
├─────────────────────────────────────────────────────────────────┤
│  slab = build_surface_slab(                                     │
│      crystal,                                                   │
│      orientation=[0, 0, 1],  # (001) surface                    │
│      depth=50.0,             # 50 Å thick                       │
│  )                                                              │
│  • Rotate to align [001] with z-axis                            │
│  • Create supercell for in-plane periodicity                    │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. GENERATE RECIPROCAL SPACE                    │
├─────────────────────────────────────────────────────────────────┤
│  recip = reciprocal_lattice_vectors(slab.cell_vectors)          │
│  hkl, g_vectors = generate_reciprocal_points(recip, ...)        │
│  • Reciprocal lattice for diffraction geometry                  │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. RHEED SIMULATION                           │
├─────────────────────────────────────────────────────────────────┤
│  ewald = build_ewald_data(slab, voltage_kv=15.0, ...)           │
│  pattern = kinematic_spot_simulator(slab, ewald, theta=2.0)     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Source Files

- `ucell/unitcell.py` - Lattice construction, reciprocal space
- `ucell/helper.py` - Supercells, surface slabs, atom scraping
- `types/rheed_types.py` - SlicedCrystal definition
