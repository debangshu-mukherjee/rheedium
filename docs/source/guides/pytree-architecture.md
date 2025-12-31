# PyTree Architecture

Rheedium uses JAX PyTrees as the foundation for all data structures, enabling GPU acceleration, automatic differentiation, and efficient functional transformations across the entire simulation pipeline.

```{figure} figures/data_flow_diagram.svg
:alt: Data flow through rheedium
:width: 100%

Data flow through rheedium's PyTree-based architecture, from input file parsing through simulation to pattern output. Each box represents a PyTree-registered data structure that can be JIT-compiled and transformed.
```

## What Are PyTrees?

A PyTree is JAX's abstraction for nested data structures containing arrays. Any Python object registered as a PyTree can be:

- **JIT-compiled** for GPU/TPU acceleration
- **Vectorized** with `jax.vmap` for batch processing
- **Differentiated** with `jax.grad` for optimization
- **Transformed** with `jax.tree_map` for element-wise operations

In rheedium, crystallographic data structures are registered as PyTrees, allowing seamless integration with JAX's transformation machinery.

## PyTree Classes in Rheedium

Rheedium defines 7 PyTree-registered classes across two modules:

### Crystal Data Structures (`types/crystal_types.py`)

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `CrystalStructure` | Bulk crystal with dual coordinate systems | `frac_positions`, `cart_positions`, `cell_lengths`, `cell_angles` |
| `EwaldData` | Angle-independent precomputed diffraction data | `wavelength_ang`, `k_magnitude`, `g_vectors`, `structure_factors` |
| `PotentialSlices` | 3D potential slices for multislice simulation | `slices`, `slice_thickness`, `x_calibration`, `y_calibration` |
| `XYZData` | Parsed XYZ file format container | `positions`, `atomic_numbers`, `lattice`, `energy` |

### RHEED-Specific Structures (`types/rheed_types.py`)

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `RHEEDPattern` | Computed diffraction pattern output | `G_indices`, `k_out`, `detector_points`, `intensities` |
| `RHEEDImage` | Experimental RHEED image data | `img_array`, `incoming_angle`, `calibration`, `detector_distance` |
| `SlicedCrystal` | Surface-oriented slab for simulation | `cart_positions`, `orientation`, `depth`, `x_extent`, `y_extent` |

```{figure} figures/crystal_structure_example.svg
:alt: Crystal structure PyTree
:width: 85%

A crystal structure visualization showing the data stored in a `CrystalStructure` PyTree: atomic positions, cell parameters, and atomic numbers are all stored as JAX arrays that can be transformed together.
```

## Registration Pattern

All PyTrees in rheedium follow the same pattern: **NamedTuple + `@register_pytree_node_class`**.

### Why NamedTuple?

1. **Immutability**: Prevents accidental mutation, essential for functional JAX code
2. **Named fields**: Self-documenting access like `crystal.cell_lengths` instead of `crystal[2]`
3. **Type hints**: Compatible with static analysis and IDE autocompletion

### Registration Example

```python
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Float, Int, Array

@register_pytree_node_class
class RHEEDPattern(NamedTuple):
    """JAX-compatible RHEED diffraction pattern."""

    G_indices: Int[Array, "N"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]

    def tree_flatten(self):
        """Flatten into (children, aux_data) for JAX."""
        children = (
            self.G_indices,
            self.k_out,
            self.detector_points,
            self.intensities,
        )
        aux_data = None  # No static metadata
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from flattened representation."""
        return cls(*children)
```

## Children vs Auxiliary Data

The `tree_flatten` method separates data into two categories:

### Children (Traced Arrays)

Arrays that participate in JAX transformations:

- Passed through `jax.jit` compilation
- Traced for automatic differentiation
- Mapped over with `jax.vmap`

**Example**: All coordinate arrays, intensities, wavevectors

### Auxiliary Data (Static Metadata)

Non-array data or arrays that should not be traced:

- Preserved unchanged through transformations
- Not differentiated
- Used for reconstruction in `tree_unflatten`

**Example**: Calibration values, string metadata, configuration flags

### Example: PotentialSlices

`PotentialSlices` stores calibration metadata as aux_data because these values are physical constants, not variables to optimize:

```python
@register_pytree_node_class
class PotentialSlices(NamedTuple):
    slices: Float[Array, "n_slices height width"]
    slice_thickness: float
    x_calibration: float
    y_calibration: float

    def tree_flatten(self):
        # Only the 3D array is a "child"
        children = (self.slices,)
        # Calibrations are aux_data (not traced)
        aux_data = (self.slice_thickness, self.x_calibration, self.y_calibration)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        slices = children[0]
        slice_thickness, x_cal, y_cal = aux_data
        return cls(slices, slice_thickness, x_cal, y_cal)
```

## Factory Functions with Validation

Since beartype cannot validate NamedTuple fields directly, rheedium uses **factory functions** that perform JAX-compatible validation before constructing PyTrees.

### The Pattern

```python
from jaxtyping import jaxtyped
from beartype import beartype
import jax.lax as lax

@jaxtyped(typechecker=beartype)
def create_rheed_pattern(
    g_indices: Int[Array, "N"],
    k_out: Float[Array, "M 3"],
    detector_points: Float[Array, "M 2"],
    intensities: Float[Array, "M"],
) -> RHEEDPattern:
    """Factory with runtime type checking."""

    # Validation happens at JIT compile time
    mm = intensities.shape[0]

    def _validate():
        # Check shape consistency
        lax.cond(
            k_out.shape == (mm, 3),
            lambda: k_out,
            lambda: lax.stop_gradient(lax.cond(False, lambda: k_out, lambda: k_out))
        )
        # Check positivity
        lax.cond(
            jnp.all(intensities >= 0),
            lambda: intensities,
            lambda: lax.stop_gradient(...)
        )

    _validate()
    return RHEEDPattern(g_indices, k_out, detector_points, intensities)
```

### Why `lax.cond` for Validation?

Standard Python `if` statements don't work inside JIT-compiled functions. `lax.cond` is JAX's traced conditional that:

1. **Evaluates at compile time** when conditions involve static shapes
2. **Raises errors via `lax.stop_gradient`** when the false branch is taken
3. **Preserves tracability** for gradient computation

## Benefits for RHEED Simulation

### 1. GPU Acceleration

PyTree registration enables seamless GPU execution:

```python
import jax

@jax.jit
def compute_pattern(crystal: CrystalStructure, voltage: float) -> RHEEDPattern:
    # Entire computation runs on GPU
    ewald = build_ewald_data(crystal, voltage)
    return kinematic_spot_simulator(crystal, ewald, theta=2.0)

# First call compiles; subsequent calls are fast
pattern = compute_pattern(my_crystal, 15.0)
```

### 2. Automatic Differentiation

Optimize structure parameters against experimental data:

```python
def loss(positions: Float[Array, "N 3"], target: RHEEDPattern) -> float:
    crystal = CrystalStructure(positions, ...)
    simulated = compute_pattern(crystal)
    return jnp.mean((simulated.intensities - target.intensities)**2)

# Gradient w.r.t. atomic positions
grad_positions = jax.grad(loss)(initial_positions, experimental_pattern)
```

### 3. Batch Processing with vmap

Compute azimuthal scans efficiently:

```python
@jax.jit
def single_angle(phi: float) -> RHEEDPattern:
    return kinematic_spot_simulator(crystal, ewald, theta=2.0, phi=phi)

# Vectorize over 360 azimuthal angles
phis = jnp.linspace(0, 360, 360)
all_patterns = jax.vmap(single_angle)(phis)
# all_patterns.intensities has shape (360, M)
```

### 4. Functional Transformations

Apply operations to all arrays in a structure:

```python
# Scale all positions by 1.01 (1% lattice expansion)
expanded = jax.tree_map(
    lambda x: x * 1.01 if x.ndim > 0 else x,
    crystal
)
```

## Data Flow Through PyTrees

```{figure} figures/pytree_hierarchy.svg
:alt: PyTree hierarchy and data flow
:width: 100%

Data flow through rheedium's PyTree structures, from input file parsing through `CrystalStructure` and `EwaldData` to the final `RHEEDPattern` output.
```

```
Input Files (CIF, XYZ, POSCAR)
        ↓
   parse_cif() / parse_xyz()
        ↓
┌───────────────────────────────────┐
│      CrystalStructure (PyTree)    │
│  ├─ frac_positions [N, 4]         │
│  ├─ cart_positions [N, 4]         │
│  ├─ cell_lengths [3]              │
│  └─ cell_angles [3]               │
└───────────────────────────────────┘
        ↓
   build_ewald_data()
        ↓
┌───────────────────────────────────┐
│       EwaldData (PyTree)          │
│  ├─ wavelength_ang                │
│  ├─ k_magnitude                   │
│  ├─ g_vectors [N, 3]              │
│  ├─ structure_factors [N]         │
│  └─ intensities [N]               │
└───────────────────────────────────┘
        ↓
   kinematic_spot_simulator()
        ↓
┌───────────────────────────────────┐
│      RHEEDPattern (PyTree)        │
│  ├─ G_indices [N]                 │
│  ├─ k_out [M, 3]                  │
│  ├─ detector_points [M, 2]        │
│  └─ intensities [M]               │
└───────────────────────────────────┘
```

## Type Aliases

Rheedium defines custom type aliases in `types/custom_types.py` for unified scalar handling:

```python
from typing import TypeAlias, Union
from jaxtyping import Float, Integer, Bool, Num, Array

# Accept both Python scalars and 0-d JAX arrays
scalar_float: TypeAlias = Union[float, Float[Array, " "]]
scalar_int: TypeAlias = Union[int, Integer[Array, " "]]
scalar_bool: TypeAlias = Union[bool, Bool[Array, " "]]
scalar_num: TypeAlias = Union[int, float, Num[Array, " "]]

# Image array types
float_image: TypeAlias = Float[Array, " H W"]
int_image: TypeAlias = Integer[Array, " H W"]
```

This allows functions to accept either Python primitives or JAX arrays transparently.

## Key Source Files

- `types/crystal_types.py` - Crystal and Ewald PyTrees
- `types/rheed_types.py` - RHEED pattern and image PyTrees
- `types/custom_types.py` - Type aliases
