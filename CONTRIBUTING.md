# Contributing to Rheedium

Thank you for your interest in contributing to Rheedium! This guide will help you get started with development and ensure your contributions align with the project's standards.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- CUDA-compatible GPU (optional, for acceleration)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dxm447/rheedium.git
   cd rheedium
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
rheedium/
├── src/rheedium/           # Main source code
│   ├── inout/             # Data I/O operations
│   ├── plots/             # Visualization utilities
│   ├── recon/             # Reconstruction algorithms
│   ├── simul/             # RHEED simulation core
│   ├── types/             # Custom type definitions
│   └── ucell/             # Unit cell calculations
├── tests/                 # Test suite
├── docs/                  # Documentation
├── data/                  # Sample data files
└── tutorials/             # Example notebooks
```

## Coding Standards

### JAX-First Development

Rheedium is built on JAX for high-performance computation. All new code must follow JAX best practices:

**Required JAX Patterns:**
- Use `jax.lax.scan` instead of Python `for` loops
- Use `jax.lax.cond` instead of `if-then-else` statements
- Use `.at[].set()` for array updates instead of in-place modifications
- Ensure functions are purely functional (no side effects)
- No global variables or mutable state

**Example:**
```python
# ❌ Wrong - Python loops and conditionals
def bad_function(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
    return jnp.array(result)

# ✅ Correct - JAX patterns
def good_function(x: Float[Array, "n"]) -> Float[Array, "n"]:
    def scan_fn(carry, xi):
        return carry, jax.lax.cond(xi > 0, lambda: xi * 2, lambda: xi)
    _, result = jax.lax.scan(scan_fn, None, x)
    return result
```

### Type Hinting with jaxtyping

All functions must include comprehensive type hints using `jaxtyping`:

```python
from jaxtyping import Float, Array, Int
from beartype import beartype
from beartype.typing import Optional

@jaxtyped(typechecker=beartype)
def simulate_pattern(
    crystal: CrystalStructure,
    voltage_kV: Float[Array, ""],
    theta_deg: Optional[Float[Array, ""]] = jnp.asarray(2.0),
) -> RHEEDPattern:
    """Function implementation..."""
```

**Type Hinting Rules:**
- All function parameters and returns must be type hinted
- Intermediate variables inside functions should be type hinted
- Use descriptive array dimension names: `"n_atoms 3"` not `"* 3"`
- Import types from `rheedium.types` for consistency

### Documentation Standards

#### Module Docstrings

Each Python file must start with a module docstring listing all functions and classes:

```python
"""
Module: ucell.unitcell
----------------------
Functions for unit cell calculations and transformations.

Functions
---------
- `reciprocal_unitcell`:
    Calculate reciprocal unit cell parameters from direct cell parameters
- `build_cell_vectors`:
    Construct unit cell vectors from lengths and angles

Classes
-------
- `CrystalStructure`:
    JAX-compatible crystal structure representation
"""
```

#### Function Docstrings

All functions must follow this comprehensive docstring format:

```python
def simulate_rheed_pattern(
    crystal: CrystalStructure,
    voltage_kV: Float[Array, ""],
    theta_deg: Float[Array, ""],
) -> RHEEDPattern:
    """
    Description
    -----------
    Compute a kinematic RHEED pattern for the given crystal using
    atomic form factors from Kirkland potentials.

    Parameters
    ----------
    - `crystal` (CrystalStructure):
        Crystal structure to simulate
    - `voltage_kV` (Float[Array, ""]):
        Accelerating voltage in kilovolts
    - `theta_deg` (Float[Array, ""]):
        Grazing angle in degrees

    Returns
    -------
    - `pattern` (RHEEDPattern):
        RHEED pattern with detector points and intensities

    Examples
    --------
    >>> import rheedium as rh
    >>> crystal = rh.inout.parse_cif("structure.cif")
    >>> pattern = rh.simul.simulate_rheed_pattern(
    ...     crystal=crystal,
    ...     voltage_kV=jnp.asarray(20.0),
    ...     theta_deg=jnp.asarray(2.0)
    ... )

    Flow
    ----
    - Generate reciprocal lattice points
    - Calculate incident wavevector
    - Find allowed reflections using Ewald sphere
    - Project onto detector plane
    - Compute intensities with atomic form factors
    - Return structured RHEED pattern
    """
```

**Required Sections:**
- `Description`: What the function does
- `Parameters`: All parameters with types and descriptions
- `Returns`: Return value with type and description
- `Examples`: Working code examples
- `Flow`: High-level algorithm steps

### Code Style

- **Variable Names**: Use descriptive `snake_case` names
- **No Comments in Code**: All explanations go in docstrings
- **Long Names Over Short**: `reciprocal_lattice_vectors` not `rlv`
- **Pure Functions**: No side effects, return new data

### JAX Validation Pattern

For factory functions creating custom types, use this JAX-compatible validation pattern:

```python
def create_crystal_structure(
    frac_positions: Float[Array, "* 4"],
    cart_positions: Float[Array, "* 4"],
) -> CrystalStructure:
    def validate_and_create():
        def check_shape():
            return lax.cond(
                frac_positions.shape[1] == 4,
                lambda: frac_positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: frac_positions, lambda: frac_positions)
                )
            )
        
        check_shape()  # Execute validation
        
        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions
        )
    
    return validate_and_create()
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=rheedium tests/

# Run specific test file
pytest tests/ucell/test_unitcell.py
```

### Writing Tests

- Place tests in corresponding directories under `tests/`
- Test files should be named `test_<module>.py`
- Use descriptive test function names
- Test both functionality and JAX compatibility

Example test:

```python
import jax.numpy as jnp
import rheedium as rh

def test_wavelength_calculation():
    """Test electron wavelength calculation for various energies."""
    energies = jnp.array([10.0, 20.0, 30.0])
    wavelengths = rh.ucell.wavelength_ang(energies)
    
    assert wavelengths.shape == (3,)
    assert jnp.all(wavelengths > 0)
    assert jnp.allclose(wavelengths[0], 0.1226, rtol=1e-3)
```

## Pull Request Process

### Before Submitting

1. **Code Quality:**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Run pre-commit
   pre-commit run --all-files
   
   # Run tests
   pytest tests/
   ```

2. **Documentation:**
   ```bash
   # Build docs locally
   cd docs/
   make html
   ```

### PR Guidelines

1. **Branch Naming:** Use descriptive names like `feature/multislice-potential` or `fix/bessel-function-accuracy`

2. **Commit Messages:** Use clear, descriptive commit messages:
   ```
   Add multislice potential calculation for crystal structures
   
   - Implement crystal_potential function using Fourier shifts
   - Add support for arbitrary grid shapes and sampling
   - Include comprehensive tests and documentation
   ```

3. **PR Description:** Include:
   - What the PR does
   - Why it's needed
   - How to test it
   - Any breaking changes

### Review Process

All PRs require:
- [ ] Passing CI tests
- [ ] Code review approval
- [ ] Documentation updates (if applicable)
- [ ] No merge conflicts

## Issue Guidelines

### Bug Reports

Include:
- Minimal reproducible example
- Expected vs actual behavior
- Environment details (Python version, JAX version, GPU/CPU)
- Error messages and stack traces

### Feature Requests

Include:
- Use case description
- Proposed API (if applicable)
- Performance considerations
- Relationship to existing functionality

## Development Guidelines

### Performance Considerations

- Profile new algorithms with `jax.profiler`
- Use `@jax.jit` for computational functions
- Consider memory usage for large crystal structures
- Test GPU compatibility when applicable

### Adding New Features

1. **Design Phase:**
   - Discuss approach in an issue first
   - Consider JAX constraints early
   - Plan type signatures and API

2. **Implementation:**
   - Start with working CPU implementation
   - Add comprehensive tests
   - Optimize for performance
   - Add GPU compatibility if relevant

3. **Documentation:**
   - Update API documentation
   - Add tutorial examples
   - Update README if needed

### Backwards Compatibility

- Maintain API compatibility when possible
- Use deprecation warnings for breaking changes
- Document migration paths in CHANGELOG

## Getting Help

- **Questions:** Open a discussion or issue
- **Chat:** Contact maintainers directly
- **Documentation:** Check the [docs](https://rheedium.readthedocs.io/)

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for their contributions
- Paper acknowledgments (for significant algorithmic contributions)
- GitHub contributors list

Thank you for contributing to Rheedium and advancing RHEED simulation capabilities!