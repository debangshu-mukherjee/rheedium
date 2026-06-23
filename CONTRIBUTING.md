# Contributing to Rheedium

Thank you for your interest in contributing to Rheedium! This guide describes how the codebase is written — type hinting, documentation, validation, testing, and tooling — so your contributions match the existing standards.

## Development Setup

### Prerequisites

- Python 3.11–3.13 (`requires-python = ">3.11, <3.14"`)
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Git
- CUDA-compatible GPU (optional, for acceleration)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debangshu-mukherjee/rheedium.git
   cd rheedium
   ```

2. **Install in development mode:**
   ```bash
   # Everything (docs, tests, notebooks, dev tooling)
   uv sync --extra dev

   # With CUDA support as well
   uv sync --extra dev_cuda
   ```

   The dependency groups are defined in `pyproject.toml`: `docs`, `test`,
   `notebooks`, `cuda`, `dev` (= docs + test + notebooks + tooling),
   `dev_cuda` (= dev + cuda), and `all`.

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
rheedium/
├── src/rheedium/          # Main source code
│   ├── audit/             # Benchmarking and realism-audit utilities
│   ├── inout/             # Data I/O (CIF, POSCAR, XYZ, HDF5, TIFF, vasprun)
│   ├── plots/             # Visualization utilities
│   ├── procs/             # Procedural models and preprocessing
│   ├── recon/             # Inverse-problem / reconstruction utilities
│   ├── simul/             # RHEED simulation core (kinematic, Ewald, ...)
│   ├── tools/             # Parallel/distributed and numerical helpers
│   ├── types/             # Custom JAX-compatible type definitions
│   └── ucell/             # Unit cell and crystallographic calculations
├── tests/                 # Test suite (mirrors src layout, see below)
├── docs/                  # Sphinx documentation
├── data/                  # Sample data files
└── tutorials/             # Paired Jupyter notebooks (.ipynb + .py)
```

Each subpackage is a namespace package exposing its public API through
`__init__.py` with an explicit `__all__`. The top-level `src/rheedium/__init__.py`
enables 64-bit precision (`jax.config.update("jax_enable_x64", True)`) and sets
CPU threading XLA flags **before** JAX is imported, and optionally initializes
multi-host distributed execution. Keep import-time side effects confined to that
module.

## Coding Standards

### JAX-First Development

Rheedium is built on JAX for differentiable, high-performance computation. All
new code must follow JAX best practices:

**Required JAX Patterns:**
- Use `jax.lax.scan` instead of Python `for` loops over array data
- Use `jax.lax.cond` / `jnp.where` instead of data-dependent `if`/`else`
- Use `.at[].set()` for array updates instead of in-place modification
- Keep functions purely functional — no side effects, no global mutable state
- Decorate computational functions with `@jax.jit` where appropriate
- Code must remain traceable for `jit`, `grad`, `vmap`, and `pmap`

**Example:**
```python
# ❌ Wrong - Python loops and conditionals over array data
def bad_function(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
    return jnp.array(result)


# ✅ Correct - vectorized JAX
@jaxtyped(typechecker=beartype)
def good_function(x: Float[Array, " n"]) -> Float[Array, " n"]:
    return jnp.where(x > 0, x * 2, x)
```

### Type Hinting with jaxtyping and beartype

Every public function is runtime-typechecked with the
`@jaxtyped(typechecker=beartype)` decorator stack and annotated with
`jaxtyping` shape/dtype specs:

```python
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, Int, Num, jaxtyped

from rheedium.types import CrystalStructure, RHEEDPattern, scalar_float


@jaxtyped(typechecker=beartype)
def simulate_pattern(
    crystal: CrystalStructure,
    voltage_kV: scalar_float,
    theta_deg: Optional[scalar_float] = 2.0,
) -> RHEEDPattern:
    """..."""
```

**Type Hinting Rules:**
- All parameters and return values are annotated; multiple returns use
  `beartype.typing.Tuple[...]`.
- Annotate intermediate variables inside function bodies too — e.g.
  `alpha_rad: Float[Array, ""] = jnp.deg2rad(alpha)`.
- Use descriptive dimension names in shape specs: `Num[Array, "N 4"]`,
  `Float[Array, " n 3"]`, scalars as `Float[Array, ""]`.
- Prefer the scalar aliases from `rheedium.types` (`scalar_float`,
  `scalar_int`, `scalar_bool`, `scalar_num`) for scalar arguments; these are
  unions accepting both Python scalars and 0-d JAX arrays. Image aliases
  (`float_jax_image`, `int_np_image`, ...) are also available.
- Import shared types from `rheedium.types`, not by re-defining them.
- Import typing constructs (`Optional`, `Union`, `Tuple`, `List`, `Dict`,
  `TypeAlias`) from `beartype.typing`, not the stdlib `typing` module.

### Custom Types and PyTrees

Structured data types live in `rheedium.types` and are **Equinox modules**
(`eqx.Module`): immutable JAX PyTrees that flow through `jit`/`grad`/`vmap`.
Static, non-array metadata fields are declared with `eqx.field(static=True)`
so they are excluded from the differentiable leaves.

```python
import equinox as eqx
from jaxtyping import Num, Array


class CrystalStructure(eqx.Module):
    """JAX-compatible Pytree with fractional and Cartesian coordinates.

    :see: :class:`~.test_crystal_types.TestCrystalStructure`
    ...
    """

    frac_positions: Num[Array, "N 4"]
    cart_positions: Num[Array, "N 4"]
    cell_lengths: Num[Array, "3"]
    cell_angles: Num[Array, "3"]
```

### Validation Pattern for Factory Functions

Custom types are constructed through `create_*` factory functions that
validate inputs. Use a two-tier approach:

- **Static shape/structure checks** that can be resolved at trace time use
  plain Python `raise ValueError`.
- **Data-dependent (traced) checks** use `equinox.error_if`, which raises at
  runtime without breaking `jit`.

```python
@jaxtyped(typechecker=beartype)
def create_crystal_structure(
    frac_positions: Num[Array, "... 4"],
    cart_positions: Num[Array, "... 4"],
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
) -> CrystalStructure:
    """Create a CrystalStructure PyTree with data validation.

    :see: :class:`~.test_crystal_types.TestCrystalStructure`
    ...
    """
    frac_positions = jnp.asarray(frac_positions)
    cart_positions = jnp.asarray(cart_positions)

    if frac_positions.shape[1] != 4:          # static -> ValueError
        raise ValueError("frac_positions must have shape (N, 4)")

    checked_cell_lengths = eqx.error_if(       # traced -> eqx.error_if
        cell_lengths,
        jnp.any(cell_lengths <= 0),
        "cell_lengths must be positive",
    )
    return CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=checked_cell_lengths,
        cell_angles=cell_angles,
    )
```

### Documentation Standards

Docstrings follow the **NumPy / numpydoc convention** (enforced by Ruff's
`pydocstyle` rules and `pydoclint`). Coverage is checked by `interrogate`
(`fail-under = 90`). Do **not** use ad-hoc section headers — stick to the
numpydoc sections below.

#### Module Docstrings

Each module starts with a one-line summary, an `Extended Summary`, a
`Routine Listings` section cross-referencing every public object, and a
`Notes` section where relevant:

```python
"""Functions for unit cell calculations and transformations.

Extended Summary
----------------
This module provides functions for crystallographic unit cell operations
including reciprocal-space calculations and lattice transformations.

Routine Listings
----------------
:func:`reciprocal_unitcell`
    Calculate reciprocal unit cell parameters from direct cell parameters.
:func:`build_cell_vectors`
    Construct unit cell vectors from lengths and angles.
:class:`CrystalStructure`
    JAX-compatible crystal structure representation.

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
"""
```

Use the correct Sphinx role in `Routine Listings`: `:func:` for functions,
`:class:` for classes/PyTrees, `:obj:` for type aliases and constants, and
`:mod:` for submodules.

#### Function and Class Docstrings

```python
@jaxtyped(typechecker=beartype)
def reciprocal_unitcell(
    a: scalar_float,
    b: scalar_float,
    c: scalar_float,
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    """
    Calculate reciprocal unit cell parameters from direct cell parameters.

    Computes reciprocal lattice parameters (a*, b*, c*, ...) from direct
    lattice parameters using crystallographic relationships.

    :see: :class:`~.test_unitcell.TestReciprocalUnitcell`

    Parameters
    ----------
    a : scalar_float
        Direct cell length a in Angstroms.
    b : scalar_float
        Direct cell length b in Angstroms.
    c : scalar_float
        Direct cell length c in Angstroms.

    Returns
    -------
    reciprocal_lengths : Float[Array, "3"]
        Reciprocal cell lengths [a*, b*, c*] in 1/Angstroms.
    reciprocal_angles : Float[Array, "3"]
        Reciprocal cell angles [α*, β*, γ*].

    Notes
    -----
    1. Convert input angles to radians if provided in degrees.
    2. Compute unit cell volume from the triple-product formula.
    3. Derive a*, b*, c* from the volume and direct cell parameters.

    See Also
    --------
    reciprocal_lattice_vectors : Generate reciprocal basis vectors.
    """
```

**Docstring conventions:**
- Open with a single imperative summary line.
- Add a `:see:` Sphinx cross-reference linking the object to its test class
  (e.g. `:see: :class:`~.test_unitcell.TestReciprocalUnitcell``). This is used
  throughout `src/` to tie each public symbol to its tests.
- `Parameters` and `Returns` repeat the type and describe each item. Name the
  return values (numpydoc `name : type` form) so `pydoclint` is satisfied.
- Use `Notes` (often a numbered list) to describe the algorithm/flow.
- Use `See Also` to point at related functions.
- Use `Attributes` for `eqx.Module` fields; `Yields` for generators; `Raises`
  where a function raises.
- Use a raw string (`r"""`) when the docstring contains LaTeX/backslashes.

### Code Style

Style is enforced by Ruff (`line-length = 79`, `target-version = "py312"`,
double quotes). The active lint rule set includes `D, E, F, B, I, N, UP, ANN,
S, A, C4, PIE, PT, RET, SIM, ARG, ERA, PL`. Key conventions:

- **Variable Names**: descriptive `snake_case`; long names over abbreviations
  (`reciprocal_lattice_vectors`, not `rlv`). Scientific single-letter symbols
  (`G`, `T`, `dE_E`) are permitted where they mirror the physics.
- **No inline comments for explanation**: explanations belong in docstrings.
  Comments are reserved for non-obvious rationale (the *why*, not the *what*).
- **Pure functions**: no side effects; return new data.
- **Imports**: sorted by isort (`I`); `jax` and `jaxtyping` are treated as
  known third-party. Imports inside functions are used only to guard optional
  dependencies or platform branches.

## Testing

The test suite uses `pytest` with `chex`, `absl.testing.parameterized`,
`hypothesis`, and `pytest-xdist`. Runtime jaxtyping/beartype checking is active
during tests via the `--jaxtyping-packages=rheedium,beartype.beartype` pytest
flag, so shape/dtype bugs surface as test failures.

### Test Layout

Tests mirror the source layout under `tests/test_rheedium/`:

```
tests/
├── conftest.py                       # JAX memory management, xdist worker sizing
├── _factories.py                     # Typed factories for test data
├── _assertions.py                    # Typed chex assertion helpers
├── _types.py                         # Shared test type aliases
├── test_data/                        # CIF/POSCAR/XYZ/npz fixtures
└── test_rheedium/
    ├── test_ucell/test_unitcell.py
    ├── test_simul/test_kinematic.py
    ├── test_types/test_crystal_types.py
    └── ...                            # one test_<module>.py per source module
```

- Test files are named `test_<module>.py`; test classes `Test*` (typically
  `chex.TestCase`); test functions `test_*`.
- Reuse the shared helpers (`_factories.make_*`, `_assertions.assert_*`)
  instead of hand-rolling fixture data; they are themselves
  `@jaxtyped(typechecker=beartype)`-decorated.

### Running Tests

```bash
# Run the whole suite (xdist auto-sizes workers from available RAM)
pytest

# Run a single module / class / test
pytest tests/test_rheedium/test_ucell/test_unitcell.py
pytest tests/test_rheedium/test_ucell/test_unitcell.py::TestBulkToSlice

# Coverage
pytest --cov=rheedium
```

`conftest.py` clears the JAX JIT cache after every test and fails tests that
leak more than `MEM_LEAK_THRESHOLD_GB` of RSS, so write tests that do not
retain large compiled artifacts.

### Writing Tests

- Prefer `chex` assertions over bare `assert` for arrays:
  `chex.assert_shape`, `chex.assert_type`, `chex.assert_tree_all_finite`,
  `chex.assert_trees_all_close`, `chex.assert_trees_all_equal`.
- Use `absl.testing.parameterized` (`@parameterized.parameters` /
  `@parameterized.named_parameters`) for table-driven cases, and `chex`
  variants (`@chex.variants`) to exercise functions under JIT and eager.
- Use `hypothesis` for property-based tests of numerical/crystallographic
  invariants.
- Test both correctness and JAX compatibility (jit/grad/vmap where relevant).

Example:

```python
import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

import rheedium as rh


class TestWavelength(chex.TestCase):
    """Tests for electron wavelength calculation."""

    def test_known_values(self) -> None:
        """Wavelength at 10 kV should match the analytic value."""
        energies: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = rh.ucell.wavelength_ang(energies)

        chex.assert_shape(wavelengths, (3,))
        chex.assert_tree_all_finite(wavelengths)
        assert bool(jnp.all(wavelengths > 0))
        chex.assert_trees_all_close(wavelengths[0], 0.1226, rtol=1e-3)
```

## Tutorial Notebooks

Tutorials live in `tutorials/` as Jupyter notebooks. Most notebooks are paired
with Jupytext percent scripts (`.ipynb` plus `.py`) so they can be edited in VS
Code/Jupyter while keeping reviewable source diffs (`[tool.jupytext]` sets
`formats = "ipynb,py:percent"`).

For remote development, open `tutorials/<notebook>.ipynb` through VS Code
Remote-SSH and select the project kernel. After editing a paired notebook, run:

```bash
uv run jupytext --sync tutorials/<notebook>.ipynb
uv run python docs/strip_notebook_outputs.py tutorials/<notebook>.ipynb
```

Commit the synced `.py` and output-stripped `.ipynb` together (the
`jupytext-sync` and `strip-notebook-outputs` pre-commit hooks do this for you).
The docs render the notebooks through `myst-nb`. Notebook outputs are generated
for documentation through a local MyST-NB/Jupyter cache rather than committed
to git:

```bash
uv run python docs/build_notebook_cache.py
cd docs
uv run make html
```

The cache lives under `docs/build/.jupyter_cache`, so it is rebuilt with the
other generated documentation artifacts.

## Pull Request Process

### Before Submitting

```bash
# Lint and format (must match CI: `ruff check src/ tests/`)
ruff check src/ tests/
ruff format src/ tests/

# Type check (primary checker is `ty`)
ty check

# Run all pre-commit hooks
pre-commit run --all-files

# Run the test suite
pytest

# Build docs locally
cd docs/
uv run make html
```

### Type Checking

`ty` is the project's type checker; its configuration lives under `[tool.ty]`
in `pyproject.toml`. A number of rules are suppressed because the
jaxtyping + beartype + JAX idioms produce false positives (e.g.
`invalid-assignment`, `invalid-argument-type`, `missing-argument`), while
`unresolved-attribute` is kept at `error` in `src/` to catch real bugs.
A stricter informational scan mirrors the CI `type-safety-scan` job and can be
run on demand:

```bash
pre-commit run --hook-stage manual strict-ty-scan
```

A `[tool.pyright]` block mirrors the same suppressions so VS Code / Pylance
matches the project's actual checker.

### Pre-commit Hooks

`pre-commit` runs (see `.pre-commit-config.yaml`):
`ruff check --fix`, `ruff format`, `jupytext --sync`, notebook output
stripping, and `ty check`. If `ruff --fix` modifies files, the commit aborts —
re-stage and commit again.

### PR Guidelines

1. **Branch Naming:** descriptive, e.g. `feature/multislice-potential` or
   `fix/bessel-function-accuracy`.

2. **Commit Messages:** clear and descriptive:
   ```
   Add multislice potential calculation for crystal structures

   - Implement crystal_potential function using Fourier shifts
   - Add support for arbitrary grid shapes and sampling
   - Include comprehensive tests and documentation
   ```

3. **PR Description:** include what the PR does, why it's needed, how to test
   it, and any breaking changes.

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
   - Discuss the approach in an issue first
   - Consider JAX constraints (tracing, shapes, purity) early
   - Plan the type signatures, custom types, and public API

2. **Implementation:**
   - Place the code in the appropriate subpackage and export it via that
     package's `__init__.py` (`__all__`)
   - Decorate with `@jaxtyped(typechecker=beartype)` and annotate fully
   - Add numpydoc docstrings with a `:see:` cross-reference to the tests
   - Add comprehensive tests mirroring the source path under
     `tests/test_rheedium/`
   - Optimize for performance and GPU compatibility where relevant

3. **Documentation:**
   - Update API documentation and `Routine Listings`
   - Add a tutorial example if it introduces user-facing functionality
   - Update the README if needed

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
