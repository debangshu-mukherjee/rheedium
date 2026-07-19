# Contributing to Rheedium

Thank you for your interest in contributing to Rheedium! This guide describes how the codebase is written — type hinting, documentation, validation, testing, and tooling — so your contributions match the existing standards.

## Core Principle: Invertible Modularity

Rheedium's modules are differentiable operators, and the boundaries between them are the boundaries at which the inverse problem is solved. A forward model built from clean but *opaque* boxes can only be run forwards; one built from boxes that never discard a gradient can be run backwards just as well — you attach a loss at any seam and solve for what produced the data (beam coherence, defect density, structure) while freezing the rest. This invertibility is the codebase's core asset.

It rests on one invariant:

> **Reductions stay explicit, late, and differentiable. No module collapses information it is not forced to.**

Concretely:

- Keep amplitudes complex; apply `|·|²` as late as possible, never inside a kernel that something downstream might want to sum coherently.
- Express averaging as an explicit weighted sum over a distribution, not as a baked-in convolution or a hidden quadrature.
- Prefer the analytic *coherent-average limit* (e.g. virtual-crystal occupancy, Debye–Waller damping) over a hard, irreversible collapse.
- Use `jnp.where` / `lax.cond` and continuous fields rather than discrete swaps or data-dependent Python control flow, so every parameter keeps a derivative.

The failure mode is silent: when a module performs a hard, non-differentiable, or premature reduction, the forward model still looks correct — only invertibility breaks, and only at that one seam. Treat any such reduction as a design smell to be justified explicitly in review, not an implementation detail. The JAX-First rules below are the mechanics of upholding this principle.

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
- **Assign before returning.** Bind a function's result to a type-annotated
  variable and return that name, rather than returning a bare expression — so the
  returned value carries an explicit type at its definition site:

  ```python
  # ❌ Avoid - bare expression in the return
  def add(a: int, b: int) -> int:
      return a + b


  # ✅ Prefer - annotated result, then return the name
  def add(a: int, b: int) -> int:
      c: int = a + b
      return c
  ```
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

**All types live in `rheedium.types` — no exceptions.** Every structured data
type — every `eqx.Module` PyTree, every `NamedTuple` / dataclass carrier, every
type alias, **and every `create_*` constructor that builds one** — is defined under
`src/rheedium/types/` and **nowhere else**. Every other subpackage (`simul`,
`procs`, `recon`, `ucell`, `inout`, `plots`, `tools`, `audit`) **imports** its
types from `rheedium.types`; it must **not** define its own PyTree, container, or
`create_*`/`*_spec` factory. Why: a single import surface, one PyTree
flatten/unflatten registration per type, one home for the validation (`create_*`)
contract, and no duplicate or near-duplicate carriers drifting across modules
(which is exactly what the inverse problem needs — `recon` compares `Distribution`
objects, so there must be *one* `Distribution`). A new type or constructor a
feature needs is added to the appropriate `types/<area>.py` first — e.g.
`crystal_types.py`, `beam_types.py`, `detector.py`, `simulation_params.py`,
`rheed_types.py`, or the `types/distributions/` subpackage — **then** imported
where it is used. A result/parameter container that "feels local" to a solver or
producer (e.g. a fit-result or problem-spec PyTree) is **still a type**: it goes in
`rheedium.types`, not beside the function that returns it.

The `create_*` part of this rule means constructors that build Rheedium-owned
structured data carriers. It does **not** move domain producers whose job is to
compute an existing carrier, such as `procs.create_surface_slab(...) ->
CrystalStructure`, out of the producer subpackage; the returned type is still
owned by `rheedium.types`, but the surface-building procedure belongs in
`procs`. It also does not apply to third-party object factories such as
`plots.create_phosphor_colormap(...)`, which returns a Matplotlib colormap rather
than a Rheedium type. Those functions must not define new carriers or new
carrier constructors locally.

Structured data types are **Equinox modules** (`eqx.Module`): immutable JAX PyTrees
that flow through `jit`/`grad`/`vmap`. Static, non-array metadata fields are
declared with `eqx.field(static=True)` so they are excluded from the differentiable
leaves.

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
validate inputs. These factories live in `rheedium.types` **next to the type they
build** (never in the consuming subpackage). Use a two-tier approach:

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
`pydocstyle` rules and a source-only `pydoclint` pass). Coverage is checked by
`interrogate` (`fail-under = 90`). Do **not** use ad-hoc section headers — stick
to the numpydoc sections below.

`pydoclint` is configured in `pyproject.toml` for the project's actual type
system: jaxtyping shape strings (for example `Float[Array, "H W"]`) are core
signature syntax, and current pydoclint cannot safely parse them for return-type
comparison. Therefore pydoclint still enforces argument order and required
`Returns` / `Yields` sections, but return type equality is left to `ty`,
jaxtyping, and beartype. The committed `.pydoclint-baseline` records existing
source docstring debt so new source violations fail without forcing unrelated
historical cleanup. Test docstrings are covered by
`tests/test_rheedium/test_testing_documentation.py` instead of pydoclint because
the test suite intentionally uses `Extended Summary` / `Notes` sections for
rendered validation docs.

#### Module Docstrings

Each module starts with a one-line summary, an `Extended Summary`, a
`Routine Listings` section cross-referencing every public object, and a
`Notes` section where relevant. For a package `__init__.py`, the
`Extended Summary` must always list **every submodule `.py` file** (as
`- :mod:`name`` entries with a one-line description) — when you add a new
submodule, add it to that listing in the same change:

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

**Every public object is listed in three places, and all three must agree:**

1. In its own **module**, at the **top** in the docstring's `Routine Listings`
   (the human- and Sphinx-facing API index), **and**
2. at the **bottom** in that module's `__all__` (the import-facing public
   surface), **and**
3. in the **subpackage `__init__.py`** — which re-exports the submodule's public
   API — repeated in `__init__.py`'s **own** `Routine Listings` *and* `__all__`,
   so the package-level docstring and export surface catalogue every public symbol
   the subpackage exposes.

```python
# src/rheedium/ucell/unitcell.py
"""Functions for unit cell calculations and transformations.

...

Routine Listings
----------------
:func:`reciprocal_unitcell`
    Calculate reciprocal unit cell parameters from direct cell parameters.
:func:`build_cell_vectors`
    Construct unit cell vectors from lengths and angles.
"""

# ... implementations ...

__all__ = [
    "reciprocal_unitcell",
    "build_cell_vectors",
]
```

```python
# src/rheedium/ucell/__init__.py — re-export + re-list the subpackage API
"""Unit cell and crystallographic calculations.

...

Routine Listings
----------------
:func:`reciprocal_unitcell`
    Calculate reciprocal unit cell parameters from direct cell parameters.
:func:`build_cell_vectors`
    Construct unit cell vectors from lengths and angles.
"""

from .unitcell import build_cell_vectors, reciprocal_unitcell

__all__ = ["build_cell_vectors", "reciprocal_unitcell"]
```

A symbol missing from any of the three is a defect: missing from a module's
`__all__` means it is not re-exported; missing from a `Routine Listings` means it
is undocumented; missing from the `__init__.py` lists means it is absent from the
subpackage's public API. When you add, rename, or remove a public function, update
**all three** (module `Routine Listings` + module `__all__` + the subpackage
`__init__.py` `Routine Listings` + `__all__`) in the same change.

**Keep the one-line summary identical across both `Routine Listings`.** Each
function's docstring opens with a single-sentence summary line. That exact
sentence is the description used under the function's `:func:` entry in **both**
the module's `Routine Listings` and the subpackage `__init__.py`'s `Routine
Listings` — the three must read verbatim the same (in the example above,
*"Calculate reciprocal unit cell parameters from direct cell parameters."*
appears identically in the function docstring, the module listing, and the
`__init__.py` listing). When you change a function's summary line, update both
`Routine Listings` descriptions to match it.

**Export once, from the module that owns it — no compatibility re-exports.** Each
public symbol has exactly **one** canonical export path: the module that *defines*
it, surfaced through *its own* subpackage's `__init__.py` (the three places above).
That is the only place it is exported from. Do **not** add a second export of the
same symbol from any other module or subpackage — not to preserve an old import
location, not "for convenience," not as a forwarding alias. A symbol importable
from two places is a bug magnet: the paths drift, callers split across both,
`__all__` / `Routine Listings` fall out of sync, and a re-exported PyTree can even
register its flatten/unflatten twice. The single re-export the codebase *does*
allow is the structural one already required — a subpackage `__init__.py` surfacing
the public API of **its own** submodules; that **is** the canonical path, not a
duplicate of it. Cross-subpackage access always goes through the owner's public
path (`from rheedium.types import Distribution`), never by re-publishing the symbol
from a second subpackage.

When a symbol **moves or is renamed, it moves**: update every import site to the
new canonical path and **delete** the old one in the *same* change — no shim, no
alias, no `DeprecationWarning`, no re-export left behind for old callers. This is
the project's zero-legacy policy; the only migration record is a `CHANGELOG.md`
note. Two implementations or two import paths never ship together.

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
  throughout `src/` to tie each public symbol to its tests. The test class carries
  the matching **back-reference** to this symbol (see *Test Code Conventions*), so
  the `:see:` link is **bidirectional** in the rendered Read the Docs — source →
  test and test → source.
- `Parameters` and `Returns` repeat the type and describe each item. Name the
  return values (numpydoc `name : type` form) so `pydoclint` is satisfied — and
  since functions **return a type-annotated variable rather than a bare
  expression** (see "Assign before returning"), the `Returns` name **must be that
  variable's name**, so the docstring, the body, and the signature all agree. For
  example, a body ending in `reciprocal_lengths: Float[Array, "3"] = ...; return
  reciprocal_lengths` documents its return as `reciprocal_lengths : Float[Array,
  "3"]`. For multiple returns, name each entry after the corresponding returned
  variable.
- **Mark static (non-traced) parameters.** If an argument is *static* — a
  compile-time constant excluded from JIT tracing rather than a traced value —
  its `Parameters` entry must say so, because changing it forces re-tracing /
  recompilation. This covers arguments passed through
  `jax.jit(static_argnames=...)` (or `static_argnums`), Python `int`/`str`/`bool`
  flags that drive shape or control flow, and any value that ends up in an
  `eqx.field(static=True)`. State it explicitly, e.g.:

  ```
  Parameters
  ----------
  n_modes : int
      Number of beam modes to expand (**static** — a compile-time constant;
      changing it triggers retracing/recompilation).
  ```
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
    """Validate :func:`~rheedium.ucell.wavelength_ang`.

    Covers the relativistic electron-wavelength relation across the RHEED
    accelerating-voltage range: known-value accuracy, positivity, and shape.

    :see: :func:`~rheedium.ucell.wavelength_ang`
    """

    def test_known_values(self) -> None:
        """Wavelength at 10/20/30 kV matches the analytic de Broglie value.

        Confirms ``wavelength_ang`` reproduces the relativistic electron
        wavelength: the result must be positive, finite, shaped ``(3,)``, and the
        10 kV entry must equal the textbook 0.1226 Angstrom (the *what*).

        Notes
        -----
        Evaluates ``wavelength_ang`` on a 3-vector of accelerating voltages and
        asserts shape, finiteness, positivity, and ``rtol=1e-3`` closeness to the
        analytic reference (the *how*).
        """
        energies: Float[Array, "3"] = jnp.array([10.0, 20.0, 30.0])
        wavelengths: Float[Array, "3"] = rh.ucell.wavelength_ang(energies)

        chex.assert_shape(wavelengths, (3,))
        chex.assert_tree_all_finite(wavelengths)
        assert bool(jnp.all(wavelengths > 0))
        chex.assert_trees_all_close(wavelengths[0], 0.1226, rtol=1e-3)
```

### Test Code Conventions

Tests are first-class source: `tests/**/*.py` is in **both** the Ruff and `ty`
scope and runs under live jaxtyping/beartype checking, so the same style
discipline as `src/` applies — with a few test-specific adaptations:

- **Type-hint test bodies and helpers exactly as in `src/`.** Every test method is
  `def test_*(self) -> None:`; annotate intermediate variables
  (`wavelengths: Float[Array, "3"] = rh.ucell.wavelength_ang(...)`); and the
  **assign-before-returning** rule applies to any helper that returns data — bind a
  type-annotated variable and return that name. Private/shared helpers carry full
  `jaxtyping` annotations (and `@jaxtyped(typechecker=beartype)` where arrays flow,
  as `_factories.make_*` / `_assertions.assert_*` do).
- **Document *what* and *how* on every test, class, and module (numpydoc).** A
  test's docstring is its specification, not a label. Open with the imperative
  summary line, then an `Extended Summary` paragraph stating **what** is verified
  (the property, invariant, or expected value — with units and tolerances), and a
  `Notes` section describing **how** (the inputs/fixtures, the assertion strategy,
  and the `jit`/`grad`/`vmap` variant exercised). The **module** docstring
  summarises that file's coverage; each **`Test<Symbol>` class** docstring names
  the symbol under test and the scope of its cases (see the `TestWavelength`
  example above). Stick to numpydoc sections — no ad-hoc headers, same as `src/`.
  Private helpers may keep a one-line summary.
- **Test docstrings are published to Read the Docs.** The test suite is rendered
  as a *Testing / Validation* reference in the Sphinx docs (autosummary over the
  `tests` package), so these docstrings are user-facing documentation of *what the
  library guarantees and how each guarantee is checked* — not private notes. Write
  them as reader-facing prose and keep them current; the `:see:` cross-reference
  makes the source ↔ test link navigable in **both** directions in the rendered
  docs.
- **Mirror the source layout, and make the `:see:` cross-reference
  bidirectional.** One `test_<module>.py` per source module; a `Test<Symbol>`
  class (a `chex.TestCase`) per public symbol; `test_*` methods. The link runs
  **both ways**:
  - the **source** symbol carries `:see: :class:`~.test_<module>.Test<Symbol>``
    pointing *forward* to its test class, **and**
  - the **test class** carries the counterpart `:see: :func:`~rheedium...``
    (or `:class:` / `:obj:`) pointing *back* to the symbol under test
    (see `TestWavelength` above).

  Once the test API is rendered in the docs (above), **both** references resolve
  in Read the Docs, so a reader navigates source → test *and* test → source. The
  two are a matched pair: add the back-reference whenever you add the forward one,
  and renaming either side means updating both.
- **No `__all__` or `Routine Listings` in test modules.** Tests are *not* a public
  API, so the three-places listing rule does **not** apply. A test module needs
  only a one-line summary + extended-summary docstring (no `Routine Listings`, no
  `__all__`).
- **Private helpers are `_`-prefixed and local; reused fixtures go in the shared
  helpers.** A helper used by one file is a `_`-prefixed function in it; anything
  reused across files lives in `tests/_factories.py` / `_assertions.py` /
  `_types.py`, not copy-pasted.

## Tutorial Notebooks

Tutorials live in `tutorials/` as Jupyter notebooks. Most notebooks are paired
with Jupytext percent scripts (`.ipynb` plus `.py`) so they can be edited in VS
Code/Jupyter while keeping reviewable source diffs (`[tool.jupytext]` sets
`formats = "ipynb,py:percent"`).

**Explanation lives in markdown cells, not code comments.** In `tutorials/`, do
**not** use inline `#` comments to explain what the code does — narrative,
motivation, and the physics belong in **markdown cells** (`# %% [markdown]` blocks
in the paired `.py`), which render in the docs and are strongly encouraged. Keep
code cells comment-free; the prose between them carries the teaching. This is the
notebook counterpart of the `src/` rule that explanations go in docstrings rather
than inline comments — here they go in markdown.

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

# Source docstring structure
pydoclint src/

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
matches the project's actual checker. `ty` remains the **only enforced** checker;
pyright also runs as a **non-blocking second opinion** (CI `pyright-scan` job)
that catches categories `ty`'s suppressions hide — e.g. `reportIndexIssue`. Run
it locally with:

```bash
pre-commit run --hook-stage manual pyright-scan
```

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
   - **Any new type, PyTree, or `create_*` constructor goes in `rheedium.types`**
     (see *Custom Types and PyTrees*) — never in the consuming subpackage; import
     it from there.
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

### API Evolution (zero-legacy)

The codebase carries **no compatibility layer**. When an API changes:

- **No shims, aliases, re-exports, or `DeprecationWarning`s** are kept alive for
  old import paths or signatures (see *Export once, from the module that owns it*).
- Update every call site and **delete** the old path in the *same* change; two
  implementations or import paths never ship together.
- The **only** migration record is a `CHANGELOG.md` note documenting the
  rename/removal and the new path.
- Prefer getting the API right over preserving a wrong one — a clean break with a
  changelog entry beats a forwarding alias that quietly rots.

### Versioning & release pins

`[project].version` in `pyproject.toml` is the **single source of truth** for the
package version (CalVer, e.g. `2026.6.8`). The `automatons/` experiment scripts
each pin that exact version in their PEP 723 header
(`dependencies = ["rheedium==<version>"]`) for reproducibility, so the two must
**never drift**:

- **Bumping `[project].version` and updating the `automatons/` pins is one
  atomic change.** When you change the version in `pyproject.toml`, update the
  PEP 723 `rheedium==<version>` pin in **every** `automatons/*.py` (and the GPU
  `rheedium[cuda]==<version>` form) in the *same* commit. A bumped `pyproject`
  version with a stale automaton pin — or vice versa — is a defect.
- **Use the canonical rewriter, don't hand-edit.** `automatons/bump_pin.py` (a
  PEP 723 script) reads the version from `pyproject.toml` and rewrites every
  automaton header to match; it is idempotent (a second run is a no-op). Run it on
  every bump rather than editing headers by hand, so all pins stay identical and in
  sync with `pyproject`.
- **A guard test enforces it.** A test asserts every `automatons/*.py` pins exactly
  the current `[project].version`; it fails CI if any pin drifts from `pyproject`.
- This only constrains the *released, pinned* form. Local development against
  unpublished changes still uses `uv run --with-editable . automatons/<x>.py`,
  which overrides the pin (see the automatons plan §3).

### Building and Releasing

Packaging is **uv end-to-end**: the build backend is `uv_build` (see
`[build-system]` in `pyproject.toml`) and releases go out with `uv publish` —
no `setuptools`, `build`, or `twine` anywhere. CI's `build` job runs
`uv build` and then installs the built wheel into a clean ephemeral
environment and imports it (`uv run --isolated --no-project --with
dist/rheedium-*.whl python -c "import rheedium"`), which verifies wheel
contents, metadata, and dependency resolution in one uv-native step.

```bash
# Build the sdist and wheel into dist/
uv build

# Smoke-test the built wheel in a clean environment
uv run --isolated --no-project --with dist/rheedium-*.whl \
  python -c "import rheedium; print(rheedium.__version__)"

# Publish to PyPI (uses a PyPI API token)
UV_PUBLISH_TOKEN=<pypi-token> uv publish
```

Release checklist:

1. Bump `[project].version` (CalVer) **and** run `automatons/bump_pin.py` in
   the same commit (see *Versioning & release pins* above); update
   `CHANGELOG.md`.
2. Run the full wall (`ruff check src/ tests/`, `pydoclint src/`, `ty check`,
   `pre-commit run --all-files`, `pytest`) at the release commit.
3. `uv build` from a clean tree; run the wheel smoke-test above and confirm
   the metadata carries `License-Expression: MIT`.
4. Tag the release commit (`v<version>`), then `uv publish`.

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
