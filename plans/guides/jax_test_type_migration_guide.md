# JAX Test Type Migration Guide

Scope: `janssen`, `rheedium`, `ptyrodactyl`

Stack:

```text
ty         Static Python/interface checking.
jaxtyping  Array, dtype, shape, and PyTree contracts in annotations.
beartype   Runtime enforcement of jaxtyping contracts.
chex       Runtime JAX testing: shapes, dtypes, trees, devices, variants, fake transforms, and backend checks.
pytest     Execution harness.
```

Core policy:

```text
Type-check tests.
Type fixtures, factories, fakes, and assertion helpers first.
Treat public API tests as typed client programs.
Keep Chex central; do not replace Chex assertions with annotations.
Use backend-specific array types by default:
  JAX array   -> Float[Array, "x y"]
  NumPy array -> Float[NDArray, "x y"]
Use ArrayLike only for functions that intentionally accept coercible inputs.
```

This guide assumes the project already uses strong type annotations in main code and weaker annotations in tests. The migration is therefore not a rewrite. It is a staged conversion of tests from untyped dynamic clients into typed clients of the package.

---

## 1. Mental model

A JAX test suite has four distinct correctness layers.

| Layer | Tool | What it proves | What it does not prove |
|---|---|---|---|
| Python interface correctness | `ty` | Names, signatures, Protocols, fixtures, fakes, helper APIs, ordinary Python typing | JAX array shape/dtype constraints |
| Shape/dtype contract documentation | `jaxtyping` | Expected array backend, dtype family, shape names, PyTree structure annotations | Runtime truth unless checked |
| Runtime annotation enforcement | `beartype` through `jaxtyping` | Function arguments and returns satisfy jaxtyping annotations at checked call boundaries | All intermediate runtime facts |
| JAX runtime behavior | `chex` | Concrete shape, dtype, tree, finiteness, device, sharding, transform, backend behavior | Static interface correctness |

Use the layers together:

```python
import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def test_encoder_output() -> None:
    x: Float[Array, "batch dim"] = jnp.ones((4, 8), dtype=jnp.float32)

    y: Float[Array, "batch hidden"] = encode(x)

    chex.assert_shape(y, (4, 16))
    chex.assert_type(y, jnp.float32)
    chex.assert_tree_all_finite(y)
```

The annotation makes the test a statically valid client. Chex verifies the computed runtime value.

---

## 2. Target end state

```text
src/
  Strongly typed.
  Public JAX APIs use Float[Array, ...], Int[Array, ...], Bool[Array, ...], etc.
  NumPy-only APIs use Float[NDArray, ...], etc.
  Runtime shape/type checking is enabled for high-value public APIs and numerical kernels.

tests/
  Included in ty.
  Test functions return None.
  Fixtures, factories, fake objects, and assertion helpers are typed.
  Public API tests act as typed client programs.
  Chex assertions are retained for concrete runtime facts.
  Dynamic pytest/Chex edges use local casts or local ignores only.
  Broad exclusions are forbidden.
```

The intended final guarantee:

```text
The test suite checks behavior.
The test suite type-checks as client code.
The test suite exercises runtime array contracts.
The test suite verifies device/transform behavior explicitly where relevant.
```

---

## 3. Type vocabulary

Create a shared file, for example `tests/_types.py`.

```python
# tests/_types.py

from typing import TypeAlias

from jax import Array
from jaxtyping import Bool, Float, Float32, Int, PRNGKeyArray, PyTree
from numpy.typing import NDArray

# Random keys
Key: TypeAlias = PRNGKeyArray

# Generic JAX arrays
JaxScalar: TypeAlias = Float[Array, ""]
JaxVector: TypeAlias = Float[Array, "x"]
JaxMatrix: TypeAlias = Float[Array, "x y"]
JaxTensor3: TypeAlias = Float[Array, "x y z"]
JaxAnyFloat: TypeAlias = Float[Array, "..."]

# Precision-specific JAX arrays
JaxF32Scalar: TypeAlias = Float32[Array, ""]
JaxF32Vector: TypeAlias = Float32[Array, "x"]
JaxF32Matrix: TypeAlias = Float32[Array, "x y"]

# Generic NumPy arrays
NpScalar: TypeAlias = Float[NDArray, ""]
NpVector: TypeAlias = Float[NDArray, "x"]
NpMatrix: TypeAlias = Float[NDArray, "x y"]
NpTensor3: TypeAlias = Float[NDArray, "x y z"]
NpAnyFloat: TypeAlias = Float[NDArray, "..."]

# Precision-specific NumPy arrays
NpF32Scalar: TypeAlias = Float32[NDArray, ""]
NpF32Vector: TypeAlias = Float32[NDArray, "x"]
NpF32Matrix: TypeAlias = Float32[NDArray, "x y"]

# Masks and labels
JaxMask: TypeAlias = Bool[Array, "x"]
NpMask: TypeAlias = Bool[NDArray, "x"]

JaxLabels: TypeAlias = Int[Array, "x"]
NpLabels: TypeAlias = Int[NDArray, "x"]

# Common ML shapes
Batch: TypeAlias = Float[Array, "batch dim"]
Logits: TypeAlias = Float[Array, "batch classes"]
Probs: TypeAlias = Float[Array, "batch classes"]
Labels: TypeAlias = Int[Array, "batch"]

# PyTrees
JaxParams: TypeAlias = PyTree[Float[Array, "..."]]
JaxGrads: TypeAlias = PyTree[Float[Array, "..."]]
NpParams: TypeAlias = PyTree[Float[NDArray, "..."]]
```

Rules:

```text
Use Float[...] when any floating dtype is acceptable.
Use Float32[...] or Float64[...] when precision is part of the contract.
Use Array for JAX arrays.
Use NDArray for NumPy arrays.
Use Array | NDArray only when the API deliberately supports both families.
Avoid ArrayLike except at explicit coercion boundaries.
```

Example of an intentional dual-backend contract:

```python
from typing import TypeAlias

from jax import Array
from jaxtyping import Float
from numpy.typing import NDArray

JaxOrNpMatrix: TypeAlias = Float[Array | NDArray, "x y"]


def accepts_either_backend(x: JaxOrNpMatrix) -> JaxOrNpMatrix:
    return x
```

Example of an intentional conversion boundary:

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from numpy.typing import NDArray


def np_to_jax(x: Float[NDArray, "x y"]) -> Float[Array, "x y"]:
    return jnp.asarray(x)
```

Do not blur this into a permissive `ArrayLike` signature unless the function really accepts scalars, NumPy arrays, JAX arrays, and other JAX-coercible inputs as part of its public contract.

---

## 4. Configure `ty` to check tests

Start with inclusion, not strictness perfection.

```toml
# pyproject.toml

[tool.ty.src]
include = [
  "src",
  "tests",
]

[tool.ty.environment]
python-version = "3.12"
root = ["./src"]
```

Set `python-version` to the project’s real supported version.

Use narrow overrides only where the test tree intentionally contains malformed code, generated files, snapshots, or intentionally unresolved imports.

```toml
[[tool.ty.overrides]]
include = [
  "tests/fixtures/bad_inputs/**",
  "tests/snapshots/**",
]

[tool.ty.overrides.rules]
possibly-unresolved-reference = "warn"
```

Bad migration end state:

```toml
[tool.ty.src]
exclude = ["tests"]
```

Acceptable targeted suppression:

```python
# Chex/PyTest dynamic edge. Keep the cast or ignore local.
variant = cast(Variant, self.variant)
```

or:

```python
monkeypatch.setattr(module, "dynamic_attr", fake_value)  # type: ignore[attr-defined]
```

Never use file-wide ignores as the default migration mechanism.

---

## 5. Migration order

Do not start by annotating every assertion. Start where type information propagates.

```text
1. Add tests to ty's checked source set.
2. Add -> None to test functions touched by current work.
3. Type fixtures.
4. Type factories.
5. Type fake objects and mocks through Protocols.
6. Add shared jaxtyping aliases.
7. Type Chex assertion helpers.
8. Type public API tests as sample user programs.
9. Add jaxtyping/beartype runtime checking at selected function boundaries.
10. Isolate Chex fake-device and fake-transform machinery.
11. Add CI gates for ty and pytest.
12. Optionally add a second static checker only for public API typing tests.
```

Priority table:

| Area | Priority | Reason |
|---|---:|---|
| Test helpers | Highest | One annotation fixes many tests. |
| Fixtures | Highest | Fixture types propagate into every dependent test. |
| Factories | Highest | Factories determine the actual test data contract. |
| Fake objects | High | Fakes often drift from real protocols. |
| Public API tests | High | These are typed user programs. |
| Chex assertion helpers | High | They centralize runtime expectations. |
| Ordinary test bodies | Medium | Useful, but lower leverage than fixtures/helpers. |
| Mock-heavy pytest glue | Low | Permit narrow local escapes. |

---

## 6. No-regret first pass

Convert this:

```python
def test_loss_decreases():
    ...
```

into this:

```python
def test_loss_decreases() -> None:
    ...
```

Then type fixtures:

```python
import pytest
import jax
from jaxtyping import PRNGKeyArray


@pytest.fixture
def key() -> PRNGKeyArray:
    return jax.random.key(0)
```

Then type factories:

```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def make_batch(batch: int, dim: int) -> Float[Array, "{batch} {dim}"]:
    return jnp.zeros((batch, dim), dtype=jnp.float32)
```

Then use factories in tests:

```python
def test_encode_shape() -> None:
    x = make_batch(batch=4, dim=8)

    y = encode(x)

    chex.assert_shape(y, (4, 16))
```

The factory now carries the shape contract. The test does not need a redundant local annotation unless it improves clarity.

---

## 7. Fixtures

Fixtures should describe what they return. Avoid untyped fixture blobs.

```python
import pytest
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


@pytest.fixture
def small_batch() -> Float[Array, "batch dim"]:
    return jnp.ones((4, 8), dtype=jnp.float32)
```

Parameterized fixtures:

```python
import pytest


@pytest.fixture(params=[2, 4, 8])
def batch_size(request: pytest.FixtureRequest) -> int:
    return int(request.param)
```

Generator fixtures:

```python
from collections.abc import Iterator

import chex
import pytest


@pytest.fixture
def fake_pmap() -> Iterator[None]:
    with chex.fake_pmap():
        yield
```

Bad:

```python
@pytest.fixture
def model():
    return make_model(...)
```

Better:

```python
@pytest.fixture
def model() -> Model:
    return make_model(...)
```

---

## 8. Factories

Factories should encode shape, dtype, and backend.

```python
import numpy as np
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Float32
from numpy.typing import NDArray


def make_jax_matrix(x: int, y: int) -> Float32[Array, "{x} {y}"]:
    return jnp.zeros((x, y), dtype=jnp.float32)


def make_np_matrix(x: int, y: int) -> Float32[NDArray, "{x} {y}"]:
    return np.zeros((x, y), dtype=np.float32)
```

Avoid factories that erase backend identity:

```python
# Bad for this codebase convention.
def make_matrix(x: int, y: int):
    return jnp.zeros((x, y))
```

Prefer explicit backend-specific names:

```python
def make_jax_logits(batch: int, classes: int) -> Float32[Array, "{batch} {classes}"]:
    return jnp.zeros((batch, classes), dtype=jnp.float32)


def make_np_logits(batch: int, classes: int) -> Float32[NDArray, "{batch} {classes}"]:
    return np.zeros((batch, classes), dtype=np.float32)
```

---

## 9. Public API tests as typed clients

A public API test should look like code a user could write.

```python
import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from your_package import step


def test_step_public_api_accepts_jax_matrix() -> None:
    x: Float[Array, "batch dim"] = jnp.ones((4, 8), dtype=jnp.float32)

    y: Float[Array, "batch dim"] = step(x)

    chex.assert_shape(y, (4, 8))
    chex.assert_type(y, jnp.float32)
    chex.assert_tree_all_finite(y)
```

NumPy-specific public API test:

```python
import chex
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from your_package import preprocess_numpy


def test_preprocess_numpy_public_api() -> None:
    x: Float[NDArray, "batch dim"] = np.ones((4, 8), dtype=np.float32)

    y: Float[NDArray, "batch dim"] = preprocess_numpy(x)

    chex.assert_shape(y, (4, 8))
    chex.assert_type(y, np.float32)
```

Conversion-boundary test:

```python
import chex
import numpy as np
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from numpy.typing import NDArray

from your_package import np_to_jax


def test_np_to_jax_public_api() -> None:
    x: Float[NDArray, "batch dim"] = np.ones((4, 8), dtype=np.float32)

    y: Float[Array, "batch dim"] = np_to_jax(x)

    chex.assert_shape(y, (4, 8))
    chex.assert_type(y, jnp.float32)
```

---

## 10. Chex remains first-class

Do not replace this:

```python
chex.assert_shape(y, (4, 8))
chex.assert_type(y, jnp.float32)
chex.assert_tree_all_finite(y)
```

with only this:

```python
y: Float[Array, "batch dim"] = f(x)
```

The annotation is a static/client-side contract. Chex verifies the actual runtime value.

Use Chex for:

```text
shape checks
rank checks
dtype checks
finite-value checks
tree structure checks
tree equality/closeness checks
device/host placement checks
sharding checks
backend restrictions
JIT/non-JIT variants
PMAP variants
fake_jit / fake_pmap debugging
multi-CPU fake-device setup
jittable assertions via chexify
```

Create typed Chex helpers for repeated checks.

```python
import chex
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float32, jaxtyped


@jaxtyped(typechecker=beartype)
def assert_float32_batch(
    x: Float32[Array, "batch dim"],
    *,
    batch: int,
    dim: int,
) -> None:
    chex.assert_shape(x, (batch, dim))
    chex.assert_type(x, jnp.float32)
    chex.assert_tree_all_finite(x)
```

Use the helper:

```python
def test_encoder_output_shape() -> None:
    x = make_jax_matrix(4, 8)

    y = encoder(x)

    assert_float32_batch(y, batch=4, dim=16)
```

The helper boundary is runtime-checked by `jaxtyping`/`beartype`; the helper body verifies concrete Chex facts.

---

## 11. Runtime type checking with jaxtyping and beartype

Use runtime checking on high-value boundaries, not every tiny function.

Good targets:

```text
public API functions
numerical kernels with nontrivial shape contracts
loss functions
metric functions
sampling functions
optimizer/update steps
test factories
Chex assertion helpers
PyTree-processing helpers
```

Poor targets:

```text
trivial one-line helpers
mocking glue
snapshot plumbing
pytest metaprogramming
intentionally invalid fixture modules
```

Pattern:

```python
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def center(x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
    return x - x.mean(axis=0)
```

Test-time checking can also be applied with the jaxtyping pytest hook.

For `janssen`:

```toml
[tool.pytest.ini_options]
addopts = "--jaxtyping-packages=janssen,beartype.beartype"
```

For `rheedium`:

```toml
[tool.pytest.ini_options]
addopts = "--jaxtyping-packages=rheedium,beartype.beartype"
```

For `ptyrodactyl`:

```toml
[tool.pytest.ini_options]
addopts = "--jaxtyping-packages=ptyrodactyl,beartype.beartype"
```

After test helpers are clean, consider extending the hook to importable test utility modules:

```toml
[tool.pytest.ini_options]
addopts = "--jaxtyping-packages=janssen,tests,beartype.beartype"
```

Avoid `from __future__ import annotations` in modules where annotations must be available for runtime checking. Stringified annotations and postponed annotations can interfere with runtime type checking.

---

## 12. JIT and runtime checking

For JIT-compiled functions, jaxtyping checks shape and dtype when JAX traces the function. The compiled code path does not carry those checks.

```python
import jax
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=beartype)
def normalize(x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
    return x / x.sum(axis=-1, keepdims=True)
```

Check decorator order in real code when combining `jax.jit`, `jax.pmap`, `jax.custom_jvp`, `eqx.filter_jit`, or other decorators. Keep one local convention per repo.

Practical rule:

```text
Use runtime jaxtyping checks to catch bad call boundaries.
Use Chex to verify the values produced under transformed execution.
```

---

## 13. Chex `chexify` and jittable assertions

Static shape/dtype assertions are often available during tracing. Concrete value assertions inside JAX-transformed code need Chex’s jittable assertion machinery.

Pattern:

```python
import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


@chex.chexify
@jax.jit
def safe_log1p_abs(x: Float[Array, "batch"]) -> Float[Array, "batch"]:
    chex.assert_shape(x, (None,))
    chex.assert_tree_all_finite(x)
    return jnp.log1p(jnp.abs(x))


def test_safe_log1p_abs() -> None:
    y = safe_log1p_abs(jnp.ones((4,), dtype=jnp.float32))
    safe_log1p_abs.wait_checks()
    chex.assert_shape(y, (4,))
```

Rule:

```text
Use normal Chex assertions outside JAX transforms.
Use chexify / with_jittable_assertions for assertions that must execute inside transformed code.
Wait for async checks when needed.
```

---

## 14. Device and fake-device testing

There are three levels of device tests.

```text
fake_pmap / fake_jit tests
  Fast local logic tests.
  Good for structural coverage.
  Not proof of real multi-device semantics.

set_n_cpu_devices tests
  Local pmap-style smoke tests on CPU.
  Good for catching device-count and pmap-shape assumptions.
  Must be configured before XLA backends initialize.

real GPU/TPU tests
  Actual backend behavior.
  Needed for backend-specific precision, sharding, placement, memory, and collective behavior.
```

### `set_n_cpu_devices`

`chex.set_n_cpu_devices(n)` must run before JAX initializes XLA backends. Do not hide it inside an ordinary fixture that may run too late.

Bad:

```python
@pytest.fixture
def fake_devices() -> None:
    chex.set_n_cpu_devices(8)  # Too late if JAX already initialized.
```

Better:

```python
# tests/conftest.py
# Keep this at the top, before importing the package under test.

import chex

chex.set_n_cpu_devices(8)
```

Best CI structure:

```text
Run CPU multi-device tests in a fresh Python process.
Set fake CPU devices before importing the package under test.
Keep real GPU/TPU tests in separate jobs.
Skip real-device tests when the device is unavailable.
```

### `fake_pmap`

`chex.fake_pmap()` patches `jax.pmap` with `jax.vmap`. This is valuable for fast local tests and debugging. It is not equivalent to real parallel execution.

Typed fixture:

```python
from collections.abc import Iterator

import chex
import pytest


@pytest.fixture
def fake_pmap() -> Iterator[None]:
    with chex.fake_pmap():
        yield
```

Usage:

```python
import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def test_pmapped_update_under_fake_pmap(fake_pmap: None) -> None:
    x: Float[Array, "devices batch dim"] = jnp.ones((4, 2, 8), dtype=jnp.float32)

    y = pmapped_update(x)

    chex.assert_shape(y, (4, 2, 8))
    chex.assert_tree_all_finite(y)
```

Use `fake_pmap` for logic. Use real or fake CPU devices for pmap structure. Use real accelerators for backend truth.

### Backend restrictions

Use Chex backend restrictions when a test must fail if compilation occurs on the wrong backend.

```python
import chex
import jax
import jax.numpy as jnp


def test_cpu_only_path() -> None:
    with chex.restrict_backends(allowed=["cpu"]):
        y = jax.jit(cpu_only_fn)(jnp.ones((4,), dtype=jnp.float32))

    chex.assert_shape(y, (4,))
```

---

## 15. Chex variants

Chex variants let the same test run under transformed and untransformed execution.

Use variants for:

```text
jitted vs non-jitted behavior
device_put vs host behavior
pmap variants
regression tests around tracing behavior
functions that should behave identically across transform modes
```

Dynamic Chex test classes may need small local casts because `self.variant` is dynamic.

```python
from collections.abc import Callable
from typing import TypeVar, cast

import chex
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

F = TypeVar("F", bound=Callable[..., object])
Variant = Callable[[F], F]


class NormalizeTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_normalize_variant(self) -> None:
        def f(x: Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
            return normalize(x)

        variant = cast(Variant, self.variant)
        vf = variant(f)

        x: Float[Array, "batch dim"] = jnp.ones((4, 8), dtype=jnp.float32)
        y = vf(x)

        chex.assert_shape(y, (4, 8))
        chex.assert_tree_all_finite(y)
```

This is an acceptable local escape. Do not disable type checking for the whole file just because Chex variants are dynamic.

---

## 16. PyTrees

Use `PyTree` for parameters, gradients, optimizer state, model state, and nested batches.

Basic aliases:

```python
from typing import TypeAlias

from jax import Array
from jaxtyping import Float, PyTree

Params: TypeAlias = PyTree[Float[Array, "..."]]
Grads: TypeAlias = PyTree[Float[Array, "..."]]
```

Runtime structure check:

```python
import chex


def test_grads_match_params(params: Params) -> None:
    grads: Grads = compute_grads(params)

    chex.assert_trees_all_equal_structs(grads, params)
```

Path-dependent leaf shape check:

```python
from beartype import beartype
from jax import Array
from jaxtyping import Float, PyTree, jaxtyped


@jaxtyped(typechecker=beartype)
def assert_matching_param_grad_shapes(
    params: PyTree[Float[Array, "?leaf"], "T"],
    grads: PyTree[Float[Array, "?leaf"], "T"],
) -> None:
    pass
```

Use Chex to verify concrete tree facts:

```python
chex.assert_trees_all_equal_structs(params, grads)
chex.assert_trees_all_equal_shapes(params, grads)
chex.assert_trees_all_equal_dtypes(params, grads)
```

---

## 17. Protocols for fakes and mocks

Fakes should be typed against the interface they are replacing.

Bad:

```python
class FakeStore:
    def get(self, key):
        return self.data[key]
```

Better:

```python
from typing import Protocol


class Store(Protocol):
    def get(self, key: str) -> bytes: ...


class FakeStore:
    def __init__(self, data: dict[str, bytes]) -> None:
        self.data = data

    def get(self, key: str) -> bytes:
        return self.data[key]
```

Use the protocol in tests:

```python
def test_loads_from_store() -> None:
    store: Store = FakeStore({"x": b"payload"})

    result = load(store, "x")

    assert result == b"payload"
```

Mocks and monkeypatching are where tests most often lie. Prefer a fake object that implements a protocol over an untyped mock.

---

## 18. Parameterized tests

Bad:

```python
@pytest.mark.parametrize("shape", [(2, 3), (4, 5)])
def test_layer(shape):
    x = jnp.ones(shape)
    ...
```

Better:

```python
from typing import TypeAlias

import pytest

Case: TypeAlias = tuple[int, int]

CASES: tuple[Case, ...] = (
    (2, 3),
    (4, 5),
)


@pytest.mark.parametrize(("batch", "dim"), CASES)
def test_layer(batch: int, dim: int) -> None:
    x = make_jax_matrix(batch, dim)

    y = layer(x)

    chex.assert_shape(y, (batch, dim))
```

For more complex cases:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerCase:
    batch: int
    dim: int
    hidden: int


CASES: tuple[LayerCase, ...] = (
    LayerCase(batch=2, dim=3, hidden=5),
    LayerCase(batch=4, dim=8, hidden=16),
)


@pytest.mark.parametrize("case", CASES)
def test_layer_case(case: LayerCase) -> None:
    x = make_jax_matrix(case.batch, case.dim)

    y = layer(x, hidden=case.hidden)

    chex.assert_shape(y, (case.batch, case.hidden))
```

---

## 19. CI gates

Initial gate:

```yaml
- name: Type check
  run: uv run ty check

- name: Test
  run: uv run pytest
```

Better split during migration:

```yaml
- name: Type check source
  run: uv run ty check src

- name: Type check tests
  run: uv run ty check tests

- name: Runtime tests
  run: uv run pytest
```

Recommended final matrix:

```yaml
- name: Type check source
  run: uv run ty check src

- name: Type check tests
  run: uv run ty check tests

- name: Unit tests
  run: uv run pytest tests/unit

- name: CPU multi-device tests
  run: uv run pytest tests/multidevice_cpu

- name: Public API typing tests
  run: uv run ty check tests/api_typing
```

Optional later-stage canary:

```yaml
- name: Public API pyright canary
  run: uv run pyright tests/api_typing
```

Do not run multiple checkers over all implementation code unless it earns its keep. Multi-checker effort is highest-value on public API typed-client tests.

---

## 20. Repository rollout plan

Apply the same sequence to `janssen`, `rheedium`, and `ptyrodactyl`.

### PR 1: Include tests in `ty`

```text
Add tests to [tool.ty.src].include.
Add narrow overrides only for known invalid fixtures/snapshots.
Do not attempt to fix all failures in this PR if the noise is large.
Record the current failure categories.
```

### PR 2: Annotate test entry points

```text
Add -> None to touched test functions.
Do not churn every file mechanically unless the repo is small.
Prefer opportunistic conversion plus helper-first conversion.
```

### PR 3: Add shared test type aliases

```text
Create tests/_types.py.
Add JAX and NumPy aliases separately.
Remove ArrayLike from default test vocabulary.
```

### PR 4: Type factories and fixtures

```text
Convert data factories to Float[Array, ...] or Float[NDArray, ...].
Convert key fixtures to PRNGKeyArray.
Convert model/config fixtures to concrete project types.
```

### PR 5: Type Chex assertion helpers

```text
Extract repeated shape/dtype/finiteness/tree checks.
Wrap high-value helpers with @jaxtyped(typechecker=beartype).
Keep Chex assertions inside the helpers.
```

### PR 6: Type public API tests

```text
For each public API, add or convert tests that act as typed user programs.
Explicitly annotate inputs and outputs.
Use Chex to verify runtime facts.
```

### PR 7: Device and transform test cleanup

```text
Move fake_pmap/fake_jit fixtures into typed helpers.
Move set_n_cpu_devices to process-start setup.
Separate fake_pmap logic tests from real pmap/device tests.
Add backend skips or backend assertions explicitly.
```

### PR 8: Runtime type-checking hook

```text
Enable --jaxtyping-packages=<package>,beartype.beartype in pytest.
Start with the package only.
Add tests utilities later if clean.
Remove future annotations from runtime-checked modules if necessary.
```

### PR 9: CI enforcement

```text
Make ty check tests required.
Make pytest with jaxtyping/beartype required.
Optionally add public API pyright/mypy canary later.
```

---

## 21. Anti-patterns

### Broadly untyped tests

```python
def test_model():
    x = make_x()
    y = model(x)
    assert y.shape == (4, 8)
```

Problem: the test may run while the test client contract rots.

Better:

```python
def test_model() -> None:
    x: Float[Array, "batch dim"] = make_jax_matrix(4, 8)
    y: Float[Array, "batch dim"] = model(x)
    chex.assert_shape(y, (4, 8))
```

### Replacing Chex with annotations

```python
# Insufficient.
y: Float[Array, "batch dim"] = model(x)
```

Better:

```python
y: Float[Array, "batch dim"] = model(x)
chex.assert_shape(y, (4, 8))
chex.assert_type(y, jnp.float32)
chex.assert_tree_all_finite(y)
```

### Erasing backend identity

```python
# Bad for this convention.
def f(x: ArrayLike) -> Array:
    ...
```

Better:

```python
def f(x: Float[Array, "x y"]) -> Float[Array, "x y"]:
    ...
```

or, for NumPy:

```python
def f_np(x: Float[NDArray, "x y"]) -> Float[NDArray, "x y"]:
    ...
```

### Hiding dynamic test machinery globally

Bad:

```python
# ty: ignore-file
```

Better:

```python
variant = cast(Variant, self.variant)
```

### Late fake-device setup

Bad:

```python
def test_pmap() -> None:
    chex.set_n_cpu_devices(8)
    ...
```

Better:

```python
# tests/conftest.py, before package/JAX backend initialization
import chex

chex.set_n_cpu_devices(8)
```

---

## 22. Completion checklist

A repo is migrated when all of these are true:

```text
[ ] tests are included in ty
[ ] test functions touched by normal development return None
[ ] shared JAX/NumPy type aliases exist
[ ] fixtures are typed
[ ] factories are typed
[ ] fake objects implement typed Protocols or concrete interfaces
[ ] repeated Chex assertions are extracted into typed helpers
[ ] public API tests are typed client programs
[ ] ArrayLike is not used as the default array annotation
[ ] Array and NDArray contracts are kept separate
[ ] Chex fake_pmap/fake_jit usage is isolated in typed helpers/fixtures
[ ] set_n_cpu_devices runs before XLA backend initialization where used
[ ] real-device tests are separate from fake-device tests
[ ] jaxtyping/beartype runtime checking is enabled for high-value boundaries
[ ] broad test exclusions are removed
[ ] CI gates ty over source and tests
[ ] CI gates pytest with runtime checking enabled
```

---

## 23. Minimal templates

### `tests/_types.py`

```python
from typing import TypeAlias

from jax import Array
from jaxtyping import Bool, Float, Float32, Int, PRNGKeyArray, PyTree
from numpy.typing import NDArray

Key: TypeAlias = PRNGKeyArray

JaxMatrix: TypeAlias = Float[Array, "x y"]
JaxF32Matrix: TypeAlias = Float32[Array, "x y"]
NpMatrix: TypeAlias = Float[NDArray, "x y"]
NpF32Matrix: TypeAlias = Float32[NDArray, "x y"]

JaxMask: TypeAlias = Bool[Array, "x"]
JaxLabels: TypeAlias = Int[Array, "x"]

JaxParams: TypeAlias = PyTree[Float[Array, "..."]]
JaxGrads: TypeAlias = PyTree[Float[Array, "..."]]
```

### `tests/_factories.py`

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float32, PRNGKeyArray
from numpy.typing import NDArray


def make_key(seed: int = 0) -> PRNGKeyArray:
    return jax.random.key(seed)


def make_jax_matrix(x: int, y: int) -> Float32[Array, "{x} {y}"]:
    return jnp.zeros((x, y), dtype=jnp.float32)


def make_np_matrix(x: int, y: int) -> Float32[NDArray, "{x} {y}"]:
    return np.zeros((x, y), dtype=np.float32)
```

### `tests/_assertions.py`

```python
import chex
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float32, jaxtyped


@jaxtyped(typechecker=beartype)
def assert_jax_f32_matrix(
    x: Float32[Array, "rows cols"],
    *,
    rows: int,
    cols: int,
) -> None:
    chex.assert_shape(x, (rows, cols))
    chex.assert_type(x, jnp.float32)
    chex.assert_tree_all_finite(x)
```

### Public API test

```python
import chex
from jax import Array
from jaxtyping import Float32

from tests._factories import make_jax_matrix
from your_package import encode


def test_encode_public_api() -> None:
    x = make_jax_matrix(4, 8)

    y: Float32[Array, "batch hidden"] = encode(x)

    chex.assert_shape(y, (4, 16))
```

---

## 24. References

Official documentation used while drafting this guide:

- Astral `ty` configuration: <https://docs.astral.sh/ty/reference/configuration/>
- jaxtyping array annotations: <https://docs.kidger.site/jaxtyping/api/array/>
- jaxtyping runtime type checking: <https://docs.kidger.site/jaxtyping/api/runtime-type-checking/>
- jaxtyping PyTree annotations: <https://docs.kidger.site/jaxtyping/api/pytree/>
- jaxtyping FAQ on static checking and `jax.jit`: <https://docs.kidger.site/jaxtyping/faq/>
- Chex documentation: <https://chex.readthedocs.io/>
- Chex API reference: <https://chex.readthedocs.io/en/latest/api.html>
- JAX typing documentation: <https://docs.jax.dev/en/latest/jax.typing.html>
