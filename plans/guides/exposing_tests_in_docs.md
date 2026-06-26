# Exposing the Test Suite in the Rendered Docs

Scope: `rheedium`

What this covers:

```text
Render the test suite as a "Test Reference" in Read the Docs, next to the
API Reference, so:
  - every test documents what it guarantees and how (numpydoc docstrings),
  - the source :see: -> test links resolve, and a test :see: -> source link
    closes the loop (bidirectional navigation),
  - the suite reads as a validation reference, not private scaffolding.
```

Companion docs:

- Conventions (the rules): [CONTRIBUTING.md](../../CONTRIBUTING.md) → *Test Code
  Conventions*.
- Gated rollout (the work plan): [testing_docs.md](../partial/testing_docs.md).

This guide is the operational reference: the structure, the build, and the rST
gotchas you hit when autodoc starts rendering test docstrings.

---

## 1. Structure

The Test Reference mirrors the API Reference and lives under the **same**
"API Reference" caption in the nav:

```text
docs/source/
  index.rst                 # toctree (caption "API Reference"): api/index + tests/index
  api/index.rst             # "API Reference"  (rheedium.<pkg> via automodule)
  api/<pkg>.rst             # one per subpackage
  tests/index.rst           # "Test Reference" (landing page + toctree)
  tests/<pkg>.rst           # one per subpackage: automodule each test_*.py module
```

`tests/<pkg>.rst` mirrors `api/<pkg>.rst` — it `automodule`s every
`tests.test_rheedium.test_<pkg>.test_*` module:

```rst
rheedium.ucell tests
====================

Validation and regression tests for :mod:`rheedium.ucell`.

.. automodule:: tests.test_rheedium.test_ucell.test_unitcell
   :members:
   :undoc-members:
   :show-inheritance:
```

**Why it imports:** every test subpackage has `__init__.py`, and `conf.py`
puts the project root on `sys.path`, so `tests.test_rheedium.test_<pkg>.test_*`
is importable by autodoc. The docs build env must include the `test` extra
(`chex`, `hypothesis`, `absl-testing`) — `uv sync --extra dev` covers it.

---

## 2. Bidirectional `:see:`

The cross-reference is a **matched pair** — add both halves together:

```python
# src/rheedium/ucell/unitcell.py
def wavelength_ang(...):
    """...
    :see: :class:`~.test_ucell.TestWavelength`     # source -> test (forward)
    """

# tests/test_rheedium/test_ucell/test_unitcell.py
class TestWavelength(chex.TestCase):
    """Validate :func:`~rheedium.ucell.wavelength_ang`.
    ...
    :see: :func:`~rheedium.ucell.wavelength_ang`    # test -> source (back)
    """
```

Once **both** the API Reference and the Test Reference are rendered, both links
resolve and a reader navigates source → test → source. A forward `:see:` with no
rendered test target (the old state) is a dead link; the Test Reference is what
makes it live.

---

## 3. Docstrings: what + how

Test docstrings are now **published documentation**, not labels. Per CONTRIBUTING:

- **Summary line** — what this test asserts.
- **`Extended Summary`** — *what* is verified (property/invariant/value, units,
  tolerances).
- **`Notes`** — *how* (inputs/fixtures, assertion strategy, the `jit`/`grad`/`vmap`
  variant). Stick to numpydoc sections; no ad-hoc headers.

The module docstring summarises the file's coverage; each `Test<Symbol>` class
docstring names the symbol under test + carries the back-`:see:`.

---

## 4. rST gotchas (autodoc renders docstrings as reStructuredText)

### 4.1 `|x|` is a substitution reference — escape it

A docstring with absolute-value / magnitude bars — `|G|`, `|k_out|`, `|T|`,
`|amplitude|^2` — makes docutils look for an undefined **substitution** `|G|` and
emit `ERROR: Undefined substitution referenced: "G"`. Fixes:

```text
Broken:   """|T| <= 1 everywhere."""
Fix A:    r"""\|T\| <= 1 everywhere."""        # escape the pipes (raw string)
Fix B:    """:math:`|T|` <= 1 everywhere."""   # math role (no escaping, no r)
Fix C:    """``|T|`` <= 1 everywhere."""        # inline literal (mind adjacency)
```

**Does adding `r` alone fix it? No.** `r` only changes how *Python* reads the
string — it stops `\|` from being an "invalid escape sequence" warning. It does
**nothing** to how *rST* reads `|G|`. The working combo is **`r` + `\|`**: the
`\|` is what tells rST to render a literal pipe; the `r` is only there so the
backslash is legal Python. `:math:` / `` `` `` need neither `r` nor escaping.

Caveats:

- Escaping a **non-raw** docstring (`"""...\|G\|..."""`) raises a Python
  `SyntaxWarning: invalid escape sequence '\|'` — always pair `\|` with `r"""`.
- Line length 79 applies to test docstrings; escaping adds characters and can
  push a line over `E501` — reword to fit.
- The bars can appear in a docstring **body** line, not just the summary — fix
  every line, not only the `"""` opener.

### 4.2 `@chex.all_variants` signature warnings

`@chex.all_variants` replaces `test_x` with generated `test_x__with_jit` /
`test_x__without_jit` methods that have **no resolvable signature**, so
`sphinx_autodoc_typehints` throws `error while formatting signature ... [autodoc]`
— one per generated method (hundreds across the suite). They are non-fatal but
drown the log. Silenced centrally in `conf.py`:

```python
suppress_warnings = ["myst.mathjax", "autodoc"]
```

You cannot skip the variant members (chex leaves no base method, so skipping
hides the test), and handler ordering cannot pre-empt the library's crash — so
suppression is the lever. The trade-off: real `autodoc` warnings on `src/` are
also hidden; revisit if the API Reference grows genuine signature issues.

### 4.3 LaTeX in docstrings needs raw strings

Any docstring with backslash math (`:math:`\alpha``, `\frac`, ...) must be
`r"""..."""`, or Python mangles the backslashes before docutils sees them. This
is already a CONTRIBUTING rule; it is doubly true now that test docstrings render.

---

## 5. Building & verifying

```bash
# Full build (notebooks executed) — the production build:
cd docs && uv run make html

# Verify the Test Reference WITHOUT executing tutorial notebooks
# (isolates test-docs issues from notebook-execution failures):
uv run sphinx-build -b html -D nb_execution_mode=off source /tmp/docs_check
```

**Known separate blocker:** `make html` executes the tutorial notebooks, and a
notebook that lags the current API (e.g. mid-rationalization renames) fails the
*whole* build before HTML is written — unrelated to the Test Reference. Use the
`nb_execution_mode=off` build to verify the test docs while that is outstanding;
fix the notebooks (the "examples never lag" guardrail) for a green production
build.

Sanity checks:

```bash
# every documented test module must import under autodoc
uv run python -c "import tests.test_rheedium.test_simul.test_simulator"

# no unescaped magnitude bars left in test docstrings
grep -rnE '"""[^"]*[^\\]\|[A-Za-z]' tests/test_rheedium/
```

---

## 6. Adding a test, keeping docs green

1. Put it in `tests/test_rheedium/test_<pkg>/test_<module>.py`, class
   `Test<Symbol>(chex.TestCase)`, methods `test_*`.
2. Write the `what`/`how` docstrings (numpydoc).
3. Add the **back-`:see:`** to the source symbol, and confirm the source carries
   the **forward** `:see:` to this class.
4. **Do not** add `__all__` or `Routine Listings` to test modules — tests are not
   a public API.
5. Escape any `|x|` bars (`r"""...\|x\|..."""`) or use `:math:`.
6. If the file is in a new subpackage, add a `docs/source/tests/<pkg>.rst` and
   list it in `docs/source/tests/index.rst`.
7. Build with `nb_execution_mode=off` and confirm no new `Undefined substitution`
   errors.

---

## 7. State of play

- **Landed:** Test Reference under the "API Reference" caption; per-subpackage
  autodoc pages; the chex-variant warnings suppressed; the `|x|` substitution
  errors escaped. The `nb_execution_mode=off` build succeeds with the test docs
  rendered (≈10 warnings, down from ≈990).
- **Outstanding (tracked in [testing_docs.md](../partial/testing_docs.md)):** the
  full `what`/`how` docstring enrichment pass (T3), the bidirectional `:see:`
  back-reference sweep across all test classes (T2), the bidirectionality guard
  test + `-W` link-clean CI (T4), and fixing the tutorial-notebook execution so
  the production `make html` is green.
