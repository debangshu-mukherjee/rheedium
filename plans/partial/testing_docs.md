# Testing Documentation & Bidirectional `:see:` Plan

Scope: `rheedium` — make the **test suite first-class, published documentation**.
Three coupled deliverables:

1. **Detailed `what` / `how` test docstrings** — every test module, `Test<Symbol>`
   class, and `test_*` method documents *what property/invariant it verifies* and
   *how it verifies it* (numpydoc), per the new `tests/` rules in
   [CONTRIBUTING.md](../../CONTRIBUTING.md).
2. **Render the testing API in Read the Docs** — a "Testing / Validation"
   reference that autodocs the `tests` package alongside the existing `api/*`
   pages, so the suite reads as *what the library guarantees and how each
   guarantee is checked*.
3. **Bidirectional `:see:`** — the source symbol's `:see:` → its test class, and
   the test class's `:see:` → the source symbol, so once both are rendered the
   cross-reference resolves **source → test and test → source** in RTD.

Status: **proposed.** The CONTRIBUTING conventions (1–3 above, as written rules)
have **landed**; this plan implements them across the suite and the docs build.
**Not on the gated roadmap chain** (framework → rationalization → recon →
automatons): this is a documentation/quality track that can land any time. It does
not touch runtime behavior — docstrings, docs config, and one guard test only.

---

## 1. Why

- The source `:see:` references (`:see: :class:`~.test_unitcell.TestReciprocalUnitcell``)
  are pervasive in `src/`, but **today they are dead links in Read the Docs** —
  the `tests` package is not in the Sphinx build, so the target does not resolve.
- The suite already encodes *what the library guarantees* (every physical
  invariant, round-trip, and known-value check). Surfacing it turns ~1100 tests
  from private scaffolding into a **validation reference** a user (or referee) can
  read — directly supporting the paper's "rigor ladder" story.
- A back-reference on each test class closes the loop: a reader on a test page
  jumps to the symbol under test, and vice-versa. The link becomes navigable both
  ways instead of one-directional and broken.

---

## 2. Workstreams

### W1 — Sphinx wiring (render the testing API)

The docs are set up to make this mechanical (one `docs/source/api/<pkg>.rst` per
subpackage, `sphinx.ext.autodoc` + `sphinx_autodoc_typehints`, `project_root`
already on `sys.path`, `tests/test_rheedium/__init__.py` present so the package
imports):

- **Landed structure:** a **Test Reference** page (`docs/source/tests/index.rst`)
  sits **under the existing "API Reference" caption** in `index.rst`, alongside
  `api/index` — so the nav reads *API Reference* + *Test Reference* under
  **API REFERENCE**. The Test Reference has one `tests/<pkg>.rst` page per
  subpackage that `automodule`s each `tests.test_rheedium.test_<pkg>.test_*`
  module (`:members: :undoc-members: :show-inheritance:`), mirroring the
  `api/<pkg>.rst` pattern. All test subpackages have `__init__.py` and
  `project_root` is on `sys.path`, so the modules import under autodoc.
- Ensure the docs build can **import** the test modules (autodoc imports what it
  documents): install the `test` extra in the docs build env, *or* extend
  `autodoc_mock_imports` for the heaviest test-only deps. Installing is preferred
  — the docstrings describe real asserted behavior.

### W2 — Bidirectional `:see:` back-references

- Add a `:see:` from every `Test<Symbol>` class **back** to its source symbol
  (`:see: :func:`~rheedium...`` / `:class:` / `:obj:`), the counterpart of the
  source's forward `:see:`.
- Reconcile the **forward** direction at the same time: confirm every public
  source symbol's `:see:` points at a test class that exists (the rationalization/
  framework churn may have moved or renamed some).

### W3 — Docstring enrichment (`what` + `how`)

The large content pass. For each test module/class/method, expand the docstring to
the numpydoc form mandated by CONTRIBUTING:

- **Module** docstring → summary + the file's coverage.
- **Class** (`Test<Symbol>`) docstring → the symbol under test, the scope of its
  cases, and the back-`:see:`.
- **Method** (`test_*`) docstring → summary line; `Extended Summary` = *what* is
  verified (property/invariant/expected value, units, tolerances); `Notes` = *how*
  (inputs/fixtures, assertion strategy, the `jit`/`grad`/`vmap` variant). No
  ad-hoc headers.

Sequence **per subpackage** (audit, inout, plots, procs, recon, simul, tools,
types, ucell) so each is one reviewable PR.

### W4 — Enforcement (keep it from rotting)

- **Bidirectionality guard test** — a test asserting: (a) every public source
  symbol with a `:see:` points to an existing test class, and (b) every
  `Test<Symbol>` class carries a `:see:` back to an existing source symbol.
- **Docs build is link-clean** — run Sphinx with nitpicky + warnings-as-errors
  (`-n -W`) for cross-references so a dead `:see:` (either direction) fails the
  build; wire it into the docs CI.
- **Docstring coverage on tests** — extend `interrogate` / pydocstyle scope so
  test modules/classes/methods are checked for the required docstring.

---

## 3. Constraints

- **No behavior change.** This plan edits docstrings, docs config, and adds one
  guard test. No test logic, no `src/` runtime code changes.
- **Docs build must stay green and reasonably fast.** Autodoc imports the test
  modules (which import `rheedium` + JAX); watch build time/memory and mock only
  what is safe to mock.
- **Follow the CONTRIBUTING conventions verbatim** — this plan is their
  implementation, not a second spec.

---

## 4. Gated phases

Each phase is one reviewable PR ending at an **objectively checkable gate** (a
command, not an opinion).

### Entry — Gate T0
The CONTRIBUTING `tests/` conventions (detailed docstrings, no `__all__`/Routine
Listings in tests, bidirectional `:see:`, published-to-RTD) are committed.
**Already satisfied.**

### Universal gate (every phase)
- `cd docs && uv run make html` builds; **no new cross-reference warnings**;
- `uv run pytest` green; `uv run ruff check tests` and `uv run ty check` clean;
- the change is docstrings/docs/guard-only — **no runtime/test-logic edit**
  (reviewer-verified).

### Phase T1 — Sphinx wiring  ·  W1
*Tasks:* `api/tests.rst` autodoc page + toctree entry; docs-build env gets the
`test` deps (or targeted `autodoc_mock_imports`); confirm the `tests` package
imports under the docs build.

**Gate TG1:** `make html` renders a "Testing & Validation" section containing the
test classes; at least one existing **source `:see:` forward link now resolves**
to its rendered test page (spot-checked + no "undefined label" warning for it).

### Phase T2 — Bidirectional back-references  ·  W2
*Tasks:* add the back-`:see:` to every `Test<Symbol>` class; fix any stale forward
`:see:` in `src/`.

**Gate TG2:** the **W4 bidirectionality guard test passes** — every source `:see:`
resolves to an existing test class and every test class has a back-`:see:` to an
existing source symbol; `make html` shows both directions resolving for a sampled
pair.

### Phase T3 — Docstring enrichment (per subpackage)  ·  W3
*Tasks:* expand module/class/method docstrings to the `what`/`how` numpydoc form,
one subpackage per PR (9 PRs: audit → inout → plots → procs → recon → simul →
tools → types → ucell).

**Gate TG3 (per subpackage):** every `test_*` in the subpackage has a summary +
`Extended Summary` (*what*) + `Notes` (*how*); each `Test<Symbol>` class and the
module have the required docstring; `interrogate` (tests-scoped) passes; docs
render the enriched pages.

### Phase T4 — Enforcement + CI  ·  W4
*Tasks:* land the bidirectionality guard test; turn on nitpicky warnings-as-errors
for `:see:`/cross-refs in the docs build; extend interrogate/pydocstyle to test
scope; wire the docs build (with test deps) into CI.

**Gate TG4:** docs CI runs `make html` **`-n -W`-clean** for cross-references; the
guard test + tests-scoped docstring coverage are required checks; a deliberately
broken `:see:` (either direction) fails CI in a dry run.

### Phase T5 — Landing page & polish
*Tasks:* a "Testing & Validation" overview page (what the suite guarantees, how to
read a test page, the bidirectional navigation); cross-link from the main docs and
the README; confirm end-to-end source ↔ test navigation.

**Gate TG5:** the overview page builds and is linked from `index.rst`; a reviewer
can navigate source → test → source for a sampled symbol in the built HTML.

### Gate summary

| Gate | Pass condition (+ universal gate) |
|------|------------------------------------|
| **T0** | CONTRIBUTING `tests/` conventions committed (**done**) |
| **TG1** | `make html` renders the testing API; a source `:see:` forward link resolves |
| **TG2** | bidirectionality guard passes; sampled pair resolves both ways |
| **TG3** | per-subpackage `what`/`how` docstrings; interrogate (tests) passes |
| **TG4** | docs CI `-n -W` clean for cross-refs; guard + coverage required; broken `:see:` fails CI |
| **TG5** | testing overview page linked; source ↔ test navigation verified in HTML |

T1–T2 are the foundation (render + link); T3 is the bulk content pass; T4 locks it
against rot; T5 polishes. T3 subpackages can be reordered/parallelized after TG2.

---

## 5. Risks

- **Autodoc import cost / failures.** Importing every test module pulls in JAX and
  the test deps at build time. Mitigation: install the `test` extra in the docs
  build; mock only safe leaves; watch build time/memory (TG1 measures it).
- **Nitpicky false positives.** `-n -W` can flag legitimate non-`:see:` refs.
  Mitigation: scope the strictness to the cross-reference classes that matter, or
  use a `nitpick_ignore` allowlist; keep the guard test as the real completeness
  check.
- **Docstring drift.** Enriched test docstrings can fall out of sync with the
  assertions. Mitigation: the summary line is reviewed in every PR; the guard test
  pins the `:see:` structure (not the prose).
- **Large content effort.** ~1100 tests is a long enrichment pass. Mitigation: T3
  is per-subpackage and additive; T1/T2/T4 deliver the rendered, linked,
  enforced skeleton **before** all prose is written, so value lands early.
- **Build-time coupling to test deps.** The docs build now depends on `chex` /
  `hypothesis` / `absl-testing`. Mitigation: add them to the docs build group
  explicitly so RTD and local builds agree.

---

## 6. Diff surface

| Path | Change |
|------|--------|
| `docs/source/api/tests.rst` | **new** — autodoc the `tests` package (per-subpackage groups) |
| `docs/source/index.rst` / `api/index.rst` | add the "Testing & Validation" toctree entry |
| `docs/source/conf.py` | ensure `tests` importable; `autodoc_mock_imports` / nitpicky settings for cross-refs |
| `pyproject.toml` | docs build resolves the `test` extra deps (combined docs+test group or RTD config) |
| `tests/test_rheedium/**/test_*.py` | back-`:see:` on each `Test<Symbol>`; enriched module/class/method docstrings (T3) |
| `tests/.../test_see_bidirectional.py` | **new** — guard asserting both `:see:` directions resolve to real symbols |
| `.github/workflows/*` | docs build (with test deps) + `-n -W` cross-ref check + tests-docstring coverage |
| `CONTRIBUTING.md` | already updated — the spec this plan implements |

---

## 7. Outcome

When complete: the test suite is a **published validation reference** on Read the
Docs, every test states *what* it guarantees and *how*, and the `:see:`
cross-reference is **bidirectional** — a reader moves source → test → source
freely, no dead links. The suite stops being invisible scaffolding and becomes
part of the documented contract, enforced by a guard test and a link-clean docs
build so it cannot silently rot.
