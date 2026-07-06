Testing & Validation
====================

This section publishes Rheedium's test suite as a validation reference: what
the library guarantees, where each guarantee is checked, and how the check is
implemented. The pages mirror ``tests/test_rheedium`` by subpackage so the
validation surface lines up with the public API surface.

How to Read
-----------

Each module page starts with the test file's coverage scope. Each ``Test*``
class groups checks for a source symbol or behavior family and carries a
``:see:`` back-reference when it validates a documented source object. Each
``test_*`` method has an ``Extended Summary`` for the property being verified
and ``Notes`` for the fixtures, parametrization, assertions, and JAX execution
path used to verify it.

Bidirectional Links
-------------------

Source docstrings point forward to the tests that validate them, and test
classes point back to the source symbols they cover. The
``tests.test_rheedium.test_see_bidirectional`` guard keeps those links
resolvable, while ``tests.test_rheedium.test_testing_documentation`` keeps this
reference documented as a first-class contract.

Coverage Map
------------

.. toctree::
   :maxdepth: 1

   audit
   inout
   physics
   plots
   procs
   recon
   simul
   tools
   types
   ucell
   core
