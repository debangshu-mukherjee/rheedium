API Reference
=============

JAX-based RHEED simulation and analysis package.

Rheedium provides a comprehensive suite of tools for simulating and analyzing
Reflection High-Energy Electron Diffraction (RHEED) patterns. Built on JAX,
it offers differentiable simulations suitable for optimization and machine
learning applications in materials science and surface physics.

Submodules
----------

.. toctree::
   :maxdepth: 1
   :hidden:

   inout
   plots
   recon
   simul
   types
   ucell

:mod:`rheedium.inout`
    Data input/output operations for crystal structures and RHEED images.

:mod:`rheedium.plots`
    Visualization tools for RHEED patterns and crystal structures.

:mod:`rheedium.recon`
    Surface reconstruction analysis and modeling utilities.

:mod:`rheedium.simul`
    RHEED pattern simulation using kinematic diffraction theory.

:mod:`rheedium.types`
    Custom type definitions and data structures for JAX compatibility.

:mod:`rheedium.ucell`
    Unit cell and crystallographic computation utilities.

Examples
--------

.. code-block:: python

    import rheedium as rh
    crystal = rh.inout.parse_cif("structure.cif")
    pattern = rh.simul.simulate_rheed_pattern(crystal)
    rh.plots.plot_rheed(pattern)

Notes
-----

All computations are JAX-compatible and support automatic differentiation
for gradient-based optimization of crystal structures and simulation
parameters.
