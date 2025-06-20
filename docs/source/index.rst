Welcome to rheedium's documentation!
====================================

rheedium is a JAX-based library for RHEED (Reflection High-Energy Electron Diffraction) 
simulation and analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   tutorials/index

Features
--------

* JAX-based for high performance and GPU acceleration
* Kinematic RHEED pattern simulation
* Crystal structure parsing from CIF files
* Atomic potential calculations using Kirkland scattering factors
* Phosphor screen visualization

Quick Start
-----------

.. code-block:: python

   import rheedium as rh
   
   # Load crystal structure
   crystal = rh.inout.parse_cif("path/to/crystal.cif")
   
   # Simulate RHEED pattern
   pattern = rh.simul.simulate_rheed_pattern(crystal, voltage_kV=20.0)
   
   # Visualize results
   rh.plots.plot_rheed(pattern)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`