API Reference
=============

This page documents all public functions and classes in the rheedium package.

Input/Output Module
-------------------

.. automodule:: rheedium.inout
   :noindex:

**Functions:**

.. autofunction:: rheedium.inout.parse_cif
.. autofunction:: rheedium.inout.symmetry_expansion  
.. autofunction:: rheedium.inout.atomic_symbol
.. autofunction:: rheedium.inout.kirkland_potentials
.. autofunction:: rheedium.inout.parse_xyz

Simulation Module
-----------------

.. automodule:: rheedium.simul
   :noindex:

**Functions:**

.. autofunction:: rheedium.simul.wavelength_ang
.. autofunction:: rheedium.simul.incident_wavevector
.. autofunction:: rheedium.simul.project_on_detector
.. autofunction:: rheedium.simul.find_kinematic_reflections
.. autofunction:: rheedium.simul.compute_kinematic_intensities
.. autofunction:: rheedium.simul.simulate_rheed_pattern
.. autofunction:: rheedium.simul.atomic_potential
.. autofunction:: rheedium.simul.crystal_potential

Unit Cell Module
----------------

.. automodule:: rheedium.ucell
   :noindex:

**Functions:**

.. autofunction:: rheedium.ucell.angle_in_degrees
.. autofunction:: rheedium.ucell.compute_lengths_angles
.. autofunction:: rheedium.ucell.parse_cif_and_scrape
.. autofunction:: rheedium.ucell.reciprocal_unitcell
.. autofunction:: rheedium.ucell.reciprocal_uc_angles
.. autofunction:: rheedium.ucell.get_unit_cell_matrix
.. autofunction:: rheedium.ucell.build_cell_vectors
.. autofunction:: rheedium.ucell.generate_reciprocal_points
.. autofunction:: rheedium.ucell.atom_scraper
.. autofunction:: rheedium.ucell.bessel_kv

Types Module
------------

.. automodule:: rheedium.types
   :noindex:

**Classes:**

.. autoclass:: rheedium.types.CrystalStructure
   :members:
   :undoc-members:

.. autoclass:: rheedium.types.PotentialSlices
   :members:
   :undoc-members:

.. autoclass:: rheedium.types.XYZData
   :members:
   :undoc-members:

.. autoclass:: rheedium.types.RHEEDPattern
   :members:
   :undoc-members:

.. autoclass:: rheedium.types.RHEEDImage
   :members:
   :undoc-members:

**Factory Functions:**

.. autofunction:: rheedium.types.make_xyz_data
.. autofunction:: rheedium.types.create_potential_slices
.. autofunction:: rheedium.types.create_crystal_structure
.. autofunction:: rheedium.types.create_rheed_pattern
.. autofunction:: rheedium.types.create_rheed_image

**Type Aliases:**

.. autodata:: rheedium.types.scalar_float
.. autodata:: rheedium.types.scalar_int
.. autodata:: rheedium.types.scalar_num
.. autodata:: rheedium.types.non_jax_number
.. autodata:: rheedium.types.float_image
.. autodata:: rheedium.types.int_image

Plotting Module
---------------

.. automodule:: rheedium.plots
   :noindex:

**Functions:**

.. autofunction:: rheedium.plots.create_phosphor_colormap
.. autofunction:: rheedium.plots.plot_rheed

Reconstruction Module
---------------------

.. automodule:: rheedium.recon
   :noindex:

*This module is currently empty and will be populated with reconstruction algorithms in future releases.*