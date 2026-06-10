Interactive Ewald Sphere
========================

This page is a lightweight proof of concept for browser-side Marimo via
``sphinx-marimo``. It intentionally avoids ``rheedium`` and JAX so it can run
inside WebAssembly on Read the Docs while still matching the same
relativistic wavelength and Ewald-sphere geometry conventions used by the
package.

Use the controls to vary the beam energy, grazing angle, lattice spacing, and
number of reciprocal rods. The notebook recomputes the Ewald-circle
intersections live in the browser.

.. marimo:: ewald_sphere_demo.py
   :height: 760px
   :width: 100%
   :click-to-load: overlay
   :load-button-text: Run Ewald Demo
