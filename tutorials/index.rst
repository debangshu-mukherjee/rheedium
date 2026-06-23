Tutorials
=========

Tutorial notebooks for Rheedium. They are authored as Jupyter notebooks paired
with Jupytext percent scripts where available, so they can be opened directly in
VS Code or Jupyter while still keeping reviewable Python sources in git.

.. note::

   For remote development, open ``tutorials/<notebook>.ipynb`` in VS Code over
   Remote-SSH and select the project kernel. After editing paired notebooks,
   run ``jupytext --sync tutorials/<notebook>.ipynb`` before committing.

.. toctree::
   :maxdepth: 1

   01_kinematic_SrTiO3
   02_kinematic_MgO
   03_kinematic_Bi2Se3
   04_finite_domain
   05_bi2se3_temp_sweep
   06_atomic_visualizer
