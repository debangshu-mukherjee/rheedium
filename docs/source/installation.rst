Installation
============

Requirements
------------

rheedium requires Python 3.8+ and the following dependencies:

* JAX
* NumPy
* SciPy
* Matplotlib
* Pandas
* beartype
* jaxtyping

Install from PyPI
-----------------

.. code-block:: bash

   pip install rheedium

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/yourusername/rheedium.git
   cd rheedium
   pip install -e .

GPU Support
-----------

For GPU acceleration, install JAX with CUDA support:

.. code-block:: bash

   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Verify Installation
-------------------

.. code-block:: python

   import rheedium as rh
   print(f"rheedium version: {rh.__version__}")