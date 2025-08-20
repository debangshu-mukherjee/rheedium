Installation
============

Requirements
------------

rheedium requires Python 3.11+ and the following dependencies:

* JAX (with optional CUDA support)
* NumPy
* SciPy
* Matplotlib
* Pandas
* beartype
* jaxtyping

Recommended: Install with uv
----------------------------

`uv <https://github.com/astral-sh/uv>`_ is the recommended package manager for rheedium. It's significantly faster than pip and handles dependency resolution more efficiently.

Install uv
~~~~~~~~~~~

.. code-block:: bash

   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv

Install rheedium with uv
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install from PyPI
   uv pip install rheedium
   
   # Or install in a new virtual environment
   uv venv rheedium-env
   source rheedium-env/bin/activate  # On Windows: rheedium-env\Scripts\activate
   uv pip install rheedium

Development Installation with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/rheedium.git
   cd rheedium
   
   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install in development mode with all dependencies
   uv pip install -e ".[dev,docs]"
   
   # Or install dev dependencies separately
   uv pip install -e .
   uv pip install pytest pytest-cov black isort flake8 mypy
   uv pip install sphinx sphinx-rtd-theme myst-parser nbsphinx

GPU Support with uv
~~~~~~~~~~~~~~~~~~~

For GPU acceleration, install JAX with CUDA support:

.. code-block:: bash

   # For CUDA 12 (recommended)
   uv pip install "jax[cuda12]"
   
   # For CUDA 11
   uv pip install "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Alternative: Install with pip
-----------------------------

If you prefer using pip:

.. code-block:: bash

   # Install from PyPI
   pip install rheedium

   # Install from source
   git clone https://github.com/yourusername/rheedium.git
   cd rheedium
   pip install -e .

   # Development installation
   pip install -e ".[dev,docs]"

Project Setup with uv
---------------------

For new projects using rheedium:

.. code-block:: bash

   # Create a new project directory
   mkdir my-rheed-project
   cd my-rheed-project
   
   # Initialize with uv
   uv venv
   source .venv/bin/activate
   
   # Install rheedium and other dependencies
   uv pip install rheedium jupyter matplotlib
   
   # Create a simple test script
   cat > test_rheedium.py << EOF
   import rheedium as rh
   import jax.numpy as jnp
   
   # Test basic functionality
   crystal = rh.types.create_crystal_structure(
       frac_positions=jnp.array([[0, 0, 0, 1]]),
       cart_positions=jnp.array([[0, 0, 0, 1]]),
       cell_lengths=jnp.array([5.0, 5.0, 5.0]),
       cell_angles=jnp.array([90.0, 90.0, 90.0])
   )
   print("rheedium is working correctly!")
   EOF
   
   # Run the test
   python test_rheedium.py

Why uv?
-------

- **Speed**: Up to 10-100x faster than pip for dependency resolution
- **Reliability**: Better conflict resolution and dependency management
- **Modern**: Built with Rust, designed for Python 3.8+
- **Compatible**: Drop-in replacement for pip with same interface
- **Memory efficient**: Lower memory usage during installation

Lock File Management
--------------------

When using uv for development, you can generate lock files for reproducible builds:

.. code-block:: bash

   # Generate uv.lock file (if using pyproject.toml)
   uv lock
   
   # Install from lock file
   uv sync
   
   # Update dependencies
   uv lock --upgrade

Verify Installation
-------------------

.. code-block:: python

   import rheedium as rh
   import jax
   import jax.numpy as jnp
   
   print(f"rheedium imported successfully")
   print(f"JAX devices: {jax.devices()}")
   print(f"JAX backend: {jax.default_backend()}")
   
   # Test basic functionality
   crystal = rh.types.create_crystal_structure(
       frac_positions=jnp.array([[0, 0, 0, 1]]),
       cart_positions=jnp.array([[0, 0, 0, 1]]),
       cell_lengths=jnp.array([5.0, 5.0, 5.0]),
       cell_angles=jnp.array([90.0, 90.0, 90.0])
   )
   print("Crystal structure creation: ✓")
   
   # Test RHEED simulation
   pattern = rh.simul.simulate_rheed_pattern(
       crystal=crystal,
       voltage_kV=jnp.asarray(10.0),
       theta_deg=jnp.asarray(2.0)
   )
   print(f"RHEED simulation: ✓ ({len(pattern.intensities)} reflections)")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors with JAX**:

.. code-block:: bash

   # Ensure you have the correct JAX version
   uv pip install --upgrade jax jaxlib

**CUDA issues**:

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi
   
   # Reinstall JAX with CUDA support
   uv pip uninstall jax jaxlib
   uv pip install "jax[cuda12]"

**Virtual environment issues**:

.. code-block:: bash

   # Recreate virtual environment
   rm -rf .venv
   uv venv
   source .venv/bin/activate
   uv pip install rheedium

Performance Tips
~~~~~~~~~~~~~~~~

- Use uv for faster dependency resolution
- Enable JAX GPU support for large simulations
- Consider using `JAX_ENABLE_X64=1` for high precision calculations
- Use `jax.jit()` to compile functions for better performance

For more help, visit the `rheedium documentation <https://rheedium.readthedocs.io>`_ or open an issue on `GitHub <https://github.com/yourusername/rheedium/issues>`_.