import os
import sys

# Add the source directory to the path
project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Project information
project = "rheedium"
copyright = "2024, Author"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# HTML output
html_theme = "alabaster"

# Mock heavy imports to avoid hanging
autodoc_mock_imports = [
    "jax",
    "jax.numpy",
    "jaxlib",
    "jaxtyping",
    "beartype",
    "scipy",
    "matplotlib",
    "pandas",
    "gemmi",
]

# Napoleon settings for NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
