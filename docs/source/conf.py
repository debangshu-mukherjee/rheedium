import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("./_ext"))

project = "rheedium"
copyright = "2024"
author = "Debangshu Mukherjee"

# The full version, including alpha/beta/rc tags
release = "2025.01.28"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    # "param_parser",  # Comment out if this extension doesn't exist
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Choose one theme consistently
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings for custom docstring format
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True
napoleon_custom_sections = ["Description", "Parameters", "Returns", "Flow"]

# Add nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue building even if there are errors

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Intersphinx mapping for external references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Custom CSS to improve type hint rendering
html_css_files = [
    "custom.css",
]

# Add these configurations to exclude imported items and type hints
autodoc_default_options = {
    "exclude-members": "Float, Array, Int, Num, Bool, beartype, jaxtyped",
    "undoc-members": True,
    "show-inheritance": True,
}

# Add nitpicky mode to catch any broken references
nitpicky = False  # Set to False initially to avoid overwhelming warnings

# Add type aliases to handle jaxtyping types cleanly
napoleon_type_aliases = {
    'Float[Array, ""]': "scalar array",
    'Float[Array, "3"]': "3D array", 
    'Float[Array, "3 3"]': "3x3 array",
    'Float[Array, "M 3"]': "Mx3 array",
    'Float[Array, "N 3"]': "Nx3 array",
    'Int[Array, ""]': "integer array",
    'Num[Array, "*"]': "numeric array",
    'Bool[Array, "*"]': "boolean array",
}

# Add any modules to ignore when warning about missing references
nitpick_ignore = [
    ("py:class", "Float"),
    ("py:class", "Array"), 
    ("py:class", "Int"),
    ("py:class", "Num"),
    ("py:class", "Bool"),
    ("py:class", "jaxtyping.Float"),
    ("py:class", "jaxtyping.Array"),
    ("py:class", "jaxtyping.Int"),
    ("py:class", "jaxtyping.Num"),
    ("py:class", "jaxtyping.Bool"),
    ("py:class", "beartype.typing.NamedTuple"),
    ("py:class", "beartype"),
    ("py:class", "jaxtyped"),
    ("py:obj", "beartype"),
    ("py:obj", "jaxtyped"),
]

# Define what to skip during documentation
def skip_member(app, what, name, obj, skip, options):
    # Skip all imported jaxtyping and beartype items
    if name in ["Float", "Array", "Int", "Num", "Bool", "beartype", "jaxtyped"]:
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)