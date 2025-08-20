import os
import sys
from datetime import datetime

import tomllib
from sphinx.application import Sphinx

# CRITICAL: Disable beartype during documentation builds
os.environ["BUILDING_DOCS"] = "1"

# CRITICAL: Proper path setup for autodoc
project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")

# Add both src and project root to path
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

print(f"Added to sys.path: {src_path}")  # Debug line

# Set up JAX to avoid issues during import
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

# Read project metadata from pyproject.toml
pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]

# Handle authors
authors_data = pyproject_data["project"]["authors"]
author = authors_data[0]["name"] if isinstance(authors_data[0], dict) else authors_data[0]

project_copyright = f"{datetime.now().year}, {author}"
release = pyproject_data["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": None,
    ".md": None,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options - default to dark mode
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3399ff",
        "color-brand-content": "#3399ff",
    },
    "sidebar_hide_name": False,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
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

# Add custom sections to napoleon (for Google style)
napoleon_custom_sections = [
    ("Algorithm", "notes_style"),  # Custom section for Algorithm (converted from Flow)
]

# nbsphinx configuration
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# IMPORTANT: Mock only truly problematic dependencies
# sphinx-autodoc-typehints should handle most type annotations
autodoc_mock_imports = [
    "pandas",
    "scipy",
    "scipy.interpolate", 
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "gemmi",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "ignore-module-all": False,  # Respect __all__
}

# Type handling with sphinx-autodoc-typehints
autodoc_typehints = "signature"  # Show types only in signature
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "all"
python_use_unqualified_type_names = True
typehints_fully_qualified = False
always_document_param_types = False  # Let Google docstrings handle this
typehints_document_rtype = False  # Let Google docstrings handle this
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

# Reduced nitpicky mode
nitpicky = False

# Type aliases for cleaner display
napoleon_type_aliases = {
    'Float[Array, " "]': "scalar float",
    'Float[Array, " 3"]': "3D float array",
    'Float[Array, " 3 3"]': "3x3 float array",
    'Float[Array, " M 3"]': "Mx3 float array",
    'Float[Array, " N 3"]': "Nx3 float array",
    'Float[Array, " N 4"]': "Nx4 float array",
    'Int[Array, " "]': "scalar int",
    'Int[Array, " N"]': "N int array",
    'Num[Array, " N"]': "N numeric array",
    'Bool[Array, " "]': "scalar bool",
    "scalar_float": "float",
    "scalar_int": "int",
    "scalar_num": "numeric",
}

# Additional type aliases for sphinx-autodoc-typehints
typehints_defaults = "comma"
typehints_type_aliases = napoleon_type_aliases

# Ignore problematic references
nitpick_ignore = [
    ("py:class", "Float"),
    ("py:class", "Array"),
    ("py:class", "Int"),
    ("py:class", "Num"),
    ("py:class", "Bool"),
    ("py:class", "jaxtyping.Float"),
    ("py:class", "jaxtyping.Array"),
    ("py:class", "beartype"),
    ("py:class", "jaxtyped"),
    ("py:obj", "beartype"),
    ("py:obj", "jaxtyped"),
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

html_css_files = ["custom.css"]
html_js_files = ["custom.js"]


def skip_member(app, what, name, obj, skip, options):
    """Skip problematic members."""
    skip_names = [
        "Float",
        "Array",
        "Int",
        "Num",
        "Bool",
        "beartype",
        "jaxtyped",
        "tree_flatten",
        "tree_unflatten",
    ]
    if name in skip_names:
        return True
    return skip


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Process signatures to handle jaxtyping annotations."""
    if signature:
        # Simplify jaxtyping annotations in signatures
        signature = signature.replace('Float[Array, " ', 'FloatArray[')
        signature = signature.replace('Int[Array, " ', 'IntArray[')
        signature = signature.replace('Bool[Array, " ', 'BoolArray[')
        signature = signature.replace('Num[Array, " ', 'NumArray[')
        signature = signature.replace('"]', ']')
    return signature, return_annotation


def setup(app: Sphinx) -> None:
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-signature", process_signature)
