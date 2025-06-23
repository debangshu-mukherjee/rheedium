import os
import sys
import tomllib
from datetime import datetime

# CRITICAL: Proper path setup for autodoc
project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")

# Add both src and project root to path
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

print(f"Added to sys.path: {src_path}")  # Debug line

# Read project metadata from pyproject.toml
pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]

# Handle authors
authors_data = pyproject_data["project"]["authors"]
if isinstance(authors_data[0], dict):
    author = authors_data[0]["name"]
else:
    author = authors_data[0]

copyright = f"{datetime.now().year}, {author}"
release = pyproject_data["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": None,
    ".md": None,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings
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

# Add custom sections to napoleon
napoleon_custom_sections = [
    ("Description", "params_style"),
    ("Parameters", "params_style"),
    ("Returns", "returns_style"),
    ("Flow", "params_style"),
    ("Examples", "examples_style"),
    ("Notes", "notes_style"),
]

# nbsphinx configuration
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# IMPORTANT: Mock problematic imports
autodoc_mock_imports = [
    "jax.config",  # This often causes issues
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "Float, Array, Int, Num, Bool, beartype, jaxtyped, tree_flatten, tree_unflatten",
}

# Type handling
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
python_use_unqualified_type_names = True

# Reduced nitpicky mode
nitpicky = False

# Type aliases for cleaner display
napoleon_type_aliases = {
    'Float[Array, ""]': "scalar array",
    'Float[Array, "3"]': "3D array",
    'Float[Array, "3 3"]': "3x3 array",
    'Float[Array, "M 3"]': "Mx3 array",
    'Float[Array, "N 3"]': "Nx3 array",
    'Float[Array, "* 4"]': "Nx4 array",
    'Int[Array, ""]': "integer array",
    'Num[Array, "*"]': "numeric array",
    "scalar_float": "float",
    "scalar_int": "int",
}

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


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
