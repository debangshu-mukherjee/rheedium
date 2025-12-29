import os
import sys
import tomllib
from datetime import datetime

# Disable JAX GPU usage during doc building to prevent timeouts
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["BUILDING_DOCS"] = "1"


# Make jaxtyped decorator a no-op during doc building to preserve docstrings
# This MUST be done before any rheedium imports
def _noop_decorator(*args, **kwargs):
    """No-op decorator for documentation building."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn


import jaxtyping

jaxtyping.jaxtyped = _noop_decorator


# Add project paths
project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

print(f"Added to sys.path: {src_path}")

# Read project metadata from pyproject.toml
pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]

authors_data = pyproject_data["project"]["authors"]
author = (
    authors_data[0]["name"]
    if isinstance(authors_data[0], dict)
    else authors_data[0]
)

project_copyright = f"{datetime.now().year}, {author}"
release = pyproject_data["project"]["version"]

# -- General configuration ---------------------------------------------------

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

# MyST-Parser configuration for LaTeX math rendering
myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ math delimiters
    "amsmath",  # Enable LaTeX math environments like \begin{equation}
]

# MathJax configuration to ensure math renders on first page load
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "startup": {
        "ready": "() => { MathJax.startup.defaultReady(); MathJax.startup.promise.then(() => { console.log('MathJax initial typesetting complete'); }); }"
    },
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
    "options": {
        "processHtmlClass": "tex2jax_process|mathjax_process|math|output_area",
    },
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_input_prompt = "In [%s]:"
nbsphinx_output_prompt = "Out [%s]:"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

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
    "source_repository": "https://github.com/debangshu-mukherjee/rheedium/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/debangshu-mukherjee/rheedium/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for NumPy style docstrings
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

# Mock imports for packages that might not be available during doc build
autodoc_mock_imports = [
    "pandas",
    "scipy",
    "scipy.interpolate",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "matplotlib.axes",
    "matplotlib.figure",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",
    "gemmi",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "ignore-module-all": False,
}

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
python_use_unqualified_type_names = True
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

# Disable nitpicky mode to avoid warnings
nitpicky = False

# Type aliases for jaxtyping annotations
napoleon_type_aliases = {
    'Float[Array, " "]': "scalar float",
    'Float[Array, " 3"]': "3D float array",
    'Float[Array, " 3 3"]': "3x3 float array",
    'Float[Array, " M 3"]': "Mx3 float array",
    'Float[Array, " N 3"]': "Nx3 float array",
    'Float[Array, " N 4"]': "Nx4 float array",
    'Int[Array, " "]': "scalar int",
    'Int[Array, " N"]': "N int array",
    'Int[Array, " 2"]': "2-element int array",
    'Num[Array, " N"]': "N numeric array",
    'Bool[Array, " "]': "scalar bool",
    "scalar_float": "float",
    "scalar_int": "int",
    "scalar_num": "numeric",
}

typehints_defaults = "comma"
typehints_type_aliases = napoleon_type_aliases

# Ignore certain types that cause issues
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

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Custom setup ------------------------------------------------------------


def skip_member(app, what, name, obj, skip, options):
    """
    Skip specific members in documentation.
    """
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
    if name.startswith("_") and not name.startswith("__"):
        return True

    # Skip NamedTuple field descriptors to avoid duplicate object descriptions
    # when both class attributes and module-level data entries are documented
    if what == "data" and hasattr(obj, "_field_types"):
        return True

    return skip


def process_signature(
    app, what, name, obj, options, signature, return_annotation
):
    """
    Process signatures to handle jaxtyping annotations.
    """
    if signature:
        signature = signature.replace('Float[Array, " ', "FloatArray[")
        signature = signature.replace('Int[Array, " ', "IntArray[")
        signature = signature.replace('Bool[Array, " ', "BoolArray[")
        signature = signature.replace('Num[Array, " ', "NumArray[")
        signature = signature.replace('"]', "]")
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-signature", process_signature)
