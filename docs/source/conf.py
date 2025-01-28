import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "rheedium"
copyright = "2024"
author = "Debangshu Mukherjee"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
