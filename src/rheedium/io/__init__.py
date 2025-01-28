"""
=========================================================

Data I/O (:mod:`rheedium.io`)

=========================================================

This package contains the modules for the loading and
unloading of datasets.
"""

# Get all functions defined in data_io.py for __all__
import inspect
import sys

from . import data_io
from .data_io import *  # This will expose all functions from data_io.py

__all__ = [
    name
    for name, obj in inspect.getmembers(sys.modules["rheedium.io.data_io"])
    if inspect.isfunction(obj) and not name.startswith("_")
]
