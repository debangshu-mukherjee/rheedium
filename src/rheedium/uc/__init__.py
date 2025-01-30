"""
=========================================================

Unit Cell Operations (:mod:`rheedium.uc`)

=========================================================

This package contains the modules for the calculations of
unit cell operations and conversion to Ewald sphere.
"""

# Get all functions and classes from both modules for __all__
import inspect
import sys

from . import helper, unitcell
from .helper import *
from .unitcell import *

__all__ = (
    # Functions and classes from helper.py
    [
        name
        for name, obj in inspect.getmembers(sys.modules["rheedium.uc.helper"])
        if (inspect.isfunction(obj) or inspect.isclass(obj))
        and not name.startswith("_")
    ]
    +
    # Functions and classes from unitcell.py
    [
        name
        for name, obj in inspect.getmembers(sys.modules["rheedium.uc.unitcell"])
        if (inspect.isfunction(obj) or inspect.isclass(obj))
        and not name.startswith("_")
    ]
)
