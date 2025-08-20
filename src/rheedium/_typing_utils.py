"""Utility module for handling type checking decorators.

This module provides conditional decorators that can be disabled during
documentation builds to allow Sphinx autodoc to properly introspect functions.
"""

import os
from collections.abc import Callable
from typing import Any, TypeVar

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Check if we're building documentation
BUILDING_DOCS = os.environ.get('BUILDING_DOCS', '').lower() in ('1', 'true', 'yes')

if BUILDING_DOCS:
    # During documentation builds, make decorators no-ops
    def jaxtyped(typechecker: Any = None) -> Callable[[F], F]:
        """No-op decorator for documentation builds."""

        def decorator(func: F) -> F:
            return func

        return decorator

    # Mock beartype to be a no-op decorator
    def beartype(func: F) -> F:
        """No-op decorator for documentation builds."""
        return func
else:
    # Normal runtime - use actual decorators
    try:
        from beartype import beartype as _beartype
        from jaxtyping import jaxtyped as _jaxtyped

        # Re-export the actual decorators
        jaxtyped = _jaxtyped
        beartype = _beartype.beartype  # Get the actual beartype decorator
    except ImportError:
        # If packages aren't installed, make them no-ops
        def jaxtyped(typechecker: Any = None) -> Callable[[F], F]:
            def decorator(func: F) -> F:
                return func

            return decorator

        def beartype(func: F) -> F:
            return func


__all__ = ['jaxtyped', 'beartype', 'BUILDING_DOCS']
