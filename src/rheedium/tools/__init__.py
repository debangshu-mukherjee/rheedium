"""Reusable numerical and workflow tools for rheedium.

Extended Summary
----------------
This package provides the shared numerical infrastructure used by the
simulation and crystallography modules. It centralises special-function
kernels (modified Bessel functions), quadrature helpers, JAX
compatibility wrappers, electron-beam utility functions, and
distributed-array sharding so that domain modules depend on a single,
well-tested toolbox rather than reimplementing low-level numerics.

All functions are JAX-compatible, JIT-safe, and support automatic
differentiation unless noted otherwise.

Routine Listings
----------------
:func:`bessel_k0`
    Modified Bessel function of the second kind, order zero.
:func:`bessel_k1`
    Modified Bessel function of the second kind, order one.
:func:`bessel_kv`
    Modified Bessel function of the second kind, arbitrary real
    order.
:func:`bucketize_grid`
    Snap a requested detector grid up to the nearest exported bucket.
:func:`deserialize_exported`
    Reload an exported forward-model artifact from bytes.
:func:`distribute_batched`
    Run a batched callable data-parallel across a device mesh.
:func:`enable_compilation_cache`
    Point JAX's persistent compilation cache at a directory.
:func:`export_forward`
    Export a forward function to a portable StableHLO artifact.
:func:`serialize_exported`
    Serialize an exported forward-model artifact to bytes.
:class:`ExportError`
    Raised when a forward model cannot be exported as-is.
:func:`gauss_hermite_nodes_weights`
    Gauss-Hermite quadrature nodes and weights for Gaussian
    averaging integrals.
:func:`incident_wavevector`
    Calculate incident electron wavevector from beam parameters.
:func:`incidence_angles_to_radians`
    Convert public grazing/azimuth degrees to internal radian angles.
:func:`interaction_constant`
    Relativistic electron interaction constant for multislice
    calculations.
:func:`jax_safe`
    Wrap a function to convert positional args to JAX arrays.
:func:`shard_array`
    Shard an array across devices for parallel processing.
:func:`wavelength_ang`
    Calculate relativistic electron wavelength in angstroms.

Notes
-----
The Bessel implementations use piecewise polynomial approximations
(Abramowitz & Stegun) for :func:`bessel_k0` and :func:`bessel_k1`,
and series / asymptotic expansions for the general-order
:func:`bessel_kv`. These are needed by the Lobato-van Dyck projected
potential, which is expressed analytically in terms of
:math:`K_0` and :math:`K_1`.

The electron-beam utilities (:func:`wavelength_ang`,
:func:`incidence_angles_to_radians`, :func:`incident_wavevector`,
:func:`interaction_constant`) live here rather than in :mod:`rheedium.simul`
to break circular import chains between simulation sub-modules.
"""

from .caching import enable_compilation_cache
from .exporting import (
    ExportError,
    bucketize_grid,
    deserialize_exported,
    export_forward,
    serialize_exported,
)
from .parallel import distribute_batched, shard_array
from .quadrature import gauss_hermite_nodes_weights
from .simul_utils import (
    incidence_angles_to_radians,
    incident_wavevector,
    interaction_constant,
    wavelength_ang,
)
from .special import bessel_k0, bessel_k1, bessel_kv
from .wrappers import jax_safe

__all__: list[str] = [
    "ExportError",
    "bessel_k0",
    "bessel_k1",
    "bessel_kv",
    "bucketize_grid",
    "deserialize_exported",
    "distribute_batched",
    "enable_compilation_cache",
    "export_forward",
    "gauss_hermite_nodes_weights",
    "incidence_angles_to_radians",
    "incident_wavevector",
    "interaction_constant",
    "jax_safe",
    "serialize_exported",
    "shard_array",
    "wavelength_ang",
]
