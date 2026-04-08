"""Differentiable procedural models and preprocessing for RHEED.

Extended Summary
----------------
This module collects forward procedural operations that prepare
experimental data, build surface slabs, and modify crystal or surface
structure in differentiable ways. The intent is that inverse
algorithms in :mod:`rheedium.recon` can optimize through these
procedural transforms against experimental RHEED data.

Routine Listings
----------------
:func:`add_adsorbate_layer`
    Add an adsorbate layer with fractional coverage to a slab.
:func:`apply_surface_reconstruction`
    Apply an m x n surface reconstruction to a slab.
:func:`create_surface_slab`
    Construct a surface slab from a bulk crystal.
:func:`gaas001_2x4`
    GaAs(001)-2x4 beta2 As-rich surface slab.
:func:`incoherent_domain_average`
    Incoherently average RHEED patterns from multiple domains.
:func:`log_intensity_transform`
    Differentiable log transform for dynamic range compression.
:func:`mgo001_bulk_terminated`
    Bulk-terminated MgO(001) rocksalt surface slab.
:func:`normalize_image`
    Normalize image intensities to [0, 1] range.
:func:`preprocess_experimental`
    Full differentiable preprocessing pipeline for experimental images.
:func:`si100_2x1`
    Si(100)-2x1 symmetric dimer row surface slab.
:func:`si111_1x1`
    Bulk-terminated Si(111) surface slab.
:func:`si111_7x7`
    Si(111)-7x7 DAS reconstruction surface slab.
:func:`soft_threshold_mask`
    Create a differentiable soft mask from a distance field.
:func:`srtio3_001_2x1`
    SrTiO3(001)-2x1 TiO2-terminated surface slab.
:func:`subtract_background`
    Subtract dark frame background with non-negative clipping.
:func:`vicinal_surface_step_splitting`
    CTR intensity modification from periodic surface steps.

Notes
-----
``rheedium.procs`` is the forward tool chest for differentiable data and
structure processing. This namespace is intentionally broader than the
inverse-facing :mod:`rheedium.recon` package.
"""

from .library import (
    gaas001_2x4,
    mgo001_bulk_terminated,
    si100_2x1,
    si111_1x1,
    si111_7x7,
    srtio3_001_2x1,
)
from .preprocessing import (
    log_intensity_transform,
    normalize_image,
    preprocess_experimental,
    soft_threshold_mask,
    subtract_background,
)
from .surface_builder import (
    add_adsorbate_layer,
    apply_surface_reconstruction,
    create_surface_slab,
)
from .surface_modifier import (
    incoherent_domain_average,
    vicinal_surface_step_splitting,
)

__all__: list[str] = [
    "add_adsorbate_layer",
    "apply_surface_reconstruction",
    "create_surface_slab",
    "gaas001_2x4",
    "incoherent_domain_average",
    "log_intensity_transform",
    "mgo001_bulk_terminated",
    "normalize_image",
    "preprocess_experimental",
    "si100_2x1",
    "si111_1x1",
    "si111_7x7",
    "soft_threshold_mask",
    "srtio3_001_2x1",
    "subtract_background",
    "vicinal_surface_step_splitting",
]
