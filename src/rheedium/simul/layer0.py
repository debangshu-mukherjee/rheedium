"""Layer-0 coherent simulation kernels and detector rendering helpers.

Layer 0 owns direct coherent or sparse simulation primitives: Ewald and
multislice kernels, detector projection, and dense-field rendering. The legacy
``rheedium.simul.simulator`` module still re-exports these names for import
compatibility; new code should prefer this module when it needs one coherent
kernel evaluation rather than ensemble orchestration.
"""

from .simulator import (
    checked_ewald_simulator,
    checked_multislice_propagate,
    checked_multislice_simulator,
    compute_kinematic_intensities_with_ctrs,
    detector_extent_mm,
    ewald_simulator,
    find_ctr_ewald_intersection,
    find_kinematic_reflections,
    kinematic_amplitude,
    log_compress_image,
    multislice_amplitude,
    multislice_detector_amplitude,
    multislice_propagate,
    multislice_simulator,
    project_on_detector_geometry,
    render_amplitude_to_field,
    render_ctr_amplitude_to_field,
    render_pattern_to_image,
    sliced_crystal_to_projected_potential_slices,
)

__all__: list[str] = [
    "checked_ewald_simulator",
    "checked_multislice_propagate",
    "checked_multislice_simulator",
    "compute_kinematic_intensities_with_ctrs",
    "detector_extent_mm",
    "ewald_simulator",
    "find_ctr_ewald_intersection",
    "find_kinematic_reflections",
    "kinematic_amplitude",
    "log_compress_image",
    "multislice_amplitude",
    "multislice_detector_amplitude",
    "multislice_propagate",
    "multislice_simulator",
    "project_on_detector_geometry",
    "render_amplitude_to_field",
    "render_ctr_amplitude_to_field",
    "render_pattern_to_image",
    "sliced_crystal_to_projected_potential_slices",
]
