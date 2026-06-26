"""Layer-1 detector-image orchestration over distribution axes.

Layer 1 binds one or more statistical ``Distribution`` axes to Layer-0
coherent kernels, reduces them through the shared averaging framework, and
applies detector PSF/normalization. The legacy ``rheedium.simul.simulator``
module still re-exports these names for import compatibility.
"""

from .simulator import (
    checked_simulate_detector_image,
    simulate_detector_image,
    simulate_detector_image_instrument,
)

__all__: list[str] = [
    "checked_simulate_detector_image",
    "simulate_detector_image",
    "simulate_detector_image_instrument",
]
