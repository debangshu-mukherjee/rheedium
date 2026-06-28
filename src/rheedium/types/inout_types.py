"""Type carriers for input/output metadata.

Routine Listings
----------------
:class:`FrameMetadata`
    Per-frame metadata extracted from TIFF tags.
"""

from beartype.typing import NamedTuple


class FrameMetadata(NamedTuple):
    """Per-frame metadata extracted from TIFF tags.

    Attributes
    ----------
    exposure_time_s : float
        Exposure time in seconds. NaN if not available.
    timestamp_s : float
        Timestamp in seconds since epoch. NaN if not available.
    description : str
        Image description string from TIFF tag. Empty if not available.
    frame_index : int
        Zero-based index of this frame in the sequence.

    Notes
    -----
    Metadata availability depends on the acquisition software. Missing fields
    default to NaN for numeric values or an empty string for text.
    """

    exposure_time_s: float
    timestamp_s: float
    description: str
    frame_index: int


__all__: list[str] = [
    "FrameMetadata",
]
