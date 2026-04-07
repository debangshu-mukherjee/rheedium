"""TIFF image sequence loading and preprocessing for RHEED analysis.

Extended Summary
----------------
This module provides utilities for loading TIFF image sequences from
RHEED experiments, extracting per-frame metadata (exposure, timestamp),
performing flat-field and background corrections, and automatically
detecting the specular beam center for detector geometry calibration.

Routine Listings
----------------
:class:`FrameMetadata`
    Per-frame metadata extracted from TIFF tags.
:func:`load_tiff_sequence`
    Load ordered TIFF stack into a JAX array.
:func:`extract_frame_metadata`
    Extract exposure time, timestamp, and description from a TIFF page.
:func:`normalize_sequence`
    Background subtraction, flat-field correction, and normalization.
:func:`detect_beam_center`
    Locate the specular spot position automatically.
:func:`load_tiff_as_rheed_image`
    Load a single TIFF frame and return a RHEEDImage PyTree.

Notes
-----
TIFF loading uses ``tifffile`` for robust multi-page and BigTIFF
support. Image data is converted to float64 JAX arrays on load.
The ``detect_beam_center`` function returns coordinates suitable for
direct assignment to ``DetectorGeometry.center_pixel``.
"""

import contextlib
import logging
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tifffile
from beartype import beartype
from beartype.typing import List, NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812

from rheedium.types import RHEEDImage, create_rheed_image, scalar_float
from rheedium.types.constants import (
    H_OVER_SQRT_2ME_ANG_VSQRT,
    RELATIVISTIC_COEFF_PER_V,
)

logger: logging.Logger = logging.getLogger(__name__)

_EXIF_EXPOSURE_TIME_TAG: int = 33434
_SINGLE_FRAME_NDIM: int = 2
_MULTIPAGE_FRAME_NDIM: int = 3


class FrameMetadata(NamedTuple):
    """Per-frame metadata extracted from TIFF tags.

    Attributes
    ----------
    exposure_time_s : float
        Exposure time in seconds. NaN if not available.
    timestamp_s : float
        Timestamp in seconds since epoch. NaN if not available.
    description : str
        Image description string from TIFF tag. Empty if not
        available.
    frame_index : int
        Zero-based index of this frame in the sequence.

    Notes
    -----
    Metadata availability depends on the acquisition software.
    Missing fields default to NaN (numeric) or empty string (text).
    """

    exposure_time_s: float
    timestamp_s: float
    description: str
    frame_index: int


@beartype
def extract_frame_metadata(
    page: tifffile.TiffPage,
    frame_index: int = 0,
) -> FrameMetadata:
    """Extract metadata from a single TIFF page.

    Parameters
    ----------
    page : tifffile.TiffPage
        A TIFF page object from ``tifffile``.
    frame_index : int, optional
        Zero-based frame index to record. Default: 0

    Returns
    -------
    metadata : FrameMetadata
        Extracted metadata with exposure time, timestamp,
        description, and frame index.

    Notes
    -----
    1. **Exposure time** --
       Read from TIFF tag 33434 (ExposureTime) if present,
       otherwise from the image description via regex.
    2. **Timestamp** --
       Read from ``page.datetime`` and convert to epoch seconds,
       or extracted from description.
    3. **Description** --
       Read from ``page.description`` tag.

    Examples
    --------
    >>> import tifffile
    >>> import rheedium as rh
    >>> with tifffile.TiffFile("frame.tif") as tif:
    ...     meta = rh.inout.extract_frame_metadata(tif.pages[0])
    >>> meta.frame_index
    0
    """
    exposure_time: float = float("nan")
    timestamp: float = float("nan")
    description: str = ""

    tags: tifffile.TiffTags = page.tags
    if _EXIF_EXPOSURE_TIME_TAG in tags:
        exposure_time = float(tags[_EXIF_EXPOSURE_TIME_TAG].value)

    try:
        dt = page.datetime
        if dt is not None:
            timestamp = dt.timestamp()
    except (AttributeError, ValueError, TypeError):
        pass

    try:
        description = str(page.description) if page.description else ""
    except (AttributeError, ValueError):
        description = ""

    if np.isnan(exposure_time) and description:
        match = re.search(r"[Ee]xposure[\s:=]*([0-9.eE+-]+)", description)
        if match:
            with contextlib.suppress(ValueError):
                exposure_time = float(match.group(1))

    if np.isnan(timestamp) and description:
        match = re.search(r"[Tt]imestamp[\s:=]*([0-9.eE+-]+)", description)
        if match:
            with contextlib.suppress(ValueError):
                timestamp = float(match.group(1))

    return FrameMetadata(
        exposure_time_s=exposure_time,
        timestamp_s=timestamp,
        description=description,
        frame_index=frame_index,
    )


@beartype
def load_tiff_sequence(
    path: Union[str, Path],
    sort_by: str = "name",
) -> Tuple[Float[Array, "T H W"], List[FrameMetadata]]:
    """Load ordered TIFF stack into a JAX float64 array.

    Extended Summary
    ----------------
    Handles both multi-page TIFFs (single file with T pages) and
    directories of single-frame TIFFs. Files are sorted by filename
    or by embedded timestamp metadata.

    Parameters
    ----------
    path : Union[str, Path]
        Path to a multi-page TIFF file or a directory containing
        single-frame TIFF files (``*.tif`` or ``*.tiff``).
    sort_by : str, optional
        Sorting method for directory of files. ``"name"`` sorts
        lexicographically by filename. ``"timestamp"`` sorts by
        embedded timestamp metadata (falls back to name if
        timestamps are unavailable). Default: ``"name"``

    Returns
    -------
    sequence : Float[Array, "T H W"]
        Image stack as float64 JAX array with shape
        ``(n_frames, height, width)``.
    metadata : List[FrameMetadata]
        Per-frame metadata in the same order as the stack.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is a directory with no TIFF files, or if
        ``sort_by`` is not ``"name"`` or ``"timestamp"``.

    Notes
    -----
    1. **Resolve path** --
       Determine if single multi-page file or directory.
    2. **Load frames** --
       Use ``tifffile`` to read image data as NumPy arrays.
    3. **Sort** --
       Apply requested sorting to directory sequences.
    4. **Extract metadata** --
       Call :func:`extract_frame_metadata` for each frame.
    5. **Convert to JAX** --
       Cast to float64 JAX array.

    Examples
    --------
    >>> import rheedium as rh
    >>> seq, meta = rh.inout.load_tiff_sequence("rheed_data/")
    >>> seq.shape
    (100, 480, 640)
    >>> meta[0].frame_index
    0
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if sort_by not in ("name", "timestamp"):
        raise ValueError(
            f"sort_by must be 'name' or 'timestamp', got '{sort_by}'"
        )

    if path.is_file():
        return _load_multipage_tiff(path)

    return _load_tiff_directory(path, sort_by)


@beartype
def _load_multipage_tiff(
    filepath: Path,
) -> Tuple[Float[Array, "T H W"], List[FrameMetadata]]:
    """Load a multi-page TIFF file.

    Parameters
    ----------
    filepath : Path
        Path to the multi-page TIFF file.

    Returns
    -------
    sequence : Float[Array, "T H W"]
        Image stack.
    metadata : List[FrameMetadata]
        Per-frame metadata.
    """
    metadata_list: List[FrameMetadata] = []
    with tifffile.TiffFile(filepath) as tif:
        np_stack: Float[NDArray, "T H W"] = tif.asarray()
        if np_stack.ndim == _SINGLE_FRAME_NDIM:
            np_stack = np_stack[np.newaxis, :, :]
        for idx, page in enumerate(tif.pages):
            metadata_list.append(extract_frame_metadata(page, idx))

    sequence: Float[Array, "T H W"] = jnp.asarray(np_stack, dtype=jnp.float64)
    return sequence, metadata_list


@beartype
def _load_tiff_directory(
    dirpath: Path,
    sort_by: str,
) -> Tuple[Float[Array, "T H W"], List[FrameMetadata]]:
    """Load a directory of single-frame TIFF files.

    Parameters
    ----------
    dirpath : Path
        Path to directory containing TIFF files.
    sort_by : str
        Sorting method: ``"name"`` or ``"timestamp"``.

    Returns
    -------
    sequence : Float[Array, "T H W"]
        Image stack.
    metadata : List[FrameMetadata]
        Per-frame metadata.

    Raises
    ------
    ValueError
        If no TIFF files found in directory.
    """
    tiff_files: List[Path] = sorted(
        list(dirpath.glob("*.tif")) + list(dirpath.glob("*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {dirpath}")

    frames: List[Float[NDArray, "H W"]] = []
    metadata_list: List[FrameMetadata] = []

    for idx, fpath in enumerate(tiff_files):
        with tifffile.TiffFile(fpath) as tif:
            frame: Float[NDArray, "H W"] = tif.asarray()
            if frame.ndim == _MULTIPAGE_FRAME_NDIM:
                frame = frame[0]
            frames.append(frame)
            metadata_list.append(extract_frame_metadata(tif.pages[0], idx))

    if sort_by == "timestamp":
        timestamps: List[float] = [m.timestamp_s for m in metadata_list]
        if not all(np.isnan(t) for t in timestamps):
            sort_indices: List[int] = sorted(
                range(len(timestamps)), key=lambda i: timestamps[i]
            )
            frames = [frames[i] for i in sort_indices]
            metadata_list = [
                FrameMetadata(
                    exposure_time_s=metadata_list[i].exposure_time_s,
                    timestamp_s=metadata_list[i].timestamp_s,
                    description=metadata_list[i].description,
                    frame_index=new_idx,
                )
                for new_idx, i in enumerate(sort_indices)
            ]
        else:
            logger.warning(
                "No valid timestamps found; falling back to name sort."
            )

    np_stack: Float[NDArray, "T H W"] = np.stack(frames, axis=0)
    sequence: Float[Array, "T H W"] = jnp.asarray(np_stack, dtype=jnp.float64)
    return sequence, metadata_list


@jaxtyped(typechecker=beartype)
def normalize_sequence(
    sequence: Float[Array, "T H W"],
    background: Optional[Float[Array, "H W"]] = None,
    flat_field: Optional[Float[Array, "H W"]] = None,
) -> Float[Array, "T H W"]:
    """Background subtraction, flat-field correction, and normalization.

    Extended Summary
    ----------------
    Applies standard detector corrections to a RHEED image sequence:

    1. Subtract dark frame (background).
    2. Divide by flat-field response (sensitivity correction).
    3. Normalize each frame independently to [0, 1].

    All operations are differentiable via ``jax.grad``.

    Parameters
    ----------
    sequence : Float[Array, "T H W"]
        Raw image stack with shape ``(n_frames, height, width)``.
    background : Float[Array, "H W"], optional
        Dark frame to subtract. If ``None``, no subtraction.
    flat_field : Float[Array, "H W"], optional
        Flat-field correction image. Each frame is divided by this
        after background subtraction. Values near zero are clamped
        to avoid division artifacts. If ``None``, no correction.

    Returns
    -------
    normalized : Float[Array, "T H W"]
        Corrected and normalized image stack with per-frame values
        in [0, 1].

    Notes
    -----
    1. **Background subtraction** --
       ``I = max(I_raw - I_dark, 0)`` for each frame.
    2. **Flat-field correction** --
       ``I = I / max(flat, epsilon)`` where epsilon = 1e-12
       prevents division by zero.
    3. **Per-frame normalization** --
       Each frame is independently rescaled to [0, 1]:
       ``I_norm = (I - min) / max(max - min, 1e-12)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> seq = jnp.ones((10, 64, 64)) * 500.0
    >>> bg = jnp.ones((64, 64)) * 100.0
    >>> normed = rh.inout.normalize_sequence(seq, background=bg)
    >>> normed.shape
    (10, 64, 64)
    """
    corrected: Float[Array, "T H W"] = sequence

    if background is not None:
        corrected = jnp.maximum(corrected - background[None, :, :], 0.0)

    if flat_field is not None:
        safe_flat: Float[Array, "H W"] = jnp.maximum(flat_field, 1e-12)
        corrected = corrected / safe_flat[None, :, :]

    frame_min: Float[Array, "T 1 1"] = jnp.min(
        corrected, axis=(1, 2), keepdims=True
    )
    frame_max: Float[Array, "T 1 1"] = jnp.max(
        corrected, axis=(1, 2), keepdims=True
    )
    frame_range: Float[Array, "T 1 1"] = jnp.maximum(
        frame_max - frame_min, 1e-12
    )
    normalized: Float[Array, "T H W"] = (corrected - frame_min) / frame_range
    return normalized


@jaxtyped(typechecker=beartype)
def detect_beam_center(
    image: Float[Array, "H W"],
    sigma_pixels: scalar_float = 5.0,
    threshold_fraction: scalar_float = 0.5,
) -> Float[Array, " 2"]:
    """Locate the specular beam center in a RHEED image.

    Extended Summary
    ----------------
    Detects the position of the specular (direct) beam spot using a
    weighted centroid approach. The image is first smoothed with a
    Gaussian filter, then a threshold is applied to isolate the bright
    spot region. The center is computed as the intensity-weighted
    centroid of the thresholded region.

    The returned coordinates are suitable for direct assignment to
    ``DetectorGeometry.center_pixel``.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Single RHEED frame (typically without sample, showing only
        the direct beam).
    sigma_pixels : scalar_float, optional
        Gaussian smoothing sigma in pixels. Larger values improve
        robustness to noise but reduce precision for small spots.
        Default: 5.0
    threshold_fraction : scalar_float, optional
        Fraction of the smoothed maximum intensity used as the
        threshold for centroid computation. Range: (0, 1).
        Default: 0.5

    Returns
    -------
    center : Float[Array, " 2"]
        Beam center as ``[row, col]`` in pixel coordinates.

    Notes
    -----
    1. **Smooth** --
       Convolve with Gaussian via FFT (same approach as
       :func:`~rheedium.simul.beam_averaging.detector_psf_convolve`).
    2. **Threshold** --
       Compute ``I_thresh = threshold_fraction * max(I_smooth)``.
       Zero out pixels below threshold.
    3. **Centroid** --
       Compute intensity-weighted mean of row and column indices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import rheedium as rh
    >>> img = jnp.zeros((128, 128))
    >>> img = img.at[60:68, 62:70].set(1000.0)
    >>> center = rh.inout.detect_beam_center(img)
    >>> center.shape
    (2,)
    """
    height: int = image.shape[0]
    width: int = image.shape[1]

    freq_y: Float[Array, " H"] = jnp.fft.fftfreq(height)
    freq_x: Float[Array, " W"] = jnp.fft.fftfreq(width)
    fy: Float[Array, "H W"]
    fx: Float[Array, "H W"]
    fy, fx = jnp.meshgrid(freq_y, freq_x, indexing="ij")
    gaussian_filter: Float[Array, "H W"] = jnp.exp(
        -2.0 * jnp.pi**2 * sigma_pixels**2 * (fy**2 + fx**2)
    )
    smoothed: Float[Array, "H W"] = jnp.real(
        jnp.fft.ifft2(jnp.fft.fft2(image) * gaussian_filter)
    )
    smoothed = jnp.maximum(smoothed, 0.0)

    peak_val: scalar_float = jnp.max(smoothed)
    threshold: scalar_float = threshold_fraction * peak_val
    masked: Float[Array, "H W"] = jnp.where(
        smoothed >= threshold, smoothed, 0.0
    )

    row_indices: Float[Array, " H"] = jnp.arange(height, dtype=jnp.float64)
    col_indices: Float[Array, " W"] = jnp.arange(width, dtype=jnp.float64)
    row_grid: Float[Array, "H W"]
    col_grid: Float[Array, "H W"]
    row_grid, col_grid = jnp.meshgrid(row_indices, col_indices, indexing="ij")

    total_intensity: scalar_float = jnp.sum(masked)
    safe_total: scalar_float = jnp.maximum(total_intensity, 1e-12)
    center_row: scalar_float = jnp.sum(masked * row_grid) / safe_total
    center_col: scalar_float = jnp.sum(masked * col_grid) / safe_total

    center: Float[Array, " 2"] = jnp.array(
        [center_row, center_col], dtype=jnp.float64
    )
    return center


@beartype
def load_tiff_as_rheed_image(
    path: Union[str, Path],
    incoming_angle_deg: float,
    voltage_kv: float,
    detector_distance_mm: float,
    calibration: Union[float, Float[Array, "2"]] = 1.0,
    background: Optional[Float[Array, "H W"]] = None,
) -> RHEEDImage:
    r"""Load a single TIFF frame and return a RHEEDImage PyTree.

    Extended Summary
    ----------------
    Convenience function that loads a TIFF file, optionally subtracts
    a background, computes the electron wavelength from the
    accelerating voltage, and packages everything into a
    ``RHEEDImage`` PyTree ready for comparison with simulated
    patterns.

    Parameters
    ----------
    path : Union[str, Path]
        Path to a single-frame TIFF file. If the file is multi-page,
        only the first frame is used.
    incoming_angle_deg : float
        Grazing incidence angle in degrees.
    voltage_kv : float
        Accelerating voltage in kilovolts. Used to compute the
        relativistic electron wavelength.
    detector_distance_mm : float
        Sample-to-detector distance in mm.
    calibration : Union[float, Float[Array, "2"]], optional
        Pixel-to-physical-units calibration. Either a scalar (same
        for both axes) or a 2-element array ``[cal_x, cal_y]``.
        Default: 1.0
    background : Float[Array, "H W"], optional
        Dark frame to subtract before packaging. If ``None``, no
        subtraction is performed.

    Returns
    -------
    rheed_image : RHEEDImage
        Validated RHEEDImage PyTree containing the image and all
        experimental parameters.

    Notes
    -----
    1. **Load image** --
       Read the TIFF via ``tifffile``. If multi-page, take frame 0.
    2. **Background subtract** --
       If provided, subtract and clip to non-negative.
    3. **Compute wavelength** --
       Relativistic de Broglie wavelength:
       :math:`\\lambda = 12.2643 / \\sqrt{V(1 + 0.978476 \\times 10^{-6} V)}`
       where :math:`V` is voltage in volts.
    4. **Package** --
       Call :func:`~rheedium.types.create_rheed_image` with all
       parameters.

    Examples
    --------
    >>> import rheedium as rh
    >>> img = rh.inout.load_tiff_as_rheed_image(
    ...     "frame_001.tif",
    ...     incoming_angle_deg=2.0,
    ...     voltage_kv=20.0,
    ...     detector_distance_mm=350.0,
    ... )
    >>> img.img_array.shape
    (480, 640)
    """
    filepath: Path = Path(path)
    with tifffile.TiffFile(filepath) as tif:
        np_img: Float[NDArray, "H W"] = tif.asarray()

    if np_img.ndim > _SINGLE_FRAME_NDIM:
        np_img = np_img[0]

    img_array: Float[Array, "H W"] = jnp.asarray(np_img, dtype=jnp.float64)

    if background is not None:
        img_array = jnp.maximum(img_array - background, 0.0)

    voltage_v: float = voltage_kv * 1000.0
    corrected_voltage: float = voltage_v * (
        1.0 + RELATIVISTIC_COEFF_PER_V * voltage_v
    )
    electron_wavelength_ang: float = H_OVER_SQRT_2ME_ANG_VSQRT / np.sqrt(
        corrected_voltage
    )

    cal: Union[Float[Array, "2"], scalar_float] = jnp.asarray(
        calibration, dtype=jnp.float64
    )

    rheed_image: RHEEDImage = create_rheed_image(
        img_array=img_array,
        incoming_angle=jnp.float64(incoming_angle_deg),
        calibration=cal,
        electron_wavelength=jnp.float64(electron_wavelength_ang),
        detector_distance=jnp.float64(detector_distance_mm),
    )
    return rheed_image


__all__: list[str] = [
    "FrameMetadata",
    "detect_beam_center",
    "extract_frame_metadata",
    "load_tiff_as_rheed_image",
    "load_tiff_sequence",
    "normalize_sequence",
]
