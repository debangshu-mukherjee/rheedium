"""HDF5 serializer and deserializer for rheedium PyTrees.

Extended Summary
----------------
Provides functions for saving and loading rheedium PyTree objects to and
from HDF5 files. The implementation follows the ``diffpes`` serializer
layout but extends it to recurse through nested child pytrees so composite
objects like :class:`rheedium.recon.ReconstructionResult` can round-trip
without flattening nested parameter trees into ad hoc blobs.

Routine Listings
----------------
:func:`load_from_h5`
    Load one or more PyTrees from an HDF5 file.
:func:`save_to_h5`
    Save one or more PyTrees to an HDF5 file.

Notes
-----
The serializer supports the public rheedium PyTree containers:
``ElectronBeam``, ``CrystalStructure``, ``EwaldData``,
``PotentialSlices``, ``XYZData``, ``RHEEDPattern``, ``RHEEDImage``,
``SlicedCrystal``, ``SurfaceConfig``, ``DetectorGeometry``, and
``ReconstructionResult``. Nested ``dict`` / ``list`` / ``tuple``
containers are also supported inside those objects.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Union
from jaxtyping import Shaped
from numpy import ndarray as NDArray  # noqa: N812

from ..recon import ReconstructionResult
from ..types import (
    CrystalStructure,
    DetectorGeometry,
    ElectronBeam,
    EwaldData,
    PotentialSlices,
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    XYZData,
)

_ATTR_AUX: str = "_aux_data_json"
_ATTR_JSON: str = "_json_value"
_ATTR_KIND: str = "_node_kind"
_ATTR_KEYS: str = "_dict_keys_json"
_ATTR_LENGTH: str = "_num_items"
_ATTR_TYPE: str = "_pytree_type"
_JSON_KIND: str = "__rheedium_json_kind__"
_KIND_DICT: str = "dict"
_KIND_JSON: str = "json"
_KIND_LIST: str = "list"
_KIND_NONE: str = "none"
_KIND_PYTREE: str = "pytree"
_KIND_TUPLE: str = "tuple"
_JSON_COMPLEX: str = "complex"
_JSON_DICT: str = "dict"
_JSON_TUPLE: str = "tuple"


@dataclass(frozen=True)
class _PyTreeMeta:
    """Serialization metadata for a supported rheedium PyTree."""

    cls: Any
    children_fields: tuple[str, ...]
    aux_encoder: Callable[[Any], Any]
    aux_decoder: Callable[[Any], Any]
    uses_tree_methods: bool = True


def _encode_none(
    _aux: None,  # noqa: ARG001
) -> None:
    """Encode ``None`` auxiliary data for JSON storage."""
    return


def _decode_none(
    _val: None,  # noqa: ARG001
) -> None:
    """Decode JSON ``null`` back to ``None`` auxiliary data."""
    return


def _scalar_to_python(
    value: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Convert a scalar array-like value to a plain Python scalar."""
    scalar_array: NDArray = np.asarray(value)
    if scalar_array.ndim != 0:
        msg = "Only scalar auxiliary values can be JSON-encoded."
        raise TypeError(msg)
    return scalar_array.item()


def _encode_json_value(  # noqa: PLR0911
    value: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Convert a Python value into a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, complex):
        return {
            _JSON_KIND: _JSON_COMPLEX,
            "real": value.real,
            "imag": value.imag,
        }

    if isinstance(value, np.generic):
        return _encode_json_value(value.item())

    if hasattr(value, "shape"):
        scalar_array: NDArray = np.asarray(value)
        if scalar_array.ndim == 0:
            return _encode_json_value(scalar_array.item())

    if isinstance(value, list):
        return [_encode_json_value(item) for item in value]

    if isinstance(value, tuple):
        return {
            _JSON_KIND: _JSON_TUPLE,
            "items": [_encode_json_value(item) for item in value],
        }

    if isinstance(value, dict):
        return {
            _JSON_KIND: _JSON_DICT,
            "items": [
                [_encode_json_value(key), _encode_json_value(item)]
                for key, item in value.items()
            ],
        }

    msg = f"Unsupported JSON value type: {type(value).__name__}"
    raise TypeError(msg)


def _decode_json_value(
    value: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Reconstruct a Python value from :func:`_encode_json_value` output."""
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]

    if isinstance(value, dict):
        marker: Any = value.get(_JSON_KIND)
        if marker == _JSON_COMPLEX:
            return complex(value["real"], value["imag"])
        if marker == _JSON_TUPLE:
            return tuple(_decode_json_value(item) for item in value["items"])
        if marker == _JSON_DICT:
            return {
                _decode_json_value(key): _decode_json_value(item)
                for key, item in value["items"]
            }
        return {key: _decode_json_value(item) for key, item in value.items()}

    return value


def _encode_scalar_tuple(
    aux: tuple[Any, ...],  # noqa: ANN401
) -> list[Any]:
    """Encode a tuple of scalar auxiliary values for JSON storage."""
    return [_scalar_to_python(item) for item in aux]


def _decode_potential_slices_aux(
    value: Any,  # noqa: ANN401
) -> tuple[Any, Any, Any]:
    """Decode PotentialSlices scalar auxiliary metadata."""
    decoded: Any = _decode_json_value(value)
    components: list[Any] = [
        jnp.asarray(component, dtype=jnp.float64) for component in decoded
    ]
    return (components[0], components[1], components[2])


def _encode_json_aux(
    aux: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Encode JSON-like auxiliary data."""
    return _encode_json_value(aux)


def _decode_json_aux(
    value: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Decode JSON-like auxiliary data."""
    return _decode_json_value(value)


_PYTREE_REGISTRY: dict[str, _PyTreeMeta] = {
    "ElectronBeam": _PyTreeMeta(
        cls=ElectronBeam,
        children_fields=ElectronBeam._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "CrystalStructure": _PyTreeMeta(
        cls=CrystalStructure,
        children_fields=CrystalStructure._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "EwaldData": _PyTreeMeta(
        cls=EwaldData,
        children_fields=EwaldData._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "PotentialSlices": _PyTreeMeta(
        cls=PotentialSlices,
        children_fields=("slices",),
        aux_encoder=_encode_scalar_tuple,
        aux_decoder=_decode_potential_slices_aux,
    ),
    "XYZData": _PyTreeMeta(
        cls=XYZData,
        children_fields=(
            "positions",
            "atomic_numbers",
            "lattice",
            "stress",
            "energy",
        ),
        aux_encoder=_encode_json_aux,
        aux_decoder=_decode_json_aux,
    ),
    "RHEEDPattern": _PyTreeMeta(
        cls=RHEEDPattern,
        children_fields=RHEEDPattern._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "RHEEDImage": _PyTreeMeta(
        cls=RHEEDImage,
        children_fields=RHEEDImage._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "SlicedCrystal": _PyTreeMeta(
        cls=SlicedCrystal,
        children_fields=SlicedCrystal._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
    "SurfaceConfig": _PyTreeMeta(
        cls=SurfaceConfig,
        children_fields=SurfaceConfig._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
        uses_tree_methods=False,
    ),
    "DetectorGeometry": _PyTreeMeta(
        cls=DetectorGeometry,
        children_fields=DetectorGeometry._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
        uses_tree_methods=False,
    ),
    "ReconstructionResult": _PyTreeMeta(
        cls=ReconstructionResult,
        children_fields=ReconstructionResult._fields,
        aux_encoder=_encode_none,
        aux_decoder=_decode_none,
    ),
}
_PYTREE_REGISTRY_BY_CLASS: dict[type[Any], _PyTreeMeta] = {
    meta.cls: meta for meta in _PYTREE_REGISTRY.values()
}


def _flatten_pytree(
    pytree: Any,  # noqa: ANN401
    meta: _PyTreeMeta,
) -> tuple[tuple[Any, ...], Any]:  # noqa: ANN401
    """Flatten a supported PyTree using its registered metadata."""
    if meta.uses_tree_methods:
        children: tuple[Any, ...]
        aux_data: Any
        children, aux_data = pytree.tree_flatten()
        return tuple(children), aux_data
    return tuple(pytree), None


def _unflatten_pytree(
    meta: _PyTreeMeta,
    aux_data: Any,  # noqa: ANN401
    children: tuple[Any, ...],  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Reconstruct a supported PyTree from serialized children."""
    if meta.uses_tree_methods:
        return meta.cls.tree_unflatten(aux_data, children)
    return meta.cls(*children)


@beartype
def _dataset_write_kwargs(
    data: Shaped[NDArray, "..."],
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, tuple[int, ...]]],
) -> dict[str, Any]:
    """Build ``create_dataset`` keyword arguments for one child array."""
    if data.ndim == 0:
        return {}

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    if shuffle:
        kwargs["shuffle"] = True
    if fletcher32:
        kwargs["fletcher32"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return kwargs


def _write_value(
    parent: Any,  # noqa: ANN401
    name: str,
    value: Any,  # noqa: ANN401
    *,
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, tuple[int, ...]]],
) -> None:
    """Write one supported value into an HDF5 group."""
    meta: Optional[_PyTreeMeta] = _PYTREE_REGISTRY_BY_CLASS.get(type(value))
    if meta is not None:
        group: Any = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_PYTREE
        group.attrs[_ATTR_TYPE] = type(value).__name__
        children: tuple[Any, ...]
        aux_data: Any
        children, aux_data = _flatten_pytree(value, meta)
        group.attrs[_ATTR_AUX] = json.dumps(meta.aux_encoder(aux_data))
        for field_name, child in zip(
            meta.children_fields,
            children,
            strict=True,
        ):
            _write_value(
                group,
                field_name,
                child,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=chunks,
            )
        return

    if value is None:
        group = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_NONE
        return

    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_DICT
        group.attrs[_ATTR_KEYS] = json.dumps(
            [_encode_json_value(key) for key in value]
        )
        group.attrs[_ATTR_LENGTH] = len(value)
        for index, item in enumerate(value.values()):
            _write_value(
                group,
                f"item_{index}",
                item,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=chunks,
            )
        return

    if isinstance(value, list):
        group = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_LIST
        group.attrs[_ATTR_LENGTH] = len(value)
        for index, item in enumerate(value):
            _write_value(
                group,
                f"item_{index}",
                item,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=chunks,
            )
        return

    if isinstance(value, tuple):
        group = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_TUPLE
        group.attrs[_ATTR_LENGTH] = len(value)
        for index, item in enumerate(value):
            _write_value(
                group,
                f"item_{index}",
                item,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=chunks,
            )
        return

    if isinstance(value, (bool, int, float, complex, str)):
        group = parent.create_group(name)
        group.attrs[_ATTR_KIND] = _KIND_JSON
        group.attrs[_ATTR_JSON] = json.dumps(_encode_json_value(value))
        return

    array_value: NDArray = np.asarray(value)
    if array_value.dtype.kind in {"O", "S", "U"}:
        msg = f"Unsupported array dtype for HDF5 storage: {array_value.dtype}"
        raise TypeError(msg)

    dataset_kwargs: dict[str, Any] = _dataset_write_kwargs(
        data=array_value,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        fletcher32=fletcher32,
        chunks=chunks,
    )
    parent.create_dataset(
        name,
        data=array_value,
        **dataset_kwargs,
    )


def _read_value(
    node: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Read one supported value from an HDF5 dataset or group."""
    if isinstance(node, h5py.Dataset):
        data: Shaped[NDArray, "..."] = node[()]
        return jnp.asarray(data)

    kind: str = str(node.attrs[_ATTR_KIND])
    if kind == _KIND_NONE:
        return None

    if kind == _KIND_JSON:
        return _decode_json_value(json.loads(str(node.attrs[_ATTR_JSON])))

    if kind in {_KIND_LIST, _KIND_TUPLE}:
        length: int = int(node.attrs[_ATTR_LENGTH])
        items: list[Any] = [
            _read_value(node[f"item_{index}"]) for index in range(length)
        ]
        return items if kind == _KIND_LIST else tuple(items)

    if kind == _KIND_DICT:
        raw_keys: Any = json.loads(str(node.attrs[_ATTR_KEYS]))
        keys: list[Any] = [_decode_json_value(item) for item in raw_keys]
        values: list[Any] = [
            _read_value(node[f"item_{index}"]) for index in range(len(keys))
        ]
        return dict(zip(keys, values, strict=True))

    if kind == _KIND_PYTREE:
        type_name: str = str(node.attrs[_ATTR_TYPE])
        if type_name not in _PYTREE_REGISTRY:
            msg = f"Unknown PyTree type: {type_name}"
            raise TypeError(msg)

        meta: _PyTreeMeta = _PYTREE_REGISTRY[type_name]
        aux_json: Any = json.loads(str(node.attrs[_ATTR_AUX]))
        aux_data: Any = meta.aux_decoder(aux_json)
        children: tuple[Any, ...] = tuple(
            _read_value(node[field_name])
            for field_name in meta.children_fields
        )
        return _unflatten_pytree(meta, aux_data, children)

    msg = f"Unknown HDF5 node kind: {kind}"
    raise TypeError(msg)


@beartype
def save_to_h5(
    path: Union[str, Path],
    /,
    *,
    compression: Optional[str] = None,
    compression_opts: Any = None,  # noqa: ANN401
    shuffle: bool = False,
    fletcher32: bool = False,
    chunks: Optional[Union[bool, tuple[int, ...]]] = None,
    **pytrees: Any,  # noqa: ANN401
) -> None:
    """Save one or more named PyTrees to an HDF5 file.

    Parameters
    ----------
    path : Union[str, Path]
        File path for the HDF5 file to create.
    compression : Optional[str], optional
        HDF5 compression filter name (for example ``"gzip"``).
        Applied to non-scalar datasets only.
    compression_opts : Any, optional
        Compression options passed through to h5py.
    shuffle : bool, optional
        If True, enable the HDF5 shuffle filter on non-scalar datasets.
    fletcher32 : bool, optional
        If True, enable HDF5 Fletcher32 checksums on non-scalar datasets.
    chunks : Optional[Union[bool, tuple[int, ...]]], optional
        Chunking policy for non-scalar datasets.
    **pytrees : Any
        Named PyTrees or nested child pytrees to serialize.

    Raises
    ------
    ValueError
        If no values are provided.
    ValueError
        If ``compression_opts`` is provided without ``compression``.
    TypeError
        If an unsupported value is encountered.
    """
    if not pytrees:
        msg = "At least one PyTree must be provided."
        raise ValueError(msg)
    if compression is None and compression_opts is not None:
        msg = "compression_opts requires compression to be set."
        raise ValueError(msg)

    file_path: Path = Path(path)
    with h5py.File(file_path, "w") as handle:
        for group_name, pytree in pytrees.items():
            _write_value(
                handle,
                group_name,
                pytree,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fletcher32=fletcher32,
                chunks=chunks,
            )


@beartype
def load_from_h5(
    path: Union[str, Path],
    name: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load one or more PyTrees from an HDF5 file.

    Parameters
    ----------
    path : Union[str, Path]
        File path to the HDF5 file to read.
    name : Optional[str], optional
        Name of a specific top-level object to load. If ``None``, all
        top-level objects are loaded and returned as a dict.

    Returns
    -------
    result : Any
        The requested PyTree when ``name`` is given, otherwise a mapping
        from top-level names to loaded PyTrees.

    Raises
    ------
    KeyError
        If ``name`` is given but is not present in the file.
    TypeError
        If the file contains an unsupported serialized node.
    """
    file_path: Path = Path(path)
    with h5py.File(file_path, "r") as handle:
        if name is not None:
            if name not in handle:
                msg = f"Group '{name}' not found in {path}"
                raise KeyError(msg)
            return _read_value(handle[name])

        result: dict[str, Any] = {}
        for group_name in handle:
            result[group_name] = _read_value(handle[group_name])
        return result


__all__: list[str] = [
    "load_from_h5",
    "save_to_h5",
]
