"""Download helper for local experimental RHEED reference datasets.

This script stages experimental datasets into the git-ignored
``tests/test_data/reference_data/experimental/`` directory. It is
intended for local benchmarking work only; downloaded payloads are not
tracked by git.

By default, the script downloads only lightweight metadata and companion
archives. Large raw detector files are downloaded only when
``--include-raw`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_EXPERIMENTAL_DIR: Final[Path] = Path(__file__).resolve().parent
_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class DatasetFile:
    """One downloadable file in an experimental dataset manifest."""

    filename: str
    url: str
    size_bytes: int
    md5: str | None
    description: str
    is_raw: bool = False


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest for one experimental RHEED dataset."""

    slug: str
    title: str
    doi: str
    doi_url: str
    record_url: str
    local_dirname: str
    summary: str
    files: tuple[DatasetFile, ...]


_DATASETS: Final[dict[str, DatasetManifest]] = {
    "sto_homoepitaxy_zenodo_8000271": DatasetManifest(
        slug="sto_homoepitaxy_zenodo_8000271",
        title=(
            'Datasets for Work "Predicting Pulsed-Laser Deposition '
            "SrTiO3 Homoepitaxy Growth Dynamics using High-Speed "
            'Reflection High-Energy Electron Diffraction"'
        ),
        doi="10.5281/zenodo.8000271",
        doi_url="https://doi.org/10.5281/zenodo.8000271",
        record_url="https://zenodo.org/records/8000271",
        local_dirname="sto_homoepitaxy_zenodo_8000271",
        summary=(
            "High-speed SrTiO3 homoepitaxy RHEED dataset. Default download "
            "pulls record metadata plus AFM/XRD companion archives; the "
            "26 GB raw HDF5 file is opt-in via --include-raw."
        ),
        files=(
            DatasetFile(
                filename="record_8000271.json",
                url="https://zenodo.org/api/records/8000271",
                size_bytes=7081,
                md5=None,
                description="Zenodo record metadata JSON.",
            ),
            DatasetFile(
                filename="AFM.zip",
                url=(
                    "https://zenodo.org/api/records/8000271/files/"
                    "AFM.zip/content"
                ),
                size_bytes=7710609,
                md5="24d2a7511dd765da97cff95530de4081",
                description="AFM companion archive.",
            ),
            DatasetFile(
                filename="XRD.zip",
                url=(
                    "https://zenodo.org/api/records/8000271/files/"
                    "XRD.zip/content"
                ),
                size_bytes=673716,
                md5="4458bd2c46a01026d68db3b1c37f8e30",
                description="XRD companion archive.",
            ),
            DatasetFile(
                filename="STO_STO_test6_06292022-standard.h5",
                url=(
                    "https://zenodo.org/api/records/8000271/files/"
                    "STO_STO_test6_06292022-standard.h5/content"
                ),
                size_bytes=28112046144,
                md5="08f062168e101d64344ec605bf55d601",
                description="Smallest raw STO RHEED HDF5 file in the record.",
                is_raw=True,
            ),
        ),
    )
}


def _format_size(size_bytes: int) -> str:
    """Format a byte count with binary units."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("Unreachable size-format branch")


def _iter_selected_files(
    manifest: DatasetManifest,
    include_raw: bool,
) -> tuple[DatasetFile, ...]:
    """Return the files selected for download under current CLI flags."""
    return tuple(
        dataset_file
        for dataset_file in manifest.files
        if include_raw or not dataset_file.is_raw
    )


def _compute_md5(path: Path) -> str:
    """Compute the MD5 hash of a local file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    dataset_file: DatasetFile,
    destination: Path,
    force: bool,
) -> None:
    """Download one manifest file to the target location."""
    if destination.exists() and not force:
        print(f"skip  {destination.name} (already exists)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(dataset_file.url, headers={"User-Agent": "rheedium"})
    try:
        with urlopen(request) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
    except (HTTPError, URLError) as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Failed to download {dataset_file.filename}: {exc}"
        ) from exc

    if dataset_file.md5 is not None:
        digest = _compute_md5(destination)
        if digest != dataset_file.md5:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dataset_file.filename}: "
                f"expected {dataset_file.md5}, got {digest}"
            )

    print(f"saved {destination.name}")


def _list_datasets() -> None:
    """Print the built-in experimental dataset manifests."""
    for manifest in _DATASETS.values():
        print(f"{manifest.slug}")
        print(f"  title: {manifest.title}")
        print(f"  doi:   {manifest.doi_url}")
        print(f"  note:  {manifest.summary}")
        for dataset_file in manifest.files:
            raw_tag = " [raw]" if dataset_file.is_raw else ""
            print(
                "  - "
                f"{dataset_file.filename} "
                f"({_format_size(dataset_file.size_bytes)}){raw_tag}"
            )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download local experimental RHEED reference datasets into the "
            "git-ignored reference_data/experimental tree."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=tuple(sorted(_DATASETS)),
        help="Dataset manifest slug to download.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset manifests and exit.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include large raw detector payloads in the download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser


def main() -> None:
    """Run the experimental dataset download helper."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.list:
        _list_datasets()
        return

    if args.dataset is None:
        parser.error("dataset is required unless --list is used")

    manifest = _DATASETS[args.dataset]
    selected_files = _iter_selected_files(
        manifest=manifest,
        include_raw=args.include_raw,
    )
    destination_dir = _EXPERIMENTAL_DIR / manifest.local_dirname

    print(f"dataset: {manifest.slug}")
    print(f"target:  {destination_dir}")
    print(f"doi:     {manifest.doi_url}")
    print("files:")
    for dataset_file in selected_files:
        raw_tag = " [raw]" if dataset_file.is_raw else ""
        print(
            "  - "
            f"{dataset_file.filename} "
            f"({_format_size(dataset_file.size_bytes)}){raw_tag}"
        )

    if args.dry_run:
        return

    for dataset_file in selected_files:
        _download_file(
            dataset_file=dataset_file,
            destination=destination_dir / dataset_file.filename,
            force=args.force,
        )


if __name__ == "__main__":
    main()
