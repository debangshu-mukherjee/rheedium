# Experimental RHEED Data Provenance

This directory stores downloaded experimental reference data used by
`rheedium.audit`.

Downloaded payloads under this directory are intentionally git-ignored.
Only provenance documents and lightweight tracking files should be kept
under version control.

Local download helper:

- `python tests/test_data/reference_data/experimental/download_reference_data.py --list`
- `python tests/test_data/reference_data/experimental/download_reference_data.py sto_homoepitaxy_zenodo_8000271 --dry-run`
- Add `--include-raw` only when the large raw HDF5 file is actually needed.

## Downloaded Datasets

### SrTiO3 Homoepitaxy Growth Dynamics

- DOI: `10.5281/zenodo.8000271`
- DOI URL: <https://doi.org/10.5281/zenodo.8000271>
- Zenodo record: <https://zenodo.org/records/8000271>
- Title: `Datasets for Work "Predicting Pulsed-Laser Deposition SrTiO3 Homoepitaxy Growth Dynamics using High-Speed Reflection High-Energy Electron Diffraction"`
- Initial local download target:
  `tests/test_data/reference_data/experimental/sto_homoepitaxy_zenodo_8000271/`
- Files selected for the initial benchmark seed:
  - `STO_STO_test6_06292022-standard.h5`
  - `AFM.zip`
  - `XRD.zip`
- Notes:
  - The full Zenodo record is much larger than a typical test fixture set.
  - The initial local seed uses the smallest raw STO HDF5 file from the record
    plus the lightweight AFM/XRD companion archives.
