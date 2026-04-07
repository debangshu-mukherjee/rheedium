# RHEED Reference Bundle

These fixtures back the `rheedium.audit` benchmark pipeline.

Current status:

- The stored images are synthetic detector references generated from
  `ewald_simulator` plus `instrument_broadened_pattern`.
- The goal of the bundle today is to validate metadata format,
  benchmark mechanics, and image-space metrics.
- Calibrated experimental references can replace these files later
  without changing the audit API.

Each case is stored as:

- `*_metadata.json`: benchmark metadata and simulation parameters
- `*.npz`: compressed detector image with a single `image` array
