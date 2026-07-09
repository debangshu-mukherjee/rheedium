# Automaton Index

This catalog lists every top-level Python script in `automatons/`.

Detector/intensity PNG artifacts use logarithmic display at their established
names and include an explicitly linear companion with a `_linear.png` suffix.

## Loop A — Online RHEED

| Script | Purpose |
|---|---|
| `rheed_ingest.py` | One detector frame to per-frame growth observables and stateless growth-state JSON. |
| `growth_monitor.py` | Rolling detector series to oscillation period, roughness trend, and transition flags. |

## Loop B — Theory In The Loop

| Script | Purpose |
|---|---|
| `forward_kinematic.py` | Structure file or smoke crystal to dual-scale kinematic RHEED detector PNGs and a raw `.npz` artifact. |
| `forward_multislice.py` | Structure file or smoke slab to dual-scale transmission multislice detector PNGs and a raw `.npz` artifact. |
| `forward_reflection.py` | Structure file or smoke slab to dual-scale edge-on reflection multislice detector PNGs and a raw `.npz` artifact. |
| `screen_xyz_ensemble.py` | Directory/glob of candidate structures to simulated RHEED ranking against a measured or simulated target. |
| `match_measured_to_simulated.py` | Measured detector image to simulated-image similarity scores, best match, and residual artifacts. |

## Loop C — Inversion

| Script | Purpose |
|---|---|
| `fit_orientation_beam.py` | Synthetic measured detector vector to fitted orientation/beam parameters and covariance. |
| `reconstruct_distribution.py` | Synthetic measured mixture to recovered incoherent distribution weights, uncertainty band, and dual-scale detector previews. |
| `invert_structure.py` | Synthetic measured detector vector to recovered structure-latent parameters. |
| `recipe_deviation.py` | Intended-vs-fitted recipe gap, z-scores, severity, and frozen deviation report. |

## Diagnostics & Ensemble

| Script | Purpose |
|---|---|
| `azimuthal_sweep.py` | Distributed azimuth sweep to per-angle detector metrics, mean image, and montage artifacts. |
| `parameter_grid.py` | Distributed energy-by-theta detector grid to numeric arrays and integrated-intensity heatmap. |
| `ensemble_average.py` | Generic `Distribution` ensemble average to averaged detector image and mode/effective-count metrics. |
| `reconstruct_orientation.py` | Synthetic orientation-mixture inverse fit to recovered weights, residual image, and gradient check. |
| `convergence_study.py` | Mode-count convergence diagnostic to monotone residual-vs-N curve and detector previews. |

## Operations

| Script | Purpose |
|---|---|
| `audit_invariants.py` | Run stateless physics-invariant checks and emit pass/fail residual artifacts. |
| `bump_pin.py` | Rewrite PEP 723 `rheedium==...` pins across automaton scripts. |
| `export_model.py` | Export a version-tagged StableHLO kinematic forward artifact with separate-process reload proof. |
