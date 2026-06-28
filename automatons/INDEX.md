# Automaton Index

This catalog lists every top-level Python script in `automatons/`.

## Loop A — Online RHEED

| Script | Purpose |
|---|---|
| `rheed_ingest.py` | One detector frame to per-frame growth observables and stateless growth-state JSON. |
| `growth_monitor.py` | Rolling detector series to oscillation period, roughness trend, and transition flags. |

## Loop B — Theory In The Loop

| Script | Purpose |
|---|---|
| `forward_kinematic.py` | Structure file or smoke crystal to kinematic RHEED detector PNG and `.npz` artifacts. |
| `forward_multislice.py` | Structure file or smoke slab to transmission multislice detector PNG and `.npz` artifacts. |
| `forward_reflection.py` | Structure file or smoke slab to edge-on reflection multislice detector PNG and `.npz` artifacts. |
| `screen_xyz_ensemble.py` | Directory/glob of candidate structures to simulated RHEED ranking against a measured or simulated target. |
| `match_measured_to_simulated.py` | Measured detector image to simulated-image similarity scores, best match, and residual artifacts. |

## Loop C — Inversion

| Script | Purpose |
|---|---|
| `fit_orientation_beam.py` | Synthetic measured detector vector to fitted orientation/beam parameters and covariance. |
| `reconstruct_distribution.py` | Synthetic measured mixture to recovered incoherent distribution weights and band. |
| `invert_structure.py` | Synthetic measured detector vector to recovered structure-latent parameters. |
| `recipe_deviation.py` | Intended-vs-fitted recipe gap, z-scores, severity, and frozen deviation report. |

## Operations

| Script | Purpose |
|---|---|
| `bump_pin.py` | Rewrite PEP 723 `rheedium==...` pins across automaton scripts. |
