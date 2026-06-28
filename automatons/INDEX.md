# Automaton Index

| Script | Loop | Purpose |
|---|---|---|
| `forward_kinematic.py` | B | Structure file or smoke crystal to kinematic RHEED detector PNG and `.npz` artifacts. |
| `forward_multislice.py` | B | Structure file or smoke slab to transmission multislice detector PNG and `.npz` artifacts. |
| `forward_reflection.py` | B | Structure file or smoke slab to edge-on reflection multislice detector PNG and `.npz` artifacts. |
| `screen_xyz_ensemble.py` | B | Directory/glob of candidate structures to simulated RHEED ranking against a measured or simulated target. |
| `match_measured_to_simulated.py` | B | Measured detector image to simulated-image similarity scores, best match, and residual artifacts. |
| `fit_orientation_beam.py` | C | Synthetic measured detector vector to fitted orientation/beam parameters and covariance. |
| `reconstruct_distribution.py` | C | Synthetic measured mixture to recovered incoherent distribution weights and band. |
| `invert_structure.py` | C | Synthetic measured detector vector to recovered structure-latent parameters. |
| `recipe_deviation.py` | C | Intended-vs-fitted recipe gap, z-scores, severity, and frozen deviation report. |
| `bump_pin.py` | Ops | Rewrite PEP 723 `rheedium==...` pins across automaton scripts. |
