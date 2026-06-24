# Plan: Rigorous Reflection-Geometry RHEED Multislice (+ Lobato as default)

Scope: `rheedium`. Two parts, one branch:
- **Part A** — add a physically correct **reflection-geometry ("edge-on")**
  multislice RHEED simulator as a new module. The existing
  `multislice_propagate` / `multislice_simulator`
  (`src/rheedium/simul/simulator.py`) are *transmission* multislice with a
  tilted incident wave (beam sent *through* the slab along the surface normal);
  they are an approximation. This part implements the beam propagating **along
  the surface** with the pattern formed by the wave that **reflects back into
  vacuum**.
- **Part B** — make **Lobato–van Dyck** the default atomic form-factor /
  projected-potential parameterization throughout the simulation pipeline
  (the kinematic path still hard-codes Kirkland), with Kirkland kept reachable.

Status: **implemented and verified in-tree** — Part A
(`crystal_to_edge_on_slices`, `reflection_multislice_propagate`,
`reflection_multislice_simulator`) plus the `EdgeOnSlices` type and exports, and
Part B (Lobato as the default form factor + kinematic-path routing) all landed
in `src/rheedium/simul/reflection_multislice.py`, `form_factors.py`, and
`ewald.py`. All five §9 acceptance tests pass
(`tests/test_rheedium/test_simul/test_reflection_multislice.py`, 17 cases:
specular geometry, Fresnel step reflectivity, CAP/vacuum convergence, Bi₂Se₃ rod
matching, and finite-output fixtures), as do `test_form_factors.py` /
`test_lobato.py`. Deviation from §8 test-guidance: the Lobato-vs-Kirkland low-q
cross-check is an order-of-magnitude check (plus an explicit "they differ" test)
rather than the suggested <1% agreement, since the two parameterizations
genuinely differ by more than 1%. Moved from `plans/partial/` to
`plans/implemented/` on completion.
This document is authoritative and self-contained; implement exactly to it.
Physical correctness (see §9 acceptance tests) is the bar, **not** "it runs".

Audience: an autonomous coding agent (codex) running in the repo with the
project venv available. Match repo conventions (§10).

---

## 0. TL;DR for the implementer

1. New module `src/rheedium/simul/reflection_multislice.py` with three public
   functions: `crystal_to_edge_on_slices`, `reflection_multislice_propagate`,
   `reflection_multislice_simulator`. Optionally a new PyTree `EdgeOnSlices`
   under `src/rheedium/types/`.
2. The method: slice the crystal **along the beam axis x**; propagate a 2D wave
   `φ(y,z)` along x by FFT multislice; absorb at the z-edges with a CAP; read
   off the **up-going** field in the vacuum region → `RHEEDPattern`.
3. Part B: add `parameterization="lobato"` to `atomic_scattering_factor` and
   route the kinematic ewald path through it.
4. Gate on physics: flat-slab specular at θ_out=θ_in, and specular reflectivity
   matching the **Fresnel step formula** (§9). Then Bi₂Se₃.
5. The atom count is irrelevant to cost — see §11. Do not worry about 4000
   atoms; worry about grid sizes and the read-off.

---

## 1. Physics background (read before coding)

### 1.1 Why reflection, not transmission
RHEED is grazing incidence (θ ≈ 1–4° from the surface). The beam travels almost
parallel to the surface and the detected pattern is the wave **reflected** back
into vacuum. Transmission multislice (beam ∥ surface normal) does not model
this; it is the wrong boundary-value problem for RHEED.

### 1.2 Edge-on real-space multislice
Rotate the multislice propagation axis to lie **along the beam** (the in-plane
direction). Slices are stacked along the beam axis `x`; the transverse plane is
`(y, z)` with `z` the surface normal. A 2D wavefield `φ(y, z)` is marched along
`x` through thin potential slabs. Where the wave turns upward (`k_z > 0`) in the
vacuum region above the surface, that is the reflected RHEED beam set.

This is the multislice analogue of the reflection treatments in Ichimiya &
Cohen, *RHEED* (2004) and Maksym & Beeby (1981). It reuses 2D FFT machinery, is
differentiable, and works directly on the explicit atomic slab.

### 1.3 Refraction & the correct reflectivity reference (important)
Electrons see an **attractive** mean inner potential `V0 > 0` (~10–20 V), so the
refractive index `n = sqrt(1 + V0/E) > 1`. **Consequence: there is NO
total-external-reflection-below-a-critical-angle** (that is an X-ray / repulsive
barrier phenomenon). Do **not** write that test.

The correct, quantitative, closed-form reference for a **flat featureless slab**
(step inner potential `V0` for `z < z_surf`, vacuum above) is the 1D
potential-step (Fresnel) reflection coefficient for the surface-normal motion:

- beam energy `E = voltage_kv * 1000` (volts/eV),
- vacuum perpendicular wavevector `k_perp_vac = k * sin(theta)`, `k = 2π/λ`,
- in-crystal perpendicular wavevector
  `k_perp_in = k * sqrt(sin(theta)**2 + V0/E)`  (parallel momentum conserved),
- amplitude `r = (k_perp_vac - k_perp_in) / (k_perp_vac + k_perp_in)`,
- **reflectivity `R = |r|**2`**.

`R → 1` smoothly as `θ → 0` (grazing), and decreases as θ grows. There is no
sharp cutoff. This `R(θ)` is the analytic target for acceptance test #2 (§9).
It also matches the package's own refraction convention
(`refraction_index = sqrt(1 + V0/voltage_v)` in `multislice_propagate`).

### 1.4 Specular & diffracted beams
In-plane momentum is conserved up to a surface reciprocal vector `G_∥`; energy
is conserved (`|k_out| = |k_in| = k`). The **specular** (00) beam has
`k_out,∥ = k_in,∥` and `k_out,z = +k sinθ` (mirror of incidence). Diffracted
beams sit at `k_out,∥ = k_in,∥ + G_∥`. For a periodic supercell, `G_∥` is the
in-plane reciprocal lattice (the same rods the kinematic `ewald_simulator`
finds) — that is acceptance test #4.

---

## 2. Existing code to reuse (exact references — verify signatures in-tree)

- `rheedium.tools.wavelength_ang(voltage_kv) -> Å`
  (`src/rheedium/tools/simul_utils.py:43`). `k = 2π/λ`.
- `rheedium.tools.incident_wavevector(lam_ang, theta_deg, phi_deg) -> [3]`
  (`simul_utils.py:116`): returns `k (cosθcosφ, cosθsinφ, −sinθ)`. **Use this
  convention** (incidence downward, `k_z<0`).
- `rheedium.tools.interaction_constant(voltage_kv, wavelength_ang) -> σ`
  (`simul_utils.py:172`), units 1/(V·Å). Transmission is `exp(i σ V_proj)`.
- `rheedium.simul.projected_potential(atomic_number, r, parameterization)
  -> V·Å` (`src/rheedium/simul/form_factors.py:616`). Lobato default. `r` is
  radial distance in Å. This is the per-atom projected potential to splat.
- `rheedium.ucell.reciprocal_lattice_vectors(*cell_lengths,*cell_angles,
  in_degrees=True) -> [3,3]` (`src/rheedium/ucell/unitcell.py:730`), includes
  2π. Use for the in-plane rods in test #4.
- `project_on_detector(k_out, detector_distance) -> [K,2]`
  (`src/rheedium/simul/simulator.py:125`).
- `create_rheed_pattern(g_indices, k_out, detector_points, intensities)
  -> RHEEDPattern` (`src/rheedium/types/rheed_types.py`). Fields:
  `G_indices, k_out, detector_points, intensities`.
- `CrystalStructure` (`src/rheedium/types/crystal_types.py`):
  `cart_positions` is `[N,4]` = `[x,y,z,Z]`; `cell_lengths=[a,b,c]`,
  `cell_angles=[α,β,γ]` (deg). MD slabs are orthogonal (90/90/90).
- `PotentialSlices` PyTree + `create_potential_slices`
  (`crystal_types.py:531/584`) — the style template for any new container.
- **Reference only, DO NOT modify**: `multislice_propagate` (`simulator.py:1666`)
  and `multislice_simulator` (`simulator.py:1832`) — the transmission versions.

---

## 3. Coordinate & unit conventions (fix these once)

- Surface normal `+z` points into vacuum. The slab's surface is the **top**:
  `z_surf = max(atom z)`. Vacuum is `z > z_surf`. Atoms occupy
  `z_atoms_min … z_surf`.
- φ=0 first (required milestone): beam azimuth along `+x`. Transverse plane
  `(y, z)`. Generalize to arbitrary φ later by rotating in-plane coords; φ=0 is
  the milestone gated by tests.
- All lengths Å; energies in V (`E = voltage_kv*1000`) / keV as in the package.
- `k = 2π/λ`. `k_x0 = k cosθ` (carrier along beam). `k_z0 = −k sinθ` (downward
  transverse tilt of the incident wave). For φ=0, `k_y0 = 0`.

---

## 4. New PyTree: `EdgeOnSlices` (optional but recommended)

Add under `src/rheedium/types/` (e.g. extend `crystal_types.py`), mirroring
`PotentialSlices` style (`eqx.Module`, a `create_edge_on_slices` factory with
validation, exported from `types/__init__.py` and listed in `__all__`):

```
EdgeOnSlices:
    slices:        Float[Array, "nx_slices ny nz"]   # projected potential V·Å, per beam-slice
    dx_slice:      Float[Array, ""]                   # Å, slab thickness along beam
    dy:            Float[Array, ""]                   # Å transverse (in-plane y)
    dz:            Float[Array, ""]                   # Å transverse (surface normal)
    y_extent:      Float[Array, ""]                   # Ly (periodic)
    z_lo:          Float[Array, ""]                   # bottom of z window
    z_surf:        Float[Array, ""]                   # surface height (top of atoms)
    cap_width:     Float[Array, ""]                   # Å absorbing-layer thickness
```

`ny, nz, nx_slices` are static ints (shape). If you prefer not to add a type,
return a tuple/dict — but a typed container matches the codebase and keeps the
public API clean.

---

## 5. Algorithm (detailed, with pseudocode)

### 5.1 `crystal_to_edge_on_slices`
```
def crystal_to_edge_on_slices(crystal, *, phi_deg=0.0, dx_slice=1.0,
        dy=0.25, dz=0.25, vacuum_above=30.0, cap_width=15.0,
        penetration_depth=None, r_cutoff=4.0, parameterization="lobato"):
    pos = crystal.cart_positions[:, :3]; Z = crystal.cart_positions[:, 3]
    Lx = crystal.cell_lengths[0]; Ly = crystal.cell_lengths[1]
    z_surf = max(pos[:,2]); z_atoms_min = min(pos[:,2])
    # z window: [z_lo, z_hi]; include penetration depth below surface + bottom CAP,
    # vacuum_above + top CAP above surface.
    z_bottom_phys = z_surf - (penetration_depth or (z_surf - z_atoms_min))
    z_lo = z_bottom_phys - cap_width
    z_hi = z_surf + vacuum_above + cap_width
    ny = round(Ly / dy); nz = round((z_hi - z_lo) / dz)
    nx_slices = ceil(Lx / dx_slice)
    # transverse grids
    y = arange(ny)*dy            # periodic over Ly
    z = z_lo + arange(nz)*dz
    # For each beam-slice i (atoms with x in [i*dx, (i+1)*dx)):
    #   V_i(y,z) = sum over those atoms of projected_potential(Z_a, r),
    #   r = sqrt(min_image(y - y_a, Ly)**2 + (z - z_a)**2), zeroed beyond r_cutoff.
    # Use jax.lax.scan / vmap over slices; assign atoms to slices by floor(x/dx).
    return EdgeOnSlices(slices=V (nx_slices,ny,nz), dx_slice, dy, dz,
                        y_extent=Ly, z_lo, z_surf, cap_width)
```
Notes:
- `r_cutoff` (≈4 Å) bounds the per-atom splat so cost is O(atoms × small patch),
  not O(atoms × ny × nz). Minimum-image only in `y` (periodic); `z` is open.
- Atoms are assigned to exactly one beam-slice by their `x` (project along beam).
- Potential is **real** here; the CAP (imaginary part) is added in the
  propagator step (§5.3), not baked into `slices`.

### 5.2 CAP (complex absorbing potential) mask
A real-space damping mask applied every step, function of z only:
```
W(z) = cap_strength * (depth_into_cap / cap_width)**2   # 0 in interior
cap_mask(z) = exp(-W(z) * dx_slice)                      # in [0,1]
```
nonzero only within `cap_width` of `z_lo` and `z_hi`. Absorbs the transmitted
(down) beam at the bottom and the reflected (up) beam at the very top after it
has left the read-off region — preventing FFT wrap-around in the non-periodic z.
`cap_strength` (V·Å scale) is a tunable; pick so test #3 (CAP convergence)
passes.

### 5.3 `reflection_multislice_propagate`
```
def reflection_multislice_propagate(slices, voltage_kv, theta_deg, phi_deg=0.0,
        cap_strength=..., bandwidth_limit=2/3):
    lam = wavelength_ang(voltage_kv); k = 2π/lam
    sigma = interaction_constant(voltage_kv, lam)
    k_x0 = k*cos(theta); k_z0 = -k*sin(theta)        # φ=0
    # transverse reciprocal grids
    ky = fftfreq(ny, dy)*2π ; kz = fftfreq(nz, dz)*2π    # be consistent: see note
    KY,KZ = meshgrid(ky,kz)
    P = exp(-1j * (dx_slice/(2*k)) * (KY**2 + KZ**2))    # paraxial propagator along x
    # initial transverse wave (vacuum plane wave with grazing downward tilt):
    phi = exp(1j * k_z0 * z[None,:]) * ones(ny)[:,None]
    band = soft_aperture(KY,KZ, bandwidth_limit)         # anti-alias
    capm = cap_mask(z)                                    # (nz,) broadcast over y
    for i in range(nx_slices):
        phi = phi * exp(1j * sigma * slices[i])           # transmit
        phi = phi * capm[None,:]                           # absorb at z-edges
        phi = ifft2( fft2(phi) * P * band )               # propagate dx along x
    return phi    # complex (ny, nz)
```
Convention note: choose ONE Fourier convention and use it consistently. With
`k=2π/λ` and reciprocal vectors carrying 2π (as the package does), define
transverse wavenumbers as `2π*fftfreq`. The paraxial propagator phase is
`-(dx/2k)(k_y²+k_z²)` (equivalent to `-π λ dx (ν_y²+ν_z²)` with ν the ordinary
fftfreq) — pick whichever, but keep it consistent with the read-off in §5.4.
Paraxiality holds: `|k_z0|/k = sinθ ≈ 0.04`.

### 5.4 Read-off (the crux) — up-going beams in vacuum → RHEEDPattern
```
def reflection_multislice_simulator(crystal, voltage_kv=30, theta_deg=2.5,
        phi_deg=0.0, detector_distance=80.0, dx_slice=1.0, dy=0.25, dz=0.25,
        vacuum_above=30.0, cap_width=15.0, parameterization="lobato"):
    slices = crystal_to_edge_on_slices(...)
    phi = reflection_multislice_propagate(slices, ...)
    # 1) restrict to the vacuum read-off band: z in [z_surf+margin, z_hi-cap_width]
    phi_vac = phi[:, z_vac_mask]                          # (ny, nz_vac)
    z_vac   = z[z_vac_mask]
    # 2) FFT over periodic y -> in-plane channel ky (these are the surface rods):
    A = fft_y(phi_vac)                                    # (ny, nz_vac), axis 0 = ky
    ky_axis = 2π*fftfreq(ny, dy)
    out = []
    for j, ky in enumerate(ky_axis):
        kz2 = k**2 - k_x0**2 - ky**2
        if kz2 <= 0: continue                             # evanescent, skip
        kz = sqrt(kz2)                                    # up-going (+)
        # project the vacuum z-profile onto the UP-going plane wave:
        up = sum( conj(exp(1j*kz*z_vac)) * A[j,:] ) / len(z_vac)
        I  = |up|**2
        k_out = (k_x0, ky, +kz)                           # |k_out| = k by construction
        out.append((k_out, I, ky_index_j))
    # 3) detector + pattern
    detector_points = project_on_detector(stack(k_out), detector_distance)
    intensities = normalize_to_max1(array(I))
    return create_rheed_pattern(g_indices=ky_channel_ids, k_out=..., 
                                detector_points=..., intensities=...)
```
Crux details (this is what test #1/#2 validate):
- The up-going projection `⟨exp(+i kz z), A(ky,·)⟩` over a finite vacuum slab is
  not perfectly orthogonal to the down-going incident wave; a longer
  `vacuum_above` improves separation (hence test #3). Normalize by the slab
  sample count (document the convention) so `R = I_specular / I_incident` is a
  pure number comparable to the Fresnel `R` of §1.3.
- Reference the **incident** amplitude for reflectivity: the incident wave is a
  unit-amplitude plane wave (`|phi_init|=1`), so `I_specular` is already the
  reflectivity if the projection is normalized consistently. Make the
  normalization explicit and test it against §1.3.
- The specular channel is `ky = 0` with `kz = +k sinθ`. Its detector point must
  be the mirror-angle position (test #1).
- Map only `kz2 > 0` channels (propagating beams); skip evanescent.

---

## 6. Public API (signatures, keyword-only options)

In `src/rheedium/simul/reflection_multislice.py`, all decorated with
`@jaxtyped(typechecker=beartype)`, numpy-style docstrings, exported from
`src/rheedium/simul/__init__.py` (add to the module docstring Routine Listings
and to `__all__`):

```python
def crystal_to_edge_on_slices(crystal: CrystalStructure, *, phi_deg: scalar_num = 0.0,
    dx_slice: scalar_float = 1.0, dy: scalar_float = 0.25, dz: scalar_float = 0.25,
    vacuum_above: scalar_float = 30.0, cap_width: scalar_float = 15.0,
    penetration_depth: scalar_float | None = None, r_cutoff: scalar_float = 4.0,
    parameterization: str = "lobato") -> EdgeOnSlices: ...

def reflection_multislice_propagate(slices: EdgeOnSlices, voltage_kv: scalar_num,
    theta_deg: scalar_num, phi_deg: scalar_num = 0.0,
    cap_strength: scalar_float = 50.0, bandwidth_limit: scalar_float = 2/3,
) -> Complex[Array, "ny nz"]: ...

def reflection_multislice_simulator(crystal: CrystalStructure, voltage_kv: scalar_num = 30.0,
    theta_deg: scalar_num = 2.5, phi_deg: scalar_num = 0.0,
    detector_distance: scalar_float = 80.0, dx_slice: scalar_float = 1.0,
    dy: scalar_float = 0.25, dz: scalar_float = 0.25, vacuum_above: scalar_float = 30.0,
    cap_width: scalar_float = 15.0, parameterization: str = "lobato",
) -> RHEEDPattern: ...
```
`parameterization` resolved at trace time (no branching on traced values).

---

## 7. Constraints / do NOT touch
- Do **not** modify `ewald_simulator`, `find_ctr_ewald_intersection`,
  `simulate_detector_image`, or the transmission `multislice_*` functions
  (except the single permitted Part-B keyword on the kinematic form-factor call,
  §8).
- Do **not** edit anything under `tutorials/` (the maintainer owns the
  notebooks) or `src/rheedium/simul/sweeps.py`.
- Pure-JAX, JIT-friendly, differentiable. Static shapes (`nx_slices/ny/nz` are
  Python ints). Use `jax.lax.scan` for the slice loop.

---

## 8. Part B — Lobato as the default form factor

The projected-potential path already defaults to `"lobato"`; the **kinematic**
path hard-codes Kirkland. Make Lobato the package-wide default, Kirkland still
reachable.

1. **`atomic_scattering_factor`** (`src/rheedium/simul/form_factors.py:841`,
   call at ~`:908`): add `parameterization: str = "lobato"`. Default →
   `lobato_form_factor`; `"kirkland"` → `kirkland_form_factor`. Keep the
   Debye–Waller multiplication unchanged.
2. **Kinematic call** (`src/rheedium/simul/ewald.py:106`, inside
   `build_ewald_data`): route through `atomic_scattering_factor` (or
   `lobato_form_factor`) so `ewald_simulator` is Lobato by default. If you add a
   `parameterization` kwarg to `build_ewald_data`/`ewald_simulator`, make it
   **keyword-only with a Lobato default and change no other behavior** (this is
   the only permitted edit to `ewald_simulator`).
3. **Form-factor curve plot** (`src/rheedium/plots/diagrams.py:590`): switch the
   displayed curve to Lobato (or add `parameterization="lobato"`). Low priority.

Do **NOT** touch `src/rheedium/audit/invariants.py` — it validates *both*
parameterizations on purpose. Keep the standalone `kirkland_*` functions.

Test guidance (avoid masking errors): where a test pins a Kirkland magic number,
either call it with `parameterization="kirkland"` to preserve the assertion, or
replace it with a physical-property assertion (positivity; monotonic decay in q;
heavier-Z ⇒ larger low-q scattering). Add: (a) a test that
`atomic_scattering_factor(...)` defaults to Lobato (== `lobato_form_factor × DW`,
not the Kirkland value); (b) a cross-check that Lobato vs Kirkland agree to <1%
at low q (|q| ≤ 1 Å⁻¹) for a mid-Z element.

---

## 9. Acceptance tests (the bar) — `tests/test_rheedium/test_simul/test_reflection_multislice.py`

Use the repo's chex / `parameterized` style. Keep grids small for speed
(`ny≈96–128`, `nz≈256`, synthetic slabs). Tests:

1. **Specular geometry (primary).** Build a synthetic flat slab: `EdgeOnSlices`
   with `slices[i] = V0*dx_slice` for `z < z_surf`, else 0 (uniform in x,y),
   `V0≈15` V. The reflected pattern must be dominated by the **specular** beam at
   `kz = +k sinθ` (mirror angle), within **~2% in angle / ≤1 detector pixel**,
   for θ ∈ {1°, 2°, 3°}. No off-specular beams (flat ⇒ only (00)).
2. **Fresnel step reflectivity (quantitative).** For the same flat slab, the
   specular reflectivity `R_sim = I_specular / I_incident` matches the closed
   form of §1.3, `R = |(k⊥_vac − k⊥_in)/(k⊥_vac + k⊥_in)|²` with
   `k⊥_in = k·sqrt(sin²θ + V0/E)`, within **~15%** across θ ∈ {0.5°,1°,2°,3°}.
   `R` must increase as θ→0. (No total-external-reflection cutoff — see §1.3.)
3. **CAP / vacuum convergence.** Specular position and `R_sim` change by **<5%**
   when `vacuum_above` and `cap_width` are each increased by 50% (no wrap-around).
4. **Weak-scattering geometry (Bi₂Se₃).** For a thin Bi₂Se₃ slab
   (`tests/test_data/bi2se3/intial.xyz`, optionally z-truncated near the
   surface), the in-plane positions (`ky` channels) of the strongest reflected
   beams coincide with the in-plane rod positions from `ewald_simulator` at the
   same θ, φ, within tolerance. Intensities need not match (kinematic vs
   dynamical).
5. **Numerics & shape.** `RHEEDPattern` intensities finite, non-negative, no
   NaN/Inf, normalized to max 1; runs for all five
   `tests/test_data/bi2se3/*.xyz` at production-ish grids.

**Acceptance commands (all must be green):**
```
.venv/bin/python -m pytest tests/test_rheedium/test_simul/test_reflection_multislice.py -x -q
.venv/bin/python -m pytest tests/test_rheedium/test_simul/test_form_factors.py tests/test_rheedium/test_simul/test_lobato.py -q
.venv/bin/python -m pytest tests/test_rheedium/test_simul -q
.venv/bin/ruff check src/rheedium/simul/reflection_multislice.py src/rheedium/simul/form_factors.py src/rheedium/simul/ewald.py tests/test_rheedium/test_simul/test_reflection_multislice.py
```
(If `-n auto` from `addopts` causes sandbox/process issues, append `-n0`.)

---

## 10. Repo conventions (must follow)
- jaxtyping + `@jaxtyped(typechecker=beartype)` on public functions; shapes in
  annotations (e.g. `Float[Array, "ny nz"]`). Scalar types from
  `rheedium.types` (`scalar_float`, `scalar_int`, `scalar_num`).
- numpy-style docstrings with a `:see:` test cross-reference line, Parameters,
  Returns, Notes (numbered steps), See Also — match neighbours in `simul/`.
- ruff line-length **79**; double-quote strings; `pyproject.toml` rules. New
  public funcs need docstrings (interrogate ≥ 90%).
- Tests typed and chex-based; reuse fixtures/factories where present.
- Export new public names from `simul/__init__.py` (`__all__` + Routine
  Listings) and any new type from `types/__init__.py`.

## 11. Performance (addressing "the atoms are tiny")
Correct — 4000 atoms is trivial. Cost is set by the **grids**, not atoms:
- transverse grid ≈ `Ly/dy × (z-window)/dz ≈ 165 × 600 ≈ 1e5` points,
- `nx_slices ≈ Lx/dx_slice ≈ 73` steps of `transmit → fft2 → propagate → ifft2`.
That is ~73 small 2D FFTs — sub-second to seconds on CPU. The per-atom splat is
O(atoms × small patch) thanks to `r_cutoff`. So slice creation will not choke.
The real knobs are `dy/dz` (accuracy vs size) and `vacuum_above`/`cap_width`
(read-off cleanliness). Keep test grids small; production can refine.

## 12. Pitfalls / known artifacts
- **Entry-edge artifact.** The incident plane wave fills the whole `(y,z)` plane
  at `x=0`, including inside the crystal — an artifact of edge-on geometry. It
  decays over propagation; read off the reflected beam in the **vacuum** band
  and after sufficient `Lx`. If needed, ignore the first few slices' transient.
- **Down/up separation.** The incident (down) wave is also present in vacuum;
  the up-going projection is only clean over a long enough vacuum slab — this is
  exactly what test #3 guards.
- **Fourier convention.** Keep `k=2π/λ`, 2π-carrying wavenumbers, and the
  propagator phase mutually consistent (§5.3). A factor-2π slip will move the
  specular off the mirror angle — test #1 catches it.
- **No total external reflection** for the attractive electron inner potential
  (§1.3). Don't assert it.
- **CAP too strong** eats the reflected beam (low R, fails #2); **too weak**
  wraps around (fails #3). Tune `cap_strength`.

## 13. Milestones (suggested order)
1. `EdgeOnSlices` + flat-slab constructor in a test helper.
2. `reflection_multislice_propagate` + read-off → **test #1 (specular angle)**.
3. Tune CAP/vacuum → **test #2 (Fresnel R)** and **#3 (convergence)**.
4. `crystal_to_edge_on_slices` (atomic projector) → **test #4 (Bi₂Se₃ rods)**.
5. `reflection_multislice_simulator` end-to-end → **test #5**.
6. Part B (Lobato default) + its tests.
7. Exports, docstrings, ruff clean; run all §9 commands green.

A prior draft of this plan lives at
`../rheedium-msrefl/SPEC_reflection_multislice.md` (worktree); **this file is the
corrected, authoritative version** — prefer it (notably the §1.3 reflectivity
physics correction).
