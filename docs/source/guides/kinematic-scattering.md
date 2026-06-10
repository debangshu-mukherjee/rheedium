# Kinematic Scattering Theory

Kinematic scattering theory provides the foundation for RHEED pattern simulation in rheedium. This guide covers the mathematical framework, from electron wavelength to diffraction intensities.

## The Kinematic Approximation

Kinematic theory assumes that electrons scatter **at most once** as they traverse the crystal. This single-scattering approximation is valid when:

- The crystal is thin (surface-sensitive RHEED geometry)
- Scattering is weak (high-energy electrons, ~10-30 keV)
- Only relative intensities matter (not absolute cross-sections)

For quantitative analysis of strong reflections or thick samples, dynamical (multiple scattering) theory is required, but kinematic theory captures the essential physics of RHEED patterns.

```{figure} figures/grazing_incidence_geometry.svg
:alt: RHEED grazing incidence geometry
:width: 100%

RHEED geometry showing the electron beam arriving at grazing angle $\theta$ from the sample surface. The surface-sensitive nature of RHEED arises from this shallow incident angle, which limits electron penetration to just a few atomic layers.
```

## Relativistic Electron Wavelength

At RHEED energies (typically 10-30 keV), electrons are mildly relativistic. The de Broglie wavelength must include relativistic corrections:

$$
\lambda = \frac{h}{\sqrt{2 m_0 e V \left(1 + \frac{eV}{2 m_0 c^2}\right)}}
$$

where:

- $h = 6.626 \times 10^{-34}$ J·s (Planck's constant)
- $m_0 = 9.109 \times 10^{-31}$ kg (electron rest mass)
- $e = 1.602 \times 10^{-19}$ C (electron charge)
- $V$ = accelerating voltage (V)
- $c = 2.998 \times 10^8$ m/s (speed of light)

### Numerical Values

| Voltage (kV) | Wavelength (Å) | Relativistic Correction |
|--------------|----------------|------------------------|
| 10 | 0.1220 | 0.98% |
| 15 | 0.0994 | 1.47% |
| 20 | 0.0859 | 1.96% |
| 30 | 0.0698 | 2.93% |

The relativistic correction $eV/(2m_0c^2)$ is small but non-negligible for accurate simulations.

```{figure} figures/wavelength_vs_voltage.svg
:alt: Electron wavelength vs accelerating voltage
:width: 80%

Relativistic electron wavelength as a function of accelerating voltage. The dashed line shows the non-relativistic approximation, which overestimates the wavelength by several percent at typical RHEED energies.
```

### Implementation

In rheedium, wavelength calculation is in `simul/simul_utils.py`:

```python
def wavelength_ang(voltage_kv: float) -> float:
    """Relativistic electron wavelength in Angstroms."""
    voltage_v = voltage_kv * 1000.0
    # Relativistic correction factor
    gamma = 1.0 + e * voltage_v / (2.0 * m_e * c**2)
    return h / jnp.sqrt(2.0 * m_e * e * voltage_v * gamma) * 1e10
```

## Structure Factor

The structure factor $F(\mathbf{G})$ encodes how atoms in the unit cell scatter into a reciprocal lattice vector $\mathbf{G}$:

$$
F(\mathbf{G}) = \sum_{j=1}^{N} f_j(|\mathbf{G}|) \, \exp(-W_j) \, \exp(i \mathbf{G} \cdot \mathbf{r}_j)
$$

where:

- $f_j(|\mathbf{G}|)$ = atomic form factor for atom $j$ (see [Form Factors](form-factors.md))
- $\exp(-W_j)$ = Debye-Waller thermal damping factor
- $\mathbf{r}_j$ = position of atom $j$ in the unit cell
- $N$ = number of atoms in the unit cell

### Physical Interpretation

Each term in the sum represents:

1. **Scattering amplitude** $f_j$: How strongly atom $j$ scatters electrons (depends on atomic number and scattering angle)
2. **Thermal damping** $\exp(-W_j)$: Reduction due to thermal vibrations (stronger at large $|\mathbf{G}|$)
3. **Phase factor** $\exp(i\mathbf{G}\cdot\mathbf{r}_j)$: Interference from atom positions

```{figure} figures/structure_factor_phases.svg
:alt: Structure factor phase diagram
:width: 70%

Argand diagram showing how phase factors from different atoms combine to form the total structure factor. Each arrow represents one atom's contribution, and their vector sum gives $F(\mathbf{G})$.
```

### Extinction Rules

The structure factor determines which reflections are allowed. For example:

- **BCC lattice**: $F = 0$ when $h + k + l$ is odd
- **FCC lattice**: $F = 0$ unless $h, k, l$ are all even or all odd
- **Diamond structure**: Additional extinctions when $h + k + l = 4n + 2$

These arise from destructive interference in the phase factor sum.

## Diffraction Intensity

The measured intensity is proportional to the squared modulus of the structure factor:

$$
I(\mathbf{G}) = |F(\mathbf{G})|^2
$$

This assumes:

- Kinematic (single-scattering) conditions
- Incoherent summation over thermal vibrations (absorbed in Debye-Waller factor)
- No absorption or anomalous dispersion

```{figure} figures/form_factor_curves.svg
:alt: Atomic form factors vs momentum transfer
:width: 85%

Atomic form factors $f(q)$ for several elements showing how scattering amplitude decreases with momentum transfer. Heavier elements scatter more strongly, but all form factors approach zero at high $q$.
```

## Simplified vs Full Structure Factors

Rheedium provides two approaches for structure factor calculation:

### Simplified Model

Uses atomic number as a proxy for scattering amplitude:

$$
F_{\text{simple}}(\mathbf{G}) = \sum_{j=1}^{N} Z_j \, \exp(i \mathbf{G} \cdot \mathbf{r}_j)
$$

- **Advantages**: Fast, no external tables needed
- **Limitations**: Ignores $q$-dependence of form factors, no thermal effects
- **Use case**: Quick pattern visualization, qualitative comparisons

### Full Model (Kirkland Form Factors)

Uses tabulated electron scattering form factors with thermal corrections:

$$
F_{\text{full}}(\mathbf{G}) = \sum_{j=1}^{N} f_j^{\text{Kirkland}}(|\mathbf{G}|) \, \exp\left(-\frac{B_j |\mathbf{G}|^2}{16\pi^2}\right) \, \exp(i \mathbf{G} \cdot \mathbf{r}_j)
$$

- **Advantages**: Quantitatively accurate intensities
- **Limitations**: Requires form factor tables, more computation
- **Use case**: Comparison with experimental data, intensity analysis

See [Form Factors](form-factors.md) for details on the Kirkland parameterization.

```{figure} figures/debye_waller_damping.svg
:alt: Debye-Waller thermal damping
:width: 80%

Debye-Waller damping factor $\exp(-W)$ at different temperatures. Higher temperatures increase atomic vibrations, causing stronger damping of high-$q$ reflections and reducing the intensity of high-order diffraction spots.
```

## Crystal Truncation Rods (CTRs)

For surface-sensitive RHEED, the reciprocal lattice consists of continuous **rods** perpendicular to the surface, not discrete points. This arises from the abrupt surface termination.

### Physical Origin

A semi-infinite crystal can be modeled as an infinite crystal multiplied by a step function $\Theta(z)$:

$$
\rho_{\text{surface}}(\mathbf{r}) = \rho_{\text{bulk}}(\mathbf{r}) \times \Theta(z)
$$

In reciprocal space, this convolution becomes:

$$
\tilde{\rho}(\mathbf{G}) = \tilde{\rho}_{\text{bulk}}(\mathbf{G}) \otimes \text{FT}[\Theta(z)]
$$

The Fourier transform of the step function is proportional to $1/q_z$, creating continuous intensity along the surface-normal direction.

```{figure} figures/ctr_origin_diagram.svg
:alt: Origin of crystal truncation rods
:width: 100%

Origin of crystal truncation rods: (left) bulk crystal with 3D periodicity produces discrete Bragg peaks, (center) surface truncation breaks z-periodicity, (right) reciprocal space shows continuous rods instead of discrete points.
```

### CTR Intensity Modulation

Along a rod at fixed $(h, k)$, the intensity varies as:

$$
I(h, k, l) \propto \frac{|F(h, k, l)|^2}{\sin^2(\pi l)}
$$

The $1/\sin^2(\pi l)$ factor produces:

- **Divergence at integer $l$**: Bragg peaks from bulk periodicity
- **Finite intensity between Bragg peaks**: Surface truncation rods
- **Minimum at half-integer $l$**: Anti-Bragg conditions

```{figure} figures/ctr_intensity_profile.svg
:alt: Crystal truncation rod intensity profile
:width: 85%

CTR intensity profile showing the characteristic $1/\sin^2(\pi l)$ modulation. Intensity diverges at Bragg peak positions (integer $l$) and reaches minima at anti-Bragg conditions.
```

### Implementation

The CTR simulator in `simul/kinematic.py` uses `kinematic_ctr_simulator()`:

```python
def kinematic_ctr_simulator(
    crystal: CrystalStructure,
    voltage_kv: float,
    theta_deg: float,
    phi_deg: float = 0.0,
    hmax: int = 3,
    kmax: int = 3,
    l_points: int = 100,
    detector_distance: float = 100.0,
) -> RHEEDPattern:
    """Simulate RHEED with continuous CTR sampling."""
```

## Ewald Sphere Intersection

The Ewald sphere determines which reciprocal lattice points (or rod positions) contribute to the diffraction pattern at a given beam geometry. See [Ewald Sphere](ewald-sphere.md) for the full treatment.

### Diffraction Condition

A reflection $\mathbf{G}$ is allowed when:

$$
|\mathbf{k}_{\text{out}}| = |\mathbf{k}_{\text{in}}|
$$

where:

$$
\mathbf{k}_{\text{out}} = \mathbf{k}_{\text{in}} + \mathbf{G}
$$

This ensures elastic scattering (energy conservation).

### Grazing Incidence Geometry

In RHEED, the incident beam arrives at grazing angle $\theta$ (typically 1-5°):

$$
\mathbf{k}_{\text{in}} = |\mathbf{k}| \begin{pmatrix} \cos\theta \cos\phi \\ \cos\theta \sin\phi \\ -\sin\theta \end{pmatrix}
$$

The small $\theta$ creates a nearly tangent intersection of the Ewald sphere with reciprocal lattice rods, producing the characteristic streaked RHEED pattern.

```{figure} figures/ewald_sphere_2d.svg
:alt: Ewald sphere 2D cross-section
:width: 90%

2D cross-section of the Ewald sphere construction in RHEED geometry. The incident beam (red arrow) arrives at grazing angle, and the sphere intersects vertical reciprocal lattice rods. Each intersection corresponds to a diffraction spot on the detector.
```

## Detector Projection

Outgoing wavevectors $\mathbf{k}_{\text{out}}$ are projected onto a flat detector screen:

$$
\begin{pmatrix} Y \\ Z \end{pmatrix}_{\text{detector}} = d \cdot \frac{1}{k_{\text{out},x}} \begin{pmatrix} k_{\text{out},y} \\ k_{\text{out},z} \end{pmatrix}
$$

where $d$ is the sample-to-detector distance.

Only forward-scattered electrons ($k_{\text{out},x} > 0$) and upward-going electrons ($k_{\text{out},z} > 0$) reach the detector.

## Workflow Summary

```
CrystalStructure
      ↓
┌─────────────────────────────────────┐
│ 1. Compute wavelength from voltage  │
│    λ = h / √(2m₀eV(1 + eV/2m₀c²))  │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ 2. Generate reciprocal lattice      │
│    G(h,k,l) = h·a* + k·b* + l·c*   │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ 3. Compute structure factors        │
│    F(G) = Σⱼ fⱼ exp(-Wⱼ) exp(iG·rⱼ)│
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ 4. Find Ewald sphere intersections  │
│    |k_out| = |k_in|                │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ 5. Project onto detector screen     │
│    (Y, Z) = d · (k_y, k_z) / k_x   │
└─────────────────────────────────────┘
      ↓
RHEEDPattern
```

## Visualizing the RHEED Pattern

The kinematic theory produces actual diffraction patterns. Here's a complete example from CIF file to rendered pattern:

```python
import rheedium as rh

# Load crystal structure
crystal = rh.io.parse_cif("MgO.cif")

# Simulate at 2° grazing incidence
pattern = rh.simul.kinematic_simulator(
    crystal,
    voltage_kv=15.0,
    theta_deg=2.0,
    phi_deg=0.0,
    hmax=5,
    kmax=5,
    surface_roughness=0.3,
)

# Render the pattern
rh.plots.plot_rheed(
    pattern,
    grid_size=400,
    spot_width=0.03,
    cmap_name="phosphor",
)
```

```{figure} figures/mgo_kinematic_rheed.svg
:alt: MgO kinematic RHEED pattern
:width: 80%

Simulated MgO RHEED pattern at 2° grazing incidence along the [100] azimuth. The vertical Laue zones and horizontal Kikuchi lines arise from kinematic scattering theory. Spot intensities are modulated by the structure factor $|F(\mathbf{G})|^2$.
```

### Effect of Structure Factor on Pattern

The structure factor determines which reflections are allowed and their intensities:

```python
import rheedium as rh

# Compare rock salt (MgO) vs. perovskite (SrTiO3) patterns
for cif_file, title in [
    ("MgO.cif", "Rock salt (MgO)"),
    ("SrTiO3.cif", "Perovskite (SrTiO₃)")
]:
    crystal = rh.io.parse_cif(cif_file)
    pattern = rh.simul.kinematic_simulator(
        crystal,
        voltage_kv=15.0,
        theta_deg=2.0,
        hmax=5,
        kmax=5,
    )
    rh.plots.plot_rheed(pattern, cmap_name="phosphor")
```

```{figure} figures/structure_factor_comparison.svg
:alt: Structure factor comparison between MgO and SrTiO3
:width: 100%

RHEED patterns from two crystal structures demonstrating how the structure factor $F(\mathbf{G})$ controls spot positions and intensities. MgO (rock salt) shows systematic absences different from SrTiO₃ (perovskite).
```

## Key Source Files

- `simul/kinematic.py` - Main simulation functions
- `simul/simulator.py` - Intensity calculations with CTRs
- `simul/simul_utils.py` - Wavelength and wavevector utilities
