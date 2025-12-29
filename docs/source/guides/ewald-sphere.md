# Ewald Sphere Construction

The Ewald sphere is a geometric construction in reciprocal space that determines which crystal planes can diffract electrons at a given beam geometry. Rheedium implements both binary and finite-domain Ewald sphere models.

## Geometric Construction

### Reciprocal Space

For a crystal with real-space lattice vectors $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$, the reciprocal lattice vectors are:

$$
\mathbf{a}^* = \frac{2\pi (\mathbf{b} \times \mathbf{c})}{V}, \quad
\mathbf{b}^* = \frac{2\pi (\mathbf{c} \times \mathbf{a})}{V}, \quad
\mathbf{c}^* = \frac{2\pi (\mathbf{a} \times \mathbf{b})}{V}
$$

where $V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$ is the unit cell volume.

Reciprocal lattice vectors are indexed by Miller indices $(h, k, l)$:

$$
\mathbf{G}_{hkl} = h \mathbf{a}^* + k \mathbf{b}^* + l \mathbf{c}^*
$$

### The Ewald Sphere

The Ewald sphere is a sphere in reciprocal space with:

- **Center**: At the tip of $-\mathbf{k}_{\text{in}}$ (the negative incident wavevector)
- **Radius**: $|\mathbf{k}| = 2\pi/\lambda$ (the wavevector magnitude)

### Diffraction Condition

A reciprocal lattice point $\mathbf{G}$ lies on the Ewald sphere when:

$$
|\mathbf{k}_{\text{out}}| = |\mathbf{k}_{\text{in}}|
$$

where $\mathbf{k}_{\text{out}} = \mathbf{k}_{\text{in}} + \mathbf{G}$.

This is the **elastic scattering condition**: the scattered electron has the same energy as the incident electron.

## Incident Wavevector

In RHEED geometry, electrons arrive at grazing angle $\theta$ with azimuthal orientation $\phi$:

$$
\mathbf{k}_{\text{in}} = |\mathbf{k}| \begin{pmatrix} \cos\theta \cos\phi \\ \cos\theta \sin\phi \\ -\sin\theta \end{pmatrix}
$$

where:

- $\theta$ = grazing incidence angle (typically 1-5°)
- $\phi$ = azimuthal angle (rotation about surface normal)
- $|\mathbf{k}| = 2\pi/\lambda$

The small grazing angle creates a nearly tangent intersection with reciprocal lattice rods, producing elongated streaks in the RHEED pattern.

```{figure} figures/grazing_incidence_geometry.svg
:alt: RHEED grazing incidence geometry
:width: 85%

RHEED grazing incidence geometry showing the beam arriving at angle $\theta$ from the surface. The specularly diffracted beam exits at the same angle on the opposite side.
```

## Binary Mode

The simplest Ewald model uses a **tolerance-based criterion** for sphere intersection:

$$
\frac{||\mathbf{k}_{\text{out}}| - |\mathbf{k}_{\text{in}}||}{|\mathbf{k}_{\text{in}}|} < \epsilon
$$

where $\epsilon$ is typically 0.05 (5% tolerance).

### Algorithm

1. For each reciprocal lattice point $\mathbf{G}$:
   - Compute $\mathbf{k}_{\text{out}} = \mathbf{k}_{\text{in}} + \mathbf{G}$
   - Check if $|k_{\text{out}}|$ is within tolerance of $|k_{\text{in}}|$
   - Filter for upward scattering: $k_{\text{out},z} > 0$
2. Return allowed indices and their intensities

### Implementation

```python
def ewald_allowed_reflections(
    ewald: EwaldData,
    theta_deg: float,
    phi_deg: float = 0.0,
    tolerance: float = 0.05,
) -> Tuple[Int[Array, "M"], Float[Array, "M 3"], Float[Array, "M"]]:
    """Find reciprocal lattice points satisfying the Ewald condition."""
```

## Finite Domain Mode

Real surfaces have finite coherent domain sizes, and electron beams have energy spread and angular divergence. These effects broaden the Ewald sphere into a **shell** with finite thickness.

### Physical Effects

| Effect | Causes | Reciprocal Space Consequence |
|--------|--------|------------------------------|
| Finite domain size | Surface defects, terraces | Rod broadening in $(k_x, k_y)$ |
| Energy spread | Thermionic emission | Shell thickness in $|k|$ |
| Beam divergence | Electron optics | Shell thickness in direction |

### Rod Broadening

A finite coherent domain of extent $L$ causes reciprocal lattice rods to have Gaussian width:

$$
\sigma_{\text{rod}} = \frac{2\pi}{L \sqrt{2\pi}}
$$

This follows from the Fourier uncertainty relation: $\Delta x \cdot \Delta k \approx 2\pi$.

### Ewald Shell Thickness

Energy spread $\Delta E/E$ and beam divergence $\Delta\theta$ combine to give shell thickness:

$$
\sigma_{\text{shell}} = |\mathbf{k}| \sqrt{\left(\frac{\Delta E}{2E}\right)^2 + (\Delta\theta)^2}
$$

For typical RHEED conditions (15 kV, thermionic gun):

- $\Delta E/E \approx 10^{-4}$ (energy spread)
- $\Delta\theta \approx 10^{-3}$ rad (divergence)
- $\sigma_{\text{shell}} \approx 0.07$ Å$^{-1}$

### Overlap Integral

Instead of a binary on/off criterion, finite domain mode computes a continuous overlap:

$$
\text{overlap} = \exp\left(-\frac{d^2}{2\sigma_{\text{eff}}^2}\right)
$$

where:

- $d = ||\mathbf{k}_{\text{out}}| - |\mathbf{k}_{\text{in}}||$ (deviation from elastic condition)
- $\sigma_{\text{eff}}^2 = \sigma_{\text{rod}}^2 + \sigma_{\text{shell}}^2$ (combined broadening)

### Implementation

```python
def rod_ewald_overlap(
    k_in: Float[Array, "3"],
    g_vectors: Float[Array, "N 3"],
    rod_sigma: Float[Array, "2"],
    shell_sigma: float,
) -> Float[Array, "N"]:
    """Compute continuous overlap between rods and Ewald shell."""
```

## EwaldData Structure

Rheedium separates **angle-independent** computations (done once) from **angle-dependent** computations (done per beam orientation):

### Angle-Independent (Precomputed)

| Field | Description | Units |
|-------|-------------|-------|
| `wavelength_ang` | Electron wavelength | Å |
| `k_magnitude` | Wavevector magnitude $2\pi/\lambda$ | Å$^{-1}$ |
| `sphere_radius` | Ewald sphere radius (= $|k|$) | Å$^{-1}$ |
| `recip_vectors` | Reciprocal lattice basis $[\mathbf{a}^*, \mathbf{b}^*, \mathbf{c}^*]$ | Å$^{-1}$ |
| `hkl_grid` | Miller indices $[N \times 3]$ | dimensionless |
| `g_vectors` | Reciprocal lattice vectors $[N \times 3]$ | Å$^{-1}$ |
| `g_magnitudes` | $|\mathbf{G}|$ for each point | Å$^{-1}$ |
| `structure_factors` | $F(\mathbf{G})$ for each point | complex |
| `intensities` | $|F(\mathbf{G})|^2$ for each point | a.u. |

### Angle-Dependent (Per Orientation)

| Quantity | Description |
|----------|-------------|
| $\mathbf{k}_{\text{in}}$ | Incident wavevector from $\theta$, $\phi$ |
| Allowed indices | Which $\mathbf{G}$ satisfy Ewald condition |
| $\mathbf{k}_{\text{out}}$ | Outgoing wavevectors |
| Detector coordinates | Projection onto screen |

### Efficiency

By precomputing structure factors, an azimuthal scan (varying $\phi$ at fixed $\theta$) only requires:

1. Rotating $\mathbf{k}_{\text{in}}$
2. Finding new Ewald intersections
3. Looking up precomputed intensities

This avoids recomputing atomic form factors and structure factors at each angle.

## Building EwaldData

```python
from rheedium.simul import build_ewald_data

ewald = build_ewald_data(
    crystal=my_crystal,        # CrystalStructure
    voltage_kv=15.0,           # Accelerating voltage
    hmax=3, kmax=3, lmax=2,    # Miller index bounds
    temperature=300.0,         # For Debye-Waller factors
    use_debye_waller=True,     # Include thermal damping
)

# Access precomputed data
print(f"Wavelength: {ewald.wavelength_ang:.4f} Å")
print(f"Number of G vectors: {len(ewald.g_vectors)}")
```

## Finding Allowed Reflections

```python
from rheedium.simul import ewald_allowed_reflections

# Binary mode (tolerance-based)
indices, k_out, intensities = ewald_allowed_reflections(
    ewald=ewald,
    theta_deg=2.0,      # Grazing angle
    phi_deg=0.0,        # Azimuthal angle
    tolerance=0.05,     # 5% tolerance
)

print(f"Found {len(indices)} allowed reflections")
```

## Finite Domain Intensities

```python
from rheedium.simul import finite_domain_intensities

overlap, weighted_intensities = finite_domain_intensities(
    ewald=ewald,
    theta_deg=2.0,
    phi_deg=0.0,
    domain_extent_ang=[100.0, 100.0, 50.0],  # Domain size in Å
    energy_spread_frac=1e-4,                  # Relative energy spread
    beam_divergence_rad=1e-3,                 # Beam divergence in radians
)

# overlap values range from 0 (no intersection) to 1 (exact)
```

## Ewald Sphere Visualizations

The Ewald sphere construction can be viewed from multiple angles to understand the geometry:

```{figure} figures/ewald_sphere_2d.svg
:alt: Ewald sphere 2D cross-section
:width: 90%

2D cross-section (xz plane) of the Ewald sphere construction. The incident beam comes from the left at grazing angle, and the sphere intersects vertical reciprocal lattice rods. Each intersection corresponds to a diffraction spot.
```

```{figure} figures/ewald_sphere_3d_perspective.svg
:alt: Ewald sphere 3D perspective view
:width: 80%

3D perspective view of the Ewald sphere with reciprocal lattice rods. The partial sphere surface shows where the beam energy constrains possible scattering directions.
```

```{figure} figures/ewald_sphere_3d_top.svg
:alt: Ewald sphere 3D top view
:width: 80%

Top-down view looking along the surface normal. The circular intersection of the Ewald sphere with the plane of rod positions determines which reflections appear on the RHEED screen.
```

## Key Source Files

- `simul/ewald.py` - Ewald sphere geometry and `build_ewald_data()`
- `simul/finite_domain.py` - Finite domain overlap calculations
- `simul/simul_utils.py` - Wavevector utilities
