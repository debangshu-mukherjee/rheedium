# Crystal Truncation Rods and Surface Effects

Crystal truncation rods (CTRs) are the hallmark of surface-sensitive diffraction. This guide covers the physics of CTRs, roughness effects, and finite coherent domain modeling.

## Origin of Crystal Truncation Rods

### Bulk vs Surface Diffraction

A bulk 3D crystal produces discrete Bragg peaks at reciprocal lattice points $\mathbf{G}_{hkl}$. When the crystal is truncated at a surface, the periodicity is broken along the surface-normal direction, creating continuous rods of intensity.

### Mathematical Derivation

Consider a semi-infinite crystal extending from $z = -\infty$ to $z = 0$:

$$
\rho(\mathbf{r}) = \rho_{\text{bulk}}(\mathbf{r}) \times \Theta(-z)
$$

where $\Theta$ is the Heaviside step function.

In reciprocal space, this convolution becomes:

$$
\tilde{\rho}(\mathbf{q}) = \tilde{\rho}_{\text{bulk}}(\mathbf{q}) \otimes \mathcal{F}[\Theta(-z)]
$$

The Fourier transform of the step function gives:

$$
\mathcal{F}[\Theta(-z)] \propto \frac{1}{iq_z} + \pi\delta(q_z)
$$

This $1/q_z$ dependence creates continuous intensity along rods perpendicular to the surface.

### CTR Intensity Profile

For a surface-terminated crystal, the intensity along a rod at fixed $(h, k)$ varies as:

$$
I(h, k, l) \propto \frac{|F(h, k, l)|^2}{\sin^2(\pi l)}
$$

where:

- $F(h, k, l)$ = structure factor at this reciprocal space point
- $l$ = continuous (non-integer) Miller index along the rod

### Features of the CTR Profile

| Position | $\sin^2(\pi l)$ | Intensity | Physical Meaning |
|----------|-----------------|-----------|------------------|
| $l = 0, 1, 2, ...$ | 0 | $\to \infty$ | Bulk Bragg peaks |
| $l = 0.5, 1.5, ...$ | 1 | Minimum | Anti-Bragg condition |
| Intermediate | Varies | Moderate | Surface truncation signal |

The divergence at integer $l$ is regularized by the crystal's finite thickness and instrumental broadening.

## Surface Roughness

Real surfaces are not perfectly flat. Surface roughness modifies the CTR intensity.

### Gaussian Height Distribution

For a surface with RMS roughness $\sigma_h$, the height distribution is:

$$
P(z) = \frac{1}{\sqrt{2\pi}\sigma_h} \exp\left(-\frac{z^2}{2\sigma_h^2}\right)
$$

### Roughness Damping Factor

Roughness reduces CTR intensity, especially at large $q_z$:

$$
I_{\text{rough}}(q_z) = I_{\text{ideal}}(q_z) \times \exp\left(-q_z^2 \sigma_h^2\right)
$$

or equivalently:

$$
I_{\text{rough}}(q_z) = I_{\text{ideal}}(q_z) \times \exp\left(-\frac{1}{2} q_z^2 \sigma_h^2 \times 2\right)
$$

### Physical Interpretation

- **Smooth surface** ($\sigma_h \to 0$): No damping, sharp CTRs
- **Rough surface** (large $\sigma_h$): Strong damping at high $q_z$
- **Typical values**: $\sigma_h \approx 1$–$5$ Å for epitaxial surfaces

### Implementation

```python
from rheedium.simul.surface_rods import roughness_damping

# q_z values along a rod
q_z = jnp.linspace(0, 5, 100)  # Å⁻¹

# Roughness damping for σ_h = 2 Å
damping = roughness_damping(q_z, sigma_h=2.0)
# damping[0] ≈ 1.0, damping at high q_z << 1
```

## Lateral Rod Profiles

CTRs have finite width perpendicular to the rod direction due to finite coherent domain size.

### Gaussian Profile

For a coherent domain of extent $\xi$:

$$
I(q_\perp) \propto \exp\left(-\frac{q_\perp^2}{2\sigma_q^2}\right)
$$

where $\sigma_q = 1/\xi$ and $q_\perp$ is the perpendicular momentum transfer.

### Lorentzian Profile

For exponentially decaying correlations:

$$
I(q_\perp) \propto \frac{1}{1 + (q_\perp \xi)^2}
$$

This profile arises from surfaces with random defect distributions.

### Choosing a Profile

| Surface Type | Typical Profile | Physical Reason |
|--------------|-----------------|-----------------|
| High-quality epitaxial | Gaussian | Well-defined domain boundaries |
| Polycrystalline | Lorentzian | Random defects |
| Intermediate | Voigt (Gaussian ⊗ Lorentzian) | Mixed contributions |

## Finite Domain Effects

### Rod Broadening from Domain Size

A finite in-plane coherent domain of extent $L$ causes reciprocal lattice rods to have width:

$$
\sigma_{\text{rod}} = \frac{2\pi}{L \sqrt{2\pi}}
$$

This follows from the Fourier uncertainty relation.

### Numerical Example

| Domain Size $L$ | Rod Width $\sigma_{\text{rod}}$ |
|-----------------|--------------------------------|
| 1000 Å | 0.0025 Å$^{-1}$ |
| 100 Å | 0.025 Å$^{-1}$ |
| 10 Å | 0.25 Å$^{-1}$ |

Small domains produce broad, diffuse rods.

### Implementation

```python
from rheedium.simul.finite_domain import extent_to_rod_sigma

# Domain extent in each direction
domain_extent = [100.0, 100.0, 50.0]  # Å

# Convert to reciprocal-space rod widths
sigma_rod = extent_to_rod_sigma(domain_extent)
# Returns [σ_x, σ_y] in Å⁻¹ (z-direction gives infinite rod, no σ_z)
```

## Ewald Shell Thickness

The electron beam has finite energy spread and angular divergence, broadening the Ewald sphere into a shell.

### Energy Spread Contribution

For relative energy spread $\Delta E / E$:

$$
\Delta k_E = k \times \frac{\Delta E}{2E}
$$

(Factor of 2 because $k \propto \sqrt{E}$)

### Beam Divergence Contribution

For angular divergence $\Delta\theta$:

$$
\Delta k_\theta = k \times \Delta\theta
$$

### Combined Shell Thickness

Adding in quadrature:

$$
\sigma_{\text{shell}} = k \sqrt{\left(\frac{\Delta E}{2E}\right)^2 + (\Delta\theta)^2}
$$

### Typical RHEED Values

For a 15 keV thermionic gun RHEED system:

| Parameter | Typical Value |
|-----------|---------------|
| $\Delta E / E$ | $10^{-4}$ |
| $\Delta\theta$ | 1 mrad |
| $k$ | 63 Å$^{-1}$ |
| $\sigma_{\text{shell}}$ | 0.07 Å$^{-1}$ |

### Implementation

```python
from rheedium.simul.finite_domain import compute_shell_sigma

sigma_shell = compute_shell_sigma(
    k_magnitude=63.0,           # Å⁻¹
    energy_spread_frac=1e-4,    # Relative
    beam_divergence_rad=1e-3,   # Radians
)
# sigma_shell ≈ 0.07 Å⁻¹
```

## Rod-Ewald Overlap

The key to finite domain simulation is computing how much each reciprocal lattice rod overlaps with the broadened Ewald shell.

### Overlap Integral

For a rod at position $\mathbf{G}$ with width $\sigma_{\text{rod}}$ intersecting a shell of thickness $\sigma_{\text{shell}}$:

$$
\text{overlap} = \exp\left(-\frac{d^2}{2\sigma_{\text{eff}}^2}\right)
$$

where:

- $d = ||\mathbf{k}_{\text{out}}| - |\mathbf{k}_{\text{in}}||$ (deviation from elastic condition)
- $\sigma_{\text{eff}}^2 = \sigma_{\text{rod}}^2 + \sigma_{\text{shell}}^2$

### Weighted Intensity

The observable intensity is:

$$
I_{\text{obs}} = |F(\mathbf{G})|^2 \times \text{overlap}
$$

### Implementation

```python
from rheedium.simul.finite_domain import rod_ewald_overlap

overlap = rod_ewald_overlap(
    k_in=k_in_vector,       # Incident wavevector
    g_vectors=g_vectors,    # Reciprocal lattice points
    rod_sigma=sigma_rod,    # Rod widths [σ_x, σ_y]
    shell_sigma=sigma_shell,# Shell thickness
)
# overlap is Array of shape (N,) with values in [0, 1]
```

## Rod-Ewald Intersection Geometry

For CTR simulations, rheedium solves the quadratic equation for where a continuous rod intersects the Ewald sphere.

### The Quadratic Equation

For a rod extending along $\mathbf{c}^*$ at fixed $(h, k)$:

$$
\mathbf{k}_{\text{out}}(l) = \mathbf{k}_{\text{in}} + h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*
$$

The Ewald condition $|\mathbf{k}_{\text{out}}| = |\mathbf{k}_{\text{in}}|$ gives:

$$
|\mathbf{k}_{\text{in}} + \mathbf{G}_{hk} + l\mathbf{c}^*|^2 = |\mathbf{k}_{\text{in}}|^2
$$

Expanding:

$$
a l^2 + b l + c = 0
$$

where:

$$
\begin{align}
a &= |\mathbf{c}^*|^2 \\
b &= 2(\mathbf{k}_{\text{in}} + \mathbf{G}_{hk}) \cdot \mathbf{c}^* \\
c &= |\mathbf{k}_{\text{in}} + \mathbf{G}_{hk}|^2 - |\mathbf{k}_{\text{in}}|^2
\end{align}
$$

### Solutions

The quadratic gives two $l$ values (entry and exit of Ewald sphere through the rod):

$$
l = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

Only solutions with:

- $l$ real (discriminant $\geq 0$)
- $k_{\text{out},z} > 0$ (upward scattering)

contribute to the RHEED pattern.

## Complete Workflow

```python
from rheedium.simul import (
    build_ewald_data,
    finite_domain_intensities,
)

# 1. Build Ewald data (angle-independent)
ewald = build_ewald_data(
    crystal=my_crystal,
    voltage_kv=15.0,
    hmax=3, kmax=3, lmax=2,
    temperature=300.0,
)

# 2. Compute finite domain intensities
overlap, intensities = finite_domain_intensities(
    ewald=ewald,
    theta_deg=2.0,
    phi_deg=0.0,
    domain_extent_ang=[100.0, 100.0, 50.0],
    energy_spread_frac=1e-4,
    beam_divergence_rad=1e-3,
)

# 3. Apply roughness damping (optional)
from rheedium.simul.surface_rods import roughness_damping

# Get q_z for each reflection
q_z = ewald.g_vectors[:, 2]  # z-component of G vectors
damping = roughness_damping(q_z, sigma_h=2.0)
intensities_rough = intensities * damping
```

## Physical Summary

| Effect | Mathematical Description | Implementation |
|--------|-------------------------|----------------|
| Surface termination | CTRs: $I \propto 1/\sin^2(\pi l)$ | `kinematic_ctr_simulator()` |
| Surface roughness | $I \times \exp(-q_z^2 \sigma_h^2)$ | `roughness_damping()` |
| Finite domain | Rod width $\sigma = 2\pi/(L\sqrt{2\pi})$ | `extent_to_rod_sigma()` |
| Energy spread | Shell width $\sigma = k \Delta E/(2E)$ | `compute_shell_sigma()` |
| Beam divergence | Shell width $\sigma = k \Delta\theta$ | `compute_shell_sigma()` |
| Combined broadening | Gaussian overlap integral | `rod_ewald_overlap()` |

## Key Source Files

- [`simul/surface_rods.py`](../../src/rheedium/simul/surface_rods.py) - CTR intensity and roughness
- [`simul/finite_domain.py`](../../src/rheedium/simul/finite_domain.py) - Domain and beam broadening
- [`simul/kinematic.py`](../../src/rheedium/simul/kinematic.py) - CTR simulator
