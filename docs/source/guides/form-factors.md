# Atomic Form Factors

Atomic form factors describe how individual atoms scatter electrons as a function of momentum transfer. Rheedium uses the Kirkland parameterization combined with Debye-Waller thermal corrections.

## Electron vs X-ray Scattering

Unlike X-ray scattering (which probes electron density), electron scattering probes the electrostatic potential of atoms. This leads to different form factor behavior:

| Property | X-ray Form Factor | Electron Form Factor |
|----------|-------------------|---------------------|
| Scattering from | Electron density | Electrostatic potential |
| Low-$q$ limit | $f(0) = Z$ | $f(0) \propto Z^{1/3}$ |
| $q$-dependence | Falls off faster | Falls off slower |
| Light elements | Weak scattering | Relatively stronger |

```{figure} figures/form_factor_curves.svg
:alt: Form factor comparison
:width: 90%

Comparison of atomic form factors $f(q)$ for different elements. The curves show the characteristic falloff with increasing momentum transfer, with heavier elements exhibiting stronger scattering amplitudes across all $q$ values.
```

## Kirkland Parameterization

Electron form factors are parameterized as a sum of six Gaussians:

$$
f_e(q) = \sum_{i=1}^{6} a_i \exp\left(-b_i s^2\right)
$$

where:

- $s = q/(4\pi) = \sin\theta/\lambda$ (scattering parameter)
- $q = |\mathbf{G}|$ (momentum transfer magnitude)
- $a_i$, $b_i$ = element-specific Kirkland coefficients

### Tabulated Coefficients

Rheedium includes Kirkland coefficients for elements Z = 1 to 103, stored in `luggage/Kirkland_Potentials.csv`. Each element has 12 parameters ($a_1$–$a_6$, $b_1$–$b_6$).

### Physical Interpretation

The Gaussian sum approximates the Fourier transform of the atomic potential:

- **Small $q$ (forward scattering)**: All Gaussians contribute, $f \approx \sum a_i$
- **Large $q$ (backscattering)**: Narrow Gaussians ($b_i$ small) dominate
- **Each term**: Represents different length scales of the atomic potential

```{figure} figures/element_comparison.svg
:alt: Atomic form factors for different elements
:width: 80%

Kirkland atomic form factors $f(q)$ for selected elements. Heavier elements scatter more strongly, but all form factors decrease with increasing momentum transfer as smaller length scales are probed.
```

### Implementation

```python
from rheedium.simul.form_factors import kirkland_form_factor

# Form factor at q = 2.5 Å⁻¹ for silicon (Z=14)
q = 2.5
f_Si = kirkland_form_factor(q, atomic_number=14)
```

Internally:

```python
def kirkland_form_factor(
    q: Float[Array, "..."],
    atomic_number: int,
) -> Float[Array, "..."]:
    """Kirkland electron form factor."""
    s = q / (4 * jnp.pi)  # Convert to scattering parameter
    s_sq = s ** 2

    # Load coefficients for this element
    a_coeffs, b_coeffs = _load_kirkland_params(atomic_number)

    # Sum of 6 Gaussians
    f = jnp.sum(a_coeffs * jnp.exp(-b_coeffs * s_sq))
    return f
```

## Debye-Waller Factor

Thermal vibrations reduce scattering intensity, especially at large momentum transfer. The Debye-Waller factor accounts for this:

$$
\exp(-W) = \exp\left(-\frac{1}{2} \langle u^2 \rangle q^2\right)
$$

where $\langle u^2 \rangle$ is the mean-square atomic displacement.

### B-Factor Convention

Crystallographers often use the temperature factor $B$:

$$
B = 8\pi^2 \langle u^2 \rangle
$$

The Debye-Waller factor then becomes:

$$
\exp(-W) = \exp\left(-\frac{B q^2}{16\pi^2}\right) = \exp\left(-\frac{B s^2}{4}\right)
$$

### Physical Meaning

| Temperature | Effect |
|-------------|--------|
| $T \to 0$ | $\langle u^2 \rangle \to u_0^2$ (zero-point motion) |
| $T$ increases | $\langle u^2 \rangle$ grows linearly |
| High $T$ | Large damping of high-$q$ reflections |

```{figure} figures/debye_waller_damping.svg
:alt: Debye-Waller damping at different temperatures
:width: 80%

Debye-Waller damping factor $\exp(-W)$ for silicon at different temperatures. Higher temperatures increase atomic vibrations, causing stronger damping especially at large momentum transfer.
```

### Temperature Dependence

Rheedium uses a simplified Debye model for $\langle u^2 \rangle$:

$$
\langle u^2 \rangle = \langle u^2 \rangle_0 \cdot \sqrt{\frac{12}{Z}} \cdot \frac{T}{T_0}
$$

where:

- $\langle u^2 \rangle_0$ = reference displacement at $T_0$
- $Z$ = atomic number (heavier atoms vibrate less)
- $T$ = temperature in Kelvin
- $T_0$ = reference temperature (typically 300 K)

### Implementation

```python
from rheedium.simul.form_factors import debye_waller_factor

# Debye-Waller factor at q = 2.5 Å⁻¹, 300 K, for Si
dw = debye_waller_factor(q=2.5, temperature=300.0, atomic_number=14)
# dw ≈ 0.95 (5% reduction)
```

## Surface Atom Enhancement

Surface atoms have fewer neighbors constraining their motion, leading to **enhanced thermal vibrations**:

$$
\langle u^2 \rangle_{\text{surface}} \approx 2 \times \langle u^2 \rangle_{\text{bulk}}
$$

This enhancement:

- Increases damping of surface-sensitive reflections
- Can be toggled in rheedium simulations
- Is particularly important for grazing-incidence RHEED

```{figure} figures/roughness_damping.svg
:alt: Surface roughness effects on scattering
:width: 85%

Surface roughness damping factor for different RMS roughness values. Surface atoms with enhanced vibrations and rough surfaces both reduce diffraction intensity at high momentum transfer, similar to the Debye-Waller effect.
```

## Combined Atomic Scattering Factor

The total atomic scattering amplitude combines form factor and thermal damping:

$$
f_{\text{total}}(q, T) = f_e(q) \times \exp(-W)
$$

In rheedium:

```python
def atomic_scattering_factor(
    q: Float[Array, "N"],
    atomic_numbers: Int[Array, "M"],
    temperature: float = 300.0,
    use_debye_waller: bool = True,
) -> Float[Array, "M N"]:
    """Combined form factor with optional thermal damping."""
```

## Structure Factor with Full Form Factors

The complete structure factor uses element-specific form factors:

$$
F(\mathbf{G}) = \sum_{j=1}^{N} f_j(|\mathbf{G}|) \, \exp\left(-\frac{B_j |\mathbf{G}|^2}{16\pi^2}\right) \, \exp(i\mathbf{G} \cdot \mathbf{r}_j)
$$

Each atom contributes:

1. **Form factor** $f_j(|\mathbf{G}|)$: Scattering amplitude at this $q$
2. **Debye-Waller** $\exp(-B_j q^2/16\pi^2)$: Thermal reduction
3. **Phase factor** $\exp(i\mathbf{G}\cdot\mathbf{r}_j)$: Interference from position

## q-Dependence Visualization

The form factor and Debye-Waller factor combine to produce q-dependent scattering:

```{figure} figures/combined_scattering.svg
:alt: Combined form factor and Debye-Waller scattering
:width: 80%

Combined atomic scattering factor showing the form factor alone (blue) and with Debye-Waller damping at 300 K (red). The shaded region shows how thermal effects reduce scattering at high $q$.
```

- **Low q**: Form factor dominates, $f \approx$ constant
- **Mid q**: Both effects contribute
- **High q**: Debye-Waller damping dominates, exponential falloff

## Practical Implications for RHEED

### Intensity Ratios

The q-dependence of form factors affects relative peak intensities:

- Low-index reflections (small $|\mathbf{G}|$): Strong, less affected by temperature
- High-index reflections (large $|\mathbf{G}|$): Weaker, more temperature-sensitive

### Element Sensitivity

| Element Type | Scattering Strength | Thermal Sensitivity |
|--------------|---------------------|---------------------|
| Heavy metals | Strong | Low (stiff) |
| Transition metals | Moderate | Moderate |
| Light elements | Weaker | High (floppy) |
| Oxygen | Weak | High |

### Temperature Effects

At elevated temperatures:

- High-order reflections become much weaker
- Pattern becomes dominated by low-order spots
- Surface atoms show enhanced damping

```{figure} figures/structure_factor_phases.svg
:alt: Structure factor phase diagram
:width: 80%

Argand diagram showing how atomic contributions combine to form the total structure factor. Each atom's phase factor $\exp(i\mathbf{G}\cdot\mathbf{r}_j)$ determines its contribution direction, and the vector sum gives $F(\mathbf{G})$.
```

## Example: Silicon Form Factor

```python
import jax.numpy as jnp
from rheedium.simul.form_factors import (
    kirkland_form_factor,
    debye_waller_factor,
)

# q values from 0 to 6 Å⁻¹
q = jnp.linspace(0.1, 6.0, 100)

# Pure form factor (no thermal effects)
f_Si = kirkland_form_factor(q, atomic_number=14)

# Debye-Waller factor at 300 K
dw_300 = debye_waller_factor(q, temperature=300.0, atomic_number=14)

# Debye-Waller factor at 600 K
dw_600 = debye_waller_factor(q, temperature=600.0, atomic_number=14)

# Combined scattering
f_total_300 = f_Si * dw_300
f_total_600 = f_Si * dw_600
```

## Key Source Files

- `simul/form_factors.py` - Form factor and Debye-Waller implementations
- `inout/luggage/Kirkland_Potentials.csv` - Kirkland coefficient table
