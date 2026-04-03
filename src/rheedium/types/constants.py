"""Physical constants for RHEED simulation.

Extended Summary
----------------
Defines fundamental physical constants and derived constants used
throughout the rheedium simulation pipeline. All values are from
CODATA 2018 (NIST) and are exact where possible (2019 SI
redefinition). Using ``Final`` annotations ensures that static
analysis tools and beartype flag any accidental reassignment.

Routine Listings
----------------
:obj:`PLANCK_CONSTANT_JS`
    Planck constant *h* in J·s.
:obj:`HBAR_JS`
    Reduced Planck constant *ℏ* in J·s.
:obj:`ELECTRON_MASS_KG`
    Electron rest mass in kg.
:obj:`ELEMENTARY_CHARGE_C`
    Elementary charge in C.
:obj:`SPEED_OF_LIGHT_MS`
    Speed of light in vacuum in m/s.
:obj:`BOLTZMANN_CONSTANT_JK`
    Boltzmann constant in J/K.
:obj:`AMU_TO_KG`
    Atomic mass unit to kg conversion factor.
:obj:`M2_TO_ANG2`
    Square metres to square Ångströms conversion factor.
:obj:`RELATIVISTIC_COEFF_PER_V`
    Relativistic correction coefficient *e / (2 m_e c²)* in V⁻¹.
:obj:`H_OVER_SQRT_2ME_ANG_VSQRT`
    Electron wavelength prefactor *h / √(2 m_e e)* in Å·V^0.5.

Notes
-----
Constants are plain Python floats, not JAX arrays, so they can be
used in both JAX (``jnp``) and NumPy (``np``) contexts without
triggering unwanted device transfers or tracing side effects.
"""

from beartype.typing import Final

PLANCK_CONSTANT_JS: Final[float] = 6.62607015e-34
"""Planck constant *h* in J·s (exact, 2019 SI)."""

HBAR_JS: Final[float] = 1.054571817e-34
"""Reduced Planck constant *ℏ = h / (2π)* in J·s."""

ELECTRON_MASS_KG: Final[float] = 9.1093837015e-31
"""Electron rest mass in kg (CODATA 2018)."""

ELEMENTARY_CHARGE_C: Final[float] = 1.602176634e-19
"""Elementary charge in C (exact, 2019 SI)."""

SPEED_OF_LIGHT_MS: Final[float] = 299792458.0
"""Speed of light in vacuum in m/s (exact)."""

BOLTZMANN_CONSTANT_JK: Final[float] = 1.380649e-23
"""Boltzmann constant in J/K (exact, 2019 SI)."""

AMU_TO_KG: Final[float] = 1.66053906660e-27
"""Atomic mass unit (Dalton) to kg conversion factor."""

M2_TO_ANG2: Final[float] = 1e20
"""Square metres to square Ångströms (1 m² = 10²⁰ Å²)."""

RELATIVISTIC_COEFF_PER_V: Final[float] = 0.978476e-6
r"""Relativistic correction coefficient in V⁻¹.

Derived as *e / (2 m_e c²)*:

.. math::

    \frac{e}{2\,m_e\,c^2}
    = \frac{1.602176634 \times 10^{-19}}
           {2 \times 9.1093837015 \times 10^{-31}
            \times (299792458)^2}
    \approx 0.978476 \times 10^{-6}\;\mathrm{V^{-1}}
"""

H_OVER_SQRT_2ME_ANG_VSQRT: Final[float] = 12.2643
r"""Electron wavelength prefactor in Å·V^0.5.

Derived as *h / √(2 m_e e)*:

.. math::

    \frac{h}{\sqrt{2\,m_e\,e}}
    = \frac{6.62607015 \times 10^{-34}}
           {\sqrt{2 \times 9.1093837015 \times 10^{-31}
                    \times 1.602176634 \times 10^{-19}}}
    \approx 12.2643\;\text{Å}\cdot\text{V}^{0.5}
"""

__all__: list[str] = [
    "AMU_TO_KG",
    "BOLTZMANN_CONSTANT_JK",
    "ELEMENTARY_CHARGE_C",
    "ELECTRON_MASS_KG",
    "H_OVER_SQRT_2ME_ANG_VSQRT",
    "HBAR_JS",
    "M2_TO_ANG2",
    "PLANCK_CONSTANT_JS",
    "RELATIVISTIC_COEFF_PER_V",
    "SPEED_OF_LIGHT_MS",
]
