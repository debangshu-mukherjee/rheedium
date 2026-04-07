import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Finite Domain Ewald Sphere Broadening

    This tutorial demonstrates how finite coherent domain size affects RHEED diffraction patterns. Real crystal surfaces are never perfectly ordered over infinite distances - defects, domain boundaries, and limited terrace sizes restrict the coherent scattering length.

    ## Physical Background

    In kinematic diffraction, we typically assume the Ewald sphere intersects infinitely sharp reciprocal lattice rods. In reality:

    1. **Finite domain size** causes reciprocal space broadening: rod width σ ∝ 1/L
    2. **Beam energy spread** and **divergence** create a finite Ewald "shell" thickness
    3. The measured intensity is the **overlap integral** between broadened rods and shell

    This tutorial covers:
    - Computing rod widths from domain extent
    - Calculating Ewald shell thickness from beam parameters
    - Visualizing the rod-Ewald overlap
    - Comparing RHEED patterns for different domain sizes
    """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from pathlib import Path
    import rheedium as rh

    # Suppress JAX GPU warning if no CUDA
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    return Path, jnp, plt, rh


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Load Crystal Structure

    We'll use MgO as our example - a simple rock-salt structure.
    """
    )
    return


@app.cell
def _(repo_root, rh):
    crystal = rh.inout.parse_cif(repo_root / "tests" / "test_data" / "MgO.cif")

    print(f"Cell parameters: a = {float(crystal.cell_lengths[0]):.3f} Å")
    print(f"Number of atoms: {crystal.cart_positions.shape[0]}")
    print(f"Atomic positions (fractional):")
    print(crystal.frac_positions)
    return (crystal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Rod Width from Domain Size

    The reciprocal lattice rod width is inversely proportional to domain size:

    $$\sigma_q = \frac{2\pi}{L \times \sqrt{2\pi}}$$

    This ensures the Gaussian approximation has the same FWHM as the true sinc² profile.
    """
    )
    return


@app.cell
def _(jnp, plt, rh):
    # Compute rod widths for different domain sizes
    domain_sizes = jnp.logspace(1, 3, 50)  # 10 to 1000 Å
    rod_sigmas = []
    for _L in domain_sizes:
        _extent = jnp.array([_L, _L, _L / 2])
        _sigma = rh.simul.extent_to_rod_sigma(
            _extent
        )  # Typical thin film: Lx = Ly, Lz smaller
        rod_sigmas.append(float(_sigma[0]))
    rod_sigmas = jnp.array(rod_sigmas)  # x-component
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.loglog(domain_sizes, rod_sigmas, "b-", linewidth=2)
    _ax.set_xlabel("Domain size L (Å)", fontsize=12)
    # Plot
    _ax.set_ylabel("Rod width σ (Å⁻¹)", fontsize=12)
    _ax.set_title("Reciprocal Lattice Rod Broadening", fontsize=14)
    _ax.grid(True, alpha=0.3)
    _ax.axhline(
        y=0.1, color="r", linestyle="--", alpha=0.5, label="σ = 0.1 Å⁻¹"
    )
    _ax.axhline(
        y=0.01, color="g", linestyle="--", alpha=0.5, label="σ = 0.01 Å⁻¹"
    )
    _ax.legend()
    plt.tight_layout()
    # Add reference lines
    plt.show()
    for _L in [20, 50, 100, 500]:
        _extent = jnp.array([float(_L), float(_L), float(_L) / 2])
        _sigma = rh.simul.extent_to_rod_sigma(_extent)
        # Print some values
        print(f"L = {_L:4d} Å → σ = {float(_sigma[0]):.4f} Å⁻¹")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Ewald Shell Thickness

    The Ewald sphere becomes a "shell" due to:
    - **Energy spread** ΔE/E: causes Δk/k = ΔE/(2E)
    - **Beam divergence** Δθ: causes Δk⊥ = k×Δθ

    Combined in quadrature:
    $$\sigma_{shell} = k \times \sqrt{\left(\frac{\Delta E}{2E}\right)^2 + \Delta\theta^2}$$
    """
    )
    return


@app.cell
def _(jnp, rh):
    # Calculate shell thickness for different voltages
    voltages_kv = jnp.array([10, 15, 20, 25, 30])
    print("Ewald shell thickness for different beam voltages:")
    print("(ΔE/E = 10⁻⁴, Δθ = 1 mrad)")
    print("=" * 45)
    for V in voltages_kv:
        lam = rh.simul.wavelength_ang(V)
        _k = 2 * jnp.pi / lam
        _sigma_shell = rh.simul.compute_shell_sigma(
            _k, energy_spread_frac=0.0001, beam_divergence_rad=0.001
        )
        print(
            f"V = {int(V):2d} kV: λ = {float(lam):.4f} Å, k = {float(_k):.1f} Å⁻¹, σ_shell = {float(_sigma_shell):.4f} Å⁻¹"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Rod-Ewald Overlap Visualization

    The overlap factor determines how much intensity each reflection contributes. Let's visualize how it depends on domain size.
    """
    )
    return


@app.cell
def _(crystal, rh):
    # Build Ewald data for MgO
    voltage_kv = 15.0
    theta_deg = 2.0  # Grazing angle

    ewald = rh.simul.build_ewald_data(
        crystal=crystal,
        voltage_kv=voltage_kv,
        hmax=3,
        kmax=3,
        lmax=2,
        temperature=300.0,
    )

    print(f"Electron wavelength: λ = {float(ewald.wavelength_ang):.4f} Å")
    print(f"Wavevector magnitude: k = {float(ewald.k_magnitude):.2f} Å⁻¹")
    print(f"Number of G vectors: {ewald.g_vectors.shape[0]}")
    return ewald, theta_deg


@app.cell
def _(ewald, jnp, plt, rh, theta_deg):
    # Compare overlap distributions for different domain sizes
    domain_sizes_test = [20, 50, 100, 500]  # Å
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10))
    _axes = _axes.flatten()
    for _ax, _L in zip(_axes, domain_sizes_test):
        domain = jnp.array([float(_L), float(_L), float(_L) / 2])
        _overlap, _intensities = rh.simul.finite_domain_intensities(
            ewald=ewald,
            theta_deg=theta_deg,
            phi_deg=0.0,
            domain_extent_ang=domain,
        )
        _ax.hist(_overlap, bins=50, range=(0, 1), alpha=0.7, edgecolor="black")
        _ax.set_xlabel("Overlap factor", fontsize=11)
        _ax.set_ylabel("Count", fontsize=11)
        _ax.set_title(f"Domain size L = {_L} Å", fontsize=12)
        _n_active = int(jnp.sum(_overlap > 0.1))
        _ax.axvline(x=0.1, color="r", linestyle="--", alpha=0.5)
        _ax.text(
            0.95,
            0.95,
            f"Active (>0.1): {_n_active}",
            transform=_ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
    plt.suptitle(
        "Overlap Factor Distribution vs Domain Size", fontsize=14, y=1.02
    )
    plt.tight_layout()  # Histogram of overlap values
    plt.show()  # Statistics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Observation**: Smaller domains have more "active" reflections (overlap > 0.1) because the broader rods intercept the Ewald shell over a wider range.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Intensity-Weighted Overlap

    The measured diffraction pattern combines the base structure factor intensity with the overlap:
    """
    )
    return


@app.cell
def _(ewald, jnp, plt, rh, theta_deg):
    # Compare intensity distributions
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    domain_large = jnp.array([1000.0, 1000.0, 500.0])
    # Large domain (nearly infinite crystal)
    overlap_large, intensities_large = rh.simul.finite_domain_intensities(
        ewald, theta_deg, 0.0, domain_large
    )
    domain_small = jnp.array([30.0, 30.0, 15.0])
    overlap_small, intensities_small = rh.simul.finite_domain_intensities(
        ewald, theta_deg, 0.0, domain_small
    )
    _axes[0].semilogy(sorted(ewald.intensities, reverse=True), "b-", alpha=0.7)
    _axes[0].set_xlabel("Reflection rank")
    # Small domain (polycrystalline)
    _axes[0].set_ylabel("Intensity (a.u.)")
    _axes[0].set_title("Base Structure Factor Intensities")
    _axes[0].grid(True, alpha=0.3)
    _axes[1].semilogy(sorted(intensities_large, reverse=True), "g-", alpha=0.7)
    _axes[1].set_xlabel("Reflection rank")
    # Plot base intensities
    _axes[1].set_ylabel("Intensity (a.u.)")
    _axes[1].set_title(f"Large Domain (L = 1000 Å)")
    _axes[1].grid(True, alpha=0.3)
    _axes[2].semilogy(sorted(intensities_small, reverse=True), "r-", alpha=0.7)
    _axes[2].set_xlabel("Reflection rank")
    _axes[2].set_ylabel("Intensity (a.u.)")
    # Plot large domain intensities
    _axes[2].set_title(f"Small Domain (L = 30 Å)")
    _axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"\nTotal intensity comparison:")
    print(f"  Base intensities:     {float(jnp.sum(ewald.intensities)):.2e}")
    # Plot small domain intensities
    print(f"  Large domain (1000Å): {float(jnp.sum(intensities_large)):.2e}")
    # Print total intensity comparison
    print(f"  Small domain (30Å):   {float(jnp.sum(intensities_small)):.2e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Effect of Beam Parameters

    Let's explore how beam quality affects the Ewald shell thickness and therefore the pattern.
    """
    )
    return


@app.cell
def _(ewald, jnp, rh):
    beam_conditions = [
        ("High quality", 1e-05, 0.0001),
        ("Standard", 0.0001, 0.001),
        ("Poor quality", 0.001, 0.005),
    ]
    _k = ewald.k_magnitude
    domain_1 = jnp.array([100.0, 100.0, 50.0])
    print("Effect of beam quality on Ewald shell thickness:")
    print("=" * 60)
    print(
        f"{'Condition':<15} {'ΔE/E':<10} {'Δθ (mrad)':<12} {'σ_shell (Å⁻¹)':<15}"
    )
    print("-" * 60)
    for _name, _dE_E, _dtheta in beam_conditions:
        _sigma_shell = rh.simul.compute_shell_sigma(_k, _dE_E, _dtheta)
        print(
            f"{_name:<15} {_dE_E:<10.1e} {_dtheta * 1000:<12.2f} {float(_sigma_shell):<15.4f}"
        )
    print("\nFor comparison, rod width at L=100Å: σ_rod = 0.025 Å⁻¹")
    return beam_conditions, domain_1


@app.cell
def _(beam_conditions, domain_1, ewald, jnp, plt, rh, theta_deg):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, (_name, _dE_E, _dtheta) in zip(_axes, beam_conditions):
        _overlap, _intensities = rh.simul.finite_domain_intensities(
            ewald=ewald,
            theta_deg=theta_deg,
            phi_deg=0.0,
            domain_extent_ang=domain_1,
            energy_spread_frac=_dE_E,
            beam_divergence_rad=_dtheta,
        )
        _ax.hist(_overlap, bins=50, range=(0, 1), alpha=0.7, edgecolor="black")
        _ax.set_xlabel("Overlap factor")
        _ax.set_ylabel("Count")
        _ax.set_title(
            f"{_name}\n(ΔE/E={_dE_E:.0e}, Δθ={_dtheta * 1000:.1f}mrad)"
        )
        _n_active = int(jnp.sum(_overlap > 0.1))
        _ax.text(
            0.95,
            0.95,
            f"Active: {_n_active}",
            transform=_ax.transAxes,
            ha="right",
            va="top",
        )
    plt.suptitle(
        "Effect of Beam Quality on Overlap Distribution", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Summary

    The finite domain broadening model accounts for two key effects:

    1. **Rod broadening** from finite coherent domain size:
       - σ_rod = 2π/(L×√(2π)) ≈ 2.5/L Å⁻¹
       - Smaller domains → broader rods → more reflections contribute

    2. **Shell broadening** from beam properties:
       - σ_shell = k×√[(ΔE/2E)² + Δθ²]
       - Typically dominated by beam divergence (≈1 mrad)

    3. **Overlap integral** gives continuous intensity weighting:
       - Replaces binary "on/off" Ewald sphere condition
       - More realistic for imperfect crystals and real beams

    ### Typical Values

    | Domain Size | Rod Width σ_rod |
    |------------|----------------|
    | 20 Å | 0.125 Å⁻¹ |
    | 100 Å | 0.025 Å⁻¹ |
    | 500 Å | 0.005 Å⁻¹ |

    | Beam Quality | Shell Width σ_shell (at 15 kV) |
    |-------------|-------------------------------|
    | FEG, collimated | ~0.006 Å⁻¹ |
    | Standard thermionic | ~0.06 Å⁻¹ |
    | Poor | ~0.3 Å⁻¹ |
    """
    )
    return


@app.cell
def _(jnp, rh):
    # Quick reference table
    print("Quick Reference: Rod Width vs Domain Size")
    print("=" * 40)
    for _L in [10, 20, 50, 100, 200, 500, 1000]:
        _extent = jnp.array([float(_L), float(_L), float(_L)])
        _sigma = rh.simul.extent_to_rod_sigma(_extent)
        print(f"L = {_L:4d} Å  →  σ = {float(_sigma[0]):.4f} Å⁻¹")
    return


if __name__ == "__main__":
    app.run()
