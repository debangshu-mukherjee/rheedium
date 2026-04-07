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
    # RHEED Simulation: SrTiO3 Perovskite Surface

    This tutorial demonstrates kinematic RHEED simulation for SrTiO3, a prototypical perovskite oxide.

    ## Setup

    Load the necessary modules.
    """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import rheedium as rh

    return jnp, rh


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Use `autoreload` to reload changed modules, you may disable this for normal use.
    """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _(rh):
    structure_file = rh.inout.parse_cif("../tests/test_data/SrTiO3.cif")
    return (structure_file,)


@app.cell
def _(structure_file):
    structure_file
    return


@app.cell
def _(jnp):
    zone_axis = jnp.array([1, 1, 1])
    thickness = jnp.asarray([12, 12, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Now, load the parameters
    """
    )
    return


@app.cell
def _():
    voltage_kV = 20.0  # Electron beam voltage in kV
    theta_deg = 2.0  # Grazing angle
    hmax, kmax = 4, 4  # In-plane reciprocal lattice bounds
    detector_distance = 100.0  # Detector distance in mm
    points_per_streak = 100  # Points to sample along each CTR streak
    return (
        detector_distance,
        hmax,
        kmax,
        points_per_streak,
        theta_deg,
        voltage_kV,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Now, we will process things step by step.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Electron wavelength (Å)
    """
    )
    return


@app.cell
def _(rh, voltage_kV):
    lam_ang = rh.simul.wavelength_ang(voltage_kV)
    lam_ang
    return (lam_ang,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build real-space lattice vectors
    """
    )
    return


@app.cell
def _(rh, structure_file):
    cell_vectors = rh.ucell.build_cell_vectors(
        *structure_file.cell_lengths, *structure_file.cell_angles
    )
    return (cell_vectors,)


@app.cell
def _(cell_vectors):
    cell_vectors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generate reciprocal lattice vectors
    """
    )
    return


@app.cell
def _(hmax, kmax, rh, structure_file):
    # For reference: generate bulk reciprocal lattice points
    lmax = 2  # For visualization only
    Gs = rh.ucell.generate_reciprocal_points(
        crystal=structure_file,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    print(f"Reciprocal lattice points (for reference): {Gs.shape[0]}")
    return (lmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Incident wavevector
    """
    )
    return


@app.cell
def _(lam_ang, rh, theta_deg):
    k_in = rh.simul.incident_wavevector(lam_ang, theta_deg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build EwaldData for efficient angle scanning

    The `build_ewald_data` function pre-computes all angle-independent quantities
    (reciprocal lattice, structure factors with Kirkland form factors and Debye-Waller).
    This enables efficient reuse when scanning beam angles.
    """
    )
    return


@app.cell
def _(hmax, kmax, lmax, rh, structure_file, voltage_kV):
    # Build EwaldData - pre-compute structure factors and reciprocal lattice
    # This only needs to be done once per crystal/voltage combination
    ewald_data = rh.simul.build_ewald_data(
        crystal=structure_file,
        voltage_kv=voltage_kV,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        temperature=300.0,  # Room temperature for Debye-Waller factors
    )

    print(f"Electron wavelength: {float(ewald_data.wavelength_ang):.4f} Å")
    print(f"Wavevector magnitude |k|: {float(ewald_data.k_magnitude):.2f} 1/Å")
    print(
        f"Number of reciprocal lattice points: {ewald_data.g_vectors.shape[0]}"
    )
    print(f"HKL grid shape: {ewald_data.hkl_grid.shape}")
    return (ewald_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Find allowed reflections for specific beam angles

    Now use `ewald_allowed_reflections` to find which reflections satisfy the Ewald
    sphere condition for a given beam orientation (theta, phi).
    """
    )
    return


@app.cell
def _(ewald_data, jnp, rh, theta_deg):
    # Find allowed reflections for theta=2°, phi=0°
    allowed_idx, k_out_allowed, intensities_allowed = (
        rh.simul.ewald_allowed_reflections(
            ewald=ewald_data,
            theta_deg=theta_deg,
            phi_deg=0.0,  # Beam along x-axis
            tolerance=0.1,  # 10% tolerance for Ewald sphere intersection
        )
    )

    # Filter out padding values (-1 indices)
    valid_mask = allowed_idx >= 0
    n_valid = int(jnp.sum(valid_mask))

    print(f"Found {n_valid} allowed reflections at theta={theta_deg}°, phi=0°")

    # Show the allowed HKL indices with highest intensity
    if n_valid > 0:
        valid_idx = allowed_idx[valid_mask]
        valid_hkl = ewald_data.hkl_grid[valid_idx]
        valid_I = intensities_allowed[valid_mask]

        # Sort by intensity (descending)
        sorted_order = jnp.argsort(-valid_I)

        print("\nTop 10 allowed reflections by intensity:")
        for i in range(min(10, n_valid)):
            j = sorted_order[i]
            h, k, l = valid_hkl[j]
            I = valid_I[j]
            print(
                f"  ({int(h):2d}, {int(k):2d}, {int(l):2d})  I = {float(I):.4f}"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Azimuthal scan using pre-computed EwaldData

    One major advantage of pre-computing EwaldData is efficient azimuthal scans.
    The structure factors only need to be computed once!
    """
    )
    return


@app.cell
def _(ewald_data, jnp, rh, theta_deg):
    # Azimuthal scan: find reflections at different phi angles
    # SrTiO3 is cubic, so 90° periodicity expected
    phi_angles = jnp.linspace(0, 90, 10)  # Scan from 0° to 90°

    print("Azimuthal scan results (SrTiO3 cubic symmetry):")
    print("-" * 50)
    for phi in phi_angles:
        idx, k_out, intensities = rh.simul.ewald_allowed_reflections(
            ewald=ewald_data,
            theta_deg=theta_deg,
            phi_deg=float(phi),
            tolerance=0.1,
        )
        n_refl = int(jnp.sum(idx >= 0))
        total_I = float(jnp.sum(intensities[idx >= 0])) if n_refl > 0 else 0.0
        print(
            f"  phi = {float(phi):5.1f}°: {n_refl:3d} reflections, total I = {total_I:.2f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulate RHEED spot pattern using discrete 3D reciprocal lattice

    The `kinematic_spot_simulator` treats the reciprocal lattice as discrete 3D points.
    This is useful for bulk-like diffraction or when only spot positions matter.
    For grazing incidence RHEED, we need a larger `lmax` to capture reflections where
    G_z exceeds the incident beam's z-component.
    """
    )
    return


@app.cell
def _(
    detector_distance,
    hmax,
    kmax,
    rh,
    structure_file,
    theta_deg,
    voltage_kV,
):
    # Generate RHEED spot pattern using discrete 3D reciprocal lattice
    # kinematic_spot_simulator finds where discrete G points intersect the Ewald sphere
    spot_pattern = rh.simul.kinematic_spot_simulator(
        crystal=structure_file,
        voltage_kv=voltage_kV,
        theta_deg=theta_deg,
        hmax=hmax,
        kmax=kmax,
        lmax=5,  # Need larger lmax for grazing incidence
        detector_distance=detector_distance,
        tolerance=0.05,
    )

    print(f"Number of spots: {len(spot_pattern.intensities)}")
    print(
        f"X-coordinate range: [{spot_pattern.detector_points[:, 0].min():.2f}, {spot_pattern.detector_points[:, 0].max():.2f}] mm"
    )
    print(
        f"Y-coordinate range: [{spot_pattern.detector_points[:, 1].min():.2f}, {spot_pattern.detector_points[:, 1].max():.2f}] mm"
    )
    return (spot_pattern,)


@app.cell
def _(rh, spot_pattern):
    # Plot the spot pattern
    rh.plots.plot_rheed(spot_pattern, grid_size=400, interp_type="linear")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulate RHEED streak pattern using Crystal Truncation Rods (CTRs)

    The `streak_simulator` models RHEED as diffraction from continuous rods in reciprocal space,
    producing the characteristic vertical streaks seen in real RHEED patterns.
    """
    )
    return


@app.cell
def _(
    detector_distance,
    hmax,
    kmax,
    points_per_streak,
    rh,
    structure_file,
    theta_deg,
    voltage_kV,
):
    # Generate RHEED streak pattern for SrTiO3
    # kinematic_ctr_simulator models continuous crystal truncation rods (CTRs)
    # and returns a RHEEDPattern directly
    streak_pattern = rh.simul.kinematic_ctr_simulator(
        crystal=structure_file,
        voltage_kv=voltage_kV,
        theta_deg=theta_deg,
        hmax=hmax,
        kmax=kmax,
        detector_distance=detector_distance,
        n_points_per_rod=points_per_streak,
    )

    print(f"Number of streak points: {len(streak_pattern.intensities)}")
    print(
        f"X-coordinate range: [{streak_pattern.detector_points[:, 0].min():.2f}, {streak_pattern.detector_points[:, 0].max():.2f}] mm"
    )
    print(
        f"Y-coordinate range: [{streak_pattern.detector_points[:, 1].min():.2f}, {streak_pattern.detector_points[:, 1].max():.2f}] mm"
    )
    return (streak_pattern,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Check how many reflections were found
    """
    )
    return


@app.cell
def _(jnp, streak_pattern):
    # Summary of streak pattern
    print(f"Number of streak points: {len(streak_pattern.intensities)}")
    print(
        f"Number of unique rods: {len(jnp.unique(streak_pattern.G_indices))}"
    )
    print(
        f"Intensity range: [{streak_pattern.intensities.min():.3f}, {streak_pattern.intensities.max():.3f}]"
    )
    print("\nSample detector coordinates (first 5):")
    print(streak_pattern.detector_points[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot the simulated RHEED pattern
    """
    )
    return


@app.cell
def _(rh, streak_pattern):
    # Plot using phosphor colormap
    # streak_pattern is already a RHEEDPattern from kinematic_ctr_simulator
    rh.plots.plot_rheed(streak_pattern, grid_size=400, interp_type="linear")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Understanding the RHEED Streak Pattern

    ## Physics of RHEED Streaks

    The `streak_simulator` properly models RHEED from surfaces:

    1. **Crystal Truncation Rods (CTRs)**: Surface breaks z-symmetry, creating continuous rods in reciprocal space
    2. **Ewald Sphere Intersection**: Each (h,k) rod intersects the Ewald sphere along an arc
    3. **Vertical Streaks**: Arc projections onto the detector create characteristic vertical lines

    ## Key Features:
    - **Streak spacing**: Determined by in-plane reciprocal lattice (∝ 1/a)
    - **Intensity modulation**: CTR intensity varies as 1/sin²(πl), brightest at Bragg positions
    - **Streak length**: Depends on Ewald sphere curvature and grazing angle
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SlicedCrystal for Oriented Surfaces

    The `SlicedCrystal` type allows creating properly oriented surface slabs for more advanced simulations.
    """
    )
    return


@app.cell
def _(jnp, rh, structure_file):
    # Create a properly oriented [111] surface slab
    slab_111 = rh.types.bulk_to_slice(
        bulk_crystal=structure_file,
        orientation=jnp.array([1, 1, 1]),  # (111) surface
        depth=20.0,  # 20 Å deep
        x_extent=150.0,  # 150 Å wide
        y_extent=150.0,  # 150 Å tall
    )

    print(f"Created [111] surface slab:")
    print(f"  Number of atoms: {slab_111.cart_positions.shape[0]}")
    print(f"  Orientation: {slab_111.orientation}")
    print(
        f"  Dimensions: {slab_111.x_extent} x {slab_111.y_extent} x {slab_111.depth} Å"
    )
    print(
        f"  Z range: {slab_111.cart_positions[:, 2].min():.2f} to {slab_111.cart_positions[:, 2].max():.2f} Å"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Proper Solution: Using SlicedCrystal for True [111] Surface

    The new `SlicedCrystal` type properly orients the crystal!
    """
    )
    return


@app.cell
def _(jnp, rh, structure_file):
    # Create a properly oriented [111] surface slab
    slab_111_1 = rh.types.bulk_to_slice(
        bulk_crystal=structure_file,
        orientation=jnp.array([1, 1, 1]),
        depth=20.0,
        x_extent=150.0,
        y_extent=150.0,
    )
    print(f"Created [111] surface slab:")
    print(
        f"  Number of atoms: {slab_111_1.cart_positions.shape[0]}"
    )  # (111) surface
    print(f"  Orientation: {slab_111_1.orientation}")  # 20 Å deep
    print(
        f"  Dimensions: {slab_111_1.x_extent} x {slab_111_1.y_extent} x {slab_111_1.depth} Å"
    )  # 150 Å wide
    print(
        f"  Z range: {slab_111_1.cart_positions[:, 2].min():.2f} to {slab_111_1.cart_positions[:, 2].max():.2f} Å"
    )  # 150 Å tall
    return (slab_111_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Alternative: Plot streak pattern directly

    The streak pattern was already plotted above using `plot_rheed`.
    """
    )
    return


@app.cell
def _(streak_pattern):
    # Additional visualization: scatter plot showing individual streaks
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        streak_pattern.detector_points[:, 0],
        streak_pattern.detector_points[:, 1],
        s=2,
        c=streak_pattern.intensities,
        cmap="hot",
        alpha=0.8,
    )
    ax.set_facecolor("black")
    ax.set_xlabel("Horizontal position (mm)")
    ax.set_ylabel("Vertical position (mm)")
    ax.set_title("SrTiO3 RHEED Streak Pattern\n20 keV, theta = 2 deg")
    ax.set_aspect("equal")
    plt.colorbar(scatter, label="Intensity")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Test Multislice Simulator

    Now let's test the new multislice (dynamical) simulator with the SlicedCrystal!
    """
    )
    return


@app.cell
def _(rh, slab_111_1, voltage_kV):
    # Convert the SlicedCrystal to PotentialSlices
    potential = rh.simul.sliced_crystal_to_potential(
        sliced_crystal=slab_111_1,
        slice_thickness=2.0,
        pixel_size=0.5,
        voltage_kv=voltage_kV,
    )
    print(f"PotentialSlices created:")
    print(f"  Shape: {potential.slices.shape}")  # 2 Å per slice
    print(
        f"  Slice thickness: {potential.slice_thickness} Å"
    )  # 0.5 Å pixels (balance between accuracy and speed)
    print(
        f"  Pixel size: ({potential.x_calibration}, {potential.y_calibration}) Å"
    )
    return (potential,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulate RHEED pattern using multislice

    This uses the dynamical diffraction algorithm that accounts for multiple scattering.
    """
    )
    return


@app.cell
def _(detector_distance, potential, rh, theta_deg, voltage_kV):
    # Simulate RHEED pattern with multislice
    pattern_multislice = rh.simul.multislice_simulator(
        potential_slices=potential,
        voltage_kv=voltage_kV,
        theta_deg=theta_deg,
        phi_deg=90.0,  # Vertical streaks
        detector_distance=detector_distance,
        detector_width=200.0,  # mm
        detector_height=200.0,  # mm
        detector_pixels_x=256,  # Reduced for speed
        detector_pixels_y=256,
    )

    print(f"Multislice RHEED pattern:")
    print(f"  Number of points: {len(pattern_multislice.intensities)}")
    print(
        f"  Intensity range: [{pattern_multislice.intensities.min():.2e}, {pattern_multislice.intensities.max():.2e}]"
    )
    return (pattern_multislice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compare kinematic vs multislice

    Kinematic approximation: single scattering, fast, good for thin samples
    Multislice: multiple scattering, slower, accurate for thick samples
    """
    )
    return


@app.cell
def _(pattern_multislice, rh):
    # Plot the multislice RHEED pattern
    rh.plots.plot_rheed(pattern_multislice, grid_size=512, interp_type="cubic")
    return


if __name__ == "__main__":
    app.run()
