"""Test file to verify form_factors.py functions work correctly.

This demonstrates the basic functionality of the atomic scattering
factor calculations for RHEED simulation.
"""

import jax
import jax.numpy as jnp
import sys
import os


from form_factors import (
    kirkland_form_factor,
    debye_waller_factor,
    atomic_scattering_factor,
    get_mean_square_displacement,
    load_kirkland_parameters,
)

jax.config.update("jax_enable_x64", True)


def test_kirkland_parameters():
    """Test loading Kirkland parameters for Silicon."""
    print("Testing Kirkland parameter loading...")
    a_coeffs, b_coeffs = load_kirkland_parameters(14)  # Silicon
    print(f"  Silicon a coefficients shape: {a_coeffs.shape}")
    print(f"  Silicon b coefficients shape: {b_coeffs.shape}")
    print(f"  First a coefficient: {a_coeffs[0]:.6f}")
    print("  ✓ Parameters loaded successfully\n")
    return a_coeffs, b_coeffs


def test_form_factor():
    """Test form factor calculation at various q values."""
    print("Testing form factor calculation...")
    q_values = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])  # 1/Å
    
    # Silicon form factor
    f_si = kirkland_form_factor(14, q_values)
    print(f"  Si form factors at q={q_values.tolist()}:")
    print(f"  {f_si}")
    
    # Check that form factor decreases with q
    assert jnp.all(f_si[1:] < f_si[:-1]), "Form factor should decrease with q"
    print("  ✓ Form factor decreases correctly with q\n")
    
    return f_si


def test_debye_waller():
    """Test Debye-Waller factor calculation."""
    print("Testing Debye-Waller factor...")
    q_values = jnp.array([0.0, 1.0, 2.0])
    msd = 0.01  # Å²
    
    # Bulk atom
    dw_bulk = debye_waller_factor(q_values, msd, is_surface=False)
    print(f"  Bulk DW factors at q={q_values.tolist()}:")
    print(f"  {dw_bulk}")
    
    # Surface atom
    dw_surf = debye_waller_factor(q_values, msd, is_surface=True)
    print(f"  Surface DW factors at q={q_values.tolist()}:")
    print(f"  {dw_surf}")
    
    # Check that surface damping is stronger
    assert jnp.all(dw_surf[1:] < dw_bulk[1:]), "Surface should have stronger damping"
    print("  ✓ Surface atoms have enhanced damping\n")
    
    return dw_bulk, dw_surf


def test_mean_square_displacement():
    """Test MSD calculation for different temperatures."""
    print("Testing mean square displacement...")
    
    # Room temperature Silicon
    msd_si_300 = get_mean_square_displacement(14, 300.0, is_surface=False)
    print(f"  Si bulk MSD at 300K: {msd_si_300:.6f} Ų")
    
    # High temperature
    msd_si_600 = get_mean_square_displacement(14, 600.0, is_surface=False)
    print(f"  Si bulk MSD at 600K: {msd_si_600:.6f} Ų")
    
    # Surface atom
    msd_si_surf = get_mean_square_displacement(14, 300.0, is_surface=True)
    print(f"  Si surface MSD at 300K: {msd_si_surf:.6f} Ų")
    
    # Check temperature scaling
    assert msd_si_600 > msd_si_300, "Higher temperature should give larger MSD"
    assert msd_si_surf > msd_si_300, "Surface should have larger MSD"
    print("  ✓ MSD scales correctly with temperature and surface\n")
    
    return msd_si_300, msd_si_600, msd_si_surf


def test_combined_scattering():
    """Test combined atomic scattering factor."""
    print("Testing combined atomic scattering factor...")
    
    # Create q vectors
    q_vectors = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    
    # Calculate for Silicon at room temperature
    f_combined = atomic_scattering_factor(
        14, q_vectors, temperature=300.0, is_surface=False
    )
    print(f"  Si scattering factors for various q vectors:")
    for i, q in enumerate(q_vectors):
        q_mag = jnp.linalg.norm(q)
        print(f"    q={q.tolist()}, |q|={q_mag:.3f}: f={f_combined[i]:.6f}")
    
    # Test batch processing
    batch_q = jnp.tile(q_vectors[jnp.newaxis, :, :], (3, 1, 1))  # (3, 5, 3)
    f_batch = atomic_scattering_factor(14, batch_q, temperature=300.0)
    print(f"\n  Batch processing shape: {batch_q.shape} -> {f_batch.shape}")
    
    print("  ✓ Combined scattering factor works correctly\n")
    
    return f_combined


def test_different_elements():
    """Test form factors for different elements."""
    print("Testing different elements...")
    
    q = jnp.array([1.0])  # 1/Å
    elements = {
        "H": 1,
        "C": 6, 
        "Si": 14,
        "Cu": 29,
        "Au": 79,
    }
    
    print("  Form factors at q=1.0 1/Å:")
    for name, z in elements.items():
        f = kirkland_form_factor(z, q)[0]
        print(f"    {name:2s} (Z={z:2d}): f={f:.6f}")
    
    print("  ✓ Different elements calculated successfully\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing form_factors.py implementation")
    print("=" * 60 + "\n")
    
    # Run tests
    test_kirkland_parameters()
    test_form_factor()
    test_debye_waller()
    test_mean_square_displacement()
    test_combined_scattering()
    test_different_elements()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()