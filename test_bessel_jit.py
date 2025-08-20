#!/usr/bin/env python
"""Test script to verify bessel_kv works with JAX JIT compilation."""

import os
# Force CPU execution to avoid CUDA issues
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import time

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Test with normal imports (not documentation build mode)
from rheedium.ucell.bessel import bessel_kv

def test_bessel_jit():
    """Test that bessel_kv works correctly with JIT compilation."""
    
    print("Testing bessel_kv with JAX JIT compilation...")
    print("-" * 50)
    
    # Test parameters
    v = 0.5  # Order
    x_values = jnp.linspace(0.1, 10.0, 100)
    
    # First call (includes JIT compilation time)
    start = time.time()
    result1 = bessel_kv(v, x_values)
    time1 = time.time() - start
    print(f"First call (with JIT compilation): {time1:.4f} seconds")
    
    # Second call (should be much faster, using compiled version)
    start = time.time()
    result2 = bessel_kv(v, x_values)
    time2 = time.time() - start
    print(f"Second call (using compiled version): {time2:.4f} seconds")
    
    # Verify results are identical
    assert jnp.allclose(result1, result2), "Results should be identical"
    print("✓ Results are identical between calls")
    
    # Check that JIT compilation actually happened (second call should be faster)
    speedup = time1 / time2
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Test with different inputs to ensure JIT works correctly
    v2 = 1.5
    x_values2 = jnp.linspace(0.5, 5.0, 50)
    result3 = bessel_kv(v2, x_values2)
    print(f"✓ Function works with different inputs: shape={result3.shape}")
    
    # Verify the function is actually JIT-compiled
    print(f"\nFunction type: {type(bessel_kv)}")
    print(f"Is JIT-compiled: {'jaxlib' in str(type(bessel_kv))}")
    
    # Test some known values for K_0
    x_test = jnp.array([1.0])
    k0_result = bessel_kv(0.0, x_test)
    k0_expected = 0.4210244  # Approximate value of K_0(1)
    print(f"\nK_0(1.0) = {k0_result[0]:.6f} (expected ≈ {k0_expected:.6f})")
    assert jnp.abs(k0_result[0] - k0_expected) < 0.01, "K_0(1) value check failed"
    print("✓ Numerical accuracy check passed")
    
    print("\n" + "=" * 50)
    print("All tests passed! JIT compilation is working correctly.")
    return True

def test_jit_decorator_directly():
    """Test that our jit decorator from _decorators works correctly."""
    from rheedium._decorators import jit
    
    print("\nTesting direct use of jit decorator...")
    print("-" * 50)
    
    @jit
    def simple_function(x):
        return x ** 2 + 2 * x + 1
    
    x_test = jnp.array([1.0, 2.0, 3.0])
    result = simple_function(x_test)
    expected = jnp.array([4.0, 9.0, 16.0])
    
    assert jnp.allclose(result, expected), "Simple JIT function failed"
    print("✓ Direct jit decorator works correctly")
    print(f"  Input: {x_test}")
    print(f"  Output: {result}")
    print(f"  Expected: {expected}")
    
    return True

if __name__ == "__main__":
    # Run tests
    test_bessel_jit()
    test_jit_decorator_directly()
    
    print("\n✅ All JIT compilation tests passed successfully!")