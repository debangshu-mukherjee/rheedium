JAX Validation Pattern for Factory Functions
============================================

Overview
--------
This document describes the standardized JAX-compatible validation pattern used for all factory functions in the rheedium codebase. This pattern ensures compile-time validation, JIT compatibility, and optimal performance.

Why This Pattern?
-----------------
Traditional Python validation using `if` statements breaks JAX's control flow tracing, preventing functions from being JIT-compiled. The JAX validation pattern solves this by:

1. **Compile-time validation**: Errors are caught during JIT compilation
2. **Zero runtime overhead**: No validation checks during execution
3. **JIT compatibility**: Functions work seamlessly with `jax.jit()`, `jax.grad()`, etc.
4. **Type safety**: JAX's compilation system enforces validation rules

The Pattern
-----------

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: python

    @jaxtyped(typechecker=beartype)
    def create_data_structure(data, param1, param2):
        """
        Factory function following JAX validation pattern.
        """
        # Convert inputs to JAX arrays
        data = jnp.asarray(data, dtype=jnp.float64)
        param1 = jnp.asarray(param1, dtype=jnp.float64)
        param2 = jnp.asarray(param2, dtype=jnp.float64)

        def validate_and_create():
            # Individual validation functions
            def check_shape():
                return lax.cond(
                    data.shape == expected_shape,
                    lambda: data,  # Pass through if valid
                    lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))  # Fail if invalid
                )
            
            def check_values():
                return lax.cond(
                    jnp.all(data >= 0),
                    lambda: data,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
                )
            
            def check_parameters():
                return lax.cond(
                    jnp.logical_and(param1 > 0, param2 > 0),
                    lambda: (param1, param2),
                    lambda: lax.stop_gradient(lax.cond(False, lambda: (param1, param2), lambda: (param1, param2)))
                )
            
            # Execute all validations (no assignment needed)
            check_shape()
            check_values()
            check_parameters()
            
            # Return original data (now guaranteed valid)
            return DataStructure(
                data=data,
                param1=param1,
                param2=param2
            )
        
        return validate_and_create()

Key Components
~~~~~~~~~~~~~~

1. **Validation Functions**: Each check is wrapped in a function using `lax.cond`
2. **No Assignment**: Call validation functions without storing results
3. **Original Data**: Return the original input data after validation
4. **Error Branch**: Use `lax.stop_gradient(lax.cond(False, ...))` for invalid cases

Common Validation Patterns
--------------------------

Shape Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    def check_shape():
        return lax.cond(
            data.shape == (expected_dim1, expected_dim2),
            lambda: data,
            lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
        )

Value Range Validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def check_range():
        return lax.cond(
            jnp.logical_and(jnp.all(data >= min_val), jnp.all(data <= max_val)),
            lambda: data,
            lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
        )

Finite Values Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def check_finite():
        return lax.cond(
            jnp.all(jnp.isfinite(data)),
            lambda: data,
            lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
        )

Multiple Parameter Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def check_multiple_params():
        return lax.cond(
            jnp.logical_and(
                param1 > 0,
                jnp.logical_and(param2 > 0, param3 > 0)
            ),
            lambda: (param1, param2, param3),
            lambda: lax.stop_gradient(lax.cond(False, lambda: (param1, param2, param3), lambda: (param1, param2, param3)))
        )

Conditional Validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def check_conditional():
        def check_scalar():
            return lax.cond(
                scalar_param > 0,
                lambda: scalar_param,
                lambda: lax.stop_gradient(lax.cond(False, lambda: scalar_param, lambda: scalar_param))
            )
        
        def check_array():
            return lax.cond(
                jnp.logical_and(
                    array_param.shape == (2,),
                    jnp.all(array_param > 0)
                ),
                lambda: array_param,
                lambda: lax.stop_gradient(lax.cond(False, lambda: array_param, lambda: array_param))
            )
        
        return lax.cond(
            param.ndim == 0,
            check_scalar,
            check_array
        )

Best Practices
--------------

1. **Always convert inputs to JAX arrays** at the beginning
2. **Use descriptive function names** for validation checks
3. **Group related validations** in the same function
4. **Document validation logic** in docstrings
5. **Test with invalid inputs** to ensure compilation errors
6. **Keep validation functions simple** and focused

Example: Complete Factory Function
----------------------------------

.. code-block:: python

    @jaxtyped(typechecker=beartype)
    def create_crystal_structure(
        frac_positions: Float[Array, "* 4"],
        cart_positions: Num[Array, "* 4"],
        cell_lengths: Num[Array, "3"],
        cell_angles: Num[Array, "3"],
    ) -> CrystalStructure:
        """
        Factory function to create a CrystalStructure instance with JAX validation.
        
        Parameters
        ----------
        frac_positions : Float[Array, "* 4"]
            Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
        cart_positions : Num[Array, "* 4"]
            Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
        cell_lengths : Num[Array, "3"]
            Unit cell lengths [a, b, c] in Ångstroms.
        cell_angles : Num[Array, "3"]
            Unit cell angles [α, β, γ] in degrees.

        Returns
        -------
        CrystalStructure
            A validated CrystalStructure instance.

        Raises
        ------
        CompilationError
            If validation fails during JIT compilation.
        """
        # Convert inputs to JAX arrays
        frac_positions = jnp.asarray(frac_positions)
        cart_positions = jnp.asarray(cart_positions)
        cell_lengths = jnp.asarray(cell_lengths)
        cell_angles = jnp.asarray(cell_angles)

        def validate_and_create():
            # Shape validations
            def check_frac_shape():
                return lax.cond(
                    frac_positions.shape[1] == 4,
                    lambda: frac_positions,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: frac_positions, lambda: frac_positions))
                )
            
            def check_cart_shape():
                return lax.cond(
                    cart_positions.shape[1] == 4,
                    lambda: cart_positions,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: cart_positions, lambda: cart_positions))
                )
            
            def check_cell_lengths_shape():
                return lax.cond(
                    cell_lengths.shape == (3,),
                    lambda: cell_lengths,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: cell_lengths, lambda: cell_lengths))
                )
            
            def check_cell_angles_shape():
                return lax.cond(
                    cell_angles.shape == (3,),
                    lambda: cell_angles,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: cell_angles, lambda: cell_angles))
                )
            
            # Consistency validations
            def check_atom_count():
                return lax.cond(
                    frac_positions.shape[0] == cart_positions.shape[0],
                    lambda: (frac_positions, cart_positions),
                    lambda: lax.stop_gradient(lax.cond(False, lambda: (frac_positions, cart_positions), lambda: (frac_positions, cart_positions)))
                )
            
            def check_atomic_numbers():
                return lax.cond(
                    jnp.all(frac_positions[:, 3] == cart_positions[:, 3]),
                    lambda: (frac_positions, cart_positions),
                    lambda: lax.stop_gradient(lax.cond(False, lambda: (frac_positions, cart_positions), lambda: (frac_positions, cart_positions)))
                )
            
            # Value validations
            def check_cell_lengths_positive():
                return lax.cond(
                    jnp.all(cell_lengths > 0),
                    lambda: cell_lengths,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: cell_lengths, lambda: cell_lengths))
                )
            
            def check_cell_angles_valid():
                return lax.cond(
                    jnp.all(jnp.logical_and(cell_angles > 0, cell_angles < 180)),
                    lambda: cell_angles,
                    lambda: lax.stop_gradient(lax.cond(False, lambda: cell_angles, lambda: cell_angles))
                )
            
            # Execute all validations
            check_frac_shape()
            check_cart_shape()
            check_cell_lengths_shape()
            check_cell_angles_shape()
            check_atom_count()
            check_atomic_numbers()
            check_cell_lengths_positive()
            check_cell_angles_valid()
            
            # Return validated structure
            return CrystalStructure(
                frac_positions=frac_positions,
                cart_positions=cart_positions,
                cell_lengths=cell_lengths,
                cell_angles=cell_angles,
            )
        
        return validate_and_create()

Testing
-------

To test that validation works correctly:

.. code-block:: python

    import jax
    import rheedium as rh

    # This should work
    @jax.jit
    def test_valid():
        crystal = rh.types.create_crystal_structure(
            frac_positions=jnp.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]]),
            cart_positions=jnp.array([[0, 0, 0, 1], [2.7155, 2.7155, 2.7155, 1]]),
            cell_lengths=jnp.array([5.431, 5.431, 5.431]),
            cell_angles=jnp.array([90, 90, 90])
        )
        return crystal.cell_lengths

    # This should fail at compilation time
    @jax.jit
    def test_invalid():
        crystal = rh.types.create_crystal_structure(
            frac_positions=jnp.array([[0, 0, 0, 1]]),  # Wrong shape
            cart_positions=jnp.array([[0, 0, 0, 1], [2.7155, 2.7155, 2.7155, 1]]),
            cell_lengths=jnp.array([5.431, 5.431, 5.431]),
            cell_angles=jnp.array([90, 90, 90])
        )
        return crystal.cell_lengths

    # Valid case works
    result = test_valid()
    
    # Invalid case fails at compilation
    try:
        result = test_invalid()
    except Exception as e:
        print("Validation failed as expected:", e)

Migration Guide
---------------

Converting from Python `if` statements:

**Before:**
.. code-block:: python

    if data.shape != expected_shape:
        raise ValueError("Invalid shape")
    if jnp.any(data < 0):
        raise ValueError("Negative values not allowed")
    return DataStructure(data=data)

**After:**
.. code-block:: python

    def validate_and_create():
        def check_shape():
            return lax.cond(
                data.shape == expected_shape,
                lambda: data,
                lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
            )
        
        def check_values():
            return lax.cond(
                jnp.all(data >= 0),
                lambda: data,
                lambda: lax.stop_gradient(lax.cond(False, lambda: data, lambda: data))
            )
        
        check_shape()
        check_values()
        return DataStructure(data=data)
    
    return validate_and_create()

Conclusion
----------
This JAX validation pattern provides compile-time safety, optimal performance, and full JIT compatibility. All factory functions in the rheedium codebase should follow this pattern to ensure consistency and reliability. 