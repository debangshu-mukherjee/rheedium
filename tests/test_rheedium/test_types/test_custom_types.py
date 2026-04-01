"""Test suite for rheedium.types.custom_types type aliases.

Verifies that type aliases accept the expected Python and JAX types
and reject incorrect types via beartype runtime checking.
"""

import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float, Integer, jaxtyped

from rheedium.types.custom_types import (
    float_image,
    int_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)


class TestScalarFloat:
    """Tests for the scalar_float type alias."""

    def test_accepts_python_float(self) -> None:
        @beartype
        def f(x: scalar_float) -> scalar_float:
            return x

        assert f(3.14) == 3.14

    def test_accepts_jax_scalar(self) -> None:
        @beartype
        def f(x: scalar_float) -> scalar_float:
            return x

        val = jnp.float64(2.5)
        result = f(val)
        assert float(result) == 2.5

    def test_rejects_string(self) -> None:
        @beartype
        def f(x: scalar_float) -> None:
            pass

        with pytest.raises(Exception):
            f("hello")

    def test_rejects_list(self) -> None:
        @beartype
        def f(x: scalar_float) -> None:
            pass

        with pytest.raises(Exception):
            f([1.0, 2.0])


class TestScalarInt:
    """Tests for the scalar_int type alias."""

    def test_accepts_python_int(self) -> None:
        @beartype
        def f(x: scalar_int) -> scalar_int:
            return x

        assert f(42) == 42

    def test_accepts_jax_int_scalar(self) -> None:
        @beartype
        def f(x: scalar_int) -> scalar_int:
            return x

        val = jnp.int32(7)
        result = f(val)
        assert int(result) == 7

    def test_rejects_string(self) -> None:
        @beartype
        def f(x: scalar_int) -> None:
            pass

        with pytest.raises(Exception):
            f("hello")


class TestScalarBool:
    """Tests for the scalar_bool type alias."""

    def test_accepts_python_bool(self) -> None:
        @beartype
        def f(x: scalar_bool) -> scalar_bool:
            return x

        assert f(True) is True
        assert f(False) is False

    def test_accepts_jax_bool_scalar(self) -> None:
        @beartype
        def f(x: scalar_bool) -> scalar_bool:
            return x

        val = jnp.bool_(True)
        result = f(val)
        assert bool(result) is True


class TestScalarNum:
    """Tests for the scalar_num type alias."""

    def test_accepts_python_int(self) -> None:
        @beartype
        def f(x: scalar_num) -> scalar_num:
            return x

        assert f(5) == 5

    def test_accepts_python_float(self) -> None:
        @beartype
        def f(x: scalar_num) -> scalar_num:
            return x

        assert f(3.14) == 3.14

    def test_accepts_jax_scalar(self) -> None:
        @beartype
        def f(x: scalar_num) -> scalar_num:
            return x

        val = jnp.float64(1.5)
        result = f(val)
        assert float(result) == 1.5

    def test_rejects_string(self) -> None:
        @beartype
        def f(x: scalar_num) -> None:
            pass

        with pytest.raises(Exception):
            f("hello")


class TestNonJaxNumber:
    """Tests for the non_jax_number type alias."""

    def test_accepts_python_int(self) -> None:
        @beartype
        def f(x: non_jax_number) -> non_jax_number:
            return x

        assert f(10) == 10

    def test_accepts_python_float(self) -> None:
        @beartype
        def f(x: non_jax_number) -> non_jax_number:
            return x

        assert f(2.5) == 2.5

    def test_rejects_jax_array(self) -> None:
        @beartype
        def f(x: non_jax_number) -> None:
            pass

        with pytest.raises(Exception):
            f(jnp.float64(1.0))


class TestFloatImage:
    """Tests for the float_image type alias."""

    def test_accepts_2d_float_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: float_image) -> float_image:
            return x

        img = jnp.ones((64, 128), dtype=jnp.float32)
        result = f(img)
        assert result.shape == (64, 128)

    def test_accepts_float64(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: float_image) -> float_image:
            return x

        img = jnp.zeros((32, 32), dtype=jnp.float64)
        result = f(img)
        assert result.shape == (32, 32)

    def test_rejects_int_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: float_image) -> None:
            pass

        img = jnp.ones((64, 64), dtype=jnp.int32)
        with pytest.raises(Exception):
            f(img)

    def test_rejects_3d_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: float_image) -> None:
            pass

        img = jnp.ones((3, 64, 64), dtype=jnp.float32)
        with pytest.raises(Exception):
            f(img)


class TestIntImage:
    """Tests for the int_image type alias."""

    def test_accepts_2d_int_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: int_image) -> int_image:
            return x

        img = jnp.ones((64, 128), dtype=jnp.int32)
        result = f(img)
        assert result.shape == (64, 128)

    def test_rejects_float_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: int_image) -> None:
            pass

        img = jnp.ones((64, 64), dtype=jnp.float32)
        with pytest.raises(Exception):
            f(img)

    def test_rejects_1d_array(self) -> None:
        @jaxtyped(typechecker=beartype)
        def f(x: int_image) -> None:
            pass

        img = jnp.ones((64,), dtype=jnp.int32)
        with pytest.raises(Exception):
            f(img)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
