"""Test suite for rheedium.types.custom_types type aliases."""

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.door import die_if_unbearable

from rheedium.types.custom_types import (
    float_jax_image,
    float_np_image,
    int_jax_image,
    int_np_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)


def _accepts(value, hint):
    die_if_unbearable(value, hint)
    return value


def _assert_rejected(value, hint):
    with pytest.raises(Exception):
        die_if_unbearable(value, hint)


class TestScalarFloat:
    """Tests for the scalar_float type alias."""

    def test_accepts_python_float(self):
        assert _accepts(3.14, scalar_float) == 3.14

    def test_accepts_jax_scalar(self):
        result = _accepts(jnp.float64(2.5), scalar_float)
        assert float(result) == 2.5

    def test_rejects_string(self):
        _assert_rejected("hello", scalar_float)

    def test_rejects_list(self):
        _assert_rejected([1.0, 2.0], scalar_float)


class TestScalarInt:
    """Tests for the scalar_int type alias."""

    def test_accepts_python_int(self):
        assert _accepts(42, scalar_int) == 42

    def test_accepts_jax_int_scalar(self):
        result = _accepts(jnp.int32(7), scalar_int)
        assert int(result) == 7

    def test_rejects_string(self):
        _assert_rejected("hello", scalar_int)


class TestScalarBool:
    """Tests for the scalar_bool type alias."""

    def test_accepts_python_bool(self):
        assert _accepts(True, scalar_bool) is True
        assert _accepts(False, scalar_bool) is False

    def test_accepts_jax_bool_scalar(self):
        result = _accepts(jnp.bool_(True), scalar_bool)
        assert bool(result) is True


class TestScalarNum:
    """Tests for the scalar_num type alias."""

    def test_accepts_python_int(self):
        assert _accepts(5, scalar_num) == 5

    def test_accepts_python_float(self):
        assert _accepts(3.14, scalar_num) == 3.14

    def test_accepts_jax_scalar(self):
        result = _accepts(jnp.float64(1.5), scalar_num)
        assert float(result) == 1.5

    def test_rejects_string(self):
        _assert_rejected("hello", scalar_num)


class TestNonJaxNumber:
    """Tests for the non_jax_number type alias."""

    def test_accepts_python_int(self):
        assert _accepts(10, non_jax_number) == 10

    def test_accepts_python_float(self):
        assert _accepts(2.5, non_jax_number) == 2.5

    def test_rejects_jax_array(self):
        _assert_rejected(jnp.float64(1.0), non_jax_number)


class TestFloatJaxImage:
    """Tests for the float_jax_image type alias."""

    def test_accepts_2d_float_array(self):
        img = jnp.ones((64, 128), dtype=jnp.float32)
        result = _accepts(img, float_jax_image)
        assert result.shape == (64, 128)

    def test_accepts_float64(self):
        img = jnp.zeros((32, 32), dtype=jnp.float64)
        result = _accepts(img, float_jax_image)
        assert result.shape == (32, 32)

    def test_rejects_int_array(self):
        img = jnp.ones((64, 64), dtype=jnp.int32)
        _assert_rejected(img, float_jax_image)

    def test_rejects_3d_array(self):
        img = jnp.ones((3, 64, 64), dtype=jnp.float32)
        _assert_rejected(img, float_jax_image)


class TestIntJaxImage:
    """Tests for the int_jax_image type alias."""

    def test_accepts_2d_int_array(self):
        img = jnp.ones((64, 128), dtype=jnp.int32)
        result = _accepts(img, int_jax_image)
        assert result.shape == (64, 128)

    def test_rejects_float_array(self):
        img = jnp.ones((64, 64), dtype=jnp.float32)
        _assert_rejected(img, int_jax_image)

    def test_rejects_1d_array(self):
        img = jnp.ones((64,), dtype=jnp.int32)
        _assert_rejected(img, int_jax_image)


class TestFloatNpImage:
    """Tests for the float_np_image type alias."""

    def test_accepts_2d_float_array(self):
        img = np.ones((64, 128), dtype=np.float32)
        result = _accepts(img, float_np_image)
        assert result.shape == (64, 128)

    def test_rejects_int_array(self):
        img = np.ones((64, 64), dtype=np.int32)
        _assert_rejected(img, float_np_image)

    def test_rejects_3d_array(self):
        img = np.ones((3, 64, 64), dtype=np.float32)
        _assert_rejected(img, float_np_image)


class TestIntNpImage:
    """Tests for the int_np_image type alias."""

    def test_accepts_2d_int_array(self):
        img = np.ones((64, 128), dtype=np.int32)
        result = _accepts(img, int_np_image)
        assert result.shape == (64, 128)

    def test_rejects_float_array(self):
        img = np.ones((64, 64), dtype=np.float32)
        _assert_rejected(img, int_np_image)

    def test_rejects_1d_array(self):
        img = np.ones((64,), dtype=np.int32)
        _assert_rejected(img, int_np_image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
