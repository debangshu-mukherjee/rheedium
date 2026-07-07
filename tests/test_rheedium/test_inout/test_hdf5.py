"""Tests for HDF5 PyTree serialization."""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Integer

import rheedium as rh
from rheedium.types import (
    CrystalStructure,
    DetectorGeometry,
    ReconResult,
    RHEEDImage,
    RHEEDPattern,
    SlicedCrystal,
    SurfaceConfig,
    XYZData,
)

pytest.importorskip("h5py")


def _assert_round_trip_equal(actual: Any, expected: Any) -> None:
    """Recursively compare nested rheedium pytrees exactly."""
    if expected is None or isinstance(expected, str):
        assert actual == expected
        return

    if isinstance(expected, dict):
        assert list(actual.keys()) == list(expected.keys())
        key: Any
        for key in expected:
            _assert_round_trip_equal(actual[key], expected[key])
        return

    if isinstance(expected, list):
        assert len(actual) == len(expected)
        actual_item: Any
        expected_item: Any
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_round_trip_equal(actual_item, expected_item)
        return

    if isinstance(expected, tuple) and hasattr(expected, "_fields"):
        assert type(actual) is type(expected)
        field_name: Any
        for field_name in expected._fields:
            _assert_round_trip_equal(
                getattr(actual, field_name),
                getattr(expected, field_name),
            )
        return

    if isinstance(expected, eqx.Module):
        assert type(actual) is type(expected)
        field: Any
        for field in dataclasses.fields(expected):
            _assert_round_trip_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
        return

    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_round_trip_equal(actual_item, expected_item)
        return

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _build_sample_pytrees() -> dict[str, object]:
    """Construct one instance of each supported public rheedium PyTree."""
    beam: Float[Array, "..."] = rh.types.create_electron_beam(
        energy_kev=15.0,
        energy_spread_ev=0.2,
        angular_divergence_mrad=0.3,
        coherence_length_transverse_angstrom=450.0,
        coherence_length_longitudinal_angstrom=850.0,
        spot_size_um=jnp.array([120.0, 40.0]),
    )

    crystal: CrystalStructure = rh.types.create_crystal_structure(
        frac_positions=jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [0.5, 0.5, 0.5, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cart_positions=jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [2.715, 2.715, 2.715, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cell_lengths=jnp.array([5.43, 5.43, 5.43], dtype=jnp.float64),
        cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
    )

    ewald: Integer[Array, "..."] = rh.types.create_ewald_data(
        wavelength_ang=jnp.asarray(0.099, dtype=jnp.float64),
        k_magnitude=jnp.asarray(63.466518, dtype=jnp.float64),
        sphere_radius=jnp.asarray(63.466518, dtype=jnp.float64),
        recip_vectors=jnp.eye(3, dtype=jnp.float64),
        hkl_grid=jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32),
        g_vectors=jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=jnp.float64,
        ),
        g_magnitudes=jnp.array([0.0, 1.0], dtype=jnp.float64),
        structure_factors=jnp.array(
            [1.0 + 0.0j, 0.5 + 0.25j],
            dtype=jnp.complex128,
        ),
        intensities=jnp.array([1.0, 0.3125], dtype=jnp.float64),
    )

    potential: Float[Array, "..."] = rh.types.create_potential_slices(
        slices=jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4),
        slice_thickness=jnp.asarray(1.5, dtype=jnp.float64),
        x_calibration=jnp.asarray(0.2, dtype=jnp.float64),
        y_calibration=jnp.asarray(0.25, dtype=jnp.float64),
    )

    xyz: Integer[Array, "..."] = rh.types.create_xyz_data(
        positions=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
        lattice=jnp.eye(3, dtype=jnp.float64) * 6.0,
        stress=jnp.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=jnp.float64,
        ),
        energy=jnp.asarray(-14.25, dtype=jnp.float64),
        forces=jnp.array(
            [
                [0.01, -0.02, 0.03],
                [-0.01, 0.02, -0.03],
                [0.00, 0.00, 0.00],
            ],
            dtype=jnp.float64,
        ),
        properties=[
            {"name": "species", "columns": 1},
            {"name": "index", "columns": 1},
        ],
        comment="water-like example",
    )

    xyz_with_nones: Integer[Array, "..."] = XYZData(
        positions=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64),
        atomic_numbers=jnp.array([29], dtype=jnp.int32),
        lattice=None,
        stress=None,
        energy=None,
        forces=None,
        properties=None,
        comment=None,
    )

    pattern: RHEEDPattern = rh.types.create_rheed_pattern(
        g_indices=jnp.array([0, 1], dtype=jnp.int32),
        k_out=jnp.array(
            [[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]],
            dtype=jnp.float64,
        ),
        detector_points=jnp.array(
            [[10.0, 5.0], [20.0, 15.0]],
            dtype=jnp.float64,
        ),
        intensities=jnp.array([100.0, 80.0], dtype=jnp.float64),
    )

    image: RHEEDImage = rh.types.create_rheed_image(
        img_array=jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        incoming_angle=jnp.asarray(2.0, dtype=jnp.float64),
        calibration=jnp.array([0.02, 0.03], dtype=jnp.float64),
        electron_wavelength=jnp.asarray(0.099, dtype=jnp.float64),
        detector_distance=jnp.asarray(800.0, dtype=jnp.float64),
    )

    sliced: SlicedCrystal = rh.types.create_sliced_crystal(
        cart_positions=jnp.array(
            [
                [0.0, 0.0, 0.0, 14.0],
                [1.5, 1.5, 2.0, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cell_lengths=jnp.array([120.0, 120.0, 20.0], dtype=jnp.float64),
        cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
        orientation=jnp.array([1, 1, 1], dtype=jnp.int32),
        depth=jnp.asarray(20.0, dtype=jnp.float64),
        x_extent=jnp.asarray(120.0, dtype=jnp.float64),
        y_extent=jnp.asarray(120.0, dtype=jnp.float64),
    )

    surface_config: Float[Array, "..."] = SurfaceConfig(
        method="explicit",
        height_fraction=0.45,
        coordination_cutoff=2.8,
        coordination_threshold=7,
        n_layers=2,
        layer_tolerance=0.35,
        explicit_mask=jnp.array([True, False, True], dtype=bool),
    )

    detector_geometry: Any = DetectorGeometry(
        distance=250.0,
        tilt_angle=7.5,
        curvature_radius=float("inf"),
        center_offset_h=1.5,
        center_offset_v=-2.0,
        psf_sigma_pixels=0.4,
    )

    reconstruction: ReconResult = ReconResult(
        params={
            "beam": beam,
            "offsets": [
                jnp.asarray(1.25, dtype=jnp.float64),
                jnp.asarray(-0.5, dtype=jnp.float64),
            ],
            "config": {
                "surface": surface_config,
                "scale": jnp.asarray(2.0, dtype=jnp.float64),
            },
        },
        latent_params={
            "offsets": [
                jnp.asarray(0.5, dtype=jnp.float64),
                jnp.asarray(-0.2, dtype=jnp.float64),
            ],
            "scale_logit": jnp.asarray(1.0, dtype=jnp.float64),
        },
        simulated=jnp.array(
            [[2.0, 1.0], [0.5, 3.0]],
            dtype=jnp.float64,
        ),
        residual=jnp.array(
            [[0.1, -0.2], [0.0, 0.3]],
            dtype=jnp.float64,
        ),
        loss=jnp.asarray(0.035, dtype=jnp.float64),
        iterations=jnp.asarray(2, dtype=jnp.int32),
        converged=jnp.asarray(True),
        solver_status="successful",
    )

    return {
        "beam": beam,
        "crystal": crystal,
        "ewald": ewald,
        "potential": potential,
        "xyz": xyz,
        "xyz_with_nones": xyz_with_nones,
        "pattern": pattern,
        "image": image,
        "sliced": sliced,
        "surface_config": surface_config,
        "detector_geometry": detector_geometry,
        "reconstruction": reconstruction,
    }


class TestHdf5RoundTrip(chex.TestCase):
    """Round-trip tests for rheedium HDF5 serialization.

    :see: :func:`~rheedium.inout.load_from_h5`
    :see: :func:`~rheedium.inout.save_to_h5`
    """

    def test_save_and_load_all_supported_pytrees(self) -> None:
        r"""Every public rheedium PyTree should survive an HDF5 round-trip.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: Every public
        rheedium PyTree should survive an HDF5 round-trip.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The result is checked with direct unittest or Chex assertions against
        the expected contract.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_hdf5``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        payload: Any = _build_sample_pytrees()

        tmpdir: str
        with TemporaryDirectory() as tmpdir:
            path: Path = Path(tmpdir) / "pytrees.h5"

            rh.inout.save_to_h5(path, **payload)
            loaded: Any = rh.inout.load_from_h5(path)

        self.assertEqual(set(loaded), set(payload))
        key: Any
        expected: Any
        for key, expected in payload.items():
            _assert_round_trip_equal(loaded[key], expected)

    def test_load_single_named_object(self) -> None:
        r"""A single group can be loaded by name.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A single group can
        be loaded by name.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The existing assertions in the function body compare the observed
        result with the expected contract for this module.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_hdf5``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        payload: Any = _build_sample_pytrees()

        tmpdir: str
        with TemporaryDirectory() as tmpdir:
            path: Path = Path(tmpdir) / "single.h5"

            rh.inout.save_to_h5(path, **payload)
            loaded: Any = rh.inout.load_from_h5(path, name="reconstruction")

        _assert_round_trip_equal(loaded, payload["reconstruction"])

    def test_missing_file_raises_file_not_found(self) -> None:
        r"""A non-existent path raises FileNotFoundError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A non-existent
        path raises FileNotFoundError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_hdf5``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmpdir: str
        with TemporaryDirectory() as tmpdir:
            path: Path = Path(tmpdir) / "missing.h5"
            with pytest.raises(FileNotFoundError):
                rh.inout.load_from_h5(path)

    def test_corrupt_file_raises_runtime_error(self) -> None:
        r"""A present but non-HDF5 file raises RuntimeError.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: A present but
        non-HDF5 file raises RuntimeError.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        The negative path is validated by asserting the expected exception
        rather than accepting silent coercion or fallback behavior.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_hdf5``, so the Test Reference
        exposes both the guarantee and the implementation path.
        """
        tmpdir: str
        with TemporaryDirectory() as tmpdir:
            path: Path = Path(tmpdir) / "corrupt.h5"
            path.write_bytes(b"this is not a valid HDF5 file")
            with pytest.raises(RuntimeError, match="Failed to open HDF5"):
                rh.inout.load_from_h5(path)


class TestOccupancyHdf5RoundTrip(chex.TestCase):
    """HDF5 round trips preserve first-class occupancy fields.

    :see: :func:`~rheedium.inout.save_to_h5`
    :see: :func:`~rheedium.inout.load_from_h5`
    """

    def test_occupancies_round_trip_exactly(self) -> None:
        r"""Verify occupancy arrays round-trip bit-exactly through HDF5.

        Extended Summary
        ----------------
        Verifies the documented behavior for this test case: the
        ``occupancies`` fields of ``CrystalStructure`` and
        ``SlicedCrystal`` are dynamic registry children, so saving and
        loading preserves the arrays exactly, and a ``None`` occupancy
        field stays ``None``.

        Notes
        -----
        It constructs the representative inputs inside the test body, keeping
        the fixture and assertion path local to the documented case.

        Numerical expectations are checked with exact array-equality
        assertions because HDF5 stores the float64 payload losslessly.

        The documented check is rendered from
        ``tests.test_rheedium.test_inout.test_hdf5``, so the Test
        Reference exposes both the guarantee and the implementation path.
        """
        crystal: CrystalStructure = rh.types.create_crystal_structure(
            frac_positions=jnp.array(
                [[0.0, 0.0, 0.0, 14.0], [0.5, 0.5, 0.5, 8.0]],
                dtype=jnp.float64,
            ),
            cart_positions=jnp.array(
                [[0.0, 0.0, 0.0, 14.0], [2.715, 2.715, 2.715, 8.0]],
                dtype=jnp.float64,
            ),
            cell_lengths=jnp.array([5.43, 5.43, 5.43], dtype=jnp.float64),
            cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
            occupancies=jnp.array([0.9, 0.4], dtype=jnp.float64),
        )
        sliced: SlicedCrystal = rh.types.create_sliced_crystal(
            cart_positions=jnp.array(
                [[0.0, 0.0, 0.0, 14.0], [1.5, 1.5, 2.0, 8.0]],
                dtype=jnp.float64,
            ),
            cell_lengths=jnp.array([120.0, 120.0, 20.0], dtype=jnp.float64),
            cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=jnp.asarray(20.0, dtype=jnp.float64),
            x_extent=jnp.asarray(120.0, dtype=jnp.float64),
            y_extent=jnp.asarray(120.0, dtype=jnp.float64),
            occupancies=jnp.array([1.0, 0.25], dtype=jnp.float64),
        )
        bare_sliced: SlicedCrystal = rh.types.create_sliced_crystal(
            cart_positions=jnp.array(
                [[0.0, 0.0, 0.0, 14.0]],
                dtype=jnp.float64,
            ),
            cell_lengths=jnp.array([120.0, 120.0, 20.0], dtype=jnp.float64),
            cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
            orientation=jnp.array([0, 0, 1], dtype=jnp.int32),
            depth=jnp.asarray(20.0, dtype=jnp.float64),
            x_extent=jnp.asarray(120.0, dtype=jnp.float64),
            y_extent=jnp.asarray(120.0, dtype=jnp.float64),
        )
        with TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "occupancies.h5"
            rh.inout.save_to_h5(
                path,
                crystal=crystal,
                sliced=sliced,
                bare_sliced=bare_sliced,
            )
            loaded: dict[str, Any] = rh.inout.load_from_h5(path)
        np.testing.assert_array_equal(
            np.asarray(loaded["crystal"].occupancies),
            np.asarray(crystal.occupancies),
        )
        np.testing.assert_array_equal(
            np.asarray(loaded["sliced"].occupancies),
            np.asarray(sliced.occupancies),
        )
        self.assertIsNone(loaded["bare_sliced"].occupancies)
