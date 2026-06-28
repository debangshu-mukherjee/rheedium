"""Tests for recon/solve.py.

Verifies the general optimistix/optax reconstruction surface on lightweight
synthetic inverse problems with deterministic linear forward models.
"""

import os
import tempfile
import time

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import rheedium.types as rh_types
from rheedium import recon
from rheedium.recon import (
    build_incoherent_intensity_library,
    fit_geometry_beam,
    multistart,
    reconstruct_distribution,
    reconstruct_incoherent_weights,
    solve,
)
from rheedium.types import (
    BeamModeDistribution,
    CrystalStructure,
    DistributionAxisSpec,
    ReconProblem,
    ReconResult,
    create_crystal_displacement_axis_spec,
    create_crystal_structure,
    create_distribution_axis_spec,
)

_LINEAR_MATRIX: Float[Array, "pixels params"] = jnp.array(
    [[2.0, 1.0], [1.0, -1.0], [0.5, 2.0]],
    dtype=jnp.float64,
)


def _linear_forward(params: Float[Array, "params"]) -> Float[Array, "pixels"]:
    """Map a two-parameter vector to synthetic detector pixels."""
    pixels: Float[Array, "pixels"] = _LINEAR_MATRIX @ params
    return pixels


def _linear_problem() -> tuple[ReconProblem, Float[Array, "params"]]:
    """Return a well-conditioned synthetic reconstruction problem."""
    true_params: Float[Array, "params"] = jnp.array(
        [1.25, -0.5],
        dtype=jnp.float64,
    )
    measured: Float[Array, "pixels"] = _linear_forward(true_params)
    problem: ReconProblem = ReconProblem(
        forward=_linear_forward,
        measured=measured,
    )
    return problem, true_params


def _parametric_spread_weights(
    samples: Float[Array, "samples"],
    width: Float[Array, ""],
) -> Float[Array, "samples"]:
    """Return normalized Gaussian-like weights on fixed samples."""
    logits: Float[Array, "samples"] = -0.5 * (samples / width) ** 2
    weights: Float[Array, "samples"] = jax.nn.softmax(logits)
    return weights


def _identity_forward(
    params: Float[Array, "params"],
) -> Float[Array, "params"]:
    """Return parameters unchanged as a synthetic forward output."""
    output: Float[Array, "params"] = params
    return output


_GEOMETRY_BEAM_MATRIX: Float[Array, "pixels params"] = jnp.array(
    [
        [1.0, 0.0, 2.0, -0.5],
        [0.0, 1.0, -1.0, 1.5],
        [1.5, -0.5, 0.25, 0.75],
        [-0.25, 1.25, 1.0, 0.5],
        [0.75, 0.5, -0.5, 1.0],
    ],
    dtype=jnp.float64,
)


def _geometry_beam_transform(
    latent: Float[Array, "params"],
) -> tuple[Float[Array, "orientation"], BeamModeDistribution]:
    """Map synthetic latent coordinates to orientation and beam modes."""
    orientation: Float[Array, "orientation"] = latent[:2]
    beam_modes: BeamModeDistribution = BeamModeDistribution(
        beta_in_plane=jnp.asarray(0.0, dtype=jnp.float64),
        beta_out_of_plane=jnp.asarray(0.0, dtype=jnp.float64),
        divergence_in_plane_rad=latent[2],
        divergence_out_of_plane_rad=latent[3],
        energy_spread_ev=jnp.asarray(0.0, dtype=jnp.float64),
        distribution_id="synthetic_geometry_beam",
    )
    return orientation, beam_modes


def _geometry_beam_forward(
    crystal: CrystalStructure,
    orientation: Float[Array, "orientation"],
    beam_modes: BeamModeDistribution,
) -> Float[Array, "pixels"]:
    """Return a linear detector fixture from orientation and beam modes."""
    del crystal
    parameter_vector: Float[Array, "params"] = jnp.array(
        [
            orientation[0],
            orientation[1],
            beam_modes.divergence_in_plane_rad,
            beam_modes.divergence_out_of_plane_rad,
        ],
        dtype=jnp.float64,
    )
    pixels: Float[Array, "pixels"] = _GEOMETRY_BEAM_MATRIX @ parameter_vector
    return pixels


def _double_well_loss(
    simulated: Float[Array, "params"],
    measured: Float[Array, "params"],  # noqa: ARG001
) -> Float[Array, ""]:
    """Return an asymmetric double-well objective with a local minimum."""
    x: Float[Array, ""] = jnp.ravel(simulated)[0]
    loss: Float[Array, ""] = (x**2 - 1.0) ** 2 + 0.2 * x
    return loss


def _small_crystal() -> CrystalStructure:
    """Return a two-atom cubic crystal carrier for recon fixtures."""
    crystal: CrystalStructure = create_crystal_structure(
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
                [2.0, 2.0, 2.0, 14.0],
            ],
            dtype=jnp.float64,
        ),
        cell_lengths=jnp.array([4.0, 4.0, 4.0], dtype=jnp.float64),
        cell_angles=jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64),
    )
    return crystal


def _displacement_modes() -> Float[Array, "modes atoms xyz"]:
    """Return one asymmetric Cartesian displacement mode."""
    modes: Float[Array, "modes atoms xyz"] = jnp.array(
        [
            [
                [0.7, 0.1, 0.2],
                [-0.2, 0.6, 0.9],
            ],
        ],
        dtype=jnp.float64,
    )
    return modes


def _crystal_amplitude_forward(
    crystal: CrystalStructure,
) -> Complex[Array, "rows cols"]:
    """Map a crystal carrier to a small differentiable amplitude image."""
    q_vectors: Float[Array, "pixels xyz"] = jnp.array(
        [
            [0.4, 0.1, 0.2],
            [0.1, 0.6, -0.2],
            [-0.3, 0.2, 0.7],
            [0.5, -0.4, 0.3],
        ],
        dtype=jnp.float64,
    )
    positions: Float[Array, "atoms xyz"] = crystal.cart_positions[:, :3]
    atomic_numbers: Float[Array, "atoms"] = crystal.cart_positions[:, 3]
    phases: Float[Array, "pixels atoms"] = q_vectors @ positions.T
    amplitudes: Complex[Array, "pixels"] = jnp.sum(
        atomic_numbers * jnp.exp(1j * phases),
        axis=1,
    )
    image: Complex[Array, "rows cols"] = jnp.reshape(amplitudes, (2, 2))
    return image


def _crystal_intensity_forward(
    crystal: CrystalStructure,
) -> Float[Array, "rows cols"]:
    """Map a crystal carrier to a real intensity image."""
    amplitudes: Complex[Array, "rows cols"] = _crystal_amplitude_forward(
        crystal
    )
    intensity: Float[Array, "rows cols"] = jnp.real(
        jnp.conj(amplitudes) * amplitudes
    )
    return intensity


class TestReconSolve(chex.TestCase):
    """Tests for the general reconstruction solver.

    :see: :class:`~rheedium.types.ReconProblem`
    :see: :func:`~rheedium.recon.multistart`
    :see: :func:`~rheedium.recon.solve`
    """

    def test_least_squares_solve_recovers_linear_parameters(self) -> None:
        r"""Least-squares solve should recover planted linear parameters.

        Extended Summary
        ----------------
        Verifies that the default Levenberg-Marquardt path solves a synthetic
        overdetermined linear inverse problem to tight tolerance.

        Notes
        -----
        It initializes from zero, solves against exact synthetic data, and
        checks final parameters, loss, convergence, and finite residuals.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()

        result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.zeros(2, dtype=jnp.float64),
            max_steps=32,
            rtol=1e-10,
            atol=1e-10,
        )

        self.assertIsInstance(result, ReconResult)
        self.assertTrue(bool(result.converged))
        chex.assert_trees_all_close(result.params, true_params, atol=1e-8)
        chex.assert_trees_all_close(result.loss, 0.0, atol=1e-12)
        chex.assert_tree_all_finite(result.residual)

    def test_optax_adamw_solve_reduces_scalar_loss(self) -> None:
        r"""Optax-backed solve should reduce the synthetic image loss.

        Extended Summary
        ----------------
        Verifies that the first-order path wraps an optax gradient
        transformation through the same public solve surface.

        Notes
        -----
        It compares the final AdamW loss against the initial loss rather than
        requiring exact convergence from a fixed small iteration budget.
        """
        problem: ReconProblem
        _true_params: Float[Array, "params"]
        problem, _true_params = _linear_problem()
        initial_latent: Float[Array, "params"] = jnp.array(
            [-2.0, 2.0],
            dtype=jnp.float64,
        )
        initial_residual: Float[Array, "pixels"] = (
            problem.residual_from_latent(initial_latent)
        )
        initial_loss: Float[Array, ""] = jnp.mean(initial_residual**2)

        result: ReconResult = solve(
            problem=problem,
            initial_latent=initial_latent,
            mode="adamw",
            max_steps=300,
            learning_rate=0.05,
            atol=1e-8,
        )

        self.assertLess(float(result.loss), float(initial_loss) * 1e-2)
        chex.assert_tree_all_finite(result.params)

    def test_multistart_returns_lowest_loss_result(self) -> None:
        r"""Multistart should choose the start with the best final loss.

        Extended Summary
        ----------------
        Verifies deterministic best-result selection across a leading start
        axis of initial latent guesses.

        Notes
        -----
        It includes the exact solution as one start and checks that the
        returned result has essentially zero loss.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()
        starts: Float[Array, "starts params"] = jnp.stack(
            [
                jnp.array([-10.0, 10.0], dtype=jnp.float64),
                true_params,
            ]
        )

        result: ReconResult = multistart(
            problem=problem,
            initial_latents=starts,
            max_steps=8,
            atol=1e-12,
        )

        chex.assert_trees_all_close(result.params, true_params, atol=1e-10)
        chex.assert_trees_all_close(result.loss, 0.0, atol=1e-12)

    def test_seeded_random_multistart_is_reproducible(self) -> None:
        r"""Seeded random multistart should be reproducible.

        Extended Summary
        ----------------
        Verifies the K3 generated-start path: a template latent plus a PRNG key
        expands into deterministic random starts, and repeated calls with the
        same seed return the same best reconstruction.

        Notes
        -----
        The linear reference problem is exactly recoverable from every basin,
        which keeps this test focused on seeded generation and deterministic
        selection.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()
        key: Array = jax.random.PRNGKey(101)
        template: Float[Array, "params"] = jnp.zeros(
            2,
            dtype=jnp.float64,
        )

        first_result: ReconResult = multistart(
            problem=problem,
            initial_latents=template,
            key=key,
            n_starts=5,
            random_scale=3.0,
            max_steps=16,
            atol=1e-10,
        )
        second_result: ReconResult = multistart(
            problem=problem,
            initial_latents=template,
            key=key,
            n_starts=5,
            random_scale=3.0,
            max_steps=16,
            atol=1e-10,
        )

        chex.assert_trees_all_close(
            first_result.params, true_params, atol=1e-8
        )
        chex.assert_trees_all_close(
            first_result.params,
            second_result.params,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            first_result.loss,
            second_result.loss,
            atol=1e-12,
        )

    def test_multistart_escapes_planted_local_minimum(self) -> None:
        r"""Multistart should escape a planted local objective basin.

        Extended Summary
        ----------------
        Verifies the K3 robustness gate on an asymmetric double-well scalar
        objective: a single cold start lands in the higher-loss positive basin,
        while multistart finds the lower-loss negative basin.

        Notes
        -----
        The problem uses the finite-budget AdamW scalar minimization mode so
        the positive local basin is not crossed by a single cold start, and
        best-start selection is based on ``ReconResult.loss``.
        """
        problem: ReconProblem = ReconProblem(
            forward=_identity_forward,
            measured=jnp.zeros(1, dtype=jnp.float64),
            loss_fn=_double_well_loss,
        )

        single_result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.array([2.0], dtype=jnp.float64),
            mode="adamw",
            max_steps=100,
            learning_rate=0.02,
            atol=1e-10,
        )
        starts: Float[Array, "starts params"] = jnp.array(
            [[2.0], [-2.0]],
            dtype=jnp.float64,
        )
        multistart_result: ReconResult = multistart(
            problem=problem,
            initial_latents=starts,
            mode="adamw",
            max_steps=100,
            learning_rate=0.02,
            atol=1e-10,
        )

        self.assertGreater(float(single_result.params[0]), 0.5)
        self.assertLess(float(multistart_result.params[0]), -0.5)
        self.assertLess(
            float(multistart_result.loss), float(single_result.loss)
        )

    def test_bracketed_initialization_converges_in_fewer_steps(self) -> None:
        r"""A bracketed start should refine faster than a cold start.

        Extended Summary
        ----------------
        Verifies the Loop-B to Loop-C handoff contract for K3: when a coarse
        bracket already places the latent near the correct basin, the same
        nonlinear least-squares solve reaches the planted solution in fewer
        reported solver steps than a distant cold start.

        Notes
        -----
        The synthetic residual ``x**2 - 4`` is nonlinear but exactly
        identifiable, so the iteration comparison is stable.
        """
        problem: ReconProblem = ReconProblem(
            forward=lambda latent: latent**2,
            measured=jnp.array([4.0], dtype=jnp.float64),
        )

        cold_result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.array([10.0], dtype=jnp.float64),
            max_steps=64,
            atol=1e-10,
            rtol=1e-10,
        )
        bracketed_result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.array([2.2], dtype=jnp.float64),
            max_steps=64,
            atol=1e-10,
            rtol=1e-10,
        )

        chex.assert_trees_all_close(
            bracketed_result.params,
            jnp.array([2.0], dtype=jnp.float64),
            atol=1e-8,
        )
        self.assertLess(
            int(bracketed_result.iterations),
            int(cold_result.iterations),
        )

    def test_fixed_budget_matches_longer_solve_with_wall_clock_budget(
        self,
    ) -> None:
        r"""Fixed-step rapid solve should match the longer convergence solve.

        Extended Summary
        ----------------
        Verifies the K2 rapid-path gate on the reference synthetic problem:
        after a warm-up call, a short fixed step budget recovers the same
        parameters as a longer solve while staying inside a generous wall-clock
        budget.

        Notes
        -----
        The timing assertion measures the warmed-cache reference problem rather
        than first compilation, keeping the budget meaningful but not brittle.
        """
        problem: ReconProblem
        true_params: Float[Array, "params"]
        problem, true_params = _linear_problem()
        initial_latent: Float[Array, "params"] = jnp.zeros(
            2,
            dtype=jnp.float64,
        )

        _warmup: ReconResult = solve(
            problem=problem,
            initial_latent=initial_latent,
            max_steps=6,
            atol=1e-10,
        )
        start_time: float = time.perf_counter()
        fixed_result: ReconResult = solve(
            problem=problem,
            initial_latent=initial_latent,
            max_steps=6,
            atol=1e-10,
        )
        elapsed_seconds: float = time.perf_counter() - start_time
        longer_result: ReconResult = solve(
            problem=problem,
            initial_latent=initial_latent,
            max_steps=32,
            atol=1e-10,
        )

        self.assertLess(elapsed_seconds, 10.0)
        chex.assert_trees_all_close(
            fixed_result.params,
            longer_result.params,
            atol=1e-9,
        )
        chex.assert_trees_all_close(
            fixed_result.params,
            true_params,
            atol=1e-8,
        )

    def test_solve_enables_requested_compilation_cache(self) -> None:
        r"""Solve should wire inversion runs through the XLA cache helper.

        Extended Summary
        ----------------
        Verifies the K6 warm-cache path by passing a compilation-cache
        directory directly to ``solve`` and checking that JAX sees the
        requested persistent cache before the optimizer path runs.

        Notes
        -----
        The test restores the previous cache directory when one was already
        configured, keeping the global JAX setting contained.
        """
        problem: ReconProblem
        _true_params: Float[Array, "params"]
        problem, _true_params = _linear_problem()
        saved_cache_dir: str | None = jax.config.jax_compilation_cache_dir
        cache_dir: str = tempfile.mkdtemp(prefix="rheedium-solve-cache-")

        try:
            result: ReconResult = solve(
                problem=problem,
                initial_latent=jnp.zeros(2, dtype=jnp.float64),
                max_steps=6,
                atol=1e-10,
                compilation_cache_dir=cache_dir,
                compilation_cache_per_arch=False,
            )

            self.assertTrue(bool(result.converged))
            self.assertEqual(
                jax.config.jax_compilation_cache_dir,
                os.path.abspath(cache_dir),
            )
        finally:
            jax.config.update(
                "jax_compilation_cache_dir",
                saved_cache_dir or cache_dir,
            )

    def test_reconstruct_incoherent_weights_recovers_planted_shape(
        self,
    ) -> None:
        r"""Incoherent weight fast path should recover planted weights.

        Extended Summary
        ----------------
        Verifies the convex linear distribution-reconstruction path for
        incoherent intensity sums.

        Notes
        -----
        It builds a small independent intensity library, mixes it with known
        simplex weights, and solves the regularized normal equations.
        """
        intensity_library: Float[Array, "samples rows cols"] = jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.5]],
                [[0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 1.0]],
            ],
            dtype=jnp.float64,
        )
        true_weights: Float[Array, "samples"] = jnp.array(
            [0.2, 0.5, 0.3],
            dtype=jnp.float64,
        )
        measured: Float[Array, "rows cols"] = jnp.einsum(
            "n,nhw->hw",
            true_weights,
            intensity_library,
        )

        weights: Float[Array, "samples"] = reconstruct_incoherent_weights(
            intensity_library=intensity_library,
            measured_image=measured,
            ridge=1e-12,
        )

        chex.assert_trees_all_close(weights, true_weights, atol=1e-8)

    def test_solve_recovers_parametric_distribution_spread(self) -> None:
        r"""Solve should recover a planted parametric distribution spread.

        Extended Summary
        ----------------
        Verifies the K2 distribution-reconstruction branch that is not the
        convex free-form fast path: a single positive latent controls a smooth
        fixed-support distribution, and ``solve`` recovers the planted spread
        from synthetic measured weights.

        Notes
        -----
        The support is intentionally lightweight and exactly identifiable,
        keeping this as a solver contract test rather than a physics benchmark.
        """
        samples: Float[Array, "samples"] = jnp.linspace(
            -1.0,
            1.0,
            5,
            dtype=jnp.float64,
        )
        true_width: Float[Array, ""] = jnp.asarray(0.35, dtype=jnp.float64)
        measured: Float[Array, "samples"] = _parametric_spread_weights(
            samples,
            true_width,
        )

        def transform(
            latent: Float[Array, "params"],
        ) -> dict[str, Float[Array, ""]]:
            width: Float[Array, ""] = recon.positive_from_unconstrained(
                latent[0],
                minimum=0.05,
            )
            params: dict[str, Float[Array, ""]] = {"width": width}
            return params

        def forward(
            params: dict[str, Float[Array, ""]],
        ) -> Float[Array, "samples"]:
            weights: Float[Array, "samples"] = _parametric_spread_weights(
                samples,
                params["width"],
            )
            return weights

        problem: ReconProblem = ReconProblem(
            forward=forward,
            measured=measured,
            transform=transform,
        )
        result: ReconResult = solve(
            problem=problem,
            initial_latent=jnp.array([0.7], dtype=jnp.float64),
            max_steps=64,
            atol=1e-10,
            rtol=1e-10,
        )

        chex.assert_trees_all_close(
            result.params["width"],
            true_width,
            atol=1e-6,
        )
        chex.assert_trees_all_close(result.simulated, measured, atol=1e-8)


def _amplitude_templates() -> Float[Array, "samples rows cols"]:
    """Return independent synthetic amplitude templates."""
    templates: Float[Array, "samples rows cols"] = jnp.array(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 1.5], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    return templates


def _one_hot_perturbation(
    base_templates: Float[Array, "samples rows cols"],
    sample: Float[Array, "samples"],
) -> Float[Array, "rows cols"]:
    """Select one amplitude template from a one-hot sample coordinate."""
    amplitude: Float[Array, "rows cols"] = jnp.einsum(
        "n,nhw->hw",
        sample,
        base_templates,
    )
    return amplitude


def _identity_forward(
    amplitude: Float[Array, "rows cols"],
) -> Float[Array, "rows cols"]:
    """Return a synthetic coherent amplitude image unchanged."""
    return amplitude


class TestGeometryBeamFit(chex.TestCase):
    """Tests for the geometry/beam convenience wrapper.

    :see: :func:`~rheedium.recon.fit_geometry_beam`
    """

    def test_fit_geometry_beam_recovers_planted_linear_fixture(self) -> None:
        r"""The convenience wrapper should recover orientation and beam params.

        Extended Summary
        ----------------
        Verifies that ``fit_geometry_beam`` builds a shared
        :class:`ReconProblem`, solves a calibrated forward closure, and returns
        a finite Laplace covariance in the physical parameter basis.

        Notes
        -----
        The fixture uses a fixed crystal carrier and a planted linear
        orientation/beam detector map so recovery is deterministic and cheap.
        """
        crystal: CrystalStructure = _small_crystal()
        true_latent: Float[Array, "params"] = jnp.array(
            [1.25, -0.4, 0.2, 0.35],
            dtype=jnp.float64,
        )
        true_orientation: Float[Array, "orientation"]
        true_beam: BeamModeDistribution
        true_orientation, true_beam = _geometry_beam_transform(true_latent)
        measured: Float[Array, "pixels"] = _geometry_beam_forward(
            crystal,
            true_orientation,
            true_beam,
        )

        orientation: Float[Array, "orientation"]
        beam_modes: BeamModeDistribution
        covariance: Float[Array, "cov cov"]
        orientation, beam_modes, covariance = fit_geometry_beam(
            crystal=crystal,
            measured=measured,
            forward=_geometry_beam_forward,
            initial_latent=jnp.zeros(4, dtype=jnp.float64),
            transform=_geometry_beam_transform,
            max_steps=32,
            atol=1e-10,
            uncertainty_regularization=1e-5,
        )

        chex.assert_trees_all_close(
            orientation,
            true_orientation,
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            beam_modes.divergence_in_plane_rad,
            true_beam.divergence_in_plane_rad,
            atol=1e-8,
        )
        chex.assert_trees_all_close(
            beam_modes.divergence_out_of_plane_rad,
            true_beam.divergence_out_of_plane_rad,
            atol=1e-8,
        )
        self.assertEqual(covariance.shape, (7, 7))
        chex.assert_tree_all_finite(covariance)


class TestReconDistributionReconstruction(chex.TestCase):
    """Tests for base-object distribution reconstruction.

    :see: :class:`~rheedium.types.DistributionAxisSpec`
    :see: :func:`~rheedium.types.create_distribution_axis_spec`
    :see: :func:`~rheedium.recon.build_incoherent_intensity_library`
    :see: :func:`~rheedium.recon.reconstruct_distribution`
    """

    def test_reconstruct_distribution_recovers_planted_axis_weights(
        self,
    ) -> None:
        r"""Base-axis reconstruction should recover planted weights.

        Extended Summary
        ----------------
        Verifies the updated plan's library-builder path: a base object and
        perturbation-axis specification build an incoherent intensity library,
        then the convex solver recovers the planted mixing distribution.

        Notes
        -----
        It uses one-hot axis samples to select independent amplitude templates,
        mixes their intensities with known weights, and checks the recovered
        distribution plus its one-sigma band.
        """
        samples: Float[Array, "samples sample_dim"] = jnp.eye(
            3,
            dtype=jnp.float64,
        )
        axis_spec: DistributionAxisSpec = create_distribution_axis_spec(
            samples=samples,
            perturbation_fn=_one_hot_perturbation,
            forward_model=_identity_forward,
            output_kind="amplitude",
            axis_id="synthetic_axis",
        )
        base_templates: Float[Array, "samples rows cols"] = (
            _amplitude_templates()
        )
        intensity_library: Float[Array, "samples rows cols"] = (
            build_incoherent_intensity_library(
                base_object=base_templates,
                axis_spec=axis_spec,
            )
        )
        true_weights: Float[Array, "samples"] = jnp.array(
            [0.2, 0.5, 0.3],
            dtype=jnp.float64,
        )
        measured: Float[Array, "rows cols"] = jnp.einsum(
            "n,nhw->hw",
            true_weights,
            intensity_library,
        )

        distribution, band = reconstruct_distribution(
            measured_image=measured,
            base_object=base_templates,
            axis_spec=axis_spec,
            ridge=1e-12,
            noise_variance=0.05,
        )

        chex.assert_trees_all_close(
            distribution.weights,
            true_weights,
            atol=1e-8,
        )
        chex.assert_trees_all_close(distribution.samples, samples, atol=1e-12)
        self.assertEqual(distribution.axis_id, "synthetic_axis")
        chex.assert_shape(band, (3,))
        chex.assert_tree_all_finite(band)

    def test_crystal_displacement_axis_recovers_planted_weights(self) -> None:
        r"""Crystal-backed displacement axis should recover planted weights.

        Extended Summary
        ----------------
        Verifies the K2 physical-carrier gate by instantiating an axis
        specification on a real :class:`rheedium.types.CrystalStructure`,
        building its incoherent intensity library, and recovering the planted
        displacement mixing distribution.

        Notes
        -----
        The forward model is a compact differentiable scattering surrogate, so
        the test exercises the carrier and reconstruction path without paying
        for a full detector simulation.
        """
        crystal: CrystalStructure = _small_crystal()
        samples: Float[Array, "samples modes"] = jnp.array(
            [[0.0], [0.15], [0.35]],
            dtype=jnp.float64,
        )
        axis_spec: DistributionAxisSpec = (
            create_crystal_displacement_axis_spec(
                samples=samples,
                displacement_modes=_displacement_modes(),
                forward_model=_crystal_amplitude_forward,
                output_kind="amplitude",
                axis_id="crystal_displacement",
            )
        )
        intensity_library: Float[Array, "samples rows cols"] = (
            build_incoherent_intensity_library(
                base_object=crystal,
                axis_spec=axis_spec,
            )
        )
        true_weights: Float[Array, "samples"] = jnp.array(
            [0.2, 0.55, 0.25],
            dtype=jnp.float64,
        )
        measured: Float[Array, "rows cols"] = jnp.einsum(
            "n,nhw->hw",
            true_weights,
            intensity_library,
        )

        distribution, band = reconstruct_distribution(
            measured_image=measured,
            base_object=crystal,
            axis_spec=axis_spec,
            ridge=1e-12,
            noise_variance=0.05,
        )

        chex.assert_trees_all_close(
            distribution.weights,
            true_weights,
            atol=1e-8,
        )
        self.assertEqual(distribution.axis_id, "crystal_displacement")
        chex.assert_tree_all_finite(band)


class TestReconForwardGradientGate(chex.TestCase):
    """Tests for finite gradients through crystal-backed recon paths.

    :see: :class:`~rheedium.types.ReconProblem`
    """

    def test_crystal_backed_problem_has_finite_latent_gradient(self) -> None:
        r"""A representative crystal-backed recon loss should differentiate.

        Extended Summary
        ----------------
        Verifies the universal finite-gradient gate across the new transform
        layer, a physical ``CrystalStructure`` carrier, and the public
        :class:`rheedium.types.ReconProblem` loss surface.

        Notes
        -----
        The latent coordinate is bounded into a displacement amplitude before
        perturbing the crystal, mirroring the structure-inversion path used by
        local solvers.
        """
        crystal: CrystalStructure = _small_crystal()
        axis_spec: DistributionAxisSpec = (
            create_crystal_displacement_axis_spec(
                samples=jnp.array([[0.0]], dtype=jnp.float64),
                displacement_modes=_displacement_modes(),
                forward_model=_crystal_intensity_forward,
                output_kind="intensity",
            )
        )
        target_crystal: CrystalStructure = axis_spec.perturbation_fn(
            crystal,
            jnp.array([0.12], dtype=jnp.float64),
        )
        measured: Float[Array, "rows cols"] = _crystal_intensity_forward(
            target_crystal
        )

        def transform(latent: Float[Array, "params"]) -> CrystalStructure:
            amplitude: Float[Array, ""] = recon.bounded_from_unconstrained(
                latent[0],
                -0.1,
                0.3,
            )
            perturbed: CrystalStructure = axis_spec.perturbation_fn(
                crystal,
                jnp.array([amplitude]),
            )
            return perturbed

        problem: ReconProblem = ReconProblem(
            forward=_crystal_intensity_forward,
            measured=measured,
            transform=transform,
        )
        latent: Float[Array, "params"] = jnp.array(
            [0.2],
            dtype=jnp.float64,
        )
        gradient: Float[Array, "params"] = jax.grad(problem.loss_from_latent)(
            latent
        )

        chex.assert_tree_all_finite(gradient)


class TestReconSolveNamespace(chex.TestCase):
    """Tests for public solve exports."""

    def test_namespace_exports_solve_entry_points(self) -> None:
        r"""Solve APIs should be re-exported from rheedium.recon.

        Extended Summary
        ----------------
        Verifies that the package-level namespace exposes the new solver
        surface documented by the reconstruction optimization plan.

        Notes
        -----
        It checks object identity between direct imports and attributes on
        ``rheedium.recon``.
        """
        self.assertFalse(hasattr(recon, "ReconProblem"))
        self.assertFalse(hasattr(recon, "ReconResult"))
        self.assertIs(rh_types.ReconProblem, ReconProblem)
        self.assertIs(rh_types.ReconResult, ReconResult)
        self.assertIs(recon.solve, solve)
        self.assertIs(recon.multistart, multistart)
        self.assertFalse(hasattr(recon, "create_distribution_axis_spec"))
        self.assertFalse(
            hasattr(recon, "create_crystal_displacement_axis_spec")
        )
        self.assertIs(
            rh_types.create_distribution_axis_spec,
            create_distribution_axis_spec,
        )
        self.assertIs(
            rh_types.create_crystal_displacement_axis_spec,
            create_crystal_displacement_axis_spec,
        )
        self.assertIs(
            recon.build_incoherent_intensity_library,
            build_incoherent_intensity_library,
        )
        self.assertIs(recon.reconstruct_distribution, reconstruct_distribution)
