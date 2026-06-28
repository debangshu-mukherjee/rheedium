"""Tests for the frozen recon inverse import surface."""

from pathlib import Path

import chex

import rheedium.types as rh_types
from rheedium import recon

_FROZEN_INVERSE_API: dict[str, str] = {
    "covariance_from_fisher": ":func:",
    "fisher_information_from_residual": ":func:",
    "fit_geometry_beam": ":func:",
    "laplace_inverse_mass_matrix": ":func:",
    "laplace_uncertainty": ":func:",
    "multistart": ":func:",
    "posterior_from_samples": ":func:",
    "recipe_deviation": ":func:",
    "recipe_deviation_report_payload": ":func:",
    "recipe_deviation_report_schema": ":func:",
    "reconstruct_distribution": ":func:",
    "sample_posterior": ":func:",
    "solve": ":func:",
    "validate_recipe_deviation_report": ":func:",
}
_FROZEN_INVERSE_TYPE_API: dict[str, str] = {
    "DistributionAxisSpec": ":class:",
    "LaplaceUncertainty": ":class:",
    "OrientationFitResult": ":class:",
    "PosteriorSamples": ":class:",
    "RECIPE_DEVIATION_SCHEMA_VERSION": ":obj:",
    "RecipeDeviationReport": ":class:",
    "ReconProblem": ":class:",
    "ReconResult": ":class:",
    "create_crystal_displacement_axis_spec": ":func:",
    "create_distribution_axis_spec": ":func:",
}


class TestReconApiFreeze(chex.TestCase):
    """Tests for the automaton-facing inverse API freeze.

    :see: :class:`~rheedium.types.ReconProblem`
    :see: :func:`~rheedium.recon.solve`
    :see: :func:`~rheedium.recon.fit_geometry_beam`
    """

    def test_frozen_inverse_api_is_exported_and_routine_listed(self) -> None:
        r"""The frozen inverse API should be importable and documented.

        Extended Summary
        ----------------
        Verifies the K6 API-freeze gate by checking that each
        automaton-facing inverse symbol is present in ``rheedium.recon``,
        exported through ``__all__``, and repeated in the package Routine
        Listings.

        Notes
        -----
        This intentionally freezes the downstream import contract without
        constraining unrelated recon helpers to a closed list.
        """
        package_doc: str = recon.__doc__ or ""
        for name, role in _FROZEN_INVERSE_API.items():
            self.assertIn(name, recon.__all__)
            self.assertTrue(hasattr(recon, name), msg=name)
            self.assertIn(f"{role}`{name}`", package_doc)

    def test_frozen_inverse_types_are_exported_from_types_only(self) -> None:
        r"""The frozen inverse carriers should live under ``rheedium.types``.

        Extended Summary
        ----------------
        Verifies the types-centralization gate by checking that inverse
        carriers and their constructors are public from ``rheedium.types`` and
        absent from ``rheedium.recon``.

        Notes
        -----
        This pairs the automaton-facing recon function freeze with the
        owner-only export rule for structured PyTree carriers.
        """
        types_doc: str = rh_types.__doc__ or ""
        for name, role in _FROZEN_INVERSE_TYPE_API.items():
            self.assertIn(name, rh_types.__all__)
            self.assertTrue(hasattr(rh_types, name), msg=name)
            self.assertIn(f"{role}`{name}`", types_doc)
            self.assertNotIn(name, recon.__all__)
            self.assertFalse(hasattr(recon, name), msg=name)

    def test_recon_package_is_marked_typed(self) -> None:
        r"""The recon package should carry a PEP 561 marker.

        Extended Summary
        ----------------
        Verifies the K6 typed-package gate for the frozen inverse API by
        checking the committed ``py.typed`` marker next to
        ``rheedium.recon.__file__``.

        Notes
        -----
        The marker lets downstream automatons and type checkers treat the
        public recon imports as typed package data.
        """
        recon_package: Path = Path(recon.__file__).parent
        self.assertTrue((recon_package / "py.typed").is_file())
