"""Differentiable crystal-defect generators.

Extended Summary
----------------
This module provides continuous defect models for modifying a
``CrystalStructure`` without changing its cell. The public APIs are
designed for inverse workflows that need gradients with respect to
occupancies, interstitial amplitudes, or species-mixing variables.

Routine Listings
----------------
:func:`apply_vacancy_field`
    Attenuate the effective scattering strength of existing sites with
    continuous occupancies.
:func:`apply_interstitial_field`
    Append candidate interstitial sites with continuous occupancies.
:func:`apply_antisite_field`
    Blend host and substituted species through complementary
    co-located site occupancies.

Notes
-----
Partial occupation is carried by the first-class ``occupancies`` field
of ``CrystalStructure``: each site's occupancy multiplies its atomic
form factor in every simulation kernel (``f_eff = occ * f_Z``), while
the atomic-number column stays integral. This keeps vacancy,
interstitial, and antisite models differentiable without requiring
hard atom deletion, discrete species swaps, or unphysical
"effective Z" encodings.

R5 return type: these APIs are sub-coherence structure modifiers. They return a
modified ``CrystalStructure`` directly, not a statistical ``Distribution``.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
)
from rheedium.ucell import build_cell_vectors


def _base_occupancies(crystal: CrystalStructure) -> Float[Array, "N_atoms"]:
    """Return the crystal's occupancies, defaulting to fully occupied."""
    if crystal.occupancies is None:
        return jnp.ones(crystal.cart_positions.shape[0], dtype=jnp.float64)
    return jnp.asarray(crystal.occupancies, dtype=jnp.float64)


def _assemble_crystal(
    frac_xyz: Float[Array, "N_atoms 3"],
    cart_xyz: Float[Array, "N_atoms 3"],
    atomic_numbers: Float[Array, "N_atoms"],
    occupancies: Float[Array, "N_atoms"],
    reference_crystal: CrystalStructure,
) -> CrystalStructure:
    """Create a CrystalStructure from coordinates, species, and occupancies."""
    frac_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [frac_xyz, atomic_numbers]
    )
    cart_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [cart_xyz, atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=reference_crystal.cell_lengths,
        cell_angles=reference_crystal.cell_angles,
        occupancies=occupancies,
    )


@jaxtyped(typechecker=beartype)
def apply_vacancy_field(
    crystal: CrystalStructure,
    site_occupancies: Float[Array, "N_atoms"],
) -> CrystalStructure:
    """Apply a continuous vacancy field to an existing crystal.

    :see: :class:`~.test_crystal_defects.TestApplyVacancyField`

    Parameters
    ----------
    crystal : CrystalStructure
        Reference crystal structure.
    site_occupancies : Float[Array, "N_atoms"]
        Continuous occupancy for each crystallographic site. Values are
        clipped to ``[0, 1]`` and multiply the crystal's existing
        per-site occupancies in the returned structure.

    Returns
    -------
    vacancy_modified_crystal : CrystalStructure
        Crystal with unchanged coordinates and atomic numbers, and
        vacancy-scaled ``occupancies``.

    Notes
    -----
    1. **Clip occupancies** --
       Restrict ``site_occupancies`` to the physical interval
       ``[0, 1]``.
    2. **Scale site occupancies** --
       Multiply the crystal's existing occupancies (ones when absent)
       by the clipped vacancy occupancies. The atomic-number column is
       untouched, so a 1% vacancy on silicon stays silicon.
    3. **Preserve geometry** --
       Return a new ``CrystalStructure`` with the same fractional and
       Cartesian coordinates as the input crystal.

    This uses a continuous occupancy-suppression model rather than hard
    atom deletion, so gradients with respect to occupancy remain
    well-defined: every simulation kernel multiplies each atom's form
    factor by its occupancy.
    """
    occupancies: Float[Array, "N_atoms"] = jnp.clip(
        jnp.asarray(site_occupancies, dtype=jnp.float64), 0.0, 1.0
    )
    new_occupancies: Float[Array, "N_atoms"] = (
        _base_occupancies(crystal) * occupancies
    )
    return _assemble_crystal(
        frac_xyz=crystal.frac_positions[:, :3],
        cart_xyz=crystal.cart_positions[:, :3],
        atomic_numbers=crystal.cart_positions[:, 3],
        occupancies=new_occupancies,
        reference_crystal=crystal,
    )


@jaxtyped(typechecker=beartype)
def apply_interstitial_field(
    crystal: CrystalStructure,
    interstitial_frac_positions: Float[Array, "N_interstitial 3"],
    interstitial_atomic_numbers: Float[Array, "N_interstitial"],
    interstitial_occupancies: Float[Array, "N_interstitial"],
) -> CrystalStructure:
    """Append continuously weighted interstitial sites to a crystal.

    :see: :class:`~.test_crystal_defects.TestApplyInterstitialField`

    Parameters
    ----------
    crystal : CrystalStructure
        Reference crystal structure.
    interstitial_frac_positions : Float[Array, "N_interstitial 3"]
        Candidate interstitial positions in fractional coordinates of
        the input cell.
    interstitial_atomic_numbers : Float[Array, "N_interstitial"]
        Atomic number assigned to each candidate interstitial species.
        Stored unchanged in the species column.
    interstitial_occupancies : Float[Array, "N_interstitial"]
        Continuous occupancy for each candidate interstitial. Values
        are clipped to ``[0, 1]`` and stored in the ``occupancies``
        field.

    Returns
    -------
    interstitial_modified_crystal : CrystalStructure
        Crystal structure with the original atoms preserved and the
        occupancy-weighted interstitial candidates appended.

    Notes
    -----
    1. **Clip occupancies** --
       Restrict interstitial occupancies to ``[0, 1]``.
    2. **Map to Cartesian coordinates** --
       Convert the candidate fractional positions with the input cell
       vectors.
    3. **Append sites** --
       Concatenate the original crystal positions with the interstitial
       candidates in both coordinate systems; concatenate the original
       occupancies (ones when absent) with the clipped interstitial
       occupancies. Atomic numbers stay integral for both populations.

    The interstitial sites are retained even at zero occupancy so the
    public contract stays shape-stable under ``jax.jit`` and
    ``jax.vmap``; a zero-occupancy site contributes exactly zero to
    every structure factor.
    """
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    clipped_occupancies: Float[Array, "N_interstitial"] = jnp.clip(
        jnp.asarray(interstitial_occupancies, dtype=jnp.float64), 0.0, 1.0
    )
    interstitial_z: Float[Array, "N_interstitial"] = jnp.asarray(
        interstitial_atomic_numbers, dtype=jnp.float64
    )
    interstitial_cart_positions: Float[Array, "N_interstitial 3"] = (
        jnp.asarray(interstitial_frac_positions, dtype=jnp.float64)
        @ cell_vectors
    )

    frac_xyz: Float[Array, "N_total 3"] = jnp.concatenate(
        [crystal.frac_positions[:, :3], interstitial_frac_positions], axis=0
    )
    cart_xyz: Float[Array, "N_total 3"] = jnp.concatenate(
        [crystal.cart_positions[:, :3], interstitial_cart_positions], axis=0
    )
    atomic_numbers: Float[Array, "N_total"] = jnp.concatenate(
        [crystal.cart_positions[:, 3], interstitial_z], axis=0
    )
    occupancies: Float[Array, "N_total"] = jnp.concatenate(
        [_base_occupancies(crystal), clipped_occupancies], axis=0
    )

    return _assemble_crystal(
        frac_xyz=frac_xyz,
        cart_xyz=cart_xyz,
        atomic_numbers=atomic_numbers,
        occupancies=occupancies,
        reference_crystal=crystal,
    )


@jaxtyped(typechecker=beartype)
def apply_antisite_field(
    crystal: CrystalStructure,
    site_mixing_fractions: Float[Array, "N_atoms"],
    replacement_atomic_numbers: Float[Array, "N_atoms"],
) -> CrystalStructure:
    """Blend host and substitute species with complementary occupancies.

    :see: :class:`~.test_crystal_defects.TestApplyAntisiteField`

    Parameters
    ----------
    crystal : CrystalStructure
        Reference crystal structure.
    site_mixing_fractions : Float[Array, "N_atoms"]
        Continuous substitute fraction for each site. Values are clipped
        to ``[0, 1]`` where ``0`` keeps the host species and ``1``
        applies the full substitute species.
    replacement_atomic_numbers : Float[Array, "N_atoms"]
        Atomic number of the substitute species at each site.

    Returns
    -------
    antisite_modified_crystal : CrystalStructure
        Crystal with ``2 N_atoms`` sites: each original site is
        represented by two co-located sites — the host species at
        occupancy ``occ * (1 - f)`` and the substitute species at
        occupancy ``occ * f`` — so the site scatters as
        ``(1 - f) * f_host(q) + f * f_substitute(q)``.

    Notes
    -----
    1. **Clip mixing fractions** --
       Restrict site fractions to ``[0, 1]``.
    2. **Split each site into two co-located sites** --
       The first ``N_atoms`` rows keep the host atomic numbers with
       occupancy ``occ * (1 - f)``; the following ``N_atoms`` rows
       carry the substitute atomic numbers at the same coordinates
       with occupancy ``occ * f``. Atomic numbers stay integral.
    3. **Preserve geometry** --
       Coordinates and cell are unchanged; only the species column of
       the appended rows and the ``occupancies`` field differ.

    This represents the antisite exactly in the kinematic model —
    the summed scattering amplitude of the pair is
    ``(1 - f) f_host + f f_sub`` — instead of the former
    virtual-crystal "effective Z" blend, which truncated to a
    different element entirely.
    """
    mixing_fractions: Float[Array, "N_atoms"] = jnp.clip(
        jnp.asarray(site_mixing_fractions, dtype=jnp.float64), 0.0, 1.0
    )
    substitute_atomic_numbers: Float[Array, "N_atoms"] = jnp.asarray(
        replacement_atomic_numbers, dtype=jnp.float64
    )
    host_atomic_numbers: Float[Array, "N_atoms"] = crystal.cart_positions[:, 3]
    base_occupancies: Float[Array, "N_atoms"] = _base_occupancies(crystal)
    host_occupancies: Float[Array, "N_atoms"] = base_occupancies * (
        1.0 - mixing_fractions
    )
    substitute_occupancies: Float[Array, "N_atoms"] = (
        base_occupancies * mixing_fractions
    )

    frac_xyz: Float[Array, "N2 3"] = jnp.concatenate(
        [crystal.frac_positions[:, :3], crystal.frac_positions[:, :3]],
        axis=0,
    )
    cart_xyz: Float[Array, "N2 3"] = jnp.concatenate(
        [crystal.cart_positions[:, :3], crystal.cart_positions[:, :3]],
        axis=0,
    )
    atomic_numbers: Float[Array, "N2"] = jnp.concatenate(
        [host_atomic_numbers, substitute_atomic_numbers], axis=0
    )
    occupancies: Float[Array, "N2"] = jnp.concatenate(
        [host_occupancies, substitute_occupancies], axis=0
    )
    return _assemble_crystal(
        frac_xyz=frac_xyz,
        cart_xyz=cart_xyz,
        atomic_numbers=atomic_numbers,
        occupancies=occupancies,
        reference_crystal=crystal,
    )


__all__: list[str] = [
    "apply_antisite_field",
    "apply_interstitial_field",
    "apply_vacancy_field",
]
