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
    Blend host and substituted species through continuous mixing
    fractions.

Notes
-----
The fourth column of the ``CrystalStructure`` position arrays is used as
an effective atomic number or scattering weight. This lets vacancy,
interstitial, and antisite models stay differentiable without requiring
hard atom deletion or discrete species swaps.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
)
from rheedium.ucell import build_cell_vectors


def _assemble_crystal(
    frac_xyz: Float[Array, "N_atoms 3"],
    cart_xyz: Float[Array, "N_atoms 3"],
    effective_atomic_numbers: Float[Array, "N_atoms"],
    reference_crystal: CrystalStructure,
) -> CrystalStructure:
    """Create a CrystalStructure from updated coordinates and weights."""
    frac_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [frac_xyz, effective_atomic_numbers]
    )
    cart_positions: Float[Array, "N_atoms 4"] = jnp.column_stack(
        [cart_xyz, effective_atomic_numbers]
    )
    return create_crystal_structure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=reference_crystal.cell_lengths,
        cell_angles=reference_crystal.cell_angles,
    )


@jaxtyped(typechecker=beartype)
def apply_vacancy_field(
    crystal: CrystalStructure,
    site_occupancies: Float[Array, "N_atoms"],
) -> CrystalStructure:
    """Apply a continuous vacancy field to an existing crystal.

    Parameters
    ----------
    crystal : CrystalStructure
        Reference crystal structure.
    site_occupancies : Float[Array, "N_atoms"]
        Continuous occupancy for each crystallographic site. Values are
        clipped to ``[0, 1]`` and multiply the effective atomic number in
        the fourth column of the returned structure.

    Returns
    -------
    vacancy_modified_crystal : CrystalStructure
        Crystal with unchanged coordinates and vacancy-weighted effective
        atomic numbers.

    Notes
    -----
    1. **Clip occupancies** --
       Restrict ``site_occupancies`` to the physical interval
       ``[0, 1]``.
    2. **Apply effective weights** --
       Multiply the original atomic-number column by the clipped
       occupancies.
    3. **Preserve geometry** --
       Return a new ``CrystalStructure`` with the same fractional and
       Cartesian coordinates as the input crystal.

    This uses a continuous scattering-weight suppression model rather
    than hard atom deletion, so gradients with respect to occupancy
    remain well-defined.
    """
    occupancies: Float[Array, "N_atoms"] = jnp.clip(
        jnp.asarray(site_occupancies, dtype=jnp.float64), 0.0, 1.0
    )
    effective_atomic_numbers: Float[Array, "N_atoms"] = (
        crystal.cart_positions[:, 3] * occupancies
    )
    return _assemble_crystal(
        frac_xyz=crystal.frac_positions[:, :3],
        cart_xyz=crystal.cart_positions[:, :3],
        effective_atomic_numbers=effective_atomic_numbers,
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

    Parameters
    ----------
    crystal : CrystalStructure
        Reference crystal structure.
    interstitial_frac_positions : Float[Array, "N_interstitial 3"]
        Candidate interstitial positions in fractional coordinates of
        the input cell.
    interstitial_atomic_numbers : Float[Array, "N_interstitial"]
        Atomic number assigned to each candidate interstitial species.
    interstitial_occupancies : Float[Array, "N_interstitial"]
        Continuous occupancy or scattering weight for each candidate
        interstitial. Values are clipped to ``[0, 1]``.

    Returns
    -------
    interstitial_modified_crystal : CrystalStructure
        Crystal structure with the original atoms preserved and the
        weighted interstitial candidates appended.

    Notes
    -----
    1. **Clip occupancies** --
       Restrict interstitial occupancies to ``[0, 1]``.
    2. **Compute effective interstitial weights** --
       Multiply each candidate atomic number by its occupancy.
    3. **Map to Cartesian coordinates** --
       Convert the candidate fractional positions with the input cell
       vectors.
    4. **Append sites** --
       Concatenate the original crystal positions with the interstitial
       candidates in both coordinate systems.

    The interstitial sites are retained even at zero occupancy so the
    public contract stays shape-stable under ``jax.jit`` and
    ``jax.vmap``.
    """
    cell_vectors: Float[Array, "3 3"] = build_cell_vectors(
        *crystal.cell_lengths, *crystal.cell_angles
    )
    clipped_occupancies: Float[Array, "N_interstitial"] = jnp.clip(
        jnp.asarray(interstitial_occupancies, dtype=jnp.float64), 0.0, 1.0
    )
    effective_atomic_numbers: Float[Array, "N_interstitial"] = (
        jnp.asarray(interstitial_atomic_numbers, dtype=jnp.float64)
        * clipped_occupancies
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
        [crystal.cart_positions[:, 3], effective_atomic_numbers], axis=0
    )

    return _assemble_crystal(
        frac_xyz=frac_xyz,
        cart_xyz=cart_xyz,
        effective_atomic_numbers=atomic_numbers,
        reference_crystal=crystal,
    )


@jaxtyped(typechecker=beartype)
def apply_antisite_field(
    crystal: CrystalStructure,
    site_mixing_fractions: Float[Array, "N_atoms"],
    replacement_atomic_numbers: Float[Array, "N_atoms"],
) -> CrystalStructure:
    """Blend host and substitute species with continuous mixing fractions.

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
        Crystal with unchanged coordinates and blended effective atomic
        numbers.

    Notes
    -----
    1. **Clip mixing fractions** --
       Restrict site fractions to ``[0, 1]``.
    2. **Blend species continuously** --
       Compute ``(1 - f) * Z_host + f * Z_substitute`` for each site.
    3. **Preserve geometry** --
       Return a structure with the same coordinates and cell as the
       reference crystal.

    This acts as a differentiable virtual-crystal-style antisite model.
    """
    mixing_fractions: Float[Array, "N_atoms"] = jnp.clip(
        jnp.asarray(site_mixing_fractions, dtype=jnp.float64), 0.0, 1.0
    )
    substitute_atomic_numbers: Float[Array, "N_atoms"] = jnp.asarray(
        replacement_atomic_numbers, dtype=jnp.float64
    )
    host_atomic_numbers: Float[Array, "N_atoms"] = crystal.cart_positions[:, 3]
    effective_atomic_numbers: Float[Array, "N_atoms"] = (
        1.0 - mixing_fractions
    ) * host_atomic_numbers + mixing_fractions * substitute_atomic_numbers
    return _assemble_crystal(
        frac_xyz=crystal.frac_positions[:, :3],
        cart_xyz=crystal.cart_positions[:, :3],
        effective_atomic_numbers=effective_atomic_numbers,
        reference_crystal=crystal,
    )


__all__: list[str] = [
    "apply_antisite_field",
    "apply_interstitial_field",
    "apply_vacancy_field",
]
