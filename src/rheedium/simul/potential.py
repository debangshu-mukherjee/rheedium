"""Crystal projected potential construction for multislice simulation.

Extended Summary
----------------
Provides ``crystal_projected_potential``, which builds the complex
projected electrostatic potential V(x, y) + i V_abs(x, y) for one
slice of a multislice calculation. The real part is the elastic
potential summed from atomic projected potentials; the imaginary
part is an absorptive potential modeling inelastic losses
(phonon scattering, plasmon excitation).

Routine Listings
----------------
:func:`crystal_projected_potential`
    Construct the complex projected potential for one multislice
    slice from atomic positions and atomic numbers.

Notes
-----
Two parameterizations are supported:
- ``"lobato"`` (default): Lobato-van Dyck (2014) Bessel-K potential.
  Preferred for RHEED — more accurate at high q, obeys all physical
  constraints.
- ``"kirkland"``: Kirkland (1998) Gaussian potential. Retained for
  cross-validation.

The summation over atoms is implemented as ``jax.vmap`` over the
atom axis followed by ``jnp.sum`` rather than ``lax.scan`` so that
gradients flow cleanly through every atomic contribution.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from rheedium.types import scalar_float

from .form_factors import projected_potential


@jaxtyped(typechecker=beartype)
def crystal_projected_potential(
    atomic_positions_angstrom: Float[Array, "N_atoms 3"],
    atomic_numbers: Int[Array, "N_atoms"],
    grid_shape: Tuple[int, int],
    cell_dimensions_angstrom: Float[Array, "2"],
    absorption_fraction: scalar_float = 0.1,
    parameterization: str = "lobato",
) -> Complex[Array, "H W"]:
    r"""Compute the complex projected potential for one slice.

    Parameters
    ----------
    atomic_positions_angstrom : Float[Array, "N_atoms 3"]
        Cartesian positions [x, y, z] of atoms in the slice in
        Angstroms. Only x and y are used; z is ignored because the
        projection has already been done by the slicing step.
    atomic_numbers : Int[Array, "N_atoms"]
        Atomic number Z of each atom (1-103).
    grid_shape : tuple[int, int]
        Real-space grid dimensions (H, W) in the surface plane.
    cell_dimensions_angstrom : Float[Array, "2"]
        Physical extent [Lx, Ly] of the simulation cell in Angstroms.
    absorption_fraction : scalar_float, optional
        Ratio kappa = V_abs / V_real. Typical: 0.05-0.15.
        Default: 0.1.
    parameterization : str, optional
        Potential model: ``"lobato"`` or ``"kirkland"``. Default:
        ``"lobato"``.

    Returns
    -------
    projected_potential_complex : Complex[Array, "H W"]
        Complex projected potential V_real + i*V_abs in V*Angstrom
        units.

    Notes
    -----
    1. Build real-space coordinate grids ``xx(H, W)`` and ``yy(H, W)``
       from ``cell_dimensions_angstrom`` and ``grid_shape``.
    2. For each atom: compute the radial distance to every grid pixel
       and evaluate the projected atomic potential via
       :func:`rheedium.simul.projected_potential`.
    3. Sum contributions from all atoms with ``jax.vmap`` + ``jnp.sum``
       (differentiable through every atom).
    4. The real part is the elastic potential. The imaginary part is
       ``absorption_fraction * V_real`` (proportional absorption
       approximation).
    5. Return ``V_real + 1j * V_abs``.
    """
    absorption_fraction_arr: Float[Array, ""] = jnp.asarray(
        absorption_fraction, dtype=jnp.float64
    )
    n_x: int = grid_shape[0]
    n_y: int = grid_shape[1]
    lx: Float[Array, ""] = cell_dimensions_angstrom[0]
    ly: Float[Array, ""] = cell_dimensions_angstrom[1]
    x_coords: Float[Array, "H"] = jnp.linspace(0.0, lx, n_x)
    y_coords: Float[Array, "W"] = jnp.linspace(0.0, ly, n_y)
    xx: Float[Array, "H W"]
    yy: Float[Array, "H W"]
    xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")

    def _atom_contribution(
        atom_pos: Float[Array, "3"],
        atom_z: Int[Array, ""],
    ) -> Float[Array, "H W"]:
        """Project a single atom's potential onto the grid."""
        dx: Float[Array, "H W"] = xx - atom_pos[0]
        dy: Float[Array, "H W"] = yy - atom_pos[1]
        r: Float[Array, "H W"] = jnp.sqrt(dx**2 + dy**2)
        v_atom: Float[Array, "H W"] = projected_potential(
            atom_z, r, parameterization
        )
        return v_atom

    contributions: Float[Array, "N_atoms H W"] = jax.vmap(_atom_contribution)(
        atomic_positions_angstrom, atomic_numbers
    )
    v_real: Float[Array, "H W"] = jnp.sum(contributions, axis=0)
    v_abs: Float[Array, "H W"] = absorption_fraction_arr * v_real
    v_complex: Complex[Array, "H W"] = v_real + 1j * v_abs
    return v_complex


__all__: list[str] = [
    "crystal_projected_potential",
]
