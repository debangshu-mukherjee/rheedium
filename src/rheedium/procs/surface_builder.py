"""Surface slab construction and reconstruction utilities.

Extended Summary
----------------
Provides functions for building surface slabs from bulk crystals,
applying m x n surface reconstructions, and adding adsorbate layers.
These are the building blocks for constructing physically realistic
surface models that go beyond ideal bulk termination.

Routine Listings
----------------
:func:`create_surface_slab`
    Construct a surface slab from a bulk crystal by cutting along
    a specified Miller plane with vacuum gap.
:func:`apply_surface_reconstruction`
    Expand the in-plane unit cell and displace surface-layer atoms
    to model an m x n surface reconstruction.
:func:`add_adsorbate_layer`
    Add an adsorbate layer whose fractional coverage is stored as
    the adsorbate site occupancy.

Notes
-----
All functions return ``CrystalStructure`` PyTrees and are compatible
with JAX transformations. Coordinate transformations use Rodrigues'
rotation formula implemented with pure JAX operations to maintain
differentiability of continuous parameters (displacements, coverage).

R5 return type: all public functions in this module are structure builders or
sub-coherence structure modifiers and therefore return ``CrystalStructure``.
Statistical surface populations are represented separately as ``Distribution``
producers in :mod:`rheedium.procs.surface_modifier`.
"""

import math

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Final, Optional, Tuple
from jaxtyping import Array, Float, Int, Num, jaxtyped

from rheedium.types import (
    CrystalStructure,
    create_crystal_structure,
    scalar_float,
)
from rheedium.ucell import build_cell_vectors
from rheedium.ucell.unitcell import reorient_to_zone_axis

_FRAC_TOL: Final[scalar_float] = 1e-6


def _gauss_reduce_2d(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-reduce a 2-D lattice basis to its shortest vectors."""
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    for _ in range(64):
        if float(b @ b) < float(a @ a):
            a, b = b, a
        m: float = round(float(a @ b) / float(a @ a))
        if m == 0:
            break
        b = b - m * a
    return a, b


def _reduce_generators_2d(
    generators: list[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a reduced primitive basis for a 2-D lattice from generators.

    The generating set may be redundant (more than two vectors); the
    routine performs iterated size reduction against the current shortest
    vector until stable, then a final Gaussian reduction on the two
    shortest linearly independent vectors.
    """
    vectors: list[np.ndarray] = [
        np.asarray(v, dtype=np.float64)
        for v in generators
        if float(np.dot(v, v)) > 1e-12
    ]
    for _ in range(64):
        vectors = [v for v in vectors if float(v @ v) > 1e-12]
        vectors.sort(key=lambda v: float(v @ v))
        pivot: np.ndarray = vectors[0]
        changed: bool = False
        reduced: list[np.ndarray] = [pivot]
        for v in vectors[1:]:
            m: float = round(float(v @ pivot) / float(pivot @ pivot))
            r: np.ndarray = v - m * pivot
            if m != 0:
                changed = True
            reduced.append(r)
        vectors = reduced
        if not changed:
            break
    vectors = [v for v in vectors if float(v @ v) > 1e-12]
    vectors.sort(key=lambda v: float(v @ v))
    first: np.ndarray = vectors[0]
    second: Optional[np.ndarray] = None
    for v in vectors[1:]:
        cross: float = float(first[0] * v[1] - first[1] * v[0])
        if abs(cross) > 1e-6:
            second = v
            break
    if second is None:
        raise ValueError("degenerate in-plane lattice: no primitive basis")
    return _gauss_reduce_2d(first, second)


def _primitive_surface_cell(  # noqa: PLR0915
    surf: CrystalStructure,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a reoriented surface cell to its primitive in-plane mesh.

    ``reorient_to_zone_axis`` returns a valid but generally non-primitive
    surface cell (for face-centred crystals its in-plane vectors are twice
    the primitive surface translation). This routine finds the true
    in-plane periodicity of the decorated atom pattern, reduces the
    in-plane basis, and refills the (smaller) atom basis.

    Returns the reduced in-plane vectors ``a_r``, ``b_r`` (2-vectors, the
    surface normal is ``+z`` in this frame), the out-of-plane vector
    ``c_vec`` (3-vector), and the refilled atom ``frac`` coordinates and
    ``(z_number, occupancy)`` columns.
    """
    lengths: list[float] = [float(x) for x in surf.cell_lengths]
    angles: list[float] = [float(x) for x in surf.cell_angles]
    cell: np.ndarray = np.asarray(
        build_cell_vectors(*lengths, *angles), dtype=np.float64
    )
    a2: np.ndarray = cell[0, :2].copy()
    b2: np.ndarray = cell[1, :2].copy()
    c_vec: np.ndarray = cell[2].copy()

    cart: np.ndarray = np.asarray(surf.cart_positions[:, :3], dtype=np.float64)
    z_num: np.ndarray = np.asarray(surf.cart_positions[:, 3], dtype=np.float64)
    occ: np.ndarray = (
        np.ones(cart.shape[0], dtype=np.float64)
        if surf.occupancies is None
        else np.asarray(surf.occupancies, dtype=np.float64)
    )
    xy: np.ndarray = cart[:, :2]
    z_cart: np.ndarray = cart[:, 2]
    basis_mat: np.ndarray = np.column_stack([a2, b2])
    tol: float = 1e-4

    def _key(
        point_xy: np.ndarray, z_val: float, z_number: float, occ_val: float
    ) -> tuple:
        frac: np.ndarray = np.linalg.solve(basis_mat, point_xy)
        frac = frac - np.floor(frac + tol)
        frac = np.where(frac > 1.0 - tol, 0.0, frac)
        return (
            round(float(frac[0]), 3),
            round(float(frac[1]), 3),
            round(float(z_val), 3),
            int(round(z_number)),
            round(float(occ_val), 2),
        )

    atom_set: set = {
        _key(xy[i], z_cart[i], z_num[i], occ[i]) for i in range(len(z_num))
    }
    candidates: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    for j in range(len(z_num)):
        same: bool = (
            abs(z_num[j] - z_num[0]) < 0.5
            and abs(z_cart[j] - z_cart[0]) < 1e-3
            and abs(occ[j] - occ[0]) < 1e-3
        )
        if same:
            candidates.append(xy[j] - xy[0])
    valid: list[np.ndarray] = []
    for t in candidates:
        maps_onto: bool = all(
            _key(xy[i] + t, z_cart[i], z_num[i], occ[i]) in atom_set
            for i in range(len(z_num))
        )
        if maps_onto:
            valid.append(np.asarray(t, dtype=np.float64))

    generators: list[np.ndarray] = [
        v for v in valid if float(np.dot(v, v)) > 1e-12
    ]
    generators.extend([a2, b2])
    a_r: np.ndarray
    b_r: np.ndarray
    a_r, b_r = _reduce_generators_2d(generators)
    if float(a_r @ b_r) > 1e-9:
        b_r = b_r - a_r
    if float(a_r @ b_r) > 1e-9:
        b_r = b_r - a_r

    a_r3: np.ndarray = np.array([a_r[0], a_r[1], 0.0])
    b_r3: np.ndarray = np.array([b_r[0], b_r[1], 0.0])
    reduced_basis: np.ndarray = np.stack([a_r3, b_r3, c_vec], axis=0)
    inv_reduced: np.ndarray = np.linalg.inv(reduced_basis.T)

    kept_frac: list[list[float]] = []
    kept_cols: list[list[float]] = []
    seen: set = set()
    for i in range(len(z_num)):
        frac: np.ndarray = inv_reduced @ cart[i]
        frac = frac - np.floor(frac + 1e-6)
        frac = np.where(frac > 1.0 - 1e-6, 0.0, frac)
        key: tuple = (
            round(float(frac[0]), 4),
            round(float(frac[1]), 4),
            round(float(frac[2]), 4),
            int(round(z_num[i])),
        )
        if key in seen:
            continue
        seen.add(key)
        kept_frac.append([float(frac[0]), float(frac[1]), float(frac[2])])
        kept_cols.append([float(z_num[i]), float(occ[i])])

    frac_arr: np.ndarray = np.asarray(kept_frac, dtype=np.float64)
    cols_arr: np.ndarray = np.asarray(kept_cols, dtype=np.float64)
    return a_r3, b_r3, c_vec, frac_arr, cols_arr


@jaxtyped(typechecker=beartype)
def create_surface_slab(  # noqa: PLR0915
    bulk_crystal: CrystalStructure,
    surface_normal_miller: Int[Array, "3"],
    slab_thickness_angstrom: Optional[scalar_float] = None,
    vacuum_gap_angstrom: scalar_float = 15.0,
    *,
    n_layers: Optional[int] = None,
    in_plane_repeat: Tuple[int, int] = (1, 1),
) -> CrystalStructure:
    """Construct a true ``(hkl)`` surface slab from a bulk crystal.

    :see: :class:`~.test_surface_builder.TestCreateSurfaceSlab`

    Parameters
    ----------
    bulk_crystal : CrystalStructure
        Bulk crystal with lattice parameters and atomic basis.
    surface_normal_miller : Int[Array, "3"]
        Miller indices [h, k, l] of the surface normal direction.
    slab_thickness_angstrom : scalar_float, optional
        Requested slab thickness in Angstroms. Converted to an integer
        number of ``(hkl)`` layer repeats via the c-axis repeat distance
        (``ceil``). Ignored when ``n_layers`` is given.
    vacuum_gap_angstrom : scalar_float, optional
        Vacuum layer thickness in Angstroms above the slab. Default 15.0.
    n_layers : int, optional
        Explicit number of ``(hkl)`` layer repeats. Takes precedence over
        ``slab_thickness_angstrom`` when both are supplied.
    in_plane_repeat : Tuple[int, int], optional
        Diagonal in-plane supercell repeats ``(na, nb)`` applied only when
        the caller asks for a supercell (default ``(1, 1)`` = no tiling).

    Returns
    -------
    slab : CrystalStructure
        Slab whose ``a``/``b`` span the ``(hkl)`` plane as the primitive
        surface mesh, ``c`` is perpendicular with length
        ``thickness + vacuum`` (or ``n_layers * d_layer + vacuum``), and
        fractional coordinates are wrapped into ``[0, 1)``.

    Notes
    -----
    1. **Reorient** -- ``reorient_to_zone_axis`` builds a cell whose
       ``a``/``b`` span the ``(hkl)`` plane; its in-plane mesh is reduced
       to the primitive surface cell (:func:`_primitive_surface_cell`).
    2. **Stack** -- ``n_layers`` copies are stacked along the surface
       cell's ``c`` (fractional z plus integer offset).
    3. **Vacuum** -- ``c`` is extended by ``vacuum_gap_angstrom`` while
       atom Cartesian ``z`` stays fixed.
    4. **Canonical frame** -- lengths/angles feed ``build_cell_vectors``
       so ``cart = frac @ build_cell_vectors(lengths, angles)``; the
       in-plane mesh is tiled only when ``in_plane_repeat`` asks for it.
    5. **Thickness precedence** -- ``n_layers`` wins; otherwise
       ``slab_thickness_angstrom`` sets both the layer count
       (``ceil(thickness / d_layer)``) and the total ``c``.

    Fractional coordinates are wrapped into ``[0, 1)`` so translating the
    slab by the returned ``a`` (or ``b``) maps the atom set onto itself.
    """
    if n_layers is None and slab_thickness_angstrom is None:
        raise ValueError("supply either n_layers or slab_thickness_angstrom")

    surface_normal_miller = jnp.asarray(surface_normal_miller, dtype=jnp.int32)
    vacuum: float = float(vacuum_gap_angstrom)

    surf: CrystalStructure = reorient_to_zone_axis(
        bulk_crystal, surface_normal_miller
    )
    a_r3: np.ndarray
    b_r3: np.ndarray
    c_vec: np.ndarray
    base_frac: np.ndarray
    base_cols: np.ndarray
    a_r3, b_r3, c_vec, base_frac, base_cols = _primitive_surface_cell(surf)

    reduced_basis: np.ndarray = np.stack([a_r3, b_r3, c_vec], axis=0)
    base_cart: np.ndarray = base_frac @ reduced_basis
    d_layer: float = abs(float(c_vec[2]))

    if n_layers is not None:
        n_stack: int = int(n_layers)
        total_c: float = n_stack * d_layer + vacuum
        keep_top: float = n_stack * d_layer - 1e-6
    else:
        thickness: float = float(slab_thickness_angstrom)
        # Snap to a whole number of layers so the physical slab span is an
        # exact multiple of the interlayer spacing; otherwise using the raw
        # requested thickness as ``total_c`` while the atom count quantizes by
        # layer breaks the bulk-density contract for arbitrary thicknesses.
        n_stack = max(int(math.ceil(thickness / d_layer - 1e-6)), 1)
        total_c = n_stack * d_layer + vacuum
        keep_top = n_stack * d_layer - 1e-6

    layer_offsets: np.ndarray = np.arange(n_stack + 1, dtype=np.float64)
    stacked_cart: np.ndarray = (
        base_cart[None, :, :]
        + layer_offsets[:, None, None] * c_vec[None, None]
    ).reshape(-1, 3)
    stacked_cols: np.ndarray = np.tile(base_cols, (n_stack + 1, 1))
    stacked_cart[:, 2] -= stacked_cart[:, 2].min()
    keep_mask: np.ndarray = stacked_cart[:, 2] <= keep_top
    kept_cart: np.ndarray = stacked_cart[keep_mask]
    kept_cols: np.ndarray = stacked_cols[keep_mask]

    a_len: float = float(np.linalg.norm(a_r3))
    b_len: float = float(np.linalg.norm(b_r3))
    cos_gamma: float = float(a_r3 @ b_r3) / (a_len * b_len)
    gamma_deg: float = float(np.degrees(np.arccos(np.clip(cos_gamma, -1, 1))))

    na: int = int(in_plane_repeat[0])
    nb: int = int(in_plane_repeat[1])
    slab_basis: np.ndarray = np.stack(
        [a_r3, b_r3, np.array([0.0, 0.0, total_c])], axis=0
    )
    inv_slab: np.ndarray = np.linalg.inv(slab_basis.T)
    frac_xyz: np.ndarray = (inv_slab @ kept_cart.T).T
    frac_xyz = frac_xyz - np.floor(frac_xyz + _FRAC_TOL)
    frac_xyz = np.where(frac_xyz > 1.0 - _FRAC_TOL, 0.0, frac_xyz)

    if na > 1 or nb > 1:
        shifts: list[np.ndarray] = []
        cols_tiled: list[np.ndarray] = []
        for i in range(na):
            for j in range(nb):
                shifted: np.ndarray = frac_xyz + np.array(
                    [i, j, 0.0], dtype=np.float64
                )
                shifted[:, 0] /= na
                shifted[:, 1] /= nb
                shifts.append(shifted)
                cols_tiled.append(kept_cols)
        frac_xyz = np.concatenate(shifts, axis=0)
        kept_cols = np.concatenate(cols_tiled, axis=0)
        a_len *= na
        b_len *= nb

    new_lengths: np.ndarray = np.array([a_len, b_len, total_c])
    new_angles: np.ndarray = np.array([90.0, 90.0, gamma_deg])
    canonical: np.ndarray = np.asarray(
        build_cell_vectors(*new_lengths, *new_angles), dtype=np.float64
    )
    cart_xyz: np.ndarray = frac_xyz @ canonical

    z_col: np.ndarray = kept_cols[:, 0:1]
    occ_out: Float[Array, "K"] = jnp.asarray(
        kept_cols[:, 1], dtype=jnp.float64
    )
    frac_with_z: Float[Array, "K 4"] = jnp.asarray(
        np.column_stack([frac_xyz, z_col]), dtype=jnp.float64
    )
    cart_with_z: Float[Array, "K 4"] = jnp.asarray(
        np.column_stack([cart_xyz, z_col]), dtype=jnp.float64
    )
    return create_crystal_structure(
        frac_positions=frac_with_z,
        cart_positions=cart_with_z,
        cell_lengths=jnp.asarray(new_lengths, dtype=jnp.float64),
        cell_angles=jnp.asarray(new_angles, dtype=jnp.float64),
        occupancies=occ_out,
    )


@jaxtyped(typechecker=beartype)
def apply_surface_reconstruction(  # noqa: PLR0915
    slab: CrystalStructure,
    reconstruction_matrix: Int[Array, "2 2"],
    surface_layer_depth_angstrom: scalar_float,
    atomic_displacements: Optional[Float[Array, "N_surface_atoms 3"]] = None,
) -> CrystalStructure:
    """Apply an m x n surface reconstruction to a slab.

    :see: :class:`~.test_surface_builder.TestApplySurfaceReconstruction`

    Parameters
    ----------
    slab : CrystalStructure
        Base surface slab (unreconstructed).
    reconstruction_matrix : Int[Array, "2 2"]
        2x2 integer matrix relating new surface cell to bulk cell.
        For Si(111) 7x7: ``[[7,0],[0,7]]``.
        For GaAs(001) 2x4: ``[[2,0],[0,4]]``.
    surface_layer_depth_angstrom : scalar_float
        Depth from top surface that defines the reconstructed region.
    atomic_displacements : Float[Array, "N_surface_atoms 3"], optional
        Displacement vectors [dx, dy, dz] in Angstroms, one per surface
        atom in the deterministic (z, y, x) ordering. A length mismatch
        with the surface-atom count raises ``ValueError`` (no silent
        truncation). ``None`` (default) expands the cell without any
        displacement (used to build supercells for adatom decoration).

    Returns
    -------
    reconstructed_slab : CrystalStructure
        Slab with expanded in-plane cell and displaced surface atoms.

    Notes
    -----
    1. **Expand the cell** -- ``a_new = m11*a1 + m12*a2``,
       ``b_new = m21*a1 + m22*a2``, ``c`` unchanged.
    2. **Signed tiling** -- the supercell corners ``(0,0),(1,0),(0,1),
       (1,1)`` are transformed into base-lattice coordinates and the base
       cell is tiled over ``floor(min)-1 .. ceil(max)+1`` on both axes, so
       shear matrices with negative entries are covered.
    3. **Membership** -- each candidate is wrapped into supercell
       fractional coordinates (``frac - floor(frac)``) with a half-open
       ``[0, 1)`` corner snap and deduplicated on rounded ``(frac, Z)``
       keys, yielding exactly ``|det M| * n_basis`` atoms.
    4. **Stable ordering** -- atoms are sorted by ``(z, y, x)`` so the
       surface-atom / displacement correspondence is deterministic; the
       top ``surface_layer_depth_angstrom`` slab defines the surface set.
    5. **Displacements** -- one displacement per surface atom (else
       ``ValueError``) is added in the canonical frame, and fractional
       coordinates are recomputed so ``cart = frac @ build_cell_vectors``.
    """
    reconstruction_matrix = jnp.asarray(reconstruction_matrix, dtype=jnp.int32)
    depth: float = float(surface_layer_depth_angstrom)

    cell_vecs: np.ndarray = np.asarray(
        build_cell_vectors(*slab.cell_lengths, *slab.cell_angles),
        dtype=np.float64,
    )
    a1: np.ndarray = cell_vecs[0]
    a2: np.ndarray = cell_vecs[1]

    m11: int = int(reconstruction_matrix[0, 0])
    m12: int = int(reconstruction_matrix[0, 1])
    m21: int = int(reconstruction_matrix[1, 0])
    m22: int = int(reconstruction_matrix[1, 1])
    det_m: int = abs(m11 * m22 - m12 * m21)
    if det_m == 0:
        raise ValueError("reconstruction_matrix must be non-singular")

    new_a_vec: np.ndarray = m11 * a1 + m12 * a2
    new_b_vec: np.ndarray = m21 * a1 + m22 * a2
    new_c_vec: np.ndarray = cell_vecs[2]
    new_cell_mat: np.ndarray = np.stack([new_a_vec, new_b_vec, new_c_vec])
    inv_new_cell: np.ndarray = np.linalg.inv(new_cell_mat.T)

    base_cart: np.ndarray = np.asarray(
        slab.cart_positions[:, :3], dtype=np.float64
    )
    z_num: np.ndarray = np.asarray(slab.cart_positions[:, 3], dtype=np.float64)
    occ: np.ndarray = (
        np.ones(base_cart.shape[0], dtype=np.float64)
        if slab.occupancies is None
        else np.asarray(slab.occupancies, dtype=np.float64)
    )
    n_basis: int = base_cart.shape[0]

    # Signed bounding box: supercell corners -> base-lattice coordinates.
    corners: np.ndarray = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64
    )
    base_coords: np.ndarray = corners @ np.array(
        [[m11, m12], [m21, m22]], dtype=np.float64
    )
    n1_lo: int = int(math.floor(base_coords[:, 0].min())) - 1
    n1_hi: int = int(math.ceil(base_coords[:, 0].max())) + 1
    n2_lo: int = int(math.floor(base_coords[:, 1].min())) - 1
    n2_hi: int = int(math.ceil(base_coords[:, 1].max())) + 1

    tol: float = 1e-6
    kept_frac: list[list[float]] = []
    kept_cols: list[list[float]] = []
    seen: set = set()
    for n1 in range(n1_lo, n1_hi + 1):
        for n2 in range(n2_lo, n2_hi + 1):
            shift: np.ndarray = n1 * a1 + n2 * a2
            for i in range(n_basis):
                cart_i: np.ndarray = base_cart[i] + shift
                frac: np.ndarray = inv_new_cell @ cart_i
                wrapped: np.ndarray = frac - np.floor(frac + tol)
                wrapped = np.where(wrapped > 1.0 - tol, 0.0, wrapped)
                key: tuple = (
                    round(float(wrapped[0]), 6),
                    round(float(wrapped[1]), 6),
                    round(float(wrapped[2]), 6),
                    int(round(z_num[i])),
                )
                if key in seen:
                    continue
                seen.add(key)
                kept_frac.append(
                    [float(wrapped[0]), float(wrapped[1]), float(wrapped[2])]
                )
                kept_cols.append([float(z_num[i]), float(occ[i])])

    expected: int = det_m * n_basis
    if len(kept_frac) != expected:
        raise ValueError(
            f"signed tiling produced {len(kept_frac)} atoms, "
            f"expected {expected}"
        )

    new_lengths: np.ndarray = np.array(
        [
            float(np.linalg.norm(new_a_vec)),
            float(np.linalg.norm(new_b_vec)),
            float(np.linalg.norm(new_c_vec)),
        ]
    )
    cos_alpha: float = float(new_b_vec @ new_c_vec) / (
        new_lengths[1] * new_lengths[2]
    )
    cos_beta: float = float(new_a_vec @ new_c_vec) / (
        new_lengths[0] * new_lengths[2]
    )
    cos_gamma: float = float(new_a_vec @ new_b_vec) / (
        new_lengths[0] * new_lengths[1]
    )
    new_angles: np.ndarray = np.degrees(
        np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0))
    )
    canonical: np.ndarray = np.asarray(
        build_cell_vectors(*new_lengths, *new_angles), dtype=np.float64
    )

    frac_mat: np.ndarray = np.asarray(kept_frac, dtype=np.float64)
    cols_mat: np.ndarray = np.asarray(kept_cols, dtype=np.float64)
    cart_mat: np.ndarray = frac_mat @ canonical

    # Stable ordering by (z, y, x) so displacement indexing is deterministic.
    order: np.ndarray = np.lexsort(
        (cart_mat[:, 0], cart_mat[:, 1], cart_mat[:, 2])
    )
    cart_mat = cart_mat[order]
    cols_mat = cols_mat[order]

    if atomic_displacements is not None:
        z_max: float = float(cart_mat[:, 2].max())
        surface_mask: np.ndarray = cart_mat[:, 2] >= z_max - depth
        n_surface: int = int(surface_mask.sum())
        displacements: np.ndarray = np.asarray(
            atomic_displacements, dtype=np.float64
        )
        if displacements.shape[0] != n_surface:
            raise ValueError(
                f"atomic_displacements has {displacements.shape[0]} rows but "
                f"the reconstruction has {n_surface} surface atoms (depth "
                f"{depth} A); supply exactly one displacement per surface "
                "atom in (z, y, x) order"
            )
        surface_idx: np.ndarray = np.nonzero(surface_mask)[0]
        cart_mat[surface_idx] += displacements

    inv_canonical: np.ndarray = np.linalg.inv(canonical)
    frac_out: np.ndarray = cart_mat @ inv_canonical
    z_col: np.ndarray = cols_mat[:, 0:1]
    return create_crystal_structure(
        frac_positions=jnp.asarray(
            np.column_stack([frac_out, z_col]), dtype=jnp.float64
        ),
        cart_positions=jnp.asarray(
            np.column_stack([cart_mat, z_col]), dtype=jnp.float64
        ),
        cell_lengths=jnp.asarray(new_lengths, dtype=jnp.float64),
        cell_angles=jnp.asarray(new_angles, dtype=jnp.float64),
        occupancies=jnp.asarray(cols_mat[:, 1], dtype=jnp.float64),
    )


@jaxtyped(typechecker=beartype)
def add_adsorbate_layer(
    slab: CrystalStructure,
    adsorbate_positions_fractional: Float[Array, "N_ads 3"],
    adsorbate_atomic_numbers: Num[Array, "N_ads"],
    coverage_fraction: scalar_float,
) -> CrystalStructure:
    """Add an adsorbate layer to a surface slab.

    :see: :class:`~.test_surface_builder.TestAddAdsorbateLayer`

    Parameters
    ----------
    slab : CrystalStructure
        Base surface slab.
    adsorbate_positions_fractional : Float[Array, "N_ads 3"]
        Fractional coordinates of adsorbate atoms in the slab cell.
    adsorbate_atomic_numbers : Num[Array, "N_ads"]
        Atomic number of each adsorbate atom.
    coverage_fraction : scalar_float
        Fractional monolayer coverage in [0, 1]. Stored as the
        adsorbate site occupancy, which weights each adsorbate's form
        factor in every simulation kernel.

    Returns
    -------
    decorated_slab : CrystalStructure
        Slab with adsorbates appended. Adsorbate atomic numbers are
        stored unchanged; the coverage enters as the adsorbates'
        ``occupancies`` entries, so each adsorbate scatters as
        ``coverage * f_Z(q)``.

    Notes
    -----
    1. Convert adsorbate fractional positions to Cartesian using
       the slab cell vectors.
    2. Clip ``coverage_fraction`` to [0, 1] and assign it as the
       occupancy of every adsorbate site; atomic numbers stay the
       element's integral Z.
    3. Concatenate slab and adsorbate positions and occupancies
       (slab occupancies default to ones when absent).
    4. Return new CrystalStructure with combined atoms.

    Modeling partial coverage as a site occupancy is appropriate for
    RHEED because the beam illuminates macroscopic areas and samples a
    statistical average over many unit cells.
    """
    coverage_fraction: Float[Array, ""] = jnp.clip(
        jnp.asarray(coverage_fraction, dtype=jnp.float64), 0.0, 1.0
    )
    adsorbate_atomic_numbers: Float[Array, "N_ads"] = jnp.asarray(
        adsorbate_atomic_numbers, dtype=jnp.float64
    )

    cell_vecs: Float[Array, "3 3"] = build_cell_vectors(
        *slab.cell_lengths, *slab.cell_angles
    )

    ads_cart: Float[Array, "N_ads 3"] = (
        adsorbate_positions_fractional @ cell_vecs
    )

    ads_cart_with_z: Float[Array, "N_ads 4"] = jnp.column_stack(
        [ads_cart, adsorbate_atomic_numbers]
    )
    ads_frac_with_z: Float[Array, "N_ads 4"] = jnp.column_stack(
        [adsorbate_positions_fractional, adsorbate_atomic_numbers]
    )

    new_cart: Num[Array, "M 4"] = jnp.concatenate(
        [slab.cart_positions, ads_cart_with_z], axis=0
    )
    new_frac: Num[Array, "M 4"] = jnp.concatenate(
        [slab.frac_positions, ads_frac_with_z], axis=0
    )
    slab_occupancies: Float[Array, "N_slab"] = (
        jnp.ones(slab.cart_positions.shape[0], dtype=jnp.float64)
        if slab.occupancies is None
        else jnp.asarray(slab.occupancies, dtype=jnp.float64)
    )
    ads_occupancies: Float[Array, "N_ads"] = coverage_fraction * jnp.ones(
        adsorbate_atomic_numbers.shape[0], dtype=jnp.float64
    )
    new_occupancies: Float[Array, "M"] = jnp.concatenate(
        [slab_occupancies, ads_occupancies], axis=0
    )

    return create_crystal_structure(
        frac_positions=new_frac,
        cart_positions=new_cart,
        cell_lengths=slab.cell_lengths,
        cell_angles=slab.cell_angles,
        occupancies=new_occupancies,
    )


__all__: list[str] = [
    "add_adsorbate_layer",
    "apply_surface_reconstruction",
    "create_surface_slab",
]
