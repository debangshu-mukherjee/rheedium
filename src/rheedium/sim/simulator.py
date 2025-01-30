import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from scipy.interpolate import griddata

from rheedium import io, sim, uc

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class RHEEDPattern(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for representing RHEED patterns.

    Attributes
    ----------
    - `G_indices` (Int[Array, "*"]):
        Indices of reciprocal-lattice vectors that satisfy reflection
    - `k_out` (Float[Array, "M 3"]):
        Outgoing wavevectors (in 1/Å) for those reflections
    - `detector_points` (Float[Array, "M 2"]):
        (Y, Z) coordinates on the detector plane, in Ångstroms.
    - `intensities` (Float[Array, "M"]):
        Intensities for each reflection.
    """

    G_indices: Int[Array, "*"]
    k_out: Float[Array, "M 3"]
    detector_points: Float[Array, "M 2"]
    intensities: Float[Array, "M"]

    def tree_flatten(self):
        return (
            (self.G_indices, self.k_out, self.detector_points, self.intensities),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
def incident_wavevector(
    lam_ang: Float[Array, ""], theta_deg: Float[Array, ""]
) -> Float[Array, "3"]:
    """
    Description
    -----------
    Build an incident wavevector k_in with magnitude (2π / λ),
    traveling mostly along +x, with a small angle theta from the x-y plane.

    Parameters
    ----------
    - `lam_ang` (Float[Array, ""]):
        Electron wavelength in angstroms
    - `theta_deg` (Float[Array, ""]):
        Grazing angle in degrees

    Returns
    -------
    - `k_in` (Float[Array, "3"]):
        The 3D incident wavevector (1/angstrom)
    """
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    theta: Float[Array, ""] = jnp.deg2rad(theta_deg)
    kx: Float[Array, ""] = k_mag * jnp.cos(theta)
    kz: Float[Array, ""] = -k_mag * jnp.sin(theta)
    k_in: Float[Array, "3"] = jnp.array([kx, 0.0, kz], dtype=jnp.float64)
    return k_in


@jaxtyped(typechecker=beartype)
def project_on_detector(
    k_out_set: Float[Array, "M 3"], detector_distance: Float[Array, ""]
) -> Float[Array, "M 2"]:
    """
    Description
    -----------
    Project wavevectors k_out onto a plane at x = detector_distance.
    Returns (M, 2) array of [Y, Z] coordinates on the detector.

    Parameters
    ----------
    - `k_out_set` (Float[Array, "M 3"]):
        (M, 3) array of outgoing wavevectors
    - `detector_distance` (Float[Array, ""):
        distance (in angstroms, or same unit) where screen is placed at x = L

    Returns
    -------
    - `coords` (Float[Array, "M 2"]):
        (M, 2) array of projected [Y, Z]
    """
    norms: Float[Array, "M 1"] = jnp.linalg.norm(k_out_set, axis=1, keepdims=True)
    directions: Float[Array, "M 3"] = k_out_set / (norms + 1e-12)
    t_vals: Float[Array, "M"] = detector_distance / (directions[:, 0] + 1e-12)
    Y: Float[Array, "M"] = directions[:, 1] * t_vals
    Z: Float[Array, "M"] = directions[:, 2] * t_vals
    coords: Float[Array, "M 2"] = jnp.stack([Y, Z], axis=-1)
    return coords


@jaxtyped(typechecker=beartype)
def find_kinematic_reflections(
    k_in: Float[Array, "3"],
    Gs: Float[Array, "M 3"],
    lam_ang: Float[Array, ""],
    z_sign: Optional[Float[Array, ""]] = jnp.asarray(1.0),
    tolerance: Optional[Float[Array, ""]] = jnp.asarray(0.05),
) -> Tuple[Int[Array, "K"], Float[Array, "K 3"]]:
    """
    Description
    -----------
    Returns indices of G for which ||k_in + G|| ~ 2π/lam
    and the z-component of (k_in + G) has the specified sign.

    Parameters
    ----------
    - `k_in` (Float[Array, "3"]):
        shape (3,)
    - `Gs` (Float[Array, "M 3]"):
        G vector
    - `lam_ang` (Float[Array, ""):
        electron wavelength in Å
    - `z_sign` (Float[Array, ""]):
        sign for z-component of k_out
    - `tolerance` (Float[Array, ""]):
        how close to the Ewald sphere in 1/Å

    Returns
    -------
    - `allowed_indices` (Int[Array, "K"]):
        Allowed indices that will kinematically reflect.
    - `k_out` (Float[Array, "K 3"]):
        Outgoing wavevectors (in 1/Å) for those reflections.
    """
    k_mag: Float[Array, ""] = 2.0 * jnp.pi / lam_ang
    k_out_candidates: Float[Array, "M 3"] = k_in[None, :] + Gs  # shape (M,3)
    norms: Float[Array, "M"] = jnp.linalg.norm(k_out_candidates, axis=1)
    cond_mag: Bool[Array, "M"] = jnp.abs(norms - k_mag) < tolerance
    cond_z: Bool[Array, "M"] = jnp.sign(k_out_candidates[:, 2]) == jnp.sign(z_sign)
    mask: Bool[Array, "M"] = jnp.logical_and(cond_mag, cond_z)
    allowed_indices: Int[Array, "K"] = jnp.where(mask)[0]
    k_out: Float[Array, "K 3"] = k_out_candidates[allowed_indices]
    return (allowed_indices, k_out)


def compute_kinematic_intensities(
    positions: Float[Array, "N 3"], G_allowed: Float[Array, "M 3"]
) -> Float[Array, "M"]:
    """
    Description
    -----------
    Given the atomic Cartesian positions (N,3) and the
    reciprocal vectors G_allowed (M,3),
    compute the kinematic intensity for each reflection:
        I(G) = | sum_j exp(i G·r_j) |^2
    ignoring atomic form factors, etc.

    Parameters
    ----------
    - `positions` (Float[Array, "N 3]):
        Atomic positions in Cartesian coordinates.
    - `G_allowed` (Float[Array, "M 3]):
        Reciprocal lattice vectors that satisfy reflection condition.

    Returns
    -------
    - `intensities` (Float[Array, "M"]):
        Intensities for each reflection.
    """

    def intensity_for_G(G_):
        phases = jnp.einsum("j,ij->i", G_, positions)
        re = jnp.sum(jnp.cos(phases))
        im = jnp.sum(jnp.sin(phases))
        return re * re + im * im

    intensities = jax.vmap(intensity_for_G)(G_allowed)
    return intensities


def simulate_rheed_pattern(
    crystal: io.CrystalStructure,
    voltage_kV: Optional[Float[Array, ""]] = jnp.asarray(10.0),
    theta_deg: Optional[Float[Array, ""]] = jnp.asarray(1.0),
    hmax: Optional[Int[Array, ""]] = jnp.asarray(3),
    kmax: Optional[Int[Array, ""]] = jnp.asarray(3),
    lmax: Optional[Int[Array, ""]] = jnp.asarray(1),
    tolerance: Optional[Float[Array, ""]] = jnp.asarray(0.05),
    detector_distance: Optional[Float[Array, ""]] = jnp.asarray(1000.0),
    z_sign: Optional[Float[Array, ""]] = jnp.asarray(1.0),
) -> RHEEDPattern:
    """
    Description
    -----------
    Compute a simple kinematic RHEED pattern for the given crystal.

    Parameters
    ----------
    - `crystal` (io.CrystalStructure):
        Crystal structure to simulate
    - `voltage_kV` (Float[Array, ""]):
        Accelerating voltage in kilovolts.
        Optional. Default: 10.0
    - `theta_deg` (Float[Array, ""]):
        Grazing angle in degrees
        Optional. Default: 1.0
    - `hmax, kmax, lmax` (Int[Array, ""]):
        Bounds on reciprocal lattice indices
        Optional. Default: 3, 3, 1
    - `tolerance` (Float[Array, ""]):
        How close to the Ewald sphere in 1/Å
        Optional. Default: 0.05
    - `detector_distance` (Float[Array, ""]):
        Distance from the sample to the detector plane in angstroms
        Optional. Default: 1000.0
    - `z_sign` (Float[Array, ""]):
        If +1, keep reflections with positive z in k_out
        Optional. Default: 1.0
    Returns
    -------
    - `pattern` (RHEEDPattern):
        A NamedTuple capturing reflection indices, k_out, and detector coords.

    Flow
    ----
    - Rebuild real-space cell vectors from (cell_lengths, cell_angles).
    - Generate reciprocal lattice points up to (hmax, kmax, lmax).
    - Compute electron wavelength in angstroms from voltage_kV (kV).
    - Build incident wavevector at grazing angle theta_deg.
    - Find which Gs satisfy the reflection condition: ||k_in + G|| ~ 2π/λ.
    - Project the resulting k_out onto a plane at x=detector_distance.
    - Return a RHEEDPattern with reflection info.
    """
    cell_vecs = uc.build_cell_vectors(
        crystal.cell_lengths[0],
        crystal.cell_lengths[1],
        crystal.cell_lengths[2],
        crystal.cell_angles[0],
        crystal.cell_angles[1],
        crystal.cell_angles[2],
    )
    Gs: Float[Array, "M 3"] = uc.generate_reciprocal_points(
        crystal=crystal,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        in_degrees=True,
    )
    lam_ang: Float[Array, ""] = uc.wavelength_ang(voltage_kV)
    k_in: Float[Array, "3"] = sim.incident_wavevector(lam_ang, theta_deg)
    allowed_indices, k_out = sim.find_kinematic_reflections(
        k_in=k_in, Gs=Gs, lam_ang=lam_ang, z_sign=z_sign, tolerance=tolerance
    )
    detector_points: Float[Array, "M 2"] = project_on_detector(
        k_out, jnp.asarray(detector_distance)
    )
    G_allowed = Gs[allowed_indices]
    atom_positions = crystal.cart_positions[:, :3]
    intensities: Float[Array, "M"] = sim.compute_kinematic_intensities(
        positions=atom_positions, G_allowed=G_allowed
    )
    pattern = RHEEDPattern(
        G_indices=allowed_indices,
        k_out=k_out,
        detector_points=detector_points,
        intensities=intensities,
    )
    return pattern


def plot_rheed(
    rheed_pattern: RHEEDPattern,
    grid_size: Optional[int] = 200,
    interp_type: Optional[str] = "cubic",
    cmap_name: Optional[str] = "phosphor",
):
    """
    Description
    -----------
    Interpolate the RHEED spots onto a uniform grid using various methods,
    then display using the phosphor colormap.

    The parameter `interp_type` controls which interpolation method is used:
      - "cubic"    => calls griddata(..., method="cubic")
      - "linear"   => calls griddata(..., method="linear")
      - "nearest"  => calls griddata(..., method="nearest")

    Parameters
    ----------
    - `rheed_pattern` (RHEEDPattern)
        Must have `detector_points` of shape (M, 2) and
        `intensities` of shape (M,).
    - `grid_size` (int)
        Controls how many grid points in Y and Z directions.
        Default is 200.
    - `interp_type` (str)
        Which interpolation approach to use. Default is "cubic".
        The parameter `interp_type` controls which interpolation method is used:
        - "cubic"    => calls griddata(..., method="cubic")
        - "linear"   => calls griddata(..., method="linear")
        - "nearest"  => calls griddata(..., method="nearest")
    - `cmap_name` (str):
        Name for your custom phosphor colormap.
        Default is 'phosphor'.
    """
    coords = rheed_pattern.detector_points
    Y = coords[:, 0]
    Z = coords[:, 1]
    intensities = rheed_pattern.intensities
    Y_np = np.asarray(Y)
    Z_np = np.asarray(Z)
    I_np = np.asarray(intensities)
    if interp_type in ("cubic", "linear", "nearest"):
        method = interp_type
    else:
        raise ValueError(
            "interp_type must be one of: 'cubic', 'linear', or 'nearest'. "
            f"Got: {interp_type}"
        )
    y_min, y_max = float(Y_np.min()), float(Y_np.max())
    z_min, z_max = float(Z_np.min()), float(Z_np.max())
    y_lin = np.linspace(y_min, y_max, grid_size)
    z_lin = np.linspace(z_min, z_max, grid_size)
    Yg, Zg = np.meshgrid(y_lin, z_lin, indexing="xy")
    grid_points = np.column_stack([Yg.ravel(), Zg.ravel()])
    interpolated = griddata(
        points=(Y_np, Z_np), values=I_np, xi=grid_points, method=method, fill_value=0.0
    )
    intensity_grid = interpolated.reshape((grid_size, grid_size))
    phosphor_cmap = io.create_phosphor_colormap(cmap_name)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(
        intensity_grid.T,
        origin="lower",
        cmap=phosphor_cmap,
        extent=[y_min, y_max, z_min, z_max],
        aspect="equal",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Interpolated Intensity (arb. units)")
    ax.set_title(f"RHEED Pattern ({method} interpolation)")
    ax.set_xlabel("Y (Å)")
    ax.set_ylabel("Z (Å)")
    plt.tight_layout()
    plt.show()
