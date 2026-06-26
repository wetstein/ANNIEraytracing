"""Material definitions and per-triangle PMT mesh classification.

Phase 1: material ID enum, property table, and geometric classification
of PMT body/hardware meshes.

Usage:
    from annieray.materials import MaterialID, MATERIAL_TABLE, classify_pmt_body
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class MaterialID(IntEnum):
    """Numeric IDs used in the hits ``material_id`` column (HMAT)."""
    UNKNOWN = 0
    GLASS = 1           # PMT glass bulb (non-active part)
    PHOTOCATHODE = 2    # Active detection surface (upper hemisphere)
    PVC = 3             # LUX/ETEL housing and wings (dark grey)
    TEFLON = 4          # Inner structure white covering, 8" PMT holders
    ACRYLIC = 5         # 10" PMT holders, LAPPD window
    STEEL = 6           # Inner structure frame
    BLACK_SHEET = 7     # Black tank liner / light-tight barrier


@dataclass
class MaterialProps:
    """Optical/visual properties for a material.

    Fields will be extended in later phases:
      refractive_index, absorption_length_mm, quantum_efficiency, ...
    """
    color: tuple[float, float, float]   # (R, G, B) each in [0, 1]
    is_sensitive: bool                   # active photocathode-like surface?


MATERIAL_TABLE: dict[MaterialID, MaterialProps] = {
    MaterialID.UNKNOWN:      MaterialProps(color=(0.80, 0.20, 0.20), is_sensitive=False),
    MaterialID.GLASS:        MaterialProps(color=(0.70, 0.78, 0.85), is_sensitive=False),
    MaterialID.PHOTOCATHODE: MaterialProps(color=(0.50, 0.22, 0.10), is_sensitive=True),
    MaterialID.PVC:          MaterialProps(color=(0.10, 0.10, 0.12), is_sensitive=False),
    MaterialID.TEFLON:       MaterialProps(color=(0.95, 0.95, 0.92), is_sensitive=False),
    MaterialID.ACRYLIC:      MaterialProps(color=(0.88, 0.90, 0.94), is_sensitive=False),
    MaterialID.STEEL:        MaterialProps(color=(0.50, 0.50, 0.52), is_sensitive=False),
    MaterialID.BLACK_SHEET:  MaterialProps(color=(0.02, 0.02, 0.02), is_sensitive=False),
}


# ---- PMT forward-axis definitions ----
# (forward_axis_index, sign, label)
PMT_FORWARD: dict[str, tuple[int, int]] = {
    "LUX":      (2,  1),   # +Z
    "ETEL":     (2, -1),   # -Z
    "Hamamatsu": (0,  1),  # +X
    "Watchboy":  (1,  1),  # +Y
    "Watchman":  (1,  1),  # +Y
}


# ---------------------------------------------------------------------------
# Per-triangle material classification helpers
# ---------------------------------------------------------------------------

def _face_normals_and_centers(tris: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Compute (N,3) face centers and unit normals from (N,3,3) triangle array."""
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]
    centers = (v0 + v1 + v2) / 3.0
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen
    return centers, normals


def _fit_bulb_sphere(centers: np.ndarray,
                     fwd_proj: np.ndarray,
                     rad_dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to the bulb (dome) portion of a PMT mesh.

    Uses the forward-most 15 % of faces and a least-squares sphere fit.
    This avoids the X/Z bias introduced by large non-spherical base
    features (e.g. the square base of 10-inch Watchboy tubes).

    Returns (sphere_center, sphere_radius).
    """
    # Forward-most 15 % captures the dome for all PMT shapes tested.
    fwd_thresh = np.percentile(fwd_proj, 85)
    mask = fwd_proj > fwd_thresh
    if mask.sum() < 10:
        # fallback — use all forward-half faces
        mask = fwd_proj > np.percentile(fwd_proj, 50)
    bc = centers[mask]
    # Least-squares sphere fit: solve A·x = b where
    #   A = [2*cx, 2*cy, 2*cz, 1],  x = [cx0, cy0, cz0, r² - cx0² - cy0² - cz0²]^T
    #   b = cx² + cy² + cz²
    A = np.ones((len(bc), 4), dtype=np.float64)
    A[:, :3] = 2.0 * bc
    rhs = (bc ** 2).sum(axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    sphere_center = sol[:3]
    sphere_radius = float(np.sqrt(sol[3] + (sphere_center ** 2).sum()))
    return sphere_center, sphere_radius


def classify_pmt_body(tris: np.ndarray,
                      pmt_type_name: str) -> np.ndarray:
    """Return ``(n_triangles,)`` :class:`MaterialID` array for a PMT body mesh.

    Parameters
    ----------
    tris:
        ``(M, 3, 3)`` float32 array — triangle vertex positions.
    pmt_type_name:
        One of ``"LUX"``, ``"ETEL"``, ``"Hamamatsu"``, ``"Watchboy"``,
        ``"Watchman"``.

    Classification logic
    --------------------
    1. Compute face centers and unit face normals.
    2. Fit a sphere to the forward bulb region.
    3. Faces on the **forward hemisphere** of the bulb → ``PHOTOCATHODE``.
    4. Faces on the **rear hemisphere** of the bulb → ``GLASS``.
    5. Faces far from the sphere (base / body) → ``GLASS``.
    6. For LUX/ETEL: faces with large off-axis extent not on bulb → ``PVC``.
    """
    centers, normals = _face_normals_and_centers(tris)
    fax, fsig = PMT_FORWARD.get(pmt_type_name, (2, 1))

    fwd = np.zeros(3, dtype=np.float64)
    fwd[fax] = fsig

    c = centers - centers.mean(axis=0)
    fwd_proj = c @ fwd
    rad_dist = np.linalg.norm(c - np.outer(fwd_proj, fwd), axis=1)

    sphere_center, sphere_radius = _fit_bulb_sphere(centers, fwd_proj, rad_dist)

    sc = centers - sphere_center
    dist_from_sphere = np.linalg.norm(sc, axis=1)

    # On-bulb: distance from sphere centre within 30 % of radius
    on_bulb = np.abs(dist_from_sphere - sphere_radius) < sphere_radius * 0.30

    # Normal check: dome faces should point radially away from the sphere center.
    # Cross-product normals from outward-wound triangles point outward,
    # so for a sphere they align with (centers - sphere_center).
    radial_align = (normals * (centers - sphere_center)).sum(axis=1) / dist_from_sphere
    on_bulb &= radial_align > 0.3

    # Forward hemisphere of the bulb: use face-normal orientation rather than
    # the sphere-center plane, because the physical equator (widest diameter)
    # is where normals transition from backward (n·fwd < 0) to forward (n·fwd > 0).
    # The sphere center often lies well below the physical equator (e.g. Y=56
    # vs Y=99 for 10-inch Watchboy), so (sc @ fwd) > 0 would incorrectly include
    # cylindrical-body faces.  A 1e-3 threshold discards numerical-noise faces
    # on base geometry.
    forward_hemi = (normals @ fwd) > 1e-3

    material_ids = np.full(tris.shape[0], MaterialID.GLASS, dtype=np.int32)

    # Photocathode = on bulb + forward hemisphere
    photocathode_mask = on_bulb & forward_hemi
    material_ids[photocathode_mask] = MaterialID.PHOTOCATHODE

    # For LUX/ETEL: identify PVC housing — rear end with large radial extent
    # and forward wings — flat panels perpendicular to the forward axis.
    is_lux_etel = pmt_type_name in ("LUX", "ETEL")
    if is_lux_etel:
        # Housing occupies the rear portion (negative fwd_proj) with
        # moderate-to-large radial distance.
        rear_mask = fwd_proj < np.percentile(fwd_proj, 25)
        housing_mask = rear_mask & (rad_dist > np.percentile(rad_dist, 40))
        material_ids[housing_mask] = MaterialID.PVC

        # Wings: flat forward-facing panels at very large radial extent.
        # Their normals point almost exactly forward (within 8°), and they
        # extend well beyond the cylindrical body.  These are NOT bulb faces
        # even though they happen to lie near the fitted sphere surface.
        nfwd = normals @ fwd
        wing_mask = (nfwd > 0.99) & (rad_dist > sphere_radius * 0.55)
        material_ids[wing_mask] = MaterialID.PVC

    return material_ids


def classify_pmt_hardware(hw_type_index: int,
                          n_triangles: int) -> np.ndarray:
    """Return ``(n_triangles,)`` :class:`MaterialID` for a hardware mesh.

    * Type index 4 → 8" hardware → ``TEFLON`` (white)
    * Type index 5 → 10" hardware → ``ACRYLIC`` (frosted)
    """
    if hw_type_index == 4:
        mid = MaterialID.TEFLON
    elif hw_type_index == 5:
        mid = MaterialID.ACRYLIC
    else:
        mid = MaterialID.UNKNOWN
    return np.full(n_triangles, mid, dtype=np.int32)
