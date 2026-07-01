"""LAPPD housing model based on Kandemir's WCSim implementation.

Provides the "ANNIE" LAPPD model: a waterproof housing with an acrylic
window, air gap, and off-center photocathode, replacing the Default model's
bare rectangle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- Default model (bare photocathode rectangle) ----
DEFAULT_HALF_SIZE = 101.0  # mm — side = 202 mm square

# ---- ANNIE model (housed LAPPD) dimensions (mm) ----
# Housing outer box half-extents
HOUSING_HALF = (165.0, 215.0, 30.0)   # X × Y × Z (Z = radial direction)

# Photocathode half-sizes
PC_HALF = (95.75, 95.75)

# Photocathode centre in the housing local frame
PC_LOCAL = (0.0, -45.0, 3.5)   # (X, Y, Z) — offset -45 mm in Y

# Position correction ratio (shifts LAPPD radially inward)
CORRECTION = 0.965


@dataclass
class LAPPDHousing:
    """Oriented-box housing with internal photocathode.

    The housing is a rectangular box (330×430×60 mm) whose axes are
    defined relative to the tank wall:
      - local X: tangential along the tank circumference
      - local Y: vertical (parallel to +Z in the structure frame)
      - local Z: radially inward (front face normal)

    The photocathode (PC) is a smaller square inside the box, offset
    in Y by -45 mm and in Z by +3.5 mm (toward the front face).

    Attributes:
        centre:    Box centre (mm) in structure frame, after 0.965 radial correction.
        axes:      Orthonormal right-handed frame: (local_X, local_Y, local_Z).
        half:      Half-extents (mm) along each local axis.
        pc_centre: World-frame photocathode centre (mm).
        pc_normal: World-frame photocathode normal (= +local_Z, inward).
        pc_half:   Photocathode half-side lengths (mm).
    """

    centre: tuple[float, float, float]          # box centre (mm), after 0.965 correction
    axes: tuple[                                # orthonormal right-handed axes
        tuple[float, float, float],             # local X (tangential along tank wall)
        tuple[float, float, float],             # local Y (vertical = +Z in structure frame)
        tuple[float, float, float],             # local Z (radially inward = front face)
    ]
    half: tuple[float, float, float] = HOUSING_HALF  # (HX, HY, HZ) half-extents in mm

    # Pre-computed world-frame photocathode
    pc_centre: tuple[float, float, float] | None = None
    pc_normal: tuple[float, float, float] | None = None
    pc_half: tuple[float, float] = PC_HALF


def build_housing(
    cad_centre: tuple[float, float, float],
    cad_normal: tuple[float, float, float],
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> LAPPDHousing:
    """Build the housed LAPPD at a CAD candidate position.

    Parameters
    ----------
    cad_centre:
        (x, y, z) from the STEP CAD (mm).
    cad_normal:
        Radially inward unit normal at the CAD position.
    z_axis:
        Vertical direction in the structure frame (default Z-up).

    Returns
    -------
    LAPPDHousing with all fields filled.
    """
    cx, cy, cz = cad_centre
    nx, ny, nz = cad_normal

    # ---- Apply radial correction ----
    # The STEP CAD candidate positions are on the CAD cylinder, but the
    # housing box is offset inward so its front face aligns with the
    # physical support column (r ≈ 1304 mm).  CORRECTION = 0.965 scales
    # the XY position to move the box centre inward by ≈ half the box
    # thickness (30 mm).  Only XY is scaled; Z (vertical) is unchanged.
    hx = cx * CORRECTION
    hy = cy * CORRECTION
    hz = cz

    # ---- Build local axes (right-handed orthonormal) ----
    # local_Z: points radially inward (front face normal)
    local_z = (nx, ny, nz)

    # local_Y: vertical, parallel to the tank Z-axis
    local_y = z_axis

    # local_X = cross(local_Y, local_Z), then re-orthogonalise
    # This gives the tangential direction around the tank circumference.
    lx = (
        local_y[1] * local_z[2] - local_y[2] * local_z[1],
        local_y[2] * local_z[0] - local_y[0] * local_z[2],
        local_y[0] * local_z[1] - local_y[1] * local_z[0],
    )
    ll = math.sqrt(lx[0]**2 + lx[1]**2 + lx[2]**2)
    if ll > 1e-12:
        local_x = (lx[0] / ll, lx[1] / ll, lx[2] / ll)
    else:
        local_x = (1.0, 0.0, 0.0)

    # Recompute local_y = cross(local_z, local_x) for orthogonality
    local_y = (
        local_z[1] * local_x[2] - local_z[2] * local_x[1],
        local_z[2] * local_x[0] - local_z[0] * local_x[2],
        local_z[0] * local_x[1] - local_z[1] * local_x[0],
    )

    # ---- Photocathode world position ----
    # PC_LOCAL = (0, -45, 3.5) mm in the housing local frame:
    #   X: centred (0)
    #   Y: offset -45 mm (shifted downward in the housing)
    #   Z: offset +3.5 mm (shifted toward front face, inside the air gap)
    # Transform to world frame using the local-axis basis vectors.
    plx, ply, plz = PC_LOCAL
    pc_world = (
        hx + plx * local_x[0] + ply * local_y[0] + plz * local_z[0],
        hy + plx * local_x[1] + ply * local_y[1] + plz * local_z[1],
        hz + plx * local_x[2] + ply * local_y[2] + plz * local_z[2],
    )

    return LAPPDHousing(
        centre=(hx, hy, hz),
        axes=(local_x, local_y, local_z),
        pc_centre=pc_world,
        pc_normal=local_z,
    )


def compute_housing_track_length(
    pos: tuple[float, float, float],
    direc: tuple[float, float, float],
    housing: LAPPDHousing,
) -> float:
    """Maximum muon track length (m) until the ray exits the housing bounding box.

    Uses slab-method ray–oriented-box intersection.  Returns the
    exit distance * 1.05 (5 % safety margin) in **metres**, or 4.0 m
    if the ray misses the box.
    """
    cx, cy, cz = housing.centre
    ax, ay, az = housing.axes
    hx, hy, hz = housing.half
    ox, oy, oz = pos
    dx, dy, dz = direc

    t_min = -1e30
    t_max = 1e30

    for aax, aay, aaz, h in [(ax[0], ax[1], ax[2], hx),
                               (ay[0], ay[1], ay[2], hy),
                               (az[0], az[1], az[2], hz)]:
        denom = dx * aax + dy * aay + dz * aaz
        oc = (ox - cx) * aax + (oy - cy) * aay + (oz - cz) * aaz
        if abs(denom) > 1e-30:
            t0 = (-h - oc) / denom
            t1 = (h - oc) / denom
        else:
            t0 = -1e30 if (-h - oc) < 0 else 1e30
            t1 = 1e30 if (h - oc) > 0 else -1e30
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_min:
            t_min = t0
        if t1 < t_max:
            t_max = t1
        if t_min > t_max:
            return 4.0

    # t_max is the ray-exit distance in mm; convert to metres * 1.05
    track_mm = max(t_max, 0.0)
    return max(track_mm * 1.05 / 1000.0, 0.5)


def housing_to_arrays(housing: LAPPDHousing) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a LAPPDHousing into kernel arrays.

    Returns
    -------
    housing_data : ndarray (1, 16) float32
        [cx,cy,cz, ax_x,ax_y,ax_z, ay_x,ay_y,ay_z, az_x,az_y,az_z, hx,hy,hz, pad]
    annie_lappd_data : ndarray (1, 7) float32
        [pcx,pcy,pcz, pcnx,pcyn,pczn, pchalf] — same layout as lappd_data.
    """
    ax, ay, az = housing.axes
    hx, hy, hz = housing.half
    hd = np.array([
        housing.centre[0], housing.centre[1], housing.centre[2],
        ax[0], ax[1], ax[2],
        ay[0], ay[1], ay[2],
        az[0], az[1], az[2],
        hx, hy, hz,
        0.0,  # padding
    ], dtype=np.float32).reshape(1, 16)

    pc = np.array([
        housing.pc_centre[0], housing.pc_centre[1], housing.pc_centre[2],
        housing.pc_normal[0], housing.pc_normal[1], housing.pc_normal[2],
        housing.pc_half[0],
    ], dtype=np.float32).reshape(1, 7)

    return hd, pc
