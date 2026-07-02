"""GPU-accelerated analytic ray tracer using Taichi."""

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti

from annieray import gdml_parser, step_parser
from annieray import pmt_loader
from annieray import pmt_mesh
from annieray.optics import OpticalMaterial
from annieray.step_parser import LAPPD_HALF_SIZE


# ---- Component IDs (written to hits[:, 8] = HCID) ----
CID_NO_HIT = 0           # photon escaped or was absorbed
CID_INNER_STRUCTURE = 1  # hit the GDPM structure mesh (steel, acrylic, etc.)
CID_PMT = 2              # hit a PMT photocathode sphere
CID_LAPPD = 3            # hit a LAPPD photocathode rectangle
CID_TANK_WALL = 4        # hit the invisible outer tank cylinder

# ---- Detector system codes (written to hits[:, 10] = HDS) ----
DET_SYS_NONE = -1          # not a detector hit (structure, tank, or none)
DET_SYS_PMT = 0            # hit recorded against a PMT detector
DET_SYS_LAPPD_DEFAULT = 1  # hit recorded against a default bare-rectangle LAPPD
DET_SYS_LAPPD_ANNIE = 2    # hit recorded against the ANNIE housed LAPPD

# ---- Column indices for the 13-column hit array produced by the kernel ----
# These names are used throughout the code to avoid magic-number indexing.
HI = 0   # hit_flag      (1 = hit, 0 = miss)
HT = 1   # t             (path length from origin to hit, mm)
HX = 2   # hit x         (detector-frame X coordinate, mm)
HY = 3   # hit y         (detector-frame Y coordinate, mm)
HZ = 4   # hit z         (detector-frame Z coordinate, mm)
HNX = 5  # hit normal x  (outward-facing normal at hit point)
HNY = 6  # hit normal y
HNZ = 7  # hit normal z
HCID = 8 # component_id  (which geometry element was hit: 1-4 above)
HDI = 9  # detector_index(which detector in the registry; -1 = not a detector)
HDS = 10 # detector_system(which detector type: 0 = PMT, 1 = LAPPD default, 2 = LAPPD annie)
HLU = 11 # local_u      (PMT: polar angle θ from direction in rad; LAPPD: position along strips in mm)
HLV = 12 # local_v      (PMT: azimuthal angle φ in rad; LAPPD: position across strips in mm)
HMAT = 13 # material_id  (MaterialID enum value, e.g. GLASS=1, PHOTOCATHODE=2, ...)

N_HIT_COLS = 14

# Column indices in the expanded (post-kernel) hit array.
# The kernel produces N_HIT_COLS columns (0..N_HIT_COLS-1).
# trace_cherenkov appends the columns below.
H_ARRIVAL = N_HIT_COLS      # 14 — arrival_time (ns)
H_WAVELEN = N_HIT_COLS + 1  # 15 — wavelength (nm)
H_BOUNCE  = N_HIT_COLS + 2  # 16 — number of surface reflections
N_EXPANDED_COLS = N_HIT_COLS + 3  # 17

# Speed of light in vacuum (mm/ns)
C_MM_NS = 299.792458

# Default refractive index of water at ~350 nm
# Used to convert photon path length to arrival time: t_arrival = t / (C/n)
N_WATER_DEFAULT = 1.34


@dataclass
class Geometry:
    """Aggregate geometry passed to the Taichi kernel.

    Fields prefixed with mesh_/pmt_/lappd_ are GPU arrays consumed directly
    by trace_kernel().  The detectors list is the human-readable registry
    that maps detector_index → detector ID, type, position, etc.
    """
    # ---- GDML structure mesh ----
    mesh_vertices: np.ndarray   # (N, 3) float32 — triangle vertex XYZ (mm)
    mesh_triangles: np.ndarray  # (M, 3) int32   — vertex-index triplets

    # ---- PMT spheres ----
    pmt_centers: np.ndarray     # (P, 3) float32 — sphere centre in structure frame (mm)
    pmt_radii: np.ndarray       # (P,)   float32 — radius per PMT (mm)
    pmt_directions: np.ndarray  # (P, 3) float32 — inward-pointing unit normal

    # ---- Default LAPPD rectangles (bare photocathode) ----
    lappd_data: np.ndarray      # (L, 7) float32 — [cx,cy,cz, nx,ny,nz, half_size]
    lappd_strip_axes: np.ndarray  # (L, 3) float32 — strip-axis unit vector per LAPPD

    # ---- Tank cylinder bounds (used for _ray_tank_intersect) ----
    tank_radius: float  # mm
    tank_z_min: float   # mm
    tank_z_max: float   # mm

    # ---- ANNIE LAPPD housing model (when --lappd-model=annie) ----
    lappd_housing_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 16), dtype=np.float32)
    )  # (H, 16) float32 — see housing_to_arrays() for layout

    annie_lappd_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 7), dtype=np.float32)
    )  # (H, 7) float32 — photocathode rect [cx,cy,cz, nx,ny,nz, half]

    # ---- Detector registry (not used by the kernel) ----
    detectors: list = field(default_factory=list)  # list[DetectorInfo]

    # ---- Per-triangle material IDs for structure mesh ----
    mesh_material_ids: np.ndarray | None = None  # (M,) int32 — MaterialID per tri

    # ---- BVH acceleration for structure mesh ----
    bvh_node_min: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    bvh_node_max: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    bvh_node_left: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_node_right: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_start: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_end: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_n_nodes: int = 0

    # ---- PMT body meshes (triangle soup per instance type, local frame) ----
    pmt_body_tris: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (T_global, 9) float32 — 3 vertices per row, local frame
    pmt_body_mat_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (T_global,) int32 — MaterialID per triangle
    pmt_body_offsets: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 0, 0], dtype=np.int32)
    )  # (5,) int32 — start index per mesh type 0-3 + sentinel

    # ---- PMT hardware (holder) meshes ----
    pmt_hw_tris: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (T_hw, 9) float32
    pmt_hw_mat_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (T_hw,) int32 — MaterialID per triangle (TEFLON=4, ACRYLIC=5)
    pmt_hw_offsets: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0], dtype=np.int32)
    )  # (3,) int32 — start index for types 4, 5 + sentinel

    # ---- Per-PMT instance data for mesh refinement ----
    pmt_rotmats: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (P, 9) float32 — flattened R^T for local-frame transform
    pmt_mesh_types: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (P,) int32 — mesh type index per PMT (body)
    pmt_hw_types: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (P,) int32 — HW type index per PMT (-1=none, 4=8", 5=10")
    pmt_instance_pos: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )  # (P, 3) float32 — instance position (mesh centroid)
    pmt_bounding_radii: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (P,) float32 — max(body, HW) bounding sphere radius

    # ---- Surfboard (obscurant PVC panel) oriented boxes ----
    surfboard_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 16), dtype=np.float32)
    )  # (S, 16) float32 — same layout as lappd_housing_data

    # ---- LAPPD correction state ----
    lappd_corrections_baked: np.ndarray | None = None  # (1,3) float32 — last-applied dx,dy,dz


def reload_lappd_corrections(geo: Geometry):
    """Read lappd_corrections.csv and apply delta to housing/kernel data."""
    corr_path = os.path.join(os.path.dirname(__file__), "lappd_corrections.csv")
    if not os.path.exists(corr_path):
        return
    n = geo.lappd_housing_data.shape[0]
    if n == 0:
        return

    new_corrs = np.zeros((n, 3), dtype=np.float32)
    with open(corr_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            if 0 <= idx < n:
                new_corrs[idx] = [float(row["dx"]), float(row["dy"]), float(row["dz"])]

    if geo.lappd_corrections_baked is None:
        geo.lappd_corrections_baked = new_corrs.copy()
        delta = new_corrs
    else:
        delta = new_corrs - geo.lappd_corrections_baked
        geo.lappd_corrections_baked = new_corrs.copy()

    for idx in range(n):
        if np.any(delta[idx] != 0.0):
            geo.lappd_housing_data[idx, 0:3] += delta[idx]
            geo.annie_lappd_data[idx, 0:3] += delta[idx]


def build_surfboards(n: int, tank_z_min: float, tank_z_max: float) -> np.ndarray:
    """Build oriented-box data for N surfboard obscurant panels.

    Surfboards are long rectangular PVC panels (2450×280×10 mm) mounted
    vertically on the octagonal columns at the forward-most vertices of
    the tank.  They absorb photons and stop muon tracks.

    Parameters
    ----------
    n : int
        Number of surfboards (0, 1, or 3).
    tank_z_min, tank_z_max : float
        Tank Z bounds (mm) — used to centre the surfboard vertically.

    Returns
    -------
    np.ndarray
        ``(n, 16)`` float32 — oriented-box layout matching
        ``lappd_housing_data``.

    Layout of the 16 columns::

        [cx, cy, cz, ax_x,ax_y,ax_z, ay_x,ay_y,ay_z, az_x,az_y,az_z, hx,hy,hz, pad]

    The local frame convention (same as the housing)::

        local_X : tangential to the tank (width direction, 280 mm total)
        local_Y : vertical, along +Z (length direction, 2450 mm total)
        local_Z : radially inward (thickness direction, 10 mm total)
    """
    if n not in (0, 1, 3):
        raise ValueError(f"n_surfboards must be 0, 1, or 3, got {n}")
    if n == 0:
        return np.empty((0, 16), dtype=np.float32)

    column_r = 1304.0
    z_centre = (tank_z_min + tank_z_max) / 2.0
    half_x = 140.0   # 280 mm / 2
    half_y = 1225.0  # 2450 mm / 2
    half_z = 5.0     # 10 mm / 2

    # User-requested offsets applied to all surfboards
    dz_vert = -114.0  # vertical (local_Y / +Z); net = +390 - 504
    dr_rad = 65.0     # +radial inward (local_Z toward tank centre); net = +50 + 15

    if n == 1:
        angles_deg = [90.0]
    else:
        angles_deg = [45.0, 90.0, 135.0]

    rows = []
    for a_deg in angles_deg:
        a = math.radians(a_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)

        # local_Z = radially inward
        az_x = -cos_a
        az_y = -sin_a
        az_z = 0.0

        # local_Y = vertical
        ay_x = 0.0
        ay_y = 0.0
        ay_z = 1.0

        # local_X = tangential = cross(local_Y, local_Z)
        ax_x = sin_a
        ax_y = -cos_a
        ax_z = 0.0

        cx = column_r * cos_a + dr_rad * az_x
        cy = column_r * sin_a + dr_rad * az_y
        cz = z_centre + dz_vert

        rows.append([
            cx, cy, cz,
            ax_x, ax_y, ax_z,
            ay_x, ay_y, ay_z,
            az_x, az_y, az_z,
            half_x, half_y, half_z,
            0.0,  # pad
        ])

    return np.array(rows, dtype=np.float32)


def build_surfboard_housings(surfboard_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build LAPPD housing arrays positioned in front of each surfboard.

    Each housing is offset from its surfboard along local_Z (radially inward)
    and shares the same local axes (aligned straight, not rotated).  The
    three housings are staggered vertically so the leftmost (45°) sits higher,
    the rightmost (135°) lower, and the centre (90°) at mid-Z, forming a
    diagonal in the visualisation frame.

    Parameters
    ----------
    surfboard_data : np.ndarray
        ``(S, 16)`` float32 — oriented-box surfboard data from build_surfboards().

    Returns
    -------
    housing_data : np.ndarray  ``(S, 16)`` float32 — lappd_housing_data layout.
    annie_data   : np.ndarray  ``(S, 7)`` float32  — annie_lappd_data layout.
    """
    from annieray.lappd_model import HOUSING_HALF, PC_LOCAL, PC_HALF

    n = surfboard_data.shape[0]
    if n == 0:
        return np.empty((0, 16), dtype=np.float32), np.empty((0, 7), dtype=np.float32)

    housing_data = np.empty((n, 16), dtype=np.float32)
    annie_data = np.empty((n, 7), dtype=np.float32)

    gap = 50.0  # mm between surfboard back face and housing front face
    hhx, hhy, hhz = HOUSING_HALF  # (165, 215, 30)

    # Vertical stagger: leftmost (45°) higher, rightmost (135°) lower, centre at mid-Z
    z_diag_offsets = [800.0, 0.0, -800.0] if n == 3 else [0.0]

    for i in range(n):
        row = surfboard_data[i]
        cx, cy, cz = row[0], row[1], row[2]
        ax = row[3:6]
        ay = row[6:9]
        az = row[9:12]
        shz = row[14]  # surfboard half_z (thickness)

        # Offset centre along local_Z (radially inward)
        offset = shz + gap + hhz
        hcx = cx + offset * az[0]
        hcy = cy + offset * az[1]
        hcz = cz + offset * az[2]

        # Vertical stagger along local_Y (global Z)
        hcz += z_diag_offsets[i]

        # Use surfboard axes directly (aligned straight, no rotation)
        hax = ax
        hay = ay
        haz = az

        # PC centre = housing centre + PC_LOCAL in housing local frame
        plx, ply, plz = PC_LOCAL  # (0, -45, 3.5)
        pcx = hcx + plx * hax[0] + ply * hay[0] + plz * haz[0]
        pcy = hcy + plx * hax[1] + ply * hay[1] + plz * haz[1]
        pcz = hcz + plx * hax[2] + ply * hay[2] + plz * haz[2]

        housing_data[i] = [
            hcx, hcy, hcz,
            hax[0], hax[1], hax[2],
            hay[0], hay[1], hay[2],
            haz[0], haz[1], haz[2],
            hhx, hhy, hhz,
            0.0,
        ]
        annie_data[i] = [
            pcx, pcy, pcz,
            haz[0], haz[1], haz[2],  # PC normal = local_Z (inward)
            PC_HALF[0],
        ]

    return housing_data, annie_data


def build_geometry(
    gdml_path: Path,
    step_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    pmt_csv_path: Optional[Path] = None,
    lappd_indices: Optional[list[int]] = None,
    no_lappd: bool = False,
    z_offset: float = 0.0,
    lappd_model: str = "default",
    bottom_rotation_deg: float = 45.0,
    bottom_spin_deg: float = 0.0,
    det_rotation_deg: float = 22.5,
    n_surfboards: int = 0,
) -> Geometry:
    """Build a Geometry from a GDML file and optional PMT/STEP data sources.

    Stages:
      1. Parse the GDML structure mesh (always required).
      2. Load PMT positions from CSV, or from STEP manifest as fallback.
      3. Load the STEP manifest (if available) for LAPPD and tank info.
      4. Build default LAPPD rectangles from manifest candidate positions.
      5. Optionally build the ANNIE LAPPD housing model.
      6. Build the detector registry (stable ID → DetectorInfo mapping).
    """
    # ---- Stage 1: parse the GDML structure mesh ----
    # Prefer a pre-built .npz cache (fast, no XML parsing).
    cache_path = gdml_path.with_suffix(".npz").with_name(
        gdml_path.stem + "_cache.npz"
    )
    if cache_path.exists():
        data = np.load(cache_path)
        verts, tris = data["vertices"], data["triangles"]
        print(f"  Structure mesh: {len(verts)} verts, {len(tris)} tris (cached)")
    else:
        verts, tris = gdml_parser.parse_gdml(gdml_path)
    if det_rotation_deg != 0.0 and verts.shape[0] > 0:
        pmt_loader.rotate_z(verts, det_rotation_deg)

    # ---- Stage 1b: build BVH for structure mesh ----
    from annieray.bvh import build_bvh
    bvh = build_bvh(verts, tris)

    # ---- Stage 2: load PMT positions ----
    # PMTs can come from: (a) Scan CSV file, (b) STEP manifest JSON, (c) STEP raw.
    # The CSV is preferred because it has per-PMT radii and type names.
    pmt_directions = np.zeros((0, 3), dtype=np.float32)
    if pmt_csv_path and pmt_csv_path.exists():
        pmt_data = pmt_loader.load_pmts(pmt_csv_path, z_offset=z_offset,
                                         bottom_rotation_deg=bottom_rotation_deg,
                                         bottom_spin_deg=bottom_spin_deg,
                                         det_rotation_deg=det_rotation_deg)
        pmt_centers = pmt_data["centers"]
        pmt_radii = pmt_data["radii"]
        pmt_directions = pmt_data["directions"]
    elif manifest_path and manifest_path.exists():
        manifest = step_parser.ComponentManifest.from_json(manifest_path)
        pmt_centers = np.array(manifest.pmt_centers, dtype=np.float32) if manifest.pmt_centers else np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.full(len(pmt_centers), manifest.pmt_radius, dtype=np.float32) if len(pmt_centers) > 0 else np.zeros(0, dtype=np.float32)
    elif step_path:
        manifest = step_parser.parse_step(step_path)
        pmt_centers = np.array(manifest.pmt_centers, dtype=np.float32) if manifest.pmt_centers else np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.full(len(pmt_centers), manifest.pmt_radius, dtype=np.float32) if len(pmt_centers) > 0 else np.zeros(0, dtype=np.float32)
    else:
        pmt_centers = np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.zeros(0, dtype=np.float32)

    # ---- Stage 3: load the STEP manifest (LAPPD candidates and tank bounds) ----
    manifest = None
    if manifest_path and manifest_path.exists():
        manifest = step_parser.ComponentManifest.from_json(manifest_path)
    elif step_path:
        manifest = step_parser.parse_step(step_path)

    # ---- Tank bounds ----
    # Used for the infinite-cylinder tank-wall intersection and for computing
    # the Z-centre when placing the ANNIE housing on the octagon.
    # The inner-structure manifest bbox does NOT reflect the tank dimensions,
    # so we always use hardcoded defaults (10 ft diameter, from detector specs).
    tank_radius = 1524.0
    tank_z_min = 19.0
    tank_z_max = 3861.0

    # Active LAPPD positions (shared by default rectangles and housing model)
    lappd_sources: list[tuple[float, float, float]] | None = None
    if not no_lappd and manifest:
        if lappd_indices is not None and manifest.lappd_candidates:
            lappd_sources = [tuple(manifest.lappd_candidates[i].center) for i in lappd_indices
                            if i < len(manifest.lappd_candidates)]
        if lappd_sources is None:
            lappd_sources = [tuple(c) for c in manifest.lappd_centers] if manifest.lappd_centers else None
        if lappd_sources is None and manifest.lappd_candidates:
            from annieray.step_parser import DEFAULT_LAPPD_INDICES
            lappd_sources = [tuple(manifest.lappd_candidates[i].center) for i in DEFAULT_LAPPD_INDICES
                            if i < len(manifest.lappd_candidates)]

    # ---- Surfboard obscurant panels (needed before Stage 5) ----
    surfboard_data = build_surfboards(n_surfboards, tank_z_min, tank_z_max)

    # ---- Stage 4: build default LAPPD rectangles ----
    # Each LAPPD is a square photocathode rectangle at a candidate position
    # from the STEP manifest.  The ANNIE model replaces one of these with
    # the full housing (Stage 5).
    lappd_data = np.zeros((0, 7), dtype=np.float32)
    lappd_strip_axes = np.zeros((0, 3), dtype=np.float32)
    housing_source: tuple[float, float, float] | None = None
    if lappd_sources:
        # If ANNIE model, place housing on an octagon vertex (22.5° from panel centre)
        if lappd_model == "annie" and len(lappd_sources) > 0:
            tank_cz = (tank_z_min + tank_z_max) / 2.0
            # Position housing so its back (+Z) face nearly touches
            # the cylindrical column at the octagon corner (r≈1304 at 22.5°).
            # The housing is 60 mm thick; centre is 30 mm inward from column.
            z_replaced = min(lappd_sources, key=lambda s: abs(s[2] - tank_cz))[2]
            from annieray.lappd_model import CORRECTION
            column_r = 1304.0
            half_thick = 30.0
            center_r = column_r - half_thick          # 1274 mm
            input_r  = center_r / CORRECTION           # pre-correction radius
            cos22 = np.cos(np.radians(22.5))
            sin22 = np.sin(np.radians(22.5))
            housing_source = (input_r * cos22, input_r * sin22, z_replaced)
            rect_sources = [s for s in lappd_sources if s[2] != z_replaced]
        else:
            rect_sources = lappd_sources

        lappd_list = []
        strip_list = []
        for cx, cy, cz in rect_sources:
            r = np.hypot(cx, cy)
            if r > 1:
                nlx = -cx / r
                nly = -cy / r
                nlz = 0.0
            else:
                nlx, nly, nlz = 0.0, 0.0, -1.0
            lappd_list.append([cx, cy, cz, nlx, nly, nlz, LAPPD_HALF_SIZE])
            # Strip axis = vertical (Z-axis) for barrel LAPPDs
            strip_list.append([0.0, 0.0, 1.0])
        lappd_data = np.array(lappd_list, dtype=np.float32) if lappd_list else np.zeros((0, 7), dtype=np.float32)
        lappd_strip_axes = np.array(strip_list, dtype=np.float32) if strip_list else np.zeros((0, 3), dtype=np.float32)
        if det_rotation_deg != 0.0 and lappd_data.shape[0] > 0:
            pmt_loader.rotate_z(lappd_data[:, :3], det_rotation_deg)
            # Also rotate the inward-normal direction (columns 3:6)
            pmt_loader.rotate_z(lappd_data[:, 3:6], det_rotation_deg)
            # Strip axis is Z-up (0,0,1) for barrel LAPPDs — stays unchanged by Z rotation

    # ---- Stage 5: ANNIE LAPPD housing model ----
    # When --lappd-model=annie, replaces the default rectangle closest to
    # the tank mid-plane with a 5-sided waterproof box (Kandemir design)
    # containing an off-centre photocathode.
    lappd_housing_data = np.empty((0, 16), dtype=np.float32)
    annie_lappd_data = np.empty((0, 7), dtype=np.float32)
    if lappd_model == "annie" and housing_source is not None:
        from annieray.lappd_model import build_housing, housing_to_arrays

        cx, cy, cz = housing_source
        r = max(np.hypot(cx, cy), 1.0)
        normal = (-cx / r, -cy / r, 0.0)

        housing = build_housing((cx, cy, cz), normal)
        lappd_housing_data, annie_lappd_data = housing_to_arrays(housing)
        if det_rotation_deg != 0.0:
            if lappd_housing_data.shape[0] > 0:
                # Columns: 0:3=centre, 3:6=ax, 6:9=ay, 9:12=az, 12:15=half (scalars), 15=pad
                pmt_loader.rotate_z(lappd_housing_data[:, 0:3], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 3:6], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 6:9], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 9:12], det_rotation_deg)
            if annie_lappd_data.shape[0] > 0:
                pmt_loader.rotate_z(annie_lappd_data[:, 0:3], det_rotation_deg)
                pmt_loader.rotate_z(annie_lappd_data[:, 3:6], det_rotation_deg)

    # ---- Stage 5b: surfboard-mounted LAPPD housings ----
    # Builds a housing in front of each surfboard (independent of Stage 5 housing source).
    if lappd_model == "annie" and n_surfboards > 0:
        sb_housing, sb_annie = build_surfboard_housings(surfboard_data)
        if sb_housing.shape[0] > 0:
            lappd_housing_data = np.concatenate([lappd_housing_data, sb_housing], axis=0)
            annie_lappd_data = np.concatenate([annie_lappd_data, sb_annie], axis=0)

    # ---- Stage 6: build detector registry ----
    # Creates a list of DetectorInfo objects with stable IDs (WCSim TubeIDs
    # 332–463 for PMTs, 1000+ for LAPPDs, 2000+ for ANNIE LAPPD) so that
    # hit records can be mapped back to hardware regardless of array indexing.
    from annieray.detectors import build_detector_registry

    pmt_types_list: list[str] = []
    pmt_det_nums: list[int] = []
    pmt_panels_list: list[int] = []
    if pmt_csv_path and pmt_csv_path.exists():
        pmt_types_list = pmt_data.get("types", [])
        pmt_det_nums = pmt_data.get("detector_nums", [])
        pmt_panels_list = pmt_data.get("panels", [])

    detectors = build_detector_registry(
        pmt_centers=pmt_centers,
        pmt_radii=pmt_radii,
        pmt_types=pmt_types_list,
        pmt_directions=pmt_directions,
        pmt_detector_nums=pmt_det_nums,
        pmt_panels=pmt_panels_list,
        lappd_rect_data=lappd_data if lappd_data.shape[0] > 0 else None,
        lappd_housing_data=lappd_housing_data if lappd_housing_data.shape[0] > 0 else None,
        annie_lappd_data=annie_lappd_data if annie_lappd_data.shape[0] > 0 else None,
    )

    # ---- Stage 7: PMT body meshes and instance data (for mesh-based ray tracing) ----
    pmt_body_tris = np.zeros((0, 9), dtype=np.float32)
    pmt_body_mat_ids = np.zeros(0, dtype=np.int32)
    pmt_body_offsets = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    pmt_hw_tris = np.zeros((0, 9), dtype=np.float32)
    pmt_hw_mat_ids = np.zeros(0, dtype=np.int32)
    pmt_hw_offsets = np.array([0, 0, 0], dtype=np.int32)
    pmt_rotmats_arr = np.zeros((0, 9), dtype=np.float32)
    pmt_mesh_types_arr = np.zeros(0, dtype=np.int32)
    pmt_hw_types_arr = np.zeros(0, dtype=np.int32)
    pmt_instance_pos_arr = np.zeros((0, 3), dtype=np.float32)
    pmt_bounding_radii_arr = np.zeros(0, dtype=np.float32)

    if pmt_csv_path and pmt_csv_path.exists() and len(pmt_centers) > 0:
        body_meshes = pmt_mesh.load_pmt_body_meshes()
        pmt_body_tris, pmt_body_mat_ids, pmt_body_offsets = \
            pmt_mesh.build_body_tris_arrays(body_meshes)

        hw_meshes = pmt_mesh.load_pmt_hw_meshes()
        pmt_hw_tris, pmt_hw_mat_ids, pmt_hw_offsets = \
            pmt_mesh.build_hw_tris_arrays(hw_meshes)

        _mesh_types = pmt_data.get("mesh_types", np.zeros(len(pmt_centers), dtype=np.int32))
        _rotmats = pmt_data.get("rotmats", np.zeros((len(pmt_centers), 9), dtype=np.float32))
        _inst_pos = pmt_data.get("instance_positions",
                                 np.zeros((len(pmt_centers), 3), dtype=np.float32))

        pmt_mesh_types_arr = np.asarray(_mesh_types, dtype=np.int32)
        pmt_rotmats_arr = np.asarray(_rotmats, dtype=np.float32)
        pmt_instance_pos_arr = np.asarray(_inst_pos, dtype=np.float32)

        # HW type per instance: Hamamatsu→4 (8" HW), Watchboy/Watchman→5 (10" HW)
        pmt_hw_types_arr = np.full(len(pmt_centers), -1, dtype=np.int32)
        for i in range(len(pmt_centers)):
            mt = int(pmt_mesh_types_arr[i])
            if mt == 2:
                pmt_hw_types_arr[i] = 4
            elif mt == 3:
                pmt_hw_types_arr[i] = 5

        # Bounding radius = max(body_br, hw_br) per instance
        pmt_bounding_radii_arr = np.zeros(len(pmt_centers), dtype=np.float32)
        for i in range(len(pmt_centers)):
            mt = int(pmt_mesh_types_arr[i])
            md = body_meshes.get(mt)
            br = md.bounding_radius if md is not None else 0.0
            hwmt = int(pmt_hw_types_arr[i])
            if hwmt >= 0:
                hw_md = hw_meshes.get(hwmt)
                if hw_md is not None:
                    br = max(br, hw_md.bounding_radius)
            pmt_bounding_radii_arr[i] = br

        n_loaded = sum(1 for v in body_meshes.values() if v is not None)
        print(f"  PMT body meshes: {n_loaded}/4 types loaded, "
              f"{pmt_body_tris.shape[0]} total tris, "
              f"{len(pmt_centers)} instances")
        if pmt_hw_tris.shape[0] > 0:
            print(f"  PMT HW meshes: {pmt_hw_tris.shape[0]} total tris")

    # Default: all structure mesh triangles are TEFLON (ID 4)
    n_tris = tris.shape[0]
    mesh_material_ids = np.full(n_tris, 4, dtype=np.int32) if n_tris > 0 else np.zeros(0, dtype=np.int32)

    return Geometry(
        mesh_vertices=verts,
        mesh_triangles=tris,
        mesh_material_ids=mesh_material_ids,
        pmt_centers=pmt_centers,
        pmt_radii=pmt_radii,
        pmt_directions=pmt_directions,
        lappd_data=lappd_data,
        lappd_strip_axes=lappd_strip_axes,
        tank_radius=tank_radius,
        tank_z_min=tank_z_min,
        tank_z_max=tank_z_max,
        lappd_housing_data=lappd_housing_data,
        annie_lappd_data=annie_lappd_data,
        surfboard_data=surfboard_data,
        detectors=detectors,
        pmt_body_tris=pmt_body_tris,
        pmt_body_mat_ids=pmt_body_mat_ids,
        pmt_body_offsets=pmt_body_offsets,
        pmt_hw_tris=pmt_hw_tris,
        pmt_hw_mat_ids=pmt_hw_mat_ids,
        pmt_hw_offsets=pmt_hw_offsets,
        pmt_rotmats=pmt_rotmats_arr,
        pmt_mesh_types=pmt_mesh_types_arr,
        pmt_hw_types=pmt_hw_types_arr,
        pmt_instance_pos=pmt_instance_pos_arr,
        pmt_bounding_radii=pmt_bounding_radii_arr,
        # BVH
        bvh_node_min=bvh.node_min,
        bvh_node_max=bvh.node_max,
        bvh_node_left=bvh.node_left,
        bvh_node_right=bvh.node_right,
        bvh_tri_start=bvh.tri_start,
        bvh_tri_end=bvh.tri_end,
        bvh_tri_ids=bvh.tri_ids,
        bvh_n_nodes=bvh.n_nodes,
    )


# ---- Taichi helper functions (single-return pattern for Taichi compat) ----


@ti.func
def _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        lo_x, lo_y, lo_z,
                        hi_x, hi_y, hi_z):
    """Slab-method ray–AABB intersection.  Returns (hit, t_near)."""
    hit = 0
    t_near = 0.0
    tmin = -1e30
    tmax = 1e30
    ok = 1

    if ti.abs(dx) > 1e-12:
        inv_d = 1.0 / dx
        t1 = (lo_x - ox) * inv_d
        t2 = (hi_x - ox) * inv_d
        tmin = ti.max(tmin, ti.min(t1, t2))
        tmax = ti.min(tmax, ti.max(t1, t2))
    else:
        if ox < lo_x or ox > hi_x:
            ok = 0

    if ok:
        if ti.abs(dy) > 1e-12:
            inv_d = 1.0 / dy
            t1 = (lo_y - oy) * inv_d
            t2 = (hi_y - oy) * inv_d
            tmin = ti.max(tmin, ti.min(t1, t2))
            tmax = ti.min(tmax, ti.max(t1, t2))
        else:
            if oy < lo_y or oy > hi_y:
                ok = 0

    if ok:
        if ti.abs(dz) > 1e-12:
            inv_d = 1.0 / dz
            t1 = (lo_z - oz) * inv_d
            t2 = (hi_z - oz) * inv_d
            tmin = ti.max(tmin, ti.min(t1, t2))
            tmax = ti.min(tmax, ti.max(t1, t2))
        else:
            if oz < lo_z or oz > hi_z:
                ok = 0

    if ok and tmin <= tmax and tmax > 1e-6:
        hit = 1
        t_near = ti.max(tmin, 0.0)

    return hit, t_near


@ti.func
def _ray_triangle_intersect(
    ox, oy, oz, dx, dy, dz,
    v0x, v0y, v0z,
    v1x, v1y, v1z,
    v2x, v2y, v2z,
):
    hit = 0
    t_hit = 0.0
    u = 0.0
    v = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x

    a = e1x * hx + e1y * hy + e1z * hz

    if ti.abs(a) > 1e-12:
        f = 1.0 / a
        sx = ox - v0x
        sy = oy - v0y
        sz = oz - v0z

        u = f * (sx * hx + sy * hy + sz * hz)
        if 0.0 <= u <= 1.0:
            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x

            v = f * (dx * qx + dy * qy + dz * qz)
            if 0.0 <= v and u + v <= 1.0:
                t_hit = f * (e2x * qx + e2y * qy + e2z * qz)
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x
                n_len = ti.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-12:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len
                hit = 1

    return hit, t_hit, u, v, nx, ny, nz


@ti.func
def _ray_sphere_intersect(ox, oy, oz, dx, dy, dz, scx, scy, scz, radius):
    """Ray–sphere intersection.  Returns (hit, t_hit, nx, ny, nz).

    For a ray along direction (dx,dy,dz) from origin (ox,oy,oz),
    solves the quadratic |O + t*D - C|² = r².
    t0 = first intersection (entry into sphere)
    t1 = second intersection (exit from sphere)
    If the origin is inside the sphere, t0 < 0 and we use t1 instead.
    """
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    cx = ox - scx  # vector from sphere centre to ray origin
    cy = oy - scy
    cz = oz - scz

    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (cx * dx + cy * dy + cz * dz)
    c = cx * cx + cy * cy + cz * cz - radius * radius

    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)  # first intersection (entry)
        t1 = (-b + sqrt_disc) / (2.0 * a)  # second intersection (exit)

        t_hit = t0
        if t_hit < 1e-6:
            t_hit = t1  # origin inside sphere → use exit point
        if t_hit >= 1e-6:
            hx = ox + dx * t_hit - scx
            hy = oy + dy * t_hit - scy
            hz = oz + dz * t_hit - scz
            n_len = ti.sqrt(hx * hx + hy * hy + hz * hz)
            if n_len > 1e-12:
                nx = hx / n_len
                ny = hy / n_len
                nz = hz / n_len
            hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_rectangle_intersect(ox, oy, oz, dx, dy, dz, rcx, rcy, rcz, rux, ruy, ruz, half):
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    n_len = ti.sqrt(rux * rux + ruy * ruy + ruz * ruz)
    if n_len > 1e-12:
        nnx = rux / n_len
        nny = ruy / n_len
        nnz = ruz / n_len

        denom = dx * nnx + dy * nny + dz * nnz
        if ti.abs(denom) > 1e-12:
            t_candidate = ((rcx - ox) * nnx + (rcy - oy) * nny + (rcz - oz) * nnz) / denom
            if t_candidate >= 1e-6:
                hx = ox + dx * t_candidate - rcx
                hy = oy + dy * t_candidate - rcy
                hz = oz + dz * t_candidate - rcz

                upx = 1.0
                upy = 0.0
                upz = 0.0
                if ti.abs(nnx) > 0.9:
                    upx = 0.0
                    upy = 1.0
                    upz = 0.0

                ux = upy * nnz - upz * nny
                uy = upz * nnx - upx * nnz
                uz = upx * nny - upy * nnx
                u_len2 = ti.sqrt(ux * ux + uy * uy + uz * uz)
                if u_len2 > 1e-12:
                    ux /= u_len2
                    uy /= u_len2
                    uz /= u_len2

                vx = nny * uz - nnz * uy
                vy = nnz * ux - nnx * uz
                vz = nnx * uy - nny * ux

                u_hit = hx * ux + hy * uy + hz * uz
                v_hit = hx * vx + hy * vy + hz * vz

                if ti.abs(u_hit) <= half and ti.abs(v_hit) <= half:
                    t_hit = t_candidate
                    nx = nnx
                    ny = nny
                    nz = nnz
                    hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_tank_intersect(ox, oy, oz, dx, dy, dz, radius):
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    a = dx * dx + dy * dy
    if a > 1e-12:
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius

        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = ti.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = t0
            if t_hit < 1e-6:
                t_hit = t1
            if t_hit >= 1e-6:
                hx = ox + dx * t_hit
                hy = oy + dy * t_hit
                n_len = ti.sqrt(hx * hx + hy * hy)
                if n_len > 1e-12:
                    nx = hx / n_len
                    ny = hy / n_len
                    nz = 0.0
                hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_box_intersect(
    ox, oy, oz, dx, dy, dz,
    bcx, bcy, bcz,
    ax_x, ax_y, ax_z,
    ay_x, ay_y, ay_z,
    az_x, az_y, az_z,
    half_x, half_y, half_z,
):
    """Oriented-box intersection via slabs.

    Returns (hit, t_hit, is_front_face) where is_front_face is True when
    the ray enters through the +Z face (window / front of housing).
    """
    hit = 0
    t_hit = 0.0
    is_front = 0

    # Transform ray origin/direction to box local frame
    ocx = ox - bcx
    ocy = oy - bcy
    ocz = oz - bcz

    lox = ocx * ax_x + ocy * ax_y + ocz * ax_z
    loy = ocx * ay_x + ocy * ay_y + ocz * ay_z
    loz = ocx * az_x + ocy * az_y + ocz * az_z

    ldx = dx * ax_x + dy * ax_y + dz * ax_z
    ldy = dx * ay_x + dy * ay_y + dz * ay_z
    ldz = dx * az_x + dy * az_y + dz * az_z

    t_min = -1e30
    t_max = 1e30
    entry_face = 0  # 1 = +Z, 2 = -Z, 3 = X, 4 = Y

    _ok = 1

    # X slabs
    if ti.abs(ldx) < 1e-12:
        if lox < -half_x or lox > half_x:
            _ok = 0
    else:
        t1 = (-half_x - lox) / ldx
        t2 = (half_x - lox) / ldx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 3
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    # Y slabs
    if _ok and ti.abs(ldy) < 1e-12:
        if loy < -half_y or loy > half_y:
            _ok = 0
    elif _ok:
        t1 = (-half_y - loy) / ldy
        t2 = (half_y - loy) / ldy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 4
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    # Z slabs
    if _ok and ti.abs(ldz) < 1e-12:
        if loz < -half_z or loz > half_z:
            _ok = 0
    elif _ok:
        t1 = (-half_z - loz) / ldz
        t2 = (half_z - loz) / ldz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 2 if ldz > 0 else 1  # entering -Z or +Z face
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    if _ok and t_min >= 1e-6:
        hit = 1
        t_hit = t_min
        is_front = 1 if entry_face == 1 else 0

    return hit, t_hit, is_front


@ti.func
def _box_hit_normal(hx, hy, hz, cx, cy, cz,
                    ax_x, ax_y, ax_z,
                    ay_x, ay_y, ay_z,
                    az_x, az_y, az_z,
                    hhx, hhy, hhz):
    """Outward-facing unit normal at a point on an oriented box surface.

    Transforms the hit point to the box local frame, determines which
    face was hit (the one closest to its half-extent), and returns the
    outward-facing normal in world coordinates.
    """
    dx = hx - cx
    dy = hy - cy
    dz = hz - cz
    lx = dx * ax_x + dy * ax_y + dz * ax_z
    ly = dx * ay_x + dy * ay_y + dz * ay_z
    lz = dx * az_x + dy * az_y + dz * az_z

    d_x = ti.abs(ti.abs(lx) - hhx)
    d_y = ti.abs(ti.abs(ly) - hhy)
    d_z = ti.abs(ti.abs(lz) - hhz)

    nx_l = 0.0
    ny_l = 0.0
    nz_l = 0.0
    if d_x <= d_y and d_x <= d_z:
        nx_l = 1.0 if lx > 0 else -1.0
    elif d_y <= d_z:
        ny_l = 1.0 if ly > 0 else -1.0
    else:
        nz_l = 1.0 if lz > 0 else -1.0

    wx = nx_l * ax_x + ny_l * ay_x + nz_l * az_x
    wy = nx_l * ax_y + ny_l * ay_y + nz_l * az_y
    wz = nx_l * ax_z + ny_l * ay_z + nz_l * az_z
    return wx, wy, wz


# ---- Local-coordinate helpers (for use in the kernel loop) ----


@ti.func
def _pmt_local_coords(nx, ny, nz, pdx, pdy, pdz):
    """Return (theta, phi) of hit on PMT hemisphere.

    nx,ny,nz = outward-facing normal at hit point (sphere surface) = (hit-centre)/r.
    pdx,pdy,pdz = PMT direction (unit, pointing toward detector centre).

    theta = polar angle from direction axis (0 at pole, pi/2 at equator).
    phi   = azimuthal angle around direction axis (rad).
    """
    # Outward normal at hit = (nx,ny,nz)
    # Polar angle from PMT direction: cos(theta) = n · dir
    cos_theta = nx * pdx + ny * pdy + nz * pdz
    if cos_theta > 1.0:
        cos_theta = 1.0
    if cos_theta < -1.0:
        cos_theta = -1.0
    theta = ti.acos(cos_theta)

    # Project outward normal onto plane perpendicular to direction
    px = nx - pdx * cos_theta
    py = ny - pdy * cos_theta
    pz = nz - pdz * cos_theta
    p_len = ti.sqrt(px * px + py * py + pz * pz)

    phi = 0.0
    if p_len > 1e-12:
        # Build orthonormal basis around direction for phi measurement
        up_x = 0.0
        up_y = 0.0
        up_z = 1.0
        if ti.abs(pdz) > 0.9:
            up_x = 1.0
            up_y = 0.0
            up_z = 0.0

        v1x = up_y * pdz - up_z * pdy
        v1y = up_z * pdx - up_x * pdz
        v1z = up_x * pdy - up_y * pdx
        v1_len = ti.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if v1_len > 1e-12:
            v1x /= v1_len
            v1y /= v1_len
            v1z /= v1_len

        v2x = pdy * v1z - pdz * v1y
        v2y = pdz * v1x - pdx * v1z
        v2z = pdx * v1y - pdy * v1x

        # Normalise projected vector and get phi
        px /= p_len
        py /= p_len
        pz /= p_len
        phi = ti.atan2(px * v2x + py * v2y + pz * v2z,
                       px * v1x + py * v1y + pz * v1z)

    return theta, phi


@ti.func
def _lappd_local_coords(hx, hy, hz, lcx, lcy, lcz, sx, sy, sz, nx, ny, nz):
    """Return (u, v) in mm for a LAPPD hit in strip-aligned coordinates.

    hx,hy,hz = hit position.
    lcx,lcy,lcz = LAPPD centre.
    sx,sy,sz = strip axis unit vector (vertical).
    nx,ny,nz = normal (inward-facing).
    u = position along strip axis (mm).
    v = position perpendicular to strip axis (mm).
    """
    relx = hx - lcx
    rely = hy - lcy
    relz = hz - lcz
    u = relx * sx + rely * sy + relz * sz
    # Perpendicular axis = cross(normal, strip_axis)
    px = ny * sz - nz * sy
    py = nz * sx - nx * sz
    pz = nx * sy - ny * sx
    p_len = ti.sqrt(px * px + py * py + pz * pz)
    v = 0.0
    if p_len > 1e-12:
        v = (relx * px + rely * py + relz * pz) / p_len
    return u, v


# ---- GPU kernel ----
#
# Each GPU thread processes one photon independently.  Every thread tests
# its photon against ALL geometry elements (triangles, spheres, rectangles,
# boxes, tank cylinder) and keeps the nearest intersection (smallest t).
#
# The kernel writes a fixed-width 13-column row per photon:
#   [hit_flag, t, x,y,z, nx,ny,nz, component_id, detector_index,
#    detector_system, local_u, local_v]
#
# Tracing order (later = higher priority for same t):
#   1. Structure mesh (triangles)            → component_id = 1
#   2. PMT spheres (fallback)                → component_id = 2
#   3. PMT mesh refinement (override sphere) → component_id = 2
#   4. Default LAPPD rectangles              → component_id = 3
#   5. ANNIE housing box (absorbs side/back) → kills photon (component_id = 0)
#   6. ANNIE LAPPD photocathode rectangle    → component_id = 3
#   7. Tank wall (infinite cylinder)         → component_id = 4


@ti.kernel
def trace_kernel(
    origins: ti.types.ndarray(ndim=2),
    directions: ti.types.ndarray(ndim=2),
    mesh_vertices: ti.types.ndarray(ndim=2),
    mesh_triangles: ti.types.ndarray(ndim=2),
    mesh_material_ids: ti.types.ndarray(ndim=1),
    bvh_node_min: ti.types.ndarray(ndim=2),
    bvh_node_max: ti.types.ndarray(ndim=2),
    bvh_node_left: ti.types.ndarray(ndim=1),
    bvh_node_right: ti.types.ndarray(ndim=1),
    bvh_tri_start: ti.types.ndarray(ndim=1),
    bvh_tri_end: ti.types.ndarray(ndim=1),
    bvh_tri_ids: ti.types.ndarray(ndim=1),
    bvh_n_nodes: ti.i32,
    pmt_centers: ti.types.ndarray(ndim=2),
    pmt_radii: ti.types.ndarray(ndim=1),
    pmt_dirs: ti.types.ndarray(ndim=2),
    pmt_body_tris: ti.types.ndarray(ndim=2),
    pmt_body_mat_ids: ti.types.ndarray(ndim=1),
    pmt_body_offsets: ti.types.ndarray(ndim=1),
    pmt_rotmats: ti.types.ndarray(ndim=2),
    pmt_mesh_types: ti.types.ndarray(ndim=1),
    pmt_hw_types: ti.types.ndarray(ndim=1),
    pmt_hw_tris: ti.types.ndarray(ndim=2),
    pmt_hw_mat_ids: ti.types.ndarray(ndim=1),
    pmt_hw_offsets: ti.types.ndarray(ndim=1),
    pmt_instance_pos: ti.types.ndarray(ndim=2),
    pmt_bounding_radii: ti.types.ndarray(ndim=1),
    lappd_data: ti.types.ndarray(ndim=2),
    lappd_strip: ti.types.ndarray(ndim=2),
    tank_radius: ti.f32,
    tank_z_min: ti.f32,
    tank_z_max: ti.f32,
    housing_data: ti.types.ndarray(ndim=2),
    annie_lappd_data: ti.types.ndarray(ndim=2),
    surfboard_data: ti.types.ndarray(ndim=2),
    hits: ti.types.ndarray(ndim=2),
):
    n_rays = origins.shape[0]
    n_tris = mesh_triangles.shape[0]
    n_pmts = pmt_centers.shape[0]
    n_lappds = lappd_data.shape[0]
    n_housings = housing_data.shape[0]
    n_surfboards = surfboard_data.shape[0]
    n_body_tris = pmt_body_tris.shape[0]
    n_hw_tris = pmt_hw_tris.shape[0]

    for i in range(n_rays):
        ox = origins[i, 0]
        oy = origins[i, 1]
        oz = origins[i, 2]
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]

        inv_len = 1.0 / ti.sqrt(dx * dx + dy * dy + dz * dz)
        dx *= inv_len
        dy *= inv_len
        dz *= inv_len

        best_t = 1e30
        best_hit = CID_NO_HIT
        best_x = 0.0
        best_y = 0.0
        best_z = 0.0
        best_nx = 0.0
        best_ny = 0.0
        best_nz = 0.0
        best_det_idx = -1
        best_det_sys = DET_SYS_NONE
        best_lu = 0.0
        best_lv = 0.0
        best_mat = 0  # MaterialID

        # ---- BVH-accelerated structure mesh (replaces brute-force loop) ----
        if bvh_n_nodes > 0:
            stack = ti.Vector([-1]*32, dt=ti.i32)
            stack[0] = bvh_n_nodes - 1  # root
            sp = 0

            while sp >= 0:
                node = stack[sp]
                sp -= 1

                if node < 0:
                    continue

                if bvh_node_left[node] == -1:  # leaf
                    for idx in range(bvh_tri_start[node], bvh_tri_end[node]):
                        t = bvh_tri_ids[idx]
                        i0 = mesh_triangles[t, 0]
                        i1 = mesh_triangles[t, 1]
                        i2 = mesh_triangles[t, 2]

                        v0x = mesh_vertices[i0, 0]
                        v0y = mesh_vertices[i0, 1]
                        v0z = mesh_vertices[i0, 2]
                        v1x = mesh_vertices[i1, 0]
                        v1y = mesh_vertices[i1, 1]
                        v1z = mesh_vertices[i1, 2]
                        v2x = mesh_vertices[i2, 0]
                        v2y = mesh_vertices[i2, 1]
                        v2z = mesh_vertices[i2, 2]

                        h, th, _u, _v, nx, ny, nz = _ray_triangle_intersect(
                            ox, oy, oz, dx, dy, dz,
                            v0x, v0y, v0z,
                            v1x, v1y, v1z,
                            v2x, v2y, v2z,
                        )

                        if h and th > 1e-6 and th < best_t:
                            best_t = th
                            best_hit = CID_INNER_STRUCTURE
                            best_x = ox + dx * th
                            best_y = oy + dy * th
                            best_z = oz + dz * th
                            best_nx = nx
                            best_ny = ny
                            best_nz = nz
                            best_det_idx = -1
                            best_det_sys = DET_SYS_NONE
                            best_mat = mesh_material_ids[t] if mesh_material_ids.shape[0] > 0 else 0
                else:  # internal node — test children, push near-first
                    ln = bvh_node_left[node]
                    hl, tl = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        bvh_node_min[ln, 0], bvh_node_min[ln, 1], bvh_node_min[ln, 2],
                        bvh_node_max[ln, 0], bvh_node_max[ln, 1], bvh_node_max[ln, 2])

                    rn = bvh_node_right[node]
                    hr, tr = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        bvh_node_min[rn, 0], bvh_node_min[rn, 1], bvh_node_min[rn, 2],
                        bvh_node_max[rn, 0], bvh_node_max[rn, 1], bvh_node_max[rn, 2])

                    if hl and hr:
                        if tl < tr:
                            if tr < best_t:
                                sp += 1
                                stack[sp] = rn
                            if tl < best_t:
                                sp += 1
                                stack[sp] = ln
                        else:
                            if tl < best_t:
                                sp += 1
                                stack[sp] = ln
                            if tr < best_t:
                                sp += 1
                                stack[sp] = rn
                    elif hl and tl < best_t:
                        sp += 1
                        stack[sp] = ln
                    elif hr and tr < best_t:
                        sp += 1
                        stack[sp] = rn

        for p in range(n_pmts):
            # If this PMT has loaded mesh data, skip the analytic sphere entirely.
            # The mesh refinement loop below will handle hit detection via a
            # bounding-sphere pre-filter + local-frame triangle test.
            pmt_mt = ti.cast(pmt_mesh_types[p], ti.i32)
            has_mesh = 0
            if n_body_tris > 0 and pmt_mt >= 0:
                _s = pmt_body_offsets[pmt_mt]
                _e = pmt_body_offsets[pmt_mt + 1]
                if _e > _s and pmt_bounding_radii[p] > 0.0:
                    has_mesh = 1

            if not has_mesh:
                pcx = pmt_centers[p, 0]
                pcy = pmt_centers[p, 1]
                pcz = pmt_centers[p, 2]
                pr = pmt_radii[p]

                hit, t_hit, nx, ny, nz = _ray_sphere_intersect(
                    ox, oy, oz, dx, dy, dz,
                    pcx, pcy, pcz,
                    pr,
                )

                if hit and t_hit > 1e-6 and t_hit < best_t:
                    hpx = ox + dx * t_hit - pcx
                    hpy = oy + dy * t_hit - pcy
                    hpz = oz + dz * t_hit - pcz
                    pdx = pmt_dirs[p, 0]
                    pdy = pmt_dirs[p, 1]
                    pdz = pmt_dirs[p, 2]
                    # Hemisphere check: dot(normal, direction) > 0 for front face
                    dot_fwd = hpx * pdx + hpy * pdy + hpz * pdz
                    if dot_fwd > 0.0:
                        best_t = t_hit
                        best_hit = CID_PMT
                        best_x = ox + dx * t_hit
                        best_y = oy + dy * t_hit
                        best_z = oz + dz * t_hit
                        best_nx = nx
                        best_ny = ny
                        best_nz = nz
                        best_det_idx = p
                        best_det_sys = DET_SYS_PMT
                        theta, phi = _pmt_local_coords(nx, ny, nz, pdx, pdy, pdz)
                        best_lu = theta
                        best_lv = phi
                        best_mat = 2  # PHOTOCATHODE (forward hemisphere)

        # ---- PMT mesh refinement (override sphere hits with accurate mesh) ----
        if n_body_tris > 0 and n_pmts > 0:
            for p in range(n_pmts):
                mt = ti.cast(pmt_mesh_types[p], ti.i32)
                if mt < 0:
                    continue
                t_start = pmt_body_offsets[mt]
                t_end = pmt_body_offsets[mt + 1]
                if t_end <= t_start:
                    continue

                ipx = pmt_instance_pos[p, 0]
                ipy = pmt_instance_pos[p, 1]
                ipz = pmt_instance_pos[p, 2]
                br = pmt_bounding_radii[p]
                if br <= 0.0:
                    continue

                # Bounding sphere pre-filter (dir is normalized → a = 1)
                ocx = ox - ipx
                ocy = oy - ipy
                ocz = oz - ipz
                b_half = ocx * dx + ocy * dy + ocz * dz
                c = ocx * ocx + ocy * ocy + ocz * ocz - br * br
                disc = b_half * b_half - c
                if disc < 0.0:
                    continue
                sqrt_disc = ti.sqrt(disc)
                t0 = -b_half - sqrt_disc
                t1 = -b_half + sqrt_disc
                t_enter = t0 if t0 > 0.0 else t1
                if t_enter < 1e-6 or t_enter >= best_t:
                    continue

                # Load rotation matrix (flattened R^T, row-major)
                rm0 = pmt_rotmats[p, 0]
                rm1 = pmt_rotmats[p, 1]
                rm2 = pmt_rotmats[p, 2]
                rm3 = pmt_rotmats[p, 3]
                rm4 = pmt_rotmats[p, 4]
                rm5 = pmt_rotmats[p, 5]
                rm6 = pmt_rotmats[p, 6]
                rm7 = pmt_rotmats[p, 7]
                rm8 = pmt_rotmats[p, 8]

                # Transform ray to local frame: v_local = R^T · v_global
                lox = ocx * rm0 + ocy * rm1 + ocz * rm2
                loy = ocx * rm3 + ocy * rm4 + ocz * rm5
                loz = ocx * rm6 + ocy * rm7 + ocz * rm8

                ldx = dx * rm0 + dy * rm1 + dz * rm2
                ldy = dx * rm3 + dy * rm4 + dz * rm5
                ldz = dx * rm6 + dy * rm7 + dz * rm8

                for t in range(t_start, t_end):
                    v0x = pmt_body_tris[t, 0]
                    v0y = pmt_body_tris[t, 1]
                    v0z = pmt_body_tris[t, 2]
                    v1x = pmt_body_tris[t, 3]
                    v1y = pmt_body_tris[t, 4]
                    v1z = pmt_body_tris[t, 5]
                    v2x = pmt_body_tris[t, 6]
                    v2y = pmt_body_tris[t, 7]
                    v2z = pmt_body_tris[t, 8]

                    _hit, _t, _u, _v, _nx, _ny, _nz = _ray_triangle_intersect(
                        lox, loy, loz, ldx, ldy, ldz,
                        v0x, v0y, v0z,
                        v1x, v1y, v1z,
                        v2x, v2y, v2z,
                    )

                    if _hit and _t > 1e-6 and _t < best_t and pmt_body_mat_ids[t] == 2:
                        # Hit position in local frame
                        hx_l = lox + ldx * _t
                        hy_l = loy + ldy * _t
                        hz_l = loz + ldz * _t

                        # Transform normal to world frame: n_world = R · n_local
                        # R is (R^T)^T, stored as [rm0, rm3, rm6] row 0, etc.
                        wnx = _nx * rm0 + _ny * rm3 + _nz * rm6
                        wny = _nx * rm1 + _ny * rm4 + _nz * rm7
                        wnz = _nx * rm2 + _ny * rm5 + _nz * rm8
                        n_len = ti.sqrt(wnx * wnx + wny * wny + wnz * wnz)
                        if n_len > 1e-12:
                            wnx /= n_len
                            wny /= n_len
                            wnz /= n_len

                        best_t = _t
                        # World hit = instance_pos + R · h_local
                        best_x = ipx + hx_l * rm0 + hy_l * rm3 + hz_l * rm6
                        best_y = ipy + hx_l * rm1 + hy_l * rm4 + hz_l * rm7
                        best_z = ipz + hx_l * rm2 + hy_l * rm5 + hz_l * rm8
                        best_nx = wnx
                        best_ny = wny
                        best_nz = wnz
                        best_hit = CID_PMT
                        best_det_idx = p
                        best_det_sys = DET_SYS_PMT
                        best_mat = 2
                        pdx = pmt_dirs[p, 0]
                        pdy = pmt_dirs[p, 1]
                        pdz = pmt_dirs[p, 2]
                        theta, phi = _pmt_local_coords(wnx, wny, wnz, pdx, pdy, pdz)
                        best_lu = theta
                        best_lv = phi

        # ---- PMT hardware (holder) mesh refinement ----
        if n_hw_tris > 0 and n_pmts > 0:
            for p in range(n_pmts):
                hwmt = ti.cast(pmt_hw_types[p], ti.i32)
                if hwmt < 4:
                    continue
                hw_start = pmt_hw_offsets[hwmt - 4]
                hw_end = pmt_hw_offsets[hwmt - 3]
                if hw_end <= hw_start:
                    continue

                ipx = pmt_instance_pos[p, 0]
                ipy = pmt_instance_pos[p, 1]
                ipz = pmt_instance_pos[p, 2]
                br = pmt_bounding_radii[p]
                if br <= 0.0:
                    continue

                ocx = ox - ipx
                ocy = oy - ipy
                ocz = oz - ipz
                b_half = ocx * dx + ocy * dy + ocz * dz
                c = ocx * ocx + ocy * ocy + ocz * ocz - br * br
                disc = b_half * b_half - c
                if disc < 0.0:
                    continue
                sqrt_disc = ti.sqrt(disc)
                t0 = -b_half - sqrt_disc
                t1 = -b_half + sqrt_disc
                t_enter = t0 if t0 > 0.0 else t1
                if t_enter < 1e-6 or t_enter >= best_t:
                    continue

                rm0 = pmt_rotmats[p, 0]
                rm1 = pmt_rotmats[p, 1]
                rm2 = pmt_rotmats[p, 2]
                rm3 = pmt_rotmats[p, 3]
                rm4 = pmt_rotmats[p, 4]
                rm5 = pmt_rotmats[p, 5]
                rm6 = pmt_rotmats[p, 6]
                rm7 = pmt_rotmats[p, 7]
                rm8 = pmt_rotmats[p, 8]

                lox = ocx * rm0 + ocy * rm1 + ocz * rm2
                loy = ocx * rm3 + ocy * rm4 + ocz * rm5
                loz = ocx * rm6 + ocy * rm7 + ocz * rm8
                ldx = dx * rm0 + dy * rm1 + dz * rm2
                ldy = dx * rm3 + dy * rm4 + dz * rm5
                ldz = dx * rm6 + dy * rm7 + dz * rm8

                for t in range(hw_start, hw_end):
                    v0x = pmt_hw_tris[t, 0]
                    v0y = pmt_hw_tris[t, 1]
                    v0z = pmt_hw_tris[t, 2]
                    v1x = pmt_hw_tris[t, 3]
                    v1y = pmt_hw_tris[t, 4]
                    v1z = pmt_hw_tris[t, 5]
                    v2x = pmt_hw_tris[t, 6]
                    v2y = pmt_hw_tris[t, 7]
                    v2z = pmt_hw_tris[t, 8]

                    _hit, _t, _u, _v, _nx, _ny, _nz = _ray_triangle_intersect(
                        lox, loy, loz, ldx, ldy, ldz,
                        v0x, v0y, v0z,
                        v1x, v1y, v1z,
                        v2x, v2y, v2z,
                    )

                    if _hit and _t > 1e-6 and _t < best_t:
                        hx_l = lox + ldx * _t
                        hy_l = loy + ldy * _t
                        hz_l = loz + ldz * _t

                        wnx = _nx * rm0 + _ny * rm3 + _nz * rm6
                        wny = _nx * rm1 + _ny * rm4 + _nz * rm7
                        wnz = _nx * rm2 + _ny * rm5 + _nz * rm8
                        n_len = ti.sqrt(wnx * wnx + wny * wny + wnz * wnz)
                        if n_len > 1e-12:
                            wnx /= n_len
                            wny /= n_len
                            wnz /= n_len

                        best_t = _t
                        best_x = ipx + hx_l * rm0 + hy_l * rm3 + hz_l * rm6
                        best_y = ipy + hx_l * rm1 + hy_l * rm4 + hz_l * rm7
                        best_z = ipz + hx_l * rm2 + hy_l * rm5 + hz_l * rm8
                        best_nx = wnx
                        best_ny = wny
                        best_nz = wnz
                        best_hit = CID_INNER_STRUCTURE
                        best_det_idx = -1
                        best_det_sys = DET_SYS_NONE
                        best_mat = pmt_hw_mat_ids[t]

        for l in range(n_lappds):
            lcx = lappd_data[l, 0]
            lcy = lappd_data[l, 1]
            lcz = lappd_data[l, 2]
            lux = lappd_data[l, 3]
            luy = lappd_data[l, 4]
            luz = lappd_data[l, 5]
            lhalf = lappd_data[l, 6]
            sx = lappd_strip[l, 0]
            sy = lappd_strip[l, 1]
            sz = lappd_strip[l, 2]

            hit, t_hit, nx, ny, nz = _ray_rectangle_intersect(
                ox, oy, oz, dx, dy, dz,
                lcx, lcy, lcz,
                lux, luy, luz,
                lhalf,
            )

            if hit and t_hit > 1e-6 and t_hit < best_t:
                best_t = t_hit
                best_hit = CID_LAPPD
                best_x = ox + dx * t_hit
                best_y = oy + dy * t_hit
                best_z = oz + dz * t_hit
                best_nx = nx
                best_ny = ny
                best_nz = nz
                best_det_idx = n_pmts + l
                best_det_sys = DET_SYS_LAPPD_DEFAULT
                uu, vv = _lappd_local_coords(
                    best_x, best_y, best_z,
                    lcx, lcy, lcz,
                    sx, sy, sz,
                    lux, luy, luz,
                )
                best_lu = uu
                best_lv = vv
                best_mat = 5  # ACRYLIC (LAPPD window)

        # ---- ANNIE LAPPD housing model (if present) ----
        for h in range(n_housings):
            hc_x, hc_y, hc_z = housing_data[h, 0], housing_data[h, 1], housing_data[h, 2]
            a_x_x, a_x_y, a_x_z = housing_data[h, 3], housing_data[h, 4], housing_data[h, 5]
            a_y_x, a_y_y, a_y_z = housing_data[h, 6], housing_data[h, 7], housing_data[h, 8]
            a_z_x, a_z_y, a_z_z = housing_data[h, 9], housing_data[h, 10], housing_data[h, 11]
            h_hx, h_hy, h_hz = housing_data[h, 12], housing_data[h, 13], housing_data[h, 14]

            box_hit, t_box, is_front = _ray_box_intersect(
                ox, oy, oz, dx, dy, dz,
                hc_x, hc_y, hc_z,
                a_x_x, a_x_y, a_x_z,
                a_y_x, a_y_y, a_y_z,
                a_z_x, a_z_y, a_z_z,
                h_hx, h_hy, h_hz,
            )

            if box_hit and t_box > 1e-6 and t_box < best_t and not is_front:
                # Absorbed by housing side/back wall
                best_t = t_box
                best_hit = CID_NO_HIT
                best_x = ox + dx * t_box
                best_y = oy + dy * t_box
                best_z = oz + dz * t_box
                best_nx = 0.0
                best_ny = 0.0
                best_nz = 0.0
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE

            # ANNIE LAPPD photocathode (inside housing, at front face)
            apc_x, apc_y, apc_z = annie_lappd_data[h, 0], annie_lappd_data[h, 1], annie_lappd_data[h, 2]
            apc_nx, apc_ny, apc_nz = annie_lappd_data[h, 3], annie_lappd_data[h, 4], annie_lappd_data[h, 5]
            apc_half = annie_lappd_data[h, 6]

            pc_hit, t_pc, pc_nx, pc_ny, pc_nz = _ray_rectangle_intersect(
                ox, oy, oz, dx, dy, dz,
                apc_x, apc_y, apc_z,
                apc_nx, apc_ny, apc_nz,
                apc_half,
            )

            if pc_hit and t_pc > 1e-6 and t_pc < best_t:
                best_t = t_pc
                best_hit = CID_LAPPD
                best_x = ox + dx * t_pc
                best_y = oy + dy * t_pc
                best_z = oz + dz * t_pc
                best_nx = pc_nx
                best_ny = pc_ny
                best_nz = pc_nz
                best_det_idx = n_pmts + n_lappds + h
                best_det_sys = DET_SYS_LAPPD_ANNIE
                uu, vv = _lappd_local_coords(
                    best_x, best_y, best_z,
                    apc_x, apc_y, apc_z,
                    a_y_x, a_y_y, a_y_z,
                    apc_nx, apc_ny, apc_nz,
                )
                best_lu = uu
                best_lv = vv
                best_mat = 2  # PHOTOCATHODE (ANNIE LAPPD)

        # ---- Surfboard obscurant panels (PVC material, configurable optics) ----
        for s in range(n_surfboards):
            scx = surfboard_data[s, 0]
            scy = surfboard_data[s, 1]
            scz = surfboard_data[s, 2]
            ax_x = surfboard_data[s, 3]
            ax_y = surfboard_data[s, 4]
            ax_z = surfboard_data[s, 5]
            ay_x = surfboard_data[s, 6]
            ay_y = surfboard_data[s, 7]
            ay_z = surfboard_data[s, 8]
            az_x = surfboard_data[s, 9]
            az_y = surfboard_data[s, 10]
            az_z = surfboard_data[s, 11]
            shx = surfboard_data[s, 12]
            shy = surfboard_data[s, 13]
            shz = surfboard_data[s, 14]

            sb_hit, t_sb, _ = _ray_box_intersect(
                ox, oy, oz, dx, dy, dz,
                scx, scy, scz,
                ax_x, ax_y, ax_z,
                ay_x, ay_y, ay_z,
                az_x, az_y, az_z,
                shx, shy, shz,
            )

            if sb_hit and t_sb > 1e-6 and t_sb < best_t:
                best_t = t_sb
                best_hit = CID_INNER_STRUCTURE
                best_x = ox + dx * t_sb
                best_y = oy + dy * t_sb
                best_z = oz + dz * t_sb
                best_nx, best_ny, best_nz = _box_hit_normal(
                    best_x, best_y, best_z,
                    scx, scy, scz,
                    ax_x, ax_y, ax_z,
                    ay_x, ay_y, ay_z,
                    az_x, az_y, az_z,
                    shx, shy, shz,
                )
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE
                best_mat = 3  # MaterialID.PVC

        hit, t_hit, nx, ny, nz = _ray_tank_intersect(
            ox, oy, oz, dx, dy, dz,
            tank_radius,
        )
        if hit and t_hit > 1e-6 and t_hit < best_t:
            hit_z = oz + dz * t_hit
            if tank_z_min <= hit_z <= tank_z_max:
                best_t = t_hit
                best_hit = CID_TANK_WALL
                best_x = ox + dx * t_hit
                best_y = oy + dy * t_hit
                best_z = hit_z
                best_nx = nx
                best_ny = ny
                best_nz = nz
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE
                best_mat = 7  # BLACK_SHEET

        hits[i, HI] = 1.0 if best_hit != CID_NO_HIT else 0.0
        hits[i, HT] = best_t
        hits[i, HX] = best_x
        hits[i, HY] = best_y
        hits[i, HZ] = best_z
        hits[i, HNX] = best_nx
        hits[i, HNY] = best_ny
        hits[i, HNZ] = best_nz
        hits[i, HCID] = best_hit * 1.0
        hits[i, HDI] = best_det_idx * 1.0
        hits[i, HDS] = best_det_sys * 1.0
        hits[i, HLU] = best_lu
        hits[i, HLV] = best_lv
        hits[i, HMAT] = best_mat * 1.0


def trace_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    geometry: Geometry,
) -> np.ndarray:
    n_rays = origins.shape[0]
    hits = np.zeros((n_rays, N_HIT_COLS), dtype=np.float32)
    # Default detector_index = -1, detector_system = -1
    hits[:, HDI] = -1.0
    hits[:, HDS] = DET_SYS_NONE * 1.0

    trace_kernel(
        origins,
        directions,
        geometry.mesh_vertices,
        geometry.mesh_triangles.astype(np.int32),
        geometry.mesh_material_ids if geometry.mesh_material_ids is not None
        else np.zeros(0, dtype=np.int32),
        geometry.bvh_node_min,
        geometry.bvh_node_max,
        geometry.bvh_node_left,
        geometry.bvh_node_right,
        geometry.bvh_tri_start,
        geometry.bvh_tri_end,
        geometry.bvh_tri_ids,
        geometry.bvh_n_nodes,
        geometry.pmt_centers,
        geometry.pmt_radii,
        geometry.pmt_directions,
        geometry.pmt_body_tris,
        geometry.pmt_body_mat_ids,
        geometry.pmt_body_offsets,
        geometry.pmt_rotmats,
        geometry.pmt_mesh_types,
        geometry.pmt_hw_types,
        geometry.pmt_hw_tris,
        geometry.pmt_hw_mat_ids,
        geometry.pmt_hw_offsets,
        geometry.pmt_instance_pos,
        geometry.pmt_bounding_radii,
        geometry.lappd_data,
        geometry.lappd_strip_axes,
        geometry.tank_radius,
        geometry.tank_z_min,
        geometry.tank_z_max,
        geometry.lappd_housing_data,
        geometry.annie_lappd_data,
        geometry.surfboard_data,
        hits,
    )
    return hits


def trace_with_optics(
    origins: np.ndarray,
    directions: np.ndarray,
    geometry: Geometry,
    optics_config: dict[int, OpticalMaterial],
    max_bounces: int = 3,
    n_water: float = N_WATER_DEFAULT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace rays with multi-bounce optical surface physics.

    Calls ``trace_rays()`` repeatedly, each time processing surface
    interactions (Fresnel reflection/transmission, diffuse reflection,
    absorption) per the per-material ``optics_config``.  Detected hits
    carry the **total** optical path length in the ``HT`` column and
    the number of surface reflections in the returned ``bounce_counts``.

    Parameters
    ----------
    origins:
        ``(N, 3)`` float32 — ray start points.
    directions:
        ``(N, 3)`` float32 — ray unit directions.
    geometry:
        Geometry bundle.
    optics_config:
        Per-material :class:`OpticalMaterial` dict from
        :func:`load_optics_config`.
    max_bounces:
        Maximum number of surface reflections per photon (the first
        trace counts as bounce 0).
    n_water:
        Refractive index of water (used for Fresnel computation).
    rng:
        NumPy random generator.

    Returns
    -------
    (hits, bounce_counts, orig_indices):
        ``hits`` is ``(M, N_HIT_COLS)`` float32 — detected hits only
        (a subset of the input rays), with ``HT`` = total accumulated
        path length.  ``bounce_counts`` is ``(M,)`` int32 — number of
        surface reflections for each hit.  ``orig_indices`` is ``(M,)``
        int32 — the original photon index for each detected hit (for
        lookups into per-photon data like creation time).  If no hits
        are detected, returns two empty arrays and an empty index array.
    """
    from annieray.optics import evaluate_hit

    if rng is None:
        rng = np.random.default_rng()

    origins = np.asarray(origins, dtype=np.float32).copy()
    directions = np.asarray(directions, dtype=np.float32).copy()
    n = len(origins)

    total_path = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=bool)
    n_bounces = np.zeros(n, dtype=np.int32)
    detected_hits: list[np.ndarray] = []
    detected_bounces: list[np.int32] = []
    detected_indices: list[int] = []

    for bounce in range(max_bounces + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break

        hits = trace_rays(origins[idx], directions[idx], geometry)

        total_path[idx] += hits[:, HT]

        next_alive = np.zeros(len(idx), dtype=bool)

        for j in range(len(idx)):
            i = idx[j]
            if hits[j, HI] == 0:
                alive[i] = False
                continue

            mat_id = int(hits[j, HMAT])
            mat_opt = optics_config.get(mat_id)
            if mat_opt is None:
                alive[i] = False
                continue

            incident_dir = directions[i]
            normal = hits[j, HNX:HNZ + 1]
            action, new_dir = evaluate_hit(mat_opt, incident_dir, normal, n_water, rng)

            if action == "detect":
                hits[j, HT] = total_path[i]
                detected_hits.append(hits[j:j + 1].copy())
                detected_bounces.append(np.int32(bounce))
                detected_indices.append(i)
                alive[i] = False
            elif action == "reflect":
                origins[i] = hits[j, HX:HZ + 1]
                directions[i] = new_dir
                next_alive[j] = True
            # absorb: drop

        alive[idx] = next_alive

    if detected_hits:
        hits_out = np.concatenate(detected_hits, axis=0)
        bounces_out = np.array(detected_bounces, dtype=np.int32)
        indices_out = np.array(detected_indices, dtype=np.int32)
        return hits_out, bounces_out, indices_out
    return (np.zeros((0, N_HIT_COLS), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32))


def _housing_from_array(arr: np.ndarray):
    """Convert a (16,) housing array back into a LAPPDHousing dataclass."""
    from annieray.lappd_model import LAPPDHousing
    cx, cy, cz = arr[0], arr[1], arr[2]
    ax = (arr[3], arr[4], arr[5])
    ay = (arr[6], arr[7], arr[8])
    az = (arr[9], arr[10], arr[11])
    hx, hy, hz = arr[12], arr[13], arr[14]
    return LAPPDHousing(
        centre=(cx, cy, cz),
        axes=(ax, ay, az),
        half=(float(hx), float(hy), float(hz)),
    )


def compute_tank_track_length(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    tank_radius: float,
    tank_z_min: float,
    tank_z_max: float,
) -> float:
    """Maximum muon track length (m) before exiting the tank cylinder.

    Returns 1.05 × the ray-to-exit distance in metres, or 4.0 m fallback.
    """
    Px, Py, Pz = muon_pos[0], muon_pos[1], muon_pos[2]
    Dx, Dy, Dz = muon_dir[0], muon_dir[1], muon_dir[2]
    R = tank_radius
    z_min = tank_z_min
    z_max = tank_z_max

    candidates = []

    # ---- Cylinder wall (radial exit) ----
    a = Dx * Dx + Dy * Dy
    if a > 1e-12:
        b = 2.0 * (Px * Dx + Py * Dy)
        c = Px * Px + Py * Py - R * R
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2.0 * a)
            if t1 > 0:
                z_at_wall = Pz + t1 * Dz
                if z_min <= z_at_wall <= z_max:
                    candidates.append(t1)

    # ---- Top cap (Z = z_max) ----
    if Dz > 0:
        t = (z_max - Pz) / Dz
        if t > 0:
            r2 = (Px + t * Dx) ** 2 + (Py + t * Dy) ** 2
            if r2 <= R * R:
                candidates.append(t)

    # ---- Bottom cap (Z = z_min) ----
    if Dz < 0:
        t = (z_min - Pz) / Dz
        if t > 0:
            r2 = (Px + t * Dx) ** 2 + (Py + t * Dy) ** 2
            if r2 <= R * R:
                candidates.append(t)

    if not candidates:
        return 4.0

    track_mm = min(candidates)
    return max(track_mm * 1.05 / 1000.0, 0.5)


def compute_track_length(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    geometry: Geometry | None,
) -> float:
    """Geometry-aware muon track length in metres.

    Delegates to ``compute_tank_track_length`` or
    ``compute_housing_track_length`` based on the geometry contents.
    Also tests against surfboard boxes and returns the shortest
    distance.  Falls back to 4.0 m when no geometry is available.
    """
    pos3 = muon_pos[:3] if len(muon_pos) > 3 else muon_pos
    dir3 = muon_dir[:3] if len(muon_dir) > 3 else muon_dir

    from annieray.lappd_model import compute_housing_track_length

    best = 1e30

    if geometry is not None and geometry.lappd_housing_data.shape[0] > 0:
        hd = geometry.lappd_housing_data[0]
        housing = _housing_from_array(hd)
        best = min(best, compute_housing_track_length(pos3, dir3, housing))

    if geometry is not None:
        for s in range(geometry.surfboard_data.shape[0]):
            row = geometry.surfboard_data[s]
            cx, cy, cz = row[0], row[1], row[2]
            ax = (row[3], row[4], row[5])
            ay = (row[6], row[7], row[8])
            az = (row[9], row[10], row[11])
            hx, hy, hz = row[12], row[13], row[14]

            ox, oy, oz = pos3
            dx, dy, dz = dir3
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
                    break
            else:
                if t_max > 0 and t_min > -1e-6:
                    track_mm = max(t_max, 0.0)
                    best = min(best, track_mm * 1.05 / 1000.0)

    if geometry is not None and best > 1e29:
        best = compute_tank_track_length(
            pos3, dir3,
            geometry.tank_radius, geometry.tank_z_min, geometry.tank_z_max,
        )

    if best > 1e29:
        return 4.0
    return max(best, 0.5)


def trace_cherenkov(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    photons_per_cm: int = 150,
    geometry: Geometry | None = None,
    rng: np.random.Generator | None = None,
    wavelength_nm: float = 350.0,
    n_water: float = N_WATER_DEFAULT,
    max_bounces: int = 0,
    optics_config: dict[int, OpticalMaterial] | None = None,
) -> np.ndarray:
    """Trace Cherenkov photons from a muon track.

    Workflow:
      1. Compute track length from geometry (tank cylinder or LAPPD housing).
      2. Call generate_cherenkov_photons() with that track length.
      3. Run the GPU kernel via trace_rays() or trace_with_optics().
      4. Expand to (N, 17) by appending arrival_time, wavelength,
         and bounce_count.

    Parameters
    ----------
    photons_per_cm : int
        Photons generated per cm along the muon track.
        Total photons ≈ ``track_length_cm × photons_per_cm``.
    geometry : Geometry | None
        If provided, the track length is automatically computed
        from the tank or LAPPD housing bounds.  When ``None``,
        a fixed 4.0 m track is used.

    Returns ``(N, N_EXPANDED_COLS)`` hit array with columns:
        0: hit_flag, 1: t (mm), 2-4: x,y,z, 5-7: nx,ny,nz,
        8: component_id, 9: detector_index, 10: detector_system,
        11: local_u, 12: local_v, 13: material_id,
        14: arrival_time (ns), 15: wavelength (nm), 16: bounce_count
    """
    from annieray.cherenkov import generate_cherenkov_photons
    from annieray.lappd_model import compute_housing_track_length
    from annieray.optics import load_optics_config

    if rng is None:
        rng = np.random.default_rng()

    # ---- Compute track length from geometry ----
    track_length = compute_track_length(muon_pos, muon_dir, geometry)

    origins, directions, create_times = generate_cherenkov_photons(
        muon_pos, muon_dir, photons_per_cm, track_length=track_length, rng=rng,
    )

    # ---- No geometry → just return the generated rays (no intersection testing) ----
    if geometry is None:
        n = origins.shape[0]
        full = np.zeros((n, N_EXPANDED_COLS), dtype=np.float32)
        full[:, HI] = 1.0
        full[:, HX:HZ + 1] = origins
        full[:, HDI] = -1.0
        full[:, HDS] = float(DET_SYS_NONE)
        full[:, H_WAVELEN] = wavelength_nm
        full[:, H_ARRIVAL] = create_times
        return full

    if max_bounces > 0:
        cfg = optics_config if optics_config is not None else load_optics_config(None)
        hits, bounce_counts, orig_indices = trace_with_optics(
            origins, directions, geometry, cfg,
            max_bounces=max_bounces, n_water=n_water, rng=rng,
        )
    else:
        hits = trace_rays(origins, directions, geometry)
        bounce_counts = np.zeros(hits.shape[0], dtype=np.int32)
        orig_indices = np.arange(hits.shape[0], dtype=np.int32)

    # ---- Expand from N_HIT_COLS to N_EXPANDED_COLS ----
    # Add arrival_time (14), wavelength (15), bounce_count (16).
    n = hits.shape[0]
    full = np.zeros((n, N_EXPANDED_COLS), dtype=np.float32)
    full[:, :N_HIT_COLS] = hits

    full[:, H_WAVELEN] = wavelength_nm
    full[:, H_BOUNCE] = bounce_counts

    c_in_water = C_MM_NS / n_water
    hit_mask = hits[:, HI] > 0.5
    if hit_mask.any():
        full[hit_mask, H_ARRIVAL] = create_times[orig_indices[hit_mask]] + hits[hit_mask, HT] / c_in_water

    return full
