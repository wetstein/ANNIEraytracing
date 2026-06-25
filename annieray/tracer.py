"""GPU-accelerated analytic ray tracer using Taichi."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti

from annieray import gdml_parser, step_parser
from annieray import pmt_loader
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

N_HIT_COLS = 13

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


def build_geometry(
    gdml_path: Path,
    step_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    pmt_csv_path: Optional[Path] = None,
    lappd_indices: Optional[list[int]] = None,
    no_lappd: bool = False,
    z_offset: float = 0.0,
    lappd_model: str = "default",
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
    verts, tris = gdml_parser.parse_gdml(gdml_path)

    # ---- Stage 2: load PMT positions ----
    # PMTs can come from: (a) Scan CSV file, (b) STEP manifest JSON, (c) STEP raw.
    # The CSV is preferred because it has per-PMT radii and type names.
    pmt_directions = np.zeros((0, 3), dtype=np.float32)
    if pmt_csv_path and pmt_csv_path.exists():
        pmt_data = pmt_loader.load_pmts(pmt_csv_path, z_offset=z_offset)
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

    # ---- Tank bounds (from manifest or hard-coded defaults) ----
    # Used for the infinite-cylinder tank-wall intersection and for computing
    # the Z-centre when placing the ANNIE housing on the octagon.
    if manifest and manifest.tank_bbox:
        tank_radius = max(
            manifest.tank_bbox.xmax - manifest.tank_bbox.xmin,
            manifest.tank_bbox.ymax - manifest.tank_bbox.ymin,
        ) / 2
        tank_z_min = manifest.tank_bbox.zmin
        tank_z_max = manifest.tank_bbox.zmax
    else:
        tank_radius = 1264.0
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

    return Geometry(
        mesh_vertices=verts,
        mesh_triangles=tris,
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
        detectors=detectors,
    )


# ---- Taichi helper functions (single-return pattern for Taichi compat) ----


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
#   2. PMT spheres                           → component_id = 2
#   3. Default LAPPD rectangles              → component_id = 3
#   4. ANNIE housing box (absorbs side/back) → kills photon (component_id = 0)
#   5. ANNIE LAPPD photocathode rectangle    → component_id = 3
#   6. Tank wall (infinite cylinder)         → component_id = 4


@ti.kernel
def trace_kernel(
    origins: ti.types.ndarray(ndim=2),
    directions: ti.types.ndarray(ndim=2),
    mesh_vertices: ti.types.ndarray(ndim=2),
    mesh_triangles: ti.types.ndarray(ndim=2),
    pmt_centers: ti.types.ndarray(ndim=2),
    pmt_radii: ti.types.ndarray(ndim=1),
    pmt_dirs: ti.types.ndarray(ndim=2),
    lappd_data: ti.types.ndarray(ndim=2),
    lappd_strip: ti.types.ndarray(ndim=2),
    tank_radius: ti.f32,
    tank_z_min: ti.f32,
    tank_z_max: ti.f32,
    housing_data: ti.types.ndarray(ndim=2),
    annie_lappd_data: ti.types.ndarray(ndim=2),
    hits: ti.types.ndarray(ndim=2),
):
    n_rays = origins.shape[0]
    n_tris = mesh_triangles.shape[0]
    n_pmts = pmt_centers.shape[0]
    n_lappds = lappd_data.shape[0]
    n_housings = housing_data.shape[0]

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

        for t in range(n_tris):
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

            hit, t_hit, u, v, nx, ny, nz = _ray_triangle_intersect(
                ox, oy, oz, dx, dy, dz,
                v0x, v0y, v0z,
                v1x, v1y, v1z,
                v2x, v2y, v2z,
            )

            if hit and t_hit > 1e-6 and t_hit < best_t:
                best_t = t_hit
                best_hit = CID_INNER_STRUCTURE
                best_x = ox + dx * t_hit
                best_y = oy + dy * t_hit
                best_z = oz + dz * t_hit
                best_nx = nx
                best_ny = ny
                best_nz = nz
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE

        for p in range(n_pmts):
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
        geometry.pmt_centers,
        geometry.pmt_radii,
        geometry.pmt_directions,
        geometry.lappd_data,
        geometry.lappd_strip_axes,
        geometry.tank_radius,
        geometry.tank_z_min,
        geometry.tank_z_max,
        geometry.lappd_housing_data,
        geometry.annie_lappd_data,
        hits,
    )
    return hits


def trace_cherenkov(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    n_photons: int,
    geometry: Geometry,
    rng: np.random.Generator | None = None,
    wavelength_nm: float = 350.0,
    n_water: float = N_WATER_DEFAULT,
) -> np.ndarray:
    """Trace Cherenkov photons from a muon track.

    Workflow:
      1. Call generate_cherenkov_photons() to get (origins, directions).
      2. Run the GPU kernel via trace_rays() → (N, 13) hit array.
      3. Expand to (N, 15) by appending arrival_time and wavelength.

    The expansion is the place to add per-photon wavelength and timing.
    Currently wavelength_nm is a single scalar; for a per-photon spectrum,
    generate_cherenkov_photons() should return a wavelength array and this
    function should stamp it into full[:, 14] per-photon instead.

    Returns (N, 15) hit array with columns:
        0: hit_flag, 1: t (mm), 2-4: x,y,z, 5-7: nx,ny,nz,
        8: component_id, 9: detector_index, 10: detector_system,
        11: local_u, 12: local_v, 13: arrival_time (ns), 14: wavelength (nm)
    """
    from annieray.cherenkov import generate_cherenkov_photons

    if rng is None:
        rng = np.random.default_rng()

    origins, directions, creationTime = generate_cherenkov_photons(
        muon_pos, muon_dir, n_photons, rng=rng,
    )
    hits = trace_rays(origins, directions, geometry)

    # ---- Expand from 13 to 15 columns ----
    # Kernel output: columns 0-12 (hit_flag through local_v).
    # We add columns 13 (arrival_time in ns) and 14 (wavelength in nm).
    n = hits.shape[0]
    full = np.zeros((n, N_HIT_COLS + 2), dtype=np.float32)
    full[:, :N_HIT_COLS] = hits

    # Col 14: wavelength
    # Currently a single value for all photons.  To support per-photon
    # wavelength sampling, replace this line with per-photon assignment
    # using a wavelength array from generate_cherenkov_photons().
    full[:, N_HIT_COLS + 1] = wavelength_nm

    # Col 13: arrival_time = photon path length / speed_of_light_in_water
    #   t = ray path length from origin to hit (mm)
    #   C = 299.79 mm/ns (vacuum)
    #   n = refractive index of water
    #   arrival_time = t / (C / n)
    #
    # NOTE: This assumes all photons start at the same vertex.  Once muon
    # propagation is added, the per-photon emission time along the track
    # must be added to this calculation.
    c_in_water = C_MM_NS / n_water
    hit_mask = hits[:, HI] > 0.5
    if hit_mask.any():
        full[hit_mask, N_HIT_COLS] = hits[hit_mask, HT] / c_in_water #+ creationTime <- This needs to have a way to call back to what photons actually hit and then find their creation time

    return full
