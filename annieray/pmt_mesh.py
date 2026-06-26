"""Shared PMT mesh loading for tracer kernel and viz server."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parent.parent / "pmt_meshes"

# Mesh type index → (filename, recenter, type_name)
PMT_BODY_SPECS: list[tuple[int, str, bool, str]] = [
    (0, "pmt_lux_bottom.gdml",    True,  "LUX"),
    (1, "pmt_etel_top.gdml",      True,  "ETEL"),
    (2, "pmt_8inch_body.gdml",    False, "Hamamatsu"),
    (3, "pmt_10inch_body.gdml",   False, "Watchboy"),
]

PMT_HW_SPECS: list[tuple[int, str]] = [
    (4, "pmt_8inch_hardware.gdml"),
    (5, "pmt_10inch_hardware.gdml"),
]


@dataclass
class PMTMeshData:
    """Per-type PMT mesh in local frame (centroid at origin)."""
    vertices: np.ndarray     # (V, 3) float32 — flat triangle soup
    material_ids: np.ndarray  # (T,) int32 — MaterialID per triangle
    bounding_radius: float    # mm — max ||vertex|| + 1 mm margin
    n_tris: int


# Module-level caches so callers (tracer, viz server) share one parse
_body_mesh_cache: dict[int, PMTMeshData] | None = None
_hw_mesh_cache: dict[int, PMTMeshData] | None = None


def parse_gdml_flattened(path: Path, recenter: bool = True
                         ) -> tuple[np.ndarray, int]:
    """Parse a GDML tessellated mesh into (flat_vertices, n_tris)."""
    tree = ET.parse(path)
    root = tree.getroot()
    positions = root.findall(".//position")
    verts = {
        p.attrib["name"]: (float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
        for p in positions
    }
    triangles = root.findall(".//triangular")
    out = []
    for tri in triangles:
        for key in ("vertex1", "vertex2", "vertex3"):
            out.extend(verts[tri.attrib[key]])
    arr = np.array(out, dtype=np.float32).reshape(-1, 3)
    if recenter:
        arr -= arr.mean(axis=0)
    return arr, len(arr) // 3


def _compute_bounding_radius(vertices: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(vertices, axis=1))) + 1.0


def _load_body_mesh_from_cache(mi: int, gn: str, tn: str) -> PMTMeshData | None:
    """Load a pre-built body mesh from ``_body_cache.npz``, or None."""
    cache_name = gn.replace(".gdml", "_body_cache.npz")
    p = MESH_DIR / cache_name
    if not p.exists():
        return None
    data = np.load(p)
    verts = data["vertices"]
    mat_ids = data["material_ids"]
    bradius = float(data["bounding_radius"])
    n_tris = len(verts) // 3
    print(f"  PMT body mesh {mi} ({tn}): {n_tris} tris (cached), "
          f"bounding radius {bradius:.1f} mm")
    return PMTMeshData(vertices=verts, material_ids=mat_ids,
                       bounding_radius=bradius, n_tris=n_tris)


def load_pmt_body_meshes() -> dict[int, PMTMeshData]:
    """Load all 4 PMT body meshes and classify per-triangle materials.

    Prefers pre-built ``_body_cache.npz`` files (fast, no GDML parsing).
    Falls back to parsing raw GDML.

    Cached internally — second call is a no-op.

    Returns dict mapping mesh type index 0-3 to PMTMeshData.
    Missing files are omitted (graceful fallback).
    """
    global _body_mesh_cache
    if _body_mesh_cache is not None:
        return _body_mesh_cache

    from annieray.materials import classify_pmt_body

    result: dict[int, PMTMeshData] = {}
    for mi, gn, rc, tn in PMT_BODY_SPECS:
        cached = _load_body_mesh_from_cache(mi, gn, tn)
        if cached is not None:
            result[mi] = cached
            continue

        p = MESH_DIR / gn
        if not p.exists():
            print(f"  PMT mesh {mi} ({gn}): NOT FOUND")
            continue
        flat, n_tris = parse_gdml_flattened(p, recenter=rc)
        tris_333 = flat.reshape(-1, 3, 3)
        mat_ids = classify_pmt_body(tris_333, tn)
        bradius = _compute_bounding_radius(flat)
        result[mi] = PMTMeshData(
            vertices=flat,
            material_ids=mat_ids,
            bounding_radius=bradius,
            n_tris=n_tris,
        )
        print(f"  PMT body mesh {mi} ({tn}): {n_tris} tris, "
              f"bounding radius {bradius:.1f} mm, "
              f"PC={int((mat_ids == 2).sum())}, "
              f"GLASS={int((mat_ids == 1).sum())}, "
              f"PVC={int((mat_ids == 3).sum())}")
    _body_mesh_cache = result
    return result


def _load_hw_mesh_from_npy(mi: int, gn: str) -> np.ndarray | None:
    """Load a decimated HW mesh from .npy cache, or None."""
    npy_name = gn.replace(".gdml", "_decimated.npy")
    p = MESH_DIR / npy_name
    if p.exists():
        arr = np.load(p)
        print(f"  PMT HW mesh {mi} (decimated {npy_name}): {len(arr)//3} tris")
        return arr
    return None


def load_pmt_hw_meshes() -> dict[int, PMTMeshData]:
    """Load hardware (holder) meshes for 8" and 10" PMTs.

    Prefers decimated ``.npy`` cache files (created by decimate_hw.py).
    Falls back to GDML parsing.

    Cached internally — second call is a no-op.

    Returns dict mapping mesh type index 4-5 to PMTMeshData.
    Missing files are omitted (graceful fallback).
    """
    global _hw_mesh_cache
    if _hw_mesh_cache is not None:
        return _hw_mesh_cache

    from annieray.materials import classify_pmt_hardware

    result: dict[int, PMTMeshData] = {}
    for mi, gn in PMT_HW_SPECS:
        # Prefer decimated npy cache
        flat = _load_hw_mesh_from_npy(mi, gn)
        if flat is None:
            p = MESH_DIR / gn
            if not p.exists():
                print(f"  PMT HW mesh {mi} ({gn}): NOT FOUND")
                continue
            flat, n_tris = parse_gdml_flattened(p, recenter=False)
        else:
            n_tris = len(flat) // 3

        mat_ids = classify_pmt_hardware(mi, n_tris)
        bradius = _compute_bounding_radius(flat)
        result[mi] = PMTMeshData(
            vertices=flat,
            material_ids=mat_ids,
            bounding_radius=bradius,
            n_tris=n_tris,
        )
        print(f"  PMT HW mesh {mi}: {n_tris} tris, "
              f"bounding radius {bradius:.1f} mm, "
              f"material={int(mat_ids[0])}")
    _hw_mesh_cache = result
    return result


def build_viz_caches() -> tuple[dict[int, bytes], dict[int, bytes],
                                dict[int, bytes], dict[int, bytes],
                                dict[int, bytes], dict[int, bytes]]:
    """Build byte-string caches needed by the viz server.

    All meshes are loaded through the shared internal caches (no re-parsing).

    Returns
    -------
    (body_verts, body_colors, body_pc, body_pvc, hw_verts, hw_colors)
    Each is ``dict[int, bytes]`` keyed by mesh type index.
    ``body_pc`` / ``body_pvc`` may contain ``b""`` for types with no
    photocathode or PVC faces respectively.
    """
    from annieray.materials import MATERIAL_TABLE, MaterialID

    body_meshes = load_pmt_body_meshes()
    hw_meshes = load_pmt_hw_meshes()

    body_verts: dict[int, bytes] = {}
    body_colors: dict[int, bytes] = {}
    body_pc: dict[int, bytes] = {}
    body_pvc: dict[int, bytes] = {}

    for mi, md in body_meshes.items():
        body_verts[mi] = md.vertices.tobytes()

        # Per-vertex RGBA from material table
        n_tris = md.n_tris
        colors = np.zeros((n_tris * 3, 4), dtype=np.uint8)
        for tri_idx in range(n_tris):
            mid = int(md.material_ids[tri_idx])
            props = MATERIAL_TABLE.get(MaterialID(mid))
            r = int(props.color[0] * 255)
            g = int(props.color[1] * 255)
            b = int(props.color[2] * 255)
            colors[tri_idx * 3] =     [r, g, b, 255]
            colors[tri_idx * 3 + 1] = [r, g, b, 255]
            colors[tri_idx * 3 + 2] = [r, g, b, 255]
        body_colors[mi] = colors.tobytes()

        # PC sub-mesh (offset inward 1 mm along face normal)
        pc_mask = md.material_ids == int(MaterialID.PHOTOCATHODE)
        pc_tri_idx = np.where(pc_mask)[0]
        if len(pc_tri_idx) > 0:
            n = len(pc_tri_idx)
            idx = np.repeat(pc_tri_idx * 3, 3) + np.tile(np.arange(3), n)
            pc_verts = md.vertices[idx].copy()
            v0 = md.vertices[pc_tri_idx * 3]
            v1 = md.vertices[pc_tri_idx * 3 + 1]
            v2 = md.vertices[pc_tri_idx * 3 + 2]
            normals = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(normals, axis=1, keepdims=True)
            nlen[nlen == 0] = 1
            normals /= nlen
            pc_verts -= np.repeat(normals * 1.0, 3, axis=0)
            body_pc[mi] = pc_verts.tobytes()
        else:
            body_pc[mi] = b""

        # PVC sub-mesh
        pvc_mask = md.material_ids == int(MaterialID.PVC)
        pvc_tri_idx = np.where(pvc_mask)[0]
        if len(pvc_tri_idx) > 0:
            n = len(pvc_tri_idx)
            idx = np.repeat(pvc_tri_idx * 3, 3) + np.tile(np.arange(3), n)
            body_pvc[mi] = md.vertices[idx].copy().tobytes()
        else:
            body_pvc[mi] = b""

    # Hardware caches (verts + colors, no PC/PVC sub-meshes)
    hw_verts: dict[int, bytes] = {}
    hw_colors: dict[int, bytes] = {}
    for mi, md in hw_meshes.items():
        hw_verts[mi] = md.vertices.tobytes()
        n_tris = md.n_tris
        colors = np.zeros((n_tris * 3, 4), dtype=np.uint8)
        for tri_idx in range(n_tris):
            mid = int(md.material_ids[tri_idx])
            props = MATERIAL_TABLE.get(MaterialID(mid))
            r = int(props.color[0] * 255)
            g = int(props.color[1] * 255)
            b = int(props.color[2] * 255)
            colors[tri_idx * 3] =     [r, g, b, 255]
            colors[tri_idx * 3 + 1] = [r, g, b, 255]
            colors[tri_idx * 3 + 2] = [r, g, b, 255]
        hw_colors[mi] = colors.tobytes()

    return body_verts, body_colors, body_pc, body_pvc, hw_verts, hw_colors


def build_body_tris_arrays(
    body_meshes: dict[int, PMTMeshData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all body-mesh triangles and build offset table.

    Returns
    -------
    body_tris : (T_global, 9) float32
        Each row is 3 vertices (v0x,v0y,v0z, v1x,…, v2z) in local frame.
    body_mat_ids : (T_global,) int32
        Material ID per triangle.
    body_offsets : (5,) int32
        Start index in body_tris for each mesh type 0-3, plus sentinel.
    """
    tri_list: list[np.ndarray] = []
    mat_list: list[np.ndarray] = []
    offsets = [0]
    for mt in range(4):
        md = body_meshes.get(mt)
        if md is not None:
            tris_9 = md.vertices.reshape(-1, 9)
            tri_list.append(tris_9)
            mat_list.append(md.material_ids)
            offsets.append(offsets[-1] + md.n_tris)
        else:
            offsets.append(offsets[-1])
    body_tris = np.concatenate(tri_list, axis=0) if tri_list else np.zeros((0, 9), dtype=np.float32)
    body_mat_ids = np.concatenate(mat_list, axis=0) if mat_list else np.zeros(0, dtype=np.int32)
    body_offsets = np.array(offsets, dtype=np.int32)
    return body_tris, body_mat_ids, body_offsets


def build_hw_tris_arrays(
    hw_meshes: dict[int, PMTMeshData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all HW-mesh triangles and build offset table.

    Returns
    -------
    hw_tris : (T_hw, 9) float32
        Each row is 3 vertices in local frame.
    hw_mat_ids : (T_hw,) int32
        Material ID per triangle (TEFLON=4 or ACRYLIC=5).
    hw_offsets : (3,) int32
        Start index for mesh types 4, 5 plus sentinel.
    """
    tri_list: list[np.ndarray] = []
    mat_list: list[np.ndarray] = []
    offsets = [0]
    for mt in (4, 5):
        md = hw_meshes.get(mt)
        if md is not None:
            tri_list.append(md.vertices.reshape(-1, 9))
            mat_list.append(md.material_ids)
            offsets.append(offsets[-1] + md.n_tris)
        else:
            offsets.append(offsets[-1])
    hw_tris = np.concatenate(tri_list, axis=0) if tri_list else np.zeros((0, 9), dtype=np.float32)
    hw_mat_ids = np.concatenate(mat_list, axis=0) if mat_list else np.zeros(0, dtype=np.int32)
    hw_offsets = np.array(offsets, dtype=np.int32)
    return hw_tris, hw_mat_ids, hw_offsets
