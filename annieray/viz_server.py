"""Interactive Cherenkov visualization server for ANNIE.

Serves a Three.js frontend with real-time ray tracing of Cherenkov cones
from a user-controlled muon track.

Usage:
    python -m annieray viz-server \
        --pmt-csv PMTPositions_Scan.txt
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from annieray.pmt_loader import rotate_z
from annieray.pmt_mesh import build_viz_caches
from annieray.tracer import build_geometry, trace_cherenkov, reload_lappd_corrections

geometry = None

pmt_instance_data: dict | None = None   # from pmt_loader

# ---- PMT mesh caching (built from shared pmt_mesh module) ----
_viz_caches = build_viz_caches()
PMT_MESH_CACHE: dict[int, bytes] = {}
PMT_COLOR_CACHE: dict[int, bytes] = {}
PMT_MESH_PC_CACHE: dict[int, bytes] = _viz_caches[2]
PMT_MESH_PVC_CACHE: dict[int, bytes] = _viz_caches[3]
_body_verts, _body_colors, _body_pc, _body_pvc, _hw_verts, _hw_colors = _viz_caches
PMT_MESH_CACHE.update(_body_verts)
PMT_MESH_CACHE.update(_hw_verts)
PMT_COLOR_CACHE.update(_body_colors)
PMT_COLOR_CACHE.update(_hw_colors)
# Print sub-mesh info matching previous style
for _mi, _d in PMT_MESH_CACHE.items():
    _n = len(_d) // 36
    _pc_n = len(PMT_MESH_PC_CACHE.get(_mi, b"")) // 36
    _pvc_n = len(PMT_MESH_PVC_CACHE.get(_mi, b"")) // 36
    print(f"  PMT mesh {_mi}: {_n} tris (PC sub: {_pc_n}, PVC sub: {_pvc_n})")

# ---- PMT tip positions cache ----
PMT_TIPS_CACHE: list[dict] | None = None  # loaded from pmt_tip_positions.csv

# ---- Scan mesh overlay cache ----
SCAN_MESH_DIR = Path("scan files by part") / "transformed"
SCAN_MESH_CACHE: dict[str, tuple[bytes, bytes]] = {}  # name -> (verts_bytes, tris_bytes)

def _load_scan_mesh(name: str, det_rotation_deg: float = 0.0):
    """Load a pre-processed scan mesh from .npy files into the cache."""
    verts_path = SCAN_MESH_DIR / f"{name}_verts.npy"
    tris_path = SCAN_MESH_DIR / f"{name}_tris.npy"
    if verts_path.exists() and tris_path.exists():
        verts = np.load(verts_path)
        tris = np.load(tris_path)
        if det_rotation_deg != 0.0:
            rotate_z(verts, det_rotation_deg)
        SCAN_MESH_CACHE[name] = (verts.tobytes(), tris.tobytes())
        return True
    return False

# ---------------------------------------------------------------------------

class VizHandler(BaseHTTPRequestHandler):
    _pmt_types: list[str] = []
    _pmt_data = None
    _corrections: dict[int, tuple[float, float, float]] = {}
    _corr_path: Path | None = None
    _lappd_corr_path: str | None = None

    def log_message(self, fmt, *args):
        pass  # quiet

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data, content_type="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._send_html()
        elif path == "/api/pmts":
            self._send_pmts()
        elif path == "/api/mesh/verts":
            self._send_mesh_verts()
        elif path == "/api/mesh/tris":
            self._send_mesh_tris()
        elif path == "/api/trace":
            self._handle_trace(params)
        elif path == "/api/housing":
            self._send_lappd_housing()
        elif path == "/api/surfboards":
            self._send_surfboards()
        elif path == "/api/lappd_correction":
            self._send_lappd_correction()
        elif path == "/api/detectors":
            self._send_detectors()
        elif path.startswith("/api/pmt_mesh/"):
            self._send_pmt_mesh(path)
        elif path.startswith("/api/pmt_mesh_pc/"):
            self._send_pmt_mesh_pc(path)
        elif path.startswith("/api/pmt_mesh_pvc/"):
            self._send_pmt_mesh_pvc(path)
        elif path.startswith("/api/pmt_mesh_colors/"):
            self._send_pmt_mesh_colors(path)
        elif path == "/api/scan_mesh/list":
            self._send_scan_mesh_list()
        elif path.startswith("/api/scan_mesh/"):
            self._send_scan_mesh(path)
        elif path == "/api/pmt_tips":
            self._send_pmt_tips()
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)
        path = urlparse(self.path).path

        if path == "/api/correction/save":
            self._save_correction(data)
        elif path == "/api/lappd_correction":
            self._save_lappd_correction(data)
        elif path == "/api/surfboard/adjust":
            self._handle_surfboard_adjust(data)
        elif path == "/api/lappd/adjust":
            self._handle_lappd_adjust(data)
        else:
            self.send_error(404)

    def _save_correction(self, data):
        tube_id = int(data["tube_id"])
        axial = float(data["axial"])
        tangential = float(data["tangential"])
        vertical = float(data["vertical"])

        dets = list(self._pmt_data["detector_nums"])
        if tube_id not in dets:
            self._send_json({"error": "tube not found"}, 400)
            return
        idx = dets.index(tube_id)
        d = self._pmt_data["directions"][idx]

        # Determine mode from direction
        if abs(d[2]) > 0.99:  # bottom or top PMT — global XYZ
            dx, dy, dz = axial, tangential, vertical
        else:  # barrel PMT — local axis basis
            e_axial = np.array([d[0], d[1], d[2]], dtype=np.float64)
            e_vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            e_tangential = np.cross(e_vertical, e_axial)
            tnorm = np.linalg.norm(e_tangential)
            if tnorm > 1e-9:
                e_tangential /= tnorm
            else:
                e_tangential = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            dx = axial * e_axial[0] + tangential * e_tangential[0] + vertical * e_vertical[0]
            dy = axial * e_axial[1] + tangential * e_tangential[1] + vertical * e_vertical[1]
            dz = axial * e_axial[2] + tangential * e_tangential[2] + vertical * e_vertical[2]

        # Read current file
        corrections = {}
        corr_path = self._corr_path
        if corr_path and corr_path.exists():
            with open(corr_path, newline="") as f:
                for row in csv.DictReader(f):
                    corrections[int(row["tube_id"])] = (
                        float(row.get("dx", 0)),
                        float(row.get("dy", 0)),
                        float(row.get("dz", 0)),
                    )

        old = corrections.get(tube_id, (0.0, 0.0, 0.0))

        # Update in-memory
        corrections[tube_id] = (dx, dy, dz)
        self._corrections[tube_id] = (dx, dy, dz)

        # Write back
        with open(corr_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tube_id", "dx", "dy", "dz"])
            for tid in sorted(corrections):
                cx, cy, cz = corrections[tid]
                writer.writerow([tid, cx, cy, cz])

        # Apply delta to instance_positions
        positions = self._pmt_data["instance_positions"]
        positions[idx, 0] += dx - old[0]
        positions[idx, 1] += dy - old[1]
        positions[idx, 2] += dz - old[2]

        # Apply same delta to kernel geometry so mesh refinement finds corrected positions
        geometry.pmt_instance_pos[idx, 0] += dx - old[0]
        geometry.pmt_instance_pos[idx, 1] += dy - old[1]
        geometry.pmt_instance_pos[idx, 2] += dz - old[2]
        geometry.pmt_centers[idx, 0] += dx - old[0]
        geometry.pmt_centers[idx, 1] += dy - old[1]
        geometry.pmt_centers[idx, 2] += dz - old[2]

        self._send_json({
            "success": True,
            "tube_id": tube_id,
            "instance_position": [float(positions[idx, 0]),
                                  float(positions[idx, 1]),
                                  float(positions[idx, 2])],
        })

    def _send_lappd_housing(self):
        if geometry.lappd_housing_data.shape[0] == 0:
            self._send_json({"housing": []})
            return
        housings = []
        for i in range(geometry.lappd_housing_data.shape[0]):
            hd = geometry.lappd_housing_data[i].tolist()
            ad = geometry.annie_lappd_data[i].tolist()
            housings.append({
                "center": [hd[0], hd[1], hd[2]],
                "axis_x": [hd[3], hd[4], hd[5]],
                "axis_y": [hd[6], hd[7], hd[8]],
                "axis_z": [hd[9], hd[10], hd[11]],
                "half": [hd[12], hd[13], hd[14]],
                "pc_center": [ad[0], ad[1], ad[2]],
                "pc_normal": [ad[3], ad[4], ad[5]],
                "pc_half": [ad[6]],
            })
        self._send_json({"housing": housings})

    def _send_surfboards(self):
        if geometry.surfboard_data.shape[0] == 0:
            self._send_json({"surfboards": []})
            return
        boards = []
        for i in range(geometry.surfboard_data.shape[0]):
            row = geometry.surfboard_data[i].tolist()
            boards.append({
                "center": [row[0], row[1], row[2]],
                "axis_x": [row[3], row[4], row[5]],
                "axis_y": [row[6], row[7], row[8]],
                "axis_z": [row[9], row[10], row[11]],
                "half": [row[12], row[13], row[14]],
            })
        self._send_json({"surfboards": boards})

    def _handle_surfboard_adjust(self, data):
        idx = int(data["index"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        cz = float(data["cz"])
        if 0 <= idx < geometry.surfboard_data.shape[0]:
            geometry.surfboard_data[idx, 0] = cx
            geometry.surfboard_data[idx, 1] = cy
            geometry.surfboard_data[idx, 2] = cz
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "invalid index"}, 400)

    def _handle_lappd_adjust(self, data):
        idx = int(data["index"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        cz = float(data["cz"])
        if 0 <= idx < geometry.lappd_housing_data.shape[0]:
            old_center = geometry.lappd_housing_data[idx, 0:3].copy()
            geometry.lappd_housing_data[idx, 0] = cx
            geometry.lappd_housing_data[idx, 1] = cy
            geometry.lappd_housing_data[idx, 2] = cz
            delta = np.array([cx, cy, cz]) - old_center
            geometry.annie_lappd_data[idx, 0:3] += delta
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "invalid index"}, 400)

    def _send_lappd_correction(self):
        if not self._lappd_corr_path or not os.path.exists(self._lappd_corr_path):
            self._send_json({"corrections": [{"idx": 0, "dx": 0, "dy": 0, "dz": 0}]})
            return
        with open(self._lappd_corr_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self._send_json({"corrections": rows})

    def _save_lappd_correction(self, data):
        idx = int(data.get("idx", 0))
        dx = float(data.get("dx", 0))
        dy = float(data.get("dy", 0))
        dz = float(data.get("dz", 0))
        if not self._lappd_corr_path:
            self._send_json({"error": "no LAPPD correction path"}, 400)
            return
        corrected = False
        rows = []
        if os.path.exists(self._lappd_corr_path):
            with open(self._lappd_corr_path, newline="") as f:
                for row in csv.DictReader(f):
                    if int(row["idx"]) == idx:
                        row["dx"], row["dy"], row["dz"] = str(dx), str(dy), str(dz)
                        corrected = True
                    rows.append(row)
        if not corrected:
            rows.append({"idx": str(idx), "dx": str(dx), "dy": str(dy), "dz": str(dz)})
        with open(self._lappd_corr_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "dx", "dy", "dz"])
            for r in rows:
                w.writerow([r["idx"], r["dx"], r["dy"], r["dz"]])
        self._send_json({"status": "ok"})

    def _send_detectors(self):
        dets = []
        for d in geometry.detectors:
            dets.append({
                "id": d.id,
                "system": d.system,
                "label": d.label,
                "position": list(d.position),
                "direction": list(d.direction),
                "panel": d.panel,
                "pmt_type": d.pmt_type,
                "radius": d.radius,
            })
        self._send_json({"detectors": dets})

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def _send_pmts(self):
        c = np.asarray(geometry.pmt_centers)
        d = np.asarray(pmt_instance_data["directions"]) if pmt_instance_data else np.empty((0, 3))
        resp = {
            "centers": c.tolist(),
            "radii": geometry.pmt_radii.tolist(),
            "types": self._pmt_types,
            "directions": d.tolist(),
            "detector_nums": pmt_instance_data["detector_nums"] if pmt_instance_data is not None else [],
            "corrections": {str(tid): list(v) for tid, v in self._corrections.items()},
        }
        if pmt_instance_data is not None:
            resp["mesh_types"] = pmt_instance_data["mesh_types"].tolist()
            resp["quaternions"] = pmt_instance_data["quaternions"].tolist()
            resp["instance_positions"] = pmt_instance_data["instance_positions"].tolist()
            # Hardware mesh type per PMT (4=8" HW, 5=10" HW, -1=none)
            hw = []
            for t in self._pmt_types:
                if t == "Hamamatsu":
                    hw.append(4)
                elif t in ("Watchboy", "Watchman"):
                    hw.append(5)
                else:
                    hw.append(-1)
            resp["hw_mesh_types"] = hw
        self._send_json(resp)

    def _send_mesh_verts(self):
        # Mesh is in GDML rest frame (Z-up structure frame).
        # Three.js camera is also Z-up — no transform needed.
        buf = geometry.mesh_vertices.astype(np.float32).tobytes()
        self._send_binary(buf)

    def _send_mesh_tris(self):
        buf = geometry.mesh_triangles.astype(np.int32).tobytes()
        self._send_binary(buf)

    def _send_pmt_mesh(self, path):
        try:
            type_idx = int(path.split("/")[-1])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid type"}, 400)
            return
        data = PMT_MESH_CACHE.get(type_idx)
        if data is None:
            self._send_json({"error": "mesh type not available"}, 404)
            return
        self._send_binary(data)

    def _send_pmt_mesh_pc(self, path):
        try:
            type_idx = int(path.split("/")[-1])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid type"}, 400)
            return
        data = PMT_MESH_PC_CACHE.get(type_idx)
        if data is None or len(data) == 0:
            self._send_json({"error": "no PC sub-mesh for this type"}, 404)
            return
        self._send_binary(data)

    def _send_pmt_mesh_pvc(self, path):
        try:
            type_idx = int(path.split("/")[-1])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid type"}, 400)
            return
        data = PMT_MESH_PVC_CACHE.get(type_idx)
        if data is None or len(data) == 0:
            self._send_json({"error": "no PVC sub-mesh for this type"}, 404)
            return
        self._send_binary(data)

    def _send_pmt_mesh_colors(self, path):
        try:
            type_idx = int(path.split("/")[-1])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid type"}, 400)
            return
        data = PMT_COLOR_CACHE.get(type_idx)
        if data is None:
            self._send_json({"error": "mesh type not available"}, 404)
            return
        self._send_binary(data, content_type="application/octet-stream")

    def _send_scan_mesh_list(self):
        names = sorted(SCAN_MESH_CACHE.keys())
        self._send_json({"meshes": names})

    def _send_scan_mesh(self, path):
        parts = path.split("/")
        if len(parts) < 4:
            self._send_json({"error": "invalid path"}, 400)
            return
        mesh_name = parts[-2]
        data_type = parts[-1]
        entry = SCAN_MESH_CACHE.get(mesh_name)
        if entry is None:
            self._send_json({"error": f"mesh '{mesh_name}' not available"}, 404)
            return
        verts_bytes, tris_bytes = entry
        if data_type == "verts":
            self._send_binary(verts_bytes)
        elif data_type == "tris":
            self._send_binary(tris_bytes)
        else:
            self._send_json({"error": "invalid data type"}, 400)

    def _send_pmt_tips(self):
        if PMT_TIPS_CACHE is None:
            self._send_json({"tips": []})
            return
        self._send_json({"tips": PMT_TIPS_CACHE})

    def _handle_trace(self, params):
        try:
            mx = float(params.get("mx", ["0"])[0])
            my = float(params.get("my", ["0"])[0])
            mz = float(params.get("mz", ["2000"])[0])
            dx = float(params.get("dx", ["0"])[0])
            dy = float(params.get("dy", ["0"])[0])
            dz = float(params.get("dz", ["-1"])[0])
            photons_per_cm = int(params.get("photons_per_cm", ["150"])[0])
            photons_per_cm = min(max(photons_per_cm, 1), 1000)
        except (ValueError, TypeError):
            self._send_json({"error": "invalid parameters"}, 400)
            return

        rng = np.random.default_rng()
        t0 = time.time()
        reload_lappd_corrections(geometry)
        hits = trace_cherenkov(
            (mx, my, mz), (dx, dy, dz), photons_per_cm, geometry, rng=rng,
        )
        elapsed = time.time() - t0

        total_hits = int(hits[:, 0].sum())
        total_photons = photons_per_cm * 401

        # Return positions by component type — these are drawn as dots
        comp = hits[:, 8]
        pmt_positions    = hits[comp == 2.0, 2:5].tolist()
        struct_positions = hits[comp == 1.0, 2:5].tolist()
        lappd_positions  = hits[comp == 3.0, 2:5].tolist()
        tank_positions   = hits[comp == 4.0, 2:5].tolist()

        self._send_json({
            "pmt_positions": pmt_positions,
            "struct_positions": struct_positions,
            "lappd_positions": lappd_positions,
            "tank_positions": tank_positions,
            "counts": {
                "pmt": len(pmt_positions),
                "struct": len(struct_positions),
                "lappd": len(lappd_positions),
                "tank": len(tank_positions),
            },
            "total_hits": total_hits,
            "total_photons": total_photons,
            "time_ms": round(elapsed * 1000, 1),
        })


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANNIE Cherenkov Visualization</title>
<style>
  body { margin:0; overflow:hidden; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  #controls {
    position:absolute; top:10px; left:10px; z-index:100;
    background:rgba(245,247,250,0.92); color:#333; padding:14px 16px;
    border-radius:8px; width:270px; font-size:13px; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 2px 12px rgba(0,0,0,0.08);
  }
  #controls h3 { margin:0 0 10px; color:#111; font-size:15px; }
  #controls label { display:block; margin:6px 0; }
  #controls .row { display:flex; gap:6px; align-items:center; }
  #controls .row input[type=number] { width:60px; background:#fff; border:1px solid rgba(0,0,0,0.2); color:#111; padding:3px 6px; border-radius:4px; }
  #controls input[type=range] { width:100%; margin:2px 0; accent-color:#4488cc; }
  #controls input[type=checkbox] { accent-color:#4488cc; }
  #controls button {
    width:100%; margin-top:8px; padding:6px;
    background:#4488cc; color:#fff; border:none; border-radius:4px;
    font-size:13px; cursor:pointer;
  }
  #controls button:hover { background:#5599dd; }
  #controls button:disabled { opacity:0.5; cursor:default; }
  #result { margin-top:8px; font-size:12px; line-height:1.5; color:#333; }
  #result b { color:#111; }
  #status {
    position:absolute; top:10px; right:10px; z-index:100;
    background:rgba(245,247,250,0.92); color:#333; padding:8px 14px;
    border-radius:6px; font-size:13px; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12);
  }
  #info {
    position:absolute; bottom:10px; left:10px; z-index:100;
    color:rgba(0,0,0,0.5); font-size:12px;
    background:rgba(255,255,255,0.7); padding:4px 10px; border-radius:4px;
  }
  .hit-summary { display:grid; grid-template-columns:1fr 1fr; gap:2px 12px; }
  .hit-summary .label { color:#666; }
  .hit-summary .value { text-align:right; color:#000; }
  #pmt-adjust {
    display:none; position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:340px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #pmt-adjust .adj-row { display:flex; align-items:center; gap:8px; margin:4px 0; justify-content:center; }
  #pmt-adjust .adj-row label { min-width:80px; text-align:right; }
  #pmt-adjust .adj-row input[type=range] { width:140px; margin:0; accent-color:#4488cc; }
  #pmt-adjust .adj-row .val { min-width:50px; text-align:left; font-family:monospace; }
  #pmt-adjust button {
    margin:6px 4px 0; padding:5px 16px; border:none; border-radius:4px; font-size:12px; cursor:pointer;
  }
  #pmt-adjust #adj-save { background:#4488cc; color:#fff; }
  #pmt-adjust #adj-save:hover { background:#5599dd; }
  #pmt-adjust #adj-reset { background:#888; color:#fff; }
  #pmt-adjust #adj-reset:hover { background:#999; }
  #pmt-adjust #adj-cancel { background:#ccc; color:#333; }
  #pmt-adjust #adj-cancel:hover { background:#ddd; }
  #housing-popup {
    display:none; position:fixed; bottom:80px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:400px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #surfboard-popup {
    display:none; position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:400px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #surfboard-popup .sb-title { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
  #surfboard-popup .sb-title h3 { margin:0; font-size:14px; }
  #surfboard-popup .sb-title button { background:none; border:none; font-size:16px; cursor:pointer; color:#666; padding:2px 6px; border-radius:4px; }
  #surfboard-popup .sb-title button:hover { background:rgba(0,0,0,0.1); color:#000; }
  #surfboard-popup .sb-row { display:flex; align-items:center; gap:6px; margin:4px 0; justify-content:center; }
  #surfboard-popup .sb-row label { min-width:80px; text-align:right; font-size:12px; }
  #surfboard-popup .sb-row input[type=number] { width:60px; text-align:center; font-family:monospace; font-size:12px; }
  #surfboard-popup .sb-row input[type=range] { width:180px; margin:0; accent-color:#4488cc; }
  #surfboard-popup button { margin:6px 4px 0; padding:5px 16px; border:none; border-radius:4px; font-size:12px; cursor:pointer; }
  #surfboard-popup #surfReset { background:#888; color:#fff; }
  #surfboard-popup #surfReset:hover { background:#999; }
</style>
</head>
<body>
<div id="controls">
  <h3>Muon Controls</h3>
  <label>Position
    <div class="row">
      X: <input type="number" id="mx" value="0" step="50">
      Y: <input type="number" id="my" value="0" step="50">
      Z: <input type="number" id="mz" value="2000" step="50">
    </div>
  </label>
  <label>θ (polar) <input type="range" id="theta" min="0" max="3.1416" step="0.01" value="3.1416"></label>
  <label>φ (azimuth) <input type="range" id="phi" min="0" max="6.2832" step="0.01" value="0"></label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label>Ambient <span id="ambientVal">0.50</span>
    <input type="range" id="ambient" min="0" max="2.0" step="0.01" value="0.50"></label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label>LAPPD dx <input type="number" id="lappdDx" step="1" value="0" style="width:70px;"></label>
  <label>LAPPD dy <input type="number" id="lappdDy" step="1" value="0" style="width:70px;"></label>
  <label>LAPPD dz <input type="number" id="lappdDz" step="1" value="0" style="width:70px;"></label>
  <button id="saveLappdCorr">Save Correction</button>
  <button id="focusLAPPD" disabled>Focus on LAPPD</button>
  <label style="margin-top:6px;"><input type="checkbox" id="lappdGrey"> Grey LAPPD</label>
  <div id="surfboard-adjust" style="display:none;"></div>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label style="margin-top:6px;"><input type="checkbox" id="structGrey"> Grey Structure</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="structOpacity" min="0" max="1.0" step="0.01" value="1.0" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="pmtGrey"> Grey PMTs</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="pmtOpacity" min="0" max="1.0" step="0.01" value="0.85" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="showHW"> Show Holders</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showRefSpheres"> Show Reference Spheres</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="refOpacity" min="0" max="0.5" step="0.01" value="0.12" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="showTubeIDs"> Show Tube IDs</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showScan"> Show Scan Overlay</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">
    Mesh <select id="scanMeshSelect" style="font-size:11px;">
      <option value="SuperStructure">SuperStructure</option>
      <option value="AllPMTs">All PMTs</option>
      <option value="BottomLayer">Bottom Layer</option>
      <option value="TopLayer">Top Layer</option>
      <option value="Panel-1">Panel 1</option>
      <option value="Panel-2">Panel 2</option>
      <option value="Panel-3">Panel 3</option>
      <option value="Panel-4">Panel 4</option>
      <option value="Panel-5">Panel 5</option>
      <option value="Panel-6">Panel 6</option>
      <option value="Panel-7">Panel 7</option>
      <option value="Panel-8">Panel 8</option>
      <option value="Panel-1-PMTs">Panel 1 PMTs</option>
      <option value="Panel-2-PMTs">Panel 2 PMTs</option>
      <option value="Panel-3-PMTs">Panel 3 PMTs</option>
      <option value="Panel-4-PMTs">Panel 4 PMTs</option>
      <option value="Panel-5-PMTs">Panel 5 PMTs</option>
      <option value="Panel-6-PMTs">Panel 6 PMTs</option>
      <option value="Panel-7-PMTs">Panel 7 PMTs</option>
      <option value="Panel-8-PMTs">Panel 8 PMTs</option>
      <option value="TopPMTs">Top PMTs</option>
      <option value="BottomPMTs">Bottom PMTs</option>
      <option value="TankLid">Tank Lid</option>
    </select>
  </label>
  <label style="margin-top:2px;"><input type="checkbox" id="showScanTips"> Show Scan Tips</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showCone" checked> Show Cone Guide</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showRayLines"> Show Ray Lines</label>
  <button id="traceBtn" style="margin-top:8px;">Trace Cherenkov Photons</button>
  <div id="traceResult" style="margin-top:6px;font-size:12px;line-height:1.5;"></div>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.15);">
  <h3 style="margin-top:6px;">View</h3>
  <label>Azimuth <span id="viewAzVal">36</span>
    <input type="range" id="viewAz" min="0" max="360" step="1" value="36"></label>
  <label>Polar <span id="viewElVal">77</span>
    <input type="range" id="viewEl" min="0" max="180" step="1" value="77"></label>
  <label>Distance <span id="viewDistVal">4416</span>
    <input type="range" id="viewDist" min="100" max="10000" step="50" value="4416"></label>
  <div style="margin-top:4px;font-size:11px;color:#666;">Real-time shadows · No server calls</div>
</div>
<div id="status">Loading geometry…</div>
<div id="info">Drag to orbit · Scroll to zoom · Right-drag to pan</div>
<div id="pmt-adjust">
  <div style="margin-bottom:6px;"><b>PMT <span id="adj-tube-id"></span></b> <span id="adj-type" style="color:#888;font-size:11px;"></span></div>
  <div id="adj-sliders">
    <div class="adj-row">
      <label id="adj-label-0">Axial</label>
      <input type="range" id="adj-0" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-0">0.0</span> mm
    </div>
    <div class="adj-row">
      <label id="adj-label-1">Tangential</label>
      <input type="range" id="adj-1" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-1">0.0</span> mm
    </div>
    <div class="adj-row">
      <label id="adj-label-2">Vertical</label>
      <input type="range" id="adj-2" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-2">0.0</span> mm
    </div>
  </div>
  <div style="margin-top:4px;">
    <button id="adj-save">Save</button>
    <button id="adj-reset">Reset</button>
    <button id="adj-cancel">Cancel</button>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ---- State ----
let muonGroup = null;
let coneVisual = null;
let spotLight = null;
let spotTarget = null;
let housingData = null;
let housingMeshes = [];
let tracePoints = null;   // group of dot meshes for traced photon hits
let rayLines = null;      // group of line meshes for sampled ray paths
let refSpheres = null;    // group of translucent reference spheres
let scanOverlay = null;   // group for scan mesh overlay
let traceResultEl = null;
let pmtGroup = null;      // group of PMT body meshes
let pmtHWGroup = null;    // group of PMT holder meshes
let pmtData = null;       // full PMT data from /api/pmts
let selectedIdx = -1;     // index of selected PMT, -1 = none
let selectedSurfboard = -1; // index of selected surfboard, -1 = none
let selectedHousing = -1; // index of selected LAPPD housing, -1 = none
let selectedMesh = null;  // the mesh object currently highlighted

// ---- DOM refs ----
const statusEl = document.getElementById('status');
const thetaSlider = document.getElementById('theta');
const phiSlider = document.getElementById('phi');

// ---- Scene ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x7088a0);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 20000);
camera.up.set(0, 0, 1);
camera.position.set(3500, 2500, 3000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NoToneMapping;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2000);
controls.update();

// Environment map for glass/reflections
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(renderer), 0.04).texture;
pmremGenerator.dispose();

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);

// Spotlight from muon vertex (Cherenkov cone light source)
spotTarget = new THREE.Object3D();
scene.add(spotTarget);

spotLight = new THREE.SpotLight(0xffffff, 1.0);
spotLight.angle = 0.73;
spotLight.penumbra = 0.08;
spotLight.decay = 0;
spotLight.distance = 0;
spotLight.castShadow = true;
spotLight.shadow.mapSize.width = 2048;
spotLight.shadow.mapSize.height = 2048;
spotLight.shadow.camera.near = 10;
spotLight.shadow.camera.far = 5000;
spotLight.shadow.bias = 0;
spotLight.shadow.normalBias = 0;
scene.add(spotLight);

// Grid helper (horizontal, Z-up frame)
const grid = new THREE.GridHelper(3000, 20, 0x8899aa, 0x667788);
grid.position.z = 2000;
grid.receiveShadow = false;
scene.add(grid);

// ---- Labeled axes + origin ----
function makeTextSprite(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'Bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, 64, 32);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(200, 100, 1);
    return sprite;
}

function addAxesAndOrigin() {
    const axisLen = 600;
    const group = new THREE.Group();

    // Axis arrows
    const axisDefs = [
        { dir: [1,0,0], color: 0xff0000, label: 'X' },
        { dir: [0,1,0], color: 0x00ff00, label: 'Y' },
        { dir: [0,0,1], color: 0x0066ff, label: 'Z' },
    ];
    for (const a of axisDefs) {
        const d = new THREE.Vector3(a.dir[0], a.dir[1], a.dir[2]);
        const arrow = new THREE.ArrowHelper(d, new THREE.Vector3(0,0,0),
            axisLen, a.color, 80, 40);
        group.add(arrow);

        // Label at tip
        const tip = d.clone().multiplyScalar(axisLen * 1.25);
        const sp = makeTextSprite(a.label, '#' + a.color.toString(16).padStart(6, '0'));
        sp.position.copy(tip);
        group.add(sp);
    }

    // Negative axis stubs (dashed or thin)
    for (const a of axisDefs) {
        const d = new THREE.Vector3(-a.dir[0], -a.dir[1], -a.dir[2]);
        const mat = new THREE.LineBasicMaterial({
            color: a.color, transparent: true, opacity: 0.25,
        });
        const pts = [new THREE.Vector3(0,0,0), d.clone().multiplyScalar(axisLen * 0.4)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        group.add(new THREE.Line(geo, mat));
    }

    // Origin sphere
    const originGeo = new THREE.SphereGeometry(20, 16, 12);
    const originMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const originMesh = new THREE.Mesh(originGeo, originMat);
    originMesh.position.set(0, 0, 0);
    group.add(originMesh);

    // "O" label
    const oLabel = makeTextSprite('O', '#000000');
    oLabel.position.set(0, -70, 0);
    group.add(oLabel);

    scene.add(group);
}

addAxesAndOrigin();

// ---- Material colours ----
const SURFACE_COLOR = new THREE.Color(0xffffff);
const GREY_COLOR = new THREE.Color(0x999999);

// ---- Load mesh ----
async function loadMesh() {
    const [vertsResp, trisResp] = await Promise.all([
        fetch('/api/mesh/verts'),
        fetch('/api/mesh/tris'),
    ]);
    const vertsBuf = await vertsResp.arrayBuffer();
    const trisBuf = await trisResp.arrayBuffer();

    const positions = new Float32Array(vertsBuf);
    const indices = new Int32Array(trisBuf);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    return geo;
}

// ---- Load PMTs ----
async function loadPMTs() {
    const resp = await fetch('/api/pmts');
    return await resp.json();
}

// ---- PMT type colours ----
const PMT_COLORS = {
    'LUX': 0xffaa00,
    'ETEL': 0x00ccff,
    'Hamamatsu': 0x44aaff,
    'Watchboy': 0xff8844,
    'Watchman': 0xff8844,
};

// ---- Muon + light update ----
function updateMuonAndLight(pos, dir) {
    const d = new THREE.Vector3(dir.x, dir.y, dir.z).normalize();
    const p = new THREE.Vector3(pos.x, pos.y, pos.z);

    // Muon vertex visual
    if (muonGroup) scene.remove(muonGroup);
    muonGroup = new THREE.Group();
    const vGeo = new THREE.SphereGeometry(20, 16, 12);
    const vMat = new THREE.MeshBasicMaterial({ color: 0xff6644 });
    const vMesh = new THREE.Mesh(vGeo, vMat);
    vMesh.position.copy(p);
    muonGroup.add(vMesh);
    const arrow = new THREE.ArrowHelper(d, p, 400, 0xff6644, 60, 30);
    muonGroup.add(arrow);
    scene.add(muonGroup);

    // Cone guide outline (wireframe, doesn't cast shadow)
    if (coneVisual) scene.remove(coneVisual);
    const coneLen = 800;
    const baseR = coneLen * Math.tan(0.73);
    const coneGeo = new THREE.ConeGeometry(baseR, coneLen, 36, 1, true);
    const coneMat = new THREE.MeshBasicMaterial({
        color: 0x66aaaa, wireframe: true, transparent: true, opacity: 0.25,
    });
    coneVisual = new THREE.Mesh(coneGeo, coneMat);
    const coneCenter = p.clone().add(d.clone().multiplyScalar(coneLen / 2));
    coneVisual.position.copy(coneCenter);
    const upDir = new THREE.Vector3(0, 1, 0);
    coneVisual.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(upDir, d.clone().negate()));
    coneVisual.castShadow = false;
    coneVisual.receiveShadow = false;
    coneVisual.visible = document.getElementById('showCone').checked;
    scene.add(coneVisual);

    // Spotlight follows muon vertex
    spotLight.position.copy(p);
    const targetPos = p.clone().add(d.clone().multiplyScalar(3000));
    spotTarget.position.copy(targetPos);
    spotLight.target = spotTarget;
}

// ---- Ambient slider ----
const ambientSlider = document.getElementById('ambient');
const ambientLabel = document.getElementById('ambientVal');
ambientSlider.addEventListener('input', () => {
    ambient.intensity = parseFloat(ambientSlider.value);
    ambientLabel.textContent = ambientSlider.value;
});

// ---- Update from controls ----
function updateScene() {
    const mx = parseFloat(document.getElementById('mx').value);
    const my = parseFloat(document.getElementById('my').value);
    const mz = parseFloat(document.getElementById('mz').value);
    const theta = parseFloat(thetaSlider.value);
    const phi = parseFloat(phiSlider.value);

    const dx = Math.sin(theta) * Math.cos(phi);
    const dy = Math.sin(theta) * Math.sin(phi);
    const dz = Math.cos(theta);

    updateMuonAndLight({ x: mx, y: my, z: mz }, { x: dx, y: dy, z: dz });
}

// ---- Trace Cherenkov photons ----
async function doTrace() {
    const mx = document.getElementById('mx').value;
    const my = document.getElementById('my').value;
    const mz = document.getElementById('mz').value;
    const theta = parseFloat(thetaSlider.value);
    const phi = parseFloat(phiSlider.value);
    const dx = Math.sin(theta) * Math.cos(phi);
    const dy = Math.sin(theta) * Math.sin(phi);
    const dz = Math.cos(theta);
    const photonsPerCm = 150;

    const btn = document.getElementById('traceBtn');
    btn.disabled = true;
    btn.textContent = 'Tracing…';
    traceResultEl = document.getElementById('traceResult');

    try {
        const url = `/api/trace?mx=${mx}&my=${my}&mz=${mz}&dx=${dx.toFixed(6)}&dy=${dy.toFixed(6)}&dz=${dz.toFixed(6)}&photons_per_cm=${photonsPerCm}`;
        const resp = await fetch(url);
        const data = await resp.json();

        // Remove previous trace dots
        if (tracePoints) scene.remove(tracePoints);

        const dotGeo = new THREE.SphereGeometry(8, 6, 6);
        const pmtMat = new THREE.MeshBasicMaterial({ color: 0x44dd88 });
        const structMat = new THREE.MeshBasicMaterial({ color: 0xff8844 });
        const lappdMat = new THREE.MeshBasicMaterial({ color: 0xdd44dd });
        const tankMat = new THREE.MeshBasicMaterial({ color: 0x4488ff });

        tracePoints = new THREE.Group();
        for (const p of data.pmt_positions) {
            const m = new THREE.Mesh(dotGeo, pmtMat);
            m.position.set(p[0], p[1], p[2]);
            tracePoints.add(m);
        }
        for (const p of data.struct_positions) {
            const m = new THREE.Mesh(dotGeo, structMat);
            m.position.set(p[0], p[1], p[2]);
            tracePoints.add(m);
        }
        for (const p of data.lappd_positions) {
            const m = new THREE.Mesh(dotGeo, lappdMat);
            m.position.set(p[0], p[1], p[2]);
            tracePoints.add(m);
        }
        for (const p of data.tank_positions) {
            const m = new THREE.Mesh(dotGeo, tankMat);
            m.position.set(p[0], p[1], p[2]);
            tracePoints.add(m);
        }
        scene.add(tracePoints);

        // Ray lines: sample up to 500 lines from muon vertex to hit positions
        if (rayLines) scene.remove(rayLines);
        const showLines = document.getElementById('showRayLines').checked;
        if (showLines && data.total_hits > 0) {
            const muonPos = new THREE.Vector3(
                parseFloat(document.getElementById('mx').value),
                parseFloat(document.getElementById('my').value),
                parseFloat(document.getElementById('mz').value),
            );
            // Collect all hit positions
            const allHits = [];
            for (const p of data.pmt_positions) allHits.push({ pos: p, color: 0x44dd88 });
            for (const p of data.struct_positions) allHits.push({ pos: p, color: 0xff8844 });
            for (const p of data.lappd_positions) allHits.push({ pos: p, color: 0xdd44dd });
            for (const p of data.tank_positions) allHits.push({ pos: p, color: 0x4488ff });
            // Sample
            const maxLines = 500;
            const step = Math.max(1, Math.floor(allHits.length / maxLines));
            const lineGroup = new THREE.Group();
            for (let i = 0; i < allHits.length; i += step) {
                const h = allHits[i];
                const geo = new THREE.BufferGeometry().setFromPoints([
                    muonPos,
                    new THREE.Vector3(h.pos[0], h.pos[1], h.pos[2]),
                ]);
                const mat = new THREE.LineBasicMaterial({
                    color: h.color,
                    transparent: true,
                    opacity: 0.25,
                });
                lineGroup.add(new THREE.Line(geo, mat));
            }
            rayLines = lineGroup;
            scene.add(rayLines);
        }

        const c = data.counts;
        traceResultEl.innerHTML = `<b>${data.total_hits}</b>/<b>${data.total_photons}</b> hits `
            + `(PMT <b>${c.pmt}</b>, `
            + `struct <b>${c.struct}</b>, `
            + `LAPPD <b>${c.lappd}</b>, `
            + `tank <b>${c.tank}</b>) `
            + `<span style="color:#888">· ${photonsPerCm} ph/cm</span> `
            + `in ${data.time_ms} ms`;
    } catch (e) {
        traceResultEl.textContent = 'Trace failed: ' + e.message;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Trace Cherenkov Photons';
    }
}

// ---- Events ----
thetaSlider.addEventListener('input', updateScene);
phiSlider.addEventListener('input', updateScene);
document.getElementById('mx').addEventListener('change', updateScene);
document.getElementById('my').addEventListener('change', updateScene);
document.getElementById('mz').addEventListener('change', updateScene);
document.getElementById('showCone').addEventListener('change', () => {
    if (coneVisual) coneVisual.visible = document.getElementById('showCone').checked;
});
document.getElementById('showRayLines').addEventListener('change', () => {
    if (rayLines) rayLines.visible = document.getElementById('showRayLines').checked;
});
document.getElementById('traceBtn').addEventListener('click', doTrace);

// ---- Init ----
async function init() {
    try {
        statusEl.textContent = 'Loading structure mesh…';
        const meshGeo = await loadMesh();

        // Flatten indexed geometry to non-indexed (indexed path has rendering issues)
        const pos = meshGeo.getAttribute('position');
        const idx = meshGeo.getIndex();
        const triCount = idx.count / 3;
        const flatPos = new Float32Array(triCount * 9);
        for (let i = 0; i < triCount; i++) {
            for (let j = 0; j < 3; j++) {
                const vi = idx.getX(i * 3 + j);
                flatPos[i * 9 + j * 3 + 0] = pos.getX(vi);
                flatPos[i * 9 + j * 3 + 1] = pos.getY(vi);
                flatPos[i * 9 + j * 3 + 2] = pos.getZ(vi);
            }
        }
        const flatGeo = new THREE.BufferGeometry();
        flatGeo.setAttribute('position', new THREE.BufferAttribute(flatPos, 3));
        flatGeo.computeVertexNormals();

        const meshMat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            roughness: 0.5,
            metalness: 0.05,
            transparent: true,
            opacity: 1.0,
            side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(flatGeo, meshMat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        statusEl.textContent = 'Loading PMTs…';
        const pmtData = await loadPMTs();

        // Load PMT mesh geometries for needed types
        const meshGeos = {};       // glass outer shell (full mesh)
        const meshPcGeos = {};     // PC inner layer (brown, 1 mm inside glass)
        const meshPvcGeos = {};    // PVC housing/wings (dark grey)
        const hwMeshGeos = {};
        const neededTypes = new Set(pmtData.mesh_types || []);
        const neededHwTypes = new Set((pmtData.hw_mesh_types || []).filter(v => v >= 0));
        for (const mt of neededTypes) {
            try {
                const meshResp = await fetch(`/api/pmt_mesh/${mt}`);
                if (meshResp.ok) {
                    const buf = await meshResp.arrayBuffer();
                    const flat = new Float32Array(buf);
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                    geo.computeVertexNormals();
                    meshGeos[mt] = geo;
                }
                // PC sub-mesh (translucent brown inner layer)
                const pcResp = await fetch(`/api/pmt_mesh_pc/${mt}`);
                if (pcResp.ok) {
                    const buf = await pcResp.arrayBuffer();
                    if (buf.byteLength > 0) {
                        const flat = new Float32Array(buf);
                        const geo = new THREE.BufferGeometry();
                        geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                        geo.computeVertexNormals();
                        meshPcGeos[mt] = geo;
                    }
                }
                // PVC sub-mesh (dark grey housing / wings)
                const pvcResp = await fetch(`/api/pmt_mesh_pvc/${mt}`);
                if (pvcResp.ok) {
                    const buf = await pvcResp.arrayBuffer();
                    if (buf.byteLength > 0) {
                        const flat = new Float32Array(buf);
                        const geo = new THREE.BufferGeometry();
                        geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                        geo.computeVertexNormals();
                        meshPvcGeos[mt] = geo;
                    }
                }
            } catch (e) {
                console.warn(`No mesh for type ${mt}`);
            }
        }
        for (const mt of neededHwTypes) {
            try {
                const [meshResp, colResp] = await Promise.all([
                    fetch(`/api/pmt_mesh/${mt}`),
                    fetch(`/api/pmt_mesh_colors/${mt}`),
                ]);
                if (meshResp.ok) {
                    const buf = await meshResp.arrayBuffer();
                    const flat = new Float32Array(buf);
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                    if (colResp.ok) {
                        const colBuf = await colResp.arrayBuffer();
                        const colors = new Uint8Array(colBuf);
                        geo.setAttribute('color', new THREE.BufferAttribute(colors, 4, true));
                    }
                    geo.computeVertexNormals();
                    hwMeshGeos[mt] = geo;
                }
            } catch (e) {
                console.warn(`No HW mesh for type ${mt}`);
            }
        }

        // Render PMTs — dual-layer: glass outer shell + PC inner coating
        pmtGroup = new THREE.Group();
        pmtHWGroup = new THREE.Group();

        // Materials (cloned per-PMT so grey/opacity toggles work individually)
        function makeGlassMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.97, 0.97, 0.99),  // faint cool white
                roughness: 0.0,
                metalness: 0.0,
                transparent: true,
                opacity: 1.0,
                side: THREE.DoubleSide,
                envMap: scene.environment,
                envMapIntensity: 1.8,
                depthWrite: false,
                transmission: 0.92,
                thickness: 0.5,
                ior: 1.5,
            });
        }
        function makePcMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.55, 0.20, 0.10),  // warm brown
                roughness: 0.4,
                metalness: 0.0,
                transparent: true,
                opacity: 0.35,   // thin-film PC — subtle tint through glass
                side: THREE.DoubleSide,
                depthWrite: true,
            });
        }
        function makePvcMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.10, 0.10, 0.12),  // dark grey
                roughness: 0.8,
                metalness: 0.0,
                side: THREE.DoubleSide,
                depthWrite: true,
            });
        }

        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const r = pmtData.radii[i];
            const d = pmtData.directions[i];
            const type = pmtData.types[i];
            const color = PMT_COLORS[type] || 0x999999;

            const mt = pmtData.mesh_types ? pmtData.mesh_types[i] : -1;
            const geo = meshGeos[mt];

            if (geo) {
                const ip = pmtData.instance_positions[i];
                const q = pmtData.quaternions[i];

                // Glass outer shell (transparent, reflective)
                const glassMat = makeGlassMat();
                const glassMesh = new THREE.Mesh(geo, glassMat);
                glassMesh.position.set(ip[0], ip[1], ip[2]);
                glassMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                glassMesh.castShadow = true;
                glassMesh.receiveShadow = true;
                glassMesh.renderOrder = 1;
                glassMesh.userData.pmtIdx = i;
                glassMesh.userData.tubeId = pmtData.detector_nums[i];
                pmtGroup.add(glassMesh);

                // PC inner layer (brown, offset 1 mm inside glass)
                const pcGeo = meshPcGeos[mt];
                if (pcGeo) {
                    const pcMatMesh = makePcMat();
                    const pcMesh = new THREE.Mesh(pcGeo, pcMatMesh);
                    pcMesh.position.set(ip[0], ip[1], ip[2]);
                    pcMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    pcMesh.castShadow = true;
                    pcMesh.receiveShadow = true;
                    pcMesh.renderOrder = 0;
                    pcMesh.userData.pmtIdx = i;
                    pcMesh.userData.tubeId = pmtData.detector_nums[i];
                    pcMesh.userData.isPC = true;
                    pmtGroup.add(pcMesh);
                }

                // PVC housing / wings (dark grey, only LUX/ETEL have these)
                const pvcGeo = meshPvcGeos[mt];
                if (pvcGeo) {
                    const pvcMat = makePvcMat();
                    const pvcMesh = new THREE.Mesh(pvcGeo, pvcMat);
                    pvcMesh.position.set(ip[0], ip[1], ip[2]);
                    pvcMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    pvcMesh.castShadow = true;
                    pvcMesh.receiveShadow = true;
                    pvcMesh.renderOrder = 2;
                    pvcMesh.userData.pmtIdx = i;
                    pvcMesh.userData.tubeId = pmtData.detector_nums[i];
                    pvcMesh.userData.isPVC = true;
                    pmtGroup.add(pvcMesh);
                }

                // Hardware mesh at same position/orientation
                const hwmt = pmtData.hw_mesh_types ? pmtData.hw_mesh_types[i] : -1;
                const hwGeo = hwMeshGeos[hwmt];
                if (hwGeo) {
                    const hwHasColors = hwGeo.getAttribute('color') != null;
                    const hwMat = new THREE.MeshPhysicalMaterial({
                        color: hwHasColors ? 0xffffff : 0x887766,
                        vertexColors: hwHasColors,
                        roughness: 0.5,
                        metalness: 0.3,
                        transparent: true,
                        opacity: 0.7,
                        side: THREE.DoubleSide,
                    });
                    const hwMesh = new THREE.Mesh(hwGeo, hwMat);
                    hwMesh.position.set(ip[0], ip[1], ip[2]);
                    hwMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    hwMesh.castShadow = true;
                    hwMesh.receiveShadow = true;
                    hwMesh.userData.pmtIdx = i;
                    hwMesh.userData.tubeId = pmtData.detector_nums[i];
                    pmtHWGroup.add(hwMesh);
                }
            } else {
                const sphereGeo = new THREE.SphereGeometry(r * 0.35, 16, 12);
                const sphereMat = new THREE.MeshStandardMaterial({
                    color: color,
                    roughness: 0.4,
                    metalness: 0.05,
                    transparent: true,
                    opacity: 0.85,
                });
                const sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.set(c[0], c[1], c[2]);
                sphere.castShadow = true;
                sphere.receiveShadow = true;
                sphere.userData.pmtIdx = i;
                sphere.userData.tubeId = pmtData.detector_nums[i];
                pmtGroup.add(sphere);
            }

            const dir = new THREE.Vector3(d[0], d[1], d[2]).normalize();
            const origin = new THREE.Vector3(c[0], c[1], c[2]);
            const arrow = new THREE.ArrowHelper(dir, origin, 50, color, 18, 10);
            pmtGroup.add(arrow);

            // Red dot at bulb tip (centers[i]) for alignment debugging
            const dotGeo = new THREE.SphereGeometry(6, 8, 6);
            const dotMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.position.set(c[0], c[1], c[2]);
            pmtGroup.add(dot);
        }
        scene.add(pmtGroup);
        pmtHWGroup.visible = false;
        scene.add(pmtHWGroup);

        // ---- Tube ID labels ----
        const labelGroup = new THREE.Group();
        labelGroup.visible = false;
        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const tid = pmtData.detector_nums[i];
            const canvas = document.createElement('canvas');
            canvas.width = 192;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgba(0,0,0,0.55)';
            ctx.beginPath();
            ctx.roundRect(0, 0, 192, 64, 8);
            ctx.fill();
            ctx.font = 'bold 28px monospace';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(tid), 96, 34);
            const tex = new THREE.CanvasTexture(canvas);
            tex.needsUpdate = true;
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, sizeAttenuation: true });
            const sprite = new THREE.Sprite(mat);
            sprite.position.set(c[0], c[1], c[2] + 200);
            sprite.scale.set(400, 140, 1);
            sprite.userData.pmtIdx = i;
            labelGroup.add(sprite);
        }
        scene.add(labelGroup);

        // Reference spheres at each PMT center+radius (translucent)
        refSpheres = new THREE.Group();
        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const r = pmtData.radii[i];
            const type = pmtData.types[i];
            const color = PMT_COLORS[type] || 0x999999;
            const sphereGeo = new THREE.SphereGeometry(r, 32, 24);
            const sphereMat = new THREE.MeshPhysicalMaterial({
                color: color,
                transparent: true,
                opacity: 0.12,
                roughness: 0.0,
                metalness: 0.0,
                depthWrite: false,
                side: THREE.DoubleSide,
            });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.set(c[0], c[1], c[2]);
            refSpheres.add(sphere);
        }
        refSpheres.visible = false;
        scene.add(refSpheres);

        // Scan overlay (loaded on demand)
        scanOverlay = new THREE.Group();
        scanOverlay.visible = false;
        scene.add(scanOverlay);

        async function loadScanMesh(name) {
            const [vertsResp, trisResp] = await Promise.all([
                fetch(`/api/scan_mesh/${name}/verts`),
                fetch(`/api/scan_mesh/${name}/tris`),
            ]);
            if (!vertsResp.ok || !trisResp.ok) return null;
            const vertsBuf = await vertsResp.arrayBuffer();
            const trisBuf = await trisResp.arrayBuffer();
            const positions = new Float32Array(vertsBuf);
            const indices = new Int32Array(trisBuf);
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setIndex(new THREE.BufferAttribute(indices, 1));
            // Flatten indexed to non-indexed (indexed path has rendering issues)
            const pos = geo.getAttribute('position');
            const idx = geo.getIndex();
            const triCount = idx.count / 3;
            const flatPos = new Float32Array(triCount * 9);
            for (let i = 0; i < triCount; i++) {
                for (let j = 0; j < 3; j++) {
                    const vi = idx.getX(i * 3 + j);
                    flatPos[i * 9 + j * 3 + 0] = pos.getX(vi);
                    flatPos[i * 9 + j * 3 + 1] = pos.getY(vi);
                    flatPos[i * 9 + j * 3 + 2] = pos.getZ(vi);
                }
            }
            const flatGeo = new THREE.BufferGeometry();
            flatGeo.setAttribute('position', new THREE.BufferAttribute(flatPos, 3));
            flatGeo.computeVertexNormals();
            return flatGeo;
        }

        statusEl.textContent = `Loaded ${pmtData.centers.length} PMTs.`;

        // Load ANNIE LAPPD housings (if present)
        const housingResp = await fetch('/api/housing');
        housingData = await housingResp.json();
        housingMeshes.length = 0;
        let housingCorrOrigins = [];  // original centers per housing (before global correction)
        if (Array.isArray(housingData.housing) && housingData.housing.length > 0) {
            const boxMat = new THREE.MeshStandardMaterial({
                color: 0x446688,
                transparent: true,
                opacity: 0.25,
                roughness: 0.6,
                metalness: 0.0,
                side: THREE.DoubleSide,
            });
            const pcMat = new THREE.MeshStandardMaterial({
                color: 0x88bbdd,
                roughness: 0.3,
                metalness: 0.1,
                side: THREE.DoubleSide,
            });
            let hidx = 0;
            for (const h of housingData.housing) {
                const origCenter = new THREE.Vector3(h.center[0], h.center[1], h.center[2]);
                const origPC = new THREE.Vector3(h.pc_center[0], h.pc_center[1], h.pc_center[2]);

                const boxGeo = new THREE.BoxGeometry(h.half[0]*2, h.half[1]*2, h.half[2]*2);
                const boxMesh = new THREE.Mesh(boxGeo, boxMat);
                boxMesh.position.copy(origCenter);
                const m4 = new THREE.Matrix4();
                m4.set(
                    h.axis_x[0], h.axis_y[0], h.axis_z[0], 0,
                    h.axis_x[1], h.axis_y[1], h.axis_z[1], 0,
                    h.axis_x[2], h.axis_y[2], h.axis_z[2], 0,
                    0, 0, 0, 1,
                );
                boxMesh.quaternion.setFromRotationMatrix(m4);
                boxMesh.castShadow = true;
                boxMesh.receiveShadow = true;
                boxMesh.userData = {
                    isHousing: true,
                    hIdx: hidx,
                    origCenter: origCenter.clone(),
                    origPC: origPC.clone(),
                    axisX: new THREE.Vector3(h.axis_x[0], h.axis_x[1], h.axis_x[2]),
                    axisY: new THREE.Vector3(h.axis_y[0], h.axis_y[1], h.axis_y[2]),
                    axisZ: new THREE.Vector3(h.axis_z[0], h.axis_z[1], h.axis_z[2]),
                };
                scene.add(boxMesh);

                const pcGeo = new THREE.PlaneGeometry(h.pc_half[0]*2, h.pc_half[0]*2);
                const pcMesh = new THREE.Mesh(pcGeo, pcMat);
                pcMesh.position.copy(origPC);
                pcMesh.quaternion.copy(boxMesh.quaternion);
                pcMesh.castShadow = true;
                pcMesh.receiveShadow = true;
                scene.add(pcMesh);

                housingMeshes.push({box: boxMesh, pc: pcMesh});
                housingCorrOrigins.push({center: origCenter.clone(), pc: origPC.clone()});
                hidx++;
            }

            // LAPPD global dx/dy/dz corrections (applied to ALL housings)
            const dxInput = document.getElementById('lappdDx');
            const dyInput = document.getElementById('lappdDy');
            const dzInput = document.getElementById('lappdDz');
            const saveBtn = document.getElementById('saveLappdCorr');

            const corrResp = await fetch('/api/lappd_correction');
            const corrData = await corrResp.json();
            let lappdCorr = {dx: 0, dy: 0, dz: 0};
            if (corrData.corrections && corrData.corrections.length > 0) {
                const c = corrData.corrections[0];
                lappdCorr = {dx: parseFloat(c.dx), dy: parseFloat(c.dy), dz: parseFloat(c.dz)};
            }
            dxInput.value = lappdCorr.dx;
            dyInput.value = lappdCorr.dy;
            dzInput.value = lappdCorr.dz;

            function applyLappdCorrection() {
                const dx = parseFloat(dxInput.value) || 0;
                const dy = parseFloat(dyInput.value) || 0;
                const dz = parseFloat(dzInput.value) || 0;
                for (let i = 0; i < housingMeshes.length; i++) {
                    const hm = housingMeshes[i];
                    const orig = housingCorrOrigins[i];
                    hm.box.position.set(orig.center.x + dx, orig.center.y + dy, orig.center.z + dz);
                    hm.pc.position.set(orig.pc.x + dx, orig.pc.y + dy, orig.pc.z + dz);
                }
            }

            applyLappdCorrection();

            saveBtn.addEventListener('click', async () => {
                const body = JSON.stringify({
                    idx: 0,
                    dx: parseFloat(dxInput.value) || 0,
                    dy: parseFloat(dyInput.value) || 0,
                    dz: parseFloat(dzInput.value) || 0,
                });
                await fetch('/api/lappd_correction', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body,
                });
                applyLappdCorrection();
            });

            // Grey LAPPD checkbox
            const greyCheck = document.getElementById('lappdGrey');
            greyCheck.addEventListener('change', () => {
                const isGrey = greyCheck.checked;
                for (const hm of housingMeshes) {
                    if (isGrey) {
                        hm.box.material.color.setHex(0x999999);
                        hm.box.material.transparent = false;
                        hm.box.material.opacity = 1.0;
                        hm.box.material.roughness = 0.5;
                        hm.box.material.metalness = 0.05;
                        hm.pc.material.color.setHex(0x999999);
                        hm.pc.material.roughness = 0.5;
                        hm.pc.material.metalness = 0.05;
                    } else {
                        hm.box.material.color.setHex(0x446688);
                        hm.box.material.transparent = true;
                        hm.box.material.opacity = 0.25;
                        hm.box.material.roughness = 0.6;
                        hm.box.material.metalness = 0.0;
                        hm.pc.material.color.setHex(0x88bbdd);
                        hm.pc.material.roughness = 0.3;
                        hm.pc.material.metalness = 0.1;
                    }
                }
            });

            document.getElementById('focusLAPPD').disabled = false;
        }

        // ---- Surfboard obscurant panels ----
        const surfResp = await fetch('/api/surfboards');
        const surfData = await surfResp.json();
        const surfMeshes = [];
        if (Array.isArray(surfData.surfboards)) {
            const surfMat = new THREE.MeshStandardMaterial({
                color: 0x333344,
                roughness: 0.8,
                metalness: 0.0,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.35,
            });
            let sbIdx = 0;
            for (const sb of surfData.surfboards) {
                const geo = new THREE.BoxGeometry(sb.half[0]*2, sb.half[1]*2, sb.half[2]*2);
                const mesh = new THREE.Mesh(geo, surfMat);
                mesh.position.set(sb.center[0], sb.center[1], sb.center[2]);
                const m4 = new THREE.Matrix4();
                m4.set(
                    sb.axis_x[0], sb.axis_y[0], sb.axis_z[0], 0,
                    sb.axis_x[1], sb.axis_y[1], sb.axis_z[1], 0,
                    sb.axis_x[2], sb.axis_y[2], sb.axis_z[2], 0,
                    0, 0, 0, 1,
                );
                mesh.quaternion.setFromRotationMatrix(m4);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = {
                    isSurfboard: true,
                    sbIdx: sbIdx,
                    origCenter: new THREE.Vector3(sb.center[0], sb.center[1], sb.center[2]),
                    axisX: new THREE.Vector3(sb.axis_x[0], sb.axis_x[1], sb.axis_x[2]),
                    axisY: new THREE.Vector3(sb.axis_y[0], sb.axis_y[1], sb.axis_y[2]),
                    axisZ: new THREE.Vector3(sb.axis_z[0], sb.axis_z[1], sb.axis_z[2]),
                };
                scene.add(mesh);
                surfMeshes.push(mesh);
                sbIdx++;
            }
        }

        // ---- Grey toggles for structure and PMTs ----
        const structGreyCheck = document.getElementById('structGrey');
        const pmtGreyCheck = document.getElementById('pmtGrey');

        structGreyCheck.addEventListener('change', () => {
            meshMat.color.setHex(structGreyCheck.checked ? 0x999999 : 0xffffff);
        });

        const structOpacitySlider = document.getElementById('structOpacity');
        structOpacitySlider.addEventListener('input', () => {
            meshMat.opacity = parseFloat(structOpacitySlider.value);
        });

        pmtGreyCheck.addEventListener('change', () => {
            const isGrey = pmtGreyCheck.checked;
            pmtGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    const idx = child.userData.pmtIdx;
                    if (idx === undefined) return;
                    if (isGrey) {
                        child.material.color.setHex(0x999999);
                        if (!child.userData.isPC && !child.userData.isPVC) {
                            child.material.transmission = 0.0;
                        }
                    } else {
                        if (child.userData.isPC) {
                            child.material.color.setRGB(0.55, 0.20, 0.10);
                        } else if (child.userData.isPVC) {
                            child.material.color.setRGB(0.10, 0.10, 0.12);
                        } else {
                            child.material.color.setRGB(0.97, 0.97, 0.99);
                            child.material.transmission = 0.92;
                        }
                    }
                }
            });
        });

        const showHWCheck = document.getElementById('showHW');
        showHWCheck.addEventListener('change', () => {
            pmtHWGroup.visible = showHWCheck.checked;
        });

        const showRefCheck = document.getElementById('showRefSpheres');
        showRefCheck.addEventListener('change', () => {
            refSpheres.visible = showRefCheck.checked;
        });

        const refOpacitySlider = document.getElementById('refOpacity');
        refOpacitySlider.addEventListener('input', () => {
            const op = parseFloat(refOpacitySlider.value);
            refSpheres.children.forEach(child => {
                if (child.isMesh && child.material) {
                    child.material.opacity = op;
                }
            });
        });

        const showTubeIDsCheck = document.getElementById('showTubeIDs');
        showTubeIDsCheck.addEventListener('change', () => {
            labelGroup.visible = showTubeIDsCheck.checked;
        });

        const showScanCheck = document.getElementById('showScan');
        const scanMeshSelect = document.getElementById('scanMeshSelect');
        let scanLoadedName = '';

        async function loadScanOverlay(name) {
            statusEl.textContent = `Loading ${name}…`;
            const geo = await loadScanMesh(name);
            if (!geo) {
                statusEl.textContent = `Scan mesh '${name}' not available.`;
                return null;
            }
            const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute('position'));
            console.log(`Scan mesh '${name}' bounds:`, box.min.toArray(), box.max.toArray());
            const mat = new THREE.MeshPhysicalMaterial({
                color: 0x44aadd,
                roughness: 0.5,
                metalness: 0.0,
                transparent: true,
                opacity: 0.65,
                side: THREE.DoubleSide,
                depthWrite: true,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.castShadow = false;
            mesh.receiveShadow = false;
            scanOverlay.add(mesh);
            scanLoadedName = name;
            statusEl.textContent = `Scan overlay '${name}' loaded.`;
            return mesh;
        }

        showScanCheck.addEventListener('change', async () => {
            if (showScanCheck.checked) {
                if (scanOverlay.children.length === 0) {
                    await loadScanOverlay(scanMeshSelect.value);
                }
                scanOverlay.visible = true;
            } else {
                scanOverlay.visible = false;
            }
        });

        scanMeshSelect.addEventListener('change', async () => {
            const name = scanMeshSelect.value;
            // Clear current mesh
            while (scanOverlay.children.length > 0) {
                const child = scanOverlay.children[0];
                child.geometry.dispose();
                child.material.dispose();
                scanOverlay.remove(child);
            }
            scanLoadedName = '';
            if (showScanCheck.checked) {
                await loadScanOverlay(name);
                scanOverlay.visible = true;
            }
        });

        // ---- PMT Scan Tips ----
        let pmtTips = null;
        const showScanTipsCheck = document.getElementById('showScanTips');
        showScanTipsCheck.addEventListener('change', () => {
            if (pmtTips) pmtTips.visible = showScanTipsCheck.checked;
        });

        async function loadPmtTips() {
            const resp = await fetch('/api/pmt_tips');
            const data = await resp.json();
            if (!data.tips || data.tips.length === 0) return;

            const tipGroup = new THREE.Group();
            const highMat = new THREE.MeshBasicMaterial({ color: 0x44dd44 });
            const medMat = new THREE.MeshBasicMaterial({ color: 0xdddd44 });
            const lowMat = new THREE.MeshBasicMaterial({ color: 0xdd4444 });

            for (const t of data.tips) {
                if (!t.found) continue;
                const r = t.reliability;
                const mat = r >= 0.7 ? highMat : r >= 0.3 ? medMat : lowMat;
                const geo = new THREE.SphereGeometry(8, 8, 6);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(t.tip_x, t.tip_y, t.tip_z);
                tipGroup.add(mesh);

                // CSV seed position (smaller yellow dot)
                const csvGeo = new THREE.SphereGeometry(4, 6, 4);
                const csvMat = new THREE.MeshBasicMaterial({ color: 0xdddd00 });
                const csvMesh = new THREE.Mesh(csvGeo, csvMat);
                csvMesh.position.set(t.csv_x, t.csv_y, t.csv_z);
                tipGroup.add(csvMesh);
            }

            tipGroup.visible = showScanTipsCheck.checked;
            scene.add(tipGroup);
            pmtTips = tipGroup;
            statusEl.textContent += ` ${data.tips.length} tips loaded.`;
        }
        loadPmtTips();

        const pmtOpacitySlider = document.getElementById('pmtOpacity');
        pmtOpacitySlider.addEventListener('input', () => {
            const op = parseFloat(pmtOpacitySlider.value);
            pmtGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    if (child.userData.isPC) {
                        child.material.opacity = 0.35;
                    } else if (child.userData.isPVC) {
                        child.material.opacity = 1.0;
                    } else {
                        child.material.opacity = op;
                    }
                }
            });
            pmtHWGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    child.material.opacity = op;
                }
            });
        });

        // ---- View controls (azimuth, elevation, distance) ----
        const viewAzSlider = document.getElementById('viewAz');
        const viewElSlider = document.getElementById('viewEl');
        const viewDistSlider = document.getElementById('viewDist');
        const viewAzVal = document.getElementById('viewAzVal');
        const viewElVal = document.getElementById('viewElVal');
        const viewDistVal = document.getElementById('viewDistVal');

        function sphericalToCartesian(azDeg, polDeg, dist) {
            const az = azDeg * Math.PI / 180;
            const pol = polDeg * Math.PI / 180;
            const x = dist * Math.sin(pol) * Math.cos(az);
            const y = dist * Math.sin(pol) * Math.sin(az);
            const z = dist * Math.cos(pol);
            return new THREE.Vector3(x, y, z);
        }

        function cartesianToSpherical(pos) {
            const dist = pos.length();
            const pol = Math.acos(Math.min(1, Math.max(-1, pos.z / dist)));
            const az = Math.atan2(pos.y, pos.x);
            return { az: (az * 180 / Math.PI + 360) % 360, pol: pol * 180 / Math.PI, dist };
        }

        function updateView() {
            const az = parseFloat(viewAzSlider.value);
            const pol = parseFloat(viewElSlider.value);
            const dist = parseFloat(viewDistSlider.value);
            const t = controls.target;
            const rel = sphericalToCartesian(az, pol, dist);
            camera.position.set(t.x + rel.x, t.y + rel.y, t.z + rel.z);
            controls.update();
        }

        viewAzSlider.addEventListener('input', updateView);
        viewElSlider.addEventListener('input', updateView);
        viewDistSlider.addEventListener('input', updateView);

        // Sync sliders when user drags with mouse
        controls.addEventListener('change', () => {
            const rel = new THREE.Vector3().copy(camera.position).sub(controls.target);
            const s = cartesianToSpherical(rel);
            viewAzSlider.value = s.az;
            viewElSlider.value = s.pol;
            viewDistSlider.value = Math.round(s.dist / 50) * 50;
            viewAzVal.textContent = Math.round(s.az);
            viewElVal.textContent = Math.round(s.pol);
            viewDistVal.textContent = Math.round(s.dist);
        });

        // Initial scene (muon at default position)
        updateScene();

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // ---- PMT click selection ----
        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const adjPanel = document.getElementById('pmt-adjust');
        let meshStartPos = null;
        let selectedStartSliders = null;
        let adjBasis = null;

        function sliderToCartesian(vals, basis) {
            if (basis.mode === 'global') {
                return new THREE.Vector3(vals[0], vals[1], vals[2]);
            }
            const a = vals[0], t = vals[1], v = vals[2];
            return new THREE.Vector3(
                a * basis.eAxial.x + t * basis.eTang.x + v * basis.eVert.x,
                a * basis.eAxial.y + t * basis.eTang.y + v * basis.eVert.y,
                a * basis.eAxial.z + t * basis.eTang.z + v * basis.eVert.z
            );
        }

        function updatePMTPreview() {
            if (selectedIdx < 0 || !adjBasis || !meshStartPos || !selectedStartSliders) return;
            const vals = [
                parseFloat(document.getElementById('adj-0').value),
                parseFloat(document.getElementById('adj-1').value),
                parseFloat(document.getElementById('adj-2').value),
            ];
            const currentCorr = sliderToCartesian(vals, adjBasis);
            const startCorr = sliderToCartesian(selectedStartSliders, adjBasis);
            const delta = currentCorr.clone().sub(startCorr);
            const newPos = meshStartPos.clone().add(delta);
            pmtGroup.children.forEach(ch => {
                if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                    ch.position.copy(newPos);
                }
            });
            pmtHWGroup.children.forEach(ch => {
                if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                    ch.position.copy(newPos);
                }
            });
        }

        function cancelAdjustment() {
            if (selectedMesh && meshStartPos) {
                pmtGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(meshStartPos);
                    }
                });
                pmtHWGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(meshStartPos);
                    }
                });
            }
            deselectPMT();
        }

        function deselectPMT() {
            if (selectedMesh) {
                selectedMesh.material.emissive.setHex(0x000000);
                selectedMesh.material.emissiveIntensity = 0;
            }
            selectedIdx = -1;
            selectedMesh = null;
            adjPanel.style.display = 'none';
            meshStartPos = null;
            selectedStartSliders = null;
            adjBasis = null;
        }

        function selectPMT(idx) {
            if (selectedMesh) {
                selectedMesh.material.emissive.setHex(0x000000);
                selectedMesh.material.emissiveIntensity = 0;
            }
            selectedIdx = idx;
            selectedMesh = null;
            const tubeId = pmtData.detector_nums[idx];
            const type = pmtData.types[idx];
            const dir = pmtData.directions[idx];
            const corr = pmtData.corrections[tubeId] || [0, 0, 0];

            document.getElementById('adj-tube-id').textContent = tubeId;
            document.getElementById('adj-type').textContent = type;
            adjPanel.style.display = 'block';

            // Compute basis and initial slider values
            let v0, v1, v2, labels;
            const eVert = new THREE.Vector3(0, 0, 1);
            const eAxial = new THREE.Vector3(dir[0], dir[1], dir[2]);
            const eTang = new THREE.Vector3().crossVectors(eVert, eAxial);
            const tLen = eTang.length();
            let mode;

            if (tLen < 1e-6) {  // bottom/top — global XYZ
                mode = 'global';
                labels = ['dX', 'dY', 'dZ'];
                v0 = corr[0]; v1 = corr[1]; v2 = corr[2];
            } else {
                mode = 'local';
                eTang.divideScalar(tLen);
                labels = ['Axial', 'Tangential', 'Vertical'];
                v0 = corr[0]*eAxial.x + corr[1]*eAxial.y + corr[2]*eAxial.z;
                v1 = corr[0]*eTang.x  + corr[1]*eTang.y  + corr[2]*eTang.z;
                v2 = corr[2];  // dot with (0,0,1)
            }

            for (let k = 0; k < 3; k++) {
                document.getElementById('adj-label-' + k).textContent = labels[k];
                const sl = document.getElementById('adj-' + k);
                const vals = [v0, v1, v2];
                sl.value = vals[k];
                document.getElementById('adj-val-' + k).textContent = parseFloat(vals[k]).toFixed(1);
            }

            // Highlight the clicked mesh — find a mesh belonging to this PMT
            const allMeshes = [];
            pmtGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx === idx) allMeshes.push(ch); });
            pmtHWGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx === idx) allMeshes.push(ch); });
            if (allMeshes.length > 0) {
                selectedMesh = allMeshes[0];
                selectedMesh.material.emissive.setHex(0x4444ff);
                selectedMesh.material.emissiveIntensity = 0.5;
            }

            // Record start state for preview
            meshStartPos = selectedMesh ? selectedMesh.position.clone() : new THREE.Vector3(0,0,0);
            selectedStartSliders = [v0, v1, v2];
            adjBasis = { mode, eAxial: eAxial.clone(), eTang: eTang.clone(), eVert: eVert.clone() };
        }

        renderer.domElement.addEventListener('click', (event) => {
            const rect = renderer.domElement.getBoundingClientRect();
            pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(pointer, camera);

            const targets = [];
            pmtGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx !== undefined) targets.push(ch); });
            pmtHWGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx !== undefined && ch.visible) targets.push(ch); });

            const pmtHits = raycaster.intersectObjects(targets);
            if (pmtHits.length > 0) {
                deselectPMT();
                deselectSurfboard();
                selectPMT(pmtHits[0].object.userData.pmtIdx);
                return;
            }

            // Check housing clicks
            const housingTargets = housingMeshes.map(hm => hm.box);
            const housingHits = raycaster.intersectObjects(housingTargets);
            if (housingHits.length > 0) {
                deselectPMT();
                deselectSurfboard();
                selectHousing(housingHits[0].object.userData.hIdx);
                return;
            }

            const surfHits = raycaster.intersectObjects(surfMeshes);
            if (surfHits.length > 0) {
                deselectPMT();
                deselectHousing();
                selectSurfboard(surfHits[0].object.userData.sbIdx);
            } else if (selectedSurfboard >= 0) {
                deselectSurfboard();
            } else if (selectedHousing >= 0) {
                deselectHousing();
            } else if (selectedIdx >= 0) {
                deselectPMT();
            }
        });

        // Slider live update
        for (let k = 0; k < 3; k++) {
            document.getElementById('adj-' + k).addEventListener('input', (e) => {
                const idx = parseInt(e.target.id.split('-')[1]);
                document.getElementById('adj-val-' + idx).textContent = parseFloat(e.target.value).toFixed(1);
                updatePMTPreview();
            });
        }

        document.getElementById('adj-save').addEventListener('click', async () => {
            if (selectedIdx < 0) return;
            const tubeId = pmtData.detector_nums[selectedIdx];
            const axial = parseFloat(document.getElementById('adj-0').value);
            const tangential = parseFloat(document.getElementById('adj-1').value);
            const vertical = parseFloat(document.getElementById('adj-2').value);

            // Capture current live-preview position before the fetch
            const currentPos = new THREE.Vector3();
            pmtGroup.children.some(ch => {
                if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                    currentPos.copy(ch.position);
                    return true;
                }
                return false;
            });

            const resp = await fetch('/api/correction/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tube_id: tubeId, axial, tangential, vertical}),
            });
            const result = await resp.json();
            if (result.success) {
                // Use the live-preview position directly (more reliable than server round-trip)
                pmtGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(currentPos);
                    }
                });
                pmtHWGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(currentPos);
                    }
                });
                // Update start state so Cancel doesn't revert
                meshStartPos = currentPos.clone();
                selectedStartSliders = [
                    parseFloat(document.getElementById('adj-0').value),
                    parseFloat(document.getElementById('adj-1').value),
                    parseFloat(document.getElementById('adj-2').value),
                ];
            }
            deselectPMT();
        });

        document.getElementById('adj-reset').addEventListener('click', () => {
            for (let k = 0; k < 3; k++) {
                document.getElementById('adj-' + k).value = '0';
                document.getElementById('adj-val-' + k).textContent = '0.0';
            }
            updatePMTPreview();
        });

        document.getElementById('adj-cancel').addEventListener('click', cancelAdjustment);

        // ---- Surfboard interactive position popup ----
        const surfPopup = document.getElementById('surfboard-popup');

        function updateSurfboardPosition() {
            const idx = selectedSurfboard;
            if (idx < 0 || idx >= surfMeshes.length) return;
            const mesh = surfMeshes[idx];
            const ud = mesh.userData;
            const dVert = parseFloat(document.getElementById('surfVert').value) || 0;
            const dRad = parseFloat(document.getElementById('surfRad').value) || 0;
            const dTang = parseFloat(document.getElementById('surfTang').value) || 0;

            const dcx = dVert * ud.axisY.x + dRad * ud.axisZ.x + dTang * ud.axisX.x;
            const dcy = dVert * ud.axisY.y + dRad * ud.axisZ.y + dTang * ud.axisX.y;
            const dcz = dVert * ud.axisY.z + dRad * ud.axisZ.z + dTang * ud.axisX.z;

            const newCx = ud.origCenter.x + dcx;
            const newCy = ud.origCenter.y + dcy;
            const newCz = ud.origCenter.z + dcz;
            mesh.position.set(newCx, newCy, newCz);

            fetch('/api/surfboard/adjust', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: idx, cx: newCx, cy: newCy, cz: newCz}),
            }).catch(() => {});
        }

        function selectSurfboard(idx) {
            selectedSurfboard = idx;
            document.getElementById('surfPopupIdx').textContent = idx + 1;
            for (const axis of ['Vert', 'Rad', 'Tang']) {
                document.getElementById('surf' + axis).value = 0;
                document.getElementById('surf' + axis + 'Num').value = 0;
            }
            surfPopup.style.display = 'block';
        }

        function deselectSurfboard() {
            if (selectedSurfboard < 0) return;
            selectedSurfboard = -1;
            surfPopup.style.display = 'none';
        }

        // Bidirectional sync: slider → number, number → slider
        const surfAxes = [
            { slider: 'surfVert', num: 'surfVertNum' },
            { slider: 'surfRad',  num: 'surfRadNum' },
            { slider: 'surfTang', num: 'surfTangNum' },
        ];
        for (const { slider, num } of surfAxes) {
            document.getElementById(slider).addEventListener('input', () => {
                document.getElementById(num).value = document.getElementById(slider).value;
                updateSurfboardPosition();
            });
            document.getElementById(num).addEventListener('input', () => {
                document.getElementById(slider).value = document.getElementById(num).value;
                updateSurfboardPosition();
            });
        }

        document.getElementById('surfReset').addEventListener('click', () => {
            const idx = selectedSurfboard;
            if (idx < 0 || idx >= surfMeshes.length) return;
            const mesh = surfMeshes[idx];
            const orig = mesh.userData.origCenter;
            mesh.position.copy(orig);
            for (const axis of ['Vert', 'Rad', 'Tang']) {
                document.getElementById('surf' + axis).value = 0;
                document.getElementById('surf' + axis + 'Num').value = 0;
            }
            fetch('/api/surfboard/adjust', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: idx, cx: orig.x, cy: orig.y, cz: orig.z}),
            }).catch(() => {});
        });

        document.getElementById('surfPopupClose').addEventListener('click', deselectSurfboard);

        // ---- LAPPD housing interactive position popup ----
        const housingPopup = document.getElementById('housing-popup');

        function updateHousingPosition() {
            const idx = selectedHousing;
            if (idx < 0 || idx >= housingMeshes.length) return;
            const hm = housingMeshes[idx];
            const ud = hm.box.userData;
            const dVert = parseFloat(document.getElementById('hVert').value) || 0;
            const dRad = parseFloat(document.getElementById('hRad').value) || 0;
            const dTang = parseFloat(document.getElementById('hTang').value) || 0;

            const dcx = dVert * ud.axisY.x + dRad * ud.axisZ.x + dTang * ud.axisX.x;
            const dcy = dVert * ud.axisY.y + dRad * ud.axisZ.y + dTang * ud.axisX.y;
            const dcz = dVert * ud.axisY.z + dRad * ud.axisZ.z + dTang * ud.axisX.z;

            const newCx = ud.origCenter.x + dcx;
            const newCy = ud.origCenter.y + dcy;
            const newCz = ud.origCenter.z + dcz;
            hm.box.position.set(newCx, newCy, newCz);

            // PC moves by the same delta
            hm.pc.position.set(
                ud.origPC.x + dcx,
                ud.origPC.y + dcy,
                ud.origPC.z + dcz,
            );

            fetch('/api/lappd/adjust', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: idx, cx: newCx, cy: newCy, cz: newCz}),
            }).catch(() => {});
        }

        function selectHousing(idx) {
            selectedHousing = idx;
            document.getElementById('hPopupIdx').textContent = idx + 1;
            for (const axis of ['Vert', 'Rad', 'Tang']) {
                document.getElementById('h' + axis).value = 0;
                document.getElementById('h' + axis + 'Num').value = 0;
            }
            housingPopup.style.display = 'block';
        }

        function deselectHousing() {
            if (selectedHousing < 0) return;
            selectedHousing = -1;
            housingPopup.style.display = 'none';
        }

        const housingAxes = [
            { slider: 'hVert', num: 'hVertNum' },
            { slider: 'hRad',  num: 'hRadNum' },
            { slider: 'hTang', num: 'hTangNum' },
        ];
        for (const { slider, num } of housingAxes) {
            document.getElementById(slider).addEventListener('input', () => {
                document.getElementById(num).value = document.getElementById(slider).value;
                updateHousingPosition();
            });
            document.getElementById(num).addEventListener('input', () => {
                document.getElementById(slider).value = document.getElementById(num).value;
                updateHousingPosition();
            });
        }

        document.getElementById('hReset').addEventListener('click', () => {
            const idx = selectedHousing;
            if (idx < 0 || idx >= housingMeshes.length) return;
            const hm = housingMeshes[idx];
            const ud = hm.box.userData;
            const orig = ud.origCenter;
            const origPC = ud.origPC;
            hm.box.position.copy(orig);
            hm.pc.position.copy(origPC);
            for (const axis of ['Vert', 'Rad', 'Tang']) {
                document.getElementById('h' + axis).value = 0;
                document.getElementById('h' + axis + 'Num').value = 0;
            }
            fetch('/api/lappd/adjust', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: idx, cx: orig.x, cy: orig.y, cz: orig.z}),
            }).catch(() => {});
        });

        document.getElementById('hPopupClose').addEventListener('click', deselectHousing);

    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

function focusLAPPD() {
    if (!housingMeshes || housingMeshes.length === 0) return;
    const hm = housingMeshes[0];
    const target = hm.box.position.clone();
    const ud = hm.box.userData;
    const normal = ud.axisZ.clone();  // inward-facing normal
    const eye = target.clone().add(normal.clone().multiplyScalar(800));
    camera.position.copy(eye);
    controls.target.copy(target);
    controls.update();
}

document.getElementById('focusLAPPD').addEventListener('click', focusLAPPD);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

init();
</script>

<!-- Surfboard position popup -->
<div id="surfboard-popup">
  <div class="sb-title">
    <h3>Surfboard <span id="surfPopupIdx">0</span></h3>
    <button id="surfPopupClose">✕</button>
  </div>
  <div class="sb-row">
    <label>Vertical</label>
    <input type="number" id="surfVertNum" step="1" value="0">
    <input type="range" id="surfVert" min="-1500" max="1500" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Radial</label>
    <input type="number" id="surfRadNum" step="1" value="0">
    <input type="range" id="surfRad" min="-800" max="800" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Tangential</label>
    <input type="number" id="surfTangNum" step="1" value="0">
    <input type="range" id="surfTang" min="-800" max="800" step="1" value="0">
  </div>
  <div>
    <button id="surfReset">Reset</button>
  </div>
</div>

<!-- LAPPD Housing position popup -->
<div id="housing-popup">
  <div class="sb-title">
    <h3>LAPPD Housing <span id="hPopupIdx">0</span></h3>
    <button id="hPopupClose">✕</button>
  </div>
  <div class="sb-row">
    <label>Vertical</label>
    <input type="number" id="hVertNum" step="1" value="0">
    <input type="range" id="hVert" min="-1500" max="1500" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Radial</label>
    <input type="number" id="hRadNum" step="1" value="0">
    <input type="range" id="hRad" min="-800" max="800" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Tangential</label>
    <input type="number" id="hTangNum" step="1" value="0">
    <input type="range" id="hTang" min="-800" max="800" step="1" value="0">
  </div>
  <div>
    <button id="hReset">Reset</button>
  </div>
</div>

</body>
</html>
"""


def run_server(args):
    global geometry, pmt_instance_data

    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f32)

    pmt_csv = args.pmt_csv
    if pmt_csv is None:
        pmt_csv = Path("PMTPositions_Scan.txt")
    else:
        pmt_csv = Path(pmt_csv)

    gdml_path = Path(args.gdml)
    step_path = Path(args.step) if args.step else None
    manifest_path = Path(args.manifest) if args.manifest else None

    print(f"Loading geometry from {gdml_path}...")
    geometry = build_geometry(
        gdml_path,
        step_path=step_path,
        manifest_path=manifest_path,
        pmt_csv_path=pmt_csv,
        no_lappd=args.no_lappd,
        z_offset=args.z_offset,
        lappd_model=args.lappd_model,
        bottom_rotation_deg=args.bottom_rot,
        bottom_spin_deg=args.bottom_spin,
        det_rotation_deg=args.det_rotation,
        n_surfboards=args.surfboard if hasattr(args, 'surfboard') else 0,
    )
    print(f"  Mesh: {geometry.mesh_vertices.shape[0]} verts, {geometry.mesh_triangles.shape[0]} tris")
    print(f"  PMTs: {geometry.pmt_centers.shape[0]}")

    from annieray.pmt_loader import load_pmts
    pmt_info = load_pmts(pmt_csv, z_offset=args.z_offset,
                         bottom_rotation_deg=args.bottom_rot,
                         bottom_spin_deg=args.bottom_spin,
                         det_rotation_deg=args.det_rotation)
    VizHandler._pmt_types = pmt_info["types"]
    VizHandler._pmt_data = pmt_info
    pmt_instance_data = pmt_info

    # ---- Apply corrections to simulated PMT positions ----
    _corr_path = Path(__file__).resolve().parent.parent / "corrections.csv"
    VizHandler._corr_path = _corr_path
    VizHandler._lappd_corr_path = str(Path(__file__).resolve().parent / "lappd_corrections.csv")
    _corrections: dict[int, tuple[float, float, float]] = {}
    if _corr_path.exists():
        with open(_corr_path, newline="") as _f:
            for _row in csv.DictReader(_f):
                _tid = int(_row["tube_id"])
                _dx = float(_row.get("dx", 0))
                _dy = float(_row.get("dy", 0))
                _dz = float(_row.get("dz", 0))
                _corrections[_tid] = (_dx, _dy, _dz)
        print(f"  Corrections: {len(_corrections)} loaded")
        _dets = pmt_instance_data["detector_nums"]
        _dirs = pmt_instance_data["directions"]
        _positions = pmt_instance_data["instance_positions"]
        _applied = 0
        _cr = args.det_rotation
        for _i, _tid in enumerate(_dets):
            if _tid in _corrections:
                _dx, _dy, _dz = _corrections[_tid]
                if _cr != 0.0:
                    _th = math.radians(_cr)
                    _c = math.cos(_th)
                    _s = math.sin(_th)
                    _dx, _dy = _dx * _c - _dy * _s, _dx * _s + _dy * _c
                _positions[_i, 0] += _dx
                _positions[_i, 1] += _dy
                _positions[_i, 2] += _dz
                # Apply same delta to kernel geometry (mesh refinement + sphere fallback)
                geometry.pmt_instance_pos[_i, 0] += _dx
                geometry.pmt_instance_pos[_i, 1] += _dy
                geometry.pmt_instance_pos[_i, 2] += _dz
                geometry.pmt_centers[_i, 0] += _dx
                geometry.pmt_centers[_i, 1] += _dy
                geometry.pmt_centers[_i, 2] += _dz
                _applied += 1
        print(f"  Corrections applied to {_applied} PMT instance positions")
    else:
        print("  No corrections.csv found — simulated detector uncorrected")
    VizHandler._corrections = _corrections

    # Load scan mesh overlays
    print("Loading scan overlays...")
    _det_rot = args.det_rotation
    for _sn in [
        "AllPMTs", "SuperStructure", "BottomLayer", "TopLayer",
        "Panel-1", "Panel-2", "Panel-3", "Panel-4", "Panel-5", "Panel-6", "Panel-7", "Panel-8",
        "Panel-1-PMTs", "Panel-2-PMTs", "Panel-3-PMTs", "Panel-4-PMTs",
        "Panel-5-PMTs", "Panel-6-PMTs", "Panel-7-PMTs", "Panel-8-PMTs",
        "TopPMTs", "BottomPMTs", "TankLid",
    ]:
        if _load_scan_mesh(_sn, det_rotation_deg=_det_rot):
            _n = len(np.load(SCAN_MESH_DIR / f"{_sn}_verts.npy"))
            print(f"  Scan mesh '{_sn}': {_n} verts")

    # Load PMT tip positions
    global PMT_TIPS_CACHE
    _tips_path = SCAN_MESH_DIR / "pmt_tip_positions.csv"
    if _tips_path.exists():
        tips = []
        _tip_rot = args.det_rotation
        with open(_tips_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tip_x = float(row["tip_x"])
                tip_y = float(row["tip_y"])
                tip_z = float(row["tip_z"])
                csv_x = float(row["csv_x"])
                csv_y = float(row["csv_y"])
                csv_z = float(row["csv_z"])
                if _tip_rot != 0.0:
                    theta = math.radians(_tip_rot)
                    c = math.cos(theta)
                    s = math.sin(theta)
                    tip_x, tip_y = tip_x * c - tip_y * s, tip_x * s + tip_y * c
                    csv_x, csv_y = csv_x * c - csv_y * s, csv_x * s + csv_y * c
                tips.append({
                    "tube_id": int(row["tube_id"]),
                    "panel": int(row["panel"]),
                    "type": row["type"],
                    "tip_x": tip_x,
                    "tip_y": tip_y,
                    "tip_z": tip_z,
                    "csv_x": csv_x,
                    "csv_y": csv_y,
                    "csv_z": csv_z,
                    "offset_mm": float(row["offset_mm"]),
                    "n_verts": int(row["n_verts"]),
                    "reliability": float(row["reliability"]),
                    "found": row["found"].strip() in ("1", "true", "True"),
                })
        PMT_TIPS_CACHE = tips
        print(f"  PMT tips: {len(tips)} loaded")
    else:
        print(f"  PMT tips: NOT FOUND ({_tips_path})")

    host = args.host
    port = args.port
    server = HTTPServer((host, port), VizHandler)
    print(f"\nViz server at http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
