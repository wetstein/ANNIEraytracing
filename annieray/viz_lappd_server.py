"""Standalone LAPPD module visualizer with Cherenkov ray tracing.

Serves a Three.js page showing the ANNIE LAPPD housing, photocathode,
and interactive Cherenkov ray tracing with configurable muon position.

Usage:
    python -m annieray viz-lappd
"""

from __future__ import annotations

import json
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

from annieray.lappd_model import build_housing, housing_to_arrays
from annieray.lappd_response import process_hit_dicts, LAPPDResponseConfig

housing_json: dict = {}
_last_trace: dict = {}


# ---------------------------------------------------------------------------
# Analytic geometry intersection helpers  (Python, no Taichi)
# ---------------------------------------------------------------------------

# Suppress the noisy module-level print in cherenkov.py on import
import io, sys as _sys
_sys_modules_import = __import__
_old_stdout = _sys.stdout
_sys.stdout = io.StringIO()
from annieray.cherenkov import generate_cherenkov_photons
_sys.stdout = _old_stdout


def _ray_box_intersect(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    cx: float, cy: float, cz: float,
    ax_x: float, ax_y: float, ax_z: float,
    ay_x: float, ay_y: float, ay_z: float,
    az_x: float, az_y: float, az_z: float,
    hx: float, hy: float, hz: float,
) -> tuple[bool, float, float, bool]:
    """Slab-test ray vs oriented box.  Returns (hit, t_entry, t_exit, is_front_face).

    ``t_entry`` / ``t_exit`` define the segment the ray spends inside the box.
    ``is_front_face`` is True when the ray enters through the +Z face (the
    acrylic window facing into the tank).
    """
    t_min = -1e30
    t_max = 1e30
    front_val = -1e30

    axes = [
        (ax_x, ax_y, ax_z, hx),
        (ay_x, ay_y, ay_z, hy),
        (az_x, az_y, az_z, hz),
    ]
    for axis_idx, (aax, aay, aaz, h) in enumerate(axes):
        denom = dx * aax + dy * aay + dz * aaz
        oc = (ox - cx) * aax + (oy - cy) * aay + (oz - cz) * aaz
        t0 = (-h - oc) / denom if abs(denom) > 1e-30 else (-1e30 if (-h - oc) < 0 else 1e30)
        t1 = (h - oc) / denom if abs(denom) > 1e-30 else (1e30 if (h - oc) > 0 else -1e30)
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_min:
            t_min = t0
            if axis_idx == 2:
                front_val = -denom
        if t1 < t_max:
            t_max = t1
        if t_min > t_max:
            return False, 0.0, 0.0, False

    is_front = front_val > 0
    return True, t_min, t_max, is_front


def _ray_rectangle_intersect(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    pcx: float, pcy: float, pcz: float,
    nrmx: float, nrmy: float, nrmz: float,
    half: float,
    up_x: float, up_y: float, up_z: float,
) -> tuple[bool, float, float, float, float]:
    """Ray vs axis-aligned (in local frame) square.

    ``up`` defines the local vertical (Y) direction of the rectangle;
    the local X is ``cross(up, normal)``.
    Returns (hit, t, nx, ny, nz).
    """
    denom = dx * nrmx + dy * nrmy + dz * nrmz
    if abs(denom) < 1e-30:
        return False, 0.0, 0.0, 0.0, 0.0

    ocx = pcx - ox
    ocy = pcy - oy
    ocz = pcz - oz
    t = (ocx * nrmx + ocy * nrmy + ocz * nrmz) / denom
    if t < 1e-6:
        return False, 0.0, 0.0, 0.0, 0.0

    px = ox + dx * t
    py = oy + dy * t
    pz = oz + dz * t

    # Local coordinates
    # local_x = cross(up, normal)
    lxx = up_y * nrmz - up_z * nrmy
    lxy = up_z * nrmx - up_x * nrmz
    lxz = up_x * nrmy - up_y * nrmx
    ll = math.sqrt(lxx * lxx + lxy * lxy + lxz * lxz)
    if ll > 1e-12:
        lxx /= ll
        lxy /= ll
        lxz /= ll

    # local_y = up (already unit)
    lyx, lyy, lyz = up_x, up_y, up_z

    u = (px - pcx) * lxx + (py - pcy) * lxy + (pz - pcz) * lxz
    v = (px - pcx) * lyx + (py - pcy) * lyy + (pz - pcz) * lyz

    if abs(u) > half or abs(v) > half:
        return False, 0.0, 0.0, 0.0, 0.0

    nx = -nrmx if denom > 0 else nrmx
    ny = -nrmy if denom > 0 else nrmy
    nz = -nrmz if denom > 0 else nrmz
    return True, t, nx, ny, nz


def _trace_cherenkov_on_lappd(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    n_photons: int,
    housing: dict,
    rng: np.random.Generator,
) -> dict:
    """Generate Cherenkov photons using the proper generator and trace against the LAPPD housing."""
    origins, directions, create_times = generate_cherenkov_photons(muon_pos, muon_dir, n_photons, rng=rng)
    result = _trace_hits(origins, directions, housing, rng, create_times)
    result["muon_pos"] = list(muon_pos)
    result["muon_dir"] = list(muon_dir)
    return result


def _trace_spot_on_lappd(
    pos: tuple[float, float, float],
    dir: tuple[float, float, float],
    n_photons: int,
    spread_deg: float,
    housing: dict,
    rng: np.random.Generator,
) -> dict:
    """Generate a directed beam of N photons at the LAPPD and return hits."""
    dx0, dy0, dz0 = dir
    norm = math.sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0)
    if norm < 1e-12:
        return {"n_photons": 0, "n_hits": 0, "hits": [], "rays": []}
    dx0 /= norm; dy0 /= norm; dz0 /= norm
    muon_dir = np.array([dx0, dy0, dz0], dtype=np.float64)

    # Build orthogonal basis for scattering
    if abs(dx0) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(muon_dir, ref)
    u /= np.linalg.norm(u)
    v = np.cross(muon_dir, u)

    origins = np.zeros((n_photons, 3), dtype=np.float32)
    directions = np.zeros((n_photons, 3), dtype=np.float32)
    spread_rad = math.radians(spread_deg)

    for i in range(n_photons):
        theta = rng.rayleigh(scale=spread_rad)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        d = (math.cos(theta) * muon_dir
             + math.sin(theta) * math.cos(phi) * u
             + math.sin(theta) * math.sin(phi) * v)
        d /= np.linalg.norm(d)
        origins[i] = pos
        directions[i] = d.astype(np.float32)

    return _trace_hits(origins, directions, housing, rng)


def _trace_hits(
    origins: np.ndarray,
    directions: np.ndarray,
    housing: dict,
    rng: np.random.Generator,
    create_times: np.ndarray | None = None,
) -> dict:
    """Trace array of (origin, direction) pairs against the LAPPD housing."""
    n_total = origins.shape[0]
    _c_water = 299.792458 / 1.34  # mm/ns in water
    h = housing
    cx, cy, cz = h["center"]
    ax0, ax1, ax2 = h["axis_x"]
    ay0, ay1, ay2 = h["axis_y"]
    az0, az1, az2 = h["axis_z"]
    hhx, hhy, hhz = h["half"]
    pcx, pcy, pcz = h["pc_center"]
    pcnx, pcny, pcnz = h["pc_normal"]
    pchalf = h["pc_half"][0]
    up_x, up_y, up_z = h["axis_y"]

    rays: list[dict] = []
    hits: list[dict] = []
    ray_sample = max(1, n_total // 500)

    for i in range(n_total):
        px, py, pz = origins[i]
        dx, dy, dz = directions[i]

        best_hit: dict | None = None

        b_hit, b_entry, b_exit, b_front = _ray_box_intersect(
            px, py, pz, dx, dy, dz, cx, cy, cz,
            ax0, ax1, ax2, ay0, ay1, ay2, az0, az1, az2, hhx, hhy, hhz,
        )

        if b_hit:
            origin_inside = b_entry < 1e-6

            if origin_inside:
                # ── Ray originates inside the housing box ──
                pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(
                    px, py, pz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z,
                )
                if pc_hit and pc_t > 1e-6:
                    best_hit = {"type": "photocathode", "x": px + dx * pc_t, "y": py + dy * pc_t, "z": pz + dz * pc_t, "nx": nx, "ny": ny, "nz": nz, "t": float(pc_t)}

            elif b_front and b_entry > 1e-6:
                # ── Entered through the front face (acrylic window) ──
                pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(
                    px, py, pz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z,
                )
                if pc_hit and pc_t > 1e-6 and pc_t <= b_exit + 1e-6:
                    best_hit = {"type": "photocathode", "x": px + dx * pc_t, "y": py + dy * pc_t, "z": pz + dz * pc_t, "nx": nx, "ny": ny, "nz": nz, "t": float(pc_t)}
                else:
                    # Hit the window but missed the PC → module surface hit
                    best_hit = {"type": "housing", "x": px + dx * b_entry, "y": py + dy * b_entry, "z": pz + dz * b_entry, "nx": az0, "ny": az1, "nz": az2, "t": float(b_entry)}

            else:
                # ── Entered through side or back face (charcoal PVC) ──
                best_hit = {"type": "housing", "x": px + dx * b_entry, "y": py + dy * b_entry, "z": pz + dz * b_entry, "nx": az0, "ny": az1, "nz": az2, "t": float(b_entry)}

        else:
            # No box intersection at all — check PC directly
            pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(
                px, py, pz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z,
            )
            if pc_hit and pc_t > 1e-6:
                best_hit = {"type": "photocathode", "x": px + dx * pc_t, "y": py + dy * pc_t, "z": pz + dz * pc_t, "nx": nx, "ny": ny, "nz": nz, "t": float(pc_t)}

        if best_hit:
            best_hit["origin"] = [float(px), float(py), float(pz)]
            best_hit["dir"] = [float(dx), float(dy), float(dz)]
            if create_times is not None:
                best_hit["arrival_time"] = float(create_times[i]) + best_hit["t"] / _c_water
            hits.append(best_hit)
        elif i % ray_sample == 0:
            rays.append({"origin": [float(px), float(py), float(pz)], "dir": [float(dx), float(dy), float(dz)]})

    ray_cap = min(len(rays), 500)
    if len(rays) > ray_cap:
        idx = rng.choice(len(rays), ray_cap, replace=False)
        rays = [rays[i] for i in sorted(idx)]

    return _sanitise({
        "muon_pos": [float(o) for o in origins[0]],
        "muon_dir": [float(d) for d in directions[0]],
        "n_photons": n_total,
        "n_hits": len(hits),
        "hits": hits,
        "rays": rays,
    })


def _trace_single_ray(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    housing: dict,
) -> dict | None:
    """Trace one ray against the LAPPD.  Returns hit dict or None."""
    h = housing
    cx, cy, cz = h["center"]
    ax0, ax1, ax2 = h["axis_x"]; ay0, ay1, ay2 = h["axis_y"]; az0, az1, az2 = h["axis_z"]
    hhx, hhy, hhz = h["half"]
    pcx, pcy, pcz = h["pc_center"]; pcnx, pcny, pcnz = h["pc_normal"]
    pchalf = h["pc_half"][0]; up_x, up_y, up_z = h["axis_y"]

    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    if norm < 1e-12: return None
    dx /= norm; dy /= norm; dz /= norm

    best_hit = None

    b_hit, b_entry, b_exit, b_front = _ray_box_intersect(ox, oy, oz, dx, dy, dz, cx, cy, cz, ax0, ax1, ax2, ay0, ay1, ay2, az0, az1, az2, hhx, hhy, hhz)

    if b_hit:
        origin_inside = b_entry < 1e-6

        if origin_inside:
            pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(ox, oy, oz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z)
            if pc_hit and pc_t > 1e-6:
                best_hit = {"type": "photocathode", "x": ox + dx * pc_t, "y": oy + dy * pc_t, "z": oz + dz * pc_t, "nx": float(nx), "ny": float(ny), "nz": float(nz), "t": float(pc_t)}

        elif b_front and b_entry > 1e-6:
            pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(ox, oy, oz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z)
            if pc_hit and pc_t > 1e-6 and pc_t <= b_exit + 1e-6:
                best_hit = {"type": "photocathode", "x": ox + dx * pc_t, "y": oy + dy * pc_t, "z": oz + dz * pc_t, "nx": float(nx), "ny": float(ny), "nz": float(nz), "t": float(pc_t)}
            else:
                best_hit = {"type": "housing", "x": ox + dx * b_entry, "y": oy + dy * b_entry, "z": oz + dz * b_entry, "nx": float(az0), "ny": float(az1), "nz": float(az2), "t": float(b_entry)}

        else:
            best_hit = {"type": "housing", "x": ox + dx * b_entry, "y": oy + dy * b_entry, "z": oz + dz * b_entry, "nx": float(az0), "ny": float(az1), "nz": float(az2), "t": float(b_entry)}

    else:
        pc_hit, pc_t, nx, ny, nz = _ray_rectangle_intersect(ox, oy, oz, dx, dy, dz, pcx, pcy, pcz, pcnx, pcny, pcnz, pchalf, up_x, up_y, up_z)
        if pc_hit and pc_t > 1e-6:
            best_hit = {"type": "photocathode", "x": ox + dx * pc_t, "y": oy + dy * pc_t, "z": oz + dz * pc_t, "nx": float(nx), "ny": float(ny), "nz": float(nz), "t": float(pc_t)}

    return best_hit


def _sanitise(obj):
    """Recursively convert numpy scalars to plain Python types."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# HTML / Three.js page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing:border-box; }
  body { margin:0; overflow:hidden; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#1a1a2e; display:flex; height:100vh; }
  #canvas-container { flex:1; position:relative; height:100vh; }
  #canvas-container canvas { display:block; }
  #status {
    position:absolute; bottom:12px; left:12px; z-index:100;
    color:rgba(255,255,255,0.5); font-size:12px;
  }
  #axis-label {
    position:absolute; bottom:12px; right:12px; z-index:100;
    color:rgba(255,255,255,0.4); font-size:11px;
    text-align:right; line-height:1.6;
  }
  .c { color:#ff6666; } .y { color:#66ff66; } .z { color:#6688ff; }
  .side-panel {
    background:rgba(30,30,60,0.92); color:#ccc; padding:16px;
    font-size:13px; overflow-y:auto; display:flex; flex-direction:column; gap:10px;
  }
  #controlPanel { width:280px; border-right:1px solid #333; }
  #displayPanel { width:380px; border-left:1px solid #333; }
  .side-panel h3 { margin:0 0 4px 0; color:#aac; font-size:14px; border-bottom:1px solid #444; padding-bottom:4px; }
  .side-panel label { display:flex; justify-content:space-between; align-items:center; font-size:12px; }
  .side-panel input[type=range] { width:120px; }
  .side-panel input[type=number] { width:70px; background:#222; color:#ccc; border:1px solid #444; padding:2px 4px; border-radius:3px; }
  .side-panel button {
    background:#4488cc; color:white; border:none; padding:6px 14px; border-radius:4px;
    cursor:pointer; font-size:13px; margin-top:4px;
  }
  .side-panel button:hover { background:#5599dd; }
  .side-panel button:active { background:#3377bb; }
  .side-panel .stat { font-size:12px; color:#8a8; }
  .side-panel .stat span { color:#aea; }
  .side-panel .hit-info { font-size:11px; color:#888; max-height:120px; overflow-y:auto; }
  .side-panel .muon-label { color:#ff8844; }
</style>
</head>
<body>
<div id="controlPanel" class="side-panel">
  <h3>Mode</h3>
  <div id="modeBtns" style="display:flex;gap:6px;">
    <button id="modeCherenkov" style="flex:1;background:#4488cc;font-size:12px;padding:4px 8px;">Cherenkov</button>
    <button id="modeSpot" style="flex:1;background:#555;font-size:12px;padding:4px 8px;">Spot Gun</button>
  </div>
  <label style="margin-top:2px;">
    <input type="checkbox" id="enableResponse"> LAPPD response pipeline
  </label>

  <h3>Source</h3>
  <label id="originX"><span style="width:80px;">X</span> <input type="number" id="mx" value="0" step="50" style="width:70px;"></label>
  <label id="originY"><span style="width:80px;">Y</span> <input type="number" id="my" value="-45" step="10" style="width:70px;"></label>
  <label id="interceptX" style="display:none;"><span style="width:80px;">X target</span> <input type="number" id="tx" value="0" step="10" style="width:70px;"></label>
  <label id="interceptY" style="display:none;"><span style="width:80px;">Y target</span> <input type="number" id="ty" value="-45" step="10" style="width:70px;"></label>
  <label>Z <input type="number" id="mz" value="500" step="50" style="width:70px;"></label>
  <label>θ (vertical °) <input type="range" id="theta" min="-90" max="90" step="1" value="0"><span id="thetaVal" style="min-width:32px;text-align:right;display:inline-block;color:#aea;">0</span></label>
  <label>φ (horizontal °) <input type="range" id="phi" min="-180" max="180" step="1" value="0"><span id="phiVal" style="min-width:36px;text-align:right;display:inline-block;color:#aea;">0</span></label>
  <label style="margin-top:-4px;font-size:11px;">
    <input type="checkbox" id="interceptMode"> Intercept mode (set PC hit)
  </label>
  <label>Photons/cm <input type="number" id="nPhotons" value="150" min="1" max="5000" step="1" style="width:90px;"></label>
  <label id="spreadLabel" style="display:none;">Spread (°) <input type="range" id="spread" min="0" max="10" step="0.1" value="2"><span id="spreadVal">2.0</span></label>
  <button id="traceBtn">Fire</button>

  <h3>Results</h3>
  <div class="stat">Hits: <span id="hitCount">-</span></div>
  <div class="stat">Hit rate: <span id="hitRate">-</span></div>
  <div class="hit-info" id="hitInfo">Adjust controls and click Fire</div>
</div>
<div id="canvas-container">
  <div id="status">LAPPD housing · drag to orbit · scroll to zoom</div>
  <div id="axis-label"><span class="c">X</span> tangential · <span class="y">Y</span> vertical · <span class="z">Z</span> radial</div>
</div>
<div id="displayPanel" class="side-panel"></div>

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

// ---- Scene setup ----
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 1, 10000);
camera.position.set(600, 400, 600);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

// ---- Lights ----
const ambient = new THREE.AmbientLight(0x404060, 0.8);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(500, 800, 600);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x6688cc, 0.5);
fillLight.position.set(-400, -200, -300);
scene.add(fillLight);

// ---- Grid & Axes ----
const grid = new THREE.GridHelper(1000, 20, 0x444466, 0x333355);
scene.add(grid);

const axes = new THREE.AxesHelper(300);
scene.add(axes);

const NUM_STRIPS = 28;

// ---- Dynamic groups ----
const rayGroup = new THREE.Group();
scene.add(rayGroup);

const hitGroup = new THREE.Group();
scene.add(hitGroup);

// ---- Muon marker (sphere + arrow + cone guide) ----
let muonGroup = new THREE.Group();
scene.add(muonGroup);
let coneVisual = null;

// ---- Intercept point marker (ring / crosshair on the PC) ----
let interceptMarker = null;
function updateInterceptMarker(pos, dir) {
  if (interceptMarker) { scene.remove(interceptMarker); interceptMarker = null; }
  if (!pos.intercept) return;

  // Mark the intercept point on the PC with a yellow/cyan ring
  const ring = new THREE.RingGeometry(4, 8, 20);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x00ffcc, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthWrite: false,
  });
  interceptMarker = new THREE.Mesh(ring, mat);
  interceptMarker.position.set(pos.intercept.x, pos.intercept.y, pos.intercept.z);
  // Orient ring to face the PC normal (toward -Z)
  const n = new THREE.Vector3(h.pc_normal[0], h.pc_normal[1], h.pc_normal[2]).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  if (Math.abs(n.dot(up)) > 0.99) up.set(1, 0, 0);
  interceptMarker.quaternion.setFromUnitVectors(up, n);
  scene.add(interceptMarker);
}

function updateMuonMarker(pos, dir) {
  while (muonGroup.children.length) muonGroup.children.length = 0;
  if (coneVisual) { scene.remove(coneVisual); coneVisual = null; }

  const p = new THREE.Vector3(pos.x, pos.y, pos.z);
  const d = new THREE.Vector3(dir.x, dir.y, dir.z).normalize();

  // Muon track line (4m from vertex backward along direction)
  const trackLen = 4000;
  const trailEnd = p.clone().add(d.clone().multiplyScalar(-trackLen));
  const trackPos = new Float32Array([p.x, p.y, p.z, trailEnd.x, trailEnd.y, trailEnd.z]);
  const trackGeo = new THREE.BufferGeometry();
  trackGeo.setAttribute('position', new THREE.BufferAttribute(trackPos, 3));
  const trackMat = new THREE.LineDashedMaterial({
    color: 0xff8844, dashSize: 20, gapSize: 15, transparent: true, opacity: 0.25,
  });
  const trackLine = new THREE.Line(trackGeo, trackMat);
  trackLine.computeLineDistances();
  muonGroup.add(trackLine);

  // Sphere at origin
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(10, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xff8844, emissive: 0xff4400, emissiveIntensity: 0.3 })
  );
  sphere.position.copy(p);
  muonGroup.add(sphere);

  // Direction arrow from origin
  const arrow = new THREE.ArrowHelper(d, p, 400, 0xff8844, 60, 30);
  muonGroup.add(arrow);

  // Cherenkov cone guide (apex at origin, pointing in muon direction)
  const coneLen = 800;
  const baseR = coneLen * Math.tan(0.73);
  const coneGeo = new THREE.ConeGeometry(baseR, coneLen, 36, 1, true);
  const coneMat = new THREE.MeshBasicMaterial({
    color: 0x66aaaa, wireframe: true, transparent: true, opacity: 0.2,
  });
  coneVisual = new THREE.Mesh(coneGeo, coneMat);
  const coneCenter = p.clone().add(d.clone().multiplyScalar(coneLen / 2));
  coneVisual.position.copy(coneCenter);
  const upDir = new THREE.Vector3(0, 1, 0);
  coneVisual.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(upDir, d.clone().negate()));
  scene.add(coneVisual);

  // Show intercept marker when in intercept mode
  updateInterceptMarker(pos, dir);
}

let currentHousing = null;

// ---- Load geometry ----
const housingResp = await fetch('/api/geometry');
const h = await housingResp.json();

// Housing box
const boxGeo = new THREE.BoxGeometry(h.half[0]*2, h.half[1]*2, h.half[2]*2);
const boxMat = new THREE.MeshStandardMaterial({
    color: 0x4488aa, transparent: true, opacity: 0.25,
    roughness: 0.5, metalness: 0.05, side: THREE.DoubleSide,
});
const boxMesh = new THREE.Mesh(boxGeo, boxMat);
boxMesh.position.set(h.center[0], h.center[1], h.center[2]);
const m4 = new THREE.Matrix4();
m4.set(
    h.axis_x[0], h.axis_y[0], h.axis_z[0], 0,
    h.axis_x[1], h.axis_y[1], h.axis_z[1], 0,
    h.axis_x[2], h.axis_y[2], h.axis_z[2], 0,
    0, 0, 0, 1,
);
boxMesh.quaternion.setFromRotationMatrix(m4);
scene.add(boxMesh);

const edgeGeo = new THREE.EdgesGeometry(boxGeo);
const edgeMat = new THREE.LineBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.6 });
const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
edgeMesh.position.copy(boxMesh.position);
edgeMesh.quaternion.copy(boxMesh.quaternion);
scene.add(edgeMesh);

// Photocathode
const pcGeo = new THREE.PlaneGeometry(h.pc_half[0]*2, h.pc_half[0]*2);
const pcMat = new THREE.MeshStandardMaterial({
    color: 0x66aadd, roughness: 0.3, metalness: 0.1, side: THREE.DoubleSide,
});
const pcMesh = new THREE.Mesh(pcGeo, pcMat);
pcMesh.position.set(h.pc_center[0], h.pc_center[1], h.pc_center[2]);
pcMesh.quaternion.copy(boxMesh.quaternion);
scene.add(pcMesh);

const pcEdgeGeo = new THREE.EdgesGeometry(pcGeo);
const pcEdgeMat = new THREE.LineBasicMaterial({ color: 0x88ddff, transparent: true, opacity: 0.4 });
const pcEdgeMesh = new THREE.LineSegments(pcEdgeGeo, pcEdgeMat);
pcEdgeMesh.position.copy(pcMesh.position);
pcEdgeMesh.quaternion.copy(pcMesh.quaternion);
scene.add(pcEdgeMesh);

currentHousing = h;

// ---- Glow texture (white tinted by vertex colors) ----
function makeWhiteGlowTexture(size) {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  const cx = size/2, cy = size/2;
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx);
  grad.addColorStop(0, 'white');
  grad.addColorStop(0.25, 'white');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.3)');
  grad.addColorStop(1, 'transparent');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}
const glowTex = makeWhiteGlowTexture(64);

// Heat colormap — blue→cyan→green→yellow→red over [tMin, tMax]
function heatColor(t, tMin, tMax) {
  const r = tMax > tMin ? (t - tMin) / (tMax - tMin) : 0.5;
  const clamped = Math.max(0, Math.min(1, r));
  // blue=0 → cyan=0.25 → green=0.5 → yellow=0.75 → red=1.0
  return [
    Math.min(1, clamped * 2),
    Math.min(1, 2 - Math.abs(clamped - 0.5) * 4),
    Math.max(0, 1 - clamped * 2),
  ];
}

// ---- Mode toggle ----
let currentMode = 'cherenkov'; // 'cherenkov' | 'spot'
const modeCherenkovBtn = document.getElementById('modeCherenkov');
const modeSpotBtn = document.getElementById('modeSpot');
const spreadLabel = document.getElementById('spreadLabel');
const spreadSlider = document.getElementById('spread');
const spreadVal = document.getElementById('spreadVal');
const traceBtn = document.getElementById('traceBtn');

function setMode(mode) {
  currentMode = mode;
  modeCherenkovBtn.style.background = mode === 'cherenkov' ? '#4488cc' : '#555';
  modeSpotBtn.style.background = mode === 'spot' ? '#4488cc' : '#555';
  spreadLabel.style.display = mode === 'spot' ? 'flex' : 'none';
  traceBtn.textContent = mode === 'spot' ? 'Fire Spot' : 'Trace Cherenkov';
}

modeCherenkovBtn.addEventListener('click', () => setMode('cherenkov'));
modeSpotBtn.addEventListener('click', () => setMode('spot'));
spreadSlider.addEventListener('input', () => {
  spreadVal.textContent = parseFloat(spreadSlider.value).toFixed(1);
});

// ---- Direction from theta/phi ----
const thetaSlider = document.getElementById('theta');
const phiSlider = document.getElementById('phi');

function getDirection() {
  // θ = elevation (°): 0=horizontal, +90=up (Y), -90=down (-Y)
  // φ = azimuth (°): 0=toward detector (-Z), +90=right (+X), -90=left (-X)
  const tr = parseFloat(thetaSlider.value) * Math.PI / 180;
  const pr = parseFloat(phiSlider.value) * Math.PI / 180;
  return {
    x: Math.cos(tr) * Math.sin(pr),
    y: Math.sin(tr),
    z: -Math.cos(tr) * Math.cos(pr),
  };
}

function getPosition() {
  const z = parseFloat(document.getElementById('mz').value) || 500;
  const dir = getDirection();
  if (document.getElementById('interceptMode').checked) {
    // Intercept mode: user sets (target_x, target_y) on the PC plane
    // and direction (θ, φ).  Compute the origin so the ray passes
    // through that intercept point.
    const tx = parseFloat(document.getElementById('tx').value) || 0;
    const ty = parseFloat(document.getElementById('ty').value) || 0;
    const pc_z = h.pc_center[2];   // PC plane Z
    const t = (pc_z - z) / dir.z;  // distance from origin to PC plane
    return {
      x: tx - dir.x * t,
      y: ty - dir.y * t,
      z: z,
      intercept: { x: tx, y: ty, z: pc_z },
    };
  }
  return {
    x: parseFloat(document.getElementById('mx').value) || 0,
    y: parseFloat(document.getElementById('my').value) || 0,
    z: z,
  };
}

// ---- Aim reticle (ring on LAPPD surface) ----
let aimReticle = null;

function updateAimReticle() {
  const pos = getPosition();
  const dir = getDirection();
  fetch(`/api/aim?ox=${pos.x}&oy=${pos.y}&oz=${pos.z}&dx=${dir.x.toFixed(6)}&dy=${dir.y.toFixed(6)}&dz=${dir.z.toFixed(6)}`)
    .then(r => r.json())
    .then(data => {
      if (aimReticle) { scene.remove(aimReticle); aimReticle = null; }
      if (data.hit) {
        const ring = new THREE.RingGeometry(6, 10, 24);
        const mat = new THREE.MeshBasicMaterial({
          color: data.type === 'photocathode' ? 0x00ff66 : 0xff8844,
          side: THREE.DoubleSide, transparent: true, opacity: 0.9,
          depthWrite: false,
        });
        aimReticle = new THREE.Mesh(ring, mat);
        aimReticle.position.set(data.x, data.y, data.z);
        // Orient ring to surface normal
        const n = new THREE.Vector3(data.nx, data.ny, data.nz).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        if (Math.abs(n.dot(up)) > 0.99) up.set(1, 0, 0);
        aimReticle.quaternion.setFromUnitVectors(up, n);
        scene.add(aimReticle);
      }
    })
    .catch(() => {});
}

function updateScene() {
  const pos = getPosition();
  const dir = getDirection();
  updateMuonMarker(pos, dir);
  if (!pos.intercept) updateAimReticle();
}

// Intercept mode toggle — swap origin X/Y for target X/Y
function updateInterceptMode() {
  const on = document.getElementById('interceptMode').checked;
  document.getElementById('originX').style.display = on ? 'none' : '';
  document.getElementById('originY').style.display = on ? 'none' : '';
  document.getElementById('interceptX').style.display = on ? '' : 'none';
  document.getElementById('interceptY').style.display = on ? '' : 'none';
  updateScene();
}
document.getElementById('interceptMode').addEventListener('change', updateInterceptMode);

thetaSlider.addEventListener('input', () => {
  document.getElementById('thetaVal').textContent = thetaSlider.value;
  updateScene();
});
phiSlider.addEventListener('input', () => {
  document.getElementById('phiVal').textContent = phiSlider.value;
  updateScene();
});
document.getElementById('mx').addEventListener('input', updateScene);
document.getElementById('my').addEventListener('input', updateScene);
document.getElementById('tx').addEventListener('input', updateScene);
document.getElementById('ty').addEventListener('input', updateScene);
document.getElementById('mz').addEventListener('input', updateScene);

setMode('cherenkov');
updateScene();

// ---- Trace ----
async function runTrace() {
  const pos = getPosition();
  const dir = getDirection();
  updateMuonMarker(pos, dir);
  const nPhotons = parseInt(document.getElementById('nPhotons').value) || 5000;

  while (rayGroup.children.length) rayGroup.remove(rayGroup.children[0]);
  while (hitGroup.children.length) hitGroup.remove(hitGroup.children[0]);

  document.getElementById('hitCount').textContent = 'tracing...';
  document.getElementById('hitRate').textContent = '';

  const spread = currentMode === 'spot' ? parseFloat(spreadSlider.value) : 2;
  const enableResp = document.getElementById('enableResponse').checked ? '&response=1' : '';
  const url = `/api/trace?x=${pos.x}&y=${pos.y}&z=${pos.z}&dx=${dir.x.toFixed(6)}&dy=${dir.y.toFixed(6)}&dz=${dir.z.toFixed(6)}&n=${nPhotons}&mode=${currentMode}&spread=${spread}${enableResp}`;

  const resp = await fetch(url);
  const data = await resp.json();

  document.getElementById('hitCount').textContent = data.n_hits;
  document.getElementById('hitRate').textContent = (data.n_hits / data.n_photons * 100).toFixed(1) + '%';

  // Ray lines
  const rayMat = new THREE.LineBasicMaterial({ color: 0x4488cc, transparent: true, opacity: 0.12 });
  for (const ray of data.rays) {
    const o = ray.origin; const d = ray.dir;
    const end = [o[0] + d[0] * 400, o[1] + d[1] * 400, o[2] + d[2] * 400];
    const verts = new Float32Array([o[0], o[1], o[2], end[0], end[1], end[2]]);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    rayGroup.add(new THREE.Line(g, rayMat));
  }

  // Hit points with glow sprites — color by arrival time
  const pcHits = data.hits.filter(h => h.type === 'photocathode');
  const hwHits = data.hits.filter(h => h.type === 'housing');

  // Determine arrival-time range for PC hits
  let tMin = 0, tMax = 1;
  if (pcHits.length > 0) {
    const times = pcHits.map(h => h.arrival_time || 0);
    tMin = Math.min(...times);
    tMax = Math.max(...times);
    if (tMax - tMin < 0.01) tMax = tMin + 0.01;
  }

  function addGlowPoints(hitList, size, colorFn) {
    if (hitList.length === 0) return;
    const n = hitList.length;
    const verts = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const h = hitList[i];
      verts[i*3] = h.x; verts[i*3+1] = h.y; verts[i*3+2] = h.z;
      const rgb = colorFn(h);
      colors[i*3] = rgb[0]; colors[i*3+1] = rgb[1]; colors[i*3+2] = rgb[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      map: glowTex, size: size, sizeAttenuation: true,
      transparent: true, opacity: 1.0, depthWrite: false,
      depthTest: false, vertexColors: true,
    });
    hitGroup.add(new THREE.Points(geo, mat));
  }
  addGlowPoints(pcHits, 28, (h) => heatColor(h.arrival_time || 0, tMin, tMax));
  addGlowPoints(hwHits, 12, () => [0.5, 0.5, 0.5]);

  // Hit info
  if (data.hits.length > 0) {
    document.getElementById('hitInfo').innerHTML =
      `<span style="color:#0f6">●</span> PC: <b>${pcHits.length}</b> &middot; ` +
      `<span style="color:#f55">●</span> housing: <b>${hwHits.length}</b>`;
  } else {
    document.getElementById('hitInfo').textContent = 'No hits';
  }

  // ---- Show response error, if any ----
  if (data.response_error) {
    const oldErr = document.getElementById('responseError');
    if (oldErr) oldErr.remove();
    const errDiv = document.createElement('div');
    errDiv.id = 'responseError';
    errDiv.style.cssText = 'color:#f66;font-size:11px;margin-top:4px;padding:4px;background:rgba(255,0,0,0.08);border-radius:3px;';
    errDiv.textContent = 'LAPPD response: ' + data.response_error;
    document.getElementById('displayPanel').appendChild(errDiv);
  } else {
    const oldErr = document.getElementById('responseError');
    if (oldErr) oldErr.remove();
  }

  // ---- Readout heatmaps (if full matrix data available) ----
  if (data.readout) {
    const detIdx = Object.keys(data.readout)[0];
    const readout = data.readout[detIdx];
    const end0 = readout.end0;  // 28 × 256 nested arrays
    const end1 = readout.end1;

    // Find global max voltage across both ends
    let maxV = 1e-6;
    for (let s = 0; s < NUM_STRIPS; s++) {
      for (let t = 0; t < 256; t++) {
        maxV = Math.max(maxV, end0[s][t], end1[s][t]);
      }
    }

    // Find peak charge time bin (sum over all strips, both ends)
    let peakBin = 0;
    let peakCharge = -1;
    for (let t = 0; t < 256; t++) {
      let ch = 0;
      for (let s = 0; s < NUM_STRIPS; s++) ch += end0[s][t] + end1[s][t];
      if (ch > peakCharge) { peakCharge = ch; peakBin = t; }
    }

    const mL = 24;             // left margin
    const stripH = 5;          // pixel height per strip (full view)
    const plotW = 256;         // 256 bins × 1px
    const plotH = NUM_STRIPS * stripH;
    const rowGap = 14;         // gap between matrices
    const labelH = 12;         // space for bottom label
    const totalW = mL + plotW;
    const totalH = (plotH + rowGap) * 2 + labelH;

    // ── Zoomed view params ──
    const halfWin = 40;        // 40 bins = 4ns
    const zStart = Math.max(0, peakBin - halfWin);
    const zEnd = Math.min(256, peakBin + halfWin);
    const zBins = zEnd - zStart;
    const zStripH = 12;        // taller strips in zoom
    const zTimeW = 3;          // wider time bins in zoom
    const zPlotW = zBins * zTimeW;
    const zPlotH = NUM_STRIPS * zStripH;
    const zTotalW = mL + zPlotW;
    const zTotalH = (zPlotH + rowGap) * 2 + labelH;

    const parent = document.getElementById('displayPanel');

    function makeCanvas(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
      const c = document.createElement('canvas');
      c.id = id;
      return c;
    }

    function heatColor(val) {
      const f = Math.min(1, val / maxV);
      const r = Math.min(1, f * 2);
      const g = Math.min(1, (f - 0.5) * 2);
      const b = Math.max(0, 1 - f * 2);
      return `rgb(${r*255|0},${g*255|0},${b*255|0})`;
    }

    function drawMatrix(ctxt, matrix, yOff, tOff, nBins, sH, tW, label) {
      for (let s = 0; s < NUM_STRIPS; s++) {
        for (let t = 0; t < nBins; t++) {
          ctxt.fillStyle = heatColor(matrix[s][tOff + t]);
          ctxt.fillRect(mL + t * tW, yOff + s * sH, tW, sH);
        }
      }
      ctxt.fillStyle = 'rgba(255,255,255,0.7)';
      ctxt.font = '10px sans-serif';
      ctxt.fillText(label, 2, yOff + 12);
      // Strip index labels
      ctxt.font = Math.min(9, sH - 1) + 'px sans-serif';
      ctxt.fillStyle = 'rgba(255,255,255,0.45)';
      for (let s = 0; s < NUM_STRIPS; s++) {
        if (s % 4 === 0) {
          ctxt.fillText(String(s + 1), 2, yOff + s * sH + sH - 1);
        }
      }
      // Time labels
      ctxt.fillStyle = 'rgba(255,255,255,0.3)';
      ctxt.font = '7px sans-serif';
      const step = Math.max(1, Math.round(50 / (tW || 1)));
      for (let t = 0; t <= nBins; t += step) {
        const ns = ((tOff + t) * 0.1).toFixed(1);
        ctxt.fillText(ns + 'ns', mL + t * tW - 8, yOff + sH * NUM_STRIPS + 10);
      }
    }

    // ── Full-range canvas ──
    const cFull = makeCanvas('readoutHeatmap');
    cFull.width = totalW;
    cFull.height = totalH;
    cFull.style.cssText = 'width:100%;height:auto;border-radius:4px;margin-top:6px;';
    parent.appendChild(cFull);
    const ctxFull = cFull.getContext('2d');

    drawMatrix(ctxFull, end0, 0, 0, 256, stripH, 1, 'End 0');
    drawMatrix(ctxFull, end1, plotH + rowGap, 0, 256, stripH, 1, 'End 1');

    ctxFull.fillStyle = 'rgba(255,255,255,0.25)';
    ctxFull.font = '9px sans-serif';
    ctxFull.fillText('Full (25.6 ns)', 2, totalH - 2);

    // ── Zoomed canvas ──
    const cZoom = makeCanvas('readoutZoom');
    cZoom.width = zTotalW;
    cZoom.height = zTotalH;
    cZoom.style.cssText = 'width:100%;height:auto;border-radius:4px;margin-top:4px;';
    parent.appendChild(cZoom);
    const ctxZoom = cZoom.getContext('2d');

    function drawZoomMatrix(ctxt, matrix, yOff, label) {
      // Highlight peak column
      ctxt.fillStyle = 'rgba(255,255,0,0.10)';
      const peakX = mL + (peakBin - zStart) * zTimeW;
      ctxt.fillRect(peakX, yOff, zTimeW, zPlotH);

      drawMatrix(ctxt, matrix, yOff, zStart, zBins, zStripH, zTimeW, label);
    }

    drawZoomMatrix(ctxZoom, end0, 0, 'End 0 (zoom)');
    drawZoomMatrix(ctxZoom, end1, zPlotH + rowGap, 'End 1 (zoom)');

    ctxZoom.fillStyle = 'rgba(255,255,255,0.25)';
    ctxZoom.font = '9px sans-serif';
    ctxZoom.fillText('Zoom 8 ns around peak', 2, zTotalH - 2);

  } else {
    const existing = document.getElementById('readoutHeatmap');
    if (existing) existing.remove();
    const existingZ = document.getElementById('readoutZoom');
    if (existingZ) existingZ.remove();
  }
}

traceBtn.addEventListener('click', runTrace);

// ---- Animation ----
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

// ---- Resize ----
window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
});

document.title = 'LAPPD Cherenkov Viewer';
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class LAPPDServer(BaseHTTPRequestHandler):
    _housing: dict = {}

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._send_html()
        elif path == "/api/geometry":
            self._send_json(self._housing)
        elif path == "/api/trace":
            self._handle_trace(qs)
        elif path == "/api/aim":
            self._handle_aim(qs)
        else:
            self.send_error(404)

    def _handle_trace(self, qs: dict):
        try:
            x = float(qs.get("x", ["0"])[0])
            y = float(qs.get("y", ["0"])[0])
            z = float(qs.get("z", ["500"])[0])
            dx = float(qs.get("dx", ["0"])[0])
            dy = float(qs.get("dy", ["0"])[0])
            dz = float(qs.get("dz", ["-1"])[0])
            n = int(qs.get("n", ["5000"])[0])
            mode = qs.get("mode", ["cherenkov"])[0]
            spread = float(qs.get("spread", ["2"])[0])
            enable_response = qs.get("response", ["0"])[0] in ("1", "true", "yes")
        except (ValueError, TypeError):
            self._send_json({"error": "invalid parameters"}, 400)
            return

        norm = math.sqrt(dx*dx + dy*dy + dz*dz)
        if norm < 1e-12:
            self._send_json({"error": "zero direction"}, 400)
            return
        dx /= norm; dy /= norm; dz /= norm

        rng = np.random.default_rng()
        if mode == "spot":
            result = _trace_spot_on_lappd(
                (x, y, z), (dx, dy, dz), n, spread,
                self._housing, rng,
            )
        else:
            result = _trace_cherenkov_on_lappd(
                (x, y, z), (dx, dy, dz), n,
                self._housing, rng,
            )

        # Optionally run the LAPPD response pipeline
        if enable_response and result.get("hits"):
            try:
                cfg = LAPPDResponseConfig()
                resp = process_hit_dicts(result["hits"], config=cfg)
                if resp:
                    for det_idx, data in resp.items():
                        # Full 28×256 voltage-time matrices as nested lists
                        s0 = data["side0"].tolist()
                        s1 = data["side1"].tolist()
                        if "readout" not in result:
                            result["readout"] = {}
                        result["readout"][det_idx] = {
                            "end0": s0,
                            "end1": s1,
                        }
            except Exception as exc:
                result["response_error"] = str(exc)
                import traceback as _tb
                _tb.print_exc()

        self._send_json(result)

    def _handle_aim(self, qs: dict):
        try:
            ox = float(qs.get("ox", ["0"])[0])
            oy = float(qs.get("oy", ["0"])[0])
            oz = float(qs.get("oz", ["500"])[0])
            dx = float(qs.get("dx", ["0"])[0])
            dy = float(qs.get("dy", ["0"])[0])
            dz = float(qs.get("dz", ["-1"])[0])
        except (ValueError, TypeError):
            self._send_json({"error": "invalid parameters"}, 400)
            return

        hit = _trace_single_ray(ox, oy, oz, dx, dy, dz, self._housing)
        if hit is None:
            self._send_json({"hit": False})
        else:
            self._send_json({"hit": True, "type": hit["type"], "x": hit["x"], "y": hit["y"], "z": hit["z"], "nx": hit["nx"], "ny": hit["ny"], "nz": hit["nz"]})


def run_server(host: str = "localhost", port: int = 8081) -> None:
    global housing_json
    housing_json = _build_housing_json()
    LAPPDServer._housing = housing_json
    server = HTTPServer((host, port), LAPPDServer)
    print(f"LAPPD Cherenkov viewer at http://{host}:{port}/")
    print("  Use the side panel to adjust muon position and trace Cherenkov light.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _build_housing_json() -> dict:
    housing = build_housing((0, 0, 0), (0, 0, 1))
    hd, ad = housing_to_arrays(housing)
    h = hd[0]
    a = ad[0]
    return {
        "center": [float(h[0]), float(h[1]), float(h[2])],
        "axis_x": [float(h[3]), float(h[4]), float(h[5])],
        "axis_y": [float(h[6]), float(h[7]), float(h[8])],
        "axis_z": [float(h[9]), float(h[10]), float(h[11])],
        "half": [float(h[12]), float(h[13]), float(h[14])],
        "pc_center": [float(a[0]), float(a[1]), float(a[2])],
        "pc_normal": [float(a[3]), float(a[4]), float(a[5])],
        "pc_half": [float(a[6])],
    }
