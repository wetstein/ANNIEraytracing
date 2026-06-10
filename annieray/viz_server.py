"""Interactive Cherenkov visualization server for ANNIE.

Serves a Three.js frontend with real-time ray tracing of Cherenkov cones
from a user-controlled muon track.

Usage:
    python -m annieray viz-server --gdml PHASE2_INNER_STRUCTURE.gdml \
        --pmt-csv PMTPositions_Scan.txt
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from annieray.tracer import build_geometry, trace_cherenkov

geometry = None
n_pmts = 0


class VizHandler(BaseHTTPRequestHandler):
    _pmt_types: list[str] = []

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
        elif path == "/api/detectors":
            self._send_detectors()
        else:
            self.send_error(404)

    def _send_lappd_housing(self):
        if geometry.lappd_housing_data.shape[0] == 0:
            self._send_json({"housing": []})
            return
        hd = geometry.lappd_housing_data[0].tolist()
        ad = geometry.annie_lappd_data[0].tolist()
        self._send_json({
            "housing": {
                "center": [hd[0], hd[1], hd[2]],
                "axis_x": [hd[3], hd[4], hd[5]],
                "axis_y": [hd[6], hd[7], hd[8]],
                "axis_z": [hd[9], hd[10], hd[11]],
                "half": [hd[12], hd[13], hd[14]],
                "pc_center": [ad[0], ad[1], ad[2]],
                "pc_normal": [ad[3], ad[4], ad[5]],
                "pc_half": [ad[6]],
            },
        })

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
        centers = geometry.pmt_centers.tolist()
        radii = geometry.pmt_radii.tolist()
        self._send_json({
            "centers": centers,
            "radii": radii,
            "types": self._pmt_types,
        })

    def _send_mesh_verts(self):
        buf = geometry.mesh_vertices.astype(np.float32).tobytes()
        self._send_binary(buf)

    def _send_mesh_tris(self):
        buf = geometry.mesh_triangles.astype(np.int32).tobytes()
        self._send_binary(buf)

    def _handle_trace(self, params):
        try:
            mx = float(params.get("mx", ["0"])[0])
            my = float(params.get("my", ["0"])[0])
            mz = float(params.get("mz", ["2000"])[0])
            dx = float(params.get("dx", ["0"])[0])
            dy = float(params.get("dy", ["0"])[0])
            dz = float(params.get("dz", ["-1"])[0])
            n = int(params.get("n", ["10000"])[0])
            n = min(max(n, 100), 1_000_000)
        except (ValueError, TypeError):
            self._send_json({"error": "invalid parameters"}, 400)
            return

        rng = np.random.default_rng()
        t0 = time.time()
        hits = trace_cherenkov(
            (mx, my, mz), (dx, dy, dz), n, geometry, rng=rng,
        )
        elapsed = time.time() - t0

        total_hits = int(hits[:, 0].sum())

        # Return positions by component type — these are drawn as dots
        comp = hits[:, 8]
        pmt_positions = hits[comp == 2.0, 2:5].tolist()
        struct_positions = hits[comp == 1.0, 2:5].tolist()
        tank_positions  = hits[comp == 4.0, 2:5].tolist()

        self._send_json({
            "pmt_positions": pmt_positions,
            "struct_positions": struct_positions,
            "tank_positions": tank_positions,
            "total_hits": total_hits,
            "total_photons": n,
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
  <label>LAPPD Radial <span id="lappdRadialVal">-62</span>
    <input type="range" id="lappdRadial" min="-200" max="200" step="1" value="-62"></label>
  <label>LAPPD Vertical <span id="lappdVertVal">740</span>
    <input type="range" id="lappdVert" min="-1000" max="1000" step="5" value="740"></label>
  <button id="focusLAPPD" disabled>Focus on LAPPD</button>
  <label style="margin-top:6px;"><input type="checkbox" id="lappdGrey"> Grey LAPPD</label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label>Bottom Rot <span id="bottomRotVal">135</span>
    <input type="range" id="bottomRot" min="-180" max="180" step="22.5" value="135"></label>
  <label style="margin-top:6px;"><input type="checkbox" id="structGrey"> Grey Structure</label>
  <label style="margin-top:2px;"><input type="checkbox" id="pmtGrey"> Grey PMTs</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showCone" checked> Show Cone Guide</label>
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
let pmtMeshes = [];
let muonGroup = null;
let coneVisual = null;
let spotLight = null;
let spotTarget = null;
let housingData = null;
let boxMesh = null;
let tracePoints = null;   // group of dot meshes for traced photon hits
let traceResultEl = null;

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

// Grid helper
const grid = new THREE.GridHelper(3000, 20, 0x8899aa, 0x667788);
grid.position.y = 2000;
grid.receiveShadow = false;
scene.add(grid);

// Axes
const axes = new THREE.AxesHelper(500);
scene.add(axes);

// ---- Material colors ----
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

const pmtMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.05,
    metalness: 0.0,
    transmission: 0.85,
    thickness: 0.5,
    ior: 1.5,
    envMapIntensity: 1.0,
    clearcoat: 0.0,
});

function createPMTSphere(center, radius) {
    const geo = new THREE.SphereGeometry(radius, 28, 22);
    const mesh = new THREE.Mesh(geo, pmtMat);
    mesh.position.set(center[0], center[1], center[2]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
}

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
    coneVisual.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(upDir, d));
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
    const n = 10000;

    const btn = document.getElementById('traceBtn');
    btn.disabled = true;
    btn.textContent = 'Tracing…';
    traceResultEl = document.getElementById('traceResult');

    try {
        const url = `/api/trace?mx=${mx}&my=${my}&mz=${mz}&dx=${dx.toFixed(6)}&dy=${dy.toFixed(6)}&dz=${dz.toFixed(6)}&n=${n}`;
        const resp = await fetch(url);
        const data = await resp.json();

        // Remove previous trace dots
        if (tracePoints) scene.remove(tracePoints);

        const dotGeo = new THREE.SphereGeometry(8, 6, 6);
        const pmtMat = new THREE.MeshBasicMaterial({ color: 0x44dd88 });
        const structMat = new THREE.MeshBasicMaterial({ color: 0xff8844 });
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
        for (const p of data.tank_positions) {
            const m = new THREE.Mesh(dotGeo, tankMat);
            m.position.set(p[0], p[1], p[2]);
            tracePoints.add(m);
        }
        scene.add(tracePoints);

        traceResultEl.innerHTML = `<b>${data.total_hits}</b>/<b>${data.total_photons}</b> hits `
            + `(PMT <b>${data.pmt_positions.length}</b>, `
            + `struct <b>${data.struct_positions.length}</b>, `
            + `tank <b>${data.tank_positions.length}</b>) `
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
            side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(flatGeo, meshMat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        statusEl.textContent = 'Loading PMTs…';
        const pmtData = await loadPMTs();
        pmtMeshes = pmtData.centers.map((c, i) => {
            const mesh = createPMTSphere(c, pmtData.radii[i]);
            mesh.userData.type = pmtData.types[i];
            scene.add(mesh);
            return mesh;
        });
        // Store original positions for bottom PMT rotation
        const origPMTPos = pmtData.centers.map(c => new THREE.Vector3(c[0], c[1], c[2]));
        const isBottomPMT = pmtData.types.map(t => t === 'LUX');
        statusEl.textContent = `Loaded ${pmtMeshes.length} PMTs.`;

        // Load ANNIE LAPPD housing (if present)
        const housingResp = await fetch('/api/housing');
        housingData = await housingResp.json();
        if (!Array.isArray(housingData.housing)) {
            const h = housingData.housing;
            const origCenter = new THREE.Vector3(h.center[0], h.center[1], h.center[2]);
            const origR = Math.hypot(origCenter.x, origCenter.y);
            const origPC = new THREE.Vector3(h.pc_center[0], h.pc_center[1], h.pc_center[2]);

            // Housing box
            const boxGeo = new THREE.BoxGeometry(h.half[0]*2, h.half[1]*2, h.half[2]*2);
            const boxMat = new THREE.MeshStandardMaterial({
                color: 0x446688,
                transparent: true,
                opacity: 0.25,
                roughness: 0.6,
                metalness: 0.0,
                side: THREE.DoubleSide,
            });
            boxMesh = new THREE.Mesh(boxGeo, boxMat);
            boxMesh.position.copy(origCenter);
            const m4 = new THREE.Matrix4();
            m4.set(
                h.axis_x[0], h.axis_y[0], h.axis_z[0], 0,
                h.axis_x[1], h.axis_y[1], h.axis_z[1], 0,
                h.axis_x[2], h.axis_y[2], h.axis_z[2], 0,
                0, 0, 0, 1,
            );
            const baseQuat = new THREE.Quaternion().setFromRotationMatrix(m4);
            boxMesh.quaternion.copy(baseQuat);
            boxMesh.castShadow = true;
            boxMesh.receiveShadow = true;
            scene.add(boxMesh);

            // Photocathode rectangle
            const pcGeo = new THREE.PlaneGeometry(h.pc_half[0]*2, h.pc_half[0]*2);
            const pcMat = new THREE.MeshStandardMaterial({
                color: 0x88bbdd,
                roughness: 0.3,
                metalness: 0.1,
                side: THREE.DoubleSide,
            });
            const pcMesh = new THREE.Mesh(pcGeo, pcMat);
            pcMesh.position.copy(origPC);
            pcMesh.quaternion.copy(boxMesh.quaternion);
            pcMesh.castShadow = true;
            pcMesh.receiveShadow = true;
            scene.add(pcMesh);

            // LAPPD position sliders
            const radialSlider = document.getElementById('lappdRadial');
            const vertSlider = document.getElementById('lappdVert');
            const radialVal = document.getElementById('lappdRadialVal');
            const vertVal = document.getElementById('lappdVertVal');

            function updateHousing() {
                const dr = parseFloat(radialSlider.value);
                const dz = parseFloat(vertSlider.value);

                const scale = (origR + dr) / origR;
                const pos = origCenter.clone().multiplyScalar(scale);
                pos.z = origCenter.z + dz;
                boxMesh.position.copy(pos);

                const pcPos = origPC.clone().multiplyScalar(scale);
                pcPos.z = origPC.z + dz;
                pcMesh.position.copy(pcPos);

                radialVal.textContent = dr;
                vertVal.textContent = dz;
            }

            radialSlider.addEventListener('input', updateHousing);
            vertSlider.addEventListener('input', updateHousing);

            // Grey LAPPD checkbox
            const greyCheck = document.getElementById('lappdGrey');
            greyCheck.addEventListener('change', () => {
                const isGrey = greyCheck.checked;
                if (isGrey) {
                    boxMat.color.setHex(0x999999);
                    boxMat.transparent = false;
                    boxMat.opacity = 1.0;
                    boxMat.roughness = 0.5;
                    boxMat.metalness = 0.05;
                    pcMat.color.setHex(0x999999);
                    pcMat.roughness = 0.5;
                    pcMat.metalness = 0.05;
                } else {
                    boxMat.color.setHex(0x446688);
                    boxMat.transparent = true;
                    boxMat.opacity = 0.25;
                    boxMat.roughness = 0.6;
                    boxMat.metalness = 0.0;
                    pcMat.color.setHex(0x88bbdd);
                    pcMat.roughness = 0.3;
                    pcMat.metalness = 0.1;
                }
            });

            document.getElementById('focusLAPPD').disabled = false;
        }

        // ---- Bottom PMT rotation slider ----
        const bottomRotSlider = document.getElementById('bottomRot');
        const bottomRotVal = document.getElementById('bottomRotVal');

        function rotateBottomPMTs() {
            const angle = parseFloat(bottomRotSlider.value) * Math.PI / 180;
            const cosA = Math.cos(angle);
            const sinA = Math.sin(angle);
            for (let i = 0; i < pmtMeshes.length; i++) {
                if (isBottomPMT[i]) {
                    const orig = origPMTPos[i];
                    const x = orig.x * cosA - orig.y * sinA;
                    const y = orig.x * sinA + orig.y * cosA;
                    pmtMeshes[i].position.set(x, y, orig.z);
                }
            }
            bottomRotVal.textContent = bottomRotSlider.value;
        }

        bottomRotSlider.addEventListener('input', rotateBottomPMTs);
        rotateBottomPMTs();  // apply default 135°

        // ---- Grey toggles for structure and PMTs ----
        const structGreyCheck = document.getElementById('structGrey');
        const pmtGreyCheck = document.getElementById('pmtGrey');

        structGreyCheck.addEventListener('change', () => {
            meshMat.color.setHex(structGreyCheck.checked ? 0x999999 : 0xffffff);
        });

        pmtGreyCheck.addEventListener('change', () => {
            const isGrey = pmtGreyCheck.checked;
            if (isGrey) {
                pmtMat.color.setHex(0x999999);
                pmtMat.transmission = 0;
                pmtMat.roughness = 0.5;
                pmtMat.metalness = 0.05;
            } else {
                pmtMat.color.setHex(0xffffff);
                pmtMat.transmission = 0.85;
                pmtMat.roughness = 0.05;
                pmtMat.metalness = 0.0;
            }
            pmtMat.needsUpdate = true;
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

    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

function focusLAPPD() {
    if (!boxMesh) return;
    const target = boxMesh.position.clone();
    const normal = new THREE.Vector3(
        housingData.housing.pc_normal[0],
        housingData.housing.pc_normal[1],
        housingData.housing.pc_normal[2],
    );
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
</body>
</html>
"""


def run_server(args):
    global geometry, n_pmts

    import taichi as ti

    ti.init(default_fp=ti.f32)

    pmt_csv = args.pmt_csv
    if pmt_csv is None:
        pmt_csv = Path("PMTPositions_Scan.txt")

    print(f"Loading geometry from {args.gdml}...")
    geometry = build_geometry(
        args.gdml,
        step_path=args.step,
        manifest_path=args.manifest,
        pmt_csv_path=pmt_csv,
        no_lappd=args.no_lappd,
        z_offset=args.z_offset,
        lappd_model=args.lappd_model,
    )
    print(f"  Mesh: {geometry.mesh_vertices.shape[0]} verts, {geometry.mesh_triangles.shape[0]} tris")
    print(f"  PMTs: {geometry.pmt_centers.shape[0]}")
    n_pmts = geometry.pmt_centers.shape[0]

    from annieray.pmt_loader import load_pmts
    pmt_info = load_pmts(pmt_csv, z_offset=args.z_offset)
    VizHandler._pmt_types = pmt_info["types"]

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
