# ANNIEraytracing

GPU-accelerated ray tracer for the ANNIE detector, implemented in Taichi.
Traces photons through a GDML-detailed inner structure, PMTs, LAPPDs, and
obscurant surfboard panels, recording hit positions, local coordinates,
arrival times, and wavelengths.

## Quick Start

```bash
# Batch-mode simulation: Cherenkov photons from a muon track
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 150

# Interactive 3D viewer with ray tracing
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt --port 8080

# With 3 surfboard obscurant panels and ANNIE LAPPD housing model
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 --lappd-model annie

# Build detector registry YAML (one-time setup for model coupling)
python -m annieray build-detector-config \
    --pmt-csv PMTPositions_Scan.txt -o detectors.yaml
```

## CLI Commands

### `batch` — Batch-mode event generation

Runs N events on the GPU, writing results to Parquet files. No display
server required.

```
python -m annieray batch [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gdml` | `PHASE2_INNER_STRUCTURE_closed.gdml` | GDML structure mesh |
| `--step` | None | STEP CAD manifest (for LAPPD candidates) |
| `--manifest` | None | Pre-cached component manifest JSON |
| `--pmt-csv` | None | PMT position file (`PMTPositions_Scan.txt`) |
| `--events` | 100 | Number of events to generate |
| `--photons-per-cm` | 150 | Photons per cm along the muon track |
| `--batch-size` | 50 | Events per GPU launch (higher = faster) |
| `--muon-fixed` | None | Fixed muon topology: `"x y z t0 dx dy dz"` (7 floats) |
| `--muon-file` | None | File with one topology per line (`x y z t0 dx dy dz`) |
| `--surfboard` | 0 | PVC surfboard panels (`0`, `1`, or `3`) |
| `--lappd-model` | `annie` | LAPPD geometry (`default` = bare rectangle, `annie` = housed) |
| `--lappd-indices` | None | Comma-separated LAPPD candidate indices from STEP |
| `--det-rotation` | 22.5 | Global Z-rotation (deg) aligning +Y with octagon corner |
| `--z-offset` | 0.0 | Vertical offset applied to all PMT positions (mm) |
| `--no-lappd` | false | Skip LAPPD rectangles entirely |
| `--max-bounces` | 0 | Multi-bounce optics (0 = single bounce, 3 = up to 3 bounces) |
| `--pmt-response` | false | Enable PMT digital model (SPE charge + TTS) |
| `--full-wf` | false | Full waveform path for PMT response (requires `--pmt-response`) |
| `--output-dir` | `results/` | Output directory for Parquet files |
| `--no-record` | false | Skip writing per-event output files |
| `--wavelength` | 350.0 | Cherenkov photon wavelength (nm) |
| `--seed` | None | Random seed for reproducibility |

**Muon topology** — By default muons are sampled uniformly in position
and direction inside the tank. With `--muon-fixed` you specify a single
topology for every event:

```bash
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --muon-fixed "0 -45 2000 0 0 0 -1" \
    --events 1000 --photons-per-cm 150
```

The 7 values are: `x y z t0 dx dy dz` (t0 in ns, position in mm,
direction as a unit vector).

**Output files** — Two Parquet files in `--output-dir`:

- `photon_hits.parquet` — per-photon hit records with event_id, position,
  normal, component, detector info, local coordinates, arrival time,
  wavelength, and bounce count.
- `pmt_responses.parquet` — when `--pmt-response` is used, per-PMT
  digitised charges and hit times.

### `viz-server` — Interactive 3D viewer

Starts a local HTTP server with a Three.js frontend. Click on PMTs or
surfboard panels to adjust positions in real time. Trace Cherenkov cones
from a user-controlled muon.

```
python -m annieray viz-server [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gdml` | `PHASE2_INNER_STRUCTURE.gdml` | GDML structure mesh |
| `--step` | None | STEP CAD manifest |
| `--manifest` | None | Pre-cached component manifest JSON |
| `--pmt-csv` | None | PMT position file |
| `--det-rotation` | 22.5 | Global Z-rotation (deg) |
| `--z-offset` | 0.0 | Vertical offset (mm) |
| `--no-lappd` | false | Skip LAPPD rectangles |
| `--lappd-model` | `annie` | LAPPD geometry model |
| `--bottom-rot` | 45.0 | Extra Z-rotation for bottom PMTs (deg) |
| `--bottom-spin` | 0.0 | Spin rotation for bottom PMTs (deg) |
| `--surfboard` | 0 | PVC surfboard panels (`0`, `1`, or `3`) |
| `--port` | 8080 | HTTP port |
| `--host` | `0.0.0.0` | Bind address |

**Interactive features:**
- **Muon controls**: set X/Y/Z position and theta/phi direction, then
  click "Trace Cherenkov" to run the GPU kernel and see hit dots.
- **PMT adjustment**: click any PMT → popup adjusts axial/tangential/
  vertical position with live preview; save persists to `corrections.csv`.
- **Surfboard adjustment**: click a surfboard panel → popup adjusts
  vertical/radial/tangential position; reset restores default.
- **LAPPD housing adjustment**: click a housing box → same popup pattern
  for per-housing vertical/radial/tangential position.
- **Toggles**: grey-structure, grey-PMT, grey-LAPPD modes; structure
  opacity slider; LAPPD global dx/dy/dz correction (saves to
  `lappd_corrections.csv`).

```bash
# Full featured viewer
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 --lappd-model annie \
    --port 8080
```

### `build-detector-config` — Detector registry YAML

Scans the geometry and writes a `detectors.yaml` mapping stable IDs to
hardware info (position, direction, type, panel). Needed for coupling
with external simulation frameworks.

```
python -m annieray build-detector-config --pmt-csv PMTPositions_Scan.txt -o detectors.yaml
```

## Surfboard Obscurant Panels

PVC panels (2450 x 280 x 10 mm) mounted vertically at the forward
octagon vertices (45°, 90°, 135°). They absorb photons and stop muon
tracks. Configurable via `--surfboard {0,1,3}`.

When `--surfboard 3` is combined with `--lappd-model annie`, three
LAPPD housings are automatically built in front of each panel, offset
radially inward and staggered vertically (leftmost higher, rightmost
lower, centre at mid-Z) to form a diagonal in the visualisation frame.

Each housing is clickable in the viz viewer for individual position
adjustment (vertical/radial/tangential sliders).

## LAPPD Housing Model

The `--lappd-model annie` flag replaces the default bare photocathode
rectangle with the full Kandemir waterproof housing: a 5-sided acrylic
box (330 x 430 x 60 mm) with an off-centre photocathode (191.5 x 191.5 mm)
at local-Z = +3.5 mm from the housing centre.

The housing is built from the `LAPPDHousing` dataclass in
`annieray/lappd_model.py` and stored in the geometry as
`lappd_housing_data` (S, 16) and `annie_lappd_data` (S, 7) arrays,
one row per housing instance.

## Interactive Position Correction

The viewer supports three levels of position adjustment:

1. **PMT corrections** — saved to `corrections.csv`. One row per PMT
   (axial, tangential, vertical in the PMT local frame). Applied at
   geometry load time and during trace operations.

2. **LAPPD global correction** — saved to `lappd_corrections.csv`.
   A single dx/dy/dz offset applied uniformly to all LAPPD housings.

3. **Per-object sliders** — click any surfboard or LAPPD housing in
   the 3D view for a popup with live-adjust sliders (vertical, radial,
   tangential in the object's local frame). Adjustments update the
   server-side geometry arrays in real time.

## How the Ray Tracing Works

### Pipeline overview

```
     GDML file          PMT CSV / STEP CAD
         |                     |
    gdml_parser.py       pmt_loader.py / step_parser.py
         |                     |
         v                     v
    build_geometry() ──────> Geometry dataclass
                                  |
                    ┌─────────────┴──────────────┐
                    v                             v
              trace_rays()                 trace_cherenkov()
                    |                             |
                    └─────────────┬───────────────┘
                                  v
                          trace_kernel()
                     (Taichi GPU kernel)
                                  |
                     ┌────────────┴────────────┐
                     v                         v
                hits ndarray            expand with arrival_time
                     |                   + wavelength + bounce
                     v
               write_hits() / BatchAccumulator
                     |
                     v
               hits.parquet
```

### The Taichi kernel (`trace_kernel`)

Each GPU thread processes one photon:

1. **Normalises** the direction vector.
2. **Scans all mesh triangles** (structure) via Möller–Trumbore.
3. **Scans all PMTs** via sphere intersection + hemisphere check.
   Computes local polar/azimuthal coordinates for angular response.
4. **Scans PMT hardware meshes** (holders) via per-PMT oriented triangle
   intersection.
5. **Scans all default LAPPDs** via oriented-rectangle intersection.
   Computes strip-aligned local coordinates (along/across strips).
6. **Finds ANNIE LAPPD housings** via oriented-box slab intersection.
   Side/back faces absorb; front face passes to a photocathode rectangle.
7. **Scans surfboard panels** via oriented-box intersection. Hits are
   recorded as `CID_INNER_STRUCTURE` with material ID 3 (PVC) for
   multi-bounce optics evaluation.
8. **Finds the tank wall** via infinite-cylinder intersection.
9. **Writes the nearest hit** with component ID, detector index/system,
   local coordinates, and material ID.

### Multi-bounce optics

With `--max-bounces N` (N > 0), `trace_with_optics()` manages N rounds
of Fresnel reflection/transmission and diffuse reflection per material.
Each bounce computes a new direction from the surface BRDF and retraces
the ray, accumulating optical path length and maintaining the bounce
count in the output.

## Batch-Mode Simulations

The `batch` command requires no display server. It:

1. Builds the geometry once.
2. For each event, draws a muon topology (position + direction).
3. Generates Cherenkov photons along the muon track.
4. Launches the GPU kernel on up to `--batch-size` events at once
   (~2.7x speedup vs per-event launches).
5. Collects hits and optionally runs the PMT digital model.
6. Writes `photon_hits.parquet` and `pmt_responses.parquet`.

### Output schema — `photon_hits.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `hit_flag` | int32 | 1 = hit, 0 = miss |
| `t` | float32 | Path length (mm) |
| `x, y, z` | float32 | Hit position (mm) |
| `nx, ny, nz` | float32 | Surface normal |
| `component_id` | int32 | 1=structure, 2=PMT, 3=LAPPD, 4=tank wall |
| `detector_index` | int32 | Index into detector registry, -1 if none |
| `detector_system` | int32 | 0=pmt, 1=lappd_default, 2=lappd_annie |
| `local_u` | float32 | Along-strip (LAPPD) / polar angle (PMT) |
| `local_v` | float32 | Across-strip (LAPPD) / azimuthal angle (PMT) |
| `material_id` | int32 | Hit material (structure materials, PVC=3) |
| `arrival_time` | float32 | Photon travel time (ns) |
| `wavelength` | float32 | Photon wavelength (nm) |
| `bounce_count` | int32 | Number of surface reflections |
| `photon_id` | int64 | Sequential photon index within event |

### PMT digital model

When `--pmt-response` is set, `process_pmt_hits()` runs after tracing:

- **Fast path** (default): assigns SPE charge (Gaussian-smear) and
  transit time spread per PMT type. See `annieray/pmt_response.py`.
- **Full waveform path** (`--full-wf`): synthesises SPE pulse trains
  with realistic pulse shapes and a hit-finding algorithm.

Output written to `pmt_responses.parquet` with columns: `event_id`,
`detector_index`, `charge_pe`, `hit_time_ns`.

## Code Map

| File | Role |
|------|------|
| `cli.py` | CLI parser, `batch`/`viz-server`/`build-detector-config` subcommands |
| `tracer.py` | Geometry dataclass, `build_geometry()`, Taichi trace kernel, all intersection functions, `trace_cherenkov()`, multi-bounce `trace_with_optics()` |
| `batch.py` | Batch-mode event loop, `BatchAccumulator` for Parquet output |
| `cherenkov.py` | Vectorised Cherenkov photon generator (~13 ms for 60K photons) |
| `pmt_response.py` | PMT digital model — fast path and full-waveform path |
| `lappd_response.py` | Taichi-accelerated LAPPD digitisation pipeline |
| `lappd_model.py` | `LAPPDHousing` dataclass, housing geometry builder, `compute_housing_track_length()` |
| `detectors.py` | `DetectorInfo`, `build_detector_registry()`, YAML I/O |
| `output.py` | Parquet writer, hit schema |
| `gdml_parser.py` | GDML mesh parser (vertex/triangle arrays) |
| `pmt_loader.py` | PMT CSV parser, mesh loader, hardware mesh builder |
| `pmt_mesh.py` | PMT body/hardware mesh loading and array building |
| `step_parser.py` | STEP CAD parser (component manifest) |
| `viz_server.py` | Interactive Three.js viewer with real-time ray tracing |
| `viz_lappd_server.py` | Standalone LAPPD module viewer |
| `optics.py` | Optical material database, Fresnel/reflectance evaluation |
| `_version.py` | Package version |
