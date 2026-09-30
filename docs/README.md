# ANNIEraytracing

GPU-accelerated ray tracer for the ANNIE detector, implemented in Taichi.
Traces photons through a GDML-detailed inner structure, PMTs, LAPPDs, and
obscurant surfboard panels, recording hit positions, local coordinates,
arrival times, and wavelengths.

Includes a **gridded likelihood-search framework**: batch events on the
GPU, then scan a grid of muon hypotheses (direction, and optionally
vertex position) and score each against the observed event using a
Poisson charge likelihood + optional Gaussian time residual likelihood,
with on-the-fly raytracing per hypothesis.

## Quick Start

```bash
# Batch-mode simulation (Cherenkov)
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 150

# Batch-mode simulation with wbLS scintillation (adds to Cherenkov)
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 150 \
    --scintillation --photons-per-cm-scint 100

# Scintillation-only batch (Cherenkov off)
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 0 \
    --scintillation --photons-per-cm-scint 100

# Direction fit for a single event (grid scan, runs raytracing per hypothesis)
python -m annieray fit output.h5 --event 0 --show

# Zoomed-in fit around the true direction
python -m annieray fit output.h5 --event 0 \
    --theta-window 10 --grid-steps 21 --show

# Interactive 3D viewer (scintillation toggle + parameters in Trace panel)
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt --port 8080
```

## CLI Commands

### `batch` — Batch-mode event generation

Runs N events on the GPU, writing results to a single HDF5 file. No
display server required.

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
| `--muon-file` | None | File with one topology per line |
| `--surfboard` | 0 | PVC surfboard panels (`0`, `1`, or `3`) This flag gets the LAPPD response if you use "3" on the flag| 
| `--lappd-model` | `annie` | LAPPD geometry (`default` / `annie`) |
| `--lappd-indices` | None | Comma-separated LAPPD candidate indices from STEP |
| `--det-rotation` | 22.5 | Global Z-rotation (deg) |
| `--z-offset` | 0.0 | Vertical offset (mm) |
| `--no-lappd` | false | Skip LAPPD rectangles |
| `--no-gdml` | false | Skip inner structure GDML mesh (tank + PMTs only) |
| `--no-pmt-holders` | false | Skip PMT body and hardware holder meshes (PMT positions still loaded) |
| `--max-bounces` | 0 | Multi-bounce optics |
| `--pmt-response` | false | Enable PMT digital model (SPE charge + TTS) |
| `--full-wf` | false | Full waveform path (requires `--pmt-response`) |
| `--light-burst` | false | Isotropic light burst (instead of muon Cherenkov) |
| `--burst-n-phots` | 1000 | Number of isotropic photons per burst |
| `--burst-position` | None | Burst centre `"x y z"` (mm) |
| `--output-dir` | `results/` | Output directory for HDF5 file |
| `--no-record` | false | Skip writing per-event output |
| `--wavelength` | 350.0 | Cherenkov photon wavelength (nm) |
| `--scintillation` | false | Enable wbLS scintillation photon generation (adds to Cherenkov) |
| `--photons-per-cm-scint` | 100 | Scintillation photons per cm of track |
| `--wavelength-scint` | 420.0 | Scintillation photon wavelength (nm) |
| `--tau-fast` | 2.0 | Fast scintillation decay time (ns) |
| `--tau-slow` | 20.0 | Slow scintillation decay time (ns) |
| `--fast-fraction` | 0.95 | Fraction of scintillation light in the fast component |
| `--optics-config` | None | YAML file with per-material optical properties |
| `--water-absorption-mm` | 0.0 | Water absorption length in mm (0 = off) |
| `--water-scattering-mm` | 0.0 | Rayleigh scattering length in mm (0 = off) |
| `--seed` | None | Random seed |

**Output** — A single `output.h5` containing tables:

| Table | Contents |
|-------|----------|
| `photon_hits` | Per-photon hit records, incl. `photon_source` (0=Cherenkov, 1=scintillation) |
| `pmt_responses` | Digitised charges and hit times (when `--pmt-response`) |
| `muon_truth` | Per-event truth (position, direction, track length, etc.) with per-source generated/detected counts |
| `detectors` | Detector registry (system, index, position, label, panel) |
| `metadata` | Tank dimensions and geometry parameters |

The `photon_hits.arrival_time` column already includes the scintillation
decay delay for scintillation photons, so Cherenkov and scintillation
timing can be studied separately via `photon_source` without any
additional book-keeping.

### `fit` — Direction fitting from batch output

Runs an on-the-fly direction grid scan for a single event. Each hypothesis
calls `trace_cherenkov()` independently to predict per-PMT hit counts,
then evaluates the Poisson charge likelihood (+ optional Gaussian time
residual likelihood).

```
python -m annieray fit output.h5 [--event 0] [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gdml` | `PHASE2_INNER_STRUCTURE_closed.gdml` | Geometry mesh |
| `--pmt-csv` | No default | PMT position file |
| `--event` | 0 | Event ID to fit |
| `--grid-theta` | `"0 180 19"` | θ grid: `"start stop steps"` (deg) |
| `--grid-phi` | `"0 360 37"` | φ grid: `"start stop steps"` (deg) |
| `--theta-window` | None | ±half-width around true direction; overrides `--grid-theta` |
| `--phi-window` | None | φ half-window; defaults to `--theta-window` |
| `--grid-steps` | 41 | Steps per axis in windowed mode |
| `--fix-vertex` | None | Fixed vertex `"x y z"` (mm) |
| `--fix-t0` | None | Fixed t0 (ns) |
| `--photons-per-cm` | auto-detect | Photons per cm for hypothesis evaluation |
| `--use-time` | false | Include time residual likelihood |
| `--time-sigma` | None | Per-PMT-type or global timing sigma (ns) |
| `--alpha` | 1.0 | Scale factor for time likelihood |
| `--seed` | 42 | RNG seed |
| `--save-grid` | None | Save likelihood surface to NPZ |
| `--show` | false | Show plot (auto: rectangular for zoom, polar for full-sky) |
| `--polar` | false | Force polar projection |
| `--no-gdml` | false | Skip inner structure GDML mesh |
| `--no-pmt-holders` | false | Skip PMT body and hardware holder meshes |

**Plot type auto-detection:**
- Full-sky grid (θ span > 170°, φ span > 350°) → polar
- Zoomed-in grid → rectangular (Δθ × Δφ heatmap)

## Testing

```bash
pytest                         # All fast tests (slow ones deselected)
pytest -m slow                 # Slow end-to-end tests (geometry + GPU kernel)
pytest tests/test_scintillation.py  # Scintillation / timing unit tests
pytest tests/test_timing_e2e.py -m slow  # End-to-end arrival-time sanity test
```

The default `pytest` run deselects the `slow` marker (configured via
`addopts` in `pyproject.toml`); run them explicitly with `-m slow`.

**Timing tests.** The final photon `arrival_time` is always computed as
`create_time + path_length / c_in_water`, where `create_time` already
includes the muon's arrival at the emission point (Cherenkov) and, for
scintillation, an additional sampled decay delay. Tests verify this:
`tests/test_scintillation.py` checks the created-time offsets directly, and
`tests/test_timing_e2e.py` traces the same muon twice with identical origins
but different `tau_fast`, confirming the reported `arrival_time` shifts by
exactly the decay-constant difference.

> Note: the generators store the muon start time `t0` in **seconds** (they
> internally convert via `*10**9` to ns), so pass `t0` as seconds, not ns.

### Gridded likelihood search workflow

The fitting framework searches over muon hypotheses and scores each one
against the observed data. Two dimensions can be scanned independently:

**Direction scan** (`fit`): fix the vertex position, scan θ × φ. The
likelihood surface is an `(n_θ, n_φ)` array of log-likelihoods
(`ScanResult.scores`).

**Position scan** (`scripts/scan_ll_vs_position*.py`): fix the direction,
scan (x, z) or (x, y) vertex positions. A fast full-sky direction fit can
be run at *each* position (`scan_ll_vs_position.py`), or a single fixed
direction traced once per position (`scan_ll_vs_position_fixed_dir.py`,
~0.5 s/position).

Typical end-to-end workflow:

```bash
# 1. Simulate events
python -m annieray batch --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 150 --pmt-response

# 2. Coarse full-sky direction fit for one event
python -m annieray fit results/output.h5 --event 0 --show

# 3. Zoom in around the best direction for precision
python -m annieray fit results/output.h5 --event 0 \
    --theta-window 10 --phi-window 10 --grid-steps 41 \
    --save-grid grid.npz --show

# 4. Sweep the vertex position with a fixed direction (fast)
python scripts/scan_ll_vs_position_fixed_dir.py results/output.h5 \
    --grid-x "-1200 1200 13" --grid-z "300 2700 13" --show

# 5. Replot a saved likelihood surface
python scripts/fit_viewer.py grid.npz --clip 10 --polar
```

The search reuses a single pre-built `Geometry` and calls
`trace_cherenkov()` once per hypothesis in `annieray/fitting.py`.
The truth direction from `muon_truth` is stored in the `ScanResult` so
that fit residuals can be computed for validation.

### Other commands

- **`viz-server`** — Interactive 3D viewer with Three.js frontend
- **`viz-lappd`** — Standalone LAPPD module viewer
- **`build-detector-config`** — Writes detector registry YAML

## Analysis scripts

Standalone plotting and scanning utilities in `scripts/`.

### `fit_viewer.py`

Reload and replot a saved likelihood surface from an NPZ file.

```bash
python scripts/fit_viewer.py grid.npz
python scripts/fit_viewer.py grid.npz --clip 10 --polar
python scripts/fit_viewer.py grid.npz --save figure.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--clip` | 20 | ΔLL clipping window |
| `--polar` | false | Force polar projection |
| `--rect` | false | Force rectangular projection |
| `--save` | None | Save figure to file |

### `scan_ll_vs_position.py`

Direction grid scan at each (x,z) spatial position. Produces 3-panel
heatmaps: best LL, best θ, best φ vs position.

```bash
python scripts/scan_ll_vs_position.py output.h5 --event 0 \
    --grid-x "1500 2500 11" --grid-z "1500 2500 11" \
    --grid-theta "0 180 9" --grid-phi "0 360 9" \
    --save scan_results.npz --show
```

### `scan_ll_vs_position_fixed_dir.py`

Fixed-direction scan over (x,z) or (x,y) positions at ~0.5 s/position.
Single raytracing call per position — much faster than running a full
direction scan at each point.

```bash
# XZ scan
python scripts/scan_ll_vs_position_fixed_dir.py output.h5 \
    --grid-x "-500 500 11" --grid-z "1500 2500 11" --show

# XY scan
python scripts/scan_ll_vs_position_fixed_dir.py output.h5 \
    --grid-x "-500 500 11" --grid-y "-500 500 11" --fix-z 2000 --show
```

| Flag | Default | Description |
|------|---------|-------------|
| `--grid-x` | `"0 2000 11"` | X range: `"start stop steps"` |
| `--grid-y` | None | Y range (use with `--fix-z`) |
| `--grid-z` | None | Z range (use with `--fix-y`) |
| `--fix-y` | None | Fixed Y coordinate |
| `--fix-z` | None | Fixed Z coordinate |
| `--fix-theta` | auto-detect | Fixed θ from muon_truth |
| `--fix-phi` | auto-detect | Fixed φ from muon_truth |
| `--clip` | auto | Colormap range (5th–95th percentile by default) |
| `--save` | None | Save results to NPZ |
| `--show` | false | Show heatmap |

### `generate_muon_grid.py`

Generate a `MuonStartsAndDirecs`-format file with a configurable angular
scan: for each (x,z) position in the standard 13×13 grid, creates n×n
muon directions evenly spaced from −half_range to +half_range in both the
vertical and horizontal planes. With `--n-steps 1` the direction is
always (0, 1, 0).

```bash
python scripts/generate_muon_grid.py > MuonStartsAndDirecs_angled.txt
python scripts/generate_muon_grid.py --n-steps 5 --half-range 45 > MuonStartsAndDirecs_5x5.txt
python scripts/generate_muon_grid.py --n-steps 1 > MuonStartsAndDirecs_1x1.txt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-steps` | 3 | Number of angle steps per axis |
| `--half-range` | 22.5 | Half-range of angles in degrees |

The output feeds `annieray batch --muon-file MuonStartsAndDirecs_angled.txt`.

### `event_display.py`

Interactive three-panel event display for batch output: top endcap PMTs,
unrolled barrel (φ vs Z, φ centered on the median LAPPD so surfboard-
mounted LAPPDs appear mid-plot), and bottom endcap PMTs. Navigate events
with ◀/▶ buttons or arrow keys.

```bash
python scripts/event_display.py results/
python scripts/event_display.py results/output.h5
```

### `load_batch.py`

Template script showing how to load HDF5 output and group hits by event
and detector: per-event hit counts, per-detector aggregates, muon truth
joins, and PMT response summaries.

```bash
python scripts/load_batch.py results/output.h5
```

## Likelihood model

### Poisson charge likelihood

Uses the per-PMT hit count (`n_hits` from `pmt_responses`), an integer
representing the number of detected photoelectrons.  The log-likelihood is:

$$LL_{\text{charge}} = \sum_i \left[ n_i \ln(\mu_i) - \mu_i - \ln(n_i!) \right]$$

where `μ_i` is the expected number of hits from the hypothesis (computed
by tracing Cherenkov photons) and `n_i` is the observed count.  PMTs with
no observed hits contribute `-μ_i`.  Implemented in `annieray/likelihood.py`.

### Gaussian time residual likelihood

When `--use-time` is set, the earliest photon arrival time at each PMT is
compared to the expected time via:

$$LL_{\text{time}} = -\frac{1}{2} \sum_i \frac{(t_i - t_{\text{exp},i})^2}{\sigma_i^2}$$

with per-PMT-type transit time sigma from `PMT_TYPE_DEFAULTS` (ETEL=1.8,
LUX=1.2, Hamamatsu=1.5, Watchboy=1.6, Watchman=1.0 ns).  The combined
score is `LL = LL_charge + α · LL_time`.

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
              trace_rays()            trace_muon_light()
                                    (Cherenkov + optional scintillation)
                    |                             |
                    └─────────────┬───────────────┘
                                  v
                          trace_kernel()
                     (Taichi GPU kernel)
                                  |
                     ┌────────────┴────────────┐
                     v                         v
                hits ndarray            expand with arrival_time
                     |                  + wavelength + bounce
                     v
               write_hits() / BatchAccumulator
                     |
                     v
               output.h5
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
7. **Scans surfboard panels** via oriented-box intersection.
8. **Finds the tank wall** via infinite-cylinder intersection.
9. **Writes the nearest hit** with component ID, detector index/system,
   local coordinates, and material ID.

## Muon direction conventions

- **θ (theta)** — polar angle from vertical. 0° = downward (−z),
  90° = horizontal (XY plane), 180° = upward (+z).
- **φ (phi)** — azimuthal angle in XY plane. 0° = along +x,
  90° = along +y.

## Surfboard Obscurant Panels

PVC panels (2450 x 280 x 10 mm) mounted vertically at the forward
octagon vertices (45°, 90°, 135°). Configurable via `--surfboard {0,1,3}`.

## LAPPD Housing Model

The `--lappd-model annie` flag replaces the default bare photocathode
rectangle with the full Kandemir waterproof housing: a 5-sided acrylic
box (330 x 430 x 60 mm) with an off-centre photocathode (191.5 x 191.5 mm).

## Scintillation (wbLS)

With `--scintillation`, the batch mode and viz-server additionally emit
scintillation photons along the muon track. Unlike Cherenkov light
(a cone around the muon direction), scintillation is emitted
**isotropically** and its emission time is **delayed** by an exponential
decay sampled from a mixture of fast and slow components:

- fast decay `--tau-fast` (default 2.0 ns) — ~95% of light (`--fast-fraction`)
- slow decay `--tau-slow` (default 20.0 ns) — the remainder

These defaults match the companion ANNIE wbLS characterisation
(Caravaca et al., arXiv:2006.00173). Photons continue to be emitted along
the track at a rate set by `--photons-per-cm-scint`.

Both photon types are concatenated and traced in a **single GPU call**, and
tagged per-photon in the expanded hit array column `H_SOURCE`
(`SOURCE_CKV = 0`, `SOURCE_SCI = 1`). This propagates to the HDF5
`photon_hits.photon_source` column and the `muon_truth` per-source generated
and detected counts, so Cherenkov/scintillation hits can be separated in
analysis without re-tracing. The LAPPD response pipeline (`process_hits`)
processes both photon types — it filters only on `detector_system`, not on
`H_SOURCE`, so scintillation photons are digitised alongside Cherenkov
photons with their own wavelength (420 nm default) and decay-delayed arrival
time.  The combined counts appear in the readout's `n_photons` /
`n_passed_qe` but are not split per-source.

In the viz-server, the two sources are drawn
in distinct colours (cyan = Cherenkov, yellow = scintillation) and can be
shown/hidden independently.

### Viewer rendering semantics

Hit markers in the viewer obey physical occlusion: they render on top of
the surface they sit on, but are hidden when a solid surface lies between
them and the camera. The surfboard obscurant panels are drawn **opaque**
(matching the real PVC), so hits placed behind a board are properly
obscured by it from the appropriate viewing angles. Markers use
`depthTest: true` with a high `renderOrder`, and the surfaces they land on
(the photocathode planes, structure mesh, boards, PMT inner layers) carry a
small `polygonOffset` so a marker sitting on a face is never z-fought away
by its own surface.

## Multi-bounce optics & water attenuation

With `--max-bounces N` (N > 0), `trace_with_optics()` manages N rounds
of Fresnel reflection/transmission and diffuse reflection per material.

Water attenuation is enabled with `--water-absorption-mm` and
`--water-scattering-mm` (via the `WaterAttenuation` config in
`annieray/optics.py`). When either is active, `trace_with_optics()`
applies exponential absorption and Rayleigh scattering along the photon
path. Both are off by default (`0.0` = disabled).

## Code Map

| File | Role |
|------|------|
| `cli.py` | CLI parser, all subcommands (`batch`, `fit`, `viz-server`, etc.) |
| `tracer.py` | Geometry dataclass, `build_geometry()`, Taichi trace kernel, `trace_muon_light()`/`trace_cherenkov()` |
| `batch.py` | Batch-mode event loop, `BatchAccumulator` for HDF5 output |
| `cherenkov.py` | Vectorised Cherenkov / isotropic photon generator |
| `scintillation.py` | Isotropic, delayed wbLS scintillation photon generator |
| `fitting.py` | `grid_scan_direction()`, `load_observed_event()`, `ScanResult`/`ObservedEvent` |
| `likelihood.py` | Poisson charge + Gaussian time residual log-likelihood |
| `io_h5.py` | HDF5 table I/O (load/save/append) |
| `pmt_response.py` | PMT digital model — fast path and full-waveform path |
| `lappd_response.py` | Taichi-accelerated LAPPD digitisation pipeline |
| `lappd_model.py` | `LAPPDHousing` dataclass, housing geometry builder |
| `detectors.py` | `DetectorInfo`, `build_detector_registry()`, YAML I/O |
| `output.py` | HDF5 writer for hit data and detector config |
| `gdml_parser.py` | GDML mesh parser (vertex/triangle arrays) |
| `pmt_loader.py` | PMT CSV parser, mesh loader, hardware mesh builder |
| `pmt_mesh.py` | PMT body/hardware mesh loading and array building |
| `step_parser.py` | STEP CAD parser (component manifest) |
| `viz_server.py` | Interactive Three.js viewer |
| `viz_lappd_server.py` | Standalone LAPPD module viewer |
| `optics.py` | Optical material database, Fresnel/reflectance evaluation |
| `_version.py` | Package version |
| **Scripts** | |
| `scripts/fit_viewer.py` | Polar/rectangular likelihood surface plot from NPZ |
| `scripts/scan_ll_vs_position.py` | Direction scan at each (x,z) spatial position |
| `scripts/scan_ll_vs_position_fixed_dir.py` | Fixed-direction scan over (x,z) or (x,y) |
| `scripts/convert_to_h5.py` | Convert old parquet output directories to HDF5 |
| `scripts/LAPPD_positionscan.py` | LAPPD hit heatmap from batch output |
| `scripts/LAPPD_positionscanH5.py` | LAPPD hit heatmap from HDF5 batch output |
| `scripts/pmt_histograms.py` | Per-PMT charge/time histograms |
| `scripts/event_display.py` | Interactive 3-panel event display |
| `scripts/generate_muon_grid.py` | Generate `MuonStartsAndDirecs` with angular scan |
| `scripts/load_batch.py` | Template for loading and grouping HDF5 output |
