# Batch Output Format

The `annieray batch` command writes up to three Parquet files and
two sidecar files into the output directory.

## Files

| File | Always written? | Description |
|------|:-:|-------------|
| `photon_hits.parquet` | yes | Per-photon detector hit records |
| `muon_truth.parquet` | yes | Per-event muon topology and efficiency |
| `pmt_responses.parquet` | only with `--pmt-response` | Per-PMT digital-model output |
| `detectors.csv` | yes | Detector registry (for display / analysis) |
| `metadata.json` | yes | Tank dimensions |

---

## `photon_hits.parquet`

One row per photon that reaches a sensitive detector (PMT or LAPPD).
Photons that miss all detectors or hit only structural elements are
**not** recorded.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number (0-based) |
| `detector_system` | int32 | 0 = PMT, 1 = LAPPD (default), 2 = LAPPD (ANNIE housing) |
| `detector_index` | int32 | **Global** detector index within the system (see *Indexing* below) |
| `local_u` | float32 | Along-strip position (LAPPD, mm) / unused for PMT |
| `local_v` | float32 | Across-strip position (LAPPD, mm) / unused for PMT |
| `arrival_time` | float32 | Photon time-of-flight from emission point (ns) |
| `wavelength` | float32 | Photon wavelength (nm) |

### Indexing

`detector_index` is a **global offset** that matches the `detector_index`
column in `detectors.csv`.  The kernel assigns indices as:

| Detector type | Index range |
|---------------|-------------|
| PMT | 0 … `n_pmts` − 1 |
| LAPPD (default) | `n_pmts` … `n_pmts` + `n_lappds` − 1 |
| LAPPD (ANNIE) | `n_pmts` + `n_lappds` … `n_pmts` + `n_lappds` + `n_housings` − 1 |

For a typical 132-PMT configuration with 3 surfboard-mounted ANNIE
LAPPD housings: PMTs are 0–131, LAPPDs are 132–134.

Join to `detectors.csv` on `(system_code, detector_index)`:

```python
hits = pd.read_parquet("photon_hits.parquet")
det  = pd.read_csv("detectors.csv")
merged = hits.merge(
    det,
    left_on=["detector_system", "detector_index"],
    right_on=["system_code", "detector_index"],
)
```

---

## `muon_truth.parquet`

One row per event.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `pos_x` | float32 | Muon start X (mm) |
| `pos_y` | float32 | Muon start Y (mm) |
| `pos_z` | float32 | Muon start Z (mm) |
| `t0` | float32 | Muon start time (ns) |
| `dir_x` | float32 | Muon direction unit vector X |
| `dir_y` | float32 | Muon direction unit vector Y |
| `dir_z` | float32 | Muon direction unit vector Z |
| `theta_deg` | float32 | Polar angle from vertical (°); 0 = upward (+Y), 180 = downward |
| `phi_deg` | float32 | Azimuthal angle in XY (°); arctan2(y, x) |
| `track_length_mm` | float32 | Track chord inside the tank (mm) |
| `n_generated` | int32 | Number of Cherenkov photons generated |
| `n_detected` | int32 | Number of photons detected by PMTs / LAPPDs |

### Direction conventions

- `theta_deg` is measured from the +Y axis (vertical).
    - θ = 0° → upward, θ = 90° → horizontal, θ = 180° → downward.
- `phi_deg` follows the standard `arctan2(y, x)` convention.
    - φ = 0° → +X, φ = 90° → +Y.

### Typical usage

```python
muons = pd.read_parquet("muon_truth.parquet")

# Per-event detection efficiency
muons["efficiency"] = muons["n_detected"] / muons["n_generated"].clip(1)

# Merge with per-event hit summaries
per_event = hits.groupby("event_id").size().rename("n_photons")
merged = muons.merge(per_event, on="event_id")
```

---

## `pmt_responses.parquet`

Only written when `--pmt-response` is passed.  One row per PMT that
registered at least one hit.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `pmt_index` | int32 | PMT index (0 … n_pmts−1) |
| `charge` | float32 | Integrated charge (pC) |
| `time` | float32 | Hit time (ns) |
| `n_hits` | int32 | Number of photon hits folded into this response |

---

## Sidecar files

### `detectors.csv`

| Column | Description |
|--------|-------------|
| `system_code` | 0=PMT, 1=lappd_default, 2=lappd_annie |
| `detector_index` | Global index (matches `photon_hits.parquet`) |
| `x, y, z` | Detector position (mm) |
| `label` | Human-readable name (e.g. `PMT_332`, `LAPPD_ANNIE_0`) |
| `panel` | PMT panel number (0=bottom, 9=top, 1–8=barrel); −1 for LAPPDs |

### `metadata.json`

```json
{
  "tank_radius_mm": 1524.0,
  "tank_z_min_mm": 19.0,
  "tank_z_max_mm": 3861.0
}
```

---

## Cross-file joins

The simplest way to assemble a complete analysis DataFrame:

```python
import pandas as pd

hits  = pd.read_parquet("results/photon_hits.parquet")
muons = pd.read_parquet("results/muon_truth.parquet")
det   = pd.read_csv("results/detectors.csv")

# Per-event hit counts
per_event = hits.groupby("event_id").size().rename("n_photons")

# Per-event per-detector hit counts
by_detector = hits.groupby(
    ["event_id", "detector_system", "detector_index"],
    as_index=False,
).agg(
    n_hits=("detector_index", "count"),
    first_arrival=("arrival_time", "min"),
)

# Attach detector positions
by_detector = by_detector.merge(
    det,
    left_on=["detector_system", "detector_index"],
    right_on=["system_code", "detector_index"],
)

# Attach muon parameters
full = by_detector.merge(muons, on="event_id")
```

---

## System codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `pmt` | 10-inch PMT (LUX / ETEL) |
| 1 | `lappd_default` | Rectangular LAPPD (original model) |
| 2 | `lappd_annie` | LAPPD in waterproof Kandemir housing |
