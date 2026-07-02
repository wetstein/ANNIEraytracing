"""Batch-mode event generation for the full ANNIE detector.

Provides:
  - Muon topology sampling (fixed, from file, or random).
  - Event loop that calls ``trace_cherenkov`` and optionally runs the
    PMT digital model.
  - ``BatchAccumulator`` for fast PyArrow-batched writes to Parquet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from annieray.tracer import (
    HI, HT, HDI, HDS, HLU, HLV, H_ARRIVAL, H_WAVELEN, H_BOUNCE,
    DET_SYS_PMT, DET_SYS_LAPPD_ANNIE, DET_SYS_NONE,
    N_HIT_COLS, N_EXPANDED_COLS,
    C_MM_NS, N_WATER_DEFAULT,
    trace_rays, trace_cherenkov, compute_track_length, Geometry,
)


# ---------------------------------------------------------------------------
# Schemas for output tables
# ---------------------------------------------------------------------------

PHOTON_HIT_SCHEMA = pa.schema([
    ("event_id", pa.int64()),
    ("detector_system", pa.int32()),
    ("detector_index", pa.int32()),
    ("local_u", pa.float32()),
    ("local_v", pa.float32()),
    ("arrival_time", pa.float32()),
    ("wavelength", pa.float32()),
])

MUON_TRUTH_SCHEMA = pa.schema([
    ("event_id", pa.int64()),
    ("pos_x", pa.float32()),
    ("pos_y", pa.float32()),
    ("pos_z", pa.float32()),
    ("t0", pa.float32()),
    ("dir_x", pa.float32()),
    ("dir_y", pa.float32()),
    ("dir_z", pa.float32()),
    ("theta_deg", pa.float32()),
    ("phi_deg", pa.float32()),
    ("track_length_mm", pa.float32()),
    ("n_generated", pa.int32()),
    ("n_detected", pa.int32()),
])

PMT_RESPONSE_SCHEMA = pa.schema([
    ("event_id", pa.int64()),
    ("pmt_index", pa.int32()),
    ("charge", pa.float32()),
    ("time", pa.float32()),
    ("n_hits", pa.int32()),
])


# ---------------------------------------------------------------------------
# Muon topology sampling
# ---------------------------------------------------------------------------


def _sample_tank_position(rng: np.random.Generator, tank_radius: float,
                          tank_z_min: float, tank_z_max: float
                          ) -> tuple[float, float, float]:
    """Uniform rejection-sampled (x, y, z) inside the tank cylinder."""
    r = tank_radius * 0.9
    while True:
        x = rng.uniform(-r, r)
        y = rng.uniform(-r, r)
        if x * x + y * y <= r * r:
            break
    z = rng.uniform(tank_z_min + 100.0, tank_z_max - 100.0)
    return (float(x), float(y), float(z))


def sample_muon_state(
    event_id: int,
    config: "BatchConfig",
    rng: np.random.Generator,
    geometry: Geometry | None = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Return ``(muon_pos, muon_dir)`` for a given event.

    *muon_pos* is a 4-tuple ``(x, y, z, t0)``.
    *muon_dir* is a 3-tuple ``(dx, dy, dz)``.
    """
    if config.muon_file is not None:
        lines = _MUON_FILE_CACHE
        idx = event_id % len(lines)
        x, y, z, t0, dx, dy, dz = lines[idx]
    elif config.muon_fixed is not None:
        x, y, z, t0, dx, dy, dz = config.muon_fixed
    else:
        # Sampled topology
        if geometry is None:
            x, y, z = _sample_tank_position(rng, 1524.0, 19.0, 3861.0)
        else:
            x, y, z = _sample_tank_position(
                rng, geometry.tank_radius, geometry.tank_z_min, geometry.tank_z_max
            )
        t0 = 0.0
        mode = config.muon_mode
        if mode == "downward":
            # Downward-going with small random scatter (≈ 5 deg)
            theta = rng.uniform(0.0, np.radians(5.0))
            phi = rng.uniform(0.0, 2.0 * np.pi)
            sin_t = np.sin(theta)
            dx = float(sin_t * np.cos(phi))
            dy = float(sin_t * np.sin(phi))
            dz = -float(np.cos(theta))
        elif mode == "isotropic":
            # Uniform direction on sphere
            theta = float(np.arccos(rng.uniform(-1.0, 1.0)))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            dx = float(np.sin(theta) * np.cos(phi))
            dy = float(np.sin(theta) * np.sin(phi))
            dz = float(np.cos(theta))
        elif mode == "beam":
            # Forward along +Y with up to 15° scatter
            theta = float(rng.uniform(0.0, np.radians(15.0)))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            dx = float(np.sin(theta) * np.cos(phi))
            dy = float(np.cos(theta))
            dz = float(np.sin(theta) * np.sin(phi))
        else:
            raise ValueError(f"Unknown muon_mode: {mode}")

    pos = (x, y, z, t0)
    direc = (dx, dy, dz)
    return pos, direc


# ---------------------------------------------------------------------------
# Muon-file cache
# ---------------------------------------------------------------------------

_MUON_FILE_CACHE: list[tuple[float, float, float, float, float, float, float]] = []


def _load_muon_file(path: Path) -> list:
    """Parse the topology file and populate the global cache."""
    global _MUON_FILE_CACHE
    _MUON_FILE_CACHE = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(
                    f"{path}:{line_no + 1} — expected 7 values "
                    f"(x y z t0 dx dy dz), got {len(parts)}"
                )
            _MUON_FILE_CACHE.append(tuple(float(p) for p in parts))
    if not _MUON_FILE_CACHE:
        raise ValueError(f"{path}: no valid topology lines found")
    return _MUON_FILE_CACHE


# ---------------------------------------------------------------------------
# BatchAccumulator
# ---------------------------------------------------------------------------


@dataclass
class BatchAccumulator:
    """Writes per-event data incrementally via ``pq.ParquetWriter``.

    Each ``append_event()`` call writes a small Arrow ``RecordBatch``
    directly to disk as a new row group.  This avoids accumulating all
    rows in Python memory, which caused progressive slowdown as the
    batch run progressed.
    """

    output_dir: Path = Path("results")

    _photon_writer: Optional[pq.ParquetWriter] = None
    _pmt_writer: Optional[pq.ParquetWriter] = None
    _muon_writer: Optional[pq.ParquetWriter] = None
    _n_photon_rows: int = 0
    _n_pmt_rows: int = 0
    _n_muon_rows: int = 0

    def append_event(
        self,
        event_id: int,
        hits: np.ndarray,
        pmt_responses: Optional[dict[int, dict]] = None,
        muon_params: Optional[dict] = None,
    ) -> None:
        """Record one event's hits, PMT responses, and muon truth."""
        n_detected = self._append_photon_hits(event_id, hits)
        if pmt_responses is not None:
            self._append_pmt_responses(event_id, pmt_responses)
        if muon_params is not None:
            self._record_muon(event_id, n_detected, **muon_params)

    # ------------------------------------------------------------------
    # Photon hits
    # ------------------------------------------------------------------

    def _lazy_photon_writer(self) -> pq.ParquetWriter:
        if self._photon_writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "photon_hits.parquet"
            self._photon_writer = pq.ParquetWriter(str(path), PHOTON_HIT_SCHEMA)
        return self._photon_writer

    def _lazy_pmt_writer(self) -> pq.ParquetWriter:
        if self._pmt_writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "pmt_responses.parquet"
            self._pmt_writer = pq.ParquetWriter(str(path), PMT_RESPONSE_SCHEMA)
        return self._pmt_writer

    def _lazy_muon_writer(self) -> pq.ParquetWriter:
        if self._muon_writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "muon_truth.parquet"
            self._muon_writer = pq.ParquetWriter(str(path), MUON_TRUTH_SCHEMA)
        return self._muon_writer

    def _append_photon_hits(self, event_id: int, hits: np.ndarray) -> int:
        """Extract per-detector hit 4-vectors and write a row group.

        Returns the number of detector hits (n_detected).
        """
        det_mask = (
            (np.abs(hits[:, HDS] - DET_SYS_PMT) < 0.5)
            | (np.abs(hits[:, HDS] - DET_SYS_LAPPD_ANNIE) < 0.5)
        )
        n_detected = int(det_mask.sum())
        if n_detected == 0:
            return 0

        sel = hits[det_mask]
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(np.full(n_detected, event_id, dtype=np.int64)),
                pa.array(sel[:, HDS].astype(np.int32)),
                pa.array(sel[:, HDI].astype(np.int32)),
                pa.array(sel[:, HLU]),
                pa.array(sel[:, HLV]),
                pa.array(sel[:, H_ARRIVAL]),
                pa.array(sel[:, H_WAVELEN]),
            ],
            schema=PHOTON_HIT_SCHEMA,
        )
        self._lazy_photon_writer().write_batch(batch)
        self._n_photon_rows += n_detected
        return n_detected

    def _append_pmt_responses(
        self, event_id: int, responses: dict[int, dict]
    ) -> None:
        if not responses:
            return

        idx = sorted(responses.keys())
        n = len(idx)

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(np.full(n, event_id, dtype=np.int64)),
                pa.array(np.array(idx, dtype=np.int32)),
                pa.array(np.array([responses[i]["charge"] for i in idx], dtype=np.float32)),
                pa.array(np.array([responses[i]["time"] for i in idx], dtype=np.float32)),
                pa.array(np.array([responses[i]["n_hits"] for i in idx], dtype=np.int32)),
            ],
            schema=PMT_RESPONSE_SCHEMA,
        )
        self._lazy_pmt_writer().write_batch(batch)
        self._n_pmt_rows += n

    def _record_muon(
        self,
        event_id: int,
        n_detected: int,
        pos: tuple,
        direc: tuple,
        track_length: float,
        n_generated: int,
    ) -> None:
        """Record one muon truth row."""
        x, y, z, t0 = pos
        dx, dy, dz = direc
        theta_deg = float(np.degrees(np.arccos(-dz)))
        phi_deg = float(np.degrees(np.arctan2(dy, dx)))

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([event_id], pa.int64()),
                pa.array([x], pa.float32()),
                pa.array([y], pa.float32()),
                pa.array([z], pa.float32()),
                pa.array([t0], pa.float32()),
                pa.array([dx], pa.float32()),
                pa.array([dy], pa.float32()),
                pa.array([dz], pa.float32()),
                pa.array([theta_deg], pa.float32()),
                pa.array([phi_deg], pa.float32()),
                pa.array([track_length], pa.float32()),
                pa.array([n_generated], pa.int32()),
                pa.array([n_detected], pa.int32()),
            ],
            schema=MUON_TRUTH_SCHEMA,
        )
        self._lazy_muon_writer().write_batch(batch)
        self._n_muon_rows += 1

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all Parquet writers (flushes data to disk)."""
        if self._photon_writer is not None:
            self._photon_writer.close()
            self._photon_writer = None
        if self._pmt_writer is not None:
            self._pmt_writer.close()
            self._pmt_writer = None
        if self._muon_writer is not None:
            self._muon_writer.close()
            self._muon_writer = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """All parameters for a batch run."""

    # Events
    n_events: int = 100
    muon_fixed: Optional[tuple[float, float, float, float, float, float, float]] = None
    muon_file: Optional[Path] = None
    muon_mode: str = "isotropic"  # "downward", "isotropic", or "beam"

    # Photon generation
    photons_per_cm: int = 150
    wavelength_nm: float = 350.0
    max_bounces: int = 0
    batch_size: int = 50

    # Response models
    pmt_response: bool = False
    pmt_full_wf: bool = False

    # I / O
    output_dir: Path = Path("results")
    record_events: bool = True

    # Reproducibility
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_batch(
    geometry: Geometry,
    config: BatchConfig,
    optics_config: Optional[dict] = None,
) -> dict[str, Path]:
    """Run N events and return paths to output files.

    Parameters
    ----------
    geometry : Geometry
        Pre-built detector geometry.
    config : BatchConfig
    optics_config : dict or None
        Per-material optical properties (for multi-bounce mode).

    Returns
    -------
    dict[str, Path]
        ``{"photon_hits": ..., "pmt_responses": ...}`` — only keys that
        were actually written.
    """
    from annieray.cherenkov import generate_cherenkov_photons
    from annieray.pmt_response import process_pmt_hits

    rng = np.random.default_rng(config.seed)
    accumulator = BatchAccumulator(output_dir=config.output_dir)

    if config.muon_file is not None:
        _load_muon_file(config.muon_file)

    batch_size = max(1, config.batch_size)
    n_water = N_WATER_DEFAULT
    c_in_water = C_MM_NS / n_water
    wave_nm = float(config.wavelength_nm)

    t_start = time.time()
    processed = 0

    while processed < config.n_events:
        batch_end = min(processed + batch_size, config.n_events)
        n_batch = batch_end - processed

        # --- Collect photons for all events in this batch ---
        batch_origins: list[np.ndarray] = []
        batch_dirs: list[np.ndarray] = []
        batch_ctimes: list[np.ndarray] = []
        event_counts: list[int] = []
        event_muon: list[tuple] = []
        event_track_length: list[float] = []
        event_n_generated: list[int] = []

        for i in range(n_batch):
            event_id = processed + i
            muon_pos, muon_dir = sample_muon_state(event_id, config, rng, geometry)
            track_length = compute_track_length(muon_pos, muon_dir, geometry)
            o, d, ct = generate_cherenkov_photons(
                muon_pos, muon_dir, config.photons_per_cm,
                track_length=track_length, rng=rng,
            )
            batch_origins.append(o)
            batch_dirs.append(d)
            batch_ctimes.append(ct)
            event_counts.append(len(o))
            event_muon.append((muon_pos, muon_dir))
            event_track_length.append(track_length)
            event_n_generated.append(len(o))

        # Concatenate into one big array
        all_origins = np.vstack(batch_origins)
        all_dirs = np.vstack(batch_dirs)
        all_ctimes = np.concatenate(batch_ctimes)

        # --- One GPU launch for the whole batch ---
        if config.max_bounces > 0:
            from annieray.tracer import trace_with_optics
            from annieray.optics import load_optics_config
            cfg = optics_config if optics_config is not None else load_optics_config(None)
            hits, bounce_counts, orig_indices = trace_with_optics(
                all_origins, all_dirs, geometry, cfg,
                max_bounces=config.max_bounces, n_water=n_water, rng=rng,
            )
        else:
            hits = trace_rays(all_origins, all_dirs, geometry)
            bounce_counts = np.zeros(hits.shape[0], dtype=np.int32)
            orig_indices = np.arange(hits.shape[0], dtype=np.int32)

        # --- Expand to N_EXPANDED_COLS across all photons ---
        n_total = hits.shape[0]
        full = np.zeros((n_total, N_EXPANDED_COLS), dtype=np.float32)
        full[:, :N_HIT_COLS] = hits
        full[:, H_WAVELEN] = wave_nm
        full[:, H_BOUNCE] = bounce_counts

        hit_mask = hits[:, HI] > 0.5
        if hit_mask.any():
            full[hit_mask, H_ARRIVAL] = (
                all_ctimes[orig_indices[hit_mask]]
                + hits[hit_mask, HT] / c_in_water
            )

        # --- Slice back per event ---
        offsets = np.empty(n_batch + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(event_counts, out=offsets[1:])

        for i in range(n_batch):
            event_id = processed + i
            sl = slice(offsets[i], offsets[i + 1])
            event_hits = full[sl]
            muon_pos, muon_dir = event_muon[i]

            pmt_responses = None
            if config.pmt_response:
                pmt_responses = process_pmt_hits(
                    event_hits, geometry, rng=rng, full_wf=config.pmt_full_wf,
                )

            if config.record_events:
                muon_params = {
                    "pos": muon_pos,
                    "direc": muon_dir,
                    "track_length": event_track_length[i],
                    "n_generated": event_n_generated[i],
                }
                accumulator.append_event(
                    event_id, event_hits, pmt_responses,
                    muon_params=muon_params,
                )

        processed += n_batch

        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{processed}/{config.n_events}] "
            f"{elapsed:.1f}s elapsed, {rate:.1f} ev/s"
        )

    elapsed = time.time() - t_start
    print(f"  Total: {elapsed:.1f}s for {config.n_events} events "
          f"({config.n_events / elapsed:.1f} ev/s)")

    if config.record_events:
        try:
            accumulator.close()
        finally:
            paths = {}
            if accumulator._n_photon_rows > 0:
                p = config.output_dir / "photon_hits.parquet"
                paths["photon_hits"] = p
                print(f"  Wrote {p} ({accumulator._n_photon_rows} photon rows)")
            if accumulator._n_pmt_rows > 0:
                p = config.output_dir / "pmt_responses.parquet"
                paths["pmt_responses"] = p
                print(f"  Wrote {p} ({accumulator._n_pmt_rows} PMT response rows)")
            if accumulator._n_muon_rows > 0:
                p = config.output_dir / "muon_truth.parquet"
                paths["muon_truth"] = p
                print(f"  Wrote {p} ({accumulator._n_muon_rows} muon truth rows)")
            return paths

    return {}
