"""Batch-mode event generation for the full ANNIE detector.

Provides:
  - Muon topology sampling (fixed, from file, or random).
  - Event loop that calls ``trace_cherenkov`` and optionally runs the
    PMT digital model.
  - ``BatchAccumulator`` for incremental row-group writes to HDF5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from annieray.io_h5 import (
    PHOTON_HIT_DTYPE, MUON_TRUTH_DTYPE, PMT_RESPONSE_DTYPE,
    append_table,
)
from annieray.tracer import (
    HI, HT, HDI, HDS, HLU, HLV, H_ARRIVAL, H_WAVELEN, H_BOUNCE,
    DET_SYS_PMT, DET_SYS_LAPPD_ANNIE, DET_SYS_NONE,
    N_HIT_COLS, N_EXPANDED_COLS,
    C_MM_NS, N_WATER_DEFAULT,
    trace_rays, trace_cherenkov, compute_track_length, Geometry,
)


# ---------------------------------------------------------------------------
# Schemas for output tables  (numpy dtypes → HDF5 datasets)
# ---------------------------------------------------------------------------

# Defined in io_h5.py:
#   PHOTON_HIT_DTYPE, MUON_TRUTH_DTYPE, PMT_RESPONSE_DTYPE

H5_OUTPUT_NAME = "output.h5"


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
    """Writes per-event data incrementally to a single HDF5 file.

    Each ``append_event()`` call appends rows to resizable HDF5 datasets.
    This avoids accumulating all rows in Python memory.
    """

    output_dir: Path = Path("results")

    _h5_file: Optional[h5py.File] = None
    _n_photon_rows: int = 0
    _n_pmt_rows: int = 0
    _n_muon_rows: int = 0

    @property
    def h5_path(self) -> Path:
        return self.output_dir / H5_OUTPUT_NAME

    def _ensure_file(self) -> h5py.File:
        if self._h5_file is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._h5_file = h5py.File(str(self.h5_path), "w")
        return self._h5_file

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

    def _append_photon_hits(self, event_id: int, hits: np.ndarray) -> int:
        """Extract per-detector hit 4-vectors and append to HDF5.

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
        arr = np.zeros(n_detected, dtype=PHOTON_HIT_DTYPE)
        arr["event_id"] = event_id
        arr["detector_system"] = sel[:, HDS].astype(np.int32)
        arr["detector_index"] = sel[:, HDI].astype(np.int32)
        arr["local_u"] = sel[:, HLU]
        arr["local_v"] = sel[:, HLV]
        arr["arrival_time"] = sel[:, H_ARRIVAL]
        arr["wavelength"] = sel[:, H_WAVELEN]

        f = self._ensure_file()
        append_table(f, "photon_hits", arr)
        self._n_photon_rows += n_detected
        return n_detected

    def _append_pmt_responses(
        self, event_id: int, responses: dict[int, dict]
    ) -> None:
        if not responses:
            return

        idx = sorted(responses.keys())
        n = len(idx)

        arr = np.zeros(n, dtype=PMT_RESPONSE_DTYPE)
        arr["event_id"] = event_id
        arr["pmt_index"] = np.array(idx, dtype=np.int32)
        arr["charge"] = np.array([responses[i]["charge"] for i in idx], dtype=np.float32)
        arr["time"] = np.array([responses[i]["time"] for i in idx], dtype=np.float32)
        arr["n_hits"] = np.array([responses[i]["n_hits"] for i in idx], dtype=np.int32)

        f = self._ensure_file()
        append_table(f, "pmt_responses", arr)
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
        # Normalize direction so theta is consistent regardless of input
        norm = np.linalg.norm([dx, dy, dz])
        if norm > 0:
            dx, dy, dz = dx/norm, dy/norm, dz/norm
        theta_deg = float(np.degrees(np.arccos(-dz)))
        phi_deg = float(np.degrees(np.arctan2(dy, dx)))

        arr = np.zeros(1, dtype=MUON_TRUTH_DTYPE)
        arr["event_id"] = event_id
        arr["pos_x"] = x
        arr["pos_y"] = y
        arr["pos_z"] = z
        arr["t0"] = t0
        arr["dir_x"] = dx
        arr["dir_y"] = dy
        arr["dir_z"] = dz
        arr["theta_deg"] = theta_deg
        arr["phi_deg"] = phi_deg
        arr["track_length_mm"] = track_length * 1000.0  # convert m → mm
        arr["n_generated"] = n_generated
        arr["n_detected"] = n_detected

        f = self._ensure_file()
        append_table(f, "muon_truth", arr)
        self._n_muon_rows += 1

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HDF5 file (flushes data to disk)."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


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

    # Light burst (replaces Cherenkov generation)
    light_burst: bool = False
    burst_n_photons: int = 1000
    burst_position: Optional[tuple[float, float, float]] = None
    burst_t0: float = 0.0

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
    from annieray.cherenkov import generate_cherenkov_photons, generate_isotropic_photons
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

            if config.light_burst:
                cx, cy, cz = config.burst_position or (0, 0, 1940)
                muon_pos = (cx, cy, cz, config.burst_t0)
                muon_dir = (0.0, 0.0, -1.0)
                track_length = 0.0
                o, d, ct = generate_isotropic_photons(
                    muon_pos, config.burst_n_photons, rng,
                )
            else:
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
            h5_path = config.output_dir / H5_OUTPUT_NAME
            if accumulator._n_photon_rows > 0:
                paths["photon_hits"] = h5_path
                print(f"  Wrote {h5_path} ({accumulator._n_photon_rows} photon rows)")
            if accumulator._n_pmt_rows > 0:
                paths["pmt_responses"] = h5_path
                print(f"  Wrote {h5_path} ({accumulator._n_pmt_rows} PMT response rows)")
            if accumulator._n_muon_rows > 0:
                paths["muon_truth"] = h5_path
                print(f"  Wrote {h5_path} ({accumulator._n_muon_rows} muon truth rows)")
            return paths

    return {}
