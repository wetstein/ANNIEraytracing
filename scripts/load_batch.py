"""Template: load batch output, group hits by event and detector.

Usage:
    python scripts/load_batch.py [--hits results/photon_hits.parquet]
                                 [--pmts results/pmt_responses.parquet]

The output Parquet files have these columns:

    photon_hits.parquet:
        event_id, detector_system, detector_index,
        local_u, local_v, arrival_time, wavelength

    pmt_responses.parquet (only with --pmt-response):
        event_id, pmt_index, charge, time, n_hits

    muon_truth.parquet:
        event_id, pos_x/y/z, t0, dir_x/y/z, theta_deg, phi_deg,
        track_length_mm, n_generated, n_detected

Detector system codes:
    0 = PMT
    1 = LAPPD (default rectangular)
    2 = LAPPD (ANNIE housing)

Detector index maps to the geometry's detector registry (stable IDs).

Muon direction conventions:
    theta_deg: polar angle from vertical (0 = upward, 180 = downward)
    phi_deg:   azimuthal angle in XY plane (arctan2(y, x))
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from annieray.io_h5 import load_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect batch output")
    parser.add_argument("--hits", default=None,
                        help="Path to HDF5 output file (replaces --hits/--pmts/--muons)")
    parser.add_argument("--pmts",
                        help="Ignored (kept for backward compat)")
    parser.add_argument("--muons",
                        help="Ignored (kept for backward compat)")
    parser.add_argument("output", nargs="?", default="results",
                        help="Batch output directory (default results/)")
    args = parser.parse_args()

    if args.hits:
        h5_path = args.hits
    else:
        h5_path = Path(args.output) / "output.h5"

    # ── Load ──────────────────────────────────────────────────────
    hits = load_table(h5_path, "photon_hits")
    print(f"photon_hits: {len(hits)} rows, {hits.event_id.nunique()} events")

    print(hits)

    # ── Per-event summary ─────────────────────────────────────────
    per_event = hits.groupby("event_id").agg(
        n_photons=("detector_index", "count"),
        n_structure=("detector_system", lambda x: (x == -1).sum()),
        n_pmt=("detector_system", lambda x: (x == 0).sum()),
        n_lappd=("detector_system", lambda x: (x == 1).sum()),
        n_annie=("detector_system", lambda x: (x == 2).sum()),
    )
    print("\n── Per-event hit counts (first 5) ──")
    print(per_event.head())

    # ── Per-detector hit counts per event ─────────────────────────
    by_detector = hits.groupby(
        ["event_id", "detector_system", "detector_index"],
        as_index=False,
    ).agg(
        n_hits=("detector_index", "count"),
        first_arrival=("arrival_time", "min"),
    )
    print("\n── Per-detector hits (first 10) ──")
    print(by_detector.head(10))

    # ── Muon truth parameters ─────────────────────────────────────
    try:
        muons = load_table(h5_path, "muon_truth")
    except (KeyError, OSError):
        muons = None
    if muons is not None:
        print(f"\n── Muon truth: {len(muons)} events ──")
        print(muons.head())

        # ── Full-length hit-count vector for PMT 42 ───────────────
        pmt42 = by_detector.query(
            "detector_system == 0 and detector_index == 42"
        )
        counts = pmt42.set_index("event_id")["n_hits"]    # event_id -> n_hits
        full = counts.reindex(muons["event_id"], fill_value=0)
        vec = full.values                                  # numpy array, len = n_events
        nz = (vec > 0).sum()
        print(f"\n── PMT 42 hit-count vector: len={len(vec)}, "
              f"{nz} non-zero ({100 * nz / len(vec):.1f}%) ──")
        print(vec[:12], "...")

        # Merge muon params with per-event hit summaries
        ev = per_event.reset_index()
        ev_mu = ev.merge(muons, on="event_id", how="left")
        print(f"\n── Per-event hits + muon params (first 5) ──")
        print(ev_mu.head())

        # Example: filter by track length
        long = ev_mu.query("track_length_mm > 3000")
        print(f"\n── Events with track > 3 m: {len(long)} ──")
        if not long.empty:
            print(long[["event_id", "track_length_mm", "n_photons"]].head())

        # Example: filter by direction (downward = theta > 90)
        downward = ev_mu.query("theta_deg > 90")
        print(f"\n── Downward-going events: {len(downward)} ──")

        # Example: efficiency vs track length
        ev_mu["efficiency"] = ev_mu["n_detected"] / ev_mu["n_generated"].clip(1)
        print(f"\n── Detection efficiency range: "
              f"{ev_mu['efficiency'].min():.4f} – {ev_mu['efficiency'].max():.4f}")
    else:
        print("\n(No muon_truth.parquet found)")

    # ── PMT response data (if available) ──────────────────────────
    try:
        pmts = load_table(h5_path, "pmt_responses")
    except (KeyError, OSError):
        pmts = None
    if pmts is not None:
        print(f"\n── PMT responses: {len(pmts)} rows, "
              f"{pmts.event_id.nunique()} events ──")
        per_event_charge = pmts.groupby("event_id").agg(
            total_charge=("charge", "sum"),
            n_hits=("n_hits", "sum"),
        )
        print(per_event_charge.head())
    else:
        print("\n(No pmt_responses in HDF5 — skip --pmt-response?)")


if __name__ == "__main__":
    main()
