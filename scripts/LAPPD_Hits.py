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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect batch output")
    parser.add_argument("--hits", default="results/photon_hits.parquet",
                        help="Path to photon_hits.parquet")
    parser.add_argument("--pmts", default="results/pmt_responses.parquet",
                        help="Path to pmt_responses.parquet")
    parser.add_argument("--muons", default="results/muon_truth.parquet",
                        help="Path to muon_truth.parquet")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────
    hits = pq.read_table(args.hits).to_pandas()
    print(f"photon_hits: {len(hits)} rows, {hits.event_id.nunique()} events")

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



    # ── Example: hits on PMT 42 in every event ────────────────────
    pmt42 = by_detector.query(
        "detector_system == 0 and detector_index == 42"
    )
    print(f"\n── PMT index 42: {len(pmt42)} events with hits ──")
    if not pmt42.empty:
        print(pmt42.head())

    lappd132_p = by_detector.query(
        "detector_system == 2 and detector_index == 132"
    )

    lappd133_p = by_detector.query(
        "detector_system == 2 and detector_index == 133"
    )

    lappd134_p = by_detector.query(
        "detector_system == 2 and detector_index == 134"
    )



    print("ASDFASDFADHJGJHGJHGJHGJHGJHGSFA")
    print(type(pmt42))

    # ── Example: all ANNIE LAPPD hits ─────────────────────────────
    annie = by_detector.query("detector_system == 2")
    print(f"\n── ANNIE LAPPD hits: {len(annie)} rows ──")
    if not annie.empty:
        print(annie.head())

    # ── Muon truth parameters ─────────────────────────────────────
    try:
        muons = pq.read_table(args.muons).to_pandas()
    except (FileNotFoundError, OSError):
        muons = None
    if muons is not None:
        print(f"\n── Muon truth: {len(muons)} events ──")
        print(muons.head())

        pmt42countsbyevent = pmt42.set_index("event_id")["n_hits"]
        pmt42counts_allevents =  pmt42countsbyevent.reindex(muons["event_id"], fill_value=0)
        vpmt42counts = pmt42counts_allevents.values

        print("\n── PMT 42 hit counts for all events (first 10) ──")
        print(vpmt42counts)

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
        pmts = pq.read_table(args.pmts).to_pandas()
    except (FileNotFoundError, OSError):
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
        print("\n(No pmt_responses.parquet found — skip --pmt-response?)")


    #Creating LAPPD hit data
    LAPPD_Indices = [132, 133, 134]
    lappd_hits = hits[hits["detector_index"].isin(LAPPD_Indices)]
    counts = lappd_hits.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)
    pivoted.columns = [f"n_pmt{c}" for c in pivoted.columns]

    tablePhotonsPerMuonHits = muons[["event_id", "pos_x", "pos_z"]].merge(pivoted, on="event_id", how="left")
    #Writing the txt file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    with open("LAPPDHits.txt","w",encoding="utf-8") as file:
        file.write(tablePhotonsPerMuonHits.to_string(index=False))


if __name__ == "__main__":
    main()
