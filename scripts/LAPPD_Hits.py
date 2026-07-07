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

#Below is for plotting
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random


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

    # ── Muon truth parameters ─────────────────────────────────────
    try:
        muons = pq.read_table(args.muons).to_pandas()
    except (FileNotFoundError, OSError):
        muons = None

    #Creating LAPPD hit data
    LAPPD_Indices = [132, 133, 134]
    lappd_hits = hits[hits["detector_index"].isin(LAPPD_Indices)]
    counts = lappd_hits.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)
    pivoted.columns = [f"n_pmt{c}" for c in pivoted.columns]

    tablePhotonsPerMuonHits = muons[["event_id", "pos_x", "pos_z"]].merge(pivoted, on="event_id", how="left").fillna(0)

    data = tablePhotonsPerMuonHits[["n_pmt132", "n_pmt133", "n_pmt134"]].to_numpy() #Selecting needed data and converting to np array
    
    #Plotting color mesh
    x_vals = np.linspace(-1.2, 1.2, 13) #make this based on data from "tablePhotonPerMuonHits" 
    y_vals = np.linspace(-1.2, 1.2, 13)
    XX,YY = np.meshgrid(x_vals, y_vals)
    fig,ax = plt.subplots()
    plt.pcolormesh(XX,YY,data[:, 0].reshape(13,13), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 0].max()))
    plt.title("LAPPD 132 Hits")
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)

    plt.colorbar()

    plt.show()


    #Plotting LAPPD 133 Hits
    fig2,ax2 = plt.subplots()
    plt.pcolormesh(XX,YY,data[:, 1].reshape(13,13), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 1].max()))
    plt.title("LAPPD 133 Hits")
    ax2.set_xticks(x_vals)
    ax2.set_yticks(y_vals)

    plt.colorbar()

    plt.show()

    #Plotting LAPPD 134 Hits
    fig3,ax3 = plt.subplots()
    plt.pcolormesh(XX,YY,data[:, 2].reshape(13,13), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 2].max()))
    plt.title("LAPPD 134 Hits")
    ax3.set_xticks(x_vals)
    ax3.set_yticks(y_vals)

    plt.colorbar()

    plt.show()
if __name__ == "__main__":
    main()
