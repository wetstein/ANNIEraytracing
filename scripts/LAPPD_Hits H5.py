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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#Below is imports for plotting
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


from annieray.io_h5 import load_table


def main() -> None:
    if len(sys.argv) < 2:
        h5_path = Path("results")
    else:
        h5_path = Path(sys.argv[1])

    if h5_path.suffix == ".h5":
        pass
    else:
        h5_path = h5_path / "output.h5"

    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    print(f"HDF5: {h5_path}")

    # ── Load ──────────────────────────────────────────────────────
    hits = load_table(h5_path, "photon_hits")
    if hits.empty:
        print("No photon_hits found.")
        return

    muons = load_table(h5_path, "muon_truth")
    if muons.empty:
        print("No muon_truth found.")
        return


    #Creating LAPPD hit data
    LAPPD_Indices = [132, 133, 134]
    lappd_hits = hits[hits["detector_index"].isin(LAPPD_Indices)]
    counts = lappd_hits.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)
    pivoted.columns = [f"n_pmt{c}" for c in pivoted.columns]

    tablePhotonsPerMuonHits = muons[["event_id", "pos_x", "pos_z"]].merge(pivoted, on="event_id", how="left").fillna(0)

    data = tablePhotonsPerMuonHits[["n_pmt132", "n_pmt133", "n_pmt134"]].to_numpy() #Selecting needed data and converting to np array
    


    xpos = tablePhotonsPerMuonHits["pos_x"].to_numpy() #Converting x positions to a numpy array
    zpos = tablePhotonsPerMuonHits["pos_z"].to_numpy() #Converting z positions to a numpy array
    
    numX = len(np.unique(xpos)) #gets the unique number of x cords | These are the columns 
    numZ = len(np.unique(zpos)) #gets the unique number of z cords | These are the rows

    #Plotting color mesh
    x_vals = np.linspace(np.min(xpos), np.max(xpos), numX)/(10**3) #m
    z_vals = np.linspace(np.min(zpos), np.max(zpos), numZ)/(10**3) #m
    XX,ZZ = np.meshgrid(x_vals, z_vals)
    
    
    #Plotting LAPPD 132 Hits
    fig,ax = plt.subplots()
    plt.pcolormesh(XX,ZZ,data[:, 0].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 0].max()))
    plt.title("LAPPD 132 Hits from Muon Vertex Positions")
    ax.set_xticks(x_vals)
    ax.set_yticks(z_vals)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)") #this is outward facing so it should be labeled as y
    cbar1 = plt.colorbar()
    cbar1.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)
    
    plt.show()


    #Plotting LAPPD 133 Hits
    fig2,ax2 = plt.subplots()
    plt.pcolormesh(XX,ZZ,data[:, 1].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 1].max()))
    plt.title("LAPPD 133 Hits from Muon Vertex Positions")
    ax2.set_xticks(x_vals)
    ax2.set_yticks(z_vals)
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    cbar2 = plt.colorbar()
    cbar2.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)

    plt.show()

    #Plotting LAPPD 134 Hits
    fig3,ax3 = plt.subplots()
    plt.pcolormesh(XX,ZZ,data[:, 2].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 2].max()))
    plt.title("LAPPD 134 Hits from Muon Vertex Positions")
    ax3.set_xticks(x_vals)
    ax3.set_yticks(z_vals)
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    cbar3 = plt.colorbar()
    cbar3.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)
    plt.show()
if __name__ == "__main__":
    main()
