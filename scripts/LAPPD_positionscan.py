"""LAPPD position-scan heatmap from batch output.

Reads ``output.h5`` produced by ``annieray batch`` and plots heatmaps
of hit counts for each ANNIE LAPPD as a function of muon (X, Z) position.

Usage:
    python scripts/LAPPD_positionscan.py results/
    python scripts/LAPPD_positionscan.py results/output.h5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from annieray.io_h5 import load_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LAPPD position-scan heatmap from batch output"
    )
    parser.add_argument("input", type=Path, default=Path("results"), nargs="?",
                        help="Batch output dir or output.h5 path")
    args = parser.parse_args()

    h5_path = args.input
    if h5_path.suffix != ".h5":
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

    # ANNIE LAPPDs have indices 132, 133, 134 (sys_code == 2)
    LAPPD_Indices = [132, 133, 134]
    lappd_hits = hits[hits["detector_index"].isin(LAPPD_Indices)]
    if lappd_hits.empty:
        print("No LAPPD hits found.")
        return

    counts = lappd_hits.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)

    # Merge with muon positions (positions in mm, convert to m for plotting)
    merged = muons[["event_id", "pos_x", "pos_z"]].merge(pivoted, on="event_id", how="left").fillna(0)

    # Build grid from actual muon positions
    xs = merged["pos_x"].values / 1000.0  # mm → m
    zs = merged["pos_z"].values / 1000.0  # mm → m
    u_x = np.unique(xs)
    u_z = np.unique(zs)

    if len(u_x) * len(u_z) != len(merged):
        print(f"Warning: muon positions do not form a {len(u_x)}x{len(u_z)} grid. "
              f"{len(u_x) * len(u_z)} != {len(merged)}. Falling back to fixed grid.")
        u_x = np.linspace(xs.min(), xs.max(), int(np.sqrt(len(merged))))
        u_z = np.linspace(zs.min(), zs.max(), int(np.sqrt(len(merged))))

    nX, nZ = len(u_x), len(u_z)
    print(f"Grid: {nX}×{nZ} = {nX * nZ} muon positions")

    x_idx = {v: i for i, v in enumerate(u_x)}
    z_idx = {v: i for i, v in enumerate(u_z)}

    data = merged[["pos_x", "pos_z"] + LAPPD_Indices].copy()
    data["pos_x"] = xs
    data["pos_z"] = zs

    titles = [f"LAPPD {idx} Hits from Muon Vertex Positions" for idx in LAPPD_Indices]

    for i, idx in enumerate(LAPPD_Indices):
        z2d = np.full((nZ, nX), np.nan)
        for _, row in data.iterrows():
            xi, zi = x_idx[row["pos_x"]], z_idx[row["pos_z"]]
            z2d[zi, xi] = row[idx]

        fig, ax = plt.subplots()
        vmax = np.nanmax(z2d)
        norm = colors.LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None
        mesh = ax.pcolormesh(
            u_x, u_z, z2d, cmap="viridis", shading="auto",
            edgecolors="r", linewidths=0.5, norm=norm,
        )
        ax.set_title(titles[i])
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Z Position (m)")
        fig.colorbar(mesh, label="Number of Hits")
        plt.show()


if __name__ == "__main__":
    main()
