"""Plot LAPPD hit counts (or % of max) as a 2D heatmap over muon vertex.

Assumes the batch was run with a regular grid of muon positions
(via --muon-file) so that unique x and z values form a rectangular
grid.  Produces one panel per ANNIE LAPPD (3 panels).

Usage:
    python scripts/lappd_scan.py results/
    python scripts/lappd_scan.py --pct results/   # show % of peak per LAPPD
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_data(output_dir: Path):
    hits = pq.read_table(str(output_dir / "photon_hits.parquet")).to_pandas()
    muons = pq.read_table(str(output_dir / "muon_truth.parquet")).to_pandas()
    det = pd.read_csv(output_dir / "detectors.csv")
    return hits, muons, det


def main():
    pct_mode = "--pct" in sys.argv
    args = [a for a in sys.argv if not a.startswith("--")]
    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} <batch_output_dir>")
        return

    output_dir = Path(args[1])
    hits, muons, det = load_data(output_dir)

    # Identify ANNIE LAPPDs (system_code == 2 in detectors.csv)
    lappd_det = det[det["system_code"] == 2].copy()
    if lappd_det.empty:
        print("No ANNIE LAPPDs found in detector registry.")
        return

    lappd_indices = lappd_det["detector_index"].values
    lappd_labels = lappd_det["label"].values
    n_lappds = len(lappd_indices)
    print(f"Found {n_lappds} ANNIE LAPPDs: indices {lappd_indices}")

    # Per-event LAPPD hit counts: (event_id, detector_index) -> n_hits
    lappd_hits = hits[hits["detector_system"] == 2]
    if lappd_hits.empty:
        print("No LAPPD hits in data.")
        return

    hit_counts = lappd_hits.groupby(["event_id", "detector_index"]).size()

    # Build a grid from muon positions
    xs = muons["pos_x"].values
    zs = muons["pos_z"].values
    u_x = np.unique(xs)
    u_z = np.unique(zs)

    if len(u_x) * len(u_z) != len(muons):
        print("Warning: muon positions do not form a complete rectangular grid "
              f"({len(u_x)} x {len(u_z)} = {len(u_x)*len(u_z)}, "
              f"expected {len(muons)}).  Falling back to scatter plot.")
        gridded = False
    else:
        gridded = True
        nx, nz = len(u_x), len(u_z)
        # Sort so heatmap axes are ascending
        u_x.sort()
        u_z.sort()
        print(f"Grid: {nx} x {nz} = {nx * nz} events")

    # Build 2D arrays of hit counts per LAPPD
    if gridded:
        x_idx = {v: i for i, v in enumerate(u_x)}
        z_idx = {v: i for i, v in enumerate(u_z)}
        maps = []
        for li in lappd_indices:
            z2d = np.zeros((nz, nx))
            for _, row in muons.iterrows():
                ev = int(row["event_id"])
                xi, zi = x_idx[row["pos_x"]], z_idx[row["pos_z"]]
                z2d[zi, xi] = hit_counts.get((ev, li), 0)
            maps.append(np.ma.masked_where(z2d == 0, z2d))

        if pct_mode:
            normed = []
            for m in maps:
                mx = m.max()
                normed.append(m / mx * 100 if mx > 0 else m)
            maps = normed
            print("Displaying normalized (% of peak per LAPPD)")

    # Plot
    fig, axes = plt.subplots(1, n_lappds, figsize=(6 * n_lappds, 5),
                             squeeze=False)
    axes = axes[0]

    vmin, vmax = 0, 0
    if gridded:
        if pct_mode:
            vmin, vmax = 0, 100
        else:
            vmax = max(m.max() for m in maps) if maps else 1

    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    for i, li in enumerate(lappd_indices):
        ax = axes[i]
        label = lappd_labels[i] if i < len(lappd_labels) else f"LAPPD {li}"

        if gridded:
            im = ax.pcolormesh(u_x, u_z, maps[i], shading="auto",
                               cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("muon start x (mm)")
            ax.set_ylabel("muon start z (mm)")
        else:
            # Fallback: scatter
            vals = np.array([
                hit_counts.get((int(row["event_id"]), li), 0)
                for _, row in muons.iterrows()
            ])
            if pct_mode:
                mx = vals.max()
                vals = vals / mx * 100 if mx > 0 else vals
            sc = ax.scatter(xs, zs, c=vals, cmap=cmap, s=40,
                            edgecolors="white", linewidth=0.3, vmin=vmin, vmax=vmax)
            ax.set_xlabel("muon start x (mm)")
            ax.set_ylabel("muon start z (mm)")

        ax.set_title(f"{label}\nindex={li}")
        ax.set_aspect("equal")

        cbar_label = "LAPPD hits (% of peak)" if pct_mode else "LAPPD hits"
        if gridded:
            fig.colorbar(im, ax=ax, label=cbar_label)
        else:
            fig.colorbar(sc, ax=ax, label=cbar_label)

    title = "LAPPD hit counts (% of peak) vs muon vertex (x, z)" if pct_mode else "LAPPD hit counts vs muon vertex (x, z)"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
