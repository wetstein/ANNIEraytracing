"""LAPPD hit counts per angle as N×N sub-grids over muon vertex.

Each scan position shows an N×N grid of small colored boxes, one per
muon direction angle.  N is auto-detected from the batch output:
each (x,z) position must have N×N events.

Usage:
    python scripts/lappd_scan_angled.py results/
    python scripts/lappd_scan_angled.py --pct results/
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

SUB_SIZE_FRAC = 0.55  # sub-grid / grid-spacing ratio


def load_data(output_dir: Path):
    hits = pq.read_table(str(output_dir / "photon_hits.parquet")).to_pandas()
    muons = pq.read_table(str(output_dir / "muon_truth.parquet")).to_pandas()
    det = pd.read_csv(output_dir / "detectors.csv")
    return hits, muons, det


def build_cell_maps(muons, hit_counts, lappd_indices):
    """Group events by (pos_x, pos_z), sort by event_id, build N×N arrays.

    Returns (cell_maps, N) where cell_maps[li][(px, pz)] is an N×N array.
    """
    grouped = muons.groupby(["pos_x", "pos_z"], sort=False)
    cell_maps = {li: {} for li in lappd_indices}
    n_positions = 0
    detected_N = None

    for (px, pz), group in grouped:
        group = group.sort_values("event_id")
        M = len(group)
        N = int(math.isqrt(M))
        if N * N != M:
            print(f"Warning: group at ({px}, {pz}) has {M} events "
                  f"(not a perfect square), skipping")
            continue
        if detected_N is None:
            detected_N = N
        elif N != detected_N:
            print(f"Warning: inconsistent N at ({px}, {pz}) — "
                  f"expected {detected_N}, got {N}, skipping")
            continue

        for li in lappd_indices:
            arr = np.zeros((N, N))
            for k in range(N * N):
                ev = int(group.iloc[k]["event_id"])
                arr[k // N, k % N] = hit_counts.get((ev, li), 0)
            cell_maps[li][(float(px), float(pz))] = arr
        n_positions += 1

    if detected_N is None:
        print("Error: no valid position tiles found")
        return {}, 0

    print(f"Built {n_positions} position tiles ({n_positions} × {detected_N}² events, "
          f"N={detected_N})")
    return cell_maps, detected_N


def main():
    pct_mode = "--pct" in sys.argv
    args = [a for a in sys.argv if not a.startswith("--")]
    if len(args) < 2:
        print(f"Usage: {args[0]} <batch_output_dir>")
        return

    output_dir = Path(args[1])
    hits, muons, det = load_data(output_dir)

    lappd_det = det[det["system_code"] == 2].copy()
    if lappd_det.empty:
        print("No ANNIE LAPPDs found in detector registry.")
        return
    lappd_indices = lappd_det["detector_index"].values
    lappd_labels = lappd_det["label"].values
    n_lappds = len(lappd_indices)
    print(f"Found {n_lappds} ANNIE LAPPDs: indices {lappd_indices}")

    lappd_hits = hits[hits["detector_system"] == 2]
    if lappd_hits.empty:
        print("No LAPPD hits in data.")
        return
    hit_counts = lappd_hits.groupby(["event_id", "detector_index"]).size()

    cell_maps, N = build_cell_maps(muons, hit_counts, lappd_indices)
    if N == 0:
        return

    # Determine grid spacing from unique sorted positions
    positions = list(cell_maps[lappd_indices[0]].keys())
    xs = np.array([p[0] for p in positions])
    zs = np.array([p[1] for p in positions])
    u_x = np.unique(xs)
    u_z = np.unique(zs)
    dx = np.diff(u_x).min() if len(u_x) > 1 else 200
    dz = np.diff(u_z).min() if len(u_z) > 1 else 200
    grid_spacing = min(dx, dz)
    sub_size = grid_spacing * SUB_SIZE_FRAC
    cell_w = sub_size / N
    print(f"Grid spacing {grid_spacing} mm, sub-grid {sub_size:.0f} mm, "
          f"cells {cell_w:.1f} mm")

    # Color mapping
    all_vals = []
    for li in lappd_indices:
        for arr in cell_maps[li].values():
            all_vals.extend(arr.ravel())
    all_vals = np.array(all_vals)
    if pct_mode:
        vmin, vmax = 0, 100
    else:
        vmin, vmax = 0, float(all_vals.max()) if all_vals.max() > 0 else 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    # Determine per-lappd max for pct_mode normalization
    lappd_max = {}
    if pct_mode:
        for li in lappd_indices:
            mx = max(arr.max() for arr in cell_maps[li].values())
            lappd_max[li] = mx if mx > 0 else 1

    # Plot
    fig, axes = plt.subplots(1, n_lappds, figsize=(6 * n_lappds, 5),
                             squeeze=False)
    axes = axes[0]

    for i, li in enumerate(lappd_indices):
        ax = axes[i]
        label = lappd_labels[i] if i < len(lappd_labels) else f"LAPPD {li}"

        for (px, pz), arr in cell_maps[li].items():
            origin_x = px - sub_size / 2
            origin_z = pz - sub_size / 2
            for ri in range(N):
                for ci in range(N):
                    val = arr[ri, ci]
                    if pct_mode:
                        val_pct = val / lappd_max[li] * 100
                        color = cmap(norm(val_pct))
                    else:
                        color = cmap(norm(val))
                    if val == 0:
                        color = (1, 1, 1, 1)
                    rect = Rectangle(
                        (origin_x + ci * cell_w, origin_z + ri * cell_w),
                        cell_w, cell_w,
                        facecolor=color, edgecolor="0.85", linewidth=0.3
                    )
                    ax.add_patch(rect)

        ax.set_xlabel("muon start x (mm)")
        ax.set_ylabel("muon start z (mm)")
        ax.set_title(f"{label}\nindex={li}")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.05)

        cbar_label = "LAPPD hits (% of peak)" if pct_mode else "LAPPD hits"
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, label=cbar_label
        )

    title_suffix = f" (N={N})" if N != 3 else ""
    title = f"LAPPD hits per angle{title_suffix}" if not pct_mode else f"LAPPD hits per angle (% of peak){title_suffix}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
