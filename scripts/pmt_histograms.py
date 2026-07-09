"""Per-PMT histograms of charge or arrival time from batch output.

Reads ``output.h5`` produced by ``annieray batch``.  PMTs are split into
four panel groups, each opening its own figure window:

  - Top endcap (panel 9)
  - Barrel panels 1–4
  - Barrel panels 5–8
  - Bottom endcap (panel 0)

Usage:
    python scripts/pmt_histograms.py results/ --charge --bins 40
    python scripts/pmt_histograms.py results/output.h5 --time --bins 50
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from annieray.io_h5 import load_table


def classify_pmts(det_df):
    """Return ``{panel_group: [(detector_index, label), ...]}``.

    Panel groups: ``top``, ``barrel_1`` (panels 1-4),
    ``barrel_2`` (panels 5-8), ``bottom``.
    """
    groups: dict[str, list[tuple[int, str]]] = {
        "top": [], "barrel_1": [], "barrel_2": [], "bottom": [],
    }
    for _, row in det_df.iterrows():
        if int(row["system_code"]) != 0:
            continue
        idx = int(row["detector_index"])
        label = str(row["label"])
        panel = int(row["panel"]) if "panel" in row and pd.notna(row["panel"]) else -1

        if panel == 9:
            groups["top"].append((idx, label))
        elif panel == 0:
            groups["bottom"].append((idx, label))
        elif 1 <= panel <= 4:
            groups["barrel_1"].append((idx, label))
        elif 5 <= panel <= 8:
            groups["barrel_2"].append((idx, label))

    for name in groups:
        groups[name].sort(key=lambda x: x[0])
    return groups


def load_values(h5_path, charge_mode, time_mode):
    """Return ``(values_dict, actual_mode)`` where mode is ``"charge"`` or ``"time"``."""
    if time_mode:
        hits = load_table(h5_path, "photon_hits")
        pmt_hits = hits[hits["detector_system"] == 0]
        if pmt_hits.empty:
            print("No PMT hits found in photon_hits.")
            return {}, "time"
        return {
            di: grp["arrival_time"].values
            for di, grp in pmt_hits.groupby("detector_index")
        }, "time"

    resp = load_table(h5_path, "pmt_responses")
    if resp.empty:
        print("Warning: no pmt_responses table. Falling back to arrival times.")
        return load_values(h5_path, charge_mode=False, time_mode=True)

    return {
        pi: grp["charge"].values
        for pi, grp in resp.groupby("pmt_index")
    }, "charge"


def plot_group(group_name, pmts, values, fig_title, mode, bins):
    """Create a figure with one histogram subplot per PMT."""
    n_pmts = len(pmts)
    if n_pmts == 0:
        return

    ncols = min(6, max(3, int(math.ceil(math.sqrt(n_pmts)))))
    nrows = int(math.ceil(n_pmts / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows))
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=0.98)

    flat_axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    all_vals = np.concatenate([values[idx] for idx, _ in pmts if values.get(idx) is not None])
    if len(all_vals) == 0:
        x_lo, x_hi = 0, 1
    elif mode == "time":
        lo, hi = np.percentile(all_vals, [2, 98])
        x_lo, x_hi = max(0, lo), hi
    else:
        lo, hi = np.percentile(all_vals, [2, 98])
        x_lo, x_hi = 0, hi if hi > 0 else 1

    xlabel = "Arrival Time (ns)" if mode == "time" else "Charge (PE)"

    for ax, (idx, label) in zip(flat_axes, pmts):
        vals = values.get(idx)
        if vals is not None and len(vals) > 0:
            ax.hist(vals, bins=bins, range=(x_lo, x_hi),
                    color="steelblue", edgecolor="white", linewidth=0.3)
        else:
            ax.text(0.5, 0.5, "no hits", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_xlim(x_lo, x_hi)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{label}", fontsize=8)

    for ax in flat_axes[n_pmts:]:
        ax.set_visible(False)

    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=11)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.93,
                        wspace=0.35, hspace=0.45)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Per-PMT histograms of charge or arrival time"
    )
    parser.add_argument("input", type=Path, default=Path("results"), nargs="?",
                        help="Batch output dir or output.h5 path")
    parser.add_argument("--bins", type=int, default=40,
                        help="Number of histogram bins (default: 40)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--charge", action="store_true", dest="charge",
                      help="Histogram of integrated charge (PE) [default]")
    mode.add_argument("--time", action="store_true", dest="time",
                      help="Histogram of arrival time (ns)")
    args = parser.parse_args()

    h5_path = args.input
    if h5_path.suffix != ".h5":
        h5_path = h5_path / "output.h5"

    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    charge_mode = args.charge or not args.time
    time_mode = args.time

    det = load_table(h5_path, "detectors")
    if det.empty:
        print("Error: no detectors table.")
        return

    groups = classify_pmts(det)
    values, actual_mode = load_values(h5_path, charge_mode, time_mode)

    mode_label = actual_mode
    titles = {
        "top": f"Top Endcap (panel 9) — {mode_label}",
        "barrel_1": f"Barrel Panels 1–4 — {mode_label}",
        "barrel_2": f"Barrel Panels 5–8 — {mode_label}",
        "bottom": f"Bottom Endcap (panel 0) — {mode_label}",
    }

    for name, pmts in groups.items():
        if not pmts:
            continue
        print(f"  {titles[name]}: {len(pmts)} PMTs")
        plot_group(name, pmts, values, titles[name], mode_label, args.bins)

    print(f"Plotted {sum(len(v) for v in groups.values())} PMTs across 4 figures.")


if __name__ == "__main__":
    main()
