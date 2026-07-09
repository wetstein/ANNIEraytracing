#!/usr/bin/env python3
"""Event display for ANNIE batch output.

Reads ``output.h5`` produced by ``annieray batch``.  Shows a three-panel
layout:

  Top view    – top endcap PMTs (panel 9) at (x, y) in tank cross-section
  Barrel      – unrolled cylinder (φ vs Z), barrel PMTs (panels 1-8) + LAPPDs
  Bottom view – bottom endcap PMTs (panel 0) at (x, y) in tank cross-section

The barrel φ axis is centered on the median LAPPD φ so that the
surfboard-mounted LAPPDs appear in the middle of the plot.

Controls: ◀/▶ buttons or left/right arrow keys to navigate events.

Usage:
    python scripts/event_display.py results/
    python scripts/event_display.py results/output.h5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

from annieray.io_h5 import load_table, read_attrs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(h5_path, first_hit_time=False, charge=False, time=False):
    hits = load_table(h5_path, "photon_hits")
    counts = hits.groupby(["event_id", "detector_system", "detector_index"]).size()

    det = load_table(h5_path, "detectors")

    meta = read_attrs(h5_path)

    first_times = None
    if first_hit_time:
        first_times = hits.groupby(
            ["event_id", "detector_system", "detector_index"]
        )["arrival_time"].min()

    pmt_charge = pmt_time = None
    if charge or time:
        pmt_df = load_table(h5_path, "pmt_responses")
        if not pmt_df.empty:
            pmt_charge = pmt_df.set_index(["event_id", "pmt_index"])["charge"]
            pmt_time = pmt_df.set_index(["event_id", "pmt_index"])["time"]

    return counts, first_times, det, meta, pmt_charge, pmt_time


# ---------------------------------------------------------------------------
# Detector classification
# ---------------------------------------------------------------------------

# Panel number semantics (from PMT scan file):
#   0  → bottom endcap (LUX, upward-facing)
#   1–8 → barrel octagon faces (inward-facing)
#   9  → top endcap (ETEL, downward-facing)


def classify_detectors(det_df):
    """Split detectors into top/barrel-PMT/barrel-LAPPD/bottom groups."""
    top = []
    barrel_pmt = []
    barrel_lappd = []
    bottom = []

    for _, row in det_df.iterrows():
        sys_code = int(row["system_code"])
        idx = int(row["detector_index"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        label = str(row["label"])
        panel = int(row["panel"]) if "panel" in row and pd.notna(row["panel"]) else -1

        is_pmt = sys_code == 0

        if is_pmt and panel == 9:
            top.append((sys_code, idx, x, y, label))
        elif is_pmt and panel == 0:
            bottom.append((sys_code, idx, x, y, label))
        elif is_pmt:
            phi = float(np.degrees(np.arctan2(y, x)))
            barrel_pmt.append((sys_code, idx, phi, z, label))
        else:
            phi = float(np.degrees(np.arctan2(y, x)))
            barrel_lappd.append((sys_code, idx, phi, z, label))

    return top, barrel_pmt, barrel_lappd, bottom


def keys_and_coords(det_list):
    """Return ``(keys, xs, ys)`` from a detector list."""
    if not det_list:
        return np.empty((0, 2), dtype=np.int32), np.array([]), np.array([])
    keys = np.array([(e[0], e[1]) for e in det_list], dtype=np.int32)
    xs = np.array([e[2] for e in det_list])
    ys = np.array([e[3] for e in det_list])
    return keys, xs, ys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ANNIE event display")
    parser.add_argument("input", type=Path, default=Path("results"),
                        nargs="?",
                        help="Batch output dir or output.h5 path (default: results/)")
    parser.add_argument("--linear", action="store_true",
                        help="Linear color scale (default: log\u2081\u2080)")
    parser.add_argument("--hide-zero", action="store_true",
                        help="Show PMTs with zero charge/hits as white")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--first-hit-time", action="store_true",
                      help="Color by first-hit arrival time (ns) from photon_hits")
    mode.add_argument("--charge", action="store_true",
                      help="Color PMTs by integrated charge (PE) from digital model")
    mode.add_argument("--time", action="store_true",
                      help="Color PMTs by response time (ns) from digital model")
    args = parser.parse_args()

    if args.input.suffix == ".h5":
        h5_path = args.input
    else:
        h5_path = args.input / "output.h5"

    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    print(f"HDF5: {h5_path}")

    counts, first_times, det_df, meta, pmt_charge, pmt_time = load_data(
        h5_path, first_hit_time=args.first_hit_time,
        charge=args.charge, time=args.time,
    )

    if (args.charge or args.time) and pmt_charge is None:
        print("Warning: no pmt_responses table in HDF5.  "
              "Re-run batch with --pmt-response to use this mode.  "
              "Falling back to raw hit counts.")
        args.charge = args.time = False

    tank_r = meta["tank_radius_mm"]
    tank_z_min = meta["tank_z_min_mm"]
    tank_z_max = meta["tank_z_max_mm"]

    events = sorted(counts.index.get_level_values("event_id").unique())
    n_events = len(events)
    if n_events == 0:
        print("No events found.")
        return

    print(f"{len(det_df)} detectors, {n_events} events")

    # ---- Classify detectors ----------------------------------------------
    top, barrel_pmt, barrel_lappd, bottom = classify_detectors(det_df)

    top_keys, top_xs, top_ys = keys_and_coords(top)
    bp_keys,  bp_xs,  bp_ys  = keys_and_coords(barrel_pmt)
    bl_keys,  bl_xs,  bl_ys  = keys_and_coords(barrel_lappd)
    bot_keys, bot_xs, bot_ys = keys_and_coords(bottom)

    # ---- Barrel φ centering on the positive Y axis -----------------------
    center_phi = 90.0  # positive Y axis in arctan2(y,x) convention

    # Wrap φ differences to [-180, 180] so the plot wraps around correctly
    bp_xs_ctr = ((bp_xs - center_phi + 180) % 360) - 180
    bl_xs_ctr = ((bl_xs - center_phi + 180) % 360) - 180

    # ---- Figure layout ---------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1, 1.5, 1],
        hspace=0.35,
        left=0.08, right=0.88, top=0.95, bottom=0.10,
    )

    ax_top = fig.add_subplot(gs[0])
    ax_barrel = fig.add_subplot(gs[1])
    ax_bottom = fig.add_subplot(gs[2])

    cmap = plt.cm.plasma.copy()
    if args.hide_zero:
        cmap.set_bad("white")
    norm = Normalize(vmin=0, vmax=None)
    use_log = not args.linear

    def raw_value(event_id: int, sc: int, di: int) -> float:
        """Get the base value for a detector in the selected mode."""
        if args.charge and sc == 0 and pmt_charge is not None:
            try:
                return float(pmt_charge.loc[(event_id, di)])
            except KeyError:
                return 0.0
        if args.time and sc == 0 and pmt_time is not None:
            try:
                return float(pmt_time.loc[(event_id, di)])
            except KeyError:
                return 0.0
        if args.first_hit_time and first_times is not None:
            try:
                return float(first_times.loc[(event_id, sc, di)])
            except KeyError:
                return 0.0
        try:
            return float(counts.loc[(event_id, sc, di)])
        except KeyError:
            return 0.0

    def color_value(event_id: int, sc: int, di: int) -> float:
        v = raw_value(event_id, sc, di)
        if args.time:
            return v
        if use_log:
            return np.log10(v + 1.0)
        return v

    def colorbar_label() -> str:
        if args.charge:
            return "charge (PE)" if args.linear else "log\u2081\u2080(charge + 1) (PE)"
        if args.time:
            return "response time (ns)"
        if args.first_hit_time:
            return "first hit time (ns)"
        return "log\u2081\u2080(n_hits + 1)" if use_log else "n_hits"

    # ---- Top view --------------------------------------------------------
    ax_top.add_patch(Circle(
        (0, 0), tank_r, fill=False,
        color="gray", linestyle="--", linewidth=0.7,
    ))
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-tank_r * 1.15, tank_r * 1.15)
    ax_top.set_ylim(-tank_r * 1.15, tank_r * 1.15)
    ax_top.set_title("Top View (panel 9 – ETEL endcap)")
    ax_top.set_xlabel("x (mm)")
    ax_top.set_ylabel("y (mm)")

    s_top = (
        ax_top.scatter(
            top_xs, top_ys, marker="o", s=40, c=np.zeros(len(top)), cmap=cmap, norm=norm,
            edgecolors="white", linewidth=0.3,
        )
        if len(top) > 0 else None
    )

    # ---- Barrel (unrolled cylinder) --------------------------------------
    all_barrel_z = np.concatenate([bp_ys, bl_ys])
    if len(all_barrel_z) > 0:
        z_lo, z_hi = all_barrel_z.min(), all_barrel_z.max()
        z_pad = max(100.0, (z_hi - z_lo) * 0.08)
        barrel_z_lo, barrel_z_hi = z_lo - z_pad, z_hi + z_pad
    else:
        barrel_z_lo, barrel_z_hi = tank_z_min, tank_z_max

    phi_half_range = 190.0
    ax_barrel.set_xlim(-phi_half_range, phi_half_range)
    ax_barrel.set_ylim(barrel_z_lo, barrel_z_hi)
    ax_barrel.set_title(f"Barrel (unrolled, centered on LAPPD φ = {center_phi:.0f}°)")
    ax_barrel.set_xlabel("φ (degrees)")
    ax_barrel.set_ylabel("z (mm)")
    ax_barrel.axhline(tank_z_min, color="gray", linestyle=":", linewidth=0.5)
    ax_barrel.axhline(tank_z_max, color="gray", linestyle=":", linewidth=0.5)

    tick_vals = np.arange(-180, 181, 60)
    tick_labels = [f"{((v + center_phi + 180) % 360 - 180):.0f}°" for v in tick_vals]
    ax_barrel.set_xticks(tick_vals)
    ax_barrel.set_xticklabels(tick_labels)

    s_bp = (
        ax_barrel.scatter(
            bp_xs_ctr, bp_ys, marker="o", s=60, c=np.zeros(len(barrel_pmt)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3, label="PMT",
        )
        if len(barrel_pmt) > 0 else None
    )
    s_bl = (
        ax_barrel.scatter(
            bl_xs_ctr, bl_ys, marker="s", s=40, c=np.zeros(len(barrel_lappd)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3, label="LAPPD",
        )
        if len(barrel_lappd) > 0 else None
    )
    if s_bp or s_bl:
        ax_barrel.legend(loc="upper right", fontsize=9, markerscale=0.9)

    # ---- Bottom view -----------------------------------------------------
    ax_bottom.add_patch(Circle(
        (0, 0), tank_r, fill=False,
        color="gray", linestyle="--", linewidth=0.7,
    ))
    ax_bottom.set_aspect("equal")
    ax_bottom.set_xlim(-tank_r * 1.15, tank_r * 1.15)
    ax_bottom.set_ylim(-tank_r * 1.15, tank_r * 1.15)
    ax_bottom.set_title("Bottom View (panel 0 – LUX endcap)")
    ax_bottom.set_xlabel("x (mm)")
    ax_bottom.set_ylabel("y (mm)")

    s_bot = (
        ax_bottom.scatter(
            bot_xs, bot_ys, marker="o", s=40, c=np.zeros(len(bottom)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3,
        )
        if len(bottom) > 0 else None
    )

    # ---- Colorbar (shared across all panels) -----------------------------
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        label=colorbar_label(),
    )

    # ---- Navigation ------------------------------------------------------
    current = [0]

    title_text = fig.text(
        0.01, 0.98, "",
        ha="left", va="top", fontsize=13, fontweight="bold",
    )

    def update(event_id):
        all_colors = []
        for scatter, keys in [
            (s_top, top_keys),
            (s_bp, bp_keys),
            (s_bl, bl_keys),
            (s_bot, bot_keys),
        ]:
            if scatter is None or len(keys) == 0:
                continue
            colors = np.zeros(len(keys))
            for j, (sc, di) in enumerate(keys):
                colors[j] = color_value(event_id, sc, di)
            if args.hide_zero:
                colors[colors == 0] = np.nan
            scatter.set_array(colors)
            all_colors.extend(colors[~np.isnan(colors)] if args.hide_zero else colors)

        norm.vmax = max(all_colors) if all_colors else 1
        title_text.set_text(f"Event {event_id}  ({current[0] + 1}/{n_events})")
        fig.canvas.draw_idle()

    def next_event(_):
        current[0] = (current[0] + 1) % n_events
        update(events[current[0]])

    def prev_event(_):
        current[0] = (current[0] - 1) % n_events
        update(events[current[0]])

    def on_key(event):
        if event.key == "right":
            next_event(event)
        elif event.key == "left":
            prev_event(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    btn_prev = Button(fig.add_axes([0.35, 0.02, 0.10, 0.04]), "\u25c0 Prev")
    btn_next = Button(fig.add_axes([0.55, 0.02, 0.10, 0.04]), "Next \u25b6")
    btn_prev.on_clicked(prev_event)
    btn_next.on_clicked(next_event)

    update(events[0])

    print("Controls: \u25c0/\u25b6 buttons or left/right arrow keys")
    plt.show()


if __name__ == "__main__":
    main()
