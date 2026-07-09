#!/usr/bin/env python3
"""Polar contour plot of 2-D likelihood surface from grid_scan_direction.

Usage:
    python scripts/fit_viewer.py <grid.npz>
    python -m annieray fit ... --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_polar(result, ax=None, show_colorbar=True):
    """Polar contour plot of log-likelihood vs (θ, φ).

    Parameters
    ----------
    result : ScanResult
        Result from ``grid_scan_direction()``.
    ax : matplotlib.axes.Axes or None
    show_colorbar : bool
    """
    if ax is None:
        ax = plt.subplot(111, projection="polar")

    theta_rad = np.radians(result.theta_grid)
    phi_rad = np.radians(result.phi_grid)
    TH, PH = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    # scores → probability scale for visualisation
    scores = result.scores
    smax = scores.max()
    smin = scores.min()
    if smax > smin:
        prob = np.exp(scores - smax)
    else:
        ax.set_title("Likelihood surface (flat — no discrimination)", pad=20)
        return ax

    levels = np.linspace(prob.min(), prob.max(), 21)
    cf = ax.contourf(PH, TH, prob, levels=levels, cmap="inferno")

    if result.true_theta is not None:
        ax.plot(
            np.radians(result.true_phi),
            np.radians(result.true_theta),
            marker="*", color="cyan", markersize=14,
            mew=1.5, mec="white", zorder=5,
            label="True",
        )

    ax.plot(
        np.radians(result.best_phi),
        np.radians(result.best_theta),
        marker="o", color="lime", markersize=10,
        mew=1.5, mec="white", zorder=5,
        label="Best fit",
    )

    ax.set_ylim(0, np.pi)
    ax.set_yticks(np.radians([30, 60, 90, 120, 150]))
    ax.set_yticklabels(["30°", "60°", "90°", "120°", "150°"])
    ax.set_title("Likelihood surface", pad=20)

    ax.legend(loc="upper right", fontsize=10)

    if show_colorbar:
        plt.colorbar(cf, ax=ax, label="Relative likelihood", pad=0.12)

    plt.tight_layout()
    return ax


def main():
    p = argparse.ArgumentParser(description="Plot grid-scan likelihood surface")
    p.add_argument("grid_file", type=Path, help="NPZ file from --save-grid")
    p.add_argument("--save", type=Path, default=None, help="Save figure to file")
    args = p.parse_args()

    data = np.load(args.grid_file)

    class FakeResult:
        theta_grid = data["theta_grid"]
        phi_grid = data["phi_grid"]
        scores = data["scores"]
        best_theta = float(data["best_theta"])
        best_phi = float(data["best_phi"])
        true_theta = float(data["true_theta"]) if "true_theta" in data else None
        true_phi = float(data["true_phi"]) if "true_phi" in data else None

    plot_polar(FakeResult())
    if args.save:
        plt.savefig(args.save, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
