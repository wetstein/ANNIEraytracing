"""Grid scan over muon track hypotheses using on-the-fly raytracing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from annieray.tracer import (
    Geometry,
    trace_cherenkov,
    DET_SYS_PMT,
    HDI,
    HDS,
    H_ARRIVAL,
)
from annieray.io_h5 import load_table
from annieray.likelihood import (
    poisson_charge_ll,
    time_residual_ll,
    total_log_likelihood,
    compute_per_pmt_sigma,
)

__all__ = [
    "ScanResult",
    "grid_scan_direction",
    "load_observed_event",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of a 2-D grid scan over muon direction."""

    theta_grid: np.ndarray       # (n_theta,)  deg
    phi_grid: np.ndarray         # (n_phi,)    deg
    scores: np.ndarray           # (n_theta, n_phi)  log-likelihood
    best_theta: float            # deg
    best_phi: float              # deg
    best_score: float
    true_theta: float | None = None
    true_phi: float | None = None
    fix_vertex: tuple[float, float, float] | None = None
    fix_t0: float | None = None
    photons_per_cm: int = 150
    timing: dict = field(default_factory=dict)


@dataclass
class ObservedEvent:
    """Observed PMT data and truth for a single event."""

    event_id: int
    pmt_counts: dict[int, int]       # {pmt_index: n_hits}
    pmt_times: dict[int, float]      # {pmt_index: time}
    all_pmt_indices: list[int]
    hit_pmt_indices: list[int]
    # Truth (for validation)
    true_pos: tuple[float, float, float] | None = None
    true_dir: tuple[float, float, float] | None = None
    true_theta: float | None = None
    true_phi: float | None = None
    true_t0: float | None = None
    true_photons_per_cm: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_from_angles(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    """(θ, φ) in degrees → (dx, dy, dz) unit vector.

    Uses the batch/muon-truth convention:
        θ =  0° → (0, 0, −1)  downward  (dz = −cos θ)
        θ = 90° → (sinφ, cosφ, 0)  horizontal
        θ = 180° → (0, 0, +1)  upward
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    st = math.sin(th)
    return (st * math.cos(ph), st * math.sin(ph), -math.cos(th))


def _count_hits_per_pmt(hits: np.ndarray) -> tuple[dict[int, int], dict[int, float]]:
    """From a (N, 17) hit array, return per-PMT hit counts and min arrival times.

    Only PMT hits (``detector_system == 0``) are considered.
    """
    pmt_mask = np.abs(hits[:, HDS] - DET_SYS_PMT) < 0.5
    pmt_hits = hits[pmt_mask]
    if len(pmt_hits) == 0:
        return {}, {}

    indices = pmt_hits[:, HDI].astype(np.int32)
    unique = np.unique(indices)

    counts: dict[int, int] = {}
    min_times: dict[int, float] = {}

    for d_idx in unique:
        mask = indices == d_idx
        these = pmt_hits[mask]
        counts[int(d_idx)] = int(mask.sum())
        min_times[int(d_idx)] = float(np.min(these[:, H_ARRIVAL]))

    return counts, min_times


# ---------------------------------------------------------------------------
# Load observed data
# ---------------------------------------------------------------------------


def load_observed_event(h5_path: Path, event_id: int) -> ObservedEvent:
    """Load observed PMT data and muon truth for *event_id*."""
    pmt = load_table(h5_path, "pmt_responses")
    event_pmt = pmt[pmt["event_id"] == event_id]

    pmt_counts: dict[int, int] = {}
    pmt_times: dict[int, float] = {}
    for _, row in event_pmt.iterrows():
        idx = int(row["pmt_index"])
        pmt_counts[idx] = int(row["n_hits"])
        pmt_times[idx] = float(row["time"])
    hit_indices = list(pmt_times.keys())

    # All PMT indices from detectors table
    det = load_table(h5_path, "detectors")
    pmt_det = det[det["system_code"] == 0]
    all_pmt_indices = sorted(int(r["detector_index"]) for _, r in pmt_det.iterrows())

    # Truth
    truth = load_table(h5_path, "muon_truth")
    event_truth = truth[truth["event_id"] == event_id]
    if not event_truth.empty:
        row = event_truth.iloc[0]
        true_pos = (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]))
        true_dir = (float(row["dir_x"]), float(row["dir_y"]), float(row["dir_z"]))
        true_theta = float(row["theta_deg"])
        true_phi = float(row["phi_deg"])
        true_t0 = float(row["t0"])
        track_cm = row["track_length_mm"] / 10.0
        # Handle old files where track_length_mm was actually stored in meters
        if track_cm < 10:
            track_cm = row["track_length_mm"] * 1000.0 / 10.0
        true_photons_per_cm = int(row["n_generated"] / max(track_cm, 1))
    else:
        true_pos = true_dir = None
        true_theta = true_phi = true_t0 = true_photons_per_cm = None

    return ObservedEvent(
        event_id=event_id,
        pmt_counts=pmt_counts,
        pmt_times=pmt_times,
        all_pmt_indices=all_pmt_indices,
        hit_pmt_indices=hit_indices,
        true_pos=true_pos,
        true_dir=true_dir,
        true_theta=true_theta,
        true_phi=true_phi,
        true_t0=true_t0,
        true_photons_per_cm=true_photons_per_cm,
    )


# ---------------------------------------------------------------------------
# Grid scan
# ---------------------------------------------------------------------------


def grid_scan_direction(
    observed: ObservedEvent,
    geometry: Geometry,
    theta_range: tuple[float, float, int],
    phi_range: tuple[float, float, int],
    fix_vertex: tuple[float, float, float] | None = None,
    fix_t0: float | None = None,
    photons_per_cm: int | None = None,
    use_time: bool = False,
    alpha: float = 1.0,
    time_sigma: float | dict[int, float] | None = None,
    rng_seed: int = 42,
    verbose: bool = True,
) -> ScanResult:
    """Grid scan over muon direction (θ, φ) with on-the-fly raytracing.

    Parameters
    ----------
    observed : ObservedEvent
        Loaded observed data for a single event.
    geometry : Geometry
        Pre-built geometry (reused across all evaluations).
    theta_range : (start, stop, steps)
        θ range in degrees (0–180).
    phi_range : (start, stop, steps)
        φ range in degrees (0–360).
    fix_vertex : (x, y, z) or None
        If None, use ``observed.true_pos``.
    fix_t0 : float or None
        If None, use ``observed.true_t0``.
    photons_per_cm : int or None
        If None, use ``observed.true_photons_per_cm`` or 150.
    """
    # Resolve fixed parameters
    vertex = fix_vertex if fix_vertex is not None else observed.true_pos
    t0 = fix_t0 if fix_t0 is not None else observed.true_t0
    ppcm = photons_per_cm or observed.true_photons_per_cm or 150

    if vertex is None:
        raise ValueError(
            "No vertex available. Provide --fix-vertex or ensure "
            "the HDF5 file has a muon_truth table."
        )

    # Timing sigma lookup
    if use_time:
        if time_sigma is None:
            resolved_sigma = compute_per_pmt_sigma(
                geometry, observed.all_pmt_indices
            )
        elif isinstance(time_sigma, dict):
            resolved_sigma = time_sigma
        else:
            resolved_sigma = {idx: time_sigma for idx in observed.all_pmt_indices}
    else:
        resolved_sigma = {}

    # Build grid
    theta_vals = np.linspace(theta_range[0], theta_range[1], theta_range[2])
    phi_vals = np.linspace(phi_range[0], phi_range[1], phi_range[2])
    n_theta = len(theta_vals)
    n_phi = len(phi_vals)
    scores = np.full((n_theta, n_phi), -np.inf)

    rng = np.random.default_rng(rng_seed)
    t_start = time.time()
    n_eval = 0
    n_total = n_theta * n_phi
    _last_report = 0.0

    for it, th in enumerate(theta_vals):
        for ip, ph in enumerate(phi_vals):
            muon_dir = _dir_from_angles(th, ph)
            muon_pos = (*vertex, t0 if t0 is not None else 0.0)

            hits = trace_cherenkov(
                muon_pos,
                muon_dir,
                photons_per_cm=ppcm,
                geometry=geometry,
                rng=rng,
            )
            n_eval += 1

            expected_counts, expected_times_min = _count_hits_per_pmt(hits)

            charge_ll = poisson_charge_ll(
                observed.pmt_counts,
                expected_counts,
                observed.all_pmt_indices,
            )

            if use_time and expected_times_min:
                time_ll = time_residual_ll(
                    observed.pmt_times,
                    expected_times_min,
                    observed.hit_pmt_indices,
                    resolved_sigma,
                )
            else:
                time_ll = 0.0

            scores[it, ip] = total_log_likelihood(charge_ll, time_ll, alpha=alpha)

            if verbose and (n_eval == 1 or time.time() - _last_report > 5.0):
                _last_report = time.time()
                elapsed = _last_report - t_start
                rate = n_eval / elapsed if elapsed > 0 else 0
                remaining = (n_total - n_eval) / rate if rate > 0 else 0
                best_sofar = float(np.max(scores))
                print(
                    f"  [{n_eval}/{n_total}] "
                    f"{elapsed:.0f}s elapsed, "
                    f"{rate:.1f} ev/s, "
                    f"{remaining:.0f}s remaining, "
                    f"best LL={best_sofar:.1f}"
                )

    elapsed = time.time() - t_start

    # Best point
    best_flat = int(np.argmax(scores))
    best_it = best_flat // n_phi
    best_ip = best_flat % n_phi

    if verbose:
        print(
            f"  Scan complete: {n_eval} evaluations in {elapsed:.1f}s "
            f"({n_eval / elapsed:.1f} ev/s)"
        )

    return ScanResult(
        theta_grid=theta_vals,
        phi_grid=phi_vals,
        scores=scores,
        best_theta=float(theta_vals[best_it]),
        best_phi=float(phi_vals[best_ip]),
        best_score=float(scores[best_it, best_ip]),
        true_theta=observed.true_theta,
        true_phi=observed.true_phi,
        fix_vertex=vertex,
        fix_t0=t0,
        photons_per_cm=ppcm,
        timing={"n_evaluations": n_eval, "elapsed_s": elapsed},
    )
