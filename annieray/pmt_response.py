"""PMT digital model: per-PMT integrated charge and time from photon hits.

Two paths:
  **Fast path** (default): each photoelectron gets SPE charge resolution
    smearing and transit time spread.  Sum charges, earliest smeared time.
  **Waveform path** (``full_wf=True``): generate synthetic SPE pulse train
    with pedestal + noise, run threshold-crossing hit finder.

The design follows the Daya Bay / WCSim approach described in
``docs/2024-11-25_PMTWaveformSim_AnalysisSoftware.pdf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from annieray.tracer import HDI, HDS, HLU, HLV, H_ARRIVAL, H_WAVELEN
from annieray.tracer import DET_SYS_PMT


# ---------------------------------------------------------------------------
# Default PMT parameters (ANNIE Phase II PMT types)
# ---------------------------------------------------------------------------

# Typical values from WCSim PMTObject classes and ANNIE characterisation:
#   spe_charge_sigma:  fractional RMS of 1-PE charge distribution  (~35 %)
#   transit_time_sigma:  single-photon transit time spread (ns)    (~1.5 ns)
#   noise_rate_mhz:     dark-noise rate in MHz                     (~1 kHz)
PMT_TYPE_DEFAULTS: dict[str, dict] = {
    "ETEL":      {"spe_charge_sigma": 0.35, "transit_time_sigma": 1.8, "noise_rate_mhz": 1e-3},
    "LUX":       {"spe_charge_sigma": 0.30, "transit_time_sigma": 1.2, "noise_rate_mhz": 1e-3},
    "Hamamatsu": {"spe_charge_sigma": 0.35, "transit_time_sigma": 1.5, "noise_rate_mhz": 2e-3},
    "Watchboy":  {"spe_charge_sigma": 0.32, "transit_time_sigma": 1.6, "noise_rate_mhz": 1e-3},
    "Watchman":  {"spe_charge_sigma": 0.28, "transit_time_sigma": 1.0, "noise_rate_mhz": 1e-3},
}


@dataclass
class PMTResponseConfig:
    """Per-PMT-type digital model configuration.

    Parameters
    ----------
    spe_charge_mean : float
        Mean SPE charge in photoelectrons (nominally 1.0 PE).
    spe_charge_sigma : float
        Fractional RMS of the SPE charge distribution (e.g. 0.35 = 35 %).
    transit_time_sigma : float
        Single-photon transit time spread (ns, sigma of Gaussian).
    transit_time_offset : float
        Mean transit time delay (ns).  Added to every hit time.
    noise_rate_mhz : float
        Dark noise rate in MHz (1 kHz = 1e-3 MHz).
    noise_charge_mean : float
        Mean charge of noise hits (PE).
    noise_charge_sigma : float
        Sigma of noise-hit charge distribution (PE).
    """

    spe_charge_mean: float = 1.0
    spe_charge_sigma: float = 0.35
    transit_time_sigma: float = 1.5
    transit_time_offset: float = 0.0

    noise_rate_mhz: float = 1e-3
    noise_charge_mean: float = 0.2
    noise_charge_sigma: float = 0.1

    # Waveform-path parameters
    pulse_rise_ns: float = 1.0       # SPE pulse rise time (ns)
    pulse_fall_ns: float = 5.0       # SPE pulse fall time (ns)
    threshold_pe: float = 0.3        # hit-finder threshold (PE)
    integration_window_ns: tuple[float, float] = (-5.0, 20.0)  # pre/post peak (ns)

    @classmethod
    def for_pmt_type(cls, pmt_type: str) -> PMTResponseConfig:
        """Return a config tuned for a given ANNIE PMT type."""
        base = PMT_TYPE_DEFAULTS.get(pmt_type, PMT_TYPE_DEFAULTS["ETEL"])
        return cls(**base)

    @classmethod
    def build_lookup(
        cls,
        geometry_detectors: list,
        global_config: Optional[PMTResponseConfig] = None,
    ) -> dict[int, PMTResponseConfig]:
        """Build a ``{detector_index: PMTResponseConfig}`` lookup.

        Uses per-PMT-type defaults when available, falling back to
        *global_config* (or the default constructor).
        """
        from annieray.detectors import DetectorInfo

        global_cfg = global_config if global_config is not None else cls()
        lookup: dict[int, PMTResponseConfig] = {}
        for d in geometry_detectors:
            if d.system == "pmt":
                if d.pmt_type:
                    lookup[d.index] = cls.for_pmt_type(d.pmt_type)
                else:
                    lookup[d.index] = global_cfg
        return lookup


# ---------------------------------------------------------------------------
# SPE pulse shape (analytical)
# ---------------------------------------------------------------------------


def _spe_pulse(
    t: np.ndarray,
    amplitude: float = 1.0,
    rise_ns: float = 1.0,
    fall_ns: float = 5.0,
) -> np.ndarray:
    """Single-photoelectron pulse shape (normalised gamma function).

    ``f(t) = A * (t / ta)^n * exp(-n*(t/ta - 1))``   for ``t > 0``

    with ``n = (rise_ns / fall_ns)`` and ``ta = rise_ns``.
    Integral is ≈ ``amplitude`` (in PE · ns).
    """
    n = max(rise_ns / fall_ns, 0.1)
    ta = rise_ns
    out = np.zeros_like(t)
    mask = t > 0
    x = t[mask] / ta
    out[mask] = amplitude * (x ** n) * np.exp(-n * (x - 1.0))
    return out


# ---------------------------------------------------------------------------
# Hit finder  (threshold crossing + charge integration)
# ---------------------------------------------------------------------------


def _find_hits(
    waveform: np.ndarray,
    time_axis: np.ndarray,
    threshold: float,
    integration_window: tuple[float, float],
) -> list[dict]:
    """Simple threshold-crossing hit finder.

    Returns a list of ``{"charge": float, "time": float}`` dicts, one per
    found pulse.  Charge is integrated over the window around the peak.
    Time is the peak bin centre.
    """
    dt = time_axis[1] - time_axis[0]
    pre, post = integration_window
    pre_bins = int(abs(pre) / dt)
    post_bins = int(post / dt)

    above = waveform > threshold
    if not above.any():
        return []

    hits = []
    i = 0
    n = len(waveform)
    while i < n:
        if not above[i]:
            i += 1
            continue
        # start of pulse
        start = i
        while i < n and above[i]:
            i += 1
        end = i - 1

        # peak within this region
        peak_idx = start + np.argmax(waveform[start:end + 1])

        # integration window around peak
        lo = max(0, peak_idx - pre_bins)
        hi = min(n, peak_idx + post_bins + 1)

        charge = np.trapz(waveform[lo:hi], time_axis[lo:hi])
        time = time_axis[peak_idx]
        hits.append({"charge": charge, "time": time})

    return hits


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def process_pmt_hits(
    hits: np.ndarray,
    geometry,
    config: Optional[PMTResponseConfig] = None,
    rng: Optional[np.random.Generator] = None,
    full_wf: bool = False,
) -> dict[int, dict]:
    """Process PMT photon hits into per-PMT integrated charge and time.

    Parameters
    ----------
    hits : ndarray, shape ``(N, N_EXPANDED_COLS)``
        Full hit array from ``trace_cherenkov()``.  PMT hits are selected
        via ``hits[:, HDS] == DET_SYS_PMT``.
    geometry : Geometry
        Needed to resolve per-PMT-type configs via ``geometry.detectors``.
    config : PMTResponseConfig or None
        Global fallback config (per-PMT-type defaults are preferred).
    rng : numpy.random.Generator or None
    full_wf : bool
        If True, use the full waveform path (slower, more realistic).
        If False (default), use the fast analytical path.

    Returns
    -------
    dict
        ``{detector_index: {"charge": float (PE), "time": float (ns),
                            "n_hits": int}}``
        Only PMTs that received at least one photon are included.
    """
    if rng is None:
        rng = np.random.default_rng()

    pmt_mask = np.abs(hits[:, HDS] - DET_SYS_PMT) < 0.5
    if not pmt_mask.any():
        return {}

    pmt_hits = hits[pmt_mask]

    # Build per-index config lookup
    cfg_lookup = PMTResponseConfig.build_lookup(
        geometry.detectors, global_config=config
    )

    if full_wf:
        return _process_waveform_path(pmt_hits, cfg_lookup, rng)
    else:
        return _process_fast_path(pmt_hits, cfg_lookup, rng)


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------


def _process_fast_path(
    pmt_hits: np.ndarray,
    cfg_lookup: dict[int, PMTResponseConfig],
    rng: np.random.Generator,
) -> dict[int, dict]:
    """SPE-resolution smearing, no explicit waveform.

    For each PMT:
      - Each hit draws a charge from ``N(spe_charge_mean, spe_charge_sigma²)``
      - Total charge = sum.
      - Earliest hit time gets ``+ N(0, transit_time_sigma²)``.
    """
    result: dict[int, dict] = {}

    # Group by detector_index
    indices = pmt_hits[:, HDI].astype(np.int32)
    unique_indices = np.unique(indices)

    for d_idx in unique_indices:
        mask = indices == d_idx
        these = pmt_hits[mask]
        cfg = cfg_lookup.get(int(d_idx))

        n = these.shape[0]

        # SPE charge smearing: each hit draws charge from Gaussian
        charges = rng.normal(
            loc=cfg.spe_charge_mean,
            scale=cfg.spe_charge_sigma * cfg.spe_charge_mean,
            size=n,
        )
        charges = np.clip(charges, 0.1 * cfg.spe_charge_mean, None)
        total_charge = float(np.sum(charges))

        # Earliest arrival time + transit time spread
        earliest = float(np.min(these[:, H_ARRIVAL]))
        tts = rng.normal(0.0, cfg.transit_time_sigma)
        time = earliest + cfg.transit_time_offset + tts

        result[int(d_idx)] = {
            "charge": total_charge,
            "time": time,
            "n_hits": n,
        }

    return result


# ---------------------------------------------------------------------------
# Waveform path
# ---------------------------------------------------------------------------


def _process_waveform_path(
    pmt_hits: np.ndarray,
    cfg_lookup: dict[int, PMTResponseConfig],
    rng: np.random.Generator,
) -> dict[int, dict]:
    """Full waveform generation + hit finding.

    For each PMT:
      1. Create a time axis covering all hits (with padding).
      2. Generate an SPE pulse for each hit.
      3. Sum pulses, add pedestal + Gaussian noise.
      4. Run threshold-crossing hit finder.
      5. Sum charges of all found pulses; take earliest peak time.
    """
    result: dict[int, dict] = {}

    indices = pmt_hits[:, HDI].astype(np.int32)
    unique_indices = np.unique(indices)

    for d_idx in unique_indices:
        mask = indices == d_idx
        these = pmt_hits[mask]
        cfg = cfg_lookup.get(int(d_idx))

        n = these.shape[0]
        times = these[:, H_ARRIVAL]

        t_min = float(np.min(times))
        t_max = float(np.max(times))

        # Time axis with generous padding
        pad_pre = 50.0
        pad_post = 100.0
        t_start = t_min - pad_pre
        t_stop = t_max + pad_post
        dt = 0.1  # ns per bin (matching lappd_response convention)
        time_axis = np.arange(t_start, t_stop + dt, dt, dtype=np.float64)
        waveform = np.zeros_like(time_axis)

        # Generate SPE pulse for each hit
        for i in range(n):
            t_hit = times[i]
            pulse = _spe_pulse(
                time_axis - t_hit,
                amplitude=cfg.spe_charge_mean,
                rise_ns=cfg.pulse_rise_ns,
                fall_ns=cfg.pulse_fall_ns,
            )
            waveform += pulse

        # Pedestal + noise
        pedestal = rng.uniform(-0.05, 0.05)
        noise = rng.normal(0.0, 0.02, size=len(time_axis))
        waveform += pedestal + noise

        # Hit finding
        found = _find_hits(
            waveform, time_axis,
            threshold=cfg.threshold_pe,
            integration_window=cfg.integration_window_ns,
        )

        if found:
            total_charge = sum(f["charge"] for f in found)
            earliest_time = min(f["time"] for f in found)
        else:
            total_charge = 0.0
            earliest_time = 0.0

        result[int(d_idx)] = {
            "charge": total_charge,
            "time": earliest_time,
            "n_hits": n,
        }

    return result
