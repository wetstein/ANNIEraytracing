"""LAPPD response simulation: GPU-accelerated digitization pipeline.

Takes photon hits from the ray tracer (ANNIE LAPPDs only) and produces
digital readout matrices matching the LApyPD pipeline specification.

Data flow
---------
  1. Filter trace output for ANNIE LAPPD hits (detector_system == 2)
  2. Group by LAPPD detector_index
  3. For each LAPPD:
     a. Extract per-photon data: parallel_pos (along strips), perp_pos (across),
        arrival_time, wavelength
     b. Run Taichi kernel pipeline: QE -> charge -> transit time ->
        strip assignment -> attenuation -> digitization
     c. Return (28, 256) float32 readout matrices for side0 and side1

References
----------
Implements the LApyPD pipeline specification from:
    extern/LApyPD/LApyPD-tutorial.ipynb
Uses LAPPD25 QE curve data from:
    extern/LApyPD/dependencies/LAPPD25_interpolated_photon_energy_qe.csv
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import taichi as ti

# ---------------------------------------------------------------------------
# Constants (from LApyPD-tutorial.ipynb)
# ---------------------------------------------------------------------------

# Strip geometry
FACE_SIZE_MM = 200.0
STRIP_WIDTH_MM = 5.2
STRIP_SPACING_MM = 1.7
NUM_STRIPS = 28
STRIP_LENGTH_MM = 200.0

# Signal propagation
SIGNAL_SPEED_FRACTION_C = 0.53
C_MM_NS = 299.792458

# Digitization
TIME_STEP_NS = 0.1
TOTAL_TIME_NS = 25.6
NUM_TIME_BINS = int(TOTAL_TIME_NS / TIME_STEP_NS)  # 256
RISE_TIME_NS_FWHM = 1.0
IMPEDANCE_OHMS = 50.0
E_C = 1.602e-19

# Path to LApyPD dependencies (relative to this file)
_LApyPD_DEPS = (
    Path(__file__).resolve().parents[1] / "extern" / "LApyPD" / "dependencies"
)

# PC half-size from lappd_model.py (used to shift PC-centered coords to strip-local)
_PC_HALF = 95.75

# ---------------------------------------------------------------------------
# QE curve loader (loads from LApyPD CSV on first use)
# ---------------------------------------------------------------------------

_QE_TABLE: np.ndarray | None = None


def _load_qe_table() -> np.ndarray:
    """Load the LAPPD25 QE curve from LApyPD CSV.

    Returns
    -------
    table : ndarray, shape (N, 2), float64
        Columns are [wavelength_nm, QE].
    """
    global _QE_TABLE
    if _QE_TABLE is not None:
        return _QE_TABLE
    path = _LApyPD_DEPS / "LAPPD25_interpolated_photon_energy_qe.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"LApyPD not installed. Run: git submodule update --init\n"
            f"  (missing {path})"
        )
    data = np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=(0, 3), dtype=np.float64,
    )
    # Filter negative QE values (non-physical)
    data = data[data[:, 1] >= 0]
    _QE_TABLE = data
    return _QE_TABLE


# ---------------------------------------------------------------------------
# LAPPDGeometry - strip geometry  (CPU)
# ---------------------------------------------------------------------------


@dataclass
class LAPPDGeometry:
    """Microstrip geometry for a single LAPPD, matching LApyPD defaults."""

    face_size: float = FACE_SIZE_MM
    strip_width: float = STRIP_WIDTH_MM
    strip_spacing: float = STRIP_SPACING_MM
    strip_length: float = STRIP_LENGTH_MM
    num_strips: int = NUM_STRIPS

    strip_centers: np.ndarray = field(init=False)  # (28,) float32 - mm
    bounds: np.ndarray = field(init=False)  # (28, 2) float32 - [left, right]

    def __post_init__(self):
        half_gap = self.strip_spacing / 2
        centers = []
        bounds = []
        for i in range(self.num_strips):
            c = half_gap + i * (self.strip_width + self.strip_spacing)
            centers.append(c)
            left = c - self.strip_width / 2 - half_gap
            right = c + self.strip_width / 2 + half_gap
            bounds.append((left, right))
        self.strip_centers = np.array(centers, dtype=np.float32)
        self.bounds = np.array(bounds, dtype=np.float32)


# ---------------------------------------------------------------------------
# LAPPDResponseConfig - pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class LAPPDResponseConfig:
    """Configurable parameters for each stage of the LAPPD response pipeline.

    All defaults match LApyPD-tutorial.ipynb.
    """

    enable_qe: bool = True

    # Charge generation (electrons)
    charge_mean: float = 5e6
    charge_std: float = (6e6 - 4e6) / 6
    charge_min: float = 1e6
    charge_max: float = 1e7

    # Transit time (ns)
    transit_mean: float = 0.075
    transit_std: float = (0.08 - 0.07) / 6
    transit_min: float = 0.07
    transit_max: float = 0.08

    # Strip charge assignment (Gaussian spread)
    strip_gauss_sigma: float = 3.0
    strip_spread: int = 5

    # Charge attenuation - maximum imbalance between ends
    max_attenuation_skew: float = 0.10

    # Digitization
    time_step_ns: float = TIME_STEP_NS
    total_time_ns: float = TOTAL_TIME_NS
    rise_time_ns_fwhm: float = RISE_TIME_NS_FWHM
    impedance_ohms: float = IMPEDANCE_OHMS


# ---------------------------------------------------------------------------
# Taichi kernel helpers
# ---------------------------------------------------------------------------

# Initialise Taichi (required before any kernel/func definitions are parsed)
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Pre-computed constants for Gaussian CDF via erf
_SQRT2 = 1.4142135623730951
_SQRT2PI = 2.5066282746310002


@ti.func
def _erf_approx(x):
    """Fast erf approximation using tanh (error < 0.5%)."""
    return ti.tanh(1.2009 * x + 0.0784 * x * x * x)


@ti.func
def _gaussian_cdf(x, mean, sigma):
    """CDF of Gaussian at x using erf approximation."""
    return _erf_approx((x - mean) / (sigma * _SQRT2)) * 0.5 + 0.5


# ---------------------------------------------------------------------------
# Taichi kernels  (one per pipeline stage)
# ---------------------------------------------------------------------------


@ti.kernel
def apply_qe_kernel(
    wavelengths: ti.types.ndarray(ndim=1),
    qe_wavelengths: ti.types.ndarray(ndim=1),
    qe_values: ti.types.ndarray(ndim=1),
    rnd: ti.types.ndarray(ndim=1),
    passed: ti.types.ndarray(ndim=1),
):
    """Kernel 1: apply quantum efficiency filter per photon.

    For each photon, linearly interpolate QE from the LAPPD25 curve
    and compare against a uniform random number.  Sets ``passed[i] = 1``
    if the photon passes the QE filter.
    """
    n = wavelengths.shape[0]
    m = qe_wavelengths.shape[0]

    for i in range(n):
        lam = wavelengths[i]
        qe = 0.0
        # Linear interpolate QE from lookup table
        if lam <= qe_wavelengths[0]:
            qe = qe_values[0]
        elif lam >= qe_wavelengths[m - 1]:
            qe = qe_values[m - 1]
        else:
            lo = 0
            hi = m - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if qe_wavelengths[mid] <= lam:
                    lo = mid
                else:
                    hi = mid
            lam_lo = qe_wavelengths[lo]
            lam_hi = qe_wavelengths[hi]
            t_frac = 0.0
            if lam_hi > lam_lo:
                t_frac = (lam - lam_lo) / (lam_hi - lam_lo)
            qe = qe_values[lo] + t_frac * (qe_values[hi] - qe_values[lo])

        passed[i] = 1.0 if rnd[i] < qe else 0.0


@ti.kernel
def sample_charge_kernel(
    passed: ti.types.ndarray(ndim=1),
    rnd: ti.types.ndarray(ndim=1),
    mean: ti.f32,
    std: ti.f32,
    clip_min: ti.f32,
    clip_max: ti.f32,
    charges: ti.types.ndarray(ndim=1),
):
    """Kernel 2: sample charge (electrons) from a clipped Gaussian."""
    for i in range(passed.shape[0]):
        if passed[i] < 0.5:
            charges[i] = 0.0
        else:
            c = mean + std * rnd[i]
            if c < clip_min:
                c = clip_min
            elif c > clip_max:
                c = clip_max
            charges[i] = c


@ti.kernel
def sample_transit_time_kernel(
    passed: ti.types.ndarray(ndim=1),
    rnd: ti.types.ndarray(ndim=1),
    mean: ti.f32,
    std: ti.f32,
    clip_min: ti.f32,
    clip_max: ti.f32,
    transit_times: ti.types.ndarray(ndim=1),
):
    """Kernel 3: sample transit time (ns) from a clipped Gaussian."""
    for i in range(passed.shape[0]):
        if passed[i] < 0.5:
            transit_times[i] = 0.0
        else:
            t = mean + std * rnd[i]
            if t < clip_min:
                t = clip_min
            elif t > clip_max:
                t = clip_max
            transit_times[i] = t


@ti.kernel
def assign_strip_charges_kernel(
    passed: ti.types.ndarray(ndim=1),
    perp_positions: ti.types.ndarray(ndim=1),
    charges: ti.types.ndarray(ndim=1),
    strip_centers: ti.types.ndarray(ndim=1),
    bounds_left: ti.types.ndarray(ndim=1),
    bounds_right: ti.types.ndarray(ndim=1),
    sigma: ti.f32,
    spread: ti.i32,
    strip_charge_map: ti.types.ndarray(ndim=2),
):
    """Kernel 4: distribute per-photon charge across microstrips.

    For each photon that passed QE:
      1. Find closest strip center index.
      2. For each strip in [idx - spread, idx + spread]:
         Compute Gaussian CDF integral over the strip bounds and allocate
         the corresponding fraction of total charge.
    """
    for i in range(passed.shape[0]):
        if passed[i] < 0.5:
            continue

        center = perp_positions[i]
        total_charge = charges[i]
        n_strips = strip_centers.shape[0]

        closest = 0
        min_dist = 1e10
        for j in range(n_strips):
            d = center - strip_centers[j]
            if d < 0:
                d = -d
            if d < min_dist:
                min_dist = d
                closest = j

        lo = closest - spread
        if lo < 0:
            lo = 0
        hi = closest + spread + 1
        if hi > n_strips:
            hi = n_strips

        s2 = sigma * _SQRT2

        for j in range(lo, hi):
            a = bounds_left[j]
            b = bounds_right[j]
            cdf_b = _gaussian_cdf(b, center, sigma)
            cdf_a = _gaussian_cdf(a, center, sigma)
            area = cdf_b - cdf_a
            if area < 0.0:
                area = 0.0
            strip_charge_map[i, j] = area * total_charge


@ti.kernel
def apply_attenuation_kernel(
    passed: ti.types.ndarray(ndim=1),
    parallel_positions: ti.types.ndarray(ndim=1),
    strip_charge_map: ti.types.ndarray(ndim=2),
    strip_length: ti.f32,
    max_skew: ti.f32,
    charge_side0: ti.types.ndarray(ndim=2),
    charge_side1: ti.types.ndarray(ndim=2),
):
    """Kernel 5: split charge between two readout ends.

    Attenuation model from LApyPD:
      pos_norm = (parallel_pos / strip_length) * 2 - 1  in [-1, 1]
      skew = -pos_norm * max_skew
      side0_fraction = 0.5 + skew / 2
      side1_fraction = 0.5 - skew / 2
    """
    for i in range(passed.shape[0]):
        if passed[i] < 0.5:
            continue

        pp = parallel_positions[i]
        if pp < 0.0:
            pp = 0.0
        elif pp > strip_length:
            pp = strip_length

        pos_norm = (pp / strip_length) * 2.0 - 1.0
        skew = -pos_norm * max_skew
        s0_frac = 0.5 + skew * 0.5
        s1_frac = 0.5 - skew * 0.5

        n_strips = strip_charge_map.shape[1]
        for j in range(n_strips):
            q = strip_charge_map[i, j]
            if q > 0.0:
                charge_side0[i, j] = q * s0_frac
                charge_side1[i, j] = q * s1_frac


@ti.kernel
def digitize_kernel(
    passed: ti.types.ndarray(ndim=1),
    hit_times: ti.types.ndarray(ndim=1),
    transit_times: ti.types.ndarray(ndim=1),
    parallel_positions: ti.types.ndarray(ndim=1),
    charge_side0: ti.types.ndarray(ndim=2),
    charge_side1: ti.types.ndarray(ndim=2),
    strip_length: ti.f32,
    signal_speed: ti.f32,
    time_step: ti.f32,
    total_time: ti.f32,
    rise_sigma: ti.f32,
    impedance: ti.f32,
    readout_s0: ti.types.ndarray(ndim=2),
    readout_s1: ti.types.ndarray(ndim=2),
):
    """Kernel 6: accumulate Gaussian voltage pulses into readout matrices.

    For each (photon, strip) with non-zero charge:
      1. Compute arrival time at each readout end.
      2. Convert charge (electrons) to Coulombs.
      3. Generate Gaussian current pulse -> voltage.
      4. Atomic-add into the readout matrix.
    """
    n_photons = passed.shape[0]
    n_strips = charge_side0.shape[1]
    n_bins = readout_s0.shape[1]

    inv_sigma = 1.0 / rise_sigma
    norm_factor = 1.0 / (rise_sigma * _SQRT2PI)
    e_C = 1.602e-19

    for i in range(n_photons):
        if passed[i] < 0.5:
            continue

        ht = hit_times[i]
        tt = transit_times[i]
        pp = parallel_positions[i]

        if pp < 0.0:
            pp = 0.0
        elif pp > strip_length:
            pp = strip_length

        t_s0 = ht + tt + pp / signal_speed
        t_s1 = ht + tt + (strip_length - pp) / signal_speed

        for j in range(n_strips):
            q_s0 = charge_side0[i, j]
            q_s1 = charge_side1[i, j]

            if q_s0 > 0.0:
                charge_C = q_s0 * e_C
                for k in range(n_bins):
                    t = k * time_step
                    dt = t - t_s0
                    I_t = charge_C * norm_factor * ti.exp(-0.5 * (dt * inv_sigma) ** 2) * 1e9
                    ti.atomic_add(readout_s0[j, k], I_t * impedance)

            if q_s1 > 0.0:
                charge_C = q_s1 * e_C
                for k in range(n_bins):
                    t = k * time_step
                    dt = t - t_s1
                    I_t = charge_C * norm_factor * ti.exp(-0.5 * (dt * inv_sigma) ** 2) * 1e9
                    ti.atomic_add(readout_s1[j, k], I_t * impedance)


# ---------------------------------------------------------------------------
# Default singleton geometry
# ---------------------------------------------------------------------------

_DEFAULT_GEOMETRY = LAPPDGeometry()


# ---------------------------------------------------------------------------
# Process hits from tracer output  (numpy array interface)
# ---------------------------------------------------------------------------


def process_hits(
    hits: np.ndarray,
    config: LAPPDResponseConfig | None = None,
    rng_seed: int = 42,
    qe_table: np.ndarray | None = None,
    geometry: LAPPDGeometry | None = None,
) -> dict[int, dict]:
    """Run LAPPD digitization pipeline on trace output.

    Parameters
    ----------
    hits : ndarray, shape (N, 17), float32
        Output of ``trace_cherenkov()``.  Only rows where
        ``detector_system == 2`` (DET_SYS_LAPPD_ANNIE) are processed.
    config : LAPPDResponseConfig or None
        Pipeline configuration.  Uses defaults when ``None``.
    rng_seed : int
        Seed for CPU-side random pre-generation.
    qe_table : ndarray, shape (M, 2) or None
        QE lookup table with columns [wavelength_nm, QE].
        Loads LAPPD25 data from LApyPD CSV when ``None``.
    geometry : LAPPDGeometry or None
        Strip geometry.  Uses default 28-strip 200 mm face when ``None``.

    Returns
    -------
    result : dict[int, dict]
        Mapping of ``detector_index`` to::

            {
                "side0": ndarray (28, 256) float32,
                "side1": ndarray (28, 256) float32,
                "n_photons": int,
                "n_passed_qe": int,
            }
    """
    if config is None:
        config = LAPPDResponseConfig()
    if qe_table is None:
        qe_table = _load_qe_table()
    geo = geometry or _DEFAULT_GEOMETRY

    # Column indices matching tracer.py conventions
    HDI = 9  # detector_index
    HDS = 10  # detector_system
    HLU = 11  # local_u (parallel pos, along strips)
    HLV = 12  # local_v (perp pos, across strips)
    H_ARRIVAL = 14
    H_WAVELEN = 15

    # Filter for ANNIE LAPPD hits only
    lappd_mask = hits[:, HDS] == 2.0
    if not lappd_mask.any():
        return {}

    lappd_hits = hits[lappd_mask]
    detector_indices = lappd_hits[:, HDI].astype(np.int32)
    unique_indices = np.unique(detector_indices)

    result: dict[int, dict] = {}
    qe_table_f = np.ascontiguousarray(qe_table.astype(np.float32))
    qe_wavelengths = np.ascontiguousarray(qe_table_f[:, 0])
    qe_values = np.ascontiguousarray(qe_table_f[:, 1])
    geo_strip_centers = np.ascontiguousarray(geo.strip_centers)
    geo_bounds_left = np.ascontiguousarray(geo.bounds[:, 0])
    geo_bounds_right = np.ascontiguousarray(geo.bounds[:, 1])

    for det_idx in unique_indices:
        mask = detector_indices == det_idx
        group = lappd_hits[mask]
        n = group.shape[0]

        # Per-photon data arrays (must be contiguous for Taichi)
        perp_pos = np.ascontiguousarray(
            group[:, HLV].astype(np.float32) + _PC_HALF
        )
        parallel_pos = np.ascontiguousarray(
            group[:, HLU].astype(np.float32) + STRIP_LENGTH_MM * 0.5
        )
        hit_time = np.ascontiguousarray(group[:, H_ARRIVAL].astype(np.float32))
        wavelength = np.ascontiguousarray(group[:, H_WAVELEN].astype(np.float32))

        # Pre-generate random numbers on CPU
        rng = np.random.default_rng(rng_seed + int(det_idx))
        if config.enable_qe:
            qe_rnd = rng.uniform(0.0, 1.0, n).astype(np.float32)
        else:
            qe_rnd = np.zeros(n, dtype=np.float32)
        charge_rnd = rng.normal(0.0, 1.0, n).astype(np.float32)
        transit_rnd = rng.normal(0.0, 1.0, n).astype(np.float32)

        # Allocate device arrays
        passed = np.zeros(n, dtype=np.float32)
        charges = np.zeros(n, dtype=np.float32)
        transit_times = np.zeros(n, dtype=np.float32)
        strip_charge_map = np.zeros((n, geo.num_strips), dtype=np.float32)
        charge_s0 = np.zeros((n, geo.num_strips), dtype=np.float32)
        charge_s1 = np.zeros((n, geo.num_strips), dtype=np.float32)
        readout_s0 = np.zeros((geo.num_strips, NUM_TIME_BINS), dtype=np.float32)
        readout_s1 = np.zeros((geo.num_strips, NUM_TIME_BINS), dtype=np.float32)

        # ---- Kernel 1: QE ----
        if config.enable_qe:
            apply_qe_kernel(
                wavelength,
                qe_wavelengths,
                qe_values,
                qe_rnd,
                passed,
            )
        else:
            passed.fill(1.0)

        n_passed = int(passed.sum())

        # ---- Kernel 2: Charge ----
        sample_charge_kernel(
            passed,
            charge_rnd,
            config.charge_mean,
            config.charge_std,
            config.charge_min,
            config.charge_max,
            charges,
        )

        # ---- Kernel 3: Transit time ----
        sample_transit_time_kernel(
            passed,
            transit_rnd,
            config.transit_mean,
            config.transit_std,
            config.transit_min,
            config.transit_max,
            transit_times,
        )

        # ---- Kernel 4: Strip charge assignment ----
        assign_strip_charges_kernel(
            passed,
            perp_pos,
            charges,
            geo_strip_centers,
            geo_bounds_left,
            geo_bounds_right,
            config.strip_gauss_sigma,
            config.strip_spread,
            strip_charge_map,
        )

        # ---- Kernel 5: Attenuation ----
        apply_attenuation_kernel(
            passed,
            parallel_pos,
            strip_charge_map,
            geo.strip_length,
            config.max_attenuation_skew,
            charge_s0,
            charge_s1,
        )

        # ---- Kernel 6: Digitization ----
        rise_sigma = config.rise_time_ns_fwhm / 2.355
        signal_speed = SIGNAL_SPEED_FRACTION_C * C_MM_NS

        digitize_kernel(
            passed,
            hit_time,
            transit_times,
            parallel_pos,
            charge_s0,
            charge_s1,
            geo.strip_length,
            signal_speed,
            config.time_step_ns,
            config.total_time_ns,
            rise_sigma,
            config.impedance_ohms,
            readout_s0,
            readout_s1,
        )

        result[int(det_idx)] = {
            "side0": readout_s0,
            "side1": readout_s1,
            "n_photons": n,
            "n_passed_qe": n_passed,
        }

    return result


# ---------------------------------------------------------------------------
# Process hit dicts  (JSON-friendly interface for the viz server)
# ---------------------------------------------------------------------------


def process_hit_dicts(
    hits: list[dict],
    detector_index: int = 0,
    config: LAPPDResponseConfig | None = None,
    rng_seed: int = 42,
    qe_table: np.ndarray | None = None,
    geometry: LAPPDGeometry | None = None,
) -> dict:
    """Run the digitization pipeline on a list of hit dicts (from the viz).

    Parameters
    ----------
    hits : list[dict]
        Each dict has keys ``x``, ``y``, ``z``, ``t``, ``origin``, ``dir``,
        and optionally ``arrival_time`` and ``type`` (only ``"photocathode"``
        hits are processed).
    detector_index : int
        LAPPD detector index to use for grouping.
    config, rng_seed, qe_table, geometry : see ``process_hits()``.

    Returns
    -------
    result : dict
        Same format as ``process_hits()`` return values per single LAPPD,
        or an empty dict if no PC hits were found.
    """
    if config is None:
        config = LAPPDResponseConfig()
    if qe_table is None:
        qe_table = _load_qe_table()
    geo = geometry or _DEFAULT_GEOMETRY

    pc_hits = [h for h in hits if h.get("type") == "photocathode"]
    if not pc_hits:
        return {}

    n = len(pc_hits)

    # Build per-photon arrays from dicts.
    # In the viz, hit positions (x, y, z) are in mm from the detector centre
    # which aligns with the PC centre.  The local coordinate axes match the
    # housing frame: axis_y = vertical = along strips, axis_x = horizontal =
    # across strips.
    #
    # We reconstruct local_u / local_v by projecting the hit position
    # onto the strip axis (vertical = +Y in the default housing frame)
    # and the perpendicular axis (horizontal = +X).

    defaults = {"arrival_time": 0.0, "wavelength": 350.0}

    # For the default housing (center at origin, Z inward, Y up, X tangential):
    #   along strips = Y  (up)
    #   across strips = X (tangential)
    # The PC centre is at (0, -45, 3.5) in housing local frame, but the viz
    # sets up the housing JSON with the box around the CAD centre which is
    # the housing centre, not the PC centre.  The PC hit positions from
    # _trace_hits() are world-frame positions on the PC plane, which is
    # at pc_center = (0, -45, 3.5) relative to housing center.

    # We compute local_u / local_v by taking the hit position relative to
    # PC centre and projecting onto the housing axes.
    # From the housing JSON, axis_y = (0, 1, 0) = vertical = along strips.
    # axis_x = (1, 0, 0) = tangential = across strips.

    hit_time = np.zeros(n, dtype=np.float32)
    wavelength = np.zeros(n, dtype=np.float32)

    # The default housing has PC centre at (0, -45, 3.5).
    # We get the PC centre from the first hit's context.
    # Actually, in the viz, the geometry is fetched from /api/geometry
    # which returns housing JSON with pc_center.  The caller should
    # pass this info.  For now, use defaults.
    # The key insight: the hit position (x, y, z) from _trace_hits() *is*
    # already in world coordinates on the PC plane.  The PC centre is at
    # pc_center = (0, -45, 3.5) in the default housing frame.
    # So the local coords in mm from PC centre are simply:
    #   rel = hit_pos - pc_center
    #   local_u = rel . axis_y (along strips)
    #   local_v = rel . axis_x (across strips)

    # We default to the canonical PC centre for a single-LAPPD viz.
    pc_cx, pc_cy, pc_cz = 0.0, -45.0, 3.5
    # Strip axis (along strips = vertical = +Y)
    ax_y = (0.0, 1.0, 0.0)
    # Perp axis (across strips = tangential = +X)
    ax_x = (1.0, 0.0, 0.0)

    lu_arr = np.zeros(n, dtype=np.float32)
    lv_arr = np.zeros(n, dtype=np.float32)
    for i, h in enumerate(pc_hits):
        rx = h["x"] - pc_cx
        ry = h["y"] - pc_cy
        rz = h["z"] - pc_cz
        lu_arr[i] = rx * ax_x[0] + ry * ax_x[1] + rz * ax_x[2]
        lv_arr[i] = rx * ax_y[0] + ry * ax_y[1] + rz * ax_y[2]
        hit_time[i] = h.get("arrival_time", defaults["arrival_time"])
        wavelength[i] = h.get("wavelength", defaults["wavelength"])

    # Build a synthetic tracer-style hit array.
    # Store PC-centred local coordinates (same convention as the tracer's
    # _lappd_local_coords), so process_hits applies the correct strip-local
    # shift: parallel_pos = HLU + strip_length/2, perp_pos = HLV + pc_half.
    syn = np.zeros((n, 17), dtype=np.float32)
    syn[:, 9] = float(detector_index)  # HDI
    syn[:, 10] = 2.0  # HDS = DET_SYS_LAPPD_ANNIE
    syn[:, 11] = lv_arr  # HLU = PC-centred, along strips
    syn[:, 12] = lu_arr  # HLV = PC-centred, across strips
    syn[:, 14] = hit_time  # H_ARRIVAL
    syn[:, 15] = wavelength  # H_WAVELEN

    return process_hits(
        syn,
        config=config,
        rng_seed=rng_seed,
        qe_table=qe_table,
        geometry=geometry,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LAPPDGeometry",
    "LAPPDResponseConfig",
    "process_hits",
    "process_hit_dicts",
    "NUM_STRIPS",
    "NUM_TIME_BINS",
    "_load_qe_table",
]
