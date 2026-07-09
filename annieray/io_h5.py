"""HDF5 I/O helpers for batch output, single-shot hits, and readback."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numpy structured dtypes  (field names → DataFrame column names)
# ---------------------------------------------------------------------------

# Batch photon hits (7 columns)
PHOTON_HIT_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("detector_system", "i4"),
    ("detector_index", "i4"),
    ("local_u", "f4"),
    ("local_v", "f4"),
    ("arrival_time", "f4"),
    ("wavelength", "f4"),
])

# Muon truth (13 columns)
MUON_TRUTH_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("pos_x", "f4"),
    ("pos_y", "f4"),
    ("pos_z", "f4"),
    ("t0", "f4"),
    ("dir_x", "f4"),
    ("dir_y", "f4"),
    ("dir_z", "f4"),
    ("theta_deg", "f4"),
    ("phi_deg", "f4"),
    ("track_length_mm", "f4"),
    ("n_generated", "i4"),
    ("n_detected", "i4"),
])

# PMT responses (5 columns)
PMT_RESPONSE_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("pmt_index", "i4"),
    ("charge", "f4"),
    ("time", "f4"),
    ("n_hits", "i4"),
])

# Detector registry (7 columns)
DETECTOR_DTYPE = np.dtype([
    ("system_code", "i4"),
    ("detector_index", "i4"),
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("label", h5py.string_dtype()),
    ("panel", "i4"),
])

# Full hit table for single-shot mode (17 columns)
FULL_HIT_DTYPE = np.dtype([
    ("hit_flag", "i4"),
    ("t", "f4"),
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("nx", "f4"),
    ("ny", "f4"),
    ("nz", "f4"),
    ("component_id", "i4"),
    ("detector_index", "i4"),
    ("detector_system", "i4"),
    ("local_u", "f4"),
    ("local_v", "f4"),
    ("arrival_time", "f4"),
    ("wavelength", "f4"),
    ("bounce_count", "i4"),
    ("photon_id", "i8"),
])

DTYPES = {
    "photon_hits": PHOTON_HIT_DTYPE,
    "muon_truth": MUON_TRUTH_DTYPE,
    "pmt_responses": PMT_RESPONSE_DTYPE,
    "detectors": DETECTOR_DTYPE,
    "hits": FULL_HIT_DTYPE,
}


# ---------------------------------------------------------------------------
# Append helper (resizable dataset)
# ---------------------------------------------------------------------------

def _ensure_dataset(f: h5py.File, name: str, dtype: np.dtype):
    """Return existing dataset *name* or create a resizable one."""
    if name in f:
        return f[name]
    return f.create_dataset(name, shape=(0,), maxshape=(None,),
                            dtype=dtype, compression="gzip")


def append_table(f: h5py.File, name: str, data: np.ndarray) -> None:
    """Append *data* (a numpy structured array) to a resizable dataset."""
    dset = _ensure_dataset(f, name, data.dtype)
    n = dset.shape[0]
    dset.resize((n + len(data),))
    dset[n:] = data


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def load_table(path: Path | str, table: str) -> pd.DataFrame:
    """Load an HDF5 table as a pandas DataFrame.

    Returns an empty DataFrame if the dataset does not exist.
    """
    with h5py.File(path, "r") as f:
        if table not in f:
            dtype = DTYPES.get(table)
            if dtype is None:
                return pd.DataFrame()
            return pd.DataFrame({n: [] for n in dtype.names})
        return pd.DataFrame(f[table][()])


def load_all(path: Path | str) -> dict[str, pd.DataFrame]:
    """Load all standard batch-output tables."""
    return {key: load_table(path, key) for key in DTYPES}


def read_attrs(path: Path | str) -> dict:
    """Read root-group attributes (e.g. tank metadata)."""
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


# ---------------------------------------------------------------------------
# Write helpers for batch mode
# ---------------------------------------------------------------------------

def write_detectors(path: Path, detectors: list) -> None:
    """Write detector registry to HDF5."""
    rows = []
    for d in detectors:
        from annieray.tracer import DET_SYS_PMT, DET_SYS_LAPPD_DEFAULT, DET_SYS_LAPPD_ANNIE
        sys_code = {
            "pmt": DET_SYS_PMT,
            "lappd_default": DET_SYS_LAPPD_DEFAULT,
            "lappd_annie": DET_SYS_LAPPD_ANNIE,
        }.get(d.system, -1)
        panel = getattr(d, "panel", -1)
        rows.append((sys_code, d.index, d.position[0], d.position[1],
                     d.position[2], str(d.label), panel))
    arr = np.array(rows, dtype=DETECTOR_DTYPE)
    with h5py.File(path, "a") as f:
        append_table(f, "detectors", arr)
    print(f"  Wrote detectors table ({len(rows)} rows) to {path}")


def write_metadata(path: Path, meta: dict) -> None:
    """Write tank metadata as HDF5 root-group attributes."""
    with h5py.File(path, "a") as f:
        for key, val in meta.items():
            f.attrs[key] = val


# ---------------------------------------------------------------------------
# Write helper for single-shot mode
# ---------------------------------------------------------------------------

def write_full_hits(path: Path, hits: np.ndarray,
                    photon_ids: np.ndarray | None = None) -> None:
    """Write (N, 17) hit array to HDF5.

    Matches the previous output.py ``write_hits()`` signature; the
    file extension should be ``.h5`` instead of ``.parquet``.
    """
    from annieray.tracer import (
        HI, HT, HX, HY, HZ, HNX, HNY, HNZ,
        HCID, HDI, HDS, HLU, HLV, HMAT,
        N_HIT_COLS, H_ARRIVAL, H_WAVELEN, H_BOUNCE, N_EXPANDED_COLS,
    )

    n = hits.shape[0]
    ncols = hits.shape[1]
    if photon_ids is None:
        photon_ids = np.arange(n, dtype=np.int64)

    if ncols < N_EXPANDED_COLS:
        full = np.full((n, N_EXPANDED_COLS), np.nan, dtype=np.float32)
        full[:, :N_HIT_COLS] = hits
        full[:, H_BOUNCE] = 0
    else:
        full = hits

    arr = np.zeros(n, dtype=FULL_HIT_DTYPE)
    arr["hit_flag"] = full[:, HI].astype(np.int32)
    arr["t"] = full[:, HT].astype(np.float32)
    arr["x"] = full[:, HX].astype(np.float32)
    arr["y"] = full[:, HY].astype(np.float32)
    arr["z"] = full[:, HZ].astype(np.float32)
    arr["nx"] = full[:, HNX].astype(np.float32)
    arr["ny"] = full[:, HNY].astype(np.float32)
    arr["nz"] = full[:, HNZ].astype(np.float32)
    arr["component_id"] = full[:, HCID].astype(np.int32)
    arr["detector_index"] = full[:, HDI].astype(np.int32)
    arr["detector_system"] = full[:, HDS].astype(np.int32)
    arr["local_u"] = full[:, HLU].astype(np.float32)
    arr["local_v"] = full[:, HLV].astype(np.float32)
    arr["arrival_time"] = full[:, H_ARRIVAL].astype(np.float32)
    arr["wavelength"] = full[:, H_WAVELEN].astype(np.float32)
    arr["bounce_count"] = full[:, H_BOUNCE].astype(np.int32)
    arr["photon_id"] = photon_ids

    with h5py.File(path, "w") as f:
        f.create_dataset("hits", data=arr, compression="gzip")
