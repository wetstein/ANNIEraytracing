"""Parquet output for ray tracer hit data and detector registry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# 16-column hit schema
#
# Columns 0-13 come from the GPU kernel (see tracer.py for column indices).
# Columns 14-16 are added by trace_cherenkov() in tracer.py:
#   14 = arrival_time (ns) — total photon path length / (C / n_water)
#   15 = wavelength (nm)
#   16 = bounce_count       — number of surface reflections (0 = direct)
HIT_SCHEMA = pa.schema([
    ("hit_flag", pa.int32()),
    ("t", pa.float32()),
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("z", pa.float32()),
    ("nx", pa.float32()),
    ("ny", pa.float32()),
    ("nz", pa.float32()),
    ("component_id", pa.int32()),
    ("detector_index", pa.int32()),
    ("detector_system", pa.int32()),
    ("local_u", pa.float32()),
    ("local_v", pa.float32()),
    ("arrival_time", pa.float32()),
    ("wavelength", pa.float32()),
    ("bounce_count", pa.int32()),
])


def write_hits(
    hits: np.ndarray,
    path: Path,
    photon_ids: Optional[np.ndarray] = None,
) -> None:
    """Write (N, 17) hit array to Parquet.

    If hits has only the kernel columns (N_HIT_COLS=14), expanded
    columns (arrival_time, wavelength, bounce_count) are filled
    with NaN / 0.
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

    # Expand to N_EXPANDED_COLS if needed
    if ncols < N_EXPANDED_COLS:
        full = np.full((n, N_EXPANDED_COLS), np.nan, dtype=np.float32)
        full[:, :N_HIT_COLS] = hits
        full[:, H_BOUNCE] = 0  # no optics info → zero bounces
    else:
        full = hits

    table = pa.table({
        "hit_flag": full[:, HI].astype(np.int32),
        "t": full[:, HT].astype(np.float32),
        "x": full[:, HX].astype(np.float32),
        "y": full[:, HY].astype(np.float32),
        "z": full[:, HZ].astype(np.float32),
        "nx": full[:, HNX].astype(np.float32),
        "ny": full[:, HNY].astype(np.float32),
        "nz": full[:, HNZ].astype(np.float32),
        "component_id": full[:, HCID].astype(np.int32),
        "detector_index": full[:, HDI].astype(np.int32),
        "detector_system": full[:, HDS].astype(np.int32),
        "local_u": full[:, HLU].astype(np.float32),
        "local_v": full[:, HLV].astype(np.float32),
        "arrival_time": full[:, H_ARRIVAL].astype(np.float32),
        "wavelength": full[:, H_WAVELEN].astype(np.float32),
        "bounce_count": full[:, H_BOUNCE].astype(np.int32),
        "photon_id": photon_ids,
    })
    pq.write_table(table, str(path))


def write_detector_config(detectors: list, path: Path) -> None:
    """Write detector registry to YAML."""
    from annieray.detectors import detector_config_to_yaml
    detector_config_to_yaml(detectors, path)
