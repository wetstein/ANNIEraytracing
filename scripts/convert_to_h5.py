"""Convert old-format batch directories (parquet + CSV + JSON) to single output.h5.

Usage:
    python scripts/convert_to_h5.py results/
    python scripts/convert_to_h5.py results/ results_angled/
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from annieray.io_h5 import (
    PHOTON_HIT_DTYPE,
    MUON_TRUTH_DTYPE,
    PMT_RESPONSE_DTYPE,
    DETECTOR_DTYPE,
    append_table,
)

TABLE_COLUMNS = {
    "photon_hits": PHOTON_HIT_DTYPE,
    "muon_truth": MUON_TRUTH_DTYPE,
    "pmt_responses": PMT_RESPONSE_DTYPE,
    "detectors": DETECTOR_DTYPE,
}

PARQUET_TABLES = ["photon_hits", "muon_truth", "pmt_responses"]


def convert_dir(dir_path: Path) -> Path:
    h5_path = dir_path / "output.h5"
    print(f"\n=== {dir_path}  ->  {h5_path} ===")

    if h5_path.exists():
        print("  output.h5 already exists, skipping.")
        return h5_path

    # 1) Read Parquet tables
    for table_name in PARQUET_TABLES:
        pq_file = dir_path / f"{table_name}.parquet"
        if not pq_file.exists():
            continue
        df = pd.read_parquet(str(pq_file))
        dtype = TABLE_COLUMNS[table_name]
        arr = df.to_records(index=False).astype(dtype)
        with h5py.File(h5_path, "a") as f:
            append_table(f, table_name, arr)
        print(f"  {table_name}: {len(df)} rows")

    # 2) Read detectors CSV
    csv_path = dir_path / "detectors.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        dtype = TABLE_COLUMNS["detectors"]
        rows = []
        for _, row in df.iterrows():
            label = str(row.get("label", row.get("Label", "")))
            panel = int(row.get("panel", row.get("Panel", -1)))
            rows.append((
                int(row["system_code"]),
                int(row["detector_index"]),
                float(row["x"]), float(row["y"]), float(row["z"]),
                label, panel,
            ))
        arr = np.array(rows, dtype=dtype)
        with h5py.File(h5_path, "a") as f:
            append_table(f, "detectors", arr)
        print(f"  detectors: {len(rows)} rows")

    # 3) Read metadata JSON as root-group attributes
    json_path = dir_path / "metadata.json"
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        with h5py.File(h5_path, "a") as f:
            for key, val in meta.items():
                f.attrs[key] = val
        print(f"  metadata: {len(meta)} attributes")

    print(f"  Done — {h5_path}")
    return h5_path


def main():
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        print("Usage: python scripts/convert_to_h5.py <dir> [<dir> ...]")
        sys.exit(1)
    for d in dirs:
        if d.is_dir():
            convert_dir(d)
        else:
            print(f"  Skipping {d} (not a directory)")


if __name__ == "__main__":
    main()
