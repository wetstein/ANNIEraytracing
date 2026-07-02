"""Detector registry: flexible ID scheme, YAML config, and builder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

LAPPD_ID_OFFSET_DEFAULT = 1000
LAPPD_ID_OFFSET_ANNIE = 2000


@dataclass
class DetectorInfo:
    """Stable record for one photosensor in the ANNIE detector.

    Each detector has a unique ID that persists across runs, independent
    of array indices in the GPU kernel.  This lets analysis code match
    hits to hardware without relying on geometry construction order.

    ID ranges:
        PMT:           332–463 (WCSim TubeIDs)
        Default LAPPD: 1000+  (one index per rectangle)
        ANNIE LAPPD:   2000+  (one index per housed LAPPD)
    """

    id: int                                  # stable unique ID
    system: str                              # "pmt" | "lappd_default" | "lappd_annie"
    label: str                               # human-readable name, e.g. "PMT_332"
    index: int                               # position in geometry arrays (-1 if loaded from YAML)
    position: tuple[float, float, float]     # centre in structure frame (mm)
    direction: tuple[float, float, float]    # inward-pointing unit normal

    # PMT-specific fields
    panel: int = -1                          # panel number 0-9 (0=bottom LUX, 9=top ETEL)
    pmt_type: str = ""                       # LUX, ETEL, Hamamatsu, Watchboy, Watchman
    radius: float = 0.0                      # PMT sphere radius (mm)

    # LAPPD-specific fields
    half_size: float = 0.0                   # photocathode half-side length (mm)
    strip_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)  # LAPPD strip direction (unit vector)


def build_detector_registry(
    pmt_centers: np.ndarray,
    pmt_radii: np.ndarray,
    pmt_types: list[str],
    pmt_directions: np.ndarray,
    pmt_detector_nums: list[int],
    pmt_panels: Optional[list[int]] = None,
    lappd_rect_data: Optional[np.ndarray] = None,
    lappd_housing_data: Optional[np.ndarray] = None,
    annie_lappd_data: Optional[np.ndarray] = None,
) -> list[DetectorInfo]:
    """Build the detector registry from raw geometry arrays."""
    detectors: list[DetectorInfo] = []

    # PMTs
    for i in range(len(pmt_centers)):
        pos = tuple(float(x) for x in pmt_centers[i])
        d = tuple(float(x) for x in pmt_directions[i])
        det_num = pmt_detector_nums[i] if i < len(pmt_detector_nums) else 0
        panel = pmt_panels[i] if pmt_panels and i < len(pmt_panels) else -1
        detectors.append(DetectorInfo(
            id=det_num,
            system="pmt",
            label=f"PMT_{det_num}",
            index=i,
            position=pos,
            direction=d,
            panel=panel,
            pmt_type=pmt_types[i] if i < len(pmt_types) else "",
            radius=float(pmt_radii[i]),
        ))

    # Default LAPPD rectangles
    n_pmts = len(pmt_centers)
    n_lappds = lappd_rect_data.shape[0] if lappd_rect_data is not None else 0
    if lappd_rect_data is not None:
        for i in range(n_lappds):
            pos = (float(lappd_rect_data[i, 0]),
                   float(lappd_rect_data[i, 1]),
                   float(lappd_rect_data[i, 2]))
            nd = (float(lappd_rect_data[i, 3]),
                  float(lappd_rect_data[i, 4]),
                  float(lappd_rect_data[i, 5]))
            half = float(lappd_rect_data[i, 6])
            det_id = LAPPD_ID_OFFSET_DEFAULT + i
            detectors.append(DetectorInfo(
                id=det_id,
                system="lappd_default",
                label=f"LAPPD_DEFAULT_{i}",
                index=n_pmts + i,
                position=pos,
                direction=nd,
                half_size=half,
                strip_axis=(0.0, 0.0, 1.0),
            ))

    # ANNIE LAPPD housing
    n_housings = lappd_housing_data.shape[0] if lappd_housing_data is not None else 0
    if lappd_housing_data is not None and n_housings > 0:
        for i in range(n_housings):
            hc = (float(lappd_housing_data[i, 0]),
                  float(lappd_housing_data[i, 1]),
                  float(lappd_housing_data[i, 2]))
            a_y = (float(lappd_housing_data[i, 6]),
                   float(lappd_housing_data[i, 7]),
                   float(lappd_housing_data[i, 8]))
            a_z = (float(lappd_housing_data[i, 9]),
                   float(lappd_housing_data[i, 10]),
                   float(lappd_housing_data[i, 11]))

            pc_pos = (float(annie_lappd_data[i, 0]),
                      float(annie_lappd_data[i, 1]),
                      float(annie_lappd_data[i, 2]))
            pc_nd = (float(annie_lappd_data[i, 3]),
                     float(annie_lappd_data[i, 4]),
                     float(annie_lappd_data[i, 5]))
            pc_half = float(annie_lappd_data[i, 6])

            det_id = LAPPD_ID_OFFSET_ANNIE + i
            detectors.append(DetectorInfo(
                id=det_id,
                system="lappd_annie",
                label=f"LAPPD_ANNIE_{i}",
                index=n_pmts + n_lappds + i,
                position=pc_pos,
                direction=pc_nd,
                half_size=pc_half,
                strip_axis=a_y,
            ))

    return detectors


def detector_config_to_yaml(detectors: list[DetectorInfo], path: Path) -> None:
    """Write detector registry to YAML."""
    data = []
    for d in detectors:
        entry = {
            "id": d.id,
            "system": d.system,
            "label": d.label,
            "position_mm": [round(v, 1) for v in d.position],
            "direction": [round(v, 6) for v in d.direction],
        }
        if d.system == "pmt":
            entry["panel"] = d.panel
            entry["pmt_type"] = d.pmt_type
            entry["radius_mm"] = d.radius
        else:
            entry["half_size_mm"] = d.half_size
            if d.system == "lappd_annie":
                entry["strip_axis"] = [round(v, 6) for v in d.strip_axis]

        data.append(entry)

    with open(path, "w") as f:
        yaml.dump({"detectors": data}, f, default_flow_style=None, sort_keys=False)


def detector_config_from_yaml(path: Path) -> list[DetectorInfo]:
    """Read detector registry from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    detectors: list[DetectorInfo] = []
    for entry in raw.get("detectors", []):
        pos = tuple(entry["position_mm"])
        nd = tuple(entry["direction"])
        half = entry.get("half_size_mm", 0.0)
        strip = tuple(entry.get("strip_axis", (0.0, 0.0, 1.0)))
        radius = entry.get("radius_mm", 0.0)

        detectors.append(DetectorInfo(
            id=entry["id"],
            system=entry["system"],
            label=entry.get("label", f"DET_{entry['id']}"),
            index=-1,
            position=pos,
            direction=nd,
            panel=entry.get("panel", -1),
            pmt_type=entry.get("pmt_type", ""),
            radius=radius,
            half_size=half,
            strip_axis=strip,
        ))

    return detectors
