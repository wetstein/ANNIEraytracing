"""Output for ray tracer hit data and detector registry (HDF5)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def write_hits(
    hits: np.ndarray,
    path: Path,
    photon_ids: Optional[np.ndarray] = None,
) -> None:
    """Write (N, 17) hit array to HDF5.

    Delegates to ``io_h5.write_full_hits()``.  The *path* should
    end in ``.h5``.
    """
    from annieray.io_h5 import write_full_hits
    write_full_hits(path, hits, photon_ids=photon_ids)


def write_detector_config(detectors: list, path: Path) -> None:
    """Write detector registry to YAML."""
    from annieray.detectors import detector_config_to_yaml
    detector_config_to_yaml(detectors, path)
