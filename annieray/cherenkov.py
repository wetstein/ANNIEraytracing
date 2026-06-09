"""Cherenkov photon generation for muon tracks in water.

This module is the single place where photon emission is modelled.
Currently implements a simple single-vertex Cherenkov cone.

TO EXTEND THE EMISSION MODEL (student task):
  - Add muon propagation along the track (distribute origins along Z)
  - Add wavelength-dependent Cherenkov angle via dispersion curve n(λ)
  - Add per-photon wavelength sampling (return wavelengths array)
  - Add delta-ray or scintillation components

The output (origins, directions) is consumed by trace_rays() in tracer.py,
which runs the GPU kernel.  Optionally a wavelengths array can be returned
and threaded through trace_cherenkov() to fill the hits[:, 14] column.
"""

import numpy as np

# Nominal Cherenkov angle for n_water = 1.34 at ~350 nm
#   cos(θ_c) = 1 / (n * β)  →  θ_c ≈ arccos(1/1.34) ≈ 0.73 rad ≈ 42°
CHERENKOV_ANGLE = 0.73

# Default wavelength when no per-photon sampling is used (nm)
DEFAULT_WAVELENGTH = 350.0


def generate_cherenkov_photons(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    #Adding in array of wavelengths | Unused currently but will replace the current DEFAULT_WAVELENGTH
    # WaveLengths = np.linspace(10,1400, num = 100) #nm from UV to IR, low spacing for now to reduce computaional load
    n: int,
    cherenkov_angle: float = CHERENKOV_ANGLE,
    rng: np.random.Generator | None = None,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Cherenkov cone photons from a muon track.

    All photons originate at a single vertex (muon_pos) and are emitted
    uniformly within a filled cone of half-angle cherenkov_angle around
    muon_dir.

    Args:
        muon_pos: Muon vertex (x, y, z) in mm.
        muon_dir: Muon direction (does not need to be unit).
        n: Number of photons to generate.
        cherenkov_angle: Cherenkov angle in radians.
        rng: NumPy random generator.
        wavelength: Photon wavelength in nm (default 350).

    Returns:
        (origins, directions) arrays, each (N, 3) float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- Normalise muon direction to unit vector ----
    mdx, mdy, mdz = muon_dir
    m_len = np.sqrt(mdx * mdx + mdy * mdy + mdz * mdz)
    mdx /= m_len
    mdy /= m_len
    mdz /= m_len

    # ---- Build two orthonormal basis vectors (v, w) perpendicular to muon_dir ----
    # We need an arbitrary vector not parallel to muon_dir to start the cross product.
    # If muon_dir is nearly along X, use Y as reference; otherwise use X.
    if abs(mdx) > 0.9:
        ref_x, ref_y, ref_z = 0.0, 1.0, 0.0   # reference = Y axis
    else:
        ref_x, ref_y, ref_z = 1.0, 0.0, 0.0   # reference = X axis

    # v = cross(muon_dir, reference), then normalise
    # v is the first basis vector in the plane perpendicular to muon_dir
    vx = ref_y * mdz - ref_z * mdy
    vy = ref_z * mdx - ref_x * mdz
    vz = ref_x * mdy - ref_y * mdx
    v_len = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= v_len
    vy /= v_len
    vz /= v_len

    # w = cross(muon_dir, v)
    # w is the second basis vector, orthogonal to both muon_dir and v
    wx = mdy * vz - mdz * vy
    wy = mdz * vx - mdx * vz
    wz = mdx * vy - mdy * vx
    # w is already unit since v and muon_dir are unit and orthogonal

    # ---- Generate random angles on the Cherenkov cone ----
    # phi: random azimuth around the muon direction (full circle)
    phi = rng.uniform(0, 2 * np.pi, n)
    # theta: random polar angle within the cone (filled cone, not just
    # the cone surface).  For a thin Cherenkov ring, sample near a fixed
    # angle instead of [0, cherenkov_angle].
    theta = cherenkov_angle #This should be the way to simulate only the Cherenkov Angle for a single muon rather than a filled cone, but the visual model still shows a cone
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # ---- Build photon direction vectors ----
    # Each photon direction = muon_dir * cosθ + (v * cosφ + w * sinφ) * sinθ
    #   - muon_dir * cosθ  →  component along the muon track
    #   - (v*cosφ + w*sinφ) * sinθ  →  perpendicular component, with
    #     random azimuth φ choosing the position around the cone
    origins = np.empty((n, 3), dtype=np.float32)
    directions = np.empty((n, 3), dtype=np.float32)

    # All photons currently start at the same vertex.
    # TODO: propagate muon along track and distribute emission points.
    origins[:, 0] = muon_pos[0]
    origins[:, 1] = muon_pos[1]
    origins[:, 2] = muon_pos[2]

    sp = np.sin(phi)
    cp = np.cos(phi)
    directions[:, 0] = mdx * cos_theta + (vx * cp + wx * sp) * sin_theta
    directions[:, 1] = mdy * cos_theta + (vy * cp + wy * sp) * sin_theta
    directions[:, 2] = mdz * cos_theta + (vz * cp + wz * sp) * sin_theta

    return origins, directions
