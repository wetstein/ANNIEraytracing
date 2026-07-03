"""Generate a MuonStartsAndDirecs file with configurable angular scan.

For each (x,z) position in the standard grid, creates n×n muon
directions: evenly spaced angles from −half_range to +half_range
in both the vertical (Y-Z) and horizontal (X-Y) planes.

When n=1 the direction is simply (0, 1, 0) — no angular spread.

Output format: <x> <y> <z> <t0> <dx> <dy> <dz>

Usage:
    python scripts/generate_muon_grid.py > MuonStartsAndDirecs_angled.txt
    python scripts/generate_muon_grid.py --n-steps 5 --half-range 45 > MuonStartsAndDirecs_5x5.txt
    python scripts/generate_muon_grid.py --n-steps 1 > MuonStartsAndDirecs_1x1.txt
"""

import argparse
import itertools
import math

NX = 13
NZ = 13
X_MIN, X_MAX = -1200.0, 1200.0
Z_MIN, Z_MAX = 300.0, 2700.0
Y = 0.0
T0 = 0.0


def direction(vert_deg: float, horiz_deg: float):
    v = math.radians(vert_deg)
    h = math.radians(horiz_deg)
    cv, sv = math.cos(v), math.sin(v)
    ch, sh = math.cos(h), math.sin(h)
    dx = cv * sh
    dy = cv * ch
    dz = sv
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / norm, dy / norm, dz / norm


def main():
    parser = argparse.ArgumentParser(
        description="Generate muon grid with angular scan"
    )
    parser.add_argument("--n-steps", type=int, default=3,
                        help="Number of angle steps per axis (default 3)")
    parser.add_argument("--half-range", type=float, default=22.5,
                        help="Half-range of angles in degrees (default 22.5)")
    args = parser.parse_args()

    n = args.n_steps
    half = args.half_range

    if n < 1:
        parser.error("--n-steps must be >= 1")

    if n == 1:
        angles = [0.0]
    elif n % 2 == 1:
        step = half * 2 / (n - 1)
        angles = [-half + i * step for i in range(n)]
    else:
        step = half * 2 / (n - 1)
        angles = [-half + i * step for i in range(n)]

    xs = [X_MIN + i * (X_MAX - X_MIN) / (NX - 1) for i in range(NX)]
    zs = [Z_MIN + i * (Z_MAX - Z_MIN) / (NZ - 1) for i in range(NZ)]

    for x, z in itertools.product(xs, zs):
        for vert, horiz in itertools.product(angles, repeat=2):
            if n == 1:
                dx, dy, dz = 0.0, 1.0, 0.0
            else:
                dx, dy, dz = direction(vert, horiz)
            print(f"{x:g} {Y:g} {z:g} {T0:g} {dx:g} {dy:g} {dz:g}")


if __name__ == "__main__":
    main()
