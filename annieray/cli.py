"""CLI for annieray ray tracer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from annieray.tracer import (
    build_geometry,
    trace_rays,
    trace_cherenkov,
    Geometry,
)
from annieray.output import write_hits, write_detector_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annieray",
        description="GPU-accelerated ray tracer for ANNIE detector",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run ray tracing")
    run.add_argument("--gdml", required=True, type=Path, help="Path to InnerStructure.gdml")
    run.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    run.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    run.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file or CSV (default: PMTPositions_Scan.txt near GDML)")
    run.add_argument("--photons", type=int, default=100000, help="Number of photons to trace")
    run.add_argument("--output", "-o", type=Path, default=Path("hits.parquet"), help="Output Parquet path")
    run.add_argument("--seed", type=int, default=None, help="Random seed")
    run.add_argument("--mode", choices=["uniform", "cherenkov"], default="uniform",
                     help="Photon generation mode")
    run.add_argument("--lappd-indices", type=str, default=None,
                     help="Comma-separated LAPPD candidate indices (default: use 3 from manifest)")
    run.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    run.add_argument("--z-offset", type=float, default=0.0,
                     help="Vertical offset (mm) — use when PMT CSV and GDML use different Z origins")
    run.add_argument("--lappd-model", choices=["default", "annie"], default="default",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")
    run.add_argument("--detector-config", type=Path, default=None,
                     help="Path to detector registry YAML (auto-built if absent)")
    run.add_argument("--wavelength", type=float, default=350.0,
                     help="Photon wavelength in nm (default: 350)")

    cherenkov = sub.add_parser("extract-manifest", help="Extract component manifest from STEP")
    cherenkov.add_argument("--step", required=True, type=Path, help="Path to STEP CAD file")
    cherenkov.add_argument("--output", "-o", type=Path, default=Path("component_manifest.json"), help="Output JSON path")

    viz = sub.add_parser("viz-server", help="Start interactive Cherenkov visualization server")
    viz.add_argument("--gdml", required=True, type=Path, help="Path to InnerStructure.gdml")
    viz.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file for component positions")
    viz.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    viz.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    viz.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    viz.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    viz.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    viz.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    viz.add_argument("--lappd-model", choices=["default", "annie"], default="default",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")

    detcfg = sub.add_parser("build-detector-config",
                            help="Build detector registry YAML from geometry")
    detcfg.add_argument("--gdml", required=True, type=Path, help="Path to InnerStructure.gdml")
    detcfg.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    detcfg.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    detcfg.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    detcfg.add_argument("--output", "-o", type=Path, default=Path("detectors.yaml"),
                        help="Output YAML path (default: detectors.yaml)")
    detcfg.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    detcfg.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    detcfg.add_argument("--lappd-model", choices=["default", "annie"], default="default",
                        help="LAPPD geometry model")

    lappd = sub.add_parser("viz-lappd", help="Start standalone LAPPD module viewer")
    lappd.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    lappd.add_argument("--port", type=int, default=8081, help="Port to bind (default: 8081)")

    return p


def _generate_uniform(geometry: Geometry, n: int, rng: np.random.Generator) -> tuple:
    """Generate random photons uniformly inside the tank volume.

    Positions are rejection-sampled inside a cylinder 90 % of the tank
    radius.  Directions are isotropic (uniform on the sphere).

    This is useful for flat efficiency scans — each photon has equal
    probability of hitting any detector regardless of emission model.

    To add a new emission model, create a function with the same
    signature (geometry, n, rng) → (origins, directions) and wire
    it into run_command() as a new --mode branch.

    Returns (origins, directions) arrays, each (N, 3) float32.
    """
    r_tank = geometry.tank_radius * 0.9
    z_min = geometry.tank_z_min + 100.0
    z_max = geometry.tank_z_max - 100.0

    origins = np.zeros((n, 3), dtype=np.float32)
    directions = np.zeros((n, 3), dtype=np.float32)

    # Rejection sampling for uniform positions in cylinder
    batch = int(n * 1.3) + 1
    accepted = 0
    while accepted < n:
        xs = rng.uniform(-r_tank, r_tank, batch)
        ys = rng.uniform(-r_tank, r_tank, batch)
        zs = rng.uniform(z_min, z_max, batch)
        rs2 = xs * xs + ys * ys
        mask = rs2 <= r_tank * r_tank
        n_mask = mask.sum()
        to_take = min(n_mask, n - accepted)
        if to_take == 0:
            continue
        origins[accepted:accepted + to_take, 0] = xs[mask][:to_take]
        origins[accepted:accepted + to_take, 1] = ys[mask][:to_take]
        origins[accepted:accepted + to_take, 2] = zs[mask][:to_take]
        accepted += to_take

    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    directions[:, 0] = np.cos(theta) * np.cos(phi)
    directions[:, 1] = np.sin(theta) * np.cos(phi)
    directions[:, 2] = np.sin(phi)

    return origins, directions

#Call generate_cherenkov for each muon, make a seperate function for multiple muon generation
def _generate_cherenkov(
    geometry: Geometry,
    n: int,
    rng: np.random.Generator,
    muon_pos: tuple = (0.0, 0.0, 2000.0,0.0), #Tank coordinate system (cartesian based on the cylinder) where z is height, the last component is time in ns
    muon_dir: tuple = (0.0, 0.0, -1.0),
    cherenkov_angle: float = 0.73,
) -> tuple:
    """Generate Cherenkov cone from a muon track (delegates to cherenkov module).

    This is a thin wrapper that calls generate_cherenkov_photons() in
    cherenkov.py.  The student modifying the emission model should work
    in cherenkov.py directly; this function just passes through.

    Returns (origins, directions) arrays, each (N, 3) float32.
    """
    from annieray.cherenkov import generate_cherenkov_photons

    return generate_cherenkov_photons(
        muon_pos, muon_dir, n,
        cherenkov_angle=cherenkov_angle, rng=rng,
    )


def run_command(args: argparse.Namespace) -> None:
    # Validate args
    if args.pmt_csv and not args.pmt_csv.exists():
        print(f"Error: PMT CSV not found: {args.pmt_csv}")
        return
    if not args.pmt_csv and not args.step and not args.manifest:
        print("Warning: no PMT data source provided (--pmt-csv, --step, or --manifest)")

    # Handle lappd-indices override
    lappd_indices = None
    if args.lappd_indices:
        import json
        lappd_indices = [int(x) for x in args.lappd_indices.split(",")]

    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, lappd_indices=lappd_indices,
                          no_lappd=args.no_lappd, z_offset=args.z_offset, lappd_model=args.lappd_model)

    print(f"  Mesh: {geom.mesh_vertices.shape[0]} vertices, {geom.mesh_triangles.shape[0]} triangles")
    print(f"  PMTs: {geom.pmt_centers.shape[0]} (radii: {set(f'{r:.1f}' for r in geom.pmt_radii)})")
    print(f"  LAPPDs: {geom.lappd_data.shape[0]}")
    if geom.lappd_housing_data.shape[0] > 0:
        print(f"  ANNIE LAPPD housing: 1 at centre {geom.lappd_housing_data[0, :3]}")
    if geom.detectors:
        print(f"  Detector registry: {len(geom.detectors)} entries")
    print(f"  Tank: R={geom.tank_radius:.0f} mm, Z=[{geom.tank_z_min:.0f}, {geom.tank_z_max:.0f}]")

    # Write detector config if requested
    if args.detector_config:
        print(f"\nWriting detector registry to {args.detector_config}...")
        write_detector_config(geom.detectors, args.detector_config)

    rng = np.random.default_rng(args.seed)

    print(f"\nGenerating {args.photons} photons ({args.mode} mode)...")
    t0 = time.time()
    if args.mode == "uniform":
        origins, directions = _generate_uniform(geom, args.photons, rng)
        hits = trace_rays(origins, directions, geom)
    else:
        hits = trace_cherenkov(
            (0.0, 0.0, 2000.0), (0.0, 0.0, -1.0),
            args.photons, geom, rng=rng,
            wavelength_nm=args.wavelength,
        )
    t_gen = time.time() - t0
    print(f"  Generated/traced in {t_gen:.2f}s")

    n_hit = int(hits[:, 0].sum())
    print(f"\nResults: {n_hit}/{args.photons} hit ({n_hit / args.photons * 100:.1f}%)")

    from annieray.tracer import CID_PMT, CID_LAPPD, CID_INNER_STRUCTURE, CID_TANK_WALL
    cid_names = {CID_INNER_STRUCTURE: "structure", CID_PMT: "PMT", CID_LAPPD: "LAPPD", CID_TANK_WALL: "tank"}
    for cid in range(1, 5):
        count = int((hits[:, 8] == cid).sum())
        if count > 0:
            print(f"  {cid_names[cid]}: {count}")

    n_det = int((hits[:, 9] >= 0).sum())
    if n_det > 0:
        print(f"  Detector hits: {n_det}")

    print(f"\nWriting {args.output}...")
    write_hits(hits, args.output)
    print("Done.")


def build_detector_config_command(args: argparse.Namespace) -> None:
    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, no_lappd=args.no_lappd,
                          z_offset=args.z_offset, lappd_model=args.lappd_model)

    print(f"  Built {len(geom.detectors)} detectors:")
    for d in geom.detectors:
        print(f"    {d.id:5d}  {d.system:16s}  {d.label}")

    print(f"\nWriting {args.output}...")
    write_detector_config(geom.detectors, args.output)
    print("Done.")


def extract_manifest_command(args: argparse.Namespace) -> None:
    from annieray.step_parser import parse_step

    print(f"Parsing {args.step}...")
    t0 = time.time()
    manifest = parse_step(args.step)
    manifest.to_json(args.output)
    t_elapsed = time.time() - t0
    print(f"Extracted {len(manifest.pmt_centers)} PMT candidates, "
          f"{len(manifest.lappd_candidates)} LAPPD candidates")
    print(f"Saved to {args.output} in {t_elapsed:.1f}s")


def main(argv: list[str] | None = None) -> None:
    import taichi as ti
    ti.init(default_fp=ti.f32)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_command(args)
    elif args.command == "extract-manifest":
        extract_manifest_command(args)
    elif args.command == "build-detector-config":
        build_detector_config_command(args)
    elif args.command == "viz-server":
        from annieray.viz_server import run_server

        run_server(args)
    elif args.command == "viz-lappd":
        from annieray.viz_lappd_server import run_server as run_lappd_server

        run_lappd_server(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
