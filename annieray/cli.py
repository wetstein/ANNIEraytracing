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
    DET_SYS_PMT, DET_SYS_LAPPD_DEFAULT, DET_SYS_LAPPD_ANNIE,
)
from annieray.optics import load_optics_config
from annieray.output import write_hits, write_detector_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annieray",
        description="GPU-accelerated ray tracer for ANNIE detector",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run ray tracing")
    run.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    run.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    run.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    run.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file or CSV (default: PMTPositions_Scan.txt near GDML)")
    run.add_argument("--photons-per-cm", type=int, default=150, help="Photons per cm along the 4 m muon track (total ≈ 401 × this value)")
    run.add_argument("--output", "-o", type=Path, default=Path("hits.parquet"), help="Output Parquet path")
    run.add_argument("--seed", type=int, default=None, help="Random seed")
    run.add_argument("--mode", choices=["uniform", "cherenkov"], default="uniform",
                     help="Photon generation mode")
    run.add_argument("--lappd-indices", type=str, default=None,
                     help="Comma-separated LAPPD candidate indices (default: use 3 from manifest)")
    run.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    run.add_argument("--z-offset", type=float, default=0.0,
                     help="Vertical offset (mm) — use when PMT CSV and GDML use different Z origins")
    run.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")
    run.add_argument("--detector-config", type=Path, default=None,
                     help="Path to detector registry YAML (auto-built if absent)")
    run.add_argument("--wavelength", type=float, default=350.0,
                     help="Photon wavelength in nm (default: 350)")
    run.add_argument("--det-rotation", type=float, default=22.5,
                     help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")
    run.add_argument("--max-bounces", type=int, default=0,
                     help="Number of surface reflections per photon (0 = off, default: 0)")
    run.add_argument("--optics-config", type=Path, default=None,
                     help="YAML file with per-material optical properties (default: built-in)")

    cherenkov = sub.add_parser("extract-manifest", help="Extract component manifest from STEP")
    cherenkov.add_argument("--step", required=True, type=Path, help="Path to STEP CAD file")
    cherenkov.add_argument("--output", "-o", type=Path, default=Path("component_manifest.json"), help="Output JSON path")

    viz = sub.add_parser("viz-server", help="Start interactive Cherenkov visualization server")
    viz.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    viz.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file for component positions")
    viz.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    viz.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    viz.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    viz.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    viz.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    viz.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    viz.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")
    viz.add_argument("--bottom-rot", type=float, default=45.0,
                     help="Extra Z-rotation (deg) for all PMTs, aligning scan mesh with structure (default: 45)")
    viz.add_argument("--bottom-spin", type=float, default=22.5,
                     help="Per-PMT spin (deg) about forward axis for bottom PMTs (default: 22.5)")
    viz.add_argument("--det-rotation", type=float, default=22.5,
                     help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")
    viz.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                     help="Number of obscurant PVC surfboards (0, 1, or 3)")

    detcfg = sub.add_parser("build-detector-config",
                            help="Build detector registry YAML from geometry")
    detcfg.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                        help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    detcfg.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    detcfg.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    detcfg.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    detcfg.add_argument("--output", "-o", type=Path, default=Path("detectors.yaml"),
                        help="Output YAML path (default: detectors.yaml)")
    detcfg.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    detcfg.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    detcfg.add_argument("--lappd-model", choices=["default", "annie"], default="default",
                        help="LAPPD geometry model")
    detcfg.add_argument("--det-rotation", type=float, default=22.5,
                        help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")

    lappd = sub.add_parser("viz-lappd", help="Start standalone LAPPD module viewer")
    lappd.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    lappd.add_argument("--port", type=int, default=8081, help="Port to bind (default: 8081)")

    batch = sub.add_parser("batch", help="Batch-mode event generation")
    batch.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                       help="Path to GDML geometry mesh")
    batch.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    batch.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    batch.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file or CSV")
    batch.add_argument("--events", type=int, default=-1,
                        help="Number of events (default: 100, or auto-count from --muon-file)")
    batch.add_argument("--muon-fixed", type=str, default=None,
                       help="Fixed muon topology: 'x y z t0 dx dy dz' (7 floats)")
    batch.add_argument("--muon-file", type=Path, default=None,
                       help="File with one topology per line: 'x y z t0 dx dy dz'")
    batch.add_argument("--muon-mode", type=str, default="isotropic",
                       choices=["downward", "isotropic", "beam"],
                       help="Muon direction sampling (default: isotropic). "
                            "'downward' = mostly -Z (atmospheric), "
                            "'isotropic' = uniform on sphere (default), "
                            "'beam' = forward along +Y (beam direction)")
    batch.add_argument("--photons-per-cm", type=int, default=150,
                       help="Photons per cm along the muon track")
    batch.add_argument("--batch-size", type=int, default=50,
                       help="Events per GPU launch (default: 50, higher = faster)")
    batch.add_argument("--output-dir", "-o", type=Path, default=Path("results"),
                       help="Output directory for Parquet files")
    batch.add_argument("--no-record", action="store_true",
                       help="Skip writing per-event output files")
    batch.add_argument("--pmt-response", action="store_true",
                       help="Enable PMT digital model")
    batch.add_argument("--full-wf", action="store_true",
                       help="Use full waveform path for PMT response")
    batch.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    batch.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                       help="Number of obscurant PVC surfboards (0, 1, or 3)")
    batch.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    batch.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                       help="LAPPD geometry model")
    batch.add_argument("--det-rotation", type=float, default=22.5,
                       help="Global Z-rotation (deg) so +Y aligns with octagon corner")
    batch.add_argument("--lappd-indices", type=str, default=None,
                       help="Comma-separated LAPPD candidate indices")
    batch.add_argument("--wavelength", type=float, default=350.0,
                       help="Photon wavelength in nm (default: 350)")
    batch.add_argument("--max-bounces", type=int, default=0,
                       help="Number of surface reflections per photon (0 = off)")
    batch.add_argument("--optics-config", type=Path, default=None,
                       help="YAML file with per-material optical properties")
    batch.add_argument("--seed", type=int, default=None, help="Random seed")

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
    photons_per_cm: int,
    rng: np.random.Generator,
    muon_pos: tuple = (0.0, 0.0, 2000.0), #Tank coordinate system (cartesian based on the cylinder) where z is height
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

    origins, directions, create_times = generate_cherenkov_photons(
        muon_pos, muon_dir, photons_per_cm,
        cherenkov_angle=cherenkov_angle, rng=rng,
    )
    return origins, directions


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
                          no_lappd=args.no_lappd, z_offset=args.z_offset, lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation, n_surfboards=args.surfboard)

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

    if args.max_bounces > 0 and args.mode == "cherenkov":
        print(f"\nMulti-bounce optics enabled: max {args.max_bounces} reflections per photon")

    total_photons = args.photons_per_cm * 401
    print(f"\nGenerating {args.photons_per_cm} photons/cm × 401 steps = {total_photons} total ({args.mode} mode)...")
    t0 = time.time()
    if args.mode == "uniform":
        origins, directions = _generate_uniform(geom, total_photons, rng)
        hits = trace_rays(origins, directions, geom)
    else:
        optics_cfg = load_optics_config(args.optics_config) if args.max_bounces > 0 else None
        hits = trace_cherenkov(
            (0.0, 0.0, 2000.0), (0.0, 0.0, -1.0),
            args.photons_per_cm, geom, rng=rng,
            wavelength_nm=args.wavelength,
            max_bounces=args.max_bounces,
            optics_config=optics_cfg,
        )
    t_gen = time.time() - t0
    print(f"  Generated/traced in {t_gen:.2f}s")

    n_hit = int(hits[:, 0].sum())
    print(f"\nResults: {n_hit}/{total_photons} hit ({n_hit / total_photons * 100:.1f}%)")

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
                          z_offset=args.z_offset, lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation)

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


def _count_muon_lines(path: Path) -> int:
    """Count valid topology lines in a muon file (skips blanks and comments)."""
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                n += 1
    return n


def batch_command(args: argparse.Namespace) -> None:
    from annieray.batch import BatchConfig, run_batch

    # Auto-detect event count from muon file
    if args.events == -1:
        if args.muon_file and args.muon_file.exists():
            args.events = _count_muon_lines(args.muon_file)
            print(f"Auto-detected {args.events} events from {args.muon_file}")
        else:
            args.events = 100

    # Parse muon-fixed
    muon_fixed = None
    if args.muon_fixed:
        parts = args.muon_fixed.split()
        if len(parts) != 7:
            print("Error: --muon-fixed requires 7 floats: x y z t0 dx dy dz")
            return
        muon_fixed = tuple(float(p) for p in parts)

    config = BatchConfig(
        n_events=args.events,
        muon_fixed=muon_fixed,
        muon_file=args.muon_file,
        muon_mode=args.muon_mode,
        photons_per_cm=args.photons_per_cm,
        wavelength_nm=args.wavelength,
        max_bounces=args.max_bounces,
        batch_size=args.batch_size,
        pmt_response=args.pmt_response,
        pmt_full_wf=args.full_wf,
        output_dir=args.output_dir,
        record_events=not args.no_record,
        seed=args.seed,
    )

    # Validate
    if args.muon_file and not args.muon_file.exists():
        print(f"Error: muon file not found: {args.muon_file}")
        return
    if args.pmt_csv and not args.pmt_csv.exists():
        print(f"Error: PMT CSV not found: {args.pmt_csv}")
        return

    # Build geometry (reuse the same geometry for all events)
    lappd_indices = None
    if args.lappd_indices:
        import json
        lappd_indices = [int(x) for x in args.lappd_indices.split(",")]

    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, lappd_indices=lappd_indices,
                          no_lappd=args.no_lappd, z_offset=args.z_offset,
                          lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation, n_surfboards=args.surfboard)

    print(f"  Mesh: {geom.mesh_vertices.shape[0]} verts, {geom.mesh_triangles.shape[0]} tris")
    print(f"  PMTs: {geom.pmt_centers.shape[0]}")
    print(f"  LAPPDs: {geom.lappd_data.shape[0]}")
    print(f"  Tank: R={geom.tank_radius:.0f} mm, Z=[{geom.tank_z_min:.0f}, {geom.tank_z_max:.0f}]")
    if geom.surfboard_data.shape[0] > 0:
        print(f"  Surfboards: {geom.surfboard_data.shape[0]} PVC panels")

    # Save companion files for event display / analysis
    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tank metadata
    meta = {
        "tank_radius_mm": geom.tank_radius,
        "tank_z_min_mm": geom.tank_z_min,
        "tank_z_max_mm": geom.tank_z_max,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  Saved {meta_path}")

    # Detector registry
    det_path = output_dir / "detectors.csv"
    with open(det_path, "w") as f:
        f.write("system_code,detector_index,x,y,z,label,panel\n")
        for d in geom.detectors:
            sys_code = {
                "pmt": DET_SYS_PMT,
                "lappd_default": DET_SYS_LAPPD_DEFAULT,
                "lappd_annie": DET_SYS_LAPPD_ANNIE,
            }.get(d.system, -1)
            panel = getattr(d, "panel", -1)
            f.write(f"{sys_code},{d.index},{d.position[0]},{d.position[1]},{d.position[2]},{d.label},{panel}\n")
    print(f"  Saved {det_path}")

    if args.muon_file:
        print(f"  Muon topology: from file ({args.muon_file})")
    elif muon_fixed:
        print(f"  Muon topology: fixed {muon_fixed}")
    else:
        mode_names = {"downward": "downward (atmospheric)",
                      "isotropic": "isotropic (totally random, default)",
                      "beam": "forward along +Y (beam-like)"}
        print(f"  Muon topology: random position, {mode_names.get(config.muon_mode, config.muon_mode)}")

    print(f"\nGenerating {config.n_events} events ({config.photons_per_cm} ph/cm)...")
    if config.pmt_response:
        mode = "waveform" if config.pmt_full_wf else "fast"
        print(f"  PMT response: enabled ({mode} path)")

    optics_cfg = None
    if args.max_bounces > 0:
        from annieray.optics import load_optics_config
        optics_cfg = load_optics_config(args.optics_config)
        print(f"  Multi-bounce optics: max {args.max_bounces} reflections")

    paths = run_batch(geom, config, optics_config=optics_cfg)
    print("Done.")


def main(argv: list[str] | None = None) -> None:
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f32)

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
    elif args.command == "batch":
        batch_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
