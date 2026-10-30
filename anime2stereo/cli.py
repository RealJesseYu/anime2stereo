"""Command line entry point: ``anime2stereo input.mp4 -o out/ [options]``."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Config, process_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anime2stereo", description="Lift 2D cel animation into stereoscopic 3D.")
    p.add_argument("input", help="input video")
    p.add_argument("-o", "--out", default="output", help="output directory")
    p.add_argument("--start", type=float, default=0.0, help="start time (s)")
    p.add_argument("--duration", type=float, default=None, help="seconds to process")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--scale", type=float, default=None, help="resize factor for input frames")
    p.add_argument("--depth", choices=["cel", "ml", "hybrid"], default="hybrid", help="depth source (default hybrid)")
    p.add_argument("--refine", choices=["none", "grabcut"], default="grabcut", help="cel mask edge refinement")
    p.add_argument("--no-two-pass", action="store_true", help="skip the mask-aware second plate pass")
    p.add_argument("--cel-threshold", type=float, default=30.0, help="residual threshold (0-255) for cel detection")
    p.add_argument("--cel-relief", type=float, default=0.5, help="0 = flat cutout cels, 1 = full ML relief inside cels")
    p.add_argument("--no-static-layers", action="store_true", help="do not lift never-moving characters out of the plate")
    p.add_argument("--shot-threshold", type=float, default=0.45)
    p.add_argument("--max-disparity", type=float, default=0.03, help="max disparity as a fraction of width")
    p.add_argument("--convergence", type=float, default=None, help="zero-parallax depth in [0,1] (default: auto)")
    p.add_argument("--formats", default="sbs,anaglyph,wiggle,debug", help="comma list of sbs,sbs-full,anaglyph,wiggle,depth,debug")
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--device", default=None, help="torch device (cuda|mps|cpu); default auto")
    p.add_argument("--no-plates", action="store_true", help="do not save clean plate PNGs")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        depth_mode=args.depth, refine=args.refine, two_pass_plate=not args.no_two_pass,
        cel_threshold=args.cel_threshold, cel_relief=args.cel_relief, static_layers=not args.no_static_layers,
        shot_threshold=args.shot_threshold,
        max_disparity_frac=args.max_disparity, convergence=args.convergence,
        ml_model=args.model, device=args.device,
        formats=tuple(s.strip() for s in args.formats.split(",") if s.strip()),
        save_plates=not args.no_plates,
    )
    process_video(args.input, Path(args.out), cfg, start=args.start, duration=args.duration,
                  max_frames=args.max_frames, scale=args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
