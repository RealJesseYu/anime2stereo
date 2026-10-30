"""End-to-end orchestration: video -> shots -> plates -> cels -> depth -> stereo outputs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import cleanplate, depth as depthmod, segmentation, shots as shotmod, stabilize, stereo, video
from .metrics import hole_fraction
from .stereo import StereoParams


@dataclass
class Config:
    depth_mode: str = "hybrid"           # cel | ml | hybrid
    refine: str = "grabcut"              # none | grabcut
    two_pass_plate: bool = True
    cel_threshold: float = 30.0
    cel_relief: float = 0.5              # 0 = flat cards, 1 = full ML relief inside cels
    static_layers: bool = True           # lift never-moving characters out of the plate via depth islands + inpainting
    shot_threshold: float = 0.45
    max_disparity_frac: float = 0.03     # fraction of frame width
    convergence: Optional[float] = None  # zero-parallax depth in [0,1]; None = auto
    ml_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: Optional[str] = None
    formats: tuple[str, ...] = ("sbs", "anaglyph", "wiggle", "debug")
    wiggle_seconds: float = 4.0
    save_plates: bool = True


@dataclass
class ShotResult:
    start: int
    end: int
    pan_px: float
    plate_never_revealed_frac: float
    mean_cel_coverage: float
    mean_cels: float
    static_layer_frac: float
    naive_hole_frac: float
    layered_hole_frac: float
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)


def _label_overlay(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = frame.copy()
    n = int(labels.max())
    rng = np.random.default_rng(3)
    for i in range(1, n + 1):
        c = rng.integers(60, 255, size=3)
        m = labels == i
        out[m] = (0.45 * out[m] + 0.55 * c).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(x) for x in c), 1)
    return out


def _montage(frame, labels, depth, ana) -> np.ndarray:
    top = np.concatenate([frame, _label_overlay(frame, labels)], axis=1)
    bot = np.concatenate([depthmod.colorize_depth(depth), ana], axis=1)
    return np.concatenate([top, bot], axis=0)


def process_shot(frames: list[np.ndarray], cfg: Config, estimator: Optional[depthmod.DepthEstimator], params: StereoParams, log=print):
    t0 = time.time()
    timings = {}
    h, w = frames[0].shape[:2]

    shifts = stabilize.estimate_shifts(frames)
    timings["pan"] = time.time() - t0
    t = time.time()
    plate = cleanplate.build_plate(frames, shifts)
    crops = [plate.crop(i, h, w) for i in range(len(frames))]
    layers = segmentation.segment_shot(frames, crops, threshold=cfg.cel_threshold, refine="none")
    if cfg.two_pass_plate:
        plate = cleanplate.refine_plate_with_masks(plate, frames, [L.mask for L in layers])
        crops = [plate.crop(i, h, w) for i in range(len(frames))]
    timings["plate"] = time.time() - t
    t = time.time()
    layers = segmentation.segment_shot(frames, crops, threshold=cfg.cel_threshold, refine=cfg.refine)
    timings["segment"] = time.time() - t

    t = time.time()
    Hc, Wc = plate.canvas.shape[:2]
    static_frac = 0.0
    if cfg.depth_mode == "cel":
        plate_depth = depthmod.ground_plane_depth(Hc, Wc)
        pd_crops = [plate.crop_mask(plate_depth, i, h, w) for i in range(len(frames))]
        depths = [depthmod.cel_depth(L, pd) for L, pd in zip(layers, pd_crops)]
    else:
        assert estimator is not None
        raw_plate = estimator.predict([plate.canvas])[0]
        raw_frames = estimator.predict(frames)
        normed = depthmod.normalise_shot([raw_plate] + raw_frames)
        plate_depth = depthmod.smooth_background_depth(normed[0])
        frame_ml = normed[1:]
        if cfg.static_layers:
            # characters that never moved are still in the plate: find them by depth,
            # inpaint behind them, and re-segment so they become cels like any other
            fg = depthmod.static_foreground(plate_depth)
            static_frac = float(fg.mean())
            if fg.any():
                plate.canvas, plate_depth = depthmod.inpaint_behind(plate.canvas, plate_depth, fg)
                crops = [plate.crop(i, h, w) for i in range(len(frames))]
                layers = segmentation.segment_shot(frames, crops, threshold=cfg.cel_threshold, refine=cfg.refine)
        pd_crops = [plate.crop_mask(plate_depth, i, h, w) for i in range(len(frames))]
        # map each frame's relative depth into the plate's scale using background pixels
        frame_ml = [depthmod.align_to_reference(fm, pd, ~L.mask) for fm, pd, L in zip(frame_ml, pd_crops, layers)]
        if cfg.depth_mode == "ml":
            depths = frame_ml
        else:
            depths = [depthmod.hybrid_depth(fm, pd, L, relief=cfg.cel_relief) for fm, pd, L in zip(frame_ml, pd_crops, layers)]
            depths = depthmod.track_and_smooth_cel_depths(depths, layers)
    timings["depth"] = time.time() - t

    # convergence: put ~75% of the scene behind the screen plane
    if cfg.convergence is None:
        sample = np.concatenate([d[::4, ::4].ravel() for d in depths[:: max(1, len(depths) // 16)]])
        zpd = float(np.percentile(sample, 75))
    else:
        zpd = cfg.convergence
    p = StereoParams(max_disparity=params.max_disparity, zpd=zpd, splat_width=params.splat_width)

    t = time.time()
    Ls, Rs = [], []
    naive_holes, layered_holes = [], []
    ones = np.ones((h, w), bool)
    for i, (f, d, L) in enumerate(zip(frames, depths, layers)):
        Lf, Rf = stereo.render_stereo(crops[i], pd_crops[i], f, d, L.labels, p)
        Ls.append(Lf)
        Rs.append(Rf)
        if i % 4 == 0:
            naive_holes.append(hole_fraction(d, ones, p))
            layered_holes.append(hole_fraction(pd_crops[i], ones, p))
    timings["render"] = time.time() - t

    res = ShotResult(
        start=0, end=len(frames), pan_px=float(np.ptp(shifts, axis=0).max()),
        plate_never_revealed_frac=float(plate.inpainted.mean()),
        mean_cel_coverage=float(np.mean([L.mask.mean() for L in layers])),
        mean_cels=float(np.mean([L.n for L in layers])), static_layer_frac=static_frac,
        naive_hole_frac=float(np.mean(naive_holes)), layered_hole_frac=float(np.mean(layered_holes)),
        seconds=time.time() - t0, timings={k: round(v, 2) for k, v in timings.items()},
    )
    return dict(L=Ls, R=Rs, depths=depths, layers=layers, plate=plate, plate_depth=plate_depth, zpd=zpd, result=res)


def process_video(
    input_path: str | Path,
    out_dir: str | Path,
    cfg: Config,
    start: float = 0.0,
    duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    scale: Optional[float] = None,
    log=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    frames, fps = video.read_frames(input_path, start=start, duration=duration, max_frames=max_frames, scale=scale)
    frames = video.deflicker(frames)
    h, w = frames[0].shape[:2]
    log(f"read {len(frames)} frames @ {fps:.2f} fps, {w}x{h}")
    shot_ranges = shotmod.detect_shots(frames, threshold=cfg.shot_threshold)
    log(f"detected {len(shot_ranges)} shots: {shot_ranges}")

    estimator = None
    if cfg.depth_mode in ("ml", "hybrid"):
        estimator = depthmod.DepthEstimator(cfg.ml_model, cfg.device)
    params = StereoParams(max_disparity=cfg.max_disparity_frac * w)

    all_L, all_R, all_debug, all_depth = [], [], [], []
    shot_results = []
    for si, (s, e) in enumerate(shot_ranges):
        log(f"shot {si}: frames {s}-{e}")
        out = process_shot(frames[s:e], cfg, estimator, params, log=log)
        r: ShotResult = out["result"]
        r.start, r.end = s, e
        shot_results.append(r)
        log(f"  pan={r.pan_px:.1f}px cels/frame={r.mean_cels:.1f} cover={r.mean_cel_coverage:.1%} static={r.static_layer_frac:.1%} "
            f"naive_holes={r.naive_hole_frac:.2%} layered_holes={r.layered_hole_frac:.2%} "
            f"zpd={out['zpd']:.2f} t={r.seconds:.1f}s {r.timings}")
        all_L += out["L"]
        all_R += out["R"]
        all_depth += out["depths"]
        if "debug" in cfg.formats:
            for i in range(e - s):
                ana = stereo.anaglyph(out["L"][i], out["R"][i])
                all_debug.append(_montage(frames[s + i], out["layers"][i].labels, out["depths"][i], ana))
        if cfg.save_plates:
            cv2.imwrite(str(out_dir / f"plate_shot{si:02d}.png"), out["plate"].canvas)
            cv2.imwrite(str(out_dir / f"plate_depth_shot{si:02d}.png"), depthmod.colorize_depth(out["plate_depth"]))

    written = {}
    if "sbs" in cfg.formats:
        written["sbs"] = str(video.write_video(out_dir / "stereo_sbs_half.mp4", [stereo.side_by_side(L, R, half=True) for L, R in zip(all_L, all_R)], fps))
    if "sbs-full" in cfg.formats:
        written["sbs_full"] = str(video.write_video(out_dir / "stereo_sbs_full.mp4", [stereo.side_by_side(L, R, half=False) for L, R in zip(all_L, all_R)], fps))
    if "anaglyph" in cfg.formats:
        written["anaglyph"] = str(video.write_video(out_dir / "stereo_anaglyph.mp4", [stereo.anaglyph(L, R) for L, R in zip(all_L, all_R)], fps))
    if "depth" in cfg.formats or "debug" in cfg.formats:
        written["depth"] = str(video.write_video(out_dir / "depth.mp4", [depthmod.colorize_depth(d) for d in all_depth], fps))
    if "debug" in cfg.formats:
        written["debug"] = str(video.write_video(out_dir / "debug_montage.mp4", all_debug, fps))
    if "wiggle" in cfg.formats:
        n = min(len(all_L), int(cfg.wiggle_seconds * fps))
        step = max(1, int(round(fps / 12)))
        wig = []
        for i in range(0, n, step):
            wig += [all_L[i], all_R[i]]
        written["wiggle"] = str(video.write_gif(out_dir / "wiggle.gif", wig, fps=12))

    summary = {
        "input": str(input_path), "frames": len(frames), "fps": fps, "size": [w, h],
        "config": asdict(cfg), "shots": [asdict(r) for r in shot_results],
        "outputs": written, "seconds": round(time.time() - t_all, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {summary['seconds']}s -> {out_dir}")
    return summary
