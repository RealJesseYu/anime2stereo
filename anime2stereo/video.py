"""Frame I/O helpers (OpenCV for reading, ffmpeg for writing when available)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_frames(
    path: str | Path,
    start: float = 0.0,
    duration: float | None = None,
    max_frames: int | None = None,
    scale: float | None = None,
) -> tuple[list[np.ndarray], float]:
    """Read BGR uint8 frames from a video.

    Args:
        path: video file.
        start: start time in seconds.
        duration: seconds to read (None = to end).
        max_frames: hard cap on number of frames.
        scale: optional resize factor (e.g. 0.5).

    Returns:
        (frames, fps)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    n_target = None
    if duration is not None:
        n_target = int(round(duration * fps))
    if max_frames is not None:
        n_target = max_frames if n_target is None else min(n_target, max_frames)

    frames: list[np.ndarray] = []
    while True:
        if n_target is not None and len(frames) >= n_target:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if scale is not None and scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path} (start={start}s)")
    return frames, float(fps)


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float, crf: int = 18) -> Path:
    """Write BGR frames to an H.264 mp4 via ffmpeg (falls back to OpenCV mp4v)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    h, w = frames[0].shape[:2]
    # yuv420p needs even dimensions
    if w % 2 or h % 2:
        w2, h2 = w - (w % 2), h - (h % 2)
        frames = [f[:h2, :w2] for f in frames]
        h, w = h2, w2
    ff = _ffmpeg()
    if ff is None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return path
    cmd = [
        ff, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def write_gif(path: str | Path, frames: Iterable[np.ndarray], fps: float, max_width: int | None = 480) -> Path:
    """Write BGR frames to a palette-optimised GIF via ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    if max_width is not None and frames[0].shape[1] > max_width:
        s = max_width / frames[0].shape[1]
        frames = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in frames]
    h, w = frames[0].shape[:2]
    ff = _ffmpeg()
    if ff is None:
        raise RuntimeError("ffmpeg is required for GIF output")
    cmd = [
        ff, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-filter_complex", "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return path


def deflicker(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Normalise per-frame global brightness (old film flickers; it breaks temporal medians).

    Each frame's luminance mean/std is matched to the shot-wide median mean/std.
    """
    stats = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stats.append((g.mean(), g.std() + 1e-6))
    means = np.array([s[0] for s in stats])
    stds = np.array([s[1] for s in stats])
    tm, ts = float(np.median(means)), float(np.median(stds))
    out = []
    for f, (m, s) in zip(frames, stats):
        gain = ts / s
        bias = tm - gain * m
        out.append(np.clip(f.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8))
    return out
