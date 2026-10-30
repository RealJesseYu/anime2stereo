"""Assemble README figures from pipeline output directories.

usage: python scripts/make_examples.py OUT_DIR [OUT_DIR ...] --dest examples/
Writes, per output dir: a 4-panel strip (original | cels | depth | anaglyph) for a
few frames, and copies the wiggle GIF and clean plate.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path, every: int, max_n: int):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while len(out) < max_n:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(f)
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dest", default="examples")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for d in args.dirs:
        d = Path(d)
        name = d.name
        m = json.loads((d / "metrics.json").read_text())
        n = m["frames"]
        every = max(1, n // (args.frames + 1))
        frames = read_frames(d / "debug_montage.mp4", every, args.frames + 1)[1:]
        # montage is 2x2 (orig, cels / depth, anaglyph) -> re-lay as 1x4 strip
        strips = []
        for f in frames:
            h, w = f.shape[0] // 2, f.shape[1] // 2
            panels = [f[:h, :w], f[:h, w:], f[h:, :w], f[h:, w:]]
            strips.append(np.concatenate(panels, axis=1))
        sheet = np.concatenate(strips, axis=0)
        s = args.width / sheet.shape[1]
        sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / f"{name}_strip.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if (d / "wiggle.gif").exists():
            shutil.copy(d / "wiggle.gif", dest / f"{name}_wiggle.gif")
        plates = sorted(d.glob("plate_shot*.png"))
        if plates:
            ims = [cv2.imread(str(p)) for p in plates[:3]]
            h = min(i.shape[0] for i in ims)
            ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims]
            cv2.imwrite(str(dest / f"{name}_plates.jpg"), np.concatenate(ims, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(name, "frames", n, "shots", len(m["shots"]))


if __name__ == "__main__":
    main()
