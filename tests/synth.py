"""Synthetic cel animation: textured static background + camera pan + moving cel."""
import cv2
import numpy as np


def make_background(h=180, w=320, pad=60, seed=0):
    rng = np.random.default_rng(seed)
    big = rng.random((h + 2 * pad, w + 2 * pad, 3)).astype(np.float32)
    big = (cv2.GaussianBlur(big, (0, 0), 6) - 0.5) * 6 + 0.45
    big = (np.clip(big, 0.05, 0.8) * 255).astype(np.uint8)
    # some hard edges so phase correlation has structure
    for i in range(0, big.shape[1], 37):
        cv2.line(big, (i, 0), (i, big.shape[0]), (30, 30, 30), 1)
    for j in range(0, big.shape[0], 29):
        cv2.line(big, (0, j), (big.shape[1], j), (200, 200, 200), 1)
    return big, pad


def make_sequence(n=40, h=180, w=320, pan_per_frame=(1.5, 0.0), seed=0):
    """Returns frames, true cel masks, true background crops (aligned to each frame)."""
    big, pad = make_background(h, w, seed=seed)
    frames, masks, bgs = [], [], []
    for t in range(n):
        ox = int(round(pad + pan_per_frame[0] * t))
        oy = int(round(pad + pan_per_frame[1] * t))
        bg = big[oy:oy + h, ox:ox + w].copy()
        f = bg.copy()
        m = np.zeros((h, w), bool)
        cx, cy, r = int(60 + 5 * t), h // 2, 28
        cv2.circle(f, (cx, cy), r, (240, 240, 240), -1)
        cv2.circle(f, (cx, cy), r, (10, 10, 10), 2)
        cv2.circle(m.view(np.uint8), (cx, cy), r + 1, 1, -1)
        frames.append(f)
        masks.append(m)
        bgs.append(bg)
    return frames, masks, bgs
