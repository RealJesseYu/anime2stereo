"""Shot (scene cut) detection so that clean plates are built per static background."""
from __future__ import annotations

import cv2
import numpy as np


def _hist(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [64], [0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def detect_shots(frames: list[np.ndarray], threshold: float = 0.45, min_len: int = 8) -> list[tuple[int, int]]:
    """Return [(start, end)) index ranges of shots.

    Uses a combination of histogram distance and mean absolute pixel difference on
    downsampled frames. A cut is declared when the combined score spikes above
    `threshold` relative to the local median.
    """
    n = len(frames)
    if n == 0:
        return []
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for f in frames]
    hists = [_hist(f) for f in frames]
    scores = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        hd = 0.5 * np.abs(hists[i] - hists[i - 1]).sum()  # 0..1
        pd = np.abs(small[i] - small[i - 1]).mean()  # 0..1
        scores[i] = hd + 2.0 * pd
    # adaptive: compare to local median so slow pans don't trigger
    cuts = [0]
    for i in range(1, n):
        lo, hi = max(1, i - 12), min(n, i + 12)
        local = np.median(scores[lo:hi]) if hi > lo else 0.0
        if scores[i] > threshold and scores[i] > 3.0 * local + 0.05 and i - cuts[-1] >= min_len:
            cuts.append(i)
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
