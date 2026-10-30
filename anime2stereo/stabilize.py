"""Global (camera pan) motion estimation.

Traditional animation moves the camera over a painted background by translation
("pan" / "truck"). A whole-frame phase correlation is easily hijacked by a large,
high-contrast moving cel, so we phase-correlate a grid of blocks and take the
response-weighted *median* block shift: the static background occupies most blocks,
the cels only a few, and the median ignores them.
"""
from __future__ import annotations

import cv2
import numpy as np


def _dft_friendly(n: int) -> int:
    """Largest even k <= n that is an optimal DFT size.

    OpenCV's phaseCorrelate zero-pads non-optimal sizes internally and its sub-pixel peak
    refinement is biased by 0.5px on odd sizes; both show up as a constant fake drift."""
    k = n
    while k > 8 and (k % 2 or cv2.getOptimalDFTSize(k) != k):
        k -= 1
    return k


def _block_shift(prev: np.ndarray, cur: np.ndarray, win: np.ndarray) -> tuple[float, float, float]:
    # NB: cv2.phaseCorrelate applies the window to its inputs *in place* (observed on
    # OpenCV 5.0), so always hand it private copies.
    (dx, dy), resp = cv2.phaseCorrelate(np.ascontiguousarray(prev).copy(), np.ascontiguousarray(cur).copy(), win)
    return dx, dy, resp


def estimate_pair_shift(prev: np.ndarray, cur: np.ndarray, grid: tuple[int, int] = (4, 6), min_response: float = 0.05,
                        min_texture: float = 2.0, max_step_frac: float = 0.25) -> tuple[float, float, float]:
    """Content motion (dx, dy) from `prev` to `cur` using a median over grid blocks.

    Returns (dx, dy, confidence) where confidence = fraction of blocks agreeing with the median.
    """
    h, w = prev.shape
    gy, gx = grid
    bh, bw = _dft_friendly(h // gy), _dft_friendly(w // gx)
    win = cv2.createHanningWindow((bw, bh), cv2.CV_32F)
    cands = []
    for j in range(gy):
        for i in range(gx):
            y0, x0 = j * (h // gy), i * (w // gx)
            a = prev[y0:y0 + bh, x0:x0 + bw]
            b = cur[y0:y0 + bh, x0:x0 + bw]
            if a.std() < min_texture:
                continue
            dx, dy, resp = _block_shift(a, b, win)
            if resp < min_response or abs(dx) > max_step_frac * bw or abs(dy) > max_step_frac * bh:
                continue
            cands.append((dx, dy, resp))
    if len(cands) < 3:
        # fall back to whole frame
        he, we = _dft_friendly(h), _dft_friendly(w)
        win_full = cv2.createHanningWindow((we, he), cv2.CV_32F)
        dx, dy, resp = _block_shift(prev[:he, :we], cur[:he, :we], win_full)
        if resp < min_response or abs(dx) > max_step_frac * w or abs(dy) > max_step_frac * h:
            return 0.0, 0.0, 0.0
        return dx, dy, 0.5
    c = np.array(cands)
    mdx, mdy = np.median(c[:, 0]), np.median(c[:, 1])
    agree = (np.abs(c[:, 0] - mdx) < 1.5) & (np.abs(c[:, 1] - mdy) < 1.5)
    if agree.sum() >= 2:
        mdx = float(np.average(c[agree, 0], weights=c[agree, 2]))
        mdy = float(np.average(c[agree, 1], weights=c[agree, 2]))
    return float(mdx), float(mdy), float(agree.mean())


def _snap_integer(anchor: np.ndarray, cur: np.ndarray, rel: np.ndarray, min_overlap: float) -> np.ndarray | None:
    """Pick the integer offset among the 4 roundings of ``rel`` that best aligns ``cur`` to ``anchor``.

    The plate is built by placing frames at integer offsets, so the final choice must be
    the integer alignment with the smallest robust photometric error, not a rounded
    sub-pixel estimate (which is a coin flip at .5).
    """
    h, w = anchor.shape
    best, best_err = None, np.inf
    for cx in (np.floor(rel[0]), np.ceil(rel[0])):
        for cy in (np.floor(rel[1]), np.ceil(rel[1])):
            r = np.array([int(cx), int(cy)])
            x0, x1 = max(0, r[0]), min(w, w + r[0])
            y0, y1 = max(0, r[1]), min(h, h + r[1])
            if (x1 - x0) * (y1 - y0) < min_overlap * h * w:
                continue
            a = anchor[y0:y1, x0:x1]
            b = cur[y0 - r[1]:y1 - r[1], x0 - r[0]:x1 - r[0]]
            err = np.minimum(np.abs(a - b), 24.0).mean()  # clipped: moving cels count only so much
            if err < best_err:
                best, best_err = r, err
    return best


def estimate_shifts(frames: list[np.ndarray], min_confidence: float = 0.3, min_overlap: float = 0.5) -> np.ndarray:
    """Per-frame alignment offset (dx, dy) that maps frame i onto frame 0's coordinates, shape (n, 2).

    Sign: phase correlation reports how far the *content* moved (+x = right, +y = down);
    the alignment offset is the negative of the accumulated content motion.

    Drift control: consecutive-pair estimates predict the offset, then the frame is
    re-registered against an *anchor* frame (frame 0, re-anchored whenever the overlap
    drops below ``min_overlap``) so sub-pixel errors do not accumulate over a long pan.
    """
    n = len(frames)
    shifts = np.zeros((n, 2), dtype=np.float64)
    if n < 2:
        return shifts
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0) for f in frames]
    h, w = gray[0].shape
    anchor, anchor_off = gray[0], np.zeros(2)  # anchor offsets are kept integer
    for i in range(1, n):
        dx, dy, conf = estimate_pair_shift(gray[i - 1], gray[i])
        if conf < min_confidence:
            dx, dy = 0.0, 0.0
        predicted = shifts[i - 1] - np.array([dx, dy])
        # residual registration against the anchor on their overlap
        rel = np.round(predicted - anchor_off).astype(int)
        x0, x1 = max(0, rel[0]), min(w, w + rel[0])
        y0, y1 = max(0, rel[1]), min(h, h + rel[1])
        corrected = predicted
        if (x1 - x0) * (y1 - y0) >= min_overlap * h * w:
            a = anchor[y0:y1, x0:x1]
            b = gray[i][y0 - rel[1]:y1 - rel[1], x0 - rel[0]:x1 - rel[0]]
            rdx, rdy, rconf = estimate_pair_shift(a, b)
            if rconf >= min_confidence and abs(rdx) < 3 and abs(rdy) < 3:
                corrected = anchor_off + rel - np.array([rdx, rdy])
            snapped = _snap_integer(anchor, gray[i], corrected - anchor_off, min_overlap)
            if snapped is not None:
                corrected = anchor_off + snapped
        else:
            anchor, anchor_off = gray[i], np.round(predicted)
            corrected = anchor_off
        shifts[i] = corrected
    return shifts


def is_panning(shifts: np.ndarray, tol: float = 1.5) -> bool:
    return bool(np.ptp(shifts, axis=0).max() > tol)
