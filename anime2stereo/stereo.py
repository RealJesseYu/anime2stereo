"""Layered depth-image-based rendering (DIBR) of a stereo pair.

Convention: ``depth`` in [0,1], 1 = nearest. ``disparity = max_disp * (depth - zpd)``
in pixels; positive disparity = in front of the screen plane (zero-parallax depth
``zpd``). For a point in front of the screen the left-eye image shifts it right and the
right-eye image shifts it left.

Rendering order per eye:
  1. forward-warp the complete clean-plate background (its own depth map), fill cracks;
  2. forward-warp each cel (RGBA) and composite over, nearest last.
Because the background behind every cel is already known (temporal inpainting), the
disocclusions that plague single-layer DIBR simply do not occur.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoParams:
    max_disparity: float = 18.0   # pixels at full output width
    zpd: float = 0.3              # zero-parallax depth (0..1); background sits behind the screen
    splat_width: int = 1          # extra pixels each sample covers to avoid 1px cracks


def forward_warp(
    img: np.ndarray,
    alpha: np.ndarray,
    disp: np.ndarray,
    sign: float,
    splat_width: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat ``img`` along x by ``sign*disp`` with a z-buffer (nearest wins).

    Fully vectorised: every valid source pixel gets a target column; ``np.maximum.at``
    resolves collisions deterministically by nearness (larger ``disp`` = nearer). Each
    sample also covers ``splat_width`` extra pixels in its shift direction so 1px cracks
    from depth gradients close; extra splats only overwrite strictly *farther* content.

    Returns (warped BGR, written mask).
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    valid = alpha > 0
    if not valid.any():
        return out, np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(valid)
    q = np.round(sign * disp[ys, xs]).astype(np.int32)
    near = (sign * q).astype(np.int32)            # larger = nearer, regardless of eye
    xt = xs + q
    zmin = np.iinfo(np.int32).min
    zbuf = np.full(h * w, zmin, dtype=np.int32)
    src = img.reshape(-1, img.shape[2]) if img.ndim == 3 else img.ravel()
    flat_out = out.reshape(-1, img.shape[2]) if img.ndim == 3 else out.ravel()
    src_idx = ys * w + xs

    # --- main splat
    ok = (xt >= 0) & (xt < w)
    tgt = ys[ok] * w + xt[ok]
    nv = near[ok]
    np.maximum.at(zbuf, tgt, nv)
    win = nv == zbuf[tgt]
    flat_out[tgt[win]] = src[src_idx[ok][win]]
    # --- extra splats: only onto strictly farther (or empty) targets
    step = 1 if sign > 0 else -1
    for extra in range(1, splat_width + 1):
        xe = xt + step * extra
        ok = (xe >= 0) & (xe < w)
        tgt = ys[ok] * w + xe[ok]
        nv = near[ok]
        cand = nv > zbuf[tgt]
        tgt, nv, si = tgt[cand], nv[cand], src_idx[ok][cand]
        if tgt.size == 0:
            continue
        zb2 = np.full(h * w, zmin, dtype=np.int32)
        np.maximum.at(zb2, tgt, nv)
        win = nv == zb2[tgt]
        flat_out[tgt[win]] = src[si[win]]
        zbuf[tgt[win]] = nv[win]
    written = (zbuf > zmin).reshape(h, w)
    return out, written


def fill_cracks(img: np.ndarray, written: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Fill small unwritten gaps by horizontal nearest-neighbour, then inpaint the rest."""
    hole = ~written
    if not hole.any():
        return img
    h, w = written.shape
    # left-nearest valid index
    idx = np.where(written, np.arange(w)[None, :], -1)
    left = np.maximum.accumulate(idx, axis=1)
    idx_r = np.where(written, np.arange(w)[None, :], w)
    right = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(h)[:, None]
    out = img.copy()
    dl = np.where(left >= 0, np.arange(w)[None, :] - left, 10**6)
    dr = np.where(right < w, right - np.arange(w)[None, :], 10**6)
    use_left = dl <= dr
    li = np.clip(left, 0, w - 1)
    ri = np.clip(right, 0, w - 1)
    fill = np.where(use_left[..., None], img[rows, li], img[rows, ri])
    small = hole & (np.minimum(dl, dr) <= max_gap)
    out[small] = fill[small]
    big = hole & ~small
    if big.any():
        out = cv2.inpaint(out, big.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return out


def render_eye(
    plate_crop: np.ndarray,
    plate_depth: np.ndarray,
    frame: np.ndarray,
    frame_depth: np.ndarray,
    labels: np.ndarray,
    params: StereoParams,
    sign: float,
) -> np.ndarray:
    """Render one eye. ``sign`` = +1 for left eye, -1 for right eye."""
    disp_bg = params.max_disparity * (plate_depth - params.zpd)
    bg, wr = forward_warp(plate_crop, np.ones(plate_depth.shape, bool), disp_bg, sign, params.splat_width)
    bg = fill_cracks(bg, wr)
    if not (labels > 0).any():
        return bg
    # all cels in one z-buffered pass (nearest wins), soft alpha at the contour
    m = labels > 0
    disp = params.max_disparity * (frame_depth - params.zpd)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)
    col, wr = forward_warp(frame, m, disp, sign, params.splat_width)
    aw, _ = forward_warp(np.repeat(a[..., None], 3, axis=2), m, disp, sign, params.splat_width)
    aw = np.clip(aw[..., 0], 0, 1)[..., None] * wr[..., None]
    return (col.astype(np.float32) * aw + bg.astype(np.float32) * (1 - aw)).astype(np.uint8)


def render_stereo(plate_crop, plate_depth, frame, frame_depth, labels, params: StereoParams) -> tuple[np.ndarray, np.ndarray]:
    L = render_eye(plate_crop, plate_depth, frame, frame_depth, labels, params, +1.0)
    R = render_eye(plate_crop, plate_depth, frame, frame_depth, labels, params, -1.0)
    return L, R


def render_naive(frame: np.ndarray, depth: np.ndarray, params: StereoParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-layer DIBR baseline (no segmentation / inpainting). Returns (L, R, hole fraction)."""
    disp = params.max_disparity * (depth - params.zpd)
    ones = np.ones(depth.shape, bool)
    L, wl = forward_warp(frame, ones, disp, +1.0, params.splat_width)
    R, wr = forward_warp(frame, ones, disp, -1.0, params.splat_width)
    holes = 0.5 * ((~wl).mean() + (~wr).mean())
    return fill_cracks(L, wl), fill_cracks(R, wr), float(holes)


# ---------------------------------------------------------------- output formats

def side_by_side(L: np.ndarray, R: np.ndarray, half: bool = True) -> np.ndarray:
    if half:
        w = L.shape[1] // 2
        L = cv2.resize(L, (w, L.shape[0]), interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, (w, R.shape[0]), interpolation=cv2.INTER_AREA)
    return np.concatenate([L, R], axis=1)


def anaglyph(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Red-cyan anaglyph (left eye = red channel)."""
    out = R.copy()
    out[..., 2] = L[..., 2]
    return out


def wiggle(L: np.ndarray, R: np.ndarray) -> list[np.ndarray]:
    return [L, R]
