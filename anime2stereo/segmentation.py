"""Cel (foreground layer) segmentation.

Cels are segmented as the residual between each frame and the pan-aligned clean plate.
That residual is cleaned with morphology tuned for film grain, then optionally snapped
to the drawn outlines with GrabCut (anime cels have hard, dark contour lines, which
makes GrabCut's colour model converge quickly).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class CelLayers:
    mask: np.ndarray      # (H,W) bool union of all cels
    labels: np.ndarray    # (H,W) int32, 0 = background, 1..k = cel instances
    n: int


def residual_mask(
    frame: np.ndarray,
    plate_crop: np.ndarray,
    threshold: float = 30.0,
    blur_sigma: float = 1.2,
    min_area_frac: float = 0.003,
) -> np.ndarray:
    """Binary cel mask from |frame - plate| (max over colour channels)."""
    f = cv2.GaussianBlur(frame, (0, 0), blur_sigma).astype(np.int16)
    p = cv2.GaussianBlur(plate_crop, (0, 0), blur_sigma).astype(np.int16)
    d = np.abs(f - p).max(axis=2).astype(np.uint8)
    m = d > threshold
    # film grain / scratches: small opening, then close to merge limbs
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = ndi.binary_fill_holes(m.astype(bool))
    # drop small components
    lab, k = ndi.label(m)
    if k:
        sizes = ndi.sum(m, lab, index=np.arange(1, k + 1))
        min_area = min_area_frac * m.size
        keep = np.zeros(k + 1, dtype=bool)
        keep[1:] = sizes >= min_area
        m = keep[lab]
    return m


def grabcut_refine(frame: np.ndarray, mask: np.ndarray, iters: int = 2, band: int = 6) -> np.ndarray:
    """Snap a coarse mask to image edges with GrabCut.

    Interior (eroded) -> sure FG, far exterior -> sure BG, a band around the boundary is
    left probable so the colour models decide where the contour line is.
    """
    if not mask.any():
        return mask
    # restrict to the (padded) bounding box of the mask: GrabCut cost scales with pixels
    ys, xs = np.where(mask)
    pad = 3 * band
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    if (y1 - y0) * (x1 - x0) < mask.size:
        sub = grabcut_refine(np.ascontiguousarray(frame[y0:y1, x0:x1]), mask[y0:y1, x0:x1], iters, band)
        out = mask.copy()
        out[y0:y1, x0:x1] = sub
        return out
    m8 = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_fg = cv2.erode(m8, k)
    maybe = cv2.dilate(m8, k)
    gc = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[m8 > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any():
        return mask
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    out = ndi.binary_fill_holes(out)
    # never let refinement wander outside the dilated band or drop the sure interior
    out &= maybe > 0
    out |= sure_fg > 0
    return out


def temporal_smooth_masks(masks: list[np.ndarray], radius: int = 1) -> list[np.ndarray]:
    """Majority vote over a small temporal window (kills single-frame flicker)."""
    n = len(masks)
    if n < 3 or radius < 1:
        return masks
    out = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        votes = np.sum([masks[j] for j in range(lo, hi)], axis=0)
        out.append(votes * 2 > (hi - lo))
    return out


def label_cels(mask: np.ndarray, dilate_px: int = 1) -> CelLayers:
    """Connected components -> cel instances. A 1px dilation pulls the outline into the cel."""
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    lab, k = ndi.label(mask)
    return CelLayers(mask=mask, labels=lab.astype(np.int32), n=int(k))


def segment_shot(
    frames: list[np.ndarray],
    plate_crops: list[np.ndarray],
    threshold: float = 30.0,
    refine: str = "grabcut",
    temporal_radius: int = 1,
) -> list[CelLayers]:
    masks = [residual_mask(f, p, threshold=threshold) for f, p in zip(frames, plate_crops)]
    masks = temporal_smooth_masks(masks, temporal_radius)
    if refine == "grabcut":
        masks = [grabcut_refine(f, m) for f, m in zip(frames, masks)]
    return [label_cels(m) for m in masks]
