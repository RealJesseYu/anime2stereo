"""Quantitative sanity checks (no ground truth exists for real anime, so we measure
what segmentation + inpainting buys us over naive single-layer DIBR)."""
from __future__ import annotations

import numpy as np

from .stereo import StereoParams, forward_warp


def hole_fraction(depth: np.ndarray, alpha: np.ndarray, params: StereoParams) -> float:
    """Fraction of output pixels left unwritten by a forward warp (before any filling), avg of both eyes."""
    disp = params.max_disparity * (depth - params.zpd)
    _, wl = forward_warp(np.zeros(depth.shape + (3,), np.uint8), alpha, disp, +1.0, params.splat_width)
    _, wr = forward_warp(np.zeros(depth.shape + (3,), np.uint8), alpha, disp, -1.0, params.splat_width)
    return float(0.5 * ((~wl).mean() + (~wr).mean()))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-6)))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))
