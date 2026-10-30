"""Depth assignment.

Three sources, all producing a float32 map in [0, 1] where 1 = nearest:

* ``cel``    – purely ordinal, no ML: background gets a gentle ground-plane gradient,
               each cel instance gets a flat depth ordered by its lowest point on screen
               (lower on screen = nearer, the classic multiplane-camera heuristic).
* ``ml``     – monocular relative depth from Depth-Anything-V2 (via transformers), run per
               frame, normalised per shot for temporal stability.
* ``hybrid`` – ML depth for the *clean plate* (computed once per shot, so the background
               is perfectly temporally stable), cels flattened to a per-instance median of
               the ML depth and forced in front of the background they occlude. This gives
               the paper-cutout / multiplane look that reads well for cel animation.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from .segmentation import CelLayers


class DepthEstimator:
    """Lazy wrapper around a transformers depth-estimation pipeline."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf", device: Optional[str] = None):
        self.model_name = model
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dev = self.device
        if dev is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        self.device = dev
        self._pipe = pipeline("depth-estimation", model=self.model_name, device=dev)

    def predict(self, frames_bgr: list[np.ndarray], batch_size: int = 8) -> list[np.ndarray]:
        """Relative inverse depth (bigger = nearer), float32, same HxW as input. Not normalised."""
        self._load()
        from PIL import Image

        out: list[np.ndarray] = []
        for i in range(0, len(frames_bgr), batch_size):
            batch = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr[i:i + batch_size]]
            res = self._pipe(batch)
            if isinstance(res, dict):
                res = [res]
            for f, r in zip(frames_bgr[i:i + batch_size], res):
                d = r["predicted_depth"]
                d = d.detach().float().cpu().numpy() if hasattr(d, "detach") else np.asarray(d, dtype=np.float32)
                d = np.squeeze(d)
                if d.shape != f.shape[:2]:
                    d = cv2.resize(d, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
                out.append(d.astype(np.float32))
        return out


def normalise_shot(depths: list[np.ndarray], lo_p: float = 1.0, hi_p: float = 99.0) -> list[np.ndarray]:
    """Map a list of raw relative-depth maps to [0,1] with *shared* percentiles (no per-frame flicker)."""
    sample = np.concatenate([d[::4, ::4].ravel() for d in depths])
    lo, hi = np.percentile(sample, [lo_p, hi_p])
    scale = max(hi - lo, 1e-6)
    return [np.clip((d - lo) / scale, 0.0, 1.0).astype(np.float32) for d in depths]


def ground_plane_depth(h: int, w: int, top: float = 0.0, bottom: float = 0.25) -> np.ndarray:
    """Background depth increasing toward the bottom of the frame."""
    return np.repeat(np.linspace(top, bottom, h, dtype=np.float32)[:, None], w, axis=1)


def cel_depth(layers: CelLayers, bg_depth: np.ndarray, near: float = 0.55, far: float = 0.95) -> np.ndarray:
    """Ordinal depth: background map + one flat depth per cel ordered by lowest pixel row."""
    depth = bg_depth.copy()
    if layers.n == 0:
        return depth
    bottoms = []
    for k in range(1, layers.n + 1):
        ys = np.where(layers.labels == k)[0]
        bottoms.append(ys.max() if ys.size else 0)
    order = np.argsort(bottoms)  # highest bottom (furthest) first
    for rank, k0 in enumerate(order):
        t = rank / max(layers.n - 1, 1)
        z = near + (far - near) * t
        depth[layers.labels == (k0 + 1)] = z
    return depth


def align_to_reference(d: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Affine-align a relative depth map to a reference using pixels in ``mask``.

    Monocular relative depth is only defined up to scale/shift per image; frames and the
    clean plate are separate inferences, so cel depths must be mapped into the plate's
    scale before they can be compared with it. Robust fit: match median and MAD.
    """
    if mask.sum() < 100:
        return d
    a, b = d[mask], ref[mask]
    ma, mb = np.median(a), np.median(b)
    sa, sb = np.median(np.abs(a - ma)) + 1e-6, np.median(np.abs(b - mb)) + 1e-6
    return np.clip((d - ma) * (sb / sa) + mb, 0.0, 1.0).astype(np.float32)


def static_foreground(plate_depth: np.ndarray, pop: float = 0.12, kernel_frac: float = 0.4, min_area_frac: float = 0.004) -> np.ndarray:
    """Find foreground 'islands' in the plate depth: things nearer than their surroundings.

    A character that never moves during a shot survives the temporal median and ends up
    in the plate. Depth exposes it: a grey-scale morphological opening with a kernel
    larger than the character estimates the background depth, and the residual marks it.
    """
    h, w = plate_depth.shape
    k = int(kernel_frac * min(h, w)) | 1
    opened = cv2.morphologyEx(plate_depth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fg = (plate_depth - opened) > pop
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n:
        sizes = ndi.sum(fg, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_area_frac * fg.size
        fg = keep[lab]
    return fg


def inpaint_behind(plate: np.ndarray, plate_depth: np.ndarray, fg: np.ndarray, dilate: int = 3, radius: int = 9):
    """Image-inpaint the plate colour and depth behind static foreground islands."""
    if not fg.any():
        return plate, plate_depth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    m = cv2.dilate(fg.astype(np.uint8), k)
    color = cv2.inpaint(plate, m * 255, radius, cv2.INPAINT_TELEA)
    d8 = (np.clip(plate_depth, 0, 1) * 255).astype(np.uint8)
    d8 = cv2.inpaint(d8, m * 255, radius, cv2.INPAINT_NS)
    depth = d8.astype(np.float32) / 255.0
    # inpainted depth must not be nearer than the surrounding background
    ring = cv2.dilate(m, k) & ~m
    if ring.any():
        depth[m > 0] = np.minimum(depth[m > 0], np.percentile(plate_depth[ring > 0], 95))
    return color, depth


def smooth_background_depth(d: np.ndarray, sigma_space: float = 9.0, sigma_color: float = 0.08) -> np.ndarray:
    """Edge-aware smoothing to remove ML speckle on painted backgrounds."""
    d32 = d.astype(np.float32)
    return cv2.bilateralFilter(d32, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def hybrid_depth(
    ml_frame_depth: np.ndarray,
    plate_depth_crop: np.ndarray,
    layers: CelLayers,
    min_pop: float = 0.12,
    relief: float = 0.5,
) -> np.ndarray:
    """Background from the (temporally static) plate depth; cels pushed in front.

    Args:
        ml_frame_depth: ML depth of this frame, already aligned to the plate's scale.
        plate_depth_crop: ML depth of the clean plate, cropped to this frame.
        layers: cel instances for this frame.
        min_pop: minimum depth gap between a cel and the background directly behind it.
        relief: 0 = each cel is a flat card (multiplane look); 1 = keep the full ML depth
            variation inside the cel; in between scales the within-cel variation.
    """
    depth = plate_depth_crop.astype(np.float32).copy()
    if layers.n == 0:
        return depth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i in range(1, layers.n + 1):
        m = layers.labels == i
        if not m.any():
            continue
        z_cel = float(np.median(ml_frame_depth[m]))
        ring = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~layers.mask
        z_bg = float(np.percentile(plate_depth_crop[ring], 90)) if ring.any() else float(np.median(plate_depth_crop[m]))
        z = min(1.0, max(z_cel, z_bg + min_pop))
        var = cv2.GaussianBlur(ml_frame_depth, (0, 0), 2.0)[m] - z_cel
        depth[m] = np.clip(z + relief * var, z_bg + 0.5 * min_pop, 1.0)
    return depth


def track_and_smooth_cel_depths(depths: list[np.ndarray], layers: list[CelLayers], alpha: float = 0.6) -> list[np.ndarray]:
    """Temporal EMA of each cel's median depth (relief kept), matching instances across frames by overlap."""
    out = [d.copy() for d in depths]
    prev_labels: np.ndarray | None = None
    prev_z: dict[int, float] = {}
    for t, (d, L) in enumerate(zip(out, layers)):
        cur_z: dict[int, float] = {}
        for i in range(1, L.n + 1):
            m = L.labels == i
            z = float(np.median(d[m]))
            if prev_labels is not None:
                ov = prev_labels[m]
                ov = ov[ov > 0]
                if ov.size:
                    j = int(np.bincount(ov).argmax())
                    if j in prev_z and ov.size > 0.3 * m.sum():
                        z = alpha * prev_z[j] + (1 - alpha) * z
            d[m] = np.clip(d[m] + (z - float(np.median(d[m]))), 0.0, 1.0)
            cur_z[i] = z
        prev_labels, prev_z = L.labels, cur_z
    return out


def colorize_depth(d: np.ndarray) -> np.ndarray:
    """0..1 depth -> BGR colormap for debugging (near = warm)."""
    return cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
