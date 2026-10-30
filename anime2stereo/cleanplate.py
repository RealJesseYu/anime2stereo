"""Clean background plate reconstruction ("temporal inpainting").

In cel animation the painted background is static for the duration of a shot (up to a
camera pan). Stacking pan-compensated frames and taking a per-pixel temporal median
removes every moving cel and recovers the background *behind* the characters wherever
it was revealed at some point in the shot. Pixels that were never revealed (a character
that never moved, or canvas area outside the pan) are filled with image inpainting.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Plate:
    canvas: np.ndarray          # (Hc, Wc, 3) uint8 clean background in canvas coords
    coverage: np.ndarray        # (Hc, Wc) int32: number of frames covering each pixel
    inpainted: np.ndarray       # (Hc, Wc) bool: pixels filled by image inpainting
    origin: tuple[int, int]     # (ox, oy): canvas coordinate of frame-0 top-left corner
    offsets: np.ndarray         # (n, 2) int: canvas offset (x, y) of each frame's top-left

    def crop(self, i: int, h: int, w: int) -> np.ndarray:
        x, y = self.offsets[i]
        return self.canvas[y:y + h, x:x + w]

    def crop_mask(self, mask: np.ndarray, i: int, h: int, w: int) -> np.ndarray:
        x, y = self.offsets[i]
        return mask[y:y + h, x:x + w]


def _masked_median_chunk(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring invalid entries. stack (n,h,w,3) uint8; valid (n,h,w) bool."""
    n = stack.shape[0]
    s = stack.astype(np.uint16)
    s[~valid] = 1000  # sentinel sorts last
    s.sort(axis=0)
    cnt = valid.sum(axis=0)  # (h,w)
    k = np.clip(cnt // 2, 0, n - 1)
    k3 = np.broadcast_to(k[None, :, :, None], (1,) + k.shape + (3,))
    med = np.take_along_axis(s, k3, axis=0)[0]
    med[cnt == 0] = 0
    return med.astype(np.uint8)


def build_plate(
    frames: list[np.ndarray],
    shifts: np.ndarray,
    max_samples: int = 120,
    inpaint_radius: int = 5,
    row_chunk: int = 48,
) -> Plate:
    """Build the clean plate for one shot.

    Args:
        frames: BGR frames of the shot.
        shifts: (n,2) cumulative (dx,dy) of each frame relative to frame 0 (from `stabilize`).
        max_samples: cap on frames used for the median (uniformly subsampled).
        inpaint_radius: radius for OpenCV Telea inpainting of never-revealed pixels.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    sh = np.round(shifts).astype(int)
    # Content moving by (+dx) between frames means the camera panned the other way; to
    # align frame i to frame 0 we place it at canvas offset = shift_i (relative to origin).
    minx, miny = sh[:, 0].min(), sh[:, 1].min()
    maxx, maxy = sh[:, 0].max(), sh[:, 1].max()
    ox, oy = -minx, -miny
    Wc, Hc = w + (maxx - minx), h + (maxy - miny)
    offsets = sh + np.array([ox, oy])

    idx = np.linspace(0, n - 1, min(n, max_samples)).round().astype(int)
    idx = np.unique(idx)
    coverage = np.zeros((Hc, Wc), dtype=np.int32)
    for i in idx:
        x, y = offsets[i]
        coverage[y:y + h, x:x + w] += 1

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    m = len(idx)
    for r0 in range(0, Hc, row_chunk):
        r1 = min(Hc, r0 + row_chunk)
        stack = np.zeros((m, r1 - r0, Wc, 3), dtype=np.uint8)
        valid = np.zeros((m, r1 - r0, Wc), dtype=bool)
        for j, i in enumerate(idx):
            x, y = offsets[i]
            # frame rows that land in [r0, r1)
            fy0, fy1 = max(r0, y), min(r1, y + h)
            if fy1 <= fy0:
                continue
            stack[j, fy0 - r0:fy1 - r0, x:x + w] = frames[i][fy0 - y:fy1 - y]
            valid[j, fy0 - r0:fy1 - r0, x:x + w] = True
        canvas[r0:r1] = _masked_median_chunk(stack, valid)

    hole = (coverage == 0)
    inpainted = hole.copy()
    if hole.any():
        canvas = cv2.inpaint(canvas, hole.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA)
    return Plate(canvas=canvas, coverage=coverage, inpainted=inpainted, origin=(ox, oy), offsets=offsets)


def refine_plate_with_masks(plate: Plate, frames: list[np.ndarray], masks: list[np.ndarray], inpaint_radius: int = 7) -> Plate:
    """Second pass: recompute the plate using only pixels that were *not* classified as cel.

    The first-pass median can still contain a character if it covered a pixel for more
    than half of the shot. Given cel masks from pass one, we take the median of
    background-labelled samples only; pixels that are cel in every frame are inpainted.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    Hc, Wc = plate.canvas.shape[:2]
    offsets = plate.offsets
    idx = np.linspace(0, n - 1, min(n, 120)).round().astype(int)
    idx = np.unique(idx)
    m = len(idx)
    canvas = np.zeros_like(plate.canvas)
    cov = np.zeros((Hc, Wc), dtype=np.int32)
    row_chunk = 48
    for r0 in range(0, Hc, row_chunk):
        r1 = min(Hc, r0 + row_chunk)
        stack = np.zeros((m, r1 - r0, Wc, 3), dtype=np.uint8)
        valid = np.zeros((m, r1 - r0, Wc), dtype=bool)
        for j, i in enumerate(idx):
            x, y = offsets[i]
            fy0, fy1 = max(r0, y), min(r1, y + h)
            if fy1 <= fy0:
                continue
            stack[j, fy0 - r0:fy1 - r0, x:x + w] = frames[i][fy0 - y:fy1 - y]
            valid[j, fy0 - r0:fy1 - r0, x:x + w] = ~masks[i][fy0 - y:fy1 - y]
        canvas[r0:r1] = _masked_median_chunk(stack, valid)
        cov[r0:r1] = valid.sum(axis=0)
    hole = cov == 0
    if hole.any():
        # seed holes with first-pass plate (better than zeros), then inpaint
        canvas[hole] = plate.canvas[hole]
        canvas = cv2.inpaint(canvas, hole.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA)
    return Plate(canvas=canvas, coverage=cov, inpainted=hole, origin=plate.origin, offsets=offsets)
