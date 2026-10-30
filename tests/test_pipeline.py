import numpy as np
import pytest

from anime2stereo import cleanplate, segmentation, stabilize, stereo, depth as depthmod, shots
from anime2stereo.metrics import iou, psnr, hole_fraction
from anime2stereo.stereo import StereoParams

from synth import make_sequence


@pytest.fixture(scope="module")
def seq():
    return make_sequence(n=40, pan_per_frame=(1.5, 0.0))


def test_pan_estimation(seq):
    frames, _, _ = seq
    shifts = stabilize.estimate_shifts(frames)
    # content moves left by ~1.5 px/frame (camera pans right) -> alignment offset grows +x
    assert abs(shifts[-1, 0] - 1.5 * 39) < 3.0
    assert abs(shifts[-1, 1]) < 2.0


def test_clean_plate_and_masks(seq):
    frames, true_masks, bgs = seq
    h, w = frames[0].shape[:2]
    shifts = stabilize.estimate_shifts(frames)
    plate = cleanplate.build_plate(frames, shifts)
    crops = [plate.crop(i, h, w) for i in range(len(frames))]
    # background recovered behind the moving cel
    ps = np.mean([psnr(c, b) for c, b in zip(crops, bgs)])
    assert ps > 30, ps
    layers = segmentation.segment_shot(frames, crops, refine="grabcut")
    ious = [iou(L.mask, m) for L, m in zip(layers, true_masks)]
    assert np.mean(ious) > 0.85, np.mean(ious)
    assert all(L.n == 1 for L in layers[2:-2])


def test_two_pass_plate_removes_static_cel():
    # a cel that moves only in the second half: first-pass median keeps it, second pass must not
    frames, true_masks, bgs = make_sequence(n=30, pan_per_frame=(0.0, 0.0))
    frames = [frames[0]] * 20 + frames[1:11]
    bgs = [bgs[0]] * 20 + bgs[1:11]
    h, w = frames[0].shape[:2]
    shifts = np.zeros((len(frames), 2))
    plate = cleanplate.build_plate(frames, shifts)
    first = psnr(plate.crop(0, h, w), bgs[0])
    masks = [segmentation.residual_mask(f, plate.crop(i, h, w)) for i, f in enumerate(frames)]
    # masks from pass 1 only see the cel once it moves; refine from a motion-only mask union
    moving = np.any(masks, axis=0)
    plate2 = cleanplate.refine_plate_with_masks(plate, frames, [moving] * len(frames))
    second = psnr(plate2.crop(0, h, w), bgs[0])
    assert second > first


def test_stereo_disparity_direction(seq):
    frames, true_masks, bgs = seq
    h, w = frames[0].shape[:2]
    f, m, bg = frames[5], true_masks[5], bgs[5]
    labels = m.astype(np.int32)
    bgd = np.zeros((h, w), np.float32)
    d = bgd.copy(); d[m] = 1.0
    p = StereoParams(max_disparity=20.0, zpd=0.0)
    L, R = stereo.render_stereo(bg, bgd, f, d, labels, p)
    # cel centroid should move right in L and left in R by ~20px each -> 40px apart
    def cx(img):
        bright = (img.mean(axis=2) > 200)
        xs = np.where(bright)[1]
        return xs.mean()
    assert 34 < cx(L) - cx(R) < 46, cx(L) - cx(R)
    # background (disparity 0) must be unchanged
    assert psnr(L[:, :20], bg[:, :20]) > 40


def test_layered_rendering_has_no_disocclusion_holes(seq):
    frames, true_masks, bgs = seq
    h, w = frames[0].shape[:2]
    m = true_masks[5]
    d = np.zeros((h, w), np.float32); d[m] = 1.0
    p = StereoParams(max_disparity=20.0, zpd=0.0)
    naive = hole_fraction(d, np.ones((h, w), bool), p)
    layered = hole_fraction(np.zeros((h, w), np.float32), np.ones((h, w), bool), p)
    assert naive > 0.01
    assert layered == 0.0


def test_cel_depth_ordering():
    labels = np.zeros((50, 50), np.int32)
    labels[5:15, 5:15] = 1     # high on screen -> far
    labels[35:45, 30:40] = 2   # low on screen -> near
    L = segmentation.CelLayers(mask=labels > 0, labels=labels, n=2)
    d = depthmod.cel_depth(L, np.zeros((50, 50), np.float32))
    assert d[40, 35] > d[10, 10] > 0


def test_shot_detection():
    frames, _, _ = make_sequence(n=20, seed=1)
    frames2, _, _ = make_sequence(n=20, seed=7)
    frames2 = [255 - f for f in frames2]
    cuts = shots.detect_shots(frames + frames2)
    assert cuts == [(0, 20), (20, 40)], cuts
