import numpy as np
import cv2

from xdfdet import augment
from xdfdet.landmarks import REGIONS, landmarks


def test_landmarks_on_a_real_face(face):
    points = landmarks(face)
    assert points is not None and len(points) == 68
    assert sorted(i for r in REGIONS.values() for i in r) == list(range(68))


def test_ssim_difference():
    a = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert augment.ssim_difference(a, a).max() < 1e-6
    assert augment.ssim_difference(a, 255 - a).mean() > 0.5


def test_polygon_selection_respects_area_limits(face):
    import random
    points = landmarks(face)
    diff = np.zeros((224, 224), np.float32)  # everything "similar"
    chosen = []
    for seed in range(20):
        random.seed(seed)
        poly = augment.select_polygon(augment.candidate_polygons(points), diff)
        if poly is not None:  # None is allowed: no candidate fits the area limits
            chosen.append(cv2.contourArea(np.array(poly, np.float32)) / 224 ** 2)
    assert chosen and all(0.02 <= r <= 0.05 for r in chosen)


def test_fills_and_star():
    frame = np.full((224, 224, 3), 100, np.uint8)
    square = [(10, 10), (60, 10), (60, 60), (10, 60)]
    assert augment.cut_polygon(frame, square, "black")[30, 30].tolist() == [0, 0, 0]
    assert augment.cut_polygon(frame, square, "white")[30, 30].tolist() == [255, 255, 255]
    assert augment.cut_polygon(frame, None, "white")[30, 30].tolist() == [100, 100, 100]
    for _ in range(50):
        pts = np.array(augment.star_polygon(224, 224))
        assert pts.min() >= 0 and pts.max() < 224
        assert (pts.max(0) - pts.min(0)).max() <= 2 * 16


def test_augmentation_levels_keep_shape():
    frame = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for level in ("flip", "standard", "intense"):
        assert augment.augmentations(level)(image=frame)["image"].shape == frame.shape
