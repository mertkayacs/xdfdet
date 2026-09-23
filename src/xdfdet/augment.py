"""Training-time perturbations: SSIM-guided polygon cutout, star cutout, Albumentations."""

import os
import random

import cv2
import numpy as np
from skimage.metrics import structural_similarity

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # albumentations checks PyPI on import otherwise

from .config import (AUGMENT_LEVELS, MAX_AREA_RATIO, MIN_AREA_RATIO, SSIM_BIN, STAR_MAX_R,
                     STAR_MIN_R)


def ssim_difference(real, fake):
    """Per-pixel 1 - SSIM between two RGB uint8 frames, averaged over channels, in [0, 1]."""
    r, f = real.astype(np.float32) / 255.0, fake.astype(np.float32) / 255.0
    diff = np.zeros(r.shape[:2], np.float32)
    for c in range(3):
        _, smap = structural_similarity(r[..., c], f[..., c], full=True, data_range=1.0)
        diff += 1.0 - smap
    return np.clip(diff / 3.0, 0, 1)


def candidate_polygons(points, n_random=6):
    """Eyes, nose and mouth from the landmarks, plus `n_random` hulls of random landmark subsets."""
    if points is None or len(points) != 68:
        return []
    polys = [points[36:42], points[42:48], points[27:36], points[48:60], points[60:68]]
    for _ in range(n_random):
        subset = [points[i] for i in random.sample(range(68), random.randint(5, 12))]
        hull = cv2.convexHull(np.array(subset, np.int32)).squeeze().tolist()
        polys.append([hull] if isinstance(hull[0], int) else hull)
    return polys


def select_polygon(polygons, diff, min_overlap=0.3, ssim_bin=SSIM_BIN):
    """Pick the polygon that covers most of the region where real and fake look alike.

    Only polygons covering 2-5% of the frame qualify. If none reaches `min_overlap`,
    the best-scoring qualifying polygon is returned anyway (None if nothing qualifies).
    """
    region = (diff <= 1.0 - ssim_bin).astype(np.uint8)
    region_total = int(region.sum())
    img_area = float(diff.shape[0] * diff.shape[1])
    best, best_key, fallback, fallback_key = None, (-1.0, -1), None, (-1.0, -1, -1.0)
    for poly in polygons:
        area = cv2.contourArea(np.array(poly, np.float32))
        if not MIN_AREA_RATIO <= area / img_area <= MAX_AREA_RATIO:
            continue
        mask = np.zeros_like(region)
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)
        overlap = int((mask & region).sum())
        rho = overlap / region_total if region_total else 0.0
        if region_total and rho >= min_overlap and \
                (rho > best_key[0] or (np.isclose(rho, best_key[0]) and overlap > best_key[1])):
            best, best_key = poly, (rho, overlap)
        if (rho, overlap, area) > fallback_key:
            fallback, fallback_key = poly, (rho, overlap, area)
    return best if best is not None else fallback


def _fill(h, w, kind):
    if kind == "black":
        return np.zeros((h, w, 3), np.uint8)
    if kind == "white":
        return np.full((h, w, 3), 255, np.uint8)
    if kind == "random":
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    raise ValueError(f"unknown fill: {kind}")


def cut_polygon(frame, polygon, fill):
    """Replace the inside of `polygon` with black, white or random pixels."""
    if polygon is None:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
    out = frame.copy()
    out[mask == 1] = _fill(h, w, fill)[mask == 1]
    return out


def star_polygon(h, w, min_r=STAR_MIN_R, max_r=STAR_MAX_R, points=5):
    """Five-pointed star at a random position. Outer radius in [min_r, max_r)."""
    outer = np.random.randint(min_r, max_r)
    inner = max(3, int(outer * 0.5))
    cx, cy = np.random.randint(outer, w - outer), np.random.randint(outer, h - outer)
    pts = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        a = np.pi * i / points
        pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
    return pts


def augmentations(level):
    """Albumentations pipeline for a level in AUGMENT_LEVELS. Applied frame by frame."""
    from albumentations import (Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
                                HueSaturationValue, OneOf, RandomBrightnessContrast,
                                ShiftScaleRotate)
    noise, blur, color, geo = AUGMENT_LEVELS[level]
    return Compose([
        GaussNoise(p=noise),
        GaussianBlur(blur_limit=(3, 7), p=blur),
        HorizontalFlip(),
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05),
            FancyPCA(alpha=0.05),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
        ], p=color),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                         border_mode=cv2.BORDER_CONSTANT, p=geo),
    ])
