"""Render the method figures used in the README, the model card and the project site.

Every step runs the package's own code on a public-domain NASA portrait that ships
with scikit-image, so the figures can be regenerated without FaceForensics++.

    python scripts/make_figures.py docs/figures
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data

from xdfdet import augment, load_model
from xdfdet.explain import overlay, region_scores, video_gradcam
from xdfdet.landmarks import REGIONS, landmarks
from xdfdet.video import frame_indices, normalize

SIZE = 448
GOLD = (196, 167, 114)
INK = (30, 42, 56)
MODELS = ("baseline", "aug-standard", "cutout-black", "aug-cutout-black")


def save(img, path):
    cv2.imwrite(str(path.with_suffix(".webp")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 84])


def detect(frame):
    from mtcnn import MTCNN
    face = max(MTCNN().detect_faces(frame), key=lambda f: f["confidence"])
    return face["box"], face["keypoints"]


def aligned_crop(frame, box, kp, size, margin=0.3):
    from xdfdet.video import _align_and_crop
    return _align_and_crop(frame, box, kp["left_eye"], kp["right_eye"], size, margin)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    frame = np.ascontiguousarray(data.astronaut())
    box, kp = detect(frame)

    # 1. Detection on the full frame, then the eye-aligned crop.
    shown = frame.copy()
    x, y, w, h = box
    cv2.rectangle(shown, (x, y), (x + w, y + h), GOLD, 3)
    for p in kp.values():
        cv2.circle(shown, p, 5, GOLD, -1)
    save(shown, out / "detect.png")
    crop = aligned_crop(frame, box, kp, SIZE)
    small = aligned_crop(frame, box, kp, 224)
    save(crop, out / "crop.png")

    # 2. 68 landmarks and the eight regions used in the analysis.
    points = landmarks(small)
    scale = SIZE / 224
    marked = crop.copy()
    for idx in REGIONS.values():
        poly = (np.array([points[i] for i in idx]) * scale).astype(np.int32)
        cv2.polylines(marked, [poly], False, GOLD, 2, cv2.LINE_AA)
    for px, py in points:
        cv2.circle(marked, (int(px * scale), int(py * scale)), 3, INK, -1, cv2.LINE_AA)
    save(marked, out / "landmarks.png")
    masks = crop.copy()
    for idx in REGIONS.values():  # the exact polygons region_scores() fills
        poly = (np.array([points[i] for i in idx]) * scale).astype(np.int32)
        layer = masks.copy()
        cv2.fillPoly(layer, [poly], GOLD)
        masks = cv2.addWeighted(layer, 0.45, masks, 0.55, 0)
        cv2.polylines(masks, [poly], True, INK, 1, cv2.LINE_AA)
    save(masks, out / "regions.png")

    # 3. Cutout: one landmark polygon inside the 2-5% area band, three fills; star on a real frame.
    random.seed(7)
    np.random.seed(7)
    polygon = None
    while polygon is None:
        polygon = augment.select_polygon(augment.candidate_polygons(points), np.zeros((224, 224), np.float32))
    big = (np.array(polygon) * scale).astype(np.int32)
    for fill in ("black", "white", "random"):
        save(augment.cut_polygon(crop, big, fill), out / f"cutout-{fill}.png")
    star = (np.array(augment.star_polygon(224, 224)) * scale).astype(np.int32)
    save(augment.cut_polygon(crop, star, "random"), out / "star.png")

    # 4. Augmentation samples (the pipeline is applied at 224 px, as in training).
    for level, seed, visible in (("standard", 3, 5), ("intense", 11, 9)):
        random.seed(seed)
        np.random.seed(seed)
        pipe = augment.augmentations(level)
        sample = small
        for _ in range(100):  # draw until the change is large enough to see at page size
            sample = pipe(image=small)["image"]
            flipped = small[:, ::-1]  # a plain flip is not what the level changes
            if min(np.abs(sample.astype(int) - small).mean(), np.abs(sample.astype(int) - flipped).mean()) > visible:
                break
        save(cv2.resize(sample, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC), out / f"aug-{level}.png")

    # 5. Grad-CAM of four released models on the same face.
    seq = np.stack([normalize(small)] * 12)
    report = {}
    for name in MODELS:
        model = load_model(name)
        score = float(model.predict(seq[None], verbose=0).mean())
        cam = video_gradcam(model, seq)
        save(overlay(crop, cam), out / f"gradcam-{name}.png")
        report[name] = {"real_probability": round(score, 4),
                        "regions": {k: round(v, 1) for k, v in region_scores(cam, points).items()}}
        print(name, report[name], flush=True)
    (out / "gradcam.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
