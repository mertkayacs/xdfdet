"""Reading frames, ImageNet normalization, and the MTCNN face-crop step."""

import cv2
import numpy as np

from .config import CROP_FRAMES, IMG_SIZE, MEAN, SEQ_LEN, STD


def frame_indices(total, count=SEQ_LEN, step=2):
    """Every `step`-th frame from the start, or evenly spaced frames if the clip is too short."""
    if total >= count * step:
        return [i * step for i in range(count)]
    return np.linspace(0, max(0, total - 1), count, dtype=int).tolist()


def _read(cap, indices, size):
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB) if ok else None)
    return frames


def _fill_gaps(frames, size):
    """Repeat the last good frame over read failures; black if none was read yet."""
    out, last = [], np.zeros((size, size, 3), np.uint8)
    for f in frames:
        last = f if f is not None else last.copy()
        out.append(last)
    return out


def read_frames(path, count=SEQ_LEN, step=2, size=IMG_SIZE):
    """Sample `count` RGB uint8 frames from one video."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = _fill_gaps(_read(cap, frame_indices(total, count, step), size), size)
    cap.release()
    return frames


def read_pair(real_path, fake_path, count=SEQ_LEN, step=2, size=IMG_SIZE):
    """Sample the same frame indices from a real video and its fake.

    A frame is kept only if both reads succeed, as in the original pipeline,
    so the two sequences stay aligned for the SSIM comparison.
    """
    cap_r, cap_f = cv2.VideoCapture(str(real_path)), cv2.VideoCapture(str(fake_path))
    total = min(int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT)))
    idx = frame_indices(total, count, step)
    real, fake = _read(cap_r, idx, size), _read(cap_f, idx, size)
    cap_r.release(), cap_f.release()
    both = [(r, f) if r is not None and f is not None else (None, None) for r, f in zip(real, fake)]
    return _fill_gaps([r for r, _ in both], size), _fill_gaps([f for _, f in both], size)


def normalize(frame):
    """uint8 RGB -> float32, ImageNet mean/std."""
    return ((frame.astype(np.float32) / 255.0 - MEAN) / STD).astype(np.float32)


def denormalize(frame):
    return np.clip((frame * STD + MEAN) * 255, 0, 255).astype(np.uint8)


# ---- face crop (thesis Sec. 5.1.1) -----------------------------------------

MARGIN = 0.3  # the thesis says "fixed margin" without a value; 30% of the box is our choice


def _align_and_crop(frame, box, left_eye, right_eye, size, margin):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    rotated = cv2.warpAffine(frame, cv2.getRotationMatrix2D((cx, cy), angle, 1.0),
                             (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    half = max(w, h) * (1 + margin) / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - frame.shape[1], y1 - frame.shape[0])
    if pad:
        rotated = cv2.copyMakeBorder(rotated, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(rotated[y0:y1, x0:x1], (size, size))


def crop_faces(path, frames=CROP_FRAMES, size=IMG_SIZE, margin=MARGIN, detector=None):
    """Detect, eye-align and crop the main face in `frames` evenly spaced frames.

    Returns RGB uint8 crops. Frames without a detection reuse the previous face box.
    Raises ValueError if no face is found anywhere in the video.
    """
    if detector is None:
        from mtcnn import MTCNN
        detector = MTCNN()
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw = []
    for idx in np.linspace(0, max(0, total - 1), frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    crops, last = [], None
    for frame in raw:
        faces = detector.detect_faces(frame)
        if faces:
            best = max(faces, key=lambda f: f["confidence"])
            kp = best["keypoints"]
            last = (best["box"], kp["left_eye"], kp["right_eye"])
        if last is not None:
            crops.append(_align_and_crop(frame, *last, size, margin))
    if not crops:
        raise ValueError(f"no face found in {path}")
    while len(crops) < frames:
        crops.insert(0, crops[0])
    return crops


def write_video(frames, path, fps=25):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
