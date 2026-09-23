"""Grad-CAM heatmaps and the eight-region attention analysis."""

import random

import cv2
import keras
import numpy as np
import tensorflow as tf

from .config import TRAIN_FRAME_STEP
from .landmarks import REGIONS, landmarks
from .model import video_score
from .video import denormalize, normalize, read_pair

GRADCAM_FRAMES = (0, 4, 7)   # frames averaged per video in the region analysis
CASES = ("TP", "TN", "FP", "FN")


def _frame_model(model):
    """Single-frame model returning (last conv activations, real-probability)."""
    if not hasattr(model, "_xdfdet_frame_model"):
        td = {type(l.layer): l.layer for l in model.layers if isinstance(l, keras.layers.TimeDistributed)}
        backbone = next(l for l in td.values() if isinstance(l, keras.Model))
        conv = [l for l in backbone.layers if isinstance(l, keras.layers.Conv2D)][-1]
        source = backbone.inputs[0]
        inner = keras.Model(source, [conv.output, backbone.output])
        inp = keras.Input(source.shape[1:])
        feats, x = inner(inp)
        x = td[keras.layers.GlobalAveragePooling2D](x)
        x = td[keras.layers.Dropout](x, training=False)
        model._xdfdet_frame_model = keras.Model(inp, [feats, td[keras.layers.Dense](x)])
    return model._xdfdet_frame_model


def gradcam(model, frame, target="auto"):
    """Grad-CAM for one normalized frame, as a (H, W) map in [0, 1].

    `target` is "real", "fake", or "auto" (whichever class the frame is predicted as).
    """
    with tf.GradientTape() as tape:
        feats, prob = _frame_model(model)(tf.convert_to_tensor(frame[None], tf.float32), training=False)
        p = tf.clip_by_value(tf.squeeze(prob, -1), 1e-6, 1 - 1e-6)
        score = {"real": tf.math.log(p), "fake": tf.math.log(1 - p)}.get(
            target, tf.where(p >= 0.5, tf.math.log(p), tf.math.log(1 - p)))
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2))[0].numpy().astype(np.float32)
    cam = np.maximum(np.mean(feats[0].numpy().astype(np.float32) * weights, axis=-1), 0)
    cam = cv2.resize(cam / cam.max() if cam.max() > 0 else cam, frame.shape[1::-1])
    return cam.astype(np.float32)


def video_gradcam(model, frames, indices=GRADCAM_FRAMES):
    """Mean Grad-CAM over a few frames of a normalized sequence, rescaled to [0, 1]."""
    cam = np.mean([gradcam(model, frames[min(i, len(frames) - 1)]) for i in indices], axis=0)
    return cam / cam.max() if cam.max() > 0 else cam


def region_scores(cam, points):
    """Mean activation (0-100) inside each landmark region."""
    cam = cam / cam.max() if cam.max() > 0 else cam
    scores = {}
    for name, idx in REGIONS.items():
        mask = np.zeros(cam.shape, np.uint8)
        cv2.fillPoly(mask, [np.array([points[i] for i in idx], np.int32)], 1)
        scores[name] = float(cam[mask == 1].mean() * 100) if mask.any() else 0.0
    return scores


def overlay(frame_rgb, cam, strength=0.7):
    """Color the frame by Grad-CAM, fading to transparent where activation is low."""
    cam = cv2.resize(cam, frame_rgb.shape[1::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (strength * cam ** 1.5)[..., None]
    return (frame_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def case_of(label, score, threshold=0.5):
    """Outcome with "fake" as the positive class, as in the thesis figures."""
    fake, predicted_fake = label == 0, score < threshold
    return {(True, True): "TP", (False, False): "TN", (False, True): "FP", (True, False): "FN"}[
        (fake, predicted_fake)]


def analyze(model, pairs, max_per_case=50, seed=0, step=TRAIN_FRAME_STEP):
    """Region scores for up to `max_per_case` test videos of each outcome.

    Frames are sampled as in training (the original analysis read the test set through
    the training data pipeline). Landmarks come from the middle frame. Returns
    {case: {"n": int, "mean": {region: value}, "std": {region: value}}}, where "std"
    is the spread across videos.
    """
    def sequences(real_path, fake_path):
        real, fake = read_pair(real_path, fake_path, step=step)
        return {1: np.stack([normalize(f) for f in real]), 0: np.stack([normalize(f) for f in fake])}

    by_case = {c: [] for c in CASES}
    for pair in pairs:
        for label, seq in sequences(*pair).items():
            by_case[case_of(label, video_score(model, seq))].append((pair, label))
    rng, result = random.Random(seed), {}
    for case, items in by_case.items():
        items = rng.sample(items, max_per_case) if len(items) > max_per_case else items
        rows = []
        for pair, label in items:
            seq = sequences(*pair)[label]
            points = landmarks(denormalize(seq[len(seq) // 2]))
            if points is not None:
                rows.append(region_scores(video_gradcam(model, seq), points))
        result[case] = {
            "n": len(rows),
            "mean": {r: float(np.mean([row[r] for row in rows])) for r in REGIONS} if rows else {},
            "std": {r: float(np.std([row[r] for row in rows])) for r in REGIONS} if rows else {},
        }
    return result


def spread_across_regions(result):
    """Standard deviation of the eight regional means, per outcome. Lower = more even attention."""
    return {case: float(np.std(list(v["mean"].values()))) for case, v in result.items() if v["mean"]}

