"""Test-set scoring with the four metrics reported in the paper."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)

from .config import EVAL_FRAME_STEP
from .model import video_score
from .video import normalize, read_pair


def score_pairs(model, pairs, step=EVAL_FRAME_STEP):
    """Video-level real-probabilities for every real and fake video in `pairs`.

    Returns (labels, scores) with label 1 = real, 0 = fake.
    """
    labels, scores = [], []
    for real_path, fake_path in pairs:
        real, fake = read_pair(real_path, fake_path, step=step)
        for frames, label in ((real, 1), (fake, 0)):
            labels.append(label)
            scores.append(video_score(model, np.stack([normalize(f) for f in frames])))
    return np.array(labels), np.array(scores)


def metrics(labels, scores, threshold=0.5):
    """AUC, F1, Brier and LogLoss as in the paper, plus accuracy and average precision.

    F1 treats "real" (label 1) as the positive class, as the thesis evaluation did.
    """
    pred = (scores > threshold).astype(int)
    return {
        "auc": roc_auc_score(labels, scores),
        "f1": f1_score(labels, pred),
        "brier": brier_score_loss(labels, scores),
        "logloss": log_loss(labels, scores),
        "accuracy": accuracy_score(labels, pred),
        "average_precision": average_precision_score(labels, scores),
    }
