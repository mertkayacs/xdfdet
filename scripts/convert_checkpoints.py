"""Convert the original Colab checkpoints (.h5, mixed precision) into the released .keras files.

The mapping below was checked against the training notebooks: optimizer step counts
match each run's epoch count, dropout rates match the notebook code, and file
timestamps match the save cells. `newbaseline.h5` (dropout 0.4) is an early
experiment outside the paper. The aug-intense checkpoint was overwritten on Drive
by aug-standard (both runs saved to justaug.h5), so it is not released.

Do not use a model returned by `keras.models.load_model` on these .h5 files. In
Keras 3.10 that path leaves EfficientNet's Normalization layer with mean 0 and
variance 1, so the loaded model skips the normalization the network was trained
with. Only the weights are taken from the files; the architecture is rebuilt.

    python scripts/convert_checkpoints.py ORIGINALS_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

import keras
import numpy as np

from xdfdet.config import CONFIGS
from xdfdet.model import build_model

ORIGINALS = {
    "baseline": "baseline_no_aug_no_cutout.h5",
    "aug-standard": "justaug.h5",
    "cutout-random": "noaug.h5",
    "cutout-black": "noaugcb.h5",
    "cutout-white": "noaugcw.h5",
    "aug-cutout-random": "rc12.h5",
    "aug-cutout-black": "bzeroc12.h5",
    "aug-cutout-white": "whitec12.h5",
}


def face_probe():
    """Two 12-frame sequences of a public-domain NASA portrait (scikit-image), normalized."""
    import cv2
    from skimage import data
    from xdfdet.video import normalize
    face = cv2.resize(np.ascontiguousarray(data.astronaut()[20:240, 120:340]), (224, 224))
    dim = np.clip(face.astype(int) * 0.6 + 40, 0, 255).astype(np.uint8)
    return np.stack([np.stack([normalize(f)] * 12) for f in (face, dim)])


def convert(src, dst):
    probe = face_probe()
    report = {}
    for name, filename in ORIGINALS.items():
        original = keras.models.load_model(src / filename, compile=False)
        weights = original.get_weights()
        dropout = next(l.layer.rate for l in original.layers
                       if isinstance(getattr(l, "layer", None), keras.layers.Dropout))
        assert dropout == CONFIGS[name].dropout, (name, dropout)

        # Training-time setup: mixed precision, ImageNet architecture, trained weights.
        keras.mixed_precision.set_global_policy("mixed_float16")
        reference = build_model(dropout=dropout)
        reference.set_weights(weights)

        keras.mixed_precision.set_global_policy("float32")
        model = build_model(dropout=dropout)
        model.set_weights(weights)
        path = dst / f"{name}.keras"
        model.save(path)

        reloaded = keras.saving.load_model(path, compile=False)
        same = all(np.array_equal(a, b) for a, b in zip(weights, reloaded.get_weights()))
        assert same, name
        report[name] = {
            "source": filename,
            "weights_identical": same,
            "video_score_mixed_float16": reference.predict(probe, verbose=0).mean(axis=(1, 2)).round(4).tolist(),
            "video_score_release": reloaded.predict(probe, verbose=0).mean(axis=(1, 2)).round(4).tolist(),
        }
        print(name, report[name], flush=True)
    (dst / "conversion.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    convert(src, dst)
