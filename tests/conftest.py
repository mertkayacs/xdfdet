import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


@pytest.fixture(scope="session")
def face():
    """A public-domain NASA portrait shipped with scikit-image, cropped to the face."""
    from skimage import data
    img = data.astronaut()[20:240, 120:340]
    return cv2.resize(np.ascontiguousarray(img), (224, 224))


def write_clip(path, frames, fps=25):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


@pytest.fixture
def clip_tree(tmp_path, face):
    """Tiny FaceForensics++-shaped folder: 8 real clips, fakes in two manipulation folders."""
    rng = np.random.default_rng(0)
    for i in range(8):
        real = [np.clip(face.astype(int) + rng.integers(-3, 4), 0, 255).astype(np.uint8) for _ in range(32)]
        fake = [f.copy() for f in real]
        for f in fake:
            f[90:130, 90:140] = 255 - f[90:130, 90:140]  # a visible "manipulation"
        write_clip(tmp_path / f"{i:03d}.mp4", real)
        folder = tmp_path / ("FaceSwap" if i % 2 == 0 else "Face2Face")
        folder.mkdir(exist_ok=True)
        write_clip(folder / f"{i:03d}_{(i + 1) % 8:03d}.mp4", fake)
    return tmp_path
