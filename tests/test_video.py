import numpy as np
import pytest

from conftest import write_clip
import cv2

from xdfdet.video import (crop_faces, denormalize, frame_indices, normalize, read_evenly, read_frames,
                          read_pair)


def test_frame_sampling_matches_original_runs():
    assert frame_indices(32, 12, step=3) == np.linspace(0, 31, 12, dtype=int).tolist()  # training
    assert frame_indices(32, 12, step=2) == list(range(0, 24, 2))                       # evaluation


def test_read_and_normalize(clip_tree):
    frames = read_frames(clip_tree / "000.mp4")
    assert len(frames) == 12 and frames[0].shape == (224, 224, 3) and frames[0].dtype == np.uint8
    x = normalize(frames[0])
    assert x.dtype == np.float32 and abs(x.mean()) < 3
    assert np.abs(denormalize(x).astype(int) - frames[0]).max() <= 1


def test_read_pair_is_aligned(clip_tree):
    real, fake = read_pair(clip_tree / "000.mp4", clip_tree / "FaceSwap" / "000_001.mp4")
    assert len(real) == len(fake) == 12
    assert np.abs(real[0][100:120, 100:130].astype(int) - fake[0][100:120, 100:130]).mean() > 50


@pytest.mark.parametrize("count", [4, 32])
def test_read_evenly_matches_seeking(clip_tree, count):
    path = clip_tree / "000.mp4"
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seeked = []
    for idx in np.linspace(0, total - 1, count, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        seeked.append(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
    frames = read_evenly(path, count)
    assert len(frames) == count and all(np.array_equal(a, b) for a, b in zip(frames, seeked))


def test_crop_faces_finds_the_face(clip_tree):
    crops = crop_faces(clip_tree / "000.mp4", frames=4)
    assert len(crops) == 4 and crops[0].shape == (224, 224, 3)


def test_crop_faces_without_face(tmp_path):
    write_clip(tmp_path / "empty.mp4", [np.full((224, 224, 3), 128, np.uint8)] * 8)
    with pytest.raises(ValueError):
        crop_faces(tmp_path / "empty.mp4", frames=4)
