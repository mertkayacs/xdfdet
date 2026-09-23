"""dlib 68-point facial landmarks and the eight regions used in the analysis."""

import bz2
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2

PREDICTOR_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Index ranges of the 68-point scheme. Left/right follow dlib's numbering.
REGIONS = {
    "jaw": range(0, 17),
    "left_eyebrow": range(17, 22),
    "right_eyebrow": range(22, 27),
    "nose": range(27, 36),
    "left_eye": range(36, 42),
    "right_eye": range(42, 48),
    "outer_mouth": range(48, 60),
    "inner_mouth": range(60, 68),
}


def predictor_path():
    """Download dlib's landmark model once. It is not redistributed with this package
    because its training data (iBUG 300-W) is licensed for research use only."""
    cache = Path(os.environ.get("XDFDET_CACHE", Path.home() / ".cache" / "xdfdet"))
    path = cache / "shape_predictor_68_face_landmarks.dat"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PREDICTOR_URL) as r:
            path.write_bytes(bz2.decompress(r.read()))
    return path


@lru_cache(maxsize=1)
def _dlib():
    import dlib
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(predictor_path()))


def landmarks(frame):
    """68 (x, y) points for the first face dlib finds in an RGB frame, or None."""
    detector, predictor = _dlib()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    return [(p.x, p.y) for p in predictor(gray, faces[0]).parts()]
