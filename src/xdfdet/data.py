"""Pairing FaceForensics++ videos and splitting them into train/val/test."""

import random
from pathlib import Path

MANIPULATIONS = ("FaceSwap", "Face2Face", "FaceShifter", "Deepfakes")


def pair_videos(root, total=1000, manipulations=MANIPULATIONS):
    """Match each real video with one fake, cycling through the manipulations.

    Expected layout (face-cropped clips, see `xdfdet crop`):
        root/000.mp4, root/001.mp4, ...            real videos
        root/<Manipulation>/000_003.mp4, ...       fakes, named <real id>_<source id>
    If the preferred manipulation is missing for a video, any other fake of it is used.
    """
    root = Path(root)
    reals = sorted(root.glob("*.mp4"))[:total]
    fakes = sorted(root.glob("*/*.mp4"))
    pairs = []
    for i, real in enumerate(reals):
        mine = [f for f in fakes if f.name.startswith(real.stem + "_")]
        wanted = [f for f in mine if f.parent.name == manipulations[i % len(manipulations)]]
        if wanted or mine:
            pairs.append((str(real), str((wanted or mine)[0])))
    return pairs


def split_pairs(pairs, train=0.70, val=0.15, seed=42):
    """Shuffle pairs with a fixed seed and cut 70/15/15. Pairs never cross splits."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    a = int(len(pairs) * train)
    b = a + int(len(pairs) * val)
    return pairs[:a], pairs[a:b], pairs[b:]
